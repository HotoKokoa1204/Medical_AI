"""Module: preprocess_pca
Stage: Script
Author: KafuuChino
Date: 2026-09-28
Description: End-to-end rolling 3-year preprocessing, PCA, and outlier pruning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Add src to sys.path to enable imports in all environments
sys_path_root = Path(__file__).resolve().parent.parent / "src"
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from agilab_lib.analysis import (  # noqa: E402
    calculate_hotelling_t2,
    calculate_spe,
    calculate_spe_limit,
    calculate_t2_limit,
    fit_pca,
    prune_multivariate_outliers,
    summarize_components,
)
from agilab_lib.preprocessing import (  # noqa: E402
    LongitudinalPreprocessor,
    deterministic_pruning,
    engineer_features,
    load_longitudinal_cohort,
    partition_stratified_group_5fold,
    split_patient_cohort,
)
from agilab_lib.visualization import (  # noqa: E402
    plot_pca_loadings_biplot,
    plot_pca_scatter,
    plot_scree,
    plot_t2_vs_spe,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Hemodialysis Rolling 3-Year Preprocessing & PCA Pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to Kidit_Master_Baseline_V2.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save processed Parquet datasets and audit trail.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save generated publication figures.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of principal components to estimate (default 5).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute end-to-end rolling 3-year cohort preprocessing and PCA pipeline.

    Follows Spec line 54:
        Patient Group Split -> Feature Engineering -> Deterministic Pruning (Train)
        -> Imputer Fit (Train) -> PCA / T2 / SPE Pruning (Train) -> Transform (Test).

    Returns:
        Integer exit code (0 for success).
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    # Resolve input data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        candidates = [
            project_root / "data" / "Kidit_Master_Baseline_V2.xlsx",
            project_root.parent.parent
            / "AGILAB_MedicalAI"
            / "data"
            / "Kidit_Master_Baseline_V2.xlsx",
            Path(
                "E:/github/MedicalAI/AGILAB_MedicalAI/data/Kidit_Master_Baseline_V2.xlsx"
            ),
            Path("E:/github/Kidit_Master_Baseline_V2.xlsx"),
            Path("E:/github/MedicalAI/data/Kidit_Master_Baseline_V2.xlsx"),
        ]
        data_path = next((p for p in candidates if p.exists()), candidates[0])

    print("=" * 75)
    print("[PIPELINE] AGILAB MedicalAI: Rolling 3-Year Preprocessing & PCA Pipeline")
    print("=" * 75)
    print(f"[INFO] Input Data Path: {data_path.resolve()}")

    if not data_path.exists():
        print(f"[ERROR] Data file not found at {data_path}")
        return 1

    # Define single output directories (no duplicate exports)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "processed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = (
        Path(args.figures_dir)
        if args.figures_dir
        else project_root / "reports" / "figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Longitudinal Rolling 3-Year Dynamic Cohort
    print("\n[Step 1/6] Loading longitudinal panel and constructing 3-year cohort...")
    cohort_df = load_longitudinal_cohort(data_path)
    n_cohort = len(cohort_df)
    n_patients = cohort_df["PatientID"].nunique()
    death_cnt = int(cohort_df["is_death_3yr"].sum())
    print(
        f"  [OK] Valid rolling 3-year cohort: {n_cohort} records across "
        f"{n_patients} unique patients."
    )
    print(
        f"  [OK] 3-Year Mortality Distribution: {death_cnt} Deceased, "
        f"{n_cohort - death_cnt} Survived (Event Rate: {death_cnt / n_cohort:.2%})"
    )

    # 2. Patient Group Split (GroupShuffleSplit on PatientID)
    print("\n[Step 2/6] Partitioning cohort strictly by PatientID (80/20)...")
    train_raw, test_raw = split_patient_cohort(
        cohort_df, train_size=0.80, random_state=42
    )
    print(
        f"  [OK] Train Split: {len(train_raw)} records "
        f"({train_raw['PatientID'].nunique()} patients, "
        f"Death Rate: {train_raw['is_death_3yr'].mean():.2%})"
    )
    print(
        f"  [OK] Test Split:  {len(test_raw)} records "
        f"({test_raw['PatientID'].nunique()} patients, "
        f"Death Rate: {test_raw['is_death_3yr'].mean():.2%})"
    )

    # Anti-leakage verification
    train_patients = set(train_raw["PatientID"])
    test_patients = set(test_raw["PatientID"])
    overlap_patients = train_patients.intersection(test_patients)
    assert len(overlap_patients) == 0, "Identity leakage: overlap detected!"
    print("  [OK] Anti-Leakage Check Passed: Train ∩ Test == ∅")

    # 3. Feature Engineering on Train and Test
    print("\n[Step 3/6] Engineering clinical features on Train and Test...")
    train_eng = engineer_features(train_raw)
    test_eng = engineer_features(test_raw)

    # Deterministic Pruning (Train only, per Spec line 54)
    print("  Applying deterministic pruning exclusively to Training split...")
    b_ktv = (train_eng["Kt_V"] < 0.5) & train_eng["Kt_V"].notna()
    b_npcr = (train_eng["nPCR"] < 0.4) & train_eng["nPCR"].notna()
    b_wgain = train_eng["ultrafiltration"] < -0.5
    b_wloss = train_eng["ultrafiltration"] > 6.0
    b_bfr = ((train_eng["BFR"] < 100) | (train_eng["BFR"] == 0)) & train_eng[
        "BFR"
    ].notna()
    b_dfr = ((train_eng["DFR"] < 300) | (train_eng["DFR"] == 0)) & train_eng[
        "DFR"
    ].notna()
    b_bun = (
        (train_eng["after_hd_BUN"] >= train_eng["before_hd_BUN"])
        & train_eng["after_hd_BUN"].notna()
        & train_eng["before_hd_BUN"].notna()
    )
    b_bun_unver = b_bun & ~((train_eng["Kt_V"] >= 1.0) & train_eng["Kt_V"].notna())

    stage1_breakdown = {
        "ktv_lt_0_5": int(b_ktv.sum()),
        "npcr_lt_0_4": int(b_npcr.sum()),
        "ultrafiltration_gain_gt_0_5": int(b_wgain.sum()),
        "ultrafiltration_loss_gt_6_0": int(b_wloss.sum()),
        "bfr_zero_or_lt_100": int(b_bfr.sum()),
        "dfr_zero_or_lt_300": int(b_dfr.sum()),
        "bun_inversion_unverified": int(b_bun_unver.sum()),
    }

    train_det = deterministic_pruning(train_eng)
    n_det_dropped = len(train_eng) - len(train_det)
    stage1_breakdown["total_dropped"] = n_det_dropped
    stage1_breakdown["remaining"] = len(train_det)

    print(
        f"  [OK] Train deterministic pruning dropped {n_det_dropped} errors. "
        f"Remaining: {len(train_det)} records."
    )

    # 4. Type-Specific Imputer Fit on Train & Inference on Test
    print("\n[Step 4/6] Fitting type preprocessor on Train & transforming Test...")
    preprocessor = LongitudinalPreprocessor()
    preprocessor.fit(train_det)

    cont_train_df, full_train_df = preprocessor.transform(train_det)
    cont_test_df, full_test_df = preprocessor.transform(test_eng)

    print(f"  [OK] Preprocessed Continuous Train Matrix: {cont_train_df.shape}")
    print(f"  [OK] Preprocessed Full Train Matrix:       {full_train_df.shape}")
    null_cnt = full_test_df.isna().sum().sum()
    print(
        f"  [OK] Uncurated Test Matrix (Zero Deletions): {full_test_df.shape} "
        f"(Nulls: {null_cnt})"
    )

    # 5. Fit PCA & Multivariate Outlier Pruning (T^2 & Jackson-Mudholkar SPE)
    print(
        f"\n[Step 5/6] Fitting PCA (k={args.n_components}) and computing T^2 / SPE..."
    )
    pca, scores_df, loadings_df = fit_pca(cont_train_df, n_components=args.n_components)
    _ = summarize_components(pca, loadings_df, top_n=5)

    recon_train = pca.inverse_transform(scores_df.values)
    t2_train = calculate_hotelling_t2(scores_df.values, pca.explained_variance_)
    spe_train = calculate_spe(cont_train_df.values, recon_train)

    # Full PCA on continuous features for genuine Jackson-Mudholkar SPE limit
    pca_full = PCA().fit(cont_train_df)
    residual_eigenvalues = pca_full.explained_variance_[args.n_components :]

    t2_lim = calculate_t2_limit(
        n_samples=len(cont_train_df),
        n_components=args.n_components,
        alpha=0.01,
    )
    spe_lim = calculate_spe_limit(
        residual_eigenvalues=residual_eigenvalues,
        alpha=0.01,
    )

    is_outlier, t2_lim, spe_lim = prune_multivariate_outliers(
        scores=scores_df,
        spe=spe_train,
        explained_variance=pca.explained_variance_,
        t2_limit=t2_lim,
        spe_limit=spe_lim,
        alpha=0.01,
        residual_eigenvalues=residual_eigenvalues,
    )

    t2_out_cnt = int(np.sum(t2_train > t2_lim))
    spe_out_cnt = int(np.sum(spe_train > spe_lim))
    intersection_cnt = int(np.sum((t2_train > t2_lim) & (spe_train > spe_lim)))
    n_pca_pruned = int(np.sum(is_outlier))

    stage2_breakdown = {
        "t2_outliers": t2_out_cnt,
        "spe_outliers": spe_out_cnt,
        "intersection_outliers": intersection_cnt,
        "total_dropped": n_pca_pruned,
    }

    # Filter preprocessed train matrices to purified subset
    inlier_mask = ~is_outlier
    full_train_clean = full_train_df[inlier_mask].copy()
    cont_train_clean = cont_train_df[inlier_mask].copy()
    clean_scores = scores_df[inlier_mask].copy()

    total_train_pruned = len(train_raw) - len(full_train_clean)
    prune_pct = (total_train_pruned / len(train_raw)) * 100.0

    print(f"  [OK] Hotelling's T^2 99% Limit: {t2_lim:.3f} (Outliers: {t2_out_cnt})")
    print(
        f"  [OK] SPE (Q-statistic) 99% Limit: {spe_lim:.3f} (Outliers: {spe_out_cnt})"
    )
    print(f"  [OK] Multivariate PCA Outliers Pruned: {n_pca_pruned} records.")
    print(
        f"  [OK] Total Train Pruned: {total_train_pruned} records ({prune_pct:.2f}%). "
        f"Purified Train Set: {len(full_train_clean)} records."
    )

    # 5b. Immutable Stratified Group 5-Fold Partitioning
    print(
        "\n[Step 5b] Partitioning purified train set into immutable 5 folds "
        "(grouped by PatientID, stratified by is_death_3yr)..."
    )
    train_det_clean = train_det.loc[full_train_clean.index]
    year_series = (
        train_det_clean["報告年度"]
        if "報告年度" in train_det_clean.columns
        else train_det_clean.get("year")
    )
    full_train_clean, audit_df = partition_stratified_group_5fold(
        df=full_train_clean,
        patient_ids=train_det_clean["PatientID"],
        years=year_series,
        targets=full_train_clean["is_death_3yr"],
    )
    fold_dist = dict(full_train_clean["fold"].value_counts().sort_index())
    print(
        f"  [OK] Assigned immutable 5-fold partition to {len(full_train_clean)} "
        f"records. Folds: {fold_dist}"
    )

    # 6. Save Cleaned Parquet Datasets & Audit Trail (Export once to output_dir)
    print("\n[Step 6/6] Saving Parquet feature matrices and publication figures...")
    train_parquet_path = output_dir / "train_cleaned_rolling_3yr.parquet"
    test_parquet_path = output_dir / "test_uncurated_rolling_3yr.parquet"
    cont_parquet_path = output_dir / "continuous_features_pca.parquet"
    folds_csv_path = output_dir / "train_cv_folds.csv"
    audit_trail_path = output_dir / "preprocessing_audit_trail.json"

    # Export standalone audit CSV with columns:
    # [record_id, PatientID, year, is_death_3yr, fold]
    audit_df.to_csv(folds_csv_path, index=False)
    print(f"  [OK] Saved Train CV Folds Audit Table: {folds_csv_path}")

    # Verify fold isolation invariants
    assert "fold" not in full_test_df.columns, "Test set must not contain fold column!"
    assert "fold" in full_train_clean.columns, "Train set must contain fold column!"
    assert full_train_clean["fold"].dtype == np.int8, "Fold column must be int8!"

    full_train_clean.to_parquet(train_parquet_path, engine="pyarrow", index=True)
    full_test_df.to_parquet(test_parquet_path, engine="pyarrow", index=True)
    cont_train_clean.to_parquet(cont_parquet_path, engine="pyarrow", index=True)

    # Build and export audit trail JSON
    audit_trail = {
        "cohort_counts": {
            "total": len(cohort_df),
            "train": len(train_raw),
            "test": len(test_raw),
        },
        "stage1_deterministic_pruning": stage1_breakdown,
        "stage2_pca_outlier_pruning": stage2_breakdown,
        "cleaned_train_count": len(full_train_clean),
        "uncurated_test_count": len(test_raw),
        "thresholds": {
            "t2_limit_0_01": float(t2_lim),
            "spe_limit_0_01": float(spe_lim),
            "alpha": 0.01,
        },
        "event_rates": {
            "train_raw_mortality_rate": float(train_raw["is_death_3yr"].mean()),
            "train_clean_mortality_rate": float(
                full_train_clean["is_death_3yr"].mean()
            ),
            "test_mortality_rate": float(full_test_df["is_death_3yr"].mean()),
        },
        "stratified_group_5fold": {
            "n_records": len(audit_df),
            "n_folds": 5,
            "fold_counts": {
                int(k): int(v)
                for k, v in full_train_clean["fold"].value_counts().items()
            },
            "csv_path": str(folds_csv_path.name),
        },
    }

    with open(audit_trail_path, "w", encoding="utf-8") as f:
        json.dump(audit_trail, f, indent=2)

    print(f"  [OK] Saved Cleaned Train Parquet (with fold): {train_parquet_path}")
    print(f"  [OK] Saved Uncurated Test Parquet (no fold): {test_parquet_path}")
    print(f"  [OK] Saved Continuous PCA Matrix: {cont_parquet_path}")
    print(f"  [OK] Saved Preprocessing Audit Trail: {audit_trail_path}")

    # 7. Generate Publication Figures (Single export, clean filenames)
    target_clean = full_train_clean["is_death_3yr"]
    scree_path = figures_dir / "scree_plot.png"
    scatter_path = figures_dir / "pca_scatter_death.png"
    biplot_path = figures_dir / "pca_loadings_biplot.png"
    t2_spe_path = figures_dir / "pca_t2_vs_spe_outliers.png"

    plot_scree(pca, save_path=scree_path, dpi=300)
    plot_pca_scatter(
        clean_scores,
        target_clean,
        pca=pca,
        save_path=scatter_path,
        dpi=300,
        target_name="3-Year Mortality",
    )
    plot_pca_loadings_biplot(
        pca,
        feature_names=list(cont_train_df.columns),
        top_n=10,
        save_path=biplot_path,
        dpi=300,
    )
    plot_t2_vs_spe(
        t2=t2_train,
        spe=spe_train,
        t2_limit=t2_lim,
        spe_limit=spe_lim,
        save_path=t2_spe_path,
        dpi=300,
    )

    print(f"  [OK] Scree plot:          {scree_path}")
    print(f"  [OK] PC1/PC2 scatter:     {scatter_path}")
    print(f"  [OK] Loadings biplot:     {biplot_path}")
    print(f"  [OK] T2 vs SPE outliers:  {t2_spe_path}")

    print("\n" + "=" * 75)
    print("[SUCCESS] Rolling 3-Year Preprocessing & PCA Pipeline Completed!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
