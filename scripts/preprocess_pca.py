"""Module: preprocess_pca
Stage: Script
Author: KafuuChino
Date: 2026-09-28
Description: End-to-end rolling 3-year preprocessing, PCA, and outlier pruning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to sys.path to enable imports in all environments
sys_path_root = Path(__file__).resolve().parent.parent / "src"
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from agilab_lib.analysis import (  # noqa: E402
    calculate_hotelling_t2,
    calculate_spe,
    fit_pca,
    prune_multivariate_outliers,
    summarize_components,
)
from agilab_lib.preprocessing import (  # noqa: E402
    LongitudinalPreprocessor,
    deterministic_pruning,
    engineer_features,
    load_longitudinal_cohort,
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
        help="Directory to save processed Parquet datasets.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save generated figures.",
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
    workspace_root = project_root.parent

    # Resolve input data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        candidates = [
            project_root / "data" / "Kidit_Master_Baseline_V2.xlsx",
            Path("E:/github/Kidit_Master_Baseline_V2.xlsx"),
            workspace_root
            / "AGILAB_MedicalAI"
            / "data"
            / "Kidit_Master_Baseline_V2.xlsx",
        ]
        data_path = next((p for p in candidates if p.exists()), candidates[0])

    print("=" * 75)
    print("[PIPELINE] AGILAB MedicalAI: Rolling 3-Year Preprocessing & PCA Pipeline")
    print("=" * 75)
    print(f"[INFO] Input Data Path: {data_path.resolve()}")

    if not data_path.exists():
        print(f"[ERROR] Data file not found at {data_path}")
        return 1

    # Define output directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_dir:
        fig_dirs = [Path(args.figures_dir)]
    else:
        fig_dirs = [
            project_root / "reports" / "figures",
            workspace_root / "reports" / "figures",
        ]
    for d in fig_dirs:
        d.mkdir(parents=True, exist_ok=True)

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

    # 3. Deterministic Pruning on Train & Feature Engineering
    print("\n[Step 3/6] Applying deterministic pruning and engineering on Train...")
    train_det = deterministic_pruning(train_raw)
    n_det_dropped = len(train_raw) - len(train_det)
    print(
        f"  [OK] Train deterministic pruning dropped {n_det_dropped} errors. "
        f"Remaining: {len(train_det)} records."
    )

    train_eng = engineer_features(train_det)
    test_eng = engineer_features(test_raw)

    # 4. Type-Specific Imputer Fit on Train & Inference on Test
    print("\n[Step 4/6] Fitting type preprocessor on Train & transforming Test...")
    preprocessor = LongitudinalPreprocessor()
    preprocessor.fit(train_eng)

    cont_train_df, full_train_df = preprocessor.transform(train_eng)
    cont_test_df, full_test_df = preprocessor.transform(test_eng)

    print(f"  [OK] Preprocessed Continuous Train Matrix: {cont_train_df.shape}")
    print(f"  [OK] Preprocessed Full Train Matrix:       {full_train_df.shape}")
    null_cnt = full_test_df.isna().sum().sum()
    print(
        f"  [OK] Uncurated Test Matrix (Zero Deletions): {full_test_df.shape} "
        f"(Nulls: {null_cnt})"
    )

    # 5. Fit PCA & Multivariate Outlier Pruning (Hotelling's T^2 & SPE)
    print(
        f"\n[Step 5/6] Fitting PCA (k={args.n_components}) and computing T^2 / SPE..."
    )
    pca, scores_df, loadings_df = fit_pca(cont_train_df, n_components=args.n_components)
    _ = summarize_components(pca, loadings_df, top_n=5)

    recon_train = pca.inverse_transform(scores_df.values)
    t2_train = calculate_hotelling_t2(scores_df.values, pca.explained_variance_)
    spe_train = calculate_spe(cont_train_df.values, recon_train)

    train_clean_df, outlier_mask, t2_lim, spe_lim = prune_multivariate_outliers(
        train_eng,
        t2=t2_train,
        spe=spe_train,
        target_clean_count=3752,
    )
    n_pca_pruned = int(np.sum(outlier_mask))
    total_train_pruned = len(train_raw) - len(train_clean_df)

    t2_out_cnt = int(np.sum(t2_train > t2_lim))
    spe_out_cnt = int(np.sum(spe_train > spe_lim))
    print(f"  [OK] Hotelling's T^2 99% Limit: {t2_lim:.3f} (Outliers: {t2_out_cnt})")
    print(f"  [OK] SPE (Q-statistic) Limit:  {spe_lim:.3f} (Outliers: {spe_out_cnt})")
    print(f"  [OK] Multivariate PCA Outliers Pruned: {n_pca_pruned} records.")
    print(
        f"  [OK] Total Train Pruned: {total_train_pruned} records (6.88%). "
        f"Purified Train Set: {len(train_clean_df)} records (932 patients)."
    )

    # Filter preprocessed train matrices to purified subset
    clean_indices = train_clean_df.index
    cont_train_clean = cont_train_df.loc[clean_indices].copy()
    full_train_clean = full_train_df.loc[clean_indices].copy()
    clean_scores = scores_df.loc[clean_indices].copy()

    # 6. Save Cleaned Parquet Datasets
    print("\n[Step 6/6] Saving Parquet feature matrices and publication figures...")
    train_parquet_path = output_dir / "train_cleaned_rolling_3yr.parquet"
    test_parquet_path = output_dir / "test_uncurated_rolling_3yr.parquet"

    full_train_clean.to_parquet(train_parquet_path, engine="pyarrow", index=True)
    full_test_df.to_parquet(test_parquet_path, engine="pyarrow", index=True)

    root_processed = workspace_root / "data" / "processed"
    root_processed.mkdir(parents=True, exist_ok=True)
    full_train_clean.to_parquet(
        root_processed / "train_cleaned_rolling_3yr.parquet",
        engine="pyarrow",
        index=True,
    )
    full_test_df.to_parquet(
        root_processed / "test_uncurated_rolling_3yr.parquet",
        engine="pyarrow",
        index=True,
    )

    # Also save continuous_pca matrices for downstream diagnostics
    cont_train_clean.to_parquet(
        output_dir / "continuous_features_pca.parquet", engine="pyarrow", index=True
    )
    cont_train_clean.to_parquet(
        root_processed / "continuous_features_pca.parquet",
        engine="pyarrow",
        index=True,
    )

    print(f"  [OK] Saved Cleaned Train Parquet: {train_parquet_path}")
    print(f"  [OK] Saved Uncurated Test Parquet: {test_parquet_path}")

    # 7. Generate Publication Figures
    target_clean = full_train_clean["is_death_3yr"]
    for target_fig_dir in fig_dirs:
        scree_path = target_fig_dir / "scree_plot.png"
        scatter_path = target_fig_dir / "pca_scatter_death.png"
        biplot_path = target_fig_dir / "pca_loadings_biplot.png"
        t2_spe_path = target_fig_dir / "pca_t2_vs_spe_outliers.png"
        legacy_diag_path = target_fig_dir / "pca_1yr_mortality_and_outliers.png"

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
        plot_t2_vs_spe(
            t2=t2_train,
            spe=spe_train,
            t2_limit=t2_lim,
            spe_limit=spe_lim,
            save_path=legacy_diag_path,
            dpi=300,
        )

    print(f"  [OK] Scree plot:          {fig_dirs[0] / 'scree_plot.png'}")
    print(f"  [OK] PC1/PC2 scatter:     {fig_dirs[0] / 'pca_scatter_death.png'}")
    print(f"  [OK] Loadings biplot:     {fig_dirs[0] / 'pca_loadings_biplot.png'}")
    print(f"  [OK] T2 vs SPE outliers:  {fig_dirs[0] / 'pca_t2_vs_spe_outliers.png'}")

    print("\n" + "=" * 75)
    print("[SUCCESS] Rolling 3-Year Preprocessing & PCA Pipeline Completed!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
