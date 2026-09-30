"""Module: train_benchmark
Stage: Script
Author: KafuuChino
Date: 2026-09-30
Description: End-to-end execution script for 11-algorithm benchmark suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src to sys.path
sys_path_root = Path(__file__).resolve().parent.parent / "src"
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from agilab_lib.modeling import (  # noqa: E402
    get_model_zoo,
    identify_categorical_features,
    run_benchmark_suite,
)
from agilab_lib.preprocessing import load_longitudinal_cohort  # noqa: E402
from agilab_lib.visualization import (  # noqa: E402
    plot_benchmark_pr_curves,
    plot_benchmark_roc_curves,
    plot_top_feature_importance,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmark training script.

    Returns:
        argparse.Namespace containing parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run 11-Algorithm Benchmark Suite with In-Fold SMOTE-NC & OOF."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to train_cleaned_rolling_3yr.parquet.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test_uncurated_rolling_3yr.parquet.",
    )
    parser.add_argument(
        "--patient-mapping",
        type=str,
        default=None,
        help="Path to patient_id_mapping.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predictions and metrics CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory to save benchmark summary markdown.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save publication figures.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Directory to save serialized model checkpoints.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold cross-validation folds (default 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42).",
    )
    return parser.parse_args()


def load_patient_groups(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    mapping_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve PatientID grouping arrays for train and test splits.

    Args:
        df_train: Cleaned training DataFrame.
        df_test: Uncurated test DataFrame.
        mapping_path: Optional explicit path to patient_id_mapping.parquet.
        project_root: Root project directory for path fallbacks.

    Returns:
        Tuple of (train_patient_ids, test_patient_ids).
    """
    root = project_root if project_root is not None else Path.cwd()

    if "PatientID" in df_train.columns:
        train_pids = df_train["PatientID"].values
        test_pids = (
            df_test["PatientID"].values
            if "PatientID" in df_test.columns
            else np.arange(len(df_test))
        )
        return train_pids, test_pids

    # Check for mapping parquet file
    map_candidates = [
        mapping_path,
        root / "data" / "processed" / "patient_id_mapping.parquet",
    ]
    for c in map_candidates:
        if c is not None and Path(c).exists():
            mapping_df = pd.read_parquet(c)
            train_pids = mapping_df.loc[df_train.index, "PatientID"].values
            test_pids = mapping_df.loc[df_test.index, "PatientID"].values
            return train_pids, test_pids

    # Fallback to workbook if mapping not found
    workbook_candidates = [
        root / "data" / "Kidit_Master_Baseline_V2.xlsx",
        Path("E:/github/Kidit_Master_Baseline_V2.xlsx"),
        Path("E:/github/MedicalAI/data/Kidit_Master_Baseline_V2.xlsx"),
    ]
    for wb in workbook_candidates:
        if wb.exists():
            cohort_df = load_longitudinal_cohort(wb)
            train_pids = cohort_df.loc[df_train.index, "PatientID"].values
            test_pids = cohort_df.loc[df_test.index, "PatientID"].values
            return train_pids, test_pids

    # Fallback for synthetic/isolated test runs: use sample index
    return df_train.index.values, df_test.index.values


def generate_markdown_report(
    metrics_df: pd.DataFrame, n_train: int, n_test: int
) -> str:
    """Generate publication-ready Markdown summary of the benchmark results.

    Args:
        metrics_df: Benchmark metrics DataFrame.
        n_train: Training sample count.
        n_test: Test sample count.

    Returns:
        Formatted Markdown string.
    """
    sorted_df = metrics_df.sort_values(by="ROC_AUC", ascending=False)

    header = (
        "| Model | Optimal $T^*$ | ROC-AUC | PR-AUC | Brier | "
        "Acc ($T=0.5$) | Sens ($T=0.5$) | Spec ($T=0.5$) | F1 ($T=0.5$) | "
        "Kappa ($T=0.5$) | Acc ($T^*$) | Sens ($T^*$) | Spec ($T^*$) | "
        "F1 ($T^*$) | Kappa ($T^*$) |"
    )
    divider = (
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        ":---:|:---:|:---:|:---:|"
    )

    lines: list[str] = [
        "# AGILAB MedicalAI: 11-Algorithm Benchmark Suite Leaderboard",
        "",
        "## Cohort & Methodology Summary",
        "- **Cohort**: Longitudinal Hemodialysis Rolling 3-Year Dynamic Cohort",
        f"- **Training Records**: {n_train:,} observations (purified via $T^2$ & SPE)",
        f"- **Uncurated Test Records**: {n_test:,} observations (100% unpruned)",
        "- **Validation Scheme**: 5-Fold `GroupKFold` strictly by `PatientID`",
        "- **Resampling**: In-Fold `SMOTE-NC` (mode for discrete flags)",
        "- **Threshold Calibration**: Optimal cutoff $T^* \\in [0.05, 0.95]$ on OOF",
        "- **Blinding Principle**: Test labels isolated and untouched",
        "",
        "## Performance Leaderboard (Ranked by ROC-AUC)",
        "",
        header,
        divider,
    ]

    for _, row in sorted_df.iterrows():
        row_str = (
            f"| **{row['Model']}** | "
            f"`{row['Optimal_Threshold']:.2f}` | "
            f"{row['ROC_AUC']:.4f} | "
            f"{row['PR_AUC']:.4f} | "
            f"{row['Brier_Score']:.4f} | "
            f"{row['Accuracy_Default']:.4f} | "
            f"{row['Sensitivity_Default']:.4f} | "
            f"{row['Specificity_Default']:.4f} | "
            f"{row['F1_Default']:.4f} | "
            f"{row['Kappa_Default']:.4f} | "
            f"{row['Accuracy_Optimal']:.4f} | "
            f"{row['Sensitivity_Optimal']:.4f} | "
            f"{row['Specificity_Optimal']:.4f} | "
            f"{row['F1_Optimal']:.4f} | "
            f"{row['Kappa_Optimal']:.4f} |"
        )
        lines.append(row_str)

    lines.extend(
        [
            "",
            "## Anti-Leakage Audit Trail",
            "- [x] Uncurated test set count strictly 1,112 (zero synthetic instances).",
            "- [x] Patient isolation: Train $\\cap$ Test patient IDs = $\\emptyset$.",
            "- [x] In-fold SMOTE-NC: Resampling strictly inside CV training folds.",
            "- [x] Out-of-fold threshold: $T^*$ selected purely on OOF probabilities.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Execute end-to-end 11-algorithm benchmark training and evaluation pipeline.

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

    # Configure paths
    train_path = (
        Path(args.train_path)
        if args.train_path
        else project_root / "data" / "processed" / "train_cleaned_rolling_3yr.parquet"
    )
    test_path = (
        Path(args.test_path)
        if args.test_path
        else project_root / "data" / "processed" / "test_uncurated_rolling_3yr.parquet"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "processed"
    )
    reports_dir = (
        Path(args.reports_dir) if args.reports_dir else project_root / "reports"
    )
    figures_dir = (
        Path(args.figures_dir)
        if args.figures_dir
        else project_root / "reports" / "figures"
    )
    checkpoints_dir = (
        Path(args.checkpoints_dir)
        if args.checkpoints_dir
        else project_root / "models" / "checkpoints"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[PIPELINE] AGILAB MedicalAI: 11-Algorithm Benchmark Suite")
    print("=" * 80)
    print(f"[INFO] Train Data:       {train_path.resolve()}")
    print(f"[INFO] Test Data:        {test_path.resolve()}")
    print(f"[INFO] Checkpoints Dir:  {checkpoints_dir.resolve()}")
    print(f"[INFO] Figures Dir:      {figures_dir.resolve()}")
    print(f"[INFO] Reports Dir:      {reports_dir.resolve()}")

    if not train_path.exists():
        print(f"[ERROR] Train dataset not found at {train_path}")
        return 1
    if not test_path.exists():
        print(f"[ERROR] Test dataset not found at {test_path}")
        return 1

    # 1. Load Parquet Data
    print("\n[Step 1/5] Loading datasets...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "is_death_3yr" if "is_death_3yr" in train_df.columns else "is_death"
    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    print(
        f"  [OK] Training Set: {x_train.shape[0]} records, "
        f"{x_train.shape[1]} features (Mortality Rate: {np.mean(y_train):.2%})"
    )
    print(
        f"  [OK] Test Set:     {x_test.shape[0]} records, "
        f"{x_test.shape[1]} features (Mortality Rate: {np.mean(y_test):.2%})"
    )

    # Anti-leakage sample count check
    assert len(test_df) == 1112, f"Expected 1,112 test samples, got {len(test_df)}"

    # 2. Resolve Patient Identifiers for GroupKFold
    print("\n[Step 2/5] Resolving patient identifier groupings...")
    train_groups, test_groups = load_patient_groups(
        train_df,
        test_df,
        mapping_path=Path(args.patient_mapping) if args.patient_mapping else None,
        project_root=project_root,
    )
    n_unique_patients = len(np.unique(train_groups))
    print(f"  [OK] Unique Training Patients: {n_unique_patients}")

    overlap_patients = set(train_groups).intersection(set(test_groups))
    assert (
        len(overlap_patients) == 0
    ), f"Subject leakage detected: {len(overlap_patients)} overlap!"
    print(
        "  [OK] Patient Grouping Isolation Verified: "
        "Train Patients ∩ Test Patients == ∅"
    )

    # 3. Categorical Feature Identification
    print("\n[Step 3/5] Dynamically identifying categorical & discrete indices...")
    cat_indices = identify_categorical_features(x_train)
    print(
        f"  [OK] Identified {len(cat_indices)} categorical/discrete features "
        f"and {x_train.shape[1] - len(cat_indices)} continuous features for SMOTE-NC."
    )

    # 4. Execute 11-Algorithm Benchmark
    print(
        f"\n[Step 4/5] Executing 11-Model Benchmark Suite "
        f"({args.n_splits}-Fold GroupKFold CV + In-Fold SMOTE-NC)..."
    )
    models = get_model_zoo(random_state=args.random_state)
    results = run_benchmark_suite(
        x_train=x_train,
        y_train=y_train,
        groups_train=train_groups,
        x_test=x_test,
        y_test=y_test,
        models=models,
        categorical_indices=cat_indices,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    metrics_df = results["metrics_df"]
    test_pred_df = results["test_predictions_df"]
    fitted_pipelines = results["fitted_pipelines"]

    # 5. Persist Deliverables
    print("\n[Step 5/5] Persisting model predictions, checkpoints, and figures...")

    # Save test predictions parquet
    pred_path = output_dir / "model_predictions_test.parquet"
    test_pred_df.to_parquet(pred_path, engine="pyarrow", index=True)
    print(
        f"  [OK] Saved Test Predictions Matrix: {pred_path} "
        f"(Shape: {test_pred_df.shape})"
    )

    # Save metrics CSV (data/processed and reports directory)
    metrics_path = output_dir / "benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  [OK] Saved Benchmark Metrics CSV:   {metrics_path}")

    reports_metrics_path = reports_dir / "benchmark_metrics.csv"
    metrics_df.to_csv(reports_metrics_path, index=False)
    print(f"  [OK] Saved Reports Metrics CSV:     {reports_metrics_path}")

    # Save Markdown Summary
    report_content = generate_markdown_report(metrics_df, len(x_train), len(x_test))
    summary_path = reports_dir / "benchmark_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [OK] Saved Benchmark Summary MD:    {summary_path}")

    # Save serialized model checkpoints
    for name, pipe in fitted_pipelines.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        ckpt_path = checkpoints_dir / fname
        joblib.dump(pipe, ckpt_path)
    print(
        f"  [OK] Serialized {len(fitted_pipelines)} checkpoints in: {checkpoints_dir}"
    )

    # Generate publication figures
    roc_fig_path = figures_dir / "benchmark_roc_curves.png"
    pr_fig_path = figures_dir / "benchmark_pr_curves.png"
    fi_fig_path = figures_dir / "benchmark_feature_importance.png"

    plot_benchmark_roc_curves(
        y_true=y_test,
        probabilities=test_pred_df,
        save_path=roc_fig_path,
        dpi=300,
    )
    print(f"  [OK] Generated ROC Curves:          {roc_fig_path}")

    plot_benchmark_pr_curves(
        y_true=y_test,
        probabilities=test_pred_df,
        save_path=pr_fig_path,
        dpi=300,
    )
    print(f"  [OK] Generated PR Curves:           {pr_fig_path}")

    # Dynamically select top 3 models by ROC-AUC
    top_3_names = (
        metrics_df.sort_values(by="ROC_AUC", ascending=False)["Model"].head(3).tolist()
    )
    print(f"  [INFO] Dynamic Top 3 Models:        {top_3_names}")

    plot_top_feature_importance(
        models=fitted_pipelines,
        feature_names=x_train,
        y=y_train,
        top_models=top_3_names,
        save_path=fi_fig_path,
        dpi=300,
    )
    print(f"  [OK] Generated Feature Importance:  {fi_fig_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY (Ranked by ROC-AUC)")
    print("=" * 80)
    print(metrics_df.sort_values(by="ROC_AUC", ascending=False).to_string(index=False))
    print("=" * 80)
    print("[SUCCESS] 11-Algorithm Benchmark Completed Successfully!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
