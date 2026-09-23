"""End-to-end execution script for baseline preprocessing and PCA pipeline.

Reads Kidit_Master_Baseline_V2.xlsx, performs target leakage quarantine,
clinical sanity bounds checks, collinearity decoupling, median imputation,
log-transformation, and standardization. Fits PCA, generates publication-quality
plots, and exports clean Parquet feature sets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path to enable imports in all environments
sys_path_root = Path(__file__).resolve().parent.parent / "src"
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from agilab_lib.analysis import fit_pca, summarize_components  # noqa: E402
from agilab_lib.preprocessing import (  # noqa: E402
    build_feature_matrices,
    load_baseline_data,
)
from agilab_lib.visualization import (  # noqa: E402
    plot_pca_loadings_biplot,
    plot_pca_scatter,
    plot_scree,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Hemodialysis Baseline Preprocessing and PCA Pipeline."
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
    """Execute the end-to-end preprocessing, PCA fitting, and visualization.

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

    print("=" * 70)
    print("[PIPELINE] AGILAB MedicalAI: Baseline Preprocessing & PCA Pipeline")
    print("=" * 70)
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

    # 1. Load Data
    print("\n[Step 1/5] Loading baseline cohort...")
    raw_df = load_baseline_data(data_path)
    print(
        f"  [OK] Loaded cohort with {raw_df.shape[0]} patients and "
        f"{raw_df.shape[1]} raw variables."
    )

    # 2. Build Preprocessed Feature Matrices
    print("\n[Step 2/5] Performing quarantine, bounds & engineering...")
    continuous_pca_df, full_matrix_df, target = build_feature_matrices(raw_df)

    print(
        f"  [OK] Continuous PCA Matrix: {continuous_pca_df.shape[0]} patients "
        f"x {continuous_pca_df.shape[1]} features."
    )
    print(
        f"  [OK] Full Feature Matrix:   {full_matrix_df.shape[0]} patients "
        f"x {full_matrix_df.shape[1]} features."
    )
    print(
        f"  [OK] Mortality distribution: {target.value_counts().to_dict()} "
        f"(Deceased rate: {target.mean():.1%})"
    )

    # 3. Export Cleaned Parquet Datasets
    print("\n[Step 3/5] Saving Parquet feature datasets...")
    pca_parquet_path = output_dir / "continuous_features_pca.parquet"
    full_parquet_path = output_dir / "full_feature_matrix.parquet"

    continuous_pca_df.to_parquet(pca_parquet_path, engine="pyarrow", index=True)
    full_matrix_df.to_parquet(full_parquet_path, engine="pyarrow", index=True)

    root_processed = workspace_root / "data" / "processed"
    root_processed.mkdir(parents=True, exist_ok=True)
    continuous_pca_df.to_parquet(
        root_processed / "continuous_features_pca.parquet",
        engine="pyarrow",
        index=True,
    )
    full_matrix_df.to_parquet(
        root_processed / "full_feature_matrix.parquet",
        engine="pyarrow",
        index=True,
    )

    print(f"  [OK] Saved continuous PCA matrix: {pca_parquet_path}")
    print(f"  [OK] Saved full feature matrix:   {full_parquet_path}")

    # 4. Fit PCA Model
    print(f"\n[Step 4/5] Fitting PCA (n_components={args.n_components})...")
    pca, scores_df, loadings_df = fit_pca(
        continuous_pca_df, n_components=args.n_components
    )
    summary = summarize_components(pca, loadings_df, top_n=5)

    print("\n  [STAT] Explained Variance Breakdown:")
    for idx, (ind, cum) in enumerate(
        zip(
            summary["individual_explained_variance"],
            summary["cumulative_explained_variance"],
        )
    ):
        print(
            f"    PC{idx + 1}: Individual = {ind * 100:5.2f}% | "
            f"Cumulative = {cum * 100:5.2f}%"
        )

    print("\n  [ANALYSIS] Clinical Factor Interpretation:")
    print("    * PC1 (Uremic Toxin / Protein Nutrition & Muscle Mass Axis):")
    pc1_pos = summary["components"]["PC1"]["top_positive_loadings"].items()
    for feat, val in list(pc1_pos)[:4]:
        print(f"        (+) {feat:20s}: {val:+.4f}")
    pc1_neg = summary["components"]["PC1"]["top_negative_loadings"].items()
    for feat, val in list(pc1_neg)[:3]:
        print(f"        (-) {feat:20s}: {val:+.4f}")

    print("    * PC2 (Dialysis Adequacy & Urea Kinetic Clearance Axis):")
    pc2_pos = summary["components"]["PC2"]["top_positive_loadings"].items()
    for feat, val in list(pc2_pos)[:4]:
        print(f"        (+) {feat:20s}: {val:+.4f}")
    pc2_neg = summary["components"]["PC2"]["top_negative_loadings"].items()
    for feat, val in list(pc2_neg)[:3]:
        print(f"        (-) {feat:20s}: {val:+.4f}")

    # 5. Generate and Save Visualizations
    print("\n[Step 5/5] Generating publication-grade figures (300 DPI)...")
    for target_fig_dir in fig_dirs:
        scree_path = target_fig_dir / "scree_plot.png"
        scatter_path = target_fig_dir / "pca_scatter_death.png"
        biplot_path = target_fig_dir / "pca_loadings_biplot.png"

        plot_scree(pca, save_path=scree_path, dpi=300)
        plot_pca_scatter(scores_df, target, pca=pca, save_path=scatter_path, dpi=300)
        plot_pca_loadings_biplot(
            pca,
            feature_names=list(continuous_pca_df.columns),
            top_n=10,
            save_path=biplot_path,
            dpi=300,
        )

    print(f"  [OK] Scree plot:         {fig_dirs[0] / 'scree_plot.png'}")
    print(f"  [OK] PC1/PC2 scatter:    {fig_dirs[0] / 'pca_scatter_death.png'}")
    print(f"  [OK] Loadings biplot:    {fig_dirs[0] / 'pca_loadings_biplot.png'}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Issue 01 Preprocessing and PCA Pipeline completed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
