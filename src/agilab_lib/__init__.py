"""AGILAB MedicalAI Core Library for Hemodialysis Data Science and Prognosis."""

from agilab_lib.analysis import fit_pca, summarize_components
from agilab_lib.preprocessing import (
    build_feature_matrices,
    clean_clinical_bounds,
    encode_categorical_features,
    engineer_features,
    load_baseline_data,
    process_continuous_features,
    quarantine_target_leakage,
)
from agilab_lib.visualization import (
    plot_pca_loadings_biplot,
    plot_pca_scatter,
    plot_scree,
)

__all__ = [
    "load_baseline_data",
    "quarantine_target_leakage",
    "clean_clinical_bounds",
    "engineer_features",
    "process_continuous_features",
    "encode_categorical_features",
    "build_feature_matrices",
    "fit_pca",
    "summarize_components",
    "plot_scree",
    "plot_pca_scatter",
    "plot_pca_loadings_biplot",
]
