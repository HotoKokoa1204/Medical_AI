"""PCA dimensionality reduction and multivariate statistical analysis.

Provides reusable analysis utilities for extracting principal components,
calculating loadings, and summarizing clinical variance profiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(
    features: pd.DataFrame, n_components: int = 5
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Fit Principal Component Analysis (PCA) on continuous clinical features.

    Args:
        features: Preprocessed and standardized feature matrix.
        n_components: Number of principal components to estimate (default 5).

    Returns:
        A tuple of:
            - pca: Fitted sklearn PCA model instance.
            - scores_df: Transformed sample coordinates (PC1 to PCn).
            - loadings_df: Feature loading vectors of shape
              (n_features, n_components).
    """
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(features)

    component_names = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=component_names, index=features.index)
    loadings_df = pd.DataFrame(
        pca.components_.T, index=features.columns, columns=component_names
    )

    return pca, scores_df, loadings_df


def summarize_components(
    pca: PCA, loadings_df: pd.DataFrame, top_n: int = 5
) -> dict[str, Any]:
    """Extract explained variance ratios and prominent loading features for each PC.

    Args:
        pca: Fitted PCA instance.
        loadings_df: DataFrame of loadings (features x components).
        top_n: Number of leading positive and negative features to report.

    Returns:
        Dictionary containing individual variance, cumulative variance,
        and top loading indicators per principal component.
    """
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    summary: dict[str, Any] = {
        "individual_explained_variance": explained_var.tolist(),
        "cumulative_explained_variance": cumulative_var.tolist(),
        "components": {},
    }

    for col in loadings_df.columns:
        sorted_loadings = loadings_df[col].sort_values(ascending=False)
        top_pos = sorted_loadings.head(top_n).to_dict()
        top_neg = sorted_loadings.tail(top_n).to_dict()

        summary["components"][col] = {
            "top_positive_loadings": top_pos,
            "top_negative_loadings": top_neg,
        }

    return summary
