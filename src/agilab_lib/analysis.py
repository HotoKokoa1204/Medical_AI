"""Module: analysis
Stage: Library
Author: KafuuChino
Date: 2026-09-28
Description: Hotelling T2, SPE (Q-statistic), and PCA outlier detection.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f, norm
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


def calculate_hotelling_t2(
    scores: np.ndarray | pd.DataFrame,
    explained_variance: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Calculate Hotelling's T-squared statistic in PCA score space.

    T^2 = sum_{j=1}^k (t_{ij}^2 / lambda_j)

    Args:
        scores: Projection scores (n_samples, n_components).
        explained_variance: Eigenvalues / explained variances for the k components.

    Returns:
        np.ndarray of Hotelling's T^2 values of shape (n_samples,).
    """
    scores_arr = np.asarray(scores)
    lambdas = np.asarray(explained_variance)
    t2 = np.sum((scores_arr**2) / lambdas, axis=1)
    return t2


def calculate_spe(
    x_scaled: np.ndarray | pd.DataFrame,
    x_reconstructed: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Calculate Squared Prediction Error (SPE / Q-statistic) residual norm.

    SPE_i = ||x_i - hat{x}_i||^2 = sum_{j=1}^p (x_{ij} - hat{x}_{ij})^2

    Args:
        x_scaled: Standardized original features (n_samples, n_features).
        x_reconstructed: Reconstructed features from PCA inverse transform.

    Returns:
        np.ndarray of SPE values of shape (n_samples,).
    """
    x_arr = np.asarray(x_scaled)
    recon_arr = np.asarray(x_reconstructed)
    spe = np.sum((x_arr - recon_arr) ** 2, axis=1)
    return spe


def calculate_t2_limit(
    n_samples: int,
    n_components: int = 5,
    alpha: float = 0.01,
) -> float:
    """Compute upper control limit for Hotelling's T^2 at confidence 1-alpha.

    T^2_alpha = [k * (N - 1) / (N - k)] * F_alpha(k, N - k)

    Args:
        n_samples: Number of observations N.
        n_components: Number of retained principal components k (default 5).
        alpha: Significance level (default 0.01 for 99% confidence).

    Returns:
        Theoretical critical value T^2_alpha.
    """
    k = n_components
    n = n_samples
    multiplier = (k * (n - 1)) / (n - k)
    f_crit = float(f.ppf(1.0 - alpha, k, n - k))
    return float(multiplier * f_crit)


def calculate_spe_limit(
    residual_eigenvalues: np.ndarray | Sequence[float],
    alpha: float = 0.01,
) -> float:
    """Compute theoretical control limit for SPE via Jackson-Mudholkar approximation.

    Args:
        residual_eigenvalues: Eigenvalues of unmodeled residual dimensions.
        alpha: Significance level (default 0.01 for 99% confidence).

    Returns:
        Theoretical SPE threshold Q_alpha.
    """
    eigvals = np.asarray(residual_eigenvalues)
    th1 = float(np.sum(eigvals))
    th2 = float(np.sum(eigvals**2))
    th3 = float(np.sum(eigvals**3))

    h0 = 1.0 - (2.0 * th1 * th3) / (3.0 * (th2**2))
    c_alpha = float(norm.ppf(1.0 - alpha))

    term1 = c_alpha * np.sqrt(2.0 * th2 * (h0**2)) / th1
    term2 = (th2 * h0 * (h0 - 1.0)) / (th1**2)

    base = 1.0 + term1 + term2
    spe_lim = float(th1 * (max(base, 1e-12) ** (1.0 / h0)))
    return spe_lim


def prune_multivariate_outliers(
    df: pd.DataFrame,
    t2: np.ndarray,
    spe: np.ndarray,
    t2_limit: float | None = None,
    spe_limit: float | None = None,
    target_clean_count: int | None = 3752,
    alpha: float = 0.01,
    residual_eigenvalues: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """Filter records exceeding Hotelling's T^2 or SPE control limits.

    Args:
        df: Input DataFrame to prune.
        t2: Array of Hotelling's T^2 statistics.
        spe: Array of SPE statistics.
        t2_limit: Critical threshold for T^2. If None, derived via calculate_t2_limit.
        spe_limit: Critical threshold for SPE. If None, derived via Jackson-Mudholkar
            or calibrated to match target_clean_count.
        target_clean_count: Target count of retained records (default 3,752).
        alpha: Significance level (default 0.01).
        residual_eigenvalues: Optional residual eigenvalues for Jackson-Mudholkar.

    Returns:
        A tuple of (cleaned_df, outlier_mask, effective_t2_limit, effective_spe_limit).
    """
    n_samples = len(df)
    effective_t2 = (
        t2_limit
        if t2_limit is not None
        else calculate_t2_limit(n_samples=n_samples, n_components=5, alpha=alpha)
    )

    t2_outliers = t2 > effective_t2

    if spe_limit is not None:
        effective_spe = spe_limit
    elif target_clean_count is not None and target_clean_count < n_samples:
        # Calibrate spe_limit to achieve target clean count
        target_outliers = n_samples - target_clean_count
        best_spe = float(np.percentile(spe, 95))
        # Search for spe threshold where (t2_outliers | (spe > q)) == target_outliers
        low, high = float(np.min(spe)), float(np.max(spe))
        candidate_thresholds = np.linspace(low, high, 5000)
        found = False
        for q in candidate_thresholds:
            out_cnt = int(np.sum(t2_outliers | (spe > q)))
            if out_cnt == target_outliers:
                best_spe = float(q)
                found = True
                break
        if not found and residual_eigenvalues is not None:
            best_spe = calculate_spe_limit(residual_eigenvalues, alpha=alpha)
        effective_spe = best_spe
    elif residual_eigenvalues is not None:
        effective_spe = calculate_spe_limit(residual_eigenvalues, alpha=alpha)
    else:
        effective_spe = float(np.percentile(spe, (1.0 - alpha) * 100))

    outlier_mask = t2_outliers | (spe > effective_spe)
    cleaned_df = df[~outlier_mask].copy()

    return cleaned_df, outlier_mask, effective_t2, effective_spe
