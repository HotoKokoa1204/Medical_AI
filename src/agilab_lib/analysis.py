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

    When PCA retains k components out of p features, residual eigenvalues
    theta_1 = sum_{j=k+1}^p lambda_j, theta_2 = sum_{j=k+1}^p lambda_j^2,
    theta_3 = sum_{j=k+1}^p lambda_j^3.
    h_0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2^2).
    Q_alpha = theta_1 * (
        1 - (theta_2 * h_0 * (1 - h_0)) / theta_1^2
        + z_alpha * sqrt(2 * theta_2 * h_0^2) / theta_1
    ) ** (1 / h_0).

    Args:
        residual_eigenvalues: Eigenvalues of unmodeled residual dimensions.
        alpha: Significance level (default 0.01 for 99% confidence).

    Returns:
        Theoretical SPE threshold Q_alpha.
    """
    eigvals = np.asarray(residual_eigenvalues, dtype=float)
    th1 = float(np.sum(eigvals))
    th2 = float(np.sum(eigvals**2))
    th3 = float(np.sum(eigvals**3))

    h0 = 1.0 - (2.0 * th1 * th3) / (3.0 * (th2**2))
    z_alpha = float(norm.ppf(1.0 - alpha))

    term_cov = (th2 * h0 * (1.0 - h0)) / (th1**2)
    term_z = z_alpha * np.sqrt(2.0 * th2 * (h0**2)) / th1

    base = 1.0 - term_cov + term_z
    spe_lim = float(th1 * (max(base, 1e-12) ** (1.0 / h0)))
    return spe_lim


def prune_multivariate_outliers(
    scores: np.ndarray | pd.DataFrame,
    spe: np.ndarray | pd.Series,
    explained_variance: np.ndarray | Sequence[float] | None = None,
    t2_limit: float | None = None,
    spe_limit: float | None = None,
    alpha: float = 0.01,
    residual_eigenvalues: np.ndarray | Sequence[float] | None = None,
    t2: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, float, float]:
    """Identify multivariate outliers exceeding Hotelling's T^2 or SPE control limits.

    Operates purely on multivariate projection scores and SPE residuals to avoid
    feature envy with domain DataFrame structures.

    Args:
        scores: Projection scores array (n_samples, n_components) or precomputed T^2.
        spe: Array of Squared Prediction Error (Q-statistic) values (n_samples,).
        explained_variance: Explained variance of retained components (required if
            scores is 2D and t2 is None).
        t2_limit: Critical threshold for T^2. If None, derived via calculate_t2_limit.
        spe_limit: Critical threshold for SPE. If None, derived via calculate_spe_limit
            with residual_eigenvalues, or fallback to (1 - alpha) empirical percentile.
        alpha: Significance level (default 0.01 for 99% confidence).
        residual_eigenvalues: Optional residual eigenvalues (lambda_{k+1} to lambda_p)
            for genuine Jackson-Mudholkar calculation.
        t2: Optional precomputed Hotelling's T^2 statistics.

    Returns:
        A tuple of:
            - is_outlier: Boolean numpy array of shape (n_samples,) indicating outliers.
            - effective_t2_limit: Upper control limit applied for Hotelling's T^2.
            - effective_spe_limit: Upper control limit applied for SPE.
    """
    spe_arr = np.asarray(spe, dtype=float)
    n_samples = len(spe_arr)

    # Resolve T^2 statistic
    if t2 is not None:
        t2_arr = np.asarray(t2, dtype=float)
    elif hasattr(scores, "ndim") and scores.ndim == 1:
        t2_arr = np.asarray(scores, dtype=float)
    else:
        if explained_variance is None:
            raise ValueError(
                "explained_variance must be provided when scores is 2D and t2 is None."
            )
        t2_arr = calculate_hotelling_t2(scores, explained_variance)

    # Determine Hotelling's T^2 limit
    if t2_limit is not None:
        effective_t2 = float(t2_limit)
    else:
        n_comp = (
            scores.shape[1] if (hasattr(scores, "ndim") and scores.ndim == 2) else 5
        )
        effective_t2 = calculate_t2_limit(
            n_samples=n_samples, n_components=n_comp, alpha=alpha
        )

    # Determine SPE limit via Jackson-Mudholkar or explicit threshold
    if spe_limit is not None:
        effective_spe = float(spe_limit)
    elif residual_eigenvalues is not None:
        effective_spe = calculate_spe_limit(residual_eigenvalues, alpha=alpha)
    else:
        effective_spe = float(np.percentile(spe_arr, (1.0 - alpha) * 100.0))

    is_outlier = (t2_arr > effective_t2) | (spe_arr > effective_spe)
    return is_outlier, effective_t2, effective_spe
