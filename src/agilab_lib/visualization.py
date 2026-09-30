"""Module: visualization
Stage: Library
Author: KafuuChino
Date: 2026-09-28
Description: Publication-grade visualization utilities for PCA and clinical data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "SimHei",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_scree(
    pca: PCA,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot Scree plot showing individual and cumulative explained variance.

    Args:
        pca: Fitted sklearn PCA model.
        save_path: Optional path to save the figure image.
        dpi: Resolution in dots per inch for saved figure (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    exp_var = pca.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(exp_var)
    n_comp = len(exp_var)
    comp_labels = [f"PC{i + 1}" for i in range(n_comp)]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)

    bars = ax1.bar(
        comp_labels,
        exp_var,
        color="#2b5c8f",
        alpha=0.85,
        edgecolor="#1a365d",
        width=0.55,
        label="Individual Explained Variance",
    )
    ax1.set_ylabel(
        "Individual Variance (%)",
        color="#1a365d",
        fontsize=11,
        fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor="#1a365d")
    ax1.set_ylim(0, max(exp_var) * 1.35)

    for bar, val in zip(bars, exp_var):
        height = bar.get_height()
        ax1.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        comp_labels,
        cum_var,
        color="#c53030",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Cumulative Explained Variance",
    )
    ax2.set_ylabel(
        "Cumulative Variance (%)",
        color="#c53030",
        fontsize=11,
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor="#c53030")
    ax2.set_ylim(0, 115)

    for i, txt in enumerate(cum_var):
        y_offset = 14 if i == 3 else 10
        x_offset = 14 if i == 3 else 0
        ax2.annotate(
            f"{txt:.1f}%",
            (comp_labels[i], cum_var[i]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#9b2c2c",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#c53030",
                "alpha": 0.9,
                "linewidth": 0.8,
            },
        )

    plt.title(
        "Scree Plot: Explained Variance by Principal Component",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def _draw_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    edgecolor: str = "none",
    facecolor: str = "none",
    alpha: float = 0.2,
) -> None:
    """Draw a 2D covariance confidence ellipse (95%) for a set of points."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = float(np.degrees(np.arctan2(*vecs[:, 0][::-1])))

    crit = float(np.sqrt(chi2.ppf(0.95, df=2)))
    width = float(2 * crit * np.sqrt(max(vals[0], 0.0)))
    height = float(2 * crit * np.sqrt(max(vals[1], 0.0)))

    ellipse = patches.Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linewidth=1.8,
        linestyle="--",
    )
    ax.add_patch(ellipse)


def plot_pca_scatter(
    scores: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    pca: PCA | None = None,
    save_path: Path | str | None = None,
    dpi: int = 300,
    target_name: str = "3-Year Mortality",
) -> plt.Figure:
    """Plot PC1 vs PC2 scatter plot colored by outcome label.

    Features 95% confidence ellipses, mean centroids, and variance labels.

    Args:
        scores: PCA projection coordinates (at least 2 components).
        target: Binary outcome labels (0 = Alive, 1 = Deceased).
        pca: Optional fitted PCA model to display variance percentages.
        save_path: Optional path to save the output image.
        dpi: Output image resolution (default 300).
        target_name: Name of target outcome for title and legend.

    Returns:
        matplotlib.pyplot.Figure object.
    """
    scores_arr = np.asarray(scores)
    target_arr = np.asarray(target).astype(int)

    pc1 = scores_arr[:, 0]
    pc2 = scores_arr[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=dpi)

    if pca is not None:
        var1_str = f" ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        var2_str = f" ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    else:
        var1_str = ""
        var2_str = ""

    alive_mask = target_arr == 0
    death_mask = target_arr == 1

    ax.scatter(
        pc1[alive_mask],
        pc2[alive_mask],
        c="#1e3d59",
        alpha=0.55,
        s=28,
        edgecolors="none",
        label=f"Survived / Alive (n={int(np.sum(alive_mask))})",
    )
    ax.scatter(
        pc1[death_mask],
        pc2[death_mask],
        c="#ff6e40",
        alpha=0.65,
        s=32,
        edgecolors="none",
        label=f"Deceased (n={int(np.sum(death_mask))})",
    )

    _draw_confidence_ellipse(
        pc1[alive_mask],
        pc2[alive_mask],
        ax,
        edgecolor="#17252a",
        facecolor="#1e3d59",
        alpha=0.15,
    )
    _draw_confidence_ellipse(
        pc1[death_mask],
        pc2[death_mask],
        ax,
        edgecolor="#b71c1c",
        facecolor="#ff6e40",
        alpha=0.18,
    )

    ax.scatter(
        np.mean(pc1[alive_mask]),
        np.mean(pc2[alive_mask]),
        color="#0d1b2a",
        marker="X",
        s=120,
        edgecolors="white",
        linewidth=1.5,
        label="Alive Centroid",
    )
    ax.scatter(
        np.mean(pc1[death_mask]),
        np.mean(pc2[death_mask]),
        color="#990000",
        marker="X",
        s=120,
        edgecolors="white",
        linewidth=1.5,
        label="Deceased Centroid",
    )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel(
        f"Principal Component 1{var1_str}\n(Uremic Toxin & Nutrition Axis)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Principal Component 2{var2_str}\n"
        "(Dialysis Adequacy & Kinetic Clearance Axis)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"Hemodialysis Cohort: PCA Space (Colored by {target_name})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_pca_loadings_biplot(
    pca: PCA,
    feature_names: Sequence[str],
    top_n: int = 10,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot PCA feature loading vectors and comparative bar charts for PC1 and PC2.

    Args:
        pca: Fitted PCA model.
        feature_names: List of continuous feature names matching PCA input order.
        top_n: Number of highest loading features to highlight per component.
        save_path: Optional path to save the output image.
        dpi: Output image resolution (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    loadings = pca.components_
    pc1_loadings = pd.Series(loadings[0], index=feature_names)
    pc2_loadings = pd.Series(loadings[1], index=feature_names)

    vector_magnitudes = np.sqrt(pc1_loadings**2 + pc2_loadings**2)
    top_features = vector_magnitudes.sort_values(ascending=False).head(top_n).index

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=dpi)

    circle = plt.Circle(
        (0, 0), 1.0, color="#cbd5e1", fill=False, linestyle="--", linewidth=1.2
    )
    ax1.add_patch(circle)

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))
    placed_positions: list[tuple[float, float]] = []

    for idx, feat in enumerate(top_features):
        x = pc1_loadings[feat]
        y = pc2_loadings[feat]
        ax1.arrow(
            0,
            0,
            x,
            y,
            head_width=0.015,
            head_length=0.02,
            fc=colors[idx],
            ec=colors[idx],
            linewidth=1.6,
            alpha=0.9,
        )

        tx = x * 1.15
        ty = y * 1.15

        for px, py in placed_positions:
            if np.hypot(tx - px, ty - py) < 0.05:
                ty += 0.038
                tx += 0.015

        placed_positions.append((tx, ty))

        ax1.text(
            tx,
            ty,
            feat,
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
            ha="left" if x >= 0 else "right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlim(-0.55, 0.55)
    ax1.set_ylim(-0.55, 0.55)
    ax1.set_xlabel(
        f"PC1 Loadings ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_ylabel(
        f"PC2 Loadings ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_title(
        "Top 10 Feature Loading Vectors (PC1 vs PC2)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_aspect("equal", "box")

    top_df = pd.DataFrame(
        {
            "PC1": pc1_loadings[top_features],
            "PC2": pc2_loadings[top_features],
        }
    ).iloc[::-1]

    y_pos = np.arange(len(top_features))
    bar_height = 0.38

    ax2.barh(
        y_pos - bar_height / 2,
        top_df["PC1"],
        height=bar_height,
        color="#2563eb",
        alpha=0.85,
        label="PC1 (Uremic/Nutrition)",
    )
    ax2.barh(
        y_pos + bar_height / 2,
        top_df["PC2"],
        height=bar_height,
        color="#10b981",
        alpha=0.85,
        label="PC2 (Adequacy/Kinetics)",
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_df.index, fontsize=9, fontweight="bold")
    ax2.set_xlabel("Loading Coefficient", fontsize=10, fontweight="bold")
    ax2.axvline(0, color="gray", linestyle="-", linewidth=0.8)
    ax2.set_title("Leading Feature Loadings Breakdown", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    plt.suptitle(
        "PCA Loadings and Feature Vectors (Kidit Hemodialysis Dynamic Cohort)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_t2_vs_spe(
    t2: np.ndarray | pd.Series,
    spe: np.ndarray | pd.Series,
    t2_limit: float,
    spe_limit: float,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot Hotelling's T^2 vs SPE outlier scatter plot with 99% control limits.

    Args:
        t2: Array or Series of Hotelling's T^2 values.
        spe: Array or Series of SPE (Q-statistic) values.
        t2_limit: Theoretical upper control limit for T^2.
        spe_limit: Theoretical upper control limit for SPE.
        save_path: Optional path to save the output figure image.
        dpi: Output image resolution (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    t2_arr = np.asarray(t2)
    spe_arr = np.asarray(spe)

    outliers = (t2_arr > t2_limit) | (spe_arr > spe_limit)
    inliers = ~outliers

    n_total = len(t2_arr)
    n_out = int(np.sum(outliers))
    pct_out = (n_out / n_total) * 100.0 if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=dpi)

    ax.scatter(
        t2_arr[inliers],
        spe_arr[inliers],
        c="#2b5c8f",
        alpha=0.55,
        s=26,
        edgecolors="none",
        label=f"Normal Inliers (n={int(np.sum(inliers))})",
    )
    ax.scatter(
        t2_arr[outliers],
        spe_arr[outliers],
        c="#e53e3e",
        alpha=0.75,
        s=34,
        edgecolors="#9b2c2c",
        linewidths=0.5,
        label=f"Statistical Outliers (n={n_out}, {pct_out:.2f}%)",
    )

    # Theoretical control limit dashed lines
    ax.axvline(
        t2_limit,
        color="#c53030",
        linestyle="--",
        linewidth=1.8,
        label=f"$T^2_{{0.01}}$ Limit ({t2_limit:.2f})",
    )
    ax.axhline(
        spe_limit,
        color="#805ad5",
        linestyle="--",
        linewidth=1.8,
        label=f"$Q_{{0.01}}$ Limit ({spe_limit:.2f})",
    )

    ax.set_xlabel(
        "Hotelling's $T^2$ (Score Space Mahalanobis Distance)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Squared Prediction Error (SPE / Q-Statistic Residual)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "Multivariate Outlier Diagnostics: $T^2$ vs SPE (99% Control Limits)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    info_box = (
        f"Significance Level: $\\alpha=0.01$ (99%)\n"
        f"$T^2$ Upper Limit: {t2_limit:.3f}\n"
        f"SPE Upper Limit: {spe_limit:.3f}\n"
        f"Pruned Outliers: {n_out} / {n_total} ({pct_out:.2f}%)"
    )
    ax.text(
        0.04,
        0.94,
        info_box,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.92,
        },
    )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_benchmark_roc_curves(
    y_true: Sequence[int] | np.ndarray,
    probabilities: dict[str, Sequence[float] | np.ndarray] | pd.DataFrame,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot overlaid ROC curves for benchmark models with AUC annotations.

    Includes 300 DPI publication styling and dashed random chance diagonal line.

    Args:
        y_true: True binary target array (0 = Alive, 1 = Deceased).
        probabilities: Dictionary or DataFrame mapping model names to predicted
            probabilities.
        save_path: Optional path to save the output figure image.
        dpi: Output image resolution (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    y_t = np.asarray(y_true, dtype=int)

    prob_dict: dict[str, np.ndarray] = {}
    if isinstance(probabilities, pd.DataFrame):
        for col in probabilities.columns:
            if col in ("is_death_3yr", "is_death"):
                continue
            if col.endswith("_prob"):
                name = col[:-5].replace("_", " ").title()
            elif "_pred" in col:
                continue
            else:
                name = col
            prob_dict[name] = np.asarray(probabilities[col], dtype=float)
    else:
        for name, probs in probabilities.items():
            prob_dict[name] = np.asarray(probs, dtype=float)

    # Compute ROC curves and AUCs
    curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_t, probs)
        roc_auc = float(auc(fpr, tpr))
        curves.append((name, fpr, tpr, roc_auc))

    # Sort descending by AUC
    curves.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=dpi)

    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(curves), 12)))

    for idx, (name, fpr, tpr, roc_auc) in enumerate(curves):
        color = cmap[idx % len(cmap)]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            linewidth=1.9,
            color=color,
            alpha=0.9,
        )

    # Random chance reference line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#64748b",
        linewidth=1.5,
        label="Random Chance (AUC = 0.500)",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        "False Positive Rate (1 - Specificity)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        "True Positive Rate (Sensitivity / Recall)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "11-Algorithm Benchmark: Receiver Operating Characteristic (ROC) Curves\n"
        "(Hemodialysis 3-Year Mortality Cohort)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_benchmark_pr_curves(
    y_true: Sequence[int] | np.ndarray,
    probabilities: dict[str, Sequence[float] | np.ndarray] | pd.DataFrame,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot overlaid Precision-Recall (PR) curves for benchmark models with PR-AUC.

    Includes horizontal baseline event rate line and 300 DPI publication styling.

    Args:
        y_true: True binary target array (0 = Alive, 1 = Deceased).
        probabilities: Dictionary or DataFrame mapping model names to predicted
            probabilities.
        save_path: Optional path to save the output figure image.
        dpi: Output image resolution (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    y_t = np.asarray(y_true, dtype=int)

    prob_dict: dict[str, np.ndarray] = {}
    if isinstance(probabilities, pd.DataFrame):
        for col in probabilities.columns:
            if col in ("is_death_3yr", "is_death"):
                continue
            if col.endswith("_prob"):
                name = col[:-5].replace("_", " ").title()
            elif "_pred" in col:
                continue
            else:
                name = col
            prob_dict[name] = np.asarray(probabilities[col], dtype=float)
    else:
        for name, probs in probabilities.items():
            prob_dict[name] = np.asarray(probs, dtype=float)

    # Compute PR curves and PR-AUCs
    curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for name, probs in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_t, probs)
        pr_auc = float(average_precision_score(y_t, probs))
        curves.append((name, precision, recall, pr_auc))

    # Sort descending by PR-AUC
    curves.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=dpi)

    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(curves), 12)))

    for idx, (name, precision, recall, pr_auc) in enumerate(curves):
        color = cmap[idx % len(cmap)]
        ax.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC = {pr_auc:.3f})",
            linewidth=1.9,
            color=color,
            alpha=0.9,
        )

    # Baseline event rate horizontal line
    baseline_rate = float(np.mean(y_t)) if len(y_t) > 0 else 0.0
    ax.axhline(
        baseline_rate,
        linestyle="--",
        color="#64748b",
        linewidth=1.5,
        label=f"Baseline Event Rate ({baseline_rate:.3f})",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        "Recall (Sensitivity)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Precision (Positive Predictive Value)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "11-Algorithm Benchmark: Precision-Recall (PR) Curves\n"
        "(Hemodialysis 3-Year Mortality Cohort)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_top_feature_importance(
    models: dict[str, Any],
    feature_names: Sequence[str],
    top_models: Sequence[str] | None = None,
    top_n: int = 15,
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot horizontal bar charts showing Top 15 feature importances for top models.

    Args:
        models: Dictionary mapping model names to fitted pipelines or estimators.
        feature_names: Names of features corresponding to the model inputs.
        top_models: Sequence of model names to plot. If None, selects up to 3 models
            having feature_importances_ or get_feature_importance().
        top_n: Number of leading features to display per model (default 15).
        save_path: Optional path to save the output figure image.
        dpi: Output image resolution (default 300).

    Returns:
        matplotlib.pyplot.Figure object.
    """
    candidate_names: list[str] = []
    if top_models is not None:
        candidate_names = [m for m in top_models if m in models]
    else:
        preferred_order = [
            "CatBoost",
            "XGBoost",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "Decision Tree",
            "Logistic Regression",
        ]
        for name in preferred_order:
            if name in models and len(candidate_names) < 3:
                candidate_names.append(name)
        if not candidate_names:
            candidate_names = list(models.keys())[:3]

    if not candidate_names:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.text(
            0.5,
            0.5,
            "No feature importance models provided.",
            ha="center",
            va="center",
        )
        return fig

    n_models = len(candidate_names)
    fig, axes = plt.subplots(
        1, n_models, figsize=(6.5 * n_models, 7.5), dpi=dpi, squeeze=False
    )
    axes_flat = axes.flatten()

    palette_colors = ["#2b5c8f", "#10b981", "#c53030", "#805ad5"]

    for idx, name in enumerate(candidate_names):
        ax = axes_flat[idx]
        model_obj = models[name]

        if hasattr(model_obj, "named_steps") and "clf" in model_obj.named_steps:
            clf = model_obj.named_steps["clf"]
        else:
            clf = model_obj

        if hasattr(clf, "feature_importances_"):
            importances = np.asarray(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "get_feature_importance"):
            importances = np.asarray(clf.get_feature_importance(), dtype=float)
        elif hasattr(clf, "coef_"):
            importances = np.abs(
                clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
            ).astype(float)
        else:
            importances = np.zeros(len(feature_names))

        total_imp = np.sum(importances)
        norm_imp = (importances / total_imp * 100.0) if total_imp > 0 else importances

        imp_series = pd.Series(norm_imp, index=list(feature_names)).sort_values(
            ascending=False
        )
        top_subset = imp_series.head(top_n).iloc[::-1]

        y_positions = np.arange(len(top_subset))
        bar_color = palette_colors[idx % len(palette_colors)]

        bars = ax.barh(
            y_positions,
            top_subset.values,
            color=bar_color,
            alpha=0.85,
            edgecolor="#1e293b",
            linewidth=0.6,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_subset.index, fontsize=9, fontweight="bold")
        ax.set_xlabel("Relative Importance (%)", fontsize=10, fontweight="bold")
        ax.set_title(f"{name} (Top {top_n})", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.1f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1e293b",
            )

        max_val = max(top_subset.values) if len(top_subset) > 0 else 1.0
        ax.set_xlim(0, max_val * 1.25)

    plt.suptitle(
        "Top Predictive Biomarkers & Features (Leading Benchmark Ensembles)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig
