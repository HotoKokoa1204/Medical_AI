"""Module: modeling
Stage: Library
Author: KafuuChino
Date: 2026-09-30
Description: Benchmark suite and in-fold SMOTE-NC modeling for hemodialysis
    mortality prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Standard discrete count indicators from hemodialysis domain
DEFAULT_DISCRETE_COLUMNS: list[str] = [
    "HD_WS",
    "HD_ST",
    "ID_U",
    "MD_U",
    "CIC",
    "genAssess",
    "actAssess",
    "Comorb_Count",
]


def get_model_zoo(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Construct factory dictionary containing 11 configured benchmark classifiers.

    Algorithms included:
        1. Logistic Regression: LogisticRegression(
            max_iter=1000, random_state=42
        )
        2. KNN: KNeighborsClassifier(n_neighbors=7)
        3. GaussianNB: GaussianNB()
        4. Decision Tree: DecisionTreeClassifier(
            max_depth=6, random_state=42
        )
        5. Random Forest: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        )
        6. Extra Trees: ExtraTreesClassifier(
            n_estimators=200, max_depth=8, random_state=42
        )
        7. AdaBoost: AdaBoostClassifier(n_estimators=100, random_state=42)
        8. Gradient Boosting: GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
        )
        9. XGBoost: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )
        10. CatBoost: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42
        )
        11. SVM: SVC(probability=True, kernel="rbf", random_state=42)

    Args:
        random_state: Seed for pseudo-random number generators in stochastic models.

    Returns:
        Dictionary mapping model names to configured classifier estimator instances.
    """
    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=0,
            random_seed=random_state,
        ),
        "SVM": SVC(probability=True, kernel="rbf", random_state=random_state),
    }
    return models


def identify_categorical_features(
    df: pd.DataFrame,
    discrete_columns: Sequence[str] | None = None,
    max_unique_binary: int = 2,
) -> list[int]:
    """Identify column indices of categorical, discrete, and binary features.

    Examines DataFrame columns to locate binary flags (nunique <= 2), object/category
    columns, and known discrete count / schedule columns to configure SMOTE-NC.

    Args:
        df: Input DataFrame containing feature columns.
        discrete_columns: Optional sequence of explicit discrete column names.
            If None, checks against known DEFAULT_DISCRETE_COLUMNS.
        max_unique_binary: Maximum unique value count to classify as binary (default 2).

    Returns:
        List of 0-based integer column indices corresponding to categorical features.
    """
    disc_set = set(
        discrete_columns if discrete_columns is not None else DEFAULT_DISCRETE_COLUMNS
    )
    categorical_indices: list[int] = []

    for idx, col in enumerate(df.columns):
        if col in ("is_death_3yr", "is_death"):
            continue
        series = df[col]
        n_unique = series.nunique()
        is_obj = pd.api.types.is_object_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        )
        is_bin = n_unique <= max_unique_binary
        is_disc = col in disc_set

        if is_obj or is_bin or is_disc:
            categorical_indices.append(idx)

    return categorical_indices


def create_smote_pipeline(
    classifier: BaseEstimator,
    categorical_indices: Sequence[int],
    random_state: int = 42,
    n_features: int | None = None,
) -> ImbPipeline:
    """Build imblearn Pipeline chaining in-fold SMOTE-NC with the classifier.

    Args:
        classifier: Underlying classification model instance.
        categorical_indices: Integer column indices designated as categorical.
        random_state: Seed for reproducible synthetic oversampling.
        n_features: Total number of features (optional, for validation).

    Returns:
        ImbPipeline instance configuring SMOTE-NC and the classifier.
    """
    cat_list = list(categorical_indices)
    if n_features is not None and len(cat_list) >= n_features:
        # If all features are categorical, fallback to RandomOverSampler
        from imblearn.over_sampling import RandomOverSampler

        resampler: Any = RandomOverSampler(random_state=random_state)
    elif len(cat_list) == 0:
        # If no categorical features, use standard SMOTE
        resampler = SMOTE(random_state=random_state)
    else:
        resampler = SMOTENC(categorical_features=cat_list, random_state=random_state)

    pipeline = ImbPipeline([("smotenc", resampler), ("clf", classifier)])
    return pipeline


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Find decision threshold in [0.05, 0.95] maximizing F1 on OOF predictions.

    Args:
        y_true: True binary target array.
        y_prob: Predicted continuous probability array in [0, 1].
        thresholds: Candidate decision thresholds. Defaults to
            np.arange(0.05, 0.955, 0.01).

    Returns:
        Tuple of (optimal_threshold, best_f1_score).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    if y_p.ndim == 2:
        y_p = y_p[:, 1]

    if thresholds is None:
        candidate_thresholds = np.round(np.arange(0.05, 0.955, 0.01), 2)
    else:
        candidate_thresholds = np.asarray(thresholds, dtype=float)

    best_threshold: float = 0.50
    best_f1: float = -1.0

    for threshold in candidate_thresholds:
        t_val = float(threshold)
        pred = (y_p >= t_val).astype(int)
        score = float(f1_score(y_t, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = t_val

    return round(best_threshold, 2), round(best_f1, 4)


def calculate_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate 8 evaluation metrics for binary classification prognosis.

    Threshold-independent metrics:
        - ROC-AUC
        - PR-AUC (Average Precision)
        - Brier Score
    Discrete classification metrics:
        - Accuracy
        - Sensitivity (Recall)
        - Specificity (True Negative Rate)
        - F1-Score
        - Cohen's Kappa

    Args:
        y_true: True binary target labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Classification decision cutoff (default 0.5).

    Returns:
        Dictionary mapping metric names to computed floating-point values.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    if y_p.ndim == 2:
        y_p = y_p[:, 1]

    y_pred = (y_p >= threshold).astype(int)

    # Threshold-independent metrics
    try:
        roc_auc_val = float(roc_auc_score(y_t, y_p))
    except ValueError:
        roc_auc_val = 0.5

    try:
        pr_auc_val = float(average_precision_score(y_t, y_p))
    except ValueError:
        pr_auc_val = float(np.mean(y_t)) if len(y_t) > 0 else 0.0

    brier_val = float(brier_score_loss(y_t, y_p))

    # Discrete classification metrics
    acc_val = float(accuracy_score(y_t, y_pred))
    sens_val = float(recall_score(y_t, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    spec_val = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    f1_val = float(f1_score(y_t, y_pred, zero_division=0))

    try:
        kappa_val = float(cohen_kappa_score(y_t, y_pred))
        if np.isnan(kappa_val):
            kappa_val = 0.0
    except Exception:
        kappa_val = 0.0

    metrics: dict[str, float] = {
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "brier_score": brier_val,
        "accuracy": acc_val,
        "sensitivity": sens_val,
        "specificity": spec_val,
        "f1_score": f1_val,
        "cohen_kappa": kappa_val,
    }
    return metrics


def evaluate_model_cv_and_test(
    model_name: str,
    pipeline: ImbPipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    groups_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Execute 5-fold GroupKFold cross-validation, OOF tuning, and test inference.

    Args:
        model_name: Identifier name of the model.
        pipeline: ImbPipeline containing SMOTE-NC resampler and classifier.
        x_train: Training feature DataFrame.
        y_train: Training binary target series/array.
        groups_train: Patient grouping identifiers for training records.
        x_test: Uncurated test feature DataFrame.
        y_test: Test binary target series/array.
        n_splits: Number of cross-validation folds (default 5).

    Returns:
        Dictionary containing OOF probabilities, optimal threshold, test
        probabilities, fitted full pipeline, and metrics under both default
        and optimal thresholds.
    """
    y_tr = np.asarray(y_train, dtype=int)
    y_te = np.asarray(y_test, dtype=int)
    groups = np.asarray(groups_train)

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y_tr), dtype=float)

    # Fold-level training & OOF prediction collection
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(x_train, y_tr, groups=groups)
    ):
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients.intersection(val_patients)
        if len(overlap) > 0:
            raise RuntimeError(
                f"Subject leakage detected in fold {fold_idx}: "
                f"{len(overlap)} overlapping patients."
            )

        x_tr_fold = x_train.iloc[train_idx]
        y_tr_fold = y_tr[train_idx]
        x_val_fold = x_train.iloc[val_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(x_tr_fold, y_tr_fold)

        probs_val = fold_pipe.predict_proba(x_val_fold)
        oof_probs[val_idx] = probs_val[:, 1] if probs_val.ndim == 2 else probs_val

    if np.isnan(oof_probs).any():
        raise ValueError(
            f"NaN values encountered in OOF predictions for model {model_name}."
        )

    # Calibrate optimal decision threshold strictly on OOF predictions
    optimal_threshold, oof_f1 = find_optimal_threshold(y_tr, oof_probs)

    # Full refit on complete training set
    full_pipe = clone(pipeline)
    full_pipe.fit(x_train, y_tr)

    # Inference on uncurated test set
    test_probs_raw = full_pipe.predict_proba(x_test)
    test_probs = test_probs_raw[:, 1] if test_probs_raw.ndim == 2 else test_probs_raw

    if np.isnan(test_probs).any():
        raise ValueError(
            f"NaN values encountered in test predictions for model {model_name}."
        )

    # Generate dual binary decisions
    test_pred_default = (test_probs >= 0.50).astype(int)
    test_pred_optimal = (test_probs >= optimal_threshold).astype(int)

    # Evaluate metrics on test set
    metrics_default = calculate_metrics(y_te, test_probs, threshold=0.50)
    metrics_optimal = calculate_metrics(y_te, test_probs, threshold=optimal_threshold)

    result: dict[str, Any] = {
        "model_name": model_name,
        "optimal_threshold": optimal_threshold,
        "oof_f1": oof_f1,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "test_pred_default": test_pred_default,
        "test_pred_optimal": test_pred_optimal,
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "fitted_pipeline": full_pipe,
    }
    return result


def run_benchmark_suite(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    groups_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    models: dict[str, BaseEstimator] | None = None,
    categorical_indices: Sequence[int] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end benchmark across all models with in-fold SMOTE-NC.

    Args:
        x_train: Cleaned training feature matrix.
        y_train: Training binary mortality labels.
        groups_train: Patient grouping IDs for GroupKFold.
        x_test: Uncurated test feature matrix.
        y_test: Test binary mortality labels.
        models: Optional dictionary of models. If None, uses get_model_zoo().
        categorical_indices: Optional categorical column indices. If None,
            automatically detected.
        n_splits: Number of cross-validation folds (default 5).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - metrics_df: Comparative DataFrame across all models and metrics.
            - test_predictions_df: Test set probabilities and dual decisions.
            - oof_predictions_df: Training set OOF probabilities.
            - fitted_pipelines: Dict of trained full pipelines per model.
            - results: Detailed result dictionary per model.
    """
    if models is None:
        models = get_model_zoo(random_state=random_state)

    if categorical_indices is None:
        categorical_indices = identify_categorical_features(x_train)

    y_tr = np.asarray(y_train, dtype=int)
    y_te = np.asarray(y_test, dtype=int)

    summary_rows: list[dict[str, Any]] = []
    test_pred_dict: dict[str, np.ndarray] = {
        "is_death_3yr": y_te,
    }
    oof_pred_dict: dict[str, np.ndarray] = {
        "is_death_3yr": y_tr,
    }
    fitted_pipelines: dict[str, ImbPipeline] = {}
    detailed_results: dict[str, Any] = {}

    for name, estimator in models.items():
        logger.info(f"Training benchmark model: {name}")
        pipe = create_smote_pipeline(
            classifier=estimator,
            categorical_indices=categorical_indices,
            random_state=random_state,
            n_features=x_train.shape[1],
        )

        res = evaluate_model_cv_and_test(
            model_name=name,
            pipeline=pipe,
            x_train=x_train,
            y_train=y_tr,
            groups_train=groups_train,
            x_test=x_test,
            y_test=y_te,
            n_splits=n_splits,
        )

        clean_col = name.lower().replace(" ", "_")
        test_pred_dict[f"{clean_col}_prob"] = res["test_probs"]
        test_pred_dict[f"{clean_col}_pred_default"] = res["test_pred_default"]
        test_pred_dict[f"{clean_col}_pred_optimal"] = res["test_pred_optimal"]
        oof_pred_dict[f"{clean_col}_oof_prob"] = res["oof_probs"]

        fitted_pipelines[name] = res["fitted_pipeline"]
        detailed_results[name] = res

        m_def = res["metrics_default"]
        m_opt = res["metrics_optimal"]

        summary_rows.append(
            {
                "Model": name,
                "ROC_AUC": m_def["roc_auc"],
                "PR_AUC": m_def["pr_auc"],
                "Brier_Score": m_def["brier_score"],
                "Accuracy_Default": m_def["accuracy"],
                "Sensitivity_Default": m_def["sensitivity"],
                "Specificity_Default": m_def["specificity"],
                "F1_Default": m_def["f1_score"],
                "Kappa_Default": m_def["cohen_kappa"],
                "Optimal_Threshold": res["optimal_threshold"],
                "OOF_F1": res["oof_f1"],
                "Accuracy_Optimal": m_opt["accuracy"],
                "Sensitivity_Optimal": m_opt["sensitivity"],
                "Specificity_Optimal": m_opt["specificity"],
                "F1_Optimal": m_opt["f1_score"],
                "Kappa_Optimal": m_opt["cohen_kappa"],
            }
        )

    metrics_df = pd.DataFrame(summary_rows)
    test_predictions_df = pd.DataFrame(test_pred_dict, index=x_test.index)
    oof_predictions_df = pd.DataFrame(oof_pred_dict, index=x_train.index)

    return {
        "metrics_df": metrics_df,
        "test_predictions_df": test_predictions_df,
        "oof_predictions_df": oof_predictions_df,
        "fitted_pipelines": fitted_pipelines,
        "results": detailed_results,
    }
