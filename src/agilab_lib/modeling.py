"""Module: modeling
Stage: Library
Author: KafuuChino
Date: 2026-09-30
Description: Benchmark suite and in-fold SMOTE-NC modeling for hemodialysis
    mortality prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, TypedDict

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

# Metadata and target columns strictly quarantined from model feature matrices
QUARANTINE_COLUMNS: tuple[str, ...] = (
    "is_death_3yr",
    "is_death",
    "PatientID",
    "fold",
)


def quarantine_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop quarantined metadata and target columns from feature DataFrame.

    Excludes columns in QUARANTINE_COLUMNS (such as 'is_death_3yr', 'is_death',
    'PatientID', and 'fold') if present, returning a cleaned feature DataFrame.

    Args:
        df: Input DataFrame potentially containing quarantined columns.

    Returns:
        DataFrame with quarantined columns removed.
    """
    cols_to_drop = [c for c in QUARANTINE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


class BenchmarkModelResult(TypedDict):
    """Structured evaluation result container for a single benchmark model."""

    model_name: str
    optimal_threshold: float
    oof_f1: float
    oof_probs: np.ndarray
    test_probs: np.ndarray
    test_pred_default: np.ndarray
    test_pred_optimal: np.ndarray
    metrics_default: dict[str, float]
    metrics_optimal: dict[str, float]
    fitted_pipeline: ImbPipeline


class BenchmarkSuiteResult(TypedDict):
    """Structured evaluation result container for the entire benchmark suite."""

    metrics_df: pd.DataFrame
    test_predictions_df: pd.DataFrame
    oof_predictions_df: pd.DataFrame
    fitted_pipelines: dict[str, ImbPipeline]
    results: dict[str, BenchmarkModelResult]


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
    Target, identifier, and fold columns ('is_death_3yr', 'is_death', 'PatientID',
    'fold') are dropped before determining column positions to prevent any index drift.

    Args:
        df: Input DataFrame containing feature columns.
        discrete_columns: Optional sequence of explicit discrete column names.
            If None, checks against known DEFAULT_DISCRETE_COLUMNS.
        max_unique_binary: Maximum unique value count to classify as binary (default 2).

    Returns:
        List of 0-based integer column indices corresponding to categorical features.
    """
    feature_df = quarantine_features(df)

    disc_set = set(
        discrete_columns if discrete_columns is not None else DEFAULT_DISCRETE_COLUMNS
    )
    categorical_indices: list[int] = []

    for idx, col in enumerate(feature_df.columns):
        series = feature_df[col]
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

    pipeline = ImbPipeline([("smotenc", resampler), ("classifier", classifier)])
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
    groups_train: pd.Series | np.ndarray | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | np.ndarray | None = None,
    n_splits: int = 5,
    folds_train: pd.Series | np.ndarray | None = None,
) -> BenchmarkModelResult:
    """Execute cross-validation, OOF threshold tuning, and test inference.

    Requires static pre-assigned folds (`folds_train` or embedded 'fold' column).
    Eliminates dynamic cross-validation fold generation. Automatically quarantines
    'fold', 'PatientID', 'is_death_3yr', and 'is_death' from feature matrix X.
    Always asserts zero subject/patient identity leakage across train and val splits
    whenever patient group identifiers are provided.

    Args:
        model_name: Identifier name of the model.
        pipeline: ImbPipeline containing SMOTE-NC resampler and classifier.
        x_train: Training feature DataFrame.
        y_train: Training binary target series/array.
        groups_train: Optional patient grouping identifiers for training records.
            Used to assert zero subject leakage across folds.
        x_test: Uncurated test feature DataFrame.
        y_test: Test binary target series/array.
        n_splits: Number of cross-validation folds (default 5).
        folds_train: Optional pre-assigned fold indices. If None and 'fold' is
            in x_train, extracts from x_train['fold'].

    Returns:
        BenchmarkModelResult containing OOF probabilities, optimal threshold, test
        probabilities, fitted full pipeline, and metrics under both default
        and optimal thresholds.

    Raises:
        ValueError: If static folds cannot be resolved, or if x_test/y_test are None,
            or if NaNs are produced.
        RuntimeError: If patient identity overlap is detected between train and
            validation partitions.
    """
    if x_test is None or y_test is None:
        raise ValueError("x_test and y_test must be provided for evaluation.")

    # 1. Resolve and extract static folds and patient groupings
    folds: np.ndarray | None = None
    if folds_train is not None:
        folds = np.asarray(folds_train)
    elif isinstance(x_train, pd.DataFrame) and "fold" in x_train.columns:
        folds = x_train["fold"].to_numpy()

    if folds is None:
        raise ValueError(
            "Static folds must be provided via folds_train or 'fold' column in x_train."
        )

    groups: np.ndarray | None = None
    if groups_train is not None:
        groups = np.asarray(groups_train)
    elif isinstance(x_train, pd.DataFrame) and "PatientID" in x_train.columns:
        groups = x_train["PatientID"].to_numpy()

    # 2. Strict quarantine of metadata columns from feature matrix X
    x_tr_clean = quarantine_features(x_train)
    x_te_clean = quarantine_features(x_test)

    y_tr = np.asarray(y_train, dtype=int)
    y_te = np.asarray(y_test, dtype=int)

    # 3. Cross-validation partition construction
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Iterate directly over static fold indices (0 to n_splits-1):
    for fold_idx in range(n_splits):
        train_idx = np.where(folds != fold_idx)[0]
        val_idx = np.where(folds == fold_idx)[0]
        if len(val_idx) == 0:
            raise ValueError(
                f"Static fold {fold_idx} contains zero validation samples."
            )
        splits.append((train_idx, val_idx))

    oof_probs = np.full(len(y_tr), np.nan, dtype=float)

    # 4. Fold-level training & OOF prediction collection
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Still assert zero patient leakage across train and val splits
        if groups is not None:
            train_patients = set(groups[train_idx])
            val_patients = set(groups[val_idx])
            overlap = train_patients.intersection(val_patients)
            if len(overlap) > 0:
                raise RuntimeError(
                    f"Subject leakage detected in fold {fold_idx}: "
                    f"{len(overlap)} overlapping patients."
                )

        x_tr_fold = x_tr_clean.iloc[train_idx]
        y_tr_fold = y_tr[train_idx]
        x_val_fold = x_tr_clean.iloc[val_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(x_tr_fold, y_tr_fold)

        probs_val = fold_pipe.predict_proba(x_val_fold)
        oof_probs[val_idx] = probs_val[:, 1] if probs_val.ndim == 2 else probs_val

    if np.isnan(oof_probs).any():
        raise ValueError(
            f"NaN values encountered in OOF predictions for model {model_name}."
        )

    # 5. Calibrate optimal decision threshold strictly on OOF predictions
    optimal_threshold, oof_f1 = find_optimal_threshold(y_tr, oof_probs)

    # 6. Full refit on complete quarantined training set
    full_pipe = clone(pipeline)
    full_pipe.fit(x_tr_clean, y_tr)

    # 7. Inference on quarantined test set
    test_probs_raw = full_pipe.predict_proba(x_te_clean)
    test_probs = test_probs_raw[:, 1] if test_probs_raw.ndim == 2 else test_probs_raw

    if np.isnan(test_probs).any():
        raise ValueError(
            f"NaN values encountered in test predictions for model {model_name}."
        )

    # 8. Generate dual binary decisions
    test_pred_default = (test_probs >= 0.50).astype(int)
    test_pred_optimal = (test_probs >= optimal_threshold).astype(int)

    # 9. Evaluate metrics on test set
    metrics_default = calculate_metrics(y_te, test_probs, threshold=0.50)
    metrics_optimal = calculate_metrics(y_te, test_probs, threshold=optimal_threshold)

    result: BenchmarkModelResult = {
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
    groups_train: pd.Series | np.ndarray | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | np.ndarray | None = None,
    models: dict[str, BaseEstimator] | None = None,
    categorical_indices: Sequence[int] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    folds_train: pd.Series | np.ndarray | None = None,
) -> BenchmarkSuiteResult:
    """Execute end-to-end benchmark across all models with in-fold SMOTE-NC.

    Supports static fold partition consumption, isolating in-fold SMOTE-NC,
    and quarantining metadata columns ('fold', 'PatientID', target labels)
    from feature matrix X.

    Args:
        x_train: Cleaned training feature matrix.
        y_train: Training binary mortality labels.
        groups_train: Optional patient grouping IDs for leakage verification.
        x_test: Uncurated test feature matrix.
        y_test: Test binary mortality labels.
        models: Optional dictionary of models. If None, uses get_model_zoo().
        categorical_indices: Optional categorical column indices. If None,
            automatically detected after feature quarantine.
        n_splits: Number of cross-validation folds (default 5).
        random_state: Random seed for reproducibility.
        folds_train: Optional pre-assigned cross-validation fold assignments.
            If None and 'fold' column exists in x_train, it is consumed automatically.

    Returns:
        BenchmarkSuiteResult containing:
            - metrics_df: Comparative DataFrame across all models and metrics.
            - test_predictions_df: Test set probabilities and dual decisions.
            - oof_predictions_df: Training set OOF probabilities.
            - fitted_pipelines: Dict of trained full pipelines per model.
            - results: Detailed BenchmarkModelResult dictionary per model.

    Raises:
        ValueError: If x_test or y_test are not provided, or if static folds
            cannot be resolved from inputs.
    """
    if x_test is None or y_test is None:
        raise ValueError("x_test and y_test must be provided for benchmark suite.")

    if models is None:
        models = get_model_zoo(random_state=random_state)

    # Resolve folds and groups before quarantine
    folds: np.ndarray | None = None
    if folds_train is not None:
        folds = np.asarray(folds_train)
    elif isinstance(x_train, pd.DataFrame) and "fold" in x_train.columns:
        folds = x_train["fold"].to_numpy()

    if folds is None:
        raise ValueError(
            "Static folds must be provided via folds_train or 'fold' column in x_train."
        )

    groups: np.ndarray | None = None
    if groups_train is not None:
        groups = np.asarray(groups_train)
    elif isinstance(x_train, pd.DataFrame) and "PatientID" in x_train.columns:
        groups = x_train["PatientID"].to_numpy()

    # Quarantine metadata columns from feature matrices
    x_tr_clean = quarantine_features(x_train)
    x_te_clean = quarantine_features(x_test)

    if categorical_indices is None:
        categorical_indices = identify_categorical_features(x_tr_clean)

    y_tr = np.asarray(y_train, dtype=int)
    y_te = np.asarray(y_test, dtype=int)

    summary_rows: list[dict[str, Any]] = []
    test_pred_dict: dict[str, np.ndarray] = {
        "is_death_3yr": y_te,
    }
    oof_pred_dict: dict[str, np.ndarray] = {
        "is_death_3yr": y_tr,
    }
    if folds is not None:
        oof_pred_dict["fold"] = folds

    fitted_pipelines: dict[str, ImbPipeline] = {}
    detailed_results: dict[str, BenchmarkModelResult] = {}

    for name, estimator in models.items():
        logger.info(f"Training benchmark model: {name}")
        pipe = create_smote_pipeline(
            classifier=estimator,
            categorical_indices=categorical_indices,
            random_state=random_state,
            n_features=x_tr_clean.shape[1],
        )

        res = evaluate_model_cv_and_test(
            model_name=name,
            pipeline=pipe,
            x_train=x_tr_clean,
            y_train=y_tr,
            groups_train=groups,
            x_test=x_te_clean,
            y_test=y_te,
            n_splits=n_splits,
            folds_train=folds,
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
    test_predictions_df = pd.DataFrame(test_pred_dict, index=x_te_clean.index)
    oof_predictions_df = pd.DataFrame(oof_pred_dict, index=x_tr_clean.index)

    return {
        "metrics_df": metrics_df,
        "test_predictions_df": test_predictions_df,
        "oof_predictions_df": oof_predictions_df,
        "fitted_pipelines": fitted_pipelines,
        "results": detailed_results,
    }
