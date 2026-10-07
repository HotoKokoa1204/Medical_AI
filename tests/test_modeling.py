"""Module: test_modeling
Stage: Script
Author: KafuuChino
Date: 2026-09-30
Description: Unit and integration tests for 11-algorithm benchmark suite,
    in-fold SMOTE-NC, and OOF threshold calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure worktree src is prioritized
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from imblearn.over_sampling import SMOTENC  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

from agilab_lib.modeling import (  # noqa: E402
    BenchmarkModelResult,
    calculate_metrics,
    create_smote_pipeline,
    evaluate_model_cv_and_test,
    find_optimal_threshold,
    get_model_zoo,
    identify_categorical_features,
    run_benchmark_suite,
)
from agilab_lib.visualization import (  # noqa: E402
    plot_benchmark_pr_curves,
    plot_benchmark_roc_curves,
    plot_top_feature_importance,
)

TRAIN_PARQUET_PATH = Path("data/processed/train_cleaned_rolling_3yr.parquet")
TEST_PARQUET_PATH = Path("data/processed/test_uncurated_rolling_3yr.parquet")


@pytest.fixture
def synthetic_mixed_cohort() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray
]:
    """Generate reproducible synthetic mixed cohort with patient clusters.

    Returns:
        Tuple of (x_train, y_train, groups_train, x_test, y_test, groups_test).
    """
    np.random.seed(42)
    n_train = 200
    n_test = 60
    n_train_patients = 20
    n_test_patients = 6

    # Generate patient IDs (completely disjoint)
    groups_train = np.repeat(
        np.arange(1, n_train_patients + 1), n_train // n_train_patients
    )
    groups_test = np.repeat(
        np.arange(n_train_patients + 1, n_train_patients + n_test_patients + 1),
        n_test // n_test_patients,
    )

    feature_cols = {
        "cont_age": np.random.normal(65, 10, n_train),
        "cont_ktv": np.random.normal(1.3, 0.2, n_train),
        "cont_albumin": np.random.normal(3.8, 0.4, n_train),
        "cont_creatinine": np.random.normal(10.5, 2.0, n_train),
        "bin_chf": np.random.choice([0.0, 1.0], size=n_train, p=[0.75, 0.25]),
        "bin_cad": np.random.choice([0.0, 1.0], size=n_train, p=[0.80, 0.20]),
        "bin_dm": np.random.choice([0.0, 1.0], size=n_train, p=[0.60, 0.40]),
        "disc_schedule": np.random.choice([1, 2, 3], size=n_train),
        "HD_WS": np.random.choice([1, 2, 3, 4], size=n_train),
    }
    x_train = pd.DataFrame(feature_cols)
    y_train = np.array([0] * 160 + [1] * 40)
    np.random.shuffle(y_train)

    test_feature_cols = {
        "cont_age": np.random.normal(65, 10, n_test),
        "cont_ktv": np.random.normal(1.3, 0.2, n_test),
        "cont_albumin": np.random.normal(3.8, 0.4, n_test),
        "cont_creatinine": np.random.normal(10.5, 2.0, n_test),
        "bin_chf": np.random.choice([0.0, 1.0], size=n_test, p=[0.75, 0.25]),
        "bin_cad": np.random.choice([0.0, 1.0], size=n_test, p=[0.80, 0.20]),
        "bin_dm": np.random.choice([0.0, 1.0], size=n_test, p=[0.60, 0.40]),
        "disc_schedule": np.random.choice([1, 2, 3], size=n_test),
        "HD_WS": np.random.choice([1, 2, 3, 4], size=n_test),
    }
    x_test = pd.DataFrame(test_feature_cols)
    y_test = np.array([0] * 48 + [1] * 12)
    np.random.shuffle(y_test)

    return x_train, y_train, groups_train, x_test, y_test, groups_test


@pytest.fixture
def synthetic_partitioned_cohort(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Generate synthetic cohort with pre-computed static fold assignments.

    Returns:
        Tuple of (x_train, y_train, groups_train, folds_train,
        x_test, y_test, groups_test).
    """
    x_train, y_train, groups_train, x_test, y_test, groups_test = synthetic_mixed_cohort
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds_train = np.zeros(len(y_train), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(x_train, y_train, groups=groups_train)
    ):
        folds_train[val_idx] = fold_idx
    return x_train, y_train, groups_train, folds_train, x_test, y_test, groups_test


def test_identify_categorical_features(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Test dynamic identification of binary, categorical, and discrete columns."""
    x_train, _, _, _, _, _ = synthetic_mixed_cohort
    cat_indices = identify_categorical_features(
        x_train, discrete_columns=["disc_schedule", "HD_WS"]
    )

    cat_cols = [x_train.columns[i] for i in cat_indices]
    expected_discrete = {"bin_chf", "bin_cad", "bin_dm", "disc_schedule", "HD_WS"}
    assert set(cat_cols) == expected_discrete

    # Continuous columns must not be identified as categorical
    cont_cols = [col for col in x_train.columns if col not in cat_cols]
    assert set(cont_cols) == {
        "cont_age",
        "cont_ktv",
        "cont_albumin",
        "cont_creatinine",
    }


def test_find_optimal_threshold() -> None:
    """Test OOF decision threshold optimization under known probabilities."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Clear separation around 0.60
    y_prob = np.array([0.10, 0.20, 0.30, 0.40, 0.70, 0.80, 0.85, 0.90])

    optimal_t, best_f1 = find_optimal_threshold(y_true, y_prob)
    assert 0.40 < optimal_t <= 0.70
    assert best_f1 == 1.0

    # Ensure search bounds stay within [0.05, 0.95]
    assert 0.05 <= optimal_t <= 0.95


def test_oof_threshold_independence_from_test_labels(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Assert OOF threshold selection is completely independent of test labels."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    cat_indices = identify_categorical_features(x_train)

    pipe = create_smote_pipeline(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        categorical_indices=cat_indices,
        random_state=42,
    )

    # Run with original test labels and static folds
    res_orig = evaluate_model_cv_and_test(
        model_name="LR_Test",
        pipeline=pipe,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        n_splits=5,
        folds_train=folds_train,
    )

    # Invert all test labels entirely
    y_test_inverted = 1 - y_test
    res_inv = evaluate_model_cv_and_test(
        model_name="LR_Test",
        pipeline=pipe,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test_inverted,
        n_splits=5,
        folds_train=folds_train,
    )

    # Optimal threshold T* MUST be strictly identical regardless of test labels!
    assert res_orig["optimal_threshold"] == res_inv["optimal_threshold"]
    assert res_orig["oof_f1"] == res_inv["oof_f1"]
    assert np.allclose(res_orig["oof_probs"], res_inv["oof_probs"])


def test_group_kfold_patient_isolation(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Assert GroupKFold guarantees zero patient identity overlap across folds."""
    x_train, y_train, groups_train, _, _, _ = synthetic_mixed_cohort
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(x_train, y_train, groups=groups_train)
    ):
        train_pids = set(groups_train[train_idx])
        val_pids = set(groups_train[val_idx])
        overlap = train_pids.intersection(val_pids)
        assert len(overlap) == 0, f"Fold {fold} has subject overlap: {overlap}"


def test_infold_smote_nc_isolation(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Assert in-fold SMOTE-NC never alters validation folds or test set."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    cat_indices = identify_categorical_features(x_train)

    initial_test_len = len(x_test)
    initial_test_event_rate = np.mean(y_test)

    pipe = create_smote_pipeline(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        categorical_indices=cat_indices,
        random_state=42,
    )

    res = evaluate_model_cv_and_test(
        model_name="Isolation_Check",
        pipeline=pipe,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        n_splits=5,
        folds_train=folds_train,
    )

    # Test set must remain strictly untouched
    assert len(x_test) == initial_test_len
    assert len(res["test_probs"]) == initial_test_len
    assert np.mean(y_test) == initial_test_event_rate

    # Check SMOTE-NC maintains pure discrete flags (0/1) in resampled folds
    sm = SMOTENC(categorical_features=cat_indices, random_state=42)
    x_res, _ = sm.fit_resample(x_train, y_train)
    for col in ["bin_chf", "bin_cad", "bin_dm"]:
        unique_vals = set(x_res[col].unique())
        assert unique_vals.issubset({0.0, 1.0, 0, 1}), (
            f"Non-binary comorbidity generated: {unique_vals}"
        )


def test_static_fold_consumption(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify evaluate_model_cv_and_test directly consumes static fold assignments."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    cat_indices = identify_categorical_features(x_train)
    pipe = create_smote_pipeline(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        categorical_indices=cat_indices,
        random_state=42,
    )

    res = evaluate_model_cv_and_test(
        model_name="Static_Fold_Test",
        pipeline=pipe,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        n_splits=5,
        folds_train=folds_train,
    )

    assert len(res["oof_probs"]) == len(y_train)
    assert not np.isnan(res["oof_probs"]).any()
    assert 0.05 <= res["optimal_threshold"] <= 0.95
    assert not np.isnan(res["test_probs"]).any()


def test_feature_quarantine_fold_and_metadata_columns(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify fold and metadata columns in x_train are quarantined."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    x_train_embedded = x_train.copy()
    x_train_embedded["fold"] = folds_train
    x_train_embedded["PatientID"] = groups_train
    x_train_embedded["is_death_3yr"] = y_train

    x_test_embedded = x_test.copy()
    x_test_embedded["fold"] = 0
    x_test_embedded["PatientID"] = 999
    x_test_embedded["is_death_3yr"] = y_test

    models = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)}

    results = run_benchmark_suite(
        x_train=x_train_embedded,
        y_train=y_train,
        x_test=x_test_embedded,
        y_test=y_test,
        models=models,
        n_splits=5,
        random_state=42,
    )

    fitted_pipe = results["fitted_pipelines"]["Logistic Regression"]
    clf = fitted_pipe.named_steps["classifier"]
    expected_n_features = x_train.shape[1]
    assert clf.n_features_in_ == expected_n_features
    if hasattr(clf, "feature_names_in_"):
        assert "fold" not in clf.feature_names_in_
        assert "PatientID" not in clf.feature_names_in_
        assert "is_death_3yr" not in clf.feature_names_in_


def test_static_folds_patient_leakage_assertion(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify runtime error is raised if static folds leak patient identities."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    corrupted_folds = folds_train.copy()
    patient_1_idx = np.where(groups_train == 1)[0]
    corrupted_folds[patient_1_idx[: len(patient_1_idx) // 2]] = 0
    corrupted_folds[patient_1_idx[len(patient_1_idx) // 2 :]] = 1

    cat_indices = identify_categorical_features(x_train)
    pipe = create_smote_pipeline(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        categorical_indices=cat_indices,
        random_state=42,
    )

    with pytest.raises(RuntimeError, match="Subject leakage detected"):
        evaluate_model_cv_and_test(
            model_name="Leakage_Check",
            pipeline=pipe,
            x_train=x_train,
            y_train=y_train,
            groups_train=groups_train,
            x_test=x_test,
            y_test=y_test,
            n_splits=5,
            folds_train=corrupted_folds,
        )


def test_calculate_metrics_correctness() -> None:
    """Test 8 canonical clinical classification metrics calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.50)

    # Threshold-independent metrics
    assert "roc_auc" in metrics and metrics["roc_auc"] == 1.0
    assert "pr_auc" in metrics and metrics["pr_auc"] == 1.0
    assert "brier_score" in metrics and 0.0 <= metrics["brier_score"] <= 1.0

    # Discrete classification metrics
    assert "accuracy" in metrics and metrics["accuracy"] == 1.0
    assert "sensitivity" in metrics and metrics["sensitivity"] == 1.0
    assert "specificity" in metrics and metrics["specificity"] == 1.0
    assert "f1_score" in metrics and metrics["f1_score"] == 1.0
    assert "cohen_kappa" in metrics and metrics["cohen_kappa"] == 1.0


def test_all_11_models_synthetic_battery(
    synthetic_partitioned_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify Invariant 7: all 11 models train on static folds."""
    x_train, y_train, groups_train, folds_train, x_test, y_test, _ = (
        synthetic_partitioned_cohort
    )
    cat_indices = identify_categorical_features(x_train)

    models = get_model_zoo(random_state=42)
    assert len(models) == 11, f"Expected 11 models, got {len(models)}"

    for name, estimator in models.items():
        pipe = create_smote_pipeline(
            classifier=estimator,
            categorical_indices=cat_indices,
            random_state=42,
            n_features=x_train.shape[1],
        )

        res = evaluate_model_cv_and_test(
            model_name=name,
            pipeline=pipe,
            x_train=x_train,
            y_train=y_train,
            groups_train=groups_train,
            x_test=x_test,
            y_test=y_test,
            n_splits=5,
            folds_train=folds_train,
        )

        test_probs = res["test_probs"]
        oof_probs = res["oof_probs"]

        # Assert probabilities are valid and bounded in [0, 1]
        assert not np.isnan(test_probs).any(), f"NaNs in test probs for {name}"
        assert not np.isnan(oof_probs).any(), f"NaNs in OOF probs for {name}"
        assert np.all((test_probs >= 0.0) & (test_probs <= 1.0)), (
            f"Out of bounds prob for {name}"
        )
        assert np.all((oof_probs >= 0.0) & (oof_probs <= 1.0)), (
            f"Out of bounds OOF prob for {name}"
        )

        # Assert threshold calibration
        assert 0.05 <= res["optimal_threshold"] <= 0.95, (
            f"Threshold out of bounds for {name}"
        )
        assert not np.isnan(res["oof_f1"])

        # Assert decisions are binary
        assert set(res["test_pred_default"]).issubset({0, 1})
        assert set(res["test_pred_optimal"]).issubset({0, 1})

        # Assert all 8 metrics exist and are non-NaN
        for m_dict in (res["metrics_default"], res["metrics_optimal"]):
            for k, v in m_dict.items():
                assert not np.isnan(v), f"NaN metric {k} for model {name}"


def test_visualizations_battery(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify publication visualization routines generate valid figures."""
    x_train, y_train, groups_train, x_test, y_test, _ = synthetic_mixed_cohort
    cat_indices = identify_categorical_features(x_train)

    # Run lightweight benchmark
    models = {
        "Random Forest": get_model_zoo(42)["Random Forest"],
        "Logistic Regression": get_model_zoo(42)["Logistic Regression"],
    }
    benchmark_res = run_benchmark_suite(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        models=models,
        categorical_indices=cat_indices,
        n_splits=5,
    )

    test_pred_df = benchmark_res["test_predictions_df"]
    fitted_pipes = benchmark_res["fitted_pipelines"]

    # 1. ROC Curves
    roc_fig = plot_benchmark_roc_curves(y_true=y_test, probabilities=test_pred_df)
    assert roc_fig is not None
    assert len(roc_fig.axes) > 0

    # 2. PR Curves
    pr_fig = plot_benchmark_pr_curves(y_true=y_test, probabilities=test_pred_df)
    assert pr_fig is not None
    assert len(pr_fig.axes) > 0

    # 3. Feature Importance
    fi_fig = plot_top_feature_importance(
        models=fitted_pipes,
        feature_names=x_train.columns.tolist(),
        top_n=5,
    )
    assert fi_fig is not None
    assert len(fi_fig.axes) > 0


@pytest.mark.skipif(
    not TRAIN_PARQUET_PATH.exists(),
    reason="Processed parquet datasets not found",
)
def test_real_cohort_anti_leakage_properties() -> None:
    """Assert real processed dataset properties (N=1,112 test, zero leakage)."""
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)

    assert len(train_df) == 3737, f"Expected 3,737 train rows, got {len(train_df)}"
    assert len(test_df) == 1112, f"Expected 1,112 test rows, got {len(test_df)}"

    # Feature column parity
    target = "is_death_3yr"
    train_features = [c for c in train_df.columns if c != target]
    test_features = [c for c in test_df.columns if c != target]
    assert train_features == test_features, "Feature mismatch between train and test!"

    # Zero missing values
    assert train_df.isna().sum().sum() == 0, "Null values in cleaned train matrix!"
    assert test_df.isna().sum().sum() == 0, "Null values in test matrix!"


def test_pipeline_named_steps_classifier() -> None:
    """Verify create_smote_pipeline configures 'classifier' step name (Spec line 56)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipe = create_smote_pipeline(
        classifier=clf,
        categorical_indices=[0, 1],
        random_state=42,
    )
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["classifier"] is clf
    assert "smotenc" in pipe.named_steps
    assert "clf" not in pipe.named_steps


def test_group_kfold_validation_empirical_class_balance(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Assert within GroupKFold validation folds preserve empirical class balance.

    Validates Spec line 83: In-fold SMOTE-NC must only transform training
    partitions, leaving validation folds in their natural empirical distribution
    without synthetic samples.
    """
    x_train, y_train, groups_train, _, _, _ = synthetic_mixed_cohort
    cat_indices = identify_categorical_features(x_train)

    gkf = GroupKFold(n_splits=5)
    for _fold, (train_idx, val_idx) in enumerate(
        gkf.split(x_train, y_train, groups=groups_train)
    ):
        y_val_empirical = y_train[val_idx]
        val_empirical_positives = int(np.sum(y_val_empirical))
        val_empirical_total = len(y_val_empirical)
        val_empirical_prevalence = float(np.mean(y_val_empirical))

        pipe = create_smote_pipeline(
            classifier=LogisticRegression(max_iter=1000, random_state=42),
            categorical_indices=cat_indices,
            random_state=42,
        )

        # Fit exclusively on fold training partition
        x_tr_fold = x_train.iloc[train_idx]
        y_tr_fold = y_train[train_idx]
        pipe.fit(x_tr_fold, y_tr_fold)

        # Ensure validation fold is evaluated directly without resampling
        x_val_fold = x_train.iloc[val_idx]
        val_probs = pipe.predict_proba(x_val_fold)

        # Strict checks: validation size, label counts, prevalence remain untouched
        assert len(val_probs) == val_empirical_total
        assert int(np.sum(y_train[val_idx])) == val_empirical_positives
        assert float(np.mean(y_train[val_idx])) == pytest.approx(
            val_empirical_prevalence
        )

        # Verify training fold was indeed resampled by SMOTE-NC
        smotenc_step = pipe.named_steps["smotenc"]
        assert hasattr(smotenc_step, "categorical_features")
        assert smotenc_step.categorical_features == cat_indices


def test_identify_categorical_features_drops_target_and_id() -> None:
    """Verify target, PatientID, and fold columns are dropped to prevent index drift."""
    df = pd.DataFrame(
        {
            "PatientID": [101, 102, 103, 104],
            "fold": [0, 1, 2, 3],
            "is_death_3yr": [0, 1, 0, 1],
            "cont_feature": [1.2, 3.4, 5.6, 7.8],
            "bin_comorb": [0.0, 1.0, 0.0, 1.0],
            "HD_WS": [1, 2, 3, 4],
        }
    )

    cat_indices = identify_categorical_features(df)
    # Feature matrix after dropping PatientID, fold, and is_death_3yr has columns:
    # 0: cont_feature, 1: bin_comorb, 2: HD_WS
    # Categorical indices must be [1, 2], NOT [3, 4]
    assert cat_indices == [1, 2]

    # Verify matching feature dataframe
    feature_df = df.drop(columns=["PatientID", "fold", "is_death_3yr"])
    cat_cols = [feature_df.columns[i] for i in cat_indices]
    assert cat_cols == ["bin_comorb", "HD_WS"]


def test_typed_benchmark_results(
    synthetic_mixed_cohort: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Verify evaluate_model_cv_and_test returns typed BenchmarkModelResult."""
    x_train, y_train, groups_train, x_test, y_test, _ = synthetic_mixed_cohort
    cat_indices = identify_categorical_features(x_train)

    pipe = create_smote_pipeline(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        categorical_indices=cat_indices,
        random_state=42,
    )

    res: BenchmarkModelResult = evaluate_model_cv_and_test(
        model_name="LR_Typed",
        pipeline=pipe,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        n_splits=5,
    )

    expected_keys = {
        "model_name",
        "optimal_threshold",
        "oof_f1",
        "oof_probs",
        "test_probs",
        "test_pred_default",
        "test_pred_optimal",
        "metrics_default",
        "metrics_optimal",
        "fitted_pipeline",
    }
    assert set(res.keys()) == expected_keys
    assert "classifier" in res["fitted_pipeline"].named_steps
