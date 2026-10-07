"""Module: test_preprocessing
Stage: Script
Author: KafuuChino
Date: 2026-09-28
Description: Unit tests for longitudinal rolling 3-year preprocessing, PCA,
    and outlier detection pipeline.
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

from agilab_lib.analysis import (  # noqa: E402
    calculate_hotelling_t2,
    calculate_spe,
    calculate_spe_limit,
    calculate_t2_limit,
    fit_pca,
    prune_multivariate_outliers,
)
from agilab_lib.preprocessing import (  # noqa: E402
    ASSESSMENT_COLUMNS,
    CATEGORICAL_COLUMNS,
    DEFAULT_CONTINUOUS_COLUMNS,
    DISCRETE_COUNT_COLUMNS,
    HIGH_MISSING_DROPS,
    IDENTIFIER_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
    ZERO_FILL_INDICATOR_COLUMNS,
    LongitudinalPreprocessor,
    clean_clinical_bounds,
    deterministic_pruning,
    encode_categorical_features,
    engineer_features,
    load_longitudinal_cohort,
    process_continuous_features,
    quarantine_target_leakage,
    split_patient_cohort,
)
from agilab_lib.visualization import (  # noqa: E402
    plot_t2_vs_spe,
)

DATA_PATH = Path("data/Kidit_Master_Baseline_V2.xlsx")
if not DATA_PATH.exists() and Path("E:/github/Kidit_Master_Baseline_V2.xlsx").exists():
    DATA_PATH = Path("E:/github/Kidit_Master_Baseline_V2.xlsx")


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking raw hemodialysis baseline records.

    Returns:
        pd.DataFrame containing representative synthetic clinical records.
    """
    return pd.DataFrame(
        {
            "PatientID": [1001, 1002, 1003, 1004],
            "year": [2010, 2011, 2012, 2013],
            "index_date": pd.to_datetime(
                ["2010-06-01", "2011-06-01", "2012-06-01", "2013-06-01"]
            ),
            "首次出現年度": [2010, 2011, 2012, 2013],
            "is_death": [0, 1, 0, 1],
            "死亡日期": [np.nan, "2015-01-01", np.nan, "2018-02-01"],
            "死亡原因大類": [np.nan, "A", np.nan, "B"],
            "死亡原因細類": [np.nan, "A-01", np.nan, "B-02"],
            "是否離開本院": [0, 1, 0, 0],
            "離開途徑": [np.nan, "轉院", np.nan, np.nan],
            "離開年度": [np.nan, 2014, np.nan, np.nan],
            "轉院原因": [np.nan, 1, np.nan, np.nan],
            "轉至何院所": [np.nan, "0101", np.nan, np.nan],
            "最終醫療狀態": [1, 9, 1, 9],
            "腎移植日期": [np.nan, np.nan, np.nan, np.nan],
            "出現年度數": [5, 4, 3, 2],
            "最後出現年度": [2015, 2015, 2015, 2015],
            "First_HD_date": pd.to_datetime(
                ["2008-06-01", np.nan, "2011-06-01", np.nan]
            ),
            "SKH_hd_date": pd.to_datetime(
                ["2008-06-01", "2010-06-01", "2011-06-01", np.nan]
            ),
            "before_hd_weight": [60.0, 70.0, 80.0, 50.0],
            "after_hd_weight": [58.0, 67.5, 77.0, 48.0],
            "Kt_V": [1.4, 0.04, 1.2, np.nan],
            "nPCR": [1.1, 0.35, 1.0, 1.2],
            "BFR": [250, 0, 200, 300],
            "DFR": [500, 500, 0, 500],
            "PTH": [150.0, 450.0, 25.0, 800.0],
            "Ferritin": [200.0, 600.0, 150.0, 1200.0],
            "Triglyceride": [120.0, 250.0, 90.0, 300.0],
            "Albumin": [4.0, 3.5, 3.8, 3.2],
            "Creatinine": [11.0, 12.5, 9.8, 10.2],
            "CaXP": [45.0, np.nan, np.nan, np.nan],
            "totalCa": [np.nan, np.nan, 9.0, np.nan],
            "ionizedCa": [np.nan, np.nan, 4.5, np.nan],
            "K": [3.5, 2.2, 4.0, 5.1],
            "isEPO": ["是", "否", "是", np.nan],
            "AVF": ["Y", "N", np.nan, "Y"],
            "AVG": ["N", "Y", "N", np.nan],
            "HBsAg": ["N", "Y", "O", np.nan],
            "Anti_HCV": ["N", "O", "Y", "N"],
            "blood_type": ["A", "B", np.nan, "O"],
            "education": [2, 4, 3, np.nan],
            "marriage": [1, 2, np.nan, 4],
            "Dialysis_type": [1, 2, 3, np.nan],
            "sex": [1, 2, 1, 2],
            "DM": [1, 0, 1, 0],
            "Hypertension": [1, 1, 0, 1],
            "HD_WS": [3, np.nan, 2, 3],
            "HD_ST": [4.0, np.nan, 3.0, 4.0],
            "CIC": [3.0, np.nan, 2.0, 3.0],
            "total_hd_time": [240.0, np.nan, 240.0, 210.0],
            "txIntervalMin": [2640.0, np.nan, 2880.0, 2640.0],
            "ID_U": [1250.0, np.nan, 1000.0, 1250.0],
            "MD_U": [250.0, np.nan, 200.0, 250.0],
            "genAssess": [5.0, np.nan, 7.0, 3.0],
            "actAssess": [70.0, np.nan, 90.0, 50.0],
            "Comorb_Count": [2, 1, 1, 1],
        }
    )


def test_quarantine_leakage(sample_raw_dataframe: pd.DataFrame) -> None:
    """Test that all target leakage and non-predictive ID fields are excluded."""
    features_df, leakage_df, y = quarantine_target_leakage(sample_raw_dataframe)

    assert "is_death" not in features_df.columns
    assert len(y) == len(sample_raw_dataframe)
    assert (y == sample_raw_dataframe["is_death"]).all()

    for col in TARGET_LEAKAGE_COLUMNS:
        assert col not in features_df.columns
        if col in sample_raw_dataframe.columns:
            assert col in leakage_df.columns

    for col in IDENTIFIER_COLUMNS:
        assert col not in features_df.columns


def test_engineer_features(sample_raw_dataframe: pd.DataFrame) -> None:
    """Test feature engineering: vintage calculation and weight decoupling."""
    df_eng = engineer_features(sample_raw_dataframe)

    assert "dialysis_vintage_years" in df_eng.columns
    assert np.isclose(df_eng.loc[0, "dialysis_vintage_years"], 2.0, atol=0.05)
    assert np.isclose(df_eng.loc[1, "dialysis_vintage_years"], 1.0, atol=0.05)

    assert "ultrafiltration" in df_eng.columns
    assert "uf_ratio" in df_eng.columns
    assert "after_hd_weight" not in df_eng.columns
    assert np.isclose(df_eng.loc[0, "ultrafiltration"], 2.0)
    assert np.isclose(df_eng.loc[0, "uf_ratio"], 2.0 / 60.0)


def test_clean_clinical_bounds(sample_raw_dataframe: pd.DataFrame) -> None:
    """Test that software artifact Kt/V < 0.5 is replaced by NaN."""
    df_clean = clean_clinical_bounds(sample_raw_dataframe)
    assert pd.isna(df_clean.loc[1, "Kt_V"])
    assert df_clean.loc[0, "Kt_V"] == 1.4


def test_process_continuous_features(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """Test imputation, log transform, dropping high missing, and scaling."""
    df_clean = clean_clinical_bounds(sample_raw_dataframe)
    df_eng = engineer_features(df_clean)

    cont_cols = [
        "before_hd_weight",
        "ultrafiltration",
        "uf_ratio",
        "dialysis_vintage_years",
        "Kt_V",
        "PTH",
        "Ferritin",
        "Triglyceride",
        "Albumin",
        "Creatinine",
        "CaXP",
        "totalCa",
        "ionizedCa",
    ]

    scaled_df, _, _, retained_cols = process_continuous_features(
        df_eng,
        continuous_columns=cont_cols,
        missing_threshold=0.20,
    )

    for col in HIGH_MISSING_DROPS:
        assert col not in scaled_df.columns
        assert col not in retained_cols

    assert scaled_df.isna().sum().sum() == 0

    for col in retained_cols:
        assert np.isclose(scaled_df[col].mean(), 0.0, atol=1e-5)


def test_encode_categorical_features(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """Test categorical encoding, binary standardization, and missing dummies."""
    encoded_df = encode_categorical_features(sample_raw_dataframe)

    assert "isEPO" in encoded_df.columns or any(
        c.startswith("isEPO_") for c in encoded_df.columns
    )
    assert any(c.startswith("blood_type_") for c in encoded_df.columns)
    assert any("nan" in c.lower() for c in encoded_df.columns)
    assert np.issubdtype(encoded_df.dtypes.iloc[0], np.number)


def test_non_mode_imputation_architecture(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """Test non-mode imputation rules per Ticket 1 (#4) spec:

    1. genAssess and actAssess: KNN-imputed via distance weights, not mode.
    2. HD_WS, HD_ST, CIC, total_hd_time, txIntervalMin: OneHotEncoded with _nan columns.
    3. ID_U and MD_U: Filled with 0.0 and accompanied by _isna binary indicator columns.
    4. Comorb_Count: Preserved as integer score.
    """
    df_clean = clean_clinical_bounds(sample_raw_dataframe)
    df_eng = engineer_features(df_clean)

    # Verify column vocabulary separation
    assert "genAssess" not in DISCRETE_COUNT_COLUMNS
    assert "actAssess" not in DISCRETE_COUNT_COLUMNS
    assert "HD_WS" not in DISCRETE_COUNT_COLUMNS
    assert "total_hd_time" not in DEFAULT_CONTINUOUS_COLUMNS
    assert "txIntervalMin" not in DEFAULT_CONTINUOUS_COLUMNS
    assert "Comorb_Count" in DISCRETE_COUNT_COLUMNS
    assert "genAssess" in ASSESSMENT_COLUMNS
    assert "actAssess" in ASSESSMENT_COLUMNS
    assert "ID_U" in ZERO_FILL_INDICATOR_COLUMNS
    assert "MD_U" in ZERO_FILL_INDICATOR_COLUMNS

    for col in ["HD_WS", "HD_ST", "CIC", "total_hd_time", "txIntervalMin"]:
        assert col in CATEGORICAL_COLUMNS

    preprocessor = LongitudinalPreprocessor()
    preprocessor.fit(df_eng)
    cont_df, full_df = preprocessor.transform(df_eng)

    # Invariant: No NaNs in full feature matrix
    assert full_df.isna().sum().sum() == 0

    # 1. genAssess and actAssess are KNN-imputed (row 1 had NaN, now imputed)
    assert "genAssess" in full_df.columns
    assert "actAssess" in full_df.columns
    assert not pd.isna(full_df.loc[1, "genAssess"])
    assert not pd.isna(full_df.loc[1, "actAssess"])
    # Row 0 had genAssess=5.0, actAssess=70.0 (preserved)
    assert np.isclose(full_df.loc[0, "genAssess"], 5.0)
    assert np.isclose(full_df.loc[0, "actAssess"], 70.0)

    # 2. HD_WS, CIC, total_hd_time, txIntervalMin have OneHotEncoded columns with _nan
    assert any(c.startswith("HD_WS_") for c in full_df.columns)
    assert any(c.startswith("HD_WS_nan") for c in full_df.columns)
    assert any(c.startswith("CIC_nan") for c in full_df.columns)
    assert any(c.startswith("total_hd_time_nan") for c in full_df.columns)
    assert any(c.startswith("txIntervalMin_nan") for c in full_df.columns)
    # Raw categorical columns must NOT be in continuous PCA matrix
    for cat_col in ["HD_WS", "HD_ST", "CIC", "total_hd_time", "txIntervalMin"]:
        assert cat_col not in cont_df.columns

    # 3. ID_U and MD_U are filled with 0.0 and have _isna indicators
    assert "ID_U" in full_df.columns
    assert "ID_U_isna" in full_df.columns
    assert "MD_U" in full_df.columns
    assert "MD_U_isna" in full_df.columns
    # Row 1 had NaN -> ID_U=0.0, ID_U_isna=1.0
    assert full_df.loc[1, "ID_U"] == 0.0
    assert full_df.loc[1, "ID_U_isna"] == 1.0
    assert full_df.loc[1, "MD_U"] == 0.0
    assert full_df.loc[1, "MD_U_isna"] == 1.0
    # Row 0 had 1250.0 / 250.0 -> ID_U=1250.0, ID_U_isna=0.0
    assert full_df.loc[0, "ID_U"] == 1250.0
    assert full_df.loc[0, "ID_U_isna"] == 0.0

    # 4. Comorb_Count is integer
    assert np.issubdtype(full_df["Comorb_Count"].dtype, np.integer)


def test_deterministic_pruning_unit() -> None:
    """Test deterministic pruning on synthetic records including BUN inversion."""
    df_test = pd.DataFrame(
        {
            "Kt_V": [0.4, 1.2, 1.3, 1.4, 1.5, 1.6, 0.8, 1.3],
            "nPCR": [1.0, 0.3, 1.1, 1.2, 1.3, 1.4, 1.1, 1.1],
            "before_hd_weight": [60.0, 60.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0],
            "after_hd_weight": [58.0, 58.0, 51.0, 53.0, 58.0, 58.0, 58.0, 58.0],
            "BFR": [200, 200, 200, 200, 0, 200, 200, 200],
            "DFR": [500, 500, 500, 500, 500, 0, 500, 500],
            "before_hd_BUN": [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
            "after_hd_BUN": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 65.0, 65.0],
        }
    )
    # Row 0: Kt_V < 0.5 (bad)
    # Row 1: nPCR < 0.4 (bad)
    # Row 2: Weight gain > 0.5 (after - before = 1.0 > 0.5) (bad)
    # Row 3: Weight loss > 6.0 (before - after = 7.0 > 6.0) (bad)
    # Row 4: BFR == 0 (bad)
    # Row 5: DFR == 0 (bad)
    # Row 6: BUN inversion with Kt_V < 1.0 (unverified inversion, bad)
    # Row 7: BUN inversion with Kt_V >= 1.0 (verified inversion, retained)
    pruned = deterministic_pruning(df_test)
    assert len(pruned) == 1
    assert pruned.index[0] == 7


def test_extreme_hypokalemia_flag(sample_raw_dataframe: pd.DataFrame) -> None:
    """Test extreme hypokalemia (K < 2.5) warning flag and retention (Spec 42)."""
    df_eng = engineer_features(sample_raw_dataframe)
    assert "is_extreme_hypokalemia" in df_eng.columns
    # Row 1 has K = 2.2 < 2.5 -> flagged as 1
    assert df_eng.loc[1, "is_extreme_hypokalemia"] == 1
    # Row 0 has K = 3.5 -> 0
    assert df_eng.loc[0, "is_extreme_hypokalemia"] == 0

    # Ensure deterministic pruning does not drop row 1 due to K < 2.5
    df_single = pd.DataFrame(
        {
            "K": [2.2],
            "Kt_V": [1.4],
            "nPCR": [1.1],
            "before_hd_weight": [60.0],
            "after_hd_weight": [58.0],
            "BFR": [250],
            "DFR": [500],
        }
    )
    pruned = deterministic_pruning(df_single)
    assert len(pruned) == 1


def test_statistical_outlier_functions() -> None:
    """Test Hotelling's T^2, SPE, and Jackson-Mudholkar limits."""
    np.random.seed(42)
    n_samples, n_features = 200, 10
    x = np.random.randn(n_samples, n_features)

    pca, scores, _ = fit_pca(pd.DataFrame(x), n_components=5)
    recon = pca.inverse_transform(scores.values)

    t2 = calculate_hotelling_t2(scores.values, pca.explained_variance_)
    spe = calculate_spe(x, recon)

    assert len(t2) == n_samples
    assert len(spe) == n_samples
    assert (t2 >= 0).all()
    assert (spe >= 0).all()

    t2_lim = calculate_t2_limit(n_samples=n_samples, n_components=5, alpha=0.01)
    assert t2_lim > 0

    full_pca = fit_pca(pd.DataFrame(x), n_components=n_features)[0]
    res_eig = full_pca.explained_variance_[5:]
    spe_lim = calculate_spe_limit(res_eig, alpha=0.01)
    assert spe_lim > 0

    # Test refactored prune_multivariate_outliers (Feature Envy resolved)
    is_outlier, eff_t2, eff_spe = prune_multivariate_outliers(
        scores=scores,
        spe=spe,
        explained_variance=pca.explained_variance_,
        residual_eigenvalues=res_eig,
        alpha=0.01,
    )
    assert len(is_outlier) == n_samples
    assert is_outlier.dtype == bool
    assert eff_t2 > 0
    assert eff_spe > 0


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Registry workbook not available")
def test_longitudinal_cohort_and_split() -> None:
    """Test longitudinal rolling 3-year dynamic cohort and patient group splitting."""
    cohort_df = load_longitudinal_cohort(DATA_PATH)

    # 1. Target dynamic cohort count: 5,141 records across 1,165 patients
    assert len(cohort_df) == 5141
    assert cohort_df["PatientID"].nunique() == 1165

    # 2. Mortality distribution: 969 deceased, 4,172 survived (18.85%)
    assert cohort_df["is_death_3yr"].sum() == 969
    assert (cohort_df["is_death_3yr"] == 0).sum() == 4172

    # 3. Patient Group Split: Train 4,029 (932 patients), Test 1,112 (233 patients)
    train_df, test_df = split_patient_cohort(
        cohort_df, train_size=0.80, random_state=42
    )

    assert len(train_df) == 4029
    assert train_df["PatientID"].nunique() == 932

    assert len(test_df) == 1112
    assert test_df["PatientID"].nunique() == 233

    # Anti-leakage: zero overlap between Train and Test patients
    train_patients = set(train_df["PatientID"])
    test_patients = set(test_df["PatientID"])
    assert len(train_patients.intersection(test_patients)) == 0


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Registry workbook not available")
def test_full_pipeline_train_prune_and_test_invariance() -> None:
    """Test Spec line 54 sequence: split -> engineer -> prune -> fit -> test."""
    cohort_df = load_longitudinal_cohort(DATA_PATH)
    train_raw, test_raw = split_patient_cohort(
        cohort_df, train_size=0.80, random_state=42
    )

    # 1. Feature Engineering
    train_eng = engineer_features(train_raw)
    test_eng = engineer_features(test_raw)

    # 2. Deterministic Pruning on Train
    train_det = deterministic_pruning(train_eng)
    assert len(train_det) == 4007  # Drops 22 records

    # 3. Type-Specific Imputer Fit & Transform
    preprocessor = LongitudinalPreprocessor()
    preprocessor.fit(train_det)

    cont_train, full_train = preprocessor.transform(train_det)
    cont_test, full_test = preprocessor.transform(test_eng)

    # Test set invariance: ZERO sample deletions (N=1,112)
    assert len(full_test) == 1112
    assert full_test.isna().sum().sum() == 0

    # Discrete count columns must be strictly integers
    for disc_col in DISCRETE_COUNT_COLUMNS:
        if disc_col in full_train.columns:
            assert np.issubdtype(full_train[disc_col].dtype, np.integer)
        if disc_col in full_test.columns:
            assert np.issubdtype(full_test[disc_col].dtype, np.integer)

    # Invariant 2 (Zero Mode Imputation): Non-mode imputation assertions
    assert "genAssess" in full_train.columns
    assert "actAssess" in full_train.columns
    assert full_train["genAssess"].isna().sum() == 0
    assert full_test["genAssess"].isna().sum() == 0
    assert full_train["actAssess"].isna().sum() == 0
    assert full_test["actAssess"].isna().sum() == 0

    assert any(c.startswith("HD_WS_nan") for c in full_train.columns)
    assert any(c.startswith("CIC_nan") for c in full_train.columns)
    for cat_col in ["HD_WS", "HD_ST", "CIC", "total_hd_time", "txIntervalMin"]:
        assert cat_col not in cont_train.columns
        assert cat_col not in cont_test.columns

    assert "ID_U_isna" in full_train.columns and "MD_U_isna" in full_train.columns
    assert "ID_U_isna" in full_test.columns and "MD_U_isna" in full_test.columns
    assert full_train["ID_U"].isna().sum() == 0
    assert full_test["ID_U"].isna().sum() == 0

    # 4. PCA & Multivariate Outlier Pruning with genuine Jackson-Mudholkar
    pca, scores, _ = fit_pca(cont_train, n_components=5)
    recon = pca.inverse_transform(scores.values)
    t2 = calculate_hotelling_t2(scores.values, pca.explained_variance_)
    spe = calculate_spe(cont_train.values, recon)

    pca_full = fit_pca(cont_train, n_components=cont_train.shape[1])[0]
    res_eig = pca_full.explained_variance_[5:]

    is_outlier, t2_lim, spe_lim = prune_multivariate_outliers(
        scores=scores,
        spe=spe,
        t2=t2,
        explained_variance=pca.explained_variance_,
        residual_eigenvalues=res_eig,
        alpha=0.01,
    )

    clean_mask = ~is_outlier
    train_clean = train_det[clean_mask]
    cont_clean = cont_train[clean_mask]

    assert len(train_clean) == 3737
    assert np.isclose(t2_lim, 15.124, atol=0.01)
    assert np.isclose(spe_lim, 43.722, atol=0.01)

    # Verify that remaining clean records satisfy control limits
    clean_scores = scores[clean_mask]
    recon_clean = pca.inverse_transform(clean_scores.values)
    clean_t2 = calculate_hotelling_t2(clean_scores.values, pca.explained_variance_)
    clean_spe = calculate_spe(cont_clean.values, recon_clean)

    assert (clean_t2 <= t2_lim).all()
    assert (clean_spe <= spe_lim).all()


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Registry workbook not available")
def test_anti_leakage_statistics() -> None:
    """Verify test set statistics do not match imputer/scaler parameters (Spec 83)."""
    cohort_df = load_longitudinal_cohort(DATA_PATH)
    train_raw, test_raw = split_patient_cohort(
        cohort_df, train_size=0.80, random_state=42
    )

    train_eng = engineer_features(train_raw)
    test_eng = engineer_features(test_raw)
    train_det = deterministic_pruning(train_eng)

    preprocessor = LongitudinalPreprocessor()
    preprocessor.fit(train_det)

    retained_cols = preprocessor.retained_cont_cols
    test_medians = test_eng[retained_cols].median().values
    train_medians = train_det[retained_cols].median().values
    fitted_imputer_medians = preprocessor.cont_imputer.statistics_

    # Fitted parameters must match training data and diverge from test data
    assert np.allclose(train_medians, fitted_imputer_medians)
    assert not np.allclose(test_medians, fitted_imputer_medians)

    # Test scaler parameters: scaler mean matches train and diverges from test
    fitted_scaler_mean = preprocessor.scaler.mean_
    test_means = test_eng[retained_cols].mean().values
    assert not np.allclose(test_means, fitted_scaler_mean)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Registry workbook not available")
def test_pruning_boundary_verification() -> None:
    """Assert cleaned train set has no Kt/V < 0.5, UF > 6.0, or BFR == 0 (Spec 87)."""
    cohort_df = load_longitudinal_cohort(DATA_PATH)
    train_raw, test_raw = split_patient_cohort(
        cohort_df, train_size=0.80, random_state=42
    )

    train_eng = engineer_features(train_raw)
    train_det = deterministic_pruning(train_eng)

    # Cleaned training set must have zero boundary violations:
    assert (train_det["Kt_V"] < 0.5).sum() == 0
    assert (train_det["ultrafiltration"] > 6.0).sum() == 0
    assert (train_det["BFR"] == 0).sum() == 0

    # In contrast, raw train set contained violations that were pruned
    assert ((train_raw["Kt_V"] < 0.5) & train_raw["Kt_V"].notna()).sum() > 0
    assert ((train_raw["BFR"] == 0) & train_raw["BFR"].notna()).sum() > 0


def test_visualization_t2_vs_spe() -> None:
    """Test plot_t2_vs_spe generation."""
    np.random.seed(42)
    t2 = np.random.uniform(0, 20, 100)
    spe = np.random.uniform(0, 80, 100)
    fig = plot_t2_vs_spe(t2, spe, t2_limit=15.124, spe_limit=55.0)
    assert fig is not None
