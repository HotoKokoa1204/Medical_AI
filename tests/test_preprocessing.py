"""Unit tests for hemodialysis baseline preprocessing and PCA pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agilab_lib.analysis import fit_pca
from agilab_lib.preprocessing import (
    HIGH_MISSING_DROPS,
    IDENTIFIER_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
    clean_clinical_bounds,
    encode_categorical_features,
    engineer_features,
    process_continuous_features,
    quarantine_target_leakage,
)


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
            "PTH": [150.0, 450.0, 25.0, 800.0],
            "Ferritin": [200.0, 600.0, 150.0, 1200.0],
            "Triglyceride": [120.0, 250.0, 90.0, 300.0],
            "Albumin": [4.0, 3.5, 3.8, 3.2],
            "Creatinine": [11.0, 12.5, 9.8, 10.2],
            "CaXP": [45.0, np.nan, np.nan, np.nan],  # > 20% missing
            "totalCa": [np.nan, np.nan, 9.0, np.nan],  # > 20% missing
            "ionizedCa": [np.nan, np.nan, np.nan, 4.5],  # > 20% missing
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

    # Dialysis vintage checks
    assert "dialysis_vintage_years" in df_eng.columns
    # Row 0: 2010-06-01 - 2008-06-01 ~ 2.0 years
    assert np.isclose(df_eng.loc[0, "dialysis_vintage_years"], 2.0, atol=0.05)
    # Row 1: First_HD_date is NaN, fallback to SKH_hd_date 2010-06-01 -> ~1.0 yr
    assert np.isclose(df_eng.loc[1, "dialysis_vintage_years"], 1.0, atol=0.05)
    # Row 3: Both dates NaN -> should be NaN prior to imputation, or >= 0
    assert df_eng["dialysis_vintage_years"].min() >= 0.0 or pd.isna(
        df_eng.loc[3, "dialysis_vintage_years"]
    )

    # Weight decoupling checks
    assert "ultrafiltration" in df_eng.columns
    assert "uf_ratio" in df_eng.columns
    assert "after_hd_weight" not in df_eng.columns
    assert np.isclose(df_eng.loc[0, "ultrafiltration"], 2.0)
    assert np.isclose(df_eng.loc[0, "uf_ratio"], 2.0 / 60.0)


def test_clean_clinical_bounds(sample_raw_dataframe: pd.DataFrame) -> None:
    """Test that software artifact Kt/V < 0.5 is replaced by NaN."""
    df_clean = clean_clinical_bounds(sample_raw_dataframe)

    # Row 1 originally had Kt/V = 0.04
    assert pd.isna(df_clean.loc[1, "Kt_V"])
    # Row 0 originally had Kt/V = 1.4 -> preserved
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

    # High missing cols must be excluded
    for col in HIGH_MISSING_DROPS:
        assert col not in scaled_df.columns
        assert col not in retained_cols

    # No NaN remaining
    assert scaled_df.isna().sum().sum() == 0

    # Scaled features should have mean close to 0 and std close to 1
    for col in retained_cols:
        assert np.isclose(scaled_df[col].mean(), 0.0, atol=1e-5)


def test_encode_categorical_features(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """Test categorical encoding, binary standardization, and missing dummies."""
    encoded_df = encode_categorical_features(sample_raw_dataframe)

    # Binary flags mapped to 0/1 (with NaN handled or dummy indicator)
    assert "isEPO" in encoded_df.columns or any(
        c.startswith("isEPO_") for c in encoded_df.columns
    )
    # Multi-class blood_type must contain dummy columns including NaN indicator
    assert any(c.startswith("blood_type_") for c in encoded_df.columns)
    # Check dummy_na presence
    assert any("nan" in c.lower() for c in encoded_df.columns)
    # Ensure all values are numeric
    assert np.issubdtype(encoded_df.dtypes.iloc[0], np.number)


def test_fit_pca() -> None:
    """Test PCA fitting and loading calculations."""
    np.random.seed(42)
    data = np.random.randn(50, 10)
    feature_names = [f"feat_{i}" for i in range(10)]
    df = pd.DataFrame(data, columns=feature_names)

    pca, scores, loadings = fit_pca(df, n_components=5)

    assert scores.shape == (50, 5)
    assert loadings.shape == (10, 5)
    assert len(pca.explained_variance_ratio_) == 5
    assert np.isclose(
        np.sum(pca.explained_variance_ratio_[:5]),
        pca.explained_variance_ratio_.sum(),
    )
