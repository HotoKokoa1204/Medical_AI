"""Preprocessing and feature engineering for hemodialysis baseline records.

Follows AGILAB coding standards with Google-style docstrings and type annotations.
Implements clinical sanity checks, target leakage quarantine, collinearity decoupling,
non-random missingness encoding, and standardized transformations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Excluded target leakage and future follow-up columns
TARGET_LEAKAGE_COLUMNS: list[str] = [
    "死亡日期",
    "死亡原因大類",
    "死亡原因細類",
    "是否離開本院",
    "離開途徑",
    "離開年度",
    "轉院原因",
    "轉至何院所",
    "最終醫療狀態",
    "腎移植日期",
    "出現年度數",
    "最後出現年度",
]

# Non-predictive identification columns
IDENTIFIER_COLUMNS: list[str] = [
    "PatientID",
    "year",
    "index_date",
    "首次出現年度",
]

# Primary prediction label
LABEL_COLUMN: str = "is_death"

# Features with >20% missingness to be discarded per specification
HIGH_MISSING_DROPS: list[str] = [
    "CaXP",
    "totalCa",
    "ionizedCa",
    "AVF_L",
    "AVG_L",
    "PC_CathL",
    "HcvRna",
    "EPOType",
]

# Heavily right-skewed biomarkers requiring log1p transformation
LOG_TRANSFORM_COLUMNS: list[str] = [
    "PTH",
    "Ferritin",
    "Triglyceride",
]

# Default candidate continuous features (<= 20% missing rate)
DEFAULT_CONTINUOUS_COLUMNS: list[str] = [
    "age",
    "before_hd_weight",
    "ultrafiltration",
    "uf_ratio",
    "dialysis_vintage_years",
    "Kt_V",
    "Kt_V_Gotch",
    "URR_auto",
    "nPCR",
    "TACurea",
    "Creatinine",
    "Albumin",
    "Hbc",
    "Hct",
    "MCV",
    "Platelet",
    "WBC",
    "RBC",
    "Total_protein",
    "AST",
    "ALT",
    "Alkaline_P",
    "Total_Bilirubin",
    "Cholesterol",
    "Triglyceride",
    "Glucose_AC",
    "total_hd_time",
    "before_hd_BUN",
    "after_hd_BUN",
    "nextTxPreBUN",
    "txIntervalMin",
    "Fe",
    "TIBC",
    "Ferritin",
    "Al",
    "Uric_acid",
    "Na",
    "K",
    "P",
    "PTH",
    "Tranferrin_saturation",
    "BFR",
    "DFR",
    "HD_WS",
    "HD_ST",
    "HD_MSA",
    "ID_U",
    "MD_U",
    "CIC",
    "genAssess",
    "actAssess",
    "Comorb_Count",
]

# Comorbidity binary flags (0 missing)
COMORBIDITY_COLUMNS: list[str] = [
    "CHF",
    "MI",
    "LVH",
    "Stroke",
    "PTX",
    "Hepatitis_LC",
    "Cirrhosis",
    "Cancer",
    "TB",
    "GI_bleeding",
    "Neuropathy",
    "Renal_osteodystrophy",
    "Carpal_tunnel",
    "Asthma",
    "Uremic_derm",
    "Hydrothorax",
    "Ascites",
    "Cachexia",
    "Hypertension",
    "CAD",
    "Cardiomyopathy",
    "DM",
    "COPD",
    "GERD",
    "Other_comorbidity",
    "Hemiplegia",
    "AIDS",
    "Metastasis",
    "Gout",
    "Hyperlipidemia",
    "Dementia",
    "Chemo_treatment",
    "Nonrenal_anemia",
]


def load_baseline_data(
    filepath: Path | str, sheet_name: int | str = 1, header: int = 2
) -> pd.DataFrame:
    """Load baseline patient cohort from the Excel registry workbook.

    Args:
        filepath: Filepath to Kidit_Master_Baseline_V2.xlsx.
        sheet_name: Target worksheet name or index (default 1 for 01_基準年總表).
        header: 0-indexed row number containing headers (default 2 for row 3).

    Returns:
        pd.DataFrame containing the raw baseline cohort.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at {filepath}")
    return pd.read_excel(path, sheet_name=sheet_name, header=header)


def quarantine_target_leakage(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Segregate post-baseline outcome and tracking columns to prevent leakage.

    Args:
        df: Raw baseline DataFrame.

    Returns:
        A tuple of (candidate_features_df, quarantined_leakage_df, target_series).
    """
    if LABEL_COLUMN in df.columns:
        target = df[LABEL_COLUMN].copy()
    else:
        target = pd.Series(dtype=int)

    leakage_present = [c for c in TARGET_LEAKAGE_COLUMNS if c in df.columns]
    leakage_df = df[leakage_present].copy()

    drop_cols = set(leakage_present + IDENTIFIER_COLUMNS + [LABEL_COLUMN])
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    candidate_features = df.drop(columns=cols_to_drop).copy()

    return candidate_features, leakage_df, target


def clean_clinical_bounds(df: pd.DataFrame, ktv_min: float = 0.5) -> pd.DataFrame:
    """Filter out software calculation artifacts and physiological errors.

    Specifically sets Kt/V < 0.5 to NaN for subsequent median imputation,
    resolving known LIS export artifacts where formula failures recorded values
    such as 0.04 despite concurrent 80% urea clearance.

    Args:
        df: DataFrame containing clinical laboratory features.
        ktv_min: Minimum physiologically plausible Kt/V value (default 0.5).

    Returns:
        Cleaned pd.DataFrame with out-of-bound Kt/V converted to NaN.
    """
    df_clean = df.copy()
    if "Kt_V" in df_clean.columns:
        invalid_mask = df_clean["Kt_V"] < ktv_min
        df_clean.loc[invalid_mask, "Kt_V"] = np.nan
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform clinical feature engineering and collinearity decoupling.

    1. Dialysis Vintage: Computes accumulated dialysis vintage in fractional years:
       vintage = (index_date - First_HD_date) / 365.25.
       When First_HD_date is missing, SKH_hd_date is used as fallback.
       Values are clipped to >= 0.0.
    2. Weight Decoupling: Retains before_hd_weight, derives ultrafiltration
       (before_hd_weight - after_hd_weight) and uf_ratio,
       and drops after_hd_weight to reduce collinearity from r=0.992 to r<0.2.

    Args:
        df: DataFrame containing raw baseline and date fields.

    Returns:
        DataFrame with engineered features added and collinear fields removed.
    """
    df_eng = df.copy()

    # Time derivation: Dialysis Vintage
    if "index_date" in df_eng.columns:
        index_date = pd.to_datetime(df_eng["index_date"], errors="coerce")
        first_hd = (
            pd.to_datetime(df_eng["First_HD_date"], errors="coerce")
            if "First_HD_date" in df_eng.columns
            else pd.Series(pd.NaT, index=df_eng.index)
        )
        skh_hd = (
            pd.to_datetime(df_eng["SKH_hd_date"], errors="coerce")
            if "SKH_hd_date" in df_eng.columns
            else pd.Series(pd.NaT, index=df_eng.index)
        )

        effective_start = first_hd.fillna(skh_hd)
        vintage_days = (index_date - effective_start).dt.total_seconds() / 86400.0
        vintage_years = vintage_days / 365.25
        df_eng["dialysis_vintage_years"] = vintage_years.clip(lower=0.0)

    # Weight Decoupling
    if "before_hd_weight" in df_eng.columns and "after_hd_weight" in df_eng.columns:
        uf = df_eng["before_hd_weight"] - df_eng["after_hd_weight"]
        uf_ratio = uf / df_eng["before_hd_weight"].replace(0, np.nan)
        df_eng["ultrafiltration"] = uf
        df_eng["uf_ratio"] = uf_ratio
        df_eng = df_eng.drop(columns=["after_hd_weight"])

    # Drop raw date objects after derivation
    for date_col in ["First_HD_date", "SKH_hd_date"]:
        if date_col in df_eng.columns:
            df_eng = df_eng.drop(columns=[date_col])

    return df_eng


def process_continuous_features(
    df: pd.DataFrame,
    continuous_columns: Sequence[str] | None = None,
    missing_threshold: float = 0.20,
) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler, list[str]]:
    """Clean, impute, log-transform, and scale continuous laboratory features.

    Args:
        df: DataFrame containing continuous clinical features.
        continuous_columns: List of candidate continuous column names.
            If None, defaults to DEFAULT_CONTINUOUS_COLUMNS.
        missing_threshold: Maximum allowable missingness proportion (default 0.20).

    Returns:
        A tuple of:
            - scaled_df: pd.DataFrame with scaled features (mean=0, std=1).
            - imputer: Fitted SimpleImputer object.
            - scaler: Fitted StandardScaler object.
            - retained_columns: List of columns retained after threshold filtering.
    """
    candidates = (
        list(continuous_columns)
        if continuous_columns is not None
        else [c for c in DEFAULT_CONTINUOUS_COLUMNS if c in df.columns]
    )

    # Filter out columns exceeding missing threshold or explicitly dropped
    retained_cols: list[str] = []
    for col in candidates:
        if col in HIGH_MISSING_DROPS:
            continue
        if col in df.columns:
            missing_rate = df[col].isna().mean()
            if missing_rate <= missing_threshold:
                retained_cols.append(col)

    subset = df[retained_cols].copy()

    # Median Imputation
    imputer = SimpleImputer(strategy="median")
    imputed_arr = imputer.fit_transform(subset)
    imputed_df = pd.DataFrame(imputed_arr, columns=retained_cols, index=df.index)

    # Log1p transformation for heavily right-skewed biomarkers
    for col in LOG_TRANSFORM_COLUMNS:
        if col in imputed_df.columns:
            imputed_df[col] = np.log1p(imputed_df[col].clip(lower=0))

    # Z-Score Standard Scaling
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(imputed_df)
    scaled_df = pd.DataFrame(scaled_arr, columns=retained_cols, index=df.index)

    return scaled_df, imputer, scaler, retained_cols


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize binary labels and One-Hot encode categorical features.

    Binary mapping:
        Chinese: '是' -> 1, '否' -> 0.
        English: 'Y' -> 1, 'N' -> 0.
        Hepatitis: 'Y' -> 1, 'N' -> 0, 'O' -> NaN.
    Multi-class with dummy_na:
        blood_type, education, marriage, Dialysis_type, HBsAg, Anti_HCV, sex.

    Args:
        df: DataFrame containing raw categorical features.

    Returns:
        pd.DataFrame containing standardized binary and one-hot encoded features.
    """
    df_cat = df.copy()

    chinese_binary_cols = [
        "isEPO",
        "VitD",
        "Antihypert",
        "IronSupp",
        "CKD",
        "ParaTx",
    ]
    chinese_map = {"是": 1.0, "否": 0.0, 1: 1.0, 0: 0.0}

    english_binary_cols = [
        "AVF",
        "AVG",
        "PC_Cath",
        "ST_Cath",
        "Hx_HD",
        "Hx_PD",
    ]
    english_map = {"Y": 1.0, "N": 0.0, "y": 1.0, "n": 0.0, 1: 1.0, 0: 0.0}

    for col in chinese_binary_cols:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].map(chinese_map)

    for col in english_binary_cols:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].map(english_map)

    for col in ["HBsAg", "Anti_HCV"]:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].replace({"O": np.nan, "o": np.nan})
            df_cat[col] = df_cat[col].map(english_map)

    one_hot_cols = [
        col
        for col in [
            "blood_type",
            "education",
            "marriage",
            "Dialysis_type",
            "sex",
            "HBsAg",
            "Anti_HCV",
            "isEPO",
            "AVF",
            "AVG",
            "PC_Cath",
            "ST_Cath",
            "ParaTx",
            "VitD",
            "Antihypert",
            "IronSupp",
            "CKD",
            "Hx_HD",
            "Hx_PD",
        ]
        if col in df_cat.columns
    ]

    dummies = pd.get_dummies(
        df_cat[one_hot_cols],
        columns=one_hot_cols,
        dummy_na=True,
        dtype=float,
    )

    comorb_present = [col for col in COMORBIDITY_COLUMNS if col in df_cat.columns]
    if comorb_present:
        comorb_df = df_cat[comorb_present].astype(float)
    else:
        comorb_df = pd.DataFrame(index=df.index)

    return pd.concat([dummies, comorb_df], axis=1)


def build_feature_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Execute end-to-end preprocessing to build PCA and full feature matrices.

    Args:
        df: Raw baseline DataFrame.

    Returns:
        A tuple of:
            - continuous_features_pca: Scaled continuous feature DataFrame.
            - full_feature_matrix: Complete feature matrix.
            - target: Outcome label Series (is_death).
    """
    candidate_df, _, target = quarantine_target_leakage(df)
    cleaned_df = clean_clinical_bounds(candidate_df)
    engineered_df = engineer_features(cleaned_df)
    scaled_cont_df, _, _, _ = process_continuous_features(engineered_df)
    encoded_cat_df = encode_categorical_features(engineered_df)

    full_feature_matrix = pd.concat([scaled_cont_df, encoded_cat_df], axis=1)
    full_feature_matrix[LABEL_COLUMN] = target.values

    return scaled_cont_df, full_feature_matrix, target
