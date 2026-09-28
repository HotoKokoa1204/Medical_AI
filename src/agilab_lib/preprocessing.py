"""Module: preprocessing
Stage: Library
Author: KafuuChino
Date: 2026-09-28
Description: Preprocessing for hemodialysis rolling 3-year mortality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "報告年度",
    "首次出現年度",
    "醫療狀態",
]

# Primary prediction labels
LABEL_COLUMN: str = "is_death"
LABEL_COLUMN_3YR: str = "is_death_3yr"

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

# Discrete count / schedule features to be imputed via integer mode
DISCRETE_COUNT_COLUMNS: list[str] = [
    "HD_WS",
    "HD_ST",
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

# Categorical features for One-Hot Encoding
CATEGORICAL_COLUMNS: list[str] = [
    "blood_type",
    "education",
    "marriage",
    "Dialysis_type",
    "AC",
    "Base",
    "isEPO",
    "AVF",
    "AVG",
    "PC_Cath",
    "ST_Cath",
    "Hx_HD",
    "Hx_PD",
    "CKD",
    "ParaTx",
    "VitD",
    "Antihypert",
    "IronSupp",
    "HBsAg",
    "Anti_HCV",
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

    Raises:
        FileNotFoundError: If the workbook does not exist at filepath.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at {filepath}")
    return pd.read_excel(path, sheet_name=sheet_name, header=header)


def load_longitudinal_cohort(
    filepath: Path | str,
    panel_sheet: str = "02_逐年面板",
    outcomes_sheet: str = "03_結局與去向",
    header: int = 2,
) -> pd.DataFrame:
    """Load longitudinal panel and construct rolling 3-year mortality cohort.

    1. Loads '02_逐年面板' and merges with '03_結局與去向' on 'PatientID'.
    2. Calculates days to death: (death_date - index_date).dt.days.
    3. Derives rolling 3-year mortality label:
       - 1: death occurred within [0, 1095] days.
       - 0: confirmed survival past 3 years (最後出現年度 >= 報告年度 + 3).
    4. Excludes observations with unconfirmed < 3-year follow-up without death.

    Args:
        filepath: Filepath to Kidit_Master_Baseline_V2.xlsx.
        panel_sheet: Sheet name for longitudinal annual panel.
        outcomes_sheet: Sheet name for outcomes and tracking.
        header: 0-indexed header row number (default 2).

    Returns:
        pd.DataFrame containing 5,141 valid dynamic longitudinal observations.

    Raises:
        FileNotFoundError: If the workbook is not found.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at {filepath}")

    df_panel = pd.read_excel(path, sheet_name=panel_sheet, header=header)
    df_outcomes = pd.read_excel(path, sheet_name=outcomes_sheet, header=header)

    merged = pd.merge(df_panel, df_outcomes, on="PatientID", how="left")

    death_date = pd.to_datetime(merged["死亡日期"], errors="coerce")
    index_date = pd.to_datetime(merged["index_date"], errors="coerce")
    days_to_death = (death_date - index_date).dt.days

    year_col = "報告年度" if "報告年度" in merged.columns else "year"
    last_year_col = "最後出現年度"

    is_death_3yr = (days_to_death >= 0) & (days_to_death <= 1095)
    confirmed_surv = (~is_death_3yr) & (merged[last_year_col] >= merged[year_col] + 3)

    valid_mask = is_death_3yr | confirmed_surv
    cohort_df = merged[valid_mask].copy()
    cohort_df[LABEL_COLUMN_3YR] = is_death_3yr[valid_mask].astype(int)

    return cohort_df


def split_patient_cohort(
    df: pd.DataFrame,
    train_size: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition longitudinal records strictly by PatientID to prevent leakage.

    Args:
        df: Cohort DataFrame containing a 'PatientID' column.
        train_size: Proportion of patient groups allocated to training (default 0.80).
        random_state: Random state seed for reproducibility (default 42).

    Returns:
        A tuple of (train_df, test_df).
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(gss.split(df, groups=df["PatientID"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df


def deterministic_pruning(df: pd.DataFrame) -> pd.DataFrame:
    """Prune deterministic software calculation errors and entry contradictions.

    Prunes rows satisfying any of:
        - Kt/V < 0.5
        - nPCR < 0.4
        - Weight gain: (after_hd_weight - before_hd_weight) > 0.5 kg
        - Weight loss: (before_hd_weight - after_hd_weight) > 6.0 kg
        - BFR < 100 or BFR == 0
        - DFR < 300 or DFR == 0

    Args:
        df: Input training DataFrame.

    Returns:
        pd.DataFrame with deterministic errors removed.
    """
    bad_mask = pd.Series(False, index=df.index)

    if "Kt_V" in df.columns:
        bad_mask |= df["Kt_V"] < 0.5

    if "nPCR" in df.columns:
        bad_mask |= df["nPCR"] < 0.4

    if "before_hd_weight" in df.columns and "after_hd_weight" in df.columns:
        weight_gain = df["after_hd_weight"] - df["before_hd_weight"]
        weight_loss = df["before_hd_weight"] - df["after_hd_weight"]
        bad_mask |= weight_gain > 0.5
        bad_mask |= weight_loss > 6.0
    elif "ultrafiltration" in df.columns:
        bad_mask |= df["ultrafiltration"] < -0.5
        bad_mask |= df["ultrafiltration"] > 6.0

    if "BFR" in df.columns:
        bad_mask |= (df["BFR"] < 100) | (df["BFR"] == 0)

    if "DFR" in df.columns:
        bad_mask |= (df["DFR"] < 300) | (df["DFR"] == 0)

    return df[~bad_mask].copy()


def clean_clinical_bounds(df: pd.DataFrame, ktv_min: float = 0.5) -> pd.DataFrame:
    """Filter out software calculation artifacts and physiological errors.

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
       (before_hd_weight - after_hd_weight) with micro-gain (<= 0.5kg) clamped to 0.0,
       derives uf_ratio, and drops after_hd_weight.
    3. Drops raw date objects after derivation.

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
        uf_clamped = uf.copy()
        micro_gain = (uf < 0) & (uf >= -0.5)
        uf_clamped.loc[micro_gain] = 0.0

        uf_ratio = uf_clamped / df_eng["before_hd_weight"].replace(0, np.nan)
        df_eng["ultrafiltration"] = uf_clamped
        df_eng["uf_ratio"] = uf_ratio
        df_eng = df_eng.drop(columns=["after_hd_weight"])

    # Drop raw date objects after derivation
    for date_col in ["First_HD_date", "SKH_hd_date"]:
        if date_col in df_eng.columns:
            df_eng = df_eng.drop(columns=[date_col])

    return df_eng


def quarantine_target_leakage(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Segregate post-baseline outcome and tracking columns to prevent leakage.

    Args:
        df: Raw or engineered DataFrame.

    Returns:
        A tuple of (candidate_features_df, quarantined_leakage_df, target_series).
    """
    if LABEL_COLUMN_3YR in df.columns:
        target = df[LABEL_COLUMN_3YR].copy()
    elif LABEL_COLUMN in df.columns:
        target = df[LABEL_COLUMN].copy()
    else:
        target = pd.Series(dtype=int)

    leakage_present = [c for c in TARGET_LEAKAGE_COLUMNS if c in df.columns]
    leakage_df = df[leakage_present].copy()

    drop_cols = set(
        leakage_present + IDENTIFIER_COLUMNS + [LABEL_COLUMN, LABEL_COLUMN_3YR]
    )
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    candidate_features = df.drop(columns=cols_to_drop).copy()

    return candidate_features, leakage_df, target


class LongitudinalPreprocessor:
    """Type-specific preprocessor fitted strictly on training data."""

    def __init__(
        self,
        continuous_cols: Sequence[str] | None = None,
        discrete_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
        missing_threshold: float = 0.20,
    ) -> None:
        """Initialize preprocessor with feature types.

        Args:
            continuous_cols: List of candidate continuous column names.
            discrete_cols: List of discrete count column names.
            categorical_cols: List of categorical column names.
            missing_threshold: Maximum allowed missing rate for continuous features.
        """
        self.continuous_cols_input = (
            list(continuous_cols) if continuous_cols is not None else None
        )
        self.discrete_cols_input = (
            list(discrete_cols)
            if discrete_cols is not None
            else list(DISCRETE_COUNT_COLUMNS)
        )
        self.categorical_cols_input = (
            list(categorical_cols)
            if categorical_cols is not None
            else list(CATEGORICAL_COLUMNS)
        )
        self.missing_threshold = missing_threshold

        self.retained_cont_cols: list[str] = []
        self.retained_disc_cols: list[str] = []
        self.retained_cat_cols: list[str] = []

        self.cont_imputer: SimpleImputer = SimpleImputer(strategy="median")
        self.scaler: StandardScaler = StandardScaler()
        self.discrete_modes: dict[str, int] = {}
        self.sex_mode: float = 1.0
        self.ohe: OneHotEncoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> LongitudinalPreprocessor:
        """Fit imputation and scaling parameters strictly on training split.

        Args:
            df: Training DataFrame (after feature engineering).

        Returns:
            Fitted LongitudinalPreprocessor instance.
        """
        candidates = (
            self.continuous_cols_input
            if self.continuous_cols_input is not None
            else [c for c in DEFAULT_CONTINUOUS_COLUMNS if c in df.columns]
        )

        self.retained_cont_cols = [
            c
            for c in candidates
            if c in df.columns
            and c not in HIGH_MISSING_DROPS
            and df[c].isna().mean() <= self.missing_threshold
        ]

        # 1. Continuous Features: Median imputer & StandardScaler
        cont_subset = df[self.retained_cont_cols].copy()
        cont_imputed = self.cont_imputer.fit_transform(cont_subset)
        cont_imp_df = pd.DataFrame(
            cont_imputed, columns=self.retained_cont_cols, index=df.index
        )

        for col in LOG_TRANSFORM_COLUMNS:
            if col in cont_imp_df.columns:
                cont_imp_df[col] = np.log1p(cont_imp_df[col].clip(lower=0))

        self.scaler.fit(cont_imp_df)

        # 2. Discrete Features: Integer mode
        self.retained_disc_cols = [
            c for c in self.discrete_cols_input if c in df.columns
        ]
        for col in self.retained_disc_cols:
            non_null = df[col].dropna()
            mode_val = int(non_null.mode().iloc[0]) if not non_null.empty else 0
            self.discrete_modes[col] = mode_val

        # 3. Sex: Mode imputation
        if "sex" in df.columns:
            non_null_sex = df["sex"].dropna()
            self.sex_mode = (
                float(non_null_sex.mode().iloc[0]) if not non_null_sex.empty else 1.0
            )

        # 4. Categorical Features: OneHotEncoder
        self.retained_cat_cols = [
            c for c in self.categorical_cols_input if c in df.columns
        ]
        cat_subset = df[self.retained_cat_cols].astype(str)
        self.ohe.fit(cat_subset)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform input features using fitted training parameters.

        Args:
            df: Input DataFrame (e.g. Test set or Train set).

        Returns:
            A tuple of (continuous_pca_df, full_feature_df).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError(
                "LongitudinalPreprocessor must be fitted before calling transform."
            )

        # 1. Transform Continuous Features
        cont_subset = df[self.retained_cont_cols].copy()
        cont_imputed = self.cont_imputer.transform(cont_subset)
        cont_imp_df = pd.DataFrame(
            cont_imputed, columns=self.retained_cont_cols, index=df.index
        )

        for col in LOG_TRANSFORM_COLUMNS:
            if col in cont_imp_df.columns:
                cont_imp_df[col] = np.log1p(cont_imp_df[col].clip(lower=0))

        scaled_arr = self.scaler.transform(cont_imp_df)
        continuous_pca_df = pd.DataFrame(
            scaled_arr, columns=self.retained_cont_cols, index=df.index
        )

        # 2. Transform Discrete Count Features
        discrete_dict: dict[str, pd.Series] = {}
        for col in self.retained_disc_cols:
            mode_val = self.discrete_modes.get(col, 0)
            discrete_dict[col] = df[col].fillna(mode_val).astype(int)
        discrete_df = pd.DataFrame(discrete_dict, index=df.index)

        # 3. Transform Sex
        if "sex" in df.columns:
            sex_series = df["sex"].fillna(self.sex_mode).astype(float)
        else:
            sex_series = pd.Series(self.sex_mode, index=df.index, name="sex")
        sex_df = pd.DataFrame({"sex": sex_series}, index=df.index)

        # 4. Transform Categorical Features via OneHotEncoder
        cat_subset = df[self.retained_cat_cols].astype(str)
        cat_encoded_arr = self.ohe.transform(cat_subset)
        cat_feature_names = self.ohe.get_feature_names_out(self.retained_cat_cols)
        categorical_df = pd.DataFrame(
            cat_encoded_arr, columns=cat_feature_names, index=df.index
        )

        # 5. Comorbidities (0 missing, numeric)
        comorb_present = [c for c in COMORBIDITY_COLUMNS if c in df.columns]
        if comorb_present:
            comorb_df = df[comorb_present].astype(float)
        else:
            comorb_df = pd.DataFrame(index=df.index)

        # Combine into full feature matrix without duplicate column names
        cont_cols_for_full = [
            c for c in self.retained_cont_cols if c not in self.retained_disc_cols
        ]
        full_feature_df = pd.concat(
            [
                continuous_pca_df[cont_cols_for_full],
                discrete_df,
                sex_df,
                categorical_df,
                comorb_df,
            ],
            axis=1,
        )

        # Add target label if present
        if LABEL_COLUMN_3YR in df.columns:
            full_feature_df[LABEL_COLUMN_3YR] = df[LABEL_COLUMN_3YR].values
        elif LABEL_COLUMN in df.columns:
            full_feature_df[LABEL_COLUMN] = df[LABEL_COLUMN].values

        return continuous_pca_df, full_feature_df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on data and transform it.

        Args:
            df: Input DataFrame.

        Returns:
            A tuple of (continuous_pca_df, full_feature_df).
        """
        return self.fit(df).transform(df)


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
        A tuple of (scaled_df, imputer, scaler, retained_columns).
    """
    candidates = (
        list(continuous_columns)
        if continuous_columns is not None
        else [c for c in DEFAULT_CONTINUOUS_COLUMNS if c in df.columns]
    )

    retained_cols: list[str] = [
        col
        for col in candidates
        if col not in HIGH_MISSING_DROPS
        and col in df.columns
        and df[col].isna().mean() <= missing_threshold
    ]

    subset = df[retained_cols].copy()

    imputer = SimpleImputer(strategy="median")
    imputed_arr = imputer.fit_transform(subset)
    imputed_df = pd.DataFrame(imputed_arr, columns=retained_cols, index=df.index)

    for col in LOG_TRANSFORM_COLUMNS:
        if col in imputed_df.columns:
            imputed_df[col] = np.log1p(imputed_df[col].clip(lower=0))

    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(imputed_df)
    scaled_df = pd.DataFrame(scaled_arr, columns=retained_cols, index=df.index)

    return scaled_df, imputer, scaler, retained_cols


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize binary labels and One-Hot encode categorical features.

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
        A tuple of (continuous_features_pca, full_feature_matrix, target).
    """
    candidate_df, _, target = quarantine_target_leakage(df)
    cleaned_df = clean_clinical_bounds(candidate_df)
    engineered_df = engineer_features(cleaned_df)
    scaled_cont_df, _, _, _ = process_continuous_features(engineered_df)
    encoded_cat_df = encode_categorical_features(engineered_df)

    full_feature_matrix = pd.concat([scaled_cont_df, encoded_cat_df], axis=1)
    if not target.empty:
        full_feature_matrix[target.name if target.name else LABEL_COLUMN] = (
            target.values
        )

    # Derive landmark 1-year mortality target if applicable
    if "死亡日期" in df.columns and "year" in df.columns:
        death_date = pd.to_datetime(df["死亡日期"], errors="coerce")
        death_year = death_date.dt.year
        baseline_year = df["year"]
        is_death_1yr = ((death_year <= baseline_year + 1) & death_date.notna()).astype(
            int
        )
        full_feature_matrix["is_death_1yr"] = is_death_1yr.values

    return scaled_cont_df, full_feature_matrix, target
