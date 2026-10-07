# ADR-0004: Type-Specific Non-Mode Imputation Architecture and Immutable Stratified Group 5-Fold Partitioning

## Context

Following the establishment of the dynamic rolling 3-year mortality framework (ADR-0002) and the 11-algorithm benchmark suite (ADR-0003), critical methodological audits identified two architectural limitations that introduced statistical bias and validation instability:

1. **Flaws of Blanket Mode Imputation on Ordinal Scales and Prescription Discretes**:
   - *Ordinal Functional Scales (`genAssess`, `actAssess`)*: In prior preprocessing pipelines, missing values across discrete columns were uniformly filled using feature modes. For clinical functional status scales (`genAssess` ranging from 1 to 9; `actAssess` Karnofsky Performance Scale ranging from 20 to 100), mode imputation imputed the most frequent population value (typically 1 for `genAssess` and 80–90 for `actAssess`). This introduced severe pathological bias by imputing unassessed, frail, or end-of-life patients as physiologically normal, collapsing variance, destroying monotonic clinical degradation trajectories, and obscuring functional mortality signals.
   - *Prescription and Scheduling Discretes (`HD_WS`, `HD_ST`, `CIC`, `total_hd_time`, `txIntervalMin`)*: These parameters represent clinical shift scheduling, vascular access types, and treatment delivery times. Treating them as numeric variables subjected to mode imputation artificially collapsed distinct therapeutic protocols into arbitrary dominant integers and eliminated informative missingness patterns (where missingness reflects unscheduled acute shifts or unrecorded session parameters).
   - *Zero-Inflated Drug Dosages (`ID_U`, `MD_U`)*: Mode imputation across medication dosage counts obscured the fundamental clinical distinction between non-prescribed status (true dose of 0.0) and unrecorded observational missingness.

2. **Experimental Variance and Instability of Dynamic `GroupKFold`**:
   - Cross-validation previously relied on dynamic, on-the-fly instantiation of `GroupKFold(n_splits=5)`. While `GroupKFold` prevented within-patient identity leakage, it lacked class stratification. In moderately imbalanced clinical panels (~18.14% 3-year mortality), random group partitioning resulted in uncontrolled fold-to-fold event rate variance.
   - Generating fold splits dynamically across separate scripts (`train_benchmark.py`, downstream hyperparameter tuning, model stacking) created reproducibility drift, invalidated out-of-fold (OOF) prediction caching, and impaired fair pairwise comparisons between candidate architectures.

3. **Reconciliation of the 15-Record Cohort Discrepancy**:
   - ADR-0002 documented a historical estimate of 3,752 purified training records (derived from 4,029 training observations with 277 estimated drops).
   - Code verification and empirical data auditing confirmed exactly 3,737 retained training records across 932 patients:
     - Initial training cohort: 4,029 records across 932 unique patients (from the 80/20 patient-level split of the 5,141 rolling 3-year panel).
     - Stage 1 Deterministic Pruning: exactly 22 records rejected due to clinical sanity violations (physiological impossibility, including software calculation artifacts where $Kt/V < 0.5$, or extreme duration/weight anomalies).
     - Stage 2 Multivariate Statistical Outlier Pruning: exactly 270 records rejected via PCA Hotelling's $T^2$ ($T^2 > 15.124$) and Squared Prediction Error ($\text{SPE} > 22.982$) at significance level $\alpha = 0.01$.
     - Total dropped training records: $22 + 270 = 292$ records (7.25% of training cohort), yielding verified $N = 3,737$.
     - The 15-record difference between the preliminary 3,752 figure and the verified 3,737 count stemmed from an initial exploratory script where deterministic filtering dropped only 7 records instead of the audited 22 records prior to PCA pruning ($4,029 - 7 - 270 = 3,752$). The production pipeline standardizes strictly on $N = 3,737$.

## Decision

1. **Type-Specific Non-Mode Imputation Architecture**:
   - Naive mode imputation is entirely banned across all discrete and ordinal clinical variables.
   - *Ordinal Functional Status (`genAssess`, `actAssess`)*:
     - Imputed using distance-weighted K-Nearest Neighbors regression: `KNNImputer(n_neighbors=5, weights="distance")`.
     - Distance metrics are computed across retained continuous physiological features (normalized age, laboratory values, vital signs), allowing imputation to reflect patient-specific clinical severity rather than arbitrary central tendencies.
   - *Prescription and Dialysis Regimens (`HD_WS`, `HD_ST`, `CIC`, `total_hd_time`, `txIntervalMin`)*:
     - Excluded from numeric imputation and processed as nominal categorical features.
     - Encoded via `OneHotEncoder(dummy_na=True, sparse_output=False, handle_unknown="ignore")`, generating explicit binary indicator categories (e.g. `HD_WS_3.0`, `HD_WS_nan`) that preserve unrecorded regimens as informative clinical states.
   - *Zero-Inflated Drug Dosages (`ID_U`, `MD_U`)*:
     - Missing values are imputed with clinical baseline `0.0`.
     - Explicit binary missing indicator columns (`ID_U_isna`, `MD_U_isna` with values $1.0$ for missing and $0.0$ for observed) are appended to the feature matrix.
   - *Comorbidity Score (`Comorb_Count`)*:
     - Retained as an integer cumulative score (verified 0% missing in the purified cohort).
   - *Cross-Sectional & Causal Isolation*:
     - Imputation is strictly cross-sectional, with zero longitudinal forward or backward filling across person-years.
     - Imputation and encoding models are fitted exclusively on training data; the uncurated test set ($N=1,112$) is transformed strictly using training-learned transformers.

2. **Immutable Stratified Group 5-Fold Partitioning**:
   - The purified training cohort ($N=3,737$, 932 unique patients) is permanently partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` from `sklearn.model_selection`.
   - *Grouping*: Grouped strictly by `PatientID`, ensuring all multi-year records for any patient are confined to exactly one fold ($\text{PatientID}_{\text{fold}_i} \cap \text{PatientID}_{\text{fold}_j} = \emptyset$).
   - *Stratification*: Stratified by 3-year mortality (`is_death_3yr`), guaranteeing identical target prevalence (~18.14%) across all 5 folds and eliminating fold-to-fold class skew.
   - *Stable Composite Identifier*: Each observation is tracked via `record_id = f"{PatientID}_{year}"`.

3. **Dual-Track Persistence**:
   - *Track 1 (Embedded)*: A dedicated `fold` column (integer type, values 0–4) is appended directly to `data/processed/train_cleaned_rolling_3yr.parquet`.
   - *Track 2 (Audit Table)*: An external, immutable CSV audit table is generated at `data/processed/train_cv_folds.csv` with schema:
     $$\text{Columns: } [\texttt{record\_id}, \texttt{PatientID}, \texttt{year}, \texttt{is\_death\_3yr}, \texttt{fold}]$$

4. **Feature Space Quarantine and Test Set Blinding**:
   - The `fold` identifier is strictly quarantined from feature space $X$ across all training, inference, and evaluation routines alongside target variables (`is_death_3yr`, `is_death`) and identifiers (`PatientID`, `record_id`).
   - The uncurated test set ($N=1,112$ records from 233 patients) remains strictly unpartitioned (contains no `fold` column) and completely blinded during all training, validation, and threshold selection phases.

## Consequences

- **Mathematical Stability of Out-of-Fold Predictions**: Pre-computed, immutable fold assignments guarantee that all 11 benchmark models and future gating/ensemble architectures share identical validation splits, enabling exact pairwise comparisons, McNemar tests, and reliable stacked meta-learning.
- **Zero Information and Identity Leakage**: Strict patient grouping eliminates temporal self-memorization, while target stratification stabilizes in-fold cross-validation loss without leaking future outcomes.
- **Clinical Fidelity and Monotonic Risk Preservation**: KNN-based functional imputation preserves the graded severity of physiological decline, preventing artificial deflation of mortality risk in frail patients. Explicit categorical encoding of prescription missingness captures protocol variation without numeric distortion.
- **Auditable Reproducibility**: The dual-track persistence strategy provides an immutable ground truth for internal benchmarking and external verification.
