# MedicalAI (Hemodialysis Prognosis Context)

This context defines the ubiquitous language and domain concepts for clinical data preprocessing, feature engineering, exploratory data analysis, and dynamic predictive modeling based on the Kidit longitudinal hemodialysis registry (2010–2025).

## Language

**Baseline Year (基準年)**:
The initial calendar year in which a patient first enters the annual hemodialysis reporting system, representing their baseline clinical, demographic, and laboratory state.
_Avoid_: First year, intake year

**Rolling Three-Year Mortality Window (滑動三年期死亡預測窗口)**:
A dynamic prognostic setup where each annual observation at year $t$ predicts whether the patient will experience all-cause mortality within the subsequent 1,095 days ($t \to t+3$), expanding usable longitudinal panel records while strictly requiring $\ge 3$ years of potential follow-up.
_Avoid_: Static mortality, cumulative death rate

**Patient-Level Grouped Split (病患級分群分割)**:
A data partitioning protocol ensuring that all longitudinal records belonging to the same unique `PatientID` are assigned exclusively to either the training set or the testing set, preventing within-subject identity leakage.
_Avoid_: Random row split, patient-mixed partition

**Dialysis Vintage (透析資歷)**:
The accumulated duration of time (expressed in fractional years) from a patient's first lifelong hemodialysis session date (`First_HD_date`) to the current assessment date (`index_date`).
_Avoid_: Dialysis age, treatment duration

**Ultrafiltration Volume (單次脫水量)**:
The net fluid weight removed during an individual hemodialysis session, calculated as pre-dialysis weight minus post-dialysis weight (`before_hd_weight - after_hd_weight`).
_Avoid_: Weight loss, fluid removal

**Ultrafiltration Ratio (脫水率)**:
The relative proportion of fluid removed during dialysis normalized to pre-dialysis body weight (`Ultrafiltration / before_hd_weight`), measuring interdialytic fluid burden.
_Avoid_: Dehydration rate, weight drop ratio

**Kt/V (透析充分性指標)**:
A dimensionless kinetic index quantifying fractional urea clearance per dialysis treatment ($K \times t / V$), bounded by physiological single-pool hemodialysis limits ($Kt/V \ge 0.5$ in viable treatments).
_Avoid_: Clearance rate, dialysis ratio

**Hotelling's T-squared (主成分得分距離)**:
A multivariate metric measuring the normalized distance of an observation from the center of the principal component subspace, accounting for variance across retained components.
_Avoid_: PC distance, score norm

**Squared Prediction Error / SPE (殘差平方和 / Q 統計量)**:
A metric measuring the magnitude of residuals outside the principal component subspace, quantifying how poorly a patient's clinical pattern is reconstructed by the model.
_Avoid_: Reconstruction loss, residual variance

**Train-First Split Principle (先分割後剔除原則)**:
The architectural sequence requiring that dataset partitioning (Train vs. Test) precede any statistical outlier pruning or imputation fitting, preventing data leakage across subsets.
_Avoid_: Pre-split filtering, global outlier removal

**Uncurated Test Set (未刪除測試集)**:
The evaluation subset retained in its raw, unfiltered state to measure real-world clinical inference robustness against natural clinical artifacts and extreme conditions.
_Avoid_: Clean test set, filtered evaluation set

**Target Leakage (目標洩漏)**:
The erroneous inclusion of downstream clinical outcome events, transfer details, or post-baseline cause-of-death indicators within predictive feature sets, leading to spurious predictive accuracy.
_Avoid_: Data cheating, future bias

**11-Algorithm Benchmark Suite (11 演算法基準模型套件)**:
A standardized evaluation battery of 11 diverse classifiers (Logistic Regression, KNN, GaussianNB, Decision Tree, Random Forest, SVM, AdaBoost, GradientBoost, Extra Trees, XGBoost, CatBoost) trained on the rolling dynamic hemodialysis panel to benchmark discrimination, calibration, and clinical utility.
_Avoid_: 11 expert models, single-model search

**In-Fold SMOTE Resampling (折內合成過採樣)**:
The disciplined application of Synthetic Minority Over-sampling Technique (SMOTE/SMOTENC) strictly within training splits or cross-validation training folds, never touching validation folds or the uncurated test set, preventing synthetic information leakage.
_Avoid_: Global SMOTE, test set oversampling

**Grouped Patient Cross-Validation (病患級分群交叉驗證)**:
A $K$-fold cross-validation scheme partitioned strictly by `PatientID` (`GroupKFold`) so that all longitudinal person-year observations belonging to the same individual are confined to either the training fold or the validation fold, preventing temporal self-memorization.
_Avoid_: Random K-fold, stratified row K-fold

**SMOTE-NC (類別連續混合合成過採樣)**:
An extension of the Synthetic Minority Over-sampling Technique specifically designed for mixed-type datasets, performing spatial interpolation solely across continuous laboratory biomarkers while preserving discrete integer and binary states (such as comorbidities and one-hot encodings) via nearest-neighbor mode assignment.
_Avoid_: Naive continuous SMOTE on binary flags, post-hoc threshold rounding

**Out-of-Fold Threshold Calibration (折外驗證門檻校準)**:
The protocol of identifying the optimal classification cutoff threshold (e.g. maximizing F1-score) strictly using pooled out-of-fold cross-validation predictions from the training set, subsequently fixing this cutoff for inference on the uncurated test set to prevent test label leakage.
_Avoid_: Test set threshold tuning, arbitrary global 0.5 without evaluation

**Immutable Stratified Group 5-Fold Partitioning (固定分層分群五折切分)**:
A permanent, pre-computed 5-fold cross-validation partition on the purified training set ($N=3,737$) constructed using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, where patient identifiers (`PatientID`) strictly define non-overlapping groups and 3-year mortality (`is_death_3yr`) enforces balanced event rates (~18%) across all folds, serving as the immutable ground-truth partition for all baseline and gating architectures.
_Avoid_: Dynamic on-the-fly cross-validation, unstratified patient grouping

**Type-Specific Non-Mode Imputation (非眾數型態專屬填補體系)**:
A principled missing-data architecture that completely bans naive mode (眾數) imputation across all discrete clinical variables:
1. _Ordinal Functional Scales_ (`genAssess`, `actAssess`): Imputed via nearest-neighbor regression (`KNNImputer`) leveraging continuous biomarkers to preserve clinical severity rankings without defaulting unassessed patients to healthy states.
2. _Prescription & Scheduling Discretes_ (`HD_WS`, `HD_ST`, `CIC`, `total_hd_time`, `txIntervalMin`): Transformed via One-Hot encoding with explicit missing tokens (`dummy_na=True`) to treat unrecorded prescriptions as distinct categorical states.
3. _Zero-Inflated Dosages_ (`ID_U`, `MD_U`): Imputed with clinical baseline zero (`0.0`) accompanied by explicit binary missing indicator features (`_isna`).
4. _Comorbidity Counts_ (`Comorb_Count`): Zero-missing integer cumulative score derived from individual diagnosis flags.
_Avoid_: Blanket mode imputation, arbitrary integer rounding, unassessed healthy bias

