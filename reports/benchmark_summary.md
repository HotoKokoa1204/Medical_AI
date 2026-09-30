# AGILAB MedicalAI: 11-Algorithm Benchmark Suite Leaderboard

## Cohort & Methodology Summary
- **Cohort**: Longitudinal Hemodialysis Rolling 3-Year Dynamic Cohort
- **Training Records**: 3,737 observations (purified via multivariate $T^2$ & Jackson-Mudholkar SPE)
- **Uncurated Test Records**: 1,112 observations (100% unpruned, zero synthetic distortion)
- **Validation Scheme**: 5-Fold `GroupKFold` strictly partitioned by `PatientID` (zero cross-fold identity leakage)
- **Resampling**: In-Fold `SMOTE-NC` (spatial interpolation for continuous analytes, mode assignment for discrete flags)
- **Threshold Calibration**: Optimal cutoff $T^* \in [0.05, 0.95]$ (step 0.01) maximizing F1-score strictly on Out-of-Fold (OOF) predictions
- **Blinding Principle**: Test labels strictly isolated and untouched during all preprocessing, tuning, and threshold selection

## Performance Leaderboard (Ranked by ROC-AUC)

| Model | Optimal $T^*$ | ROC-AUC | PR-AUC | Brier | Acc ($T=0.5$) | Sens ($T=0.5$) | Spec ($T=0.5$) | F1 ($T=0.5$) | Kappa ($T=0.5$) | Acc ($T^*$) | Sens ($T^*$) | Spec ($T^*$) | F1 ($T^*$) | Kappa ($T^*$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | `0.52` | 0.9186 | 0.6886 | 0.1036 | 0.8804 | 0.7208 | 0.9148 | 0.6811 | 0.6077 | 0.8867 | 0.7056 | 0.9257 | 0.6881 | 0.6189 |
| **CatBoost** | `0.38` | 0.9177 | 0.7157 | 0.0831 | 0.8921 | 0.6954 | 0.9344 | 0.6954 | 0.6299 | 0.8768 | 0.7513 | 0.9038 | 0.6836 | 0.6079 |
| **AdaBoost** | `0.51` | 0.9147 | 0.6920 | 0.1892 | 0.8615 | 0.7563 | 0.8842 | 0.6593 | 0.5742 | 0.8678 | 0.6802 | 0.9082 | 0.6458 | 0.5648 |
| **XGBoost** | `0.41` | 0.9067 | 0.7086 | 0.0879 | 0.8831 | 0.6599 | 0.9311 | 0.6667 | 0.5958 | 0.8741 | 0.7005 | 0.9115 | 0.6635 | 0.5863 |
| **Extra Trees** | `0.50` | 0.9049 | 0.6777 | 0.1180 | 0.8768 | 0.7157 | 0.9115 | 0.6730 | 0.5975 | 0.8768 | 0.7157 | 0.9115 | 0.6730 | 0.5975 |
| **Gradient Boosting** | `0.44` | 0.9017 | 0.6960 | 0.0929 | 0.8768 | 0.6751 | 0.9202 | 0.6600 | 0.5849 | 0.8750 | 0.7005 | 0.9126 | 0.6651 | 0.5885 |
| **Logistic Regression** | `0.58` | 0.8911 | 0.6612 | 0.1085 | 0.8489 | 0.6650 | 0.8885 | 0.6093 | 0.5165 | 0.8543 | 0.6142 | 0.9060 | 0.5990 | 0.5101 |
| **SVM** | `0.64` | 0.8226 | 0.6045 | 0.2028 | 0.6673 | 0.7310 | 0.6536 | 0.4377 | 0.2520 | 0.6682 | 0.7310 | 0.6546 | 0.4384 | 0.2531 |
| **KNN** | `0.72` | 0.8107 | 0.4850 | 0.1806 | 0.7554 | 0.7360 | 0.7596 | 0.5160 | 0.3714 | 0.8327 | 0.5482 | 0.8940 | 0.5373 | 0.4353 |
| **GaussianNB** | `0.95` | 0.7970 | 0.5108 | 0.2377 | 0.7536 | 0.6954 | 0.7661 | 0.5000 | 0.3532 | 0.7716 | 0.6853 | 0.7902 | 0.5153 | 0.3777 |
| **Decision Tree** | `0.38` | 0.7871 | 0.4284 | 0.1492 | 0.8112 | 0.6447 | 0.8470 | 0.5474 | 0.4315 | 0.7842 | 0.7157 | 0.7989 | 0.5402 | 0.4101 |

## Anti-Leakage Audit Trail
- [x] Uncurated test set sample count strictly equals 1,112 (zero synthetic instances, zero dropped rows).
- [x] Grouped patient cross-validation: Train $\cap$ Test patient IDs = $\emptyset$.
- [x] In-fold SMOTE-NC isolation: Resampling occurs strictly inside each CV training fold and on full train set.
- [x] Out-of-fold threshold calibration: $T^*$ selected purely on OOF probabilities with test labels completely blinded.
