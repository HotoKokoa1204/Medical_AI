# AGILAB MedicalAI: 11-Algorithm Benchmark Suite Leaderboard

## Cohort & Methodology Summary
- **Cohort**: Longitudinal Hemodialysis Rolling 3-Year Dynamic Cohort
- **Training Records**: 3,737 observations (purified via $T^2$ & SPE)
- **Uncurated Test Records**: 1,112 observations (100% unpruned)
- **Validation Scheme**: 5-Fold Stratified Group Partition (`PatientID`)
- **Resampling**: In-Fold `SMOTE-NC` (mode for discrete flags)
- **Threshold Calibration**: Optimal cutoff $T^* \in [0.05, 0.95]$ on OOF
- **Blinding Principle**: Test labels isolated and untouched

## Performance Leaderboard (Ranked by ROC-AUC)

| Model | Optimal $T^*$ | ROC-AUC | PR-AUC | Brier | Acc ($T=0.5$) | Sens ($T=0.5$) | Spec ($T=0.5$) | F1 ($T=0.5$) | Kappa ($T=0.5$) | Acc ($T^*$) | Sens ($T^*$) | Spec ($T^*$) | F1 ($T^*$) | Kappa ($T^*$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | `0.50` | 0.9211 | 0.6873 | 0.1039 | 0.8822 | 0.7157 | 0.9180 | 0.6828 | 0.6107 | 0.8822 | 0.7157 | 0.9180 | 0.6828 | 0.6107 |
| **CatBoost** | `0.42` | 0.9158 | 0.7109 | 0.0822 | 0.8939 | 0.6954 | 0.9366 | 0.6990 | 0.6346 | 0.8876 | 0.7360 | 0.9202 | 0.6988 | 0.6299 |
| **XGBoost** | `0.46` | 0.9120 | 0.7209 | 0.0826 | 0.8921 | 0.6954 | 0.9344 | 0.6954 | 0.6299 | 0.8885 | 0.7107 | 0.9268 | 0.6931 | 0.6250 |
| **AdaBoost** | `0.51` | 0.9095 | 0.6799 | 0.1896 | 0.8579 | 0.7259 | 0.8863 | 0.6441 | 0.5568 | 0.8633 | 0.6650 | 0.9060 | 0.6329 | 0.5491 |
| **Extra Trees** | `0.48` | 0.9035 | 0.6808 | 0.1199 | 0.8741 | 0.7208 | 0.9071 | 0.6698 | 0.5925 | 0.8660 | 0.7563 | 0.8896 | 0.6667 | 0.5843 |
| **Gradient Boosting** | `0.44` | 0.9012 | 0.7034 | 0.0896 | 0.8786 | 0.6904 | 0.9191 | 0.6683 | 0.5941 | 0.8741 | 0.7310 | 0.9049 | 0.6729 | 0.5956 |
| **Logistic Regression** | `0.48` | 0.8910 | 0.7020 | 0.1055 | 0.8561 | 0.6751 | 0.8951 | 0.6244 | 0.5360 | 0.8534 | 0.6853 | 0.8896 | 0.6236 | 0.5335 |
| **KNN** | `0.58` | 0.8192 | 0.4841 | 0.1810 | 0.7572 | 0.7665 | 0.7552 | 0.5280 | 0.3851 | 0.7986 | 0.6853 | 0.8230 | 0.5466 | 0.4238 |
| **SVM** | `0.67` | 0.8159 | 0.5501 | 0.1980 | 0.6673 | 0.7310 | 0.6536 | 0.4377 | 0.2520 | 0.8309 | 0.5228 | 0.8973 | 0.5228 | 0.4201 |
| **Decision Tree** | `0.59` | 0.8051 | 0.4827 | 0.1387 | 0.8219 | 0.7056 | 0.8470 | 0.5840 | 0.4750 | 0.8219 | 0.7056 | 0.8470 | 0.5840 | 0.4750 |
| **GaussianNB** | `0.95` | 0.7971 | 0.5103 | 0.1937 | 0.8004 | 0.6599 | 0.8306 | 0.5394 | 0.4174 | 0.8129 | 0.6497 | 0.8481 | 0.5517 | 0.4369 |

## Anti-Leakage Audit Trail
- [x] Uncurated test set count strictly 1,112 (zero synthetic instances).
- [x] Patient isolation: Train $\cap$ Test patient IDs = $\emptyset$.
- [x] In-fold SMOTE-NC: Resampling strictly inside CV training folds.
- [x] Out-of-fold threshold: $T^*$ selected purely on OOF probabilities.
- [x] Static fold consumption: 5-fold partition with zero leakage.
