# AGILAB MedicalAI: 11-Algorithm Benchmark Suite Leaderboard

## Cohort & Methodology Summary
- **Cohort**: Longitudinal Hemodialysis Rolling 3-Year Dynamic Cohort
- **Training Records**: 3,737 observations (purified via multivariate $T^2$ & Jackson-Mudholkar SPE)
- **Uncurated Test Records**: 1,112 observations (100% unpruned, zero synthetic distortion)
- **Test Set Ground Truth**: 197 Deceased (17.72%), 915 Survived (82.28%)
- **Validation Scheme**: 5-Fold `GroupKFold` strictly partitioned by `PatientID` (zero cross-fold identity leakage)
- **Resampling**: In-Fold `SMOTE-NC` (spatial interpolation for continuous analytes, mode assignment for discrete flags)
- **Threshold Calibration**: Optimal cutoff $T^* \in [0.05, 0.95]$ (step 0.01) maximizing F1-score strictly on Out-of-Fold (OOF) predictions
- **Blinding Principle**: Test labels strictly isolated and untouched during all preprocessing, tuning, and threshold selection

## Performance Leaderboard (Ranked by ROC-AUC, with Explicit Recall & Precision)

| Rank | Model | Optimal $T^*$ | ROC-AUC | PR-AUC | Brier | Recall / Sens ($T^*$) | True Positives (TP / 197) | Precision ($T^*$) | False Positives (FP) | Spec ($T^*$) | F1 ($T^*$) | Acc ($T^*$) | Kappa ($T^*$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | `0.52` | **0.9186** | 0.6886 | 0.1036 | 70.6% | 139 / 197 | **67.1%** | **68** | **0.9257** | **0.6881** | **0.8867** | **0.6189** |
| 🥈 | **CatBoost** | `0.38` | **0.9177** | **0.7157** | **0.0831** | **75.1%** | **148 / 197** | 62.7% | 88 | 0.9038 | 0.6836 | 0.8768 | 0.6079 |
| 🥉 | **AdaBoost** | `0.51` | 0.9147 | 0.6920 | 0.1892 | 68.0% | 134 / 197 | 61.5% | 84 | 0.9082 | 0.6458 | 0.8678 | 0.5648 |
| 4 | **XGBoost** | `0.41` | 0.9067 | 0.7086 | 0.0879 | 70.1% | 138 / 197 | 63.0% | 81 | 0.9115 | 0.6635 | 0.8741 | 0.5863 |
| 5 | **Extra Trees** | `0.50` | 0.9049 | 0.6777 | 0.1180 | 71.6% | 141 / 197 | 63.5% | 81 | 0.9115 | 0.6730 | 0.8768 | 0.5975 |
| 6 | **Gradient Boosting** | `0.44` | 0.9017 | 0.6960 | 0.0929 | 70.1% | 138 / 197 | 63.3% | 80 | 0.9126 | 0.6651 | 0.8750 | 0.5885 |
| 7 | **Logistic Regression** | `0.58` | 0.8911 | 0.6612 | 0.1085 | 61.4% | 121 / 197 | 58.5% | 86 | 0.9060 | 0.5990 | 0.8543 | 0.5101 |
| 8 | **SVM** | `0.64` | 0.8226 | 0.6045 | 0.2028 | 73.1% | 144 / 197 | 31.3% | 316 | 0.6546 | 0.4384 | 0.6682 | 0.2531 |
| 9 | **KNN** | `0.72` | 0.8107 | 0.4850 | 0.1806 | 54.8% | 108 / 197 | 52.7% | 97 | 0.8940 | 0.5373 | 0.8327 | 0.4353 |
| 10 | **GaussianNB** | `0.95` | 0.7970 | 0.5108 | 0.2377 | 68.5% | 135 / 197 | 41.3% | 192 | 0.7902 | 0.5153 | 0.7716 | 0.3777 |
| 11 | **Decision Tree** | `0.38` | 0.7871 | 0.4284 | 0.1492 | 71.6% | 141 / 197 | 43.4% | 184 | 0.7989 | 0.5402 | 0.7842 | 0.4101 |

---

## Default Threshold ($T=0.50$) Comparison (Ranked by ROC-AUC)

| Rank | Model | ROC-AUC | PR-AUC | Brier | Recall / Sens ($T=0.5$) | True Positives (TP / 197) | Precision ($T=0.5$) | False Positives (FP) | Spec ($T=0.5$) | F1 ($T=0.5$) | Acc ($T=0.5$) | Kappa ($T=0.5$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | **0.9186** | 0.6886 | 0.1036 | 72.1% | 142 / 197 | 64.5% | 78 | 0.9148 | 0.6811 | 0.8804 | 0.6077 |
| 🥈 | **CatBoost** | **0.9177** | **0.7157** | **0.0831** | 69.5% | 137 / 197 | **69.5%** | **60** | **0.9344** | **0.6954** | **0.8921** | **0.6299** |
| 🥉 | **AdaBoost** | 0.9147 | 0.6920 | 0.1892 | **75.6%** | **149 / 197** | 58.4% | 106 | 0.8842 | 0.6593 | 0.8615 | 0.5742 |
| 4 | **XGBoost** | 0.9067 | 0.7086 | 0.0879 | 66.0% | 130 / 197 | 67.4% | 63 | 0.9311 | 0.6667 | 0.8831 | 0.5958 |
| 5 | **Extra Trees** | 0.9049 | 0.6777 | 0.1180 | 71.6% | 141 / 197 | 63.5% | 81 | 0.9115 | 0.6730 | 0.8768 | 0.5975 |
| 6 | **Gradient Boosting** | 0.9017 | 0.6960 | 0.0929 | 67.5% | 133 / 197 | 64.6% | 73 | 0.9202 | 0.6600 | 0.8768 | 0.5849 |
| 7 | **Logistic Regression** | 0.8911 | 0.6612 | 0.1085 | 66.5% | 131 / 197 | 56.2% | 102 | 0.8885 | 0.6093 | 0.8489 | 0.5165 |
| 8 | **SVM** | 0.8226 | 0.6045 | 0.2028 | 73.1% | 144 / 197 | 31.2% | 317 | 0.6536 | 0.4377 | 0.6673 | 0.2520 |
| 9 | **KNN** | 0.8107 | 0.4850 | 0.1806 | 73.6% | 145 / 197 | 39.7% | 220 | 0.7596 | 0.5160 | 0.7554 | 0.3714 |
| 10 | **GaussianNB** | 0.7970 | 0.5108 | 0.2377 | 69.5% | 137 / 197 | 39.0% | 214 | 0.7661 | 0.5000 | 0.7536 | 0.3532 |
| 11 | **Decision Tree** | 0.7871 | 0.4284 | 0.1492 | 64.5% | 127 / 197 | 47.6% | 140 | 0.8470 | 0.5474 | 0.8112 | 0.4315 |

---

## Anti-Leakage Audit Trail
- [x] Uncurated test set sample count strictly equals 1,112 (zero synthetic instances, zero dropped rows).
- [x] Grouped patient cross-validation: Train $\cap$ Test patient IDs = $\emptyset$.
- [x] In-fold SMOTE-NC isolation: Resampling occurs strictly inside each CV training fold and on full train set.
- [x] Out-of-fold threshold calibration: $T^*$ selected purely on OOF probabilities with test labels completely blinded.
