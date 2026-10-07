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
| **Random Forest** | `0.50` | 0.9194 | 0.6878 | 0.1046 | 0.8831 | 0.7310 | 0.9158 | 0.6890 | 0.6173 | 0.8831 | 0.7310 | 0.9158 | 0.6890 | 0.6173 |
| **CatBoost** | `0.37` | 0.9173 | 0.7234 | 0.0818 | 0.8867 | 0.6904 | 0.9290 | 0.6834 | 0.6144 | 0.8840 | 0.7614 | 0.9104 | 0.6993 | 0.6280 |
| **XGBoost** | `0.50` | 0.9137 | 0.7332 | 0.0829 | 0.8840 | 0.6751 | 0.9290 | 0.6734 | 0.6029 | 0.8840 | 0.6751 | 0.9290 | 0.6734 | 0.6029 |
| **AdaBoost** | `0.52` | 0.9076 | 0.6860 | 0.1914 | 0.8606 | 0.7310 | 0.8885 | 0.6501 | 0.5644 | 0.8696 | 0.6193 | 0.9235 | 0.6272 | 0.5482 |
| **Extra Trees** | `0.49` | 0.9025 | 0.6766 | 0.1202 | 0.8714 | 0.7259 | 0.9027 | 0.6667 | 0.5877 | 0.8696 | 0.7462 | 0.8962 | 0.6697 | 0.5895 |
| **Gradient Boosting** | `0.48` | 0.9000 | 0.7069 | 0.0901 | 0.8849 | 0.6954 | 0.9257 | 0.6816 | 0.6114 | 0.8813 | 0.7056 | 0.9191 | 0.6780 | 0.6054 |
| **Logistic Regression** | `0.45` | 0.8851 | 0.6937 | 0.1087 | 0.8525 | 0.6802 | 0.8896 | 0.6204 | 0.5297 | 0.8435 | 0.6954 | 0.8754 | 0.6116 | 0.5154 |
| **KNN** | `0.58` | 0.8225 | 0.4907 | 0.1806 | 0.7500 | 0.7716 | 0.7454 | 0.5223 | 0.3761 | 0.7995 | 0.6802 | 0.8251 | 0.5458 | 0.4235 |
| **SVM** | `0.67` | 0.8163 | 0.5504 | 0.1985 | 0.6673 | 0.7310 | 0.6536 | 0.4377 | 0.2520 | 0.8309 | 0.5228 | 0.8973 | 0.5228 | 0.4201 |
| **Decision Tree** | `0.61` | 0.7984 | 0.4648 | 0.1390 | 0.8291 | 0.6396 | 0.8699 | 0.5701 | 0.4651 | 0.8291 | 0.6396 | 0.8699 | 0.5701 | 0.4651 |
| **GaussianNB** | `0.95` | 0.7970 | 0.5102 | 0.1938 | 0.8004 | 0.6599 | 0.8306 | 0.5394 | 0.4174 | 0.8129 | 0.6497 | 0.8481 | 0.5517 | 0.4369 |

## Anti-Leakage Audit Trail
- [x] Uncurated test set count strictly 1,112 (zero synthetic instances).
- [x] Patient isolation: Train $\cap$ Test patient IDs = $\emptyset$.
- [x] In-fold SMOTE-NC: Resampling strictly inside CV training folds.
- [x] Out-of-fold threshold: $T^*$ selected purely on OOF probabilities.
- [x] Static fold consumption: 5-fold partition with zero leakage.
