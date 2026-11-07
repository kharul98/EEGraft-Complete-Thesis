# Thesis execution and results chapter draft

**Working title:** Reliability and Interpretability of a Pretrained EEG Foundation Model for Alzheimer's Disease Classification

## Abstract

This study evaluated Alzheimer's disease (AD) versus healthy controls (CN) using a pretrained CBraMod EEG encoder and a custom dual-scale residual fusion decoder. The cohort comprised 65 participants: 52 for development and 13 for a locked within-dataset assessment. Five patient-disjoint development folds compared eight neural and classical model variants. DSRF achieved 77.06% pooled development balanced accuracy; XGBoost achieved 76.61%. Fixed final models achieved held-out balanced accuracies of 70.24% (DSRF) and 77.38% (XGBoost). Synthetic noise and channel-removal tests revealed that clean classification and reliability do not coincide. Results are exploratory and do not establish clinical diagnostic validity.

## Methods and reproducibility

Original OpenNeuro ds004504 EEG records were analyzed for 36 AD and 29 CN patients. A fixed 39/13/13 train/validation/test partition was used; 52 development subjects were further divided into five stratified outer folds with separate inner validation. The locked 13-person test (7 AD, 6 CN) was used only after final DSRF and XGBoost selection and fitting. Subjects, not windows, defined partitions.

The pipeline used 19 channels, a 0.3-40 Hz bandpass, 200 Hz sampling rate, 4-second EEG windows (up to 24 per person), and subject-level aggregation of mean window softmax probabilities. CBraMod pretrained weights were strictly loaded and the encoder frozen. DSRF fused high-level pretrained tokens with multiscale convolutional raw-signal features through a learned gate and residual. Baselines included SVM, Random Forest and XGBoost on spectral and Hjorth features, plus an EEGNet-inspired network trained from scratch.

## Table 1 - Five-fold development model comparison

| Model | Pooled BAcc (%) | Mean fold BAcc (%) | Fold SD (pp) | Macro-F1 (%) | Correct /52 |
| --- | --- | --- | --- | --- | --- |
| DSRF | 77.06 | 77.17 | 13.71 | 76.79 | 40/52 |
| XGBoost | 76.61 | 76.67 | 8.99 | 76.61 | 40/52 |
| EEGNet-inspired | 74.89 | 74.5 | 10.6 | 74.77 | 39/52 |
| Random Forest | 72.71 | 73.0 | 6.84 | 72.71 | 38/52 |
| High only | 72.26 | 72.33 | 9.97 | 72.42 | 38/52 |
| SVM | 71.81 | 72.33 | 12.04 | 72.04 | 38/52 |
| No skip | 71.44 | 72.33 | 8.05 | 71.06 | 37/52 |
| Low only | 62.74 | 62.5 | 10.24 | 62.79 | 33/52 |

DSRF achieved the highest pooled development balanced accuracy but XGBoost was very close, with both predicting 40 of 52 development patients correctly. All between-model comparisons were exploratory; the best model was selected after examining development results.

## Table 2 - Capacity-matched skip-connection study

| Rule | Pooled BAcc (%) | Macro-F1 (%) | Correct /52 |
| --- | --- | --- | --- |
| residual | 77.06 | 76.79 | 40/52 |
| blend | 73.16 | 72.92 | 38/52 |

Both matched variants used identical modules and effective parameter counts. The comparison contrasted an additive identity residual against a gated blend. The observed residual advantage was not statistically conclusive.

## Table 3 - Repeated synthetic perturbation robustness

| model | condition | mean_BAcc_pct | BAcc_min_pct | BAcc_max_pct | mean_flip_pct | mean_Brier |
| --- | --- | --- | --- | --- | --- | --- |
| DSRF | noise_10pct | 71.87 | 69.72 | 74.06 | 23.08 | 0.3435 |
| DSRF | noise_20pct | 70.07 | 67.99 | 73.61 | 27.69 | 0.41 |
| DSRF | one_channel_zeroed | 76.78 | 74.89 | 78.34 | 7.31 | 0.3629 |
| No skip | noise_10pct | 71.14 | 68.44 | 76.69 | 20.0 | 0.3987 |
| No skip | noise_20pct | 69.75 | 66.27 | 74.96 | 28.08 | 0.4665 |
| No skip | one_channel_zeroed | 71.65 | 67.17 | 74.06 | 17.69 | 0.3904 |
| High only | noise_10pct | 69.66 | 66.19 | 72.26 | 20.77 | 0.3625 |
| High only | noise_20pct | 72.19 | 70.54 | 74.44 | 25.0 | 0.4021 |
| High only | one_channel_zeroed | 69.48 | 67.47 | 72.26 | 6.92 | 0.3544 |

Robustness included five seeded perturbation realizations on the SAME 52 development patients. Repeats are not independent patient cohorts. Noise magnitudes were relative to within-window signal standard deviation; channel-removal blanked one electrode per EEG window. Label-flip rates must be interpreted alongside balanced accuracy and Brier scores. A constant-label classifier has zero flips but only 50% binary balanced accuracy.

## Explainability

Full-window (up to 24) channel-occlusion analysis on 52 development patients: AD: T5 (11.03 pp), Pz (7.66 pp), F8 (7.60 pp), Fz (6.82 pp), C3 (5.09 pp); CN: O2 (31.50 pp), T6 (16.01 pp), O1 (15.90 pp), Fp2 (14.96 pp), F3 (12.52 pp). Masking sensitivity and gates are descriptive, not physiological causal evidence. Same predictions as full CV: 52/52.

The explanation procedure identifies sensitivity to model inputs rather than clinically validated brain biomarkers. Interpretation is sensitive to the masking design and the number of explained windows.

## Table 4 - Final one-time held-out results (13 patients)

| Model | BAcc (%) | Accuracy (%) | Macro-F1 (%) | AD recall (%) | CN recall (%) | AUROC | Brier | Correct /13 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSRF | 70.24 | 69.23 | 69.05 | 57.14 | 83.33 | 0.857 | 0.3146 | 9/13 |
| XGBoost | 77.38 | 76.92 | 76.92 | 71.43 | 83.33 | 0.905 | 0.2976 | 10/13 |

DSRF patient-level BAcc exploratory 95% stratified bootstrap interval: 46.43%-92.86%; XGBoost interval: 53.57%-100.00%. Paired DSRF-minus-XGBoost difference: -7.14 percentage points (interval -28.57 to +14.29 pp). Exploratory exact McNemar p=1.00000, with 1 patients correct only for DSRF and 2 correct only for XGBoost.

## Discussion and limitations

The contribution is a documented patient-level evaluation of an existing pretrained EEG encoder with a custom, interpretable decoder and a reliability-focused stress-test design. Small held-out sample size, one source cohort, exploratory model-selection comparisons, possible demographic or recording confounding, artificial rather than real-world acquisition corruption, and the absence of external validation limit generalization. Calibration measurements on only 13 held-out subjects are especially uncertain.

## Conclusion

Pretrained EEG representations with dual-scale residual fusion can yield competitive exploratory Alzheimer's-versus-control discrimination, but prediction stability under corruption is distinct from clean performance. Findings do not predict future Alzheimer's risk or establish a clinical diagnostic system. Independent prospective validation would be required.

## Reproducibility and writing tasks remaining

The five outer development folds, model weights, final training protocol, final held-out predictions, paired statistics, robustness records and explainability outputs have been stored. Add verified citations, ethics/data-use statements and institution-specific thesis formatting separately. Do not claim external validation or clinical applicability.
