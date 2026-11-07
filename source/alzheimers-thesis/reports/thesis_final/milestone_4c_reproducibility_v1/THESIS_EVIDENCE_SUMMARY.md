# Short-EEG dementia classifier reliability: consolidated exploratory evidence

**Objective:** Evaluate whether short-EEG dementia classifiers remain reliable across time and recording conditions, and examine uncertainty, clinical severity and spectral feature sensitivity.

This is a methodological reliability study, **not** a prospective Alzheimer-risk predictor or independent clinical validation.

## 1. Study and validation design

- Eyes-closed cohort: 23 AD + 23 healthy (AD task, n=46), and 18 FTD + the same 23 healthy (FTD task, n=41).
- StandardScaler and class-weighted RBF SVM, five repeats of five outer participant-stratified folds, with nested training-only model selection.
- CV repetitions share participants and are correlated. Healthy controls also overlap across the two tasks.

## 2. Temporal prediction reliability

| Endpoint | AD vs HC | FTD vs HC |
|---|---:|---:|
| Original first-minute BACC | 73.48% | 73.72% |
| Stable and correct | 46.96% | 53.17% |
| Stable but wrong | 8.70% | 13.17% |
| Predicted class changes | 44.35% | 33.66% |

Stable predictions can be wrong. The percentages represent correlated participant-repeat observations.

## 3. Calibrated uncertainty and abstention

| Measure | AD vs HC | FTD vs HC |
|---|---:|---:|
| First-minute Brier score | 0.1861 | 0.1917 |
| Training-prevalence Brier baseline | 0.2513 | 0.2479 |
| First-minute error-detection AUROC | 0.6565 | 0.6663 |
| Stable-only error-detection AUROC | 0.5019 | 0.6239 |

Sigmoid probability calibration used only outer-training participants. Original SVM labels were preserved in selective prediction.
Selection rankings were retrospective and cohort-relative; they are not deployable abstention thresholds.

- AD, 60s confidence, 80% coverage: BACC gain +7.21 percentage points; conditional 95% CI [+1.86, +10.75] percentage points.
- AD, 300s consensus, 70% coverage: BACC gain +6.78 percentage points; conditional 95% CI [+1.84, +12.43] percentage points.
- FTD, 60s confidence, 80% coverage: BACC gain +3.67 percentage points; conditional 95% CI [-2.20, +7.54] percentage points.
- FTD, 300s consensus, 70% coverage: BACC gain +6.57 percentage points; conditional 95% CI [+0.99, +11.50] percentage points.

## 4. MMSE clinical severity

- AD patient-only: n=23, median MMSE=20.0.
- FTD patient-only: n=18, median MMSE=22.0.
- All selected patients had MMSE and age; healthy controls had constant MMSE=30.
- None of the four primary within-diagnosis MMSE tests survived Holm correction.
- Exploratory secondary AD MMSE versus stable-wrong rate: rho=+0.555, permutation p=0.0048, BH q=0.0480.
This secondary association is hypothesis-generating, not proof of disease-severity prediction.

## 5. Cross-condition robustness

Matched 30-second eyes-closed and eyes-open/photic EEG, from the same held-out people.

| Measure | AD vs HC | FTD vs HC |
|---|---:|---:|
| Matched participants | 19 AD + 22 HC | 13 FTD + 22 HC |
| Eyes-closed BACC | 66.94% | 73.64% |
| Eyes-open/photic BACC | 58.11% | 69.41% |
| Cross-condition agreement | 60.98% | 65.71% |
| EO minus EC BACC | -8.83 percentage points | -4.23 percentage points |

- AD paired BACC difference conditional 95% CI: [-22.92, +5.41] percentage points.
- FTD paired BACC difference conditional 95% CI: [-21.36, +12.76] percentage points.

Both paired BACC intervals include zero. Eyes-open state, photic stimulation, recording positions and preprocessing differences cannot be separated causally.
The 30-second rule recovered 14 participants relative to the original 60-second eligibility test; 10 of 64 remained ineligible.

## 6. Temporal model explanation stability

| Mean explanation drift | AD vs HC | FTD vs HC |
|---|---:|---:|
| Stable-correct | 0.6589 | 0.4296 |
| Stable-wrong | 0.9076 | 0.6279 |
| Unstable | 0.9442 | 0.5546 |

These effects are nonadditive local SVM-margin perturbations, not SHAP values or causal biomarkers.
The theta/alpha family contains three related features, while each other group contains one.

## 7. Predictive feature-group ablation

| Removed features | AD BACC change | FTD BACC change |
|---|---:|---:|
| Delta | +0.87 percentage points | +0.00 percentage points |
| Theta/alpha family | -9.57 percentage points | -10.51 percentage points |
| Beta | -4.35 percentage points | -2.27 percentage points |
| Gamma | -6.52 percentage points | -12.49 percentage points |

FTD gamma removal was the largest observed FTD loss: -12.49 percentage points; conditional bootstrap CI [-22.13, -4.08] percentage points; Holm-adjusted p=0.0688.

**None of the eight group-ablation effects survived multiple-comparison correction.**

## 8. Defensible thesis conclusion

Participant-disjoint internal evaluation demonstrates that short-EEG diagnostic predictions may vary across recording time and acquisition conditions. A nontrivial fraction of temporally consistent predictions are incorrect. Calibrated uncertainty can identify some errors retrospectively but cannot guarantee correctness.

The study does not demonstrate independent clinical validation, prospective Alzheimer risk prediction, deployable clinical abstention, or causal frequency-band biomarkers.

## 9. Important limitations

- Small, quality-selected cohorts; shared controls and repeated-CV dependence.
- Post-hoc analyses and multiple comparisons.
- Conditional uncertainty intervals do not capture all dataset-selection and model-development uncertainty.
- Eyes-open/photic condition and protocol differences are confounded.
- Cross-sectional MMSE cannot establish longitudinal disease progression or future conversion.
- Feature perturbations are nonadditive and are not direct physiological explanations.

## 10. Reproducibility

All numerical results were read from completed milestone CSV files and cross-checked where possible.
See evidence_master.csv, claim_audit.csv, sha256_inventory.csv and the source research scripts.
**Raw EEG recordings are excluded from the archive.** Dataset access is needed for independent signal reprocessing.
