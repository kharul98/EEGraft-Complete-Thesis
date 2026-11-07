# EEG Dementia Thesis - Research Milestone Register

Research status: November 2026.

## Research question

Can EEG-based dementia classifiers be reliable across time,
recording conditions, subjects and signal perturbations,
rather than merely achieving high classification accuracy?

## Phase A: Original AD/FTD reliability study

Reported analytical cohort: 64 participants:
23 Alzheimer's disease (AD), 18 frontotemporal dementia
(FTD), and 23 healthy controls.

Tasks:
- AD versus healthy controls.
- FTD versus healthy controls.

Features:
- Relative delta power
- Relative theta power
- Relative alpha power
- Relative beta power
- Relative gamma power
- Theta/alpha ratio

Method:
Standardized class-balanced RBF support vector machine.
Subject-disjoint repeated nested cross-validation:
five outer folds, five repeats, three inner folds.
Models trained on first-minute features and evaluated
on the held-out participants' five EEG minutes.

Reported results:

| Metric | AD vs CN | FTD vs CN |
| --- | ---: | ---: |
| First-minute BAcc | 73.48% | 73.72% |
| Prediction changed in five minutes | 44.35% | 33.66% |
| Stable but wrong | 8.70% | 13.17% |
| Eyes-open minus eyes-closed BAcc | -8.83 pp | -4.23 pp |

The recording-condition intervals included zero.
Eight feature-ablation comparisons yielded no
Holm-adjusted statistically significant result.

Repeated-CV assessments are correlated and do not
represent additional independent participants.

## Phase B: Independent Isfahan MCI investigation

61 recordings: 29 MCI and 32 healthy.
Five-minute quality screening retained 49 and excluded 12.

EEG_1: 11 MCI and 14 healthy retained.
EEG_2: 12 MCI and 12 healthy retained.

| Metric | EEG_1 | EEG_2 |
| --- | ---: | ---: |
| First-minute BAcc | 40.78% | 55.83% |
| MCI sensitivity | 67.27% | 68.33% |
| Healthy specificity | 14.29% | 43.33% |
| Stable but wrong | 59.20% | 35.83% |
| Unstable predictions | 5.60% | 20.00% |

A constant-label classifier has 50% balanced accuracy.
This is MCI classification, not AD/FTD diagnosis.
Cross-release participant linkage and units require caution.

## Phase C: External Figshare EEG audit

Archive inventory contained:
305 AD, 56 MCI and 110 control MATLAB files.
These numbers are recording files, NOT participant counts.

README participant counts:
59 AD, 7 MCI and 102 controls.
Only 57 explicit AD participant folders were verified
in the earlier inspection.

Sampling rates and channel formats differed.
No confirmed external AD validation metric was produced.

## Phase D: Pretrained CBraMod and custom DSRF

Later OpenNeuro ds004504 AD-versus-CN analytical cohort:
65 participants, including 36 AD and 29 controls.

39 training, 13 validation and 13 locked-test subjects.
52 development subjects evaluated in five outer folds.

Pretrained CBraMod checkpoint SHA256:
0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178

A custom dual-scale residual fusion decoder combines:
- High-level pretrained EEG token features
- Low-level temporal convolutional features
- Learned gates
- Residual fusion
- Class-attention pooling

Development pooled balanced accuracy:
DSRF 77.06%; XGBoost 76.61%;
EEGNet-inspired 74.89%;
Random Forest 72.71%;
High-only 72.26%;
SVM 71.81%;
No-skip 71.44%;
Low-only 62.74%.

Matched residual versus gated-blend comparison:
77.06% versus 73.16% pooled balanced accuracy.
The difference was exploratory and inconclusive.

## Phase E: EEG model interpretability

Out-of-fold explanation of 52 development participants.
The full-window analysis agreed with all 52 original
cross-validation patient decisions.

Highest reported electrode masking sensitivities:

AD: T5, Pz, F8, Fz, C3.
CN: O2, T6, O1, Fp2, F3.

Electrode masking reveals model sensitivity, not
a causal or clinically validated brain biomarker.

## Phase F: Repeated reliability experiments

Five seeded perturbation realizations per condition
on the same 52 development patients.

DSRF mean balanced accuracy:
Clean 77.06%
10% Gaussian noise 71.87%
20% Gaussian noise 70.07%
One channel zeroed 76.78%.

These are artificial perturbations, not
real-world acquisition validation.

## Phase G: Locked final holdout

Final frozen DSRF: 5 training epochs on 52 development subjects.
Final XGBoost: trained on the same 52 development subjects.

Original 13 held-out patients: 7 AD, 6 controls.

DSRF:
Balanced accuracy 70.24%;
accuracy 69.23%;
macro-F1 69.05%;
AUROC 0.857;
9/13 correctly classified.

XGBoost:
Balanced accuracy 77.38%;
accuracy 76.92%;
macro-F1 76.92%;
AUROC 0.905;
10/13 correctly classified.

DSRF minus XGBoost balanced accuracy: -7.14 pp.
Exploratory paired McNemar p = 1.0.

XGBoost was stronger on the small final holdout.
Neither system has external clinical diagnostic validation.

## Important methodological boundaries

- The original 64-person study and the later 65-person
  analytical cohort have not been reconciled at the
  participant-identity level.
- Five-minute temporal stability was evaluated in the
  original SVM research, not directly replicated by
  the later 24-window DSRF classification pipeline.
- The Isfahan MCI experiment is a separate disease task.
- No independent external AD validation has been established.
- Statistical comparisons are exploratory.
- These models do not predict future Alzheimer's risk.
- Original early-study code and provenance must be verified
  against the saved files before claiming full replication.
