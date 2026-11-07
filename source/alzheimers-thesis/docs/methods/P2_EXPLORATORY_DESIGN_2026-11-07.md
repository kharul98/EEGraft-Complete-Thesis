# P2 Exploratory Adaptive EEG Development

Date: 2026-11-07

Status: EXPLORATORY, NOT CONFIRMATORY

## Objectives

1. Extract robust prefix-valid spectral measurements.
2. Evaluate whether elementary stopping policies can
   generate an informative duration-performance trade-off.
3. Identify preprocessing and model-design weaknesses
   before selecting the final proposed method.

## Cohort

NEMAR on004504 v1.0.0

88 participants:
- 36 AD
- 29 cognitively normal
- 23 FTD

Primary exploratory task: AD versus CN.
Secondary exploratory task: AD versus FTD.

## Feature engineering

Causal 1-40 Hz Butterworth filtering.
Ten-second chronological signal chunks.
State carried between adjacent chunks.

Per-epoch Welch spectral powers.
Mean and median aggregation over epochs/channels.

Duration checkpoints:
30, 60, 120, 180, and 300 seconds.

No future signal may influence an earlier prefix.

Raw amplitude above 200 microvolts is only a
descriptive quality proxy, not an artifact diagnosis.

## Classifier feasibility

Participant-level repeated stratified 5-fold CV,
three repeats.

Training-only standardization.
Training-only inner 3-fold regularization selection.

Logistic regression is a feasibility baseline,
not a final proposed architecture.

## Exploratory stopping policies

Fixed durations:
30, 60, 120, 180, 300 seconds.

Adaptive baselines:
confidence-only,
spectral-stability-only,
confidence-and-stability,
confidence-and-quality.

Combined prototype:
confidence, stability, and quality together.

Confidence thresholds are arbitrary pilot choices.
Classifier probabilities are not calibrated.

Quality and stability thresholds use training data only.

No held-out test labels may influence a stopping
decision or model selection.

## Primary descriptive outputs

- Balanced accuracy
- AD sensitivity
- Non-AD specificity
- ROC-AUC
- Mean and median stopping duration
- Early stopping rate
- False-negative AD count

## Statistical limitations

Repeated test folds do not create new participants.
No test-time threshold tuning.
No confirmatory significance testing.
No formal equivalence or non-inferiority claim.
No clinical diagnostic claim.

Results will guide design but cannot confirm
the final hypothesis.

## Pending research work

- Validated artifact-aware preprocessing
- Calibrated prediction or other justified
  uncertainty estimation
- Fair time-budget-matched adaptive baselines
- Training-only policy optimization
- Robustness and independent validation
- Statistical precision and hypothesis protocol
- Ablations of each proposed component
- Confirmatory evaluation without test reuse

## Research contribution status

NOT YET ESTABLISHED.

A complex algorithm or favorable pilot result
must not be mistaken for novelty.
