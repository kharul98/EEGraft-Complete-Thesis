# P2 Training-Only Spectral Reliability Gate

Status: EXPLORATORY development, 2026-11-07.

## Primary question

Can participant-specific EEG acquisition stopping
offer a useful accuracy-duration trade-off compared
with fixed recording durations?

Primary task: AD versus cognitively normal controls.

## Data

NEMAR on004504 v1.0.0, 65 participants:
36 AD and 29 controls.

Uses previously verified causal, ten-second,
epoch-median spectral feature prefixes.

Checkpoints: 30, 60, 120, 180, 300 seconds.

## Candidate policy

At each checkpoint from 60 to 180 seconds:

- Measure spectral change from the preceding prefix.
- Optionally require a sufficiently large classifier margin.
- Optionally require sufficiently low spectral variability.
- Otherwise acquire the next prefix.
- Stop at 300 seconds if earlier conditions do not pass.

All decisions use the current or earlier prefix only.

## Selection and controls

Outer evaluation: stratified five-fold CV, one
exploratory repeat, participant-level splits.

Classifier: standardized logistic regression.
C selected by inner three-fold CV.

Within each outer training partition, produce
three-fold cross-fitted predictions using a
prespecified C=1 logistic model.

Select stopping parameters using training subjects
only, without accessing outer-test labels.

Training-only empirical eligibility:

- Balanced accuracy no more than 0.03 below fixed 60s.
- AD sensitivity no more than 0.05 below fixed 60s.
- Mean acquisition time no greater than 120 seconds.

Among eligible candidates maximize:
balanced_accuracy - 0.04 * mean_seconds / 300.

If none are eligible, use fixed 60 seconds.

These are empirical development guardrails,
NOT statistical sensitivity guarantees.

## Comparison policies

- Fixed 30, 60, 120, 180 and 300 seconds.
- Stability-only with training 75th-percentile cutoff.
- Label-independent randomized 60/120-second comparator
  matched to the training-estimated gate time budget.
- Training-selected reliability gate.

## Important limitations

Uncalibrated probabilities are used only as margins.
There is no validated clinical risk control.

The dataset has already informed method development.
Results are exploratory and not an independent test.

Do not select or alter policy parameters using
outer-test results.

No formal non-inferiority or novelty claim.

Generate no figures at this routine development step.
