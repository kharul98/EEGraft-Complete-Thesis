# P2 Prefix Processing and Robustness Experiment

Date: 2026-11-07
Status: Exploratory methodological development

## Question 1: Future-information dependence

Compare zero-phase filtering of each 30-300-second
prefix with cropping a zero-phase-filtered 300-second
recording.

Use the same raw EEG and identical filter settings.

Measure waveform differences and relative spectral
feature differences.

Confirm invariance of causal filtering.

The 300-second reference is identical by construction.

This tests filtering look-ahead only, not ICA or ASR.

## Question 2: Robust amplitude limiting

Apply within-epoch median +/- 8 robust standard
deviations to completed 10-second raw EEG epochs.

Carry causal filtering state across epochs.

Evaluate the five predefined duration checkpoints.

This is a fixed exploratory preprocessing comparator,
not validated artifact correction.

No recording or participant is excluded.

## Question 3: Classification impact

Compare limited epoch-median features against the
previous unmodified epoch-median baseline.

Use exactly matched participant-level repeated
cross-validation folds and training-only model tuning.

Evaluate AD-versus-controls and AD-versus-FTD.

Outcomes remain exploratory.

No further threshold search is authorized by a
favorable or unfavorable result.

## Scientific restrictions

No confirmatory significance claims.
No equivalence or non-inferiority claims.
No clinical usefulness claims.
No external-validation claims.

Do not treat repeated folds as independent people.

No scientific figures are required at this routine
development stage.

Generate figures only for important verified
scientific milestones.
