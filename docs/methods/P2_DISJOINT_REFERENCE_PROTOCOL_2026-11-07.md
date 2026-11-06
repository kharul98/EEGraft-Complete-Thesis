# P2 Disjoint-Reference EEG Reliability

Status: Exploratory methodological experiment.

## Research question

Do short causal EEG prefixes reproduce spectral
measurements from a later non-overlapping interval?

Does online feature stability predict disagreement
with that later interval?

## Data and eligibility

NEMAR on004504 v1.0.0.
All 88 previously verified participants.

No participant exclusions.

## Measurement construction

Causal 1-40 Hz Butterworth preprocessing.

Ten-second chronological EEG epochs.

Epoch-median relative spectral features.

Early prefix checkpoints:
30, 60, 120, 180 seconds.

Late reference:
240-300 seconds.

Early and reference intervals never overlap.

The 300-second full prefix is NOT evaluated
against the reference because they would overlap.

## Main descriptive endpoint

Total-variation distance between normalized
five-band spectral vectors.

Lower values indicate greater agreement.

The late reference is not ground truth.

## Additional checks

- Exact reproduction of previous causal features.
- Participant-paired 60-vs-120 and 60-vs-180 contrasts.
- Participant-resampled exploratory intervals.
- Online drift correlation with later disagreement.
- Group-matched different-subject negative control.
- Comparison with a delayed 60-second window.
- Reliability between the two late 30-second halves.

## Limitations

Same recording, same participant, different time window.
Not independent-patient validation.

Within-session vigilance or artifact changes may affect
early-to-late agreement.

Differences between recording duration conditions
may partly reflect temporal proximity to the reference.

No diagnostic performance or clinical claim.

No formal statistical equivalence claim.

No figures required at this routine research step.
