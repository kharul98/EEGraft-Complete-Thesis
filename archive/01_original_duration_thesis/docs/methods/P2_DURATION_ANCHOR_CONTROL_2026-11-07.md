# P2 Duration versus Temporal-Anchor Control

Status: EXPLORATORY

## Scientific question

Does improved agreement with a later EEG reference
persist when recording duration changes but the
measurement endpoint is held fixed?

## Dataset

AHEPA / NEMAR on004504 v1.0.0.
All 88 previously verified participants.

## Existing reference

240-300 seconds of the same recording.

Reference interval is disjoint from all evaluated windows.

The reference is not an independent recording session
or a clinical ground truth.

## Analysis designs

1. Starting at zero:
   0-30, 0-60, 0-120, 0-180 seconds.

2. Ending at 180 seconds:
   150-180, 120-180, 60-180, 0-180 seconds.

3. Equal 60-second windows:
   0-60, 60-120, 120-180, 180-240 seconds.

All windows use the same chronological causal filter
with state carried across the recording.

Consequently, later windows inherit filter state
from previous samples. They are position-control
analysis windows, not standalone new acquisitions.

## Endpoint

Normalized five-band spectral total-variation distance
to the 240-300-second reference.

Lower means closer spectral agreement.

## Statistical unit

Unique participant, n=88.

Participant-paired descriptive comparisons.

5000 stratified participant-bootstrap resamples.

All comparisons remain exploratory and post-hoc.
No clinical, equivalence, or novelty claims.

No figure generation is needed at this step.
