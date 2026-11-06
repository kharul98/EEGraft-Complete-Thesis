# RESEARCH STATE

Date: 2026-11-06

## Current status

The project began with DARWIN handwriting as a feasibility candidate.

DARWIN data were successfully downloaded and hashed.

A current literature check identified a major novelty collision:
a peer-reviewed PLOS ONE article published on 2026-10-26 already performs
repeated nested validation, task-level reduction, calibration, and
cross-validated SHAP analysis on DARWIN.

Therefore DARWIN is retained only as a fallback.

## Active primary candidate

Working question:

How does the amount of available resting-state EEG affect the stability of
Alzheimer-related spectral measurements and leakage-controlled subject-level
classification performance?

Candidate dataset:

OpenNeuro ds004504 / NEMAR on004504.

The dataset contains resting-state eyes-closed EEG from:
- 36 Alzheimer's disease participants
- 23 frontotemporal dementia participants
- 29 healthy controls

The scientific protocol is NOT YET LOCKED.

The next feasibility gate is to verify the actual downloaded preprocessed
recordings, participant labels, channel structure, sampling rates, and usable
post-cleaning recording durations.

## Scientific principle

We will not claim novelty from building another classifier.

The possible contribution is a controlled recording-duration robustness study
using the same subjects at multiple fixed EEG durations, with participant-level
evaluation and uncertainty reporting.

## Status

DARWIN: FALLBACK
EEG duration study: ACTIVE CANDIDATE
Protocol: NOT LOCKED
Final experiments: NOT STARTED
