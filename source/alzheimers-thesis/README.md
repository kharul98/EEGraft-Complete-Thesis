# Alzheimer's EEG Recording-Duration Thesis

Reproducible bachelor's thesis investigating how the amount of available
resting-state EEG affects Alzheimer's-related spectral measurements and
subject-level classification.

## Current scientific direction

The active study investigates recording-duration sensitivity using the
NEMAR/OpenNeuro on004504 resting-state EEG dataset.

Verified dataset structure:

- 88 participants
- 36 Alzheimer's disease
- 23 frontotemporal dementia
- 29 cognitively normal controls
- 19 EEG channels
- 500 Hz sampling
- 88/88 preprocessed recordings readable
- minimum usable recording duration: 306.1 seconds

The current candidate duration conditions are:

- 30 seconds
- 60 seconds
- 120 seconds
- 180 seconds
- 300 seconds

The primary planned comparison is Alzheimer's disease versus cognitively
normal controls.

Alzheimer's disease versus frontotemporal dementia is retained as a
prespecified disease-specificity analysis.

## Research principles

This repository prioritizes:

- participant-level independence
- leakage-controlled evaluation
- predeclared experimental decisions
- uncertainty and effect-size reporting
- preservation of negative results
- exact dataset provenance
- reproducible experiment records
- claims proportional to evidence

## Data

Large EEG files are intentionally excluded from GitHub.

The original dataset is:

NEMAR on004504, version v1.0.0

DOI:

10.82901/nemar.on004504

Dataset manifests, version information, hashes, code and permitted derived
results are retained in this repository.

## Research records

Important project records include:

- `RESEARCH_STATE.md`
- `RESEARCH_HISTORY.md`
- `OPEN_ITEMS.md`
- `PREREGISTRATION.md` when the protocol is frozen
- `EXPERIMENT_REGISTRY.md` once experiments begin
- `docs/defense/DECISION_LOG.md`
- `docs/defense/NOVELTY_EVIDENCE.md`
- `docs/defense/METHOD_JUSTIFICATIONS.md`

## Status

Dataset feasibility: VERIFIED

Primary scientific protocol: pending final freeze before primary experiments.

Primary experimental results: not yet executed at the time of this README.
