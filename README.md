# EEGraft: Complete Thesis Research Archive

**EEGraft: A Gated Dual-Scale Residual Fusion Decoder and
Subject-Level Reliability Assessment for Alzheimer's EEG Classification**

This repository consolidates the EEGraft research project,
including its earlier EEG classification investigations,
custom architecture development, reliability experiments,
baseline comparisons, evaluation records, research figures,
and thesis artifacts.

## Research objective

The project investigates EEG-based Alzheimer's disease
classification with emphasis on subject-level reliability,
model comparison, robustness, and honest evaluation.

The work includes a custom gated dual-scale residual fusion
decoder built using representations from a pretrained
EEG foundation model.

## Repository structure

- `source/alzheimers-thesis/`
  Updated primary research code, configurations, and results.

- `archive/01_original_duration_thesis/`
  Historical source from the original EEG duration repository.

- `imports/`
  Additional local research directories imported during consolidation.

- `evidence/`
  Approved derived evaluation outputs, figures, and research evidence.

- `manuscript/`
  Available thesis PDF and Overleaf source artifacts.

- `docs/`
  Research timeline, source lineage, reproducibility notes,
  and verification inventories.

## Research progression

1. Initial feasibility and dataset selection.
2. Recording-duration and temporal reliability investigations.
3. Spectral-feature baseline experiments.
4. Subject-level classification and cross-validation.
5. Independent MCI EEG negative-control investigation.
6. Pretrained CBraMod evaluation.
7. Development of the dual-scale residual fusion decoder.
8. Baseline and decoder-ablation comparisons.
9. Robustness, stability, and interpretation analyses.
10. Locked holdout evaluation.
11. Subsequent external EEG transfer experiments.
12. Final thesis documentation and reproducibility audit.

The latest external experiments must be evaluated from
their actual imported execution records. This README
does not independently certify numerical results.

## Research integrity

The repository distinguishes experimental outcomes,
exploratory analyses, held-out evaluations, and external
validation. No clinical diagnostic authorization is claimed.

## Data availability

Raw EEG recordings are not redistributed by default.
Obtain datasets through their official authorized sources.
Dataset versions, access conditions, and acquisition
instructions should be documented in `docs/DATASETS.md`.

## Historical provenance

The original GitHub repositories were:

- kharul98/alzheimers-eeg-duration-thesis
- kharul98/eeg-dementia-thesis-backup-20261107T215131Z

Their history is preserved through Git ancestry and
the original research archive directory.

The root `PROJECT_HISTORY.md` and
`FILES_SHA256SUM.txt` originated from the November 7
backup. The older checksum manifest is historical,
not a verification of the subsequently updated tree.

## Reproducibility

Consult `docs/REPRODUCIBILITY.md` and the actual
experiment configurations before executing a study.

Model files are managed with Git LFS where applicable.
