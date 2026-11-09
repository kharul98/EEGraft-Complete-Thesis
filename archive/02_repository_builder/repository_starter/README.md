# EEGraft - Complete Thesis Research Archive

**Official thesis title:** EEGraft: A Gated Dual-Scale Residual Fusion Decoder and Subject-Level Reliability Assessment for Alzheimer's EEG Classification

A **chronological guide and archival consolidation** of an undergraduate EEG dementia research project. The repository preserves the original duration-study and broader research-backup source trees as separately imported Git subtrees, and provides the final editable LaTeX thesis in `thesis/overleaf/`.

> **Research boundary:** This is retrospective, within-dataset EEG classification research, **not a clinical diagnosis or prediction of future Alzheimer's disease**. The two main experimental tracks are related but have distinct participant selections and protocols. EEGraft is the project's study identity; CBraMod is an externally published pretrained model and DSRF is the custom decoder.

## Start here: research in chronological stages

| Stage | Question or activity | Guide |
|---|---|---|
| 00 | Dataset feasibility, participant independence, preprocessing | [00 - Dataset and foundations](phases/00_dataset_and_protocol.md) |
| 01 | Earlier spectral/SVM five-minute temporal reliability (AD/CN, FTD/CN) | [01 - Temporal reliability](phases/01_temporal_reliability.md) |
| 02 | Recording duration and time-anchor analysis | [02 - Recording duration](phases/02_duration_and_anchors.md) |
| 03 | Additional model exploration and external-dataset feasibility checks | [03 - Feasibility and controls](phases/03_controls_and_external_feasibility.md) |
| 04 | Frozen CBraMod backbone with custom dual-scale residual fusion decoder | [04 - Decoder architecture](phases/04_cbramod_dsrf.md) |
| 05 | Subject-disjoint development comparisons and ablations | [05 - Development results](phases/05_development_and_ablations.md) |
| 06 | Synthetic perturbation tests and channel masking | [06 - Reliability tests](phases/06_robustness_and_explanations.md) |
| 07 | One-time locked 13-participant test | [07 - Held-out assessment](phases/07_final_holdout.md) |
| 08 | Complete thesis text, 13-figure report package and reproducibility | [08 - Final writeup](phases/08_writeup_and_figures.md) |

## Principal results (different protocols; do not pool)

| Track | Participants | Measure | Recorded result |
|---|---|---|---|
| Original temporal SVM | 64 analytical participants (23 AD, 18 FTD, 23 CN) | First-minute balanced accuracy AD/CN | 73.48% |
| Original temporal SVM | Same | Predictions changing within five minutes, AD/CN | 44.35% |
| Later CBraMod-DSRF | 65 AD/CN (52 development, 13 held out) | Development DSRF pooled balanced accuracy | 77.06% |
| Later CBraMod-DSRF | Same | Development XGBoost pooled balanced accuracy | 76.61% |
| One-time within-cohort holdout | 13 (7 AD, 6 CN) | DSRF balanced accuracy | 70.24% |
| One-time within-cohort holdout | Same | XGBoost balanced accuracy | 77.38% |

The XGBoost comparator did **better** on the small final holdout. No independent external AD diagnostic validation is claimed. See [Results and evidence](docs/RESULTS_AND_EVIDENCE.md).

## Repository structure

```text
EEGraft-Complete-Thesis/
  README.md                              <- chronological entrypoint
  phases/                               <- staged navigational research record
  docs/                                 <- evidence, boundaries, data and reproducibility
  archive/
    01-legacy-duration-study/           <- unmodified upstream contents + Git history (on build)
    02-research-backup/                 <- unmodified upstream contents + Git history (on build)
  thesis/overleaf/                      <- corrected editable LaTeX, figures and references
  tools/                                <- audit / validation scripts
```

`archive/` is filled automatically by `prepare_eegraft_repository.sh` from the separate builder package. Folder numbers express the **reading order**, not a claim that every commit in both histories occurred in that order.

## Using the repository

1. Read the [research questions](docs/RESEARCH_QUESTIONS.md) and [protocol map](docs/PROTOCOL_MAP.md).
2. Navigate the numbered `phases/` pages to find the saved experiments and reports within the two imported source trees.
3. Open `thesis/overleaf/main.tex` in Overleaf or compile locally with `latexmk -pdf` and Biber.
4. Use the [reproducibility and data guide](docs/REPRODUCIBILITY.md) to locate datasets and checkpoint requirements.
5. Review [known limitations](docs/LIMITATIONS_AND_AUDIT.md) before publication or viva.

## Original sources (retained separately)

- https://github.com/kharul98/alzheimers-eeg-duration-thesis
- https://github.com/kharul98/eeg-dementia-thesis-backup-20261107T215131Z

No original repository is deleted, renamed, or force-pushed by the builder. Both histories are imported without squash commits. Imported historical commits may contain material not obvious from the latest files; **start with a private repository and audit history before making it public**.

## Privacy and provenance

Original student handwritten signatures and signed PDFs are **excluded** from the bundled LaTeX source. The thesis compiles with blank signature space; keep signed copies private. Raw EEG datasets, third-party source archives, credentials and personal health records are not included in this starter. The backup repository itself already omits raw EEG data and may contain Git LFS checkpoint pointers; see the [LFS note](docs/REPRODUCIBILITY.md).

The 13 report figure positions include some derived numerical charts; they should not be represented as an exact recovered copy of the earlier approved 13-image set. See [figure notes](docs/FIGURES.md) and `thesis/overleaf/reproducibility/FIGURE_PROVENANCE.json`.
