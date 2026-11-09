# EEGraft Scientific Verification Report

## Evidence tiers

**Tier A — independently checked in the available source/material:** NEMAR original source protocol and archived result tables (not raw signal rerun), 13-person confusion-matrix arithmetic, public NEMAR/CAUEEG dataset paper and official CAUEEG split definitions, fixed LaTeX figure references, and 19 selected file hashes. We recalculated BAcc and accuracy from all four locked confusion matrices.

**Tier B — reported from earlier local experiment execution, not independently replayed here:** CAUEEG Code 19 locked-test AUROCs/Brier/bootstrap intervals; Codes 21–23 eye-state/timing/threshold comparisons; Codes 24–25 age/age-adjusted model analysis; Isfahan release-specific SVM aggregates; original five-seed stress outcomes and electrode-occlusion details. The user-provided aggregate logs and PNG/PDF assets support the report wording, but original restricted EDF/test predictions and training checkpoints are not mounted.

**Tier C — exploratory interpretation:** threshold-dependent label stability, observed eyes-open/closed associations, age-sensitivity implications and incremental EEG value. These are not prospective or causal findings; their uncertainty reflects recording-level resampling and may understate variation if patients contributed multiple recordings.

## Critical numerical verification

| Experiment | Data | Verified arithmetic or reported source | Interpretation |
|---|---|---|---|
| NEMAR source | 88 people: 36 AD, 23 FTD, 29 CN | NEMAR public dataset | source population |
| Study A | 64 people, AD/CN n=46, FTD/CN n=41 | 108+20+102=230; 109+27+69=205 correlated repeat states | not 435 independent people |
| Study B | 65 AD/CN people: 52 development and 13 locked | DSRF matrix [[4,3],[1,5]] => BAcc 70.24%, 9/13; XGB [[5,2],[1,5]] => BAcc 77.38%, 10/13 | observed source locked outcome favors XGB |
| CAUEEG | original 1,379 recordings; Dementia task 1,187; official no-overlap 1,122 (950/82/90) | 436 Normal + 395 MCI + 291 Dementia = 1,122 | recordings, not verified unique people |
| CAUEEG primary | 57 binary test recordings (35 Normal, 22 Dementia) | DSRF [[24,11],[7,15]] => 68.38% BAcc; XGB [[6,29],[0,22]] => 58.57% BAcc | threshold 0.50, frozen source models |
| CAUEEG secondary | same 57 locked records | XGB training-selected 0.70 BAcc 66.10%; Platt DSRF 65.58%, XGB 61.88% | exploratory threshold/calibration only |
| CAUEEG test uncertainty | 5,000 reporting-bootstrap resamples | paired 95% BAcc difference interval [-3.31,+22.08] pp | crosses zero; superiority not confirmed |
| Whole-recording unanimity | 24 windows for each of 57 recordings | DSRF stable correct=4, stable wrong=0; XGB stable correct=13, stable wrong=12 | consistency is not correctness |
| Eye-state audit | 404 training and 42 validation recordings with eligible pairs | XGB train flip at 0.50=29/404 (7.18%); 0.70=134/404 (33.17%) | threshold effects on labels, not retraining |
| CAUEEG age | train Dementia minus Normal mean age +11.23 years; validation +13.43 years | age-only validation AUROC .8397; adding DSRF +.0118 with 95% CI [-.0860,+.1081] | validation incremental EEG benefit not established |
| Isfahan MCI | 61 screened, 49 retained | release BAcc 40.78%/55.83%; stable-wrong 59.20%/35.83% | independent negative control, not AD validation |

## History and protocol correction

The earlier historical duration audit expected 690 AD rows but encountered 1,380 because the archived CSV contains **both** 690 SVM and 690 logistic-regression rows. This is not 690 duplicate EEG records. Source-to-source checks matched all first-minute SVM prediction keys after model filtering. The halted audit's downstream report was **not** independently rerun, and the duration figure is therefore omitted from the final main manuscript.

## Technical restrictions and chronology

- Source backbone is **frozen CBraMod**, not a newly invented foundation encoder; raw branch convolution widths 9 and 41, class attention and gated residual addition are described by the archived implementation.
- Source NEMAR 500 Hz becomes 200 Hz; CAUEEG starts at 200 Hz; 19 aligned EEG channels; 4-second non-overlapping windows; at most 24 spaced windows per recording. No new ICA or per-recording z-score transformation was added in the final adapter.
- 0.50 is the locked CAUEEG primary rule for **both** models. The XGBoost 0.70 threshold and train-fitted Platt variants are **secondary** and not new independent test sets.
- The source Brier convention and CAUEEG single-class binary Brier convention are **not directly compared**.
- CAUEEG Dementia is a broader target label than NEMAR AD. Clinical diagnosis, prospective risk and patient-independent confidence intervals are not established.

## Literature verification

The original CAUEEG paper is Kim, Youn and Paik, *NeuroImage* 272 (2023) 120054, DOI `10.1016/j.neuroimage.2023.120054`. The official source repository is `https://github.com/ipis-mjkim/caueeg-dataset` and explicitly describes approval-controlled research access. NEMAR/OpenNeuro source is Miltiadous and colleagues, OpenNeuro `ds004504` / NEMAR `on004504`.

## Overall status

**Compiled and internally reconciled from available evidence, NOT independently reproducible end-to-end from this ZIP alone.** No new raw EEG fitting, locked-test tuning, or human data fabrication occurred during the editorial revision.
