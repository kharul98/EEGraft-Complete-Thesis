# Figure-Text-Data Consistency Check

**Evidence authority:** where source EEG signals or individual predictions were unavailable, percentages were compared with saved executed aggregate summaries in the project and prior figure report. They were not re-executed from raw restricted recordings. Four dense figures were deliberately rebuilt at readable size, preserving their verified data and primary/secondary protocol distinction.

| Number | Original filename retained | Thesis subject | Evidence and sample unit | Caption/text agreement | Status |
|---|---|---|---|---|---|
| 1.1 | `fig_01_roadmap.pdf` | Overall three-study methodology | Three-study roadmap; 88 NEMAR people, CAUEEG recordings; schematic | Referenced | Approved original retained |
| 4.1 | `fig_04_01_cohort_flow.pdf` | Cohort selection across the four analyses | Cohort counts; A/B people, C/Isfahan recordings; simplified, source verified | Referenced | Replaced at larger print size; data unchanged |
| 4.2 | `fig_04_02_dataset_evidence.pdf` | Dataset metadata and available EEG evidence | Dataset metadata and available source EEG excerpts; privacy limits noted | Referenced | Approved original retained |
| 4.3 | `fig_04_03_preprocessing.png` | EEG preprocessing and window construction | Fixed Study B/C signal adapter and schematic windows (no measured EEG invented) | Check text reference | Approved original retained |
| 5.1 | `fig_05_01_architecture.png` | EEGraft CBraMod--DSRF architecture | Frozen CBraMod + DSRF source design, 19 x 4 x 200, kernels 9/41 | Referenced | Approved original retained |
| 5.2 | `fig_05_02_threshold_protocol.pdf` | CAUEEG frozen transfer and threshold protocol | Official 950/82/90 CAUEEG recordings; original 0.50, secondary 0.70 | Check text reference | Replaced at larger print size; data unchanged |
| 6.1 | `fig_06_01_development.pdf` | Five-fold source development comparison | 52-person source development comparison; DSRF 77.06%, XGBoost 76.61% | Referenced | Approved original retained |
| 6.2 | `fig_06_02_ablations_robustness.png` | Decoder ablations and synthetic stress tests | 52-person ablation and five seeded artificial stress tests | Referenced | Approved original retained |
| 6.3 | `fig_06_03_source_holdout.png` | Locked source confusion matrices | 13-person source holdout, DSRF 9/13, XGBoost 10/13 | Referenced | Approved original retained |
| 6.4 | `fig_06_04_external.pdf` | CAUEEG locked external classification | 57 CAUEEG test recordings; observed matrices 24/11/7/15; 6/29/0/22 | Referenced | Replaced at larger print size; data unchanged |
| 7.1 | `fig_07_01_temporal.pdf` | Five-minute temporal accuracy and prediction states | NEMAR study A 46/41 people; correlated 230/205 repeat states | Referenced | Approved original retained |
| 7.2 | `fig_07_02_mci.pdf` | Isfahan MCI stability negative control | Isfahan 61 screened, 49 retained across two releases | Check text reference | Approved original retained |
| 7.3 | `fig_07_04_unanimous.pdf` | Unanimous correct and wrong external predictions | CAUEEG unanimous 4/0/53 DSRF, 13/12/32 XGBoost | Check text reference | Approved original retained |
| 7.4 | `fig_07_03_threshold_results.pdf` | CAUEEG threshold sensitivity and calibration | CAUEEG Code 19/23 aggregates, development flips and test BAcc/Brier | Referenced | Replaced at larger print size; data unchanged |
| 7.5 | `fig_07_05_eye_audit.pdf` | Eye-state and elapsed-time audit | CAUEEG eye-state/time 404 train, 42 validation; exploratory | Check text reference | Approved original retained |
| 7.6 | `fig_07_06_age_audit.pdf` | CAUEEG age confounding and incremental EEG value | CAUEEG age validation difference 11.23/13.43 years, AUROC gain +0.0118 | Check text reference | Approved original retained |
| A.1 | `fig_A_01_electrode_occlusion.pdf` | Out-of-fold electrode-occlusion sensitivity | NEMAR source channel occlusion 52 development people | Referenced | Approved original retained |
| A.2 | `fig_A_02_reproducibility.pdf` | Reproducibility evidence chain | NEMAR reproducibility hash references; excerpted provenance | Check text reference | Approved original retained |
| A.3 | `fig_A_03_nemar_confounds.pdf` | NEMAR demographic and condition-sensitivity audits | NEMAR age/condition source audit; observational context | Check text reference | Approved original retained |

**Examples of checked arithmetic:** NEMAR 36+23+29=88; source 52+13=65; CAUEEG 950+82+90=1,122; binary test 35+22=57; 24+11+7+15=57; 6+29+0+22=57; DSRF BAcc=(15/22+24/35)/2=68.38%; XGBoost BAcc=(22/22+6/35)/2=58.57%.

**Remaining validation boundaries:** Restricted CAUEEG individual score arrays and the source original duration replay were not in this upload. Isfahan preview units and raw excerpts retain source-level limits. Figure/measure totals are source-reported where raw evidence is inaccessible. A document build is not evidence of clinical validity.
