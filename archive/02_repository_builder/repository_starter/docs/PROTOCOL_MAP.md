# Protocol and cohort boundaries

| Research stream | Population | Evaluation | Meaning / caution |
|---|---|---|---|
| Earlier temporal-reliability SVM | NEMAR on004504: 64 analytical participants = 23 AD, 18 FTD, 23 controls | Repeated nested subject-disjoint CV, five outer folds × five repeats, three inner folds, first-minute training, five-minute prediction assessments | Reports temporal label changes, correct/stable/wrong rates; repeated folds are not new participants |
| Focused duration study | Later duration investigation refers to a **65-person** AD/CN analytic cohort, recording segments 30, 60, 120, 180, 300 seconds | Duration and anchor studies | Dataset/identity reconciliation with original 64 not established; do not infer a universally optimal duration |
| Earlier nine-model saved comparison | 46 analytical participants | Saved out-of-fold metrics | Distinct analysis; do not directly equate with later 52-person development |
| Isfahan MCI extension | 61 recordings; 49 passed QC (23 MCI, 26 HC) | Two separate EEG releases | MCI/HC, **not external AD verification**; participant linkage and units uncertain |
| External Figshare archive audit | 471 MAT files, not 471 people | Compatibility and signal/channel-format audit | Independent AD diagnostic validation **not performed** |
| CBraMod–DSRF AD/CN | 65 individuals = 36 AD, 29 CN; 52 development, 13 locked test (7 AD, 6 CN) | Five patient-disjoint outer folds; final once-only evaluation | Frozen pretrained encoder, custom decoder; within-dataset only |
| Robustness | Same 52 development subjects | Five random seeds of artificial noise/masking conditions | Seeds are repeats, **not additional cohorts** |

**Important distinctions:** The original five-minute SVM label-stability protocol was not reproduced identically with the 24-window DSRF pipeline. Development model selection does not establish external validation. All reported comparisons are exploratory unless a source file explicitly justifies a stronger claim.
