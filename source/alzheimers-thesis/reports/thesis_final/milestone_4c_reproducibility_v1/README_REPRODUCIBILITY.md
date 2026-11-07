# Thesis evidence and reproducibility packet

This package was created without retraining models or
downloading new EEG recordings.

## Contents

- THESIS_EVIDENCE_SUMMARY.md: consolidated results
- evidence_master.csv: numerical results and intervals
- claim_audit.csv: supported and unsupported claims
- sha256_inventory.csv: source checksums and file sizes
- environment.json: package versions
- scripts/: available research scripts
- reports/: saved derived milestone results and figures

## Experiment sequence

1. Original eyes-closed preprocessing and classification
2. Milestone 1A: temporal stability and correctness
3. Milestone 1B: training-only calibration
4. Milestone 1C: selective prediction and error detection
5. Milestone 2A: MMSE metadata availability
6. Milestone 2B: clinical severity associations
7. Milestone 3A: matched eyes-open dataset metadata
8. Milestone 3B: eyes-open EEG acquisition and QC
9. Milestone 3C: 30-second continuity assessment
10. Milestone 3D: matched cross-condition transfer
11. Milestone 3E: spectral and prediction-transition audit
12. Milestone 4A: temporal explanation stability
13. Milestone 4B: feature-group ablation
14. Milestone 4C: this evidence package

## Important reproducibility limitations

This archive contains scripts, derived results, figures,
and provenance. It does NOT contain raw EEG recordings.

Signal processing and all source-level experiments require
original dataset access and the established project layout.

Dataset releases:
- OpenNeuro ds004504: eyes-closed EEG
- OpenNeuro ds006036 / NEMAR on006036: eyes-open/photic EEG

The eyes-open dataset has the SAME participants, not
an independent external patient cohort.

Review source-data redistribution requirements before
publicly distributing participant-level derived results.

All conclusions are exploratory and conditional on the
study cohorts and repeated cross-validation models.
