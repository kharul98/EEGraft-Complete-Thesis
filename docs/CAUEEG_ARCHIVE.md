# CAUEEG External EEG Research Archive

## Source

Original local research directory:

/home/user4/research/eegraft-caueeg

Research conducted: November 9, 2026.

## Research workflow

The numbered execution scripts document:

1. Dataset inspection and cohort checks
2. EDF and metadata quality control
3. Frozen model smoke testing
4. Development-set inference
5. Development-set evaluation
6. Reliability analysis
7. Training-only threshold selection
8. Training-only probability calibration
9. Validation variant comparison
10. Locked final test preflight
11. Locked-test inference
12. Locked-test evaluation
13. Event alignment investigation
14. Condition analysis
15. Reliability confound analysis
16. Age and label audit
17. Incremental information audit
18. Research claims synthesis

## Directory organization

- scripts/: Numbered experiment scripts
- protocol/: Frozen experimental specifications
- results/: Saved experimental summaries
- manifests/: Original execution metadata
- logs/: Available execution logs
- configs/: Available configuration files
- src/: Supporting source files
- tests/: Available tests

## Reproducibility requirements

Verify numerical findings from the original stored
execution outputs and corresponding configurations.

The archive may contain aggregate JSON summaries
without all underlying prediction-level files.

Missing model checkpoints, predictions or raw inputs
must not be silently treated as reproduced evidence.

The raw CAUEEG archive is deliberately not uploaded.
It must be retrieved from an authorized source.

## Historical research integrity

Original development and locked-test evaluations
must remain distinct.

Thresholds and calibration parameters must not
be fitted using locked-test labels.

This archive records experimental work; copying
files does not constitute independent re-execution.
