
## 2026-11-07 AUTHENTIC SCIENTIFIC FIGURE POLICY

Source: Additional user-supplied thesis instructions.

Every scientific figure must be based on real recordings,
executed analyses, saved model outputs, or calculated statistics.

No invented EEG, spectrograms, predictions, improvements,
confidence intervals, participant cases, or significance.

Conceptual diagrams must describe implemented methods.
Exploratory results must remain explicitly exploratory.

### P0 - Essential

1. Real raw EEG waveform with correct units and provenance.
2. Actual duration-window construction.
3. Real PSD and implemented spectral-feature example.
4. EXP002 spectral stability across durations.
5. EXP003 classification performance across durations.
6. Paired effects with complete confidence intervals.
7. EXP004 permutation negative control.
8. Implemented processing and evaluation pipeline.

### P1 - Important supplementary evidence

1. Temporal-anchor sensitivity.
2. AD-versus-FTD performance.
3. Demographic and quality-only negative controls.
4. Prediction errors and calibration where justified.
5. Adaptive stopping versus fixed recording duration.
6. Reproducible EEG analysis demonstration.

### Requirements for every figure

- Unique figure ID and scientific question.
- Source experiment and actual input paths.
- Dataset and participant-selection rule.
- Exact script and source-code commit.
- Figure output path and SHA-256.
- Correct units, axes, legends and captions.
- Numerical source values or reproducible extraction.
- Interpretation and explicit limitations.
- Status: PLANNED, GENERATED, VERIFIED or INCLUDED.

Generate figures from existing evidence whenever possible.
Do not rerun model experiments just to make figures attractive.
Do not silently overwrite historical figures.
Do not treat repeated folds or overlapping epochs as
independent participants.

Use PNG at 300 DPI and vector SVG where appropriate.

A figure is VERIFIED only after numerical,
provenance and visual inspection.
