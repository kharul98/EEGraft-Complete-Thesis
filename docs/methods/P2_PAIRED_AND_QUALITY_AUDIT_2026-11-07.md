# P2 Paired Uncertainty and EEG Quality Audit

Date: 2026-11-07
Status: Exploratory, post-hoc diagnostics

## Scientific purpose

Assess the uncertainty of selected pilot performance
differences and examine whether raw EEG quality
descriptors contain diagnostic-group information.

## Paired evaluation

The same participant IDs are compared between methods.

Each subject's three repeated-CV predictions remain
clustered during stratified participant resampling.

Bootstrap intervals are conditional on the fitted
models and saved out-of-fold predictions.

The comparisons were selected after inspecting
earlier results.

They are not confirmatory hypothesis tests.

## Quality negative control

Inputs:
- Proportion of raw samples above 200 microvolts
- Median epoch RMS amplitude
- Median epoch peak-to-peak amplitude

All inputs are computed from the recording prefix
without using diagnostic labels.

The classifier uses participant-level outer CV,
training-only scaling and inner-CV model tuning.

Quality prediction alone cannot establish a causal
source of confounding.

## Pending decisions

1. Whether epoch-median spectral features provide a
   sufficiently reliable improvement to retain.

2. Whether amplitude and recording-quality information
   creates a serious acquisition-confounding risk.

3. Whether the adaptive-stopping hypothesis should
   proceed, be redesigned, or be rejected.

4. Which hypotheses and policies can be evaluated
   without reusing exploratory test outcomes.

## Research restrictions

No clinical inference.
No confirmatory significance claim.
No non-inferiority conclusion.
No external validation claim.
No final novelty claim.
