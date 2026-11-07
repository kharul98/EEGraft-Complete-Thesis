# Post-hoc Analysis Boundary

Date: 2026-11-06

## Purpose

The primary recording-duration experiment had already been frozen, executed,
verified, hashed, and committed before the analyses documented below were
defined.

These analyses are therefore explicitly classified as:

- robustness analyses,
- sensitivity analyses,
- or exploratory analyses.

They do not replace the preregistered primary endpoint.

## Primary results that remain unchanged

- participant cohort
- diagnostic labels
- 30/60/120/180/300-second conditions
- start/center/end temporal anchors
- spectral feature extraction
- primary L2 logistic regression
- nested participant-level evaluation
- balanced accuracy primary endpoint
- paired bootstrap against 300 seconds
- shuffled-label control
- AD-vs-FTD disease-specificity analysis

## Added analyses

### 1. Absolute bootstrap confidence intervals

Reason:
The preregistered bootstrap estimates paired differences versus 300 seconds.
Absolute uncertainty around each condition is useful for manuscript
interpretation.

Classification:
Robustness analysis.

### 2. Temporal-anchor prediction variability

Reason:
Group-level accuracy differences do not show how much an individual
participant's predicted probability changes between start, center and end
segments.

Classification:
Exploratory robustness analysis.

### 3. Regional spectral effect patterns

Reason:
The primary biomarker analysis focused on prespecified global markers.
Regional relative-power effects can help identify whether the observed
spectral pattern is spatially diffuse or concentrated.

Classification:
Exploratory analysis.

Multiple testing:
Benjamini-Hochberg correction is applied within each diagnostic comparison
and duration.

### 4. Age/sex sensitivity

Reason:
Age and sex could contribute to diagnostic discrimination.

A model combining EEG with age and sex is compared descriptively with the
already completed EEG-only and demographic-only analyses.

Classification:
Sensitivity analysis.

## Interpretation rule

No post-hoc result may be described as preregistered.

Post-hoc analyses may strengthen, qualify, or weaken the primary
interpretation, but they may not silently redefine the primary endpoint.
