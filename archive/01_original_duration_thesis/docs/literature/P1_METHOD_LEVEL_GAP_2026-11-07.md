# P1 Methods-Level Research Gap Investigation

Date: 2026-11-07
Status: PROVISIONAL — NOT A FROZEN CONFIRMATORY PROTOCOL

## Objective

Identify a publication-oriented methodological contribution
for EEG-based Alzheimer's disease research.

Existing duration analyses from 2026-11-06 are preserved as
exploratory pilot evidence.

## Closest Existing Work

### Tzimourta et al. (2019)
DOI: https://doi.org/10.3390/brainsci9040081

Examined EEG epochs of 5–12 seconds using spectral/statistical
features and Random Forest classification.

Implication:
Epoch length is not a new research topic.

### Cejnek et al. (2021)
DOI: https://doi.org/10.1007/s11517-021-02427-6

Used novelty measures obtained from adaptive EEG filtering
to support AD/MCI classification.

Implication:
Adaptive signal filtering is already established.
It must not be confused with an adaptive decision about
how long to acquire EEG.

### Sequential dementia assessment (2023)
DOI: https://doi.org/10.1038/s41598-023-32867-z

Investigated adaptive selection of biological and cognitive
markers for dementia assessment.

Implication:
Sequential assessment itself cannot be claimed as new.

### Ahmadi et al. (2024)
DOI: https://doi.org/10.3389/fnhum.2024.1437965

Investigated Bayesian dynamic stopping in EEG-based
brain-computer interfacing.

Implication:
Dynamic stopping is already known in EEG research.

### Tawhid et al. (2025)
DOI: https://doi.org/10.1186/s40708-025-00260-3

Investigated CNN-based AD classification across EEG
segment lengths, including ablation analyses.

Implication:
A CNN or optimal segment length is insufficient novelty.

### AHEPA benchmark review (2026)
DOI: https://doi.org/10.1007/s11571-026-10464-w

Documented the substantial effect of validation design
on reported dementia-classification performance.

Implication:
Participant-level evaluation is compulsory, not novel.

## Provisional Research Gap

The screened literature does not yet establish an adequately
validated combination of:

1. Prefix-valid EEG processing using only data available
   by the current recording time.

2. A participant-adaptive acquisition decision using signal
   quality, feature stability and predictive uncertainty.

3. A direct comparison against fixed recording durations
   and simpler dynamic stopping methods.

4. Independent-patient assessment of recording-time savings,
   false-negative risk and calibration.

This is a candidate gap, not proof that no prior paper
has addressed the same combined problem.

A broader methods-level novelty search is still required.

## Proposed Working Research Question

Can a prefix-valid, quality- and uncertainty-aware
acquisition policy reduce resting-state EEG recording
time while preserving clinically relevant classification
reliability compared with matched fixed-duration policies?

## Hypotheses To Test

H1: The proposed acquisition policy reduces participant-level
recording duration relative to a predefined fixed-duration
baseline at an acceptable error-risk constraint.

H2: Prefix-only processing changes short-duration biomarker
estimates and/or prediction reliability compared with
full-recording preprocessing followed by cropping.

H3: Quality and stability information adds measurable value
beyond confidence-only or fixed-duration stopping.

H4: Performance remains interpretable under independent-cohort
testing, allowing for acquisition-domain shift.

These hypotheses are not confirmed.

## Required Baselines

- Fixed 30, 60, 120, 180 and 300-second acquisition
- Confidence-only stopping
- Spectral-stability-only stopping
- Quality-only stopping
- Full proposed stopping policy
- Full-recording-preprocessed crop comparison as an
  explicitly offline methodological reference

## Required Ablations

- Remove signal-quality information
- Remove spectral-stability information
- Remove predictive uncertainty
- Replace prefix-valid preprocessing with the offline
  reference, clearly identifying the information difference

## Evaluation Requirements

- Subject-independent train/test partitions
- Training-only feature scaling and tuning
- Stopping thresholds selected without test-label access
- Clearly defined patient-level prediction
- Recording-time distribution and average duration
- Balanced accuracy, sensitivity, specificity and AUC
- False-negative and abstention rates
- Calibration and probability uncertainty
- Paired effect sizes and participant-level confidence intervals
- Statistical precision and multiplicity analysis
- Independent cohort where scientifically feasible
- Site, age, sex and acquisition-condition sensitivity
- No diagnostic deployment claim from retrospective testing

Repeated epochs from the same person are never treated
as independent participants.

## External Cohort Priority

1. BrainLat:
   Published eyes-closed EEG, approximately 10 minutes,
   35 AD, 42 controls and 19 bvFTD participants.
   Synapse access and montage harmonization pending.

   Dataset paper:
   https://doi.org/10.1038/s41597-023-02806-8

2. Cejnek / Figshare:
   Publicly reported AD/MCI/control dataset.
   Need to verify accessible subject files, true counts,
   recording durations and acquisition-rate confounding.

   https://doi.org/10.6084/m9.figshare.5450293

3. ADFSU OSF:
   Short published segments may make it unsuitable for
   30–300-second recording-time evaluation.

4. OpenNeuro ds006036:
   Same participants as ds004504.
   May support condition robustness, not independent-patient
   external validation.

## Statistical Feasibility Risk

The development cohort has only 36 AD and 29 controls.

Strong non-inferiority or small-effect claims may be
underpowered even if the proposed policy performs well.

A precision and power plan must precede final method testing.

## Current Decision

P1 DIRECTION: PROVISIONAL GO

P2 MODEL DEVELOPMENT: NOT YET APPROVED

NOVELTY: NOT YET ESTABLISHED

EXTERNAL VALIDATION: NOT YET APPROVED

NEXT GATE:
Review external cohort access, finalize gap assessment,
and perform a statistical feasibility analysis before
freezing the P2 protocol.
