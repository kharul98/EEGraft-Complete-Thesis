# PREREGISTRATION
## Resting-State EEG Duration Robustness in Alzheimer's Disease

Protocol frozen: 2026-11-06
Status: LOCKED BEFORE PRIMARY CLASSIFICATION RESULTS

## Working title

How Much Resting-State EEG Is Enough?
Recording-Duration Sensitivity of Alzheimer Spectral Biomarkers and
Subject-Level Classification

## Primary research question

How does the amount of available resting-state eyes-closed EEG affect
the stability of spectral biomarkers and subject-level discrimination
between Alzheimer's disease and cognitively normal controls?

## Primary estimation objective

Estimate classification performance at:

- 30 seconds
- 60 seconds
- 120 seconds
- 180 seconds
- 300 seconds

using exactly the same participants at every duration.

300 seconds is the longest common duration available to all 88
participants in the verified preprocessed derivative dataset.

## Primary population

Alzheimer's disease (A): 36 participants
Cognitively normal controls (C): 29 participants

Total primary analysis N = 65.

## Secondary disease-specificity population

Alzheimer's disease (A): 36
Frontotemporal dementia (F): 23

Total secondary analysis N = 59.

Purpose:

Determine whether apparent Alzheimer-related EEG patterns distinguish
AD specifically from another dementia syndrome or merely distinguish
dementia from healthy aging.

## Dataset

NEMAR on004504 version v1.0.0.

NEMAR DOI:
10.82901/nemar.on004504

Verified local manifest SHA-256:

f15e50c76c36422bc3e8b4061b0dadacaaf4c1e5df02e676eee131a5b64102e2

Verified structure:

- 88 unique participants
- 36 AD
- 23 FTD
- 29 CN
- 19 EEG channels
- one consistent channel layout
- 500 Hz sampling
- 88/88 readable preprocessed derivative recordings
- minimum derivative duration = 306.1 seconds
- median duration = 804.52 seconds
- maximum duration = 1281.72 seconds

## Preprocessing provenance

The study uses the dataset-provided denoised derivative recordings.

According to dataset documentation, preprocessing includes:

- 0.5-45 Hz Butterworth band-pass filtering
- A1-A2 rereferencing
- Artifact Subspace Reconstruction
- RunICA
- ICLabel-guided rejection of eye and jaw artifact components

We do not claim that this preprocessing was developed or executed by
our research team.

## Independent observational unit

One participant.

No EEG epoch, channel or time window is treated as an independent
patient.

## Duration construction

Each participant is evaluated at exactly:

30, 60, 120, 180 and 300 seconds.

For each duration, three temporal placements are evaluated:

1. START
2. CENTER
3. END

Each placement contains exactly the requested amount of EEG.

This tests whether duration findings depend on where in the recording
the segment happens to be selected.

## Spectral estimation

Power spectral density is estimated using Welch's method.

Sampling rate:
500 Hz

Welch segment:
4 seconds

Overlap:
50 percent

Analysis frequency range:
1-45 Hz

Frequency bands:

- delta: 1-4 Hz
- theta: 4-8 Hz
- alpha: 8-13 Hz
- beta: 13-30 Hz
- gamma: 30-45 Hz

Relative power is normalized by total 1-45 Hz power.

## Spatial aggregation

Five prespecified regions:

Frontal:
Fp1, Fp2, F7, F3, Fz, F4, F8

Central:
C3, Cz, C4

Temporal:
T3, T4, T5, T6

Parietal:
P3, Pz, P4

Occipital:
O1, O2

Features include global and regional relative band power.

Prespecified global summary markers:

- delta relative power
- theta relative power
- alpha relative power
- beta relative power
- gamma relative power
- slowing ratio:
  (delta + theta) / (alpha + beta)
- alpha/theta ratio
- peak alpha frequency in 7-13 Hz

## Primary model

L2-regularized logistic regression.

Reasons:

- interpretable baseline
- appropriate for small participant count
- duration, not architecture, is the primary experimental variable
- avoids introducing unnecessary model capacity

Class weighting:
balanced

## Hyperparameter selection

C values:

0.01, 0.1, 1, 10, 100

C is selected only inside the training data of each outer fold.

## Evaluation

Repeated stratified nested cross-validation.

Outer:
5 folds x 10 repeats

Inner:
4-fold stratified CV

All durations and temporal placements use the same participant-level
outer split structure for each diagnostic comparison.

No participant appears simultaneously in training and test within a
fold.

## Primary endpoint

Balanced accuracy for AD versus cognitively normal controls.

## Secondary classification outcomes

- ROC-AUC
- sensitivity
- specificity
- F1
- Matthews correlation coefficient
- Brier score

## Primary duration comparison

Each shorter duration is compared against the 300-second condition at
the same temporal placement.

For repeated CV, each participant receives out-of-fold probabilities
during each repeat.

Probabilities are averaged across repeats to produce one final
out-of-fold probability per participant and condition.

## Uncertainty

Paired stratified participant-level bootstrap.

5000 bootstrap samples.

The bootstrap resamples participants separately within each diagnostic
class and preserves the pairing of predictions between duration
conditions.

Report:

- estimated balanced-accuracy difference
- 95 percent bootstrap confidence interval

## Spectral stability analysis

For every spectral feature and temporal placement, estimates at shorter
durations are compared with the corresponding 300-second estimate.

Report:

- Spearman correlation across participants
- mean absolute feature difference

No claim that correlation alone establishes absolute agreement.

## Biological-pattern analysis

Prespecified global biomarkers are compared between:

1. AD and CN
2. AD and FTD

using:

- Mann-Whitney U
- Cliff's delta effect size

The emphasis is on effect magnitude and duration stability rather than
significance hunting.

## Demographic baseline

Age and sex alone are evaluated using the same participant-level nested
cross-validation framework.

MMSE is excluded from predictive modeling.

Reason:

MMSE is directly associated with clinical cognitive status and is
perfectly 30 in every control participant in this dataset. Including it
would undermine an EEG-specific interpretation.

## Negative control

For the primary AD-versus-CN task at the 300-second CENTER condition,
diagnostic labels are randomly permuted 100 times.

The complete training and evaluation pipeline is rerun for each
permutation using subject-level CV.

This tests whether the pipeline itself can manufacture strong apparent
performance in the absence of label information.

## Interpretation rules

A higher classification score is not sufficient to establish clinical
diagnostic validity.

Possible conclusions include:

1. Short EEG durations preserve both biomarker estimates and predictive
   performance.

2. Biomarkers stabilize before classification performance stabilizes.

3. Classification performance stabilizes despite unstable individual
   biomarkers.

4. Results are highly dependent on temporal segment placement.

5. AD-vs-CN effects do not distinguish AD from FTD, suggesting a more
   general dementia-related signal.

6. No stable duration effect exists.

All are acceptable outcomes.

## Prohibited claims

We will not claim:

- prospective clinical diagnostic validation
- causal biomarkers
- population-wide generalizability
- disease specificity unless AD-vs-FTD evidence supports it
- novelty as "first-ever" without broader literature evidence
- independence of EEG windows as if they were patients

## Completion criteria

The core experiment is complete when:

- all 88 recordings have validated feature extraction
- all primary participants occur at every duration
- participant-level predictions are saved
- nested CV completes
- bootstrap intervals complete
- feature-stability analyses complete
- negative control completes
- results are hashed
- conclusions are based only on verified output artifacts
