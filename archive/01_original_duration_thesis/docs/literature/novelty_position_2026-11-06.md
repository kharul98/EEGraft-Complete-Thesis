# Novelty Position - 2026-11-06

## Important prior work

### Tzimourta et al. 2019

Title:
EEG Window Length Evaluation for the Detection of Alzheimer's Disease
over Different Brain Regions

Journal:
Brain Sciences, 9(4), 81

DOI:
10.3390/brainsci9040081

Design:
- 14 Alzheimer's disease participants
- 10 cognitively normal participants
- non-overlapping EEG windows from 5 to 12 seconds
- statistical and spectral features
- Random Forest classification
- whole-brain and regional analyses

The reported classification accuracy increased as window length increased,
with the 12-second window giving the best result among the tested windows.

Consequence for our thesis:

We MUST NOT claim:
- first study of EEG window length in Alzheimer's disease
- first study showing that EEG duration affects classification
- novelty based solely on changing EEG segment duration

## How our study differs

Our study investigates a different methodological question:

How much TOTAL AVAILABLE resting-state EEG is required for stable
participant-level spectral measurement and discrimination?

Key distinctions:

1. Duration range:
   - Prior study: 5-12 second epochs
   - Our study: 30-300 seconds of available EEG

2. Cohort:
   - Prior study: 24 participants
   - Our primary analysis: 65 participants (36 AD, 29 CN)
   - Disease-specificity analysis: 59 participants (36 AD, 23 FTD)

3. Participant consistency:
   - Every duration from 30 to 300 seconds uses exactly the same participants.

4. Temporal placement:
   - Start, center, and end recording segments are evaluated separately.

5. Evaluation:
   - Subject-level repeated nested cross-validation
   - paired participant-level bootstrap
   - shuffled-label negative control

6. Measurement question:
   - We explicitly quantify spectral-feature stability relative to 300 seconds,
     rather than studying classification accuracy alone.

7. Disease specificity:
   - AD-vs-FTD is evaluated to test whether apparent AD-vs-control signals are
     specific to Alzheimer's disease or reflect broader dementia-associated EEG
     alterations.

## Supporting methodological literature

Babiloni et al. 2021
Measures of resting state EEG rhythms for clinical trials in Alzheimer's
disease: Recommendations of an expert panel.
Alzheimer's & Dementia 17(9):1528-1553.
DOI: 10.1002/alz.12311

Relevant point:
Standardization of resting-state EEG acquisition and selection of artifact-free
periods is important. Relative spectral power can show useful repeatability
across recordings ranging from tens of seconds to several minutes.

Kamal, Campbell, and Taler
Effects of the Duration of a Resting-State EEG Recording in Healthy Aging and
Mild Cognitive Impairment.
Clinical EEG and Neuroscience.
DOI: 10.1177/1550059420983624

Relevant point:
Recording duration and temporal location can matter because longer recordings
could theoretically introduce vigilance/drowsiness changes.

## Current defensible contribution statement

A leakage-controlled recording-duration robustness audit of Alzheimer's
resting-state EEG that jointly evaluates:

- spectral biomarker stability,
- temporal segment sensitivity,
- subject-level classification,
- uncertainty,
- shuffled-label behavior,
- and disease specificity against FTD.

## Claim status

"First-ever":
NOT ESTABLISHED.

"Novel classifier":
NOT CLAIMED.

"Methodological robustness contribution":
SUPPORTED AS THE CURRENT POSITION, subject to final literature review.
