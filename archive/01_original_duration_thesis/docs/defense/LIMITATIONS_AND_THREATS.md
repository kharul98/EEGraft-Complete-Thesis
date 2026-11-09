# Limitations and Threats to Validity

## 1. Single dataset and acquisition center

All primary results come from one public cohort collected under one clinical
acquisition setting.

Implication:
Internal participant-level validation does not establish transportability to
other hospitals, EEG devices, populations, or protocols.

Defense:
We report internal robustness and do not call the results external clinical
validation.

## 2. Modest participant count

The primary AD-vs-CN analysis contains 65 participants.

Implication:
Performance estimates remain uncertain even with repeated nested validation.

Defense:
We preserve participant-level independence, report paired bootstrap intervals,
and avoid treating EEG windows as independent patients.

## 3. Released preprocessing pipeline

We use dataset-provided denoised derivatives.

Implication:
Our conclusions inherit choices made by the original preprocessing pipeline.

Defense:
Preprocessing is explicitly treated as dataset provenance rather than our
scientific contribution.

## 4. Duration segments are nested

Short start, center, and end segments are contained within their corresponding
300-second segments.

Implication:
Duration comparisons are deliberately paired and correlated.

Defense:
This is a feature, not an independence assumption. The analysis uses paired
participant-level comparisons.

## 5. Temporal-position effects

The same duration can perform differently depending on whether the segment
comes from the start, center, or end of the recording.

Implication:
A single arbitrarily selected segment can give an incomplete impression.

Defense:
Three prespecified temporal placements were analyzed.

## 6. Disease specificity

Strong AD-vs-CN classification does not imply an Alzheimer's-specific EEG
signature.

Evidence:
AD-vs-FTD discrimination is substantially weaker.

Defense:
FTD was retained specifically as a disease-specificity comparator.

## 7. MMSE separation

Controls have MMSE 30 while AD participants have substantially lower scores.

Implication:
MMSE could trivially aid classification and obscure the EEG question.

Defense:
MMSE was excluded from predictive modeling.

## 8. No formal equivalence margin

The study uses paired bootstrap confidence intervals for differences against
300 seconds but did not preregister a clinical non-inferiority/equivalence
margin.

Implication:
A confidence interval containing zero must not be translated into proof that
two durations are equivalent.

Defense:
We describe shorter durations as showing similar observed performance only
when appropriate, not as formally equivalent.

## 9. Prior window-length literature exists

A 2019 study already evaluated 5-12 second EEG windows for Alzheimer's
classification.

Implication:
The thesis cannot claim first use of recording/window duration in AD EEG.

Defense:
Our contribution concerns longer available recording durations, measurement
stability, temporal position, participant-level inference, and disease
specificity.
