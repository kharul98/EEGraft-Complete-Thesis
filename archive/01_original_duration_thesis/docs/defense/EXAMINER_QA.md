# Examiner Q&A

## Why did you choose 300 seconds as the reference?

The minimum verified duration among all 88 denoised recordings was 306.1
seconds. Therefore 300 seconds was the longest round duration that preserved
the entire cohort without excluding participants based on recording length.

## Why not use each participant's entire recording?

That would make the amount of data differ substantially between participants
and diagnostic groups. Fixed durations isolate recording duration as the
experimental variable.

## Why 30, 60, 120, 180 and 300 seconds?

They cover short to moderate acquisition periods while remaining available for
every participant. The sequence also provides progressively larger amounts of
EEG without changing the cohort.

## Why use start, center and end segments?

A duration result could otherwise depend on one arbitrary location in the
recording. The three anchors test temporal-position sensitivity.

## Why logistic regression instead of a deep neural network?

The central experimental variable is recording duration, not model
architecture. A regularized linear classifier reduces model capacity and makes
duration effects easier to interpret in a small-N cohort.

## Why repeated nested cross-validation?

The outer folds estimate performance, while the inner folds select
regularization strength without using outer-test labels. Repetition reduces
dependence on one arbitrary fold assignment.

## Why is the participant the independent unit?

Multiple samples from one EEG recording belong to the same person and are
correlated. Treating windows as independent patients would inflate effective
sample size and can cause leakage.

## Why exclude MMSE?

MMSE is strongly tied to diagnostic cognitive status. In this dataset every
control has MMSE 30. Including it would make an EEG classification experiment
partly a cognitive-score classification experiment.

## Why include AD versus FTD?

AD versus healthy controls can identify a dementia-related signal without
showing Alzheimer's specificity. FTD provides a disease comparator that tests
whether the apparent EEG pattern distinguishes AD from another dementia.

## What does the permutation test show?

When diagnostic labels were destroyed, the pipeline produced a null balanced
accuracy distribution centered near chance. The observed 300-second center
score exceeded all 100 permuted scores. This supports that the internal
pipeline is using label-associated signal rather than generating high scores
under random labels.

## Does a bootstrap confidence interval containing zero prove equivalence?

No. The study did not preregister an equivalence or non-inferiority margin.
Such an interval means we did not detect a clear paired difference at the
available sample size. It is not proof of equivalence.

## Has EEG duration already been studied in Alzheimer's disease?

Yes. Tzimourta et al. (2019) evaluated 5-12 second EEG windows in a smaller AD
cohort. Therefore our novelty is not the generic idea that duration matters.

Our contribution focuses on 30-300 seconds of available EEG, spectral
measurement stability, temporal-position robustness, participant-level
repeated nested evaluation, paired uncertainty, permutation control, and
AD-vs-FTD disease specificity.

## Can this model diagnose Alzheimer's disease clinically?

No. The study is retrospective and internally validated on a single public
cohort. External prospective validation would be required before a clinical
diagnostic claim.
