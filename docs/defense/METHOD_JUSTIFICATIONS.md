# METHOD JUSTIFICATIONS

## Participant-level independence

EEG windows from one participant are correlated observations.

Therefore windows from the same participant must never be treated as
independent people during performance estimation.

All train/test partitioning must occur at the participant level.

## Why use the released preprocessed derivatives first?

The dataset authors provide denoised derivatives produced from a documented EEG
artifact-removal pipeline.

Using those derivatives allows the thesis to focus on the research question
(recording-duration robustness) rather than spending most of the deadline
reimplementing ICA/ASR artifact rejection.

This does not prevent a later preprocessing sensitivity analysis if needed.

## Why simple classifiers are acceptable

The main independent variable is recording duration.

If a highly flexible model is changed at the same time, interpretation becomes
harder.

Simple models allow the duration effect to be measured under a controlled
pipeline. More complex models may be secondary comparisons, not the scientific
contribution.

## Why the duration grid is not locked yet

Raw recording lengths are reported in the dataset documentation, but artifact
removal can change usable post-cleaning duration.

The duration grid must therefore be selected using the actual derivative files
before final modeling begins.
