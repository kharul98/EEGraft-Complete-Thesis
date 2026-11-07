# Thesis Evidence Consolidation

## Working research title

Evaluating the Reliability of EEG-Based Alzheimer's Disease and Frontotemporal Dementia Classification Through Temporal Stability and Calibration

## Primary research

The main study examines whether EEG-based dementia classifiers remain accurate and consistent across recording time, rather than evaluating classification accuracy alone.

The original cohort contains 23 Alzheimer's disease (AD), 18 frontotemporal dementia (FTD), and 23 healthy-control participants.

### Previously audited primary findings

| Result | AD vs healthy | FTD vs healthy |
|---|---:|---:|
| First-minute balanced accuracy | 0.7348 | 0.7372 |
| Unstable five-minute predictions | 44.35% | 33.66% |
| Stable-but-wrong predictions | 8.70% | 13.17% |

These primary numbers were reported in the earlier Milestone 4C audit. Their exact machine-readable source files still need to be reconciled before final chapter sign-off.

### Additional primary findings

The matched acquisition-condition analysis reported eyes-open minus eyes-closed balanced accuracy differences of -0.0883 for AD and -0.0423 for FTD. Both confidence intervals included zero, so a statistically confirmed condition effect was not established.

Eight feature-ablation comparisons were checked; none remained significant after Holm correction.

## Independent Isfahan MCI extension

This is a separate MCI-versus-healthy task, not direct external validation of AD diagnosis.

Of 61 recordings, 49 passed the fixed five-minute signal-quality screen (23 MCI and 26 healthy).

| Measure | EEG_1 | EEG_2 |
|---|---:|---:|
| First-minute balanced accuracy | 0.4078 | 0.5583 |
| Stable-but-wrong | 0.5920 | 0.3583 |
| Unstable | 0.0560 | 0.2000 |
| Always-MCI baseline BACC | 0.5000 | 0.5000 |

EEG_1 was strongly biased toward predicting MCI. Constant-label negative controls showed that stable incorrect predictions can occur even without useful diagnostic discrimination.

The two Isfahan releases were assessed separately. Cross-release participant identity remains unconfirmed. EDF physical-unit fields were blank, and microvolts were assumed.

## Figshare independent AD data feasibility

The archive contains 471 MATLAB files, including 305 AD, 56 MCI, and 110 control files.

There were 44 AD recording-folder proxies with a standard 19-column file of nominal duration at least 60 seconds, but only one comparable healthy-control proxy.

Most other controls used a different 22-row format with an unverified common electrode mapping. Thus the original 19-channel, 60-second external AD validation was not performed.

This is a dataset compatibility limitation, not a negative classification result.

## Current scientific conclusion

EEG-based dementia predictions can be reasonably accurate at one time point yet change over time or remain consistently incorrect. Temporal consistency, diagnostic accuracy, and probability reliability are different properties and should be evaluated separately.

The available findings do not establish clinical diagnostic safety or broad cross-dataset generalization.

## Remaining thesis tasks

1. Reconcile original AD/FTD result numbers with their saved source tables.
2. Prepare final reproducible tables and figures.
3. Finalize Methods and Results chapters.
4. Write Discussion, limitations, and conclusion.
5. Complete supervisor review and submission.

