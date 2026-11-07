# Cross-diagnostic temporal reliability audit

Status: exploratory, post-hoc analysis.

## Data and verification

- AD versus healthy: 23 AD + 23 controls.
- FTD versus healthy: 18 FTD + 23 controls.
- The exact same 23 controls are reused.
- Five repeated participant-level nested-CV evaluations.
- Five nonoverlapping held-out one-minute EEG blocks.
- No model was retrained for this audit.

## Main observations

AD/HC mean reversal rate:
0.4435

FTD/HC class-balanced reversal rate:
0.3411

FTD minus AD reversal-rate difference:
-0.1024

AD minute-5 minus minute-1 balanced accuracy:
-0.0913

FTD minute-5 minus minute-1 balanced accuracy:
-0.0495

## Interpretation boundaries

The finding is a within-dataset replication
of the same methodology in another
diagnostic classification contrast.

It is not external-site validation,
proof of common disease biomarkers,
prospective early diagnosis, or
evidence of state-of-the-art accuracy.

The positive-class cohorts differ in size.
Healthy controls are shared across tasks.
Bootstrap comparisons resample those
shared controls jointly and preserve
within-participant CV repetitions.

Bootstrap intervals are conditional on
the existing fitted cross-validation
predictions; they do not fully represent
new-cohort or model-retraining uncertainty.

The original EEG window-selection
algorithm was not recovered from source.
FTD eligibility relies on the previously
frozen manifest. The quality and
representativeness limitations remain.
