# Publication evidence freeze

This is an evidence summary, not a finished thesis
manuscript or a preregistered study.

The same 46 participants were examined repeatedly
during exploratory method development.

## Verified study scope

- OpenNeuro ds004504: 88 total participants.
- Primary cohort: 46 (23 AD, 23 healthy).
- FTD is excluded from the binary endpoint.
- Of 19 excluded AD/healthy participants, 16 lacked
  a selected uninterrupted window and 3 have REVIEW
  status requiring final objective justification.
- Selected-window continuity audit reported no errors.
- Historical 300-second spectral reproduction:
  maximum absolute difference 1.77e-10.
- The outcome is classification of retrospective
  dataset diagnostic labels, not early detection.

## Reproducible findings

- First-minute trained, mean all-block BACC:
  0.7174.
- Position-matched trained, mean all-block BACC:
  0.7148.
- Five-block trained, mean all-block BACC:
  0.7209.
- Overall augmented minus original:
  +0.0035, conditional 95% interval
  [-0.0348,
   +0.0392].
- No established overall advantage of augmentation.
- Test minute five BACC: original 0.6435;
  position-matched 0.7000; augmented 0.7043.
- The augmented model is not established as superior
  to the minute-five position-matched model.
- Within-recording reversal: 102/230
  subject-repeat observations (44.3%).
  Conditional subject-bootstrap 95% interval:
  [0.3261, 0.5652].
- 27/46 participants reversed at least once
  across five correlated CV repetitions.
- Highest observed duration-model BACC:
  0.8130
  at 180 seconds,
  using RBF_SVM.
- This does not establish an optimal EEG duration.

## What is not established

- No state-of-the-art accuracy result.
- No superiority of five-block augmentation.
- No proven optimal acquisition duration.
- No demonstrated causal biomarker mechanism.
- No independent external validation.
- No guarantee of clinical applicability.
- No claim of 60-second acquisition-to-prediction;
  preprocessing may use the longer original session.
- Bootstrap uncertainty is conditional on fitted
  cross-validation results, and repeated observations
  are correlated.
- CBraMod conclusions apply to the frozen-feature
  configuration actually evaluated.

## Before journal submission

1. Document exclusion reasons for sub-027,
   sub-030 and sub-064.
2. Verify model training code and signal preprocessing
   provenance, including CBraMod input configuration.
3. Preserve participant-level splits and dependencies.
4. Verify all citations and literature comparisons.
5. Do not introduce new models after examining
   these test results.

## Evidence files

- input_fingerprints.json
- model_comparison_from_saved_oof.csv
- cohort_flow.csv
- review_cases_to_justify.csv
- duration_bacc_summary.csv
- training_strategy_comparison.csv
- paired_comparisons_existing_CIs.csv
- claim_registry.csv
- temporal_training_strategies.png
- duration_performance_sd.png
