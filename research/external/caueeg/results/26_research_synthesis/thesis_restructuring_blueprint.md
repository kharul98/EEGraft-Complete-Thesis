# EEGraft: Final Research Positioning Blueprint

## Working title

EEGraft: Gated Dual-Scale Residual Fusion and Threshold-Aware Reliability Assessment Across Alzheimer's and Dementia EEG Cohorts

## Central research problem

A model can appear accurate or temporally stable while
its decisions remain sensitive to threshold choice,
recording conditions, demographic composition, or
cross-cohort clinical task differences.

This thesis develops a gated dual-scale residual fusion
decoder and evaluates these risks rather than claiming
that a high classification score alone proves reliability.

## Research contributions

1. Implemented EEGraft decoder and development ablations.
2. Internal source-cohort evaluation with a locked holdout.
3. Frozen transfer to the CAUEEG dementia benchmark.
4. Correctness-aware, threshold-aware temporal evaluation.
5. Explicit age, label, and recording-condition audits.

These are case-study contributions, not a proof of
global first-in-kind novelty.

## Essential final evidence

### External locked test

Binary test: 57 recordings (35 Normal, 22 Dementia).

DSRF original BAcc: 68.38%
XGBoost original BAcc: 58.57%
XGBoost training-threshold BAcc:
66.10%

DSRF minus original XGBoost paired 95% interval:
[-3.31 pp, +22.08 pp].

The interval crosses zero. Do not claim established
between-model superiority.

### Threshold-sensitive stability

Training: 404 paired condition recordings.

DSRF threshold 0.50 flip rate: 35.15%
XGBoost threshold 0.50 flip rate:
7.18%
XGBoost training threshold 0.70 flip rate:
33.17%

Validation: 42 paired condition recordings.

XGBoost original flip rate:
11.90%
XGBoost adapted flip rate:
35.71%

These are exploratory condition-pair results.
They do not establish a causal eye-state effect.

### Age confounding

Age-only validation AUROC: 0.8397

Adding EEGraft DSRF to linear age:

Training OOF AUROC improvement:
+0.0457

Training conditional recording-bootstrap interval:
[0.025086722913452153, 0.06774810413315381]

Validation AUROC improvement:
+0.0118

Validation conditional recording-bootstrap interval:
[-0.08602941176470591, 0.10810661764705862]

The validation interval includes zero. There is
no independently established incremental EEG gain.

## Proposed revised chapter hierarchy

### Chapter 1: Introduction

Reframe the problem around trustworthy EEG
classification under time and cohort shift.
State the limited research questions.
Do not imply that accuracy equals clinical reliability.

### Chapter 2: Literature Review

Organize the review by leakage, pretrained encoders,
cross-dataset generalization, temporal consistency,
decision-threshold effects, and demographic confounding.

Verify citations and novelty wording before submission.

### Chapter 3: Requirements and Constraints

Retain the original engineering context.
Add reproducibility and restricted-data safeguards.

### Chapter 4: Data and Experimental Protocols

Clearly separate:
Study A: original temporal SVM experiments.
Study B: source AD/CN decoder development and
13-person held-out evaluation.
Study C: CAUEEG recording-level transfer experiment.

Never pool these cohorts into one accuracy estimate.

### Chapter 5: Methodology

Keep the original DSRF architecture and ablations.
Explain pretrained encoder freezing, score aggregation,
decision thresholds, calibration, and error-aware
stability measurements.

State which analyses were planned before the locked
test and which were introduced afterward.

### Chapter 6: Classification Results

Lead with Study B development and source holdout.
Report favorable and unfavorable source results.
Then present original frozen CAUEEG test models,
their uncertainty, and exploratory adaptations.

### Chapter 7: Reliability and Confounding

Make threshold-dependent label stability the main
new observation.

Present stable-but-wrong counts next to accuracy.
Then explain event alignment, recording condition,
time ordering, age imbalance, and heterogeneous labels.

Distinguish training, validation, and already-used
locked-test evidence.

### Chapter 8: Discussion and Limitations

Discuss why an apparently stable model can mislead.
Acknowledge source-versus-target label mismatch,
absence of verified patient IDs, small validation cohort,
age confounding, post-hoc analysis, and uncertainty.

### Chapter 9: Conclusion

Conclude that EEGraft is a viable evaluated decoder
whose broader reliability cannot be summarized by
one accuracy or stability metric.

Do not claim clinical diagnostic readiness.

## Figure policy

Preserve authentic original architectural and EEG
figures where still relevant.

Update the research overview to show all three studies.

Promote a threshold-sensitivity comparison as a
main research figure.

Promote the external frozen model comparison and
a clear age-confounding summary.

Move redundant ablations, synthetic stress figures,
and exploratory plots to appendices where appropriate.

Keep the full provenance and figure numbering
consistent with the final manuscript.

## Final writing constraints

- No further optimization using locked-test results.
- No invented statistical significance.
- No claim of a proven first-in-world discovery.
- No unsupported clinical impact or risk prediction.
- No claim that CAUEEG Dementia equals confirmed AD.
- No recording-level results described as patient-level.
- No artificial or synthetic EEG presented as raw evidence.
- No restricted participant-level records in public exports.
- Preserve all unfavorable results.
