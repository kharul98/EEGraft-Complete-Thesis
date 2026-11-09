# Evidence-backed results and source locations

The source-of-truth records below are preserved in the imported backup subtree after the builder is run.

## Early temporal SVM study

| Measure | AD versus CN | FTD versus CN |
|---|---:|---:|
| First-minute balanced accuracy | 73.48% | 73.72% |
| Five-minute prediction changed | 44.35% | 33.66% |
| Stable but incorrect prediction | 8.70% | 13.17% |
| Eyes-open minus eyes-closed balanced accuracy | -8.83 percentage points | -4.23 percentage points |

Source summary: [`archive/02-research-backup/PROJECT_HISTORY.md`](../archive/02-research-backup/PROJECT_HISTORY.md) and historical milestone reports within the archive. The original primary percentages require further source-row reconciliation before a fully independent replication claim; the recording-condition intervals include zero.

## Later model comparison (same 52 development people)

| Model | Pooled development balanced accuracy | Correct |
|---|---:|---:|
| Custom DSRF | 77.06% | 40/52 |
| XGBoost | 76.61% | 40/52 |
| EEGNet-inspired | 74.89% | 39/52 |
| Random forest | 72.71% | 38/52 |
| High-only | 72.26% | 38/52 |
| SVM | 71.81% | 38/52 |
| No-skip | 71.44% | 37/52 |
| Low-only | 62.74% | 33/52 |

Additional matched structure comparison: residual 77.06% versus gated blend 73.16% pooled development balanced accuracy; the observed advantage was not statistically conclusive.

Source: [`thesis_results_ready.md`](../archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/thesis_execution_package_2026_v1/thesis_results_ready.md).

## Five-seed synthetic robustness (same 52 people)

| Condition | DSRF mean balanced accuracy |
|---|---:|
| Clean | 77.06% |
| 10% synthetic Gaussian noise | 71.87% |
| 20% synthetic Gaussian noise | 70.07% |
| One electrode channel zeroed | 76.78% |

Artificial stress conditions do not prove robustness across real devices or hospitals.

## One-time held-out assessment (13 subjects)

| Model | Balanced accuracy | Accuracy | Macro-F1 | AUROC | Correct |
|---|---:|---:|---:|---:|---:|
| DSRF | 70.24% | 69.23% | 69.05% | 0.857 | 9/13 |
| XGBoost | **77.38%** | **76.92%** | **76.92%** | 0.905 | 10/13 |

DSRF minus XGBoost: -7.14 percentage points balanced accuracy, exploratory paired 95% CI -28.57 to +14.29 pp and exact McNemar p=1.0. No superiority conclusion follows.

Machine-readable source: [`final_metrics.json`](../archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/locked_final_test_2026_v1/final_metrics.json); final fixed configuration: [`final_protocol_2026.json`](../archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/final_protocol_2026.json).

**Evidence hierarchy:** locked evaluation > development comparisons > exploratory perturbation tests > feasibility audits. Cohorts and outcome protocols differ; figures and tables must not be interpreted as one pooled experiment.
