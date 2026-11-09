# Original Alzheimer and FTD Results: Recomputed Evidence

This table was independently calculated from saved held-out predictions, without fitting new models.
The unit of analysis is a participant within a diagnostic task. Five CV repeats are correlated.

| Measure | AD vs healthy | FTD vs healthy |
|---|---:|---:|
| Participants | 46 | 41 |
| Original 60-second balanced accuracy | 73.48% | 73.72% |
| Calibrated-threshold balanced accuracy | 74.35% | 69.32% |
| Calibrated Brier score | 0.1861 | 0.1917 |
| Training-prior Brier baseline | 0.2513 | 0.2479 |
| Stable and correct across five minutes | 46.96% | 53.17% |
| Stable but wrong across five minutes | 8.70% | 13.17% |
| Changed prediction across five minutes | 44.35% | 33.66% |

The first-minute balanced accuracy is the mean of five participant-disjoint cross-validation repeats.
Stability rates use 230 AD/healthy and 205 FTD/healthy correlated subject-repeat assessments, not 435 independent individuals.

## Interpretation boundaries

- A stable prediction can still be wrong; temporal consistency is not clinical validity.
- Calibration supplies probabilities but does not guarantee improved classification.
- The original task has no completed independent external AD-versus-healthy validation.
- The MCI extension is a different disease comparison and is reported separately.
- Matched acquisition-condition effects and frequency-band ablations were not statistically confirmed.
- No longitudinal conversion or clinical deployment performance has been established.

## Additional source agreement

- AD_vs_HC: 0 mismatches across 230 older 60-second predictions.
- FTD_vs_HC: 0 mismatches across 205 older 60-second predictions.
- FTD minute summary comparison: MATCH
