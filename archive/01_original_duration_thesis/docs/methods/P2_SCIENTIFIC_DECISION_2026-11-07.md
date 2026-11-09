# P2 Exploratory Scientific Decision

Date: 2026-11-07
Decision: CORRECT

## Paired performance observations

AD versus controls:

Epoch median 300 s minus original 300 s:
BA difference +0.0634, exploratory 95% CI [-0.0241, 0.1544].

Epoch median 300 s minus original 60 s:
BA difference +0.0104, CI [-0.0760, 0.0971].
Additional average recording time: 240 seconds.

Stability-only minus fixed 60 s:
BA difference +0.0150, CI [-0.0115, 0.0485].
Additional average recording time: 17.85 seconds.

AD versus FTD:

Epoch median 300 s minus original 300 s:
BA difference +0.0294, CI [-0.1091, 0.1727].

## Negative-control observations

Quality-only AD-versus-control balanced accuracy:

30 s: 0.5837
60 s: 0.5447
300 s: 0.5034

Quality-only prediction does not establish the
presence or absence of acquisition confounding.

## Scientific interpretation

All selected paired confidence intervals include zero.

Comparisons are post-hoc and exploratory.
Intervals condition on saved repeated-CV predictions.
No multiplicity correction or independent-cohort
confirmation was performed.

The current combined adaptive policy was dominated
on the observed accuracy-time comparison.

Its present form is not retained as the proposed
final contribution.

Robust epoch-based spectral estimation and
stability-only stopping remain candidates/baselines.

AD-versus-FTD discrimination remains weak.

## Next methodology priority

Investigate artifact-aware prefix-valid processing
without future-information leakage.

Compare against simple fixed-duration and
adaptive strategies under matching time budgets.

Do not freeze a confirmatory protocol or claim
novelty until design and independent validation
requirements have been resolved.

## Visualization requirement

Preserve authentic result figures, their actual
numerical inputs, scripts, hashes, captions,
statistical limitations and Git provenance.

Do not modify experimental results for visualization.
