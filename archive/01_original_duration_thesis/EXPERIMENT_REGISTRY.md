# EXPERIMENT REGISTRY

## EXP001 - EEG recording-duration spectral feature extraction

Status:
EXECUTED / STRUCTURALLY VERIFIED / INTERPRETATION PENDING

Purpose:
Extract prespecified spectral features from equal-duration participant-level
EEG segments.

Code commit:
5701ae79233b6d8c6e83147faab027270b0ef7da

Dataset:
NEMAR on004504 v1.0.0

Manifest SHA-256:
f15e50c76c36422bc3e8b4061b0dadacaaf4c1e5df02e676eee131a5b64102e2

Durations:
30, 60, 120, 180, 300 seconds

Temporal anchors:
start, center, end

Output:
results/duration_features/eeg_duration_features.csv

Interpretation:
Pending verification.

---

## EXP002 - Spectral feature stability analysis

Status:
EXECUTED / STRUCTURALLY VERIFIED / INTERPRETATION PENDING

Purpose:
Estimate how closely shorter-duration spectral measurements reproduce the
300-second reference.

Code commit:
5701ae79233b6d8c6e83147faab027270b0ef7da

Output directory:
results/spectral_stability/

Interpretation:
Pending verification.

---

## EXP003 - Nested duration classification experiment

Status:
EXECUTED / STRUCTURALLY VERIFIED / INTERPRETATION PENDING

Purpose:
Estimate AD-vs-CN classification performance as a function of recording
duration, with AD-vs-FTD as a prespecified disease-specificity analysis.

Code commit:
5701ae79233b6d8c6e83147faab027270b0ef7da

Evaluation:
5-fold repeated nested participant-level CV, 10 repeats.

Primary metric:
Balanced accuracy.

Output directory:
results/duration_experiment/

Interpretation:
Pending verification.

---

## EXP004 - Shuffled-label negative control

Status:
EXECUTED / STRUCTURALLY VERIFIED / INTERPRETATION PENDING

Purpose:
Test whether the modeling pipeline generates strong apparent AD-vs-CN
performance when diagnostic information is destroyed by label permutation.

Condition:
300-second center segment.

Permutations:
100

Output:
results/duration_experiment/permutation_null.csv
results/duration_experiment/permutation_summary.json

Interpretation:
Pending verification.
