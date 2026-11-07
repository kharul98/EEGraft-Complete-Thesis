# Milestone 5H: Isfahan classifier error audit

No models fitted. This audit uses saved out-of-fold predictions from Milestone 5G.

Each recording appears five times across correlated cross-validation repeats.

## EEG_1
- Minute-1 BACC 0.4078; MCI sensitivity 0.6727; NORMAL specificity 0.1429
- Minute-1 TP/FN/TN/FP across correlated repeats: 37/18/10/60
- Fraction predicted MCI: 0.7760
- Temporal state counts by true class are saved in a separate CSV.

## EEG_2
- Minute-1 BACC 0.5583; MCI sensitivity 0.6833; NORMAL specificity 0.4333
- Minute-1 TP/FN/TN/FP across correlated repeats: 41/19/26/34
- Fraction predicted MCI: 0.6250
- Temporal state counts by true class are saved in a separate CSV.

## Interpretation boundaries

- No new models were trained.
- BACC=0.5 is a comparison reference, not a formal permutation test.
- Existing conditional bootstrap intervals do not account for all retraining uncertainty.
- Do not reverse class labels or select new thresholds using held-out performance.
- Cross-release participant identity remains unconfirmed.
- MCI versus NORMAL is not equivalent to Alzheimer dementia diagnosis.
