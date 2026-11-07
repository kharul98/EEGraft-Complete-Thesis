# Isfahan MCI Reliability: Negative-Control Audit

All analyses use previously frozen held-out predictions.
No SVM models were retrained or modified.
No label inversion or threshold optimization was performed.

Repeated cross-validation observations are correlated.

## EEG_1

- MCI=11; NORMAL=14.
- First-minute SVM BACC: 0.4078; accuracy: 0.3760.
- MCI sensitivity: 0.6727; healthy specificity: 0.1429.
- SVM stable-wrong: 0.5920; unstable: 0.0560.
- Always-MCI BACC: 0.5000; stable-wrong: 0.5600.
- Always-NORMAL BACC: 0.5000; stable-wrong: 0.4400.
- Conditional first-minute BACC 95% interval: [0.3318, 0.4841].

## EEG_2

- MCI=12; NORMAL=12.
- First-minute SVM BACC: 0.5583; accuracy: 0.5583.
- MCI sensitivity: 0.6833; healthy specificity: 0.4333.
- SVM stable-wrong: 0.3583; unstable: 0.2000.
- Always-MCI BACC: 0.5000; stable-wrong: 0.5000.
- Always-NORMAL BACC: 0.5000; stable-wrong: 0.5000.
- Conditional first-minute BACC 95% interval: [0.4333, 0.6750].

## Interpretation boundaries

- Constant-label classifiers are negative controls, not clinically useful models.
- Temporal stability alone does not establish diagnostic validity.
- No statistical significance or superiority claim is established.
- Reported results are conditional on five-minute clean EEG eligibility.
- EEG physical units were missing; microvolts were explicitly assumed.
- Cross-release participant identity is not fully established.
- EEG_1 and EEG_2 were evaluated separately.
- MCI detection is different from Alzheimer dementia diagnosis.
- No additional classification model was fitted or tuned.
