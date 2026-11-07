# Milestone 5G: Separate-release MCI vs NORMAL reliability

The two Isfahan archive releases were trained and evaluated separately.
No data from one release entered training or tuning for the other.
Each recording was held out once per outer repeat.
All five minute predictions came from a model trained only on first-minute features of other recordings.
Intervals are conditional, recording-clustered bootstrap summaries of fixed CV predictions.
They do not account for retraining variation in a new population.

## EEG_1 (n=25)
- Minute-1 BACC: 0.4078 [0.3318, 0.4841]
- Minute-5 minus minute-1 BACC: +0.0182 [-0.0123, +0.0506]
- Unstable rate: 0.0560; stable-wrong rate: 0.5920

## EEG_2 (n=24)
- Minute-1 BACC: 0.5583 [0.4333, 0.6750]
- Minute-5 minus minute-1 BACC: -0.0417 [-0.1333, +0.0417]
- Unstable rate: 0.2000; stable-wrong rate: 0.3583

## Limitations
- Cross-release participant linkage is not established; no pooled cohort estimate is reported.
- MCI versus healthy is a different clinical task from AD/FTD versus healthy.
- EDF physical units were missing; microvolts were assumed and documented.
- Acquisition, preprocessing, artifacts, demographics, and cohort selection may differ from the original study.
- No probability calibration, clinical diagnosis, or prospective screening performance is established.
- Small cohorts produce substantial uncertainty; repeated CV measurements are correlated.
