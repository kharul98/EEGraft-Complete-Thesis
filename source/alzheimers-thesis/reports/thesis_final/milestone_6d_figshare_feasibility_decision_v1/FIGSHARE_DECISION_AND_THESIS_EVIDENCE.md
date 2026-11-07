# Figshare External Alzheimer Validation: Feasibility Decision

**Decision: Do not train or report external AD validation on this archive under the original 19-channel, 60-second protocol.**

The audited archive contains 471 MATLAB files: 305 AD, 56 MCI, and 110 control recordings. These are not participant counts. Folder/file proxies total 57 AD, 7 MCI, and 102 control.

A standard 19-column recording with nominal duration of at least 60 seconds occurs in 44 AD folders but only 1 control folder.

The remaining 100 root-level control files use segmenty matrices with 22 rows. Of these, 82 have nominal duration >=60 seconds under an unconfirmed 128 Hz assumption.

Ten export files have nonstandard column counts.

A shared electrode-to-column mapping, physical units, and chronological continuity across fragments have not been verified.

**Interpretation:** This is a dataset compatibility limitation, not a negative test of the existing Alzheimer classifier. Figshare AD-versus-control balanced accuracy has not been measured.

## Completed MCI Extension: Different Clinical Task

| Release | MCI / Healthy | BACC | Stable-wrong | Unstable | Constant-label BACC |
|---|---:|---:|---:|---:|---:|
| EEG_1 | 11/14 | 0.4078 | 0.5920 | 0.0560 | 0.5000 |
| EEG_2 | 12/12 | 0.5583 | 0.3583 | 0.2000 | 0.5000 |

The MCI results use five repeated, correlated subject-held-out assessments. They do not represent independent observations of 245 individuals.

EEG_1 was biased toward predicting MCI. The negative control demonstrated that temporal stability does not necessarily indicate diagnostic correctness.

## Thesis Integration

Keep original AD/FTD classification, temporal stability, calibration, and ablation analyses as the main research results.

Report Isfahan MCI as a separate methodological extension and Figshare as an audited dataset that did not meet the intended external validation requirements.

Next: consolidate the original AD/FTD evidence, prepare final tables, complete the Methods, Results, Discussion, and Conclusion chapters, and prepare for supervisor review.
