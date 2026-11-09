# Reliability of EEG-Based Dementia Classification
## Temporal Consistency, Recording-Duration Sensitivity, Calibration, and Foundation-Model Evaluation

Bachelor's Thesis | November 2026
Student: [Name] | Supervisor: [Name] | University: [University]


# Abstract
This thesis evaluates whether EEG-based dementia predictions remain reliable beyond one accuracy figure. A 64-person selected Alzheimer's disease (AD), frontotemporal dementia (FTD), and control study investigated SVM label stability across five minutes; a separate preregistered 65-person AD/control analysis investigated available EEG duration from 30 to 300 seconds. The original SVM first-minute balanced accuracy was 73.48% for AD and 73.72% for FTD; 44.35% and 33.66% of correlated subject-repeat decisions changed over five minutes, while 8.70% and 13.17% remained consistently wrong. The recording-duration study observed 83.05% AD/control balanced accuracy at 180 seconds (center), without demonstrating a universal optimal recording length. A later frozen pretrained-CBraMod model with a custom dual-scale residual fusion (DSRF) decoder achieved 77.06% development balanced accuracy, compared with 76.61% for XGBoost. On a once-used 13-person holdout, DSRF scored 70.24% and XGBoost 77.38%. Separate MCI and external-feasibility experiments exposed stable erroneous predictions and dataset incompatibilities. These findings support measuring accuracy, temporal stability, calibration, and robustness separately, not clinical diagnosis or future Alzheimer-risk prediction.


# 1. Introduction and Research Objectives
Electroencephalography captures time-varying brain electrical activity. Even when retrospective group discrimination appears useful, a model can make different decisions about the same person at different time points, or repeat a consistent but incorrect decision. This creates a reliability problem beyond conventional accuracy. The thesis asks whether temporal consistency guarantees correctness; how recording duration and temporal position affect spectral measures and prediction; what calibration and error detection add; how recording conditions and external cohorts challenge predictions; and whether a pretrained foundation-model decoder is more reliable than simpler classifiers. These are empirical methodological questions; the thesis does not propose a clinical diagnostic instrument.


# 2. Literature and Novel Contribution
The OpenNeuro/NEMAR cohort and preprocessed derivatives were published by Miltiadous and collaborators [1,2]. Earlier Alzheimer's EEG window-duration research exists [3], making a claim of first-ever duration analysis inappropriate. The thesis's duration contribution is its paired 30–300-second comparison across start, center and end segments under repeated subject-disjoint nested validation and uncertainty controls. The later neural experiments reuse the existing pretrained CBraMod architecture [4] rather than inventing a new foundation model; they add a task-specific residual decoder and matched architectural comparisons. EEGNet-like baselines are inspired by the published EEGNet family [5]. The principal contribution is a structured, audited reliability evaluation with appropriately reported negative findings.


# 3. Data and Participant Lineage
The NEMAR/OpenNeuro source has 88 people: 36 AD, 23 FTD and 29 controls. The selected original temporal dataset has 23 AD, 18 FTD and 23 controls (64 people); the 23 controls are shared across its two binary tasks, so 46 AD/control and 41 FTD/control assessments do not represent 87 distinct patients. The preregistered duration primary comparison uses 65 AD/control people; its AD/FTD secondary analysis uses 59. The foundation-model study also uses 65 AD/control people, with 52 development and 13 locked test patients. The original and later datasets overlap and must not be presented as independent cohorts.


| Source/selection | Composition | N |
| --- | --- | --- |
| NEMAR source | 36 AD / 23 FTD / 29 controls | 88 |
| Original temporal | 23 AD / 18 FTD / 23 controls | 64 |
| Earlier neural baseline | 23 AD / 23 controls | 46 |
| Preregistered duration | 36 AD / 29 controls | 65 |
| Foundation development | 29 AD / 23 controls | 52 |
| Foundation holdout | 7 AD / 6 controls | 13 |
| Isfahan MCI archive | 29 MCI / 32 controls | 61 |


Direct participant-ID checking confirms that the original and later studies share 46 people and their union contains 83 of the 88 NEMAR participants. No external AD patient cohort was independently validated.


# 4. Methods
## 4.1 Five-Minute Temporal SVM Reliability
Six interpretable spectral features (relative delta, theta, alpha, beta and gamma power plus theta/alpha ratio) were used in standardized class-balanced RBF SVMs. Fivefold repeated outer cross-validation (five repetitions) held patients out, with training-only nested hyperparameter selection. First-minute models were evaluated on five held-out minutes without refitting. Outcomes distinguished stable-correct, stable-wrong, and unstable labels. Sigmoid probability calibration, selective prediction, MMSE associations, recording-condition testing, temporal explanation drift, and feature-group ablation were examined with leakage-aware designs. Repeated patient-repetition records are correlated.

## 4.2 Preregistered Duration Protocol
The 65-person AD/control analysis used dataset-provided denoised EEG at 500 Hz, Welch PSD from four-second windows with 50% overlap and spectral bands from 1 to 45 Hz. Prespecified durations 30, 60, 120, 180 and 300 seconds were evaluated at start, center and end positions. A class-balanced L2 logistic-regression baseline used repeated 5x10 outer subject-stratified nested CV with fourfold inner tuning. Paired bootstrap intervals compared each shorter duration to 300 seconds. A 100-permutation label-shuffling negative control and a separate 59-person AD/FTD comparison were included.

## 4.3 Foundation Model and Reliability Stress Tests
Nineteen EEG channels were resampled to 200 Hz and processed into at most 24 four-second windows per person. The official pretrained CBraMod was strictly loaded and frozen; trainable DSRF fused pretrained tokens with multiscale raw-signal features through learned gates and an additive residual pathway. Eight architectures or classical learners were assessed over five development folds. A parameter-matched residual-versus-blend ablation, 24-window channel occlusion, and five repeated 10%/20% noise and channel-zeroing experiments were performed on development patients only. Final five-epoch DSRF and fixed XGBoost models trained on the same 52 people and were each assessed once on the same locked 13-person holdout.


# 5. Results
## 5.1 Original AD and FTD Accuracy, Stability and Calibration


| Metric | AD/control | FTD/control |
| --- | --- | --- |
| First-minute BAcc | 73.48% | 73.72% |
| Calibrated BAcc | 74.35% | 69.32% |
| Calibrated Brier | 0.1861 | 0.1917 |
| Training-prior Brier | 0.2513 | 0.2479 |
| Stable-correct | 46.96% | 53.17% |
| Stable-wrong | 8.70% | 13.17% |
| Prediction changed | 44.35% | 33.66% |


These rates represent 230 AD and 205 FTD correlated subject-repeat assessments. The eight frequency-family ablation effects did not survive Holm correction. Descriptive eyes-open/photic versus eyes-closed balanced-accuracy changes were -8.83 pp for AD and -4.23 pp for FTD, with intervals containing zero. Calibration altered probabilistic quality, but should not be described as an across-task classification improvement.

## 5.2 Preregistered Duration Results


| Center duration | AD/control BAcc | AUROC |
| --- | --- | --- |
| 30 s | 80.27% | 0.839 |
| 60 s | 78.21% | 0.842 |
| 120 s | 79.93% | 0.852 |
| 180 s | 83.05% | 0.861 |
| 300 s | 81.32% | 0.852 |


The highest observed center result was 83.05% at 180 seconds; 300 seconds reached 81.32% and 60 seconds 78.21%. The paired 180-minus-300 difference was +1.72 pp (exploratory 95% interval 0 to +5.17), not evidence for a universally optimal duration or formal equivalence. The 300-second-center shuffled-label check observed balanced accuracy of 79.93% against a null mean of 49.82% (100 permutations, empirical p=0.0099); this checks internal discrimination, not a duration effect. AD/FTD discrimination was substantially weaker and often near chance. Exploratory adaptive stopping did not demonstrate a consistently better accuracy–time tradeoff.

## 5.3 Earlier 46-Person Multi-Model Comparison


| Model | BAcc | AUROC |
| --- | --- | --- |
| LogisticRegression | 71.74% | 0.873 |
| SVM | 76.09% | 0.813 |
| RandomForest | 69.57% | 0.749 |
| NestedSelection | 69.57% | 0.807 |
| BasicCNN | 71.74% | 0.775 |
| EEGNetV1 | 52.17% | 0.575 |
| EEGNetV2 | 69.57% | 0.807 |
| EEGConformer | 71.74% | 0.839 |
| CBraModFrozen | 71.74% | 0.731 |


This earlier benchmark is not numerically comparable to the later DSRF 52-patient development study or its held-out test; it demonstrates that classical, CNN, EEGNet, Conformer and frozen-CBraMod baselines were attempted before the custom decoder.

## 5.4 Later Eight-Model Development Comparison


| Model | Pooled BAcc % | Macro-F1 % | Correct |
| --- | --- | --- | --- |
| DSRF | 77.06 | 76.79 | 40/52 |
| XGBoost | 76.61 | 76.61 | 40/52 |
| EEGNet-inspired | 74.89 | 74.77 | 39/52 |
| Random Forest | 72.71 | 72.71 | 38/52 |
| High only | 72.26 | 72.42 | 38/52 |
| SVM | 71.81 | 72.04 | 38/52 |
| No skip | 71.44 | 71.06 | 37/52 |
| Low only | 62.74 | 62.79 | 33/52 |


DSRF obtained the highest observed pooled development BAcc (77.06%) but was practically close to XGBoost (76.61%); each classified 40 of 52 development patients correctly. Comparisons are exploratory.

## 5.5 Capacity-Matched Ablation and Explanation


| Fusion | Pooled BAcc % | Correct |
| --- | --- | --- |
| residual | 77.06 | 40/52 |
| blend | 73.16 | 38/52 |


Matched residual fusion exceeded gated blend numerically (77.06% vs 73.16% BAcc), without conclusive paired statistical evidence. Full-window occlusion rankings were AD: T5, Pz, F8, Fz, C3; controls: O2, T6, O1, Fp2, F3. These are model sensitivities, not causal clinical EEG biomarkers.

## 5.6 Repeated Synthetic Robustness


| Decoder | Perturbation | Mean BAcc % | Flip % | Brier |
| --- | --- | --- | --- | --- |
| DSRF | noise_10pct | 71.87 | 23.08 | 0.3435 |
| DSRF | noise_20pct | 70.07 | 27.69 | 0.41 |
| DSRF | one_channel_zeroed | 76.78 | 7.31 | 0.3629 |
| No skip | noise_10pct | 71.14 | 20.0 | 0.3987 |
| No skip | noise_20pct | 69.75 | 28.08 | 0.4665 |
| No skip | one_channel_zeroed | 71.65 | 17.69 | 0.3904 |
| High only | noise_10pct | 69.66 | 20.77 | 0.3625 |
| High only | noise_20pct | 72.19 | 25.0 | 0.4021 |
| High only | one_channel_zeroed | 69.48 | 6.92 | 0.3544 |


DSRF's mean BAcc dropped to 71.87% and 70.07% under 10% and 20% Gaussian noise, with mean label-flip rates of 23.08% and 27.69%. A one-electrode zeroing condition yielded 76.78% BAcc. Repeated perturbations are not independent patient replications; a constant-label classifier can have zero flips and 50% BAcc.

## 5.7 Locked 13-Patient Final Test


| Model | BAcc | Accuracy | Macro-F1 | AUROC | Correct |
| --- | --- | --- | --- | --- | --- |
| DSRF | 70.24% | 69.23% | 69.05% | 0.857 | 9/13 |
| XGBoost | 77.38% | 76.92% | 76.92% | 0.905 | 10/13 |


DSRF identified 4/7 AD and 5/6 controls; XGBoost identified 5/7 AD and 5/6 controls. The paired BAcc difference DSRF minus XGBoost was -7.14 pp (exploratory 95% interval -28.57 to +14.29 pp; exact McNemar p=1.0). This small within-cohort holdout neither establishes superiority nor clinical diagnostic validity.

## 5.8 Isfahan MCI and External Feasibility


| Release | MCI/healthy | First-minute BAcc | Stable-wrong | Unstable |
| --- | --- | --- | --- | --- |
| EEG_1 | 11/14 | 40.78% | 59.20% | 5.60% |
| EEG_2 | 12/12 | 55.83% | 35.83% | 20.00% |


The EEG_1 classifier was strongly biased toward MCI. An always-MCI negative control also had 56% stable-wrong labels, demonstrating that stability alone is not evidence of useful discrimination. Figshare had 471 MATLAB files but no verified comparable 19-channel control population; no external AD classification accuracy was produced. Dryad and BrainLat results remained recording-level spectral-feasibility investigations with unresolved independence, montage and site confounds.


# 6. Discussion and Scientific Contribution
The studies consistently show why accuracy, temporal consistency, calibration, signal quality, recording duration and external transportability are distinct. The original SVM often changed predictions across five minutes, while the Isfahan negative control could remain consistently wrong. The preregistered duration study found variable outcomes but no formally optimal acquisition time. The custom DSRF decoder improved upon some simpler development variants but not the final XGBoost comparator on 13 held-out patients, and it remained vulnerable to synthetic noise. The thesis contribution is a reproducible reliability-focused empirical program, including negative results and conservative uncertainty statements. It is not a new foundation model or a deployable diagnostic tool.


# 7. Limitations and Ethics
Multiple analytical cohorts overlap within one original 88-person EEG source and must not be counted as independent validation. The original selected 46-patient AD/control group has exclusions and three publication-review cases requiring final objective explanations. Repeated CV estimates share participants and are correlated; the 13-patient test has wide intervals. Eyes-open/photic shifts are confounded with recording protocols; channel occlusion is not physiological causality. The foundation classifier used up to 24 four-second EEG windows per person, not five continuous minutes of temporal stability. The Isfahan releases are not proven independent patients and their EDF unit fields required assumptions; Figshare, Dryad and BrainLat cannot be claimed as successful external clinical AD validation.

Ethics statement to finalize locally: This is secondary analysis of publicly released, de-identified datasets without new participant recruitment or patient intervention. Confirm the university-required ethics statement and each source dataset's data-use and redistribution terms before submission or public redistribution.


# 8. Conclusion
EEG dementia classifiers can be internally discriminative while unstable across recording time, stable while wrong, and vulnerable to changes in acquisition and corruption. Strong architecture or a single high accuracy figure does not guarantee reliability. Patient-level validation, negative controls, sensitivity studies and cautious reporting are essential. Independent prospective clinical validation remains future work.


# References
[1] Miltiadous A, Tzimourta KD, Afrantou T, et al. A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. Data. 2023;8(6):95. DOI: 10.3390/data8060095

[2] Miltiadous A, et al. A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects. NEMAR on004504 v1.0.0. 2026. DOI: 10.82901/nemar.on004504

[3] Tzimourta KD, et al. EEG Window Length Evaluation for the Detection of Alzheimer's Disease over Different Brain Regions. Brain Sciences. 2019;9(4):81. DOI: 10.3390/brainsci9040081

[4] Wang J, Zhao S, Luo Z, Zhou Y, Jiang H, Li S, Li T, Pan G. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR. 2025. https://openreview.net/forum?id=NPNUHgHF2w

[5] Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng. 2018;15(5):056013. DOI: 10.1088/1741-2552/aace8c


# Appendix A. Evidence and Provenance
Original temporal and calibration evidence: reports/thesis_final/milestone_7c_original_ad_ftd_recomputed_v2/ and milestone_4c_reproducibility_v1/. Duration protocol and metrics: PREREGISTRATION.md, tables/ and results/duration_experiment/. Earlier model baselines: reports/thesis_final/publication_evidence_v1/. Isfahan MCI: reports/thesis_final/milestone_5i_isfahan_negative_controls_v1/. Foundation benchmark, ablation, explanations, perturbation and final results: eeg_foundation_thesis_starter/eeg_foundation_thesis/results/. The frozen final protocol and test status.json must remain unmodified. University cover, supervisor approval, ethics signoff and independently verifiable external clinical validation are not supplied by this evidence package.

