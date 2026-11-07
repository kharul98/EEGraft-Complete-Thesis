# Prespecified experiment matrix — do not fill scores before evaluation

| Model ID | Backbone | Reused pretrained? | Classification decoder | Hypothesis being tested |
|---|---|---|---|---|
| B1 | None | No | SVM + band power/Hjorth | Classical benchmark |
| B2 | None | No | Random Forest + same features | Ensemble benchmark |
| B3 | None | No | XGBoost + same features | Boosting benchmark |
| B4 | None | No | EEGNet-inspired scratch | Compact neural benchmark |
| F1 | CBraMod | YES | Global average + linear | Pretrained linear probe |
| F2 | CBraMod | YES | Global average + MLP | Baseline decoder capacity |
| A1 | CBraMod | YES | DSRF, full gated skip | Proposed architecture |
| A2 | CBraMod | YES | DSRF, fusion without residual | Contribution of residual path |
| A3 | CBraMod | YES | DSRF, high-only | Necessity of low-level evidence |
| A4 | CBraMod | YES | DSRF, low-only | Necessity of pretrained high-level evidence |
| A5 | CBraMod | NO | DSRF, scratch encoder | Benefit of pretrained weights |
| A6 | CBraMod | YES | DSRF, final two layers tuned | Benefit / risk of partial fine-tuning |

Primary endpoint: **subject-level balanced accuracy**. Secondary: macro-F1, calibration error, multiclass Brier, per-class recall, AUROC (if test has all classes), and subject-level bootstrap confidence intervals. All model hyperparameters selected using only TRAIN/VAL subjects; reserve TEST for final comparison.

Stability analysis: noisy signals, channel blanking, label flip rate, degraded balanced accuracy, and constant-majority controls. Interpretability: class-specific decoder attention is descriptive; confirm with gradient x input and channel occlusion; also visualize learned residual gate by channel/patch; repeat for multiple subjects, class groups and perturbations.

Statistical constraints: If patient counts are very small, single-split scores and bootstrap CIs are unstable. Run 3-5 pre-specified patient-stratified seeds, **with matched folds across all methods**, or patient-grouped nested CV if appropriate. Avoid a new test split every time a model is added. All matched model comparisons MUST use the exact same held-out subject IDs.

Avoid unsupported novelty claims: gated fusion, low/high features, CNN skips and EEG explainability exist in the literature. The contribution can be the controlled empirical design and an adaptation of these ideas to a pretrained CBraMod EEG backbone, not claiming to invent skip connections or a new foundation model.

Key sources: https://github.com/wjq-learning/CBraMod ; https://github.com/DL4mHealth/LEAD ; https://github.com/935963004/LaBraM ; https://github.com/ycq091044/BIOT
