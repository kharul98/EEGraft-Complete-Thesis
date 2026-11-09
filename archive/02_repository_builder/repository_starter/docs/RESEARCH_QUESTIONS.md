# Research questions and thesis logic

**Overall question.** What does it mean for EEG-based Alzheimer's classification to be reliable beyond a single accuracy estimate, and how does a custom decoder attached to a pretrained EEG foundation model perform under subject-disjoint evaluation?

- **RQ1 (early investigation):** How consistent and correct are spectral EEG classifier predictions across recording time, and can a model be stably wrong?
- **RQ2 (duration and controls):** How do recording duration and acquisition conditions influence measured spectral/classification performance, and what limits interpretation?
- **RQ3 (later architecture):** Can a custom gated dual-scale residual fusion (DSRF) decoder effectively combine frozen CBraMod representations with low-level temporal features?
- **RQ4 (comparisons):** How does DSRF compare with SVM, random forest, XGBoost, EEGNet-inspired networks, and structural ablations on patient-disjoint development/test partitions?
- **RQ5 (reliability under stress):** How do synthetic noise, electrode masking, probability error, and label changes alter interpretation of apparent accuracy?

**Why two tracks are one thesis:** The early experiments establish the importance of checking temporal consistency and incorrect-but-stable decisions. The later experiments investigate whether a pretrained representation plus a learned decoder offers an effective classification option and how robust it is under controlled perturbations. The second track does **not** replicate the first track's five-minute protocol. They contribute different evidence to one reliability-focused research argument.

Do not claim that the original 64-person and later 65-person cohorts were pooled or reconciled, or that the DSRF directly solved the SVM temporal-flip problem.
