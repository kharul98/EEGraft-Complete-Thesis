# RESEARCH HISTORY

## 2026-11-06 - Initial environment

Verified:
- WSL2 Ubuntu 26.04.1 LTS
- NVIDIA RTX 4080 SUPER, 16376 MiB
- approximately 31 GiB RAM visible to WSL
- approximately 954 GiB free disk space
- Python 3.14.4
- Git 2.53.0

## 2026-11-06 - DARWIN feasibility

DARWIN was downloaded through the UCI interface.
Raw feature and target snapshots were saved.
SHA-256 hashes were generated.

This established technical feasibility but did not establish novelty.

## 2026-11-06 - DARWIN novelty collision

A focused current literature search found:

Fatima T, Lashari TA, Moetesum M, Lashari SA.
"Handwriting-based Alzheimer's disease classification on the DARWIN dataset:
Repeated nested validation, task-level reduction, and cross-validated
explainability."
PLOS ONE. Published 2026-10-26.
DOI: 10.1371/journal.pone.0361117

This paper already includes:
- repeated nested cross-validation
- leakage-contained preprocessing
- classifier comparison
- task reduction
- calibration
- SHAP
- task-selection stability

Decision:
Do not make the planned DARWIN leakage/task-selection analysis the main thesis.
Retain DARWIN as fallback only.

## 2026-11-06 - EEG candidate selected for feasibility testing

Candidate dataset:
OpenNeuro ds004504 / NEMAR on004504.

Provisional scientific direction:
recording-duration sensitivity of Alzheimer's EEG spectral features and
subject-level classification.

Reason:
The question is practical, measurable, paired within subjects, and does not
depend on inventing a novel deep-learning architecture.

The protocol remains unfrozen until the actual post-cleaning recording
durations and data structure are verified.

## 2026-11-06 - EEG derivative retrieval correction

The initial nemar-py 0.3.3 command failed with:

"No files matched the BIDS query: datatype=eeg; scope=derivatives."

This was classified as a retrieval-interface failure, not a dataset failure.

The authoritative NEMAR/OpenNeuro repository was verified to contain
preprocessed and denoised EEG under derivatives/.

The retrieval method was changed before any scientific analysis to the
dataset's official git-annex repository mechanism.

The failed console output was retained as provenance rather than deleted.
