# EEGraft Reproducibility Guide

## 1. Restore the source

Clone the repository and retrieve Git LFS objects.

    git clone https://github.com/kharul98/EEGraft-Complete-Thesis.git
    cd EEGraft-Complete-Thesis
    git lfs pull

## 2. Locate the research code

    cd source/alzheimers-thesis

This contains the primary research working tree.

The earlier research code is separately preserved under
archive/01_original_duration_thesis.

## 3. Establish the Python environment

Identify the environment associated with the particular
experiment before installing dependencies.

Where a suitable requirements file exists:

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.lock.txt

This command is an example; confirm the actual dependency
file and Python version for the selected experiment.

## 4. Restore authorized datasets

Retrieve datasets from official sources.

Verify versions, data integrity, inclusion rules,
and recorded preprocessing configurations.

Do not infer participant identity or source provenance
from filename similarities alone.

## 5. Restore checkpoints

Retrieve Git LFS files and confirm checkpoint integrity.

Match checkpoint identifiers to saved run configurations,
model architecture, and input preprocessing.

## 6. Reproduce evaluation

Use the exact scripts, configurations, split manifests,
stored predictions, and thresholds for the relevant run.

Maintain subject-level independence.

Do not refit calibration or choose a threshold using
locked-test labels.

## 7. Verify outputs

Compare reproduced outputs with the archived metrics
and prediction-level evidence.

Any missing source file, checkpoint, or configuration
must be documented as a reproducibility gap.
