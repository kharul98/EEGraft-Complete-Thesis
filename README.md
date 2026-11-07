# EEG Dementia Thesis - Private Research Backup

This repository contains a timestamped backup of the
current EEG dementia thesis working files.

See PROJECT_HISTORY.md for the research timeline,
methodology, numerical results, and limitations.

## Layout

source/     - copied WSL thesis source trees
documents/  - research reports found in Windows Downloads
SOURCE_ROOTS.txt - original source directories
EXCLUDED_DATA_PATHS.txt - data directories not uploaded
FILES_SHA256SUM.txt - content hashes for restoration checks

All source trees were copied without changing the originals.

## Included

Where present in the copied source trees:
- Python scripts and configuration
- Research manifests and metadata
- CSV experiment results and out-of-fold predictions
- Development and held-out evaluation summaries
- Figures and research documentation
- Trained checkpoint files through Git LFS
- Existing repository documentation

## Excluded intentionally

Raw EEG recordings, datasets, third-party raw archives,
virtual environments, caches, and credentials.

Before reproducing, retrieve the datasets from their
authorized sources and rebuild the environment.

## Restoration

Clone this private repository and run:

    git lfs pull
    sha256sum --quiet -c FILES_SHA256SUM.txt

See SOURCE_ROOTS.txt to reconstruct the directory layout.

The backup reflects the filesystem state at the time
of the snapshot. It does not automatically synchronize
future Ubuntu changes to GitHub.
