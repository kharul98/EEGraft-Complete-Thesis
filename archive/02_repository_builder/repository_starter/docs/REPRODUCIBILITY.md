# Reproducibility and data access

## Original repositories

- Legacy duration analysis: `archive/01-legacy-duration-study/`
- Later multi-workspace backup: `archive/02-research-backup/`

**Unmodified original source files and Git histories** are brought into the new repository via Git subtree without squashing. They are not blended or overwritten. Read the protocol configuration and run manifests inside each source tree before executing any experiment.

## Data

Primary EEG: NEMAR/OpenNeuro `on004504` (`ds004504`) version 1.0.0, dataset DOI `10.82901/nemar.on004504`. Dataset-level count 88 participants (36 AD, 23 FTD, 29 CN). The later AD/CN analytic cohort has 65 people; original 64-person study selected a different analytical subset. Raw EEG files are intentionally excluded from GitHub and this package. Obtain authorized copies from the repository cited in the original source documentation.

The later CBraMod frozen pretraining weights are third-party material. Do not assume they may be redistributed without reviewing their terms. Check SHA256 and model configuration against archived `results/final_protocol_2026.json` and `PROJECT_HISTORY.md`.

## Checkpoints / Git LFS

The backup may use **Git LFS** for trained checkpoints. A Git subtree import preserves Git commit history and LFS **pointer files**, but it does not automatically migrate LFS binary objects to the new GitHub repository. To make standalone checkpoint downloads work, an authorized maintainer must `git lfs fetch --all` from the original remote and `git lfs push --all` to the new one, subject to license, authorization, quota, and visibility. The import script deliberately skips automatic LFS downloads to prevent unexpected bandwidth or licensing consequences.

## Thesis (sanitized distribution)

- Main source: `thesis/overleaf/main.tex`
- Library: `thesis/overleaf/bibliography/references.bib`
- Chapters: `thesis/overleaf/chapters/`
- Reproducibility audit: `thesis/overleaf/reproducibility/`
- Figures: `thesis/overleaf/images/`

Student handwritten signature images and signed PDFs are not included. The optional-signature macro already provides blank signature space when they are absent. Copy signatures into a **local, untracked** Overleaf project only when authorized for a final submission.

**Build in Overleaf:** New Project -> Upload Project -> select the `thesis/overleaf` folder packaged as ZIP, choose `main.tex`, use pdfLaTeX + Biber. For local build install TeX Live with required packages, `latexmk` and Biber.

## Record validation

Run `python3 tools/verify_repository.py` from the final repository root. Confirm no raw data/private files are present in the tracked current tree, confirm key source files and headline evidence, and run a separate review of historical Git objects before publishing.
