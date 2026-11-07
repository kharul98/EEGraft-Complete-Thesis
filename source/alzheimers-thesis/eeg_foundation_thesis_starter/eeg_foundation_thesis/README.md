# EEG Alzheimer's Thesis — Pretrained CBraMod + DSRF Decoder

**Project status (2026-11-08):** core original CBraMod backbone source is vendored (two original model files, MIT licensed), a **new low/high dual-scale residual decoder** is implemented, along with split-safe preprocessing, neural and classical baselines, ablations, reliability and interpretability utilities. **Synthetic smoke tests pass.** The actual EEG dataset and official pretrained checkpoint were not accessible inside the authoring environment, so **no real-data training and no real-world accuracy claims have been made.**

## What is ours versus what is reused?

- **Original upstream source (NOT ours):** `vendor/CBraMod/models/cbramod.py` and `criss_cross_transformer.py` from [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod), with upstream MIT license retained. The model was published at ICLR 2025.
- **Original published pretrained weights (NOT ours):** `weighting666/CBraMod` on [Hugging Face](https://huggingface.co/weighting666/CBraMod/tree/main). These weights download separately; **they are not in the ZIP**.
- **Our proposed research component:** `src/model.py` implements Dual-Scale Residual Fusion (DSRF), a learnable low-level temporal convolution path fused with high-level pretrained token features using gated residual skips and class-specific attention. **We have not demonstrated its originality relative to every past paper, nor any performance gains yet.**
- **Our experimental pipeline:** the rest of `src/` and `scripts/` runs ablations, SVM/Random Forest/XGBoost, an EEGNet-inspired scratch baseline, subject-level metrics, and gradient/occlusion explanations.

## 1. Setup (PC or Google Colab)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Install a CUDA PyTorch build matching your PC first if you have a GPU:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install xgboost          # optional, for XGBoost baseline
python scripts/smoke_test.py
```

The smoke test uses **random data and a randomly initialized mini encoder** only. Do not cite any resulting synthetic outputs in the thesis.

## 2. Obtain and verify original pretrained weights

```bash
python scripts/fetch_weights.py
```

The downloader gets `pretrained_weights.pth` from **weighting666/CBraMod** and saves it as `checkpoints/cbramod_official.pth`. It uses strict state-dict loading against the original CBraMod architecture and prints the SHA-256 hash. To inspect the entire upstream repo rather than just its two model files:

```bash
git clone https://github.com/wjq-learning/CBraMod.git external/CBraMod_full_official
```

The optional full clone is for provenance/reference only; our scripts import the vendored exact backbone source, **not** the downstream classifier in the original repository.

**OFFLINE LIMITATION:** The pretrained weights could not be downloaded in the authoring environment. On your research PC, the download step needs Internet access. The model training scripts **fail explicitly if these weights are missing**, unless you select the scratch experimental control.

## 3. Prepare your actual EEG CSV manifest

You previously examined 19-channel CSV recordings, including healthy and Alzheimer groups. This project does **not** ship or invent patient data.

An input CSV should contain **one row per sampled time point**, and named electrode columns (`Fp1`, `Fp2`, `F3`, etc.). The reader requires all 19 standard electrodes and aligns their order. If your CSV contains different labels/referencing, you must audit it rather than silently padding.

A manifest file has these columns:

| path | subject_id | label | fs_hz | unit |
| --- | --- | --- | --- | --- |
| /data/healthy/CN1.csv | real_subject_1 | healthy | 200 | uV |
| /data/severe/AD1.csv | real_subject_2 | severe | 200 | uV |

**The example entries are placeholders.** Supply your **actual** sample rate and units, obtained from dataset documentation. Do not infer frequency from CSV row count or assume a filename uniquely represents a person. If multiple CSV files belong to one patient, give every such file the same `subject_id`.

A folder-scanning helper can draft a manifest, but must be reviewed:

```bash
python scripts/make_manifest.py \
  --class-dir healthy=/path/to/healthy_folder \
  --class-dir severe=/path/to/severe_folder \
  --fs-hz 200 --unit uV --output manifest.csv
python scripts/inspect_manifest.py --manifest manifest.csv
```

Add `--class-dir mild=...` only if your labels genuinely include a third group. If an existing validated train/validation/test split exists, add a `split` column (`train`, `val`, `test`) and we **preserve** it. Otherwise, new subject-stratified partitions are generated with a fixed seed; they will be identical across all model runs using the same manifest and seed. Never split EEG windows or individual CSV rows independently.

Preprocessing: raw units are converted to microvolts, a 0.3–40 Hz bandpass is applied, signals are resampled to 200 Hz, then scaled by **100 µV** before forming non-overlapping whole-second patches. Optional common-average referencing is `--car`; it is OFF by default until the input recording reference is verified. These choices must be checked against original data provenance and compared with CBraMod upstream preprocessing. Data below 200 Hz are upsampled for interface compatibility but no missing frequencies are recovered. A 40 Hz lowpass removes 50/60 Hz mains noise without assuming which country used the equipment. Always report filter, channel order, reference, and units.

## 4. Train first pretrained baseline, then proposed architecture

Use **the same subject split for every model**:

```bash
python scripts/train.py --manifest manifest.csv \
  --run-name pretrained_linear --model linear --train-mode frozen

python scripts/train.py --manifest manifest.csv \
  --run-name proposed_fusion --model fusion --train-mode frozen \
  --robustness --batch-size 8
```

**Only after these complete:**

```bash
python scripts/train.py --manifest manifest.csv \
  --run-name without_skip --model no_skip --train-mode frozen --batch-size 8
python scripts/train.py --manifest manifest.csv \
  --run-name high_only --model high_only --train-mode frozen --batch-size 8
python scripts/train.py --manifest manifest.csv \
  --run-name low_only --model low_only --train-mode frozen --batch-size 8
python scripts/train.py --manifest manifest.csv \
  --run-name last2_fusion --model fusion --train-mode last2 --batch-size 8
python scripts/train.py --manifest manifest.csv \
  --run-name scratch_fusion --model fusion --train-mode scratch --batch-size 8
```

- `frozen`: original pretrained CBraMod weights are loaded, and kept fixed; train only decoder.
- `last2`: original weights loaded; train only final 2 transformer layers and decoder.
- `full`: original weights loaded; fine-tune all encoder layers (higher overfitting risk).
- `scratch`: randomly initialized full CBraMod (a control; **not** using pretrained weights).

For a typical 8 GB VRAM workstation, start with **4-second windows and batch size 8**, reduce to 4 if you run out of GPU memory. No high-end GPU is required just to run the frozen encoder and lightweight decoder. We have not benchmarked actual speed or VRAM use on your PC.

## 5. Literature-style baselines

```bash
python scripts/run_classical.py --manifest manifest.csv --model svm
python scripts/run_classical.py --manifest manifest.csv --model random_forest
python scripts/run_classical.py --manifest manifest.csv --model xgboost
python scripts/train.py --manifest manifest.csv --run-name eegnet_like \
  --model eegnet_like --train-mode scratch
```

Classical baselines use EEG band powers + Hjorth features (not raw samples); their classifiers use training subjects only. `eegnet_like` is an **EEGNet-inspired implementation**; it is not a claim to reproduce the reference EEGNet benchmark exactly. For formal literature-level parity, add an audited implementation from `braindecode` later. **LEAD 2026** is another particularly relevant published Alzheimer-specific foundation model, but **is not included as a running model in this initial package**. Avoid cross-paper score comparisons unless task, data, labels, montage, and split are identical.

## 6. Inspect performance and reliability

Every run saves `results/<run_name>/best.pt`, `metrics.json`, `history.csv`, `subject_splits.csv`, and per-patient probability estimates. Metrics report patient-level balanced accuracy, macro-F1, AUROC where meaningful, Brier score, calibration error, predicted-class collapse fraction and confusion matrix; bootstrap intervals require enough subjects. `--robustness` also evaluates relative 10% RMS noise and one blanked EEG channel, **together with** accuracy and a constant-majority-class control. A classifier that predicts the same class for all patients is perfectly stable but not useful.

```bash
python scripts/compare_results.py --reference proposed_fusion
```

This creates `results/comparison_table.csv` and a **paired, patient-level bootstrap** comparison. Do not tune hyperparameters based on the held-out test set. For stronger conclusions run multiple prespecified subject seeds/folds, each with the same partitions for all models and a final untouched test set where feasible.

## 7. Interpretability

```bash
python scripts/explain.py --run-dir results/proposed_fusion
```

Creates per-patient **gradient × input** channel importances, **channel occlusion** logit/probability sensitivity, class-specific **decoder attention maps**, and (for fusion) **residual gate maps**. These are **model explanations, not medical proof**. Inspect whether channels/time regions remain consistent across patients and whether removing artifacts changes conclusions. Attention alone is *not* causal evidence, and channel-importance changes do not establish a brain region is pathologically damaged.

## 8. Research integrity and unresolved dependencies

- Verify actual raw EEG sampling rate, reference, unit, channel order, recording duration, and diagnostic label definitions before any result is reported.
- Retain previous SVM/XGBoost/etc results as historical benchmarks only if they used the **same subject-level split and evaluation unit**; otherwise rerun them fairly with this pipeline.
- The proposed DSRF mechanism is a **design hypothesis**, not a proven contribution. An improvement must outperform matched heads and `no_skip` ablation and survive repeated patient-level evaluation.
- Check subject/dataset overlap with CBraMod/LEAD pretraining datasets. If patient recordings overlap, do not call it clean external testing.
- No patient-level prospective Alzheimer's **risk prediction** is possible from binary/labelled current-diagnosis classification alone. This code is for research, not clinical diagnosis.

## Published references

1. Wang et al. (2025). *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.* ICLR. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html) · [Source](https://github.com/wjq-learning/CBraMod) · [Weights](https://huggingface.co/weighting666/CBraMod)
2. Jiang et al. (2024). *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI.* ICLR. [Source](https://github.com/935963004/LaBraM)
3. Yang et al. (2023). *BIOT: Biosignal Transformer for Cross-data Learning in the Wild.* NeurIPS. [Source](https://github.com/ycq091044/BIOT)
4. Wang et al. (2026). *LEAD: An EEG Foundation Model for Alzheimer's Disease Detection.* TMLR. [Source & weights](https://github.com/DL4mHealth/LEAD)
5. Lawhern et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.* J Neural Eng. [Paper](https://pubmed.ncbi.nlm.nih.gov/29932424/)

**Status is implementation-readiness, not trained model performance.**
