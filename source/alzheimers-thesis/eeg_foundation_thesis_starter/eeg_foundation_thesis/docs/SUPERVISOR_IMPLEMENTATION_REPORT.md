# Supervisor implementation report — EEG foundation model + custom decoder

Date: 2026-11-08. Status: code complete for initial offline implementation and synthetic smoke tests; **real EEG evaluation and the official weight download remain to be completed on the research PC**.

## Supervisor requirement 1 — foundation models pretrained on large-scale EEG

Selected **CBraMod (Wang et al., ICLR 2025)** because it provides published pretraining, original source, public weights, 200-sample/one-second token representation and support for a 19-electrode montage. Alternatives to reference and potentially evaluate: **LaBraM (ICLR 2024)**, **EEGPT (NeurIPS 2024)** and **BIOT (NeurIPS 2023)**. Critically, **LEAD (TMLR 2026)** already studies an EEG foundation model for AD detection and must be discussed in the novelty positioning.

Source repo: https://github.com/wjq-learning/CBraMod

Pretrained weights: https://huggingface.co/weighting666/CBraMod

LEAD: https://github.com/DL4mHealth/LEAD

## Requirement 2 — use the original repository code and pretrained weights

- Bundled `vendor/CBraMod/models/cbramod.py` is the original MIT-licensed source (GitHub blob SHA in `vendor/CBraMod/PROVENANCE.md`); the companion transformer implementation is functionally equivalent to upstream, with minor unused-comment trims.
- `python scripts/fetch_weights.py` downloads the **original publicly released** `pretrained_weights.pth`. It fails if inaccessible rather than swapping in random weights.
- `src/model.py:load_encoder` performs PyTorch `load_state_dict(..., strict=True)` *before* replacing the original reconstruction projection with identity to expose 200-dimensional tokens.
- Train-mode `frozen` preserves all pretrained parameters and trains only the custom head; `last2` unfreezes the final two transformer layers; `scratch` is an explicit no-pretraining control.
- A checkpoint SHA-256 is saved into experiment metadata for provenance. **No weight file was available to download in the current authoring environment**; real pretrained loading has not been executed yet.

## Requirement 3 — propose a custom decoder

Input EEG windows `x` have shape **[batch, 19, P, 200]**, where `P` = 1-second patches (usually four).

**High-level route:** `H = LayerNorm(Linear(CBraMod(x)))`, with shape `[B,19,P,128]`.

**Low-level route:** two parallel Conv1D branches with 9- and 41-sample kernels, GroupNorm, GELU, temporal pooling, and projection to `[B,19,P,128]`, denoted `L`.

**Learned skip route:** `G = sigmoid(Linear(concat(H,L)))`; `F = LayerNorm(H + G*L)`; trained channel- and patch-specific feature gates let the decoder preserve low-level waveform evidence and combine it with global pretrained spatial/temporal features. This is the **proposed Dual-Scale Residual Fusion (DSRF) decoder**.

**Classification:** a class-specific softmax attention over the `19*P` tokens creates a class-specific pooled vector. The class weight vector converts each pooled vector into one class logit. This supports **2 or 3 labels** without changing the pretrained encoder.

`no_skip`, `high_only`, and `low_only` disable specific paths, making the mechanism scientifically testable. The name is internal, not a claim of established novelty. Skip connections, gating, and multiscale EEG convolution have prior literature; exhaustive novelty review remains necessary.

## Requirement 4 — match literature baselines and evaluate rigorously

For the same patient-held-out split, windowing and preprocessing: SVM on bandpower+Hjorth; Random Forest on the same features; XGBoost on the same features; a compact **EEGNet-inspired** neural baseline (not the official EEGNet implementation); pretrained CBraMod linear and MLP probes; proposed decoder and architectural ablations. Official EEGNet and LEAD can be added as separate reproducibility studies. Reporting is on **subject-level predictions**, after averaging window probabilities: balanced accuracy primary, macro F1, per-class recall, AUROC, Brier, calibration, majority-class collapse check, and paired patient-bootstrap comparisons. Small datasets require strict caveats and repeated patient-group splits.

Previous SVM/XGBoost results count as comparable only when original subject IDs, partitions and evaluation unit match those used here; never paste old percentages into the new comparison table by assumption.

## Requirement 5 — interpretability and intelligent skip analysis

- Class-specific decoder attention maps display allocation of token *weight* (not causal evidence).
- Gradient times input maps and channel occlusion quantify model sensitivity on held-out subjects.
- Proposed DSRF additionally returns mean residual gate strength by electrode and 1-second patch, plotted independently. The gate visualization must be interpreted **alongside** occlusion and predictions, not as proof of a disease biomarker.
- Robustness tests measure degraded balanced accuracy and prediction flips under low-amplitude white noise or one channel blanked. Constant-majority predictions provide a stability counterexample (zero flips, low BAcc).

## Reproducible run order (research PC)

```bash
pip install -r requirements.txt
pip install xgboost
python scripts/smoke_test.py
python scripts/fetch_weights.py
python scripts/make_manifest.py --class-dir healthy=/verified/healthy --class-dir severe=/verified/severe --fs-hz ACTUAL_HZ --unit uV --output manifest.csv
# edit and inspect subject_id, labels, sample rates and units; never treat this draft as verified
python scripts/inspect_manifest.py --manifest manifest.csv
python scripts/train.py --manifest manifest.csv --run-name pretrained_linear --model linear --train-mode frozen --batch-size 8
python scripts/train.py --manifest manifest.csv --run-name proposed_fusion --model fusion --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest manifest.csv --run-name without_skip --model no_skip --train-mode frozen --batch-size 8
python scripts/run_classical.py --manifest manifest.csv --model svm
python scripts/run_classical.py --manifest manifest.csv --model random_forest
python scripts/run_classical.py --manifest manifest.csv --model xgboost
python scripts/train.py --manifest manifest.csv --run-name eegnet_like --model eegnet_like --train-mode scratch --batch-size 8
python scripts/explain.py --run-dir results/proposed_fusion
python scripts/compare_results.py --reference proposed_fusion
```

**Replace** `ACTUAL_HZ` only with the documented acquisition frequency. If a third diagnosis/severity group exists and is clinically validated, add its `--class-dir` argument. Do not run clinical interpretation until checking the data license, reference montage and subject grouping.

## What can honestly be claimed today?

**Complete:** design and working code for CBraMod integration, strict pretrained-weight loader, proposed decoder and low/high skip ablations, binary/three-class outputs, split-aware CSV input, train/test outputs, classical comparators, interpretability visualizations, synthetic smoke tests.

**Not complete:** official checkpoint physically retrieved; confirmed exact local EEG acquisition settings and patient identifiers; trained real AD model; measured real accuracies, calibration or clinical relevance; official EEGNet or LEAD replication; publication-level novelty assessment. No demonstration of future AD risk prediction from a present-label classifier.
