# Implementation QA and evidence log

Test date: 2026-11-08.

- PASS: `python scripts/smoke_test.py` (CPU) — loads an upstream-compatible two-layer **random state_dict with strict=True**, tests pretrained interface shape, decoder gradients, frozen encoder, 2/3-class outputs, 4 decoder ablations, class-attention normalization and patient-group CSV splits.
- PASS: temporary randomly generated 15-subject, 3-label CSV dataset, no disease features — `inspect_manifest.py`, an EEGNet-inspired 1-epoch scratch training run, `run_classical.py --model svm`, `compare_results.py`, `explain.py`, all with no runtime exceptions. These outputs were removed from the distributed ZIP.
- PASS: pretrained checkpoint absence triggers an explicit error BEFORE results directory creation; no silent replacement with random weights.
- PASS: metadata includes sources, model flags, patient counts, subject-level predictions, and checkpoint hash if present.

NOT tested here: actual official **CBraMod pretrained** checkpoint downloaded from Hugging Face; original personal 19-channel EEG data; real learning or performance; 8GB GPU performance; XGBoost optional dependency in an actual experiment; official EEGNet; cross-dataset external validation. Real study outcomes are UNKNOWN.

Reproduce the smoke test: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/smoke_test.py`.

This package intentionally includes no test-result numbers, no demonstration patient predictions, and no randomly initialized checkpoint that could be confused with genuine pretrained weights.
