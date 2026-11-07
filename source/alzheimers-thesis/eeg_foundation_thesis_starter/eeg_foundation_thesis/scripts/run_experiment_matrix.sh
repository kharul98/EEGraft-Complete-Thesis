#!/bin/sh
# Run after all dataset checks. Edit MANIFEST below first.
set -eu
MANIFEST="${1:?Usage: sh scripts/run_experiment_matrix.sh /path/to/verified_manifest.csv}"
python scripts/fetch_weights.py
python scripts/inspect_manifest.py --manifest "$MANIFEST"
python scripts/run_classical.py --manifest "$MANIFEST" --model svm
python scripts/run_classical.py --manifest "$MANIFEST" --model random_forest
python scripts/run_classical.py --manifest "$MANIFEST" --model xgboost
python scripts/train.py --manifest "$MANIFEST" --run-name eegnet_like --model eegnet_like --train-mode scratch --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name pretrained_linear --model linear --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name pretrained_mlp --model mlp --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name proposed_fusion --model fusion --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name without_skip --model no_skip --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name high_only --model high_only --train-mode frozen --robustness --batch-size 8
python scripts/train.py --manifest "$MANIFEST" --run-name low_only --model low_only --train-mode frozen --robustness --batch-size 8
# More expensive controlled experiments, conduct AFTER initial matrix:
# python scripts/train.py --manifest "$MANIFEST" --run-name scratch_fusion --model fusion --train-mode scratch --batch-size 8
# python scripts/train.py --manifest "$MANIFEST" --run-name last2_fusion --model fusion --train-mode last2 --batch-size 8
