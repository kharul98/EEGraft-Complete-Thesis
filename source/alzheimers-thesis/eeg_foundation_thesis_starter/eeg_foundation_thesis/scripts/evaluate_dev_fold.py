"""Evaluate an outer development fold after inner-validation model selection."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data import build_windows, load_manifest
from src.model import FoundationClassifier, sha256_file
from scripts.train import get_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--manifest', required=True)
parser.add_argument('--run-name', required=True)
parser.add_argument('--batch-size', type=int, default=8)

args = parser.parse_args()

root = Path(__file__).resolve().parents[1]

canonical = pd.read_csv(
    root / 'manifests/ds004504_ad_cn_preprocessed.csv',
    dtype={'subject_id': str}
)

reserved = set(
    canonical.loc[canonical.split == 'test', 'subject_id']
)

expected_dev = set(
    canonical.loc[
        canonical.split.isin(['train', 'val']),
        'subject_id'
    ]
)

df = load_manifest(args.manifest)

assert len(reserved) == 13
assert len(expected_dev) == 52
assert len(df) == 52
assert set(df.subject_id) == expected_dev
assert not reserved.intersection(df.subject_id)
assert set(df['split']) == {'train', 'val', 'test'}

outer_ids = set(
    df.loc[df.split == 'test', 'subject_id']
)

assert 10 <= len(outer_ids) <= 11

workdir = root / 'results' / args.run_name

out_metrics = workdir / 'outer_dev_metrics.json'
out_predictions = (
    workdir / 'outer_dev_subject_predictions.csv'
)

if out_metrics.exists() or out_predictions.exists():
    raise SystemExit(
        'Outer-fold results already exist. Refusing overwrite.'
    )

report = json.loads(
    (workdir / 'metrics.json').read_text()
)

if (
    report.get('test_not_evaluated') is not True
    or report.get('test') is not None
):
    raise RuntimeError(
        'Training was not validation-only.'
    )

if report.get('config', {}).get('validation_only') is not True:
    raise RuntimeError(
        'Missing validation-only training flag.'
    )

allowed = {'fusion', 'no_skip', 'high_only', 'low_only'}

if report.get('model') not in allowed:
    raise ValueError('Unexpected decoder type.')

ckpt = torch.load(
    workdir / 'best.pt',
    map_location='cpu',
    weights_only=True
)

if str(Path(args.manifest).resolve()) != ckpt['manifest']:
    raise RuntimeError(
        'Checkpoint was trained using a different manifest.'
    )

if (
    ckpt['model_name'] != report['model']
    or ckpt['train_mode'] != 'frozen'
):
    raise RuntimeError(
        'Wrong architecture or encoder-freezing regime.'
    )

weights = root / 'checkpoints/cbramod_official.pth'

if ckpt['checkpoint_sha256'] != sha256_file(weights):
    raise RuntimeError(
        'Official pretrained checkpoint fingerprint mismatch.'
    )

classes = list(ckpt['classes'])

if classes != sorted(df.label.unique().tolist()):
    raise RuntimeError('Class mapping mismatch.')

model = FoundationClassifier(
    num_classes=len(classes),
    checkpoint=str(weights),
    decoder=ckpt['model_name'],
    freeze_encoder=True
)

model.load_state_dict(
    ckpt['model_state'],
    strict=True
)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model.to(device).eval()

config = report['config']

store = build_windows(
    df,
    'test',
    classes=classes,
    seconds=int(config['seconds']),
    max_per_subject=int(config['max_windows_per_subject']),
    apply_car=bool(config['car']),
    high_hz=float(config['high_hz']),
    artifact_limit_uv=config['artifact_limit_uv']
)

metrics, (
    subject_ids,
    targets,
    probabilities
) = get_metrics(
    model,
    store,
    device,
    classes,
    batch_size=args.batch_size
)

if set(subject_ids) != outer_ids:
    raise RuntimeError(
        'Outer patient IDs do not match inference output.'
    )

predictions = pd.DataFrame({
    'subject_id': subject_ids,
    'true_label': [
        classes[int(v)] for v in targets
    ],
    'predicted_label': [
        classes[int(v)]
        for v in probabilities.argmax(axis=1)
    ]
})

for i, cls in enumerate(classes):
    predictions[f'p_{cls}'] = probabilities[:, i]

predictions.to_csv(
    out_predictions,
    index=False
)

out_metrics.write_text(
    json.dumps({
        'run_name': args.run_name,
        'n_outer_subjects': len(subject_ids),
        'outer_development_metrics': metrics,
        'original_final_test_evaluated': False
    }, indent=2)
)

print('\n========== OUTER DEVELOPMENT EVALUATION ==========')
print('Experiment:', args.run_name)
print('Patients:', len(subject_ids))
print('Balanced accuracy:', round(metrics['balanced_accuracy'], 4))
print('Macro-F1:', round(metrics['macro_f1'], 4))
print('Original 13 final-test patients excluded.')
