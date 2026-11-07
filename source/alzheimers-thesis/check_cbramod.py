from pathlib import Path
import sys
import csv

import torch
import mne

root = Path.cwd()
repo = root / "third_party/CBraMod"
weights = root / "pretrained_weights/cbramod_official.pth"

sys.path.insert(0, str(repo.resolve()))

from models.cbramod import CBraMod

print("===== CBRAMOD PRETRAINED READINESS =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if not weights.exists():
    raise FileNotFoundError("Pretrained checkpoint missing")

model = CBraMod()

state = torch.load(
    weights,
    map_location="cpu",
    weights_only=True
)

# Verify exact compatibility with official weights.
model.load_state_dict(state, strict=True)

print("Pretrained weights: LOADED")

model.proj_out = torch.nn.Identity()
model = model.to(device)
model.eval()

parameters = sum(
    p.numel() for p in model.parameters()
)
print("Model parameters:", parameters)

# A small synthetic input, not patient EEG.
# Shape: batch, channels, 1-second patches, samples.
sample = torch.randn(
    1, 19, 10, 200, device=device
) * 0.1

with torch.inference_mode():
    output = model(sample)

print("Input shape:", tuple(sample.shape))
print("Embedding shape:", tuple(output.shape))
print("Finite output:", bool(torch.isfinite(output).all()))

assert output.shape == (1, 19, 10, 200)
assert torch.isfinite(output).all()

print("\n===== DATASET CHANNEL AUDIT =====")

manifest = root / "reports/openneuro_5min_manifest_v2.csv"

with manifest.open() as f:
    rows = [
        row for row in csv.DictReader(f)
        if row["status"] == "CANDIDATE"
        and row["group"] in ("A", "C")
    ]

assert len(rows) == 46

raw = mne.io.read_raw_eeglab(
    rows[0]["file"],
    preload=False,
    verbose="ERROR"
)

picks = mne.pick_types(raw.info, eeg=True, exclude=[])
channels = [raw.ch_names[p] for p in picks]

print("Dataset EEG channels:", channels)

expected = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4",
    "F8", "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

aliases = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6"
}

normalized = [
    aliases.get(name, name)
    for name in channels
]

missing = sorted(set(expected) - set(normalized))
extra = sorted(set(normalized) - set(expected))

print("Expected pretrained channels:", expected)
print("Missing channels:", missing)
print("Unexpected channels:", extra)

if not missing and not extra and len(set(normalized)) == 19:
    print("Channel mapping: COMPATIBLE")
else:
    print("Channel mapping: NEEDS REVIEW")

print("\n===== CBRAMOD READINESS COMPLETE =====")
print("No patient EEG data modified.")
