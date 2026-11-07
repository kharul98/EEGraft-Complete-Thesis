"""Inspect a real manifest's channels, recording durations and leakage-free partitions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CHANNELS, assign_splits, load_manifest, read_raw_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview-files", type=int, default=5)
    args = p.parse_args()
    df = load_manifest(args.manifest)
    print("Recordings:", len(df), "Subject count:", df.subject_id.nunique())
    print("Recording class counts:", df.label.value_counts().to_dict())
    print("SUBJECT class counts:", df.drop_duplicates("subject_id").label.value_counts().to_dict())
    print("fs_hz:", df.fs_hz.value_counts().to_dict(), "units:", df.unit.value_counts().to_dict())
    print("Model channel order:", CHANNELS)
    for _, row in df.head(args.preview_files).iterrows():
        x = read_raw_csv(row)
        print(Path(row.path).name, "-> 200Hz shape:", x.shape,
              "approx_seconds:", x.shape[1] / 200,
              "max_uV_after_processing:", round(float(np.abs(x).max() * 100), 2))
    split = assign_splits(df, seed=args.seed)
    print("Split class counts BY SUBJECT:")
    print(split.drop_duplicates("subject_id").groupby(["split", "label"]).size().to_string())
