"""Build a DRAFT manifest from labelled folders. Manually verify subject identities.

Usage:
  python scripts/make_manifest.py --class-dir healthy=/absolute/folder/healthy \
      --class-dir severe=/absolute/folder/severe --fs-hz 200 --unit uV \
      --output my_manifest.csv

IMPORTANT: Every CSV must represent one subject, OR its basename must identify that
subject. If multiple files/epochs belong to one person, edit subject_id accordingly.
"""
import argparse
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--class-dir", action="append", required=True,
                   help="label=/absolute/or/relative/folder (repeat for each class)")
    p.add_argument("--fs-hz", type=float, required=True, help="Actual recording sample rate")
    p.add_argument("--unit", choices=["uV", "mV", "V"], required=True,
                   help="Actual units in the CSV (NOT normalized values)")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for item in args.class_dir:
        if "=" not in item:
            raise SystemExit("--class-dir value must be label=folder")
        label, folder = item.split("=", 1)
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        files = sorted(folder.rglob("*.csv"))
        print(label, "CSV files:", len(files))
        for file in files:
            rows.append({"path": str(file), "subject_id": file.stem,
                         "label": label, "fs_hz": args.fs_hz, "unit": args.unit})
    if not rows:
        raise SystemExit("No CSV files found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("DRAFT manifest created:", out)
    print("REVIEW subject_id carefully. Repeated recordings for one person MUST share a subject_id.")
    print("DO NOT interpret filename stem as verified patient identity without checking the dataset notes.")
