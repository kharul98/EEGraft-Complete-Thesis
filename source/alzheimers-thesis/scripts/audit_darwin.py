from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
import sklearn
import scipy
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = ROOT / "docs" / "audit"

RAW.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("========================================")
print("DARWIN DATASET FEASIBILITY AUDIT")
print("========================================")

dataset = fetch_ucirepo(id=732)

print("Dataset fetched successfully.")

metadata = dataset.metadata
print("Dataset name:", getattr(metadata, "name", None))

X = dataset.data.features.copy()
y_df = dataset.data.targets.copy()

ids = None
try:
    if dataset.data.ids is not None:
        ids = dataset.data.ids.copy()
except Exception:
    ids = None

print()
print("----- BASIC STRUCTURE -----")
print("Feature table shape:", X.shape)
print("Target table shape:", y_df.shape)

if ids is not None:
    print("ID table shape:", ids.shape)
else:
    print("ID table: not separately provided by API")

print()
print("Feature columns:")
print(list(X.columns))

print()
print("Target columns:")
print(list(y_df.columns))

if y_df.shape[1] != 1:
    print("WARNING: expected one target column but found:", y_df.shape[1])

target_col = y_df.columns[0]
y = y_df[target_col]

print()
print("----- TARGET AUDIT -----")
print("Target name:", target_col)
print("Number of unique target values:", y.nunique(dropna=False))
print("Target counts:")
print(y.value_counts(dropna=False).to_string())

print()
print("----- MISSING VALUES -----")
print("Missing feature values:", int(X.isna().sum().sum()))
print("Missing target values:", int(y.isna().sum()))

if ids is not None:
    print("Missing ID values:", int(ids.isna().sum().sum()))

print()
print("----- DUPLICATION -----")
print("Exact duplicate feature rows:", int(X.duplicated().sum()))
print(
    "Exact duplicate feature+target rows:",
    int(pd.concat([X, y_df], axis=1).duplicated().sum()),
)

if ids is not None:
    print("Duplicate ID rows:", int(ids.duplicated().sum()))
    for col in ids.columns:
        print(
            f"ID column {col}: "
            f"{ids[col].nunique(dropna=False)} unique / {len(ids)} rows"
        )

possible_id_columns = [
    c for c in X.columns
    if str(c).strip().lower() in {
        "id", "subject", "subject_id", "participant",
        "participant_id", "patient", "patient_id"
    }
]

print("Possible ID columns inside feature table:", possible_id_columns)

for col in possible_id_columns:
    print(
        f"{col}: {X[col].nunique(dropna=False)} unique / {len(X)} rows"
    )

print()
print("----- FEATURE TYPES -----")
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric feature columns:", len(numeric_cols))
print("Non-numeric feature columns:", len(non_numeric_cols))
print("Non-numeric names:", non_numeric_cols)

print()
print("----- CONSTANT FEATURES -----")
constant_cols = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]

print("Constant columns count:", len(constant_cols))
print("Constant columns:", constant_cols)

print()
print("----- TASK NUMBER DETECTION -----")
task_numbers = set()

for col in X.columns:
    match = re.search(r"(\d+)$", str(col))
    if match:
        task_numbers.add(int(match.group(1)))

print("Detected task numbers:", sorted(task_numbers))
print("Detected number of tasks:", len(task_numbers))

print()
print("----- TARGET-LIKE FEATURE NAMES -----")
suspicious = [
    c for c in X.columns
    if any(
        token in str(c).lower()
        for token in [
            "class",
            "label",
            "target",
            "diagnosis",
            "disease",
            "alzheimer",
            "status"
        ]
    )
]

print("Suspicious predictor names:", suspicious)

features_path = RAW / "darwin_features.csv"
target_path = RAW / "darwin_target.csv"

X.to_csv(features_path, index=False)
y_df.to_csv(target_path, index=False)

saved = {
    "features_file": str(features_path.relative_to(ROOT)),
    "target_file": str(target_path.relative_to(ROOT)),
    "features_sha256": sha256_file(features_path),
    "target_sha256": sha256_file(target_path),
    "n_rows": int(len(X)),
    "n_feature_columns": int(X.shape[1]),
    "n_target_columns": int(y_df.shape[1]),
    "target_counts": {
        str(k): int(v)
        for k, v in y.value_counts(dropna=False).items()
    },
    "missing_features": int(X.isna().sum().sum()),
    "missing_target": int(y.isna().sum()),
    "duplicate_feature_rows": int(X.duplicated().sum()),
    "detected_tasks": sorted(task_numbers),
    "detected_task_count": len(task_numbers),
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
}

audit_json = AUDIT / "darwin_feasibility_audit.json"

with audit_json.open("w", encoding="utf-8") as f:
    json.dump(saved, f, indent=2)

print()
print("----- SAVED ARTIFACTS -----")
print(features_path)
print(target_path)
print(audit_json)

print()
print("Feature SHA256:", saved["features_sha256"])
print("Target SHA256:", saved["target_sha256"])

print()
print("========================================")
print("AUDIT SUMMARY")
print("========================================")
print("Participants/rows:", len(X))
print("Feature columns:", X.shape[1])
print("Target counts:", saved["target_counts"])
print("Missing features:", saved["missing_features"])
print("Missing targets:", saved["missing_target"])
print("Duplicate feature rows:", saved["duplicate_feature_rows"])
print("Detected tasks:", saved["detected_task_count"])
print("========================================")
