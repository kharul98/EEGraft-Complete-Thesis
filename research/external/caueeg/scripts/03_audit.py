import json
from collections import Counter
from pathlib import Path

import pandas as pd

D = Path.home() / "research/caueeg"
M = D / "manifests"

matches = list((D / "raw").rglob("annotation.json"))
if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one annotation.json, found {len(matches)}"
    )

root = matches[0].parent
benchmark = root / "dementia-no-overlap.json"
if not benchmark.is_file():
    raise FileNotFoundError(
        "Official dementia-no-overlap.json is required. "
        "Do not silently substitute the overlapping benchmark."
    )

annotation = json.loads(matches[0].read_text())
task = json.loads(benchmark.read_text())

if not isinstance(annotation.get("data"), list):
    raise ValueError("Unexpected annotation.json structure")

def serial_of(row):
    s = str(row["serial"]).strip()
    if not s.isdigit():
        raise ValueError(f"Invalid recording serial: {s}")
    return s.zfill(5)

splits = {}
for source_key, split_name in [
    ("train_split", "train"),
    ("validation_split", "validation"),
    ("test_split", "test"),
]:
    entries = task.get(source_key)
    if not isinstance(entries, list):
        raise ValueError(f"Missing official {source_key}")
    for entry in entries:
        s = serial_of(entry)
        if s in splits:
            raise ValueError(f"Recording crosses splits: {s}")
        splits[s] = split_name

rows = []
seen = set()
symptom_counts = Counter()

for entry in annotation["data"]:
    s = serial_of(entry)
    if s in seen:
        raise ValueError(f"Duplicate annotation serial: {s}")
    seen.add(s)

    symptoms = entry.get("symptom", [])
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if not isinstance(symptoms, list):
        raise ValueError(f"Unexpected symptom type for {s}")

    tokens = sorted({
        str(x).strip().lower()
        for x in symptoms
        if str(x).strip()
    })
    symptom_counts.update(tokens)

    edf = root / "signal" / "edf" / f"{s}.edf"
    event = root / "event" / f"{s}.json"

    rows.append({
        "serial": s,
        "age": entry.get("age"),
        "symptom_tokens": "|".join(tokens),
        "split": splits.get(s, "not_in_benchmark"),
        "edf_path": str(edf),
        "edf_exists": edf.is_file(),
        "event_exists": event.is_file(),
    })

unknown = set(splits) - seen
if unknown:
    raise ValueError(
        f"Official split references {len(unknown)} unknown recordings"
    )

df = pd.DataFrame(rows).sort_values("serial")
M.mkdir(parents=True, exist_ok=True)
df.to_csv(M / "recordings_audit.csv", index=False)
(M / "dataset_root.txt").write_text(str(root) + "\n")

template = pd.DataFrame({
    "serial": df.serial,
    "patient_id": [""] * len(df)
})
template.to_csv(M / "patient_map_TEMPLATE.csv", index=False)

summary = {
    "n_annotation_recordings": len(df),
    "n_edf_present": int(df.edf_exists.sum()),
    "n_edf_missing": int((~df.edf_exists).sum()),
    "official_split_counts": {
        str(k): int(v)
        for k, v in df.split.value_counts().items()
    },
    "symptom_token_counts": dict(symptom_counts),
    "annotation_keys": sorted(annotation.keys()),
    "annotation_entry_keys": sorted(annotation["data"][0].keys()),
    "patient_identity_verified": False,
    "note": "Recording serial is not automatically a patient ID."
}

(M / "annotation_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print(json.dumps(summary, indent=2))
print("\nPrivate recording audit:", M / "recordings_audit.csv")
print("Patient-linkage template:", M / "patient_map_TEMPLATE.csv")
