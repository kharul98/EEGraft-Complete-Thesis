import hashlib
import json
from pathlib import Path

import pandas as pd

D = Path.home() / "research/caueeg"
M = D / "manifests"
W = Path.home() / "research/eegraft-caueeg"

if (D / "derived/zero_shot_v1/test_windows.csv").exists():
    raise RuntimeError(
        "Test predictions already exist. Cohort is locked; "
        "do not rebuild it after viewing test results."
    )

annotation = pd.read_csv(
    M / "recordings_audit.csv",
    dtype={"serial": str}
)
qc = pd.read_csv(M / "edf_qc.csv", dtype={"serial": str})

df = annotation.merge(
    qc[["serial", "status", "fs_hz", "duration_s"]],
    on="serial",
    validate="one_to_one"
)

def strict_label(symptoms):
    tokens = set(str(symptoms).lower().split("|"))
    forbidden = {"mci", "normal", "abnormal"}

    if {"ad", "dementia"}.issubset(tokens) and not tokens & forbidden:
        return "AD"

    if "normal" in tokens and not tokens & {
        "ad", "dementia", "mci", "abnormal"
    }:
        return "CN"

    return None

df["label"] = df.symptom_tokens.map(strict_label)

eligible = df[
    df.split.isin(["train", "validation", "test"])
    & df.label.notna()
    & (df.status == "ok")
    & df.edf_exists
].copy()

if eligible.empty:
    raise RuntimeError("No eligible AD/CN recordings; review audit results.")

patient_map = M / "patient_map.csv"
verified = patient_map.is_file()

if verified:
    pm = pd.read_csv(patient_map, dtype=str)

    if not {"serial", "patient_id"}.issubset(pm.columns):
        raise ValueError("Patient map requires serial,patient_id")

    pm = pm[["serial", "patient_id"]].copy()
    pm["serial"] = pm.serial.str.strip().str.zfill(5)
    pm["patient_id"] = pm.patient_id.fillna("").str.strip()

    if pm.serial.duplicated().any():
        raise ValueError("Duplicate serial in patient map")

    eligible = eligible.merge(pm, on="serial", how="left", validate="one_to_one")
    eligible["patient_id"] = eligible.patient_id.fillna("").str.strip()

    if (eligible.patient_id == "").any():
        raise ValueError(
            "Patient mapping is incomplete for eligible recordings. "
            "Do not guess missing patient IDs."
        )

    if (eligible.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient crosses official train/validation/test splits")

    if (eligible.groupby("patient_id").label.nunique() > 1).any():
        raise ValueError("Conflicting labels for the same patient")

    # Prespecified primary analysis: one recording per patient.
    eligible = eligible.sort_values(["patient_id", "serial"])
    eligible["primary"] = (
        eligible.groupby("patient_id").cumcount() == 0
    )
else:
    eligible["patient_id"] = ""
    eligible["primary"] = True

eligible = eligible.sort_values(["split", "serial"])

columns = [
    "serial", "patient_id", "label", "split", "primary",
    "edf_path", "age", "symptom_tokens", "fs_hz", "duration_s"
]

cohort_path = M / "cohort.csv"
eligible[columns].to_csv(cohort_path, index=False)

primary = eligible[eligible.primary]
counts = (
    primary.groupby(["split", "label"])
    .size()
    .rename("count")
    .reset_index()
)

summary = {
    "patient_ids_verified": verified,
    "analysis_unit": "patient" if verified else "recording_exploratory",
    "n_eligible_recordings": len(eligible),
    "n_primary_cases": len(primary),
    "primary_counts": counts.to_dict(orient="records"),
    "patient_map_sha256": (
        hashlib.sha256(patient_map.read_bytes()).hexdigest()
        if verified else None
    ),
    "cohort_sha256": hashlib.sha256(
        cohort_path.read_bytes()
    ).hexdigest(),
}

(M / "cohort_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print(json.dumps(summary, indent=2))
if not verified:
    print(
        "\nWARNING: Patient IDs are unverified. "
        "Only exploratory recording-level results are permitted."
    )
