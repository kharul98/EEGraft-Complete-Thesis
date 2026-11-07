#!/usr/bin/env python3
"""M7B: read-only source audit for original AD/FTD thesis findings."""

from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import re

ROOT = Path.cwd()
BASE = ROOT / "reports/thesis_final"
OUT = BASE / "milestone_7b_primary_source_reconciliation_v1"

OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "4C evidence summary":
        "milestone_4c_reproducibility_v1/"
        "THESIS_EVIDENCE_SUMMARY.md",

    "4C claim audit":
        "milestone_4c_reproducibility_v1/"
        "claim_audit.csv",

    "4C evidence master":
        "milestone_4c_reproducibility_v1/"
        "evidence_master.csv",

    "AD temporal summary":
        "duration_locked_v1/"
        "block_prediction_stability_summary.csv",

    "AD held-out predictions":
        "duration_locked_v1/"
        "held_out_predictions.csv",

    "AD duration features":
        "duration_locked_v1/"
        "duration_features.csv",

    "FTD temporal summary":
        "ftd_extension_v1/locked_study/"
        "temporal_repeat_results.csv",

    "FTD minute metrics":
        "ftd_extension_v1/locked_study/"
        "minute_wise_metrics.csv",

    "FTD held-out predictions":
        "ftd_extension_v1/locked_study/"
        "held_out_duration_predictions.csv",

    "FTD duration features":
        "ftd_extension_v1/locked_study/"
        "duration_features.csv",

    "Calibration metrics":
        "milestone_1b_calibration_v1/"
        "calibration_metrics_by_repeat_minute.csv",

    "Calibration predictions":
        "milestone_1b_calibration_v1/"
        "calibrated_held_out_block_predictions.csv",
}

NEEDLES = (
    "0.7348", "73.48",
    "0.7372", "73.72",
    "44.35", "33.66",
    "8.70", "13.17",
    "0.0883", "0.0423",
    "holm", "stable",
    "ablation", "calibrat",
    "condition",
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def freeze(path, content):
    if path.exists():

        if path.read_text(
            encoding="utf-8"
        ) != content:

            raise RuntimeError(
                "Frozen output changed: " + str(path)
            )
    else:
        path.write_text(
            content,
            encoding="utf-8"
        )


def short(value, limit=96):
    text = str(value).replace(
        "\r", " "
    ).replace(
        "\n", " "
    ).strip()

    if len(text) > limit:
        return text[:limit - 3] + "..."

    return text


def inspect_csv(path):

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        if (
            not columns
            or len(columns) != len(set(columns))
        ):
            raise RuntimeError(
                "Missing or duplicate CSV columns: "
                + str(path)
            )

        count = 0
        examples = []
        subjects = set()
        distinct = defaultdict(set)

        watched = [
            col
            for col in columns
            if re.search(
                r"true|actual|pred|label|class|"
                r"diagnos|minute|block|duration|"
                r"repeat|state|status|bacc|fold",
                col,
                re.I
            )
        ]

        for row in reader:

            count += 1

            if count <= 2:
                examples.append(row)

            for col in columns:

                if re.search(
                    r"(participant|subject).*id|"
                    r"^(subject|participant)$",
                    col,
                    re.I
                ):
                    subjects.add(row.get(col, ""))

            for col in watched:

                if len(distinct[col]) < 20:
                    distinct[col].add(
                        row.get(col, "")
                    )

    return {
        "row_count": count,
        "columns": columns,
        "examples": examples,

        "watched_unique_values": {
            col: sorted(values)[:14]
            for col, values in distinct.items()
            if len(values) <= 14
        },

        "n_distinct_subject_ids_when_named":
            len(subjects),
    }


# ========================================
# 1. LOCATE AND HASH PRIMARY RESULTS
# ========================================

print(
    "===== MILESTONE 7B: "
    "ORIGINAL AD/FTD SOURCE MAP =====",
    flush=True
)

entries = []
results = {}
missing = []

for label, relative in SOURCES.items():

    path = BASE / relative

    if not path.is_file():

        missing.append(
            str(path.relative_to(ROOT))
        )

        continue

    item = {
        "label": label,
        "path": str(
            path.relative_to(ROOT)
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

    if path.suffix.lower() == ".csv":
        item.update(
            inspect_csv(path)
        )

    entries.append(item)
    results[label] = item

if missing:

    print(
        "SOURCE FILES NOT FOUND:"
    )

    for path in missing:
        print(" ", path)

    raise RuntimeError(
        "Primary source inventory incomplete."
    )

print(
    f"PASS: {len(entries)} primary evidence "
    "files located and SHA256 recorded."
)


# ========================================
# 2. INSPECT SOURCE SCHEMAS
# ========================================

print(
    "\n===== ORIGINAL RESULTS: "
    "SOURCE SCHEMAS ====="
)

for item in entries:

    if "row_count" not in item:
        continue

    print(
        f"\n{item['label']} | "
        f"rows={item['row_count']} | "
        "distinct subject IDs (if available)="
        f"{item['n_distinct_subject_ids_when_named']}"
    )

    print(
        "  Columns:",
        " | ".join(item["columns"])
    )

    for index, row in enumerate(
        item["examples"],
        1
    ):

        watched = [
            k for k in item["columns"]

            if re.search(
                r"subject|particip|true|actual|"
                r"pred|label|class|diagnos|"
                r"minute|block|duration|repeat|"
                r"state|status|bacc|fold",
                k,
                re.I
            )
        ]

        preview_columns = (
            watched or item["columns"]
        )[:8]

        preview = "; ".join(
            f"{key}="
            f"{short(row.get(key, ''), 48)}"

            for key in preview_columns
        )

        print(
            f"  Example {index}:",
            preview
        )

    unique_values = list(
        item[
            "watched_unique_values"
        ].items()
    )

    for column, values in unique_values[:9]:

        print(
            "  Distinct",
            column + ":",

            ", ".join(
                short(value, 30)
                for value in values
            )
        )


# ========================================
# 3. AUDIT REPRODUCIBILITY CLAIM RECORDS
# ========================================

print(
    "\n===== 4C CLAIM AUDIT AND "
    "EVIDENCE MASTER EXCERPTS ====="
)

for label in (
    "4C claim audit",
    "4C evidence master"
):

    item = results[label]

    print(
        f"\n--- {label}: "
        f"{item['row_count']} rows ---"
    )

    path = ROOT / item["path"]

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for number, row in enumerate(
            reader,
            1
        ):

            if number > 35:

                print(
                    "... further rows omitted. "
                    "Original CSV retained."
                )

                break

            values = [
                f"{key}={short(value, 110)}"

                for key, value in row.items()

                if (
                    value is not None
                    and str(value).strip()
                )
            ]

            print(
                f"  Row {number}:",
                " ; ".join(values)[:440]
            )


# ========================================
# 4. CHECK HUMAN-READABLE PRIMARY SUMMARY
# ========================================

print(
    "\n===== 4C HUMAN-READABLE "
    "SUMMARY EXCERPTS ====="
)

summary_file = (
    BASE / SOURCES["4C evidence summary"]
)

lines = summary_file.read_text(
    encoding="utf-8"
).splitlines()

matches = [
    (number, text.strip())

    for number, text in enumerate(
        lines,
        1
    )

    if any(
        token in text.lower()
        for token in NEEDLES
    )
]

for number, text in matches[:32]:

    print(
        f"  L{number}:",
        short(text, 210)
    )

print(
    "Matching lines:",
    len(matches),

    "| showing:",
    min(len(matches), 32)
)


# ========================================
# 5. SAVE SOURCE INVENTORY
# ========================================

record = {
    "milestone": "7B",

    "purpose":
        "Original AD/FTD primary source "
        "reconciliation preflight",

    "files": entries,

    "primary_claim_values_independently_recomputed":
        False,

    "no_models_trained": True,
}

freeze(
    OUT / "primary_source_inventory.json",

    json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
        sort_keys=True
    ) + "\n"
)


# ========================================
# 6. COMPLETE
# ========================================

print(
    "\n===== MILESTONE 7B COMPLETE ====="
)

print(
    "Files verified:",
    len(entries)
)

print(
    "Source map:",
    OUT / "primary_source_inventory.json"
)

print(
    "Original AD/FTD metrics have NOT YET "
    "been independently recalculated "
    "at this step."
)

print(
    "No existing files changed; "
    "no EEG processing or model training."
)

