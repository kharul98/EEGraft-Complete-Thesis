#!/usr/bin/env python3
"""Milestone 3A: metadata, file, and event inventory for matched EO/photic EEG."""

from pathlib import Path
from collections import Counter
from urllib.request import Request, urlopen
from urllib.parse import urlparse, unquote
import csv
import hashlib
import io
import json
import re
import shutil


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"

OUT = BASE / "milestone_3a_cross_condition_preflight_v1"
OUT.mkdir(parents=True, exist_ok=True)

EC_MANIFEST = (
    ROOT / "reports" / "openneuro_5min_manifest_v2.csv"
)

EC_METADATA = (
    ROOT / "data" / "eeg"
    / "on004504_v1.0.0" / "participants.tsv"
)

RELEASE_URL = (
    "https://data.nemar.org/on006036/v1.0.0"
)

GH_RAW = (
    "https://raw.githubusercontent.com/"
    "OpenNeuroDatasets/ds006036/main"
)

EO_ROOTS = [
    ROOT / "data/eeg/on006036_v1.0.0",
    ROOT / "data/eeg/on006036",
    ROOT / "data/eeg/ds006036",
    ROOT / "data/on006036",
    ROOT / "on006036",
]


# ==========================================
# FILE UTILITIES
# ==========================================

def read_table(path, sep):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(
                f,
                delimiter=sep
            )
        )


def save_table(name, rows):

    if not rows:
        return

    cols = list(
        dict.fromkeys(
            k
            for row in rows
            for k in row
        )
    )

    with (OUT / name).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=cols
        )

        writer.writeheader()
        writer.writerows(rows)


def get_small(url, limit=12000000):

    request = Request(
        url,
        headers={
            "User-Agent":
                "research-metadata-audit/1.0"
        }
    )

    with urlopen(
        request,
        timeout=22
    ) as response:

        data = response.read(
            limit + 1
        )

    if len(data) > limit:
        raise RuntimeError(
            "Metadata file too large: " + url
        )

    return data


def unique(rows, field):

    output = {}

    for row in rows:

        sid = row[field].strip()

        if sid in output:
            raise RuntimeError(
                "Duplicate subject: " + sid
            )

        output[sid] = row

    return output


# ==========================================
# 1. VERIFY THE ORIGINAL COHORT
# ==========================================

print(
    "===== MILESTONE 3A: "
    "CROSS-CONDITION DATASET PREFLIGHT =====",
    flush=True
)

if (
    not EC_MANIFEST.is_file()
    or not EC_METADATA.is_file()
):
    raise FileNotFoundError(
        "Missing frozen eyes-closed "
        "manifest or participants.tsv"
    )

manifest = unique(
    read_table(
        EC_MANIFEST,
        ","
    ),
    "subject"
)

ec_meta = unique(
    read_table(
        EC_METADATA,
        "\t"
    ),
    "participant_id"
)

if (
    len(manifest) != 88
    or len(ec_meta) != 88
):
    raise RuntimeError(
        "Eyes-closed cohort is not "
        "the original 88 participants"
    )

selected = {
    sid: row
    for sid, row in manifest.items()
    if (
        row["status"] == "CANDIDATE"
        and row["group"] in (
            "A", "F", "C"
        )
    )
}

counts = Counter(
    row["group"]
    for row in selected.values()
)

if counts != {
    "A": 23,
    "F": 18,
    "C": 23
}:
    raise RuntimeError(
        "Frozen selected cohort changed: "
        + str(counts)
    )

print(
    "Eyes-closed source: 88 participants; "
    "selected 23 AD, 18 FTD, 23 HC"
)


# ==========================================
# 2. FIND EYES-OPEN PARTICIPANT METADATA
# ==========================================

local_roots = [
    p
    for p in EO_ROOTS
    if p.is_dir()
]

local_tsv = [
    p / "participants.tsv"
    for p in local_roots
    if (
        p / "participants.tsv"
    ).is_file()
]

metadata_origin = ""

if local_tsv:

    eo_path = local_tsv[0]
    metadata_origin = "local dataset"

else:

    eo_path = (
        OUT / "on006036_v1.0.0_participants.tsv"
    )

    if eo_path.exists():

        metadata_origin = (
            "cached NEMAR/GitHub metadata "
            "(see source record)"
        )

    else:

        failures = []

        sources = [
            (
                RELEASE_URL + "/participants.tsv",
                "NEMAR v1.0.0"
            ),
            (
                GH_RAW + "/participants.tsv",
                "GitHub main (unpinned fallback)"
            )
        ]

        for url, origin in sources:

            try:

                data = get_small(
                    url,
                    limit=100000
                )

                first = (
                    data.decode("utf-8-sig")
                    .splitlines()[0]
                )

                if (
                    "participant_id" not in first
                    or "Group" not in first
                ):
                    raise RuntimeError(
                        "Unexpected participants.tsv header"
                    )

                eo_path.write_bytes(data)
                metadata_origin = origin

                break

            except Exception as exc:

                failures.append(
                    f"{url}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not metadata_origin:

            print(
                "Failed to obtain EO participants.tsv:"
            )

            for item in failures:
                print(
                    "  " + item[:250]
                )

            raise RuntimeError(
                "Download on006036 participants.tsv, "
                "then rerun"
            )


# ==========================================
# 3. VERIFY PAIRED SUBJECT IDENTITIES
# ==========================================

eo_meta = unique(
    read_table(
        eo_path,
        "\t"
    ),
    "participant_id"
)

if (
    set(eo_meta) != set(manifest)
    or len(eo_meta) != 88
):
    raise RuntimeError(
        "Eyes-open and eyes-closed participant "
        "IDs do not match exactly"
    )

label_disagreements = [
    sid
    for sid in manifest
    if (
        eo_meta[sid]["Group"].strip()
        != manifest[sid]["group"]
    )
]

if label_disagreements:

    raise RuntimeError(
        "Diagnostic-label mismatch: "
        + str(label_disagreements)
    )

age_changes = []
mmse_changes = []

for sid in manifest:

    if (
        str(
            eo_meta[sid].get("Age", "")
        ).strip()
        !=
        str(
            ec_meta[sid].get("Age", "")
        ).strip()
    ):
        age_changes.append(sid)

    if (
        str(
            eo_meta[sid].get("MMSE", "")
        ).strip()
        !=
        str(
            ec_meta[sid].get("MMSE", "")
        ).strip()
    ):
        mmse_changes.append(sid)

save_table(
    "paired_participant_inventory.csv",
    [
        {
            "subject": sid,
            "group":
                manifest[sid]["group"],
            "eyes_closed_selected":
                int(sid in selected),
            "in_AD_vs_HC":
                int(
                    sid in selected
                    and manifest[sid]["group"]
                    in ("A", "C")
                ),
            "in_FTD_vs_HC":
                int(
                    sid in selected
                    and manifest[sid]["group"]
                    in ("F", "C")
                ),
            "age_matches":
                int(
                    sid not in age_changes
                ),
            "MMSE_matches":
                int(
                    sid not in mmse_changes
                )
        }
        for sid in sorted(manifest)
    ]
)

print(
    "\n===== PAIRED PARTICIPANT METADATA ====="
)

print(
    "Eyes-open metadata source:",
    metadata_origin
)

print(
    "Metadata file:",
    eo_path
)

print(
    "PASS: all 88 IDs and "
    "disease labels match exactly"
)

print(
    "Age differences:",
    len(age_changes),
    "MMSE differences:",
    len(mmse_changes)
)

if age_changes or mmse_changes:

    print(
        "WARNING: Metadata differences "
        "require inspection before modeling"
    )


# ==========================================
# 4. CHECK LOCAL EYES-OPEN EEG FILES
# ==========================================

local_sets = {}

for root in local_roots:

    for path in root.rglob("*.set"):

        if "derivatives" not in path.parts:
            continue

        match = re.search(
            r"sub-\d{3}",
            str(path)
        )

        if (
            match
            and match.group() in selected
        ):

            local_sets.setdefault(
                match.group(),
                []
            ).append(
                str(path)
            )

local_count = sum(
    bool(
        local_sets.get(sid)
    )
    for sid in selected
)

print(
    "\n===== LOCAL EEG DATA AVAILABILITY ====="
)

print(
    "Existing local EO dataset roots:",
    [
        str(p)
        for p in local_roots
    ] or "none"
)

print(
    "Selected subjects with local "
    "derivative .set files:",
    f"{local_count}/64"
)

free = (
    shutil.disk_usage(ROOT).free
    / (1024 ** 3)
)

print(
    f"Available disk space: "
    f"{free:.2f} GiB"
)

print(
    "NEMAR CLI:",
    shutil.which("nemar")
    or "not installed"
)

print(
    "npm:",
    shutil.which("npm")
    or "not installed"
)

print(
    "datalad:",
    shutil.which("datalad")
    or "not installed"
)

if free < 4:

    print(
        "WARNING: Less than 4 GiB free; "
        "check space before EEG download"
    )

save_table(
    "local_selected_derivative_files.csv",
    [
        {
            "subject": sid,
            "derivative_set": path
        }
        for sid, paths in sorted(
            local_sets.items()
        )
        for path in paths
    ]
)


# ==========================================
# 5. INSPECT PINNED NEMAR FILE INVENTORY
# ==========================================

# The manifest provides a remote file list.
# It does not download full EEG recordings.

manifest_path = (
    OUT / "nemar_v1.0.0_manifest.json"
)

manifest_status = "unavailable"

files = []

if not manifest_path.exists():

    try:

        body = get_small(
            RELEASE_URL + "/manifest.json",
            limit=15000000
        )

        parsed = json.loads(
            body
        )

        if not isinstance(
            parsed,
            (dict, list)
        ):
            raise RuntimeError(
                "Unexpected manifest JSON type"
            )

        manifest_path.write_bytes(
            body
        )

    except Exception as exc:

        print(
            "NEMAR manifest unavailable "
            "(nonfatal):",
            str(exc)[:180]
        )

if manifest_path.exists():

    try:

        parsed = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(parsed, list):

            files = parsed

        elif isinstance(parsed, dict):

            for key in (
                "files",
                "entries",
                "items",
                "data"
            ):

                if isinstance(
                    parsed.get(key),
                    list
                ):

                    files = parsed[key]
                    break

        manifest_status = (
            "readable"
            if files
            else "unrecognized JSON structure"
        )

    except Exception as exc:

        manifest_status = (
            "parse error: "
            + str(exc)[:150]
        )


remote = []

for record in files:

    if isinstance(
        record,
        str
    ):

        rel = record
        url = ""
        nbytes = ""

    elif isinstance(
        record,
        dict
    ):

        rel = next(
            (
                str(record[k])
                for k in (
                    "path",
                    "relative_path",
                    "filepath",
                    "name"
                )
                if record.get(k)
            ),
            ""
        )

        url = str(
            record.get(
                "url",
                ""
            )
        )

        if not rel and url:

            rel = unquote(
                urlparse(url).path
            )

        nbytes = record.get(
            "size_bytes",
            record.get(
                "size",
                ""
            )
        )

    else:
        continue

    match = re.search(
        r"sub-\d{3}",
        rel
    )

    if (
        match
        and match.group() in selected
    ):

        remote.append({
            "subject": match.group(),
            "file": rel,
            "url": url,
            "size_bytes_if_known": nbytes,
            "is_derivative":
                int(
                    "derivatives/" in rel
                ),
            "is_set":
                int(
                    rel.endswith(".set")
                ),
            "is_fdt":
                int(
                    rel.endswith(".fdt")
                ),
            "is_event_tsv":
                int(
                    rel.endswith(
                        "_events.tsv"
                    )
                )
        })


save_table(
    "remote_selected_file_inventory.csv",
    remote
)

print(
    "\n===== PINNED REMOTE FILE INVENTORY ====="
)

print(
    "NEMAR v1.0.0 manifest:",
    manifest_status
)

print(
    "Selected-subject manifest entries:",
    len(remote)
)

print(
    "Selected derivative .set files:",
    sum(
        r["is_set"]
        and r["is_derivative"]
        for r in remote
    )
)

print(
    "Selected derivative .fdt files:",
    sum(
        r["is_fdt"]
        and r["is_derivative"]
        for r in remote
    )
)

print(
    "Selected event sidecars:",
    sum(
        r["is_event_tsv"]
        for r in remote
    )
)


# ==========================================
# 6. INSPECT PHOTOSTIMULATION EVENT FORMAT
# ==========================================

# Only a small events TSV is requested
# for one selected participant per group.
# These events may indicate different
# photic-stimulation conditions within 60s.

preview_rows = []

print(
    "\n===== PHOTOSTIMULATION EVENT PREVIEW ====="
)

for group in (
    "A",
    "F",
    "C"
):

    sid = sorted(
        s
        for s in selected
        if selected[s]["group"] == group
    )[0]

    rel = (
        f"{sid}/eeg/"
        f"{sid}_task-photomark_events.tsv"
    )

    url = (
        RELEASE_URL + "/" + rel
    )

    try:

        content = get_small(
            url,
            limit=150000
        ).decode(
            "utf-8-sig"
        )

        events = list(
            csv.DictReader(
                io.StringIO(content),
                delimiter="\t"
            )
        )

        if (
            not events
            or "onset" not in events[0]
        ):
            raise RuntimeError(
                "No valid events/onset column"
            )

        within = 0

        for event in events:

            try:

                within += (
                    float(
                        event["onset"]
                    ) < 60
                )

            except (
                ValueError,
                TypeError
            ):
                pass

        preview_rows.append({
            "group": group,
            "subject": sid,
            "events_total":
                len(events),
            "events_start_before_60s":
                within,
            "columns":
                ", ".join(
                    events[0].keys()
                ),
            "first_three_rows":
                json.dumps(
                    events[:3],
                    ensure_ascii=False
                )
        })

        print(
            f"Events {group} {sid}: "
            f"{len(events)} total; "
            f"{within} start before 60s; "
            f"columns="
            f"{list(events[0].keys())}"
        )

    except Exception as exc:

        preview_rows.append({
            "group": group,
            "subject": sid,
            "events_total": "unavailable",
            "error": str(exc)[:180]
        })

        print(
            f"Events {group} {sid}: "
            f"unavailable "
            f"({type(exc).__name__}: "
            f"{str(exc)[:100]})"
        )

save_table(
    "event_preview.csv",
    preview_rows
)


# ==========================================
# 7. SAVE PROVENANCE AND FINISH
# ==========================================

provenance = {
    "eyes_closed_manifest_sha256":
        hashlib.sha256(
            EC_MANIFEST.read_bytes()
        ).hexdigest(),

    "eyes_closed_metadata_sha256":
        hashlib.sha256(
            EC_METADATA.read_bytes()
        ).hexdigest(),

    "eyes_open_metadata_sha256":
        hashlib.sha256(
            eo_path.read_bytes()
        ).hexdigest(),

    "eyes_open_metadata_origin":
        metadata_origin,

    "pinned_nemar_release":
        RELEASE_URL,

    "nemar_manifest_sha256":
        hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        if manifest_path.exists()
        else None,

    "selected_unique_subjects":
        len(selected),

    "local_derivative_subjects":
        local_count,

    "disk_free_gib":
        round(
            free,
            3
        ),

    "next_protocol":
        "Eyes-closed outer-training participant "
        "models must not see any eyes-open "
        "or eyes-closed samples from held-out "
        "subject; test quality-eligible "
        "eyes-open 60s, without label-based selection"
}

(
    OUT / "preflight_provenance.json"
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True
    ) + "\n",
    encoding="utf-8"
)

print(
    "\n===== MILESTONE 3A COMPLETE ====="
)

print(
    "Output:",
    OUT
)

print(
    "No EEG recordings downloaded; "
    "no models trained; "
    "no old results changed."
)

print(
    "A 60s eyes-open test is feasible "
    "in principle; five 60s blocks "
    "cannot be assumed."
)

