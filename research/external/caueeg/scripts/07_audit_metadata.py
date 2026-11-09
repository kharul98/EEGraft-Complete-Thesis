from pathlib import Path
from collections import Counter
import csv
import io
import json
import os

HOME = Path.home()
WORKSPACE = HOME / "research" / "eegraft-caueeg"
DATA_ROOT = HOME / "research" / "caueeg"

LAYOUT = WORKSPACE / "protocol" / "dataset_layout.json"
SUMMARY = WORKSPACE / "results" / "07_metadata_audit_summary.json"
MANIFEST = DATA_ROOT / "manifests" / "07_recording_audit.csv"

SPLITS = ("train_split", "validation_split", "test_split")

PATIENT_KEYS = {
    "patient_id", "patientid", "subject_id", "subjectid",
    "person_id", "participant_id", "patient_uid", "patient_no", "pid"
}

VALID_LABELS = {"normal", "mci", "dementia"}

AD_HINTS = {
    "ad", "alzheimer", "alzheimers", "alzheimer_disease",
    "alzheimers_disease", "dementia_ad", "ad_dementia"
}


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serial_of(value):
    if value is None or isinstance(value, bool):
        return None

    token = str(value).strip()

    if token.lower().endswith(".edf"):
        token = Path(token).stem

    if not token.isascii() or not token.isdigit() or len(token) > 8:
        return None

    return token.zfill(5)


def records_from(value):
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, (str, int)):
                yield {"serial": item}

    elif isinstance(value, dict):
        if isinstance(value.get("data"), list):
            yield from records_from(value["data"])
        else:
            for key, item in value.items():
                if isinstance(item, dict):
                    record = dict(item)
                    record.setdefault("serial", key)
                    yield record
                elif isinstance(item, (str, int)):
                    yield {"serial": key, "class_label": item}


def label_map_from(task):
    containers = [task]

    for key in ("config", "config_data", "task_config"):
        containers.append(task.get(key))

    for block in containers:
        if not isinstance(block, dict):
            continue

        direct = block.get("class_label_to_name")

        if isinstance(direct, list):
            return {
                str(i): str(name).lower().strip()
                for i, name in enumerate(direct)
            }

        if isinstance(direct, dict):
            return {
                str(k): str(v).lower().strip()
                for k, v in direct.items()
            }

        reverse = block.get("class_name_to_label")

        if isinstance(reverse, dict):
            return {
                str(v): str(k).lower().strip()
                for k, v in reverse.items()
            }

    return {}


def category(record, mapping):
    raw = record.get("class_name", record.get("class"))

    if raw is not None:
        value = str(raw).lower().strip()
        if value in VALID_LABELS:
            return value

    raw = record.get("class_label", record.get("label"))
    found = mapping.get(str(raw), "")

    if found in VALID_LABELS:
        return found

    return "unresolved"


def safe_symptoms(record):
    value = record.get("symptom", [])

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    return [
        token.lower().strip()
        for token in value
        if isinstance(token, str)
    ]


def preserve_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing file matches:", path.name)
            return True

        print("[PRESERVED] Different file already exists:", path)
        print("Nothing was overwritten.")
        return False

    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)

    os.chmod(path, 0o600)

    print("[SAVED]", path)
    return True


def run():
    print("\n1. LOADING VERIFIED DATASET")

    if not LAYOUT.is_file():
        print("[BLOCKED] Dataset layout missing. Complete Code 06.")
        return

    config = read_json(LAYOUT)

    annotation_path = Path(config["annotation"])
    split_path = Path(config["split_file"])
    edf_dir = Path(config["edf_directory"])

    if (
        not annotation_path.is_file()
        or not split_path.is_file()
        or not edf_dir.is_dir()
    ):
        print("[BLOCKED] Required dataset files are missing.")
        return

    annotation = read_json(annotation_path)
    task = read_json(split_path)

    if not isinstance(annotation, (dict, list)):
        print("[BLOCKED] Unexpected annotation format.")
        return

    if not isinstance(task, dict):
        print("[BLOCKED] Unexpected official split format.")
        return

    print("[OK] Metadata loaded.")

    print("\n2. ANNOTATION STRUCTURE")

    if isinstance(annotation, dict):
        print("Annotation top-level keys:", sorted(annotation)[:20])
        source = annotation.get("data", annotation)
    else:
        print("Annotation structure: list")
        source = annotation

    print("Official task keys:", sorted(task)[:20])

    annotations = list(records_from(source))

    print("Annotation records:", len(annotations))

    if annotations:
        print(
            "Annotation field names:",
            sorted(annotations[0])[:35]
        )

    ann_serials = [
        serial_of(record.get("serial"))
        for record in annotations
    ]

    ann_counts = Counter(
        serial for serial in ann_serials
        if serial is not None
    )

    invalid_annotation = sum(
        serial is None for serial in ann_serials
    )

    duplicate_annotations = sum(
        count - 1
        for count in ann_counts.values()
        if count > 1
    )

    patient_counts = Counter()

    for record in annotations:
        for key, value in record.items():
            if (
                key.lower() in PATIENT_KEYS
                and value not in (None, "", [])
            ):
                patient_counts[key] += 1

    ad_possible = sum(
        bool(set(safe_symptoms(record)) & AD_HINTS)
        for record in annotations
    )

    symptom_counts = Counter(
        token
        for record in annotations
        for token in safe_symptoms(record)
        if token in ("normal", "mci", "dementia", "ad")
    )

    print("\n3. ANNOTATION COMPLETENESS")

    print("Invalid annotation serials:", invalid_annotation)
    print("Duplicate annotation serials:", duplicate_annotations)
    print("Broad symptom categories:", dict(symptom_counts))
    print("Potential AD-specific symptom records:", ad_possible)

    print(
        "Patient ID field presence (counts only):",
        dict(patient_counts) or "None detected"
    )

    print("\n4. OFFICIAL NO-OVERLAP SPLITS")

    mapping = label_map_from(task)

    print(
        "Class-label mapping:",
        mapping or "Not supplied in task JSON"
    )

    edf_serials = {
        serial_of(path.stem)
        for path in edf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".edf"
    }

    edf_serials.discard(None)

    print("EDF recordings:", len(edf_serials))

    rows = []
    split_serials = {}
    split_counts = {}

    duplicate_split = 0
    invalid_split = 0
    missing_edf = 0
    missing_annotation = 0
    unresolved_labels = 0

    for split_key in SPLITS:
        if split_key not in task:
            print("[BLOCKED] Missing official split:", split_key)
            return

        items = list(records_from(task[split_key]))

        print("\nSplit:", split_key)
        print("Entries:", len(items))

        names = Counter()
        serial_list = []

        for item in items:
            serial = serial_of(item.get("serial"))

            if serial is None:
                invalid_split += 1
                continue

            serial_list.append(serial)

            label = category(item, mapping)

            names[label] += 1

            if label == "unresolved":
                unresolved_labels += 1

            has_edf = serial in edf_serials
            has_annotation = serial in ann_counts

            missing_edf += not has_edf
            missing_annotation += not has_annotation

            rows.append({
                "serial": serial,
                "split": split_key.removesuffix("_split"),
                "class_name": label,
                "class_label_raw": str(
                    item.get("class_label", item.get("label", ""))
                ),
                "edf_exists": int(has_edf),
                "annotation_exists": int(has_annotation),
            })

        duplicate_split += sum(
            count - 1
            for count in Counter(serial_list).values()
            if count > 1
        )

        split_serials[split_key] = set(serial_list)
        split_counts[split_key] = dict(sorted(names.items()))

        print("Class distribution:", dict(sorted(names.items())))

    print("\n5. SPLIT INTEGRITY")

    overlaps = {
        "train_validation": len(
            split_serials["train_split"]
            & split_serials["validation_split"]
        ),
        "train_test": len(
            split_serials["train_split"]
            & split_serials["test_split"]
        ),
        "validation_test": len(
            split_serials["validation_split"]
            & split_serials["test_split"]
        ),
    }

    print("Invalid split serials:", invalid_split)
    print("Duplicate serials within splits:", duplicate_split)
    print("Cross-split recording overlaps:", overlaps)
    print("Split recordings missing EDF:", missing_edf)
    print("Split recordings missing annotation:", missing_annotation)
    print("Unresolved diagnosis labels:", unresolved_labels)

    print("\n6. SCIENTIFIC VALIDITY CHECK")

    label_counts = Counter(
        row["class_name"] for row in rows
    )

    print("Overall official split labels:", dict(label_counts))

    print(
        "NOTE: Dementia does not automatically mean "
        "Alzheimer's disease."
    )

    print(
        "NOTE: The official no-overlap benchmark was designed "
        "to exclude patient overlaps, but we cannot independently "
        "verify patient identity from recording serials alone."
    )

    issues = {
        "invalid_annotation_serials": invalid_annotation,
        "duplicate_annotation_serials": duplicate_annotations,
        "invalid_split_serials": invalid_split,
        "duplicate_within_split": duplicate_split,
        "cross_split_serial_overlaps": sum(overlaps.values()),
        "split_missing_edf": missing_edf,
        "split_missing_annotation": missing_annotation,
        "unresolved_labels": unresolved_labels,
    }

    ready = (
        bool(rows)
        and not any(issues.values())
        and label_counts["normal"] > 0
        and label_counts["dementia"] > 0
    )

    summary = {
        "dataset": "CAUEEG",
        "benchmark": "dementia-no-overlap",
        "annotation_record_count": len(annotations),
        "edf_file_count": len(edf_serials),
        "split_entry_count": len(rows),
        "split_class_counts": split_counts,
        "overall_split_class_counts": dict(
            sorted(label_counts.items())
        ),
        "recording_overlap_counts": overlaps,
        "issue_counts": issues,
        "patient_id_fields_found": dict(patient_counts),
        "potential_ad_symptom_count_needs_clinical_review": ad_possible,
        "dementia_is_not_equivalent_to_confirmed_alzheimers": True,
        "patient_identity_independently_verified": False,
        "audit_status": (
            "READY_FOR_EDF_QC"
            if ready
            else "REVIEW_REQUIRED"
        ),
    }

    print("\n7. SAVING AUDIT RESULTS")

    summary_text = json.dumps(
        summary,
        indent=2,
        ensure_ascii=True
    ) + "\n"

    private_buffer = io.StringIO()

    writer = csv.DictWriter(
        private_buffer,
        fieldnames=[
            "serial",
            "split",
            "class_name",
            "class_label_raw",
            "edf_exists",
            "annotation_exists",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

    saved_summary = preserve_write(SUMMARY, summary_text)
    saved_manifest = preserve_write(
        MANIFEST,
        private_buffer.getvalue()
    )

    if not (saved_summary and saved_manifest):
        print("AUDIT_STATUS: FILE_CONFLICT_REVIEW_REQUIRED")
    else:
        print("AUDIT_STATUS:", summary["audit_status"])

    print(
        "Restricted CAUEEG annotations and recording manifests "
        "must remain local."
    )


try:
    run()

except Exception as exc:
    print("[AUDIT ERROR]", type(exc).__name__, str(exc))

finally:
    print("\n========== CODE 07 FINISHED ==========")
