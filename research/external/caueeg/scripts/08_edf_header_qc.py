from pathlib import Path
from collections import Counter, defaultdict
import ast
import csv
import io
import json
import math
import os
import re
import statistics

HOME = Path.home()
WORKSPACE = HOME / "research" / "eegraft-caueeg"
DATA_ROOT = HOME / "research" / "caueeg"

MODEL_ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

LAYOUT_FILE = WORKSPACE / "protocol" / "dataset_layout.json"
MANIFEST_FILE = DATA_ROOT / "manifests" / "07_recording_audit.csv"

SUMMARY_FILE = WORKSPACE / "results" / "08_edf_header_summary.json"
DETAIL_FILE = DATA_ROOT / "manifests" / "08_edf_header_audit.csv"

ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

ALLOWED_REFERENCES = {
    "", "AVG", "REF", "LE", "A1", "A2",
    "M1", "M2", "CAR"
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def original_channels():
    source = MODEL_ROOT / "src" / "data.py"

    tree = ast.parse(source.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "CHANNELS"
                ):
                    result = ast.literal_eval(node.value)

                    if len(result) != 19 or len(set(result)) != 19:
                        raise ValueError(
                            "Original CHANNELS must contain 19 unique names"
                        )

                    return list(result)

    raise ValueError("Could not locate original CHANNELS")


def field(raw, start, width):
    return raw[start:start + width].decode(
        "latin-1", errors="replace"
    ).strip()


def parse_number(raw, start, width, cast=float):
    value = field(raw, start, width)
    return cast(value)


def channel_name(label, canonical):
    value = label.upper().strip()
    value = re.sub(r"^EEG[\s:._-]*", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("_", "-")

    if "-" in value:
        base, reference = value.split("-", 1)
    else:
        base, reference = value, ""

    if reference not in ALLOWED_REFERENCES:
        return None, reference

    base = ALIASES.get(base, base)

    return canonical.get(base), reference


def read_edf_header(path, expected_channels):
    canonical = {
        name.upper(): name
        for name in expected_channels
    }

    file_size = path.stat().st_size

    with path.open("rb") as handle:
        base = handle.read(256)

        if len(base) != 256:
            raise ValueError("Truncated EDF fixed header")

        n_signals = parse_number(base, 252, 4, int)
        header_bytes = parse_number(base, 184, 8, int)
        n_records = parse_number(base, 236, 8, int)
        record_seconds = parse_number(base, 244, 8, float)

        if not 1 <= n_signals <= 256:
            raise ValueError("Invalid number of EDF signals")

        if header_bytes != 256 + n_signals * 256:
            raise ValueError("Invalid EDF header length")

        if (
            not math.isfinite(record_seconds)
            or record_seconds <= 0
        ):
            raise ValueError("Invalid EDF data record duration")

        signal_header = handle.read(n_signals * 256)

        if len(signal_header) != n_signals * 256:
            raise ValueError("Truncated EDF signal header")

    columns = [
        ("labels", 16),
        ("transducers", 80),
        ("units", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefilter", 80),
        ("samples", 8),
        ("reserved", 32),
    ]

    parsed = {}
    offset = 0

    for name, width in columns:
        parsed[name] = [
            field(signal_header, offset + i * width, width)
            for i in range(n_signals)
        ]
        offset += n_signals * width

    samples = [int(value) for value in parsed["samples"]]

    if any(value <= 0 for value in samples):
        raise ValueError("Nonpositive samples per EDF record")

    bytes_per_record = 2 * sum(samples)

    if n_records == -1:
        payload_bytes = file_size - header_bytes

        if (
            payload_bytes < 0
            or payload_bytes % bytes_per_record != 0
        ):
            raise ValueError(
                "Cannot infer EDF record count from file size"
            )

        n_records = payload_bytes // bytes_per_record

    if n_records <= 0:
        raise ValueError("EDF contains no complete data records")

    expected_size = header_bytes + n_records * bytes_per_record

    if expected_size != file_size:
        raise ValueError(
            "EDF file length differs from header declaration"
        )

    duration = n_records * record_seconds

    positions = {}
    references = {}
    units = {}
    rates = {}

    duplicate_channels = []

    for i, label in enumerate(parsed["labels"]):
        name, reference = channel_name(label, canonical)

        if name is None:
            continue

        if name in positions:
            duplicate_channels.append(name)
            continue

        positions[name] = i
        references[name] = reference
        units[name] = parsed["units"][i]
        rates[name] = samples[i] / record_seconds

    missing = [
        name for name in expected_channels
        if name not in positions
    ]

    detected_order = [
        name for name, index in sorted(
            positions.items(), key=lambda entry: entry[1]
        )
    ]

    needs_reorder = detected_order != expected_channels

    issues = []

    if missing:
        issues.append("missing_eeg_channels")

    if duplicate_channels:
        issues.append("duplicate_eeg_channels")

    eeg_rates = list(rates.values())

    if eeg_rates and (
        max(eeg_rates) - min(eeg_rates) > 0.001
    ):
        issues.append("mixed_eeg_sample_rates")

    normalized_units = {
        value.lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace(" ", "")
        for value in units.values()
    }

    accepted_units = {
        "uv", "microv", "microvolt", "microvolts",
        "mv", "v", "nv"
    }

    if not normalized_units or not normalized_units.issubset(
        accepted_units
    ):
        issues.append("unrecognized_eeg_units")

    if len(normalized_units) > 1:
        issues.append("mixed_eeg_units")

    if duration < 4:
        issues.append("shorter_than_one_4s_window")

    return {
        "serial": path.stem,
        "n_signals": n_signals,
        "mapped_eeg_channels": len(positions),
        "duration_seconds": round(duration, 3),
        "eeg_sampling_rate": (
            round(eeg_rates[0], 3)
            if eeg_rates and max(eeg_rates) - min(eeg_rates) < 0.001
            else ""
        ),
        "eeg_units": ";".join(sorted(units.values())),
        "eeg_references": ";".join(sorted(set(references.values()))),
        "missing_channels": ";".join(missing),
        "needs_channel_reordering": int(needs_reorder),
        "has_96_seconds": int(duration >= 96),
        "issues": ";".join(issues),
    }


def preserve_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path.name)
            return True

        print("[PRESERVED] Existing output differs:", path)
        print("No file overwritten.")
        return False

    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)

    os.chmod(path, 0o600)

    print("[SAVED]", path)
    return True


def run():
    print("\n1. Loading original model configuration")

    channels = original_channels()

    print("[OK] Original channel count:", len(channels))
    print("Required channel order:", channels)

    if not LAYOUT_FILE.is_file() or not MANIFEST_FILE.is_file():
        print("[BLOCKED] Complete Codes 06 and 07 first.")
        return

    layout = read_json(LAYOUT_FILE)

    edf_dir = Path(layout["edf_directory"])

    if not edf_dir.is_dir():
        print("[BLOCKED] EDF directory is missing.")
        return

    with MANIFEST_FILE.open(
        newline="", encoding="utf-8"
    ) as handle:
        manifest = list(csv.DictReader(handle))

    official = {
        row["serial"]: row
        for row in manifest
    }

    paths = sorted(
        path for path in edf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".edf"
    )

    print("\n2. Scanning actual EDF headers")
    print("EDF files:", len(paths))
    print("Official split recordings:", len(official))
    print("No EEG signal samples will be loaded.")

    rows = []
    failure_examples = []
    unit_counts = Counter()
    reference_counts = Counter()
    rate_counts = Counter()
    issue_counts = Counter()

    durations = []
    official_issues = Counter()
    official_totals = Counter()
    reorder_count = 0
    long_enough_count = 0

    for index, path in enumerate(paths, start=1):
        try:
            result = read_edf_header(path, channels)
            result["header_readable"] = 1

        except Exception as exc:
            result = {
                "serial": path.stem,
                "n_signals": "",
                "mapped_eeg_channels": 0,
                "duration_seconds": "",
                "eeg_sampling_rate": "",
                "eeg_units": "",
                "eeg_references": "",
                "missing_channels": "",
                "needs_channel_reordering": "",
                "has_96_seconds": 0,
                "issues": "invalid_edf_header",
                "header_readable": 0,
            }

            if len(failure_examples) < 10:
                failure_examples.append({
                    "serial": path.stem,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:150],
                })

        original = official.get(path.stem)

        result["official_split"] = (
            original["split"] if original else "outside_benchmark"
        )
        result["official_class"] = (
            original["class_name"] if original else ""
        )

        rows.append(result)

        split = result["official_split"]

        if original is not None:
            official_totals[split] += 1

        problems = [
            item for item in result["issues"].split(";")
            if item
        ]

        for problem in problems:
            issue_counts[problem] += 1

            if original is not None:
                official_issues[split] += 1

        if result["eeg_sampling_rate"] != "":
            rate_counts[
                str(result["eeg_sampling_rate"])
            ] += 1

        if result["eeg_units"]:
            unit_counts[result["eeg_units"]] += 1

        if result["eeg_references"] is not None:
            reference_counts[
                result["eeg_references"]
            ] += 1

        if result["duration_seconds"] != "":
            durations.append(result["duration_seconds"])

        if result["needs_channel_reordering"] == 1:
            reorder_count += 1

        if result["has_96_seconds"] == 1:
            long_enough_count += 1

        if index % 250 == 0:
            print(f"[SCANNED] {index}/{len(paths)}")

    print(f"[SCANNED] {len(paths)}/{len(paths)}")

    print("\n3. EEG quality-control summary")

    print("EDF header issues:", dict(issue_counts))
    print("EEG sampling rates (Hz):", dict(rate_counts))
    print("Physical EEG units:", dict(unit_counts))
    print("Reference types:", dict(reference_counts))
    print("Recordings requiring channel reordering:", reorder_count)
    print("Recordings at least 96 seconds:", long_enough_count)

    if durations:
        print(
            "Recording duration (seconds):",
            "min =", round(min(durations), 2),
            "median =", round(statistics.median(durations), 2),
            "max =", round(max(durations), 2)
        )

    print("\n4. Official benchmark split QC")

    print("Valid EDF counts by split:", dict(official_totals))
    print("QC issue counts by split:", dict(official_issues))

    for item in failure_examples:
        print(
            "[HEADER ERROR]",
            item["serial"],
            item["error_type"],
            item["message"]
        )

    print("\n5. Diagnostic annotation cross-check")

    annotation = read_json(Path(layout["annotation"]))

    annotation_by_serial = {
        str(item["serial"]).zfill(5): item
        for item in annotation.get("data", [])
        if isinstance(item, dict) and "serial" in item
    }

    ad_tag_counts = Counter()
    class_totals = Counter()

    for row in manifest:
        key = (row["split"], row["class_name"])
        class_totals[key] += 1

        metadata = annotation_by_serial.get(row["serial"], {})
        symptoms = metadata.get("symptom", [])

        if isinstance(symptoms, str):
            symptoms = [symptoms]

        if not isinstance(symptoms, list):
            symptoms = []

        tokens = {
            token.lower().strip()
            for token in symptoms
            if isinstance(token, str)
        }

        if "ad" in tokens:
            ad_tag_counts[key] += 1

    for split in ("train", "validation", "test"):
        for label in ("normal", "mci", "dementia"):
            key = (split, label)
            print(
                split,
                label,
                "recordings =", class_totals[key],
                "AD-tagged =", ad_tag_counts[key]
            )

    print(
        "\nImportant: An 'ad' symptom tag is not independently "
        "verified patient-level Alzheimer's diagnosis."
    )

    print("\n6. Saving EDF audit")

    columns = [
        "serial",
        "official_split",
        "official_class",
        "header_readable",
        "n_signals",
        "mapped_eeg_channels",
        "duration_seconds",
        "eeg_sampling_rate",
        "eeg_units",
        "eeg_references",
        "missing_channels",
        "needs_channel_reordering",
        "has_96_seconds",
        "issues",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

    scanned_serials = {row["serial"] for row in rows}
    missing_official = set(official) - scanned_serials

    status = (
        "READY_FOR_WAVEFORM_SMOKE_TEST"
        if (
            not official_issues
            and not missing_official
            and len(official) == len(manifest)
        )
        else "REVIEW_REQUIRED"
    )

    summary = {
        "total_edf_files": len(paths),
        "official_split_recordings": len(manifest),
        "outside_official_split": len(paths) - len(official),
        "issue_counts": dict(issue_counts),
        "official_issue_counts": dict(official_issues),
        "missing_official_edf_count": len(missing_official),
        "sampling_rates_hz": dict(rate_counts),
        "physical_units": dict(unit_counts),
        "references": dict(reference_counts),
        "reordering_required_count": reorder_count,
        "duration_at_least_96_seconds_count": long_enough_count,
        "ad_tag_by_split_and_class": {
            f"{split}/{label}": ad_tag_counts[(split, label)]
            for split in ("train", "validation", "test")
            for label in ("normal", "mci", "dementia")
        },
        "diagnostic_note": (
            "Official CAUEEG dementia labels are not automatically "
            "equivalent to confirmed Alzheimer diagnoses."
        ),
        "waveforms_verified": False,
        "audit_status": status,
    }

    summary_text = json.dumps(
        summary, indent=2, sort_keys=True
    ) + "\n"

    ok1 = preserve_write(DETAIL_FILE, buffer.getvalue())
    ok2 = preserve_write(SUMMARY_FILE, summary_text)

    if ok1 and ok2:
        print("\nEDF_QC_STATUS:", status)
    else:
        print("\nEDF_QC_STATUS: OUTPUT_CONFLICT")

    print(
        "Patient-level identifiers and restricted dataset files "
        "have not been uploaded."
    )


try:
    run()

except Exception as exc:
    print("[ERROR]", type(exc).__name__, str(exc))
    print("EDF_QC_STATUS: FAILED")

finally:
    print("\n========== CODE 08 FINISHED ==========")
