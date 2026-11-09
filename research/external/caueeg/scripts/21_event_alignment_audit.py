from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
import re

import numpy as np

HOME = Path.home()
WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

MANIFEST = DATA / "manifests/07_recording_audit.csv"
LAYOUT = WORK / "protocol/dataset_layout.json"

INFERENCE = WORK / "results/11_train_val_inference_summary.json"
FINAL_PLAN = WORK / "protocol/caueeg_final_test_plan_v1.json"

PRIVATE = DATA / "derived/caueeg_transfer_v1"

PRIVATE_WINDOWS = (
    PRIVATE / "analysis/21_event_aligned_windows.csv"
)

PRIVATE_RECORDINGS = (
    PRIVATE / "analysis/21_event_alignment_records.csv"
)

SUMMARY = WORK / "results/21_event_alignment_summary.json"

FS = 200
WINDOW_SECONDS = 4
WINDOW_SAMPLES = 800
MARGIN_SECONDS = 2
MARGIN_SAMPLES = 400

EXPECTED = {
    "train": {
        "normal": 367,
        "mci": 334,
        "dementia": 249
    },
    "validation": {
        "normal": 34,
        "mci": 28,
        "dementia": 20
    }
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(4194304), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def save_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path.name)
            return True
        print("[CONFLICT] Existing output preserved:", path)
        return False

    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )

    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)

    print("[SAVED]", path)
    return True


def csv_text(rows, fields):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def verify_setup():
    print("\n1. VERIFYING EXISTING EXPERIMENT")

    for path in (
        MANIFEST, LAYOUT, INFERENCE, FINAL_PLAN
    ):
        require(path.is_file(), f"Missing: {path}")

    inference = load_json(INFERENCE)
    plan = load_json(FINAL_PLAN)
    layout = load_json(LAYOUT)

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Training/validation predictions are incomplete"
    )

    require(
        inference["total_recordings"] == 1032
        and inference["total_windows"] == 24768,
        "Development prediction counts changed"
    )

    require(
        plan["version"] == "caueeg_final_test_plan_v1",
        "Unexpected final test plan"
    )

    require(
        plan["model_hashes"] == inference["model_hashes"],
        "Model provenance mismatch"
    )

    edf_dir = Path(layout["edf_directory"])

    require(
        edf_dir.is_dir(),
        "Original EDF directory missing"
    )

    event_dir = edf_dir.parents[1] / "event"

    require(
        event_dir.is_dir(),
        "CAUEEG event directory missing"
    )

    print("[OK] Original inference summary verified")
    print("[OK] Frozen final test protocol preserved")
    print("[OK] CAUEEG event folder found")
    print("[OK] No new model inference required")

    return event_dir, inference, plan


def load_manifest():
    print("\n2. SELECTING TRAINING AND VALIDATION ONLY")

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as stream:
        all_rows = list(csv.DictReader(stream))

    require(
        len(all_rows) == 1122,
        "Unexpected official benchmark size"
    )

    selected = [
        row for row in all_rows
        if row["split"] in ("train", "validation")
    ]

    require(
        len(selected) == 1032,
        "Expected exactly 1032 development recordings"
    )

    counts = {
        split: dict(Counter(
            row["class_name"]
            for row in selected
            if row["split"] == split
        ))
        for split in ("train", "validation")
    }

    require(
        counts == EXPECTED,
        "Original class or split counts changed"
    )

    require(
        len({row["serial"] for row in selected}) == 1032,
        "Duplicate development recording"
    )

    print("Training recordings:", 950)
    print("Validation recordings:", 82)
    print("Test event files opened: 0")
    print("Test waveform files opened: 0")
    print("Test predictions opened: 0")

    return sorted(
        selected,
        key=lambda row: (
            row["split"] != "train",
            row["serial"]
        )
    )


def parse_event_file(path):
    if not path.is_file():
        return [], "missing"

    try:
        data = load_json(path)
    except (ValueError, UnicodeError, OSError):
        return [], "unreadable"

    if isinstance(data, dict):
        if "event" in data:
            data = data["event"]
        elif "events" in data:
            data = data["events"]

    if not isinstance(data, list):
        return [], "unknown_format"

    events = []

    for index, item in enumerate(data):
        if (
            not isinstance(item, (list, tuple))
            or len(item) < 2
        ):
            return [], "unknown_format"

        sample, label = item[0], item[1]

        if isinstance(sample, bool):
            return [], "unknown_format"

        try:
            sample_number = float(sample)
        except (ValueError, TypeError):
            return [], "unknown_format"

        if (
            not math.isfinite(sample_number)
            or abs(sample_number - round(sample_number)) > 1e-6
            or not isinstance(label, str)
        ):
            return [], "unknown_format"

        events.append((
            int(round(sample_number)),
            label.strip(),
            index
        ))

    if not events:
        return [], "empty"

    return events, "parsed"


def event_kind(label):
    value = re.sub(
        r"[_-]+", " ", label.lower().strip()
    )
    value = " ".join(value.split())

    if re.search(r"\beyes?\s+open\b", value):
        return "eyes_open"

    if re.search(r"\beyes?\s+clos(?:ed|e)\b", value):
        return "eyes_closed"

    if re.search(r"\bphotic\b.*\bon\b", value):
        return "photic_on"

    if re.search(r"\bphotic\b.*\boff\b", value):
        return "photic_off"

    if re.search(r"\b(paused|pause|stop recording)\b", value):
        return "paused"

    if re.search(
        r"\b(resumed|resume|continue recording|start recording)\b",
        value
    ):
        return "recording_active"

    if re.search(
        r"artifact|artefact|blink|sweat|swallow|"
        r"movement|move|drows|electrode|muscle|"
        r"cough|yawn|impedance",
        value
    ):
        return "annotated_disturbance"

    return "other"


STATE_CHANGING = {
    "eyes_open",
    "eyes_closed",
    "photic_on",
    "photic_off",
    "paused",
    "recording_active"
}


def validate_events(events, duration_samples):
    invalid = [
        event for event in events
        if event[0] < 0 or event[0] >= duration_samples
    ]

    inversions = sum(
        events[i][0] < events[i - 1][0]
        for i in range(1, len(events))
    )

    if invalid:
        status = "outside_duration"
    elif inversions:
        status = "unordered_timestamps"
    else:
        status = "aligned"

    sorted_events = sorted(
        events, key=lambda event: (event[0], event[2])
    )

    return sorted_events, status, len(invalid), inversions


def window_condition(events, start_sample):
    end_sample = start_sample + WINDOW_SAMPLES

    expanded_start = max(
        0, start_sample - MARGIN_SAMPLES
    )
    expanded_end = end_sample + MARGIN_SAMPLES

    eyes = "unknown"
    photic_active = False
    recording_active = True

    for sample, label, _ in events:
        if sample > start_sample:
            break

        kind = event_kind(label)

        if kind == "eyes_open":
            eyes = "eyes_open"

        elif kind == "eyes_closed":
            eyes = "eyes_closed"

        elif kind == "photic_on":
            photic_active = True

        elif kind == "photic_off":
            photic_active = False

        elif kind == "paused":
            recording_active = False
            eyes = "unknown"

        elif kind == "recording_active":
            recording_active = True
            # A new recording start does not establish eye state.
            if sample > 0:
                eyes = "unknown"

    nearby = [
        event_kind(label)
        for sample, label, _ in events
        if expanded_start <= sample <= expanded_end
    ]

    if "annotated_disturbance" in nearby:
        return "excluded", "annotated_disturbance_near_window"

    if any(kind in STATE_CHANGING for kind in nearby):
        return "excluded", "event_boundary_near_window"

    if not recording_active:
        return "excluded", "recording_paused"

    if photic_active:
        return "excluded", "photic_stimulation_active"

    if eyes == "unknown":
        return "excluded", "eye_state_not_explicit"

    return eyes, "eligible"


def load_saved_prediction(row, hashes):
    split = row["split"]
    serial = row["serial"]

    path = PRIVATE / split / (serial + ".json")

    require(
        path.is_file(),
        "Missing development prediction"
    )

    item = load_json(path)

    require(
        item["version"] == "caueeg_transfer_v1"
        and item["split"] == split
        and item["serial"] == serial
        and item["class_name"] == row["class_name"]
        and item["model_hashes"] == hashes,
        "Frozen prediction provenance mismatch"
    )

    starts = item["window_start_seconds"]
    duration = float(item["recording_duration_seconds"])

    require(
        len(starts) == 24
        and np.isfinite(duration)
        and duration >= 96,
        "Invalid recording window metadata"
    )

    total_samples = int(round(duration * FS))
    available = total_samples // WINDOW_SAMPLES

    expected_indices = np.unique(
        np.linspace(
            0,
            available - 1,
            num=24,
            dtype=int
        )
    )

    expected_starts = [
        int(index * WINDOW_SECONDS)
        for index in expected_indices
    ]

    require(
        starts == expected_starts,
        "Stored EEG window positions differ from Code 11"
    )

    for prefix in ("dsrf", "xgb"):
        values = np.asarray(
            item[prefix + "_window_p_ad"],
            dtype=float
        )

        require(
            values.shape == (24,)
            and np.isfinite(values).all()
            and np.all((values >= 0) & (values <= 1)),
            "Invalid stored window probabilities"
        )

    return item, total_samples


def audit(event_dir, rows, hashes):
    print("\n3. AUDITING EVENT TIME ALIGNMENT")

    recording_rows = []
    window_rows = []

    status_counts = Counter()
    window_counts = Counter()
    class_condition = Counter()

    paired_any = Counter()
    paired_two_each = Counter()

    total_events = 0
    out_of_range_events = 0
    unordered_recordings = 0

    for index, row in enumerate(rows, start=1):
        serial = row["serial"]
        split = row["split"]
        label = row["class_name"]

        prediction, total_samples = load_saved_prediction(
            row, hashes
        )

        event_path = event_dir / (serial + ".json")

        events, parse_status = parse_event_file(
            event_path
        )

        status = parse_status
        invalid_count = 0
        inversion_count = 0

        if parse_status == "parsed":
            events, status, invalid_count, inversion_count = (
                validate_events(events, total_samples)
            )

        status_counts[(split, status)] += 1

        total_events += len(events)
        out_of_range_events += invalid_count

        if inversion_count > 0:
            unordered_recordings += 1

        eligible = Counter()

        dsrf = prediction["dsrf_window_p_ad"]
        xgb = prediction["xgb_window_p_ad"]

        for window_index, start in enumerate(
            prediction["window_start_seconds"]
        ):
            if status == "aligned":
                condition, reason = window_condition(
                    events,
                    int(round(start * FS))
                )

            else:
                condition = "excluded"
                reason = "event_metadata_" + status

            window_counts[(split, condition)] += 1
            class_condition[(split, label, condition)] += 1

            if condition in ("eyes_open", "eyes_closed"):
                eligible[condition] += 1

            window_rows.append({
                "serial": serial,
                "split": split,
                "class_name": label,
                "window_index": window_index,
                "window_start_seconds": start,
                "window_end_seconds": start + WINDOW_SECONDS,
                "condition": condition,
                "reason": reason,
                "dsrf_source_ad_score": dsrf[window_index],
                "xgb_source_ad_score": xgb[window_index]
            })

        has_both = (
            eligible["eyes_open"] >= 1
            and eligible["eyes_closed"] >= 1
        )

        has_two_each = (
            eligible["eyes_open"] >= 2
            and eligible["eyes_closed"] >= 2
        )

        paired_any[split] += int(has_both)
        paired_two_each[split] += int(has_two_each)

        recording_rows.append({
            "serial": serial,
            "split": split,
            "class_name": label,
            "duration_seconds":
                prediction["recording_duration_seconds"],
            "event_status": status,
            "event_count": len(events),
            "out_of_range_events": invalid_count,
            "timestamp_inversions": inversion_count,
            "eyes_open_windows": eligible["eyes_open"],
            "eyes_closed_windows": eligible["eyes_closed"],
            "has_both_states": int(has_both),
            "has_two_windows_per_state": int(has_two_each)
        })

        if index % 200 == 0 or index == len(rows):
            print(
                f"[AUDIT] {index}/{len(rows)} recordings",
                flush=True
            )

    require(
        len(recording_rows) == 1032
        and len(window_rows) == 24768,
        "Incomplete event-alignment audit"
    )

    return {
        "recording_rows": recording_rows,
        "window_rows": window_rows,
        "status_counts": status_counts,
        "window_counts": window_counts,
        "class_condition": class_condition,
        "paired_any": paired_any,
        "paired_two_each": paired_two_each,
        "total_events": total_events,
        "out_of_range_events": out_of_range_events,
        "unordered_recordings": unordered_recordings
    }


def main():
    event_dir, inference, plan = verify_setup()

    rows = load_manifest()

    audited = audit(
        event_dir,
        rows,
        inference["model_hashes"]
    )

    print("\n4. EVENT METADATA QUALITY")

    for split in ("train", "validation"):
        total = 950 if split == "train" else 82

        print("\nPartition:", split.upper())

        for status in (
            "aligned",
            "outside_duration",
            "unordered_timestamps",
            "missing",
            "unreadable",
            "unknown_format",
            "empty"
        ):
            count = audited["status_counts"][
                (split, status)
            ]

            print(f"  {status}: {count}/{total}")

        print(
            "  Eyes-open eligible windows:",
            audited["window_counts"][
                (split, "eyes_open")
            ]
        )

        print(
            "  Eyes-closed eligible windows:",
            audited["window_counts"][
                (split, "eyes_closed")
            ]
        )

        print(
            "  Recordings with both eye states:",
            audited["paired_any"][split]
        )

        print(
            "  Recordings with >=2 windows per state:",
            audited["paired_two_each"][split]
        )

    print("\nTotal parsed event entries:",
          audited["total_events"])

    print("Out-of-duration event entries:",
          audited["out_of_range_events"])

    print("Recordings with unordered events:",
          audited["unordered_recordings"])

    print("\n5. SAVING LOCAL AUDIT RESULTS")

    record_rows = audited["recording_rows"]
    window_rows = audited["window_rows"]

    record_ok = save_once(
        PRIVATE_RECORDINGS,
        csv_text(record_rows, list(record_rows[0]))
    )

    window_ok = save_once(
        PRIVATE_WINDOWS,
        csv_text(window_rows, list(window_rows[0]))
    )

    sufficient_pairing = (
        audited["paired_two_each"]["train"] >= 20
    )

    report = {
        "version": "caueeg_event_alignment_audit_v1",
        "status": "EVENT_ALIGNMENT_AUDIT_COMPLETED",

        "dataset": "CAUEEG",
        "benchmark": "dementia-no-overlap",

        "development_recordings": 1032,
        "development_windows": 24768,

        "test_waveforms_opened": 0,
        "test_event_files_opened": 0,
        "test_prediction_files_opened": 0,

        "event_file_directory": str(event_dir),

        "timing": {
            "unit": "sample_index",
            "sampling_hz": FS,
            "window_seconds": WINDOW_SECONDS,
            "event_boundary_margin_seconds": MARGIN_SECONDS
        },

        "event_quality": {
            split: {
                status: audited["status_counts"][
                    (split, status)
                ]
                for status in (
                    "aligned",
                    "outside_duration",
                    "unordered_timestamps",
                    "missing",
                    "unreadable",
                    "unknown_format",
                    "empty"
                )
            }
            for split in ("train", "validation")
        },

        "eligible_window_counts": {
            split: {
                condition: audited["window_counts"][
                    (split, condition)
                ]
                for condition in (
                    "eyes_open",
                    "eyes_closed",
                    "excluded"
                )
            }
            for split in ("train", "validation")
        },

        "by_class_eligible_windows": {
            split: {
                label: {
                    condition: audited["class_condition"][
                        (split, label, condition)
                    ]
                    for condition in (
                        "eyes_open",
                        "eyes_closed"
                    )
                }
                for label in (
                    "normal",
                    "mci",
                    "dementia"
                )
            }
            for split in ("train", "validation")
        },

        "paired_state_coverage": {
            split: {
                "at_least_one_window_each":
                    audited["paired_any"][split],
                "at_least_two_windows_each":
                    audited["paired_two_each"][split]
            }
            for split in ("train", "validation")
        },

        "out_of_range_event_entries":
            audited["out_of_range_events"],

        "recordings_with_timestamp_inversions":
            audited["unordered_recordings"],

        "enough_training_pairs_for_preliminary_analysis":
            sufficient_pairing,

        "original_model_hashes":
            inference["model_hashes"],

        "final_test_plan_sha256":
            sha256(FINAL_PLAN),

        "analysis_status": (
            "EXPLORATORY_POST_TEST_REVIEW"
        ),

        "limitations": [
            "This audit was designed after viewing locked-test results.",
            "No newly independent held-out test is available.",
            "Event timestamps may be misaligned with EDF duration.",
            "Recordings with invalid event chronology are excluded conservatively.",
            "Eye state remains unknown until explicitly annotated.",
            "Photic and disturbance annotations are not exhaustive.",
            "Eligible windows are observational, not randomized.",
            "Window observations from one recording are correlated.",
            "Patient identity cannot be independently verified.",
            "This audit does not establish causal recording-condition effects."
        ]
    }

    report_ok = save_once(
        SUMMARY,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    if not (record_ok and window_ok and report_ok):
        print("\nEVENT_ALIGNMENT_STATUS: OUTPUT_CONFLICT")

    elif sufficient_pairing:
        print(
            "\nEVENT_ALIGNMENT_STATUS: "
            "READY_FOR_EXPLORATORY_CONDITION_ANALYSIS"
        )

    else:
        print(
            "\nEVENT_ALIGNMENT_STATUS: "
            "LIMITED_PAIRED_CONDITION_COVERAGE"
        )

    print("\nOriginal model files modified: 0")
    print("Additional model training: 0")
    print("Test signal, event and prediction files opened: 0")
    print("All recording-level details remain local.")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print("EVENT_ALIGNMENT_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 21 FINISHED ==========",
        flush=True
    )
