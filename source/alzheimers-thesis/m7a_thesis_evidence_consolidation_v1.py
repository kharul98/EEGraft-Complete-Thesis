#!/usr/bin/env python3
"""M7A: consolidate frozen thesis evidence and locate primary result sources."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path.cwd()
BASE = ROOT / "reports/thesis_final"

OUT = BASE / "milestone_7a_thesis_evidence_v1"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "mci_quality":
        BASE / "milestone_5d_isfahan_labels_and_qc_v2"
        / "first_five_minute_quality.csv",

    "mci_cv":
        BASE / "milestone_5g_isfahan_nested_reliability_v1"
        / "summary_conditional_subject_bootstrap.csv",

    "mci_controls":
        BASE / "milestone_5i_isfahan_negative_controls_v1"
        / "constant_label_comparison.csv",

    "figshare_decision":
        BASE / "milestone_6d_figshare_feasibility_decision_v1"
        / "locked_decision.json",
}


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for part in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(part)

    return h.hexdigest()


def freeze(path, content):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                "Refusing to overwrite changed output: "
                + str(path)
            )
    else:
        path.write_text(content, encoding="utf-8")


print(
    "===== MILESTONE 7A: "
    "FINAL THESIS EVIDENCE CONSOLIDATION =====",
    flush=True
)

for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {name}: {path}"
        )

quality = read_csv(FILES["mci_quality"])
mci_cv = read_csv(FILES["mci_cv"])
mci_controls = read_csv(FILES["mci_controls"])

figshare = json.loads(
    FILES["figshare_decision"].read_text(
        encoding="utf-8"
    )
)

if (
    len(quality) != 61
    or len(mci_cv) != 2
    or len(mci_controls) != 2
):
    raise RuntimeError(
        "Frozen Isfahan evidence counts changed"
    )

eligible = [
    r for r in quality
    if r["status"] == "ELIGIBLE_FIXED_300S"
]

if len(eligible) != 49:
    raise RuntimeError(
        "Isfahan eligible recording count changed"
    )

observed = Counter(
    (r["release"], r["diagnosis"])
    for r in eligible
)

expected = {
    ("EEG_1", "MCI"): 11,
    ("EEG_1", "NORMAL"): 14,
    ("EEG_2", "MCI"): 12,
    ("EEG_2", "NORMAL"): 12,
}

if dict(observed) != expected:
    raise RuntimeError(
        "MCI/healthy group counts changed"
    )

cv_by_release = {
    r["release"]: r
    for r in mci_cv
}

control_by_release = {
    r["release"]: r
    for r in mci_controls
}

if (
    set(cv_by_release) != {"EEG_1", "EEG_2"}
    or set(control_by_release) != {"EEG_1", "EEG_2"}
):
    raise RuntimeError(
        "MCI release summaries incomplete"
    )

for release in ("EEG_1", "EEG_2"):

    a = cv_by_release[release]
    b = control_by_release[release]

    for left, right in (
        ("minute1_BACC", "model_first_minute_BACC"),
        ("stable_wrong_rate", "model_stable_wrong"),
        ("unstable_rate", "model_unstable"),
    ):

        if abs(
            float(a[left]) - float(b[right])
        ) > 1e-10:
            raise RuntimeError(
                f"M5G/M5I disagreement: {release}, {left}"
            )


if (
    figshare["decision"]
    != "NO_GO_FOR_ORIGINAL_19_CHANNEL_60S_EXTERNAL_AD_VALIDATION"
):
    raise RuntimeError(
        "Figshare decision changed"
    )

if (
    int(figshare["figshare_mat_files"]) != 471
    or int(
        figshare[
            "standard_19_column_nominal_60s_AD_proxies"
        ]
    ) != 44
    or int(
        figshare[
            "standard_19_column_nominal_60s_CONTROL_proxies"
        ]
    ) != 1
):
    raise RuntimeError(
        "Figshare feasibility findings changed"
    )

print(
    "PASS: Isfahan MCI quality, nested CV, "
    "and negative-control results agree."
)

print(
    "PASS: Figshare external AD feasibility "
    "decision verified."
)


# ========================================
# DISCOVER ORIGINAL AD/FTD EVIDENCE FILES
# ========================================

print(
    "\n===== ORIGINAL AD/FTD SOURCE DISCOVERY ====="
)

primary_dirs = [
    BASE / "milestone_4c_reproducibility_v1",
    BASE / "duration_locked_v1",
    BASE / "ftd_extension_v1",
    BASE / "milestone_1b_calibration_v1",
    BASE / "milestone_3c_boundary_feasibility_v1",
]

keywords = {
    "summary": 4,
    "result": 3,
    "metric": 3,
    "temporal": 3,
    "stability": 3,
    "bootstrap": 3,
    "reproduc": 2,
    "calibrat": 2,
    "ablation": 2,
    "condition": 2,
    "duration": 1,
    "prediction": 1,
    "audit": 1,
}

candidates = []

for folder in primary_dirs:

    if not folder.is_dir():

        print(
            "NOT FOUND:",
            folder.relative_to(ROOT)
        )

        continue

    all_files = [
        p for p in folder.rglob("*")
        if p.is_file()
    ]

    print(
        folder.relative_to(ROOT),
        "| files:",
        len(all_files)
    )

    for path in all_files:

        if path.suffix.lower() not in (
            ".csv", ".json", ".md",
            ".txt", ".tsv", ".log"
        ):
            continue

        relative = str(
            path.relative_to(ROOT)
        )

        lower = relative.lower()

        score = sum(
            weight
            for key, weight in keywords.items()
            if key in lower
        )

        if score > 0:

            candidates.append({
                "path": relative,
                "size_bytes": path.stat().st_size,
                "priority_score": score,
            })


candidates.sort(
    key=lambda r: (
        -r["priority_score"],
        r["path"]
    )
)

print(
    "\nTop candidate primary-evidence files:"
)

for item in candidates[:18]:

    print(
        f"  priority={item['priority_score']} "
        f"{item['path']}"
    )

print(
    "Total indexed candidate files:",
    len(candidates)
)


# ========================================
# WRITE A HUMAN-READABLE EVIDENCE BRIEF
# ========================================

lines = [
    "# Thesis Evidence Consolidation",
    "",
    "## Working research title",
    "",
    "Evaluating the Reliability of EEG-Based "
    "Alzheimer's Disease and Frontotemporal "
    "Dementia Classification Through Temporal "
    "Stability and Calibration",
    "",
    "## Primary research",
    "",
    "The main study examines whether EEG-based "
    "dementia classifiers remain accurate and "
    "consistent across recording time, rather "
    "than evaluating classification accuracy alone.",
    "",
    "The original cohort contains 23 Alzheimer's "
    "disease (AD), 18 frontotemporal dementia "
    "(FTD), and 23 healthy-control participants.",
    "",
    "### Previously audited primary findings",
    "",
    "| Result | AD vs healthy | FTD vs healthy |",
    "|---|---:|---:|",
    "| First-minute balanced accuracy | 0.7348 | 0.7372 |",
    "| Unstable five-minute predictions | 44.35% | 33.66% |",
    "| Stable-but-wrong predictions | 8.70% | 13.17% |",
    "",
    "These primary numbers were reported in the "
    "earlier Milestone 4C audit. Their exact "
    "machine-readable source files still need "
    "to be reconciled before final chapter sign-off.",
    "",
    "### Additional primary findings",
    "",
    "The matched acquisition-condition analysis "
    "reported eyes-open minus eyes-closed balanced "
    "accuracy differences of -0.0883 for AD and "
    "-0.0423 for FTD. Both confidence intervals "
    "included zero, so a statistically confirmed "
    "condition effect was not established.",
    "",
    "Eight feature-ablation comparisons were checked; "
    "none remained significant after Holm correction.",
    "",
    "## Independent Isfahan MCI extension",
    "",
    "This is a separate MCI-versus-healthy task, "
    "not direct external validation of AD diagnosis.",
    "",
    "Of 61 recordings, 49 passed the fixed five-minute "
    "signal-quality screen (23 MCI and 26 healthy).",
    "",
    "| Measure | EEG_1 | EEG_2 |",
    "|---|---:|---:|",
]

for label, key in (
    ("First-minute balanced accuracy",
     "model_first_minute_BACC"),
    ("Stable-but-wrong",
     "model_stable_wrong"),
    ("Unstable",
     "model_unstable"),
    ("Always-MCI baseline BACC",
     "always_MCI_BACC"),
):

    one = float(
        control_by_release["EEG_1"][key]
    )

    two = float(
        control_by_release["EEG_2"][key]
    )

    lines.append(
        f"| {label} | {one:.4f} | {two:.4f} |"
    )

lines.extend([
    "",
    "EEG_1 was strongly biased toward predicting "
    "MCI. Constant-label negative controls showed "
    "that stable incorrect predictions can occur "
    "even without useful diagnostic discrimination.",
    "",
    "The two Isfahan releases were assessed separately. "
    "Cross-release participant identity remains "
    "unconfirmed. EDF physical-unit fields were "
    "blank, and microvolts were assumed.",
    "",
    "## Figshare independent AD data feasibility",
    "",
    "The archive contains 471 MATLAB files, "
    "including 305 AD, 56 MCI, and 110 control files.",
    "",
    "There were 44 AD recording-folder proxies "
    "with a standard 19-column file of nominal "
    "duration at least 60 seconds, but only "
    "one comparable healthy-control proxy.",
    "",
    "Most other controls used a different "
    "22-row format with an unverified common "
    "electrode mapping. Thus the original "
    "19-channel, 60-second external AD validation "
    "was not performed.",
    "",
    "This is a dataset compatibility limitation, "
    "not a negative classification result.",
    "",
    "## Current scientific conclusion",
    "",
    "EEG-based dementia predictions can be "
    "reasonably accurate at one time point "
    "yet change over time or remain consistently "
    "incorrect. Temporal consistency, diagnostic "
    "accuracy, and probability reliability are "
    "different properties and should be "
    "evaluated separately.",
    "",
    "The available findings do not establish "
    "clinical diagnostic safety or broad "
    "cross-dataset generalization.",
    "",
    "## Remaining thesis tasks",
    "",
    "1. Reconcile original AD/FTD result numbers "
    "with their saved source tables.",
    "2. Prepare final reproducible tables and figures.",
    "3. Finalize Methods and Results chapters.",
    "4. Write Discussion, limitations, and conclusion.",
    "5. Complete supervisor review and submission.",
    "",
])

freeze(
    OUT / "THESIS_EVIDENCE_BRIEF.md",
    "\n".join(lines) + "\n"
)


# ========================================
# SAVE SOURCE INVENTORIES
# ========================================

if candidates:

    import io

    buf = io.StringIO()

    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "path",
            "size_bytes",
            "priority_score"
        ],
        lineterminator="\n"
    )

    writer.writeheader()
    writer.writerows(candidates)

    freeze(
        OUT / "PRIMARY_SOURCE_CANDIDATES.csv",
        buf.getvalue()
    )

source_hashes = {
    name: digest(path)
    for name, path in FILES.items()
}

freeze(
    OUT / "VERIFIED_INPUT_SHA256.json",
    json.dumps(
        source_hashes,
        indent=2,
        sort_keys=True
    ) + "\n"
)


# ========================================
# COMPLETION
# ========================================

print(
    "\n===== MILESTONE 7A COMPLETE ====="
)

print(
    "Primary source candidates indexed:",
    len(candidates)
)

print(
    "MCI and Figshare summaries verified."
)

print(
    "Original AD/FTD figures carried forward "
    "from the previous audit, pending "
    "direct source-file reconciliation."
)

print(
    "Evidence brief:",
    OUT / "THESIS_EVIDENCE_BRIEF.md"
)

print(
    "No model training or previous "
    "results modified."
)

