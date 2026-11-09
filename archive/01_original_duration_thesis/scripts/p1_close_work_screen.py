from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
from datetime import datetime, timezone

import csv
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/literature"
OUT.mkdir(parents=True, exist_ok=True)


PAPERS = [
    (
        "2019 window length",
        "10.3390/brainsci9040081",
        "EEG epoch-length effects in AD",
        "Cannot claim first AD EEG duration analysis"
    ),
    (
        "2020 recording duration",
        "10.1177/1550059420983624",
        "Recording-duration effects in aging and MCI",
        "Must distinguish new contribution from duration stability"
    ),
    (
        "2021 EEG recommendations",
        "10.1002/alz.12311",
        "Standardization and repeatability of resting EEG",
        "Must justify acquisition and biomarker protocols"
    ),
    (
        "2021 independent cohort",
        "10.1007/s11517-021-02427-6",
        "Adaptive-filter novelty features for AD/MCI",
        "Must separate signal novelty detection from adaptive stopping"
    ),
    (
        "2023 sequential diagnosis",
        "10.1038/s41598-023-32867-z",
        "Cost-aware sequential dementia assessment",
        "Cannot claim adaptive assessment is wholly new"
    ),
    (
        "2025 CNN duration",
        "10.1186/s40708-025-00260-3",
        "CNN and EEG segment-length analysis",
        "Must establish an advantage beyond duration benchmarking"
    ),
    (
        "2026 AD-FTD EEG",
        "10.1186/s12984-026-01897-9",
        "Spectral, nonlinear and graph-based AD/FTD analysis",
        "Cannot claim first multi-dimensional AD/FTD EEG comparison"
    ),
    (
        "2026 cross-dataset model",
        "10.1038/s41598-026-57069-1",
        "Explainable EEG classification and cross-dataset evaluation",
        "External evaluation alone is not novel"
    ),
    (
        "2026 graph fusion",
        "10.1177/20552076261495018",
        "Prior-constrained dynamic graphs and multi-expert fusion",
        "Another graph architecture is insufficient novelty"
    ),
    (
        "2026 AHEPA review",
        "10.1007/s11571-026-10464-w",
        "Evaluation rigor across the AHEPA benchmark",
        "Subject-level validation is mandatory, not novel"
    )
]


def fetch_paper(item):
    label, doi, overlap, implication = item

    row = {
        "screen_label": label,
        "doi": doi,
        "published_title": "",
        "year": "",
        "journal": "",
        "metadata_status": "UNVERIFIED",
        "known_overlap_to_review": overlap,
        "implication_for_our_study": implication,
        "full_methods_review": "PENDING"
    }

    try:
        url = (
            "https://api.crossref.org/works/"
            + quote(doi, safe="")
        )

        request = Request(
            url,
            headers={"User-Agent": "alzheimers-thesis-p1/1.0"}
        )

        with urlopen(request, timeout=25) as response:
            data = json.load(response)["message"]

        dates = (
            data.get("published")
            or data.get("issued")
            or {}
        )

        year_parts = dates.get("date-parts") or [[None]]

        row["published_title"] = (
            data.get("title") or [""]
        )[0]

        row["journal"] = (
            data.get("container-title") or [""]
        )[0]

        row["year"] = year_parts[0][0]
        row["metadata_status"] = "RETRIEVED"

    except Exception as exc:
        row["metadata_status"] = (
            "UNVERIFIED: " + str(exc)[:120]
        )

    return row


with ThreadPoolExecutor(max_workers=4) as executor:
    rows = list(executor.map(fetch_paper, PAPERS))


csv_path = OUT / "p1_close_work_evidence_2026-11-07.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)


note = """# P1 Literature Correction and Gap Status

Date: 2026-11-07
Stage: Exploratory literature screening

## Citation correction

The earlier DOI 10.1186/s40537-025-01107-0
did not resolve.

The correct verified 2025 Brain Informatics article DOI is:

10.1186/s40708-025-00260-3

The original metadata screening file is preserved unchanged.

## Novelty status

The following are NOT independently novel contributions:

- changing EEG window length alone
- applying a CNN or transformer alone
- adding generic uncertainty estimation
- using graph features alone
- validating on a second dataset alone
- using standard participant-level cross-validation

## Candidate methodological question

Can an acquisition-aware system make reliable decisions
using only EEG already observed, and reduce recording
burden relative to properly matched fixed-duration and
adaptive baselines?

## Important distinction

Novelty detection from adaptive filtering is not
necessarily sequential acquisition stopping.

Full-text methods must be compared before a novelty
claim is approved.

## External-validation risk

An external cohort may be confounded by acquisition
differences between diagnostic groups.

A different dataset is not automatically a valid
independent clinical validation.

## Status

RESEARCH GAP: PROVISIONAL

FINAL METHODOLOGY: NOT FROZEN

CONFIRMATORY CLAIM: NOT ESTABLISHED
"""

(OUT / "p1_gap_status_2026-11-07.md").write_text(note)


print("===== CLOSEST-WORK LITERATURE SCREEN =====")
print("Papers:", len(rows))
print(
    "DOI metadata retrieved:",
    sum(x["metadata_status"] == "RETRIEVED" for x in rows)
)

for item in rows:
    print()
    print(item["screen_label"])
    print("DOI:", item["doi"])
    print("Status:", item["metadata_status"])
    print("Title:", item["published_title"])
    print("Overlap:", item["known_overlap_to_review"])
    print("Implication:", item["implication_for_our_study"])

print()
print("Outputs:", csv_path)
print("RESEARCH GAP STATUS: PROVISIONAL")
print("P1 CLOSEST-WORK SCREEN COMPLETE")
