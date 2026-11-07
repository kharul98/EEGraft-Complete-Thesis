#!/usr/bin/env python3
"""Milestone 5A: independent EEG dataset access and metadata preflight."""

from pathlib import Path
from urllib.request import Request, urlopen
import importlib.util
import json
import re
import shutil

import mne


ROOT = Path.cwd()

OUT = (
    ROOT / "reports/thesis_final"
    / "milestone_5a_external_preflight_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

ISFAHAN_URL = (
    "https://misp.mui.ac.ir/en/eeg-data-0"
)

ISFAHAN_FILES = {
    "EEG_1.zip":
        "1hG5v_COjPNzejRaL9XJAFERee9i2_V04",

    "EEG_2.zip":
        "1gS7b9C1XuDp7c-yGgbDzZePIJxI9eRwF"
}

FIGSHARE_API = (
    "https://api.figshare.com/v2/articles/5450293"
)

PREVIOUS_REPORT = (
    ROOT / "reports/thesis_final"
    / "milestone_4c_reproducibility_v1"
    / "THESIS_EVIDENCE_SUMMARY.md"
)


def get_text(url, limit=500000):

    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 Academic EEG Research"
        }
    )

    with urlopen(
        request,
        timeout=20
    ) as response:

        data = response.read(
            limit + 1
        )

    if len(data) > limit:
        raise RuntimeError(
            "Metadata response exceeds limit"
        )

    return data.decode(
        "utf-8-sig",
        errors="replace"
    )


print(
    "===== MILESTONE 5A: "
    "INDEPENDENT EEG DATASET PREFLIGHT =====",
    flush=True
)

# ==========================================
# 1. CONFIRM COMPLETED THESIS EVIDENCE
# ==========================================

if not PREVIOUS_REPORT.is_file():

    raise RuntimeError(
        "Milestone 4C report is missing"
    )

print(
    "PASS: Existing thesis evidence "
    "package is present."
)


# ==========================================
# 2. AUDIT OFFICIAL ISFAHAN SOURCE
# ==========================================

print(
    "\n===== ISFAHAN UNIVERSITY EEG REPOSITORY ====="
)

isfahan = {
    "official_page": ISFAHAN_URL,
    "page_reachable": False,
    "public_drive_links": {},
    "possible_subject_counts": {
        "MCI": 29,
        "healthy": 32,
        "total": 61
    },
    "unique_subjects_verified": False,
    "actual_EEG_files_verified": False,
    "format": "EDF reported; not yet locally verified"
}

try:

    html = get_text(
        ISFAHAN_URL
    )

    isfahan[
        "page_reachable"
    ] = True

    online_ids = set(
        re.findall(
            r"drive\.google\.com/file/d/"
            r"([A-Za-z0-9_-]+)",
            html
        )
    )

    print(
        "Official university page: REACHABLE"
    )

    print(
        "Google Drive file links found:",
        len(online_ids)
    )

except Exception as exc:

    online_ids = set()

    print(
        "University page request failed:",
        type(exc).__name__,
        str(exc)[:180]
    )

    print(
        "This does not prove the dataset "
        "is unavailable."
    )


for filename, file_id in ISFAHAN_FILES.items():

    url = (
        "https://drive.google.com/file/d/"
        + file_id
        + "/view"
    )

    on_page = (
        file_id in online_ids
    )

    isfahan[
        "public_drive_links"
    ][filename] = {
        "url": url,
        "file_id": file_id,
        "found_on_current_page":
            on_page
    }

    print(
        filename,
        "-",
        url
    )

    if isfahan["page_reachable"]:

        print(
            "  Current page link verified:",
            on_page
        )


print(
    "\nReported potential cohort: "
    "29 MCI + 32 healthy = 61 recordings."
)

print(
    "Important: participant uniqueness "
    "across both archives remains UNVERIFIED."
)

print(
    "No clinical-label or raw EEG "
    "files inspected yet."
)


# ==========================================
# 3. CHECK FIGSHARE ARTICLE METADATA
# ==========================================

print(
    "\n===== FIGSHARE AD/MCI/CONTROL BACKUP ====="
)

figshare = {
    "article_id": 5450293,
    "api_url": FIGSHARE_API,
    "api_reachable": False,
    "article_title": None,
    "license": None,
    "files": [],
    "raw_EEG_compatibility_verified": False
}

try:

    article = json.loads(
        get_text(
            FIGSHARE_API,
            limit=2000000
        )
    )

    figshare[
        "api_reachable"
    ] = True

    figshare[
        "article_title"
    ] = article.get(
        "title"
    )

    license_info = article.get(
        "license"
    )

    if isinstance(
        license_info,
        dict
    ):

        figshare[
            "license"
        ] = {
            "name":
                license_info.get("name"),

            "url":
                license_info.get("url")
        }

    else:

        figshare[
            "license"
        ] = license_info

    print(
        "Figshare API: REACHABLE"
    )

    print(
        "Article:",
        figshare["article_title"]
    )

    print(
        "License:",
        figshare["license"]
    )

    for file in article.get(
        "files",
        []
    ):

        item = {
            "name":
                file.get("name"),

            "id":
                file.get("id"),

            "size_bytes":
                file.get("size"),

            "download_url":
                file.get("download_url")
        }

        figshare[
            "files"
        ].append(item)

        size = item["size_bytes"]

        size_gib = (
            round(
                size / (1024 ** 3),
                3
            )
            if isinstance(
                size,
                (int, float)
            )
            else "unknown"
        )

        print(
            "File:",
            item["name"],
            "| GiB:",
            size_gib,
            "| ID:",
            item["id"]
        )

except Exception as exc:

    print(
        "Figshare metadata request failed:",
        type(exc).__name__,
        str(exc)[:180]
    )

    print(
        "This does not prove "
        "Figshare data are unavailable."
    )


print(
    "Caution: Published analyses describe "
    "different AD/MCI and control "
    "recording representations."
)

print(
    "Subject identities, channel labels, "
    "and continuous recording availability "
    "must be verified before validation."
)


# ==========================================
# 4. CHECK LOCAL ENVIRONMENT
# ==========================================

print(
    "\n===== LOCAL ENVIRONMENT AND STORAGE ====="
)

free_gib = (
    shutil.disk_usage(ROOT).free
    / (1024 ** 3)
)

gdown_available = (
    importlib.util.find_spec(
        "gdown"
    ) is not None
)

edf_reader_available = hasattr(
    mne.io,
    "read_raw_edf"
)

print(
    "Available disk:",
    f"{free_gib:.2f} GiB"
)

print(
    "MNE version:",
    mne.__version__
)

print(
    "EDF reading supported:",
    edf_reader_available
)

print(
    "gdown installed:",
    gdown_available
)


# ==========================================
# 5. CHECK EXISTING LOCAL EXTERNAL DATA
# ==========================================

existing = []

data_root = (
    ROOT / "data"
)

if data_root.is_dir():

    for path in sorted(
        data_root.iterdir()
    ):

        if any(
            term in path.name.lower()
            for term in (
                "isfahan",
                "figshare",
                "mci",
                "cejnek"
            )
        ):

            existing.append(
                str(path.relative_to(ROOT))
            )

print(
    "Existing matching external data "
    "directories:",
    existing if existing else "none detected"
)


# ==========================================
# 6. SAVE THE FEASIBILITY AUDIT
# ==========================================

summary = {
    "milestone":
        "5A_external_dataset_preflight",

    "original_thesis_complete":
        True,

    "recommended_next_dataset":
        "Isfahan University MCI/healthy",

    "isfahan":
        isfahan,

    "figshare":
        figshare,

    "environment": {
        "free_disk_gib":
            round(free_gib, 3),

        "mne_version":
            mne.__version__,

        "edf_reader_available":
            edf_reader_available,

        "gdown_available":
            gdown_available,

        "existing_external_paths":
            existing
    },

    "critical_design_rules": [
        "Never treat EEG epochs as independent participants",
        "Check uniqueness across Isfahan archives",
        "Use participant-disjoint nested cross-validation",
        "Do not use MCI as an AD diagnosis",
        "Fit all scaling and calibration on training subjects",
        "Freeze test eligibility before reading predictions",
        "Do not report zero-shot AD validation on MCI",
        "Audit control recording format before Figshare transfer"
    ]
}

report = (
    OUT / "external_dataset_preflight.json"
)

report.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True
    ) + "\n",
    encoding="utf-8"
)

print(
    "\n===== MILESTONE 5A COMPLETE ====="
)

print(
    "Saved:",
    report
)

print(
    "No full EEG data downloaded."
)

print(
    "No existing models retrained, "
    "modified, or overwritten."
)

