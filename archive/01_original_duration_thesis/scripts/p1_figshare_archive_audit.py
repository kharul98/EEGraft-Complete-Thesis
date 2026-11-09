from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from collections import Counter
from io import BytesIO
from datetime import datetime, timezone
import hashlib
import json
import os
import zipfile

from scipy.io import whosmat

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT / "data/eeg/external/figshare_5450293"
)

OUT = (
    ROOT / "docs/audit/p1_figshare_archive_evidence.json"
)

API = "https://api.figshare.com/v2/articles/5450293"

DATA.mkdir(parents=True, exist_ok=True)

def load_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "alzheimers-thesis-research/1.0",
            "Accept": "application/json"
        }
    )
    with urlopen(request, timeout=90) as response:
        return json.load(response)

def md5_file(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()

def category(name):
    parts = [
        p.upper()
        for p in PurePosixPath(name).parts
    ]
    for label in ("AD", "MCI", "CONTROL"):
        if label in parts:
            return label
    return "OTHER"

print("===== FIGSHARE ARCHIVE METADATA =====", flush=True)

article = load_json(API)
files = article.get("files", [])

matching = [
    f for f in files
    if f.get("name", "").lower() == "dataset.zip"
]

if len(matching) != 1:
    raise RuntimeError("Expected one dataset.zip")

file_info = matching[0]
expected_size = int(file_info["size"])
expected_md5 = (
    file_info.get("computed_md5")
    or file_info.get("supplied_md5")
)

url = file_info["download_url"]

parsed = urlparse(url)
assert parsed.scheme == "https"
assert parsed.hostname in {
    "ndownloader.figshare.com",
    "figshare.com",
    "api.figshare.com"
}

destination = DATA / "dataset.zip"
partial = DATA / "dataset.zip.part"

print("DOI:", article.get("doi"), flush=True)
print("Expected bytes:", expected_size, flush=True)
print("Checksum available:", bool(expected_md5), flush=True)

if not destination.exists():
    offset = partial.stat().st_size if partial.exists() else 0

    if offset > expected_size:
        raise RuntimeError("Partial exceeds expected size")

    if offset < expected_size:
        headers = {
            "User-Agent": "alzheimers-thesis-research/1.0"
        }

        if offset:
            headers["Range"] = f"bytes={offset}-"

        request = Request(url, headers=headers)

        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)

            if offset and status == 206:
                content_range = response.headers.get(
                    "Content-Range", ""
                )
                if not content_range.startswith(
                    f"bytes {offset}-"
                ):
                    raise RuntimeError(
                        "Invalid HTTP resume range"
                    )
                mode = "ab"
            else:
                mode = "wb"

            with partial.open(mode) as output:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)

    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            "Incomplete download; rerun this script"
        )

    if expected_md5:
        if md5_file(partial).lower() != expected_md5.lower():
            raise RuntimeError("Figshare MD5 mismatch")

    os.replace(partial, destination)

assert destination.stat().st_size == expected_size

actual_md5 = md5_file(destination)

if expected_md5:
    assert actual_md5.lower() == expected_md5.lower()

print("Archive size: PASS", flush=True)
print(
    "Archive checksum:",
    "PASS" if expected_md5 else "NOT PROVIDED",
    flush=True
)

with zipfile.ZipFile(destination) as archive:
    members = [
        item for item in archive.infolist()
        if not item.is_dir()
    ]

    extensions = Counter(
        PurePosixPath(item.filename).suffix.lower()
        for item in members
    )

    groups = Counter(
        category(item.filename)
        for item in members
    )

    mat_files = [
        item for item in members
        if item.filename.lower().endswith(".mat")
    ]

    print(
        "Checking archive CRC integrity...",
        flush=True
    )

    bad = archive.testzip()

    if bad is not None:
        raise RuntimeError(
            "Archive CRC failure"
        )

    sample_shapes = {}

    for label in ("AD", "MCI", "CONTROL"):
        candidates = [
            item for item in mat_files
            if category(item.filename) == label
            and item.file_size < 30_000_000
        ]

        if not candidates:
            sample_shapes[label] = {
                "status": "NO_SMALL_MAT_SAMPLE"
            }
            continue

        item = candidates[0]

        try:
            data = archive.read(item)

            variables = whosmat(BytesIO(data))

            sample_shapes[label] = {
                "status": "INSPECTED",
                "variables": [
                    {
                        "variable": name,
                        "shape": list(shape),
                        "matlab_type": dtype
                    }
                    for name, shape, dtype in variables
                ]
            }

        except Exception as exc:
            sample_shapes[label] = {
                "status": "INSPECTION_FAILED",
                "reason": str(exc)[:200]
            }

report = {
    "date_utc":
        datetime.now(timezone.utc).isoformat(),
    "source_article": API,
    "doi": article.get("doi"),
    "zip_bytes": expected_size,
    "md5_verified": bool(expected_md5),
    "md5": actual_md5,
    "crc_verified": True,
    "archive_file_count": len(members),
    "mat_file_count": len(mat_files),
    "file_extensions": dict(extensions),
    "files_by_group_path": dict(groups),
    "sample_matlab_shapes": sample_shapes,
    "chronological_continuity_verified": False,
    "independent_patient_status":
        "NOT VERIFIED FROM FILE COUNTS",
    "validation_status":
        "PENDING RECORDING STRUCTURE AND DEVICE REVIEW"
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")

print("\n===== FIGSHARE DATASET STRUCTURE =====")
print("Archive CRC: PASS")
print("Files:", report["archive_file_count"])
print("MATLAB files:", report["mat_file_count"])
print("Group path counts:", dict(groups))
print("File extensions:", dict(extensions))

print("\n===== MATLAB STRUCTURE SAMPLES =====")
for label, result in sample_shapes.items():
    print(label, json.dumps(result))

print("\nCONTINUOUS RECORDING STATUS: UNVERIFIED")
print("INDEPENDENT VALIDATION STATUS: PENDING")
print("FIGSHARE ARCHIVE AUDIT COMPLETE")
