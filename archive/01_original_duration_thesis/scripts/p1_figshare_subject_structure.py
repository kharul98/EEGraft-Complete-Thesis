from pathlib import Path, PurePosixPath
from collections import Counter
from io import BytesIO
import json
import zipfile

from scipy.io import whosmat

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/eeg/external/figshare_5450293/dataset.zip"
OUT = ROOT / "docs/audit/p1_figshare_subject_structure.json"

GROUPS = ("AD", "MCI", "CONTROL")

def group_and_folder(path):
    parts = PurePosixPath(path).parts
    upper = [part.upper() for part in parts]

    for group in GROUPS:
        if group in upper:
            index = upper.index(group)
            folder = "/".join(parts[index + 1:-1])
            return group, folder or "(none)"

    return "OTHER", "(unknown)"

if not ARCHIVE.is_file():
    raise RuntimeError("Verified Figshare archive is missing")

report = {
    "archive": str(ARCHIVE.relative_to(ROOT)),
    "groups": {},
    "chronological_continuity_verified": False,
    "subject_identity_verified": False
}

with zipfile.ZipFile(ARCHIVE) as archive:
    entries = sorted(
        (
            item for item in archive.infolist()
            if not item.is_dir()
            and item.filename.lower().endswith(".mat")
        ),
        key=lambda item: item.filename
    )

    print("===== FIGSHARE SUBJECT STRUCTURE =====")
    print("MATLAB files:", len(entries))

    for group in GROUPS:
        selected = [
            item for item in entries
            if group_and_folder(item.filename)[0] == group
        ]

        folders = Counter(
            group_and_folder(item.filename)[1]
            for item in selected
        )

        indices = sorted({
            0,
            len(selected) // 3,
            (2 * len(selected)) // 3,
            len(selected) - 1
        }) if selected else []

        samples = []

        for index in indices:
            item = selected[index]

            sample = {
                "path": item.filename,
                "uncompressed_bytes": item.file_size
            }

            try:
                if item.file_size > 50_000_000:
                    raise ValueError("Sample exceeds 50 MB inspection cap")

                variables = whosmat(
                    BytesIO(archive.read(item))
                )

                sample["variables"] = [
                    {
                        "name": name,
                        "shape": list(shape),
                        "dtype": dtype
                    }
                    for name, shape, dtype in variables
                ]
            except Exception as exc:
                sample["inspection_error"] = str(exc)[:200]

            samples.append(sample)

        report["groups"][group] = {
            "mat_file_count": len(selected),
            "distinct_parent_folders": len(folders),
            "folder_file_counts": dict(folders),
            "samples": samples
        }

        print()
        print("GROUP:", group)
        print("MAT files:", len(selected))
        print("Distinct parent folders:", len(folders))
        print("First folder counts:", folders.most_common(12))

        for sample in samples:
            print(
                "SAMPLE:",
                sample["path"],
                sample.get("variables", sample.get("inspection_error"))
            )

OUT.write_text(json.dumps(report, indent=2) + "\n")

print()
print("IMPORTANT: Folders are NOT verified participant IDs.")
print("Chronological acquisition order: UNVERIFIED")
print("FIGSHARE STRUCTURE AUDIT COMPLETE")
