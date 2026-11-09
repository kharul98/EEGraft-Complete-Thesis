from pathlib import Path
import json
import shutil
import stat
import subprocess
import zipfile
import re

HOME = Path.home()
WORKSPACE = HOME / "research" / "eegraft-caueeg"
DATA_ROOT = HOME / "research" / "caueeg"

SOURCE_CONFIG = WORKSPACE / "protocol" / "dataset_source.json"
LAYOUT_CONFIG = WORKSPACE / "protocol" / "dataset_layout.json"

EXTRACT_DIR = DATA_ROOT / "imported_caueeg"


def gb(value):
    return round(value / (1024 ** 3), 2)


def inspect_archive(archive):
    print("\n1. Inspecting archive")

    with zipfile.ZipFile(archive) as zf:
        entries = zf.infolist()

        if not entries:
            raise ValueError("Archive contains no files.")

        total_bytes = 0
        encrypted = False
        unsafe = []
        seen = set()

        for info in entries:
            name = info.filename.replace("\\", "/")
            parts = name.split("/")

            if (
                name.startswith("/")
                or any(p == ".." for p in parts)
                or re.match(r"^[A-Za-z]:", name)
                or "\x00" in name
            ):
                unsafe.append(name)
                continue

            normalized = name.rstrip("/")

            if normalized in seen and not info.is_dir():
                unsafe.append(name)

            seen.add(normalized)

            mode = info.external_attr >> 16

            if stat.S_IFMT(mode) == stat.S_IFLNK:
                unsafe.append(name)

            if not info.is_dir():
                total_bytes += info.file_size

            if info.flag_bits & 1:
                encrypted = True

        print("Archive:", archive)
        print("Compressed size:", gb(archive.stat().st_size), "GiB")
        print("Uncompressed size:", gb(total_bytes), "GiB")
        print("Entries:", len(entries))
        print("Encrypted:", encrypted)

        if unsafe:
            print("[BLOCKED] Unsafe or duplicate archive paths detected.")
            for name in unsafe[:10]:
                print(" ", name)
            return None

        free_bytes = shutil.disk_usage(DATA_ROOT).free

        print("Available disk space:", gb(free_bytes), "GiB")

        if free_bytes < total_bytes * 1.1:
            print("[BLOCKED] Insufficient free disk space.")
            return None

        return encrypted


def find_dataset(root):
    annotations = sorted(root.rglob("annotation.json"))

    if not annotations:
        return None, 0

    for annotation in annotations:
        candidate = annotation.parent

        edf_dir = candidate / "signal" / "edf"

        if edf_dir.is_dir():
            count = sum(
                1 for path in edf_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".edf"
            )

            if count:
                return candidate, count

    return None, 0


def run():
    print("\n1. Loading registered CAUEEG archive")

    if not SOURCE_CONFIG.is_file():
        print("[BLOCKED] Run Code 05 first.")
        return

    config = json.loads(SOURCE_CONFIG.read_text())
    archive = Path(config["source_path"])

    if not archive.is_file():
        print("[BLOCKED] Archive not found:", archive)
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("[FOUND]", archive)

    print("\n2. Checking existing extraction")

    if EXTRACT_DIR.exists():
        dataset, count = find_dataset(EXTRACT_DIR)

        if dataset is not None:
            print("[FOUND] Existing extracted dataset.")
        elif any(EXTRACT_DIR.iterdir()):
            print("[BLOCKED] Extraction directory already contains files.")
            print("Directory:", EXTRACT_DIR)
            print("No files were overwritten.")
            print("Check whether an earlier extraction was interrupted.")
            return
    else:
        dataset = None
        count = 0

    if dataset is None:
        encrypted = inspect_archive(archive)

        if encrypted is None:
            return

        print("\n3. Choosing extraction tool")

        sevenzip = (
            shutil.which("7zz")
            or shutil.which("7z")
            or shutil.which("7za")
        )

        if encrypted and not sevenzip:
            print("[BLOCKED] Encrypted ZIP requires 7-Zip.")
            print("Install it using:")
            print("sudo apt update")
            print("sudo apt install -y p7zip-full")
            print("Then rerun Code 06.")
            return

        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        print("\n4. Extracting CAUEEG")
        print("Destination:", EXTRACT_DIR)
        print("Original archive will remain unchanged.")

        if sevenzip:
            print("Using:", sevenzip)
            print("Enter your archive password if prompted.")
            print("Do not paste the password into this chat.")

            command = [
                sevenzip,
                "x",
                str(archive),
                "-o" + str(EXTRACT_DIR),
                "-y",
            ]

            result = subprocess.run(command)

            if result.returncode != 0:
                print("[FAILED] Extraction returned:", result.returncode)
                print("Partial files may remain in:", EXTRACT_DIR)
                print("Do not rerun extraction over partial files.")
                return

        else:
            print("Using Python ZIP extractor.")

            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(EXTRACT_DIR)
            except Exception as exc:
                print("[FAILED]", type(exc).__name__, str(exc))
                print("Partial files may remain in:", EXTRACT_DIR)
                return

        print("[OK] Extraction command completed.")

        dataset, count = find_dataset(EXTRACT_DIR)

    print("\n5. Verifying dataset structure")

    if dataset is None:
        print("[INCOMPLETE] Could not find annotation.json")
        print("together with signal/edf/*.edf")
        print("Check the extracted directory:")
        print(EXTRACT_DIR)
        return

    print("[FOUND] Dataset root:", dataset)
    print("[FOUND] EDF recordings:", count)

    required_files = [
        "annotation.json",
        "dementia-no-overlap.json",
    ]

    for filename in required_files:
        path = dataset / filename
        print(
            "[FOUND]" if path.is_file() else "[MISSING]",
            filename,
        )

    print("\n6. Checking annotation readability")

    try:
        annotation = json.loads(
            (dataset / "annotation.json").read_text(encoding="utf-8")
        )
        print("[OK] annotation.json is valid JSON")
        print("Top-level type:", type(annotation).__name__)

    except Exception as exc:
        print("[FAILED] Annotation JSON:", exc)
        return

    if not (dataset / "dementia-no-overlap.json").is_file():
        print("[BLOCKED] Official dementia split file missing.")
        return

    if count < 1:
        print("[BLOCKED] No EDF recordings found.")
        return

    print("\n7. Saving dataset layout")

    record = {
        "dataset_root": str(dataset),
        "edf_directory": str(dataset / "signal" / "edf"),
        "annotation": str(dataset / "annotation.json"),
        "split_file": str(dataset / "dementia-no-overlap.json"),
        "edf_recording_count": count,
        "source_archive": str(archive),
    }

    if LAYOUT_CONFIG.exists():
        previous = json.loads(LAYOUT_CONFIG.read_text())

        if previous == record:
            print("[OK] Existing layout matches.")
        else:
            print("[WARNING] Existing layout differs.")
            print("Preserving original configuration.")
            return
    else:
        LAYOUT_CONFIG.write_text(json.dumps(record, indent=2))
        print("[SUCCESS] Layout configuration saved.")

    print("\nDATASET_STATUS: EXTRACTED_AND_VERIFIED")


try:
    run()
except Exception as exc:
    print("\n[ERROR]", type(exc).__name__, str(exc))
finally:
    print("\n========== CODE 06 FINISHED ==========")

