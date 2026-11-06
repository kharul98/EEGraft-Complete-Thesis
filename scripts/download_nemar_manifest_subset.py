from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import hashlib
import json
import os
import shutil
import sys
import time


DATASET_ID = "on004504"
VERSION = "v1.0.0"

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "eeg" / f"{DATASET_ID}_{VERSION}"
DOCS = ROOT / "docs" / "dataset"
AUDIT = ROOT / "docs" / "audit"

MANIFEST_URL = (
    f"https://data.nemar.org/{DATASET_ID}/{VERSION}/manifest.json"
)

MANIFEST_PATH = DOCS / f"{DATASET_ID}_{VERSION}_manifest.json"
SELECTED_PATH = DOCS / f"{DATASET_ID}_{VERSION}_selected_files.tsv"

WORKERS = 8

ROOT_METADATA = {
    "participants.tsv",
    "participants.json",
    "dataset_description.json",
    "README",
    "README.md",
    "CHANGES",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def get_manifest() -> list[dict]:
    print("Manifest URL:", MANIFEST_URL)

    req = Request(
        MANIFEST_URL,
        headers={"User-Agent": "alzheimers-thesis-research/1.0"},
    )

    with urlopen(req, timeout=60) as response:
        content = response.read()

    MANIFEST_PATH.write_bytes(content)

    manifest = json.loads(content)

    if not isinstance(manifest, list):
        raise RuntimeError(
            f"Expected manifest JSON array; got {type(manifest).__name__}"
        )

    return manifest


def get_path(item: dict) -> str:
    for key in ("path", "file", "name", "bids_path"):
        value = item.get(key)

        if isinstance(value, str) and value:
            return value.lstrip("/")

    return ""


def get_url(item: dict, path: str) -> str:
    value = item.get("url")

    if isinstance(value, str) and value:
        if value.startswith(("https://", "http://")):
            return value

        return urljoin(
            f"https://data.nemar.org/{DATASET_ID}/{VERSION}/",
            value,
        )

    return (
        f"https://data.nemar.org/"
        f"{DATASET_ID}/{VERSION}/{path}"
    )


def get_expected_size(item: dict) -> int | None:
    for key in (
        "size",
        "size_bytes",
        "bytes",
        "content_length",
    ):
        value = item.get(key)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    return None


def selected(item: dict) -> bool:
    path = get_path(item)

    return (
        path.startswith("derivatives/")
        or path in ROOT_METADATA
    )


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]

    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return str(value)


def download_one(item: dict) -> tuple[str, str, int]:
    path = get_path(item)
    url = get_url(item, path)
    expected = get_expected_size(item)

    destination = OUT_ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not destination.is_symlink():
        local_size = destination.stat().st_size

        if expected is None or local_size == expected:
            return path, "already-valid", local_size

    temp = destination.with_name(destination.name + ".part")

    if temp.exists():
        temp.unlink()

    last_error = None

    for attempt in range(1, 4):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent":
                        "alzheimers-thesis-research/1.0"
                },
            )

            with urlopen(request, timeout=180) as response:
                with temp.open("wb") as out:
                    shutil.copyfileobj(
                        response,
                        out,
                        length=1024 * 1024,
                    )

            actual = temp.stat().st_size

            if expected is not None and actual != expected:
                raise RuntimeError(
                    f"Size mismatch for {path}: "
                    f"expected {expected}, got {actual}"
                )

            os.replace(temp, destination)

            return path, "downloaded", actual

        except Exception as exc:
            last_error = exc

            if temp.exists():
                temp.unlink()

            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed {path}: {last_error!r}"
    )


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = get_manifest()

    print()
    print("Manifest entries:", len(manifest))

    if manifest:
        print("Manifest first-entry keys:")
        print(sorted(manifest[0].keys()))

    files = [
        item
        for item in manifest
        if selected(item)
    ]

    derivative_files = [
        item
        for item in files
        if get_path(item).startswith("derivatives/")
    ]

    print("Selected total files:", len(files))
    print("Selected derivative files:", len(derivative_files))

    expected_known = [
        get_expected_size(item)
        for item in files
        if get_expected_size(item) is not None
    ]

    if expected_known:
        print(
            "Known selected byte total:",
            human_bytes(sum(expected_known)),
        )

    with SELECTED_PATH.open("w", encoding="utf-8") as f:
        f.write("path\texpected_size\turl\n")

        for item in files:
            path = get_path(item)
            size = get_expected_size(item)
            url = get_url(item, path)

            f.write(
                f"{path}\t"
                f"{'' if size is None else size}\t"
                f"{url}\n"
            )

    print()
    print("Downloading with", WORKERS, "workers...")

    completed = []
    failures = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_map = {
            pool.submit(download_one, item): item
            for item in files
        }

        done = 0

        for future in as_completed(future_map):
            item = future_map[future]
            path = get_path(item)

            try:
                result = future.result()
                completed.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "path": path,
                        "error": repr(exc),
                    }
                )

            done += 1

            if done % 20 == 0 or done == len(files):
                print(
                    f"Completed {done}/{len(files)} "
                    f"(failures={len(failures)})"
                )

    failure_path = (
        AUDIT
        / f"{DATASET_ID}_{VERSION}_download_failures.json"
    )

    failure_path.write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )

    print()
    print("----- DOWNLOAD RESULT -----")
    print("Successful:", len(completed))
    print("Failed:", len(failures))

    downloaded_bytes = sum(
        size for _, _, size in completed
    )

    print(
        "Materialized selected bytes:",
        human_bytes(downloaded_bytes),
    )

    print(
        "Manifest SHA256:",
        sha256_file(MANIFEST_PATH),
    )

    print()
    print("----- LOCAL DERIVATIVE COUNTS -----")

    set_files = list(
        OUT_ROOT.rglob("derivatives/**/*.set")
    )

    fdt_files = list(
        OUT_ROOT.rglob("derivatives/**/*.fdt")
    )

    print(".set:", len(set_files))
    print(".fdt:", len(fdt_files))

    regular_files = [
        p
        for p in (OUT_ROOT / "derivatives").rglob("*")
        if p.is_file() and not p.is_symlink()
    ]

    print(
        "Regular derivative files:",
        len(regular_files),
    )

    if failures:
        print()
        print("FIRST FAILURES:")

        for failure in failures[:10]:
            print(failure)

        sys.exit(1)

    print()
    print("MATERIALIZATION SUCCESSFUL")


if __name__ == "__main__":
    main()
