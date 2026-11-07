#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import hashlib
import json
import os
import time


PROJECT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT
    / "docs/dataset/on004504_v1.0.0_manifest.json"
)

DATA = (
    PROJECT
    / "data/eeg/on004504_v1.0.0"
)

OUT = (
    PROJECT
    / "docs/audit/p1_raw_manifest_download_summary.json"
)

MANIFEST_SHA256 = (
    "f15e50c76c36422bc3e8b4061b0dadacaaf4c1e5df02e676eee131a5b64102e2"
)

WORKERS = 5
CHUNK_SIZE = 2 * 1024 * 1024


def verify(path: Path, entry: dict) -> bool:

    if not path.is_file():
        return False

    if path.is_symlink():
        return False

    if path.stat().st_size != entry["size"]:
        return False

    algorithm = entry["checksum_algorithm"].lower()

    if algorithm == "sha256":
        h = hashlib.sha256()

    elif algorithm == "git":
        h = hashlib.sha1()

        h.update(
            f"blob {entry['size']}\0".encode()
        )

    else:
        raise ValueError(
            f"Unsupported checksum algorithm: {algorithm}"
        )

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(CHUNK_SIZE),
            b""
        ):
            h.update(block)

    return (
        h.hexdigest().lower()
        == entry["checksum"].lower()
    )


def download(entry: dict) -> dict:

    relative = entry["path"]

    if (
        not relative.startswith("sub-")
        or "/eeg/" not in relative
        or ".." in Path(relative).parts
        or Path(relative).is_absolute()
    ):
        raise ValueError(
            f"Unsafe or unexpected path: {relative}"
        )

    url = entry["bytes_url"]
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.netloc != "data.nemar.org"
        or not parsed.path.startswith(
            "/on004504/v1.0.0/sub-"
        )
    ):
        raise ValueError(
            f"Unexpected source URL: {url}"
        )

    destination = DATA / relative

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():

        if verify(destination, entry):
            return {
                "path": relative,
                "status": "already-verified",
                "size": entry["size"]
            }

        raise RuntimeError(
            f"Existing file checksum mismatch: {relative}"
        )

    partial = destination.with_name(
        destination.name + ".part"
    )

    expected_size = int(
        entry["size"]
    )

    error = ""

    for attempt in range(1, 5):

        try:

            offset = (
                partial.stat().st_size
                if partial.exists()
                else 0
            )

            if offset > expected_size:
                raise RuntimeError(
                    f"Partial file exceeds expected size: {relative}"
                )

            if offset < expected_size:

                headers = {
                    "User-Agent":
                        "alzheimers-thesis-p1/1.0"
                }

                if offset:
                    headers["Range"] = f"bytes={offset}-"

                request = Request(
                    url,
                    headers=headers
                )

                with urlopen(
                    request,
                    timeout=180
                ) as response:

                    status = getattr(
                        response,
                        "status",
                        200
                    )

                    append = bool(
                        offset and status == 206
                    )

                    mode = (
                        "ab" if append else "wb"
                    )

                    with partial.open(mode) as output:

                        while True:

                            block = response.read(
                                CHUNK_SIZE
                            )

                            if not block:
                                break

                            output.write(block)

            if not verify(partial, entry):
                raise RuntimeError(
                    f"Partial checksum mismatch: {relative}"
                )

            os.replace(
                partial,
                destination
            )

            return {
                "path": relative,
                "status": "downloaded-verified",
                "size": expected_size
            }

        except Exception as exc:

            error = str(exc)

            if attempt < 4:
                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        f"{relative}: {error}"
    )


def main() -> int:

    manifest_bytes = MANIFEST.read_bytes()

    actual_sha = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    if actual_sha != MANIFEST_SHA256:
        raise RuntimeError(
            "Pinned manifest hash mismatch"
        )

    manifest = json.loads(
        manifest_bytes
    )

    entries = [
        entry
        for entry in manifest
        if entry["path"].startswith("sub-")
    ]

    raw = [
        entry
        for entry in entries
        if entry["path"].endswith("_eeg.set")
    ]

    if len(raw) != 88 or len(entries) != 264:
        raise RuntimeError(
            f"Unexpected selection: "
            f"{len(raw)} raw, {len(entries)} total"
        )

    if len({
        entry["path"]
        for entry in entries
    }) != len(entries):
        raise RuntimeError(
            "Duplicate paths in manifest"
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "===== RAW EEG AND SIDECAR DOWNLOAD =====",
        flush=True
    )

    print(
        "Selected files:", len(entries),
        "Raw recordings:", len(raw),
        "Total GB:",
        round(
            sum(x["size"] for x in entries) / 1e9,
            3
        ),
        flush=True
    )

    success = []
    failures = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        jobs = {
            executor.submit(download, entry):
                entry["path"]
            for entry in entries
        }

        for index, future in enumerate(
            as_completed(jobs),
            1
        ):

            try:
                success.append(
                    future.result()
                )

            except Exception as exc:

                failures.append({
                    "path": jobs[future],
                    "error": str(exc)[:500]
                })

                print(
                    "FAILED",
                    jobs[future],
                    str(exc)[:160],
                    flush=True
                )

            if (
                index % 20 == 0
                or index == len(entries)
            ):

                print(
                    f"Progress: {index}/{len(entries)}",
                    f"verified={len(success)}",
                    f"failed={len(failures)}",
                    flush=True
                )

    report = {
        "run_utc":
            datetime.now(timezone.utc).isoformat(),

        "manifest_sha256":
            MANIFEST_SHA256,

        "selected":
            len(entries),

        "raw_eeg_count":
            len(raw),

        "verified":
            len(success),

        "failures":
            failures,

        "already_verified":
            sum(
                x["status"] == "already-verified"
                for x in success
            ),

        "downloaded_verified":
            sum(
                x["status"] == "downloaded-verified"
                for x in success
            ),

        "bytes_total":
            sum(x["size"] for x in entries),

        "note":
            "Checksum-verified raw acquisition; no modeling."
    }

    OUT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print(
        "===== RAW DOWNLOAD SUMMARY =====",
        flush=True
    )

    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "failures"
            },
            indent=2
        ),
        flush=True
    )

    if failures:
        print(
            "INCOMPLETE: Failed files recorded.",
            flush=True
        )
        return 2

    print(
        "ALL 264 RAW EEG/SIDECAR HASHES VERIFIED",
        flush=True
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
