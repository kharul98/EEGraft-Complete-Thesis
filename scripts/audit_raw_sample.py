from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import hashlib
import json

import numpy as np
import mne

root = Path("data/eeg/on004504_v1.0.0")
manifest_path = Path(
    "docs/dataset/on004504_v1.0.0_manifest.json"
)

expected_manifest_sha = (
    "f15e50c76c36422bc3e8b4061b0dadacaaf4c1e5df02e676eee131a5b64102e2"
)

manifest_bytes = manifest_path.read_bytes()
actual_manifest_sha = hashlib.sha256(
    manifest_bytes
).hexdigest()

assert actual_manifest_sha == expected_manifest_sha, (
    "Manifest integrity check failed"
)

manifest = json.loads(manifest_bytes)

raw_entries = sorted(
    (
        entry for entry in manifest
        if entry["path"].startswith("sub-")
        and entry["path"].endswith("_eeg.set")
    ),
    key=lambda entry: entry["path"]
)

assert len(raw_entries) == 88

entry = raw_entries[0]
path = root / entry["path"]
url = entry["bytes_url"]

parsed = urlparse(url)

assert parsed.scheme == "https"
assert parsed.netloc == "data.nemar.org"
assert parsed.path.startswith("/on004504/v1.0.0/")

print("===== RAW SAMPLE DOWNLOAD =====", flush=True)
print("Path:", path, flush=True)
print("Expected bytes:", entry["size"], flush=True)
print("Source:", url, flush=True)

path.parent.mkdir(parents=True, exist_ok=True)

if not path.exists():
    temporary = path.with_name(path.name + ".part")

    if temporary.exists():
        raise RuntimeError(
            f"Existing partial download: {temporary}"
        )

    with urlopen(url, timeout=120) as response:
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

    if temporary.stat().st_size != entry["size"]:
        raise RuntimeError(
            "Downloaded file size mismatch"
        )

    temporary.rename(path)

if path.stat().st_size != entry["size"]:
    raise RuntimeError("Raw file size mismatch")

print("Downloaded bytes:", path.stat().st_size)

algorithm = entry["checksum_algorithm"].lower()

if algorithm == "sha256":
    h = hashlib.sha256()
elif algorithm == "git":
    h = hashlib.sha1()
    h.update(
        f"blob {path.stat().st_size}\0".encode()
    )
else:
    raise RuntimeError(
        f"Unexpected checksum algorithm: {algorithm}"
    )

with path.open("rb") as f:
    while True:
        chunk = f.read(4 * 1024 * 1024)
        if not chunk:
            break
        h.update(chunk)

assert h.hexdigest() == entry["checksum"], (
    "Raw EEG checksum mismatch"
)

print("RAW FILE CHECKSUM: PASS", flush=True)

print()
print("===== RAW EEG INSPECTION =====", flush=True)

raw = mne.io.read_raw_eeglab(
    str(path),
    preload=False,
    verbose="ERROR"
)

print("Sampling rate:", raw.info["sfreq"])
print("Channels:", len(raw.ch_names))
print("Channel names:", raw.ch_names)
print("Samples:", raw.n_times)
print("Duration seconds:", round(
    raw.n_times / raw.info["sfreq"], 3
))
print("Annotations:", len(raw.annotations))

sample = raw.get_data(
    start=0,
    stop=min(
        raw.n_times,
        int(raw.info["sfreq"] * 5)
    )
)

assert np.isfinite(sample).all()

print("First 5s finite:", True)

print()
print("===== DERIVATIVE COMPARISON =====", flush=True)

candidates = list(
    (root / "derivatives").rglob(path.name)
)

print("Matching derivative files:", len(candidates))

if len(candidates) == 1:
    derivative = mne.io.read_raw_eeglab(
        str(candidates[0]),
        preload=False,
        verbose="ERROR"
    )

    print("Derivative sampling rate:",
          derivative.info["sfreq"])
    print("Derivative channels:",
          len(derivative.ch_names))
    print("Derivative duration seconds:",
          round(
              derivative.n_times
              / derivative.info["sfreq"], 3
          ))

    print(
        "Channel layout identical:",
        raw.ch_names == derivative.ch_names
    )
else:
    print(
        "Derivative matching needs inspection."
    )

print()
print("===== RAW SAMPLE AUDIT COMPLETE =====")
