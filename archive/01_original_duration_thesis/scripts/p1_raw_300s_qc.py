from pathlib import Path
import numpy as np
import pandas as pd
import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/eeg/on004504_v1.0.0"
OUT = ROOT / "results/p1_raw_300s_quality.csv"

inventory = pd.read_csv(
    ROOT / "results/p1_raw_recording_inventory.csv"
)

assert len(inventory) == 88
assert inventory.participant_id.nunique() == 88

rows = []

print("===== RAW EEG QUALITY SCREEN =====", flush=True)

for index, item in enumerate(
    inventory.itertuples(index=False), 1
):
    row = {
        "participant_id": item.participant_id,
        "group": item.group,
        "error": ""
    }

    try:
        path = ROOT / item.path

        raw = mne.io.read_raw_eeglab(
            str(path),
            preload=False,
            verbose="ERROR"
        )

        sf = float(raw.info["sfreq"])
        assert sf == 500.0

        x = raw.get_data(
            picks="eeg",
            start=0,
            stop=int(sf * 300)
        )

        assert x.shape == (19, 150000)

        finite = np.isfinite(x)
        row["finite_fraction"] = float(finite.mean())

        if not finite.all():
            raise ValueError("Nonfinite EEG samples")

        uv = x * 1e6

        channel_rms = np.sqrt(
            np.mean(uv * uv, axis=1)
        )

        channel_std = np.std(uv, axis=1)

        row.update({
            "median_channel_rms_uv":
                float(np.median(channel_rms)),
            "p99_abs_uv":
                float(np.percentile(np.abs(uv), 99)),
            "max_abs_uv":
                float(np.max(np.abs(uv))),
            "fraction_over_200uv":
                float(np.mean(np.abs(uv) > 200)),
            "near_flat_channels":
                int(np.sum(channel_std < 0.05)),
            "annotations":
                int(len(raw.annotations))
        })

    except Exception as exc:
        row["error"] = str(exc)[:300]

    rows.append(row)

    if index % 22 == 0:
        print(
            f"Inspected {index}/88 participants",
            flush=True
        )

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\n===== RAW QUALITY SUMMARY =====")
print("Participants:", len(df))
print("Read/quality errors:", int(df.error.ne("").sum()))
print("Groups:", df.group.value_counts().to_dict())

valid = df[df.error.eq("")]

if len(valid):
    print("Finite recordings:", len(valid))
    print(
        "Median channel RMS (uV):",
        round(valid.median_channel_rms_uv.median(), 3)
    )
    print(
        "Median 99th-percentile amplitude (uV):",
        round(valid.p99_abs_uv.median(), 3)
    )
    print(
        "Subjects with any >200 uV samples:",
        int((valid.fraction_over_200uv > 0).sum())
    )
    print(
        "Subjects with near-flat channels:",
        int((valid.near_flat_channels > 0).sum())
    )

print("\nIMPORTANT:")
print("Amplitude flags are exploratory QC descriptors.")
print("They are NOT validated artifact labels.")
print("No participants have been excluded.")

if len(valid) != 88:
    raise SystemExit("RAW QUALITY SCREEN REQUIRES REVIEW")

print("RAW QUALITY SCREEN COMPLETE")
