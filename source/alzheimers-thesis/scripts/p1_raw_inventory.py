#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import mne


PROJECT = Path(__file__).resolve().parents[1]

DATA = (
    PROJECT
    / "data/eeg/on004504_v1.0.0"
)

SUMMARY = (
    PROJECT
    / "docs/audit/p1_raw_manifest_download_summary.json"
)

OUT = (
    PROJECT
    / "results/p1_raw_recording_inventory.csv"
)


def main() -> int:

    verification = json.loads(
        SUMMARY.read_text()
    )

    if (
        verification["verified"] != 264
        or verification["failures"]
    ):
        raise RuntimeError(
            "Complete verified acquisition required."
        )

    participants = pd.read_csv(
        DATA / "participants.tsv",
        sep="\t"
    )

    if (
        participants["participant_id"].nunique()
        != 88
    ):
        raise RuntimeError(
            "Expected 88 unique participants."
        )

    metadata = participants.set_index(
        "participant_id"
    )

    files = sorted(
        DATA.glob(
            "sub-*/eeg/*_eeg.set"
        )
    )

    if len(files) != 88:
        raise RuntimeError(
            f"Expected 88 raw EEG files; found {len(files)}"
        )

    rows = []

    for index, file in enumerate(files, 1):

        subject = (
            file.parent.parent.name
        )

        record = {
            "participant_id":
                subject,

            "group":
                (
                    str(metadata.loc[subject, "Group"])
                    if subject in metadata.index
                    else "UNKNOWN"
                ),

            "path":
                str(
                    file.relative_to(PROJECT)
                )
        }

        try:

            raw = mne.io.read_raw_eeglab(
                str(file),
                preload=False,
                verbose="ERROR"
            )

            sampling_rate = float(
                raw.info["sfreq"]
            )

            duration = (
                raw.n_times / sampling_rate
            )

            sample = raw.get_data(
                start=0,
                stop=min(
                    raw.n_times,
                    int(sampling_rate * 2)
                ),
                picks="eeg"
            )

            record.update({
                "sfreq":
                    sampling_rate,

                "channels":
                    len(raw.ch_names),

                "duration_seconds":
                    round(duration, 4),

                "eligible_300s":
                    duration >= 300,

                "first_2s_finite":
                    bool(
                        np.isfinite(sample).all()
                    ),

                "annotations":
                    len(raw.annotations),

                "channel_names":
                    "|".join(raw.ch_names),

                "error":
                    ""
            })

        except Exception as exc:

            record.update({
                "sfreq":
                    None,

                "channels":
                    None,

                "duration_seconds":
                    None,

                "eligible_300s":
                    False,

                "first_2s_finite":
                    False,

                "annotations":
                    None,

                "channel_names":
                    "",

                "error":
                    str(exc)[:300]
            })

        rows.append(record)

        if index % 22 == 0:

            print(
                f"Read {index}/88 raw EEG headers",
                flush=True
            )

    df = (
        pd.DataFrame(rows)
        .sort_values("participant_id")
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT,
        index=False
    )

    print()
    print(
        "===== FULL RAW EEG INVENTORY ====="
    )

    print(
        "Recordings:",
        len(df)
    )

    print(
        "Group counts:",
        df.group.value_counts().to_dict()
    )

    print(
        "Read errors:",
        int(
            (df.error != "").sum()
        )
    )

    print(
        "Sampling rates:",
        sorted(
            df.sfreq.dropna().unique().tolist()
        )
    )

    print(
        "Channel counts:",
        sorted(
            df.channels.dropna().unique().tolist()
        )
    )

    print(
        "Minimum duration:",
        round(
            df.duration_seconds.min(),
            3
        )
    )

    print(
        "Eligible for 300 seconds:",
        int(df.eligible_300s.sum())
    )

    print(
        "Finite first two seconds:",
        int(df.first_2s_finite.sum())
    )

    if (
        (df.error != "").any()
        or not df.eligible_300s.all()
        or not df.first_2s_finite.all()
    ):

        print(
            "INVENTORY REQUIRES SCIENTIFIC REVIEW"
        )

        return 2

    print(
        "ALL 88 RAW EEG HEADERS VERIFIED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
