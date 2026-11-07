#!/usr/bin/env python3
"""M5D: independently verify Isfahan workbook labels; fixed 300s signal-quality audit."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re

import numpy as np
from openpyxl import load_workbook
import mne
from scipy.signal import butter, sosfiltfilt


ROOT = Path.cwd()
DATA = ROOT / 'data/external/isfahan_mci_v1'
BASE = ROOT / 'reports/thesis_final'
OUT = BASE / 'milestone_5d_isfahan_labels_and_qc_v1'
OUT.mkdir(parents=True, exist_ok=True)

RELEASES = {
    'EEG_1': {'n': 27, 'MCI': 11, 'NORMAL': 16},
    'EEG_2': {'n': 34, 'MCI': 18, 'NORMAL': 16}
}

CHANS = (
    'Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 '
    'T5 P3 Pz P4 T6 O1 O2'
).split()

FS = 256
MIN_CLEAN_PER_MINUTE = 15
EPOCH_SAMPLES = 2 * FS
MINUTES = 5
TOTAL_SAMPLES = MINUTES * 60 * FS

SOS = butter(
    4,
    [0.5, 45.0],
    btype='bandpass',
    fs=FS,
    output='sos'
)


# ==========================================
# 1. UTILITIES
# ==========================================

def sha256(path):

    h = hashlib.sha256()

    with path.open('rb') as fh:

        for block in iter(
            lambda: fh.read(1024 * 1024),
            b''
        ):
            h.update(block)

    return h.hexdigest()


def save_csv(path, rows):

    if not rows:
        raise RuntimeError(
            'Refusing to save empty table: '
            + path.name
        )

    columns = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with path.open(
        'w',
        encoding='utf-8',
        newline=''
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def lock_text(path, value):

    if (
        path.exists()
        and path.read_text(
            encoding='utf-8'
        ) != value
    ):
        raise RuntimeError(
            'Frozen source or protocol changed: '
            + str(path)
        )

    if not path.exists():

        path.write_text(
            value,
            encoding='utf-8'
        )


def numeric_id(value):

    if isinstance(value, bool) or value is None:
        raise ValueError(
            'Invalid participant file index'
        )

    if isinstance(value, (int, float)):

        number = float(value)

        if (
            not np.isfinite(number)
            or not number.is_integer()
        ):
            raise ValueError(
                'Nonintegral participant file index'
            )

        return int(number)

    s = str(value).strip()

    if re.fullmatch(
        r'0*[1-9][0-9]*',
        s
    ) is None:
        raise ValueError(
            'Unrecognized participant file index'
        )

    return int(s)


# ==========================================
# 2. VERIFY WORKBOOK TO EDF MAPPING
# ==========================================

def load_labels(release, cfg):

    root = DATA / release

    if not root.is_dir():
        raise FileNotFoundError(root)

    books = sorted(
        p
        for p in root.rglob('*.xlsx')
        if p.name.lower() == 'states.xlsx'
    )

    if len(books) != 1:
        raise RuntimeError(
            f'{release}: expected one States.xlsx, '
            f'found {len(books)}'
        )

    book = books[0]

    wb = load_workbook(
        book,
        read_only=True,
        data_only=True
    )

    try:

        if 'Sheet1' not in wb.sheetnames:
            raise RuntimeError(
                f'{release}: Sheet1 missing'
            )

        ws = wb['Sheet1']

        # Prevent an incorrectly declared Excel
        # used range from hiding records.
        ws.reset_dimensions()

        records = [
            list(row)
            for row in ws.iter_rows(
                values_only=True
            )
            if any(
                v is not None and str(v).strip()
                for v in row
            )
        ]

    finally:
        wb.close()

    if len(records) != cfg['n'] + 1:
        raise RuntimeError(
            f'{release}: expected {cfg["n"]} '
            'data rows and one header, '
            f'got {len(records)}'
        )

    head = records[0]

    if (
        len(head) < 2
        or str(head[1]).strip().casefold()
           != 'status'
    ):
        raise RuntimeError(
            f'{release}: column B header '
            'is not status'
        )

    mapping = {}

    for row in records[1:]:

        if (
            len(row) < 2
            or any(
                v is not None and str(v).strip()
                for v in row[2:]
            )
        ):
            raise RuntimeError(
                f'{release}: unexpected workbook layout'
            )

        i = numeric_id(row[0])

        label = str(
            row[1]
        ).strip().upper()

        if label not in (
            'MCI',
            'NORMAL'
        ):
            raise RuntimeError(
                f'{release}: unsupported label '
                f'{label!r} for file {i}'
            )

        if i in mapping:
            raise RuntimeError(
                f'{release}: duplicate file number {i}'
            )

        mapping[i] = label

    if set(mapping) != set(
        range(1, cfg['n'] + 1)
    ):
        raise RuntimeError(
            f'{release}: missing, unexpected, '
            'or nonsequential file numbers'
        )

    actual = Counter(
        mapping.values()
    )

    if any(
        actual[group] != cfg[group]
        for group in ('MCI', 'NORMAL')
    ):
        raise RuntimeError(
            f'{release}: label counts differ '
            f'from reported release: {dict(actual)}'
        )

    disk_edfs = {
        p.name
        for p in root.rglob('*.edf')
    }

    expected = {
        f'{i}.edf'
        for i in mapping
    }

    if disk_edfs != expected:
        raise RuntimeError(
            f'{release}: EDF files differ '
            'from workbook file numbers: '
            f'missing={sorted(expected - disk_edfs)} '
            f'extra={sorted(disk_edfs - expected)}'
        )

    rows = []

    for i in sorted(mapping):

        path = next(
            p
            for p in root.rglob(
                f'{i}.edf'
            )
            if p.name == f'{i}.edf'
        )

        rows.append({
            'subject_id':
                f'{release}-{i:03d}',

            'release': release,

            'release_local_file_id': i,

            'diagnosis': mapping[i],

            'binary_label_MCI':
                int(mapping[i] == 'MCI'),

            'edf_path':
                str(path.relative_to(ROOT)),

            'edf_bytes':
                path.stat().st_size,

            'metadata_path':
                str(book.relative_to(ROOT)),

            'metadata_sha256':
                sha256(book),

            'label_source':
                'States.xlsx Sheet1 columns A '
                '(file ID) and B (status)'
        })

    return rows, mapping, book


# ==========================================
# 3. FIRST-FIVE-MINUTE SIGNAL QUALITY
# ==========================================

def screen_first_300s(manifest_row):

    row = {
        'subject_id':
            manifest_row['subject_id'],

        'release':
            manifest_row['release'],

        'diagnosis':
            manifest_row['diagnosis'],

        'status': 'UNASSESSED',

        'quality_window_start_s': 0,

        'quality_window_end_s': 300,

        'sampling_rate_hz': '',

        'recording_duration_min': '',

        'eeg_channels': '',

        'n_boundaries_first_300s': '',

        'clean_epochs_total': '',

        'clean_epochs_each_minute': '',

        'raw_clean_epochs_each_minute': '',

        'median_filtered_epoch_ptp_uV': '',

        'median_filtered_epoch_std_uV': '',

        'median_raw_epoch_ptp_uV': '',

        'reason': ''
    }

    path = (
        ROOT / manifest_row['edf_path']
    )

    try:

        raw = mne.io.read_raw_edf(
            str(path),
            preload=False,
            verbose='ERROR'
        )

        try:

            sfreq = float(
                raw.info['sfreq']
            )

            picks = mne.pick_types(
                raw.info,
                eeg=True,
                exclude=[]
            )

            names = [
                raw.ch_names[j].strip()
                for j in picks
            ]

            row['sampling_rate_hz'] = sfreq

            row['recording_duration_min'] = round(
                raw.n_times / sfreq / 60,
                3
            )

            row['eeg_channels'] = len(picks)

            if (
                abs(sfreq - FS) > 1e-8
                or len(picks) != 19
                or names != CHANS
            ):

                row['status'] = (
                    'INCOMPATIBLE_HEADER'
                )

                row['reason'] = (
                    'Non-256Hz recording or '
                    'montage/order differs from '
                    'declared canonical 19 channels'
                )

                return row

            if raw.n_times < TOTAL_SAMPLES:

                row['status'] = 'TOO_SHORT'

                row['reason'] = (
                    'Fewer than 300 seconds of EEG'
                )

                return row

            n_boundaries = sum(
                'boundary' in str(text).lower()
                and 0.002 < float(t) < 299.998

                for t, text in zip(
                    raw.annotations.onset,
                    raw.annotations.description
                )
            )

            row['n_boundaries_first_300s'] = (
                int(n_boundaries)
            )

            if n_boundaries:

                row['status'] = (
                    'CONTINUITY_FAIL'
                )

                row['reason'] = (
                    'Recorded boundary in first 300s'
                )

                return row

            # MNE returns EDF EEG in volts.
            # Convert to microvolts.

            uv = raw.get_data(
                picks=picks,
                start=0,
                stop=TOTAL_SAMPLES
            ) * 1e6

            if (
                uv.shape
                != (19, TOTAL_SAMPLES)
                or not np.isfinite(uv).all()
            ):

                row['status'] = (
                    'INVALID_SIGNAL'
                )

                row['reason'] = (
                    'Unexpected array shape '
                    'or nonfinite samples'
                )

                return row

            # A fixed 0.5–45 Hz bandpass.
            # This does not reproduce all
            # preprocessing in ds004504.

            filtered = sosfiltfilt(
                SOS,
                uv,
                axis=1
            )

            if not np.isfinite(
                filtered
            ).all():

                row['status'] = (
                    'INVALID_SIGNAL'
                )

                row['reason'] = (
                    'Nonfinite bandpassed samples'
                )

                return row

            # ==================================
            # 4. APPLY ORIGINAL ARTIFACT RULES
            # ==================================

            def epoch_qc(data):

                epochs = data.reshape(
                    19,
                    150,
                    EPOCH_SAMPLES
                ).transpose(
                    1,
                    0,
                    2
                )

                ptp = np.ptp(
                    epochs,
                    axis=2
                )

                std = np.std(
                    epochs,
                    axis=2
                )

                clean = ~(
                    (
                        (ptp > 200).sum(axis=1)
                        >= 3
                    )
                    | (ptp > 300).any(axis=1)
                    | (std < 0.5).any(axis=1)
                )

                counts = (
                    clean.reshape(5, 30)
                    .sum(axis=1)
                )

                return (
                    counts,
                    float(np.median(ptp)),
                    float(np.median(std))
                )

            counts, median_ptp, median_std = (
                epoch_qc(filtered)
            )

            raw_counts, raw_ptp, _ = (
                epoch_qc(uv)
            )

            row['clean_epochs_total'] = (
                int(counts.sum())
            )

            row['clean_epochs_each_minute'] = (
                '|'.join(
                    map(str, counts.tolist())
                )
            )

            row['raw_clean_epochs_each_minute'] = (
                '|'.join(
                    map(str, raw_counts.tolist())
                )
            )

            row[
                'median_filtered_epoch_ptp_uV'
            ] = round(median_ptp, 4)

            row[
                'median_filtered_epoch_std_uV'
            ] = round(median_std, 4)

            row[
                'median_raw_epoch_ptp_uV'
            ] = round(raw_ptp, 4)

            if (
                median_ptp < 0.02
                or median_ptp > 10000
            ):

                row['status'] = (
                    'SCALE_WARNING'
                )

                row['reason'] = (
                    'Implausible median amplitude; '
                    'inspect EDF physical units '
                    'before modeling'
                )

            elif (
                counts < MIN_CLEAN_PER_MINUTE
            ).any():

                row['status'] = (
                    'INSUFFICIENT_CLEAN_EPOCHS'
                )

                row['reason'] = (
                    'At least one first-five-minute '
                    'block below 15 of 30 clean epochs'
                )

            else:

                row['status'] = (
                    'ELIGIBLE_FIXED_300S'
                )

                row['reason'] = (
                    'All five 60s blocks passed '
                    'fixed filter and artifact screen'
                )

            return row

        finally:
            raw.close()

    except Exception as exc:

        row['status'] = (
            'READ_OR_PROCESSING_ERROR'
        )

        row['reason'] = (
            f'{type(exc).__name__}: '
            f'{str(exc)[:250]}'
        )

        return row


# ==========================================
# 5. LOCK PROTOCOL
# ==========================================

print(
    '===== MILESTONE 5D: '
    'VERIFIED LABELS AND '
    'FIXED-FIVE-MINUTE QC =====',
    flush=True
)

provenance = (
    BASE
    / 'milestone_5b_isfahan_audit_v1'
    / 'source_archive_provenance.json'
)

if not provenance.is_file():
    raise FileNotFoundError(provenance)

protocol = {
    'version':
        'm5d_isfahan_labels_and_qc_v1',

    'dataset':
        'two Isfahan MCI/healthy EEG archives',

    'identity':
        'release-local EDF ID matched '
        'to States.xlsx Sheet1 column A; '
        'diagnosis from column B',

    'participant_id':
        'release-aware, not proof of '
        'cross-release unique person',

    'sampling_rate_hz': 256,

    'montage': CHANS,

    'quality_window_seconds': [0, 300],

    'quality_window_selection':
        'fixed first five minutes for everyone; '
        'no sliding or optimization',

    'filter':
        '4th-order Butterworth zero-phase '
        'SOS 0.5-45Hz on full five-minute signal',

    'unit':
        'MNE EDF voltage units converted '
        'to microvolts',

    'epoch_seconds': 2,

    'clean_epoch_rule':
        'reject if >=3 EEG channels '
        'ptp>200uV or any ptp>300uV '
        'or any std<0.5uV',

    'minimum_clean_epochs_per_minute': 15,

    'note':
        'No ICA/ASR or other corrections; '
        'acquisition/reference may differ '
        'from original OpenNeuro derivatives; '
        'no classification'
}

lock_text(
    OUT / 'locked_protocol.json',
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True
    ) + '\n'
)


# ==========================================
# 6. FREEZE VERIFIED CLINICAL MANIFEST
# ==========================================

manifest = []
mappings = {}

for release, config in RELEASES.items():

    rows, mapping, book = load_labels(
        release,
        config
    )

    manifest.extend(rows)

    mappings[release] = mapping

    print(
        f'{release}: EXACT label mapping passed: '
        f'MCI={config["MCI"]}, '
        f'NORMAL={config["NORMAL"]}; '
        f'EDF IDs 1..{config["n"]}; '
        f'workbook SHA256={sha256(book)[:14]}...',
        flush=True
    )


if (
    len(manifest) != 61
    or len(set(
        r['subject_id']
        for r in manifest
    )) != 61
):
    raise RuntimeError(
        'Manifest count or ID uniqueness mismatch'
    )


source_locks = {
    'source_archive_provenance_sha256':
        sha256(provenance),

    'metadata_workbook_sha256': {
        release: sha256(
            next(
                p
                for p in (
                    DATA / release
                ).rglob('*.xlsx')
                if p.name.lower() == 'states.xlsx'
            )
        )
        for release in RELEASES
    }
}

lock_text(
    OUT / 'locked_sources.json',
    json.dumps(
        source_locks,
        indent=2,
        sort_keys=True
    ) + '\n'
)

manifest_path = (
    OUT
    / 'release_aware_verified_subject_manifest.csv'
)

if manifest_path.exists():

    old_manifest = manifest_path.read_text(
        encoding='utf-8'
    )

    if old_manifest:

        # This is a new milestone output.
        # Do not silently alter an existing
        # frozen subject-to-label mapping.
        import io

        buffer = io.StringIO()

        fields = list(manifest[0].keys())

        writer = csv.DictWriter(
            buffer,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(manifest)

        if old_manifest != buffer.getvalue():
            raise RuntimeError(
                'Existing frozen subject manifest '
                'differs from current workbook mapping'
            )

else:

    save_csv(
        manifest_path,
        manifest
    )


# ==========================================
# 7. CROSS-RELEASE IDENTITY SAFEGUARD
# ==========================================

shared_file_ids = (
    set(mappings['EEG_1'])
    & set(mappings['EEG_2'])
)

discordant = sum(
    mappings['EEG_1'][i]
    != mappings['EEG_2'][i]
    for i in shared_file_ids
)

print(
    '\n===== CROSS-RELEASE '
    'LABEL/IDENTITY STATUS ====='
)

print(
    'Overlapping numeric filenames:',
    len(shared_file_ids),
    '; different diagnosis for '
    'same numeric filename:',
    discordant
)

print(
    'The two archives have release-local IDs. '
    'Filename collisions are not a '
    'clinical participant linkage.'
)

print(
    'Cross-release individual identities remain '
    'unverified; do not call all 61 independent '
    'people based on spreadsheets alone.'
)


# ==========================================
# 8. FIXED FIRST-FIVE-MINUTE QUALITY AUDIT
# ==========================================

quality = []

for i, mr in enumerate(
    manifest,
    1
):

    result = screen_first_300s(mr)

    quality.append(result)

    if (
        i <= 3
        or i % 10 == 0
        or i == 61
    ):

        print(
            f'  QC {i}/61 '
            f'{mr["subject_id"]} '
            f'{mr["diagnosis"]}: '
            f'{result["status"]}; '
            'filtered clean per minute='
            f'{result["clean_epochs_each_minute"]}; '
            'raw median PTP='
            f'{result["median_raw_epoch_ptp_uV"]}uV; '
            'filtered median PTP='
            f'{result["median_filtered_epoch_ptp_uV"]}uV',
            flush=True
        )

    if i % 10 == 0:

        save_csv(
            OUT
            / 'first_five_minute_quality_partial.csv',
            quality
        )


save_csv(
    OUT / 'first_five_minute_quality.csv',
    quality
)


# ==========================================
# 9. PRINT ELIGIBILITY RESULTS
# ==========================================

print(
    '\n===== FIXED FIRST-FIVE-MINUTE '
    'QUALITY BY RELEASE AND DIAGNOSIS ====='
)

for release in RELEASES:

    for group in (
        'MCI',
        'NORMAL'
    ):

        rows = [
            r for r in quality
            if (
                r['release'] == release
                and r['diagnosis'] == group
            )
        ]

        counter = Counter(
            r['status']
            for r in rows
        )

        print(
            f'{release} {group}: '
            f'n={len(rows)}; '
            f'{dict(counter)}'
        )


print(
    '\n===== QUALITY ISSUES / DIAGNOSTICS ====='
)

for status, count in Counter(
    r['status']
    for r in quality
).most_common():

    print(
        f'{status}: {count}/61'
    )


issues = [
    r for r in quality
    if (
        r['status']
        != 'ELIGIBLE_FIXED_300S'
    )
]

for r in issues[:20]:

    print(
        f'  {r["subject_id"]} '
        f'{r["diagnosis"]}: '
        f'{r["status"]}; '
        f'clean={r["clean_epochs_each_minute"]}; '
        f'{r["reason"]}'
    )

if len(issues) > 20:

    print(
        f'  ... {len(issues) - 20} '
        'additional issues are in CSV'
    )


# ==========================================
# 10. FINISH AND SAVE AUDIT STATUS
# ==========================================

(
    OUT / 'audit_status.json'
).write_text(
    json.dumps(
        {
            'n_recordings': 61,

            'verified_recording_labels':
                dict(Counter(
                    r['diagnosis']
                    for r in manifest
                )),

            'quality_counts':
                dict(Counter(
                    r['status']
                    for r in quality
                )),

            'release_unique_ids_not_proven_globally':
                True,

            'models_trained':
                False,

            'baseline_AD_FTD_results_modified':
                False
        },
        indent=2,
        sort_keys=True
    ) + '\n',
    encoding='utf-8'
)

print(
    '\n===== MILESTONE 5D COMPLETE ====='
)

print(
    'Verified label manifest:',
    OUT / 'release_aware_verified_subject_manifest.csv'
)

print(
    'Quality audit:',
    OUT / 'first_five_minute_quality.csv'
)

print(
    'No models trained; '
    'no previous EEG or thesis results modified.'
)

