#!/usr/bin/env python3
"""M5E: read-only physical-unit and amplitude calibration audit of Isfahan EDF."""

from pathlib import Path
from collections import Counter
import csv
import numpy as np
import mne

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
SRC = BASE / 'milestone_5d_isfahan_labels_and_qc_v1'
OUT = BASE / 'milestone_5e_isfahan_unit_diagnosis_v1'
OUT.mkdir(parents=True, exist_ok=True)

FS = 256

FACTORS = {
    'as_MNE_read': 1.0,
    'times_1e_minus_3': 1e-3,
    'times_1e_minus_6': 1e-6,
    'times_1e_minus_9': 1e-9,
}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise RuntimeError('No rows to save')

    fields = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def edf_signal_header(path):
    # EDF contains a 256-byte global header followed
    # by field-major signal headers.
    # Do not interpret these as separate 256-byte
    # structures for individual channels.

    with path.open('rb') as f:
        common = f.read(256)

        if len(common) != 256:
            raise RuntimeError(f'Short EDF header: {path}')

        try:
            n = int(
                common[252:256].decode('ascii').strip()
            )

            declared_bytes = int(
                common[184:192].decode('ascii').strip()
            )

        except ValueError as exc:
            raise RuntimeError(
                f'Unparseable EDF header: {path}'
            ) from exc

        if (
            not 19 <= n <= 64
            or declared_bytes < 256 + 256 * n
        ):
            raise RuntimeError(
                f'Unexpected EDF header dimensions: {path}'
            )

        blob = f.read(n * 256)

        if len(blob) != n * 256:
            raise RuntimeError(
                f'Incomplete EDF signal header: {path}'
            )

    cursor = 0
    sections = {}

    for key, width in (
        ('label', 16),
        ('transducer', 80),
        ('dimension', 8),
        ('physical_min', 8),
        ('physical_max', 8),
        ('digital_min', 8),
        ('digital_max', 8),
        ('prefilter', 80),
        ('samples_per_record', 8),
        ('reserved', 32),
    ):

        sections[key] = [
            blob[
                cursor + j * width:
                cursor + (j + 1) * width
            ].decode(
                'latin1',
                errors='replace'
            ).strip()

            for j in range(n)
        ]

        cursor += n * width

    if cursor != len(blob):
        raise RuntimeError(
            'EDF header parsing offset mismatch'
        )

    result = []

    for j in range(n):
        values = {
            key: items[j]
            for key, items in sections.items()
        }

        values['header_index'] = j
        result.append(values)

    return result


def clean_epoch_count(ptp_uV, std_uV):
    # Original artifact rules:
    # >=3 channels with PTP >200 uV,
    # OR any channel PTP >300 uV,
    # OR any channel STD <0.5 uV.

    good = ~(
        (ptp_uV > 200).sum(axis=1) >= 3
    )

    good &= ~(
        (ptp_uV > 300).any(axis=1)
    )

    good &= ~(
        (std_uV < 0.5).any(axis=1)
    )

    return int(good.sum())


# ==========================================
# 1. VERIFY FROZEN SOURCES
# ==========================================

manifest = read_csv(
    SRC / 'release_aware_verified_subject_manifest.csv'
)

quality = read_csv(
    SRC / 'first_five_minute_quality.csv'
)

if len(manifest) != 61 or len(quality) != 61:
    raise RuntimeError(
        'M5D frozen cohort size changed'
    )

quality_by_id = {
    row['subject_id']: row
    for row in quality
}

if (
    len(quality_by_id) != 61
    or set(quality_by_id)
       != {r['subject_id'] for r in manifest}
):
    raise RuntimeError(
        'M5D quality/manifest subject mismatch'
    )

if Counter(
    r['status'] for r in quality
) != {'SCALE_WARNING': 61}:
    raise RuntimeError(
        'Unexpected M5D quality statuses'
    )

print(
    '===== MILESTONE 5E: '
    'EDF UNIT/CALIBRATION DIAGNOSIS =====',
    flush=True
)

print(
    'Original clinical labels retained; '
    'no EEG units corrected; '
    'no models trained.'
)


# ==========================================
# 2. AUDIT ALL 61 EDF SIGNAL HEADERS
# ==========================================

rows = []
examples = {}
dimensions = Counter()
mne_units = Counter()

for i, subject in enumerate(manifest, 1):

    path = ROOT / subject['edf_path']

    headers_list = edf_signal_header(path)

    headers = {
        r['label'].strip().lower(): r
        for r in headers_list
    }

    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        verbose='ERROR'
    )

    try:

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        if (
            len(picks) != 19
            or abs(
                float(raw.info['sfreq']) - FS
            ) > 1e-8
            or raw.n_times < FS * 60
        ):
            raise RuntimeError(
                'Sampling/channel/duration mismatch: '
                + subject['subject_id']
            )

        channels = [
            raw.ch_names[j]
            for j in picks
        ]

        if any(
            name.lower() not in headers
            for name in channels
        ):
            raise RuntimeError(
                'MNE/EDF channel labels disagree: '
                + subject['subject_id']
            )

        source_units = [
            headers[name.lower()]['dimension']
            for name in channels
        ]

        if len(set(source_units)) != 1:
            raise RuntimeError(
                'Mixed declared physical EEG units: '
                + subject['subject_id']
            )

        header_unit = source_units[0]

        dimensions[
            (subject['release'], header_unit)
        ] += 1

        original_units = (
            getattr(raw, '_orig_units', {}) or {}
        )

        interpreted = tuple(sorted(set(
            str(
                original_units.get(
                    name,
                    '<missing>'
                )
            )
            for name in channels
        )))

        mne_units[
            (subject['release'], interpreted)
        ] += 1

        # MNE returns EEG in volts.
        # This reproduces the units used in
        # the previous quality audit.
        # No correction is applied here.

        signal_uV = raw.get_data(
            picks=picks,
            start=0,
            stop=FS * 60
        ) * 1e6

        if (
            signal_uV.shape != (19, FS * 60)
            or not np.isfinite(signal_uV).all()
        ):
            raise RuntimeError(
                'Invalid first-minute signal: '
                + subject['subject_id']
            )

        epochs = signal_uV.reshape(
            19,
            30,
            FS * 2
        ).transpose(
            1, 0, 2
        )

        ptp = np.ptp(
            epochs,
            axis=2
        )

        std = np.std(
            epochs,
            axis=2
        )

        median_ptp = float(
            np.median(ptp)
        )

        median_std = float(
            np.median(std)
        )

        row = {
            'subject_id':
                subject['subject_id'],

            'release':
                subject['release'],

            'diagnosis':
                subject['diagnosis'],

            'edf_dimension':
                header_unit,

            'mne_original_units':
                '|'.join(interpreted),

            'mne_first_minute_median_ptp_uV':
                median_ptp,

            'mne_first_minute_median_std_uV':
                median_std,
        }

        # Hypothetical scale factors only.
        # These do NOT modify stored signals.

        for name, factor in FACTORS.items():

            row[
                name + '_median_ptp_uV'
            ] = median_ptp * factor

            row[
                name + '_clean_epochs_of_30'
            ] = clean_epoch_count(
                ptp * factor,
                std * factor
            )

        rows.append(row)

        # Retain one example from each
        # release/diagnostic group.

        example_key = (
            subject['release'],
            subject['diagnosis']
        )

        if example_key not in examples:

            examples[example_key] = (
                subject['subject_id'],
                interpreted,
                [
                    headers[name.lower()]
                    for name in channels[:3]
                ],
                median_ptp
            )

    finally:
        raw.close()

    if i % 10 == 0 or i == 61:

        print(
            f'  Header+signal diagnostics: '
            f'{i}/61',
            flush=True
        )


# ==========================================
# 3. SAVE FULL AUDIT
# ==========================================

save_csv(
    OUT / 'all_61_edf_unit_and_scaling_diagnostics.csv',
    rows
)


# ==========================================
# 4. REPORT DECLARED PHYSICAL DIMENSIONS
# ==========================================

print(
    '\n===== PHYSICAL DIMENSIONS '
    'DECLARED IN EDF HEADERS ====='
)

for (
    release,
    dimension
), n in sorted(
    dimensions.items()
):

    print(
        f'{release}: '
        f'declared physical dimension='
        f'{dimension!r}: '
        f'{n} recordings'
    )


print(
    '\n===== MNE INTERPRETED ORIGINAL UNITS ====='
)

for (
    release,
    units
), n in sorted(
    mne_units.items()
):

    print(
        f'{release}: '
        f'MNE original units={units}: '
        f'{n} recordings'
    )


# ==========================================
# 5. DISPLAY RAW EDF CALIBRATION VALUES
# ==========================================

print(
    '\n===== REPRESENTATIVE EDF '
    'PHYSICAL/DIGITAL HEADER VALUES ====='
)

for (
    release,
    group
), (
    subject_id,
    interpreted,
    example_headers,
    median
) in sorted(
    examples.items()
):

    print(
        f'{subject_id} {group}: '
        'first-minute median MNE PTP='
        f'{median:,.2f} uV; '
        f'interpreted={interpreted}'
    )

    for h in example_headers:

        print(
            f'  {h["label"]}: '
            f'physical dimension='
            f'{h["dimension"]!r}; '
            'physical min/max='
            f'{h["physical_min"]}/'
            f'{h["physical_max"]}; '
            'digital min/max='
            f'{h["digital_min"]}/'
            f'{h["digital_max"]}; '
            'samples/record='
            f'{h["samples_per_record"]}'
        )


# ==========================================
# 6. COMPARE HYPOTHETICAL UNIT SCALES
# ==========================================

print(
    '\n===== CANDIDATE SCALE FACTORS: '
    'DIAGNOSTIC ONLY ====='
)

print(
    'These factors are hypothetical and '
    'applied only to temporary calculations.'
)

print(
    'No correction has been accepted '
    'or written to an EEG file.'
)

for name, factor in FACTORS.items():

    medians = np.array([
        r[name + '_median_ptp_uV']
        for r in rows
    ])

    passing = sum(
        int(
            r[name + '_clean_epochs_of_30']
        ) >= 15
        for r in rows
    )

    plausible = int(
        np.count_nonzero(
            (medians >= 1)
            & (medians <= 300)
        )
    )

    print(
        f'{name} (factor={factor:g}): '
        'median first-minute epoch PTP '
        'across recordings='
        f'{np.median(medians):.3f} uV; '
        'median within 1-300uV range='
        f'{plausible}/61; '
        'at least 15/30 clean epochs '
        '(UNFILTERED)='
        f'{passing}/61'
    )


# ==========================================
# 7. FINISH
# ==========================================

print(
    '\n===== MILESTONE 5E '
    'DIAGNOSTIC COMPLETE ====='
)

print(
    'Saved:',
    OUT / 'all_61_edf_unit_and_scaling_diagnostics.csv'
)

print(
    'No scaling applied, no EEG resaved, '
    'no exclusions finalized, '
    'no model training.'
)

