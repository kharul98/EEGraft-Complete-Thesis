#!/usr/bin/env python3
"""M6C: Figshare 60s duration, fragment, and channel-format audit."""

from collections import Counter, defaultdict
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import hashlib
import json
import re

import numpy as np
from scipy.io import loadmat


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

ARCHIVE = (
    ROOT / 'data/external/cejnek_figshare_v1/dataset.zip'
)

M6A = BASE / 'milestone_6a_figshare_archive_preflight_v1'
M6B = BASE / 'milestone_6b_figshare_mat_structure_v1'

OUT = BASE / 'milestone_6c_figshare_60s_compatibility_v1'
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(
        'r',
        newline='',
        encoding='utf-8-sig'
    ) as f:
        return list(csv.DictReader(f))


def checksum(path):
    h = hashlib.sha256()

    with path.open('rb') as f:
        for part in iter(
            lambda: f.read(2 * 1024 * 1024),
            b''
        ):
            h.update(part)

    return h.hexdigest()


def frozen(path, content):
    if path.exists():

        if path.read_text(
            encoding='utf-8'
        ) != content:
            raise RuntimeError(
                'Frozen M6C output differs: ' + str(path)
            )

    else:
        path.write_text(
            content,
            encoding='utf-8'
        )


def frozen_csv(path, rows):
    if not rows:
        raise RuntimeError('No results to save')

    buf = StringIO()

    fields = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    writer = csv.DictWriter(
        buf,
        fieldnames=fields,
        lineterminator='\n'
    )

    writer.writeheader()
    writer.writerows(rows)

    frozen(path, buf.getvalue())


# ==========================================
# 1. VERIFY SOURCES
# ==========================================

print(
    '===== MILESTONE 6C: '
    'FIGSHARE 60s COMPATIBILITY =====',
    flush=True
)

meta = json.loads(
    (
        M6A / 'source_and_status.json'
    ).read_text(encoding='utf-8')
)

source = (
    M6B / 'mat_variable_and_subject_folder_audit.csv'
)

rows = read_csv(source)

if len(rows) != 471:
    raise RuntimeError(
        'Expected 471 M6B MATLAB inventory records'
    )

if (
    not ARCHIVE.is_file()
    or checksum(ARCHIVE) != meta['archive_sha256']
):
    raise RuntimeError(
        'Original archive checksum mismatch'
    )

print(
    'PASS: M6A archive checksum and '
    'M6B 471-file inventory verified.'
)


# ==========================================
# 2. READ SAMPLING-RATE DOCUMENTATION
# ==========================================

with ZipFile(ARCHIVE) as zf:

    txts = [
        item.filename
        for item in zf.infolist()
        if item.filename.lower().endswith('.txt')
    ]

    if len(txts) != 1:
        raise RuntimeError(
            'Expected one archive README'
        )

    readme = zf.read(
        txts[0]
    ).decode(
        'utf-8-sig',
        errors='replace'
    )

    members = set(
        zf.namelist()
    )


rates = {}

for line in readme.splitlines():

    parts = line.strip().split()

    if len(parts) < 8:
        continue

    subject = parts[-1].lower()
    rate = parts[-2]

    if (
        re.fullmatch(
            r'(ad|mci|normal)\d+',
            subject
        )
        and rate in ('128', '256')
    ):

        if subject in rates:
            raise RuntimeError(
                'Duplicate README rate: ' + subject
            )

        rates[subject] = int(rate)


print(
    'README subject sampling rates:',
    dict(Counter(rates.values()))
)


# ==========================================
# 3. ASSESS EVERY MATLAB FILE
# ==========================================

file_rows = []
by_subject = defaultdict(list)

for r in rows:

    member = r['archive_member']

    if member not in members:
        raise RuntimeError(
            'Inventory entry absent from ZIP: ' + member
        )

    path = PurePosixPath(member)

    group = r['diagnostic_folder']
    folder = r['folder_subject_hint']
    variable = r['primary_variable']

    shape = tuple(
        int(x)
        for x in r['primary_shape'].split('x')
    )

    if (
        group not in ('AD', 'MCI', 'CONTROL')
        or len(shape) != 2
    ):
        raise RuntimeError(
            'Unexpected group or MAT dimensions: ' + member
        )

    if variable == 'export':

        if shape[1] != 19 or not folder:
            raise RuntimeError(
                'Unexpected export structure: ' + member
            )

        n_samples = shape[0]

        layout = 'samples_by_19'

        subject_id = group + '/' + folder

        rate = rates.get(
            folder.lower()
        )

        basis = 'README_subject_specific'

        if rate is None:
            raise RuntimeError(
                'README sampling rate missing: ' + folder
            )

    elif (
        group == 'CONTROL'
        and variable == 'segmenty'
    ):

        if folder or shape[0] < 16:
            raise RuntimeError(
                'Unexpected segmenty structure: ' + member
            )

        n_samples = shape[1]

        layout = (
            str(shape[0])
            + '_rows_by_samples'
        )

        subject_id = (
            group + '/file_' + path.stem
        )

        # Assumption for feasibility estimates
        # only, not verified per-file metadata.

        rate = 128

        basis = (
            'TENTATIVE_128_'
            'inferred_not_per_file_verified'
        )

    else:
        raise RuntimeError(
            'Unrecognized recording format: ' + member
        )

    if n_samples <= 0:
        raise RuntimeError(
            'Empty signal: ' + member
        )

    duration = n_samples / rate

    result = {
        'member': member,
        'diagnosis': group,
        'recording_proxy_id': subject_id,
        'format': variable,
        'matrix_layout': layout,
        'shape': r['primary_shape'],
        'sampling_rate_hz': rate,
        'rate_evidence': basis,
        'n_samples': n_samples,
        'nominal_duration_sec':
            round(duration, 6),
        'single_fragment_at_least_60s':
            int(duration >= 60),
        'single_fragment_at_least_300s':
            int(duration >= 300),
        'continuity_across_fragments_verified':
            False,
        'electrode_labels_verified':
            False
    }

    file_rows.append(result)

    by_subject[
        subject_id
    ].append(result)


# ==========================================
# 4. CHECK MISSING AD PARTICIPANT FOLDERS
# ==========================================

present_ad = {
    int(s.split('/AD')[1])
    for s in by_subject

    if (
        s.startswith('AD/AD')
        and s.split('/AD')[1].isdigit()
    )
}

missing_ad = sorted(
    set(range(1, 60)) - present_ad
)

print(
    'Missing AD folder numbers '
    'among AD1..AD59:',
    missing_ad
)

print(
    'CONTROL segmenty row-count distribution:',
    dict(Counter(
        int(r['shape'].split('x')[0])

        for r in file_rows
        if r['format'] == 'segmenty'
    ))
)


# ==========================================
# 5. SUMMARIZE SINGLE-FILE DURATION
# ==========================================

print(
    '\n===== PER-FILE DURATION AND FORMAT ====='
)

groups = Counter(
    (
        r['diagnosis'],
        r['format'],
        r['sampling_rate_hz']
    )

    for r in file_rows
)

for (
    group,
    fmt,
    rate
), n in sorted(groups.items()):

    durations = [
        r['nominal_duration_sec']

        for r in file_rows

        if (
            r['diagnosis'],
            r['format'],
            r['sampling_rate_hz']
        ) == (
            group,
            fmt,
            rate
        )
    ]

    print(
        f'{group}: {fmt}, '
        f'nominal {rate} Hz; '
        f'files={n}; '
        f'median={np.median(durations):.2f}s; '
        f'longest={max(durations):.2f}s; '
        'files with >=60s='
        f'{sum(d >= 60 for d in durations)}'
    )


# ==========================================
# 6. GROUP RECORDINGS BY FOLDER/FILE PROXY
# ==========================================

subject_rows = []

for subject_id, fragments in sorted(
    by_subject.items()
):

    group = fragments[0]['diagnosis']

    maximum = max(
        r['nominal_duration_sec']
        for r in fragments
    )

    cumulative = sum(
        r['nominal_duration_sec']
        for r in fragments
    )

    subject_rows.append({
        'recording_proxy_id': subject_id,
        'diagnosis': group,
        'n_files': len(fragments),
        'nominal_total_seconds_IF_contiguous':
            round(cumulative, 4),
        'longest_single_fragment_seconds':
            round(maximum, 4),
        'has_single_fragment_60s':
            int(maximum >= 60),
        'has_single_fragment_300s':
            int(maximum >= 300),
        'total_fragments_at_least_60s':
            int(cumulative >= 60),
        'actual_continuity_across_fragments':
            'UNVERIFIED',
        'participant_identity_status':
            (
                'folder_hint'
                if '/file_' not in subject_id
                else 'control_file_proxy_not_confirmed'
            )
    })


proxy_counts = Counter(
    r['diagnosis']
    for r in subject_rows
)

if proxy_counts != {
    'AD': 57,
    'MCI': 7,
    'CONTROL': 102
}:
    raise RuntimeError(
        'Unexpected participant-folder/file-proxy '
        'counts: ' + str(proxy_counts)
    )


print(
    '\n===== RECORDING-PROXY '
    'DURATION FEASIBILITY ====='
)

for group in (
    'AD',
    'MCI',
    'CONTROL'
):

    subset = [
        r for r in subject_rows
        if r['diagnosis'] == group
    ]

    single60 = sum(
        r['has_single_fragment_60s']
        for r in subset
    )

    potential60 = sum(
        r['total_fragments_at_least_60s']
        for r in subset
    )

    single300 = sum(
        r['has_single_fragment_300s']
        for r in subset
    )

    print(
        f'{group}: {len(subset)} '
        'participant-folder/file-proxy units; '
        f'single-file >=60s={single60}; '
        'potential total >=60s if concatenated '
        f'(NOT verified)={potential60}; '
        f'single-file >=300s={single300}'
    )


# ==========================================
# 7. SAMPLE ACTUAL SIGNAL ARRAYS
# ==========================================

lookups = {
    r['member']: r
    for r in file_rows
}


def choose(group, fmt, rate=None, count=2):

    subset = sorted(
        [
            r for r in file_rows

            if (
                r['diagnosis'] == group
                and r['format'] == fmt
                and (
                    rate is None
                    or r['sampling_rate_hz'] == rate
                )
            )
        ],

        key=lambda r: r['member']
    )

    if not subset:
        return []

    indices = sorted(set(
        np.linspace(
            0,
            len(subset) - 1,
            num=min(count, len(subset)),
            dtype=int
        ).tolist()
    ))

    return [
        subset[i]['member']
        for i in indices
    ]


selected = list(dict.fromkeys(
    choose('AD', 'export', 256, 2)
    + choose('AD', 'export', 128, 2)
    + choose('CONTROL', 'export', 128, 2)
    + choose('CONTROL', 'segmenty', 128, 3)
))

samples = []

with ZipFile(ARCHIVE) as zf:

    for member in selected:

        row = lookups[member]

        info = zf.getinfo(member)

        if info.file_size > 64 * 1024**2:
            raise RuntimeError(
                'Sample MAT exceeds memory cap: '
                + member
            )

        payload = loadmat(
            BytesIO(zf.read(member)),
            variable_names=[row['format']]
        )

        array = np.asarray(
            payload[row['format']]
        )

        expected = tuple(
            int(x)
            for x in row['shape'].split('x')
        )

        if (
            array.shape != expected
            or array.ndim != 2
        ):
            raise RuntimeError(
                'Actual MAT shape differs '
                'from M6B: ' + member
            )

        finite = float(
            np.isfinite(array).mean()
        )

        channel_major = (
            array.T
            if row['format'] == 'export'
            else array
        )

        stds = np.nanstd(
            np.where(
                np.isfinite(channel_major),
                channel_major,
                np.nan
            ),
            axis=1
        )

        samples.append({
            'member': member,
            'diagnosis': row['diagnosis'],
            'format': row['format'],
            'shape': row['shape'],
            'rate_basis': row['rate_evidence'],
            'finite_fraction': finite,
            'min_numeric_value':
                float(np.nanmin(array)),
            'max_numeric_value':
                float(np.nanmax(array)),
            'first_16_median_std_IN_ORIGINAL_UNITS':
                float(np.median(stds[:16])),
            'last_rows_median_std_IN_ORIGINAL_UNITS':
                float(np.median(stds[16:])),
            'number_of_channels_or_rows':
                int(channel_major.shape[0]),
            'electrode_order_proven': False,
            'physical_units_proven': False
        })


print(
    '\n===== SAMPLE SIGNAL CHECK '
    '(NO UNIT/COLUMN ASSUMPTIONS) ====='
)

for r in samples:

    print(
        f"{r['member']}: "
        f"{r['shape']} {r['format']}; "
        'first-16-row median STD='
        f"{r['first_16_median_std_IN_ORIGINAL_UNITS']:.3f} "
        'original units; '
        f"rows={r['number_of_channels_or_rows']}; "
        'electrode labels UNVERIFIED'
    )


# ==========================================
# 8. FREEZE DIAGNOSTIC OUTPUTS
# ==========================================

frozen_csv(
    OUT / 'all_mat_nominal_durations.csv',
    file_rows
)

frozen_csv(
    OUT / 'recording_proxy_duration_feasibility.csv',
    subject_rows
)

frozen_csv(
    OUT / 'sample_mat_signal_diagnostics.csv',
    samples
)

frozen(
    OUT / 'input_sha256.json',

    json.dumps(
        {
            'original_archive_sha256':
                checksum(ARCHIVE),

            'm6b_inventory_sha256':
                checksum(source)
        },
        sort_keys=True,
        indent=2
    ) + '\n'
)


# ==========================================
# 9. COMPLETE
# ==========================================

print(
    '\n===== MILESTONE 6C COMPLETE ====='
)

print(
    'Only sample signal arrays inspected; '
    'no EEG extracted or concatenated.'
)

print(
    'No models fitted and no existing '
    'thesis results modified.'
)

print(
    'Electrode identity and physical units '
    'remain unverified.'
)

print(
    'The 128Hz assumption for 100 CONTROL '
    'root files remains TENTATIVE.'
)

print(
    'Saved:',
    OUT
)

