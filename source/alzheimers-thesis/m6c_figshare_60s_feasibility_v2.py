#!/usr/bin/env python3
"""M6C-v2: tolerant MATLAB shape/duration audit; no model training."""
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import StringIO
import csv
import hashlib
import json
import re
import statistics

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
ARCHIVE = ROOT / 'data/external/cejnek_figshare_v1/dataset.zip'
M6A = BASE / 'milestone_6a_figshare_archive_preflight_v1/source_and_status.json'
M6B = BASE / 'milestone_6b_figshare_mat_structure_v1/mat_variable_and_subject_folder_audit.csv'
OUT = BASE / 'milestone_6c_figshare_60s_compatibility_v2'
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_table(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_frozen(path, data):
    if path.exists():
        if path.read_text(encoding='utf-8') != data:
            raise RuntimeError(
                'Existing frozen output differs: ' + str(path)
            )
    else:
        path.write_text(data, encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        raise RuntimeError('No CSV rows to save: ' + str(path))

    columns = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    text = StringIO(newline='')
    writer = csv.DictWriter(
        text,
        fieldnames=columns,
        lineterminator='\n'
    )
    writer.writeheader()
    writer.writerows(rows)
    save_frozen(path, text.getvalue())


print(
    '===== MILESTONE 6C-v2: NONSTANDARD MATRIX-AWARE AUDIT =====',
    flush=True
)

if not all(p.is_file() for p in (ARCHIVE, M6A, M6B)):
    raise FileNotFoundError(
        'Required M6A archive or M6B inventory not found'
    )

meta = json.loads(M6A.read_text(encoding='utf-8'))
actual_hash = digest(ARCHIVE)

if actual_hash != meta['archive_sha256']:
    raise RuntimeError('M6A archive SHA256 mismatch')

inventory = read_table(M6B)

if len(inventory) != 471:
    raise RuntimeError(
        'Expected 471 MATLAB files in M6B inventory'
    )

with ZipFile(ARCHIVE) as zf:
    names = set(zf.namelist())
    textnames = [
        n for n in names if n.lower().endswith('.txt')
    ]

    if len(textnames) != 1:
        raise RuntimeError('Expected exactly one README .txt')

    readme = zf.read(textnames[0]).decode(
        'utf-8-sig',
        errors='replace'
    )

rates = {}

for line in readme.splitlines():
    parts = line.strip().split()

    if len(parts) >= 8:
        subject = parts[-1].lower()

        if (
            re.fullmatch(r'(ad|mci|normal)\d+', subject)
            and parts[-2] in ('128', '256')
        ):
            if subject in rates:
                raise RuntimeError(
                    'Repeated sampling-rate key: ' + subject
                )

            rates[subject] = int(parts[-2])

print(
    'PASS: M6A archive SHA256 and '
    '471 MATLAB metadata records verified.'
)

print(
    'README sampling-rate entries:',
    dict(sorted(Counter(rates.values()).items()))
)

records = []
groups = defaultdict(list)
anomalies = []

for item in inventory:

    member = item['archive_member']

    if member not in names:
        raise RuntimeError(
            'Missing archive member: ' + member
        )

    group = item['diagnostic_folder']
    variable = item['primary_variable']
    folder = item['folder_subject_hint']
    shape_text = item['primary_shape']

    if group not in ('AD', 'MCI', 'CONTROL'):
        raise RuntimeError(
            'Unknown diagnostic group: ' + member
        )

    try:
        shape = tuple(
            int(x) for x in shape_text.split('x')
        )
    except ValueError:
        shape = ()

    n_samples = None
    status = 'UNRESOLVED_SHAPE'
    rate = None
    rate_evidence = 'UNKNOWN'
    channel_or_row_count = None

    if variable == 'export':

        if not folder:
            raise RuntimeError(
                'export matrix lacks subject folder: '
                + member
            )

        subject = f'{group}/{folder}'
        rate = rates.get(folder.lower())

        if rate is None:
            raise RuntimeError(
                'Missing README rate for folder: '
                + folder
            )

        rate_evidence = 'README_SUBJECT_SPECIFIC'

        if len(shape) == 2 and min(shape) > 0:

            nrows, ncols = shape
            channel_or_row_count = ncols

            if ncols == 19:

                n_samples = nrows
                status = 'STANDARD_EXPORT_19_COLUMNS'

            elif 1 <= ncols <= 64 and nrows > ncols:

                # Nominal duration only.
                # Nonstandard channel count.
                # Anatomical mapping unverified.

                n_samples = nrows
                status = (
                    'NONSTANDARD_EXPORT_WIDTH_'
                    'INFERRED_SAMPLE_AXIS'
                )

            else:

                status = (
                    'NONSTANDARD_EXPORT_'
                    'AMBIGUOUS_SAMPLE_AXIS'
                )

        if status != 'STANDARD_EXPORT_19_COLUMNS':
            anomalies.append(
                (member, shape_text, status)
            )

    elif variable == 'segmenty' and group == 'CONTROL':

        if folder:
            raise RuntimeError(
                'Unexpected segmenty subject folder: '
                + member
            )

        subject = (
            'CONTROL/file_'
            + PurePosixPath(member).stem
        )

        # Feasibility assumption, not
        # verified per-recording metadata.

        rate = 128
        rate_evidence = 'TENTATIVE_NOT_PER_FILE_VERIFIED'

        if (
            len(shape) == 2
            and 16 <= shape[0] <= 64
            and shape[1] > shape[0]
        ):

            n_samples = shape[1]
            channel_or_row_count = shape[0]

            status = (
                'SEGMENTY_ROWS_BY_SAMPLES_'
                'TENTATIVE_128HZ'
            )

        else:

            status = 'SEGMENTY_AMBIGUOUS_SAMPLE_AXIS'

            anomalies.append(
                (member, shape_text, status)
            )

    else:
        raise RuntimeError(
            'Unknown MATLAB recording format: ' + member
        )

    duration = (
        n_samples / rate
        if n_samples is not None
        else None
    )

    record = {
        'archive_member': member,
        'diagnosis': group,
        'recording_proxy_id': subject,
        'variable': variable,
        'shape': shape_text,
        'channel_or_row_count': channel_or_row_count,
        'nominal_sampling_rate_hz': rate,
        'sampling_rate_evidence': rate_evidence,
        'duration_status': status,
        'nominal_samples': n_samples,
        'nominal_duration_seconds':
            round(duration, 6)
            if duration is not None else '',
        'nominal_single_file_ge_60s':
            int(duration >= 60)
            if duration is not None else '',
        'same_19_EEG_channels_verified': False,
        'continuous_5_min_recording_verified': False,
    }

    records.append(record)
    groups[subject].append(record)


# ==========================================
# VERIFY COUNTS
# ==========================================

counts = Counter(
    r['diagnosis'] for r in records
)

if counts != {
    'AD': 305,
    'MCI': 56,
    'CONTROL': 110
}:
    raise RuntimeError(
        'Archive group counts changed: ' + str(counts)
    )

proxy_counts = Counter(
    key.split('/')[0]
    for key in groups
)

if proxy_counts != {
    'AD': 57,
    'MCI': 7,
    'CONTROL': 102
}:
    raise RuntimeError(
        'Subject/file proxy counts changed: '
        + str(proxy_counts)
    )


# ==========================================
# NONSTANDARD FILES
# ==========================================

print(
    '\n===== NONSTANDARD AND AMBIGUOUS MATRICES ====='
)

print(
    'Nonstandard export / ambiguous segmenty files:',
    len(anomalies)
)

print('Shape/status distributions:')

distribution = Counter(
    (shape, status)
    for _, shape, status in anomalies
)

for (shape, status), n in sorted(
    distribution.items()
):

    print(
        f'  {shape}: {status}: {n} file(s)'
    )

print('Example unusual files (up to 30):')

for member, shape, status in anomalies[:30]:

    print(
        f'  {member}: {shape} | {status}'
    )


# ==========================================
# DURATION BY DATA FORMAT
# ==========================================

print(
    '\n===== PER-FILE DURATION AND FORMAT ====='
)

for group in ('AD', 'MCI', 'CONTROL'):

    for variable in ('export', 'segmenty'):

        items = [
            r for r in records
            if (
                r['diagnosis'] == group
                and r['variable'] == variable
            )
        ]

        if not items:
            continue

        lengths = [
            float(r['nominal_duration_seconds'])
            for r in items
            if r['nominal_duration_seconds'] != ''
        ]

        standard60 = sum(
            r['duration_status']
                == 'STANDARD_EXPORT_19_COLUMNS'
            and r['nominal_duration_seconds'] != ''
            and float(r['nominal_duration_seconds']) >= 60
            for r in items
        )

        tentative60 = sum(
            r['nominal_duration_seconds'] != ''
            and float(r['nominal_duration_seconds']) >= 60
            for r in items
        )

        if lengths:

            message = (
                f'{group} {variable}: '
                f'files={len(items)}, '
                f'durations resolved={len(lengths)}, '
                f'median={statistics.median(lengths):.2f}s'
            )

        else:

            message = (
                f'{group} {variable}: '
                f'files={len(items)}, durations resolved=0'
            )

        print(
            message
            + f'; single-file >=60s any format={tentative60}'
            + f', standard 19-column={standard60}'
        )


# ==========================================
# PARTICIPANT-FOLDER / FILE-PROXY SUMMARY
# ==========================================

proxy_rows = []

for subject, items in sorted(groups.items()):

    lengths = [
        float(r['nominal_duration_seconds'])
        for r in items
        if r['nominal_duration_seconds'] != ''
    ]

    known_total = sum(lengths)
    max_single = max(lengths) if lengths else 0.0

    max_standard = max(
        (
            float(r['nominal_duration_seconds'])
            for r in items
            if (
                r['duration_status']
                    == 'STANDARD_EXPORT_19_COLUMNS'
                and r['nominal_duration_seconds'] != ''
            )
        ),
        default=0.0
    )

    proxy_rows.append({
        'recording_proxy_id': subject,
        'diagnosis': items[0]['diagnosis'],
        'n_mat_files': len(items),
        'n_unknown_duration':
            len(items) - len(lengths),
        'maximum_single_file_nominal_seconds':
            round(max_single, 5),
        'maximum_standard_19col_file_seconds':
            round(max_standard, 5),
        'sum_of_known_fragment_seconds_NOT_PROOF_OF_CONTINUITY':
            round(known_total, 5),
        'has_single_file_nominal_ge_60s_any_format':
            int(max_single >= 60),
        'has_single_STANDARD_19col_file_ge_60s':
            int(max_standard >= 60),
        'sum_of_known_fragments_ge_60s_NOT_CONTIGUOUS':
            int(known_total >= 60),
        'independent_participant_identity_verified':
            False,
    })


print(
    '\n===== RECORDING-PROXY DURATION FEASIBILITY ====='
)

for group in ('AD', 'MCI', 'CONTROL'):

    s = [
        r for r in proxy_rows
        if r['diagnosis'] == group
    ]

    single_any = sum(
        r['has_single_file_nominal_ge_60s_any_format']
        for r in s
    )

    single_standard = sum(
        r['has_single_STANDARD_19col_file_ge_60s']
        for r in s
    )

    summed = sum(
        r['sum_of_known_fragments_ge_60s_NOT_CONTIGUOUS']
        for r in s
    )

    unresolved = sum(
        r['n_unknown_duration'] > 0
        for r in s
    )

    print(
        f'{group}: {len(s)} participant-folder/file proxies; '
        f'single-file >=60s any structure={single_any}; '
        f'single-file >=60s standard 19-column={single_standard}; '
        f'summed fragments >=60s (CONTINUITY UNVERIFIED)={summed}; '
        f'proxies with unresolved duration={unresolved}'
    )


# ==========================================
# SAVE WITHOUT MODIFYING PREVIOUS MILESTONES
# ==========================================

save_csv(
    OUT / 'all_mat_nominal_durations_with_shape_flags.csv',
    records
)

save_csv(
    OUT / 'recording_proxy_duration_feasibility.csv',
    proxy_rows
)

summary = {
    'zip_sha256': actual_hash,
    'm6b_inventory_sha256': digest(M6B),
    'mat_files': len(records),
    'proxy_counts': dict(proxy_counts),
    'nonstandard_or_ambiguous_matrices':
        len(anomalies),
    'no_models_fitted': True,
    'no_files_extracted': True,
    'physical_units_verified': False,
    'common_channel_montage_verified': False,
    'fragment_continuity_verified': False,
    'control_root_file_participant_identity_verified':
        False,
}

save_frozen(
    OUT / 'audit_summary.json',
    json.dumps(
        summary,
        indent=2,
        sort_keys=True
    ) + '\n'
)


print(
    '\n===== MILESTONE 6C-v2 COMPLETE ====='
)

print(
    'All 471 MATLAB structures accounted for, '
    'including unusual shapes.'
)

print(
    'No signal concatenation, EEG extraction, '
    'classifier training, or prior result changes.'
)

print('Results:', OUT)

