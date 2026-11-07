#!/usr/bin/env python3
"""Milestone 5C: read-only diagnostic metadata audit for Isfahan EEG."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re

from openpyxl import load_workbook


ROOT = Path.cwd()

DATA = ROOT / 'data/external/isfahan_mci_v1'

OUT = (
    ROOT / 'reports/thesis_final'
    / 'milestone_5c_isfahan_label_audit_v1'
)

OUT.mkdir(parents=True, exist_ok=True)

# Reported cohort counts, not yet validated labels.
RELEASES = {
    'EEG_1': (27, 11, 16),
    'EEG_2': (34, 18, 16)
}

SAFE_WORDS = {
    'mci', 'amci', 'healthy', 'normal',
    'control', 'controls', 'hc', 'cn',
    'patient', 'patients', 'state', 'states',
    'group', 'class', 'label', 'diagnosis',
    'subject', 'subject id', 'id', 'number',
    'no', 'case', 'mmse', 'score', 'status',
    'age', 'male', 'female', 'sex', 'gender',
    'yes', '0', '1', '2', '3'
}


# ==========================================
# 1. GENERAL UTILITIES
# ==========================================

def sha256(path):

    h = hashlib.sha256()

    with path.open('rb') as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b''
        ):

            h.update(block)

    return h.hexdigest()


def write_csv(path, rows):

    if not rows:
        return

    columns = list(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))

    with path.open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def token(value):

    if value is None:
        return ''

    if isinstance(value, bool):
        return str(int(value))

    if isinstance(value, (int, float)):

        if float(value).is_integer():
            return str(int(value))

        return str(value)

    return re.sub(
        r'\s+',
        ' ',
        str(value).strip()
    ).strip()


def safe_preview(value):

    value = token(value)

    if not value:
        return '-'

    if re.fullmatch(
        r'-?\d+(?:\.\d+)?',
        value
    ):
        return value[:22]

    if value.casefold() in SAFE_WORDS:
        return value[:26]

    # Do not print arbitrary names or
    # other free-text metadata in chat logs.
    return f'<text:{len(value)}chars>'


def column_letter(index):

    result = ''
    index += 1

    while index:

        index, remainder = divmod(
            index - 1,
            26
        )

        result = (
            chr(65 + remainder)
            + result
        )

    return result


def remove_optional_header(values, n):

    headers = {
        'id', 'subject', 'subject id',
        'patient id', 'number', 'no',
        'file', 'filename', 'state',
        'states', 'group', 'class',
        'label', 'diagnosis', 'status'
    }

    if (
        len(values) == n + 1
        and values[0].casefold() in headers
    ):
        return values[1:]

    return values


def is_complete_id_vector(values, n):

    values = remove_optional_header(
        [
            token(v)
            for v in values
            if token(v)
        ],
        n
    )

    if len(values) != n:
        return False

    try:

        ids = [
            int(value)
            for value in values
        ]

    except ValueError:
        return False

    return (
        len(set(ids)) == n
        and set(ids) == set(
            range(1, n + 1)
        )
    )


def candidate_class_distribution(
    values,
    n,
    n_mci,
    n_hc
):

    values = remove_optional_header(
        [
            token(v).casefold()
            for v in values
            if token(v)
        ],
        n
    )

    if len(values) != n:
        return None

    counts = Counter(values)

    if (
        len(counts) != 2
        or sorted(counts.values())
           != sorted((n_mci, n_hc))
    ):
        return None

    # Counts alone do not prove which
    # code denotes MCI or healthy.
    return counts


# ==========================================
# 2. START AUDIT
# ==========================================

print(
    '===== MILESTONE 5C: '
    'ISFAHAN STATES.XLSX LABEL AUDIT =====',
    flush=True
)

print(
    'Reported: EEG_1 = 11 MCI + 16 HC; '
    'EEG_2 = 18 MCI + 16 HC.'
)

print(
    'No clinical labels will be assigned '
    'solely from numerical codes or filenames.'
)

workbook_rows = []
candidate_rows = []
id_rows = []
file_rows = []


# ==========================================
# 3. PROCESS BOTH DATASET RELEASES
# ==========================================

for release, (
    expected_n,
    expected_mci,
    expected_hc
) in RELEASES.items():

    folder = DATA / release

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    edfs = sorted(
        folder.rglob('*.edf')
    )

    if len(edfs) != expected_n:

        raise RuntimeError(
            f'{release}: expected '
            f'{expected_n} EDF files, '
            f'found {len(edfs)}'
        )

    eeg_ids = []

    for path in edfs:

        match = re.fullmatch(
            r'(\d+)\.edf',
            path.name,
            flags=re.I
        )

        if match:
            eeg_ids.append(
                int(match.group(1))
            )

        file_rows.append({
            'release': release,
            'file':
                str(path.relative_to(ROOT)),
            'file_id':
                match.group(1)
                if match else '',
            'file_bytes':
                path.stat().st_size
        })

    ids_unique = (
        len(set(eeg_ids))
        == expected_n
    )

    ids_sequential = (
        ids_unique
        and set(eeg_ids)
        == set(
            range(
                1,
                expected_n + 1
            )
        )
    )

    books = sorted(
        path
        for path in folder.rglob(
            '*.xlsx'
        )
        if path.name.lower()
        == 'states.xlsx'
    )

    print(
        f'\n===== {release}: '
        'EXCEL WORKBOOK STRUCTURE ====='
    )

    print(
        f'EDF files={len(edfs)}; '
        f'numeric file IDs={len(eeg_ids)}; '
        f'unique={ids_unique}; '
        f'IDs 1..{expected_n}={ids_sequential}'
    )

    print(
        'EDF ID sample:',
        sorted(eeg_ids)[:12]
    )

    if len(books) != 1:

        raise RuntimeError(
            f'{release}: expected exactly '
            f'one States.xlsx; '
            f'found {len(books)}'
        )

    path = books[0]

    print(
        'Workbook:',
        path.relative_to(ROOT)
    )

    print(
        'Workbook SHA256:',
        sha256(path)
    )

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True
    )

    release_candidates = 0

    for sheet in workbook.worksheets:

        # Prevent Excel's declared used
        # range from silently omitting data.
        sheet.reset_dimensions()

        data = []

        for row in sheet.iter_rows(
            values_only=True
        ):

            if any(
                token(v)
                for v in row
            ):
                data.append(
                    list(row)
                )

        width = max(
            (
                len(row)
                for row in data
            ),
            default=0
        )

        height = len(data)

        workbook_rows.append({
            'release': release,
            'workbook':
                str(path.relative_to(ROOT)),
            'sheet': sheet.title,
            'nonempty_rows': height,
            'max_columns': width,
            'sha256': sha256(path)
        })

        print(
            f'Sheet {sheet.title!r}: '
            f'nonempty rows={height}, '
            f'max columns={width}'
        )

        if not data:
            continue


        # ==================================
        # 4. PRINT SAFE CELL PREVIEW
        # ==================================

        print(
            'First 12 nonempty rows '
            '(free-text values redacted):'
        )

        for row_number, row in enumerate(
            data[:12],
            1
        ):

            preview = ' | '.join(
                f'{column_letter(j)}:'
                f'{safe_preview(value)}'

                for j, value in enumerate(
                    row[:14]
                )

                if token(value)
            )

            print(
                f'  row {row_number:>2}: '
                f'{preview or "<empty>"}'
            )

        if width > 14:

            print(
                f'Note: all {width} columns '
                'were scanned; only the first '
                '14 are shown in this preview.'
            )


        # ==================================
        # 5. SCAN ID AND CLASS-CODE VECTORS
        # ==================================

        vectors = []

        # Entire columns.
        for j in range(width):

            values = [
                row[j]
                if j < len(row)
                else None
                for row in data
            ]

            vectors.append((
                'column',
                column_letter(j),
                values
            ))

        # Entire rows.
        for i, row in enumerate(
            data,
            1
        ):

            vectors.append((
                'row',
                str(i),
                row
            ))

        candidate_ids = []
        candidate_labels = []

        for orientation, location, values in vectors:

            if is_complete_id_vector(
                values,
                expected_n
            ):

                candidate_ids.append((
                    orientation,
                    location
                ))

                id_rows.append({
                    'release': release,
                    'sheet': sheet.title,
                    'orientation':
                        orientation,
                    'location':
                        location,
                    'complete_unique_ids_1_to_n':
                        True
                })

            distribution = (
                candidate_class_distribution(
                    values,
                    expected_n,
                    expected_mci,
                    expected_hc
                )
            )

            if distribution is None:
                continue

            candidate_labels.append((
                orientation,
                location,
                distribution
            ))

            release_candidates += 1

            candidate_rows.append({
                'release': release,
                'sheet': sheet.title,
                'orientation':
                    orientation,
                'location':
                    location,
                'n_values':
                    expected_n,
                'code_counts':
                    json.dumps(
                        dict(distribution),
                        sort_keys=True
                    ),
                'diagnosis_mapping_confirmed':
                    False
            })

        print(
            'Complete ID vectors 1..n:',
            candidate_ids[:12]
            if candidate_ids
            else 'NONE DETECTED'
        )

        print(
            'Candidate diagnostic-code vectors:'
        )

        if candidate_labels:

            for (
                orientation,
                location,
                distribution
            ) in candidate_labels[:12]:

                print(
                    f'  {orientation} {location}: '
                    f'{dict(distribution)}'
                )

        else:

            print(
                '  NONE; workbook layout '
                'requires closer inspection.'
            )


        # ==================================
        # 6. LOW-CARDINALITY CODE COUNTS
        # ==================================

        print(
            'Low-cardinality distributions '
            '(first 12 columns):'
        )

        for j in range(
            min(12, width)
        ):

            values = [
                token(row[j])
                if j < len(row)
                else ''
                for row in data
            ]

            counts = Counter(
                value.casefold()
                for value in values
                if value
            )

            if (
                not counts
                or len(counts) > 8
            ):
                continue

            display = ', '.join(
                f'{safe_preview(value)}:{count}'
                for value, count
                in counts.most_common(8)
            )

            print(
                f'  {column_letter(j)}: '
                f'{display}'
            )

    workbook.close()

    print(
        f'{release}: candidate class vectors '
        f'found={release_candidates}; '
        'diagnostic mapping NOT finalized.'
    )


# ==========================================
# 7. SAVE METADATA AUDIT
# ==========================================

write_csv(
    OUT / 'workbook_structure.csv',
    workbook_rows
)

write_csv(
    OUT / 'candidate_label_code_distributions.csv',
    candidate_rows
)

write_csv(
    OUT / 'explicit_id_vector_locations.csv',
    id_rows
)

write_csv(
    OUT / 'edf_file_inventory.csv',
    file_rows
)


# ==========================================
# 8. CROSS-RELEASE IDENTITY WARNING
# ==========================================

print(
    '\n===== CROSS-RELEASE IDENTITY SAFEGUARD ====='
)

print(
    'Each release contains its own '
    'numbered EEG files.'
)

print(
    'Repeated numeric filenames do NOT '
    'prove participants are identical.'
)

print(
    'Different EEG file bytes do NOT '
    'prove participants are distinct.'
)

print(
    'Clinical code meanings and '
    'file-to-label linkage must be '
    'verified before classification.'
)

print(
    'Participant independence between '
    'releases remains UNVERIFIED.'
)

summary = {
    'EEG_1_recordings': 27,
    'EEG_2_recordings': 34,
    'total_EEG_files': len(file_rows),
    'candidate_class_vectors':
        len(candidate_rows),
    'diagnosis_mapping_confirmed':
        False,
    'cross_release_unique_subjects_confirmed':
        False,
    'next_step':
        'Confirm diagnostic code meanings '
        'and EDF-to-spreadsheet linkage '
        'before freezing a subject manifest.'
}

(
    OUT / 'metadata_status.json'
).write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True
    ) + '\n',
    encoding='utf-8'
)

print(
    '\n===== MILESTONE 5C '
    'METADATA PREFLIGHT COMPLETE ====='
)

print(
    'Saved:',
    OUT
)

print(
    'No classifiers trained; '
    'no EEG or previous thesis results altered.'
)

