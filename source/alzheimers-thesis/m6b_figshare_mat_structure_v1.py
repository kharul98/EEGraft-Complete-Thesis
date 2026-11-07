#!/usr/bin/env python3
"""M6B: metadata-only audit of the Figshare MATLAB EEG archive."""

from collections import Counter, defaultdict
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import hashlib
import json

from scipy.io import whosmat


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

SRC = (
    BASE
    / 'milestone_6a_figshare_archive_preflight_v1'
)

OUT = (
    BASE
    / 'milestone_6b_figshare_mat_structure_v1'
)

ARCHIVE = (
    ROOT
    / 'data/external/cejnek_figshare_v1/dataset.zip'
)

OUT.mkdir(parents=True, exist_ok=True)

MAX_MAT_BYTES = 256 * 1024**2
GROUPS = ('AD', 'MCI', 'CONTROL')


# ==========================================
# 1. FILE AND OUTPUT UTILITIES
# ==========================================

def sha256(path):

    h = hashlib.sha256()

    with path.open('rb') as f:

        for block in iter(
            lambda: f.read(2 * 1024**2),
            b''
        ):
            h.update(block)

    return h.hexdigest()


def frozen(path, content):

    if path.exists():

        if path.read_text(
            encoding='utf-8'
        ) != content:

            raise RuntimeError(
                'Frozen M6B output differs: '
                + str(path)
            )

    else:

        path.write_text(
            content,
            encoding='utf-8'
        )


def frozen_csv(path, records):

    if not records:
        raise RuntimeError(
            'Empty inventory: ' + str(path)
        )

    cols = list(dict.fromkeys(
        k for row in records for k in row
    ))

    buf = StringIO()

    writer = csv.DictWriter(
        buf,
        fieldnames=cols,
        lineterminator='\n'
    )

    writer.writeheader()
    writer.writerows(records)

    frozen(path, buf.getvalue())


# ==========================================
# 2. VERIFY ORIGINAL FIGSHARE ARCHIVE
# ==========================================

print(
    '===== MILESTONE 6B: '
    'FIGSHARE MATLAB STRUCTURE AUDIT =====',
    flush=True
)

if not ARCHIVE.is_file():
    raise FileNotFoundError(ARCHIVE)

meta_path = SRC / 'source_and_status.json'

if not meta_path.is_file():
    raise FileNotFoundError(meta_path)

meta = json.loads(
    meta_path.read_text(encoding='utf-8')
)

if (
    int(meta['archive_size_bytes'])
    != ARCHIVE.stat().st_size
):
    raise RuntimeError(
        'Archive differs from M6A size'
    )

hashval = sha256(ARCHIVE)

if hashval != meta['archive_sha256']:
    raise RuntimeError(
        'Archive differs from M6A SHA256'
    )

print(
    'PASS: Original M6A ZIP checksum '
    'and size verified.'
)


# ==========================================
# 3. SCAN MATLAB VARIABLE METADATA
# ==========================================

matrows = []
textrows = []

folder_groups = defaultdict(
    lambda: defaultdict(list)
)

root_group = Counter()
member_count = Counter()
variable_freq = Counter()
read_errors = []

with ZipFile(ARCHIVE) as zf:

    infos = zf.infolist()

    mat_infos = [
        z for z in infos
        if (
            not z.is_dir()
            and z.filename.lower().endswith('.mat')
        )
    ]

    if len(mat_infos) != 471:
        raise RuntimeError(
            'M6A MATLAB file count differs: '
            + str(len(mat_infos))
        )

    for info in infos:

        pp = PurePosixPath(
            info.filename.replace('\\', '/')
        )

        if (
            pp.is_absolute()
            or '..' in pp.parts
            or not pp.parts
        ):

            raise RuntimeError(
                'Unsafe ZIP member name: '
                + info.filename
            )

        if info.is_dir():
            continue

        group = pp.parts[0].upper()

        if (
            group not in GROUPS
            and pp.suffix.lower() != '.txt'
        ):

            raise RuntimeError(
                'Unexpected unclassified file: '
                + info.filename
            )

        if pp.suffix.lower() == '.txt':

            if info.file_size <= 100000:

                doc = zf.read(info).decode(
                    'utf-8-sig',
                    errors='replace'
                )

                textrows.append((
                    pp.as_posix(),
                    doc
                ))

            continue

        if pp.suffix.lower() != '.mat':
            continue

        if len(pp.parts) >= 3:

            folder = pp.parts[1]
            grouping = 'EXPLICIT_SUBFOLDER'

            folder_groups[
                group
            ][folder].append(
                pp.as_posix()
            )

        else:

            folder = ''
            grouping = 'ROOT_FILE_UNRESOLVED'
            root_group[group] += 1

        member_count[group] += 1

        row = {
            'archive_member':
                pp.as_posix(),

            'diagnostic_folder':
                group,

            'folder_subject_hint':
                folder,

            'subject_grouping':
                grouping,

            'mat_file_bytes':
                info.file_size,

            'zip_crc32_hex':
                f'{info.CRC:08x}',

            'mat_read_status':
                '',

            'mat_variables':
                '',

            'primary_variable':
                '',

            'primary_shape':
                '',

            'primary_class':
                '',

            'possible_channel_dimensions':
                '',
        }

        if info.file_size > MAX_MAT_BYTES:

            row['mat_read_status'] = (
                'TOO_LARGE_TO_PROBE'
            )

        else:

            try:

                # Read only MATLAB variable
                # metadata, not EEG arrays.
                # One ZIP member is held in
                # memory at a time.

                variables = whosmat(
                    BytesIO(zf.read(info))
                )

                specs = [
                    {
                        'name':
                            str(name),

                        'shape':
                            tuple(
                                int(x)
                                for x in shape
                            ),

                        'class':
                            str(dtype)
                    }

                    for name, shape, dtype
                    in variables
                ]

                row['mat_variables'] = '|'.join(

                    f"{v['name']}:"
                    f"{'x'.join(map(str, v['shape']))}:"
                    f"{v['class']}"

                    for v in specs[:12]
                )

                if specs:

                    preferred = next(
                        (
                            v
                            for name in (
                                'export',
                                'segmenty',
                                'data',
                                'EEG',
                                'eeg'
                            )
                            for v in specs

                            if (
                                v['name'].lower()
                                == name.lower()
                            )
                        ),
                        None
                    )

                    primary = preferred or max(
                        specs,

                        key=lambda v: (
                            len(v['shape']),
                            1 if v['class'] in (
                                'double',
                                'single'
                            ) else 0,

                            v['shape'][0]
                            if v['shape']
                            else 0
                        )
                    )

                    row['primary_variable'] = (
                        primary['name']
                    )

                    row['primary_shape'] = (
                        'x'.join(
                            map(
                                str,
                                primary['shape']
                            )
                        )
                    )

                    row['primary_class'] = (
                        primary['class']
                    )

                    row[
                        'possible_channel_dimensions'
                    ] = '|'.join(
                        str(v)

                        for v in sorted(
                            set(primary['shape'])
                            & {16, 19, 20, 21}
                        )
                    )

                    variable_freq[
                        (
                            group,
                            primary['name'],
                            primary['class']
                        )
                    ] += 1

                    row['mat_read_status'] = (
                        'MAT_HEADER_OK'
                    )

                else:

                    row['mat_read_status'] = (
                        'NO_VARIABLES'
                    )

            except Exception as exc:

                row['mat_read_status'] = (
                    'MAT_HEADER_ERROR'
                )

                read_errors.append((
                    pp.as_posix(),
                    type(exc).__name__,
                    str(exc)[:150]
                ))

        matrows.append(row)

        if len(matrows) % 100 == 0:

            print(
                f'  Inspected {len(matrows)}/471 '
                'MATLAB files',
                flush=True
            )


# ==========================================
# 4. SAVE FILE AND FOLDER AUDITS
# ==========================================

frozen_csv(
    OUT / 'mat_variable_and_subject_folder_audit.csv',
    matrows
)

folderrows = []

for group in GROUPS:

    for folder, files in sorted(
        folder_groups[group].items()
    ):

        folderrows.append({
            'diagnostic_folder':
                group,

            'subject_folder':
                folder,

            'n_mat_files_in_folder':
                len(files),

            'first_mat_file':
                sorted(files)[0]
        })

if folderrows:

    frozen_csv(
        OUT / 'explicit_subject_folder_counts.csv',
        folderrows
    )


# ==========================================
# 5. READ ARCHIVE DOCUMENTATION
# ==========================================

print(
    '\n===== DOCUMENTATION WITHIN ARCHIVE ====='
)

if textrows:

    for name, body in textrows:

        print(
            'Text file:',
            name,
            '| characters:',
            len(body)
        )

        print(
            '--- README preview '
            '(maximum 4000 characters) ---'
        )

        print(
            body[:4000].replace('\x00', '')
        )

        print(
            '--- END README preview ---'
        )

else:

    print(
        'No readable small text files found.'
    )


# ==========================================
# 6. SUBJECT-FOLDER AND FILE COUNTS
# ==========================================

print(
    '\n===== PARTICIPANT-FOLDER '
    'AND FILE COUNTS ====='
)

for group in GROUPS:

    folders = folder_groups[group]

    n_with_folder = sum(
        len(files)
        for files in folders.values()
    )

    distributions = Counter(
        len(files)
        for files in folders.values()
    )

    print(
        f'{group}: '
        f'MATLAB files={member_count[group]}; '
        'explicit participant-like '
        f'folders={len(folders)}; '
        'files within folders='
        f'{n_with_folder}; '
        'ungrouped root files='
        f'{root_group[group]}'
    )

    print(
        '  Files-per-folder distribution:',
        dict(sorted(distributions.items()))
    )

    print(
        '  Example folders:',
        sorted(folders)[:12]
    )

    if root_group[group]:

        roots = [
            r['archive_member']
            for r in matrows

            if (
                r['diagnostic_folder'] == group
                and r['subject_grouping']
                    == 'ROOT_FILE_UNRESOLVED'
            )
        ]

        print(
            '  Example unresolved root files:',
            roots[:12]
        )


# ==========================================
# 7. MATLAB VARIABLE FORMATS
# ==========================================

print(
    '\n===== MATLAB VARIABLE FORMATS ====='
)

for (
    group,
    name,
    dtype
), n in sorted(
    variable_freq.items()
):

    print(
        f'{group}: '
        f'{name} ({dtype}): '
        f'{n} files'
    )


for group in GROUPS:

    for grouping in (
        'EXPLICIT_SUBFOLDER',
        'ROOT_FILE_UNRESOLVED'
    ):

        sample = [
            r for r in matrows

            if (
                r['diagnostic_folder'] == group
                and r['subject_grouping'] == grouping
            )
        ]

        if not sample:
            continue

        print(
            f'{group} / {grouping}: '
            'representative MAT shapes:'
        )

        for row in (
            sample[:2]
            + sample[-1:]
        ):

            print(
                ' ',
                row['archive_member'],
                '=>',
                row['mat_variables'][:200],
                '| status:',
                row['mat_read_status']
            )


# ==========================================
# 8. AUDIT STATUS
# ==========================================

status = Counter(
    r['mat_read_status']
    for r in matrows
)

print(
    '\n===== STRUCTURAL AUDIT STATUS ====='
)

print(
    'MAT header statuses:',
    dict(status)
)

for member, etype, message in read_errors[:8]:

    print(
        '  PARSE ERROR:',
        member,
        etype,
        message
    )

print(
    'Sampling rates, channel ordering, '
    'chronological continuity and'
)

print(
    'independent subject IDs remain '
    'UNVERIFIED; no classifier fitted.'
)


# ==========================================
# 9. FREEZE SUMMARY
# ==========================================

summary = {
    'archive_sha256':
        hashval,

    'mat_files':
        len(matrows),

    'file_counts_by_group':
        dict(member_count),

    'folders_with_mat_by_group': {
        g: len(folder_groups[g])
        for g in GROUPS
    },

    'root_mat_files_by_group':
        dict(root_group),

    'mat_header_status':
        dict(status),

    'no_eeg_extracted':
        True,

    'participant_independence_verified':
        False,

    'channel_alignment_verified':
        False,

    'sampling_rate_verified':
        False,

    'minute_wise_temporal_continuity_verified':
        False,

    'no_model_training':
        True
}

frozen(
    OUT / 'audit_summary.json',
    json.dumps(
        summary,
        indent=2,
        sort_keys=True
    ) + '\n'
)

print(
    '\n===== MILESTONE 6B COMPLETE ====='
)

print(
    'Saved:',
    OUT
)

print(
    'No file extraction, signal preprocessing '
    'or model training.'
)

