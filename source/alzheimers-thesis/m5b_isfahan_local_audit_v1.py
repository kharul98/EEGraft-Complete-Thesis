#!/usr/bin/env python3
"""Milestone 5B: local Isfahan ZIP acquisition, safe extraction, EDF metadata audit."""

from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from zipfile import ZipFile, BadZipFile
import csv
import hashlib
import json
import os
import re
import shutil
import stat

import mne


ROOT = Path.cwd()
OUT = ROOT / 'reports/thesis_final/milestone_5b_isfahan_audit_v1'
DATA = ROOT / 'data/external/isfahan_mci_v1'

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

OFFICIAL = {
    'EEG_1': 382725035,
    'EEG_2': 523487067
}

MAX_EXPANDED = 20 * 1024**3
MAX_MEMBER = 5 * 1024**3


# ==========================================
# 1. FILE UTILITIES
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


def write_csv(path, rows):

    if not rows:
        return

    columns = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with path.open(
        'w',
        encoding='utf-8',
        newline=''
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 2. FIND BROWSER DOWNLOADS
# ==========================================

def find_downloads():

    roots = [
        Path.home() / 'Downloads',
        ROOT / 'downloads'
    ]

    for drive in ('c', 'd', 'e'):

        roots.append(
            Path('/mnt') / drive / 'Downloads'
        )

        user_base = (
            Path('/mnt') / drive / 'Users'
        )

        if not user_base.is_dir():
            continue

        for user in user_base.iterdir():

            if not user.is_dir():
                continue

            roots.append(
                user / 'Downloads'
            )

            roots.append(
                user / 'OneDrive' / 'Downloads'
            )

            roots.extend(
                user.glob('OneDrive*/Downloads')
            )

    unique_roots = sorted(set(
        p for p in roots if p.is_dir()
    ))

    matches = defaultdict(list)

    pattern = re.compile(
        r'^EEG[_ -]*([12])(?:\s*\(\d+\))?\.zip$',
        re.I
    )

    for directory in unique_roots:

        for p in directory.iterdir():

            if not p.is_file():
                continue

            match = pattern.fullmatch(p.name)

            if match:

                key = 'EEG_' + match.group(1)

                matches[key].append(p)

    return unique_roots, matches


def select_archives():

    roots, found = find_downloads()

    print(
        '===== LOCAL BROWSER DOWNLOAD DISCOVERY ====='
    )

    print('Downloads directories detected:')

    for directory in roots:
        print(' ', directory)

    chosen = {}

    for key, official_size in OFFICIAL.items():

        override = os.environ.get(
            'ISFAHAN_ZIP' + key[-1],
            ''
        ).strip()

        candidates = (
            [Path(override).expanduser()]
            if override
            else found[key]
        )

        valid = [
            p for p in candidates
            if (
                p.is_file()
                and p.stat().st_size == official_size
            )
        ]

        print(key + ':')

        for p in candidates:

            size = (
                p.stat().st_size
                if p.is_file()
                else 'not found'
            )

            print(
                f'  {p} ({size} bytes)'
            )

        if not valid:

            raise RuntimeError(
                f'Cannot find an exact-size {key}.zip '
                f'({official_size} bytes). '
                'If downloads are in a custom folder, '
                'set ISFAHAN_ZIP1 and ISFAHAN_ZIP2 '
                'to the full local file paths.'
            )

        valid.sort(
            key=lambda p: (
                p.name.lower()
                != key.lower() + '.zip',
                str(p)
            )
        )

        chosen[key] = valid[0]

        print(
            '  SELECTED:',
            chosen[key]
        )

        if len(valid) > 1:

            print(
                '  Multiple exact-size copies found; '
                'using the first listed.'
            )

    if (
        chosen['EEG_1'].resolve()
        == chosen['EEG_2'].resolve()
    ):

        raise RuntimeError(
            'The two ZIP paths refer to the same file'
        )

    return chosen


# ==========================================
# 3. SECURE ZIP EXTRACTION
# ==========================================

def check_member(info, seen):

    raw = info.filename.replace(
        '\\',
        '/'
    )

    path = PurePosixPath(raw)

    if (
        not raw
        or path.is_absolute()
        or '..' in path.parts
        or not path.parts
    ):

        raise RuntimeError(
            'Unsafe ZIP member: ' + repr(raw)
        )

    if (
        ':' in path.parts[0]
        or '\x00' in raw
    ):

        raise RuntimeError(
            'Unsafe ZIP member path'
        )

    if stat.S_ISLNK(
        info.external_attr >> 16
    ):

        raise RuntimeError(
            'ZIP contains a symbolic link'
        )

    if not info.is_dir():

        if raw in seen:

            raise RuntimeError(
                'Duplicate internal ZIP filename'
            )

        seen.add(raw)

    if info.file_size > MAX_MEMBER:

        raise RuntimeError(
            'ZIP member exceeds safety size cap'
        )

    return path


def extract_archive(key, source):

    target = DATA / key

    target.mkdir(
        parents=True,
        exist_ok=True
    )

    listed = []
    eeg_files = []

    reused = 0
    written = 0

    with ZipFile(source) as zf:

        infos = zf.infolist()

        if not infos:

            raise RuntimeError(
                key + ' ZIP is empty'
            )

        expanded = sum(
            i.file_size
            for i in infos
            if not i.is_dir()
        )

        if expanded > MAX_EXPANDED:

            raise RuntimeError(
                'Expanded ZIP exceeds safety cap'
            )

        seen = set()

        print(
            f'{key}: archive entries={len(infos)}, '
            f'expanded size='
            f'{expanded / 1024**3:.3f} GiB'
        )

        for info in infos:

            member = check_member(
                info,
                seen
            )

            if info.is_dir():
                continue

            if info.flag_bits & 1:

                raise RuntimeError(
                    'Encrypted ZIP member unsupported'
                )

            destination = target.joinpath(
                *member.parts
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if destination.is_symlink():

                raise RuntimeError(
                    'Unexpected extracted symbolic link'
                )

            if (
                destination.is_file()
                and destination.stat().st_size
                    == info.file_size
            ):

                action = 'reused_size_checked'
                reused += 1

            else:

                if destination.exists():

                    raise RuntimeError(
                        'Existing extracted file '
                        'has a different size'
                    )

                partial = destination.with_name(
                    destination.name + '.incomplete'
                )

                if partial.exists():
                    partial.unlink()

                try:

                    with (
                        zf.open(info) as src,
                        partial.open('wb') as dst
                    ):

                        shutil.copyfileobj(
                            src,
                            dst,
                            length=2 * 1024**2
                        )

                    if (
                        partial.stat().st_size
                        != info.file_size
                    ):

                        raise RuntimeError(
                            'Extracted file size mismatch'
                        )

                    # Reading the complete ZIP member
                    # also validates its stored CRC.

                    os.replace(
                        partial,
                        destination
                    )

                finally:

                    if partial.exists():
                        partial.unlink()

                action = 'extracted_crc_checked'
                written += 1

            extension = destination.suffix.lower()

            listed.append({
                'archive': key,
                'member': member.as_posix(),
                'uncompressed_bytes':
                    info.file_size,
                'zip_crc32_hex':
                    f'{info.CRC:08x}',
                'extension': extension,
                'status': action,
                'extracted_path':
                    str(destination)
            })

            if extension in (
                '.edf',
                '.bdf'
            ):

                eeg_files.append((
                    key,
                    member.as_posix(),
                    destination
                ))

            if (
                written
                and written % 20 == 0
                and action == 'extracted_crc_checked'
            ):

                print(
                    f'  {key}: {written} '
                    'files extracted...',
                    flush=True
                )

    print(
        f'{key}: extracted={written}, '
        f'reused={reused}; '
        f'EDF/BDF files={len(eeg_files)}'
    )

    return listed, eeg_files


# ==========================================
# 4. EEG MONTAGE / LABEL UTILITIES
# ==========================================

STANDARD = {
    s.lower()
    for s in (
        'Fp1 Fp2 F7 F3 Fz F4 F8 '
        'T3 C3 Cz C4 T4 T5 P3 Pz '
        'P4 T6 O1 O2'
    ).split()
}

ALIAS = {
    't7': 't3',
    't8': 't4',
    'p7': 't5',
    'p8': 't6'
}


def canonical_channel(name):

    normalized = re.sub(
        r'^\s*EEG[\s_-]*',
        '',
        name,
        flags=re.I
    ).strip()

    normalized = re.split(
        r'[-\s]+',
        normalized
    )[0].lower()

    return ALIAS.get(
        normalized,
        normalized
    )


def guess_group(name):

    text = str(name).lower()

    mci = bool(re.search(
        r'(?<![a-z])(?:a?mci)[-_\s]*\d*(?![a-z])',
        text
    ))

    healthy = bool(re.search(
        r'(?<![a-z])'
        r'(?:healthy|controls?|normal|hc|cn)'
        r'[-_\s]*\d*(?![a-z])',
        text
    ))

    if mci and healthy:
        return 'AMBIGUOUS'

    if mci:
        return 'MCI_PATH_HINT'

    if healthy:
        return 'HC_PATH_HINT'

    return 'UNLABELED'


# ==========================================
# 5. READ EEG HEADERS WITHOUT LOADING SIGNAL
# ==========================================

def audit_header(key, member, path):

    row = {
        'archive': key,
        'member': member,
        'relative_path':
            str(path.relative_to(ROOT)),
        'file_bytes':
            path.stat().st_size,
        'group_from_path_only':
            guess_group(member),
        'read_status': 'UNASSESSED',
        'sfreq_hz': '',
        'duration_minutes': '',
        'n_eeg_channels': '',
        'n_common_19_labels': '',
        'all_19_present': '',
        'channel_names': '',
        'error': ''
    }

    try:

        if path.suffix.lower() == '.edf':

            raw = mne.io.read_raw_edf(
                str(path),
                preload=False,
                verbose='ERROR'
            )

        else:

            raw = mne.io.read_raw_bdf(
                str(path),
                preload=False,
                verbose='ERROR'
            )

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        names = [
            raw.ch_names[i]
            for i in picks
        ]

        normalized = {
            canonical_channel(x)
            for x in names
        }

        sfreq = float(
            raw.info['sfreq']
        )

        row.update({
            'read_status': 'HEADER_OK',
            'sfreq_hz':
                round(sfreq, 6),
            'duration_minutes':
                round(
                    raw.n_times / sfreq / 60,
                    4
                ),
            'n_eeg_channels':
                len(picks),
            'n_common_19_labels':
                len(
                    STANDARD.intersection(
                        normalized
                    )
                ),
            'all_19_present':
                int(
                    STANDARD.issubset(normalized)
                ),
            'channel_names':
                '|'.join(names[:40])
        })

        raw.close()

    except Exception as exc:

        row['read_status'] = 'READ_ERROR'

        row['error'] = (
            f'{type(exc).__name__}: '
            f'{str(exc)[:220]}'
        )

    return row


# ==========================================
# 6. VERIFY ORIGINAL THESIS IS PRESENT
# ==========================================

print(
    '===== MILESTONE 5B: '
    'LOCAL ISFAHAN EEG AUDIT =====',
    flush=True
)

old_report = (
    ROOT / 'reports/thesis_final'
    / 'milestone_4c_reproducibility_v1'
    / 'THESIS_EVIDENCE_SUMMARY.md'
)

if not old_report.is_file():

    raise FileNotFoundError(
        'Original thesis evidence package missing'
    )


# ==========================================
# 7. LOCATE AND EXTRACT BOTH ARCHIVES
# ==========================================

sources = select_archives()

print(
    '\n===== ZIP FILE VALIDATION '
    'AND SAFE EXTRACTION =====',
    flush=True
)

listing = []
eeg_files = []
provenance = {}

for key in (
    'EEG_1',
    'EEG_2'
):

    path = sources[key]

    if path.stat().st_size != OFFICIAL[key]:

        raise RuntimeError(
            'Source ZIP size does not match '
            'the official file'
        )

    try:

        list_rows, eeg_rows = extract_archive(
            key,
            path
        )

    except BadZipFile as exc:

        raise RuntimeError(
            'Invalid or corrupted ZIP: '
            + str(path)
        ) from exc

    listing.extend(list_rows)
    eeg_files.extend(eeg_rows)

    provenance[key] = {
        'source_path':
            str(path),
        'size_bytes':
            path.stat().st_size,
        'sha256':
            sha256(path),
        'archive_entries':
            len(list_rows),
        'edf_bdf_files':
            len(eeg_rows)
    }


write_csv(
    OUT / 'zip_member_inventory.csv',
    listing
)

(
    OUT / 'source_archive_provenance.json'
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True
    ) + '\n',
    encoding='utf-8'
)


# ==========================================
# 8. INSPECT ARCHIVE CONTENTS
# ==========================================

print(
    '\n===== ARCHIVE CONTENTS '
    'AND DIAGNOSTIC LABEL HINTS ====='
)

for key in (
    'EEG_1',
    'EEG_2'
):

    members = [
        row for row in listing
        if row['archive'] == key
    ]

    extensions = Counter(
        row['extension']
        for row in members
    )

    print(
        f'{key}: extensions='
        f'{dict(extensions)}'
    )

    for row in members[:10]:

        print(
            '  ',
            row['member'][:170]
        )


# ==========================================
# 9. READ ALL EEG FILE HEADERS
# ==========================================

print(
    '\n===== EDF HEADER '
    'AND MONTAGE AUDIT =====',
    flush=True
)

headers = []

for k, (
    source,
    member,
    path
) in enumerate(
    eeg_files,
    1
):

    headers.append(
        audit_header(
            source,
            member,
            path
        )
    )

    if (
        k % 10 == 0
        or k == len(eeg_files)
    ):

        print(
            f'  Inspected {k}/{len(eeg_files)} '
            'EEG headers',
            flush=True
        )


write_csv(
    OUT / 'edf_header_audit.csv',
    headers
)


for key in (
    'EEG_1',
    'EEG_2'
):

    subset = [
        r for r in headers
        if r['archive'] == key
    ]

    readable = [
        r for r in subset
        if r['read_status'] == 'HEADER_OK'
    ]

    label_hints = Counter(
        r['group_from_path_only']
        for r in subset
    )

    rates = Counter(
        r['sfreq_hz']
        for r in readable
    )

    montage = sum(
        int(r['all_19_present'])
        for r in readable
    )

    durations = sorted(
        float(r['duration_minutes'])
        for r in readable
    )

    print(
        f'{key}: recordings={len(subset)}, '
        f'headers readable={len(readable)}'
    )

    print(
        '  Path label hints:',
        dict(label_hints)
    )

    print(
        '  Sampling rates:',
        dict(rates)
    )

    print(
        '  Full canonical 19-channel '
        'montage detected:',
        f'{montage}/{len(readable)}'
    )

    if durations:

        print(
            '  Duration min/median/max:',
            f'{durations[0]:.2f} / '
            f'{durations[len(durations)//2]:.2f} / '
            f'{durations[-1]:.2f} min'
        )

    for row in subset[:2]:

        print(
            f'  Example: {row["member"]}; '
            'channels='
            f'{row["channel_names"][:200]}'
        )

    for row in subset:

        if row['read_status'] == 'READ_ERROR':

            print(
                f'  READ_ERROR {row["member"]}: '
                f'{row["error"]}'
            )


# ==========================================
# 10. CHECK DUPLICATES ACROSS RELEASES
# ==========================================

print(
    '\n===== CROSS-ARCHIVE DUPLICATE WARNINGS ====='
)

by_name = defaultdict(list)
by_size = defaultdict(list)

for key, member, path in eeg_files:

    by_name[
        path.name.lower()
    ].append((
        key,
        member,
        path
    ))

    by_size[
        path.stat().st_size
    ].append((
        key,
        member,
        path
    ))


name_collisions = [
    entries
    for entries in by_name.values()
    if len({
        x[0] for x in entries
    }) > 1
]


identical_files = []

for entries in by_size.values():

    if len({
        x[0] for x in entries
    }) < 2:
        continue

    hashes = defaultdict(list)

    for entry in entries:

        hashes[
            sha256(entry[2])
        ].append(entry)

    identical_files.extend(
        items
        for items in hashes.values()
        if len({
            x[0] for x in items
        }) > 1
    )


print(
    'Repeated EDF/BDF basenames '
    'across archives:',
    len(name_collisions)
)

for entries in name_collisions[:8]:

    print(
        '  NAME:',
        ' | '.join(
            x[0] + '/' + x[1]
            for x in entries
        )
    )


print(
    'Identical EEG files across archives '
    '(SHA256):',
    len(identical_files)
)

for entries in identical_files[:8]:

    print(
        '  IDENTICAL:',
        ' | '.join(
            x[0] + '/' + x[1]
            for x in entries
        )
    )


print(
    'NOTE: No filename/hash match does '
    'NOT prove participants are distinct.'
)


# ==========================================
# 11. SAVE AUDIT SUMMARY
# ==========================================

summary = {
    'sources':
        provenance,

    'n_eeg_files':
        len(eeg_files),

    'n_readable_headers':
        sum(
            r['read_status'] == 'HEADER_OK'
            for r in headers
        ),

    'path_label_hints':
        dict(Counter(
            r['group_from_path_only']
            for r in headers
        )),

    'duplicate_basename_groups':
        len(name_collisions),

    'identical_file_groups':
        len(identical_files),

    'participant_uniqueness_verified':
        False,

    'diagnostic_labels_verified_from_metadata':
        False,

    'original_AD_FTD_thesis_modified':
        False,

    'note':
        'File-level audit only. '
        'Path label guesses are not yet '
        'verified clinical labels.'
}

(
    OUT / 'audit_summary.json'
).write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True
    ) + '\n',
    encoding='utf-8'
)


# ==========================================
# 12. FINISH
# ==========================================

print(
    '\n===== MILESTONE 5B COMPLETE ====='
)

print(
    'Extracted recordings:',
    DATA
)

print(
    'Audit results:',
    OUT
)

print(
    'No model fitted; original AD/FTD '
    'reports and EEG unchanged.'
)

