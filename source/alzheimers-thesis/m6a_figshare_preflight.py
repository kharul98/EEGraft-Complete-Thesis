#!/usr/bin/env python3
"""M6A: independent AD archive inventory; no EEG processing or training."""

from pathlib import Path, PurePosixPath
from collections import Counter
from urllib.request import Request, urlopen
from zipfile import ZipFile
from io import StringIO
import csv
import hashlib
import json
import os
import shutil

ROOT = Path.cwd()
OUT = ROOT / 'reports/thesis_final/milestone_6a_figshare_archive_preflight_v1'
DEST = ROOT / 'data/external/cejnek_figshare_v1'

OUT.mkdir(parents=True, exist_ok=True)
DEST.mkdir(parents=True, exist_ok=True)

API = 'https://api.figshare.com/v2/articles/5450293'
FILE_ID = 9423433


def get_json(url):
    req = Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 EEG research archive audit'}
    )
    with urlopen(req, timeout=45) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(2 * 1024 * 1024),
            b''
        ):
            h.update(chunk)
    return h.hexdigest()


def freeze(path, content):
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError(
                'Existing frozen result differs: ' + str(path)
            )
    else:
        path.write_text(content, encoding='utf-8')


def candidate_paths():
    yield DEST / 'dataset.zip'

    for drive in ('c', 'd', 'e'):
        base = Path('/mnt') / drive / 'Users'

        if base.is_dir():
            for user in base.iterdir():
                if user.is_dir():
                    for d in (
                        user / 'Downloads',
                        user / 'OneDrive' / 'Downloads'
                    ):
                        if d.is_dir():
                            yield from d.glob('dataset*.zip')

    d = Path.home() / 'Downloads'
    if d.is_dir():
        yield from d.glob('dataset*.zip')


print(
    '===== MILESTONE 6A: FIGSHARE '
    'INDEPENDENT-AD ARCHIVE PREFLIGHT =====',
    flush=True
)

article = get_json(API)

files = [
    f for f in article.get('files', [])
    if int(f.get('id', -1)) == FILE_ID
]

if len(files) != 1:
    raise RuntimeError(
        'Figshare file ID not found in official metadata'
    )

f = files[0]
size = int(f['size'])

if size <= 0 or size > 2 * 1024**3:
    raise RuntimeError('Unexpected archive size')

print('Article:', article.get('title'))
print('Official archive:', f['name'])
print('Official size:', size, 'bytes')
print(
    'Declared license:',
    (article.get('license') or {}).get('name')
)

archive = next(
    (
        p for p in candidate_paths()
        if p.is_file() and p.stat().st_size == size
    ),
    None
)

if archive is None:
    archive = DEST / 'dataset.zip'

    if archive.exists():
        raise RuntimeError(
            'Existing dataset.zip size differs; '
            'no overwrite attempted'
        )

    url = (
        f.get('download_url')
        or f'https://figshare.com/ndownloader/files/{FILE_ID}'
    )

    print(
        'No exact-size browser download found. '
        'Downloading official ZIP...',
        flush=True
    )

    part = DEST / 'dataset.zip.incomplete'

    if part.exists():
        part.unlink()

    try:
        req = Request(
            url,
            headers={
                'User-Agent':
                    'Mozilla/5.0 EEG research archive audit'
            }
        )

        with (
            urlopen(req, timeout=90) as response,
            part.open('wb') as destination
        ):
            shutil.copyfileobj(
                response,
                destination,
                length=2 * 1024 * 1024
            )

        if part.stat().st_size != size:
            raise RuntimeError(
                f'Download size mismatch: '
                f'{part.stat().st_size} vs {size}'
            )

        os.replace(part, archive)

    except Exception:
        if part.exists():
            part.unlink()
        raise

else:
    print(
        'Found existing matching archive; '
        'no download needed.'
    )

print('Using archive:', archive)
print('ZIP size matches Figshare metadata: PASS')

rows = []

with ZipFile(archive) as zf:
    infos = zf.infolist()

    if not infos or len(infos) > 20000:
        raise RuntimeError(
            'Empty or unexpectedly large ZIP directory'
        )

    for info in infos:
        path = PurePosixPath(
            info.filename.replace('\\', '/')
        )

        if (
            path.is_absolute()
            or '..' in path.parts
            or not path.parts
        ):
            raise RuntimeError(
                'Unsafe archive member path'
            )

        if info.is_dir():
            continue

        parts = [
            s.lower()
            for s in path.parts
        ]

        group = 'UNCLASSIFIED'

        for segment in parts[:-1]:
            if segment in (
                'ad', 'alzheimer', 'alzheimers'
            ):
                group = 'AD'
                break

            if segment in ('mci', 'amci'):
                group = 'MCI'
                break

            if segment in (
                'control', 'controls',
                'normal', 'healthy', 'hc', 'cn'
            ):
                group = 'CONTROL'
                break

        rows.append({
            'archive_member': path.as_posix(),
            'group_path_hint': group,
            'extension': path.suffix.lower(),
            'uncompressed_bytes': info.file_size,
            'compressed_bytes': info.compress_size,
            'crc32_hex': f'{info.CRC:08x}'
        })

if not rows:
    raise RuntimeError('Archive contains no files')

counts = Counter(
    (r['group_path_hint'], r['extension'])
    for r in rows
)

print('\n===== ARCHIVE STRUCTURE =====')
print('Total files:', len(rows))

print(
    'Extension counts:',
    dict(Counter(
        r['extension'] for r in rows
    ))
)

print('Group/extension path hints:')

for (group, ext), count in sorted(counts.items()):
    print(
        ' ',
        group,
        ext or '(no extension)',
        count
    )

print('First 25 archive entries:')

for r in rows[:25]:
    print(
        ' ',
        r['archive_member'][:180]
    )

buf = StringIO()

writer = csv.DictWriter(
    buf,
    fieldnames=list(rows[0]),
    lineterminator='\n'
)

writer.writeheader()
writer.writerows(rows)

freeze(
    OUT / 'archive_members.csv',
    buf.getvalue()
)

metadata = {
    'article_id': 5450293,
    'file_id': FILE_ID,
    'official_name': f['name'],
    'official_size_bytes': size,
    'archive_size_bytes': archive.stat().st_size,
    'archive_sha256': sha256(archive),
    'archive_path': str(archive),
    'archive_members': len(rows),
    'mat_files': sum(
        r['extension'] == '.mat'
        for r in rows
    ),
    'subject_identity_verified': False,
    'channel_harmonization_verified': False,
    'continuous_EEG_verified': False,
    'training_performed': False
}

freeze(
    OUT / 'source_and_status.json',
    json.dumps(
        metadata,
        indent=2,
        sort_keys=True
    ) + '\n'
)

print('\n===== MILESTONE 6A COMPLETE =====')
print('ZIP inventory written to:', OUT)
print(
    'No archive extraction, '
    'no EEG processing, no training.'
)
print(
    'Subject counts and channel compatibility '
    'are NOT yet established.'
)

