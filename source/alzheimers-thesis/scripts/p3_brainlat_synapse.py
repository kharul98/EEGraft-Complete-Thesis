#!/usr/bin/env python3
"""List or selectively download BrainLat AD/HC EEGLAB files from Synapse.

Authentication is obtained from Synapse's local client config. Never write
Synapse tokens into scripts or Git.
"""
import argparse
import re
import shutil
from collections import deque
from pathlib import Path

import pandas as pd
import synapseclient

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/audit/p3_brainlat_synapse_inventory.csv'
DOWNLOADS = ROOT / 'data/brainlat'
GROUPS = {'AD': ('1_AD', 'syn53222482'), 'CN': ('5_HC', 'syn53222486')}
PATTERN = re.compile(r'sub-[0-9]+', re.IGNORECASE)


def list_children(syn, root_id, root_name, group):
    folders = deque([(root_id, root_name)])
    scanned = set()
    files = []
    while folders:
        folder_id, path = folders.popleft()
        if folder_id in scanned:
            continue
        scanned.add(folder_id)
        for child in syn.getChildren(folder_id, includeTypes=['folder', 'file', 'link']):
            name = child['name']
            child_id = child['id']
            kind = child.get('type', '')
            remote_path = f'{path}/{name}'
            if 'Folder' in kind:
                folders.append((child_id, remote_path))
            else:
                suffix = Path(name).suffix.lower()
                if suffix not in ('.set', '.fdt'):
                    continue
                match = PATTERN.search(name)
                if match is None:
                    continue
                parts = Path(remote_path).parts
                site = 'AR' if 'AR' in parts else ('CL' if 'CL' in parts else 'UNKNOWN')
                files.append(dict(group=group, site=site,
                                  participant_id=match.group(0),
                                  extension=suffix, synapse_id=child_id,
                                  remote_path=remote_path,
                                  name=name))
    return files, len(scanned)


def inventory(syn):
    files = []
    total_folders = 0
    for group, (group_name, group_id) in GROUPS.items():
        print(f'Indexing {group_name} ({group_id})...', flush=True)
        found, folders = list_children(syn, group_id, group_name, group)
        files.extend(found)
        total_folders += folders
        print(f'  indexed {len(found)} EEG .set/.fdt files in {folders} folders', flush=True)
    df = pd.DataFrame(files, columns=['group', 'site', 'participant_id',
                                      'extension', 'synapse_id', 'remote_path', 'name'])
    if df.empty or not {'.set', '.fdt'}.issubset(set(df.extension)):
        raise ValueError('Incomplete Synapse listing. Check access and folder contents.')
    if df.duplicated(['remote_path']).any():
        raise ValueError('Duplicated remote file path')
    if not df.group.isin(['AD', 'CN']).all():
        raise ValueError('Unexpected group')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(['group','site','participant_id','extension']).to_csv(MANIFEST, index=False)
    sets = df[df.extension.eq('.set')]
    print('\n===== BRAINLAT RELEASED FILE INVENTORY =====')
    print(sets.groupby(['group', 'site']).agg(recordings=('remote_path', 'size'),
                                               participant_ids=('participant_id','nunique')).to_string())
    print('EEGLAB .set files:', len(sets), 'companion .fdt files:', int(df.extension.eq('.fdt').sum()))
    print('Unique site-qualified subject keys:', sets[['group','site','participant_id']].drop_duplicates().shape[0])
    print('Metadata index:', MANIFEST)
    print('Raw recordings downloaded: NONE')
    return df


def download(syn, df, limit):
    sets = df[df.extension.eq('.set')].copy()
    if sets.duplicated(['group','site','participant_id']).any():
        raise ValueError('Multiple .set recordings per group/site/participant; manual review required')
    sites = {'AR', 'CL'}
    if set(sets.site) - sites:
        raise ValueError('Unexpected site; inspect inventory')
    if limit > 0:
        sets = (sets.sort_values(['group','site','participant_id'])
                .groupby(['group','site'],group_keys=False).head(limit))
    keys = sets[['group','site','participant_id']].drop_duplicates()
    selected = df.merge(keys,on=['group','site','participant_id'],how='inner')
    selected = selected.sort_values(['group','site','participant_id','extension'])
    selected = selected.drop_duplicates('remote_path')
    print(f'===== DOWNLOAD PLAN: {len(sets)} .set recordings, {len(selected)} total files =====',flush=True)
    print('Destination:', DOWNLOADS)
    done = 0
    for i, rec in enumerate(selected.itertuples(index=False), 1):
        target = DOWNLOADS / Path(rec.remote_path).parent
        target.mkdir(parents=True, exist_ok=True)
        expected = target / rec.name
        if expected.is_file() and expected.stat().st_size > 0:
            print(f'{i}/{len(selected)} EXISTING {rec.remote_path}', flush=True)
            done += 1
            continue
        free = shutil.disk_usage(target).free
        if free < 8 * 1024**3:
            print('STOP DOWNLOAD: available disk space below 8 GiB; existing files kept')
            break
        try:
            entity = syn.get(rec.synapse_id, downloadLocation=str(target),
                             downloadFile=True, followLink=True,
                             ifcollision='keep.local')
            actual = Path(entity.path) if getattr(entity, 'path', None) else expected
            if not actual.is_file() or actual.stat().st_size == 0:
                print(f'{i}/{len(selected)} FAILED validation: {rec.remote_path}', flush=True)
                continue
            if actual.name != rec.name:
                print(f'{i}/{len(selected)} FILENAME MISMATCH: {actual.name} != {rec.name}', flush=True)
                continue
            print(f'{i}/{len(selected)} DOWNLOADED {rec.remote_path} ({actual.stat().st_size} bytes)', flush=True)
            done += 1
        except Exception as exc:
            print(f'{i}/{len(selected)} DOWNLOAD FAILED {rec.remote_path}: {type(exc).__name__}: {exc}',flush=True)
    print(f'===== DOWNLOADS PRESENT OR RECEIVED: {done}/{len(selected)} =====')
    print('Raw EEG is under data/brainlat: DO NOT git add it')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='download EEG files after inventory')
    parser.add_argument('--per-group-site', type=int, default=2,
                        help='max subjects per AD/CN x AR/CL cell; 0 means all')
    args = parser.parse_args()
    if args.per_group_site < 0:
        raise ValueError('Per-group-site limit cannot be negative')
    syn = synapseclient.Synapse()
    syn.login()
    if args.download:
        if not MANIFEST.is_file():
            raise FileNotFoundError('First run without --download to create the metadata index')
        df = pd.read_csv(MANIFEST)
        download(syn,df,args.per_group_site)
    else:
        inventory(syn)


if __name__ == '__main__':
    main()
