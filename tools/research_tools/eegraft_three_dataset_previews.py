#!/usr/bin/env python3
"""Evidence-grounded previews of NEMAR, Isfahan MCI and CAUEEG train EEG.
Does not touch CAUEEG validation/test EEG. Never fabricates waveforms.
"""
import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

HOME = Path.home()
OUT = HOME / 'research' / 'eegraft-dataset-previews'
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [Path(x).expanduser() for x in os.environ.get(
    'EEG_SEARCH_ROOTS',
    ':'.join(str(HOME / p) for p in ('research', 'alzheimers-thesis', 'data', 'datasets', 'Downloads'))
).split(':') if x]

# Optional precise folder settings to avoid automated discovery mistakes.
OVERRIDE = {k: os.environ.get(k + '_ROOT', '') for k in ('NEMAR', 'ISFAHAN', 'CAUEEG')}
BAD_DIRS = {'.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__',
            'derived', 'results', 'output', 'outputs', 'figures', 'plots', 'images', 'cache'}
FORMATS = {'.set', '.edf', '.bdf', '.vhdr', '.fif'}


def inspect_roots():
    seen = set()
    inventory = []
    roots = [Path(v).expanduser() for v in OVERRIDE.values() if v] + SEARCH_ROOTS
    for base in roots:
        if not base.exists():
            continue
        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for cur, dirs, files in os.walk(base):
                depth = len(Path(cur).relative_to(base).parts)
                dirs[:] = [d for d in dirs if d not in BAD_DIRS and depth < 12]
                for name in files:
                    if Path(name).suffix.lower() in FORMATS:
                        candidates.append(Path(cur) / name)
        for p in candidates:
            path = str(p.resolve())
            if path not in seen:
                seen.add(path)
                inventory.append(p)
    return inventory


def within(p, configured):
    if not configured:
        return True
    root = Path(configured).expanduser().resolve()
    return p.resolve() == root or root in p.resolve().parents


def choose_nemar(inventory):
    options = [p for p in inventory if p.suffix.lower() == '.set' and within(p, OVERRIDE['NEMAR'])]
    exact = [p for p in options if any(x in str(p).lower() for x in ('ds004504', 'on004504', 'nemar', 'miltiadous'))]
    if exact:
        return sorted(exact, key=str)
    if OVERRIDE['NEMAR'] and options:
        return sorted(options, key=str)
    raise RuntimeError('NEMAR .set files not identified. Set NEMAR_ROOT to the source EEG dataset folder.')


def choose_isfahan(inventory):
    options = [p for p in inventory if p.suffix.lower() in {'.edf', '.bdf', '.set', '.vhdr', '.fif'}
               and within(p, OVERRIDE['ISFAHAN'])]
    keywords = ('isfahan', 'eeg_1', 'eeg_2', 'mci', 'kashefpoor', 'iranian')
    selected = [p for p in options if any(k in str(p).lower() for k in keywords)
                and 'caueeg' not in str(p).lower()]
    if not selected and OVERRIDE['ISFAHAN']:
        selected = [p for p in options if 'caueeg' not in str(p).lower()]
    if not selected:
        raise RuntimeError('Isfahan EDF/EEG files not identified. Set ISFAHAN_ROOT to its EEG_1/EEG_2 folder.')
    return sorted(selected, key=lambda p: ('eeg_1' not in str(p).lower(), str(p)))


def training_normal_serials():
    layout = HOME / 'research/eegraft-caueeg/protocol/dataset_layout.json'
    if not layout.exists():
        raise RuntimeError('Missing CAUEEG dataset_layout.json. A training-only official split is mandatory.')
    record = json.loads(layout.read_text())
    split_path = Path(record['split_file']).expanduser()
    if not split_path.is_file():
        raise RuntimeError('Official CAUEEG split file not found: ' + str(split_path))
    split = json.loads(split_path.read_text())
    ids = set()
    for item in split['train_split']:
        if str(item['class_name']).strip().lower() == 'normal':
            ids.add(str(int(str(item['serial']))).zfill(5))
    if len(ids) != 367:
        raise RuntimeError(f'Expected 367 CAUEEG training Normal recordings; found {len(ids)}')
    return ids, split_path


def choose_caueeg(inventory):
    serials, split = training_normal_serials()
    options = [p for p in inventory if p.suffix.lower() in ('.edf', '.bdf')
               and within(p, OVERRIDE['CAUEEG'])
               and ('caueeg' in str(p).lower() or OVERRIDE['CAUEEG'])]
    choices = []
    for p in options:
        # Match an exact numeric token of the basename, never a substring.
        numeric_tokens = re.findall(r'\d+', p.stem)
        if any(str(int(x)).zfill(5) in serials for x in numeric_tokens):
            choices.append(p)
    if not choices:
        raise RuntimeError('No official CAUEEG *training Normal* EDF matched its serial. '
                           'Set CAUEEG_ROOT to the raw EDF folder; do not use test recordings.')
    return sorted(choices, key=str)


def standard_name(name):
    letters = re.sub(r'[^a-zA-Z0-9]', '', name).upper()
    for prefix in ('EEG', 'POL', 'CH'):
        if letters.startswith(prefix):
            letters = letters[len(prefix):]
    return letters


def get_sample(p):
    ext = p.suffix.lower()
    if ext == '.set':
        raw = mne.io.read_raw_eeglab(p, preload=False, verbose='ERROR')
    elif ext in ('.edf', '.bdf'):
        loader = mne.io.read_raw_edf if ext == '.edf' else mne.io.read_raw_bdf
        raw = loader(p, preload=False, verbose='ERROR')
    elif ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(p, preload=False, verbose='ERROR')
    elif ext == '.fif':
        raw = mne.io.read_raw_fif(p, preload=False, verbose='ERROR')
    else:
        raise RuntimeError('Unsupported source file extension')
    try:
        sf = float(raw.info['sfreq'])
        duration = float(raw.n_times / sf)
        if not (sf >= 100 and duration >= 8):
            raise RuntimeError('Recording too short or impossible sampling frequency')
        eeg = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False,
                             ecg=False, exclude=[])
        if len(eeg) < 4:
            raise RuntimeError('Not enough EEG channels identified: ' + str(raw.ch_names[:10]))
        picks = []
        for target in ('FP1', 'F3', 'C3', 'P3', 'O1', 'O2'):
            for idx in eeg:
                label = standard_name(raw.ch_names[idx])
                if label == target or label.startswith(target + 'REF') or label.startswith(target + 'AVG'):
                    if int(idx) not in picks:
                        picks.append(int(idx))
                    break
        for idx in eeg:
            if len(picks) >= 6:
                break
            if int(idx) not in picks:
                picks.append(int(idx))
        picks = picks[:6]
        # A deterministic sample interval; no optimization for a prettier trace.
        start_seconds = 30.0 if duration >= 36 else 1.0
        start = int(round(start_seconds * sf))
        stop = min(raw.n_times, start + int(round(6.0 * sf)))
        if stop - start < 4 * sf:
            raise RuntimeError('Too little recording available for a real excerpt')
        signal_uv = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
        if not np.isfinite(signal_uv).all():
            raise RuntimeError('Non-finite sample values. Refusing to interpolate or fabricate data.')
        t = np.arange(stop - start) / sf
        return {'source': p, 'sample': signal_uv, 'time': t, 'sf': sf,
                'duration': duration, 'channels': [raw.ch_names[i] for i in picks],
                'total_eeg_channels': len(eeg), 'start_seconds': start_seconds,
                'sample_count': int(stop-start)}
    finally:
        raw.close()


def render(data, index, title, subtitle, study_note, restricted=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    navy = '#122940'
    cyan = '#2276a8'
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8.3, 9.5), dpi=210)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.16, right=0.95, top=0.79, bottom=0.15, hspace=0.30)
    fig.text(0.07, 0.965, f'DATASET {index:02d}  |  {title}', fontsize=19,
             fontweight='bold', color=navy, va='top')
    fig.text(0.07, 0.919, subtitle, fontsize=11.5, color='#35566f', va='top')
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.893, 0.893], color=cyan, linewidth=2.1))
    fig.text(0.07, 0.873, study_note, fontsize=10.0, color=navy, va='top')
    fig.text(0.07, 0.832,
             f'Observed file: {data["total_eeg_channels"]} EEG channels   |   '
             f'{data["sf"]:g} Hz   |   {data["duration"] / 60:.1f} min',
             fontsize=10, color='#35566f', va='top')
    for i, ax in enumerate(axs):
        # Remove only each displayed segment median to make channel amplitudes visible.
        uv = data['sample'][i] - np.median(data['sample'][i])
        ax.plot(data['time'], uv, linewidth=0.86, color=cyan if i % 2 == 0 else '#386b7d')
        ax.set_ylabel(data['channels'][i].replace('EEG ', '')[:15], rotation=0,
                      ha='right', va='center', labelpad=13,
                      color=navy, fontweight='bold', fontsize=10)
        ax.grid(axis='both', color='#e8edf1', linewidth=0.6)
        span = max(abs(np.min(uv)), abs(np.max(uv)), 1)
        ax.set_ylim(-span * 1.13, span * 1.13)
        ax.tick_params(labelsize=8, colors='#4d677b', length=2)
        ax.set_xlim(float(data['time'][0]), float(data['time'][-1]))
        if i == 3:
            ax.text(-0.12, 0.5, 'μV', transform=ax.transAxes, rotation=90,
                    color='#5a6f7f', fontsize=8, ha='center', va='center')
    axs[-1].set_xlabel('Time within displayed 6-second EEG excerpt (s)',
                       fontsize=10, color=navy, labelpad=8)
    fig.text(0.07, 0.092,
             f'Authentic raw-signal excerpt at {data["start_seconds"]:g} s from one recording. '
             'Median-centered per channel ONLY for display.',
             fontsize=8.8, color='#445d72')
    fig.text(0.07, 0.063,
             'No generated signal, artifact removal, resampling, or model prediction shown.',
             fontsize=8.8, color='#445d72')
    if restricted:
        fig.text(0.07, 0.035, 'RESTRICTED DATA: check CAUEEG permission before public distribution.',
                 color='#9e4c3e', fontsize=8.6, fontweight='bold')
    output = OUT / f'{index:02d}_{title.replace(" ", "_")}_Real_EEG_Preview.png'
    fig.savefig(output, dpi=230, facecolor='white')
    plt.close(fig)
    return output


def main():
    print('1. Discovering existing local EEG files (metadata and filenames only)...')
    files = inspect_roots()
    print('EEG-format file count:', len(files))
    if not files:
        raise RuntimeError('No EEG recording files found. Set EEG_SEARCH_ROOTS to your dataset folders.')
    configs = [
        ('NEMAR', choose_nemar, 1, 'NEMAR ds004504',
         'Original AD, FTD and healthy-control cohort',
         '88 participants: 36 AD | 23 FTD | 29 controls', False),
        ('ISFAHAN', choose_isfahan, 2, 'Isfahan MCI EEG',
         'Independent mild-cognitive-impairment negative control',
         '61 screened; 49 retained (EEG_1: 25 | EEG_2: 24)', False),
        ('CAUEEG', choose_caueeg, 3, 'CAUEEG',
         'Frozen external transfer: Normal versus broad Dementia',
         '1,122 benchmark recordings: train 950 | val 82 | test 90', True),
    ]
    evidence = []
    produced = []
    for key, picker, number, title, subtitle, note, restricted in configs:
        print(f'\n2. {key}: locating and verifying source signal...')
        try:
            candidates = picker(files)
            last_error = ''
            selected = None
            for path in candidates[:40]:
                try:
                    selected = get_sample(path)
                    break
                except Exception as exc:
                    last_error = f'{type(exc).__name__}: {exc}'
            if selected is None:
                raise RuntimeError(f'{len(candidates)} candidate files; none readable. Last: {last_error}')
            image_path = render(selected, number, title, subtitle, note, restricted)
            produced.append(image_path)
            evidence.append({
                'dataset': key, 'source_file_PRIVATE': str(selected['source'].resolve()),
                'sampling_frequency_hz': selected['sf'],
                'original_duration_s': selected['duration'],
                'original_eeg_channel_count': selected['total_eeg_channels'],
                'displayed_channels': selected['channels'],
                'start_seconds': selected['start_seconds'],
                'sample_count': selected['sample_count'],
                'displayed_source_samples_sha256': hashlib.sha256(
                    np.ascontiguousarray(selected['sample']).tobytes()).hexdigest(),
                'png': str(image_path),
                'restricted_release_review_needed': restricted,
            })
            print('[SAVED]', image_path)
            print('   Sampling rate:', selected['sf'], 'Hz')
            print('   EEG channels in actual file:', selected['total_eeg_channels'])
            print('   Source recording (private):', selected['source'])
        except Exception as exc:
            print(f'[NOT GENERATED] {key}: {exc}')
    provenance = OUT / 'PRIVATE_Source_Provenance.json'
    previous_umask = os.umask(0o077)
    try:
        provenance.write_text(json.dumps(evidence, indent=2) + '\n')
        provenance.chmod(0o600)
    finally:
        os.umask(previous_umask)
    print('\n' + '=' * 62)
    print('GENERATED PREVIEWS:', len(produced), '/ 3')
    for p in produced:
        print(' -', p)
    print('PRIVATE provenance:', provenance)
    if len(produced) != 3:
        print('NEEDS_PATH_CHECK: set appropriate *_ROOT env variable(s) and rerun.')
    else:
        print('THREE_DATASET_PREVIEWS_STATUS: COMPLETED')
    print('IMPORTANT: confirm CAUEEG reuse permissions before publishing its raw excerpt.')

if __name__ == '__main__':
    main()
