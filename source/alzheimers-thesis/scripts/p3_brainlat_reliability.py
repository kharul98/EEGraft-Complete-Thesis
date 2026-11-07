#!/usr/bin/env python3
"""Frozen BrainLat external-cohort spectral reliability assay.

Use only released, already-preprocessed BrainLat EEG. Do NOT conflate with
causal, raw-on-device acquisition. No diagnostic training or group inference.
"""
import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import welch

NAMES = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2")
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))
CHECKPOINTS = (30, 60, 120, 180)
# Four pilot recordings were analyzed before full-cohort lock.
# Exclude from external primary inferential summary; retain in audit CSV.
PILOT_KEYS = {
    ('AR', 'AD', 'sub-30001'), ('CL', 'AD', 'sub-30003'),
    ('AR', 'CN', 'sub-100012'), ('CL', 'CN', 'sub-10001')
}


def mapping():
    src = mne.channels.make_standard_montage('biosemi128')
    dst = mne.channels.make_standard_montage('standard_1020')
    a = np.array([src.get_positions()['ch_pos'][n] for n in src.ch_names])
    b = np.array([dst.get_positions()['ch_pos'][n] for n in NAMES])
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    degree = np.rad2deg(np.arccos(np.clip(a @ b.T, -1, 1)))
    ix, iy = linear_sum_assignment(degree)
    correspondence = {NAMES[j]: (src.ch_names[i], float(degree[i, j])) for i, j in zip(ix, iy)}
    assert len(correspondence) == 19
    assert len({v[0] for v in correspondence.values()}) == 19
    if max(v[1] for v in correspondence.values()) > 11.0:
        raise RuntimeError('Unacceptable electrode mapping error')
    return correspondence


def metadata(path):
    name = path.name
    found = re.search(r'(sub-[0-9]+)', name)
    if found is None:
        raise ValueError(f'Missing BrainLat subject ID: {name}')
    subject = found.group(1)
    ancestry = list(path.parts)
    site = next((x for x in ancestry if x in ('AR', 'CL')), None)
    if '1_AD' in ancestry:
        group = 'AD'
    elif '5_HC' in ancestry:
        group = 'CN'
    elif subject.startswith('sub-3'):
        group = 'AD'  # Only for manually-uploaded four-file feasibility samples
    elif subject.startswith('sub-1'):
        group = 'CN'
    else:
        raise ValueError('Cannot infer group from path')
    return subject, group, site or 'UNKNOWN'


def epoch_bands(epoch, fs):
    samples = int(round(fs * 4))
    frequencies, power = welch(epoch, fs=fs, nperseg=samples,
                               noverlap=samples // 2, axis=-1)
    total_mask = (frequencies >= 1) & (frequencies <= 40)
    total = np.trapezoid(power[:, total_mask], frequencies[total_mask], axis=-1)
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError('Non-positive or non-finite EEG spectrum')
    channels = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies <= high)
        channels.append(np.trapezoid(power[:, mask], frequencies[mask], axis=-1) / total)
    return np.stack(channels, axis=-1)


def average_epochs(epochs, first, last):
    return np.median(epochs[first:last], axis=(0, 1))


def tv(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a /= a.sum()
    b /= b.sum()
    return float(np.abs(a - b).sum() / 2)


def run(args):
    source = Path(args.input_dir).resolve()
    target = Path(args.output_dir).resolve()
    paths = sorted(source.rglob('*.set'))
    paths = [p for p in paths if ('_eeg' in p.name.lower())]
    print('EEGLAB .set files found:', len(paths), flush=True)
    if not paths:
        print('NO DATA: no EEG .set files under', source)
        return 2
    montage = mapping()
    select_names = [montage[n][0] for n in NAMES]
    rows, rejected = [], []
    for i, path in enumerate(paths, 1):
        try:
            subject, group, site = metadata(path)
            raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose='ERROR')
            sf = float(raw.info['sfreq'])
            if abs(sf - 512) > 1e-7:
                raise ValueError(f'Unexpected sampling frequency {sf}')
            if not set(select_names).issubset(raw.ch_names) or len(raw.ch_names) != 128:
                raise ValueError('Missing expected BioSemi128 channels')
            if raw.n_times < 300 * 512:
                raise ValueError(f'Recording shorter than 300 seconds: {raw.n_times / sf:.1f}s')
            picks = [raw.ch_names.index(x) for x in select_names]
            spectra = []
            for block in range(30):
                samples = raw.get_data(picks=picks, start=block * 5120, stop=(block + 1) * 5120)
                if samples.shape != (19, 5120) or not np.isfinite(samples).all():
                    raise ValueError('Invalid EEG block or missing data')
                spectra.append(epoch_bands(samples, 512))
            spectra = np.stack(spectra)
            ref = average_epochs(spectra, 24, 30)
            p60 = average_epochs(spectra, 0, 6)
            p120 = average_epochs(spectra, 0, 12)
            p180 = average_epochs(spectra, 0, 18)
            f60 = average_epochs(spectra, 12, 18)
            f120 = average_epochs(spectra, 6, 18)
            ref_a = average_epochs(spectra, 24, 27)
            ref_b = average_epochs(spectra, 27, 30)
            record = dict(group=group, site=site, participant_id=subject,
                          pilot_seen=(site, group, subject) in PILOT_KEYS,
                          source_path=str(path.relative_to(source)), recording_seconds=float(raw.n_times / sf),
                          prefix_30_tv=tv(average_epochs(spectra, 0, 3), ref),
                          prefix_60_tv=tv(p60, ref), prefix_120_tv=tv(p120, ref),
                          prefix_180_tv=tv(p180, ref),
                          fixed_end_60_tv=tv(f60, ref), fixed_end_120_tv=tv(f120, ref),
                          fixed_end_180_tv=tv(p180, ref),
                          late_halves_tv=tv(ref_a, ref_b))
            record['primary_delta_180_minus_60'] = record['fixed_end_180_tv'] - record['fixed_end_60_tv']
            rows.append(record)
            print(f'{i}/{len(paths)} OK {site}/{group}/{subject} primary_delta={record["primary_delta_180_minus_60"]:+.6f}', flush=True)
        except Exception as exc:
            rejected.append(dict(source_path=str(path.relative_to(source)), reason=f'{type(exc).__name__}: {exc}'))
            print(f'{i}/{len(paths)} INELIGIBLE {path.name}: {exc}', flush=True)
    if not rows:
        print('NO ELIGIBLE RECORDINGS, no outcome tables written')
        return 3
    df = pd.DataFrame(rows).sort_values(['site', 'group', 'participant_id']).reset_index(drop=True)
    if df.duplicated(['site', 'group', 'participant_id']).any():
        print('DUPLICATE site-qualified participant IDs, outputs not saved')
        return 4
    target.mkdir(parents=True, exist_ok=True)
    df.to_csv(target / 'brainlat_reliability_participant.csv', index=False, float_format='%.12g')
    pd.DataFrame(rejected, columns=['source_path', 'reason']).to_csv(target / 'brainlat_reliability_exclusions.csv', index=False)
    eligible = df.loc[~df.pilot_seen].reset_index(drop=True)
    group_site_counts = eligible.groupby(['site','group']).size()
    four_cells = [('AR','AD'), ('AR','CN'), ('CL','AD'), ('CL','CN')]
    is_full = all(group_site_counts.get(cell,0) >= 8 for cell in four_cells)
    bootstrap_ci = None
    if is_full:
        rng = np.random.default_rng(20261107)
        sampled = []
        for site, group in four_cells:
            ids = np.flatnonzero((eligible.site == site) & (eligible.group == group))
            sampled.append(rng.choice(ids,size=(5000,len(ids)),replace=True))
        all_samples = np.concatenate(sampled,axis=1)
        interval = np.quantile(eligible.primary_delta_180_minus_60.to_numpy()[all_samples].mean(axis=1),[.025,.975])
        bootstrap_ci = [float(interval[0]),float(interval[1])]
    info = dict(status='exploratory_external' if is_full else 'feasibility_only',
                n_participants=int(len(df)), n_untouched_external=int(len(eligible)), n_pilot_preseen=int(df.pilot_seen.sum()), n_ineligible=len(rejected),
                n_by_group_site=df.groupby(['site', 'group']).size().rename('n').reset_index().to_dict('records'),
                primary='fixed-endpoint 0-180 vs 120-180, reference 240-300, normalized five-band TV',
                pilot_inclusive_mean=float(df.primary_delta_180_minus_60.mean()),
                mean_primary_delta=float(eligible.primary_delta_180_minus_60.mean()) if len(eligible) else None,
                median_primary_delta=float(eligible.primary_delta_180_minus_60.median()) if len(eligible) else None,
                site_group_stratified_bootstrap_95pct_ci=bootstrap_ci,
                exploratory_cell_minimum_n=8,
                approximate_channel_mapping={k:{'biosemi':v[0], 'angle_degrees':v[1]} for k,v in montage.items()},
                restrictions=['Released BrainLat is offline preprocessed, not causal/raw.',
                              'Clinical class labels are not used for prediction.',
                              '19 mapped electrodes are approximate, not physically identical to NEMAR.',
                              'Reference is another time window in same session, not an independent clinical ground truth.',
                              'No causal-time or independent-diagnostic claim.'])
    (target / 'brainlat_reliability_summary.json').write_text(json.dumps(info, indent=2) + '\n')
    print('=== BrainLat external feasibility ===')
    print('Eligible:', len(df), 'Excluded:', len(rejected))
    print('Untouched external participants:',len(eligible),'pilot-previously-seen:',int(df.pilot_seen.sum()))
    print('Primary effect mean (untouched):',round(info['mean_primary_delta'], 6) if info['mean_primary_delta'] is not None else 'not available', '(negative favors 180s)')
    print('Status:', info['status'])
    print('Stratified 95% bootstrap interval:', bootstrap_ci)
    print('Saved', target)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    raise SystemExit(run(ap.parse_args()))
