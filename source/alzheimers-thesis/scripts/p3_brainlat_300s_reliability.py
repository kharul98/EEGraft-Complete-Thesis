#!/usr/bin/env python3
"""BrainLat offline-preprocessed EEG: frozen Dryad endpoint, 19 spatially matched sensors."""
from pathlib import Path
from collections import Counter
from math import gcd
import gc
import importlib.util
import json

import mne
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/p3_brainlat_readiness.csv'
OUT = ROOT / 'results/p3_brainlat_300s_reliability.csv'
EXCLUDED = ROOT / 'results/p3_brainlat_300s_exclusions.csv'
SUMMARY = ROOT / 'docs/audit/p3_brainlat_300s_summary.json'
TARGET = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
          'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3',
          'Pz', 'P4', 'P8', 'O1', 'O2']


def montage_selection():
    bio = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']
    standard = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    names = list(bio)

    def direction(point):
        value = np.asarray(point, dtype=float)
        return value / np.linalg.norm(value)

    cost = np.array([
        [np.linalg.norm(direction(standard[t]) - direction(bio[n])) for n in names]
        for t in TARGET
    ])
    rows, columns = linear_sum_assignment(cost)
    selected = {TARGET[int(r)]: names[int(c)] for r, c in zip(rows, columns)}
    assert len(selected) == 19 and len(set(selected.values())) == 19
    return [selected[t] for t in TARGET], selected, float(cost[rows, columns].max())


def site_from_path(path):
    tokens = {part.upper() for part in path.parts}
    sites = tokens.intersection({'AR', 'CL', 'CO', 'MX', 'PE'})
    return next(iter(sites)) if len(sites) == 1 else 'UNSPECIFIED'


def describe(group):
    if group.empty:
        return {'n': 0}
    d = group['delta_180_minus_60'].to_numpy(dtype=float)
    return {
        'n': int(len(group)),
        'mean_180_minus_60': float(d.mean()),
        'median_180_minus_60': float(np.median(d)),
        'fraction_negative': float((d < 0).mean()),
        'median_fixed_60_tv': float(group.fixed_end_60_tv.median()),
        'median_fixed_120_tv': float(group.fixed_end_120_tv.median()),
        'median_fixed_180_tv': float(group.fixed_end_180_tv.median()),
    }


def main():
    print('===== BRAINLAT REAL EEG: 300-SECOND RELIABILITY =====', flush=True)
    if not INPUT.is_file():
        print('ERROR: BrainLat readiness CSV missing. No analysis performed.')
        return

    module_path = ROOT / 'scripts/p3_dryad_ad_signal_stage.py'
    if not module_path.is_file():
        print('ERROR: Frozen Dryad analysis function missing.')
        return

    spec = importlib.util.spec_from_file_location('dryad_frozen', module_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if frozen.FS != 500 or frozen.NEEDED != 150000:
        raise ValueError('Unexpected frozen Dryad analysis settings')

    channel_names, mapping, maximum_distance = montage_selection()
    print('Fixed 19-channel BioSemi128 sensor mapping:', mapping)
    print('Maximum unit-sphere montage distance:', round(maximum_distance, 4))
    print('EEG is already offline-preprocessed; online causal claims are NOT tested.')

    inventory = pd.read_csv(INPUT)
    needed_cols = {'cohort_candidate', 'source_path', 'readable', 'eligible_300s_length_only'}
    if not needed_cols.issubset(inventory.columns):
        raise ValueError('BrainLat readiness CSV is missing expected columns')
    eligible = inventory.loc[
        inventory['cohort_candidate'].isin(['AD', 'HC'])
        & inventory['readable'].eq(True)
        & inventory['eligible_300s_length_only'].eq(True)
    ].sort_values(['cohort_candidate', 'source_path'])
    print('Length-eligible records:', eligible.groupby('cohort_candidate').size().to_dict())

    results, exclusions = [], []
    for index, entry in enumerate(eligible.itertuples(index=False), 1):
        path = Path(entry.source_path)
        label = entry.cohort_candidate
        site = site_from_path(path)
        raw = None
        try:
            if not path.is_file():
                raise FileNotFoundError('Source SET file not found')
            raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose='ERROR')
            rate = float(raw.info['sfreq'])
            rounded_rate = int(round(rate))
            if rounded_rate <= 0 or abs(rate - rounded_rate) > 1e-6:
                raise ValueError('Noninteger source sampling frequency')
            n_src = rounded_rate * 300
            if raw.n_times < n_src:
                raise ValueError('Recording has fewer than 300 seconds')

            types = dict(zip(raw.ch_names, raw.get_channel_types()))
            if any(types.get(ch) != 'eeg' for ch in channel_names):
                raise ValueError('One or more mapped scalp EEG channels missing')
            if any(ch in raw.info['bads'] for ch in channel_names):
                raise ValueError('Mapped scalp channel flagged bad')

            values = raw.get_data(picks=channel_names, start=0, stop=n_src).T
            if values.shape != (n_src, 19) or not np.isfinite(values).all():
                raise ValueError('Invalid channel data dimensions or finite values')
            rms_uv = float(np.median(np.sqrt(np.mean(values ** 2, axis=0))) * 1e6)
            if rms_uv <= 0:
                raise ValueError('Zero median channel RMS')

            ratio = gcd(500, rounded_rate)
            x = resample_poly(values, 500 // ratio, rounded_rate // ratio, axis=0)
            if x.shape != (150000, 19):
                raise ValueError(f'Unexpected resampled EEG shape: {x.shape}')
            metrics = frozen.analyze(x)
            delta = metrics['fixed_end_180_tv'] - metrics['fixed_end_60_tv']
            if not np.isfinite(delta) or not all(0 <= v <= 1 for v in metrics.values()):
                raise ValueError('Invalid spectral metric')

            results.append({
                'cohort': label, 'site_from_folder': site, 'recording_code': path.stem,
                'source_sfreq_hz': rate, 'median_raw_channel_rms_uv': rms_uv,
                'delta_180_minus_60': delta, **metrics,
            })
        except Exception as exc:
            exclusions.append({
                'cohort': label, 'site_from_folder': site, 'recording_code': path.stem,
                'reason': f'{type(exc).__name__}: {str(exc)[:150]}'
            })
        finally:
            if raw is not None:
                raw.close()
            del raw
            gc.collect()
        if index % 5 == 0 or index == len(eligible):
            print(f'Processed {index}/{len(eligible)} records', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    pd.DataFrame(exclusions, columns=['cohort','site_from_folder','recording_code','reason']).to_csv(EXCLUDED, index=False)
    if df.empty:
        print('No valid results. See exclusions file; previous results unchanged.')
        return
    df.to_csv(OUT, index=False, float_format='%.12g')
    cohorts = {g: describe(df[df.cohort == g]) for g in ('AD','HC')}
    by_site = {
        f'{group}/{site}': describe(part)
        for (group, site), part in df.groupby(['cohort', 'site_from_folder'])
    }
    summary = {
        'status': 'EXPLORATORY, PREPROCESSED EEG, SELECTION-BIASED HC',
        'data': 'BrainLat downloaded physical SET/FDT files',
        'endpoint': 'Fixed-end 180s minus fixed-end 60s TV versus disjoint 240-300s reference',
        'primary_windows_seconds': {'fixed_60': [120,180], 'fixed_180': [0,180], 'reference': [240,300]},
        'reliability_function': 'Exact frozen p3_dryad_ad_signal_stage.analyze',
        'resampling': 'scipy.signal.resample_poly to 500 Hz before frozen analysis',
        'mapped_channel_labels': mapping,
        'max_montage_unit_sphere_distance': maximum_distance,
        'eligible_readiness_rows': int(len(eligible)),
        'successful_recordings': int(len(df)),
        'excluded_recordings': len(exclusions),
        'cohorts': cohorts,
        'by_site': by_site,
        'limitations': [
            'BrainLat EEG has already been offline filtered and artifact corrected.',
            '19 electrodes are geometrically nearest BioSemi128 sensors, not identical 10-20 electrodes.',
            '36 HC records had missing FDT; usable controls are heavily selected.',
            'Folder site labels are provisional and require metadata verification.',
            'No independent causal online validation or diagnosis claim can be made.'
        ]
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + '\n')
    print('\n===== BRAINLAT RELIABILITY RESULTS =====')
    print('Successful:', len(df), '| Excluded:', len(exclusions))
    for example in exclusions[:5]:
        print('Example exclusion:', example['cohort'], example['recording_code'], example['reason'])
    for cohort, stats in cohorts.items():
        print(cohort, ':', stats)
    print('\n===== SITE STRATIFICATION =====')
    for k, stats in by_site.items():
        print(k, ':', stats)
    print('\n===== INPUT FILE PROTOCOL CHECK =====')
    for group in ('AD','HC'):
        t = df.loc[df.cohort == group, 'recording_code'].str.lower()
        print(group, 'rs-HEP:', int(t.str.contains('rs-hep').sum()), 'rs:', int((~t.str.contains('rs-hep')).sum()))
    print('No p-values, patient-level CIs, or online-causality claims.')
    print('===== BRAINLAT RELIABILITY STAGE COMPLETE =====')


if __name__ == '__main__':
    main()
