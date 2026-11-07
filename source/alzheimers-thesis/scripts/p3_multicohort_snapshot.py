#!/usr/bin/env python3
"""Descriptive multisource duration-reliability evidence snapshot; no raw EEG read."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
AUDIT = ROOT / 'docs' / 'audit'


def required_csv(name, required):
    path = RESULTS / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing prerequisite: {path}')
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return frame


def series_checked(frame, column, expected, identity_cols):
    if len(frame) != expected:
        raise ValueError(f'Expected {expected} rows, found {len(frame)} for {column}')
    if frame.duplicated(identity_cols).any():
        raise ValueError(f'Duplicate analysis-unit keys: {identity_cols}')
    arr = pd.to_numeric(frame[column], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(arr).all() or np.any(np.abs(arr) > 1):
        raise ValueError(f'Nonfinite or out-of-bounds TV changes in {column}')
    return arr


def describe(cohort, subset, arr, resample_index):
    rng = np.random.default_rng(20261107 + resample_index)
    replicates = rng.choice(arr, size=(10000, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(replicates, [0.025, 0.975])
    return {
        'cohort': cohort,
        'subset': subset,
        'n_candidate_units': len(arr),
        'mean_delta': float(arr.mean()),
        'median_delta': float(np.median(arr)),
        'fraction_negative': float((arr < 0).mean()),
        'mean_unit_bootstrap_lo': float(lo),
        'mean_unit_bootstrap_hi': float(hi),
    }


def main():
    ad = required_csv('p3_dryad_ad_signal_slots.csv',
                      ['slot_column', 'slot_row', 'primary_delta_180_minus_60'])
    ad = ad.sort_values(['slot_column', 'slot_row']).drop_duplicates('slot_column', keep='first')
    hc = required_csv('p3_dryad_hc_signal_slots.csv', ['slot', 'primary_delta_180_minus_60'])
    brain = required_csv('p3_brainlat_site_outlier_audit.csv',
                         ['cohort', 'site_from_metadata', 'subject_id', 'delta_180_minus_60'])
    ad_d = series_checked(ad, 'primary_delta_180_minus_60', 43, ['slot_column'])
    hc_d = series_checked(hc, 'primary_delta_180_minus_60', 96, ['slot'])
    series_checked(brain, 'delta_180_minus_60', 36, ['cohort', 'subject_id'])
    if not set(brain.cohort).issubset({'AD', 'HC'}):
        raise ValueError('Unrecognized BrainLat diagnosis label')
    if 'UNMATCHED' in set(brain.site_from_metadata):
        raise ValueError('Unmatched BrainLat site labels')

    rows = [
        describe('Dryad', 'AD | first per candidate column', ad_d, 1),
        describe('Dryad', 'HC | unique recording array', hc_d, 2),
    ]
    for i, (label, filt) in enumerate([
        ('AD | all sites', brain.cohort.eq('AD')),
        ('AD | Argentina', brain.cohort.eq('AD') & brain.site_from_metadata.eq('AR')),
        ('AD | Chile', brain.cohort.eq('AD') & brain.site_from_metadata.eq('CL')),
        ('HC | Argentina', brain.cohort.eq('HC') & brain.site_from_metadata.eq('AR')),
    ], start=3):
        values = brain.loc[filt, 'delta_180_minus_60'].to_numpy(float)
        expected = { 'AD | all sites': 28, 'AD | Argentina': 10,
                    'AD | Chile': 18, 'HC | Argentina': 8 }[label]
        if len(values) != expected:
            raise ValueError(f'Unexpected BrainLat {label} count: {len(values)}')
        rows.append(describe('BrainLat', label, values, i))

    output = pd.DataFrame(rows)
    AUDIT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULTS / 'p3_multicohort_exploratory_snapshot.csv', index=False, float_format='%.12g')

    age_path = AUDIT / 'p3_dryad_age_sensitivity.json'
    if age_path.is_file():
        age_report = json.loads(age_path.read_text())
        age_contrast = age_report.get('nearest_age_matching_without_replacement', {}).get('AD_minus_HC_mean_effect')
    else:
        age_contrast = None

    report = {
        'protocol': 'Fixed-end 180 s minus 60 s normalized spectral TV, disjoint 240-300 s reference',
        'status': 'DESCRIPTIVE FEASIBILITY, NOT A POOLED CAUSAL OR DIAGNOSTIC CLAIM',
        'table': rows,
        'dryad_age_matched_AD_minus_HC_difference': age_contrast,
        'limitations': [
            'Dryad AD MATLAB columns and HC recording slots are not confirmed independent patients.',
            'BrainLat AD uses rs-HEP and HC uses rs; file variants confound group comparison.',
            'All eligible BrainLat HC files are Argentina-site; no Chile HC comparator.',
            'BrainLat HC missing FDT creates severe complete-case selection.',
            'BrainLat EEG was already offline-preprocessed, and sensors are spatially approximated.',
            'Bootstrap interval is conditional on candidate units; not a validated patient-level CI.',
            'BrainLat all-sites AD overlaps its site rows; do not treat rows as independent cohorts.',
            'AHEPA not recomputed here; verify metric and per-diagnosis source before synthesis.',
            'No between-dataset pooling, p-values, or significance assertions.'
        ],
    }
    (AUDIT / 'p3_multicohort_exploratory_snapshot.json').write_text(json.dumps(report, indent=2) + '\n')
    print('===== FROZEN CROSS-DATASET EXPLORATORY SNAPSHOT =====')
    print(output.to_string(index=False, formatters={c:lambda x:f'{x:+.6f}' for c in ('mean_delta','median_delta','mean_unit_bootstrap_lo','mean_unit_bootstrap_hi')}))
    print('\nDryad matched AD-HC contrast (from existing age sensitivity):', age_contrast)
    print('Intervals are candidate-unit bootstraps ONLY; not validated patient-level confidence intervals.')
    print('No pooled effect. BrainLat AD all-sites row overlaps its site-specific rows.')
    print('\n===== AHEPA ANCHOR RESULT FILE CANDIDATES =====')
    files = [p for p in RESULTS.rglob('*.csv') if any(term in p.name.lower() for term in ('anchor','temporal','disjoint','reliability'))]
    if not files:
        print('No obvious AHEPA anchor CSV found by filename; existing AHEPA results unchanged.')
    for p in sorted(files)[:15]:
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
            print(str(p.relative_to(ROOT)), '| fields:', ', '.join(cols[:18]))
        except Exception as exc:
            print(str(p.relative_to(ROOT)), '| header could not be read:', type(exc).__name__)
    print('\nNo EEG arrays were loaded or modified. No figures created.')
    print('===== SNAPSHOT COMPLETE =====')

if __name__ == '__main__':
    main()
