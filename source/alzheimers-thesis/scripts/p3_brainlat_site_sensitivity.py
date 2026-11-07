#!/usr/bin/env python3
"""Read-only site and outlier sensitivity for existing BrainLat reliability CSV.
Site lookups transcribed from BrainLat Records_AD_EEG_data.csv and Records_HC_EEG_data.csv.
No raw EEG data are read or changed.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path.home() / 'alzheimers-thesis'
SOURCE = ROOT / 'results/p3_brainlat_300s_reliability.csv'
REPORT = ROOT / 'docs/audit/p3_brainlat_site_outlier_summary.json'
OUT = ROOT / 'results/p3_brainlat_site_outlier_audit.csv'

SITE_IDS = {
    ('AD', 'AR'): '''sub-30001 sub-30002 sub-30004 sub-30008 sub-30009 sub-30011 sub-30012 sub-30013 sub-30015 sub-30017 sub-30018 sub-30020 sub-30022 sub-30026 sub-30029 sub-30031''',
    ('AD', 'CL'): '''sub-30003 sub-30006 sub-30010 sub-30014 sub-30016 sub-30021 sub-30023 sub-30019 sub-30024 sub-30025 sub-30027 sub-30028 sub-30033 sub-30035 sub-30034 sub-30005 sub-30007 sub-30030 sub-30032''',
    ('HC', 'AR'): '''sub-10002 sub-10003 sub-10004 sub-10006 sub-10007 sub-10009 sub-100012 sub-100015 sub-100018 sub-100020 sub-100024 sub-100026 sub-100028 sub-100030 sub-100031 sub-100033 sub-100035 sub-100038 sub-100022''',
    ('HC', 'CL'): '''sub-100010 sub-10008 sub-100021 sub-100011 sub-100013 sub-100016 sub-100019 sub-100042 sub-100040 sub-100041 sub-100043 sub-100044 sub-100046 sub-100045 sub-100032 sub-10001 sub-10005 sub-100017 sub-100023 sub-100025 sub-100027 sub-100036 sub-100037 sub-100039 sub-100014 sub-100029 sub-100034''',
}

site_lookup = {}
for (group, site), ids in SITE_IDS.items():
    for subject_id in ids.split():
        key = (group, subject_id)
        if key in site_lookup:
            raise ValueError(f'Duplicate site metadata assignment for {key}')
        site_lookup[key] = site


def describe(part):
    arr = part['delta_180_minus_60'].to_numpy(float)
    n = len(arr)
    if not n:
        return {'n': 0}
    mean_loo = [(arr.sum() - arr[i]) / (n - 1) for i in range(n)] if n > 1 else []
    return {
        'n': n,
        'mean_delta': float(arr.mean()),
        'median_delta': float(np.median(arr)),
        'fraction_negative': float(np.mean(arr < 0)),
        'negative_count': int(np.sum(arr < 0)),
        'mean_leave_one_out_min': float(min(mean_loo)) if mean_loo else None,
        'mean_leave_one_out_max': float(max(mean_loo)) if mean_loo else None,
        'max_abs_delta': float(np.max(np.abs(arr))),
        'median_channel_rms_uv': float(part['median_raw_channel_rms_uv'].median()),
    }


def main():
    print('===== BRAINLAT METADATA-LINKED SITE SENSITIVITY =====')
    if not SOURCE.is_file():
        print('ERROR: Existing result CSV not found:', SOURCE)
        return
    df = pd.read_csv(SOURCE)
    required = {'cohort', 'recording_code', 'delta_180_minus_60', 'median_raw_channel_rms_uv'}
    if not required.issubset(df.columns):
        raise ValueError('Missing columns: ' + repr(sorted(required - set(df.columns))))

    df['subject_id'] = df['recording_code'].astype(str).str.extract(r'(sub-\d+)', expand=False)
    if df['subject_id'].isna().any():
        raise ValueError('Some recording codes lack valid subject IDs')
    if df.duplicated(['cohort', 'subject_id']).any():
        raise ValueError('Multiple recordings from one candidate participant; re-group before inference')
    df['site_from_metadata'] = [
        site_lookup.get((group, sid), 'UNMATCHED')
        for group, sid in zip(df['cohort'], df['subject_id'])
    ]
    df['variant'] = np.where(df.recording_code.str.contains('rs-HEP', case=False, regex=False),
                             'rs-HEP',
                             np.where(df.recording_code.str.contains('_rs_', case=False, regex=False), 'rs', 'other'))
    for col in ('delta_180_minus_60', 'median_raw_channel_rms_uv'):
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite values in {col}')

    print('Previously analyzed recordings:', len(df))
    print('Site labels unmatched:', int((df.site_from_metadata == 'UNMATCHED').sum()))
    print('Unique recording IDs:', df.subject_id.nunique())

    print('\n===== BY COHORT AND SITE =====')
    groups = {}
    for (group, site), part in df.groupby(['cohort', 'site_from_metadata'], sort=True):
        stats = describe(part)
        groups[group + '/' + site] = stats
        print(group + '/' + site + ':', stats)

    print('\n===== ACQUISITION FILE VARIANTS =====')
    print(pd.crosstab(df['cohort'], df['variant']).to_string())
    print('\n===== COHORT DISTRIBUTIONS =====')
    cohorts = {}
    for group, part in df.groupby('cohort', sort=True):
        stats = describe(part)
        cohorts[group] = stats
        print(group + ':', stats)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'EXPLORATORY - metadata site sensitivity, no new EEG processing',
        'metadata_source': 'BrainLat Records_AD_EEG_data.csv and Records_HC_EEG_data.csv',
        'n_recordings': len(df),
        'unmatched_site_records': int((df.site_from_metadata == 'UNMATCHED').sum()),
        'by_site': groups,
        'by_cohort': cohorts,
        'limitations': [
            'BrainLat AD and HC are completely separated by rs-HEP versus rs filename variants.',
            '36 downloaded HC recordings were unreadable because of missing external FDT files.',
            'BrainLat EEG was offline-preprocessed; no causal online validation.',
            'Site labels are sourced from metadata records, not inferred from signal shape.',
            'Leave-one-out ranges are descriptive sensitivity measures, not confidence intervals.',
        ],
    }
    df.to_csv(OUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('\nSaved:', REPORT)
    print('No raw EEG read or changed. No hypothesis significance tested.')
    print('===== BRAINLAT SITE SENSITIVITY COMPLETE =====')

if __name__ == '__main__':
    main()
