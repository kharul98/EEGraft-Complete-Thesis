#!/usr/bin/env python3
"""M6D: freeze Figshare feasibility decision and verified thesis evidence."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path.cwd()
BASE = ROOT / 'reports' / 'thesis_final'

M6A = BASE / 'milestone_6a_figshare_archive_preflight_v1'
M6B = BASE / 'milestone_6b_figshare_mat_structure_v1'
M6C = BASE / 'milestone_6c_figshare_60s_compatibility_v2'
M5G = BASE / 'milestone_5g_isfahan_nested_reliability_v1'
M5I = BASE / 'milestone_5i_isfahan_negative_controls_v1'

OUT = BASE / 'milestone_6d_figshare_feasibility_decision_v1'

ARCHIVE = (
    ROOT / 'data/external/cejnek_figshare_v1/dataset.zip'
)

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    'm6a_metadata':
        M6A / 'source_and_status.json',

    'm6b_inventory':
        M6B / 'mat_variable_and_subject_folder_audit.csv',

    'm6c_file_durations':
        M6C / 'all_mat_nominal_durations_with_shape_flags.csv',

    'm6c_subject_proxies':
        M6C / 'recording_proxy_duration_feasibility.csv',

    'm5g_summary':
        M5G / 'summary_conditional_subject_bootstrap.csv',

    'm5i_negative_controls':
        M5I / 'constant_label_comparison.csv',
}


def sha256(path):
    h = hashlib.sha256()

    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(2 * 1024 * 1024),
            b''
        ):
            h.update(chunk)

    return h.hexdigest()


def csv_rows(path):
    with path.open(
        encoding='utf-8-sig',
        newline=''
    ) as f:
        return list(csv.DictReader(f))


def freeze(path, content):
    if path.is_file():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError(
                'Frozen output differs: ' + str(path)
            )
    else:
        path.write_text(
            content,
            encoding='utf-8'
        )


# ==========================================
# 1. VERIFY EXISTING AUDITS
# ==========================================

print(
    '===== MILESTONE 6D: '
    'FIGSHARE FEASIBILITY CLOSEOUT =====',
    flush=True
)

for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(
            f'{name}: {path}'
        )

if not ARCHIVE.is_file():
    raise FileNotFoundError(ARCHIVE)

m6a = json.loads(
    FILES['m6a_metadata'].read_text(
        encoding='utf-8'
    )
)

if sha256(ARCHIVE) != m6a['archive_sha256']:
    raise RuntimeError(
        'Figshare ZIP checksum differs from M6A'
    )

original = csv_rows(FILES['m6b_inventory'])
durations = csv_rows(FILES['m6c_file_durations'])
proxies = csv_rows(FILES['m6c_subject_proxies'])
controls = csv_rows(FILES['m5i_negative_controls'])
gsummary = csv_rows(FILES['m5g_summary'])

if (
    len(original) != 471
    or len(durations) != 471
    or len(proxies) != 166
):
    raise RuntimeError(
        'M6A/B/C inventory totals changed'
    )

if len(controls) != 2 or len(gsummary) != 2:
    raise RuntimeError(
        'M5G/M5I summary totals changed'
    )

original_members = {
    r['archive_member']
    for r in original
}

duration_members = {
    r['archive_member']
    for r in durations
}

if original_members != duration_members:
    raise RuntimeError(
        'M6B and M6C inventories disagree'
    )

if len(duration_members) != 471:
    raise RuntimeError(
        'Duplicate MATLAB archive paths'
    )


# ==========================================
# 2. VERIFY GROUP COUNTS
# ==========================================

by_group = Counter(
    r['diagnosis']
    for r in durations
)

by_proxy = Counter(
    r['diagnosis']
    for r in proxies
)

if by_group != {
    'AD': 305,
    'MCI': 56,
    'CONTROL': 110
}:
    raise RuntimeError(
        'Unexpected Figshare file distribution'
    )

if by_proxy != {
    'AD': 57,
    'MCI': 7,
    'CONTROL': 102
}:
    raise RuntimeError(
        'Unexpected recording-proxy distribution'
    )


def standard60(group):
    return sum(
        r['diagnosis'] == group
        and int(
            r['has_single_STANDARD_19col_file_ge_60s']
        ) == 1
        for r in proxies
    )


ad_standard = standard60('AD')
hc_standard = standard60('CONTROL')

segmenty = [
    r for r in durations
    if (
        r['diagnosis'] == 'CONTROL'
        and r['variable'] == 'segmenty'
    )
]

segmenty_60 = sum(
    r['nominal_duration_seconds'] != ''
    and float(
        r['nominal_duration_seconds']
    ) >= 60
    for r in segmenty
)

nonstandard = sum(
    r['variable'] == 'export'
    and r['duration_status']
        != 'STANDARD_EXPORT_19_COLUMNS'
    for r in durations
)

if (
    ad_standard,
    hc_standard,
    len(segmenty),
    segmenty_60,
    nonstandard
) != (44, 1, 100, 82, 10):
    raise RuntimeError(
        'M6C-v2 findings changed; re-review needed'
    )


# ==========================================
# 3. VERIFY MCI NEGATIVE-CONTROL RESULTS
# ==========================================

mci_by_release = {
    r['release']: r
    for r in controls
}

g_by_release = {
    r['release']: r
    for r in gsummary
}

if (
    set(mci_by_release) != {'EEG_1', 'EEG_2'}
    or set(g_by_release) != {'EEG_1', 'EEG_2'}
):
    raise RuntimeError(
        'MCI release summary incomplete'
    )

for release in ('EEG_1', 'EEG_2'):
    a = mci_by_release[release]
    b = g_by_release[release]

    if abs(
        float(a['model_first_minute_BACC'])
        - float(b['minute1_BACC'])
    ) > 1e-10:
        raise RuntimeError(
            'M5G/M5I BACC disagreement: ' + release
        )

    if abs(
        float(a['model_stable_wrong'])
        - float(b['stable_wrong_rate'])
    ) > 1e-10:
        raise RuntimeError(
            'M5G/M5I state disagreement: ' + release
        )


# ==========================================
# 4. PRINT VERIFIED FINDINGS
# ==========================================

print(
    'PASS: Figshare archive checksum and '
    '471-file inventory unchanged.'
)

print(
    'Figshare recording proxies: '
    'AD=57, MCI=7, CONTROL=102.'
)

print(
    'Standard 19-column, nominal >=60s: '
    f'AD={ad_standard}; CONTROL={hc_standard}.'
)

print(
    'CONTROL segmenty (22-row format): '
    f'{len(segmenty)} files; '
    f'nominal >=60s={segmenty_60}.'
)

print(
    'Nonstandard export matrix widths: '
    f'{nonstandard} files.'
)

print(
    'No common electrode-map verification '
    'or source-compatible control cohort.'
)


# ==========================================
# 5. FREEZE FORMAL DECISION
# ==========================================

decision = {
    'milestone': '6D',

    'decision':
        'NO_GO_FOR_ORIGINAL_19_CHANNEL_60S_EXTERNAL_AD_VALIDATION',

    'reason':
        'Only one CONTROL proxy has a standard '
        '19-column nominal >=60s file. Most '
        'controls use an unmapped 22-row '
        'segmenty layout.',

    'figshare_mat_files': 471,

    'figshare_recording_proxies':
        dict(by_proxy),

    'standard_19_column_nominal_60s_AD_proxies':
        ad_standard,

    'standard_19_column_nominal_60s_CONTROL_proxies':
        hc_standard,

    'control_segmenty_files':
        len(segmenty),

    'control_segmenty_nominal_60s_files_at_tentative_128Hz':
        segmenty_60,

    'nonstandard_export_files':
        nonstandard,

    'channel_identity_verified': False,
    'control_segmenty_sampling_rate_per_file_verified': False,
    'fragment_continuity_verified': False,
    'external_validation_performed': False,
    'classifier_trained': False,
    'previous_results_modified': False,

    'source_sha256': {
        name: sha256(path)
        for name, path in FILES.items()
    },

    'archive_sha256': sha256(ARCHIVE),
}

freeze(
    OUT / 'locked_decision.json',
    json.dumps(
        decision,
        sort_keys=True,
        indent=2
    ) + '\n'
)


# ==========================================
# 6. CREATE THESIS-READY EVIDENCE SUMMARY
# ==========================================

lines = [
    '# Figshare External Alzheimer Validation: Feasibility Decision',
    '',

    '**Decision: Do not train or report external '
    'AD validation on this archive under the '
    'original 19-channel, 60-second protocol.**',

    '',

    'The audited archive contains 471 MATLAB files: '
    '305 AD, 56 MCI, and 110 control recordings. '
    'These are not participant counts. '
    'Folder/file proxies total 57 AD, '
    '7 MCI, and 102 control.',

    '',

    'A standard 19-column recording with nominal '
    'duration of at least 60 seconds occurs in '
    '44 AD folders but only 1 control folder.',

    '',

    'The remaining 100 root-level control files '
    'use segmenty matrices with 22 rows. '
    'Of these, 82 have nominal duration >=60 seconds '
    'under an unconfirmed 128 Hz assumption.',

    '',

    'Ten export files have nonstandard column counts.',

    '',

    'A shared electrode-to-column mapping, physical '
    'units, and chronological continuity across '
    'fragments have not been verified.',

    '',

    '**Interpretation:** This is a dataset '
    'compatibility limitation, not a negative '
    'test of the existing Alzheimer classifier. '
    'Figshare AD-versus-control balanced '
    'accuracy has not been measured.',

    '',

    '## Completed MCI Extension: Different Clinical Task',
    '',

    '| Release | MCI / Healthy | BACC | Stable-wrong | '
    'Unstable | Constant-label BACC |',

    '|---|---:|---:|---:|---:|---:|',
]

for release in ('EEG_1', 'EEG_2'):
    r = mci_by_release[release]

    lines.append(
        f"| {release} | "
        f"{r['n_MCI']}/{r['n_NORMAL']} | "
        f"{float(r['model_first_minute_BACC']):.4f} | "
        f"{float(r['model_stable_wrong']):.4f} | "
        f"{float(r['model_unstable']):.4f} | "
        f"{float(r['always_MCI_BACC']):.4f} |"
    )

lines.extend([
    '',

    'The MCI results use five repeated, '
    'correlated subject-held-out assessments. '
    'They do not represent independent '
    'observations of 245 individuals.',

    '',

    'EEG_1 was biased toward predicting MCI. '
    'The negative control demonstrated that '
    'temporal stability does not necessarily '
    'indicate diagnostic correctness.',

    '',

    '## Thesis Integration',
    '',

    'Keep original AD/FTD classification, temporal '
    'stability, calibration, and ablation analyses '
    'as the main research results.',

    '',

    'Report Isfahan MCI as a separate methodological '
    'extension and Figshare as an audited dataset '
    'that did not meet the intended external '
    'validation requirements.',

    '',

    'Next: consolidate the original AD/FTD evidence, '
    'prepare final tables, complete the Methods, '
    'Results, Discussion, and Conclusion chapters, '
    'and prepare for supervisor review.',

    '',
])

freeze(
    OUT / 'FIGSHARE_DECISION_AND_THESIS_EVIDENCE.md',
    '\n'.join(lines)
)


# ==========================================
# 7. FINISH
# ==========================================

print(
    '\n===== MILESTONE 6D COMPLETE ====='
)

print(
    'Decision: NO-GO for original 19-channel '
    'Figshare external AD validation.'
)

for release in ('EEG_1', 'EEG_2'):
    r = mci_by_release[release]

    print(
        f'{release} negative-control verified: '
        f"BACC={float(r['model_first_minute_BACC']):.4f}; "
        f"stable-wrong={float(r['model_stable_wrong']):.4f}; "
        f"unstable={float(r['model_unstable']):.4f}"
    )

print(
    'Frozen outputs:', OUT
)

print(
    'Next phase: final thesis results '
    'consolidation and chapter writing.'
)

print(
    'No EEG modified, no classifier trained, '
    'no earlier milestone overwritten.'
)

