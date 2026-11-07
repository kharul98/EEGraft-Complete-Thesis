#!/usr/bin/env python3
"""Milestone 4C: audit and package completed short-EEG thesis results."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import zipfile

import numpy as np
import scipy
import sklearn
import mne


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

OUT = BASE / 'milestone_4c_reproducibility_v1'
OUT.mkdir(parents=True, exist_ok=True)


INPUTS = {
    '1a_states':
        BASE / 'milestone_1a_reliability_v1'
        / 'subject_repeat_states.csv',

    '1b_cal':
        BASE / 'milestone_1b_calibration_v1'
        / 'calibration_metrics_by_repeat_minute.csv',

    '1c_auc':
        BASE / 'milestone_1c_selective_prediction_v1'
        / 'error_detection_auroc.csv',

    '1c_coverage':
        BASE / 'milestone_1c_selective_prediction_v1'
        / 'risk_coverage_summary.csv',

    '2a_clinical':
        BASE / 'milestone_2a_clinical_metadata_v1'
        / 'clinical_eligibility_summary.csv',

    '2b_mmse':
        BASE / 'milestone_2b_mmse_associations_v1'
        / 'mmse_association_results.csv',

    '3c_eligibility':
        BASE / 'milestone_3c_boundary_feasibility_v1'
        / 'duration_continuity_and_30s_qc.csv',

    '3d_bootstrap':
        BASE / 'milestone_3d_30s_condition_transfer_v1'
        / 'paired_condition_subject_bootstrap_intervals.csv',

    '3d_repeats':
        BASE / 'milestone_3d_30s_condition_transfer_v1'
        / 'paired_condition_cv_repeat_metrics.csv',

    '3e_transitions':
        BASE / 'milestone_3e_condition_shift_audit_v1'
        / 'paired_prediction_transitions_by_group.csv',

    '3e_spectral':
        BASE / 'milestone_3e_condition_shift_audit_v1'
        / 'paired_band_shifts_by_group.csv',

    '4a_states':
        BASE / 'milestone_4a_temporal_explanation_v1'
        / 'stability_state_explanation_summary.csv',

    '4a_drift':
        BASE / 'milestone_4a_temporal_explanation_v1'
        / 'state_explanation_drift_bootstrap_contrasts.csv',

    '4a_groups':
        BASE / 'milestone_4a_temporal_explanation_v1'
        / 'average_group_score_sensitivity.csv',

    '4b_ablation':
        BASE / 'milestone_4b_feature_ablation_v1'
        / 'ablation_paired_first_minute_comparisons.csv',

    '4b_minute':
        BASE / 'milestone_4b_feature_ablation_v1'
        / 'ablation_BACC_by_minute.csv',
}

TASKS = {
    'AD_vs_HC': 46,
    'FTD_vs_HC': 41
}

GROUPS = {
    'AD_vs_HC': {
        'A': 23,
        'C': 23
    },
    'FTD_vs_HC': {
        'F': 18,
        'C': 23
    }
}


# ==========================================
# 1. FILE AND NUMERIC UTILITIES
# ==========================================

def read_csv(path):

    if not path.is_file():
        raise FileNotFoundError(
            'Required milestone CSV missing: '
            + str(path)
        )

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:

        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            'Empty results table: ' + str(path)
        )

    return rows


def table_write(path, rows):

    if not rows:
        raise RuntimeError(
            'Cannot save empty table: ' + str(path)
        )

    columns = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with path.open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def one(rows, desc, **filters):

    matches = [
        r for r in rows
        if all(
            str(r.get(k)) == str(v)
            for k, v in filters.items()
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one {desc} row for '
            f'{filters}, got {len(matches)}'
        )

    return matches[0]


def num(row, column):

    try:
        value = float(row[column])

    except (
        KeyError,
        ValueError,
        TypeError
    ) as exc:

        raise RuntimeError(
            f'Bad numeric column {column}'
        ) from exc

    if not np.isfinite(value):
        raise RuntimeError(
            f'Nonfinite numeric column {column}'
        )

    return value


def optional_num(row, column):

    if row.get(column, '') in ('', None):
        return ''

    return num(row, column)


def average(rows, column):

    return float(np.mean([
        num(r, column)
        for r in rows
    ]))


def hashfile(path):

    h = hashlib.sha256()

    with path.open('rb') as f:

        for chunk in iter(
            lambda: f.read(2**20),
            b''
        ):
            h.update(chunk)

    return h.hexdigest()


def pc(x):
    return f'{100*x:.2f}%'


def signed_pp(x):
    return f'{100*x:+.2f} percentage points'


def signed_interval(lo, hi):

    return (
        f'[{100*lo:+.2f}, {100*hi:+.2f}] '
        'percentage points'
    )


def relative(path):
    return str(path.relative_to(ROOT))


# ==========================================
# 2. LOAD ALL COMPLETED EXPERIMENT TABLES
# ==========================================

print(
    '===== MILESTONE 4C: FINAL EVIDENCE AUDIT =====',
    flush=True
)

tables = {
    name: read_csv(path)
    for name, path in INPUTS.items()
}

print(
    f'PASS: {len(INPUTS)} essential '
    'milestone tables loaded.',
    flush=True
)

metrics = []


def add(
    task,
    milestone,
    metric,
    estimate,
    source,
    low='',
    high='',
    note='',
    unit='fraction',
    comparator='',
    p='',
    adjusted=''
):

    metrics.append({
        'task': task,
        'milestone': milestone,
        'metric': metric,
        'estimate': estimate,
        'unit': unit,
        'ci_low': low,
        'ci_high': high,
        'comparator': comparator,
        'uncorrected_p': p,
        'adjusted_p': adjusted,
        'interpretation_caveat': note,
        'source_csv': relative(INPUTS[source])
    })


# ==========================================
# 3. MILESTONE 1A: STABILITY STATES
# ==========================================

states = tables['1a_states']
state_rates = {}

for task, n in TASKS.items():

    rows = [
        r for r in states
        if r['task'] == task
    ]

    keys = {
        (
            r['subject'],
            int(r['repeat'])
        )
        for r in rows
    }

    if (
        len(rows) != 5*n
        or len(keys) != 5*n
    ):
        raise RuntimeError(
            '1A participant-repeat mismatch: '
            + task
        )

    counts = Counter(
        r['state']
        for r in rows
    )

    expected_states = {
        'stable_correct',
        'stable_wrong',
        'unstable'
    }

    if set(counts) != expected_states:
        raise RuntimeError(
            'Unexpected stability states: ' + task
        )

    state_rates[task] = {
        state: counts[state] / len(rows)
        for state in counts
    }

    for state in sorted(counts):

        add(
            task,
            '1A',
            state + '_fraction',
            state_rates[task][state],
            '1a_states',
            note='Five correlated CV repetitions.'
        )

    for group, expected_n in GROUPS[task].items():

        actual = sum(
            r['group'] == group
            for r in rows
        )

        if actual != 5*expected_n:
            raise RuntimeError(
                '1A group count mismatch: '
                + task + '/' + group
            )


# ==========================================
# 4. MILESTONE 1B: CALIBRATION
# ==========================================

calibration = {}

calibration_fields = (
    'original_svm_bacc',
    'calibrated_threshold_bacc',
    'brier',
    'training_prior_brier',
    'nll',
    'ece_five_equal_width_bins',
    'auc'
)

for task in TASKS:

    first = [
        r for r in tables['1b_cal']
        if (
            r['task'] == task
            and int(r['minute']) == 1
        )
    ]

    if len(first) != 5:
        raise RuntimeError(
            'Missing calibration repeats: ' + task
        )

    calibration[task] = {
        field: average(first, field)
        for field in calibration_fields
    }

    for field, value in calibration[task].items():

        add(
            task,
            '1B',
            'first_minute_' + field,
            value,
            '1b_cal',
            note='Five correlated CV repeats; '
                 'calibration fitted within training folds.'
        )


# ==========================================
# 5. MILESTONE 1C: ERROR DETECTION
# ==========================================

uncertainty = {}

for task in TASKS:

    error = one(
        tables['1c_auc'],
        'all-errors AUROC',
        task=task,
        method='first_minute_original_label_support',
        uncertainty_target='all_first_minute_errors'
    )

    stable = one(
        tables['1c_auc'],
        'stable-only AUROC',
        task=task,
        method='first_minute_original_label_support',
        uncertainty_target='stable_wrong_among_unanimous'
    )

    baseline = one(
        tables['1c_coverage'],
        '100% coverage',
        task=task,
        method='first_minute_original_label_support',
        target_coverage='1.0'
    )

    if abs(
        num(baseline, 'mean_selected_BACC')
        - calibration[task]['original_svm_bacc']
    ) > 1e-9:

        raise RuntimeError(
            '1B versus 1C BACC mismatch: ' + task
        )

    uncertainty[task] = {
        'all_auc':
            num(error, 'auroc_higher_is_better'),

        'stable_auc':
            num(stable, 'auroc_higher_is_better')
    }

    for label, row in (
        ('all_error_detection_AUROC', error),
        ('stable_wrong_detection_AUROC', stable)
    ):

        add(
            task,
            '1C',
            label,
            num(row, 'auroc_higher_is_better'),
            '1c_auc',
            low=num(row, 'conditional_ci_low'),
            high=num(row, 'conditional_ci_high'),
            note='Retrospective error ranking.'
        )

    for method, coverage, name in (
        (
            'first_minute_original_label_support',
            '0.8',
            'one_minute_confidence_80pct'
        ),
        (
            'five_minute_label_consensus',
            '0.7',
            'five_minute_consensus_70pct'
        )
    ):

        row = one(
            tables['1c_coverage'],
            'selective performance',
            task=task,
            method=method,
            target_coverage=coverage
        )

        add(
            task,
            '1C',
            name + '_selected_BACC',
            num(row, 'mean_selected_BACC'),
            '1c_coverage',
            note='Retrospective ranking, '
                 'not a deployable threshold.'
        )

        add(
            task,
            '1C',
            name + '_BACC_gain',
            num(row, 'BACC_gain_over_full_coverage'),
            '1c_coverage',
            low=num(row, 'conditional_gain_ci_low'),
            high=num(row, 'conditional_gain_ci_high'),
            comparator='100% retained',
            note='Improvement only among retained participants.'
        )


# ==========================================
# 6. MILESTONE 2: CLINICAL SEVERITY
# ==========================================

clinical = {}

for group, expected_n in (
    ('A', 23),
    ('F', 18),
    ('C', 23)
):

    row = one(
        tables['2a_clinical'],
        'clinical availability',
        group=group,
        subset='SELECTED'
    )

    if (
        int(row['n']) != expected_n
        or int(row['mmse_available']) != expected_n
        or int(row['age_available']) != expected_n
    ):

        raise RuntimeError(
            'Clinical metadata completeness changed: '
            + group
        )

    clinical[group] = row


mmse = {}

for task, n in (
    ('AD_vs_HC', 23),
    ('FTD_vs_HC', 18)
):

    rows = [
        r for r in tables['2b_mmse']
        if r['task'] == task
    ]

    if (
        len(rows) != 5
        or any(
            int(r['n_patients']) != n
            for r in rows
        )
    ):
        raise RuntimeError(
            'MMSE tests missing: ' + task
        )

    for row in rows:

        label = row['metric']
        mmse[(task, label)] = row

        adjusted = (
            row['Holm_adjusted_p_four_primary_tests']
            if row['priority'] == 'primary'
            else row['BH_FDR_q_across_10_raw_tests']
        )

        add(
            task,
            '2B',
            'MMSE_spearman_' + label,
            num(row, 'spearman_rho'),
            '2b_mmse',
            low=optional_num(
                row,
                'spearman_bootstrap_ci_low'
            ),
            high=optional_num(
                row,
                'spearman_bootstrap_ci_high'
            ),
            p=num(
                row,
                'permutation_p_two_sided'
            ),
            adjusted=float(adjusted),
            note=(
                'Four-test Holm-corrected primary analysis.'
                if row['priority'] == 'primary'
                else 'Exploratory secondary; BH across ten tests.'
            )
        )


# ==========================================
# 7. MILESTONE 3C: EO ELIGIBILITY
# ==========================================

qc = tables['3c_eligibility']

if len(qc) != 64:
    raise RuntimeError(
        '3C must contain 64 participants'
    )

for group, expected_n in (
    ('A', 19),
    ('F', 13),
    ('C', 22)
):

    rows = [
        r for r in qc
        if r['group'] == group
    ]

    eligible = sum(
        r['30s_status'] == 'ELIGIBLE_30S'
        for r in rows
    )

    if eligible != expected_n:
        raise RuntimeError(
            '30-second eligibility mismatch: '
            + group
        )

    add(
        group,
        '3C',
        'eligible_30s_count',
        eligible,
        '3c_eligibility',
        unit='participants',
        note=f'{len(rows)} participants initially selected.'
    )


# ==========================================
# 8. MILESTONES 3D/3E: TRANSFER
# ==========================================

condition = {}

condition_metrics = (
    'EC_BACC',
    'EO_BACC',
    'EO_minus_EC_BACC',
    'same_pred_fraction',
    'mean_EO_minus_EC_raw_margin'
)

for task, expected_n in (
    ('AD_vs_HC', 41),
    ('FTD_vs_HC', 35)
):

    condition[task] = {}

    for name in condition_metrics:

        row = one(
            tables['3d_bootstrap'],
            'paired condition metric',
            task=task,
            metric=name
        )

        if int(row['n_matched']) != expected_n:
            raise RuntimeError(
                'Matched participant count changed: '
                + task
            )

        value = num(row, 'estimate')
        condition[task][name] = value

        add(
            task,
            '3D',
            name,
            value,
            '3d_bootstrap',
            low=num(
                row,
                'conditional_95CI_low'
            ),
            high=num(
                row,
                'conditional_95CI_high'
            ),
            note='Same participants; condition transfer, '
                 'not independent validation.'
        )

    agreement = one(
        tables['3e_transitions'],
        '3E label agreement',
        task=task,
        group='ALL_MATCHED',
        metric='label_agreement'
    )

    if abs(
        num(agreement, 'rate')
        - condition[task]['same_pred_fraction']
    ) > 1e-10:

        raise RuntimeError(
            '3D versus 3E agreement mismatch: '
            + task
        )

    repeat_rows = [
        r for r in tables['3d_repeats']
        if r['task'] == task
    ]

    if len(repeat_rows) != 5:
        raise RuntimeError(
            'Expected five condition-transfer CV repeats'
        )

    full_ec = average(
        repeat_rows,
        'full_EC_BACC'
    )

    condition[task]['full_EC_BACC'] = full_ec

    add(
        task,
        '3D',
        'full_EC_BACC_30s',
        full_ec,
        '3d_repeats',
        note='Full original EC cohort; '
             'not the matched EO-eligible subset.'
    )


# ==========================================
# 9. MILESTONE 4A: EXPLANATION STABILITY
# ==========================================

explanation = {}

for task, n in TASKS.items():

    rows = [
        r for r in tables['4a_states']
        if r['task'] == task
    ]

    if (
        len(rows) != 3
        or sum(
            int(r['n_correlated_subject_repeats'])
            for r in rows
        ) != 5*n
    ):

        raise RuntimeError(
            '4A stability-state counts missing: '
            + task
        )

    explanation[task] = {}

    for row in rows:

        state = row['state']

        count = int(
            row['n_correlated_subject_repeats']
        )

        if abs(
            count/(5*n)
            - state_rates[task][state]
        ) > 1e-10:

            raise RuntimeError(
                '4A versus 1A state mismatch: '
                + task
            )

        value = num(
            row,
            'mean_explanation_drift'
        )

        explanation[task][state] = value

        add(
            task,
            '4A',
            'explanation_drift_' + state,
            value,
            '4a_states',
            note='Nonadditive local SVM-margin perturbation.'
        )

    contrast = one(
        tables['4a_drift'],
        'explanation drift contrast',
        task=task,
        comparison='unstable_minus_stable_correct'
    )

    add(
        task,
        '4A',
        'unstable_minus_stable_correct_drift',
        num(contrast, 'difference'),
        '4a_drift',
        low=num(
            contrast,
            'conditional_95CI_low'
        ),
        high=num(
            contrast,
            'conditional_95CI_high'
        ),
        note='Conditional exploratory participant bootstrap.'
    )

    for group in (
        'delta',
        'theta_alpha_family',
        'beta',
        'gamma'
    ):

        row = one(
            tables['4a_groups'],
            'local feature-group effect',
            task=task,
            group=group
        )

        add(
            task,
            '4A',
            'group_score_effect_' + group,
            num(
                row,
                'mean_absolute_effect'
            ),
            '4a_groups',
            note='Theta/alpha groups three related inputs; '
                 'other groups contain one.'
        )


# ==========================================
# 10. MILESTONE 4B: GROUP ABLATION
# ==========================================

ablation = {}

methods = (
    'without_delta',
    'without_theta_alpha_family',
    'without_beta',
    'without_gamma'
)

for task in TASKS:

    full = one(
        tables['4b_minute'],
        'full-model ablation baseline',
        task=task,
        method='full_6_features',
        minute='1'
    )

    expected_bacc = (
        calibration[task]['original_svm_bacc']
    )

    if abs(
        num(full, 'mean_repeated_CV_BACC')
        - expected_bacc
    ) > 1e-9:

        raise RuntimeError(
            '4B does not reproduce 1B BACC: '
            + task
        )

    for method in methods:

        row = one(
            tables['4b_ablation'],
            'feature ablation',
            task=task,
            ablation=method
        )

        if abs(
            num(row, 'full_first_minute_BACC')
            - expected_bacc
        ) > 1e-9:

            raise RuntimeError(
                'Ablation baseline mismatch: '
                + task
            )

        ablation[(task, method)] = row

        add(
            task,
            '4B',
            'ablation_BACC_difference_' + method,
            num(
                row,
                'ablated_minus_full_BACC'
            ),
            '4b_ablation',
            low=num(
                row,
                'conditional_CI_low'
            ),
            high=num(
                row,
                'conditional_CI_high'
            ),
            p=num(
                row,
                'paired_signflip_p_approx'
            ),
            adjusted=num(
                row,
                'Holm_p_across_eight_comparisons'
            ),
            comparator='full six-feature SVM',
            note='Eight Holm-adjusted exploratory contrasts.'
        )


if any(
    num(r, 'Holm_p_across_eight_comparisons')
    < 0.05
    for r in tables['4b_ablation']
):

    raise RuntimeError(
        'A corrected ablation passed 0.05; '
        'narrative requires revision'
    )


# ==========================================
# 11. WRITE CONSOLIDATED THESIS REPORT
# ==========================================

lines = []


def line(text=''):
    lines.append(text)


line(
    '# Short-EEG dementia classifier reliability: '
    'consolidated exploratory evidence'
)

line()

line(
    '**Objective:** Evaluate whether short-EEG dementia '
    'classifiers remain reliable across time and recording '
    'conditions, and examine uncertainty, clinical severity '
    'and spectral feature sensitivity.'
)

line()

line(
    'This is a methodological reliability study, '
    '**not** a prospective Alzheimer-risk predictor '
    'or independent clinical validation.'
)

line()

line('## 1. Study and validation design')

line()

line(
    '- Eyes-closed cohort: 23 AD + 23 healthy '
    '(AD task, n=46), and 18 FTD + the same '
    '23 healthy (FTD task, n=41).'
)

line(
    '- StandardScaler and class-weighted RBF SVM, '
    'five repeats of five outer participant-stratified '
    'folds, with nested training-only model selection.'
)

line(
    '- CV repetitions share participants and are correlated. '
    'Healthy controls also overlap across the two tasks.'
)

line()

line('## 2. Temporal prediction reliability')

line()

line('| Endpoint | AD vs HC | FTD vs HC |')
line('|---|---:|---:|')

line(
    '| Original first-minute BACC | '
    f"{pc(calibration['AD_vs_HC']['original_svm_bacc'])} | "
    f"{pc(calibration['FTD_vs_HC']['original_svm_bacc'])} |"
)

for state, label in (
    ('stable_correct', 'Stable and correct'),
    ('stable_wrong', 'Stable but wrong'),
    ('unstable', 'Predicted class changes')
):

    line(
        f'| {label} | '
        f"{pc(state_rates['AD_vs_HC'][state])} | "
        f"{pc(state_rates['FTD_vs_HC'][state])} |"
    )

line()

line(
    'Stable predictions can be wrong. The percentages '
    'represent correlated participant-repeat observations.'
)

line()

line('## 3. Calibrated uncertainty and abstention')

line()

line('| Measure | AD vs HC | FTD vs HC |')
line('|---|---:|---:|')

for field, label in (
    ('brier', 'First-minute Brier score'),
    (
        'training_prior_brier',
        'Training-prevalence Brier baseline'
    )
):

    line(
        f'| {label} | '
        f"{calibration['AD_vs_HC'][field]:.4f} | "
        f"{calibration['FTD_vs_HC'][field]:.4f} |"
    )

line(
    '| First-minute error-detection AUROC | '
    f"{uncertainty['AD_vs_HC']['all_auc']:.4f} | "
    f"{uncertainty['FTD_vs_HC']['all_auc']:.4f} |"
)

line(
    '| Stable-only error-detection AUROC | '
    f"{uncertainty['AD_vs_HC']['stable_auc']:.4f} | "
    f"{uncertainty['FTD_vs_HC']['stable_auc']:.4f} |"
)

line()

line(
    'Sigmoid probability calibration used only '
    'outer-training participants. Original SVM labels '
    'were preserved in selective prediction.'
)

line(
    'Selection rankings were retrospective and cohort-relative; '
    'they are not deployable abstention thresholds.'
)

line()

for task, label in (
    ('AD_vs_HC', 'AD'),
    ('FTD_vs_HC', 'FTD')
):

    for method, coverage, desc in (
        (
            'first_minute_original_label_support',
            '0.8',
            '60s confidence, 80% coverage'
        ),
        (
            'five_minute_label_consensus',
            '0.7',
            '300s consensus, 70% coverage'
        )
    ):

        row = one(
            tables['1c_coverage'],
            'selective result',
            task=task,
            method=method,
            target_coverage=coverage
        )

        line(
            f"- {label}, {desc}: BACC gain "
            f"{signed_pp(num(row, 'BACC_gain_over_full_coverage'))}; "
            'conditional 95% CI '
            f"{signed_interval(num(row, 'conditional_gain_ci_low'), num(row, 'conditional_gain_ci_high'))}."
        )

line()

line('## 4. MMSE clinical severity')

line()

line(
    f"- AD patient-only: n={clinical['A']['n']}, "
    f"median MMSE={clinical['A']['mmse_median']}."
)

line(
    f"- FTD patient-only: n={clinical['F']['n']}, "
    f"median MMSE={clinical['F']['mmse_median']}."
)

line(
    '- All selected patients had MMSE and age; '
    'healthy controls had constant MMSE=30.'
)

line(
    '- None of the four primary within-diagnosis '
    'MMSE tests survived Holm correction.'
)

secondary = mmse[
    ('AD_vs_HC', 'stable_wrong_rate')
]

line(
    '- Exploratory secondary AD MMSE versus '
    'stable-wrong rate: '
    f"rho={num(secondary, 'spearman_rho'):+.3f}, "
    f"permutation p={num(secondary, 'permutation_p_two_sided'):.4f}, "
    f"BH q={num(secondary, 'BH_FDR_q_across_10_raw_tests'):.4f}."
)

line(
    'This secondary association is hypothesis-generating, '
    'not proof of disease-severity prediction.'
)

line()

line('## 5. Cross-condition robustness')

line()

line(
    'Matched 30-second eyes-closed and '
    'eyes-open/photic EEG, from the same held-out people.'
)

line()

line('| Measure | AD vs HC | FTD vs HC |')
line('|---|---:|---:|')

line(
    '| Matched participants | '
    '19 AD + 22 HC | 13 FTD + 22 HC |'
)

for field, label in (
    ('EC_BACC', 'Eyes-closed BACC'),
    ('EO_BACC', 'Eyes-open/photic BACC'),
    ('same_pred_fraction', 'Cross-condition agreement')
):

    line(
        f'| {label} | '
        f"{pc(condition['AD_vs_HC'][field])} | "
        f"{pc(condition['FTD_vs_HC'][field])} |"
    )

line(
    '| EO minus EC BACC | '
    f"{signed_pp(condition['AD_vs_HC']['EO_minus_EC_BACC'])} | "
    f"{signed_pp(condition['FTD_vs_HC']['EO_minus_EC_BACC'])} |"
)

line()

for task, label in (
    ('AD_vs_HC', 'AD'),
    ('FTD_vs_HC', 'FTD')
):

    row = one(
        tables['3d_bootstrap'],
        'paired BACC difference',
        task=task,
        metric='EO_minus_EC_BACC'
    )

    line(
        f'- {label} paired BACC difference '
        'conditional 95% CI: '
        f"{signed_interval(num(row, 'conditional_95CI_low'), num(row, 'conditional_95CI_high'))}."
    )

line()

line(
    'Both paired BACC intervals include zero. '
    'Eyes-open state, photic stimulation, recording positions '
    'and preprocessing differences cannot be separated causally.'
)

line(
    'The 30-second rule recovered 14 participants relative '
    'to the original 60-second eligibility test; '
    '10 of 64 remained ineligible.'
)

line()

line('## 6. Temporal model explanation stability')

line()

line('| Mean explanation drift | AD vs HC | FTD vs HC |')
line('|---|---:|---:|')

for state, label in (
    ('stable_correct', 'Stable-correct'),
    ('stable_wrong', 'Stable-wrong'),
    ('unstable', 'Unstable')
):

    line(
        f'| {label} | '
        f"{explanation['AD_vs_HC'][state]:.4f} | "
        f"{explanation['FTD_vs_HC'][state]:.4f} |"
    )

line()

line(
    'These effects are nonadditive local SVM-margin '
    'perturbations, not SHAP values or causal biomarkers.'
)

line(
    'The theta/alpha family contains three related features, '
    'while each other group contains one.'
)

line()

line('## 7. Predictive feature-group ablation')

line()

line('| Removed features | AD BACC change | FTD BACC change |')
line('|---|---:|---:|')

for method, label in (
    ('without_delta', 'Delta'),
    ('without_theta_alpha_family', 'Theta/alpha family'),
    ('without_beta', 'Beta'),
    ('without_gamma', 'Gamma')
):

    ad = num(
        ablation[('AD_vs_HC', method)],
        'ablated_minus_full_BACC'
    )

    ftd = num(
        ablation[('FTD_vs_HC', method)],
        'ablated_minus_full_BACC'
    )

    line(
        f'| {label} | '
        f'{signed_pp(ad)} | '
        f'{signed_pp(ftd)} |'
    )

line()

gamma = ablation[
    ('FTD_vs_HC', 'without_gamma')
]

line(
    'FTD gamma removal was the largest observed FTD loss: '
    f"{signed_pp(num(gamma, 'ablated_minus_full_BACC'))}; "
    'conditional bootstrap CI '
    f"{signed_interval(num(gamma, 'conditional_CI_low'), num(gamma, 'conditional_CI_high'))}; "
    'Holm-adjusted p='
    f"{num(gamma, 'Holm_p_across_eight_comparisons'):.4f}."
)

line()

line(
    '**None of the eight group-ablation effects '
    'survived multiple-comparison correction.**'
)

line()

line('## 8. Defensible thesis conclusion')

line()

line(
    'Participant-disjoint internal evaluation demonstrates '
    'that short-EEG diagnostic predictions may vary across '
    'recording time and acquisition conditions. '
    'A nontrivial fraction of temporally consistent '
    'predictions are incorrect. Calibrated uncertainty '
    'can identify some errors retrospectively but '
    'cannot guarantee correctness.'
)

line()

line(
    'The study does not demonstrate independent clinical '
    'validation, prospective Alzheimer risk prediction, '
    'deployable clinical abstention, or causal '
    'frequency-band biomarkers.'
)

line()

line('## 9. Important limitations')

line()

line(
    '- Small, quality-selected cohorts; shared controls '
    'and repeated-CV dependence.'
)

line(
    '- Post-hoc analyses and multiple comparisons.'
)

line(
    '- Conditional uncertainty intervals do not capture '
    'all dataset-selection and model-development uncertainty.'
)

line(
    '- Eyes-open/photic condition and protocol differences '
    'are confounded.'
)

line(
    '- Cross-sectional MMSE cannot establish longitudinal '
    'disease progression or future conversion.'
)

line(
    '- Feature perturbations are nonadditive and are '
    'not direct physiological explanations.'
)

line()

line('## 10. Reproducibility')

line()

line(
    'All numerical results were read from completed '
    'milestone CSV files and cross-checked where possible.'
)

line(
    'See evidence_master.csv, claim_audit.csv, '
    'sha256_inventory.csv and the source research scripts.'
)

line(
    '**Raw EEG recordings are excluded from the archive.** '
    'Dataset access is needed for independent signal reprocessing.'
)


report = OUT / 'THESIS_EVIDENCE_SUMMARY.md'

report.write_text(
    '\n'.join(lines) + '\n',
    encoding='utf-8'
)

table_write(
    OUT / 'evidence_master.csv',
    metrics
)


# ==========================================
# 12. CREATE EXPLICIT CLAIM AUDIT
# ==========================================

claims = [
    {
        'claim':
            'Short-EEG SVMs have measurable internal '
            'participant-CV classification performance.',
        'status': 'SUPPORTED_INTERNALLY',
        'qualification':
            'Not independently externally validated.',
        'evidence': '1B, 4B'
    },
    {
        'claim':
            'Predicted classes can change over five EEG minutes.',
        'status': 'SUPPORTED_DESCRIPTIVELY',
        'qualification':
            'Correlated CV repetitions.',
        'evidence': '1A'
    },
    {
        'claim':
            'Stable predictions can be wrong.',
        'status': 'SUPPORTED_DESCRIPTIVELY',
        'qualification':
            'Stability does not establish clinical correctness.',
        'evidence': '1A, 1B'
    },
    {
        'claim':
            'Calibration gives probabilistic information.',
        'status': 'SUPPORTED_INTERNALLY',
        'qualification':
            'Better Brier than prevalence baseline; '
            'not necessarily the best calibration method.',
        'evidence': '1B'
    },
    {
        'claim':
            'Uncertainty-based abstention is deployable.',
        'status': 'NOT_SUPPORTED',
        'qualification':
            'Retrospective cohort-relative rankings '
            'are not prospective thresholds.',
        'evidence': '1C'
    },
    {
        'claim':
            'MMSE reliably predicts classifier instability.',
        'status': 'NOT_CONFIRMED',
        'qualification':
            'Primary Holm-adjusted associations are uncertain.',
        'evidence': '2B'
    },
    {
        'claim':
            'AD MMSE correlates with stable-wrong frequency.',
        'status': 'HYPOTHESIS_GENERATING',
        'qualification':
            'Secondary n=23 analysis, BH q near 0.05.',
        'evidence': '2B'
    },
    {
        'claim':
            'Predictions vary between EC and EO/photic recordings.',
        'status': 'SUPPORTED_DESCRIPTIVELY',
        'qualification':
            'Within-participant condition transfer.',
        'evidence': '3D, 3E'
    },
    {
        'claim':
            'Cross-condition BACC reliably declines.',
        'status': 'NOT_CONFIRMED',
        'qualification':
            'Paired BACC intervals include zero.',
        'evidence': '3D'
    },
    {
        'claim':
            'Local explanation drift differs by stability state.',
        'status': 'SUPPORTED_CONDITIONALLY',
        'qualification':
            'Post-hoc local perturbation analysis.',
        'evidence': '4A'
    },
    {
        'claim':
            'A frequency-band group is proven necessary.',
        'status': 'NOT_CONFIRMED',
        'qualification':
            'No ablation survives Holm correction.',
        'evidence': '4B'
    },
    {
        'claim':
            'The classifier predicts future Alzheimer onset.',
        'status': 'NOT_TESTED',
        'qualification':
            'No longitudinal conversion labels '
            'or prospective screening cohort.',
        'evidence': 'Study design'
    }
]

table_write(
    OUT / 'claim_audit.csv',
    claims
)


# ==========================================
# 13. ENVIRONMENT AND REPRODUCTION NOTES
# ==========================================

environment = {
    'python': sys.version,
    'platform': platform.platform(),
    'numpy': np.__version__,
    'scipy': scipy.__version__,
    'scikit_learn': sklearn.__version__,
    'mne': mne.__version__
}

(
    OUT / 'environment.json'
).write_text(
    json.dumps(
        environment,
        indent=2,
        sort_keys=True
    ) + '\n',
    encoding='utf-8'
)


readme = """# Thesis evidence and reproducibility packet

This package was created without retraining models or
downloading new EEG recordings.

## Contents

- THESIS_EVIDENCE_SUMMARY.md: consolidated results
- evidence_master.csv: numerical results and intervals
- claim_audit.csv: supported and unsupported claims
- sha256_inventory.csv: source checksums and file sizes
- environment.json: package versions
- scripts/: available research scripts
- reports/: saved derived milestone results and figures

## Experiment sequence

1. Original eyes-closed preprocessing and classification
2. Milestone 1A: temporal stability and correctness
3. Milestone 1B: training-only calibration
4. Milestone 1C: selective prediction and error detection
5. Milestone 2A: MMSE metadata availability
6. Milestone 2B: clinical severity associations
7. Milestone 3A: matched eyes-open dataset metadata
8. Milestone 3B: eyes-open EEG acquisition and QC
9. Milestone 3C: 30-second continuity assessment
10. Milestone 3D: matched cross-condition transfer
11. Milestone 3E: spectral and prediction-transition audit
12. Milestone 4A: temporal explanation stability
13. Milestone 4B: feature-group ablation
14. Milestone 4C: this evidence package

## Important reproducibility limitations

This archive contains scripts, derived results, figures,
and provenance. It does NOT contain raw EEG recordings.

Signal processing and all source-level experiments require
original dataset access and the established project layout.

Dataset releases:
- OpenNeuro ds004504: eyes-closed EEG
- OpenNeuro ds006036 / NEMAR on006036: eyes-open/photic EEG

The eyes-open dataset has the SAME participants, not
an independent external patient cohort.

Review source-data redistribution requirements before
publicly distributing participant-level derived results.

All conclusions are exploratory and conditional on the
study cohorts and repeated cross-validation models.
"""

(
    OUT / 'README_REPRODUCIBILITY.md'
).write_text(
    readme,
    encoding='utf-8'
)


# ==========================================
# 14. COLLECT DERIVED ARTIFACTS
# ==========================================

allowed_extensions = {
    '.csv',
    '.json',
    '.md',
    '.png',
    '.txt'
}

source_files = set()

# Existing research scripts.
source_files.update(
    p for p in ROOT.glob('m*_*.py')
    if p.is_file()
)

source_files.update(
    p for p in ROOT.glob('ftd*_v1.py')
    if p.is_file()
)

# Saved milestone outputs, excluding current output.
for directory in sorted(
    BASE.glob('milestone_*')
):

    if (
        not directory.is_dir()
        or directory == OUT
    ):
        continue

    source_files.update(
        p
        for p in directory.rglob('*')
        if (
            p.is_file()
            and p.suffix.lower()
            in allowed_extensions
        )
    )

# Original small derived feature/prediction sources.
baseline_files = [
    'reports/openneuro_5min_manifest_v2.csv',

    'data/eeg/on004504_v1.0.0/participants.tsv',

    'reports/thesis_final/duration_locked_v1/'
    'duration_features.csv',

    'reports/thesis_final/duration_locked_v1/'
    'five_60s_blocks.csv',

    'reports/thesis_final/duration_locked_v1/'
    'held_out_predictions.csv',

    'reports/thesis_final/temporal_repeated_v1/'
    'all_repeats_heldout_block_predictions.csv',

    'reports/thesis_final/ftd_extension_v1/'
    'locked_study/duration_features.csv',

    'reports/thesis_final/ftd_extension_v1/'
    'locked_study/five_60s_blocks.csv',

    'reports/thesis_final/ftd_extension_v1/'
    'locked_study/held_out_duration_predictions.csv',

    'reports/thesis_final/ftd_extension_v1/'
    'locked_study/held_out_60s_block_predictions.csv'
]

for name in baseline_files:

    path = ROOT / name

    if path.is_file():
        source_files.add(path)


# ==========================================
# 15. BUILD CHECKSUM INVENTORY
# ==========================================

current_outputs = {
    p
    for p in OUT.iterdir()
    if (
        p.is_file()
        and p.name not in (
            'sha256_inventory.csv',
            'thesis_evidence_packet.zip'
        )
    )
}

all_sources = sorted(
    source_files | current_outputs,
    key=lambda p: str(p)
)

inventory = []

for path in all_sources:

    rel = path.relative_to(ROOT)

    if path in current_outputs:

        archive_name = (
            Path('results_summary')
            / path.name
        )

    elif path.suffix == '.py':

        archive_name = (
            Path('scripts')
            / path.name
        )

    else:

        archive_name = rel

    inventory.append({
        'archive_path':
            archive_name.as_posix(),

        'workspace_path':
            rel.as_posix(),

        'size_bytes':
            path.stat().st_size,

        'sha256':
            hashfile(path)
    })


if len({
    r['archive_path']
    for r in inventory
}) != len(inventory):

    raise RuntimeError(
        'Archive filename collision'
    )


table_write(
    OUT / 'sha256_inventory.csv',
    inventory
)


# ==========================================
# 16. CREATE AND VERIFY ZIP ARCHIVE
# ==========================================

archive = (
    OUT / 'thesis_evidence_packet.zip'
)

if archive.exists():
    archive.unlink()

with zipfile.ZipFile(
    archive,
    'w',
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6
) as zf:

    for path, item in zip(
        all_sources,
        inventory
    ):

        zf.write(
            path,
            item['archive_path']
        )

    zf.write(
        OUT / 'sha256_inventory.csv',
        'results_summary/sha256_inventory.csv'
    )


with zipfile.ZipFile(archive) as zf:

    bad = zf.testzip()

    if bad:
        raise RuntimeError(
            'ZIP integrity failure: ' + bad
        )


# ==========================================
# 17. FINAL CROSS-FILE VALIDATION
# ==========================================

print(
    '\n===== FINAL CROSS-FILE CHECKS ====='
)

for task in TASKS:

    print(
        f'{task}: '
        'baseline BACC='
        f"{calibration[task]['original_svm_bacc']:.4f}; "
        'stable-wrong='
        f"{state_rates[task]['stable_wrong']:.4f}; "
        'EO-EC BACC='
        f"{condition[task]['EO_minus_EC_BACC']:+.4f}"
    )

print(
    'PASS: All eight ablation Holm p-values >= 0.05.'
)

print(
    'PASS: Original SVM BACC reconciled '
    'across Milestones 1B, 1C and 4B.'
)

print(
    'PASS: Temporal state counts reconciled '
    'across Milestones 1A and 4A.'
)

print(
    'PASS: Matched label agreement reconciled '
    'across Milestones 3D and 3E.'
)

print(
    '\n===== MILESTONE 4C COMPLETE ====='
)

print(
    'Evidence report:',
    report
)

print(
    'Master numerical results:',
    OUT / 'evidence_master.csv'
)

print(
    'Claim audit:',
    OUT / 'claim_audit.csv'
)

print(
    'Source checksums:',
    OUT / 'sha256_inventory.csv'
)

print(
    'Research evidence archive:',
    archive
)

print(
    f'ZIP verified: {len(all_sources)+1} files, '
    f'{archive.stat().st_size / (1024**2):.2f} MiB'
)

print(
    'No model training, no new EEG downloads, '
    'no earlier milestone outputs changed.'
)

