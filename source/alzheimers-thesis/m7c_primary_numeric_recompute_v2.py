#!/usr/bin/env python3
"""M7C: independently recompute original AD/FTD results from frozen predictions."""

from pathlib import Path
from collections import Counter, defaultdict
from io import StringIO
import csv
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
OUT = BASE / 'milestone_7c_original_ad_ftd_recomputed_v2'
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    'source_map':
        BASE / 'milestone_7b_primary_source_reconciliation_v1'
        / 'primary_source_inventory.json',

    'pred':
        BASE / 'milestone_1b_calibration_v1'
        / 'calibrated_held_out_block_predictions.csv',

    'metric':
        BASE / 'milestone_1b_calibration_v1'
        / 'calibration_metrics_by_repeat_minute.csv',

    'master':
        BASE / 'milestone_4c_reproducibility_v1'
        / 'evidence_master.csv',

    'claim':
        BASE / 'milestone_4c_reproducibility_v1'
        / 'claim_audit.csv',

    'AD_duration':
        BASE / 'duration_locked_v1'
        / 'held_out_predictions.csv',

    'FTD_duration':
        BASE / 'ftd_extension_v1/locked_study'
        / 'held_out_duration_predictions.csv',

    'FTD_minutes':
        BASE / 'ftd_extension_v1/locked_study'
        / 'minute_wise_metrics.csv',

    'states_1a':
        BASE / 'milestone_1a_reliability_v1'
        / 'subject_repeat_states.csv',
}

TASKS = {
    'AD_vs_HC': 46,
    'FTD_vs_HC': 41
}


# ==========================================
# 1. HELPERS
# ==========================================

def digest(path):

    h = hashlib.sha256()

    with path.open('rb') as f:

        for buf in iter(
            lambda: f.read(1024 * 1024),
            b''
        ):
            h.update(buf)

    return h.hexdigest()


def load(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding='utf-8-sig',
        newline=''
    ) as f:
        return list(csv.DictReader(f))


def freeze(path, content):

    if path.exists():

        if path.read_text(
            encoding='utf-8'
        ) != content:

            raise RuntimeError(
                'Frozen output changed: '
                + str(path)
            )

    else:

        path.write_text(
            content,
            encoding='utf-8'
        )


def freeze_csv(path, rows):

    if not rows:
        raise RuntimeError(
            'Cannot save empty CSV: '
            + str(path)
        )

    cols = list(dict.fromkeys(
        k
        for r in rows
        for k in r
    ))

    buff = StringIO(newline='')

    writer = csv.DictWriter(
        buff,
        fieldnames=cols,
        lineterminator='\n'
    )

    writer.writeheader()
    writer.writerows(rows)

    freeze(
        path,
        buff.getvalue()
    )


def agree(label, a, b, tol=1e-9):

    if not (
        math.isfinite(float(a))
        and math.isfinite(float(b))
    ):
        raise RuntimeError(
            'Nonfinite claim: ' + label
        )

    if not math.isclose(
        float(a),
        float(b),
        rel_tol=0,
        abs_tol=tol
    ):

        raise RuntimeError(
            f'{label}: '
            f'recalculated={a:.12g}, '
            f'saved={b:.12g}'
        )


def balanced(y, pred):

    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    if not np.array_equal(
        np.unique(y),
        [0, 1]
    ):
        raise RuntimeError(
            'Missing binary class'
        )

    return float(
        0.5 * (
            np.mean(
                pred[y == 1] == 1
            )
            +
            np.mean(
                pred[y == 0] == 0
            )
        )
    )


# ==========================================
# 2. VERIFY FROZEN SOURCE INTEGRITY
# ==========================================

print(
    '===== MILESTONE 7C: '
    'PRIMARY AD/FTD NUMERIC RECOMPUTATION =====',
    flush=True
)

for name, path in PATHS.items():

    if not path.is_file():

        raise FileNotFoundError(
            f'{name}: {path}'
        )


source_map = json.loads(
    PATHS['source_map'].read_text(
        encoding='utf-8'
    )
)

if len(
    source_map.get('files', [])
) != 12:

    raise RuntimeError(
        'Expected M7B manifest of 12 source files'
    )


for item in source_map['files']:

    path = ROOT / item['path']

    if (
        not path.is_file()
        or digest(path) != item['sha256']
    ):

        raise RuntimeError(
            'M7B source changed: '
            + item['path']
        )

print(
    'PASS: M7B 12-file SHA256 '
    'inventory verified.'
)


preds = load(PATHS['pred'])
metrics = load(PATHS['metric'])
master = load(PATHS['master'])
claims = load(PATHS['claim'])
states_1a = load(PATHS['states_1a'])

if (
    len(preds),
    len(metrics),
    len(master),
    len(claims)
) != (2175, 50, 81, 12):

    raise RuntimeError(
        'Unexpected primary-source row counts'
    )

print(
    'PASS: 2,175 held-out minute predictions; '
    '50 metric records; 81 evidence entries.'
)


# ==========================================
# 3. ORGANIZE EVIDENCE MASTER
# ==========================================

master_by_key = defaultdict(list)

for r in master:

    master_by_key[
        (r['task'], r['metric'])
    ].append(r)


def master_value(task, metric):

    found = master_by_key[
        (task, metric)
    ]

    if (
        len(found) != 1
        or not found[0]['estimate']
    ):

        raise RuntimeError(
            f'Missing/ambiguous 4C entry: '
            f'{task}/{metric}'
        )

    return float(
        found[0]['estimate']
    )


# ==========================================
# 4. RECONSTRUCT PARTICIPANT-LEVEL DATA
# ==========================================

by_task_repeat_minute = defaultdict(list)
by_task_subject_repeat = defaultdict(dict)

subject_labels = {}
subject_folds = {}
subject_groups = {}
keys = set()


for row in preds:

    task = row['task']
    repeat = int(row['repeat'])
    minute = int(row['minute'])
    sid = row['subject']
    fold = int(row['fold'])

    y = int(row['true_label'])

    original = int(
        row['original_svm_label']
    )

    calibrated = int(
        row['calibrated_threshold_label']
    )

    prob = float(
        row['calibrated_positive_probability']
    )

    prior = float(
        row['train_fold_prevalence']
    )

    key = (
        task,
        repeat,
        sid,
        minute
    )

    if (
        task not in TASKS
        or not sid
        or repeat not in range(1, 6)
        or minute not in range(1, 6)
        or fold not in range(1, 6)
        or y not in (0, 1)
        or original not in (0, 1)
        or calibrated not in (0, 1)
        or key in keys
        or not (0 <= prob <= 1)
        or not (0 <= prior <= 1)
        or not math.isfinite(prob)
        or not math.isfinite(prior)
    ):

        raise RuntimeError(
            'Malformed or repeated prediction: '
            + str(key)
        )

    keys.add(key)

    subjkey = (
        task,
        sid
    )

    foldkey = (
        task,
        repeat,
        sid
    )

    if (
        subjkey in subject_labels
        and subject_labels[subjkey] != y
    ):

        raise RuntimeError(
            'Subject diagnosis changes: '
            + str(subjkey)
        )

    if (
        foldkey in subject_folds
        and subject_folds[foldkey] != fold
    ):

        raise RuntimeError(
            'Outer test fold changes '
            'across minutes: '
            + str(foldkey)
        )

    if (
        subjkey in subject_groups
        and subject_groups[subjkey] != row['group']
    ):

        raise RuntimeError(
            'Subject group label changes: '
            + str(subjkey)
        )

    subject_labels[subjkey] = y
    subject_folds[foldkey] = fold
    subject_groups[subjkey] = row['group']

    by_task_repeat_minute[
        (task, repeat, minute)
    ].append(row)

    by_task_subject_repeat[
        foldkey
    ][minute] = row


if (
    len(subject_labels) != 87
    or len(by_task_subject_repeat) != 435
):

    raise RuntimeError(
        'Expected 87 task-specific people '
        'and 435 repeat-assessments'
    )


for task, expected in TASKS.items():

    subs = [
        (s, y)
        for (t, s), y in subject_labels.items()
        if t == task
    ]

    if len(subs) != expected:

        raise RuntimeError(
            'Wrong participants for ' + task
        )

    class_sizes = sorted(
        Counter(
            y for s, y in subs
        ).values()
    )

    expected_sizes = (
        [23, 23]
        if task == 'AD_vs_HC'
        else [18, 23]
    )

    if class_sizes != sorted(
        expected_sizes
    ):

        raise RuntimeError(
            'Incorrect disease/control split: '
            + task
        )

    for repeat in range(1, 6):

        subjects = {
            s
            for t, r, s in by_task_subject_repeat
            if t == task and r == repeat
        }

        if subjects != {
            s for s, _ in subs
        }:

            raise RuntimeError(
                'Missing held-out subject in repeat'
            )

        for sid in subjects:

            if set(
                by_task_subject_repeat[
                    (task, repeat, sid)
                ]
            ) != set(range(1, 6)):

                raise RuntimeError(
                    'Missing or duplicate minute: '
                    + sid
                )


print(
    'PASS: 46 AD/HC and 41 FTD/HC '
    'participants; five minutes '
    'per held-out repeat.'
)


# ==========================================
# 5. RECOMPUTE ALL CALIBRATION METRICS
# ==========================================

metric_lookup = {}

for row in metrics:

    key = (
        row['task'],
        int(row['repeat']),
        int(row['minute'])
    )

    if (
        key in metric_lookup
        or key[0] not in TASKS
        or key[1] not in range(1, 6)
        or key[2] not in range(1, 6)
    ):

        raise RuntimeError(
            'Duplicate/unexpected metric: '
            + str(key)
        )

    metric_lookup[key] = row


if len(metric_lookup) != 50:

    raise RuntimeError(
        'Metric lookup count incorrect'
    )


recomputed = []

verified_metric_fields = (
    'original_svm_bacc',
    'calibrated_threshold_bacc',
    'brier',
    'training_prior_brier',
    'nll',
    'auc'
)


for key in sorted(metric_lookup):

    task, repeat, minute = key

    rows = sorted(
        by_task_repeat_minute[key],
        key=lambda r: r['subject']
    )

    if (
        len(rows) != TASKS[task]
        or int(
            metric_lookup[key]['n_subjects']
        ) != TASKS[task]
    ):

        raise RuntimeError(
            'Metric/prediction count mismatch: '
            + str(key)
        )

    y = np.array([
        int(r['true_label'])
        for r in rows
    ], dtype=int)

    orig = np.array([
        int(r['original_svm_label'])
        for r in rows
    ], dtype=int)

    cal = np.array([
        int(r['calibrated_threshold_label'])
        for r in rows
    ], dtype=int)

    p = np.array([
        float(
            r['calibrated_positive_probability']
        )
        for r in rows
    ])

    prior = np.array([
        float(r['train_fold_prevalence'])
        for r in rows
    ])

    clipped = np.clip(
        p,
        1e-15,
        1 - 1e-15
    )

    measures = {
        'original_svm_bacc':
            balanced(y, orig),

        'calibrated_threshold_bacc':
            balanced(y, cal),

        'brier':
            float(np.mean((p - y)**2)),

        'training_prior_brier':
            float(np.mean((prior - y)**2)),

        'nll':
            float(np.mean(
                -y * np.log(clipped)
                - (1 - y) * np.log(1 - clipped)
            )),

        'auc':
            float(roc_auc_score(y, p))
    }

    for field in verified_metric_fields:

        agree(
            f'{key} {field}',
            measures[field],
            float(
                metric_lookup[key][field]
            ),
            tol=2e-9
        )

    recomputed.append({
        'task': task,
        'repeat': repeat,
        'minute': minute,
        'n': len(rows),
        **measures
    })


print(
    'PASS: All 50 original/calibrated '
    'BACC, Brier, prior-Brier, NLL '
    'and AUROC rows recalculated.'
)


# ==========================================
# 6. RECALCULATE FIVE-MINUTE STATES
# ==========================================

primary = []
state_records = []


for task in TASKS:

    first = [
        r for r in recomputed
        if (
            r['task'] == task
            and r['minute'] == 1
        )
    ]

    if len(first) != 5:

        raise RuntimeError(
            'Five repeats missing for ' + task
        )

    m = {
        field: float(np.mean([
            r[field]
            for r in first
        ]))
        for field in verified_metric_fields
    }

    metric_names = (
        (
            'original_svm_bacc',
            'first_minute_original_svm_bacc'
        ),
        (
            'calibrated_threshold_bacc',
            'first_minute_calibrated_threshold_bacc'
        ),
        (
            'brier',
            'first_minute_brier'
        ),
        (
            'training_prior_brier',
            'first_minute_training_prior_brier'
        ),
        (
            'nll',
            'first_minute_nll'
        ),
        (
            'auc',
            'first_minute_auc'
        )
    )

    for field, master_field in metric_names:

        agree(
            task + ' / ' + master_field,
            m[field],
            master_value(
                task,
                master_field
            )
        )

    for (
        t,
        repeat,
        sid
    ), minute_map in sorted(
        by_task_subject_repeat.items()
    ):

        if t != task:
            continue

        y = int(
            minute_map[1]['true_label']
        )

        preds5 = [
            int(
                minute_map[i][
                    'original_svm_label'
                ]
            )
            for i in range(1, 6)
        ]

        correct_count = sum(
            p == y
            for p in preds5
        )

        if correct_count == 5:

            state = 'stable_correct'

        elif correct_count == 0:

            state = 'stable_wrong'

        else:

            state = 'unstable'

        state_records.append({
            'task': task,
            'repeat': repeat,
            'subject': sid,
            'true_label': y,
            'n_correct_minutes':
                correct_count,
            'state': state,
            'five_predictions':
                '|'.join(
                    map(str, preds5)
                )
        })

    subject_states = [
        r for r in state_records
        if r['task'] == task
    ]

    count = Counter(
        r['state']
        for r in subject_states
    )

    if len(subject_states) != (
        TASKS[task] * 5
    ):

        raise RuntimeError(
            'State count inconsistent for '
            + task
        )

    state_names = (
        (
            'stable_correct',
            'stable_correct_fraction'
        ),
        (
            'stable_wrong',
            'stable_wrong_fraction'
        ),
        (
            'unstable',
            'unstable_fraction'
        )
    )

    for state, master_field in state_names:

        agree(
            task + ' / ' + master_field,
            count[state] / len(subject_states),
            master_value(
                task,
                master_field
            )
        )

    primary.append({
        'task':
            task,

        'unique_participants':
            TASKS[task],

        'held_out_subject_repeat_assessments':
            len(subject_states),

        'first_minute_original_BACC':
            m['original_svm_bacc'],

        'first_minute_calibrated_BACC':
            m['calibrated_threshold_bacc'],

        'first_minute_calibrated_Brier':
            m['brier'],

        'first_minute_training_prior_Brier':
            m['training_prior_brier'],

        'first_minute_NLL':
            m['nll'],

        'first_minute_AUROC':
            m['auc'],

        'stable_correct_rate':
            count['stable_correct']
            / len(subject_states),

        'stable_wrong_rate':
            count['stable_wrong']
            / len(subject_states),

        'unstable_rate':
            count['unstable']
            / len(subject_states),

        'stable_correct_count_correlated':
            count['stable_correct'],

        'stable_wrong_count_correlated':
            count['stable_wrong'],

        'unstable_count_correlated':
            count['unstable'],
    })


# ==========================================
# 7. PRINT VERIFIED PRIMARY FINDINGS
# ==========================================

print(
    '\n===== INDEPENDENTLY RECOMPUTED '
    'MAIN RESULTS ====='
)

for r in primary:

    print(
        f"{r['task']}: "
        f"n={r['unique_participants']}; "
        "original BACC="
        f"{r['first_minute_original_BACC']:.6f}; "
        "calibrated BACC="
        f"{r['first_minute_calibrated_BACC']:.6f}; "
        "Brier="
        f"{r['first_minute_calibrated_Brier']:.6f}; "
        "training-prior Brier="
        f"{r['first_minute_training_prior_Brier']:.6f}"
    )

    print(
        "  Five-minute states across "
        f"{r['held_out_subject_repeat_assessments']} "
        "correlated subject-repeats: "
        "stable-correct="
        f"{r['stable_correct_rate']:.6%} "
        f"({r['stable_correct_count_correlated']}), "
        "stable-wrong="
        f"{r['stable_wrong_rate']:.6%} "
        f"({r['stable_wrong_count_correlated']}), "
        "unstable="
        f"{r['unstable_rate']:.6%} "
        f"({r['unstable_count_correlated']})"
    )


print(
    'PASS: All 18 main calibration/stability '
    'claims match the 4C evidence master.'
)


# ==========================================
# 8. CROSS-CHECK ORIGINAL DURATION RESULTS
# ==========================================

print(
    '\n===== LEGACY 60-SECOND '
    'SOURCE AGREEMENT ====='
)

legacy_agreement = []


for task, srcname in (
    ('AD_vs_HC', 'AD_duration'),
    ('FTD_vs_HC', 'FTD_duration')
):

    rows = load(
        PATHS[srcname]
    )

    # The original AD duration file contains both Logistic
    # Regression and RBF SVM predictions. This historical
    # comparison must select the original RBF SVM model.
    if task == 'AD_vs_HC':
        models = {r.get('model') for r in rows}
        if models != {'LogisticRegression', 'RBF_SVM'}:
            raise RuntimeError(
                'Unexpected AD model families: ' + str(models)
            )
        rows = [
            r for r in rows
            if r['model'] == 'RBF_SVM'
        ]

    expected_total = (
        TASKS[task] * 5 * 3
    )

    if len(rows) != expected_total:

        raise RuntimeError(
            f'{task} duration rows '
            f'{len(rows)} != {expected_total}'
        )

    first = [
        r for r in rows
        if int(r['duration_sec']) == 60
    ]

    if len(first) != (
        TASKS[task] * 5
    ):

        raise RuntimeError(
            'Unexpected legacy 60-second '
            'row count: '
            + task
        )

    mismatch = 0
    examined = set()

    for r in first:

        key = (
            task,
            int(r['repeat']),
            r['subject'],
            1
        )

        if (
            key not in keys
            or key in examined
        ):

            raise RuntimeError(
                'Missing/duplicate legacy ID: '
                + str(key)
            )

        examined.add(key)

        matched = by_task_subject_repeat[
            (key[0], key[1], key[2])
        ][1]

        if (
            int(r['true_label'])
                != int(matched['true_label'])
            or int(r['predicted_label'])
                != int(
                    matched['original_svm_label']
                )
        ):

            mismatch += 1

    legacy_agreement.append({
        'source': srcname,
        'task': task,
        'tested': len(first),
        'mismatched_60s_labels': mismatch
    })

    print(
        f'{task}: {len(first)} matched '
        'subject/repeat keys; '
        f'{mismatch} original-label mismatches.'
    )

    if mismatch:

        print(
            '  NOTE: Historical outputs are '
            'not numerically identical; '
            'retain separate provenance.'
        )


ftd_minute = load(
    PATHS['FTD_minutes']
)

if len(ftd_minute) != 5:

    raise RuntimeError(
        'Expected five FTD minute summaries'
    )


ftd_disagreements = []

for row in ftd_minute:

    minute = int(
        row['minute']
    )

    mean_bacc = float(
        np.mean([
            r['original_svm_bacc']
            for r in recomputed
            if (
                r['task'] == 'FTD_vs_HC'
                and r['minute'] == minute
            )
        ])
    )

    diff = (
        mean_bacc
        - float(row['mean_BACC'])
    )

    if abs(diff) > 1e-9:

        ftd_disagreements.append(
            (minute, diff)
        )


print(
    'FTD five-minute summary agreement:',
    'MATCH'
    if not ftd_disagreements
    else f'MISMATCH {ftd_disagreements}'
)


# ==========================================
# 9. DOCUMENT ORIGINAL 1A SOURCE
# ==========================================

print(
    '\n===== ORIGINAL 1A '
    'STATE ARTIFACT ====='
)

print(
    f'Rows={len(states_1a)}; '
    'columns='
    f'{list(states_1a[0]) if states_1a else []}'
)

print(
    'Primary state fractions above were '
    'recalculated from saved 1B original '
    'SVM labels.'
)

print(
    'They were not copied from the '
    'earlier 1A state table.'
)


# ==========================================
# 10. SAVE RECOMPUTED EVIDENCE
# ==========================================

freeze_csv(
    OUT / 'recomputed_first_minute_primary_results.csv',
    primary
)

freeze_csv(
    OUT / 'recomputed_metrics_by_repeat_minute.csv',
    recomputed
)

freeze_csv(
    OUT / 'recomputed_subject_repeat_temporal_states.csv',
    state_records
)

freeze_csv(
    OUT / 'legacy_60s_crosscheck.csv',
    legacy_agreement
)

source_hashes = {
    name: digest(path)
    for name, path in PATHS.items()
}

freeze(
    OUT / 'locked_inputs_sha256.json',

    json.dumps(
        source_hashes,
        indent=2,
        sort_keys=True
    ) + '\n'
)


# ==========================================
# 11. GENERATE THESIS-READY RESULTS TEXT
# ==========================================

lines = [
    '# Original Alzheimer and FTD Results: '
    'Recomputed Evidence',
    '',

    'This table was independently calculated '
    'from saved held-out predictions, without '
    'fitting new models.',

    'The unit of analysis is a participant '
    'within a diagnostic task. Five CV '
    'repeats are correlated.',
    '',

    '| Measure | AD vs healthy | FTD vs healthy |',
    '|---|---:|---:|'
]

ad, ftd = primary

for label, key, pct in (
    (
        'Participants',
        'unique_participants',
        False
    ),
    (
        'Original 60-second balanced accuracy',
        'first_minute_original_BACC',
        True
    ),
    (
        'Calibrated-threshold balanced accuracy',
        'first_minute_calibrated_BACC',
        True
    ),
    (
        'Calibrated Brier score',
        'first_minute_calibrated_Brier',
        False
    ),
    (
        'Training-prior Brier baseline',
        'first_minute_training_prior_Brier',
        False
    ),
    (
        'Stable and correct across five minutes',
        'stable_correct_rate',
        True
    ),
    (
        'Stable but wrong across five minutes',
        'stable_wrong_rate',
        True
    ),
    (
        'Changed prediction across five minutes',
        'unstable_rate',
        True
    )
):

    def fmt(x):

        if pct:
            return f'{x * 100:.2f}%'

        if isinstance(x, float):
            return f'{x:.4f}'

        return str(x)

    lines.append(
        f'| {label} | '
        f'{fmt(ad[key])} | '
        f'{fmt(ftd[key])} |'
    )


lines.extend([
    '',

    'The first-minute balanced accuracy is '
    'the mean of five participant-disjoint '
    'cross-validation repeats.',

    'Stability rates use 230 AD/healthy and '
    '205 FTD/healthy correlated subject-repeat '
    'assessments, not 435 independent individuals.',

    '',

    '## Interpretation boundaries',
    '',

    '- A stable prediction can still be wrong; '
    'temporal consistency is not clinical validity.',

    '- Calibration supplies probabilities but '
    'does not guarantee improved classification.',

    '- The original task has no completed '
    'independent external AD-versus-healthy validation.',

    '- The MCI extension is a different disease '
    'comparison and is reported separately.',

    '- Matched acquisition-condition effects '
    'and frequency-band ablations were not '
    'statistically confirmed.',

    '- No longitudinal conversion or clinical '
    'deployment performance has been established.',

    '',

    '## Additional source agreement',
    ''
])


for row in legacy_agreement:

    lines.append(
        f"- {row['task']}: "
        f"{row['mismatched_60s_labels']} "
        "mismatches across "
        f"{row['tested']} older "
        "60-second predictions."
    )


lines.append(
    '- FTD minute summary comparison: '
    + (
        'MATCH'
        if not ftd_disagreements
        else 'MISMATCH - review needed'
    )
)

lines.append('')


freeze(
    OUT / 'PRIMARY_RESULTS_RECOMPUTED_FOR_THESIS.md',
    '\n'.join(lines)
)


# ==========================================
# 12. COMPLETE
# ==========================================

print(
    '\n===== MILESTONE 7C COMPLETE ====='
)

print(
    'Saved:',
    OUT
)

print(
    'Original AD/FTD first-minute performance, '
    'calibration metrics, and five-minute states '
    'recomputed directly from held-out predictions.'
)

print(
    'Results checked against the '
    'Milestone 4C evidence master.'
)

print(
    'No new training, no modified predictions, '
    'and no previous files overwritten.'
)

