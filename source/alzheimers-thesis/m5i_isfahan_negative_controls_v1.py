#!/usr/bin/env python3
"""M5I: frozen negative-control audit; no model retraining."""

from pathlib import Path
from collections import Counter, defaultdict
from io import StringIO
import csv
import hashlib
import json

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

M5G = BASE / 'milestone_5g_isfahan_nested_reliability_v1'
M5H = BASE / 'milestone_5h_isfahan_error_diagnostics_v1'
OUT = BASE / 'milestone_5i_isfahan_negative_controls_v1'

OUT.mkdir(parents=True, exist_ok=True)

INPUTS = {
    'predictions':
        M5G / 'held_out_predictions_by_minute.csv',
    'states':
        M5G / 'subject_repeat_temporal_states.csv',
    'summary':
        M5G / 'summary_conditional_subject_bootstrap.csv',
    'confusion':
        M5H / 'first_minute_and_all_minutes_class_confusion.csv',
    'class_states':
        M5H / 'temporal_states_by_true_class.csv'
}

EXPECTED = {
    'EEG_1': (11, 14),
    'EEG_2': (12, 12)
}


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding='utf-8-sig',
        newline=''
    ) as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()

    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b''
        ):
            h.update(chunk)

    return h.hexdigest()


def freeze(path, text):
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise RuntimeError(
                'Frozen output changed: ' + str(path)
            )
    else:
        path.write_text(text, encoding='utf-8')


def freeze_csv(path, rows):
    if not rows:
        raise RuntimeError('Empty results')

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    buf = StringIO()

    writer = csv.DictWriter(
        buf,
        fieldnames=fields,
        lineterminator='\n'
    )

    writer.writeheader()
    writer.writerows(rows)

    freeze(path, buf.getvalue())


print(
    '===== MILESTONE 5I: '
    'FROZEN NEGATIVE CONTROLS =====',
    flush=True
)

tables = {
    key: load(path)
    for key, path in INPUTS.items()
}

expected_rows = {
    'predictions': 1225,
    'states': 245,
    'summary': 2,
    'confusion': 10,
    'class_states': 12
}

actual_rows = {
    key: len(rows)
    for key, rows in tables.items()
}

if actual_rows != expected_rows:
    raise RuntimeError(
        'Audited M5G/M5H row counts changed: '
        + str(actual_rows)
    )


# ========================================
# 1. RECONSTRUCT HELD-OUT PREDICTIONS
# ========================================

predictions = {}
true_labels = {}

for row in tables['predictions']:

    release = row['release']
    sid = row['subject_id']
    repeat = int(row['repeat'])
    minute = int(row['minute'])

    y = int(row['true_MCI'])
    predicted = int(row['predicted_MCI'])

    key = (
        release,
        repeat,
        sid,
        minute
    )

    if (
        release not in EXPECTED
        or not sid.startswith(release + '-')
        or repeat not in range(1, 6)
        or minute not in range(1, 6)
        or y not in (0, 1)
        or predicted not in (0, 1)
        or key in predictions
        or int(row['correct']) != int(y == predicted)
    ):
        raise RuntimeError(
            'Invalid held-out prediction: ' + str(key)
        )

    if (
        sid in true_labels
        and true_labels[sid] != y
    ):
        raise RuntimeError(
            'Changing diagnostic label: ' + sid
        )

    predictions[key] = predicted
    true_labels[sid] = y


if (
    len(predictions) != 1225
    or len(true_labels) != 49
):
    raise RuntimeError(
        'Incomplete prediction matrix'
    )


# ========================================
# 2. VERIFY FROZEN TEMPORAL STATES
# ========================================

saved_states = {}

for row in tables['states']:

    key = (
        row['release'],
        int(row['repeat']),
        row['subject_id']
    )

    if key in saved_states:
        raise RuntimeError(
            'Duplicate temporal state'
        )

    saved_states[key] = row['state']


summary = {
    row['release']: row
    for row in tables['summary']
}

confusion = {
    (
        row['release'],
        int(row['minute'])
    ): row
    for row in tables['confusion']
}

if (
    len(summary) != 2
    or len(confusion) != 10
):
    raise RuntimeError(
        'Summary or confusion tables incomplete'
    )


# ========================================
# 3. CALCULATE NEGATIVE CONTROLS
# ========================================

results = []
classwise = []
subjectwise = []

report = [
    '# Isfahan MCI Reliability: Negative-Control Audit',
    '',
    'All analyses use previously frozen held-out predictions.',
    'No SVM models were retrained or modified.',
    'No label inversion or threshold optimization was performed.',
    '',
    'Repeated cross-validation observations are correlated.',
    ''
]

for release, (n_mci, n_normal) in EXPECTED.items():

    subjects = sorted(
        sid for sid in true_labels
        if sid.startswith(release + '-')
    )

    if (
        len(subjects) != n_mci + n_normal
        or Counter(
            true_labels[sid]
            for sid in subjects
        ) != {
            1: n_mci,
            0: n_normal
        }
    ):
        raise RuntimeError(
            'Unexpected class membership: ' + release
        )

    states = Counter()
    first_minute = Counter()

    states_by_class = defaultdict(Counter)

    for sid in subjects:

        y = true_labels[sid]

        recording_states = []
        recording_first_predictions = []

        for repeat in range(1, 6):

            sequence = [
                predictions[
                    (
                        release,
                        repeat,
                        sid,
                        minute
                    )
                ]
                for minute in range(1, 6)
            ]

            if all(p == y for p in sequence):
                state = 'stable_correct'

            elif all(p != y for p in sequence):
                state = 'stable_wrong'

            else:
                state = 'unstable'

            if saved_states.get(
                (release, repeat, sid)
            ) != state:
                raise RuntimeError(
                    'Saved temporal state mismatch: ' + sid
                )

            states[state] += 1
            states_by_class[y][state] += 1

            first_minute[
                (y, sequence[0])
            ] += 1

            recording_states.append(state)
            recording_first_predictions.append(
                sequence[0]
            )

        subjectwise.append({
            'release': release,
            'subject_id': sid,
            'diagnosis':
                'MCI' if y else 'NORMAL',

            'fraction_first_minute_predicted_MCI':
                sum(recording_first_predictions) / 5,

            'fraction_stable_wrong':
                recording_states.count(
                    'stable_wrong'
                ) / 5,

            'fraction_unstable':
                recording_states.count(
                    'unstable'
                ) / 5
        })

    tp = first_minute[(1, 1)]
    fn = first_minute[(1, 0)]
    tn = first_minute[(0, 0)]
    fp = first_minute[(0, 1)]

    sensitivity = tp / (n_mci * 5)
    specificity = tn / (n_normal * 5)

    bacc = (
        sensitivity + specificity
    ) / 2

    model_accuracy = (
        tp + tn
    ) / (len(subjects) * 5)

    model_predicted_mci = (
        tp + fp
    ) / (len(subjects) * 5)

    n_assessments = len(subjects) * 5

    if sum(states.values()) != n_assessments:
        raise RuntimeError(
            'Incomplete temporal-state results'
        )

    stable_wrong = (
        states['stable_wrong'] / n_assessments
    )

    unstable = (
        states['unstable'] / n_assessments
    )

    prior = summary[release]

    # Reconcile everything with audited M5G/H.
    if (
        abs(
            bacc
            - float(prior['minute1_BACC'])
        ) > 1e-10

        or abs(
            bacc
            - float(confusion[(release, 1)]['BACC'])
        ) > 1e-10

        or abs(
            stable_wrong
            - float(prior['stable_wrong_rate'])
        ) > 1e-10

        or abs(
            unstable
            - float(prior['unstable_rate'])
        ) > 1e-10
    ):
        raise RuntimeError(
            'Numerical disagreement with M5G/H: '
            + release
        )

    always_mci_wrong = (
        n_normal / len(subjects)
    )

    always_normal_wrong = (
        n_mci / len(subjects)
    )

    result = {
        'release': release,
        'n_recordings': len(subjects),
        'n_MCI': n_mci,
        'n_NORMAL': n_normal,

        'model_first_minute_BACC': bacc,
        'model_first_minute_accuracy':
            model_accuracy,

        'model_MCI_sensitivity':
            sensitivity,

        'model_NORMAL_specificity':
            specificity,

        'model_false_positive_rate':
            1 - specificity,

        'model_predicted_MCI_fraction':
            model_predicted_mci,

        'model_stable_correct':
            states['stable_correct'] / n_assessments,

        'model_stable_wrong':
            stable_wrong,

        'model_unstable':
            unstable,

        'always_MCI_BACC': 0.5,

        'always_MCI_accuracy':
            n_mci / len(subjects),

        'always_MCI_stable_correct':
            n_mci / len(subjects),

        'always_MCI_stable_wrong':
            always_mci_wrong,

        'always_MCI_unstable': 0.0,

        'always_NORMAL_BACC': 0.5,

        'always_NORMAL_accuracy':
            n_normal / len(subjects),

        'always_NORMAL_stable_correct':
            n_normal / len(subjects),

        'always_NORMAL_stable_wrong':
            always_normal_wrong,

        'always_NORMAL_unstable': 0.0,

        'model_minus_constant_BACC':
            bacc - 0.5,

        'model_minus_always_MCI_stable_wrong':
            stable_wrong - always_mci_wrong,

        'conditional_BACC_CI_low':
            float(
                prior[
                    'minute1_BACC_conditional_95CI_low'
                ]
            ),

        'conditional_BACC_CI_high':
            float(
                prior[
                    'minute1_BACC_conditional_95CI_high'
                ]
            )
    }

    results.append(result)

    for y, group, n_group in (
        (1, 'MCI', n_mci),
        (0, 'NORMAL', n_normal)
    ):

        for state in (
            'stable_correct',
            'stable_wrong',
            'unstable'
        ):

            classwise.append({
                'release': release,
                'diagnosis': group,
                'state': state,
                'n_unique_recordings':
                    n_group,
                'n_correlated_assessments':
                    n_group * 5,
                'count_correlated':
                    states_by_class[y][state],
                'fraction':
                    states_by_class[y][state]
                    / (n_group * 5)
            })

    print(
        f'\n===== {release}: '
        'CONSTANT-LABEL NEGATIVE CONTROL ====='
    )

    print(
        f'Model: BACC={bacc:.4f}, '
        f'accuracy={model_accuracy:.4f}, '
        f'predicted-MCI={model_predicted_mci:.4f}'
    )

    print(
        f'Model: stable-wrong={stable_wrong:.4f}, '
        f'unstable={unstable:.4f}'
    )

    print(
        f'Always-MCI: BACC=0.5000, '
        f'accuracy={n_mci/len(subjects):.4f}, '
        f'stable-wrong={always_mci_wrong:.4f}, '
        f'unstable=0.0000'
    )

    print(
        f'Always-NORMAL: BACC=0.5000, '
        f'accuracy={n_normal/len(subjects):.4f}, '
        f'stable-wrong={always_normal_wrong:.4f}, '
        f'unstable=0.0000'
    )

    print(
        f'NORMAL false-positive rate='
        f'{1-specificity:.4f}; '
        f'MCI sensitivity={sensitivity:.4f}'
    )

    report.extend([
        f'## {release}',
        '',

        f'- MCI={n_mci}; NORMAL={n_normal}.',

        f'- First-minute SVM BACC: '
        f'{bacc:.4f}; accuracy: '
        f'{model_accuracy:.4f}.',

        f'- MCI sensitivity: {sensitivity:.4f}; '
        f'healthy specificity: {specificity:.4f}.',

        f'- SVM stable-wrong: '
        f'{stable_wrong:.4f}; '
        f'unstable: {unstable:.4f}.',

        f'- Always-MCI BACC: 0.5000; '
        f'stable-wrong: {always_mci_wrong:.4f}.',

        f'- Always-NORMAL BACC: 0.5000; '
        f'stable-wrong: {always_normal_wrong:.4f}.',

        f'- Conditional first-minute BACC '
        f'95% interval: '
        f'[{result["conditional_BACC_CI_low"]:.4f}, '
        f'{result["conditional_BACC_CI_high"]:.4f}].',

        ''
    ])


# ========================================
# 4. SAVE AUDITED RESULTS
# ========================================

freeze_csv(
    OUT / 'constant_label_comparison.csv',
    results
)

freeze_csv(
    OUT / 'states_by_true_class.csv',
    classwise
)

freeze_csv(
    OUT / 'per_recording_prediction_persistence.csv',
    subjectwise
)

freeze(
    OUT / 'input_sha256.json',
    json.dumps(
        {
            key: sha256(path)
            for key, path in INPUTS.items()
        },
        indent=2,
        sort_keys=True
    ) + '\n'
)


# ========================================
# 5. FREEZE INTERPRETATION BOUNDARIES
# ========================================

report.extend([
    '## Interpretation boundaries',
    '',

    '- Constant-label classifiers are negative '
    'controls, not clinically useful models.',

    '- Temporal stability alone does not '
    'establish diagnostic validity.',

    '- No statistical significance or '
    'superiority claim is established.',

    '- Reported results are conditional '
    'on five-minute clean EEG eligibility.',

    '- EEG physical units were missing; '
    'microvolts were explicitly assumed.',

    '- Cross-release participant identity '
    'is not fully established.',

    '- EEG_1 and EEG_2 were evaluated separately.',

    '- MCI detection is different from '
    'Alzheimer dementia diagnosis.',

    '- No additional classification '
    'model was fitted or tuned.',

    ''
])

freeze(
    OUT / 'ISFAHAN_FINAL_INTERPRETATION.md',
    '\n'.join(report)
)

print(
    '\n===== MILESTONE 5I COMPLETE ====='
)

print(
    'PASS: negative-control metrics '
    'cross-checked against M5G and M5H.'
)

print(
    'Outputs:',
    OUT
)

print(
    'No retraining, new exclusions, '
    'threshold changes, or pooled releases.'
)

