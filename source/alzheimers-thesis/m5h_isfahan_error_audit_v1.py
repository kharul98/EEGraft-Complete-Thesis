#!/usr/bin/env python3
"""M5H: audit held-out class confusion and stable-wrong behavior, no refitting."""
from pathlib import Path
from collections import Counter, defaultdict
from io import StringIO
import csv
import hashlib
import json
import math
import numpy as np

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
SRC = BASE / 'milestone_5g_isfahan_nested_reliability_v1'
OUT = BASE / 'milestone_5h_isfahan_error_diagnostics_v1'
OUT.mkdir(parents=True, exist_ok=True)

INPUTS = {
    'predictions': SRC / 'held_out_predictions_by_minute.csv',
    'states': SRC / 'subject_repeat_temporal_states.csv',
    'folds': SRC / 'nested_cv_folds_and_hyperparameters.csv',
    'metrics': SRC / 'metrics_by_release_repeat_minute.csv',
    'summary': SRC / 'summary_conditional_subject_bootstrap.csv',
}

EXPECTED = {
    'EEG_1': {'MCI': 11, 'NORMAL': 14},
    'EEG_2': {'MCI': 12, 'NORMAL': 12}
}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def freeze_text(path, body):
    if path.exists():
        if path.read_text(encoding='utf-8') != body:
            raise RuntimeError(
                'Refusing to replace changed output: ' + str(path)
            )
    else:
        path.write_text(body, encoding='utf-8')


def freeze_csv(path, records):
    if not records:
        raise RuntimeError('Empty output ' + str(path))

    fields = list(dict.fromkeys(
        k for r in records for k in r
    ))

    buf = StringIO()
    wr = csv.DictWriter(
        buf,
        fieldnames=fields,
        lineterminator='\n'
    )
    wr.writeheader()
    wr.writerows(records)
    freeze_text(path, buf.getvalue())


def bacc(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    p = y == 1
    n = y == 0

    if not p.any() or not n.any():
        raise RuntimeError(
            'Class missing from BACC calculation'
        )

    return float(
        (
            (pred[p] == 1).mean()
            + (pred[n] == 0).mean()
        ) / 2
    )


# ==========================================
# 1. VERIFY SAVED EXPERIMENT RESULTS
# ==========================================

print(
    '===== MILESTONE 5H: HELD-OUT ERROR AUDIT =====',
    flush=True
)

raw = {
    key: read_csv(path)
    for key, path in INPUTS.items()
}

expected_counts = {
    'predictions': 1225,
    'states': 245,
    'folds': 50,
    'metrics': 50,
    'summary': 2
}

actual_counts = {
    key: len(rows)
    for key, rows in raw.items()
}

if actual_counts != expected_counts:
    raise RuntimeError(
        'M5G source table counts changed: '
        + str(actual_counts)
    )

subject_truth = {}
heldout = {}

for r in raw['predictions']:

    release = r['release']
    sid = r['subject_id']
    repeat = int(r['repeat'])
    minute = int(r['minute'])
    true = int(r['true_MCI'])
    pred = int(r['predicted_MCI'])
    correct = int(r['correct'])
    margin = float(r['raw_SVM_margin'])

    key = (
        release,
        repeat,
        sid,
        minute
    )

    if (
        release not in EXPECTED
        or repeat not in range(1, 6)
        or minute not in range(1, 6)
        or true not in (0, 1)
        or pred not in (0, 1)
        or key in heldout
        or not math.isfinite(margin)
        or correct != int(true == pred)
        or not sid.startswith(release + '-')
    ):
        raise RuntimeError(
            'Invalid or repeated prediction: '
            + str(key)
        )

    # SVC classes were ordered [0, 1].
    # Confirm decision-margin orientation.

    if (
        abs(margin) > 1e-9
        and pred != int(margin > 0)
    ):
        raise RuntimeError(
            'SVM margin/prediction disagree: '
            + str(key)
        )

    if (
        sid in subject_truth
        and subject_truth[sid] != true
    ):
        raise RuntimeError(
            'Conflicting true labels for ' + sid
        )

    subject_truth[sid] = true
    heldout[key] = pred


if len(subject_truth) != 49:
    raise RuntimeError(
        'Expected 49 recording IDs'
    )


# ==========================================
# 2. VERIFY RELEASE AND CLASS MEMBERSHIP
# ==========================================

for release, expected in EXPECTED.items():

    members = [
        sid
        for sid in subject_truth
        if sid.startswith(release + '-')
    ]

    observed = Counter(
        'MCI' if subject_truth[sid] else 'NORMAL'
        for sid in members
    )

    if (
        len(members) != sum(expected.values())
        or observed != expected
    ):
        raise RuntimeError(
            'Class totals mismatch: ' + release
        )

    for sid in members:

        for repeat in range(1, 6):

            if any(
                (
                    release,
                    repeat,
                    sid,
                    minute
                ) not in heldout

                for minute in range(1, 6)
            ):
                raise RuntimeError(
                    'Incomplete minute sequence: ' + sid
                )


# ==========================================
# 3. VERIFY PARTICIPANT-DISJOINT CV
# ==========================================

tests = defaultdict(Counter)

for r in raw['folds']:

    release = r['release']
    repeat = int(r['repeat'])

    train = r['train_subject_ids'].split('|')
    test = r['test_subject_ids'].split('|')

    if (
        release not in EXPECTED
        or repeat not in range(1, 6)
        or len(set(train)) != len(train)
        or len(set(test)) != len(test)
        or set(train) & set(test)
        or len(train) != int(r['n_train'])
        or len(test) != int(r['n_test'])
    ):
        raise RuntimeError(
            'Cross-validation split integrity failure'
        )

    if not all(
        sid.startswith(release + '-')
        for sid in train + test
    ):
        raise RuntimeError(
            'A fold mixes dataset releases'
        )

    members = {
        sid
        for sid in subject_truth
        if sid.startswith(release + '-')
    }

    if set(train + test) != members:
        raise RuntimeError(
            'Unexpected CV participants'
        )

    tests[(release, repeat)].update(test)


for release in EXPECTED:

    members = {
        sid
        for sid in subject_truth
        if sid.startswith(release + '-')
    }

    for repeat in range(1, 6):

        expected_tests = Counter({
            sid: 1
            for sid in members
        })

        if tests[(release, repeat)] != expected_tests:
            raise RuntimeError(
                'Not every subject held out once: '
                + release
            )


# ==========================================
# 4. RECONCILE TEMPORAL STATES
# ==========================================

state_lookup = {}

for r in raw['states']:

    key = (
        r['release'],
        int(r['repeat']),
        r['subject_id']
    )

    if key in state_lookup:
        raise RuntimeError(
            'Repeated temporal state'
        )

    release, repeat, sid = key
    y = subject_truth[sid]

    sequence = [
        heldout[(release, repeat, sid, minute)]
        for minute in range(1, 6)
    ]

    n_correct = sum(
        int(p == y)
        for p in sequence
    )

    if n_correct == 5:
        state = 'stable_correct'

    elif n_correct == 0:
        state = 'stable_wrong'

    else:
        state = 'unstable'

    if (
        r['state'] != state
        or int(r['n_correct_minutes']) != n_correct
        or r['diagnosis']
            != ('MCI' if y else 'NORMAL')
        or r['prediction_sequence']
            != '|'.join(map(str, sequence))
    ):
        raise RuntimeError(
            'Saved temporal state mismatch: '
            + str(key)
        )

    state_lookup[key] = state


if len(state_lookup) != 245:
    raise RuntimeError(
        'Incomplete temporal state lookup'
    )


# ==========================================
# 5. LOAD PREVIOUS BACC SUMMARIES
# ==========================================

metric_lookup = {
    (
        r['release'],
        int(r['repeat']),
        int(r['minute'])
    ): r

    for r in raw['metrics']
}

summary_lookup = {
    r['release']: r
    for r in raw['summary']
}

if (
    len(metric_lookup) != 50
    or set(summary_lookup) != set(EXPECTED)
):
    raise RuntimeError(
        'Missing metric or summary rows'
    )


# ==========================================
# 6. CLASS-WISE ERROR AUDIT
# ==========================================

per_repeat = []
class_rows = []
state_rows = []
hyper_rows = []

report = [
    '# Milestone 5H: Isfahan classifier error audit',
    '',
    'No models fitted. This audit uses saved '
    'out-of-fold predictions from Milestone 5G.',
    '',
    'Each recording appears five times across '
    'correlated cross-validation repeats.',
    ''
]


for release in EXPECTED:

    members = sorted(
        sid
        for sid in subject_truth
        if sid.startswith(release + '-')
    )

    for minute in range(1, 6):

        all_y = []
        all_p = []

        for repeat in range(1, 6):

            y = np.array([
                subject_truth[sid]
                for sid in members
            ])

            p = np.array([
                heldout[(
                    release,
                    repeat,
                    sid,
                    minute
                )]
                for sid in members
            ])

            all_y.extend(y.tolist())
            all_p.extend(p.tolist())

            result = bacc(y, p)

            stored = float(
                metric_lookup[(
                    release,
                    repeat,
                    minute
                )]['BACC']
            )

            if not np.isclose(
                result,
                stored,
                atol=1e-12,
                rtol=0
            ):
                raise RuntimeError(
                    'Per-repeat BACC mismatch'
                )

            per_repeat.append({
                'release': release,
                'repeat': repeat,
                'minute': minute,
                'BACC': result,

                'sensitivity_MCI':
                    float((p[y == 1] == 1).mean()),

                'specificity_NORMAL':
                    float((p[y == 0] == 0).mean()),

                'predicted_MCI_fraction':
                    float(p.mean()),

                'TP':
                    int(((y == 1) & (p == 1)).sum()),

                'TN':
                    int(((y == 0) & (p == 0)).sum()),

                'FP':
                    int(((y == 0) & (p == 1)).sum()),

                'FN':
                    int(((y == 1) & (p == 0)).sum())
            })


        y = np.array(all_y)
        p = np.array(all_p)

        row = {
            'release': release,
            'minute': minute,

            'recording_repeat_assessments':
                len(y),

            'unique_recordings':
                len(members),

            'TP':
                int(((y == 1) & (p == 1)).sum()),

            'TN':
                int(((y == 0) & (p == 0)).sum()),

            'FP':
                int(((y == 0) & (p == 1)).sum()),

            'FN':
                int(((y == 1) & (p == 0)).sum()),

            'MCI_sensitivity':
                float((p[y == 1] == 1).mean()),

            'NORMAL_specificity':
                float((p[y == 0] == 0).mean()),

            'BACC':
                bacc(y, p),

            'fraction_predicted_MCI':
                float(p.mean()),

            'MCI_predicted_MCI_fraction':
                float(p[y == 1].mean()),

            'NORMAL_predicted_MCI_fraction':
                float(p[y == 0].mean())
        }

        class_rows.append(row)

        if minute == 1:

            if not np.isclose(
                row['BACC'],
                float(
                    summary_lookup[release][
                        'minute1_BACC'
                    ]
                ),
                atol=1e-12,
                rtol=0
            ):
                raise RuntimeError(
                    'First-minute summary BACC mismatch'
                )

            print(
                f'\n===== {release}: '
                'FIRST-MINUTE ERROR BREAKDOWN ====='
            )

            print(
                f"MCI sensitivity="
                f"{row['MCI_sensitivity']:.4f}; "
                f"NORMAL specificity="
                f"{row['NORMAL_specificity']:.4f}"
            )

            print(
                f"TP={row['TP']}, "
                f"FN={row['FN']}; "
                f"TN={row['TN']}, "
                f"FP={row['FP']} "
                "(across 5 correlated repeats)"
            )

            print(
                f"Predicted MCI fraction="
                f"{row['fraction_predicted_MCI']:.4f}; "
                f"among MCI="
                f"{row['MCI_predicted_MCI_fraction']:.4f}, "
                f"among NORMAL="
                f"{row['NORMAL_predicted_MCI_fraction']:.4f}"
            )

            print(
                f"BACC={row['BACC']:.4f}; "
                "always-one-class comparator "
                "BACC=0.5000"
            )

            report.extend([
                f'## {release}',

                f"- Minute-1 BACC "
                f"{row['BACC']:.4f}; "
                f"MCI sensitivity "
                f"{row['MCI_sensitivity']:.4f}; "
                f"NORMAL specificity "
                f"{row['NORMAL_specificity']:.4f}",

                f"- Minute-1 TP/FN/TN/FP "
                f"across correlated repeats: "
                f"{row['TP']}/{row['FN']}/"
                f"{row['TN']}/{row['FP']}",

                f"- Fraction predicted MCI: "
                f"{row['fraction_predicted_MCI']:.4f}"
            ])


    # ======================================
    # 7. TEMPORAL STATES BY TRUE DIAGNOSIS
    # ======================================

    print(
        f'===== {release}: '
        'TEMPORAL STATES BY TRUE CLASS ====='
    )

    for group in (
        'MCI',
        'NORMAL'
    ):

        ids = [
            sid
            for sid in members

            if subject_truth[sid]
               == (1 if group == 'MCI' else 0)
        ]

        states = [
            state_lookup[(
                release,
                repeat,
                sid
            )]

            for repeat in range(1, 6)
            for sid in ids
        ]

        counts = Counter(states)

        for state in (
            'stable_correct',
            'stable_wrong',
            'unstable'
        ):

            state_rows.append({
                'release': release,

                'true_group': group,

                'unique_recordings':
                    len(ids),

                'recording_repeat_assessments':
                    len(states),

                'state': state,

                'count_correlated':
                    counts[state],

                'fraction':
                    counts[state] / len(states)
            })

        print(
            f"{group}: "
            f"stable-correct="
            f"{counts['stable_correct']}/{len(states)}, "
            f"stable-wrong="
            f"{counts['stable_wrong']}/{len(states)}, "
            f"unstable="
            f"{counts['unstable']}/{len(states)}"
        )

    report.append(
        '- Temporal state counts by true class '
        'are saved in a separate CSV.'
    )
    report.append('')


# ==========================================
# 8. MODEL SELECTION AUDIT
# ==========================================

for release in EXPECTED:

    folds = [
        r for r in raw['folds']
        if r['release'] == release
    ]

    choices = Counter(
        (
            r['best_C'],
            r['best_gamma']
        )
        for r in folds
    )

    print(
        f'\n===== {release}: '
        'INNER-CV MODEL SELECTION ====='
    )

    for (c, gamma), n in sorted(
        choices.items()
    ):

        hyper_rows.append({
            'release': release,
            'C': c,
            'gamma': gamma,
            'n_outer_folds_selected': n
        })

        print(
            f'C={c}, gamma={gamma}: '
            f'{n}/{len(folds)} folds'
        )


# ==========================================
# 9. SAVE DIAGNOSTICS
# ==========================================

freeze_csv(
    OUT / 'first_minute_and_all_minutes_class_confusion.csv',
    class_rows
)

freeze_csv(
    OUT / 'per_repeat_minute_diagnostics.csv',
    per_repeat
)

freeze_csv(
    OUT / 'temporal_states_by_true_class.csv',
    state_rows
)

freeze_csv(
    OUT / 'nested_CV_hyperparameter_frequencies.csv',
    hyper_rows
)

freeze_text(
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

report.extend([
    '## Interpretation boundaries',
    '',
    '- No new models were trained.',
    '- BACC=0.5 is a comparison reference, '
    'not a formal permutation test.',
    '- Existing conditional bootstrap intervals '
    'do not account for all retraining uncertainty.',
    '- Do not reverse class labels or select '
    'new thresholds using held-out performance.',
    '- Cross-release participant identity '
    'remains unconfirmed.',
    '- MCI versus NORMAL is not equivalent '
    'to Alzheimer dementia diagnosis.',
    ''
])

freeze_text(
    OUT / 'AUDIT_FINDINGS.md',
    '\n'.join(report)
)


# ==========================================
# 10. COMPLETE
# ==========================================

print(
    '\n===== MILESTONE 5H COMPLETE ====='
)

print(
    'PASS: 1,225 prediction rows, '
    '245 temporal states, '
    '50 folds, and 50 metrics cross-audited.'
)

print(
    'No new training; all previous '
    'thesis results unchanged.'
)

print(
    'Saved:',
    OUT
)

