#!/usr/bin/env python3
"""M5G: release-separated, first-minute nested-CV MCI reliability experiment."""

from pathlib import Path
from collections import Counter, defaultdict
from io import StringIO
import csv
import hashlib
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

SRC = (
    BASE
    / 'milestone_5f_isfahan_spectral_features_v2'
)

QCBASE = (
    BASE
    / 'milestone_5d_isfahan_labels_and_qc_v2'
)

OUT = (
    BASE
    / 'milestone_5g_isfahan_nested_reliability_v1'
)

OUT.mkdir(parents=True, exist_ok=True)

FEATS = (
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma',
    'theta_alpha_ratio'
)

RELEASES = {
    'EEG_1': {'MCI': 11, 'NORMAL': 14},
    'EEG_2': {'MCI': 12, 'NORMAL': 12},
}

REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 3
NBOOT = 2500


# ==========================================
# 1. UTILITIES
# ==========================================

def sha256(path):

    h = hashlib.sha256()

    with path.open('rb') as fh:

        for part in iter(
            lambda: fh.read(1024 * 1024),
            b''
        ):
            h.update(part)

    return h.hexdigest()


def read_csv(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as fh:

        return list(csv.DictReader(fh))


def freeze(path, content):

    if path.exists():

        if path.read_text(
            encoding='utf-8'
        ) != content:

            raise RuntimeError(
                'Refusing to overwrite changed '
                'frozen output: ' + str(path)
            )

    else:

        path.write_text(
            content,
            encoding='utf-8'
        )


def freeze_csv(path, rows):

    if not rows:
        raise RuntimeError(
            'Empty output: ' + str(path)
        )

    fields = list(dict.fromkeys(
        k
        for r in rows
        for k in r
    ))

    buf = StringIO(newline='')

    writer = csv.DictWriter(
        buf,
        fieldnames=fields,
        lineterminator='\n'
    )

    writer.writeheader()
    writer.writerows(rows)

    freeze(
        path,
        buf.getvalue()
    )


def rates(y, pred):

    positive = y == 1
    negative = y == 0

    if (
        not positive.any()
        or not negative.any()
    ):
        raise RuntimeError(
            'Missing class in performance assessment'
        )

    sensitivity = float(
        np.mean(
            pred[positive] == 1
        )
    )

    specificity = float(
        np.mean(
            pred[negative] == 0
        )
    )

    bacc = (
        sensitivity + specificity
    ) / 2

    return (
        float(bacc),
        sensitivity,
        specificity
    )


# ==========================================
# 2. LOAD AND VERIFY FROZEN INPUTS
# ==========================================

print(
    '===== MILESTONE 5G: '
    'SEPARATE-RELEASE NESTED CV =====',
    flush=True
)

feature_file = (
    SRC / 'five_minute_spectral_features.csv'
)

manifest_file = (
    QCBASE
    / 'release_aware_verified_subject_manifest.csv'
)

qc_file = (
    QCBASE
    / 'first_five_minute_quality.csv'
)

protocol_file = (
    SRC / 'locked_protocol.json'
)

features = read_csv(feature_file)
manifest = read_csv(manifest_file)
quality = read_csv(qc_file)

if (
    len(features) != 245
    or len(manifest) != 61
    or len(quality) != 61
):

    raise RuntimeError(
        'Frozen input counts changed'
    )

with protocol_file.open(
    encoding='utf-8'
) as fh:

    source_protocol = json.load(fh)

if (
    source_protocol.get('version') != 'm5f_v2'
    or source_protocol.get('n_eligible') != 49
):

    raise RuntimeError(
        'Not the audited M5F-v2 feature protocol'
    )

manifest_by_id = {
    r['subject_id']: r
    for r in manifest
}

qc_by_id = {
    r['subject_id']: r
    for r in quality
}

if (
    len(manifest_by_id) != 61
    or len(qc_by_id) != 61
):

    raise RuntimeError(
        'Duplicate manifest or QC subject'
    )

eligible = {
    sid
    for sid, r in qc_by_id.items()
    if r['status'] == 'ELIGIBLE_FIXED_300S'
}

if len(eligible) != 49:
    raise RuntimeError(
        'Eligibility changed'
    )


# ==========================================
# 3. CHECK ALL 245 FEATURE VECTORS
# ==========================================

by_subject = defaultdict(dict)

for row in features:

    sid = row['subject_id']
    minute = int(row['minute'])

    if (
        sid not in eligible
        or minute not in range(1, 6)
        or minute in by_subject[sid]
    ):

        raise RuntimeError(
            'Unexpected, repeated, or '
            'ineligible feature row: ' + sid
        )

    if (
        row['release']
        != manifest_by_id[sid]['release']

        or row['diagnosis']
        != manifest_by_id[sid]['diagnosis']
    ):

        raise RuntimeError(
            'Label or release mismatch: ' + sid
        )

    expected_label = int(
        row['diagnosis'] == 'MCI'
    )

    if (
        int(row['true_label_MCI'])
        != expected_label
    ):

        raise RuntimeError(
            'Wrong numeric label: ' + sid
        )

    if int(row['clean_epochs']) < 15:

        raise RuntimeError(
            'Ineligible minute included: ' + sid
        )

    vector = np.array(
        [
            float(row[k])
            for k in FEATS
        ],
        dtype=float
    )

    if (
        not np.isfinite(vector).all()
        or np.any(vector < 0)
    ):

        raise RuntimeError(
            'Invalid feature: ' + sid
        )

    by_subject[sid][minute] = vector


if (
    set(by_subject) != eligible

    or any(
        set(minutes) != set(range(1, 6))
        for minutes in by_subject.values()
    )
):

    raise RuntimeError(
        'Expected five vectors for '
        'each eligible recording'
    )


# ==========================================
# 4. FREEZE EXPERIMENTAL PROTOCOL
# ==========================================

protocol = {

    'version': 'm5g_v1',

    'model':
        'StandardScaler + '
        'class-weight-balanced RBF SVC',

    'grid': {
        'C': [0.1, 1, 10],
        'gamma': [
            'scale',
            0.01,
            0.1
        ]
    },

    'inner_selection':
        '3-fold stratified CV balanced accuracy '
        'using only outer-training '
        'first-minute vectors',

    'outer_assessment':
        '5-fold stratified CV x 5 repeats; '
        'each recording tested once per repeat',

    'outer_seeds': [
        42 + 100 * r
        for r in range(REPEATS)
    ],

    'inner_seed_rule':
        '1000 + 100*repeat_index + '
        'outer_fold_index; zero-based indexes',

    'training_minutes': [1],

    'prediction_minutes': [1, 2, 3, 4, 5],

    'release_handling':
        'Completely separate training, '
        'tuning, and testing by release',

    'cross_release_identity':
        'Unknown; no pooling and no '
        'cross-release transfer in this experiment',

    'classification':
        'Original SVM hard prediction; '
        'no threshold tuning or calibration',

    'temporal_state_rule':
        'stable-correct if all five predictions '
        'correct; stable-wrong if all five wrong; '
        'otherwise unstable',

    'bootstrap':
        '2500 class-stratified subject-clustered '
        'resamples, same subject multiplicities '
        'across all five repeats and minutes; '
        'fixed predictions and no model refitting',

    'source_checksums': {

        'M5F_v2_features_sha256':
            sha256(feature_file),

        'M5F_v2_protocol_sha256':
            sha256(protocol_file),

        'M5D_v2_manifest_sha256':
            sha256(manifest_file),

        'M5D_v2_QC_sha256':
            sha256(qc_file)
    }
}

freeze(
    OUT / 'locked_protocol.json',
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True
    ) + '\n'
)

print(
    'PASS: 49 recordings with five '
    'validated minutes; no pooled training.',
    flush=True
)


# ==========================================
# 5. RELEASE-SEPARATED NESTED CV
# ==========================================

prediction_rows = []
fold_rows = []
state_rows = []
metric_rows = []
summary_rows = []

for release, expected in RELEASES.items():

    subjects = sorted(
        sid
        for sid in eligible
        if (
            manifest_by_id[sid]['release']
            == release
        )
    )

    labels = np.array(
        [
            int(
                manifest_by_id[sid]['diagnosis']
                == 'MCI'
            )
            for sid in subjects
        ],
        dtype=int
    )

    observed = Counter(
        manifest_by_id[sid]['diagnosis']
        for sid in subjects
    )

    if dict(observed) != expected:

        raise RuntimeError(
            'Release class counts changed: '
            + release
        )

    X = np.stack([
        [
            by_subject[sid][minute]
            for minute in range(1, 6)
        ]
        for sid in subjects
    ])

    if X.shape != (
        len(subjects),
        5,
        6
    ):

        raise RuntimeError(
            'Unexpected feature matrix dimensions'
        )

    all_pred = np.full(
        (
            REPEATS,
            len(subjects),
            5
        ),
        -1,
        dtype=int
    )

    all_margin = np.full(
        (
            REPEATS,
            len(subjects),
            5
        ),
        np.nan
    )

    for repeat in range(REPEATS):

        cv_outer = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=42 + 100 * repeat
        )

        test_counts = np.zeros(
            len(subjects),
            dtype=int
        )

        for outer_fold, (
            train,
            test
        ) in enumerate(
            cv_outer.split(
                X[:, 0, :],
                labels
            )
        ):

            if set(train).intersection(test):

                raise RuntimeError(
                    'Participant leakage '
                    'in outer cross-validation'
                )

            cv_inner = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=(
                    1000
                    + 100 * repeat
                    + outer_fold
                )
            )

            base = Pipeline([
                (
                    'scale',
                    StandardScaler()
                ),
                (
                    'svc',
                    SVC(
                        kernel='rbf',
                        class_weight='balanced'
                    )
                )
            ])

            search = GridSearchCV(
                base,

                {
                    'svc__C': [
                        0.1,
                        1,
                        10
                    ],

                    'svc__gamma': [
                        'scale',
                        0.01,
                        0.1
                    ]
                },

                cv=cv_inner,
                scoring='balanced_accuracy',
                refit=True,
                n_jobs=1,
                error_score='raise'
            )

            # Only the first minute of
            # outer-training participants
            # is used for model selection
            # and final training.

            search.fit(
                X[train, 0, :],
                labels[train]
            )

            best = search.best_estimator_

            if not np.array_equal(
                best.classes_,
                [0, 1]
            ):

                raise RuntimeError(
                    'Unexpected SVM class order'
                )

            fold_rows.append({

                'release': release,

                'repeat':
                    repeat + 1,

                'outer_fold':
                    outer_fold + 1,

                'n_train':
                    len(train),

                'n_test':
                    len(test),

                'train_MCI':
                    int(labels[train].sum()),

                'test_MCI':
                    int(labels[test].sum()),

                'best_C':
                    search.best_params_[
                        'svc__C'
                    ],

                'best_gamma':
                    search.best_params_[
                        'svc__gamma'
                    ],

                'best_inner_BACC':
                    float(search.best_score_),

                'train_subject_ids':
                    '|'.join(
                        subjects[j]
                        for j in train
                    ),

                'test_subject_ids':
                    '|'.join(
                        subjects[j]
                        for j in test
                    )
            })

            test_counts[test] += 1

            # The same trained model
            # evaluates all five minutes
            # of each held-out recording.

            for minute in range(5):

                predictions = best.predict(
                    X[test, minute, :]
                ).astype(int)

                margins = (
                    best.decision_function(
                        X[test, minute, :]
                    ).astype(float)
                )

                all_pred[
                    repeat,
                    test,
                    minute
                ] = predictions

                all_margin[
                    repeat,
                    test,
                    minute
                ] = margins

                for j, answer, score in zip(
                    test,
                    predictions,
                    margins
                ):

                    prediction_rows.append({

                        'release': release,

                        'repeat':
                            repeat + 1,

                        'outer_fold':
                            outer_fold + 1,

                        'subject_id':
                            subjects[j],

                        'true_MCI':
                            int(labels[j]),

                        'minute':
                            minute + 1,

                        'predicted_MCI':
                            int(answer),

                        'raw_SVM_margin':
                            float(score),

                        'correct':
                            int(
                                answer == labels[j]
                            )
                    })

        if (
            not np.all(test_counts == 1)

            or np.any(
                all_pred[repeat] < 0
            )

            or not np.isfinite(
                all_margin[repeat]
            ).all()
        ):

            raise RuntimeError(
                'Missing, repeated, or invalid '
                'held-out predictions'
            )

        print(
            f'{release}: nested CV repeat '
            f'{repeat + 1}/{REPEATS} verified',
            flush=True
        )


    # ======================================
    # 6. MINUTE-WISE PERFORMANCE
    # ======================================

    for repeat in range(REPEATS):

        for minute in range(5):

            bacc, sens, spec = rates(
                labels,
                all_pred[
                    repeat,
                    :,
                    minute
                ]
            )

            metric_rows.append({

                'release': release,

                'repeat':
                    repeat + 1,

                'minute':
                    minute + 1,

                'n_subjects':
                    len(subjects),

                'n_MCI':
                    int(labels.sum()),

                'n_NORMAL':
                    int(
                        (labels == 0).sum()
                    ),

                'BACC': bacc,

                'MCI_sensitivity':
                    sens,

                'NORMAL_specificity':
                    spec
            })


        # ==================================
        # 7. FIVE-MINUTE TEMPORAL STATES
        # ==================================

        for idx, sid in enumerate(subjects):

            correct = (
                all_pred[
                    repeat,
                    idx
                ]
                == labels[idx]
            )

            if correct.all():

                state = 'stable_correct'

            elif not correct.any():

                state = 'stable_wrong'

            else:

                state = 'unstable'

            state_rows.append({

                'release': release,

                'repeat':
                    repeat + 1,

                'subject_id':
                    sid,

                'diagnosis':
                    manifest_by_id[
                        sid
                    ]['diagnosis'],

                'n_correct_minutes':
                    int(correct.sum()),

                'state':
                    state,

                'prediction_sequence':
                    '|'.join(
                        str(z)
                        for z in all_pred[
                            repeat,
                            idx
                        ]
                    )
            })


    # ======================================
    # 8. REPEATED-CV SUMMARY
    # ======================================

    state_arr = np.empty(
        (
            REPEATS,
            len(subjects)
        ),
        dtype=object
    )

    for repeat in range(REPEATS):

        right = (
            all_pred[repeat]
            == labels[:, None]
        )

        state_arr[repeat] = np.where(
            right.all(axis=1),

            'stable_correct',

            np.where(
                (~right).all(axis=1),
                'stable_wrong',
                'unstable'
            )
        )

    point_bacc = [

        np.mean([
            rates(
                labels,
                all_pred[
                    repeat,
                    :,
                    minute
                ]
            )[0]

            for repeat in range(REPEATS)
        ])

        for minute in range(5)
    ]

    point_states = {

        state: float(
            np.mean(
                state_arr == state
            )
        )

        for state in (
            'stable_correct',
            'stable_wrong',
            'unstable'
        )
    }


    # ======================================
    # 9. PARTICIPANT-CLUSTERED BOOTSTRAP
    # ======================================

    rng = np.random.default_rng(
        20261107
        + (
            1
            if release == 'EEG_1'
            else 2
        )
    )

    idx0 = np.where(
        labels == 0
    )[0]

    idx1 = np.where(
        labels == 1
    )[0]

    draws = {
        f'minute{minute}_BACC': []
        for minute in range(1, 6)
    }

    draws.update({

        f'{state}_rate': []

        for state in point_states
    })

    draws[
        'minute5_minus_minute1_BACC'
    ] = []

    for _ in range(NBOOT):

        # Stratify by diagnosis, but
        # resample entire recordings.
        #
        # The same sampled recording IDs
        # are reused across repeats and
        # all five minutes.

        sampled = np.concatenate([

            rng.choice(
                idx0,
                size=len(idx0),
                replace=True
            ),

            rng.choice(
                idx1,
                size=len(idx1),
                replace=True
            )
        ])

        bacc_sample = [

            np.mean([

                rates(
                    labels[sampled],
                    all_pred[
                        repeat,
                        sampled,
                        minute
                    ]
                )[0]

                for repeat in range(REPEATS)
            ])

            for minute in range(5)
        ]

        for minute, value in enumerate(
            bacc_sample,
            1
        ):

            draws[
                f'minute{minute}_BACC'
            ].append(
                float(value)
            )

        draws[
            'minute5_minus_minute1_BACC'
        ].append(
            float(
                bacc_sample[4]
                - bacc_sample[0]
            )
        )

        for state in point_states:

            draws[
                f'{state}_rate'
            ].append(
                float(
                    np.mean(
                        state_arr[
                            :,
                            sampled
                        ] == state
                    )
                )
            )


    # ======================================
    # 10. CONDITIONAL CONFIDENCE INTERVALS
    # ======================================

    result = {

        'release': release,

        'n_recordings':
            len(subjects),

        'n_MCI':
            int(labels.sum()),

        'n_NORMAL':
            int(
                (labels == 0).sum()
            ),

        'bootstrap_replicates':
            NBOOT
    }

    for minute, value in enumerate(
        point_bacc,
        1
    ):

        key = f'minute{minute}_BACC'

        result[key] = float(value)

        result[
            key
            + '_conditional_95CI_low'
        ] = float(
            np.percentile(
                draws[key],
                2.5
            )
        )

        result[
            key
            + '_conditional_95CI_high'
        ] = float(
            np.percentile(
                draws[key],
                97.5
            )
        )

    key = (
        'minute5_minus_minute1_BACC'
    )

    result[key] = float(
        point_bacc[4]
        - point_bacc[0]
    )

    result[
        key + '_conditional_95CI_low'
    ] = float(
        np.percentile(
            draws[key],
            2.5
        )
    )

    result[
        key + '_conditional_95CI_high'
    ] = float(
        np.percentile(
            draws[key],
            97.5
        )
    )

    for state, value in point_states.items():

        key = f'{state}_rate'

        result[key] = value

        result[
            key
            + '_conditional_95CI_low'
        ] = float(
            np.percentile(
                draws[key],
                2.5
            )
        )

        result[
            key
            + '_conditional_95CI_high'
        ] = float(
            np.percentile(
                draws[key],
                97.5
            )
        )

    summary_rows.append(result)


    # ======================================
    # 11. PRINT RESULTS
    # ======================================

    print(
        f'\n===== {release}: '
        'HELD-OUT MCI RELIABILITY ====='
    )

    print(
        'Minute BACC:',
        ' | '.join(
            f'{value:.4f}'
            for value in point_bacc
        )
    )

    print(
        'First-minute BACC='
        f'{point_bacc[0]:.4f}, '
        'conditional 95% CI '
        f'[{result["minute1_BACC_conditional_95CI_low"]:.4f}, '
        f'{result["minute1_BACC_conditional_95CI_high"]:.4f}]'
    )

    print(
        'Fifth-minus-first BACC='
        f'{result["minute5_minus_minute1_BACC"]:+.4f}, '
        'conditional 95% CI '
        f'[{result["minute5_minus_minute1_BACC_conditional_95CI_low"]:+.4f}, '
        f'{result["minute5_minus_minute1_BACC_conditional_95CI_high"]:+.4f}]'
    )

    print(
        'Five-minute states:',
        ', '.join(
            f'{state}={value:.4f}'
            for state, value in point_states.items()
        )
    )


# ==========================================
# 12. CROSS-FILE OUTPUT VERIFICATION
# ==========================================

if (
    len(prediction_rows)
    != 49 * REPEATS * 5

    or len(state_rows)
    != 49 * REPEATS

    or len(metric_rows)
    != 2 * REPEATS * 5

    or len(fold_rows)
    != 2 * REPEATS * OUTER_FOLDS
):

    raise RuntimeError(
        'Unexpected nested-CV output counts'
    )


# ==========================================
# 13. SAVE REPRODUCIBLE RESULTS
# ==========================================

freeze_csv(
    OUT
    / 'held_out_predictions_by_minute.csv',
    prediction_rows
)

freeze_csv(
    OUT
    / 'nested_cv_folds_and_hyperparameters.csv',
    fold_rows
)

freeze_csv(
    OUT
    / 'subject_repeat_temporal_states.csv',
    state_rows
)

freeze_csv(
    OUT
    / 'metrics_by_release_repeat_minute.csv',
    metric_rows
)

freeze_csv(
    OUT
    / 'summary_conditional_subject_bootstrap.csv',
    summary_rows
)


# ==========================================
# 14. SAVE INTERPRETATION SAFEGUARDS
# ==========================================

note = [

    '# Milestone 5G: Separate-release '
    'MCI vs NORMAL reliability',

    '',

    'The two Isfahan archive releases '
    'were trained and evaluated separately.',

    'No data from one release entered '
    'training or tuning for the other.',

    'Each recording was held out once '
    'per outer repeat.',

    'All five minute predictions came from '
    'a model trained only on first-minute '
    'features of other recordings.',

    'Intervals are conditional, '
    'recording-clustered bootstrap '
    'summaries of fixed CV predictions.',

    'They do not account for retraining '
    'variation in a new population.',

    ''
]

for result in summary_rows:

    note.extend([

        f'## {result["release"]} '
        f'(n={result["n_recordings"]})',

        f'- Minute-1 BACC: '
        f'{result["minute1_BACC"]:.4f} '
        f'[{result["minute1_BACC_conditional_95CI_low"]:.4f}, '
        f'{result["minute1_BACC_conditional_95CI_high"]:.4f}]',

        f'- Minute-5 minus minute-1 BACC: '
        f'{result["minute5_minus_minute1_BACC"]:+.4f} '
        f'[{result["minute5_minus_minute1_BACC_conditional_95CI_low"]:+.4f}, '
        f'{result["minute5_minus_minute1_BACC_conditional_95CI_high"]:+.4f}]',

        f'- Unstable rate: '
        f'{result["unstable_rate"]:.4f}; '
        f'stable-wrong rate: '
        f'{result["stable_wrong_rate"]:.4f}',

        ''
    ])

note.extend([

    '## Limitations',

    '- Cross-release participant linkage '
    'is not established; no pooled '
    'cohort estimate is reported.',

    '- MCI versus healthy is a different '
    'clinical task from AD/FTD versus healthy.',

    '- EDF physical units were missing; '
    'microvolts were assumed and documented.',

    '- Acquisition, preprocessing, artifacts, '
    'demographics, and cohort selection '
    'may differ from the original study.',

    '- No probability calibration, clinical '
    'diagnosis, or prospective screening '
    'performance is established.',

    '- Small cohorts produce substantial '
    'uncertainty; repeated CV measurements '
    'are correlated.',

    ''
])

freeze(
    OUT / 'RESULTS_AND_LIMITATIONS.md',
    '\n'.join(note)
)


# ==========================================
# 15. FINISH
# ==========================================

print(
    '\n===== MILESTONE 5G COMPLETE ====='
)

print(
    'Nested-CV folds:',
    len(fold_rows)
)

print(
    'Held-out minute predictions:',
    len(prediction_rows)
)

print(
    'Subject-repeat temporal states:',
    len(state_rows)
)

print(
    'Results:',
    OUT
)

print(
    'No pooled cross-release model, '
    'no original AD/FTD baseline outputs changed.'
)

