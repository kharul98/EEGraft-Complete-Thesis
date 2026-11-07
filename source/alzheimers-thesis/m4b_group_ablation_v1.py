#!/usr/bin/env python3
"""Milestone 4B: participant-disjoint nested-CV leave-feature-group-out ablation."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
OUT = BASE / 'milestone_4b_feature_ablation_v1'
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / 'reports/openneuro_5min_manifest_v2.csv'

TASKS = {
    'AD_vs_HC': {
        'positive': 'A',
        'n_pos': 23,
        'n_hc': 23,
        'first': BASE / 'duration_locked_v1/duration_features.csv',
        'blocks': BASE / 'duration_locked_v1/five_60s_blocks.csv',
        'reference': BASE / 'temporal_repeated_v1/all_repeats_heldout_block_predictions.csv',
    },
    'FTD_vs_HC': {
        'positive': 'F',
        'n_pos': 18,
        'n_hc': 23,
        'first': BASE / 'ftd_extension_v1/locked_study/duration_features.csv',
        'blocks': BASE / 'ftd_extension_v1/locked_study/five_60s_blocks.csv',
        'reference': BASE / 'ftd_extension_v1/locked_study/held_out_60s_block_predictions.csv',
    },
}

FEATURES = (
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma',
    'theta_alpha_ratio'
)

GROUPS = {
    'delta': ('delta',),
    'theta_alpha_family': (
        'theta', 'alpha', 'theta_alpha_ratio'
    ),
    'beta': ('beta',),
    'gamma': ('gamma',),
}

METHODS = {
    'full_6_features': tuple(range(len(FEATURES)))
}

for group, removed in GROUPS.items():
    METHODS['without_' + group] = tuple(
        i for i, f in enumerate(FEATURES)
        if f not in removed
    )

REPEATS = 5
BLOCKS = 5
NBOOT = 4000
NPERM = 10000


# ==========================================
# 1. UTILITIES
# ==========================================

def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:
        return list(csv.DictReader(f))


def index(rows, keyfn, what):
    result = {}

    for r in rows:
        key = keyfn(r)

        if key in result:
            raise RuntimeError(
                f'Duplicate {what}: {key}'
            )

        result[key] = r

    return result


def save(filename, rows):
    if not rows:
        raise RuntimeError(
            'Empty output: ' + filename
        )

    fields = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with (OUT / filename).open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def model():
    return Pipeline([
        ('scale', StandardScaler()),
        (
            'clf',
            SVC(
                kernel='rbf',
                class_weight='balanced'
            )
        ),
    ])


def bacc(y, pred):
    sensitivity = (
        pred[y == 1] == 1
    ).mean()

    specificity = (
        pred[y == 0] == 0
    ).mean()

    return float(
        0.5 * (
            sensitivity + specificity
        )
    )


def holm(pvalues):
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    previous = 0.0

    for rank, j in enumerate(order):
        previous = max(
            previous,
            min(
                1.0,
                (len(pvalues) - rank)
                * float(pvalues[j])
            )
        )

        out[j] = previous

    return out


# ==========================================
# 2. LOCK PROTOCOL AND INPUTS
# ==========================================

print(
    '===== MILESTONE 4B: TRAIN-ONLY GROUP ABLATION =====',
    flush=True
)

paths = {'manifest': MANIFEST}

for task, cfg in TASKS.items():
    for k in (
        'first',
        'blocks',
        'reference'
    ):
        paths[
            f'{task}_{k}'
        ] = cfg[k]

checksums = {
    k: hashlib.sha256(
        v.read_bytes()
    ).hexdigest()
    for k, v in paths.items()
}

protocol = {
    'version':
        'm4b_feature_group_ablation_v1',

    'research_status':
        'post-hoc exploratory '
        'leave-one-feature-group-out ablation',

    'model':
        'StandardScaler + balanced RBF SVC, '
        '3-fold train-only nested '
        'hyperparameter search',

    'outer_folds':
        'Five repeats of five participant-'
        'stratified folds, identical to prior thesis',

    'feature_groups': GROUPS,

    'baseline':
        'all six original EEG relative-band features',

    'held_out_unit':
        'participant, not a two-second epoch '
        'or a minute',

    'training':
        'only first-minute training-participant '
        'features; no held-out participant samples',

    'test':
        'all five held-out 60s blocks; '
        'first-minute BACC is main endpoint',

    'reference_audit':
        'full-model predictions and raw '
        'decision scores must reproduce saved results',

    'bootstrap':
        '4000 stratified participant-cluster '
        'bootstraps preserving five repeats',

    'permutation':
        '10000 paired participant-level sign '
        'flips; approximate post-hoc inference',

    'multiplicity':
        'Holm adjustment across eight '
        'first-minute group-removal contrasts',

    'interpretation':
        'complementary predictive contribution, '
        'not causal biomarker importance',
}

for name, data in [
    ('locked_protocol.json', protocol),
    ('input_sha256.json', checksums)
]:
    p = OUT / name

    s = json.dumps(
        data,
        sort_keys=True,
        indent=2
    ) + '\n'

    if p.exists() and p.read_text() != s:
        raise RuntimeError(
            'Locked protocol or source changed: '
            + name
        )

    if not p.exists():
        p.write_text(s)


# ==========================================
# 3. LOAD FROZEN COHORT
# ==========================================

manifest = index(
    read(MANIFEST),
    lambda r: r['subject'],
    'manifest subject'
)

all_preds = []
all_folds = []
minute_rows = []
state_rows = []
comparison_rows = []


# ==========================================
# 4. PROCESS EACH DIAGNOSTIC TASK
# ==========================================

for task_index, (task, cfg) in enumerate(
    TASKS.items()
):

    cohort = {
        sid: r['group']
        for sid, r in manifest.items()
        if (
            r['status'] == 'CANDIDATE'
            and r['group']
            in (cfg['positive'], 'C')
        )
    }

    expected = {
        cfg['positive']: cfg['n_pos'],
        'C': cfg['n_hc']
    }

    if Counter(cohort.values()) != expected:
        raise RuntimeError(
            'Frozen cohort mismatch: ' + task
        )

    ids = sorted(cohort)
    n = len(ids)

    y = np.array([
        int(
            cohort[s] == cfg['positive']
        )
        for s in ids
    ], dtype=int)

    first = index(
        (
            r for r in read(cfg['first'])
            if int(r['duration_sec']) == 60
        ),
        lambda r: r['subject'],
        'first-minute features'
    )

    blocks = index(
        read(cfg['blocks']),
        lambda r: (
            r['subject'],
            int(r['block_number'])
        ),
        '60s block features'
    )

    previous = index(
        read(cfg['reference']),
        lambda r: (
            int(r['repeat']),
            r['subject'],
            int(r['block'])
        ),
        'saved held-out predictions'
    )

    if (
        set(first) != set(ids)
        or len(blocks) != n * BLOCKS
        or len(previous)
           != REPEATS * n * BLOCKS
    ):
        raise RuntimeError(
            'Features or predictions incomplete: '
            + task
        )

    X = np.array([
        [
            float(first[s][f])
            for f in FEATURES
        ]
        for s in ids
    ])

    XB = np.array([
        [
            [
                float(
                    blocks[(s, b)][f]
                )
                for f in FEATURES
            ]
            for b in range(
                1,
                BLOCKS + 1
            )
        ]
        for s in ids
    ])

    if (
        not np.isfinite(X).all()
        or not np.isfinite(XB).all()
        or not np.allclose(
            X,
            XB[:, 0, :],
            atol=1e-12,
            rtol=0
        )
    ):
        raise RuntimeError(
            'First-minute feature consistency '
            'failed: ' + task
        )


    # ======================================
    # 5. BUILD HELD-OUT OUTPUT ARRAYS
    # ======================================

    predictions = {
        method: np.full(
            (REPEATS, n, BLOCKS),
            -1,
            dtype=int
        )
        for method in METHODS
    }

    scores = {
        method: np.full(
            (REPEATS, n, BLOCKS),
            np.nan
        )
        for method in METHODS
    }

    fold_array = np.full(
        (REPEATS, n),
        -1,
        dtype=int
    )


    # ======================================
    # 6. IDENTICAL OUTER CV SPLITS
    # ======================================

    for rep in range(REPEATS):

        outer = StratifiedKFold(
            5,
            shuffle=True,
            random_state=42 + 100 * rep
        )

        for fold, (tr, te) in enumerate(
            outer.split(X, y),
            1
        ):

            for i in te:

                saved = previous[
                    (
                        rep + 1,
                        ids[i],
                        1
                    )
                ]

                if int(saved['fold']) != fold:
                    raise RuntimeError(
                        'Outer fold drift: '
                        + task + ' ' + ids[i]
                    )

            inner = StratifiedKFold(
                3,
                shuffle=True,
                random_state=(
                    1000
                    + 100 * rep
                    + fold
                )
            )


            # ==================================
            # 7. FIT FULL AND ABLATED MODELS
            # ==================================

            for method, keep in METHODS.items():

                selected = list(keep)

                search = GridSearchCV(
                    model(),
                    {
                        'clf__C': [
                            0.1,
                            1.0,
                            10.0
                        ],
                        'clf__gamma': [
                            'scale',
                            0.01,
                            0.1
                        ]
                    },
                    cv=inner,
                    scoring='balanced_accuracy',
                    refit=True,
                    n_jobs=1,
                    error_score='raise'
                )

                # Only training participants.
                # Excluded feature groups are
                # absent before fitting/scaling.

                search.fit(
                    X[tr][:, selected],
                    y[tr]
                )

                fitted = search.best_estimator_

                flat = (
                    XB[te][:, :, selected]
                    .reshape(
                        -1,
                        len(selected)
                    )
                )

                labels = (
                    fitted.predict(
                        flat
                    ).reshape(
                        len(te),
                        BLOCKS
                    )
                )

                margins = (
                    fitted.decision_function(
                        flat
                    ).reshape(
                        len(te),
                        BLOCKS
                    )
                )

                predictions[method][
                    rep,
                    te,
                    :
                ] = labels

                scores[method][
                    rep,
                    te,
                    :
                ] = margins

                all_folds.append({
                    'task': task,
                    'repeat': rep + 1,
                    'fold': fold,
                    'method': method,

                    'kept_features':
                        '|'.join(
                            FEATURES[j]
                            for j in selected
                        ),

                    'n_outer_train': len(tr),
                    'n_outer_test': len(te),

                    'best_C':
                        search.best_params_['clf__C'],

                    'best_gamma':
                        search.best_params_['clf__gamma'],
                })


                # =================================
                # 8. REPRODUCE HISTORICAL FULL SVM
                # =================================

                if method == 'full_6_features':

                    for k, i in enumerate(te):

                        sid = ids[i]

                        for b in range(BLOCKS):

                            prior = previous[
                                (
                                    rep + 1,
                                    sid,
                                    b + 1
                                )
                            ]

                            if (
                                int(prior['fold']) != fold

                                or int(
                                    prior['true_label']
                                ) != y[i]

                                or prior['group']
                                   != cohort[sid]

                                or int(
                                    prior['predicted_label']
                                ) != int(labels[k, b])

                                or not np.isclose(
                                    float(
                                        prior['decision_score']
                                    ),
                                    margins[k, b],
                                    atol=1e-7,
                                    rtol=0
                                )
                            ):
                                raise RuntimeError(
                                    'Historical SVM disagreement: '
                                    f'{task} {sid} '
                                    f'repeat={rep+1} '
                                    f'block={b+1}'
                                )

            fold_array[
                rep,
                te
            ] = fold

        print(
            f'{task}: nested ablation CV '
            f'repeat {rep+1}/{REPEATS} verified',
            flush=True
        )


    # ======================================
    # 9. VALIDATE COMPLETE HELD-OUT ARRAYS
    # ======================================

    if (
        np.any(fold_array < 1)
        or any(
            np.any(p < 0)
            or not np.isfinite(
                scores[m]
            ).all()
            for m, p in predictions.items()
        )
    ):
        raise RuntimeError(
            'Incomplete held-out predictions'
        )


    # ======================================
    # 10. SAVE PREDICTIONS AND STATES
    # ======================================

    for method, cube in predictions.items():

        for rep in range(REPEATS):

            for i, sid in enumerate(ids):

                first_correct = int(
                    cube[rep, i, 0] == y[i]
                )

                unanimous = bool(
                    np.all(
                        cube[rep, i]
                        == cube[rep, i, 0]
                    )
                )

                if unanimous:

                    state = (
                        'stable_correct'
                        if first_correct
                        else 'stable_wrong'
                    )

                else:
                    state = 'unstable'

                state_rows.append({
                    'task': task,
                    'subject': sid,
                    'group': cohort[sid],
                    'repeat': rep + 1,
                    'method': method,
                    'state': state,
                    'first_minute_correct':
                        first_correct,
                })

                for b in range(BLOCKS):

                    all_preds.append({
                        'task': task,
                        'subject': sid,
                        'group': cohort[sid],
                        'true_label': int(y[i]),
                        'repeat': rep + 1,
                        'fold': int(
                            fold_array[rep, i]
                        ),
                        'minute': b + 1,
                        'method': method,

                        'predicted_label':
                            int(cube[rep, i, b]),

                        'decision_score':
                            float(
                                scores[method][
                                    rep, i, b
                                ]
                            ),

                        'correct':
                            int(
                                cube[rep, i, b]
                                == y[i]
                            ),
                    })


        # ==================================
        # 11. ACCURACY ACROSS FIVE MINUTES
        # ==================================

        for b in range(BLOCKS):

            bb = float(np.mean([
                bacc(
                    y,
                    cube[rep, :, b]
                )
                for rep in range(REPEATS)
            ]))

            minute_rows.append({
                'task': task,
                'method': method,
                'minute': b + 1,
                'mean_repeated_CV_BACC': bb,

                'mean_repeated_CV_accuracy':
                    float(
                        (
                            cube[:, :, b]
                            == y[None, :]
                        ).mean()
                    ),

                'n_independent_participants': n,
                'n_correlated_repeats':
                    REPEATS,
            })


    # ======================================
    # 12. PAIRED BACC COMPARISONS
    # ======================================

    first_bacc = {
        method: next(
            r['mean_repeated_CV_BACC']
            for r in minute_rows
            if (
                r['task'] == task
                and r['method'] == method
                and r['minute'] == 1
            )
        )
        for method in METHODS
    }

    pos_ix = np.flatnonzero(
        y == 1
    )

    hc_ix = np.flatnonzero(
        y == 0
    )

    rng = np.random.default_rng(
        20261107 + task_index
    )

    pos_boot = rng.choice(
        pos_ix,
        size=(NBOOT, len(pos_ix)),
        replace=True
    )

    hc_boot = rng.choice(
        hc_ix,
        size=(NBOOT, len(hc_ix)),
        replace=True
    )

    flip = rng.choice(
        [-1, 1],
        size=(NPERM, n)
    )

    baseline = predictions[
        'full_6_features'
    ][:, :, 0]

    baseline_correct = (
        baseline == y[None, :]
    ).astype(float)


    print(
        '\n===== '
        + task
        + ': FEATURE-GROUP ABLATION ====='
    )

    print(
        'Full six-feature first-minute '
        f'BACC={first_bacc["full_6_features"]:.4f}; '
        'original hard predictions and margins '
        'verified on all five minutes.'
    )


    for method in METHODS:

        pred_cube = predictions[method]

        stable = np.all(
            pred_cube
            == pred_cube[:, :, :1],
            axis=2
        )

        stable_wrong = (
            stable
            & (
                pred_cube[:, :, 0]
                != y[None, :]
            )
        )

        unstable_rate = float(
            (~stable).mean()
        )

        stable_wrong_rate = float(
            stable_wrong.mean()
        )

        print(
            f'  {method:28s} '
            f'BACC(1min)={first_bacc[method]:.4f}, '
            f'unstable={unstable_rate:.4f}, '
            f'stable-wrong={stable_wrong_rate:.4f}'
        )

        if method == 'full_6_features':
            continue

        # Average all five correlated
        # CV repeat outcomes per participant.

        delta = (
            (
                pred_cube[:, :, 0]
                == y[None, :]
            ).astype(float)
            -
            baseline_correct
        ).mean(axis=0)

        observed = 0.5 * (
            delta[pos_ix].mean()
            + delta[hc_ix].mean()
        )

        # Stratified participant bootstrap.
        # Entire five-repeat history stays
        # attached to each participant.

        boot = 0.5 * (
            delta[pos_boot].mean(axis=1)
            + delta[hc_boot].mean(axis=1)
        )

        low, high = np.percentile(
            boot,
            [2.5, 97.5]
        )

        # Approximate paired sign-flip test.
        # Flip each participant's aggregated
        # difference, not individual CV repeats.

        weights = np.where(
            y == 1,
            0.5 / len(pos_ix),
            0.5 / len(hc_ix)
        )

        perm = (
            flip
            * (delta * weights)[None, :]
        ).sum(axis=1)

        p = float(
            (
                1 + np.count_nonzero(
                    np.abs(perm)
                    >= abs(observed) - 1e-12
                )
            )
            / (NPERM + 1)
        )

        if not np.isclose(
            observed,
            (
                first_bacc[method]
                - first_bacc['full_6_features']
            ),
            atol=1e-12
        ):
            raise RuntimeError(
                'Paired BACC contrast mismatch'
            )

        comparison_rows.append({
            'task': task,
            'ablation': method,

            'full_first_minute_BACC':
                first_bacc['full_6_features'],

            'ablated_first_minute_BACC':
                first_bacc[method],

            'ablated_minus_full_BACC':
                float(observed),

            'conditional_CI_low':
                float(low),

            'conditional_CI_high':
                float(high),

            'paired_signflip_p_approx':
                p,

            'Holm_p_across_eight_comparisons':
                '',

            'fraction_first_minute_labels_changed':
                float(
                    (
                        pred_cube[:, :, 0]
                        != baseline
                    ).mean()
                ),

            'change_in_sensitivity':
                float(
                    delta[pos_ix].mean()
                ),

            'change_in_specificity':
                float(
                    delta[hc_ix].mean()
                ),

            'n_patients':
                len(pos_ix),

            'n_healthy':
                len(hc_ix),
        })

        print(
            '    vs full: '
            f'BACC delta={observed:+.4f}; '
            'conditional 95% CI '
            f'[{low:+.4f}, {high:+.4f}]; '
            f'paired sign-flip p={p:.4f}'
        )


    # ======================================
    # 13. FIVE-MINUTE SENSITIVITY RESULTS
    # ======================================

    print(
        '\n  Minutes 1-5 BACC '
        '(same trained model applied '
        'to later held-out blocks):'
    )

    for method in METHODS:

        vals = [
            next(
                r['mean_repeated_CV_BACC']
                for r in minute_rows
                if (
                    r['task'] == task
                    and r['method'] == method
                    and r['minute'] == minute
                )
            )
            for minute in range(
                1,
                BLOCKS + 1
            )
        ]

        print(
            f'    {method:28s} '
            + ' / '.join(
                f'{x:.4f}'
                for x in vals
            )
        )


# ==========================================
# 14. MULTIPLE-COMPARISON ADJUSTMENT
# ==========================================

if len(comparison_rows) != 8:
    raise RuntimeError(
        'Expected exactly eight comparisons'
    )

q = holm(
    np.array([
        r['paired_signflip_p_approx']
        for r in comparison_rows
    ])
)

for row, adjusted in zip(
    comparison_rows,
    q
):
    row[
        'Holm_p_across_eight_comparisons'
    ] = float(adjusted)


# ==========================================
# 15. SAVE RESULTS
# ==========================================

save(
    'held_out_group_ablation_predictions.csv',
    all_preds
)

save(
    'ablation_nested_CV_folds.csv',
    all_folds
)

save(
    'ablation_BACC_by_minute.csv',
    minute_rows
)

save(
    'ablation_stability_states_by_subject_repeat.csv',
    state_rows
)

save(
    'ablation_paired_first_minute_comparisons.csv',
    comparison_rows
)


# ==========================================
# 16. PRINT CORRECTED COMPARISONS
# ==========================================

print(
    '\n===== MULTIPLICITY-ADJUSTED '
    'FIRST-MINUTE COMPARISONS ====='
)

for r in comparison_rows:

    print(
        f"{r['task']} / {r['ablation']}: "
        "BACC difference="
        f"{r['ablated_minus_full_BACC']:+.4f}, "
        "CI "
        f"[{r['conditional_CI_low']:+.4f}, "
        f"{r['conditional_CI_high']:+.4f}], "
        "Holm p="
        f"{r['Holm_p_across_eight_comparisons']:.4f}"
    )


# ==========================================
# 17. SAVE COMPUTING ENVIRONMENT
# ==========================================

environment = {
    'python': sys.version,
    'platform': platform.platform(),
    'numpy': np.__version__,
    'scipy': scipy.__version__,
    'scikit_learn': sklearn.__version__,
}

(
    OUT / 'environment.json'
).write_text(
    json.dumps(
        environment,
        indent=2,
        sort_keys=True
    ) + '\n'
)


# ==========================================
# 18. COMPLETE
# ==========================================

print(
    '\n===== MILESTONE 4B COMPLETE ====='
)

print(
    'Output:',
    OUT
)

print(
    'No prior model, EEG feature, '
    'or cross-condition eligibility changed.'
)

print(
    'Differences are exploratory, conditional '
    'on these participants and correlated '
    'cross-validation models.'
)

print(
    'Feature ablation tests predictive dependence; '
    'it does not establish causal biomarkers.'
)

