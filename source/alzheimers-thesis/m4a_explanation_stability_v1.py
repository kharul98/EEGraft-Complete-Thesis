#!/usr/bin/env python3
"""Milestone 4A: temporal stability of grouped local SVM score sensitivities."""

from pathlib import Path
from collections import Counter
import csv, hashlib, json
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'

OUT = BASE / 'milestone_4a_temporal_explanation_v1'
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / 'reports/openneuro_5min_manifest_v2.csv'

STATES = (
    BASE / 'milestone_1a_reliability_v1'
    / 'subject_repeat_states.csv'
)

TASKS = {
    'AD_vs_HC': {
        'positive': 'A',
        'n_pos': 23,
        'n_hc': 23,

        'first':
            BASE / 'duration_locked_v1'
            / 'duration_features.csv',

        'blocks':
            BASE / 'duration_locked_v1'
            / 'five_60s_blocks.csv',

        'reference':
            BASE / 'temporal_repeated_v1'
            / 'all_repeats_heldout_block_predictions.csv'
    },

    'FTD_vs_HC': {
        'positive': 'F',
        'n_pos': 18,
        'n_hc': 23,

        'first':
            BASE / 'ftd_extension_v1/locked_study'
            / 'duration_features.csv',

        'blocks':
            BASE / 'ftd_extension_v1/locked_study'
            / 'five_60s_blocks.csv',

        'reference':
            BASE / 'ftd_extension_v1/locked_study'
            / 'held_out_60s_block_predictions.csv'
    }
}

FEATS = [
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma',
    'theta_alpha_ratio'
]

GROUPS = {
    'delta': [0],
    'theta_alpha_family': [1, 2, 5],
    'beta': [3],
    'gamma': [4]
}

GROUP_NAMES = list(GROUPS)

STATE_NAMES = [
    'stable_correct',
    'stable_wrong',
    'unstable'
]

REPEATS = 5
BLOCKS = 5
NBOOT = 3000

MIN_EFFECT = 0.05
MIN_NORM = 0.05


# ==========================================
# 1. UTILITIES
# ==========================================

def read(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as fh:

        return list(csv.DictReader(fh))


def index(rows, fn, what):

    d = {}

    for row in rows:

        key = fn(row)

        if key in d:
            raise RuntimeError(
                f'Duplicate {what}: {key}'
            )

        d[key] = row

    return d


def save(name, rows):

    if not rows:
        raise RuntimeError(
            'Empty results: ' + name
        )

    keys = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with (OUT / name).open(
        'w',
        newline='',
        encoding='utf-8'
    ) as fh:

        w = csv.DictWriter(
            fh,
            fieldnames=keys
        )

        w.writeheader()
        w.writerows(rows)


def svm():

    return Pipeline([
        (
            'scale',
            StandardScaler()
        ),
        (
            'clf',
            SVC(
                kernel='rbf',
                class_weight='balanced'
            )
        )
    ])


def avg_finite(x):

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]

    return (
        float(a.mean())
        if len(a)
        else float('nan')
    )


def format_num(x):

    return (
        f'{x:.4f}'
        if np.isfinite(x)
        else 'not estimable'
    )


# ==========================================
# 2. LOCK ANALYSIS AND INPUTS
# ==========================================

print(
    '===== MILESTONE 4A: '
    'TEMPORAL EXPLANATION STABILITY =====',
    flush=True
)

paths = {
    'manifest': MANIFEST,
    'states': STATES
}

for task, cfg in TASKS.items():

    for key in (
        'first',
        'blocks',
        'reference'
    ):

        paths[
            f'{task}_{key}'
        ] = cfg[key]


checksum = {
    key: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    for key, path in paths.items()
}


protocol = {
    'version': 'm4a_v1',

    'status':
        'Exploratory local model-sensitivity '
        'audit, original diagnostic cohorts',

    'unit':
        'Participant, preserving all five '
        'correlated CV repeats',

    'model':
        'Original repeated nested '
        'subject-CV RBF-SVM, '
        'trained on first 60s',

    'validation':
        'All saved held-out SVM scores '
        'must reproduce to 1e-7',

    'reference':
        'Actual outer-training participant '
        'nearest training feature medians',

    'explanation':
        'Signed SVM decision-function change '
        'when one feature group is replaced '
        'by its training reference',

    'groups': GROUPS,

    'top_effect_threshold':
        MIN_EFFECT,

    'cosine_norm_threshold':
        MIN_NORM,

    'temporal_analysis':
        'Compare held-out blocks 2-5 '
        'with held-out block 1',

    'bootstrap':
        '3000 stratified participant-cluster '
        'resamples preserving repeats',

    'limitations':
        'Nonadditive correlated features; '
        'replacements may be off-manifold; '
        'not SHAP or causal attribution'
}


for filename, data in (
    ('locked_protocol.json', protocol),
    ('input_sha256.json', checksum)
):

    serialized = (
        json.dumps(
            data,
            indent=2,
            sort_keys=True
        ) + '\n'
    )

    path = OUT / filename

    if (
        path.exists()
        and path.read_text() != serialized
    ):
        raise RuntimeError(
            'Locked input or protocol changed: '
            + filename
        )

    if not path.exists():
        path.write_text(serialized)


manifest = index(
    read(MANIFEST),
    lambda r: r['subject'],
    'manifest subject'
)

states = index(
    read(STATES),
    lambda r: (
        r['task'],
        r['subject'],
        int(r['repeat'])
    ),
    'Milestone 1A state'
)


all_blocks = []
all_subjects = []
state_summaries = []
contrasts = []
all_groups = []


# ==========================================
# 3. PROCESS BOTH DIAGNOSTIC TASKS
# ==========================================

for task, cfg in TASKS.items():

    eligible = {
        sid: r['group']
        for sid, r in manifest.items()

        if (
            r['status'] == 'CANDIDATE'
            and r['group']
            in (cfg['positive'], 'C')
        )
    }

    expected_groups = {
        cfg['positive']: cfg['n_pos'],
        'C': cfg['n_hc']
    }

    if Counter(
        eligible.values()
    ) != expected_groups:

        raise RuntimeError(
            'Frozen cohort drift: ' + task
        )

    ids = sorted(eligible)
    n = len(ids)

    y = np.array([
        int(
            eligible[s]
            == cfg['positive']
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

        'minute features'
    )

    prior = index(
        read(cfg['reference']),

        lambda r: (
            int(r['repeat']),
            r['subject'],
            int(r['block'])
        ),

        'original scores'
    )

    if (
        set(first) != set(ids)
        or len(blocks) != n * BLOCKS
        or len(prior)
           != n * REPEATS * BLOCKS
    ):
        raise RuntimeError(
            'Expected feature/prediction '
            'coverage not found for ' + task
        )


    # ======================================
    # 4. BUILD ORIGINAL FEATURE ARRAYS
    # ======================================

    X = np.array([
        [
            float(first[s][f])
            for f in FEATS
        ]

        for s in ids
    ], dtype=float)

    XB = np.array([
        [
            [
                float(
                    blocks[(s, b)][f]
                )

                for f in FEATS
            ]

            for b in range(
                1,
                BLOCKS + 1
            )
        ]

        for s in ids
    ], dtype=float)

    if (
        not np.isfinite(X).all()
        or not np.isfinite(XB).all()
    ):
        raise RuntimeError(
            'Nonfinite spectral input'
        )

    if not np.allclose(
        X,
        XB[:, 0, :],
        rtol=0,
        atol=1e-12
    ):
        raise RuntimeError(
            'First-minute features '
            'do not match block 1'
        )


    impact = np.full(
        (
            REPEATS,
            n,
            BLOCKS,
            len(GROUPS)
        ),
        np.nan
    )

    margin = np.full(
        (
            REPEATS,
            n,
            BLOCKS
        ),
        np.nan
    )

    state_codes = np.full(
        (
            REPEATS,
            n
        ),
        -1,
        dtype=int
    )


    # ======================================
    # 5. EXACT HISTORICAL MODEL REPRODUCTION
    # ======================================

    for rep in range(REPEATS):

        outer = StratifiedKFold(
            5,
            shuffle=True,
            random_state=42 + 100 * rep
        )

        for fold, (
            tr,
            te
        ) in enumerate(
            outer.split(X, y),
            1
        ):

            inner = StratifiedKFold(
                3,
                shuffle=True,

                random_state=(
                    1000
                    + 100 * rep
                    + fold
                )
            )

            search = GridSearchCV(
                svm(),

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

            search.fit(
                X[tr],
                y[tr]
            )

            model = search.best_estimator_

            if list(
                model.classes_
            ) != [0, 1]:

                raise RuntimeError(
                    'Unexpected class ordering'
                )

            test = XB[te].reshape(
                -1,
                len(FEATS)
            )

            scores = (
                model.decision_function(
                    test
                ).reshape(
                    len(te),
                    BLOCKS
                )
            )

            labels = (
                model.predict(
                    test
                ).reshape(
                    len(te),
                    BLOCKS
                )
            )


            # Verify every saved held-out
            # prediction and raw SVM score.

            for j, i in enumerate(te):

                sid = ids[i]

                for b in range(BLOCKS):

                    old = prior[
                        (
                            rep + 1,
                            sid,
                            b + 1
                        )
                    ]

                    if (
                        int(old['fold']) != fold
                        or int(
                            old['true_label']
                        ) != y[i]

                        or old['group']
                           != eligible[sid]

                        or int(
                            old['predicted_label']
                        ) != labels[j, b]

                        or not np.isclose(
                            float(
                                old['decision_score']
                            ),

                            scores[j, b],
                            rtol=0,
                            atol=1e-7
                        )
                    ):
                        raise RuntimeError(
                            'Historical SVM mismatch: '
                            f'{task} {sid} '
                            f'repeat={rep+1} '
                            f'block={b+1}'
                        )


            # =================================
            # 6. TRAINING-ONLY FEATURE REFERENCE
            # =================================

            # Select an actual training participant
            # nearest to training-set medians.
            #
            # No held-out diagnosis or data
            # are used to choose this reference.

            train_median = np.median(
                X[tr],
                axis=0
            )

            train_scale = np.std(
                X[tr],
                axis=0
            )

            distances = (
                (
                    (
                        X[tr] - train_median
                    )

                    / np.maximum(
                        train_scale,
                        1e-12
                    )
                ) ** 2
            ).sum(axis=1)

            anchor = X[
                tr[
                    np.argmin(distances)
                ]
            ].copy()


            # =================================
            # 7. LOCAL GROUPED SVM SENSITIVITY
            # =================================

            # Score effect =
            # original SVM decision score
            # minus score after replacing
            # one linked feature group.
            #
            # These effects are nonadditive.

            impacts = np.empty(
                (
                    len(test),
                    len(GROUP_NAMES)
                ),
                dtype=float
            )

            flat_raw = scores.reshape(-1)

            for g, (
                name,
                indices
            ) in enumerate(
                GROUPS.items()
            ):

                changed = test.copy()

                changed[:, indices] = (
                    anchor[indices]
                )

                perturbed_score = (
                    model.decision_function(
                        changed
                    )
                )

                impacts[:, g] = (
                    flat_raw
                    - perturbed_score
                )


            impact[
                rep,
                te,
                :,
                :
            ] = impacts.reshape(
                len(te),
                BLOCKS,
                -1
            )

            margin[
                rep,
                te,
                :
            ] = scores


        print(
            f'{task}: verified and '
            f'explained repeat '
            f'{rep+1}/{REPEATS}',
            flush=True
        )


    if (
        not np.isfinite(
            impact
        ).all()

        or not np.isfinite(
            margin
        ).all()
    ):
        raise RuntimeError(
            'Missing model explanations'
        )


    # ======================================
    # 8. TEMPORAL EXPLANATION METRICS
    # ======================================

    drift = np.full(
        (REPEATS, n),
        np.nan
    )

    cosine = np.full(
        (REPEATS, n),
        np.nan
    )

    top_persistence = np.full(
        (REPEATS, n),
        np.nan
    )


    for rep in range(REPEATS):

        for i, sid in enumerate(ids):

            saved = states.get(
                (
                    task,
                    sid,
                    rep + 1
                )
            )

            if (
                saved is None

                or saved['group']
                   != eligible[sid]

                or saved['state']
                   not in STATE_NAMES
            ):
                raise RuntimeError(
                    'Invalid saved 1A state: '
                    f'{task} {sid} '
                    f'repeat={rep+1}'
                )

            state = saved['state']

            state_codes[
                rep,
                i
            ] = STATE_NAMES.index(
                state
            )


            z = impact[
                rep,
                i
            ]

            # Mean Euclidean change in the
            # four group sensitivity scores
            # from minute 1 to minutes 2-5.

            drift[
                rep,
                i
            ] = float(
                np.linalg.norm(
                    z[1:] - z[0],
                    axis=1
                ).mean()
            )


            # Directional agreement of
            # the 4-dimensional effect vector.

            norms = np.linalg.norm(
                z,
                axis=1
            )

            valid = (
                (norms[0] >= MIN_NORM)

                & (
                    norms[1:]
                    >= MIN_NORM
                )
            )

            if np.any(valid):

                denom = (
                    norms[1:][valid]
                    * norms[0]
                )

                values = (
                    z[1:][valid] @ z[0]
                ) / denom

                cosine[
                    rep,
                    i
                ] = float(
                    np.clip(
                        values,
                        -1,
                        1
                    ).mean()
                )


            # Dominant group agreement.
            # Weak effects are marked
            # uninformative, not forced
            # into an arbitrary ranking.

            tops = np.argmax(
                np.abs(z),
                axis=1
            )

            informative = (
                np.max(
                    np.abs(z),
                    axis=1
                )
                >= MIN_EFFECT
            )

            valid_top = (
                informative[0]
                & informative[1:]
            )

            if np.any(valid_top):

                top_persistence[
                    rep,
                    i
                ] = float(
                    (
                        tops[1:][valid_top]
                        == tops[0]
                    ).mean()
                )


            all_subjects.append({
                'task': task,
                'subject': sid,
                'group': eligible[sid],
                'repeat': rep + 1,
                'state': state,

                'mean_first_to_later_group_effect_drift':
                    float(
                        drift[rep, i]
                    ),

                'mean_first_to_later_signed_effect_cosine':
                    cosine[rep, i],

                'first_to_later_top_group_persistence':
                    top_persistence[
                        rep,
                        i
                    ],

                'first_minute_dominant_group':
                    (
                        GROUP_NAMES[
                            tops[0]
                        ]

                        if informative[0]

                        else 'none_above_threshold'
                    ),

                'first_minute_max_absolute_group_effect':
                    float(
                        np.max(
                            np.abs(
                                z[0]
                            )
                        )
                    )
            })


            for b in range(BLOCKS):

                all_blocks.append({
                    'task': task,
                    'subject': sid,
                    'group': eligible[sid],
                    'repeat': rep + 1,
                    'block': b + 1,
                    'state': state,

                    'original_raw_svm_margin':
                        float(
                            margin[
                                rep,
                                i,
                                b
                            ]
                        ),

                    'dominant_group':
                        (
                            GROUP_NAMES[
                                tops[b]
                            ]

                            if informative[b]

                            else 'none_above_threshold'
                        ),

                    **{
                        'effect_' + name:
                            float(
                                z[b, g]
                            )

                        for g, name
                        in enumerate(
                            GROUP_NAMES
                        )
                    }
                })


    # ======================================
    # 9. DISPLAY FEATURE-GROUP EFFECTS
    # ======================================

    print(
        f'\n===== {task}: '
        'TEMPORAL GROUP-EXPLANATION '
        'STABILITY ====='
    )

    print(
        'Group effects are raw SVM score '
        'differences, not probabilities '
        'or additive SHAP values.'
    )

    for g, name in enumerate(
        GROUP_NAMES
    ):

        mean_abs = float(
            np.abs(
                impact[..., g]
            ).mean()
        )

        all_groups.append({
            'task': task,
            'group': name,
            'mean_absolute_effect':
                mean_abs
        })

        print(
            f'  {name:20s}: '
            'mean absolute local '
            'SVM-margin effect='
            f'{mean_abs:.4f}'
        )


    # ======================================
    # 10. COMPARE STABILITY STATES
    # ======================================

    for st, state in enumerate(
        STATE_NAMES
    ):

        mask = (
            state_codes == st
        )

        if not mask.any():
            raise RuntimeError(
                'Missing stability state: '
                + state
            )

        row = {
            'task': task,
            'state': state,

            'n_correlated_subject_repeats':
                int(
                    mask.sum()
                ),

            'mean_explanation_drift':
                float(
                    drift[mask].mean()
                ),

            'mean_signed_effect_cosine':
                avg_finite(
                    cosine[mask]
                ),

            'mean_dominant_group_persistence':
                avg_finite(
                    top_persistence[mask]
                ),

            'fraction_with_valid_cosine':
                float(
                    np.isfinite(
                        cosine[mask]
                    ).mean()
                ),

            'fraction_with_valid_top_comparison':
                float(
                    np.isfinite(
                        top_persistence[mask]
                    ).mean()
                )
        }

        state_summaries.append(
            row
        )

        print(
            f'  {state:15s}: '
            f'n='
            f'{row["n_correlated_subject_repeats"]}; '

            f'drift='
            f'{row["mean_explanation_drift"]:.4f}, '

            f'cosine='
            f'{format_num(row["mean_signed_effect_cosine"])}, '

            f'top-group persistence='
            f'{format_num(row["mean_dominant_group_persistence"])} '

            f'(valid top='
            f'{row["fraction_with_valid_top_comparison"]:.1%})'
        )


    # ======================================
    # 11. PARTICIPANT-CLUSTER BOOTSTRAP
    # ======================================

    positive_indices = np.flatnonzero(
        y == 1
    )

    healthy_indices = np.flatnonzero(
        y == 0
    )

    rng = np.random.default_rng(
        20261107
        + (
            0
            if task == 'AD_vs_HC'
            else 100
        )
    )

    comparisons = (
        (
            'unstable_minus_stable_correct',
            2,
            0
        ),

        (
            'stable_wrong_minus_stable_correct',
            1,
            0
        )
    )

    for comparison, st1, st2 in comparisons:

        observed = float(
            drift[
                state_codes == st1
            ].mean()

            -

            drift[
                state_codes == st2
            ].mean()
        )

        samples = []

        for _ in range(NBOOT):

            indices = np.r_[
                rng.choice(
                    positive_indices,
                    len(
                        positive_indices
                    ),
                    replace=True
                ),

                rng.choice(
                    healthy_indices,
                    len(
                        healthy_indices
                    ),
                    replace=True
                )
            ]

            values = drift[
                :,
                indices
            ]

            labels = state_codes[
                :,
                indices
            ]

            if (
                np.any(
                    labels == st1
                )

                and np.any(
                    labels == st2
                )
            ):

                samples.append(
                    float(
                        values[
                            labels == st1
                        ].mean()

                        -

                        values[
                            labels == st2
                        ].mean()
                    )
                )


        if len(samples) >= NBOOT * 0.8:

            lo, hi = np.percentile(
                samples,
                [2.5, 97.5]
            )

        else:

            lo = float('nan')
            hi = float('nan')


        contrasts.append({
            'task': task,
            'comparison': comparison,
            'difference': observed,

            'conditional_95CI_low':
                float(lo),

            'conditional_95CI_high':
                float(hi),

            'valid_bootstrap_resamples':
                len(samples)
        })


        print(
            f'  {comparison}: '
            f'{observed:+.4f}, '
            'conditional CI '
            f'[{format_num(lo)}, '
            f'{format_num(hi)}]'
        )


    # ======================================
    # 12. PLOT GROUP SENSITIVITY BY MINUTE
    # ======================================

    fig, ax = plt.subplots(
        figsize=(7.5, 4.3)
    )

    for g, name in enumerate(
        GROUP_NAMES
    ):

        values = np.abs(
            impact[..., g]
        ).mean(
            axis=(0, 1)
        )

        ax.plot(
            range(
                1,
                BLOCKS + 1
            ),
            values,
            marker='o',
            label=name
        )

    ax.set(
        xlabel='Held-out minute',

        ylabel=(
            'Mean absolute local '
            'SVM score effect'
        ),

        title=(
            task
            + ': feature-group '
            'sensitivity by minute'
        ),

        xticks=range(
            1,
            6
        )
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUT / (
            task
            + '_group_effect_over_time.png'
        ),
        dpi=180
    )

    plt.close(fig)


# ==========================================
# 13. SAVE ALL RESULTS
# ==========================================

save(
    'held_out_group_score_effects.csv',
    all_blocks
)

save(
    'subject_repeat_explanation_stability.csv',
    all_subjects
)

save(
    'stability_state_explanation_summary.csv',
    state_summaries
)

save(
    'state_explanation_drift_bootstrap_contrasts.csv',
    contrasts
)

save(
    'average_group_score_sensitivity.csv',
    all_groups
)


# ==========================================
# 14. COMPLETE
# ==========================================

print(
    '\n===== MILESTONE 4A COMPLETE ====='
)

print(
    'Output directory:',
    OUT
)

print(
    'Original historical SVM scores exactly '
    'reconciled for both tasks.'
)

print(
    'Perturbation reference used '
    'outer-training data only; held-out '
    'diagnoses never used for fitting.'
)

print(
    'Local score sensitivities are nonadditive '
    'perturbation diagnostics, NOT causal '
    'EEG biomarkers or SHAP values.'
)

print(
    'All statistical intervals are conditional '
    'on this dataset, cohort, and '
    'correlated CV runs.'
)

