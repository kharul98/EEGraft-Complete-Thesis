#!/usr/bin/env python3
"""Milestone 3E: paired spectral drift and held-out prediction transitions."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / 'reports' / 'thesis_final'
SOURCE = BASE / 'milestone_3d_30s_condition_transfer_v1'
OUT = BASE / 'milestone_3e_condition_shift_audit_v1'
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / 'reports/openneuro_5min_manifest_v2.csv'

FEATURES_FILE = (
    SOURCE / 'condition_30s_spectral_features.csv'
)

PREDICTIONS_FILE = (
    SOURCE / 'held_out_paired_condition_predictions.csv'
)

FEATURES = (
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma',
    'theta_alpha_ratio'
)

TASKS = {
    'AD_vs_HC': ('A', 19, 22),
    'FTD_vs_HC': ('F', 13, 22)
}

GROUPS = ('A', 'F', 'C')
REPEATS = 5
NBOOT = 3000

RNG = np.random.default_rng(20261107)


# ==========================================
# 1. UTILITIES
# ==========================================

def load(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:
        return list(csv.DictReader(f))


def save(filename, rows):

    if not rows:
        raise RuntimeError(
            'Empty output: ' + filename
        )

    columns = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with (OUT / filename).open(
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


def unique(rows, key, description):

    result = {}

    for row in rows:

        k = key(row)

        if k in result:
            raise RuntimeError(
                f'Duplicate {description}: {k}'
            )

        result[k] = row

    return result


def ci_of_subject_mean(x):

    x = np.asarray(x, dtype=float)

    draws = RNG.integers(
        0,
        len(x),
        size=(NBOOT, len(x))
    )

    vals = x[draws].mean(axis=1)

    return [
        float(t)
        for t in np.quantile(
            vals,
            [0.025, 0.975]
        )
    ]


# ==========================================
# 2. LOCK SOURCES AND ANALYSIS
# ==========================================

print(
    '===== MILESTONE 3E: '
    'CONDITION-SHIFT FAILURE-MODE AUDIT =====',
    flush=True
)

inputs = {
    'manifest': MANIFEST,
    'features': FEATURES_FILE,
    'predictions': PREDICTIONS_FILE
}

checks = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in inputs.items()
}

protocol = {
    'status':
        'Exploratory descriptive post-hoc audit, '
        'not another trained model',

    'data':
        'Only frozen EC/EO 30s spectral features '
        'and paired held-out predictions',

    'spectral_shift':
        'EO minus EC; also divided by SD of '
        '64 EC participant features for '
        'descriptive comparability',

    'transition_states':
        'correct both, wrong both, '
        'EC-correct/EO-wrong, '
        'EC-wrong/EO-correct',

    'resampling':
        '3000 stratified participant-cluster '
        'bootstraps; keep five CV '
        'repetitions together',

    'interpretation':
        'Do not ascribe feature shifts to one '
        'physiological cause or call this '
        'external validation'
}

for name, data in (
    ('source_sha256.json', checks),
    ('locked_protocol.json', protocol)
):

    path = OUT / name

    encoded = (
        json.dumps(
            data,
            indent=2,
            sort_keys=True
        ) + '\n'
    )

    if (
        path.is_file()
        and path.read_text() != encoded
    ):
        raise RuntimeError(
            'Locked input changed: ' + name
        )

    if not path.is_file():
        path.write_text(encoded)


# ==========================================
# 3. VALIDATE FROZEN COHORTS AND FEATURES
# ==========================================

manifest = unique(
    load(MANIFEST),
    lambda r: r['subject'],
    'manifest subject'
)

selected = {
    sid: r
    for sid, r in manifest.items()
    if (
        r['status'] == 'CANDIDATE'
        and r['group'] in GROUPS
    )
}

if (
    len(selected) != 64
    or Counter(
        r['group']
        for r in selected.values()
    ) != {
        'A': 23,
        'F': 18,
        'C': 23
    }
):
    raise RuntimeError(
        'Frozen EC participants changed'
    )

features = unique(
    load(FEATURES_FILE),
    lambda r: (
        r['condition'],
        r['subject']
    ),
    'condition features'
)

ec = {
    sid: features[
        ('eyes_closed', sid)
    ]
    for sid in selected
}

eo = {
    sid: row
    for (condition, sid), row
    in features.items()
    if condition == 'eyes_open_photic'
}

if (
    len(features) != 118
    or len(ec) != 64
    or len(eo) != 54
    or Counter(
        selected[sid]['group']
        for sid in eo
    ) != {
        'A': 19,
        'F': 13,
        'C': 22
    }
):
    raise RuntimeError(
        'Unexpected EC/EO feature coverage'
    )

for sid in selected:

    r = ec[sid]

    if (
        r['group'] != selected[sid]['group']
        or int(r['duration_sec']) != 30
    ):
        raise RuntimeError(
            'EC feature-group/duration mismatch: '
            + sid
        )

    if sid in eo:

        q = eo[sid]

        if (
            q['group'] != r['group']
            or int(q['duration_sec']) != 30
        ):
            raise RuntimeError(
                'EO feature-group/duration mismatch: '
                + sid
            )

    records = (
        [r, eo[sid]]
        if sid in eo
        else [r]
    )

    for record in records:

        if not all(
            np.isfinite(
                float(record[f])
            )
            for f in FEATURES
        ):
            raise RuntimeError(
                'Nonfinite features: ' + sid
            )

        fraction = float(
            record['clean_fraction']
        )

        if not 0.5 <= fraction <= 1.0:
            raise RuntimeError(
                'Invalid clean epoch fraction: '
                + sid
            )


# ==========================================
# 4. STANDARDIZE PAIRED FEATURE SHIFTS
# ==========================================

EC_MAT = np.array([
    [
        float(ec[sid][f])
        for f in FEATURES
    ]
    for sid in sorted(selected)
])

EC_SD = np.std(
    EC_MAT,
    axis=0,
    ddof=1
)

if np.any(EC_SD <= 1e-12):
    raise RuntimeError(
        'EC spectral feature variation '
        'too small for standardized shifts'
    )

print(
    'PASS: 64 EC and 54 paired EO feature '
    'vectors, matching group labels '
    'and 30s protocol'
)

print(
    'EC-standardized changes use all '
    '64 frozen EC feature rows for '
    'descriptive scaling only.'
)

subject_features = {}

for sid in sorted(eo):

    delta = np.array([
        float(eo[sid][f])
        - float(ec[sid][f])
        for f in FEATURES
    ])

    zdelta = delta / EC_SD

    subject_features[sid] = {
        'group': selected[sid]['group'],
        'delta': delta,
        'zdelta': zdelta,

        'distance':
            float(np.linalg.norm(zdelta)),

        'ec_clean':
            float(
                ec[sid]['clean_fraction']
            ),

        'eo_clean':
            float(
                eo[sid]['clean_fraction']
            )
    }


# ==========================================
# 5. GROUP-SPECIFIC FREQUENCY-BAND SHIFTS
# ==========================================

band_rows = []

heatmap = np.zeros(
    (3, len(FEATURES))
)

for gi, group in enumerate(GROUPS):

    ids = [
        sid for sid in sorted(eo)
        if selected[sid]['group'] == group
    ]

    raw = np.array([
        subject_features[sid]['delta']
        for sid in ids
    ])

    standardized = raw / EC_SD

    for j, feature in enumerate(FEATURES):

        lo, hi = ci_of_subject_mean(
            standardized[:, j]
        )

        heatmap[gi, j] = (
            standardized[:, j].mean()
        )

        band_rows.append({
            'group': group,
            'feature': feature,
            'n_paired_subjects': len(ids),

            'mean_EO_minus_EC_raw':
                float(raw[:, j].mean()),

            'median_EO_minus_EC_raw':
                float(np.median(raw[:, j])),

            'mean_EC_standardized_shift':
                float(
                    standardized[:, j].mean()
                ),

            'conditional_standardized_CI_low':
                lo,

            'conditional_standardized_CI_high':
                hi,

            'fraction_EO_higher':
                float(
                    np.mean(raw[:, j] > 0)
                )
        })


save(
    'paired_band_shifts_by_group.csv',
    band_rows
)


print(
    '\n===== SPECTRAL FEATURE SHIFT '
    'BY DIAGNOSIS ====='
)

for group in GROUPS:

    subset = [
        r for r in band_rows
        if r['group'] == group
    ]

    print(
        f'{group} '
        f'(n={subset[0]["n_paired_subjects"]}): '
        'mean standardized EO-minus-EC shift'
    )

    for r in subset:

        print(
            f'  {r["feature"]:18s}: '
            f'{r["mean_EC_standardized_shift"]:+.3f} '
            'EC SD; conditional 95% CI '
            f'[{r["conditional_standardized_CI_low"]:+.3f}, '
            f'{r["conditional_standardized_CI_high"]:+.3f}]'
        )


# ==========================================
# 6. AUDIT HELD-OUT PREDICTION TRANSITIONS
# ==========================================

pred = unique(
    load(PREDICTIONS_FILE),

    lambda r: (
        r['task'],
        r['subject'],
        int(r['repeat'])
    ),

    'paired prediction'
)

if len(pred) != 5 * (46 + 41):
    raise RuntimeError(
        'Incomplete saved prediction coverage'
    )

TRANSITIONS = (
    'correct_both',
    'wrong_both',
    'EC_correct_EO_wrong',
    'EC_wrong_EO_correct'
)

subject_rows = []
group_rows = []

for task, (
    positive,
    expected_pos,
    expected_hc
) in TASKS.items():

    ids = sorted(
        sid for sid in selected
        if selected[sid]['group']
        in (positive, 'C')
    )

    eligible = sorted(
        sid for sid in ids
        if sid in eo
    )

    if (
        len(eligible)
        != expected_pos + expected_hc
    ):
        raise RuntimeError(
            'Task-specific paired cohort mismatch: '
            + task
        )

    per_subject = {}

    for sid in ids:

        g = selected[sid]['group']
        y = int(g == positive)

        pair = []

        for repeat in range(
            1,
            REPEATS + 1
        ):

            r = pred[
                (task, sid, repeat)
            ]

            if (
                r['group'] != g
                or int(r['true_label']) != y
                or int(r['EO_eligible'])
                   != int(sid in eo)
                or int(r['outer_fold'])
                   not in (1, 2, 3, 4, 5)
            ):
                raise RuntimeError(
                    f'Prediction metadata mismatch: '
                    f'{task} {sid} repeat {repeat}'
                )

            ec_pred = int(
                r['EC_pred_30s']
            )

            if ec_pred not in (0, 1):
                raise RuntimeError(
                    'Unexpected EC prediction'
                )

            if sid not in eo:

                if (
                    r['EO_pred_30s'] != ''
                    or r['EO_margin_30s'] != ''
                ):
                    raise RuntimeError(
                        'Ineligible participant '
                        'has EO prediction'
                    )

                continue

            eo_pred = int(
                r['EO_pred_30s']
            )

            if eo_pred not in (0, 1):
                raise RuntimeError(
                    'Unexpected EO prediction'
                )

            ec_ok = (
                ec_pred == y
            )

            eo_ok = (
                eo_pred == y
            )

            pair.append([
                int(ec_ok and eo_ok),
                int(not ec_ok and not eo_ok),
                int(ec_ok and not eo_ok),
                int(not ec_ok and eo_ok)
            ])

        if sid not in eo:
            continue

        if len(pair) != REPEATS:
            raise RuntimeError(
                'Incomplete repeated paired '
                'predictions: ' + sid
            )

        values = np.mean(
            np.asarray(
                pair,
                dtype=float
            ),
            axis=0
        )

        if not np.isclose(
            values.sum(),
            1.0
        ):
            raise RuntimeError(
                'Transition partition failed: '
                + sid
            )

        per_subject[sid] = values

        info = subject_features[sid]

        subject_rows.append({
            'task': task,
            'subject': sid,
            'group': g,

            'n_correlated_cv_repeats':
                REPEATS,

            **{
                TRANSITIONS[j] + '_fraction':
                    float(values[j])
                for j in range(4)
            },

            'label_agreement_fraction':
                float(
                    values[0] + values[1]
                ),

            'EC_correct_fraction':
                float(
                    values[0] + values[2]
                ),

            'EO_correct_fraction':
                float(
                    values[0] + values[3]
                ),

            'EO_minus_EC_correct_fraction':
                float(
                    values[3] - values[2]
                ),

            'paired_spectral_shift_EC_SD_norm':
                info['distance'],

            'EC_clean_fraction':
                info['ec_clean'],

            'EO_clean_fraction':
                info['eo_clean']
        })

    posids = [
        sid for sid in eligible
        if selected[sid]['group'] == positive
    ]

    hcids = [
        sid for sid in eligible
        if selected[sid]['group'] == 'C'
    ]

    posmat = np.array([
        per_subject[sid]
        for sid in posids
    ])

    hcmat = np.array([
        per_subject[sid]
        for sid in hcids
    ])

    if (
        posmat.shape != (expected_pos, 4)
        or hcmat.shape != (expected_hc, 4)
    ):
        raise RuntimeError(
            'Unexpected patient/control '
            'transition array'
        )

    def record(group, mat, draws):

        observed = mat.mean(
            axis=0
        )

        boot = mat[draws].mean(
            axis=1
        )

        metrics = {
            'correct_both':
                (
                    observed[0],
                    boot[:, 0]
                ),

            'wrong_both':
                (
                    observed[1],
                    boot[:, 1]
                ),

            'EC_correct_EO_wrong':
                (
                    observed[2],
                    boot[:, 2]
                ),

            'EC_wrong_EO_correct':
                (
                    observed[3],
                    boot[:, 3]
                ),

            'label_agreement':
                (
                    observed[0] + observed[1],
                    boot[:, 0] + boot[:, 1]
                ),

            'EO_minus_EC_accuracy':
                (
                    observed[3] - observed[2],
                    boot[:, 3] - boot[:, 2]
                )
        }

        for name, (
            value,
            samples
        ) in metrics.items():

            lo, hi = np.quantile(
                samples,
                [0.025, 0.975]
            )

            group_rows.append({
                'task': task,
                'group': group,

                'n_independent_participants':
                    len(mat),

                'metric': name,
                'rate': float(value),

                'conditional_CI_low':
                    float(lo),

                'conditional_CI_high':
                    float(hi)
            })

        print(
            f'{task} / {group} '
            f'({len(mat)} subjects): '
            f'correct-both={observed[0]:.3f}, '
            f'wrong-both={observed[1]:.3f}, '
            'EC-correct/EO-wrong='
            f'{observed[2]:.3f}, '
            'EC-wrong/EO-correct='
            f'{observed[3]:.3f}; '
            'label-agreement='
            f'{observed[0] + observed[1]:.3f}'
        )

    print(
        f'\n===== {task}: '
        'HELD-OUT PREDICTION TRANSITIONS ====='
    )

    dp = RNG.integers(
        0,
        len(posmat),
        size=(NBOOT, len(posmat))
    )

    dh = RNG.integers(
        0,
        len(hcmat),
        size=(NBOOT, len(hcmat))
    )

    record(
        positive,
        posmat,
        dp
    )

    record(
        'C',
        hcmat,
        dh
    )

    both = np.concatenate(
        [posmat, hcmat]
    )

    all_boot = np.concatenate(
        [
            posmat[dp],
            hcmat[dh]
        ],
        axis=1
    ).mean(axis=1)

    all_observed = both.mean(
        axis=0
    )

    all_metrics = (
        (
            'correct_both',
            all_observed[0],
            all_boot[:, 0]
        ),

        (
            'wrong_both',
            all_observed[1],
            all_boot[:, 1]
        ),

        (
            'EC_correct_EO_wrong',
            all_observed[2],
            all_boot[:, 2]
        ),

        (
            'EC_wrong_EO_correct',
            all_observed[3],
            all_boot[:, 3]
        ),

        (
            'label_agreement',
            all_observed[0] + all_observed[1],
            all_boot[:, 0] + all_boot[:, 1]
        ),

        (
            'EO_minus_EC_accuracy',
            all_observed[3] - all_observed[2],
            all_boot[:, 3] - all_boot[:, 2]
        )
    )

    for name, observed, vals in all_metrics:

        lo, hi = np.quantile(
            vals,
            [0.025, 0.975]
        )

        group_rows.append({
            'task': task,
            'group': 'ALL_MATCHED',

            'n_independent_participants':
                len(both),

            'metric': name,
            'rate': float(observed),

            'conditional_CI_low':
                float(lo),

            'conditional_CI_high':
                float(hi)
        })

    agreement = float(
        all_observed[0]
        + all_observed[1]
    )

    known_agreement = (
        0.6098
        if task == 'AD_vs_HC'
        else 0.6571
    )

    if (
        abs(
            agreement - known_agreement
        ) > 0.00015
    ):
        raise RuntimeError(
            'Original Milestone 3D agreement '
            'not reproduced: '
            + str(agreement)
        )

    ecb = 0.5 * (
        (
            posmat[:, 0]
            + posmat[:, 2]
        ).mean()

        +

        (
            hcmat[:, 0]
            + hcmat[:, 2]
        ).mean()
    )

    eob = 0.5 * (
        (
            posmat[:, 0]
            + posmat[:, 3]
        ).mean()

        +

        (
            hcmat[:, 0]
            + hcmat[:, 3]
        ).mean()
    )

    ec_known, eo_known = (
        (0.6694, 0.5811)
        if task == 'AD_vs_HC'
        else (0.7364, 0.6941)
    )

    if (
        abs(ecb - ec_known) > 0.00015
        or abs(eob - eo_known) > 0.00015
    ):
        raise RuntimeError(
            'Milestone 3D BACC not reproduced: '
            f'{ecb}, {eob}'
        )

    print(
        'PASS: matched EC BACC='
        f'{ecb:.4f}; '
        f'EO BACC={eob:.4f}; '
        f'EO-EC={eob - ecb:+.4f}; '
        f'agreement={agreement:.4f}'
    )


# ==========================================
# 7. SAVE PREDICTION TRANSITIONS
# ==========================================

save(
    'paired_prediction_transitions_by_group.csv',
    group_rows
)

save(
    'subjectwise_spectral_shift_and_prediction_transitions.csv',
    subject_rows
)


# ==========================================
# 8. SUMMARIZE OVERALL SPECTRAL DRIFT
# ==========================================

print(
    '\n===== PAIRED SPECTRAL-SHIFT '
    'MAGNITUDE ====='
)

for group in GROUPS:

    rows = [
        v
        for sid, v in subject_features.items()
        if v['group'] == group
    ]

    shifts = np.array([
        v['distance']
        for v in rows
    ])

    clean_delta = np.array([
        v['eo_clean'] - v['ec_clean']
        for v in rows
    ])

    low, high = ci_of_subject_mean(
        shifts
    )

    print(
        f'{group}: n={len(rows)}; '
        'mean 6-feature standardized '
        f'shift norm={shifts.mean():.3f} '
        '(conditional CI '
        f'[{low:.3f}, {high:.3f}]); '
        'mean EO-minus-EC clean fraction='
        f'{clean_delta.mean():+.3f}'
    )


# ==========================================
# 9. SAVE SPECTRAL SHIFT FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(9, 3.6)
)

scale = max(
    float(
        np.max(
            np.abs(heatmap)
        )
    ),
    0.01
)

image = ax.imshow(
    heatmap,
    aspect='auto',
    cmap='coolwarm',
    vmin=-scale,
    vmax=scale
)

ax.set_xticks(
    range(len(FEATURES)),
    labels=FEATURES,
    rotation=25,
    ha='right'
)

ax.set_yticks(
    range(len(GROUPS)),
    labels=GROUPS
)

ax.set_title(
    'Paired EO/photic minus EC spectral '
    'features (EC SD units)'
)

fig.colorbar(
    image,
    ax=ax,
    label='Mean paired difference / pooled EC SD'
)

fig.tight_layout()

fig.savefig(
    OUT / 'cross_condition_spectral_shift_heatmap.png',
    dpi=180
)

plt.close(fig)


# ==========================================
# 10. METHODS AND LIMITATIONS
# ==========================================

notes = """# Milestone 3E: Descriptive failure-mode audit

- Uses only frozen 30-second paired features and held-out
  EC/EO predictions. No retraining.
- All five correlated CV repetitions are kept within
  each participant during bootstrap resampling.
- Model agreement and the four transition states use
  the original hard SVM predictions.
- Standardized spectral shifts are descriptive and use
  the EC standard deviation across 64 selected people.
  This normalization is not used for model fitting.
- Uncertainty intervals are conditional on selected
  participants and previously fitted models.
- Eyes-open state, intermittent photic stimulation,
  recording position and preprocessing are confounded.
  Spectral changes do not identify a causal mechanism.
- EO quality screening excluded 10 participants from
  the original 64. This is not independent validation.
- Exploratory analysis without confirmatory hypothesis
  testing or multiple-comparison significance claims.
"""

(
    OUT / 'METHODS_AND_LIMITATIONS.md'
).write_text(
    notes,
    encoding='utf-8'
)


print(
    '\n===== MILESTONE 3E COMPLETE ====='
)

print(
    'Saved:',
    OUT
)

print(
    'No new classifier trained and no '
    'EC/EO eligibility rules changed.'
)

