#!/usr/bin/env python3
"""Milestone 3D: matched 30s eyes-closed-to-photic transfer, participant-disjoint."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
OUT = BASE / 'milestone_3d_30s_condition_transfer_v1'
OUT.mkdir(parents=True, exist_ok=True)

M = ROOT / 'reports/openneuro_5min_manifest_v2.csv'
QC = BASE / 'milestone_3c_boundary_feasibility_v1/duration_continuity_and_30s_qc.csv'
OLD_AD = BASE / 'duration_locked_v1/duration_features.csv'
OLD_FTD = BASE / 'ftd_extension_v1/locked_study/duration_features.csv'
OLD_FOLDS = BASE / 'milestone_1b_calibration_v1/calibrated_held_out_block_predictions.csv'
META = ROOT / 'data/eeg/on004504_v1.0.0/participants.tsv'

FEATS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'theta_alpha_ratio']
BANDS = [
    ('delta', 1, 4),
    ('theta', 4, 8),
    ('alpha', 8, 13),
    ('beta', 13, 30),
    ('gamma', 30, 45)
]

REPEATS = 5
NBOOT = 3000
FS = 500


def read(path, delim=','):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=delim))


def idx(rows, fn, what):
    d = {}
    for row in rows:
        k = fn(row)
        if k in d:
            raise RuntimeError(f'Duplicate {what}: {k}')
        d[k] = row
    return d


def save(name, rows):
    if not rows:
        raise RuntimeError('No rows: ' + name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def signal_epochs(data):
    if data.shape[0] != 19 or data.shape[1] % 1000:
        raise RuntimeError('Unexpected signal shape')
    e = data.reshape(19, -1, 1000).transpose(1, 0, 2)
    p = np.ptp(e, axis=2)
    sd = np.std(e, axis=2)
    clean = ~(
        ((p > 200).sum(axis=1) >= 3)
        | (p > 300).any(axis=1)
        | (sd < 0.5).any(axis=1)
    )
    return e, clean


def spectral(data, min_epochs):
    if not np.isfinite(data).all():
        raise RuntimeError('Nonfinite EEG')

    ep, clean = signal_epochs(data)
    n = int(clean.sum())

    if n < min_epochs:
        return None, n

    freq, psd = welch(
        ep[clean],
        fs=FS,
        nperseg=1000,
        noverlap=0,
        axis=-1,
        window='hann',
        detrend='constant',
        scaling='density'
    )

    denom = psd[..., (freq >= 1) & (freq <= 45)].sum(axis=-1)

    if np.any(denom <= 0) or not np.isfinite(denom).all():
        raise RuntimeError('Invalid PSD denominator')

    values = {}

    for name, lo, hi in BANDS:
        mask = (
            (freq >= lo)
            & ((freq <= hi) if name == 'gamma' else (freq < hi))
        )
        values[name] = float(
            np.median(psd[..., mask].sum(axis=-1) / denom)
        )

    values['theta_alpha_ratio'] = (
        values['theta'] / max(values['alpha'], 1e-12)
    )

    return values, n


def raw_file(path, ref_channels):
    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = mne.io.read_raw_eeglab(
        str(path),
        preload=False,
        verbose='ERROR'
    )

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch = {raw.ch_names[j].strip().lower() for j in picks}

    if (
        abs(float(raw.info['sfreq']) - FS) > 1e-6
        or len(picks) != 19
        or ch != ref_channels
    ):
        raise RuntimeError(
            'Sampling rate or EEG montage mismatch: ' + str(path)
        )

    return raw, picks


def has_boundary(raw, start, end):
    return any(
        'boundary' in str(desc).lower()
        and start + 0.002 < float(t) < end - 0.002
        for t, desc in zip(
            raw.annotations.onset,
            raw.annotations.description
        )
    )


def get_uv(raw, picks, start_s, duration_s):
    a = int(round(start_s * FS))
    b = a + int(round(duration_s * FS))

    if a < 0 or b > raw.n_times:
        raise RuntimeError('EEG window outside available samples')

    sig = raw.get_data(
        picks=picks,
        start=a,
        stop=b
    ) * 1e6

    if sig.shape != (19, int(round(duration_s * FS))):
        raise RuntimeError('EEG window shape mismatch')

    return sig


# =========================================
# 1. LOCK PROTOCOL AND SOURCES
# =========================================

protocol = {
    'status': 'Post-hoc within-participant recording-condition transfer',
    'source': 'on004504 v1.0.0, selected 300s EC windows',
    'target': 'on006036 v1.0.0, earliest boundary-safe 30s photic window selected in milestone 3C',
    'duration_s': 30,
    'epoch_s': 2,
    'min_clean_epochs': 8,
    'spectral_features': FEATS,
    'validation': '5 repeats x 5 participant-stratified outer folds; 3-fold inner tuning on EC training participants only',
    'selection': 'same fixed 10s start grid; first clean eligible 30s within frozen EC 300s window',
    'model': 'StandardScaler + class-weighted RBF SVM; C=[0.1,1,10]; gamma=[scale,0.01,0.1]',
    'primary': 'paired EO minus EC balanced accuracy on the same EO-quality-eligible held-out participants',
    'secondary': 'EC/EO label agreement, group sensitivity/specificity, paired SVM margin shift',
    'uncertainty': 'stratified participant-cluster bootstrap, repetitions kept together; shared HC not independently resampled for cross-task claims',
    'limitations': 'Same cohort, different recording state and photic stimulus; 10 quality exclusions; exploratory'
}

encoded = json.dumps(protocol, sort_keys=True, indent=2) + '\n'
pth = OUT / 'locked_protocol.json'

if pth.exists() and pth.read_text() != encoded:
    raise RuntimeError('Protocol lock changed')

if not pth.exists():
    pth.write_text(encoded)

inputs = [M, QC, OLD_AD, OLD_FTD, OLD_FOLDS, META]

checks = {
    str(p.relative_to(ROOT)):
    hashlib.sha256(p.read_bytes()).hexdigest()
    for p in inputs
}

ctext = json.dumps(checks, sort_keys=True, indent=2) + '\n'
cl = OUT / 'source_sha256.json'

if cl.exists() and cl.read_text() != ctext:
    raise RuntimeError('Input checksums changed')

if not cl.exists():
    cl.write_text(ctext)


# =========================================
# 2. VERIFY MATCHED COHORTS
# =========================================

print(
    '===== MILESTONE 3D: MATCHED 30S CONDITION TRANSFER =====',
    flush=True
)

manifest = idx(
    read(M),
    lambda r: r['subject'],
    'manifest'
)

qc = idx(
    read(QC),
    lambda r: r['subject'],
    '30s QC'
)

selected = {
    s: r for s, r in manifest.items()
    if r['status'] == 'CANDIDATE'
    and r['group'] in ('A', 'F', 'C')
}

if (
    Counter(r['group'] for r in selected.values())
    != {'A': 23, 'F': 18, 'C': 23}
    or set(qc) != set(selected)
):
    raise RuntimeError('Original frozen cohort changed')

eo_ids = {
    s for s in selected
    if qc[s]['30s_status'] == 'ELIGIBLE_30S'
}

if (
    Counter(selected[s]['group'] for s in eo_ids)
    != {'A': 19, 'F': 13, 'C': 22}
):
    raise RuntimeError('30s EO eligibility changed')

print(
    'PASS: EC 23 AD / 18 FTD / 23 HC; '
    'EO 19 AD / 13 FTD / 22 HC',
    flush=True
)


# =========================================
# 3. EXTRACT AND VERIFY SPECTRAL FEATURES
# =========================================

oldA = idx(
    (
        r for r in read(OLD_AD)
        if int(r['duration_sec']) == 60
    ),
    lambda r: r['subject'],
    'old 60s AD features'
)

oldF = idx(
    (
        r for r in read(OLD_FTD)
        if int(r['duration_sec']) == 60
    ),
    lambda r: r['subject'],
    'old 60s FTD features'
)

ref_sid = sorted(
    s for s in selected
    if selected[s]['group'] == 'A'
)[0]

ref_file = Path(selected[ref_sid]['file'])

if not ref_file.is_absolute():
    ref_file = ROOT / ref_file

ref = mne.io.read_raw_eeglab(
    str(ref_file),
    preload=False,
    verbose='ERROR'
)

refp = mne.pick_types(ref.info, eeg=True, exclude=[])

ref_channels = {
    ref.ch_names[j].strip().lower()
    for j in refp
}

if len(refp) != 19 or len(ref_channels) != 19:
    raise RuntimeError('Bad reference montage')

EC = {}
EO = {}
feature_rows = []
audit_rows = []
max_old_diff = 0.0

for k, sid in enumerate(sorted(selected), 1):

    r = selected[sid]
    raw, picks = raw_file(Path(r['file']), ref_channels)

    sec = float(r['start_sec'])
    end = float(r['end_sec'])

    if abs(end - sec - 300) > 1e-6 or has_boundary(raw, sec, end):
        raise RuntimeError(
            sid + ': frozen EC window continuity violation'
        )

    # Reconstruct 60s EC features and compare them
    # with the original thesis results before 30s modeling.

    sig60 = get_uv(raw, picks, sec, 60)
    old_feat, n60 = spectral(sig60, 15)

    if old_feat is None:
        raise RuntimeError(
            sid + ': EC first 60s below clean-epoch criterion'
        )

    prior = oldF[sid] if r['group'] == 'F' else oldA[sid]

    if int(prior['clean_epochs']) != n60:
        raise RuntimeError(
            sid + ': EC 60s epoch count differs'
        )

    delta = max(
        abs(old_feat[f] - float(prior[f]))
        for f in FEATS
    )

    max_old_diff = max(max_old_diff, delta)

    if delta > 1e-9:
        raise RuntimeError(
            f'{sid}: EC reference 60s feature mismatch, '
            f'max delta={delta}'
        )

    # Select first eligible 30s inside existing
    # frozen EC 300s window, same QC thresholds.

    chosen = None

    for offset in range(0, 271, 10):

        if has_boundary(
            raw,
            sec + offset,
            sec + offset + 30
        ):
            continue

        candidate, nclean = spectral(
            get_uv(
                raw,
                picks,
                sec + offset,
                30
            ),
            8
        )

        if candidate is not None:
            chosen = (
                offset,
                candidate,
                nclean
            )
            break

    if chosen is None:
        raise RuntimeError(
            sid + ': no eligible EC 30s '
            'within fixed 300s; stop before training'
        )

    off, f, nclean = chosen

    EC[sid] = f

    feature_rows.append({
        'condition': 'eyes_closed',
        'subject': sid,
        'group': r['group'],
        'start_sec': sec + off,
        'duration_sec': 30,
        'clean_epochs': nclean,
        'clean_fraction': nclean / 15,
        **f
    })

    audit = {
        'subject': sid,
        'group': r['group'],
        'EC_30s_offset_in_original_window': off,
        'EC_30s_clean_epochs': nclean,
        'EC_60s_original_feature_max_diff': delta,
        'EO_eligible': int(sid in eo_ids),
        'EO_30s_start_sec': '',
        'EO_30s_clean_epochs': ''
    }

    if sid in eo_ids:

        q = qc[sid]
        eo_path = Path(q['source_file'])

        eo_raw, eo_picks = raw_file(
            eo_path,
            ref_channels
        )

        eo_start = float(
            q['30s_selected_start_sec']
        )

        if has_boundary(
            eo_raw,
            eo_start,
            eo_start + 30
        ):
            raise RuntimeError(
                sid + ': 3C EO boundary violation'
            )

        e_feat, e_n = spectral(
            get_uv(
                eo_raw,
                eo_picks,
                eo_start,
                30
            ),
            8
        )

        if (
            e_feat is None
            or e_n != int(q['30s_clean_epochs'])
        ):
            raise RuntimeError(
                sid + ': 3C EO quality decision not reproduced'
            )

        EO[sid] = e_feat

        feature_rows.append({
            'condition': 'eyes_open_photic',
            'subject': sid,
            'group': r['group'],
            'start_sec': eo_start,
            'duration_sec': 30,
            'clean_epochs': e_n,
            'clean_fraction': e_n / 15,
            **e_feat
        })

        audit['EO_30s_start_sec'] = eo_start
        audit['EO_30s_clean_epochs'] = e_n

    audit_rows.append(audit)

    if k % 10 == 0 or k == 64:
        print(
            f'Extracted/validated: {k}/64',
            flush=True
        )

if (
    len(EC) != 64
    or len(EO) != 54
    or len(feature_rows) != 118
):
    raise RuntimeError('Feature count mismatch')

save(
    'condition_30s_spectral_features.csv',
    feature_rows
)

save(
    'feature_reconstruction_quality_audit.csv',
    audit_rows
)

print(
    'PASS: all 64 EC 60s reference features reproduced '
    f'(max absolute diff={max_old_diff:.3g}).'
)

print(
    'PASS: 64 EC and 54 EO 30s feature vectors; '
    'all EO QC decisions reproduced.'
)


# =========================================
# 4. VERIFY ORIGINAL FOLDS
# =========================================

prev = idx(
    (
        r for r in read(OLD_FOLDS)
        if int(r['minute']) == 1
    ),
    lambda r: (
        r['task'],
        int(r['repeat']),
        r['subject']
    ),
    'historical outer fold'
)

if len(prev) != 5 * (46 + 41):
    raise RuntimeError(
        'Historical fold coverage mismatch'
)


# =========================================
# 5. DESCRIBE EO ELIGIBILITY BIAS
# =========================================

meta = idx(
    read(META, '\t'),
    lambda r: r['participant_id'],
    'clinical meta'
)

sel_rows = []

for group in ('A', 'F', 'C'):

    subsets = [
        (
            'EO_ELIGIBLE',
            [
                s for s in sorted(selected)
                if selected[s]['group'] == group
                and s in eo_ids
            ]
        ),
        (
            'EO_INELIGIBLE',
            [
                s for s in sorted(selected)
                if selected[s]['group'] == group
                and s not in eo_ids
            ]
        )
    ]

    for status, subset in subsets:

        ages = np.array([
            float(meta[s]['Age'])
            for s in subset
        ], dtype=float)

        mmse = np.array([
            float(meta[s]['MMSE'])
            for s in subset
        ], dtype=float)

        sel_rows.append({
            'group': group,
            'status': status,
            'n': len(subset),
            'median_age':
                float(np.median(ages))
                if len(ages) else '',
            'median_MMSE':
                float(np.median(mmse))
                if len(mmse) else ''
        })

save(
    'eo_eligibility_clinical_descriptives.csv',
    sel_rows
)


# =========================================
# 6. REPEATED NESTED EC-ONLY SVM TRAINING
# =========================================

def model():
    return Pipeline([
        ('scale', StandardScaler()),
        (
            'clf',
            SVC(
                kernel='rbf',
                class_weight='balanced'
            )
        )
    ])


def bm(y, p):
    return 0.5 * (
        (p[y == 1] == 1).mean()
        + (p[y == 0] == 0).mean()
    )


all_preds = []
all_folds = []
task_summaries = []
boot_results = []
task_plots = []

rng = np.random.default_rng(20261107)

for task, positive in [
    ('AD_vs_HC', 'A'),
    ('FTD_vs_HC', 'F')
]:

    ids = sorted(
        s for s in selected
        if selected[s]['group'] in (positive, 'C')
    )

    y = np.array([
        int(selected[s]['group'] == positive)
        for s in ids
    ], dtype=int)

    X = np.array([
        [EC[s][f] for f in FEATS]
        for s in ids
    ], dtype=float)

    matched = np.array([
        i for i, s in enumerate(ids)
        if s in eo_ids
    ], dtype=int)

    Xeo = np.array([
        [EO[ids[i]][f] for f in FEATS]
        for i in matched
    ], dtype=float)

    matched_map = {
        i: j
        for j, i in enumerate(matched)
    }

    yc = y[matched]

    expected = {
        'AD_vs_HC': {
            'n': 46,
            'm': 41,
            'pos': 19,
            'hc': 22
        },
        'FTD_vs_HC': {
            'n': 41,
            'm': 35,
            'pos': 13,
            'hc': 22
        }
    }[task]

    if (
        len(ids) != expected['n']
        or len(matched) != expected['m']
        or Counter(yc.tolist()) != {
            1: expected['pos'],
            0: expected['hc']
        }
    ):
        raise RuntimeError(
            task + ': matched sizes wrong'
        )

    pc = np.full(
        (REPEATS, len(ids)),
        -1,
        dtype=int
    )

    sc = np.full(
        (REPEATS, len(ids)),
        np.nan
    )

    po = np.full(
        (REPEATS, len(matched)),
        -1,
        dtype=int
    )

    so = np.full(
        (REPEATS, len(matched)),
        np.nan
    )

    fm = np.full(
        (REPEATS, len(ids)),
        -1,
        dtype=int
    )

    for rep in range(REPEATS):

        outer = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42 + 100 * rep
        )

        for fold, (tr, te) in enumerate(
            outer.split(X, y),
            1
        ):

            for i in te:

                if int(
                    prev[
                        (task, rep + 1, ids[i])
                    ]['fold']
                ) != fold:
                    raise RuntimeError(
                        task + f': CV fold drift '
                        f'for {ids[i]} repeat {rep+1}'
                    )

            inner = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=(
                    1000 + 100 * rep + fold
                )
            )

            search = GridSearchCV(
                model(),
                {
                    'clf__C': [0.1, 1.0, 10.0],
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

            # Fit using EC training participants only.
            search.fit(
                X[tr],
                y[tr]
            )

            fit = search.best_estimator_

            # EC test: original held-out participants.
            pc[rep, te] = fit.predict(
                X[te]
            )

            sc[rep, te] = fit.decision_function(
                X[te]
            )

            fm[rep, te] = fold

            # EO test: only eligible participants
            # from this same held-out EC fold.
            em = [
                i for i in te
                if i in matched_map
            ]

            if em:

                positions = np.array([
                    matched_map[i]
                    for i in em
                ], dtype=int)

                po[
                    rep, positions
                ] = fit.predict(
                    Xeo[positions]
                )

                so[
                    rep, positions
                ] = fit.decision_function(
                    Xeo[positions]
                )

            all_folds.append({
                'task': task,
                'repeat': rep + 1,
                'fold': fold,
                'n_train_EC': len(tr),
                'n_test_EC': len(te),
                'n_test_EO': len(em),
                'best_C':
                    search.best_params_['clf__C'],
                'best_gamma':
                    search.best_params_['clf__gamma']
            })

        print(
            f'{task}: nested EC-only CV '
            f'{rep+1}/5 completed',
            flush=True
        )

    if (
        (pc < 0).any()
        or (po < 0).any()
        or (fm < 1).any()
        or not np.isfinite(sc).all()
        or not np.isfinite(so).all()
    ):
        raise RuntimeError(
            task + ': incomplete held-out predictions'
        )

    for rep in range(REPEATS):

        for i, sid in enumerate(ids):

            j = matched_map.get(i)

            all_preds.append({
                'task': task,
                'repeat': rep + 1,
                'subject': sid,
                'group': selected[sid]['group'],
                'true_label': int(y[i]),
                'outer_fold': int(fm[rep, i]),
                'EO_eligible': int(j is not None),
                'EC_pred_30s': int(pc[rep, i]),
                'EC_margin_30s': float(sc[rep, i]),
                'EO_pred_30s':
                    int(po[rep, j])
                    if j is not None else '',
                'EO_margin_30s':
                    float(so[rep, j])
                    if j is not None else ''
            })

    # =====================================
    # 7. PAIRED CONDITION METRICS
    # =====================================

    def met(indices):

        yy = y[indices]

        if (
            not np.any(yy == 1)
            or not np.any(yy == 0)
        ):
            raise RuntimeError(
                'Empty diagnostic class '
                'in matched subset'
            )

        # All five correlated CV repeats are
        # kept together for each sampled subject.

        positions = np.array([
            matched_map[i]
            for i in indices
        ])

        c = pc[:, indices]
        e = po[:, positions]

        zc = sc[:, indices]
        ze = so[:, positions]

        pos = yy == 1
        hc = ~pos

        bc = 0.5 * (
            (c[:, pos] == 1).mean()
            + (c[:, hc] == 0).mean()
        )

        be = 0.5 * (
            (e[:, pos] == 1).mean()
            + (e[:, hc] == 0).mean()
        )

        return {
            'EC_BACC': float(bc),
            'EO_BACC': float(be),
            'EO_minus_EC_BACC': float(be - bc),
            'EC_sensitivity':
                float((c[:, pos] == 1).mean()),
            'EO_sensitivity':
                float((e[:, pos] == 1).mean()),
            'EC_specificity':
                float((c[:, hc] == 0).mean()),
            'EO_specificity':
                float((e[:, hc] == 0).mean()),
            'same_pred_fraction':
                float((c == e).mean()),
            'mean_EO_minus_EC_raw_margin':
                float((ze - zc).mean())
        }

    obs = met(matched)

    full_ec = float(
        np.mean([
            bm(y, pc[rep])
            for rep in range(REPEATS)
        ])
    )

    pos_ix = matched[y[matched] == 1]
    hc_ix = matched[y[matched] == 0]

    samples = {
        name: np.empty(NBOOT)
        for name in obs
    }

    for b in range(NBOOT):

        sample = np.r_[
            rng.choice(
                pos_ix,
                len(pos_ix),
                replace=True
            ),
            rng.choice(
                hc_ix,
                len(hc_ix),
                replace=True
            )
        ]

        values = met(sample)

        for name in samples:
            samples[name][b] = values[name]

    for name, value in obs.items():

        lo, hi = np.percentile(
            samples[name],
            [2.5, 97.5]
        )

        boot_results.append({
            'task': task,
            'metric': name,
            'estimate': value,
            'conditional_95CI_low': float(lo),
            'conditional_95CI_high': float(hi),
            'n_matched': len(matched),
            'n_positive': len(pos_ix),
            'n_HC': len(hc_ix)
        })

    rank_corr = {}

    for group, subset in [
        ('patients', pos_ix),
        ('HC', hc_ix)
    ]:

        a = sc[:, subset].mean(axis=0)

        b = so[
            :,
            np.array([
                matched_map[i]
                for i in subset
            ])
        ].mean(axis=0)

        rho, pvalue = spearmanr(a, b)

        rank_corr[group] = (
            float(rho),
            float(pvalue)
        )

    for rep in range(REPEATS):

        mt = np.array([
            matched_map[i]
            for i in matched
        ])

        task_summaries.append({
            'task': task,
            'repeat': rep + 1,
            'n_EC_full': len(ids),
            'n_EO_matched': len(matched),
            'full_EC_BACC':
                bm(y, pc[rep]),
            'matched_EC_BACC':
                bm(yc, pc[rep, matched]),
            'matched_EO_BACC':
                bm(yc, po[rep, mt]),
            'matched_label_agreement':
                float(
                    (
                        pc[rep, matched]
                        == po[rep, mt]
                    ).mean()
                )
        })

    task_plots.append((
        task,
        obs['EC_BACC'],
        obs['EO_BACC']
    ))

    print(
        f'\n===== {task}: CROSS-CONDITION RESULTS ====='
    )

    print(
        f'Full-cohort EC 30s BACC={full_ec:.4f} '
        f'({len(ids)} participants)'
    )

    print(
        f'Matched EC 30s BACC={obs["EC_BACC"]:.4f}; '
        f'EO photic 30s BACC={obs["EO_BACC"]:.4f} '
        f'({len(matched)} participants)'
    )

    print(
        f'EO minus EC BACC='
        f'{obs["EO_minus_EC_BACC"]:+.4f}'
    )

    print(
        f'EC sensitivity={obs["EC_sensitivity"]:.4f}; '
        f'EO sensitivity={obs["EO_sensitivity"]:.4f}'
    )

    print(
        f'EC specificity={obs["EC_specificity"]:.4f}; '
        f'EO specificity={obs["EO_specificity"]:.4f}'
    )

    print(
        f'Cross-condition class agreement='
        f'{obs["same_pred_fraction"]:.4f}'
    )

    print(
        f'Mean paired raw SVM-margin change='
        f'{obs["mean_EO_minus_EC_raw_margin"]:+.4f}'
    )

    print(
        f'Within-patient score-rank rho='
        f'{rank_corr["patients"][0]:+.3f}; '
        f'within-HC rho='
        f'{rank_corr["HC"][0]:+.3f} '
        '(descriptive)'
    )

    for name in [
        'EO_minus_EC_BACC',
        'same_pred_fraction',
        'EO_sensitivity',
        'EO_specificity',
        'mean_EO_minus_EC_raw_margin'
    ]:

        lo, hi = np.percentile(
            samples[name],
            [2.5, 97.5]
        )

        print(
            f'  {name}: {obs[name]:+.4f}; '
            f'conditional 95% CI '
            f'[{lo:+.4f}, {hi:+.4f}]'
        )


# =========================================
# 8. SAVE REPRODUCIBLE RESULTS AND FIGURE
# =========================================

save(
    'held_out_paired_condition_predictions.csv',
    all_preds
)

save(
    'ec_only_fold_selection_audit.csv',
    all_folds
)

save(
    'paired_condition_cv_repeat_metrics.csv',
    task_summaries
)

save(
    'paired_condition_subject_bootstrap_intervals.csv',
    boot_results
)

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

xx = np.arange(
    len(task_plots)
)

width = 0.34

ax.bar(
    xx - width / 2,
    [t[1] for t in task_plots],
    width,
    label='EC matched 30s'
)

ax.bar(
    xx + width / 2,
    [t[2] for t in task_plots],
    width,
    label='EO photic matched 30s'
)

ax.set(
    xticks=xx,
    xticklabels=[
        t[0] for t in task_plots
    ],
    ylim=(0, 1),
    ylabel='Participant-level balanced accuracy',
    title='Same held-out participants, EC vs EO/photic'
)

ax.legend()
fig.tight_layout()

fig.savefig(
    OUT / 'matched_condition_bacc.png',
    dpi=180
)

plt.close(fig)


# =========================================
# 9. ELIGIBILITY DESCRIPTION
# =========================================

print(
    '\n===== EO ELIGIBILITY AND CLINICAL DESCRIPTIVES ====='
)

for row in sel_rows:

    print(
        f'{row["group"]} {row["status"]}: '
        f'n={row["n"]}, '
        f'median age={row["median_age"]}, '
        f'median EC-MMSE={row["median_MMSE"]}'
    )


# =========================================
# 10. COMPLETE
# =========================================

print(
    '\n===== MILESTONE 3D COMPLETE ====='
)

print(
    'Outputs:',
    OUT
)

print(
    'All training, model selection, and preprocessing '
    'decisions used EC only; '
    'no EO labels were used for fitting.'
)

print(
    'This is within-cohort condition transfer, '
    'NOT independent external-patient validation.'
)

