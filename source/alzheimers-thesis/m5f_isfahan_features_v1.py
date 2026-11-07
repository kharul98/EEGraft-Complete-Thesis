#!/usr/bin/env python3
"""M5F: fixed five-minute EEG band features with frozen QC parity; no model fits."""

from pathlib import Path
from collections import Counter
from io import StringIO
import csv
import hashlib
import json
import re
import sys

import mne
import numpy as np
import scipy
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch


ROOT = Path.cwd()
BASE = ROOT / 'reports/thesis_final'
SRC = BASE / 'milestone_5d_isfahan_labels_and_qc_v2'
OUT = BASE / 'milestone_5f_isfahan_spectral_features_v1'
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = SRC / 'release_aware_verified_subject_manifest.csv'
QC = SRC / 'first_five_minute_quality.csv'

CHANS = (
    'Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 '
    'T5 P3 Pz P4 T6 O1 O2'
).split()

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

EXPECTED = {
    ('EEG_1', 'MCI'): 11,
    ('EEG_1', 'NORMAL'): 14,
    ('EEG_2', 'MCI'): 12,
    ('EEG_2', 'NORMAL'): 12,
}

FS = 256
NEPOCH = 150
NSAMP = 512

SOS = butter(
    4,
    [0.5, 45],
    btype='bandpass',
    fs=FS,
    output='sos'
)


# ==========================================
# 1. HELPERS
# ==========================================

def sha(path):

    h = hashlib.sha256()

    with path.open('rb') as f:

        for part in iter(
            lambda: f.read(1024 * 1024),
            b''
        ):
            h.update(part)

    return h.hexdigest()


def read(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:
        return list(csv.DictReader(f))


def freeze(path, value):

    if (
        path.exists()
        and path.read_text(encoding='utf-8') != value
    ):
        raise RuntimeError(
            'Frozen output changed: ' + str(path)
        )

    if not path.exists():

        path.write_text(
            value,
            encoding='utf-8'
        )


def freeze_csv(path, rows):

    if not rows:
        raise RuntimeError(
            'Empty result: ' + str(path)
        )

    fields = list(dict.fromkeys(
        k for r in rows for k in r
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


# ==========================================
# 2. VALIDATE FROZEN COHORT
# ==========================================

print(
    '===== MILESTONE 5F: '
    'MCI FIVE-MINUTE SPECTRAL FEATURE AUDIT =====',
    flush=True
)

manifest = read(MANIFEST)
qcs = read(QC)

if len(manifest) != 61 or len(qcs) != 61:
    raise RuntimeError(
        'M5D-v2 cohort has changed'
    )

m_by_id = {
    r['subject_id']: r
    for r in manifest
}

q_by_id = {
    r['subject_id']: r
    for r in qcs
}

if (
    len(m_by_id) != 61
    or len(q_by_id) != 61
    or set(m_by_id) != set(q_by_id)
):
    raise RuntimeError(
        'Missing or duplicated subject IDs'
    )

if Counter(
    r['status'] for r in qcs
) != {
    'ELIGIBLE_FIXED_300S': 49,
    'INSUFFICIENT_CLEAN_EPOCHS': 12
}:
    raise RuntimeError(
        'M5D-v2 quality-status counts have changed'
    )

eligible = sorted(
    sid for sid, r in q_by_id.items()
    if r['status'] == 'ELIGIBLE_FIXED_300S'
)

counts = Counter(
    (
        m_by_id[s]['release'],
        m_by_id[s]['diagnosis']
    )
    for s in eligible
)

if dict(counts) != EXPECTED:
    raise RuntimeError(
        f'Eligibility-group counts changed: {dict(counts)}'
    )

for sid in m_by_id:

    if (
        m_by_id[sid]['diagnosis']
        != q_by_id[sid]['diagnosis']
        or m_by_id[sid]['release']
        != q_by_id[sid]['release']
    ):
        raise RuntimeError(
            'Manifest/quality label mismatch: ' + sid
        )


# ==========================================
# 3. LOCK EXTRACTION PROTOCOL
# ==========================================

protocol = {
    'version': 'm5f_v1',

    'n_eligible': 49,

    'unit_assumption':
        'Blank EDF physical dimensions treated '
        'as uV; MNE units="uV", then returned '
        'SI volts converted to uV for QC',

    'filter':
        'Butterworth 4th order 0.5-45 Hz, '
        'SOS zero-phase, full fixed first '
        '300 seconds',

    'epochs':
        '2s, nonoverlapping; same three '
        'artifact rules as M5D-v2',

    'quality_gate':
        'At least 15 clean epochs in each of '
        'five fixed minutes; exact equality '
        'with frozen M5D-v2 clean counts',

    'PSD':
        'scipy.signal.welch on each clean '
        '2s epoch and 19 channels; Hann; '
        'nperseg=512; noverlap=0; '
        'constant detrend; density scaling; '
        'mean PSD across clean epochs and '
        'channels within each minute',

    'bands_hz': BANDS,

    'integration':
        'scipy.integrate.trapezoid, '
        'endpoints shared; relative to '
        '1-45 Hz total power',

    'ratio':
        'theta relative power / alpha relative power',

    'unit_of_analysis':
        'recording within release; '
        'cross-release identity not established',

    'no_models_fitted': True,

    'legacy_parity_status':
        'NOT YET VERIFIED; compare original '
        'feature extraction before modeling',
}

checks = {
    'manifest_sha256': sha(MANIFEST),
    'frozen_quality_sha256': sha(QC),
    'mne_version': mne.__version__,
    'scipy_version': scipy.__version__,
    'numpy_version': np.__version__,
}

freeze(
    OUT / 'locked_protocol.json',
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True
    ) + '\n'
)

freeze(
    OUT / 'input_checksums.json',
    json.dumps(
        checks,
        indent=2,
        sort_keys=True
    ) + '\n'
)

print(
    'PASS: frozen cohort 49/61, with '
    'EEG_1 (11 MCI, 14 HC), '
    'EEG_2 (12 MCI, 12 HC)'
)

print(
    'PASS: explicit uV assumption and '
    'original five-minute QC rules retained'
)


# ==========================================
# 4. EXTRACT FIVE ONE-MINUTE FEATURE VECTORS
# ==========================================

feature_rows = []
subject_rows = []

for number, sid in enumerate(eligible, 1):

    mr = m_by_id[sid]
    qr = q_by_id[sid]

    path = ROOT / mr['edf_path']

    if (
        not path.is_file()
        or path.stat().st_size
           != int(mr['edf_bytes'])
    ):
        raise RuntimeError(
            'EDF source missing or changed size: ' + sid
        )

    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        units='uV',
        verbose='ERROR'
    )

    try:

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        if (
            len(picks) != 19
            or [raw.ch_names[k] for k in picks] != CHANS
            or abs(
                float(raw.info['sfreq']) - FS
            ) > 1e-8
            or raw.n_times < 300 * FS
        ):
            raise RuntimeError(
                'EDF channel/rate/duration drift: ' + sid
            )

        x = raw.get_data(
            picks=picks,
            start=0,
            stop=300 * FS
        ) * 1e6

    finally:
        raw.close()

    if (
        x.shape != (19, 300 * FS)
        or not np.isfinite(x).all()
    ):
        raise RuntimeError(
            'Invalid signal array: ' + sid
        )

    filtered = sosfiltfilt(
        SOS,
        x,
        axis=-1
    )

    epochs = filtered.reshape(
        19,
        NEPOCH,
        NSAMP
    ).transpose(
        1, 0, 2
    )

    ptp = np.ptp(
        epochs,
        axis=2
    )

    std = np.std(
        epochs,
        axis=2
    )

    keep = ~(
        ((ptp > 200).sum(axis=1) >= 3)
        | (ptp > 300).any(axis=1)
        | (std < 0.5).any(axis=1)
    )

    per_minute = (
        keep.reshape(5, 30)
        .sum(axis=1)
    )

    frozen = np.array([
        int(v)
        for v in qr['clean_epochs_each_minute'].split('|')
    ])

    if (
        per_minute.shape != (5,)
        or not np.array_equal(per_minute, frozen)
    ):
        raise RuntimeError(
            f'Signal QC mismatch: {sid}, '
            f'recalculated={per_minute.tolist()}, '
            f'frozen={frozen.tolist()}'
        )

    if np.any(per_minute < 15):
        raise RuntimeError(
            'Previously eligible recording now fails: ' + sid
        )

    subject_rows.append({
        'subject_id': sid,
        'release': mr['release'],
        'diagnosis': mr['diagnosis'],
        'n_minutes': 5,
        'clean_epochs_5_minutes':
            int(per_minute.sum()),
        'clean_epochs_by_minute':
            '|'.join(map(str, per_minute)),
        'edf_path': mr['edf_path']
    })

    for minute in range(5):

        block = epochs[
            minute * 30:
            (minute + 1) * 30
        ]

        accepted = block[
            keep[
                minute * 30:
                (minute + 1) * 30
            ]
        ]

        frequencies, power = welch(
            accepted,
            fs=FS,
            window='hann',
            nperseg=NSAMP,
            noverlap=0,
            detrend='constant',
            scaling='density',
            axis=-1
        )

        if power.shape != (
            len(accepted),
            19,
            NSAMP // 2 + 1
        ):
            raise RuntimeError(
                'Welch PSD dimensions unexpected: ' + sid
            )

        pooled = power.mean(
            axis=(0, 1)
        )

        total_mask = (
            (frequencies >= 1)
            & (frequencies <= 45)
        )

        total = float(
            trapezoid(
                pooled[total_mask],
                frequencies[total_mask]
            )
        )

        if (
            not np.isfinite(total)
            or total <= 0
        ):
            raise RuntimeError(
                'Nonpositive spectral total: ' + sid
            )

        vals = {}

        for name, (lo, hi) in BANDS.items():

            mask = (
                (frequencies >= lo)
                & (frequencies <= hi)
            )

            vals[name] = float(
                trapezoid(
                    pooled[mask],
                    frequencies[mask]
                ) / total
            )

        if (
            abs(sum(vals.values()) - 1) > 1e-8
            or any(
                not np.isfinite(v) or v < 0
                for v in vals.values()
            )
        ):
            raise RuntimeError(
                'Spectral partition failed: ' + sid
            )

        if vals['alpha'] <= 1e-12:
            raise RuntimeError(
                'Alpha fraction too small for ratio: ' + sid
            )

        vals['theta_alpha_ratio'] = (
            vals['theta'] / vals['alpha']
        )

        feature_rows.append({
            'subject_id': sid,
            'release': mr['release'],
            'diagnosis': mr['diagnosis'],
            'true_label_MCI':
                int(mr['diagnosis'] == 'MCI'),
            'minute': minute + 1,
            'clean_epochs':
                int(per_minute[minute]),
            **vals
        })

    if (
        number <= 3
        or number % 10 == 0
        or number == len(eligible)
    ):
        print(
            f'Features audited: '
            f'{number}/{len(eligible)}; '
            f'{sid}; clean epochs '
            f'{per_minute.tolist()}',
            flush=True
        )


# ==========================================
# 5. SAVE FEATURES AND DESCRIPTIVES
# ==========================================

if (
    len(feature_rows) != 49 * 5
    or len(subject_rows) != 49
):
    raise RuntimeError(
        'Incomplete feature output'
    )

freeze_csv(
    OUT / 'five_minute_spectral_features.csv',
    feature_rows
)

freeze_csv(
    OUT / 'eligible_recording_inventory.csv',
    subject_rows
)

summary = []

for (
    release,
    diagnosis
), expected in sorted(EXPECTED.items()):

    people = [
        r for r in subject_rows
        if (
            r['release'] == release
            and r['diagnosis'] == diagnosis
        )
    ]

    rows = [
        r for r in feature_rows
        if (
            r['release'] == release
            and r['diagnosis'] == diagnosis
            and r['minute'] == 1
        )
    ]

    if (
        len(people) != expected
        or len(rows) != expected
    ):
        raise RuntimeError(
            'Per-release subject count changed'
        )

    result = {
        'release': release,
        'diagnosis': diagnosis,
        'n_eligible': expected,

        'median_five_minute_clean_epoch_count':
            float(np.median([
                r['clean_epochs_5_minutes']
                for r in people
            ]))
    }

    for band in list(BANDS) + [
        'theta_alpha_ratio'
    ]:
        result[
            'median_first_minute_' + band
        ] = float(np.median([
            r[band]
            for r in rows
        ]))

    summary.append(result)

freeze_csv(
    OUT / 'first_minute_descriptives_by_release_and_diagnosis.csv',
    summary
)

print(
    '\n===== FIXED FIVE-MINUTE '
    'FEATURE EXTRACTION ====='
)

for r in summary:

    print(
        f"{r['release']} {r['diagnosis']}: "
        f"n={r['n_eligible']}, "
        'median clean 2s epochs / 5min='
        f"{r['median_five_minute_clean_epoch_count']:.1f}, "
        "first-minute theta="
        f"{r['median_first_minute_theta']:.4f}, "
        "alpha="
        f"{r['median_first_minute_alpha']:.4f}, "
        "theta/alpha="
        f"{r['median_first_minute_theta_alpha_ratio']:.3f}"
    )


# ==========================================
# 6. INSPECT OUR ORIGINAL FEATURE RECIPE
# ==========================================

print(
    '\n===== LEGACY SPECTRAL IMPLEMENTATION '
    'COMPARISON (SOURCE SNIPPETS) ====='
)

legacy = ROOT / 'm3d_condition_transfer_v1.py'

if legacy.is_file():

    lines = legacy.read_text(
        encoding='utf-8'
    ).splitlines()

    patterns = (
        r'\bwelch\(',
        r'\btrapezoid\(',
        r'\btrapz\(',
        r'\brelative\b',
        r'\btheta_alpha_ratio\b',
        r'\bbands\b'
    )

    hits = [
        i
        for i, text in enumerate(lines)
        if any(
            re.search(pattern, text, re.I)
            for pattern in patterns
        )
    ]

    intervals = []

    for i in hits:

        a = max(0, i - 7)
        b = min(len(lines), i + 10)

        if intervals and a <= intervals[-1][1]:

            intervals[-1] = (
                intervals[-1][0],
                max(
                    b,
                    intervals[-1][1]
                )
            )

        else:
            intervals.append((a, b))

    shown = 0

    for a, b in intervals:

        if shown >= 95:
            print(
                '... further snippets omitted; '
                'inspect local m3d script if necessary'
            )
            break

        print(
            f'--- {legacy.name} '
            f'lines {a+1}-{b} ---'
        )

        for j in range(
            a,
            min(b, a + 95 - shown)
        ):

            print(
                f'{j+1:>4}: {lines[j]}'
            )

            shown += 1

else:

    print(
        'Original m3d_condition_transfer_v1.py '
        'not found; source parity must be '
        'checked before training.'
    )


# ==========================================
# 7. FINISH
# ==========================================

print(
    '\n===== MILESTONE 5F '
    'FEATURE PREFLIGHT COMPLETE ====='
)

print(
    '49 release-aware recordings; '
    '245 minute-level feature vectors; '
    'exact five-minute QC parity.'
)

print(
    'No SVM fitted, no combined-cohort CV, '
    'and no cross-release independence assumed.'
)

print(
    'Legacy feature-extraction parity '
    'is NOT YET asserted; review snippets '
    'before modeling.'
)

print(
    'Saved:',
    OUT
)

