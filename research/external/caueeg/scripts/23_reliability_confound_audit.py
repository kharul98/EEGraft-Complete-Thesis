from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os

import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

HOME = Path.home()

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

ANALYSIS = DATA / "derived/caueeg_transfer_v1/analysis"

PLAN_FILE = WORK / "protocol/caueeg_final_test_plan_v1.json"
AUDIT21 = WORK / "results/21_event_alignment_summary.json"
REPORT22 = WORK / "results/22_condition_analysis.json"

WINDOWS21 = ANALYSIS / "21_event_aligned_windows.csv"
RECORDS21 = ANALYSIS / "21_event_alignment_records.csv"
PAIRED22 = ANALYSIS / "22_condition_paired_recordings.csv"

OUTPUT = WORK / "results/23_reliability_confound_audit.json"

MODELS = {
    "DSRF": "dsrf_source_ad_score",
    "XGBoost": "xgb_source_ad_score"
}

EXPECTED = {
    "train": 404,
    "validation": 42
}

SEED = 20261109
BOOTSTRAPS = 5000
MAX_MATCH_GAP_SECONDS = 60


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4194304), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing result matches:", path.name)
            return True

        print("[PRESERVED] Different result already exists:", path)
        return False

    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)

    print("[SAVED]", path)
    return True


def verify():
    print("\n1. VERIFYING RESEARCH PROVENANCE")

    for path in (
        PLAN_FILE, AUDIT21, REPORT22,
        WINDOWS21, RECORDS21, PAIRED22
    ):
        require(path.is_file(), "Missing: " + str(path))

    plan = read_json(PLAN_FILE)
    audit = read_json(AUDIT21)
    previous = read_json(REPORT22)

    require(
        plan["version"] == "caueeg_final_test_plan_v1",
        "Unexpected final test plan"
    )

    require(
        audit["status"] == "EVENT_ALIGNMENT_AUDIT_COMPLETED",
        "Code 21 incomplete"
    )

    require(
        previous["status"] == "CONDITION_ANALYSIS_COMPLETED",
        "Code 22 incomplete"
    )

    require(
        previous["provenance"]["final_test_plan_sha256"]
        == sha256(PLAN_FILE),
        "Frozen test plan changed"
    )

    for key, path in (
        ("code21_summary_sha256", AUDIT21),
        ("code21_record_csv_sha256", RECORDS21),
        ("code21_window_csv_sha256", WINDOWS21)
    ):
        require(
            previous["provenance"][key] == sha256(path),
            "Previous audit file changed: " + key
        )

    require(
        previous["test_eeg_opened"] == 0
        and previous["test_event_files_opened"] == 0
        and previous["test_predictions_opened"] == 0,
        "Prior condition-analysis provenance invalid"
    )

    require(
        audit["original_model_hashes"] == plan["model_hashes"],
        "Original model provenance mismatch"
    )

    variants = plan["secondary_exploratory_variants"]

    thresholds = variants["training_selected_thresholds"]

    require(
        float(thresholds["DSRF"]) == 0.5
        and float(thresholds["XGBoost"]) == 0.7,
        "Original training-selected thresholds changed"
    )

    require(
        variants["report_all_variants_without_test_selection"]
        is True,
        "Frozen variant-selection policy changed"
    )

    require(
        previous["analysis_type"] == "post_hoc_exploratory",
        "Code 22 exploratory designation changed"
    )

    print("[OK] Original test plan unchanged")
    print("[OK] Code 21 event audit verified")
    print("[OK] Code 22 analysis verified")
    print("[OK] Previously selected threshold 0.70 verified")
    print("[OK] No model training or new test analysis required")

    return plan, previous


def load_cases(previous):
    print("\n2. RECONSTRUCTING PAIRED CONDITION CASES")

    paired = read_csv(PAIRED22)
    windows = read_csv(WINDOWS21)

    require(
        len(paired) == 446,
        "Expected 446 paired development recordings"
    )

    require(
        len(windows) == 24768,
        "Expected 24768 development EEG windows"
    )

    window_lookup = defaultdict(list)

    for row in windows:
        split = row["split"]

        require(
            split in EXPECTED,
            "A nondevelopment partition was encountered"
        )

        condition = row["condition"]

        if condition not in ("eyes_open", "eyes_closed"):
            continue

        case_key = (split, row["serial"])

        item = {
            "condition": condition,
            "time_seconds": float(
                row["window_start_seconds"]
            ) + 2.0
        }

        for model, column in MODELS.items():
            item[model] = float(row[column])

            require(
                np.isfinite(item[model])
                and 0 <= item[model] <= 1,
                "Invalid saved model score"
            )

        window_lookup[case_key].append(item)

    result = {
        "train": [],
        "validation": []
    }

    seen = set()

    for row in paired:
        split = row["split"]
        serial = row["serial"]

        key = (split, serial)

        require(
            split in EXPECTED and key not in seen,
            "Invalid or duplicate paired recording"
        )

        seen.add(key)

        eligible = window_lookup[key]

        opened = [
            w for w in eligible
            if w["condition"] == "eyes_open"
        ]

        closed = [
            w for w in eligible
            if w["condition"] == "eyes_closed"
        ]

        require(
            len(opened) >= 2
            and len(closed) >= 2,
            "Insufficient paired eye-state windows"
        )

        require(
            len(opened) == int(row["n_open"])
            and len(closed) == int(row["n_closed"]),
            "Condition coverage changed"
        )

        case = {
            "split": split,
            "serial": serial,
            "class_name": row["class_name"],
            "windows": eligible
        }

        for model in MODELS:
            open_score = float(
                np.mean([w[model] for w in opened])
            )

            closed_score = float(
                np.mean([w[model] for w in closed])
            )

            require(
                np.isclose(
                    open_score,
                    float(row[model + "_open_score"]),
                    atol=1e-10
                ),
                "Eyes-open score does not match Code 22"
            )

            require(
                np.isclose(
                    closed_score,
                    float(row[model + "_closed_score"]),
                    atol=1e-10
                ),
                "Eyes-closed score does not match Code 22"
            )

            case[model] = {
                "open_score": open_score,
                "closed_score": closed_score
            }

        result[split].append(case)

    for split, expected in EXPECTED.items():
        require(
            len(result[split]) == expected,
            "Paired cohort count changed"
        )

        require(
            Counter(
                case["class_name"]
                for case in result[split]
            ) == Counter(
                previous[
                    "training" if split == "train"
                    else "validation"
                ]["class_counts"]
            ),
            "Diagnostic cohort composition changed"
        )

    print("[OK] Training paired cases:", len(result["train"]))
    print("[OK] Validation paired cases:",
          len(result["validation"]))
    print("[OK] Scores independently reconstructed")
    print("Test prediction files read: 0")

    return result


def calibrate(score, parameters):
    clipped = float(
        np.clip(score, 1e-6, 1 - 1e-6)
    )

    logit = np.log(
        clipped / (1 - clipped)
    )

    return float(expit(
        parameters["coefficient"] * logit
        + parameters["intercept"]
    ))


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, len(values),
        size=(BOOTSTRAPS, len(values))
    )

    estimates = values[indices].mean(axis=1)

    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975))
    ]


def variant_flip(case, model, threshold, params=None):
    opened = case[model]["open_score"]
    closed = case[model]["closed_score"]

    if params is not None:
        opened = calibrate(opened, params)
        closed = calibrate(closed, params)

    return int(
        (opened >= threshold)
        != (closed >= threshold)
    )


def threshold_sensitivity(cases, plan, split):
    print("\n3. THRESHOLD SENSITIVITY:", split.upper())

    variants = plan["secondary_exploratory_variants"]
    thresholds = variants["training_selected_thresholds"]
    calibration = variants["platt_calibration_parameters"]

    configurations = {
        "DSRF_original": ("DSRF", 0.5, None),

        "DSRF_training_threshold": (
            "DSRF",
            float(thresholds["DSRF"]),
            None
        ),

        "DSRF_calibrated": (
            "DSRF", 0.5, calibration["DSRF"]
        ),

        "XGBoost_original": (
            "XGBoost", 0.5, None
        ),

        "XGBoost_training_threshold": (
            "XGBoost",
            float(thresholds["XGBoost"]),
            None
        ),

        "XGBoost_calibrated": (
            "XGBoost",
            0.5,
            calibration["XGBoost"]
        )
    }

    flip_arrays = {}
    results = {}

    for name, (model, cutoff, params) in configurations.items():
        flips = np.asarray([
            variant_flip(c, model, cutoff, params)
            for c in cases
        ], dtype=float)

        flip_arrays[name] = flips

        results[name] = {
            "threshold": cutoff,
            "n_recordings": len(cases),
            "condition_flip_count": int(flips.sum()),
            "condition_flip_fraction": float(flips.mean()),
            "recording_bootstrap_95ci": bootstrap_ci(
                flips, SEED + (0 if split == "train" else 1)
            )
        }

        print(
            f"{name}: {int(flips.sum())}/{len(cases)} "
            f"({flips.mean():.2%})"
        )

    comparisons = {}

    for name, first, second in (
        (
            "original_DSRF_minus_original_XGBoost",
            "DSRF_original",
            "XGBoost_original"
        ),
        (
            "original_DSRF_minus_threshold_XGBoost",
            "DSRF_original",
            "XGBoost_training_threshold"
        ),
        (
            "calibrated_DSRF_minus_calibrated_XGBoost",
            "DSRF_calibrated",
            "XGBoost_calibrated"
        )
    ):
        differences = (
            flip_arrays[first] - flip_arrays[second]
        )

        comparisons[name] = {
            "difference_in_flip_fraction": float(
                differences.mean()
            ),

            "recording_bootstrap_95ci": bootstrap_ci(
                differences,
                SEED + (0 if split == "train" else 1)
            ),

            "first_only_flip_count": int(
                np.sum(
                    (flip_arrays[first] == 1)
                    & (flip_arrays[second] == 0)
                )
            ),

            "second_only_flip_count": int(
                np.sum(
                    (flip_arrays[first] == 0)
                    & (flip_arrays[second] == 1)
                )
            )
        }

        print(
            "Paired difference:",
            name,
            f"{differences.mean():+.2%}"
        )

    return {
        "variants": results,
        "paired_comparisons": comparisons
    }


def time_adjusted_model(cases, model):
    """
    Recording fixed-effects linear regression:

        score_it = alpha_i
                   + beta_eye * EyesOpen_it
                   + beta_time * TimeMinutes_it
                   + error_it

    Uses one cluster per recording.

    This adjusts only for a linear time trend.
    It does NOT establish a causal eye-state effect.
    """

    matrices = []
    outcomes = []

    for case in cases:
        windows = case["windows"]

        open_indicator = np.asarray([
            int(w["condition"] == "eyes_open")
            for w in windows
        ], dtype=float)

        time_minutes = np.asarray([
            w["time_seconds"] / 60.0
            for w in windows
        ], dtype=float)

        y = np.asarray([
            w[model] for w in windows
        ], dtype=float)

        x = np.column_stack([
            open_indicator,
            time_minutes
        ])

        # Eliminate recording-specific intercepts.
        x = x - x.mean(axis=0)
        y = y - y.mean()

        matrices.append(x)
        outcomes.append(y)

    pooled_x = np.concatenate(matrices)
    pooled_y = np.concatenate(outcomes)

    require(
        np.linalg.matrix_rank(pooled_x) == 2,
        "Eye state and elapsed time cannot be separated"
    )

    xtx_inverse = np.linalg.inv(
        pooled_x.T @ pooled_x
    )

    beta = (
        xtx_inverse @ pooled_x.T @ pooled_y
    )

    cluster_meat = np.zeros((2, 2))

    for x, y in zip(matrices, outcomes):
        residual = y - x @ beta
        score = x.T @ residual
        cluster_meat += np.outer(score, score)

    n_clusters = len(matrices)
    n_observations = len(pooled_y)

    correction = (
        n_clusters / (n_clusters - 1)
    ) * (
        (n_observations - 1)
        / (n_observations - 2)
    )

    covariance = (
        correction
        * xtx_inverse
        @ cluster_meat
        @ xtx_inverse
    )

    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    critical = float(
        student_t.ppf(
            0.975, df=n_clusters - 1
        )
    )

    pvalues = []

    for coefficient, se in zip(beta, standard_errors):
        if se > 0:
            statistic = abs(coefficient / se)

            p = float(
                2 * student_t.sf(
                    statistic, df=n_clusters - 1
                )
            )
        else:
            p = None

        pvalues.append(p)

    correlation = float(
        np.corrcoef(
            pooled_x[:, 0],
            pooled_x[:, 1]
        )[0, 1]
    )

    return {
        "recordings": n_clusters,
        "eligible_windows": n_observations,

        "eye_state_coefficient_adjusted_for_linear_time":
            float(beta[0]),

        "eye_state_coefficient_95ci": [
            float(beta[0] - critical * standard_errors[0]),
            float(beta[0] + critical * standard_errors[0])
        ],

        "eye_state_exploratory_p": pvalues[0],

        "time_coefficient_per_minute": float(beta[1]),

        "time_coefficient_95ci": [
            float(beta[1] - critical * standard_errors[1]),
            float(beta[1] + critical * standard_errors[1])
        ],

        "time_exploratory_p": pvalues[1],

        "within_record_eye_state_time_correlation":
            correlation,

        "within_record_state_time_vif": float(
            1 / (1 - correlation ** 2)
        ),

        "method": (
            "Linear recording fixed effects, "
            "recording-clustered sandwich covariance"
        ),

        "interpretation": (
            "Exploratory time-adjusted association only. "
            "Nonlinear time trends, vigilance, artifacts, "
            "and other recording-condition factors "
            "remain potential confounders."
        )
    }


def closest_pairs(cases, model):
    """
    Greedy one-to-one matches between opposite eye states.
    Only pairs <= 60 seconds apart are retained.
    Results are aggregated once per recording.
    """

    paired_results = []

    for case in cases:
        opened = [
            w for w in case["windows"]
            if w["condition"] == "eyes_open"
        ]

        closed = [
            w for w in case["windows"]
            if w["condition"] == "eyes_closed"
        ]

        candidates = []

        for i, a in enumerate(opened):
            for j, b in enumerate(closed):
                distance = abs(
                    a["time_seconds"] - b["time_seconds"]
                )

                if distance <= MAX_MATCH_GAP_SECONDS:
                    candidates.append(
                        (distance, i, j)
                    )

        candidates.sort()

        used_open = set()
        used_closed = set()
        matches = []

        for distance, i, j in candidates:
            if i in used_open or j in used_closed:
                continue

            used_open.add(i)
            used_closed.add(j)

            matches.append((
                opened[i], closed[j], distance
            ))

        if not matches:
            continue

        differences = [
            a[model] - b[model]
            for a, b, _ in matches
        ]

        label_flips = [
            int(
                (a[model] >= 0.5)
                != (b[model] >= 0.5)
            )
            for a, b, _ in matches
        ]

        paired_results.append({
            "mean_score_difference": float(
                np.mean(differences)
            ),
            "mean_absolute_difference": float(
                np.mean(np.abs(differences))
            ),
            "mean_window_label_flip": float(
                np.mean(label_flips)
            ),
            "mean_time_separation": float(
                np.mean([
                    d for _, _, d in matches
                ])
            ),
            "n_matched_window_pairs": len(matches)
        })

    n = len(paired_results)

    if n == 0:
        return {
            "status": "no_close_pairs"
        }

    score_deltas = [
        r["mean_score_difference"]
        for r in paired_results
    ]

    return {
        "status": "computed",
        "maximum_pair_separation_seconds":
            MAX_MATCH_GAP_SECONDS,

        "recordings_with_close_opposite_state_pairs": n,

        "total_matched_window_pairs": sum(
            r["n_matched_window_pairs"]
            for r in paired_results
        ),

        "median_pair_time_separation_seconds": float(
            np.median([
                r["mean_time_separation"]
                for r in paired_results
            ])
        ),

        "mean_recording_score_difference": float(
            np.mean(score_deltas)
        ),

        "recording_bootstrap_95ci": (
            bootstrap_ci(score_deltas, SEED + 23)
            if n >= 10 else None
        ),

        "mean_recording_absolute_score_difference": float(
            np.mean([
                r["mean_absolute_difference"]
                for r in paired_results
            ])
        ),

        "mean_recording_window_label_flip_fraction": float(
            np.mean([
                r["mean_window_label_flip"]
                for r in paired_results
            ])
        ),

        "warning": (
            "Nearby opposite-condition windows may still "
            "differ in elapsed time, alertness and artifacts. "
            "Matching is descriptive, not causal."
        )
    }


def main():
    plan, previous = verify()

    cases = load_cases(previous)

    results = {}

    for split in ("train", "validation"):
        results[split] = {}

        results[split]["threshold_sensitivity"] = (
            threshold_sensitivity(
                cases[split],
                plan,
                split
            )
        )

        print(
            "\n4. WITHIN-RECORDING LINEAR TIME ADJUSTMENT:",
            split.upper()
        )

        time_adjustment = {}
        close_pairs = {}

        for model in MODELS:
            fit = time_adjusted_model(
                cases[split], model
            )

            time_adjustment[model] = fit

            matched = closest_pairs(
                cases[split], model
            )

            close_pairs[model] = matched

            print("\nModel:", model)

            print(
                "  Time-adjusted eyes-open coefficient:",
                f"{fit['eye_state_coefficient_adjusted_for_linear_time']:+.5f}"
            )

            print(
                "  Exploratory 95% interval:",
                fit["eye_state_coefficient_95ci"]
            )

            print(
                "  Elapsed-time slope per minute:",
                f"{fit['time_coefficient_per_minute']:+.5f}"
            )

            print(
                "  State/time correlation:",
                f"{fit['within_record_eye_state_time_correlation']:+.4f}"
            )

            print(
                "  Close-pair eligible recordings:",
                matched.get(
                    "recordings_with_close_opposite_state_pairs", 0
                )
            )

            if matched["status"] == "computed":
                print(
                    "  Close-pair mean score shift:",
                    f"{matched['mean_recording_score_difference']:+.5f}"
                )

        results[split]["time_adjustment"] = time_adjustment
        results[split]["close_pair_sensitivity"] = close_pairs

    print("\n5. SAVING POST-HOC CONFOUNDING AUDIT")

    report = {
        "status": "RELIABILITY_CONFOUND_AUDIT_COMPLETED",

        "version": "caueeg_reliability_confound_v1",

        "analysis_type": "post_hoc_exploratory",

        "cohorts": EXPECTED,

        "results": results,

        "controls": {
            "original_threshold": 0.5,
            "training_selected_xgboost_threshold": 0.7,
            "platt_parameters_from_frozen_plan": True,
            "recording_fixed_effects": True,
            "linear_elapsed_time_covariate": True,
            "cluster_unit": "recording",
            "nearby_opposite_state_window_matching": True,
            "maximum_nearby_time_gap_seconds": 60,
            "recording_bootstrap_resamples": BOOTSTRAPS,
            "random_seed": SEED
        },

        "provenance": {
            "plan_sha256": sha256(PLAN_FILE),
            "code21_windows_sha256": sha256(WINDOWS21),
            "code22_paired_sha256": sha256(PAIRED22),
            "code22_report_sha256": sha256(REPORT22)
        },

        "test_waveforms_accessed": 0,
        "test_events_accessed": 0,
        "test_predictions_accessed": 0,

        "original_models_modified": False,
        "new_hyperparameters_selected": False,

        "limitations": [
            "The research question was refined after viewing the locked test results.",
            "Validation is not an independent prospective confirmation.",
            "Only recordings with valid event alignment and sufficient eye-state windows are analyzed.",
            "Eye-state windows have systematically different recording times.",
            "Linear time adjustment cannot eliminate nonlinear confounding.",
            "Eye-state condition is observational, not randomized.",
            "Nearby window matching does not establish causation.",
            "The score distributions of different model families are not necessarily on comparable scales.",
            "Changing a decision threshold changes apparent label stability.",
            "Recordings may not represent independent patients.",
            "No clinical Alzheimer's biomarker or diagnostic benefit is established."
        ]
    }

    saved = save_once(OUTPUT, report)

    if saved:
        print(
            "\nCONFOUND_AUDIT_STATUS: COMPLETED"
        )
    else:
        print(
            "\nCONFOUND_AUDIT_STATUS: OUTPUT_CONFLICT"
        )

    print("Test EEG, events and predictions read: 0")
    print("Original model weights modified: 0")
    print("Test-based tuning performed: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print(
        "CONFOUND_AUDIT_STATUS: FAILED"
    )

finally:
    print(
        "\n========== CODE 23 FINISHED ==========",
        flush=True
    )
