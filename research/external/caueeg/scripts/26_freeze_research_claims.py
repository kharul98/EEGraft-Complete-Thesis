from pathlib import Path
import hashlib
import json
import os

HOME = Path.home()
WORK = HOME / "research/eegraft-caueeg"

RESULTS = WORK / "results"
PROTOCOL = WORK / "protocol"
OUT = RESULTS / "26_research_synthesis"

SOURCES = {
    "plan": PROTOCOL / "caueeg_final_test_plan_v1.json",
    "test": RESULTS / "19_locked_test_evaluation.json",
    "event": RESULTS / "21_event_alignment_summary.json",
    "condition": RESULTS / "22_condition_analysis.json",
    "confound": RESULTS / "23_reliability_confound_audit.json",
    "age": RESULTS / "24_age_label_audit.json",
    "incremental": RESULTS / "25_age_incremental_information_audit.json"
}

STATUSES = {
    "test": "LOCKED_EXTERNAL_TEST_EVALUATION_COMPLETED",
    "event": "EVENT_ALIGNMENT_AUDIT_COMPLETED",
    "condition": "CONDITION_ANALYSIS_COMPLETED",
    "confound": "RELIABILITY_CONFOUND_AUDIT_COMPLETED",
    "age": "AGE_LABEL_AUDIT_COMPLETED",
    "incremental": "AGE_INCREMENTAL_AUDIT_COMPLETED"
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(4194304), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing file matches:", path.name)
            return True
        print("[CONFLICT] Existing file preserved:", path)
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


def pct(value):
    return f"{100 * float(value):.2f}%"


def pp(value):
    return f"{100 * float(value):+.2f} pp"


def main():
    print("\n1. VERIFYING RESEARCH EVIDENCE")

    docs = {}

    for key, path in SOURCES.items():
        require(
            path.is_file(),
            "Missing source: " + str(path)
        )
        docs[key] = read_json(path)

    for key, expected in STATUSES.items():
        require(
            docs[key].get("status") == expected,
            "Incomplete report: " + key
        )
        print("[VERIFIED]", key)

    plan_hash = sha256(SOURCES["plan"])

    require(
        docs["test"]["final_test_plan_sha256"]
        == plan_hash,
        "Locked-test plan mismatch"
    )

    require(
        docs["event"]["final_test_plan_sha256"]
        == plan_hash,
        "Event analysis plan mismatch"
    )

    require(
        docs["condition"]["provenance"][
            "final_test_plan_sha256"
        ] == plan_hash,
        "Condition analysis plan mismatch"
    )

    require(
        docs["confound"]["provenance"][
            "plan_sha256"
        ] == plan_hash,
        "Threshold audit plan mismatch"
    )

    require(
        docs["age"]["provenance"][
            "final_test_plan_sha256"
        ] == plan_hash,
        "Age audit plan mismatch"
    )

    require(
        docs["incremental"]["source_hashes"][
            "final_test_plan_sha256"
        ] == plan_hash,
        "Incremental analysis plan mismatch"
    )

    require(
        docs["incremental"]["source_hashes"][
            "code24_report_sha256"
        ] == sha256(SOURCES["age"]),
        "Code 24 evidence changed"
    )

    require(
        docs["test"]["primary_binary_recordings"] == 57,
        "Unexpected external test population"
    )

    require(
        docs["test"]["model_weights_modified"] is False
        and docs["test"][
            "hyperparameters_reselected_after_test"
        ] is False,
        "Test integrity violation"
    )

    require(
        docs["incremental"][
            "test_predictions_accessed"
        ] == 0,
        "Code 25 unexpectedly accessed test predictions"
    )

    print("[OK] All evidence traces to the frozen protocol")

    test = docs["test"]
    age = docs["age"]
    incremental = docs["incremental"]
    confound = docs["confound"]

    dsrf = test["model_variants"]["DSRF"]["original"]
    xgb = test["model_variants"]["XGBoost"]["original"]
    xgb_tuned = test["model_variants"]["XGBoost"]["threshold"]

    difference_ci = test["uncertainty"][
        "paired_balanced_accuracy_difference_95ci"
    ]["DSRF_original_minus_XGBoost_original"]

    require(
        difference_ci[0] <= 0 <= difference_ci[1],
        "Original model comparison interpretation changed"
    )

    train_flips = confound["results"]["train"][
        "threshold_sensitivity"
    ]["variants"]

    validation_flips = confound["results"]["validation"][
        "threshold_sensitivity"
    ]["variants"]

    age_only_auc = age["age_baseline_analysis"][
        "model_comparisons"
    ]["Age only"]["validation"]["auroc"]

    train_add = incremental[
        "paired_incremental_information"
    ]["train_oof"]["linear_DSRF_added_to_age"]

    val_add = incremental[
        "paired_incremental_information"
    ]["validation"]["linear_DSRF_added_to_age"]

    require(
        train_add["auroc_difference_95ci"][0] > 0,
        "Training age increment interpretation changed"
    )

    require(
        val_add["auroc_difference_95ci"][0] <= 0
        <= val_add["auroc_difference_95ci"][1],
        "Validation increment interpretation changed"
    )

    print("\n2. BUILDING SCIENTIFIC CLAIM REGISTER")

    claims = [
        {
            "id": "C1",
            "subject": "Custom EEGraft decoder",
            "classification": "technical contribution",
            "supported": (
                "Implemented a gated dual-scale residual fusion "
                "decoder with frozen pretrained CBraMod features."
            ),
            "not_supported": (
                "A globally unprecedented architecture or "
                "a consistently superior classifier."
            )
        },
        {
            "id": "C2",
            "subject": "External discrimination",
            "classification": "primary locked-test observation",
            "supported": (
                f"Original DSRF BAcc {pct(dsrf['balanced_accuracy'])}; "
                f"original XGBoost {pct(xgb['balanced_accuracy'])}."
            ),
            "not_supported": (
                "Statistically established superiority of DSRF "
                "over XGBoost or clinically valid AD diagnosis."
            )
        },
        {
            "id": "C3",
            "subject": "Threshold dependence",
            "classification": "post-hoc exploratory",
            "supported": (
                "XGBoost eyes-state flip frequency changes "
                "markedly between the frozen 0.50 rule "
                "and its training-selected 0.70 rule."
            ),
            "not_supported": (
                "Threshold adjustment eliminates all instability "
                "or proves an architectural advantage."
            )
        },
        {
            "id": "C4",
            "subject": "Stable but wrong",
            "classification": "descriptive reliability evidence",
            "supported": (
                "Some recordings receive unanimous incorrect "
                "window labels despite high temporal consistency."
            ),
            "not_supported": (
                "The more variable model is therefore always "
                "more reliable or clinically safer."
            )
        },
        {
            "id": "C5",
            "subject": "Eye-state associations",
            "classification": "post-hoc observational",
            "supported": (
                "Saved model scores change with annotated recording "
                "conditions in a subset with aligned event metadata."
            ),
            "not_supported": (
                "Eye state is the causal source of all variation."
            )
        },
        {
            "id": "C6",
            "subject": "Age confounding",
            "classification": "post-hoc confounding audit",
            "supported": (
                "Normal and Dementia groups differ substantially "
                "in age; age alone predicts CAUEEG labels."
            ),
            "not_supported": (
                "EEG discrimination is age-independent, "
                "or age accounts for every EEG association."
            )
        },
        {
            "id": "C7",
            "subject": "Incremental EEG information",
            "classification": "development signal, not confirmed",
            "supported": (
                "Age-plus-DSRF improves training OOF AUROC, "
                "while the small validation improvement has "
                "an uncertainty interval spanning zero."
            ),
            "not_supported": (
                "Independently established age-adjusted "
                "EEG diagnostic value."
            )
        },
        {
            "id": "C8",
            "subject": "Clinical interpretation",
            "classification": "explicit limitation",
            "supported": (
                "CAUEEG provides a broader clinical dementia "
                "classification setting at recording level."
            ),
            "not_supported": (
                "Confirmed Alzheimer diagnosis, prospective "
                "risk prediction, or patient-level validation."
            )
        }
    ]

    ledger = {
        "status": "SCIENTIFIC_CLAIMS_FROZEN",
        "research_position": (
            "Threshold-aware and error-aware EEG reliability "
            "under cross-cohort clinical task shift"
        ),
        "working_title": (
            "EEGraft: Gated Dual-Scale Residual Fusion "
            "and Threshold-Aware Reliability Assessment "
            "Across Alzheimer's and Dementia EEG Cohorts"
        ),
        "claims": claims,
        "source_hashes": {
            key: sha256(path)
            for key, path in SOURCES.items()
        },
        "statistical_boundaries": {
            "external_test_unit": "recording",
            "verified_patient_identity": False,
            "post_hoc_analyses_confirmatory": False,
            "new_test_set_available": False,
            "more_test_tuning_allowed": False,
            "world_first_claim_supported": False
        }
    }

    print("\n3. BUILDING THESIS RESTRUCTURING BLUEPRINT")

    xgb_original_flip = train_flips[
        "XGBoost_original"
    ]["condition_flip_fraction"]

    xgb_adapted_flip = train_flips[
        "XGBoost_training_threshold"
    ]["condition_flip_fraction"]

    dsrf_flip = train_flips[
        "DSRF_original"
    ]["condition_flip_fraction"]

    validation_xgb_original = validation_flips[
        "XGBoost_original"
    ]["condition_flip_fraction"]

    validation_xgb_adapted = validation_flips[
        "XGBoost_training_threshold"
    ]["condition_flip_fraction"]

    blueprint = f"""# EEGraft: Final Research Positioning Blueprint

## Working title

{ledger['working_title']}

## Central research problem

A model can appear accurate or temporally stable while
its decisions remain sensitive to threshold choice,
recording conditions, demographic composition, or
cross-cohort clinical task differences.

This thesis develops a gated dual-scale residual fusion
decoder and evaluates these risks rather than claiming
that a high classification score alone proves reliability.

## Research contributions

1. Implemented EEGraft decoder and development ablations.
2. Internal source-cohort evaluation with a locked holdout.
3. Frozen transfer to the CAUEEG dementia benchmark.
4. Correctness-aware, threshold-aware temporal evaluation.
5. Explicit age, label, and recording-condition audits.

These are case-study contributions, not a proof of
global first-in-kind novelty.

## Essential final evidence

### External locked test

Binary test: 57 recordings (35 Normal, 22 Dementia).

DSRF original BAcc: {pct(dsrf['balanced_accuracy'])}
XGBoost original BAcc: {pct(xgb['balanced_accuracy'])}
XGBoost training-threshold BAcc:
{pct(xgb_tuned['balanced_accuracy'])}

DSRF minus original XGBoost paired 95% interval:
[{pp(difference_ci[0])}, {pp(difference_ci[1])}].

The interval crosses zero. Do not claim established
between-model superiority.

### Threshold-sensitive stability

Training: 404 paired condition recordings.

DSRF threshold 0.50 flip rate: {pct(dsrf_flip)}
XGBoost threshold 0.50 flip rate:
{pct(xgb_original_flip)}
XGBoost training threshold 0.70 flip rate:
{pct(xgb_adapted_flip)}

Validation: 42 paired condition recordings.

XGBoost original flip rate:
{pct(validation_xgb_original)}
XGBoost adapted flip rate:
{pct(validation_xgb_adapted)}

These are exploratory condition-pair results.
They do not establish a causal eye-state effect.

### Age confounding

Age-only validation AUROC: {age_only_auc:.4f}

Adding EEGraft DSRF to linear age:

Training OOF AUROC improvement:
{train_add['auroc_improvement']:+.4f}

Training conditional recording-bootstrap interval:
{train_add['auroc_difference_95ci']}

Validation AUROC improvement:
{val_add['auroc_improvement']:+.4f}

Validation conditional recording-bootstrap interval:
{val_add['auroc_difference_95ci']}

The validation interval includes zero. There is
no independently established incremental EEG gain.

## Proposed revised chapter hierarchy

### Chapter 1: Introduction

Reframe the problem around trustworthy EEG
classification under time and cohort shift.
State the limited research questions.
Do not imply that accuracy equals clinical reliability.

### Chapter 2: Literature Review

Organize the review by leakage, pretrained encoders,
cross-dataset generalization, temporal consistency,
decision-threshold effects, and demographic confounding.

Verify citations and novelty wording before submission.

### Chapter 3: Requirements and Constraints

Retain the original engineering context.
Add reproducibility and restricted-data safeguards.

### Chapter 4: Data and Experimental Protocols

Clearly separate:
Study A: original temporal SVM experiments.
Study B: source AD/CN decoder development and
13-person held-out evaluation.
Study C: CAUEEG recording-level transfer experiment.

Never pool these cohorts into one accuracy estimate.

### Chapter 5: Methodology

Keep the original DSRF architecture and ablations.
Explain pretrained encoder freezing, score aggregation,
decision thresholds, calibration, and error-aware
stability measurements.

State which analyses were planned before the locked
test and which were introduced afterward.

### Chapter 6: Classification Results

Lead with Study B development and source holdout.
Report favorable and unfavorable source results.
Then present original frozen CAUEEG test models,
their uncertainty, and exploratory adaptations.

### Chapter 7: Reliability and Confounding

Make threshold-dependent label stability the main
new observation.

Present stable-but-wrong counts next to accuracy.
Then explain event alignment, recording condition,
time ordering, age imbalance, and heterogeneous labels.

Distinguish training, validation, and already-used
locked-test evidence.

### Chapter 8: Discussion and Limitations

Discuss why an apparently stable model can mislead.
Acknowledge source-versus-target label mismatch,
absence of verified patient IDs, small validation cohort,
age confounding, post-hoc analysis, and uncertainty.

### Chapter 9: Conclusion

Conclude that EEGraft is a viable evaluated decoder
whose broader reliability cannot be summarized by
one accuracy or stability metric.

Do not claim clinical diagnostic readiness.

## Figure policy

Preserve authentic original architectural and EEG
figures where still relevant.

Update the research overview to show all three studies.

Promote a threshold-sensitivity comparison as a
main research figure.

Promote the external frozen model comparison and
a clear age-confounding summary.

Move redundant ablations, synthetic stress figures,
and exploratory plots to appendices where appropriate.

Keep the full provenance and figure numbering
consistent with the final manuscript.

## Final writing constraints

- No further optimization using locked-test results.
- No invented statistical significance.
- No claim of a proven first-in-world discovery.
- No unsupported clinical impact or risk prediction.
- No claim that CAUEEG Dementia equals confirmed AD.
- No recording-level results described as patient-level.
- No artificial or synthetic EEG presented as raw evidence.
- No restricted participant-level records in public exports.
- Preserve all unfavorable results.
"""

    print("\n4. SAVING CLAIMS AND THESIS BLUEPRINT")

    registry_path = OUT / "claims_registry.json"
    blueprint_path = OUT / "thesis_restructuring_blueprint.md"

    ok1 = write_once(
        registry_path,
        json.dumps(
            ledger, indent=2, sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    ok2 = write_once(
        blueprint_path,
        blueprint
    )

    if ok1 and ok2:
        print("\nRESEARCH_SYNTHESIS_STATUS: COMPLETED")
    else:
        print("\nRESEARCH_SYNTHESIS_STATUS: OUTPUT_CONFLICT")

    print("New experiments performed: 0")
    print("Original model weights modified: 0")
    print("Individual patient records exported: 0")
    print("Existing thesis files modified: 0")
    print("Further test-based tuning performed: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )
    print("RESEARCH_SYNTHESIS_STATUS: FAILED")

finally:
    print("\n========== CODE 26 FINISHED ==========")
