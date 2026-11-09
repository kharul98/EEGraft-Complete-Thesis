from pathlib import Path
import json
import hashlib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

R = Path(__file__).resolve().parent
F = R / "eeg_foundation_thesis_starter/eeg_foundation_thesis"
S = R / "reports/thesis_final"
O = R / "thesis_closeout_2026_11_08"

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_csv(p, **kwargs):
    need(p.is_file(), f"Missing {p}")
    return pd.read_csv(p, **kwargs)

def read_json(p):
    return json.loads(p.read_text())

def put(p, content):
    data = content.encode() if isinstance(content, str) else content
    if p.exists():
        need(p.read_bytes() == data, f"Existing artifact changed: {p}")
    else:
        with p.open("xb") as f:
            f.write(data)

def tab(columns, rows):
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |"
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(x).replace("|", "/").replace("\n", " ")
                for x in row
            ) + " |"
        )
    return "\n".join(lines)

def run():
    need(R.name == "alzheimers-thesis", "Wrong project root")

    m = S / "milestone_7c_original_ad_ftd_recomputed_v2"
    original = read_csv(
        m / "recomputed_first_minute_primary_results.csv"
    ).set_index("task")

    legacy = read_csv(m / "legacy_60s_crosscheck.csv")
    need(
        len(legacy) == 2
        and int(legacy.tested.sum()) == 435
        and legacy.mismatched_60s_labels.eq(0).all(),
        "Original M7C audit not repaired"
    )

    ad = read_csv(
        S / "duration_locked_v1/held_out_predictions.csv",
        dtype={"subject": str}
    )
    ftd = read_csv(
        S / "ftd_extension_v1/locked_study/held_out_duration_predictions.csv",
        dtype={"subject": str}
    )

    need(
        set(ad.model) == {"RBF_SVM", "LogisticRegression"},
        "Historical model mismatch"
    )
    ad = ad[ad.model == "RBF_SVM"]

    def ids(frame, group):
        return set(frame.loc[frame.group == group, "subject"])

    A, C = ids(ad, "A"), ids(ad, "C")
    FD, FC = ids(ftd, "F"), ids(ftd, "C")

    need(
        (len(A), len(FD), len(C), len(FC)) == (23, 18, 23, 23)
        and C == FC,
        "Original participant selection mismatch"
    )

    older = A | FD | C
    need(len(older) == 64, "Original cohort should have 64 people")

    manifest = read_csv(
        F / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )
    need(
        len(manifest) == 65 and manifest.subject_id.is_unique,
        "Foundation cohort mismatch"
    )
    need(
        manifest.label.value_counts().to_dict() == {"AD": 36, "CN": 29},
        "Foundation class counts changed"
    )
    need(
        manifest.split.value_counts().to_dict()
        == {"train": 39, "val": 13, "test": 13},
        "Original locked partition changed"
    )

    later = set(manifest.subject_id)
    need(
        len(A & later) == 23
        and len(C & later) == 23
        and not (FD & later),
        "Original-to-foundation participant mapping mismatch"
    )

    hold = F / "results/locked_final_test_2026_v1"
    status = read_json(hold / "status.json")
    need(
        status["state"] == "complete"
        and status["test_evaluated_once"] is True,
        "Final held-out archive not frozen"
    )

    for filename, checksum in status["output_sha256"].items():
        need(
            h(hold / filename) == checksum,
            "Locked-test checksum mismatch: " + filename
        )

    fm = read_json(hold / "final_metrics.json")
    patients = read_csv(
        hold / "paired_subjects.csv",
        dtype={"subject_id": str}
    )

    need(
        len(patients) == 13
        and set(patients.subject_id)
        == set(manifest.loc[manifest.split == "test", "subject_id"]),
        "Final-test subject mismatch"
    )

    for model, column in [
        ("DSRF", "dsrf_label"),
        ("XGBoost", "xgboost_label")
    ]:
        score = balanced_accuracy_score(
            patients.true_label, patients[column]
        )
        need(
            abs(score - fm["models"][model]["balanced_accuracy"]) < 1e-10,
            model + " test metrics changed"
        )

    duration = read_csv(
        R / "tables/table_primary_ad_vs_cn_performance.csv"
    )
    need(len(duration) == 15, "Duration experiment incomplete")
    center = duration[
        duration.anchor == "center"
    ].sort_values("duration_seconds")

    dev = read_csv(
        F / "results/thesis_execution_package_2026_v1/"
        "table_1_eight_models.csv"
    )
    need(len(dev) == 8, "Eight-model benchmark incomplete")

    previous = read_csv(
        S / "publication_evidence_v1/"
        "model_comparison_from_saved_oof.csv"
    )
    need(
        len(previous) == 9
        and previous.n_participants.eq(46).all(),
        "Earlier 46-person model benchmark incomplete"
    )

    ab = read_csv(
        F / "results/thesis_execution_package_2026_v1/"
        "table_2_matched_skip.csv"
    )
    robust = read_csv(
        F / "results/devcv_robustness_repeated_v1/"
        "summary_repeated.csv"
    )
    need(
        len(ab) == 2 and len(robust) == 9,
        "Ablation or robustness results incomplete"
    )

    explanations = read_csv(
        F / "results/devcv_dsr_fusion_interpret_full24_2026/"
        "per_subject_channel.csv",
        dtype={"subject_id": str}
    )
    need(
        len(explanations) == 52 * 19
        and explanations.subject_id.nunique() == 52,
        "Full-window explanations incomplete"
    )
    need(
        set(explanations.subject_id)
        == set(manifest.loc[manifest.split != "test", "subject_id"]),
        "Explanation subject leakage"
    )

    isfahan = read_csv(
        S / "milestone_5i_isfahan_negative_controls_v1/"
        "constant_label_comparison.csv"
    )
    need(not isfahan.empty, "Isfahan negative-control output missing")

    snapshot = read_json(
        R / "results/manuscript_summary/numeric_snapshot.json"
    )
    need(
        abs(
            float(center.loc[
                center.duration_seconds == 180, "balanced_accuracy"
            ].iloc[0])
            - snapshot["best_primary_condition"]["balanced_accuracy"]
        ) < 1e-10,
        "Original duration result changed"
    )

    info = {
        "original_n": 64,
        "original_AD": 23,
        "original_FTD": 18,
        "original_CN": 23,
        "foundation_n": 65,
        "overlap_original64_with_later65": len(older & later),
        "union_unique": len(older | later),
        "development_n": 52,
        "heldout_n": 13,
        "m7c_60s_disagreements": 0,
        "final_DSRF_bacc": fm["models"]["DSRF"]["balanced_accuracy"],
        "final_XGBoost_bacc": fm["models"]["XGBoost"]["balanced_accuracy"],
        "test_reexecuted": False,
        "models_retrained": False
    }

    sections = []
    def add(s):
        sections.append(s.strip() + "\n")

    add("""# Reliability of EEG-Based Dementia Classification
## Temporal Consistency, Recording-Duration Sensitivity, Calibration, and Foundation-Model Evaluation

Bachelor's Thesis | November 2026
Student: [Name] | Supervisor: [Name] | University: [University]
""")

    add("""# Abstract
This thesis evaluates whether EEG-based dementia predictions remain reliable beyond one accuracy figure. A 64-person selected Alzheimer's disease (AD), frontotemporal dementia (FTD), and control study investigated SVM label stability across five minutes; a separate preregistered 65-person AD/control analysis investigated available EEG duration from 30 to 300 seconds. The original SVM first-minute balanced accuracy was 73.48% for AD and 73.72% for FTD; 44.35% and 33.66% of correlated subject-repeat decisions changed over five minutes, while 8.70% and 13.17% remained consistently wrong. The recording-duration study observed 83.05% AD/control balanced accuracy at 180 seconds (center), without demonstrating a universal optimal recording length. A later frozen pretrained-CBraMod model with a custom dual-scale residual fusion (DSRF) decoder achieved 77.06% development balanced accuracy, compared with 76.61% for XGBoost. On a once-used 13-person holdout, DSRF scored 70.24% and XGBoost 77.38%. Separate MCI and external-feasibility experiments exposed stable erroneous predictions and dataset incompatibilities. These findings support measuring accuracy, temporal stability, calibration, and robustness separately, not clinical diagnosis or future Alzheimer-risk prediction.
""")

    add("""# 1. Introduction and Research Objectives
Electroencephalography captures time-varying brain electrical activity. Even when retrospective group discrimination appears useful, a model can make different decisions about the same person at different time points, or repeat a consistent but incorrect decision. This creates a reliability problem beyond conventional accuracy. The thesis asks whether temporal consistency guarantees correctness; how recording duration and temporal position affect spectral measures and prediction; what calibration and error detection add; how recording conditions and external cohorts challenge predictions; and whether a pretrained foundation-model decoder is more reliable than simpler classifiers. These are empirical methodological questions; the thesis does not propose a clinical diagnostic instrument.
""")

    add("""# 2. Literature and Novel Contribution
The OpenNeuro/NEMAR cohort and preprocessed derivatives were published by Miltiadous and collaborators [1,2]. Earlier Alzheimer's EEG window-duration research exists [3], making a claim of first-ever duration analysis inappropriate. The thesis's duration contribution is its paired 30–300-second comparison across start, center and end segments under repeated subject-disjoint nested validation and uncertainty controls. The later neural experiments reuse the existing pretrained CBraMod architecture [4] rather than inventing a new foundation model; they add a task-specific residual decoder and matched architectural comparisons. EEGNet-like baselines are inspired by the published EEGNet family [5]. The principal contribution is a structured, audited reliability evaluation with appropriately reported negative findings.
""")

    add("""# 3. Data and Participant Lineage
The NEMAR/OpenNeuro source has 88 people: 36 AD, 23 FTD and 29 controls. The selected original temporal dataset has 23 AD, 18 FTD and 23 controls (64 people); the 23 controls are shared across its two binary tasks, so 46 AD/control and 41 FTD/control assessments do not represent 87 distinct patients. The preregistered duration primary comparison uses 65 AD/control people; its AD/FTD secondary analysis uses 59. The foundation-model study also uses 65 AD/control people, with 52 development and 13 locked test patients. The original and later datasets overlap and must not be presented as independent cohorts.
""")

    flow = [
        ("NEMAR source", "36 AD / 23 FTD / 29 controls", 88),
        ("Original temporal", "23 AD / 18 FTD / 23 controls", 64),
        ("Earlier neural baseline", "23 AD / 23 controls", 46),
        ("Preregistered duration", "36 AD / 29 controls", 65),
        ("Foundation development", "29 AD / 23 controls", 52),
        ("Foundation holdout", "7 AD / 6 controls", 13),
        ("Isfahan MCI archive", "29 MCI / 32 controls", 61)
    ]

    add(tab(["Source/selection", "Composition", "N"], flow))
    add(
        f"Direct participant-ID checking confirms that the original "
        f"and later studies share {len(older & later)} people and their "
        f"union contains {len(older | later)} of the 88 NEMAR participants. "
        f"No external AD patient cohort was independently validated."
    )

    add("""# 4. Methods
## 4.1 Five-Minute Temporal SVM Reliability
Six interpretable spectral features (relative delta, theta, alpha, beta and gamma power plus theta/alpha ratio) were used in standardized class-balanced RBF SVMs. Fivefold repeated outer cross-validation (five repetitions) held patients out, with training-only nested hyperparameter selection. First-minute models were evaluated on five held-out minutes without refitting. Outcomes distinguished stable-correct, stable-wrong, and unstable labels. Sigmoid probability calibration, selective prediction, MMSE associations, recording-condition testing, temporal explanation drift, and feature-group ablation were examined with leakage-aware designs. Repeated patient-repetition records are correlated.

## 4.2 Preregistered Duration Protocol
The 65-person AD/control analysis used dataset-provided denoised EEG at 500 Hz, Welch PSD from four-second windows with 50% overlap and spectral bands from 1 to 45 Hz. Prespecified durations 30, 60, 120, 180 and 300 seconds were evaluated at start, center and end positions. A class-balanced L2 logistic-regression baseline used repeated 5x10 outer subject-stratified nested CV with fourfold inner tuning. Paired bootstrap intervals compared each shorter duration to 300 seconds. A 100-permutation label-shuffling negative control and a separate 59-person AD/FTD comparison were included.

## 4.3 Foundation Model and Reliability Stress Tests
Nineteen EEG channels were resampled to 200 Hz and processed into at most 24 four-second windows per person. The official pretrained CBraMod was strictly loaded and frozen; trainable DSRF fused pretrained tokens with multiscale raw-signal features through learned gates and an additive residual pathway. Eight architectures or classical learners were assessed over five development folds. A parameter-matched residual-versus-blend ablation, 24-window channel occlusion, and five repeated 10%/20% noise and channel-zeroing experiments were performed on development patients only. Final five-epoch DSRF and fixed XGBoost models trained on the same 52 people and were each assessed once on the same locked 13-person holdout.
""")

    add("""# 5. Results
## 5.1 Original AD and FTD Accuracy, Stability and Calibration
""")

    metrics = [
        ("First-minute BAcc", "first_minute_original_BACC", True),
        ("Calibrated BAcc", "first_minute_calibrated_BACC", True),
        ("Calibrated Brier", "first_minute_calibrated_Brier", False),
        ("Training-prior Brier", "first_minute_training_prior_Brier", False),
        ("Stable-correct", "stable_correct_rate", True),
        ("Stable-wrong", "stable_wrong_rate", True),
        ("Prediction changed", "unstable_rate", True)
    ]

    rows = []
    for description, key, percent in metrics:
        vals = [
            float(original.loc[task, key])
            for task in ("AD_vs_HC", "FTD_vs_HC")
        ]
        formatted = [
            f"{100 * x:.2f}%" if percent else f"{x:.4f}"
            for x in vals
        ]
        rows.append((description, *formatted))

    add(tab(["Metric", "AD/control", "FTD/control"], rows))

    add("""These rates represent 230 AD and 205 FTD correlated subject-repeat assessments. The eight frequency-family ablation effects did not survive Holm correction. Descriptive eyes-open/photic versus eyes-closed balanced-accuracy changes were -8.83 pp for AD and -4.23 pp for FTD, with intervals containing zero. Calibration altered probabilistic quality, but should not be described as an across-task classification improvement.

## 5.2 Preregistered Duration Results
""")

    add(tab(
        ["Center duration", "AD/control BAcc", "AUROC"],
        [
            (
                f"{int(row.duration_seconds)} s",
                f"{100 * row.balanced_accuracy:.2f}%",
                f"{row.roc_auc:.3f}"
            )
            for _, row in center.iterrows()
        ]
    ))

    add("""The highest observed center result was 83.05% at 180 seconds; 300 seconds reached 81.32% and 60 seconds 78.21%. The paired 180-minus-300 difference was +1.72 pp (exploratory 95% interval 0 to +5.17), not evidence for a universally optimal duration or formal equivalence. The 300-second-center shuffled-label check observed balanced accuracy of 79.93% against a null mean of 49.82% (100 permutations, empirical p=0.0099); this checks internal discrimination, not a duration effect. AD/FTD discrimination was substantially weaker and often near chance. Exploratory adaptive stopping did not demonstrate a consistently better accuracy–time tradeoff.

## 5.3 Earlier 46-Person Multi-Model Comparison
""")

    add(tab(
        ["Model", "BAcc", "AUROC"],
        [
            (
                row.model,
                f"{100 * row.balanced_accuracy:.2f}%",
                f"{row.AUROC:.3f}"
            )
            for _, row in previous.iterrows()
        ]
    ))

    add("""This earlier benchmark is not numerically comparable to the later DSRF 52-patient development study or its held-out test; it demonstrates that classical, CNN, EEGNet, Conformer and frozen-CBraMod baselines were attempted before the custom decoder.

## 5.4 Later Eight-Model Development Comparison
""")

    add(tab(
        ["Model", "Pooled BAcc %", "Macro-F1 %", "Correct"],
        dev[
            ["Model", "Pooled BAcc (%)", "Macro-F1 (%)", "Correct /52"]
        ].itertuples(index=False, name=None)
    ))

    add("""DSRF obtained the highest observed pooled development BAcc (77.06%) but was practically close to XGBoost (76.61%); each classified 40 of 52 development patients correctly. Comparisons are exploratory.

## 5.5 Capacity-Matched Ablation and Explanation
""")

    add(tab(
        ["Fusion", "Pooled BAcc %", "Correct"],
        ab[
            ["Rule", "Pooled BAcc (%)", "Correct /52"]
        ].itertuples(index=False, name=None)
    ))

    add("""Matched residual fusion exceeded gated blend numerically (77.06% vs 73.16% BAcc), without conclusive paired statistical evidence. Full-window occlusion rankings were AD: T5, Pz, F8, Fz, C3; controls: O2, T6, O1, Fp2, F3. These are model sensitivities, not causal clinical EEG biomarkers.

## 5.6 Repeated Synthetic Robustness
""")

    add(tab(
        ["Decoder", "Perturbation", "Mean BAcc %", "Flip %", "Brier"],
        robust[
            [
                "model", "condition", "mean_BAcc_pct",
                "mean_flip_pct", "mean_Brier"
            ]
        ].itertuples(index=False, name=None)
    ))

    add("""DSRF's mean BAcc dropped to 71.87% and 70.07% under 10% and 20% Gaussian noise, with mean label-flip rates of 23.08% and 27.69%. A one-electrode zeroing condition yielded 76.78% BAcc. Repeated perturbations are not independent patient replications; a constant-label classifier can have zero flips and 50% BAcc.

## 5.7 Locked 13-Patient Final Test
""")

    final_rows = []
    for name in ("DSRF", "XGBoost"):
        result = fm["models"][name]
        final_rows.append((
            name,
            f"{100 * result['balanced_accuracy']:.2f}%",
            f"{100 * result['accuracy']:.2f}%",
            f"{100 * result['macro_f1']:.2f}%",
            f"{result['auroc_ovr_macro']:.3f}",
            f"{int(round(result['accuracy'] * 13))}/13"
        ))

    add(tab(
        ["Model", "BAcc", "Accuracy", "Macro-F1", "AUROC", "Correct"],
        final_rows
    ))

    add("""DSRF identified 4/7 AD and 5/6 controls; XGBoost identified 5/7 AD and 5/6 controls. The paired BAcc difference DSRF minus XGBoost was -7.14 pp (exploratory 95% interval -28.57 to +14.29 pp; exact McNemar p=1.0). This small within-cohort holdout neither establishes superiority nor clinical diagnostic validity.

## 5.8 Isfahan MCI and External Feasibility
""")

    add(tab(
        ["Release", "MCI/healthy", "First-minute BAcc", "Stable-wrong", "Unstable"],
        [
            ("EEG_1", "11/14", "40.78%", "59.20%", "5.60%"),
            ("EEG_2", "12/12", "55.83%", "35.83%", "20.00%")
        ]
    ))

    add("""The EEG_1 classifier was strongly biased toward MCI. An always-MCI negative control also had 56% stable-wrong labels, demonstrating that stability alone is not evidence of useful discrimination. Figshare had 471 MATLAB files but no verified comparable 19-channel control population; no external AD classification accuracy was produced. Dryad and BrainLat results remained recording-level spectral-feasibility investigations with unresolved independence, montage and site confounds.
""")

    add("""# 6. Discussion and Scientific Contribution
The studies consistently show why accuracy, temporal consistency, calibration, signal quality, recording duration and external transportability are distinct. The original SVM often changed predictions across five minutes, while the Isfahan negative control could remain consistently wrong. The preregistered duration study found variable outcomes but no formally optimal acquisition time. The custom DSRF decoder improved upon some simpler development variants but not the final XGBoost comparator on 13 held-out patients, and it remained vulnerable to synthetic noise. The thesis contribution is a reproducible reliability-focused empirical program, including negative results and conservative uncertainty statements. It is not a new foundation model or a deployable diagnostic tool.
""")

    add("""# 7. Limitations and Ethics
Multiple analytical cohorts overlap within one original 88-person EEG source and must not be counted as independent validation. The original selected 46-patient AD/control group has exclusions and three publication-review cases requiring final objective explanations. Repeated CV estimates share participants and are correlated; the 13-patient test has wide intervals. Eyes-open/photic shifts are confounded with recording protocols; channel occlusion is not physiological causality. The foundation classifier used up to 24 four-second EEG windows per person, not five continuous minutes of temporal stability. The Isfahan releases are not proven independent patients and their EDF unit fields required assumptions; Figshare, Dryad and BrainLat cannot be claimed as successful external clinical AD validation.

Ethics statement to finalize locally: This is secondary analysis of publicly released, de-identified datasets without new participant recruitment or patient intervention. Confirm the university-required ethics statement and each source dataset's data-use and redistribution terms before submission or public redistribution.
""")

    add("""# 8. Conclusion
EEG dementia classifiers can be internally discriminative while unstable across recording time, stable while wrong, and vulnerable to changes in acquisition and corruption. Strong architecture or a single high accuracy figure does not guarantee reliability. Patient-level validation, negative controls, sensitivity studies and cautious reporting are essential. Independent prospective clinical validation remains future work.
""")

    add("""# References
[1] Miltiadous A, Tzimourta KD, Afrantou T, et al. A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. Data. 2023;8(6):95. DOI: 10.3390/data8060095

[2] Miltiadous A, et al. A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects. NEMAR on004504 v1.0.0. 2026. DOI: 10.82901/nemar.on004504

[3] Tzimourta KD, et al. EEG Window Length Evaluation for the Detection of Alzheimer's Disease over Different Brain Regions. Brain Sciences. 2019;9(4):81. DOI: 10.3390/brainsci9040081

[4] Wang J, Zhao S, Luo Z, Zhou Y, Jiang H, Li S, Li T, Pan G. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR. 2025. https://openreview.net/forum?id=NPNUHgHF2w

[5] Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng. 2018;15(5):056013. DOI: 10.1088/1741-2552/aace8c
""")

    add("""# Appendix A. Evidence and Provenance
Original temporal and calibration evidence: reports/thesis_final/milestone_7c_original_ad_ftd_recomputed_v2/ and milestone_4c_reproducibility_v1/. Duration protocol and metrics: PREREGISTRATION.md, tables/ and results/duration_experiment/. Earlier model baselines: reports/thesis_final/publication_evidence_v1/. Isfahan MCI: reports/thesis_final/milestone_5i_isfahan_negative_controls_v1/. Foundation benchmark, ablation, explanations, perturbation and final results: eeg_foundation_thesis_starter/eeg_foundation_thesis/results/. The frozen final protocol and test status.json must remain unmodified. University cover, supervisor approval, ethics signoff and independently verifiable external clinical validation are not supplied by this evidence package.
""")

    md = "\n\n".join(sections) + "\n"
    O.mkdir(parents=True, exist_ok=True)

    put(O / "INTEGRATED_THESIS_MANUSCRIPT.md", md)
    put(
        O / "evidence_audit.json",
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )

    put(
        O / "cohort_flow.csv",
        "Selection,Composition,N\n"
        + "".join(f'"{a}","{b}",{c}\n' for a, b, c in flow)
    )

    sources = [
        m / "recomputed_first_minute_primary_results.csv",
        m / "legacy_60s_crosscheck.csv",
        R / "tables/table_primary_ad_vs_cn_performance.csv",
        S / "publication_evidence_v1/model_comparison_from_saved_oof.csv",
        F / "results/thesis_execution_package_2026_v1/table_1_eight_models.csv",
        F / "results/devcv_robustness_repeated_v1/summary_repeated.csv",
        hold / "final_metrics.json",
        hold / "status.json",
        F / "results/final_protocol_2026.json"
    ]

    put(
        O / "source_hashes.csv",
        "path,sha256\n"
        + "".join(f"{p.relative_to(R)},{h(p)}\n" for p in sources)
    )

    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    docx = O / "INTEGRATED_THESIS_MANUSCRIPT.docx"

    if not docx.exists():
        doc = Document()
        sec = doc.sections[0]
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11)
        sec.left_margin = sec.right_margin = Inches(0.76)
        sec.top_margin = sec.bottom_margin = Inches(0.77)

        normal = doc.styles["Normal"]
        normal.font.name = "Cambria"
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)

        for style_name, size in [("Heading 1", 15), ("Heading 2", 12)]:
            style = doc.styles[style_name]
            style.font.name = "Aptos Display"
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor(25, 53, 85)

        lines = md.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([
                        x.strip()
                        for x in lines[i].strip().strip("|").split("|")
                    ])
                    i += 1

                rows = [
                    row for row in rows
                    if not all(set(cell) <= set("-: ") for cell in row)
                ]

                if not rows:
                    continue

                table = doc.add_table(rows=1, cols=len(rows[0]))
                table.style = "Table Grid"

                for j, value in enumerate(rows[0]):
                    cell = table.rows[0].cells[j]
                    cell.text = value
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "E8EFF4")
                    cell._tc.get_or_add_tcPr().append(shade)

                for values in rows[1:]:
                    cells = table.add_row().cells
                    for j, value in enumerate(values):
                        cells[j].text = value

                for row in table.rows:
                    for cell in row.cells:
                        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.space_after = Pt(1)
                            for run in paragraph.runs:
                                run.font.size = Pt(
                                    8 if len(rows[0]) >= 5 else 9
                                )

                doc.add_paragraph("")
                continue

            if line.startswith("# "):
                doc.add_heading(line[2:], level=1)
            elif line.startswith("## "):
                doc.add_heading(line[3:], level=2)
            else:
                doc.add_paragraph(line)
            i += 1

        figures = [
            R / "figures/fig01_primary_balanced_accuracy.png",
            F / "results/thesis_execution_package_2026_v1/"
                "figure_2_final_holdout.png"
        ]

        for fig in figures:
            if fig.is_file():
                doc.add_picture(str(fig), width=Inches(6.25))

        doc.core_properties.title = (
            "Reliability of EEG-Based Dementia Classification"
        )
        doc.core_properties.author = ""
        doc.save(docx)

    files = [
        "INTEGRATED_THESIS_MANUSCRIPT.md",
        "INTEGRATED_THESIS_MANUSCRIPT.docx",
        "evidence_audit.json",
        "cohort_flow.csv",
        "source_hashes.csv"
    ]

    put(
        O / "closeout_index.sha256",
        "".join(f"{h(O / filename)}  {filename}\n" for filename in files)
    )

    print("\n========== THESIS CLOSEOUT ==========")
    print("EVIDENCE_CLOSEOUT=PASS")
    print("COHORT_LINEAGE=64 original, 65 later, 46 shared, union 83")
    print("ORIGINAL_SVM_AUDIT=435/435 labels agree")
    print("LOCKED_TEST=13 subjects, no inference rerun")
    print("THESIS_WORD_DOCUMENT=" + str(docx))
    print("THESIS_MARKDOWN_DOCUMENT=" + str(
        O / "INTEGRATED_THESIS_MANUSCRIPT.md"
    ))

if __name__ == "__main__":
    run()
