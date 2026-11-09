# Chapter-by-chapter defensibility, consistency and clarity review

This is an editorial self-check, not an independent certification of experiment integrity. The source of numerical facts is the saved project history and execution summaries; see `SOURCE_MANIFEST.md`.

## Chapter 1 — Introduction
- **Defensibility:** Establishes only retrospective AD/FTD and control classification, not future risk prediction. RQ2 treats decoder competitiveness as a testable question, not an achieved superiority claim.
- **Consistency:** Defines two tracks at the outset and never presents the original SVM's five-minute results as a DSRF measurement.
- **Clarity:** Replaces an unqualified accuracy claim with the distinction among correctness, stability, and robustness.
- **Viva checkpoint:** Why are two studies in one thesis? One evaluation problem motivates a decoder experiment, but there is no direct matched five-minute SVM/DSRF comparison.

## Chapter 2 — Literature Review
- **Defensibility:** Table contains 30 paper entries, with source citations and method-specific relevance. Latest 2025–2026 literature includes CBraMod, LEAD, AHEPA, EEG-FM-Bench and EEG-FM-Compass.
- **Consistency:** Separates AD EEG studies from general EEG model papers and method/evaluation literature.
- **Clarity:** Does not compare unrelated accuracy metrics in a single ranking or claim first discovery of window-length effects.
- **Final bibliographic audit:** Complete `and others` author names in several BibTeX entries from publisher metadata before archival submission; check institutional bibliography style.

## Chapter 3 — Requirements, Impacts and Constraints
- **Defensibility:** No unverified energy, hardware cost or ethics clearance number. The source data are publicly released, anonymized recordings.
- **Consistency:** Frozen pretrained encoder and one-time holdout remain the core technical constraints.
- **Clarity:** Medical risks and research limitations are expressed without calling this a clinical application.
- **Viva checkpoint:** Does freezing the backbone establish reliability? No; it is a design/resource choice.

## Chapter 4 — Dataset and Experimental Design
- **Defensibility:** 88 full dataset, 64 early analysis, 65 later analysis are separate counts. Unknown participant overlap is disclosed.
- **Consistency:** Original five-minute SVM protocol and later maximum 24 four-second decoder windows are not mixed.
- **Clarity:** Both 39/13/13 initial split and 52/13 final development/test structure are explained.
- **Source audit needed:** Recover original 64-person inclusion IDs and check early machine-readable result tables.

## Chapter 5 — Proposed Methodology and Decoder
- **Defensibility:** CBraMod is credited to the original authors. DSRF is described as an implemented fusion decoder; unverified layer-by-layer details are not invented.
- **Consistency:** Exact final learning and XGBoost parameters come from saved `final_protocol_2026.json`.
- **Clarity:** Equations describe abstract signal fusion and subject averaging without claiming external biomarker interpretation.
- **Source audit needed:** Compare the simplified fusion formula and raw-convolution kernel shapes against committed code before describing those details as mathematically identical.

## Chapter 6 — Experimental Results
- **Defensibility:** DSRF leads pooled development BAcc by only 0.45 pp; XGBoost leads the small locked holdout by 7.14 pp. Paired 95% interval crosses zero and McNemar p=1.0.
- **Consistency:** Older 46-person model audit is reported separately; early numbers are never ranked against 65-person models.
- **Clarity:** Confusion matrices show the underlying participant counts and prevent unsupported statistical interpretations.
- **Viva checkpoint:** Was the custom model better? It was narrowly higher on development BAcc but not on the final locked test.

## Chapter 7 — Reliability, Robustness and Explainability
- **Defensibility:** Stability is never treated as correctness. Synthetic conditions are not called real-world validation, and input masking is not causal physiology.
- **Consistency:** MCI is labeled a separate negative-control disease task, not external AD confirmation.
- **Clarity:** Every figure has a source-aware caption and explanatory paragraph.
- **Source audit needed:** Reconcile the exact original repeated-CV denominator used for reported five-minute change percentages.

## Chapter 8 — Discussion
- **Defensibility:** States limitations of internal and external validity; does not extrapolate to patient diagnosis.
- **Consistency:** Explains why two related but non-identical experiments can support one thesis narrative.
- **Clarity:** Separates observed point estimate differences from their uncertainty and model causal explanations.
- **Viva checkpoint:** Can these models identify someone likely to develop AD in the future? No, not without longitudinal labels and validation.

## Chapter 9 — Conclusion
- **Defensibility:** Conclusions bounded by actual cohorts, protocols and locked test outcomes.
- **Consistency:** Recaps both research tracks without pooling them or overstating neural-model gains.
- **Clarity:** Future clinical work is identified as proposed, not complete.

## Report-level layout checks
- 12-point `report` class and reference A4 margins retained.
- One primary supervisor, no co-supervisor names or rows; committee signature lines left blank.
- Student signature crops come only from supplied team-details PDF, not generated likenesses.
- Abstract is one readable paragraph plus a separate keywords line.
- All nine chapters, front matter, TOC, figure/table/abbreviation lists and bibliography are present.
- All 13 included figure files have provenance entries, but they are not a byte-identical recovery of the earlier approved figure ZIP.
- Report compiles with pdfLaTeX/Biber and no overfull hbox errors after fixes.
