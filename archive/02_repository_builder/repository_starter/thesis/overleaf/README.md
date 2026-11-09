# EEGraft thesis — Overleaf-ready report package

**Locked title:** EEGraft: A Gated Dual-Scale Residual Fusion Decoder and Subject-Level Reliability Assessment for Alzheimer's EEG Classification

This is a full nine-chapter **editable LaTeX thesis draft**, aligned with the 12pt A4 report formatting found in the supplied IdentityGate reference ZIP. It includes cover, declaration, approval, ethics, abstract, dedication, acknowledgement, ToC, lists of figures/tables/abbreviations, 30-paper Chapter 2 literature comparison, all nine chapters, two appendices, 35 bibliographic source entries, 13 numbered figures, author signature crops taken only from the submitted PDF, and a compiled PDF preview when local compilation succeeds.

## Open in Overleaf
1. In Overleaf select **New Project > Upload Project** and upload the ZIP. Main file: `main.tex`.
2. Set compiler **pdfLaTeX**. The bibliography is **biblatex + Biber**; Overleaf handles Biber automatically. If building locally: `latexmk -pdf -interaction=nonstopmode -use-biber main.tex`, and `makeindex main.nlo -s nomencl.ist -o main.nls` for the abbreviation list, then rerun LaTeX.
3. Edit student/supervisor details and submission date in `core/metadata.tex` if the university requires an updated version. Confirm names, signatures and the approval wording before final institutional submission.
4. All figures are in `images/`; the references compile with those filenames. `reproducibility/FIGURE_PROVENANCE.json` documents the source of each image. Do not change scientific values merely to improve presentation.

## Template match
- Class: `report`, 12 pt, Computer/Latin Modern default serif inherited from reference.
- A4 paper; left 1.25 in, right/top/bottom 1 in.
- Reference `titlesec` chapter-title format and IEEE-style bibliography with Biber.
- Front matter layout and single-supervisor approval area adapted from uploaded sample. **No co-supervisors are entered; committee signatures are blank.**

## Scientific interpretation
This report distinguishes the 64-person SVM temporal-reliability study from the later 65-person CBraMod–DSRF study; the two subject lists have not been reconciled. It reports the observed advantage of XGBoost over the proposed DSRF model on the small 13-person final holdout. It makes no clinical validation or future-disease-risk claims.

## Figure caveat
The 13 figures in this ZIP comprise **4 previously accessible user-supplied method/duration images** and **9 report plots based on saved research aggregates**. This is **not** the exact earlier approved 13-image ZIP, because that separate archive was not attached to this request and was not accessible here. The scientific findings and counts remain those saved in the project records; no raw EEG snippets were invented. If the exact 13-image archive is supplied, replace only after a one-to-one caption and evidence audit.

## Outstanding final audit
- Confirm official semester/month and primary supervisor name/title.
- Confirm the student signature crops are acceptable to your institution.
- Cross-check the early temporal results against original machine-readable tables and resolve original analytic participant IDs.
- Cross-check specific low-level decoder layers/feature shapes against the final implementation source code.
- Review 2026 literature metadata and complete abbreviated author lists where necessary.
- Confirm the final original figure set if a byte-for-byte match is mandatory.

This is a detailed submission-style draft, **not** a claim that a supervisor has approved its scientific contents.

## Formatting repair update (2026-11-08)

The revised 62-page PDF fixes title-page merging (including all four student IDs), replaces tiny signature thumbnails with higher-resolution crops from the supplied team PDF (manually checked for correct name-to-signature mapping), keeps the full approval committee on one page, repairs Figure 7.1 title/legend overlap, and improves longtables and page flow. Four detailed figures use landscape pages. The original research tables, values, bibliography and thesis title are unchanged. See `reproducibility/FORMATTING_AUDIT_REPORT.md`.

The extracted **student** signatures were provided for layout reproduction; this does not mean the thesis has received new supervisor or university approval. All committee signature lines stay blank.
