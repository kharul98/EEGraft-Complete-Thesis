# EEGraft final publication-quality revision source

**Title:** EEGraft: Gated Dual-Scale Residual Fusion and Threshold-Aware Reliability Assessment for EEG-Based Dementia Classification.

**Main entrypoint:** `main.tex`. This is the current 80-page nine-chapter A4-portrait research manuscript. Its four revised vector figures are in `images_final/` with editable `.pptx` sources in `editable_figures/`. The source preserves 19 selected figures overall and 30 chronologically ordered literature-table studies.

**Build locally or in Overleaf:** use a complete TeX Live installation with pdfLaTeX, Biber and MakeIndex. The included `latexmkrc` handles the abbreviation nomenclature. For a manual build:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
makeindex main.nlo -s nomencl.ist -o main.nls
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

**Audit record:** `audit_reports/` includes novelty evidence, individual bibliography entries and verification limits, manuscript corrections, a figure-text matrix, rubric mapping and PDF quality checks. Research-model checkpoint binaries and restricted individual CAUEEG records are not redistributed in this manuscript source.

**Signatures:** the five originally supplied student/supervisor signature image files were preserved byte-for-byte and remain in their original intended locations. The request confirms their usage authorization; other institutional signatures/approvals are a separate university process.

**Scope and limitations:** This is a reviewed research thesis, not evidence of publication acceptance or clinical certification. The historic duration pipeline and restricted test source archives were not independently replayed in this editing environment. The exact archived threshold/Platt selection provenance and restricted-data distribution permissions require source-owner review.

**Final correction-only revision (9 November 2026):** Five malformed citations are repaired; all 36 Biber reference keys resolve, the bibliography is kept at 12-point type with compact leading, PDF metadata is populated, and Appendix B.4's referenced Markdown audit is included under `revision_final/`. Figure 5.1 uses a print-enlarged arrangement of the previously approved source artwork, with the original PNG preserved beside it. The rebuilt document has 80 portrait A4 pages. See `audit_reports/EEGraft_Final_Correction_Checklist.md`.
