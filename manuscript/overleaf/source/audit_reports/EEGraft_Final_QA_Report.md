# EEGraft Final Quality Assurance Report

**Compilation:** Biber succeeded with **36 cited keys**, three completed PDF LaTeX passes after source edits (additional passes after final figure-height change); no fatal LaTeX errors. A fresh extraction of the staged, source-only project was independently compiled with pdfLaTeX, Biber, MakeIndex and two further pdfLaTeX passes. The clean-room PDF has 81 pages, and **each page's extracted text exactly matches** the working final PDF.

**PDF:** 81 pages; A4 portrait (595.28 x 841.89 PDF points), all page rotations zero. All 81 pages were rasterized and reviewed in seven contact sheets; selected dense figure pages were inspected at enlarged scale. Programmatic text bounding-box audit found **0 pages** with out-of-canvas text; unexpected near-empty pages: **[]**. The clean-room PDF has the same 81 page texts as the working build.

**References:** Table 2.1 has 30 rows, years in descending order. BibLaTeX `ieee` + Biber `sorting=none`, 36 citation keys, no missing sources. 37 unique clickable URI destinations in PDF. Individual DOI/URL network redirect outcomes are not claimed fully retested; see detailed audit.

**Figures:** 19 LaTeX figure calls and existing image source files. Four compressed figures rebuilt in PptxGenJS and rendered as vector PDFs with scientific values maintained: Figures 4.1, 5.2, 6.4 and 7.4. Original images preserve aspect ratio. Other original plots were not regenerated. All four revised PPTX source figures passed the independent slide-overflow test. A few older, wide research figures remain smaller than ideal when scaled to portrait A4; they were preserved without stretching or altering scientific evidence.

**Signature images:** All 5 original signature image files are **byte-for-byte identical** to the uploaded archive (SHA-256 match). The user's current request explicitly confirms signatory authorization. No new stroke was drawn and no signature position was changed. Coordinator and department-head blanks remain unaltered.

**Output audit limitations:** This technical QA cannot authenticate every source medical recording, every historical checkpoint or provider authorization for restricted CAUEEG signals. Printing at actual A4 size and final institutional formatting/approval checks belong to the authors/supervisor.

**Source ZIP:** Only the current LaTeX source, bibliography, 19 selected figure assets, available editable sources, original signature bytes and up-to-date audit reports are packaged; auxiliary build files and stale previous-review artifacts are excluded.
