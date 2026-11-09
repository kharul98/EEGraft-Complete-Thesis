# EEGraft Thesis: Formatting and Signature Audit

**Status:** Corrected LaTeX project successfully compiled and visually reviewed (2026-11-08).  
**Source PDF:** Previously delivered 63-page EEGraft thesis preview.  
**Final PDF:** 62 A4 pages, 12-point `report` class.  
**Scope:** Typography, image fidelity, figure placement, longtable layout, page breaks, and front matter. Scientific content and saved numerical findings are unchanged.

## Verified document features

- Four student names and IDs appear on independent, centered lines on the title page; IDs checked against the supplied team-details document.
- Student declaration uses four *original* signature images extracted at higher resolution and correctly assigned by inspecting the uploaded signed source page. No synthetic or redrawn signatures were produced.
- Supervisor: one named primary supervisor only, with a blank approval line. Thesis Coordinator and Head of Department have blank signature lines. No co-supervisors are added. A signature appearing on an earlier document was **not** transplanted as approval of this thesis.
- All **9 chapters**, **13 numbered figures**, **34 bibliography entries**, and the existing 30-paper literature-review table remain present.
- Bibliography `.bib` and `core/metadata.tex` are **byte-identical** to the earlier project.
- The frozen numerical values, datasets, methods, and claims in the nine chapter source files were not rewritten; figure/order/layout commands and table markup were changed only for presentation.
- No missing figures, no overfull horizontal/vertical boxes, and no undefined citation/reference warnings in the final LaTeX build. **11** benign underfull horizontal-box warnings remain in short/narrow text rows; these are not visible collisions.
- All page thumbnails were reviewed as contact sheets; selected problem pages and the first three pages were re-rendered at higher resolution.

## Page-by-page confirmed repairs

| Original PDF page(s) | Final PDF page(s) | Problem confirmed | Repair | Result |
|---|---|---|---|---|
| 1 | 1 | Thesis title, "by", student names and IDs ran into adjacent text blocks | Replaced ambiguous `\vspace`/inline layout with explicit centered paragraphs, independent author and ID rows and controlled vertical spacing | Fixed, all four IDs readable |
| 2 | 2 | Student signature crops were tiny and a second row sat too close to the first | Recovered embedded PDF images (with original alpha masks), correct person mapping, balanced dimensions, larger row separation | Fixed; no redrawn strokes |
| 3--4 | 3 | Approval committee spilled to a separate mostly empty page | Compact three-person committee layout with one supervisor and blank signature lines | Fixed on one page |
| 22--24 | 21--22 | Literature table showed cramped full justification, broken header, awkward wraps; Chapter 2 had a near-empty orphan page | Ragged-right `longtable` columns, rebalanced width, padded leading, prevented an artificial split of EEGraft, and tuned the table's pagination | Two continuous, legible table pages and no orphan |
| 25 | 23 | Chapter 3 heading split inside "Constraints" | Uniform, nonhyphenating chapter and section headings | Fixed |
| 28 | 26 | Risk-management table floated onto a sparsely occupied page | Kept the risk table with its section and used ragged-right text cells | Fixed |
| 33 | 31 | Single sentence fragment carried onto a nearly blank page | Improved float spacing/flow and pagination | Fixed |
| 17, 32, 36 and related figures | 17, 30, 34 | Detailed method and decoder diagrams were too small in portrait | Gave Fig. 1.1, 4.2 and 5.1 dedicated, properly rotated landscape pages (plus the wide duration figure 7.2) | Enlarged without changing image content |
| 45 | 44 | Fig. 7.1 title and legend overlapped | Repositioned title/legend in `source/rebuild_numeric_figures.py` and regenerated plot from the *same* arrays | Fixed; legend and bars legible |
| 47 | 46 | MCI plot title/legend were crowded | Added vertical clearance without modifying any values | Fixed |
| Various | Various | Float captions separated from their intended explanation and inconsistent white space | Figure-caption setup, local `[H]` placement of Fig. 7.1, shortened excessive float gaps and avoided orphan/widow lines | Improved |

## Current figure locations

| Figure | Final PDF page | Content |
|---|---:|---|
| 1.1 | 17 | Overall methodology (landscape) |
| 4.1 | 27 | Two analytical cohorts |
| 4.2 | 30 | EEG preprocessing (landscape) |
| 5.1 | 34 | CBraMod--DSRF decoder (landscape) |
| 6.1 | 39 | Development models |
| 6.2 | 40 | Component ablations |
| 6.3 | 41 | Held-out uncertainty |
| 7.1 | 44 | SVM temporal reliability; overlapping title/legend fixed |
| 7.2 | 45 | Duration study (landscape) |
| 7.3 | 46 | MCI negative control |
| 7.4 | 47 | Synthetic robustness |
| 7.5 and 7.6 | 48 | Perturbation flip rates and channel sensitivity |

## Changed source files

- `main.tex`
- `images/signature_towsif.png`
- `images/signature_ratul.png`
- `images/signature_iftikhar.png`
- `images/fig05_temporal_reliability.png`
- `images/fig07_mci_controls.png`
- `images/signature_zawad.png`
- `chapters/chapter_7.tex`
- `chapters/chapter_5.tex`
- `chapters/chapter_4.tex`
- `chapters/chapter_1.tex`
- `chapters/chapter_3.tex`
- `chapters/chapter_2.tex`
- `source/rebuild_numeric_figures.py`
- `core/declaration.tex`
- `core/approval.tex`
- `core/titlepage.tex`

## Remaining visual limitations

1. A few **originally rasterized** methodology diagrams still contain small internal labels, although printing them larger in landscape improves readability. Creating faithful vector sources would require the original editable design files; the existing scientific images were not recreated or altered.
2. The complete previously approved 13-image archive is not available in this project. The existing package still uses four previously supplied images plus nine plots derived from saved aggregates. This audit did not silently replace the scientific content of any of those figures.
3. Final university approval/signature authorization and semester/date confirmation remain institutional matters; committee lines were intentionally left blank.

## Separate scientific-review flags (NOT formatting errors)

- The 64-person SVM study and 65-person DSRF study have not been participant-ID reconciled; do not combine their metrics.
- The 13-person locked holdout is small and does not constitute external clinical validation.
- Historical temporal-study numbers and selected early model experiment details need source-table/code verification before claiming full independent reproducibility.
- The originally approved exact figure archive should be supplied if byte-for-byte figure identity is a university requirement.
- Bibliographic metadata and 2026 publications require a final scholar/supervisor check; these references were not altered in a formatting task.

**Scientific integrity note:** The DSRF development BAcc (77.06%) and the locked holdout DSRF/XGBoost BAcc (70.24%/77.38%) remain unchanged. No new experimental result was created or substituted during this audit.
