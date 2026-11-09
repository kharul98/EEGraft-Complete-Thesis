# EEGraft — Final Correction-Only Checklist

**Base:** `EEGraft_Final_Publication_Ready_Overleaf.zip` (81-page 9-chapter source)  
**Output:** `EEGraft_Final_Corrected_Overleaf.zip` and `EEGraft_Final_Corrected_Thesis.pdf`

| Requested correction | Verified outcome |
|---|---|
| Raw citation keys on former pp. 22–23 | **Fixed.** Replaced five accidental `\\cite` source sequences with valid `\cite{...}` commands in Chapter 2. Biber finds **36 cited keys**; PDF-wide text extraction finds **zero** listed raw cite-key strings or unresolved citation commands. |
| IEEE reference typography | **Fixed for case-sensitive technical terms.** BibLaTeX/IEEE numeric style preserved. Protected EEG, Alzheimer’s, CAUEEG, CBraMod, AHEPA, SHAP, SVM and other technical names in BibTeX titles. All **36 cited references** appear in citation order. The 30-paper comparison table is unchanged. |
| Almost-empty last bibliography page | **Fixed.** Bibliography occupies PDF pages **71–73**; final entry [36] appears on **page 73** with the preceding sources, not alone. Font remains **12 pt**; only bibliography leading and redundant entry gaps were compacted. |
| Figures 4.2, 4.3 and 7.4 | **Inspected** in final A4 rendering; no observed clipping, overlap or outside-page elements. Preserved existing data/graphics rather than introducing a new figure. Some small secondary labels remain more comfortable when viewed digitally at increased zoom. |
| Figure 5.1 print legibility | **Improved.** Original approved source PNG remains unchanged. The manuscript uses a larger vertical reflow of pixels from that original image; no waveform, activation or numerical result was invented. Both representation and output stages and all four component descriptions are visible on the printed page. |
| Appendix B.4 Markdown file | **Fixed.** The complete Markdown filename prints on one line on final PDF page **80**, and `revision_final/EEGraft_Scientific_Verification_Report.md` is included in the ZIP. |
| PDF metadata | **Fixed.** Title, all four student authors, Subject and Keywords are populated and verified with PDF metadata extraction. |
| Authorized signatures | **Preserved.** The five PNG files for the four students and supervisor have byte-identical SHA-256 digests to the input ZIP. PDF pages 2–3 retain the same five embedded signature images. |
| Pagination and layout | **Verified.** **80 pages**, every page 595.276 × 841.890 pt, rotation 0. No blank pages, raw citation keys, or outside-page extracted text blocks detected. Full-document low-resolution page renders and enlarged target-page renders reviewed. |
| Reproducibility | **Verified after clean extraction.** Recompiled with pdfLaTeX, Biber and MakeIndex; 36 citations resolved, no fatal LaTeX errors, no missing images. |
| Original science and structure | **Preserved.** Nine chapters, original model outputs, table values, research questions, source evidence and confirmed signatures were not edited. Only the source-level corrections and architecture print arrangement listed above were made. |
| DOI and URL verification | **Partially externally verifiable.** All **36** target addresses were extracted and checked for correct citation-key linkage; the compiled bibliography contains **36 unique clickable URI destinations**. Live access was **attempted for all 36**; some reached official publisher/proceedings/repository pages, while others were blocked, rate-limited or returned crawler errors. These network failures are **not classified as broken DOIs** without independent corroboration. The Holm 1979 DOI has Crossref-indexed corroboration, but its resolver returned an HTTP 404 in the available browsing environment. No claim of 100% live-link verification is made. |

**Submission note:** Restricted original EEG/test evidence has not been rerun as part of this LaTeX correction pass. Supervisor/institutional approval remains separate from successful compilation. This is the corrected manuscript, not an assertion of peer-review acceptance.
