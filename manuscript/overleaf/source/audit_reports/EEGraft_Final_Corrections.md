# EEGraft Final Publication-Stage Corrections

This is a **targeted revision of the uploaded 79-page nine-chapter Overleaf source**, not a new experiment or a replacement thesis design.

| File / location | Change | Verification |
|---|---|---|
| `core/abstract.tex` | One plain-English paragraph of 129 words, led by what was built and observed; no unsupported clinical claim. | Source review and compilation |
| Chapter 1, Contributions | Five concrete engineering, experimental, reliability, threshold and confounding contributions; explicit denominator distinction. | Matched source results and Chapter 9 |
| Chapter 2, Prior work and research gap | Direct comparisons with DICE-Net, EEG Conformer, CBraMod, LEAD and 2026 EEG foundation model benchmarks; narrowed defensible novelty. | Public literature source check; Table 2.1 unchanged (30 rows, descending years) |
| Chapter 5, decoder explanation | Explained frozen pretrained path, kernels 9 and 41, gated residual and class-specific attention as testable design choices rather than invented algorithms. | Existing implementation documentation and architecture figure |
| Chapter 7, reliability synthesis | Replaced repeated generic caveats with reproducible evaluation practices. | Aggregated NEMAR, Isfahan and CAUEEG evidence |
| Chapters 8-9 | Led with substantive contributions, preserved negative source holdout and confidence intervals, consolidated threats, improved closing conclusion. | Results consistency audit |
| `references.bib` | Corrected author lists for Zheng, DICE-Net, Akbar, LEADNet and the NEMAR dataset; completed 2026 Measurement volume; added Holm indexed DOI; removed unused Breiman BibTeX record (not cited or in Table 2.1). | Publisher/PubMed/index metadata and Biber |
| Fig. 4.1 | Replaced compressed cohort-flow artwork with simplified, editable portrait vector layout using original data counts. | PDF render and counts audit |
| Fig. 5.2 | Replaced tiny horizontal protocol with vertical source -> development -> locked-test diagram; primary vs secondary rules explicit. | PDF render and locked-protocol check |
| Fig. 6.4 | Enlarged and simplified the two locked CAUEEG confusion matrices; reduced duplicate metrics. | Count/balanced-accuracy arithmetic |
| Fig. 7.4 | Separated observed development condition flips, locked-test balanced accuracy and Brier scores. | Saved Code 19-23 aggregate consistency |
| Declaration and approval | **No changes** to images or page placement; original supplied, author-confirmed signatures preserved. | SHA-256 comparison to uploaded ZIP |
| `main.tex` | Preserved A4 portrait and IEEE BibLaTeX/Biber; adjusted URL line-breaking and proportional figure-height limits. | Clean PDF render and reference audit |

**What was not done:** No new EEG training, no test-set tuning, no raw CAUEEG signal release, no invented literature or budget, no imitated signature, and no universal novelty claim. The final filename is a revision label, not an assertion of peer review or university approval.
