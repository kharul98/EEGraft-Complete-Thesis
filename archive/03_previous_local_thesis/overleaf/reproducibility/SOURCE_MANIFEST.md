# EEGraft evidence provenance

Verified from connected public GitHub repository `kharul98/eeg-dementia-thesis-backup-20261107T215131Z` (main branch, Nov 8 2026).

1. `PROJECT_HISTORY.md`: Phase A original 64-person temporal SVM; Phase B Isfahan MCI; Phase C Figshare file audit; Phase D 65-person CBraMod DSRF; Phases E-F explainability and perturbations; Phase G locked test. The historical 64 and 65 participant identity lists are not reconciled.
2. `source/alzheimers-thesis/reports/thesis_final/milestone_7a_thesis_evidence_v1/THESIS_EVIDENCE_BRIEF.md`: early 64-person temporal summary and feasibility caveats. Explicitly says original machine-readable temporal summary still needs reconciliation.
3. `source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/thesis_execution_package_2026_v1/thesis_results_ready.md`: development table, matched blend ablation, five-seed perturbation means, channel sensitivity, final test summary.
4. `source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/final_protocol_2026.json`: locked DSRF and XGBoost settings, participant splitting, SHA256 of pretrained weights and data manifest.
5. `source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/locked_final_test_2026_v1/final_metrics.json`: test confusion matrices, per-class recall, accuracy, macro-F1, AUROC, Brier, ECE, paired bootstrap/McNemar, SHA256 artifacts.
6. `source/alzheimers-thesis/reports/thesis_final/audit/verified_model_metrics.csv`: historic 46-person model comparison; not matched to later 65-person protocol.

Original older repository: `https://github.com/kharul98/alzheimers-eeg-duration-thesis` and `docs/FIGURE_PLAN.md`, `docs/FIGURE_REGISTRY.md`. Its Figure Plan requires real derived evidence, selection rules and figure provenance; registry entries for some early true EEG data figures are `GENERATED_AWAITING_VISUAL_QA` rather than verified.

## Important restrictions
- The manuscript does not include a complete executable dataset or downloaded pretrained weights. It is not, by itself, a full environment reproduction bundle.
- The early 64-person temporal source-table and participant-ID reconciliation need a final independent audit.
- Exact original 13 finalized illustrations were not present in the current attachment batch. Four were recoverable from prior accessible images, nine plots redrawn from recorded aggregate numbers, and the figure provenance manifest identifies which. Do not present those nine as recovered original images.
- The source and user-supplied materials sometimes use different cohort sizes due to separate experiments. Do not collapse them into one 65-person sample.
- The later `final_protocol_2026.json` may carry `heldout_status: NOT_EVALUATED` because it was saved *before* evaluation. The separately saved `final_metrics.json` documents that the locked holdout was subsequently evaluated.
