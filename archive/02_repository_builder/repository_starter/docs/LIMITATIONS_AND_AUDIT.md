# Limitations and issues for human review

This repository is an **archival and navigational consolidation**, not a new experiment. Creating it does not validate or change scientific results.

1. The original 64-person analytical cohort and the later 65-person cohort have not been reconciled at the participant-ID level. Never report them as one combined sample.
2. The five-minute label-stability investigation belongs to the earlier SVM track. Later DSRF noise/channel tests are a different reliability protocol.
3. The 46-person nine-model study is a distinct analysis, not directly equivalent to the 52-person development evaluation.
4. Original SVM first-minute and temporal-stability claims are recorded in retrospective audit summaries; full row-level provenance reconciliation is still desirable.
5. The held-out set is 13 people from the same source dataset; no independent prospective or clinical validation was performed.
6. XGBoost was stronger than DSRF on the final holdout. DSRF is implemented custom architecture; architectural novelty and superiority are not established.
7. Masking scores show model sensitivity, **not causal biomarkers**.
8. Two Isfahan EEG releases concern MCI versus controls, not independent AD diagnosis; units and participant linkage require caution.
9. External Figshare AD data had incompatible formats; file counts must never be stated as participant counts.
10. Paper-level bibliographic metadata in the thesis should be checked by a researcher against publication sources before submission; layout compilation does not guarantee citation accuracy.
11. The Overleaf package contains 13 figure positions, but only part of the earlier hand-approved image set was recovered; the remainder were rebuilt from saved experimental numbers. Exact historical visual identity is not verified.
12. When importing complete Git histories, historical files may include personally identifying or sensitive data even if current branches appear clean. Private visibility is recommended until audited.

These constraints are preserved intentionally rather than rewritten away.
