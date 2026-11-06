# Thesis Figure Registry


## Figures generated from existing evidence (2026-11-07)

### P2-F01

- **figure_id:** "P2-F01"
- **section:** "Results"
- **purpose:** "Display all paired differences, including intervals crossing zero."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P2 paired uncertainty"
- **input_artifacts:** ["results/p2_paired_uncertainty.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p2_posthoc_effects.png", "figures/2026-11-07_p2_evidence/p2_posthoc_effects.svg"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p2_posthoc_effects.png": "e1ee7a34c0d3e75c292987c5478cefba0116424fe58502c4b923c9b3dd6f8802", "figures/2026-11-07_p2_evidence/p2_posthoc_effects.svg": "5e2eeb8d91b0ac0da2b688f939e82e052797e456f09706dfdbaf3a4248c0b2f9"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Display all paired differences, including intervals crossing zero."
- **caveat:** "Exploratory comparisons; full retraining uncertainty not captured."

### P2-F02

- **figure_id:** "P2-F02"
- **section:** "Results"
- **purpose:** "Show whether amplitude-based quality descriptors predict diagnosis."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P2 quality negative control"
- **input_artifacts:** ["results/p2_quality_control_metrics.csv", "results/figure_sources/p2_quality_ba_plot_values.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p2_quality_negative_control.png", "figures/2026-11-07_p2_evidence/p2_quality_negative_control.svg"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p2_quality_negative_control.png": "7e11ba68c1c7fd7f17bc756ce5433973e8370b22f0ca3f4503f5b3d5b68f3bbd", "figures/2026-11-07_p2_evidence/p2_quality_negative_control.svg": "d2246ddcd941224c41535ccd8e25e0871ed0e88491f2f2e2e0cff79208ca6092"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Show whether amplitude-based quality descriptors predict diagnosis."
- **caveat:** "Near-chance prediction does not rule out acquisition confounding."

### P2-F03

- **figure_id:** "P2-F03"
- **section:** "Results"
- **purpose:** "Compare all stopping policies on observed accuracy and recording time."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P2 adaptive stopping feasibility"
- **input_artifacts:** ["results/p2_adaptive_pilot_repeat_metrics.csv", "results/figure_sources/p2_policy_tradeoff_plot_values.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p2_policy_tradeoff.png", "figures/2026-11-07_p2_evidence/p2_policy_tradeoff.svg"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p2_policy_tradeoff.png": "2036ab14d26803d5f37d505c0e49eec9d914f031162aa2ff1e3497847ab6ac8c", "figures/2026-11-07_p2_evidence/p2_policy_tradeoff.svg": "b106615b8eb815fabd2d8f0766d65e69fd6b4db6e7cafb2a723ba31deaa87fc9"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Compare all stopping policies on observed accuracy and recording time."
- **caveat:** "Repeated-CV estimates and exploratory policy choices."

### P0-F01

- **figure_id:** "P0-F01"
- **section:** "Dataset and methods"
- **purpose:** "Show actual raw EEG waveforms and physical amplitude units."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P1 raw dataset audit"
- **input_artifacts:** ["data/eeg/on004504_v1.0.0/sub-001/eeg/sub-001_task-eyesclosed_eeg.set", "results/p1_raw_recording_inventory.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p0_raw_eeg_excerpt.png"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p0_raw_eeg_excerpt.png": "e92fe252d58ec3cf156c633c5ed85bc0e028936bbe51bfba4dacf6cb3782d905"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Show actual raw EEG waveforms and physical amplitude units."
- **caveat:** "One reproducibly selected recording; not population evidence."

### P0-F02

- **figure_id:** "P0-F02"
- **section:** "Methods"
- **purpose:** "Show how the actual 30–300-second acquisition prefixes overlap."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P2 prefix extraction"
- **input_artifacts:** ["data/eeg/on004504_v1.0.0/sub-001/eeg/sub-001_task-eyesclosed_eeg.set", "results/figure_sources/p0_prefix_rms_plot_values.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p0_prefix_duration_construction.png", "figures/2026-11-07_p2_evidence/p0_prefix_duration_construction.svg"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p0_prefix_duration_construction.png": "2a435aa11da1dd0aabe46e904efa247976cb19ac0b4471364781df9784c230e5", "figures/2026-11-07_p2_evidence/p0_prefix_duration_construction.svg": "11dea3d7cb8ddba30c9ba36dbb6872be12b4e9ec6aab353ecdaa1508591b5e05"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Show how the actual 30–300-second acquisition prefixes overlap."
- **caveat:** "Nested segments are dependent observations."

### P0-F03

- **figure_id:** "P0-F03"
- **section:** "Methods"
- **purpose:** "Display real 60-second and 300-second EEG spectral estimates."
- **dataset:** "NEMAR on004504 v1.0.0"
- **source_experiment:** "P2 causal preprocessing"
- **input_artifacts:** ["data/eeg/on004504_v1.0.0/sub-001/eeg/sub-001_task-eyesclosed_eeg.set", "results/figure_sources/p0_psd_plot_values.csv"]
- **script:** "scripts/p2_generate_evidence_figures.py"
- **source_code_commit:** "463468adf7c524472f3f72f70f43e16099adcb50"
- **selection_rule:** "First sorted participant for single-recording figures; all eligible participants for group-level figures"
- **outputs:** ["figures/2026-11-07_p2_evidence/p0_causal_psd_example.png", "figures/2026-11-07_p2_evidence/p0_causal_psd_example.svg"]
- **sha256:** {"figures/2026-11-07_p2_evidence/p0_causal_psd_example.png": "148e91848b5144e3f6bc0055310d31ce360af37cc751442560c3aa494bfd5189", "figures/2026-11-07_p2_evidence/p0_causal_psd_example.svg": "91226f7812eccff98016c3ea7ac410fe3e001a05760a91dd9254ee6c04bd641d"}
- **status:** "GENERATED_AWAITING_VISUAL_QA"
- **interpretation:** "Display real 60-second and 300-second EEG spectral estimates."
- **caveat:** "Single participant; no disease-specific claim."

