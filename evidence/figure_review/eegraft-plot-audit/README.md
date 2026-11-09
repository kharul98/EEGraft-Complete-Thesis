# EEGraft: figure variation candidates (visual review only)

The original thesis PDF and LaTeX source are NOT edited by this package.
Running `make_candidates.py` creates/recreates figures only in `output_candidates/`.

## What was inspected

The user-supplied 66-page EEGraft thesis PDF dated 8 November 2026.
Original figures in Chapter 6 and Chapter 7 were reviewed for repeated bar-plot design.

## Recommended figure replacements

| PDF figure | Original PNG in the thesis source | Candidate PNG | Recommended treatment |
| --- | --- | --- | --- |
| 6.1 | images/fig08_development_models.png | candidate_fig6_1_ranked_dot.png | Replace: Cleveland ranked-dot plot |
| 6.2 | images/fig09_decoder_ablations.png | candidate_fig6_2_two_panel_dots.png | Replace: matched fusion rule vs other ablations in separate panels |
| 6.3 | images/fig13_holdout_uncertainty.png | (none) | Keep the original interval/error-bar display |
| 7.1 | images/fig05_temporal_reliability.png | candidate_fig7_1_state_composition.png | Replace: mutually exclusive repeated-evaluation states as 100%-stacked composition |
| 7.2 | images/fig06_duration.png | (none) | Keep the existing line + confidence interval figure |
| 7.3 | images/fig07_mci_controls.png | candidate_fig7_3_mci_dumbbell.png | Replace: paired metric display for two SEPARATE recording releases (not paired subjects) |
| 7.4 | images/fig10_synthetic_robustness.png | candidate_fig7_4_noise_line.png | Replace: 0/10/20 noise dose trend, channel zeroed separate |
| 7.5 | images/fig11_perturbation_flips.png | candidate_fig7_5_accuracy_flip_scatter.png | Replace: BAcc vs label-change tradeoff for 3 conditions |
| 7.6 | images/fig12_channel_occlusion.png | candidate_fig7_6_channel_rankings.png | Replace: independent AD/CN rankings; no spurious channel pairing |

The image names in the second column were checked in a retained earlier Overleaf source bundle;
verify against your ACTIVE LaTeX project before replacing anything.

## Scientific integrity

All seven candidate figures use the SAVED AGGREGATE numerical values from the supplied
thesis (and its original figure script). No new subject-level observations or model results
have been created. The five perturbation repeats are NOT independent people.
The Study A 230/205 observations are correlated participant-repeat evaluations,
not 230/205 unique people. Isfahan MCI is NOT external Alzheimer's validation.
Channel masking is input sensitivity, NOT clinical or causal biomarker evidence.

The source's raw EEG artwork, original duration chart, and holdout confidence
interval chart remain unchanged. A radar plot is not used because the available
class-specific channel lists are not defined on the same axes. The full 19-channel
verified sensitivities would be needed before creating scalp topographic maps.

## Generate on Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python make_candidates.py
```

Results are PNG (300 DPI), SVG (editable vector) and a CONTACT_SHEET.png.
Do not copy any candidate into the thesis `images/` directory until reviewed.
