# Milestone 3E: Descriptive failure-mode audit

- Uses only frozen 30-second paired features and held-out
  EC/EO predictions. No retraining.
- All five correlated CV repetitions are kept within
  each participant during bootstrap resampling.
- Model agreement and the four transition states use
  the original hard SVM predictions.
- Standardized spectral shifts are descriptive and use
  the EC standard deviation across 64 selected people.
  This normalization is not used for model fitting.
- Uncertainty intervals are conditional on selected
  participants and previously fitted models.
- Eyes-open state, intermittent photic stimulation,
  recording position and preprocessing are confounded.
  Spectral changes do not identify a causal mechanism.
- EO quality screening excluded 10 participants from
  the original 64. This is not independent validation.
- Exploratory analysis without confirmatory hypothesis
  testing or multiple-comparison significance claims.
