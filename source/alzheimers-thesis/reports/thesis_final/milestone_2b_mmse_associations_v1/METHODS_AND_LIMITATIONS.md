# Milestone 2B: clinical severity evidence

- Analysis unit: individual patient.
- Cohorts: 23 selected AD and 18 selected FTD.
- Healthy controls excluded because MMSE is constant.
- Correlated CV predictions averaged once per participant.
- Primary exploratory outcomes: disease probability and
  temporal class-reversal rate.
- Secondary outcomes: temporal probability SD, original-label
  confidence support, and stable-wrong rate.
- Spearman correlations use within-group permutation tests
  and participant-bootstrap confidence intervals.
- Age adjustment is an approximate partial-rank sensitivity
  analysis using Freedman-Lane permutation.
- Holm correction: four primary raw tests.
- BH FDR correction: all ten raw tests.
- All tests remain exploratory because the dataset has
  already been examined during model development.
- MMSE is cross-sectional; associations are not evidence
  of causality, clinical diagnostic utility or prognosis.
- Window selection may limit severity representativeness.
- No classifiers were retrained.
