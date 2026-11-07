# P1 External Validation Dataset Assessment

Date: 2026-11-07
Status: Exploratory evidence review

## Confirmed primary data

NEMAR on004504 v1.0.0:
- 88 unique participants
- 36 AD, 29 controls, 23 FTD
- 19 EEG channels, 500 Hz
- All 88 raw recordings are at least 300 seconds
- 264 raw EEG and associated files verified by checksum

## Figshare 5450293

https://doi.org/10.6084/m9.figshare.5450293

The public archive contains EEG from an independently
collected AD/MCI/control cohort.

Cejnek et al. 2021 describe approximately 15-minute
sessions and preprocessing that selects artifact-free
segments of 1000 samples at 128 Hz.

Important unresolved questions:
- Are chronological recording sequences retained?
- Are original timestamps or acquisition order available?
- How many usable unique participants are represented?
- Are disease groups associated with recording hardware?
- Are file structures comparable across diagnostic groups?

Do not infer 300 seconds of continuous EEG solely
from a 15-minute original acquisition description.

## OSF correction

The previously screened OSF node jbysn is titled:

"Data for 'Accounting for the complex hierarchical
topology of EEG phase-based functional connectivity
in network binarisation'"

It is not the intended verified Alzheimer cohort.

Remove this node from the candidate validation shortlist.
Preserve the original API screening as historical evidence.

## BrainLat

https://doi.org/10.1038/s41597-023-02806-8

BrainLat is a potential independent patient cohort.

Metadata HTTP 200 does NOT prove EEG file access.

Access requirements, participant identifiers,
channel harmonization and recording quality
must be evaluated before external testing.

## Scientific decision

Primary raw dataset: VERIFIED

Independent validation: NOT YET VERIFIED

Novelty: NOT YET ESTABLISHED

Final P2 methodology: NOT YET FROZEN
