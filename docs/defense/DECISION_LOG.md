# DEFENSE DECISION LOG

This file records major choices before final results are inspected.

For each major decision we record:
- WHAT
- HOW
- WHEN
- WHY
- WHERE it affects the pipeline
- alternatives considered
- limitations
- evidence that could change the decision
- likely examiner question
- concise defense answer

---

## D001 - Do not continue DARWIN as the primary thesis

WHEN:
2026-11-06, before final thesis experiments.

WHAT:
DARWIN handwriting was downgraded from primary direction to fallback.

HOW:
A focused current literature search was performed after technical data
feasibility was established but before protocol freezing.

WHY:
A PLOS ONE study published on 2026-10-26 already performs a very similar
integrated analysis on the same dataset, including repeated nested validation,
task reduction, calibration, feature/task stability, and cross-validated SHAP.

WHERE:
Research-topic selection and novelty positioning.

ALTERNATIVES:
1. Continue DARWIN and attempt another classifier.
2. Continue DARWIN and slightly modify feature selection.
3. Pivot before major experimental investment.

DECISION:
Option 3.

LIMITATION:
A literature search cannot prove that no related work exists anywhere.

WHAT WOULD CHANGE THIS:
Evidence of a clearly distinct and meaningful DARWIN question with strong
scientific value and no close published precedent.

EXAMINER QUESTION:
"Why did you abandon the first dataset after already downloading it?"

DEFENSE ANSWER:
"Because dataset availability is not sufficient justification for a thesis.
Before freezing the protocol we checked the current literature and found a
peer-reviewed study published only days earlier that substantially overlapped
our intended contribution. We therefore changed direction before running the
main experiments rather than manufacture a weak novelty claim."

---

## D002 - Investigate recording-duration robustness in dementia EEG

WHEN:
2026-11-06, before downloading the main EEG derivatives.

WHAT:
Investigate whether Alzheimer's-related EEG measurements and classification
remain stable when the amount of available recording is systematically reduced.

HOW:
Use multiple fixed durations from the same participant recordings and maintain
participant-level evaluation.

WHY:
Recording duration is a directly actionable acquisition decision.
A model that performs well using a long recording may not remain useful when
only a short recording is available. The same-subject design permits paired
comparisons and avoids confusing duration effects with different cohorts.

WHERE:
Dataset construction, feature extraction, statistical analysis, and final
research question.

ALTERNATIVES:
1. Build a new deep neural architecture.
2. Compare standard classifiers only.
3. Analyze recording-duration sensitivity and feature stability.

DECISION:
Option 3 is the active candidate.

LIMITATION:
The precise duration grid cannot be selected until post-cleaning recording
lengths are inspected.

WHAT WOULD CHANGE THIS:
If too many participants lack sufficient clean EEG for paired duration
comparisons, the study will be redesigned or the fallback activated.

EXAMINER QUESTION:
"Why study recording duration instead of proposing a new neural network?"

DEFENSE ANSWER:
"The scientific uncertainty we identified concerns measurement reliability,
not architecture capacity. A more complicated network would not answer how
much EEG is required for stable evidence. We therefore chose the method that
matches the research question rather than choosing deep learning first."

---

## D003 - EEG derivative downloader correction

WHEN:
2026-11-06, during dataset feasibility testing and before protocol freeze.

WHAT:
The initial nemar-py derivative download command was abandoned and replaced
with direct retrieval from the official NEMAR git-annex dataset repository.

HOW:
The first command requested datatype=eeg with scope=derivatives through
nemar-py 0.3.3. The client returned no matching files.

The dataset's authoritative NEMAR/OpenNeuro documentation and repository were
then checked. They explicitly contain a derivatives directory holding
preprocessed and denoised EEG recordings.

The official repository's native git-annex mechanism was therefore selected.

WHY:
A downloader-specific query failure is not evidence that the scientific data
do not exist. Continuing to modify query syntax without checking the
authoritative dataset structure would waste time and could introduce an
unverified data source.

WHERE:
Data acquisition and provenance stage only.

SCIENTIFIC IMPACT:
None yet. No modeling or feature extraction had started, and no scientific
result depended on the failed download.

ALTERNATIVES CONSIDERED:
1. Continue debugging nemar-py filters.
2. Download the entire archive including unnecessary raw EEG.
3. Retrieve only the official derivative directory through git-annex.

DECISION:
Option 3.

LIMITATION:
The released derivative preprocessing is inherited from the dataset authors.
Our thesis must document that pipeline rather than claiming that we performed
the artifact removal ourselves.

EXAMINER QUESTION:
"Why did you use the provided derivatives rather than preprocess raw EEG?"

DEFENSE ANSWER:
"Our research question concerns recording-duration robustness rather than
artifact-removal algorithm design. The dataset authors publish a documented
denoised derivative generated using filtering, ASR, ICA and ICLabel. Using the
released derivative isolates our research question and makes the analysis more
reproducible within the available scope. We explicitly treat preprocessing as
an inherited dataset procedure, not as our contribution."

---

## D004 - Materialize NEMAR v1.0.0 through its published manifest

WHEN:
2026-11-06, during feasibility testing and before protocol freeze.

WHAT:
The actual EEG bytes were retrieved from the version-pinned NEMAR
v1.0.0 data manifest rather than relying on unresolved git-annex
placeholders.

HOW:
The authoritative manifest at data.nemar.org was downloaded and
only the derivatives directory plus required root metadata were
materialized. The published dataset version remained v1.0.0.

WHY:
The git repository cloned successfully but the annex content
remote did not materialize the large files locally. The NEMAR
data API explicitly provides version manifests and stable file
URLs for this purpose.

WHERE:
Dataset acquisition only.

SCIENTIFIC IMPACT:
No research endpoint, cohort definition, label, preprocessing
method, or model was changed. This was solely a data-transfer
correction before analysis.

DEFENSE ANSWER:
"We pinned the dataset release and retrieved its actual file
content through the archive's authoritative version manifest.
This preserved exact dataset provenance while avoiding dependence
on a local git-annex remote configuration problem."
