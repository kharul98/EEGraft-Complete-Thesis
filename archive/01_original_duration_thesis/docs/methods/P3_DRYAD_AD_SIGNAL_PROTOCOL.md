# P3 Dryad AD Signal Feasibility Protocol

Status: Exploratory external-cohort development.

Source: Dryad 10.5061/dryad.8gtht76pw.

Use the first 300 seconds of each eligible 19-channel
recording reference. Sampling rate assumed to be
500 Hz, as described in the original publication.

Processing:
- Causal fourth-order Butterworth 1-40 Hz.
- Ten-second chronological epochs.
- Four-second Welch spectral estimates.
- Median five-band relative spectral features.

Primary comparison:
0-180 seconds versus 120-180 seconds, relative to
a disjoint 240-300-second reference.

Metric:
Normalized five-band total-variation distance.

All 49 candidate long signal references are considered.
Use the first eligible recording per MATLAB column
for the primary descriptive calculation and the last
eligible recording for sensitivity analysis.

Columns are candidate clusters, NOT verified people.

No formal significance test, diagnostic classification,
or clinical inference is permitted at this stage.

Resting-state eye condition and uninterrupted temporal
continuity remain to be verified from provenance.
