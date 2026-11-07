# Data readiness gate (must be checked before scientific training)

- [ ] Identify the exact **subject/patient count** per class; do not treat rows/windows as people.
- [ ] Confirm subject IDs across repeated EEG files and sessions.
- [ ] Verify the **actual sampling frequency** for each recording from original documentation.
- [ ] Verify units (V / mV / microvolts / already-normalized) and all 19 signal channels.
- [ ] Confirm electrode references (referential, bipolar, average); align with target montage.
- [ ] Confirm whether healthy/mild/severe are true clinical diagnostic labels or other severity groupings.
- [ ] Check original dataset license, access, consent, provenance and allowed usage.
- [ ] Check for subject overlap with any pretrained corpus when claiming out-of-dataset generalization.
- [ ] Determine whether 4-second windows are feasible. Prefer same windowing for all methods.
- [ ] Keep patient-level split locked and save `subject_splits.csv`.
- [ ] Inspect class imbalance and how many TEST patients actually exist in each class.
- [ ] If using common-average reference, justify and document it. Do not silently change it.
- [ ] Check amplitude scaling (`microvolts / 100`) and evaluate 0.3-40Hz preprocessing sensitivity.
- [ ] Obtain official CBraMod checkpoint and confirm strict load and SHA-256 before using the term pretrained.
- [ ] Never report smoke-test outputs, fake-data accuracy or randomized weights as trained thesis results.
