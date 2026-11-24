# Classifier on classical inputs: validation

Defects found (%), real validation defects, flagged atom within 1.5 sigma of the defect.
A: positions and species only; B: A plus atom energy and local pressure (needs the
potential); C: A plus radial densities (positions and species only). Made by
`scripts/pilots/descriptor_classifier.py run`.

| Method | vacancy | interstitial | frenkel | antisite | all |
|---|---|---|---|---|---|
| Learned finder (EGNN, calibrated) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 |
| Trees on A (odds) | 55.5 | 38.5 | 41.0 | 49.5 | 46.6 |
| Trees on B (odds) | 56.0 | 42.5 | 45.5 | 47.0 | 48.0 |
| Trees on C (odds) | 56.5 | 43.0 | 41.8 | 48.5 | 48.0 |
| Logistic regression on A (odds) | 44.0 | 32.5 | 42.5 | 41.0 | 39.8 |
| cell size | 42.0 | 16.0 | 30.6 | 12.0 | 24.7 |
| unlike neighbours | 25.5 | 29.0 | 31.3 | 31.0 | 29.0 |
| atom energy | 25.5 | 29.5 | 36.6 | 32.5 | 30.5 |
| local pressure | 8.0 | 21.5 | 16.4 | 13.0 | 14.6 |

McNemar p (exact, not corrected for multiple tests), finder against each:

- trees A: vacancy 0.845, interstitial 0.0012, frenkel 0.222, antisite 0.164
- trees B: vacancy 1, interstitial 0.073, frenkel 0.749, antisite 0.0201
- trees C: vacancy 1, interstitial 0.0801, frenkel 0.291, antisite 0.0576
- logistic A: vacancy <0.001, interstitial <0.001, frenkel 0.281, antisite <0.001
- cell size: vacancy <0.001, interstitial <0.001, frenkel 0.0038, antisite <0.001
- unlike neighbours: vacancy <0.001, interstitial <0.001, frenkel 0.0013, antisite <0.001
- atom energy: vacancy <0.001, interstitial <0.001, frenkel 0.0357, antisite <0.001
- local pressure: vacancy <0.001, interstitial <0.001, frenkel <0.001, antisite <0.001
