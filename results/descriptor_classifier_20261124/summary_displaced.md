# Classifier on classical inputs: validation (displaced)

Defects found (%), real validation defects, flagged atom within 1.5 sigma of the defect.
A, C, D: positions and species only; B, E: also atom energy and local pressure (need the
potential). See the docstring of `scripts/pilots/descriptor_classifier.py`.

| Method | vacancy | interstitial | frenkel | antisite | all |
|---|---|---|---|---|---|
| Learned finder (EGNN, calibrated) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 |
| Trees on A (odds) | 54.0 | 37.0 | 46.3 | 46.5 | 45.9 |
| Trees on B (odds) | 55.5 | 42.0 | 41.8 | 46.0 | 46.7 |
| Trees on C (odds) | 56.0 | 40.0 | 41.8 | 49.5 | 47.3 |
| Trees on D (odds) | 57.5 | 40.5 | 45.5 | 50.5 | 48.8 |
| Trees on E (odds) | 56.5 | 39.5 | 49.2 | 48.5 | 48.4 |
| Logistic regression on A (odds) | 39.5 | 31.0 | 28.4 | 27.0 | 31.7 |
| cell size | 42.0 | 16.0 | 30.6 | 12.0 | 24.7 |
| unlike neighbours | 25.5 | 29.0 | 31.3 | 31.0 | 29.0 |
| atom energy | 25.5 | 29.5 | 36.6 | 32.5 | 30.5 |
| local pressure | 8.0 | 21.5 | 16.4 | 13.0 | 14.6 |

McNemar p (exact, not corrected for multiple tests), finder against each:

- trees A: vacancy 0.487, interstitial <0.001, frenkel 0.871, antisite 0.0166
- trees B: vacancy 0.832, interstitial 0.0533, frenkel 0.291, antisite 0.0115
- trees C: vacancy 1, interstitial 0.006, frenkel 0.302, antisite 0.133
- trees D: vacancy 0.824, interstitial 0.007, frenkel 0.766, antisite 0.256
- trees E: vacancy 1, interstitial 0.0014, frenkel 0.878, antisite 0.0652
- logistic A: vacancy <0.001, interstitial <0.001, frenkel <0.001, antisite <0.001
- cell size: vacancy <0.001, interstitial <0.001, frenkel 0.0038, antisite <0.001
- unlike neighbours: vacancy <0.001, interstitial <0.001, frenkel 0.0013, antisite <0.001
- atom energy: vacancy <0.001, interstitial <0.001, frenkel 0.0357, antisite <0.001
- local pressure: vacancy <0.001, interstitial <0.001, frenkel <0.001, antisite <0.001
