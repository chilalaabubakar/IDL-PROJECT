# Classifier on classical inputs: validation (tuned)

Defects found (%), real validation defects, flagged atom within 1.5 sigma of the defect.
A, C, D: positions and species only; B, E: also atom energy and local pressure (need the
potential). See the docstring of `scripts/pilots/descriptor_classifier.py`.

| Method | vacancy | interstitial | frenkel | antisite | all |
|---|---|---|---|---|---|
| Learned finder (EGNN, calibrated) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 |
| Trees on A (odds) | 55.5 | 42.0 | 40.3 | 50.0 | 47.5 |
| Trees on B (odds) | 56.0 | 42.5 | 45.5 | 47.0 | 48.0 |
| Trees on C (odds) | 57.5 | 49.0 | 47.0 | 50.5 | 51.4 |
| Trees on D (odds) | 56.0 | 46.5 | 46.3 | 50.0 | 50.0 |
| Trees on E (odds) | 56.0 | 50.0 | 49.2 | 52.5 | 52.2 |
| Logistic regression on A (odds) | 44.0 | 32.5 | 42.5 | 41.0 | 39.8 |
| cell size | 42.0 | 16.0 | 30.6 | 12.0 | 24.7 |
| unlike neighbours | 25.5 | 29.0 | 31.3 | 31.0 | 29.0 |
| atom energy | 25.5 | 29.5 | 36.6 | 32.5 | 30.5 |
| local pressure | 8.0 | 21.5 | 16.4 | 13.0 | 14.6 |

McNemar p (exact, not corrected for multiple tests), finder against each:

- trees A: vacancy 0.845, interstitial 0.0294, frenkel 0.121, antisite 0.233
- trees B: vacancy 1, interstitial 0.073, frenkel 0.749, antisite 0.0201
- trees C: vacancy 0.832, interstitial 1, frenkel 1, antisite 0.201
- trees D: vacancy 1, interstitial 0.503, frenkel 0.839, antisite 0.175
- trees E: vacancy 1, interstitial 0.664, frenkel 0.845, antisite 0.627
- logistic A: vacancy <0.001, interstitial <0.001, frenkel 0.281, antisite <0.001
- cell size: vacancy <0.001, interstitial <0.001, frenkel 0.0038, antisite <0.001
- unlike neighbours: vacancy <0.001, interstitial <0.001, frenkel 0.0013, antisite <0.001
- atom energy: vacancy <0.001, interstitial <0.001, frenkel 0.0357, antisite <0.001
- local pressure: vacancy <0.001, interstitial <0.001, frenkel <0.001, antisite <0.001
