# Physics recheck of the 2D point defects: a quarter of the Frenkel pairs are no defects

Made while checking the physics for the report on the two-dimensional study
(`paper/point_defects_report.tex`).

**Status, 11 November 2026:** the defect measurements below are final. The re-evaluation of
finding and repair on real defects only (`scripts/pilots/recheck_real_defects.sh`) is running;
its tables will be added here as `summary.md` and written up below.

## What was found

`defect_physics.json`, from `scripts/pilots/point_defect_physics.py`.

**Frenkel pairs that annihilated.** In a Frenkel pair an atom is moved 1.2-3σ into a gap and
the glass is relaxed. In about a quarter of the cases, the moved atom pushed a neighbour of its
own species into the empty site and took that neighbour's place. The relaxed glass is then the
original glass with two identical atoms exchanged: the same structure and the same energy.
There is no defect left to find or repair.

| Split | Frenkel pairs | The original glass again |
|---|---|---|
| Training | 6,377 | 1,735 (27%) |
| Validation | 199 | 65 (33%) |
| Test | 200 | 50 (25%) |

The generator only checked whether the moved atom had returned to *its own* site
(`recombined_tol`), so these passed as valid. Vacancies, interstitials and antisites are not
affected (none is the original again).

Consequences:

- **Test.** Every earlier Frenkel number counted these 50 glasses as defects. Finding had
  nothing to find there, and any repair step could only make the glass worse.
- **Training.** The finder was trained with 27% of its Frenkel examples labelled "displaced
  atom" and "next to an empty site" in a glass that had neither. This explains part of its
  weakness on Frenkel pairs, for example that it called most "displaced" atoms intact: in a
  third of the validation pairs they were.

**The fix.** `glassdiff.data.point_defects.already_original` detects these cases (the relaxed
glass matches the original one to one within 0.05σ, identical atoms allowed to trade places),
and `real_defects` leaves them out. The stored dataset files are unchanged. The option
`--real-defects-only` applies the filter in `scripts/eval_point_defects.py`,
`scripts/eval_point_defect_repair.py`, `scripts/pilots/compare_finders.py` and
`scripts/pilots/baseline_routes.py`; `real_defects_only: true` does it in a training config.

## What the relaxed defects are (test glasses, real defects)

Median, with the middle half of the cases in brackets.

| | Vacancy | Interstitial | Frenkel pair | Antisite |
|---|---|---|---|---|
| Real defects | 200 | 200 | 150 | 200 |
| Other atoms moved by more than 0.1σ | 14 [3, 43] | 21.5 [11.5, 40] | 44 [21, 69] | 28.5 [15, 55] |
| Largest move of another atom (σ) | 0.33 | 0.32 | 0.64 | 0.42 |
| Energy above the original (ε) | | | 3.2 [1.5, 5.0] | 2.8 [1.4, 4.3] |
| Energy higher than the original | | | 91% | 89% |
| Energy lower than the original | | | 8% | 11% |

- **Vacancies partly close.** Right after removal, the most open point at the site is 0.87σ
  from the nearest atom. After relaxation it is 0.74σ [0.66, 0.83]. Around atoms of
  defect-free glasses the same measure is 0.63σ, and 0.70σ at the 95th percentile. So 60.5%
  of relaxed vacancies are still more open than normal glass and 39.5% are not. This agrees
  with early simulations of Lennard-Jones glasses (Bennett et al., *Philos. Mag. A* 40, 485
  (1979); Delaye and Limoge, *J. Phys. I* 3, 2079 (1993)).
- **Interstitials.** The extra B atom sits 0.85σ from its nearest neighbour (B atoms in
  defect-free glass: 0.87σ) and has a higher energy (−3.0 against −3.4).
- **An antisite does not always cost energy.** For 11% of the antisites the swapped and
  relaxed glass is lower in energy than the original. In a crystal that cannot happen; a
  quickly cooled glass is not in its lowest state, and exchanging two atoms can lead it to a
  lower one (the principle of swap Monte Carlo). The original is still the target for
  repair, but in these cases it is not the more stable structure.

## Files

- `defect_physics.json`: the measurements above.
- `glass_level.json`: is a glass defective at all, with the final finder on real defects only
  (defect-free glasses flagged: 68% raw, 2% calibrated; AUC 0.75).
