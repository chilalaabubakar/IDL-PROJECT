# Point defects: finding and repairing standard defects with a denoising diffusion model

A summary of the whole study in plain language, for readers without a background in
materials science or deep learning: [Point_Defects_Study_Summary.docx](Point_Defects_Study_Summary.docx).
The report: [paper/point_defects_report.pdf](../paper/point_defects_report.pdf).

## Why this study

The first version of the project asked a diffusion model to *place* defects (D1−, D1+, D2)
that we had defined ourselves by neighbour counts, chosen so that each covered 1-5% of the
atoms. Those labels follow no established defect concept, and placing a defect needs no AI:
removing an atom makes a vacancy. This study uses the **standard point defects of
computational materials science** and asks the questions where a learned model can matter:

1. **Find**: given a relaxed glass that contains a defect, which atoms are wrong?
2. **Repair**: what is the defect-free glass it came from?

The model is the one we already have: the periodic EGNN denoiser trained with DM2's loss.
That loss teaches the network to remove a perturbation from a glass. Here the perturbation is
a physical defect instead of only random noise.

## The four defects

Each defect is made by a known operation on a known defect-free glass (an inherent structure
from `data/ka2d_256`), and the defected glass is then energy-minimised with all atoms free,
so it sits in a minimum like a real defected sample. Because the operation is known, so is
the original glass: the exact target for repair. No detection thresholds are needed.

| Defect | Operation | Atoms |
|---|---|---|
| Vacancy | remove one atom (either species) | 255 |
| Interstitial | add one small B atom at the largest gap within 2σ of a random point | 257 |
| Frenkel pair | move one atom to the largest gap 1.2-3σ from its site (the radiation-damage defect: a vacancy plus an interstitial) | 256 |
| Antisite | swap the species of an A atom and its nearest B atom (the chemical-disorder defect of compounds) | 256 |

In a crystal these defects are defined against the lattice. In a glass there is no lattice,
so the defect is defined by the operation, and after relaxation it spreads over a few
neighbours; how well it can still be found is part of the question.

For each example the dataset stores (`glassdiff.data.point_defects`):

- the relaxed defected glass;
- a per-atom **role** for detection: intact, next to an empty site (vacancy, or the empty
  site of a Frenkel pair: the first-shell neighbours of the removed or moved atom, r < 1.4
  σ_ab as the dataset's neighbour rule), extra atom, displaced atom, wrong species;
- where the defect is;
- the **repair input**: the relaxed defected glass with the atom count and species put right.
  For a vacancy, an atom of the missing species is placed in the largest gap within 1σ of
  the empty site. For an interstitial, the extra atom is removed. For an antisite, the
  species are swapped back. A Frenkel pair is left unchanged. Atom i of the repair input is
  atom i of the original glass.

Finding decides *where* to add, remove or swap an atom; repair then has to move the atoms
back to the original glass.

## Data

```bash
pip install -e ".[md]"                     # LAMMPS (export LD_LIBRARY_PATH to MPICH's lib)
python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml
```

Glasses keep their split: training defects come from the 800 training glasses (8 random
sites each, 6,400 examples per defect), validation and test defects from the 100 validation
and 100 test glasses (2 sites each, 200 per defect). Minimisation is LAMMPS FIRE, as for the
dataset's inherent structures (`glassdiff.md.minimize`), about 0.1 s per example. A run is
deterministic: every (split, defect) has its own random stream and every minimisation runs
in one single-threaded LAMMPS instance. `meta.json` records the settings, counts, the
largest residual force (below 1e-8) and the SHA256 of every file; each split also pins its
source glasses by SHA256. The validation and test splits are in the repository (3 MB
each); the training split (about 25,600 examples, 95 MB) is not, and takes about 30-45
minutes to make on 4 cores (most of it the serial gap search; the minimisations are
parallel).

## Models and training

Both models are the periodic EGNN of the main study (0.53 M parameters), started from the
trained unconditional denoiser (`egnn_uncond_smax1`).

**Repair** (`configs/train/heal_point_defects.yaml`). DM2 trains on a defect-free glass x0
shaken by random noise, x = x0 + σε, and the network d_θ(x, σ) learns to predict the shift
x − x0. The best possible prediction is d = −σ² ∇ log p_σ(x) (Tweedie's formula), so the
correction −d points towards more probable glass structures: a learned restoring force. On
test glasses its direction agrees with the physical force (cosine 0.81 at σ = 0.05, closer
than the straight line back to the original, 0.68; `results/point_defects_20261107/`). We change only
what is removed. The starting point is the repair input of a defected glass,

    x = x_repair + σε,     target = x − x0   (minimum image, mean shift removed),

so the network learns to remove the noise *and* undo the defect. A fifth of the batches are
defect-free glasses (x_repair = x0), where the loss is exactly DM2's, so the model also
learns to leave intact glass alone. It is then used like DM2's sampler, starting from the
defected glass.

**Repair, version 2** (`configs/train/heal_point_defects_v2.yaml`). The same loss with each
atom the defect moved by more than 0.1σ weighted 11 times, noise drawn log-uniformly
between 0.001 and 0.3, and Frenkel pairs given with the displaced atom back in the gap at its
site; see "Results so far" for why.

**Detection** (`configs/train/detect_point_defects.yaml`). The same EGNN trunk with a
per-atom head over the five roles (`EGNNAtomClassifier`), class-weighted cross-entropy;
a fifth of the batches are defect-free glasses with every atom intact.

```bash
python scripts/train_point_defects.py --config configs/train/heal_point_defects.yaml
python scripts/train_point_defects.py --config configs/train/detect_point_defects.yaml
```

20,000 updates each, about 1.5-2 h on a T4. On Kaggle, use
[`notebooks/kaggle_point_defects.ipynb`](../notebooks/kaggle_point_defects.ipynb): it makes
the training split, trains both models (one per GPU), evaluates them and pushes the results.

## Evaluation

`scripts/eval_point_defects.py`, on the 800 test defects:

- **Find**: the top-scored atom must lie within 1.5σ of the defect. With `--calibrate` the
  finder's probabilities are corrected for its training class weights (logit adjustment). Score: the trained
  detector's probability for the role the repair needs; without a detector, the size of
  the denoiser's output at σ = 0.05 (no potential needed). Reference that needs the
  potential: the atom whose energy is highest for its species. Chance: the fraction of
  atoms within 1.5σ of the defect.
- **Repair**, then FIRE minimisation (all atoms free, max force 1e-6, the minimiser of every
  earlier evaluation) to see which energy minimum the glass settles in. **Restored** = it
  is the original glass: every original atom has an atom of its species within 0.05σ, one
  to one, so identical atoms may trade places.
  - Vacancy: put an atom of the missing species in the largest gap within 1.5σ of the
    flagged atom; minimise, or repair the region with the denoiser first.
  - Interstitial: remove the flagged atom; minimise, or repair first.
  - Frenkel pair: (a) the region around both sites is given; shake it (0.35σ) and minimise,
    or shake and repair first. Shaking alone is the physics baseline: it tests whether the
    learned model adds anything beyond a random kick. (b) Nothing is given: flag the
    displaced atom, list up to 8 gaps 1-3.2σ from it, move it to the gap the model prefers
    (the finder's "next to an empty site" probability around the gap), then minimise or
    repair and minimise; with the potential, try every gap and keep the lowest energy.
  - Antisite: swap back the neighbouring A-B pair the model prefers (or, with the
    potential, the one with the lowest energy) near the flagged atom; minimise.

```bash
# before training (the reference in results/point_defects_20261107/)
python scripts/eval_point_defects.py --healer results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt \
    --out results/point_defects_20261107/eval_untrained.json
# after training
python scripts/eval_point_defects.py --healer runs/<pd_heal>/ckpt.pt --detector runs/<pd_detect>/ckpt.pt \
    --device cuda --out <json>
```

Generation uses LAMMPS FIRE and evaluation the PyTorch FIRE. They reach the same minimum
from a defect placed in a gap (checked on test glasses), but LAMMPS FIRE cannot start from
overlapping atoms (a shaken region), and from such a start the minimum reached depends on the
minimiser; so one minimiser is used for every method in the evaluation.

## Results so far and caveats

Before any training (`results/point_defects_20261107/README.md`), the denoiser trained only
on defect-free glasses finds vacancies more often than the per-atom energy (44% vs 29.5%),
so it restores more vacancy glasses (41-43.5% vs 26%). It is weaker for the other three
defects and rarely undoes one by itself (Frenkel pairs: 8% restored vs 0.5% by shaking
alone). That is the motivation for the training above.

After the first training (`results/point_defects_trained_20261107/README.md`):

| Defect | Found by the trained finder | Found by the energy | Restored, with the trained models | Restored, energy route |
|---|---|---|---|---|
| Vacancy | **66%** | 29.5% | **51-54%** | 26% |
| Interstitial | **46.5%** | 28% | **34.5-35%** | 21% |
| Antisite | **44.5%** | 29% | **27%** | 13.5% |
| Frenkel pair | 26% | 28.5% | 10% (region given) | 0.5% (shake only) |

The finder works and beats the potential's own energies for three defects. The repair
model learned to remove noise but not the defect: a defect moves only 30-40 of the 256
atoms, and the random noise filled almost all of the loss. Version 2
(`configs/train/heal_point_defects_v2.yaml`, `notebooks/kaggle_point_defects_v2.ipynb`)
gives the defect atoms 11 times the weight, trains mostly at small noise, and puts a
Frenkel pair's atom back in the gap at its site before repair. Which gap is chosen
separately, by the finder or by energy, with no region given.

After repair version 2 (`results/point_defects_v2_20261109/README.md`):

- **Version 2 learned the defect:** in one step it removes 12-26% of the defect-atom error,
  against 6-9% for version 1.
- **It adds nothing once minimisation follows:** restored rates stay at 51.5, 34 and 9%
  for vacancies, interstitials and region-given Frenkel pairs.
- **Without a potential:** it moves the defect atoms closer to the original, but leaves
  overlapping atoms in up to half of the glasses.
- **Headroom is in finding:** given the true defect location, minimisation restores
  53-80% of glasses, against 19-51% with the finder.
- **Frenkel pairs with nothing given:** 19% are restored; a pure physics search reaches
  18.5%.

The main result is therefore the learned finder (no potential), plus fixing the atom count
or species, plus minimisation. The repair model is a secondary result with a clear failure
mode. The no-potential test is `scripts/eval_point_defect_repair.py`.

Improving the finder (`results/point_defects_finder_20261109/README.md`):

- **Calibration.** The finder's probabilities were inflated by the class weights of
  training. Subtracting the log weights (logit adjustment, `--calibrate`) cuts false alarms
  on defect-free glasses from 59% to 1% and improves finding by 1-4 points for every defect.
- **Calibrated results.** Found: 67, 47.5, 46.5 and 30% (vacancy, interstitial, antisite,
  Frenkel pair). Restored: 51, 36, 27 and 19.5%.
- **What stays hard.** Telling a defective glass from a clean one remains weak (AUC 0.72).
  For Frenkel pairs, finding the empty site first did not help (validation only).

A longer finder run (`results/point_defects_finder_v2_20261110/README.md`):

- **Training.** 40,000 more updates at a lower learning rate
  (`configs/train/detect_point_defects_v2.yaml`).
- **Small, real gain.** Found: 69.5, 50, 49 and 37% (vacancy, interstitial, antisite,
  Frenkel pair), 3.6 points more over all 800 defects (paired McNemar p < 0.001; clear for
  Frenkel pairs alone, p = 0.013). Restored: 53.5, 38, 30 and 18%, within the noise of the
  first finder's.
- **Levelled off.** Validation finding moved by 2 points or less over the last 20,000
  updates, so further gains need a change of model or input, not more training. This finder
  is the one reported from here on.

The physics recheck and the report (`results/point_defects_recheck_20261111/README.md`,
`paper/point_defects_report.pdf`):

- **A quarter of the Frenkel pairs are no defects.** The moved atom took the place of a
  neighbour of its own species, which filled the empty site, so the relaxed glass is the
  original again (50 of 200 test pairs, 27% of the training pairs).
  `glassdiff.data.point_defects.real_defects` leaves them out (`--real-defects-only` in the
  evaluation scripts, `real_defects_only: true` in a training config). **Every Frenkel number
  above this point includes them.**
- **Two baselines without a potential or learning** (`scripts/pilots/baseline_routes.py`):
  the size of each atom's Voronoi cell (free volume) and its number of unlike neighbours
  (chemical order). An antisite is a defect of chemical order: after relaxation the two
  exchanged atoms have normal cells but too few unlike neighbours.
- **The atom energy is mostly a count of A-B bonds** (correlation −0.90 and −0.84 for A and
  B atoms), so the bond count finds about as many defects as the energy, without the
  potential.
- **Final test numbers** on the 750 real defects; brackets resample whole glasses:

| Defect | Found: finder | Found: best without learning | Restored: finder | Restored: best without learning | Restored: true location given |
|---|---|---|---|---|---|
| Vacancy | **69.5%** [62, 76] | 47.5% (cell size) | **53.5%** [46, 61] | 40.5% (cell size) | 67% |
| Interstitial | **50%** [42, 57] | 28% (unlike neighbours, energy) | **38%** [31, 46] | 21.5% (unlike neighbours) | 80% |
| Antisite | **49%** [42, 56] | 31.5% (unlike neighbours) | **30%** [24, 36] | 8% (energy) | 68.5% |
| Frenkel pair (150 real) | **45.3%** [38, 53] | 32.7% (energy) | 12.7% [8, 18] | 7.3% (energy) | 42% |

  The finder's lead is significant on the same defects (McNemar) in every case except
  restoring Frenkel pairs (p = 0.17).

Ablations of the finder (`results/point_defects_ablations_20261112/README.md`): six runs of
2,000 updates, one setting changed per run, scored on the same 750 defects.

- **Starting from the denoiser matters most:** 44.7% of the defects found, against 23.7% from
  random starting weights at the same number of updates.
- **Reach:** one message-passing layer (2.5σ) is clearly worse than two or four.
- **Class weights** change how readily atoms are labelled defective, not which atom is ranked
  first.
- **Training without the annihilated Frenkel pairs** brought no gain.

To reproduce the test numbers and the report:

```bash
bash scripts/pilots/recheck_real_defects.sh   # ~2 h on 4 CPU cores, resumable
bash scripts/pilots/finder_ablations.sh       # the six ablation runs, ~4.5 h on 4 CPU cores
python scripts/pilots/finder_ablations.py     # experiments on the finder
python paper/make_report_numbers.py && python paper/make_report_figures.py
```

Caveats: a two-dimensional model glass; one defect per glass; the type of defect is given;
the finder was trained with the annihilated Frenkel pairs still labelled as defects and has
not been retrained; "restored" asks for the exact original minimum, which is strict in a
glass where many nearby minima are equally natural.

## Next: amorphous silica

The same study in amorphous SiO₂, with oxygen vacancies, oxygen interstitials, oxygen Frenkel
pairs and silicon vacancies in DM2's glasses: [SILICA_POINT_DEFECTS.md](SILICA_POINT_DEFECTS.md).

**Stage 1 is done** (`results/silica_point_defects_20261110-1342Z/README.md`), with no learned
model:

- Given the defect site, minimisation restores 76-96% of glasses.
- The standard coordination analysis (no potential) finds only 11-29% of defects and restores
  1.5-52.5%, because the relaxed defects look like the glasses' natural defects.
- The per-atom energy (needs the potential) is strong for silicon vacancies and interstitials
  only.

As in 2D, the headroom is in finding, so a learned 3D finder (stage 2) is justified.
