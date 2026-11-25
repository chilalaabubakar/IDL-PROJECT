# Extending the point-defect study for npj Computational Materials

This branch (`npj-extensions`) starts from `dabo-point-defects` at the commit that added the
journal manuscript (`paper/npj_manuscript.tex`). The manuscript in its current form is likely
to be rejected for five reasons: the classical measures are not trained while the finder is;
the glass is ideal (2D, zero temperature, one defect of known type); the finder fails when the
atoms are slightly displaced; the main finder was trained once and the test set was looked at
during development; and there is no link to real materials. Each item below answers one of
these points.

| # | Item | Question it answers | Compute | Suggested owners | Needs |
|---|---|---|---|---|---|
| 1 | Fresh test set, three seeds | Do the results hold on defects nobody has looked at, over training runs? | 15 min CPU; 2 × 4-5 h GPU | Nicholaus, Abubakar | none |
| 2 | Trained classifier on classical inputs | Does the finder still lead when the classical side is also trained? | hours, CPU | Stacy, Michel | 1 |
| 3 | Displaced and finite-temperature glasses | Does the finder work when atoms vibrate? | GPU hours, MD on CPU | Nicholaus, Abubakar | 1 |
| 4 | Presence and type of defect | Can the finder tell whether and what, not only where? | GPU hours | Abubakar, Stacy | 1 |
| 5 | Three dimensions, silica, images | Does it carry over to a real material? | weeks | whole team | 1-3 |

Owners follow the team's workstreams and are only a suggestion.

## Rules for this branch

1. **Make the fresh test set first and do not look at it.** Every choice (settings, rules of
   the classical measures, when to stop training) is made on the validation glasses. The fresh
   test set is scored once, at the end, for the final version of each method.
2. **Write the success test before the run.** Each item below states what result would
   support the claim and what would weaken it. Report the result either way.
3. **Three seeds for every trained model** that enters a table, reported as mean and spread.
4. **Every number comes from a file.** Each analysis writes `results/<name>_<date>/` with a
   `README.md` and a JSON file; the manuscript reads the numbers through `\pd{...}` macros
   (`paper/make_manuscript_tables.py`), as now.
5. **Same success tests as the manuscript:** found = flagged atom within 1.5σ of the defect;
   restored = same inherent structure after correction and minimisation; intervals by
   bootstrap over glasses; McNemar with Holm's correction.

## 1. Fresh test set and three seeds

**Why.** The main finder was trained once, and some choices were made after earlier test
results had been seen (manuscript, Discussion). A new test set and repeated training remove
both objections.

**Data (made on 24 November 2026, not yet scored).** 200 glasses (none rejected; energy
per atom -3.680 ± 0.013, as the training glasses) and 400 defects of each type, all relaxed
below a largest force of 10⁻⁸. 117 Frenkel pairs undid themselves (29%, against 25% in the
original test set), which leaves 1,483 real test defects: 400 vacancies, 400 interstitials,
400 antisites and 283 Frenkel pairs. SHA256: glasses `eca8525e...`, defects `0af9d167...`
(full values in the two `meta.json` files). They were made with:

```bash
python scripts/make_dataset.py --config configs/data/ka2d_256_fresh.yaml        # 200 glasses, seeds from 30000
python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects_fresh.yaml splits=[test]
```

The first makes 200 new test glasses by the same recipe, with seeds never used before (train
0-, validation 10000-, test 20000-). The second makes 2 defects of each type per glass, with
new sites (seed 1), so 400 per type before the undone Frenkel pairs are removed. Twice the
glasses of the old test set narrows the intervals from about ±7 to about ±5 points. Before
scoring, check that the files still match the SHA256 in `meta.json`.

**Training.** The main finder is two stages: `configs/train/detect_point_defects.yaml`
(20,000 updates from the denoiser, seed 0), then `configs/train/detect_point_defects_v2.yaml`
(40,000 at half the learning rate, starting from stage 1). Train two more runs of both stages,
with new seeds and `init_from` of stage 2 pointing at the matching stage 1. The notebook
`notebooks/kaggle_finder_seeds.ipynb` does both stages for runs 2 and 3 (seeds 4-5 and
6-7; the first run used 0 and 1) and pushes `results/kaggle_seeds_<date>/`. Do the
same for the transformer and the CNN if they stay in the paper (their three seeds exist; only
the scoring on the fresh set is new).

**Scoring.** `scripts/pilots/recheck_real_defects.sh` with the data path changed to the fresh
set, for the three finder seeds and the classical measures (about 2 h on 4 CPU cores per
finder for the restored test).

**Success test.** The claim holds if, averaged over the three seeds, the finder leads the best
classical measure for vacancies, interstitials and antisites with Holm-corrected p < 0.05. Any
type where it does not is reported as such.

## 2. Trained classifier on classical inputs

**Why.** The finder learns from 6,400 labelled defects per type; the classical measures are
fixed rules on one atom. A reviewer will ask whether a classifier trained on classical inputs
does as well. If it does, the paper is about learned features against hand-made ones, not
about learning against classical physics.

**Inputs per atom.** Species; Voronoi cell area, number of neighbours and number of unlike
neighbours (each as a z-score for the species); and the same three averaged over the first
neighbour shell, with the shell's minimum and maximum. Version A uses only these (positions
and species, the fair match to the finder). Version B adds the atom energy and the local
pressure and their shell averages (needs the potential). Version C adds species-resolved
radial counts within 2.5σ in 16 bins, the same input the transformer starts from.

**Model.** Gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingClassifier`) and,
as a simple reference, logistic regression; five roles, the same class weights as the finder;
flag the atom with the highest probability of the role, as the finder does. Tune on the
validation set only. Subsample intact atoms in training if memory is short.

**Reuse.** `scripts/pilots/baseline_routes.py` (`voronoi_area`, `unlike_neighbours`,
`species_stats`, and `evaluate`, which also runs the repair), `scripts/pilots/structural_signatures.py`
(local pressure), `src/glassdiff/physics/ka_potential.py` (`ka_virial`, per-atom energy).

**New.** `scripts/pilots/descriptor_classifier.py`, writing `results/descriptor_classifier_<date>/`;
on Kaggle, `notebooks/kaggle_descriptor_classifier.ipynb`.

**Success test.** If version A comes within 5 points of the finder for a defect type (on the
fresh test set, mean over seeds), the manuscript drops the claim that learning beats
classical measures for that type and reports the comparison of learned and hand-made
features instead. If the finder still leads by 10 points or more, the main claim stands, and
it is much stronger than now.

**Status (24 November 2026), validation only:** `results/descriptor_classifier_20261124/`.
Trees tuned on 100 held-out training glasses find 50.0% of the validation defects from
positions and species alone and 52.2% with energy and pressure; the finder finds 52.2% and
the best single measure 30.5%. No difference is significant for any defect type. The
finder's lead in the manuscript comes from training on labelled defects, not from the graph
network. The fresh test set decides at the end.

## 3. Displaced and finite-temperature glasses

**Why.** A random displacement of 0.02σ lowers the finder from 54% to 38% of defects found
(manuscript, Fig. 4c). At any finite temperature, or on measured positions, it would fail.

**Two kinds of input.**
- *Displaced:* every atom shifted by Gaussian noise of standard deviation 0.01-0.1σ. The test
  exists: `scripts/pilots/final_report_analysis.py sensitivity`.
- *Thermal:* short molecular dynamics of each defected glass at fixed temperatures well below
  the glass transition (for example T = 0.05, 0.1 and 0.2; check the glass transition of this
  mixture in Brüning et al. 2009 before fixing the values), with snapshots taken after the
  atoms have settled into vibration. Keep a snapshot only if minimising it gives back the
  defected glass it started from (the defect has not moved or healed). `data/ka2d_256` already
  stores a T = 0.01 snapshot of every defect-free glass (`save_thermal`).

**Training.** Add an option `input_noise` to `scripts/train_point_defects.py` that shifts the
input atoms by Gaussian noise of a random size in each batch (for example uniform in 0-0.1σ),
and fine-tune the main finder from its weights. Then try training on thermal snapshots.
Train the classical classifier of item 2 on the same inputs so the comparison stays fair.

**Reference.** Minimising the displaced or thermal glass first and then running the finder
needs the potential, but shows how much is lost by reading vibrating atoms directly.

**New.** `input_noise` in `scripts/train_point_defects.py`; `scripts/make_thermal_defects.py`
(LAMMPS, `glassdiff.md`); `results/thermal_finder_<date>/`.

**Success test.** At 0.02σ the trained finder loses at most 5 points; at the thermal
temperatures it still leads the best classical measure, scored on the same inputs.

**Status (24 November 2026).** Displaced inputs are in place: `input_noise` and
`val_input_noise` in `scripts/train_point_defects.py` (tested),
`configs/train/detect_point_defects_displaced.yaml` (continues the main finder, up to
0.1σ), `notebooks/kaggle_finder_displaced.ipynb` (three runs on a GPU, then the
evaluation), the tuned trees of item 2 retrained on displaced inputs
(`descriptor_classifier.py run --input-noise 0.1 --tag displaced`) and
`scripts/pilots/displaced_inputs.py` (every method against the size of the displacement,
validation glasses). Thermal snapshots are not started.

**First result (validation, `results/displaced_inputs_20261124/`).** Every model trained on
relaxed glasses breaks down: at 0.02σ the main finder finds 39.5% (52.2% without shifts)
and at 0.05σ 16.5%. Trees trained on displaced inputs lose 1-4 points without shifts but
keep 44.7-46.2% at 0.02σ and about 40.5% at 0.05σ. The finder trained on displaced inputs
(Kaggle) is scored next to them with the same shifts.

**Finder trained on displaced inputs (25 November 2026, validation).** Three runs: 49.2% of
defects without shifts, 47.0% at 0.02σ, 43.1% at 0.05σ, 33.6% at 0.1σ (main finder: 52.2,
39.5, 16.5, 8.2). It leads the trees trained on the same inputs by 1-2 points at 0.02σ (not
significant), 2.5 at 0.05σ and 4-5 at 0.1σ (significant in every run). Success test passed
for independent shifts; thermal snapshots below.

**Thermal snapshots (25 November 2026, validation, `results/thermal_snapshots_20261125/`).**
Molecular dynamics at T = 0.02, 0.05, 0.1 and 0.2 for 10 time units; 87%, 75%, 61% and 43% of
the defects survive. The finder trained on displaced inputs finds 45.4%, 43.4%, 40.9% and
28.7% of the surviving defects (main finder 46.4, 41.2, 34.5, 22.9) and leads trees trained on
the same inputs by about 3, 4, 6 and 1 points (significant in every run at T = 0.1). Thermal
motion is where the learned network gains most on hand-made inputs.

## 4. Presence and type of defect

**Why.** The finder is told the type of defect and assumes there is one. On a mix of
defect-free and defected glasses its highest probability separates the two with an area
under the ROC curve of only 0.75.

**Model.** Add a head on the finder's features for the whole glass: the mean and the maximum
of the atoms' features, then a small network with five outputs (no defect, vacancy,
interstitial, Frenkel pair, antisite). Train it with the role head; defect-free glasses are
the negative examples (the batch sampler already draws them with probability 0.2).

**Watch the atom count.** A clean glass has 256 atoms, a vacancy 255 and an interstitial 257,
so the number of atoms gives these two types away. Remove the cue before claiming anything:
make glasses with different numbers of atoms (for example 240 to 272; note that
`lammps_quench.make_glass` starts from a square lattice, so N must be a perfect square or the
start must change), or report results with and without the count.

**Labels for Frenkel pairs.** The operation that made a Frenkel pair does not always describe
what is left after relaxation (only 66 of 150 test pairs are still pairs). Train and score
the type both with the operation and with the site-occupancy class
(`glassdiff.data.point_defects.site_occupancy`).

**Success test.** Area under the ROC curve for "has a defect" above 0.9; type accuracy
reported per type; found and restored measured end to end, with the predicted type.

## 5. Three dimensions, silica and images

Do these in order; each step uses what the earlier ones built.

**(a) 3D Kob-Andersen glass.** The standard 80:20 mixture in three dimensions. The periodic
EGNN and the potential work with any dimension (the EGNN was used in 3D for DM2's silica);
the LAMMPS quench in `src/glassdiff/md/lammps_quench.py` is written for 2D (`dimension 2`,
square lattice start) and needs a 3D version. The four defects and the success tests carry
over directly.

**(b) Amorphous silica.** Stage 1 is done: `docs/SILICA_POINT_DEFECTS.md` and
`results/silica_point_defects_20261110-1342Z/` show that the standard tools (coordination,
energy) find a minority of oxygen vacancies, interstitials, Frenkel pairs and silicon
vacancies, while repair from the true location mostly works, so finding is again the
bottleneck. Next: more glasses (DM2's training glasses in `data/sio2_dm2`), a denoiser
(`scripts/silica/train.py`, `configs/silica/train_uncond_dm2.yaml`) and a finder trained on
silica defects, scored against the same tools.

**(c) Images of two-dimensional silica.** Huang et al. (2012) imaged a two-dimensional silica
glass atom by atom in the electron microscope. Steps: find out whether their atomic positions
or images are available and under what terms (paper's supplement, or ask the authors);
choose a model of two-dimensional silica from the literature and simulate glasses with
defects; draw them as images like the CNN finder's input, adding blur and noise like the
microscope; train the CNN finder on them and apply it to the measured images. Without exact
answers in the measured images, report what is found and compare with what the authors
identified. This is the result that would interest the editors most, but it depends on data
access, so start (a) or (b) while that is settled.

**Data sources found (25 November 2026).**
- Huang et al., Nano Lett. 12, 1081 (2012), TEM on graphene (Cornell): no coordinate file is
  public; the journal's supplement page could not be read automatically. Sadjadi et al.,
  Phys. Rev. B 96, 201405(R) (2017) refined the coordinates of its large sample (9,492 Si and
  19,330 O atoms, 270 x 270 Å², 1,811 rings): "refined coordinates ... (available upon
  request)"; they thank "Berlin and Cornell groups for the coordinates of their networks".
  Ask the Cornell group for the coordinates and images, and the authors of the 2017 paper for
  the refined set.
- Lichtenstein, Heyde, Freund, J. Phys. Chem. C (2012) and Phys. Rev. Lett. (2012), STM on
  Ru(0001) (Berlin): according to Dirindin and Coslovich (arXiv:2410.10500, 2024), the
  supplements list atom coordinates of three bilayer samples (82, 67 and 290 rings; Tables S3,
  S4 and S2). Public now; download the supplements and check.
- Kumar et al., J. Phys.: Condens. Matter (2014): two larger samples on graphene shown only as
  ring figures (444 and 449 rings), digitized by Dirindin and Coslovich, who state their data
  will be on Materials Cloud.
These are real structures without exact defect labels: they can test whether a finder trained
on simulated silica flags what the authors identify, not score it against an exact answer.

## When an item is finished

Add its numbers to `paper/make_manuscript_tables.py`, update the manuscript and the
Supplementary Information, and rebuild:

```bash
python paper/make_manuscript_tables.py
cd paper && latexmk -pdf -outdir=build npj_supplementary.tex npj_manuscript.tex
cp build/npj_supplementary.pdf build/npj_manuscript.pdf .
```

Items 1 and 2 change the main claim and Table 1; item 3 changes Fig. 4 and the limits; item 4
adds a result section; item 5 adds a section, or becomes its own paper.
