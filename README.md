# Placing Defects at Chosen Sites in a Model Glass with a Diffusion Model

Course project for 11-785 Introduction to Deep Learning, College of Engineering, Carnegie
Mellon University Africa. Stacy Cheruto, Nicholaus Ladislaus, Abubakar Chilala, Gentille
Uwera and Michel Sangwa; supervised by teaching assistant Boniface Godwin.

This branch (`main`) holds the first study, reported in the midterm report. The final study,
which finds and repairs standard point defects, is on the branch
[`dabo-point-defects`](https://github.com/chilalaabubakar/IDL-PROJECT/tree/dabo-point-defects).
The two branches are kept apart.

## The question

A defect in a glass is a small group of atoms arranged differently from the rest. It decides
how a glass breaks, ages or absorbs light, so studies need many realistic copies of the same
defect at chosen places. Today one either waits for it in a slow melt-and-cool simulation or
inserts it by hand, which can leave unnatural surroundings.

We train a diffusion model to rebuild a small region of an existing glass with a requested
defect at its centre, like image inpainting, and check after relaxation whether the defect is
really there and whether its surroundings look natural. The work builds on DM2
([Yang & Schwalbe-Koda, npj Comput. Mater. 2026](https://doi.org/10.1038/s41524-025-01901-1);
[code](https://github.com/digital-synthesis-lab/DM2)).

## Reports

- [Midterm report](paper/midterm_report.pdf): the study in the format of the course rubric.
- [Workshop paper](paper/main.pdf): an 8-page write-up of the 2D results.
- [Team summary](docs/Project_Summary.docx) (Word), in plain language.
- Plans: [project](docs/PROJECT_PLAN.md), [implementation](docs/IMPLEMENTATION_PLAN.md),
  [dataset](docs/DATASET.md), [3D silica](docs/SILICA_3D_PLAN.md) and its
  [quickstart](docs/SILICA_QUICKSTART.md), [compute estimate](docs/compute/README.md).

## Data

- **Glass.** The two-dimensional Kob–Andersen mixture: 65% large A and 35% small B atoms with
  Lennard-Jones forces in a periodic box. Each glass is melted, cooled and relaxed with LAMMPS:
  800 training, 100 validation and 100 test glasses of 256 atoms; 2,000 glasses of 64 atoms
  for a symmetry test; and test sets of 1,024 atoms and of slower cooling for generalisation
  ([`data/`](data/README.md)).
- **Defects,** defined by counting neighbours with rules fixed on the training glasses:
  **D1−** an A atom with at most 5 neighbours (1.7 per glass), **D1+** an A atom with at
  least 8 (1.4 per glass), **D2** a B atom with at least two B neighbours within 1.2σ
  (10 per glass). A library of 10,545 natural defects from the training glasses is used for
  hand insertion.

## Model and baselines

- **Model.** A periodic EGNN denoiser trained as in DM2, with four additions: a defect label
  at the requested site, held atoms outside the region, the noise level as an input, and
  classifier-free guidance of strength *w*. Sampling uses DM2's sampler with 100 steps.
- **Baselines.** *Hand insertion*: paste a natural defect with its neighbours within 1.5σ,
  then relax. *Local melt-quench*: melt and cool only the region in LAMMPS. *Chance*: fill the
  region with the model without a label.
- **Experiments.** 13 trained models: two unlabelled, three seeds for each of two noise
  ranges, three for a symmetry test (EGNN, MPNN, flat MLP) and two with a range of region
  sizes.
- **Metrics.** *Success*: after relaxation, an atom of the requested type within 0.5σ of the
  site carries the requested label. *Local W1*: distance between the atom energies within 3σ
  of the site and those around natural defects; natural against natural gives the floor.
  95% bootstrap intervals over 256 requests.

## Results

Rebuilding a region inside an existing glass. Model rows: mean ± standard deviation over three
training runs (D1+: one run).

| Method | D2 success | D2 local W1 | D1− success | D1− local W1 | D1+ success |
|---|---|---|---|---|---|
| Chance (no label) | 0.03 | 0.062 | 0.02 | 0.053 | 0.00 |
| Local melt-quench | 0.05 | 0.065 | 0.00 | 0.024 | 0.01 |
| Hand insertion | 0.30 | 0.087 | 0.06 | 0.032 | 0.02 |
| Model, w = 2 | 0.45 ± 0.11 | **0.051** | **0.22** | **0.022** | 0.03 |
| Model, w = 6 | 0.71 ± 0.08 | 0.072 | 0.21 | 0.038 | 0.04 |
| Model, w chosen per model | **0.74 ± 0.04** | 0.076 | | | |
| Natural floor | | 0.013 | | 0.016 | |

- The model places D2 defects 1.5–2.4 times as often as hand insertion and D1− defects 3.5
  times as often, with surroundings closer to natural ones.
- Stronger guidance places more D2 defects but makes their surroundings less natural.
  Choosing *w* for each trained model on validation glasses halves the spread between runs.
- No method places D1+. A natural D1+ moved into another glass with its 47 nearest atoms
  disappears 97% of the time: it is held in place by glass more than 3.5σ away.
- Successful D2 placements survive 250 time units of molecular dynamics in 87% of cases
  (hand insertion 91%, natural defects 95%). On 100 new glasses D2 success is 0.36–0.51
  (hand insertion 0.23).
- The EGNN makes more natural glasses than the MPNN; the flat MLP does not learn. DM2's own
  silica model, rerun by us, reproduces its published structure.

Analyses: [round 1](results/colab_20261004-2352Z/ANALYSIS.md),
[round 2](results/colab_20261005-1006Z/ANALYSIS.md), [round 3](results/round3_ANALYSIS.md),
[training seeds](results/seeds/README.md), [sensitivity checks](results/sensitivity/README.md),
[physical checks](results/credibility_followup_20261018/README.md),
[DM2 reproduction](results/stage0/README.md).

## What came next

After the midterm, Prof. Ismaila Dabo reviewed the project. He pointed out that these defect
types follow no standard concept in materials science and that defects can be made without a
learned model. The final study therefore makes the four standard point defects with physics
and uses deep learning to *find* them; it is on the branch `dabo-point-defects`.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS, ",silica" for the 3D model
ruff check src tests scripts && pytest -q
```

LAMMPS from pip needs `libmpi.so.12`, which the `mpich` wheel provides; put its directory on
`LD_LIBRARY_PATH` ([`docs/DATASET.md`](docs/DATASET.md), section 4).

## Run

```bash
# data: committed in data/; to regenerate (about 30 CPU-minutes on 4 cores, identical result)
python scripts/make_dataset.py      --config configs/data/ka2d_256.yaml
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256
python scripts/build_patches.py     --config configs/defects/v0.yaml data=data/ka2d_256
# train: without labels, then with labels
python scripts/train.py --config configs/train/uncond.yaml
python scripts/train.py --config configs/train/cond.yaml
# sample, relax, evaluate (once per method, defect and task)
python scripts/sample.py   --config configs/sampler/cfg.yaml ckpt=runs/<cond>/ckpt.pt data=data/ka2d_256 defect=D2 request.mode=host
python scripts/relax.py    --config configs/eval/default.yaml run=runs/<sample-run>
python scripts/evaluate.py --config configs/eval/default.yaml run=runs/<sample-run> data=data/ka2d_256
python scripts/aggregate_results.py --config configs/eval/default.yaml runs_dir=runs
```

The GPU runs were made on Colab and Kaggle; [`notebooks/`](notebooks/README.md) lists each
notebook and what it runs. The generalisation runs on larger and differently cooled glasses
are described in [`docs/GAP2_GENERALIZATION.md`](docs/GAP2_GENERALIZATION.md) (Kaggle),
[`docs/GAP2_COLAB.md`](docs/GAP2_COLAB.md) and [`docs/GAP2_BRIDGES2.md`](docs/GAP2_BRIDGES2.md).

## Layout

```
configs/        data, defects, model, training, sampler and evaluation settings
data/           the generated glasses and DM2's silica glasses (see data/README.md)
docs/           plans, the feasibility pilot, the team summary
notebooks/      Colab and Kaggle notebooks, the course proof-of-concept notebook
paper/          midterm report and workshop paper: LaTeX, figures, figure scripts
results/        tables, figures and analyses of every experiment round
scripts/        one command per pipeline step; scripts/silica/ for 3D
slurm/          PSC Bridges-2 jobs
src/glassdiff/  the library: periodic EGNN, MPNN, MLP, DM2 sampler with clamping,
                RePaint and guidance, Kob-Andersen potential, FIRE relaxation,
                defect detectors, metrics, hand insertion, 3D silica
tests/          unit, acceptance and end-to-end tests (CPU)
```

## Team

| Workstream | Member |
|---|---|
| Data and physics | Nicholaus Ladislaus |
| Defects and evaluation | Stacy Cheruto |
| Models and training | Abubakar Chilala |
| Sampling and conditioning | Gentille Uwera |
| Baselines and writing | Michel Sangwa |
| Supervision | Boniface Godwin |
