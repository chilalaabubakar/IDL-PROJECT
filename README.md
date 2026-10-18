# Controlled Defect Generation in Amorphous Materials via Conditional Diffusion

IDL group project (CMU LTI): S. Cheruto, N. Ladislaus, A. Chilala, G. Uwera, M. Sangwa.

We ask a diffusion model for a specific defect at a chosen location in a 2D binary glass
(Kob–Andersen 65:35) and check, after physical relaxation, whether it is really there and
whether its surroundings look like naturally formed ones. The work builds on DM2
([Yang & Schwalbe-Koda, npj Comput. Mater. 2026](https://doi.org/10.1038/s41524-025-01901-1);
[code](https://github.com/digital-synthesis-lab/DM2)).

## Documents
- [Project plan](docs/PROJECT_PLAN.md): research questions, method, evaluation, stages, timeline, risks
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): conventions, tickets per workstream, dependencies, MVP definition
- [Dataset plan](docs/DATASET.md): the 2D KA dataset we generate, defect definitions, external datasets
- [Feasibility pilot](docs/pilot/): LAMMPS script and measured outputs
- [Team summary](docs/Project_Summary.docx) (Word): plain-language summary of what was done, the data, models, training runs, results, next steps and compute
- [Workshop paper](paper/main.pdf) (about 8.5 pages, plain language; LaTeX source in `paper/`)
- [3D silica extension plan](docs/SILICA_3D_PLAN.md): the next step toward a materials-journal paper (defects in amorphous SiO₂, physical validation)
- [3D silica quickstart](docs/SILICA_QUICKSTART.md): data, defect labels, the conditional 3D model and how to start training it (Kaggle or PSC Bridges-2)
- [Compute estimate](docs/compute/README.md): GPU-hours, CPU core-hours and ACCESS credits for the rest of the project, from measured timings

## Status

The MVP pipeline is implemented and tested (67 tests, including an end-to-end run of
every script). It covers dataset generation, defect labelling, the periodic EGNN and
the MPNN/MLP ablations, training, the DM2 sampler with clamping, RePaint and CFG
conditioning, both baselines (hand insertion, local melt-quench), relaxation and
metrics with bootstrap CIs. See
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §2 for the ticket-by-ticket
status.

**Results** (round 1: [`results/colab_20261004-2352Z/ANALYSIS.md`](results/colab_20261004-2352Z/ANALYSIS.md),
round 2: [`results/colab_20261005-1006Z/ANALYSIS.md`](results/colab_20261005-1006Z/ANALYSIS.md),
round 3: [`results/round3_ANALYSIS.md`](results/round3_ANALYSIS.md); training seeds:
[`results/seeds/`](results/seeds/README.md)): averaged over three training runs, the
conditional EGNN places D2 defects inside existing glasses 1.5–2.4× as often as hand
insertion (0.45 ± 0.11 at guidance w = 2, 0.71 ± 0.08 at w = 6, vs 0.30), with surroundings
closer on the original all-request local-energy metric, and D1− defects 3.5× as often (0.22 ± 0.02 vs 0.06); local
melt-quench almost never succeeds. In the whole-box task it reaches D2 success 0.58 ± 0.03
with pooled local-energy W1 near its finite-sample natural reference. D1+ fails for every method. D2 success varies
between training runs more than the evaluation noise (each run ends with a different
effective guidance strength). The DM2
sampler's step count acts like a cooling rate (100 steps give the natural energy), and in
the N = 64 symmetry ladder the EGNN makes the most natural glasses while a flat MLP does
not learn. Stage 0 ([`results/stage0/`](results/stage0/README.md)): DM2's own pretrained
a-SiO₂ model, rerun on CPU, reproduces its training glasses' g(r) and tetrahedral structure,
with somewhat more disorder (like a 100× faster quench); the course proof-of-concept
notebook reproduces the proposal's Fig. 1. Sensitivity checks
([`results/sensitivity/`](results/sensitivity/README.md)): the method ranking holds when the
defect cutoffs, success radius, realism window and regenerated-disk size are varied; the
absolute success rates move by up to about 0.1.

## Physical credibility: scope of the claim

[Host-aware structure and finite-temperature follow-up](results/credibility_followup_20261018/README.md)
shows that successful D2 placements remain structurally different from natural defects.
For the tested radius-6 setting, host-aware successful-only energy W1 is 0.029 versus a
count-matched natural median of 0.012, and outer-host displacement is larger than for
hand insertion. At T=0.01 after 50 LJ time units, the tracked D2 persists in 80/96
velocity trajectories for generated defects, 90/96 for successful hand insertions and
96/96 for natural defects (32 independent hosts per group). These observations support
short-term persistence for many generated defects; they do not establish natural-equivalent
surroundings or long-term stability. The new matched seed/sham repeat has only eight
requests per model so far. Historical tables average energy realism over successes and
failures unless explicitly labelled successful-only.

Full matched replication and longer stability checks: see the
[completion workflow](docs/GAP1_COMPLETION.md) and
[Kaggle notebook](notebooks/kaggle_gap1_completion.ipynb).

## Generalization study (gap 2)

Run the dedicated [Kaggle generalization notebook](notebooks/kaggle_gap2_generalization.ipynb)
for fresh 256/1024-atom hosts at three cooling rates, all three frozen training seeds,
and matched controls. See the [protocol and resume instructions](docs/GAP2_GENERALIZATION.md).
The preparation checks are complete; production placement-transfer results require running
this notebook. It supports checkpointing across multiple Kaggle sessions.

## Run on Colab (GPU)

Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Colab:
<https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_train.ipynb>

For the current experiment round only, use the shorter
[`notebooks/colab_round3.ipynb`](notebooks/colab_round3.ipynb):
<https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_round3.ipynb>
(setup, round 3, summary and push; it reuses the models and results of rounds 1–2 in Drive).
Out of Colab compute? [`notebooks/kaggle_round3.ipynb`](notebooks/kaggle_round3.ipynb) runs
the same round on Kaggle without Drive: it loads earlier models and results from this
repository and pushes its progress back every 45 minutes, so it can continue across Kaggle
sessions.
The sensitivity checks run in [`notebooks/kaggle_sensitivity.ipynb`](notebooks/kaggle_sensitivity.ipynb)
(one Kaggle session, ~4 h; analysis in `results/sensitivity/`), and the training seeds in
[`notebooks/kaggle_seeds.ipynb`](notebooks/kaggle_seeds.ipynb) (one session on T4 x2, ~10–11 h).

The full notebook checks the GPU, measures training speed, trains the unconditional and
conditional EGNN (resuming automatically after a disconnect), runs every experiment and
baseline, and builds the summary table. Runs and results are kept in Google Drive (`MyDrive/idl-project/`);
the last cell pushes a compact copy (summary, metrics, figures, trained weights) to
`results/colab_<date>/` in this repository, using a GitHub token stored in Colab Secrets.

## Quickstart (any GPU machine)

```bash
pip install torch                       # CUDA build
pip install -e ".[dev,md]"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH   # directory holding libmpi.so.12
# 1. data: already committed in data/. To regenerate (~30 CPU-minutes on 4 cores, bit-identical):
python scripts/make_dataset.py      --config configs/data/ka2d_256.yaml
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256
python scripts/build_patches.py     --config configs/defects/v0.yaml data=data/ka2d_256
# 2. train (unconditional, then conditional)
python scripts/train.py --config configs/train/uncond.yaml
python scripts/train.py --config configs/train/cond.yaml
# 3. sample -> relax -> evaluate (repeat per method, defect, task)
python scripts/sample.py   --config configs/sampler/cfg.yaml ckpt=runs/<cond>/ckpt.pt data=data/ka2d_256 defect=D2 request.mode=host
python scripts/relax.py    --config configs/eval/default.yaml run=runs/<sample-run>
python scripts/evaluate.py --config configs/eval/default.yaml run=runs/<sample-run> data=data/ka2d_256
python scripts/plot_samples.py --config configs/eval/default.yaml run=runs/<sample-run> data=data/ka2d_256
python scripts/aggregate_results.py --config configs/eval/default.yaml runs_dir=runs
```

More commands (baselines, floor row, the other datasets): `docs/IMPLEMENTATION_PLAN.md` §6
and `docs/DATASET.md` §4.

## Layout
```
configs/        YAML configs: data, defects, model, train, sampler, eval
data/           generated datasets (committed, ~20 MB; see data/README.md)
docs/           plans and the feasibility pilot
notebooks/      Stage 0 notebook, figures, Kaggle training notebooks
results/        small summary tables and figures that go in the report
runs/           training / sampling / evaluation run directories (gitignored)
scripts/        command-line entry points (one per pipeline step); scripts/silica/ for 3D
slurm/          PSC Bridges-2 job scripts (3D silica)
src/glassdiff/
  types.py      Structures, Request, label conventions
  geometry.py   periodic minimum-image vectors, neighbour lists
  md/           LAMMPS melt-quench generation, local melt-quench baseline
  physics/      Kob–Andersen potential and batched FIRE relaxation (PyTorch)
  analysis/     descriptors, defect detectors, structure statistics
  data/         dataset I/O, training/eval requests, patch library
  models/       denoisers: EGNN-PBC (main), MPNN, MLP
  diffusion/    noise and loss, DM2 score-dynamics sampler, clamp/RePaint/CFG
  eval/         success, realism, diversity, cost metrics; bootstrap CIs
  baselines.py  hand insertion (B2)
  viz.py        configuration plots
  silica/       3D silica: data, defect detectors, requests, conditional NequIP, LAMMPS
  utils/        configs, seeding, run directories
tests/          unit, acceptance and end-to-end pipeline tests (CPU, ~30 s)
```

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS, ",silica" for the 3D model
ruff check src tests scripts && pytest -q
```
LAMMPS via pip needs `libmpi.so.12`. The `mpich` wheel provides it; put its directory on
`LD_LIBRARY_PATH` (see docs/DATASET.md §4).
