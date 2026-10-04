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

## Status

The MVP pipeline is implemented and tested (64 tests, including an end-to-end run of
every script). It covers dataset generation, defect labelling, the periodic EGNN and
the MPNN/MLP ablations, training, the DM2 sampler with clamping, RePaint and CFG
conditioning, both baselines (hand insertion, local melt-quench), relaxation and
metrics with bootstrap CIs. Full-size training runs need a GPU. See
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §2 for the ticket-by-ticket
status.

## Run on Colab (GPU)

Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Colab:
<https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_train.ipynb>

It checks the GPU, measures training speed, trains the unconditional and conditional EGNN
(resuming automatically after a disconnect), runs every experiment and baseline, and builds
the summary table. Runs and results are kept in Google Drive (`MyDrive/idl-project/`).

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
notebooks/      Stage 0 notebook, figures
results/        small summary tables and figures that go in the report
runs/           training / sampling / evaluation run directories (gitignored)
scripts/        command-line entry points (one per pipeline step)
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
  utils/        configs, seeding, run directories
tests/          unit, acceptance and end-to-end pipeline tests (CPU, ~30 s)
```

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS (dataset generation)
ruff check src tests scripts && pytest -q
```
LAMMPS via pip needs `libmpi.so.12`. The `mpich` wheel provides it; put its directory on
`LD_LIBRARY_PATH` (see docs/DATASET.md §4).
