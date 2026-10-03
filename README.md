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

## Layout
```
configs/        YAML configs: data, defects, model, train, sampler, eval
data/           generated datasets (gitignored)
docs/           plans and the feasibility pilot
notebooks/      Stage 0 notebook, figures
results/        small summary tables and figures that go in the report
runs/           training / sampling / evaluation run directories (gitignored)
scripts/        command-line entry points (one per pipeline step)
src/glassdiff/
  types.py      Structures, Request, label conventions            (implemented)
  geometry.py   periodic minimum-image vectors, neighbour lists   (implemented)
  md/           LAMMPS melt-quench generation, local melt-quench baseline
  physics/      Kob–Andersen potential and batched FIRE relaxation (PyTorch)
  analysis/     descriptors, defect detectors, structure statistics
  data/         dataset I/O, training/eval requests, patch library
  models/       denoisers: EGNN-PBC (main), MPNN, MLP
  diffusion/    noise and loss, DM2 score-dynamics sampler, clamp/RePaint/CFG
  eval/         success, realism, diversity, cost metrics; bootstrap CIs
  utils/        configs, seeding, run directories                 (implemented)
tests/          unit tests; acceptance tests for each ticket (skipped until implemented)
```

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS (dataset generation)
ruff check src tests scripts && pytest -q
```
LAMMPS via pip needs `libmpi.so.12`. The `mpich` wheel provides it; put its directory on
`LD_LIBRARY_PATH` (see docs/DATASET.md §4).
