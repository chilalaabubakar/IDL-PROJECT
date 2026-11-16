# Finding and Repairing Point Defects in a Model Glass with a Denoising Network

IDL (11-785) group project, Carnegie Mellon University Africa: S. Cheruto, N. Ladislaus,
A. Chilala, G. Uwera, M. Sangwa; teaching assistant B. Godwin.

> **Two studies, two branches.** This branch (`dabo-point-defects`) holds the second study,
> made after a review by Prof. Ismaila Dabo: finding and repairing standard point defects.
> The first study (placing defects at chosen sites with a conditional diffusion model, the
> midterm report and the workshop paper) is on the branch
> [`main`](https://github.com/chilalaabubakar/IDL-PROJECT/tree/main).
> The two branches are kept separate and are not merged into each other.

In a crystal, a missing, extra or misplaced atom is found by comparing the structure with
its regular pattern. A glass has no pattern, and after a defect is made the neighbouring
atoms shift and largely hide it. We ask whether a neural network can find the defect from
the positions and species of the atoms alone, and whether the glass can then be returned to
the defect-free structure it came from.

We make the four standard point defects (vacancy, interstitial, Frenkel pair, antisite) in a
two-dimensional model glass by known operations and relax them, so the right answer is
always known. One graph neural network is trained in two stages: first to remove random
noise from glasses (a denoiser, following DM2:
[Yang & Schwalbe-Koda, npj Comput. Mater. 2026](https://doi.org/10.1038/s41524-025-01901-1)),
then to label the atoms that belong to a defect (the finder).

## Documents

- [Report](paper/point_defects_report.pdf) (23 pages): the complete study.
- [Summary in plain language](docs/Point_Defects_Study_Summary.docx) (Word): the materials
  science, the deep learning and what the study shows, for readers new to either field.
- [Method and results in detail](docs/POINT_DEFECTS.md).
- [Prof. Dabo's recommendations](docs/DABO_RECOMMENDATIONS.md): his ten points, our answers
  and the plan.
- [Silica, stage 1](docs/SILICA_POINT_DEFECTS.md): the same question in amorphous SiO₂,
  without a learned model yet.
- [Dataset](docs/DATASET.md): how the glasses were made.

## Main results

On 750 real test defects. The finder uses no interatomic potential; the best method
without learning is the best of cell size, number of unlike neighbours and atom energy.

| Defect | Found: finder | Found: best without learning | Restored: finder | Restored: best without learning | Restored: true location given |
|---|---|---|---|---|---|
| Vacancy | **69.5%** | 47.5% | **53.5%** | 40.5% | 67% |
| Interstitial | **50%** | 28% | **38%** | 21.5% | 80% |
| Antisite | **49%** | 31.5% | **30%** | 8% | 68.5% |
| Frenkel pair | **45.3%** | 32.7% | 12.7% | 7.3% | 42% |

- Once the defect is found, ordinary energy minimisation does the repair; a network trained
  to undo the defect directly is not needed.
- Starting the finder from the denoiser doubles what it finds after the same short
  training (43.8% against 21.9%, mean of three seeds).
- Training is stable: the EGNN shows gradient spikes, but with gradient clipping no run
  failed and three seeds find the same share of the defects to within one point.
- A transformer and a CNN, trained like the EGNN from random weights, find 53.6% and 54.0% of
  all defects against 43.5% for the EGNN, without gradient spikes. They match the finder of
  the table above (54.0%), which starts from the denoiser and has three times the training.
- Earlier work identified inserted vacancies and interstitials in a one-species glass with
  classical measures (Laakkonen & Nieminen, 1988). Their probe for interstitials, local
  pressure, finds 19% of ours (finder 50%). What is new here is a learned finder, scored on
  unseen defects in a two-species glass, against classical measures on the same defects.
- Numbers and tests: [`results/point_defects_recheck_20261111/`](results/point_defects_recheck_20261111/README.md);
  experiments and ablations: [`results/point_defects_ablations_20261112/`](results/point_defects_ablations_20261112/README.md);
  seeds and stability: [`results/point_defects_seeds_20261114/`](results/point_defects_seeds_20261114/README.md);
  transformer and CNN: [`results/point_defects_architectures_20261115/`](results/point_defects_architectures_20261115/README.md).

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS, ",silica" for the 3D code
ruff check src tests scripts && pytest -q
```

## Reproduce

```bash
# 1. defects: validation and test splits are committed; the training split takes ~30-45 min
python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml splits=[train]
# 2. finder and repair network (they start from the denoiser stored in results/)
python scripts/train_point_defects.py --config configs/train/detect_point_defects.yaml
python scripts/train_point_defects.py --config configs/train/heal_point_defects_v2.yaml
# 3. test evaluation, experiments on the finder, and the report
bash scripts/pilots/recheck_real_defects.sh
bash scripts/pilots/finder_ablations.sh && python scripts/pilots/finder_ablations.py
bash scripts/pilots/finder_seeds.sh && python scripts/pilots/finder_seeds.py
python scripts/pilots/finder_architectures.py evaluate --out results/point_defects_architectures_20261115
python paper/make_report_numbers.py && python paper/make_report_figures.py
cd paper && latexmk -pdf -outdir=build point_defects_report.tex
```

The training runs of the report were made on Kaggle GPUs with the notebooks in
[`notebooks/`](notebooks/README.md).

## Layout

```
configs/        settings: data, model, training (configs/train/*point_defects*.yaml), silica
data/           the glasses (ka2d_256), the point defects made in them, DM2's silica glasses
docs/           method, review and answers, silica stage 1, plain-language summary
notebooks/      Kaggle and Colab notebooks of this study
paper/          the report: LaTeX source, figures, and the scripts that write its numbers
results/        every result of this study, and the denoiser it starts from
scripts/        make_point_defects, train_point_defects, eval_point_defects*, pilots/ (tests,
                baselines, ablations), silica/; plus the shared tools listed below
src/glassdiff/  the library: periodic EGNN, Kob-Andersen potential, FIRE minimiser, data
tests/          unit and end-to-end tests (CPU)
```

## What this branch shares with `main`

The second study is built on the code and data of the first. These parts come from the
first study and are kept here because this study needs them:

- the `glassdiff` library (`src/`) and its tests, extended here with the finder, the
  point-defect data and a faster minimiser; and the tools that make the glasses and train
  and sample the denoiser (`scripts/make_dataset.py`, `train.py`, `sample.py`, `relax.py`,
  `evaluate.py` and the scripts they are tested with);
- the glasses in `data/ka2d_256/` and DM2's silica glasses in `data/sio2_dm2/`;
- the denoiser in `results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/`, trained in the
  first study on defect-free glasses. The finder and the repair network start from it.

Everything else of the first study (its plans, analyses, notebooks, results, midterm report
and workshop paper, and the smaller data sets) is on `main` only. Code comments that cite
`docs/PROJECT_PLAN.md`, `docs/IMPLEMENTATION_PLAN.md` (tickets), `docs/SILICA_3D_PLAN.md` or
`docs/SILICA_QUICKSTART.md` refer to documents on `main`.
