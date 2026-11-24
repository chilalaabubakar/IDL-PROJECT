# Finding Point Defects in a Model Glass with Deep Learning

Course project for 11-785 Introduction to Deep Learning, College of Engineering, Carnegie
Mellon University Africa. Stacy Cheruto, Nicholaus Ladislaus, Abubakar Chilala, Gentille
Uwera and Michel Sangwa; supervised by teaching assistant Boniface Godwin.

This branch (`dabo-point-defects`) holds the final study. The earlier study, in which a
diffusion model placed self-defined defects (midterm report), is on the branch
[`main`](https://github.com/chilalaabubakar/IDL-PROJECT/tree/main). The two branches are
kept apart.

## The question

A point defect is a missing, extra, displaced or exchanged atom: a vacancy, an interstitial,
a Frenkel pair or an antisite. In a crystal it is easy to see, because it breaks a regular
pattern. A glass has no pattern, and once the atoms settle around a defect they largely hide
it. We ask whether a neural network can find the defect from the positions and species of
the atoms alone, and whether the glass can then be returned to the structure it came from.

We make the four defects ourselves in a simulated glass, so the right answer is always known,
and compare the network with classical measures on the same unseen defects.

## Reports

- [Final report](paper/final_report.pdf): 10 pages, organised by the course rubric, with an
  appendix that answers the teaching assistants' reviews of the midterm report.
- [Full report](paper/point_defects_report.pdf) (25 pages): every detail of the study.
- [Plain-language summary](docs/Point_Defects_Study_Summary.docx) (Word), for readers new to
  materials science or to deep learning.
- [Method and results in detail](docs/POINT_DEFECTS.md) and
  [Prof. Dabo's review with our answers](docs/DABO_RECOMMENDATIONS.md).

## Data

- **Glass.** The two-dimensional Kob–Andersen mixture: 65% large A and 35% small B atoms
  with Lennard-Jones forces, 256 atoms in a periodic box of side 14.6σ. Each glass is melted,
  cooled and relaxed to an energy minimum with LAMMPS: 800 training, 100 validation and 100
  test glasses ([`data/ka2d_256`](data/README.md)).
- **Defects.** Remove an atom (vacancy), add a small B atom (interstitial), move an atom
  1.2–3σ into a gap (Frenkel pair) or swap an A atom with its nearest B atom (antisite), then
  relax again. 6,400 training defects of each type and 750 test defects
  ([`data/ka2d_256_point_defects`](data/README.md)). A quarter of the Frenkel pairs undo
  themselves on relaxation and are left out of the test.

## Models

| Model | What it is | Parameters |
|---|---|---|
| Denoiser | periodic EGNN, 4 layers, 128 features; learns to undo random shifts of the atoms (DM2's training) | 530,696 |
| Finder | the same network started from the denoiser, with a head that gives every atom one of five roles | 523,021 |
| Transformer finder | every atom attends to all others, with a learned bias from their distance | 576,677 |
| CNN finder | the glass drawn as a 128 × 128 picture, a U-Net, scores read at each atom | 498,309 |
| Repair network | the denoiser trained to undo a defect instead of noise | 530,696 |

The finder is trained with class-weighted cross-entropy and calibrated afterwards (logit
adjustment); AdamW, batches of 16 glasses of one defect type, gradient norm cut at 1. The
final report gives every layer, its output shape and the equations.

## Results

On the 750 test defects. A defect is *found* when the flagged atom is within 1.5σ of it, and
*restored* when correcting the atoms there and minimising the energy gives back the original
glass. The best classical measure is the best of cell size, unlike neighbours and atom energy.

| Defect | Found: finder | Found: best classical | Restored: finder | Restored: best classical | Restored: true location given |
|---|---|---|---|---|---|
| Vacancy | **69.5%** | 47.5% | **53.5%** | 40.5% | 67% |
| Interstitial | **50%** | 28% | **38%** | 21.5% | 80% |
| Antisite | **49%** | 31.5% | **30%** | 8% | 68.5% |
| Frenkel pair | **45.3%** | 32.7% | 12.7% | 7.3% | 42% |

- The finder needs no interatomic potential once trained. It also beats local pressure, the
  classical probe for interstitials (19%).
- It finds vacancies of small B atoms as well as those of large A atoms (70% and 69%); the
  atom energy finds 11% and 40%.
- It does not overfit: 50.5% of training, 52.2% of validation and 54.0% of test defects found.
- Starting from the denoiser doubles what the finder finds after a short training
  (43.8% against 21.9%, three seeds each).
- A transformer and a CNN, trained like the EGNN from random weights, find 53.6% and 54.0%
  against 43.5%. The EGNN shows gradient spikes, the other two none; with the gradient cut,
  no run failed and repeated runs agree to within one point.
- Once the defect is found, energy minimisation does the repair. A repair network moves the
  right atoms but leaves overlapping atoms without minimisation.
- Weak points: shifting every atom at random by 0.02σ lowers the finder from 54% to 38%; only
  66 of the 150 test Frenkel pairs are still Frenkel pairs after relaxation; and the finder
  says where a defect is, not whether there is one.

Numbers, tests and the scripts that make them:
[`results/point_defects_recheck_20261111/`](results/point_defects_recheck_20261111/README.md) (test results),
[`results/point_defects_ablations_20261112/`](results/point_defects_ablations_20261112/README.md) (ablations),
[`results/point_defects_seeds_20261114/`](results/point_defects_seeds_20261114/README.md) (seeds and stability),
[`results/point_defects_architectures_20261115/`](results/point_defects_architectures_20261115/README.md) (transformer and CNN),
[`results/final_report_20261117/`](results/final_report_20261117/README.md) (training against test, features, shaken inputs).

## Limits

The defects are put in by hand and serve as test cases with a known answer, not as a model of
how faults form in real glass. The glass is two-dimensional, at zero temperature, with one
defect of known type per glass. The finder needs no potential at use, but its training data
come from one. The main finder was trained once.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"          # add ",md" for LAMMPS, ",silica" for the 3D code
ruff check src tests scripts && pytest -q
```

## Reproduce

```bash
# defects: validation and test splits are committed; the training split takes 30-45 min
python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml splits=[train]
# finder and repair network, both started from the denoiser in results/
python scripts/train_point_defects.py --config configs/train/detect_point_defects.yaml
python scripts/train_point_defects.py --config configs/train/heal_point_defects_v2.yaml
# test evaluation and experiments
bash scripts/pilots/recheck_real_defects.sh
bash scripts/pilots/finder_ablations.sh && python scripts/pilots/finder_ablations.py
bash scripts/pilots/finder_seeds.sh && python scripts/pilots/finder_seeds.py
python scripts/pilots/finder_architectures.py evaluate --out results/point_defects_architectures_20261115
# reports
python paper/make_report_numbers.py && python paper/make_report_figures.py
for w in shapes splits features sensitivity; do python scripts/pilots/final_report_analysis.py $w; done
python paper/make_final_numbers.py && python paper/make_final_figures.py
cd paper && latexmk -pdf -outdir=build final_report.tex point_defects_report.tex
```

The GPU training runs were made on Kaggle and Colab with the notebooks in
[`notebooks/`](notebooks/README.md).

## Layout

```
configs/        data, model and training settings
data/           the glasses, the point defects made in them, DM2's silica glasses
docs/           method, review and answers, silica stage 1, plain-language summary
notebooks/      Kaggle and Colab notebooks
paper/          both reports: LaTeX, figures, and the scripts that write their numbers
results/        every result, and the denoiser the finder starts from
scripts/        data, training and evaluation; pilots/ holds baselines and experiments
src/glassdiff/  the library: periodic EGNN, transformer, CNN, Kob-Andersen potential,
                FIRE minimiser, defect data
tests/          unit and end-to-end tests (CPU)
```

## Team

| Workstream | Member |
|---|---|
| Data and physics | Nicholaus Ladislaus |
| Defects and evaluation | Stacy Cheruto |
| Models and training | Abubakar Chilala |
| Repair procedure | Gentille Uwera |
| Baselines and writing | Michel Sangwa |
| Supervision | Boniface Godwin |

We thank Prof. Ismaila Dabo (Materials Science and Engineering, Carnegie Mellon University)
for the review that led to this study.

## Shared with `main`

This study builds on the code and data of the first one: the `glassdiff` library and its
tests, the tools that make the glasses and train the denoiser, the glasses in
`data/ka2d_256/`, DM2's silica glasses in `data/sio2_dm2/`, and the denoiser in
`results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/`. Everything else of the first
study is on `main` only; code comments that cite `docs/PROJECT_PLAN.md`,
`docs/IMPLEMENTATION_PLAN.md`, `docs/SILICA_3D_PLAN.md` or `docs/SILICA_QUICKSTART.md` refer
to documents there.
