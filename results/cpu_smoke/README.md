# CPU smoke run: a pipeline check, not results

Everything here was produced end to end in a 4-core CPU container to prove the pipeline
works on real data: dataset → train → sample → relax → evaluate → aggregate → plot.
**The model is far too small and too briefly trained for these numbers to answer the
research questions.** They show what every table and figure will look like and give the
baselines and the floor a first measurement. Real runs need a GPU (README quickstart).

| Item | Smoke run | Planned GPU run |
|---|---|---|
| Model | EGNN-PBC, hidden 64, 3 layers (108k parameters) | hidden 128, 4 layers |
| Training | samples used the step-1,000 checkpoint (batch 8, lr 1e-3, EMA 0.99); the run continued to 2,289 updates in 80 min on 2 busy threads | 40,000 updates (`configs/train/uncond.yaml`) |
| Sampling | 650 steps (600 noisy + 50 final), 8 samples | 3,000 steps, 256 samples |
| Defect | D2 only | D1−, D1+, D2 |

## Summary (`summary.md`, from `scripts/aggregate_results.py`)

| method | defect | mode | n | success after relax | success before relax | local W1 (per-atom PE) | local W1 (bond angle) | overlap rate | s per success |
|---|---|---|---|---|---|---|---|---|---|
| floor (natural train vs natural test) | D2 | — | 512 | — | — | 0.013 [0.009, 0.019] | 0.001 [0.001, 0.002] | 0 | — |
| B1 unconditional, Task A | D2 | centre | 8 | 0 / 8 | 0 / 8 | 0.079 [0.072, 0.101] | 0.017 [0.014, 0.021] | 0 | ∞ |
| B5 RePaint (U = 1), Task B | D2 | host | 8 | 0 / 8 | 0 / 8 | 0.072 [0.060, 0.094] | 0.020 [0.017, 0.026] | 0 | ∞ |
| B2 hand insertion | D2 | hand_insert | 32 | 0.31 [0.16, 0.47] | 1.00 | 0.083 [0.060, 0.106] | 0.009 [0.007, 0.012] | 0 (0.88 before relaxing) | 143 |
| B6 local melt-quench | D2 | host | 16 | 0.06 [0.00, 0.19] | 0.06 | 0.062 [0.035, 0.106] | 0.007 [0.006, 0.012] | 0 | 609 |

Brackets are 95% bootstrap CIs over samples. Costs are CPU wall-clock on a contended
machine, so compare them only to each other.

## What the numbers already say

* **Hand insertion shows the "vanishing defect" failure.** Every pasted D2 patch is a D2
  before relaxation, but only 31% survive relaxation. Before relaxation 88% of samples
  have overlapping atoms. This is exactly the failure mode the proposal predicted for
  current practice.
* **Local melt-quench is about the chance rate.** It succeeds 1 time in 16, where about
  11% of small atoms are D2 naturally. Its surroundings are the most realistic of the
  baselines.
* **The unconditional model behaves as an unconditional model should.** It gets 0/8
  successes, consistent with chance rates of about 4% (Task A) and 11% (Task B). After
  relaxation every sample is a valid glass: no overlaps, no crystallization
  (|Ψ6| ≈ 0.04–0.05), and PE/atom −3.684 against −3.680 for natural glasses. But its
  local W1 is about 6× the floor, and atoms move 0.25σ on relaxation. The short CPU
  training is the expected cause.
* **The floor rows are what every method must be compared against.** Without them,
  W1 = 0.07 would mean nothing.

## Figures

* `natural_test_glasses.png`: four test-split glasses, natural defects ringed
* `hand_insert_D2.png`, `local_melt_quench_D2.png`, `B1_uncond_taskA.png`,
  `B5_repaint_taskB.png`: four relaxed samples per method, with the requested location
  (+ and the r_tol circle), pinned atoms hatched, defects ringed, and success per panel
* `training_curve.png`: validation loss / σ² per noise level (1 = predicting zero), for the full
  2,289-update run; at the end the model removes 55% of the noise at σ = 0.02, 70% at 0.1,
  32% at 0.3 and 11% at 0.5, and it was still improving

## Model files

`model/` holds the CPU-trained EGNN: `ckpt_step1000.pt` (used for the samples above),
`ckpt_step2289.pt` (end of the run), with the run's `config.yaml`, `log.csv` and
`git.txt`. Load either with `glassdiff.models.registry.load_denoiser(path)` or pass it as
`ckpt=` to `scripts/sample.py`. They exist so the sampling and evaluation scripts can be
tried without training first; they are not models to report results with.

## Reproduce

```bash
python scripts/train.py --config configs/train/uncond.yaml name=egnn_uncond_cpu_smoke \
  model_args.hidden=64 model_args.n_layers=3 optim.batch_size=8 optim.lr=1e-3 optim.ema=0.99 \
  max_minutes=80 val_every=250 threads=2
python scripts/sample.py --config configs/sampler/dm2.yaml ckpt=<ckpt> data=data/ka2d_256 defect=D2 \
  n_samples=8 batch_size=8 schedule.n_noisy=600 schedule.n_final=50 name=B1_uncond_taskA
python scripts/sample.py --config configs/sampler/repaint.yaml ckpt=<ckpt> data=data/ka2d_256 defect=D2 \
  n_samples=8 batch_size=8 schedule.n_noisy=600 schedule.n_final=50 request.mode=host \
  strategy.n_resample=1 name=B5_repaint_taskB
python scripts/hand_insert.py --config configs/eval/default.yaml data=data/ka2d_256 defect=D2 n_samples=32
python scripts/local_melt_quench.py --config configs/data/ka2d_256.yaml data=data/ka2d_256 defect=D2 n_samples=16
# then, for each run: relax.py, evaluate.py, plot_samples.py; and evaluate.py floor=true defect=D2
```
