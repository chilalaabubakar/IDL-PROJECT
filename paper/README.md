# Workshop paper

*Placing Defects Where You Ask: Generating Defects at Chosen Sites in a Model Glass with a
Diffusion Model*: about 8.5 pages of main text in plain language, then references and a
3-page appendix. `main.pdf` is the compiled version.

## Build

```bash
# from the repository root
python paper/make_examples.py part=model   # once, ~10 min on 4 CPU cores
python paper/make_examples.py part=hand
python paper/make_examples.py part=lmq     # needs LAMMPS
python paper/make_figures.py               # all figures, or e.g. `make_figures.py 1 3`
cd paper && latexmk -pdf -outdir=build main.tex && cp build/main.pdf main.pdf
```

Figures are drawn at their printed size (5.5 in wide) in the style of
[figures4papers](https://github.com/ChenLiu-1996/figures4papers) (`figstyle.py`): Helvetica-type
font, blues for the model, reds for the baselines, grey for natural references, lowercase bold
panel letters, vector PDF plus 300-dpi PNG.

| Figure | Content | Data |
|---|---|---|
| 1 `fig1_overview` | (a) numbered workflow; (b) natural defects in a test glass; (c) a Task B request on a real host; (d) a real sampling trajectory and its relaxed result | `data/ka2d_256/test.npz`, `paper/data/examples_D2_model.npz` |
| 2 `fig2_architecture` | the deep learning models: (a) conditional periodic EGNN denoiser, (b) one EGNN layer, (c) classifier-free guidance, (d) the symmetry-ablation backbones (EGNN, MPNN, flat MLP) | `src/glassdiff/models/`, Table 3 numbers |
| 3 `fig3_taskB` | one D2 request answered by the model, hand insertion and local melt-quench | `paper/data/examples_D2_*.npz` |
| 4 `fig4_seeds` | success vs. local W1 per training seed | `results/*/experiments/` (via `scripts/plot_seeds.py`) |
| 5 `fig5_sensitivity` (appendix) | sensitivity to evaluation settings | `results/*/experiments/`, `results/sensitivity/` |

`make_examples.py` answers the same six D2 requests (first six test glasses, sites drawn with
seed 7) with each method on CPU, relaxes them and records success, so Figures 1d and 3 show
real samples, not illustrations.

## Before submitting

- **Double-blind venues:** set `\newcommand{\anonymous}{1}` in `main.tex`. This hides the
  authors, the acknowledgements and the repository link.
- **Official style file:** the layout imitates the NeurIPS / AI4Mat format (5.5 × 9 in text,
  Times). To use the official `.sty`, drop it into this folder, delete the `geometry` and
  `mathptmx` lines, and add `\usepackage[preprint]{neurips_2025}` (or the venue's name).
- **AI-use statement:** many venues now ask for one. There is a marked place in the
  acknowledgements.
- **Numbers:** every number comes from `results/`:
  - `round3_ANALYSIS.md`, round 2's `ANALYSIS.md`;
  - `results/sensitivity/README.md`, `results/seeds/README.md`, `results/stage0/`.

  If a result is re-run, update the text from there.
