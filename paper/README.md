# Workshop paper

*Placing Defects Where You Ask: Location-Specific Defect Generation in a Model Glass with
Conditional Diffusion*: 8 pages of main text, then references and a 3-page appendix.
`main.pdf` is the compiled version.

## Build

```bash
python paper/make_figures.py          # from the repository root: figures from results/
cd paper && latexmk -pdf -outdir=build main.tex && cp build/main.pdf main.pdf
```

`make_figures.py` crops Figure 1 from the Task B sample plots, and draws Figures 2 and 3
with `scripts/plot_seeds.py` and `scripts/plot_sensitivity.py` at paper size.

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
