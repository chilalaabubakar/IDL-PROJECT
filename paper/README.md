# The reports

`final_report.tex` / `final_report.pdf`: **the final course report** (11-785), organised by the
course rubric: overview, related work, methodology, baselines and extensions, results and
analysis, discussion, future directions, conclusions, contributions. Its model description
table (`fr_table_models.tex`) follows the layout of Table 1 of Henderson et al., Nucleic
Acids Research 2019. It uses the numbers and tables of the long report below and adds:

```bash
python scripts/pilots/final_report_analysis.py shapes        # output shapes, parameter counts
python scripts/pilots/final_report_analysis.py splits        # training / validation / test
python scripts/pilots/final_report_analysis.py features      # learned features (t-SNE)
python scripts/pilots/final_report_analysis.py sensitivity   # shaken inputs (~25 min)
python paper/make_final_numbers.py                           # fr_numbers.tex, fr_table_*.tex
python paper/make_final_figures.py                           # figures/fr_fig_*
cd paper && latexmk -pdf -outdir=build final_report.tex && cp build/final_report.pdf .
```

`point_defects_report.tex` / `point_defects_report.pdf`: **the report on the two-dimensional
point-defect study** (finding and repairing vacancies, interstitials, Frenkel pairs and
antisites), written after Prof. Dabo's review. Every number in it is read from
`results/point_defects_recheck_20261111/`, `results/point_defects_ablations_20261112/`,
`results/point_defects_seeds_20261114/` and `results/point_defects_architectures_20261115/`.
(The midterm report and the workshop paper of the first study are on the branch `main`.)

```bash
# from the repository root
bash scripts/pilots/recheck_real_defects.sh   # the test evaluation, ~2 h on 4 CPU cores, resumable
python scripts/pilots/finder_ablations.py     # experiments on the finder (after finder_ablations.sh)
python scripts/pilots/finder_seeds.py         # the same settings with three seeds, training stability
python scripts/pilots/finder_architectures.py evaluate --out results/point_defects_architectures_20261115
                                              # transformer and CNN against the EGNN (after the two notebooks)
python paper/make_report_numbers.py           # pd_numbers.tex, pd_table_*.tex, summary.md
python paper/make_report_figures.py           # figures/pd_fig1..5 (or e.g. `make_report_figures.py 3`)
cd paper && latexmk -pdf -outdir=build point_defects_report.tex && cp build/point_defects_report.pdf .
```

A number in the text is a macro (`\pd{found-finder-vacancy}`) defined in `pd_numbers.tex`, so
the text cannot drift from the results; an undefined one prints in red. The pipeline figure
uses one test vacancy stored in `data/pd_report_example.npz` (remade if the file is deleted).
`figures/pd_fig4_network` is the deep learning model (network, one layer, training order).
The ablation table (`pd_table_ablations.tex`, written by `make_report_numbers.py` when all
six settings have three seeds) and its text (`pd_section_ablations.tex`) are read only when
the table exists. The same holds for the stability table and the table of the three finders
(`pd_table_stability.tex`, `pd_table_finders.tex`) and their text (`pd_section_finders.tex`).

Figures are drawn at their printed size (5.5 in wide) with `figstyle.py`: one typeface, a
colour-blind-safe palette, lowercase bold panel letters, vector PDF plus 300-dpi PNG.

| Figure | Content | Data |
|---|---|---|
| `pd_fig1_defects` | the four point defects: the operation, and the glass after relaxation | `data/ka2d_256_point_defects/test.npz` |
| `pd_fig4_network` | the deep learning model: network, one layer, training order | none |
| `pd_fig2_pipeline` | find, correct, minimise, compare, on one test vacancy | `paper/data/pd_report_example.npz` |
| `pd_fig3_results` | found and restored, finder against the three other measures | `paper/data/pd_report_results.json` |
| `pd_fig5_finders` | transformer, CNN and EGNN as finders: gradient at every update, defects found | `results/point_defects_architectures_20261115/`, `results/kaggle_arch_20261115-*/training/*/steps.csv` |
