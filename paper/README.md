# The report

`point_defects_report.tex` / `point_defects_report.pdf`: **the report on the two-dimensional
point-defect study** (finding and repairing vacancies, interstitials, Frenkel pairs and
antisites), written after Prof. Dabo's review. Every number in it is read from
`results/point_defects_recheck_20261111/` and `results/point_defects_ablations_20261112/`.
(The midterm report and the workshop paper of the first study are on the branch `main`.)

```bash
# from the repository root
bash scripts/pilots/recheck_real_defects.sh   # the test evaluation, ~2 h on 4 CPU cores, resumable
python scripts/pilots/finder_ablations.py     # experiments on the finder (after finder_ablations.sh)
python paper/make_report_numbers.py           # pd_numbers.tex, pd_table_*.tex, summary.md
python paper/make_report_figures.py           # figures/pd_fig1..4 (or e.g. `make_report_figures.py 3`)
cd paper && latexmk -pdf -outdir=build point_defects_report.tex && cp build/point_defects_report.pdf .
```

A number in the text is a macro (`\pd{found-finder-vacancy}`) defined in `pd_numbers.tex`, so
the text cannot drift from the results; an undefined one prints in red. The pipeline figure
uses one test vacancy stored in `data/pd_report_example.npz` (remade if the file is deleted).
`figures/pd_fig4_network` is the deep learning model (network, one layer, training order).
The ablation table (`pd_table_ablations.tex`, written by `make_report_numbers.py` when all
six ablation runs are finished) and its text (`pd_section_ablations.tex`) are read only when
the table exists.

Figures are drawn at their printed size (5.5 in wide) with `figstyle.py`: one typeface, a
colour-blind-safe palette, lowercase bold panel letters, vector PDF plus 300-dpi PNG.

| Figure | Content | Data |
|---|---|---|
| `pd_fig1_defects` | the four point defects: the operation, and the glass after relaxation | `data/ka2d_256_point_defects/test.npz` |
| `pd_fig4_network` | the deep learning model: network, one layer, training order | none |
| `pd_fig2_pipeline` | find, correct, minimise, compare, on one test vacancy | `paper/data/pd_report_example.npz` |
| `pd_fig3_results` | found and restored, finder against the three other measures | `paper/data/pd_report_results.json` |
