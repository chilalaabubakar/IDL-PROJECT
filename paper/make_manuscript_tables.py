"""Tables and numbers of the journal manuscript (npj_manuscript.tex, npj_supplementary.tex).

    python paper/make_manuscript_tables.py

Reads results/ and the final report's tables and writes
    paper/ms_numbers.tex        \\pdset{key}{value} for numbers not in pd_numbers / fr_numbers
    paper/ms_table_results.tex  Table 1: fr_table_results.tex with the journal caption
    paper/ms_table_design.tex   Table 2: ablations (three seeds) and the three kinds of network
    paper/ms_si_models.tex      Supplementary table: fr_table_models.tex with the journal caption
    paper/ms_si_physics.tex     Supplementary table: what the relaxed defects are
    paper/ms_si_occupancy.tex   Supplementary table: the site-occupancy check and its validation
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KINDS = ("vacancy", "interstitial", "antisite", "frenkel")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def design_table() -> str:
    seeds = json.loads((RES / "point_defects_seeds_20261114/seeds.json").read_text())["variants"]
    arch = json.loads(
        (RES / "point_defects_architectures_20261115/architectures.json").read_text()
    )["models"]

    def row(label: str, start: str, updates: str, f: dict, runs: int) -> str:
        cells = [pct(f[k]["mean"]) for k in KINDS]
        all_ = pct(f["all"]["mean"])
        if runs > 1 and f["all"].get("sd") is not None:
            all_ += f" $\\pm$ {100 * f['all']['sd']:.1f}"
        return f"{label} & {start} & {updates} & {runs} & " + " & ".join(cells) + f" & {all_} \\\\"

    rows = [r"\multicolumn{9}{@{}l}{\emph{Ablations of the EGNN finder (2,000 updates)}} \\"]
    for key, label, start in (
        ("reference", "Reference: 4 layers, class weights", "denoiser"),
        ("scratch", "Random initialisation", "random"),
        ("layers2", "Random initialisation, 2 layers", "random"),
        ("layers1", "Random initialisation, 1 layer", "random"),
        ("no_weights", "No class weights", "denoiser"),
        ("real_only", "Without self-annihilated Frenkel pairs", "denoiser"),
    ):
        v = seeds[key]
        rows.append(row(label, start, "2,000", v["found"], len(v["seeds"])))
    rows += ["\\addlinespace", r"\multicolumn{9}{@{}l}{\emph{Kind of network}} \\"]
    for key, label, start, updates in (
        ("attention", "Transformer", "random", "20,000"),
        ("cnn", "CNN (U-Net)", "random", "20,000"),
        ("egnn_scratch", "EGNN", "random", "20,000"),
        ("egnn_pretrained", "EGNN", "denoiser", "20,000"),
        ("egnn_pretrained_60k", "EGNN (finder of Table 1)", "denoiser", "60,000"),
    ):
        v = arch[key]
        rows.append(row(label, start, updates, v["found"], len(v["seeds"])))
    return (
        "% written by make_manuscript_tables.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Effect of design choices.} Share of the test defects found (\\%), "
        "calibrated. Each row is the mean over training runs that differ only in the random "
        "seed (Runs); the spread between runs (standard deviation) follows the $\\pm$. "
        "Start: initial weights, from the denoiser or random. The two lower EGNN rows from the "
        "denoiser are single runs. Short ablation runs are scored with their final weights, "
        "the 20,000-update runs with the moving average of the weights.}\n"
        "\\label{tab:design}\n\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}lllcccccl@{}}\n\\toprule\n"
        "Finder & Start & Updates & Runs & Vacancy & Interst. & Antisite & Frenkel & All \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def physics(num: dict) -> str:
    p = json.loads((RES / "point_defects_recheck_20261111/defect_physics.json").read_text())
    k = p["kinds"]

    def q(d: dict, digits: int) -> str:
        return f"{d['median']:.{digits}f} [{d['q25']:.{digits}f}, {d['q75']:.{digits}f}]"

    order = ("vacancy", "interstitial", "frenkel", "antisite")
    rows = [
        "Real defects in the test & " + " & ".join(str(k[x]["real_defects"]) for x in order) + " \\\\",
        "Other atoms moved $>0.1\\sigma$ & " + " & ".join(q(k[x]["other_atoms_moved_over_0.1"], 0) for x in order) + " \\\\",
        "Largest move of another atom ($\\sigma$) & " + " & ".join(q(k[x]["largest_move_of_another_atom"], 2) for x in order) + " \\\\",
        "Energy above the original ($\\epsilon$) & -- & -- & " + " & ".join(q(k[x]["energy_above_original"], 1) for x in ("frenkel", "antisite")) + " \\\\",
    ]  # fmt: skip
    v, i = k["vacancy"], k["interstitial"]
    num.update(
        {
            "ph-vac-open-before": f"{v['open_space_right_after_removal']['median']:.2f}",
            "ph-vac-open-after": f"{v['open_space_after_relaxation']['median']:.2f}",
            "ph-open-free": f"{p['open_space_in_defect_free_glass']['median']:.2f}",
            "ph-open-free-p95": f"{p['open_space_in_defect_free_glass']['p95']:.2f}",
            "ph-int-nn": f"{i['extra_atom_nearest_neighbour']['median']:.2f}",
            "ph-int-nn-free": f"{i['B_atom_nearest_neighbour_defect_free']['median']:.2f}",
            "ph-int-e": f"{i['extra_atom_energy']['median']:.1f}",
            "ph-int-e-free": f"{i['B_atom_energy_defect_free']['median']:.1f}",
            "ph-fp-above": f"{100 * k['frenkel']['energy_above_original']['fraction_above']:.0f}",
            "ph-as-above": f"{100 * k['antisite']['energy_above_original']['fraction_above']:.0f}",
            "ph-as-below": f"{100 * k['antisite']['energy_above_original']['fraction_below']:.0f}",
            "ph-fp-e": f"{k['frenkel']['energy_above_original']['median']:.1f}",
            "ph-as-e": f"{k['antisite']['energy_above_original']['median']:.1f}",
            "ph-moved-min": f"{min(k[x]['other_atoms_moved_over_0.1']['median'] for x in order):.0f}",
            "ph-moved-max": f"{max(k[x]['other_atoms_moved_over_0.1']['median'] for x in order):.0f}",
        }
    )
    return (
        "% written by make_manuscript_tables.py\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{\\textbf{The relaxed defects on the test glasses.} Medians with the "
        "interquartile range in brackets. Energies are compared with the original glass, so "
        "only defects that keep the number of atoms are listed.}\n"
        "\\label{tab:physics}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        " & Vacancy & Interstitial & Frenkel pair & Antisite \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def occupancy() -> str:
    occ = json.loads((RES / "point_defects_recheck_20261111/defect_occupancy.json").read_text())
    rows = []
    for split, name in (("test", "Test"), ("val", "Validation")):
        v = occ["validation"][split]

        def share(entry, cls):
            return f"{100 * entry['classes'].get(cls, 0) / entry['examples']:.1f}"

        rows.append(
            f"{name} & {share(v['vacancy'], 'vacancy')} & {share(v['interstitial'], 'interstitial')} & "
            f"{share(v['antisite'], 'antisite-like')} & {v['frenkel_undone']['classes'].get('none', 0)}/{v['frenkel_undone']['examples']} \\\\"
        )  # fmt: skip
    f = occ["validation"]["test"]["frenkel_real"]["classes"]
    classes = (("pair", "Frenkel pair"), ("close pair", "Close pair"), ("antisite-like", "Antisite-like"),
               ("complex", "Complex"), ("none", "No defect"))  # fmt: skip
    rows2 = [f"{label} & {f.get(c, 0)} \\\\" for c, label in classes]
    return (
        "% written by make_manuscript_tables.py\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{\\textbf{Site-occupancy check.} Top: share of defects of known kind that the "
        "occupancy rule classifies as expected (\\%; last column: Frenkel pairs that undid "
        "themselves classified as no defect). Bottom: classes of the 150 real test Frenkel "
        "pairs.}\n"
        "\\label{tab:occupancy}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Split & Vacancy & Interstitial & Antisite & Undone Frenkel pairs \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n"
        "\\begin{tabular}{@{}lc@{}}\n\\toprule\nClass of the relaxed Frenkel pair & Test defects \\\\\n\\midrule\n"
        + "\n".join(rows2)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def copy_with(src: str, dst: str, swaps: tuple[tuple[str, str], ...]) -> None:
    """Copy a table of the final report with a few strings changed (caption, references)."""
    text = (ROOT / "paper" / src).read_text()
    for old, new in swaps:
        if old not in text:
            raise SystemExit(f"{src}: '{old}' not found")
        text = text.replace(old, new)
    (ROOT / "paper" / dst).write_text(f"% copy of {src} written by make_manuscript_tables.py\n" + text)


def main() -> None:
    num: dict[str, str] = {}
    copy_with(
        "fr_table_results.tex",
        "ms_table_results.tex",
        (("\\caption{Test results (\\%).", "\\caption{\\textbf{Finding and restoring the test defects.} Shares in \\%."),),  # fmt: skip
    )
    copy_with(
        "fr_table_models.tex",
        "ms_si_models.tex",
        (
            ("\\caption{Deep learning models explored in this study:", "\\caption{\\textbf{Models explored in this study.} For each:"),  # fmt: skip
            ("Table~\\ref{tab:shapes}", "Supplementary Table~\\ref{tab:shapes}"),
        ),
    )
    (ROOT / "paper/ms_table_design.tex").write_text(design_table())
    (ROOT / "paper/ms_si_physics.tex").write_text(physics(num))
    (ROOT / "paper/ms_si_occupancy.tex").write_text(occupancy())
    lines = [f"\\pdset{{{k}}}{{{v}}}" for k, v in sorted(num.items())]
    (ROOT / "paper/ms_numbers.tex").write_text(
        "% written by make_manuscript_tables.py\n" + "\n".join(lines) + "\n"
    )
    print("numbers:", len(num))


if __name__ == "__main__":
    main()
