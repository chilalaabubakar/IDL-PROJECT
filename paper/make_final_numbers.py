"""Numbers and tables of the final course report (final_report.tex) that the long report lacks.

    python paper/make_final_numbers.py

Reads results/final_report_20261117/ (scripts/pilots/final_report_analysis.py) and writes
    paper/fr_numbers.tex        \\pdset{key}{value}, used as \\pd{key} like pd_numbers.tex
    paper/fr_table_shapes.tex   output shape and parameters of every block of the networks
    paper/fr_table_splits.tex   the finder on training, validation and test defects
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/final_report_20261117"
KINDS = ("vacancy", "interstitial", "antisite", "frenkel")


def pct(x: float) -> str:
    return f"{100 * x:.1f}".rstrip("0").rstrip(".")


def shape(s: list[int]) -> str:
    """[16, 256, 128] -> 16\\times256\\times128 (B = 16 glasses, N = 256 atoms)."""
    return "$" + r"\times".join(str(v) for v in s) + "$"


def shapes_table(sh: dict) -> str:
    def blocks(name):
        return {b["block"]: b for b in sh[name]["blocks"]}

    e, d = blocks("finder (EGNN)"), blocks("denoiser (EGNN)")
    t, c = blocks("transformer"), blocks("cnn")
    rows = [
        r"\multicolumn{3}{@{}l}{\emph{EGNN finder} (the denoiser is the same with a noise-level input and the vector head)} \\",
        r"Input: positions, species & $16\times256\times2$, $16\times256$ & 0 \\",
        rf"Neighbour graph and distance features (16 radial functions) & {shape(e['rbf']['shape'][0])} & 0 \\",
        rf"Input embedding (denoiser: with noise level) & {shape(e['embed']['shape'][0])} & {e['embed']['parameters']:,} ({d['embed']['parameters']:,}) \\",
        rf"$4\times$\{{EGNN layer\}}: features $h$, vector $v$ & {shape(e['layers.0']['shape'][0])}, {shape(e['layers.0']['shape'][1])} & $4\times{e['layers.0']['parameters']:,}$ \\",
        rf"Role head \{{Dense 128, Dense 5\}} (denoiser: vector head, sum of $v$) & {shape(e['head']['shape'][0])} ($16\times256\times2$) & {e['head']['parameters']:,} (0) \\",
        rf"\emph{{Total}} & & {sh['finder (EGNN)']['total']:,} ({sh['denoiser (EGNN)']['total']:,}) \\",
        r"\addlinespace",
        r"\multicolumn{3}{@{}l}{\emph{Transformer finder}} \\",
        rf"Embedding: species, and neighbours of each species by distance & {shape(t['env']['shape'][0])} & {t['species']['parameters'] + t['env']['parameters']:,} \\",
        rf"Pair features from every distance (32 radial functions) & {shape(t['pair']['shape'][0])} & {t['pair']['parameters']:,} \\",
        rf"$4\times$\{{attention with distance bias (8 heads), feed-forward 256\}} & {shape(t['blocks.0']['shape'][0])} & $4\times{t['blocks.0']['parameters']:,}$ \\",
        rf"Layer norm, role head \{{Dense 128, Dense 5\}} & {shape(t['head']['shape'][0])} & {t['norm']['parameters'] + t['head']['parameters']:,} \\",
        rf"\emph{{Total}} & & {sh['transformer']['total']:,} \\",
        r"\addlinespace",
        r"\multicolumn{3}{@{}l}{\emph{CNN finder} (U-Net)} \\",
        r"Glass drawn as a picture, one channel per species & $16\times2\times128\times128$ & 0 \\",
    ]
    enc = [c[k] for k in c if k.startswith("enc.")]
    dec = [c[k] for k in c if k.startswith("dec.")]

    def levels(blocks):  # 16x16x128x128, ... -> 16 x {16, 32, ...} x {128, 64, ...}^2
        ch = ", ".join(str(b["shape"][0][1]) for b in blocks)
        px = ", ".join(str(b["shape"][0][2]) for b in blocks)
        return rf"$16\times\{{{ch}\}}\times\{{{px}\}}^2$"

    rows += [
        rf"Encoder: $5\times$\{{Conv $3\times3$, Conv $3\times3$\}}, average pooling between levels & {levels(enc)} & {sum(b['parameters'] for b in enc):,} \\",
        rf"Decoder: $4\times$\{{upsample, join skip, Conv $3\times3$, Conv $3\times3$\}} & {levels(dec)} & {sum(b['parameters'] for b in dec):,} \\",
    ]
    rows += [
        rf"Read at each atom, add species; role head \{{Dense 64, Dense 5\}} & {shape(c['head']['shape'][0])} & {c['species']['parameters'] + c['head']['parameters']:,} \\",
        rf"\emph{{Total}} & & {sh['cnn']['total']:,} \\",
    ]
    return (
        "% written by make_final_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Output shape and number of parameters of every block, for one batch of 16 "
        "glasses of 256 atoms (shapes: glasses $\\times$ atoms $\\times$ features; for the CNN: "
        "glasses $\\times$ channels $\\times$ pixels $\\times$ pixels). The neighbour graph holds up "
        "to 64 neighbours per atom (about 22 are within the cutoff). Measured on the trained "
        "networks.}\n"
        "\\label{tab:shapes}\n\\scriptsize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xlr@{}}\n\\toprule\n"
        "Block & Output shape & Parameters \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )


def splits_table(sp: dict, num: dict) -> str:
    label = dict(train="Training (200 per type)", val="Validation", test="Test")
    rows = []
    for split in ("train", "val", "test"):
        s = sp[split]
        n = sum(s[k]["examples"] for k in KINDS)
        wce = sum(s[k]["weighted_ce"] * s[k]["examples"] for k in KINDS) / n
        ce = sum(s[k]["ce"] * s[k]["examples"] for k in KINDS) / n
        num[f"split-{split}-wce"], num[f"split-{split}-ce"] = f"{wce:.3f}", f"{ce:.3f}"
        num[f"split-{split}-all"] = pct(s["all"]["found"])
        for k in KINDS:
            num[f"split-{split}-{k}"] = pct(s[k]["found"])
        rows.append(
            f"{label[split]} & {n} & {wce:.3f} & {ce:.3f} & "
            + " & ".join(pct(s[k]["found"]) for k in KINDS)
            + f" & {pct(s['all']['found'])} \\\\"
        )
    return (
        "% written by make_final_numbers.py\n"
        "\\begin{table}[tb]\n\\centering\n"
        "\\caption{The finder of the main results on training, validation and test defects (real "
        "defects only). Loss: the class-weighted cross-entropy it was trained on, and the plain "
        "cross-entropy, per atom. Found (\\%): calibrated, as everywhere in the results.}\n"
        "\\label{tab:splits}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrcccccccc@{}}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{Loss} & \\multicolumn{5}{c}{Found (\\%)} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-9}\n"
        "Defects & $n$ & weighted & plain & Vacancy & Interst. & Antisite & Frenkel & All \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def results_table() -> str:
    """Found and restored in one table (the two tables of the long report, without repeats)."""
    res = json.loads((ROOT / "paper/data/pd_report_results.json").read_text())["kinds"]
    label = dict(vacancy="Vacancy", interstitial="Interstitial", antisite="Antisite", frenkel="Frenkel pair")  # fmt: skip
    methods = ("finder", "geometry", "bonds", "energy")

    def cell(d: dict, bold: bool, ci: bool) -> str:
        r = pct(d["rate"])
        r = f"\\textbf{{{r}}}" if bold else r
        return f"{r} {{\\scriptsize[{100 * d['lo']:.0f}, {100 * d['hi']:.0f}]}}" if ci else r

    rows = {"found": [], "restored": []}
    for what in rows:
        for k in KINDS:
            r = res[k][what]
            best = max(methods, key=lambda m: r[m]["rate"])
            cells = [cell(r[m], m == best, ci=True) for m in methods]
            last = pct(res[k]["chance"]) if what == "found" else cell(r["ceiling"], False, ci=True)
            n = f"{res[k]['n']}" if what == "found" else ""
            rows[what].append(f"{label[k]} & {n} & " + " & ".join(cells) + f" & {last} \\\\")
    return (
        "% written by make_final_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Test results (\\%). Found: the flagged atom lies within $1.5\\sigma$ of the "
        "defect (for a Frenkel pair, of either end; the finder flags the displaced atom itself in "
        "\\pd{frenkel-exact-finder}\\%, the atom energy in \\pd{frenkel-exact-energy}\\%). "
        "Restored: the original glass is recovered after correcting the atoms at the flagged atom "
        "and minimising the energy. Brackets: 95\\% intervals. Bold: the best method. Only the atom "
        "energy uses the potential to flag the atom; every repair ends with minimisation.}\n"
        "\\label{tab:found}\\label{tab:restored}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Defect & $n$ & Learned finder & Cell size & Unlike neighb. & Atom energy & Reference \\\\\n"
        "\\midrule\n"
        "\\multicolumn{6}{@{}l}{\\emph{Found}} & \\emph{random atom} \\\\\n"
        + "\n".join(rows["found"])
        + "\n\\addlinespace\n"
        "\\multicolumn{6}{@{}l}{\\emph{Restored}} & \\emph{true location given} \\\\\n"
        + "\n".join(rows["restored"])
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main() -> None:
    num: dict[str, str] = {}
    (ROOT / "paper/fr_table_results.tex").write_text(results_table())
    sh = json.loads((RES / "shapes.json").read_text())
    (ROOT / "paper/fr_table_shapes.tex").write_text(shapes_table(sh))
    for name, key in (("finder (EGNN)", "egnn"), ("denoiser (EGNN)", "den"), ("transformer", "tr"), ("cnn", "cnn")):  # fmt: skip
        num[f"size-{key}"] = f"{sh[name]['total']:,}"
    num["size-layer"] = f"{sh['finder (EGNN)']['blocks'][2]['parameters']:,}"
    if (RES / "splits.json").exists():
        sp = json.loads((RES / "splits.json").read_text())
        (ROOT / "paper/fr_table_splits.tex").write_text(splits_table(sp, num))
    if (RES / "features.json").exists():
        f = json.loads((RES / "features.json").read_text())
        num["knn-den"] = f"{100 * f['denoiser_knn_balanced_accuracy']:.0f}"
        num["knn-fin"] = f"{100 * f['finder_knn_balanced_accuracy']:.0f}"
        num["feat-atoms"] = f"{f['atoms']:,}"
    if (RES / "sensitivity.json").exists():
        s = json.loads((RES / "sensitivity.json").read_text())
        key = {"finder": "fin", "transformer": "tr", "cnn": "cnn", "egnn_scratch": "scr",
               "cell size": "cell", "unlike neighbours": "bonds", "atom energy": "energy"}  # fmt: skip
        for level in s["levels"]:
            for name, short in key.items():
                v = s["found"][str(level)][name]["all"]
                num[f"noise-{short}-{int(round(1000 * level))}"] = pct(v)
        f0, f20 = (s["found"][lv]["finder"]["all"] for lv in ("0.0", "0.02"))
        num["noise-drop-fin"] = f"{100 * (f0 - f20):.0f}"
        top = max(v["all"] for v in s["found"][str(s["levels"][-1])].values())
        num["noise-max-last"] = pct(top)
    lines = [f"\\pdset{{{k}}}{{{v}}}" for k, v in sorted(num.items())]
    (ROOT / "paper/fr_numbers.tex").write_text(
        "% written by make_final_numbers.py\n" + "\n".join(lines) + "\n"
    )
    print("numbers:", len(num))


if __name__ == "__main__":
    main()
