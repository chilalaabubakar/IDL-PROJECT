"""Numbers and tables of the point-defect report, written from the result files.

    python paper/make_report_numbers.py

Reads results/point_defects_recheck_20261111/ (the test evaluation on real defects, made in
pieces by scripts/pilots/recheck_real_defects.sh):
    chunks/finder_*.json     learned finder and the energy routes (scripts/eval_point_defects.py)
    chunks/baseline_*.json   cell size, and the energy route for antisites
    chunks/bonds_*.json      number of unlike neighbours
    chunks/ceiling_*.json    repair with the true location given
    glass_level.json         is a glass defective at all
    repair_networks.json     the repair networks without a potential
    defect_physics.json      what the relaxed defects are
and writes
    paper/pd_numbers.tex            \\pdset{key}{value} for every number used in the text
    paper/pd_table_found.tex        Table "found"
    paper/pd_table_restored.tex     Table "restored"
    paper/data/pd_report_results.json   the same rates with intervals and tests, for the figure

Intervals are 95% and resample whole test glasses (10,000 bootstrap samples), because the two
defects made in one glass are not independent. Where a rate is 0 or 100%, resampling gives no
spread, and the exact binomial (Clopper-Pearson) interval is given instead. Methods are
compared on the same defects with McNemar's exact test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/point_defects_recheck_20261111"
KINDS = ("vacancy", "interstitial", "antisite", "frenkel")  # the order of the tables
LABEL = {
    "vacancy": "Vacancy",
    "interstitial": "Interstitial",
    "antisite": "Antisite",
    "frenkel": "Frenkel pair",
}
FINDER_ROUTE = {
    "vacancy": "flag by model: insert, minimise",
    "interstitial": "flag by model: remove, minimise",
    "frenkel": "flag by model, pick by model: move, minimise",
    "antisite": "search near model flag, pick by model: swap, minimise",
}
BASELINES = ("geometry", "bonds", "energy")  # cell size, unlike neighbours, atom energy
METHODS = ("finder", *BASELINES)
HEAD = {
    "finder": "Learned finder",
    "geometry": "Cell size",
    "bonds": "Unlike neighbours",
    "energy": "Atom energy",
}
ENERGY_ROUTE = {  # routes that use only the potential
    "vacancy": "flag by energy: insert, minimise",
    "interstitial": "flag by energy: remove, minimise",
    "frenkel": "flag by energy, pick by energy: lowest of all minimised tries",
}


def pct(x: float) -> str:
    """0.695 -> '69.5', 0.5 -> '50'."""
    return f"{100 * x:.1f}".rstrip("0").rstrip(".")


def interval(each, glass, rng) -> tuple[float, float]:
    """95% interval of a rate, resampling whole glasses."""
    each, glass = np.asarray(each, dtype=float), np.asarray(glass)
    ids = np.unique(glass)
    total = np.array([each[glass == g].sum() for g in ids])
    count = np.array([(glass == g).sum() for g in ids])
    draw = rng.integers(0, len(ids), size=(10_000, len(ids)))
    rates = total[draw].sum(1) / count[draw].sum(1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    if lo == hi:  # all or none: resampling has no spread, so the exact binomial interval
        ci = binomtest(int(each.sum()), len(each)).proportion_ci(0.95, method="exact")
        lo, hi = ci.low, ci.high
    return float(lo), float(hi)


def mcnemar(a, b) -> dict:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return dict(only_first=only_a, only_second=only_b, p=float(p))


def pieces(tool: str, kind: str) -> list[dict]:
    """The pieces of one tool and defect kind, in the order of the test examples."""
    files = sorted(
        (RES / "chunks").glob(f"{tool}_{kind}_*.json"), key=lambda f: int(f.stem.split("_")[-1])
    )
    if len(files) != 4:
        raise SystemExit(f"{tool} {kind}: {len(files)} of 4 pieces are there; run the recheck")
    return [json.loads(f.read_text())["kinds"][kind] for f in files]


def join(parts: list[dict], *keys: str) -> list:
    out: list = []
    for part in parts:
        for key in keys:
            part = part[key]
        out += part
    return out


def main() -> None:
    phys = json.loads((RES / "defect_physics.json").read_text())
    level = json.loads((RES / "glass_level.json").read_text())
    rng = np.random.default_rng(0)
    out: dict = {"kinds": {}}
    for k in KINDS:
        e, b, r = pieces("finder", k), pieces("baseline", k), pieces("ceiling", k)
        c = pieces("bonds", k)
        glass = join(e, "each", "glass")
        # the same defects, in the same order, in every tool
        assert glass == join(b, "glass") == join(c, "glass") == join(r, "each", "glass"), k
        found = {
            "finder": join(e, "each", "found", "model"),
            "geometry": join(b, "geometry", "found_each"),
            "bonds": join(c, "bonds", "found_each"),
            "energy": join(e, "each", "found", "energy"),
        }
        assert found["energy"] == join(b, "energy", "found_each"), k  # the two scripts agree
        restored = {
            "finder": join(e, "each", "restored", FINDER_ROUTE[k]),
            "geometry": join(b, "geometry", "restored_each"),
            "bonds": join(c, "bonds", "restored_each"),
            "energy": join(e, "each", "restored", ENERGY_ROUTE[k])
            if k in ENERGY_ROUTE
            else join(b, "energy", "restored_each"),
            "ceiling": join(r, "each", "restored"),
        }
        if k in ("vacancy", "interstitial"):  # the energy routes of the two scripts agree
            assert restored["energy"] == join(b, "energy", "restored_each"), k
        chance = float(np.average([x["chance"] for x in b], weights=[x["examples"] for x in b]))
        res = dict(n=len(glass), chance=chance, found={}, restored={}, tests={})
        if k == "frenkel":  # how often the right atom and the right gap are chosen
            res["flagged_displaced_atom"] = {
                who: float(
                    np.average(
                        [x["repair"][f"flag by {who}: the displaced atom"] for x in e],
                        weights=[x["examples"] for x in e],
                    )
                )
                for who in ("model", "energy")
            }
        for what, table in (("found", found), ("restored", restored)):
            for m, each in table.items():
                lo, hi = interval(each, glass, rng)
                res[what][m] = dict(rate=float(np.mean(each)), lo=lo, hi=hi)
        for other in BASELINES:
            res["tests"][f"found: finder vs {other}"] = mcnemar(found["finder"], found[other])
            res["tests"][f"restored: finder vs {other}"] = mcnemar(
                restored["finder"], restored[other]
            )
        out["kinds"][k] = res
    out["glass_level"] = level["glass_level"]
    out["left_out_already_original"] = level["left_out_already_original"]
    (ROOT / "paper/data").mkdir(exist_ok=True)
    (ROOT / "paper/data/pd_report_results.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---- numbers used in the text
    num: dict[str, str] = {}
    K = out["kinds"]
    for k in KINDS:
        for m in METHODS:
            num[f"found-{m}-{k}"] = pct(K[k]["found"][m]["rate"])
        for m in (*METHODS, "ceiling"):
            num[f"restored-{m}-{k}"] = pct(K[k]["restored"][m]["rate"])
        # the best of the three baselines, and the finder's lead over it
        for what in ("found", "restored"):
            best = max(K[k][what][m]["rate"] for m in BASELINES)
            num[f"{what}-best-{k}"] = pct(best)
            num[f"lead-{what}-{k}"] = f"{100 * (K[k][what]['finder']['rate'] - best):.0f}"
        num[f"n-{k}"] = str(K[k]["n"])
        num[f"chance-{k}"] = pct(K[k]["chance"])
    for m in BASELINES:
        for what in ("found", "restored"):
            rates = [K[k][what][m]["rate"] for k in KINDS]
            num[f"{what}-{m}-min"], num[f"{what}-{m}-max"] = pct(min(rates)), pct(max(rates))
    ceil = [K[k]["restored"]["ceiling"]["rate"] for k in KINDS]
    num["restored-ceiling-min"], num["restored-ceiling-max"] = pct(min(ceil)), pct(max(ceil))
    three = ("vacancy", "interstitial", "antisite")
    ratio = [K[k]["found"]["finder"]["rate"] / K[k]["found"]["energy"]["rate"] for k in three]
    num["ratio-finder-energy-min"] = f"{min(ratio):.1f}"
    num["ratio-finder-energy-max"] = f"{max(ratio):.1f}"
    gap = [100 * (K[k]["found"]["finder"]["rate"] - K[k]["found"]["energy"]["rate"]) for k in three]
    num["gap-finder-energy-min"] = f"{min(gap):.0f}"
    num["gap-finder-energy-max"] = f"{max(gap):.0f}"
    lead = [float(num[f"lead-found-{k}"]) for k in three]
    num["lead-found-min"], num["lead-found-max"] = f"{min(lead):.0f}", f"{max(lead):.0f}"
    # the least significant of the finder's comparisons for the three defects it wins
    for what in ("found", "restored"):
        worst = max(K[k]["tests"][f"{what}: finder vs {m}"]["p"] for k in three for m in BASELINES)
        num[f"p-worst-{what}"] = "p<0.001" if worst < 0.001 else f"p\\le {worst:.3f}"
    num["n-test"] = str(sum(K[k]["n"] for k in KINDS))
    for who, name in (("model", "finder"), ("energy", "energy")):  # exactly the displaced atom
        num[f"frenkel-exact-{name}"] = f"{100 * K['frenkel']['flagged_displaced_atom'][who]:.0f}"
    chance = [K[k]["chance"] for k in KINDS]
    num["chance-min"], num["chance-max"] = f"{100 * min(chance):.0f}", f"{100 * max(chance):.0f}"
    g = out["glass_level"]
    num["glass-clean-raw"] = pct(g["raw"]["defect_free_flagged"])
    num["glass-clean-cal"] = pct(g["calibrated"]["defect_free_flagged"])
    num["glass-def-cal"] = pct(g["calibrated"]["defected_flagged"])
    num["glass-auc"] = f"{g['calibrated']['auc']:.2f}"
    more_open = phys["kinds"]["vacancy"]["fraction_more_open_than_95pct_of_defect_free"]
    num["vacancy-open"] = f"{100 * more_open:.0f}"
    num["vacancy-closed"] = f"{100 * (1 - more_open):.0f}"
    for k in KINDS:
        for key, t in K[k]["tests"].items():
            what, other = key.split(": finder vs ")
            num[f"p-{what}-{other}-{k}"] = "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
    lines = [f"\\pdset{{{k}}}{{{v}}}" for k, v in sorted(num.items())]
    (ROOT / "paper/pd_numbers.tex").write_text(
        "% written by make_report_numbers.py\n" + "\n".join(lines) + "\n"
    )

    # ---- tables
    def cell(d: dict, bold: bool = False) -> str:
        rate = f"\\textbf{{{pct(d['rate'])}}}" if bold else pct(d["rate"])
        return f"{rate} {{\\scriptsize[{100 * d['lo']:.0f}, {100 * d['hi']:.0f}]}}"

    head = " & ".join(HEAD[m] for m in METHODS)
    rows = []
    for k in KINDS:
        f = K[k]["found"]
        best = max(f, key=lambda m: f[m]["rate"])
        rows.append(
            f"{LABEL[k]} & {K[k]['n']} & "
            + " & ".join(cell(f[m], m == best) for m in METHODS)
            + f" & {pct(K[k]['chance'])} \\\\"
        )
    (ROOT / "paper/pd_table_found.tex").write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Defects found (\\%): the flagged atom lies within $1.5\\sigma$ of the defect. "
        "Brackets: 95\\% intervals from resampling whole test glasses. Only the atom energy "
        "uses the interatomic potential. Bold: the best method.}\n"
        "\\label{tab:found}\n\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{No potential} & Potential & \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\n"
        f"Defect & $n$ & {head} & Random atom \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    rows = []
    for k in KINDS:
        r = K[k]["restored"]
        best = max(METHODS, key=lambda m: r[m]["rate"])
        rows.append(
            f"{LABEL[k]} & "
            + " & ".join(cell(r[m], m == best) for m in METHODS)
            + f" & {cell(r['ceiling'])} \\\\"
        )
    (ROOT / "paper/pd_table_restored.tex").write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Original glasses restored (\\%) after the atoms are corrected at the flagged "
        "atom and the energy is minimised (every route uses the potential for this last step). "
        "Columns: how the atom was flagged. Last column: the true location given. "
        "Brackets as in Table~\\ref{tab:found}.}\n"
        "\\label{tab:restored}\n\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        f"Defect & {head} & Location given \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    # ---- the same as Markdown, next to the results
    def md_cell(d: dict) -> str:
        return f"{pct(d['rate'])}% [{100 * d['lo']:.0f}, {100 * d['hi']:.0f}]"

    md = [
        "<!-- written by paper/make_report_numbers.py -->",
        "",
        "Found (the flagged atom within 1.5σ of the defect):",
        "",
        "| Defect | n | " + " | ".join(HEAD[m] for m in METHODS) + " | Random atom |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in KINDS:
        f = K[k]["found"]
        md.append(
            f"| {LABEL[k]} | {K[k]['n']} | "
            + " | ".join(md_cell(f[m]) for m in METHODS)
            + f" | {pct(K[k]['chance'])}% |"
        )
    md += [
        "",
        "Restored (the original glass after correcting the atoms and minimising):",
        "",
        "| Defect | " + " | ".join(HEAD[m] for m in METHODS) + " | True location given |",
        "|---|---|---|---|---|---|",
    ]
    for k in KINDS:
        r = K[k]["restored"]
        md.append(
            f"| {LABEL[k]} | "
            + " | ".join(md_cell(r[m]) for m in METHODS)
            + f" | {md_cell(r['ceiling'])} |"
        )
    md += [
        "",
        "Learned finder against each baseline on the same defects (McNemar's exact test; the "
        "counts are defects that only one of the two gets right):",
        "",
        "| Defect | Comparison | Only the finder | Only the baseline | p |",
        "|---|---|---|---|---|",
    ]
    for k in KINDS:
        for key, t in K[k]["tests"].items():
            p_value = "< 0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
            what, other = key.split(": finder vs ")
            md.append(
                f"| {LABEL[k]} | {what}: finder vs {HEAD[other].lower()} | {t['only_first']} | "
                f"{t['only_second']} | {p_value} |"
            )
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if "flagged_displaced_atom" in K["frenkel"]:
        print(
            "Frenkel, flagged exactly the displaced atom:", K["frenkel"]["flagged_displaced_atom"]
        )
    print("numbers:", len(num))


if __name__ == "__main__":
    main()
