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
and, from the finder experiments when their files are there,
    results/point_defects_ablations_20261112/ablations.json   (scripts/pilots/finder_ablations.py)
    results/point_defects_seeds_20261114/seeds.json           (scripts/pilots/finder_seeds.py)
    results/point_defects_architectures_20261115/architectures.json
                                                      (scripts/pilots/finder_architectures.py)
    paper/pd_table_experiments.tex  Table "experiments": the comparisons made on the way
    paper/pd_table_ablations.tex    Table "ablations": six settings, three seeds each
    paper/pd_table_stability.tex    Table "stability": gradients and spikes during training
    paper/pd_table_finders.tex      Table "finders": transformer and CNN against the EGNN

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
ABL = ROOT / "results/point_defects_ablations_20261112/ablations.json"
SEEDS = ROOT / "results/point_defects_seeds_20261114/seeds.json"
ARCH = ROOT / "results/point_defects_architectures_20261115/architectures.json"
ABLATION_ROWS = (  # run, what changed, compared with
    ("reference", "Reference: starts from the denoiser, 4 layers, class weights", None),
    ("scratch", "Random starting weights", "reference"),
    ("layers2", "Random starting weights, 2 layers (reach $5\\sigma$)", "scratch"),
    ("layers1", "Random starting weights, 1 layer (reach $2.5\\sigma$)", "scratch"),
    ("no_weights", "No class weights", "reference"),
    ("real_only", "Trained without the Frenkel pairs that undid themselves", "reference"),
)
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


def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down correction for multiple comparisons (adjusted p values, same order)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def p_text(p: float) -> str:
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$" if p < 0.1 else f"$p={p:.2f}$"


def one(x: float) -> str:
    """0.6297 -> '63.0': a mean over seeds, always with one decimal."""
    return f"{100 * x:.1f}"


def span(values, digits: int = 0, scale: float = 1.0) -> str:
    """Lowest and highest of the values, '38--86'; one number when they are equal."""
    lo, hi = f"{scale * min(values):.{digits}f}", f"{scale * max(values):.{digits}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def experiments(K: dict) -> dict[str, str]:
    """Tables "experiments" and "ablations" and their numbers, from the finder experiments."""
    num: dict[str, str] = {}
    rn = json.loads((RES / "repair_networks.json").read_text())["kinds"]
    removed = {
        v: [1 - rn[k][f"{v}, one step"]["defect_rms"] / rn[k]["nothing"]["defect_rms"] for k in rn]
        for v in ("v1", "v2")
    }
    overlap = {n: [rn[k][f"v2, {n}"]["overlap"] for k in rn] for n in ("one step", "10 steps")}

    def span(values) -> str:
        return f"{100 * min(values):.0f}--{100 * max(values):.0f}"

    num["repair-removed-v1"], num["repair-removed-v2"] = span(removed["v1"]), span(removed["v2"])
    num["repair-overlap-1"], num["repair-overlap-10"] = (
        span(overlap["one step"]),
        span(overlap["10 steps"]),
    )
    level = json.loads((RES / "glass_level.json").read_text())["glass_level"]
    rows = [
        "1 & Training on defects & denoiser alone $\\to$ trained finder & found: vacancy / "
        "interstitial / antisite & 44 / 12 / 15 $\\to$ "
        + " / ".join(pct(K[k]["found"]["finder"]["rate"]) for k in KINDS[:3])
        + " \\\\"
    ]
    abl = json.loads(ABL.read_text()) if ABL.exists() else None
    if abl:
        t, tests = abl["trained"], abl["trained_tests"]
        a, b = t["20,000 updates"]["all"]["rate"], t["60,000 updates"]["all"]["rate"]
        u = t["60,000 updates, not calibrated"]["all"]["rate"]
        p_len = tests["60,000 against 20,000 updates"]["all"]["p"]
        p_cal = tests["calibrated against not calibrated"]["all"]["p"]
        num.update({"exp-20k": pct(a), "exp-60k": pct(b), "exp-uncal": pct(u)})
        num["exp-p-length"], num["exp-p-cal"] = p_text(p_len), p_text(p_cal)
        rows += [
            f"2 & Training length & 20,000 $\\to$ 60,000 updates & found, all {sum(K[k]['n'] for k in KINDS)} "
            f"defects & {pct(a)} $\\to$ {pct(b)} ({p_text(p_len)}) \\\\",
            "3 & Calibration of the finder & off $\\to$ on & defect-free glasses flagged; found, "
            f"all defects & {pct(level['raw']['defect_free_flagged'])} $\\to$ "
            f"{pct(level['calibrated']['defect_free_flagged'])}; {pct(u)} $\\to$ {pct(b)} "
            f"({p_text(p_cal)}) \\\\",
        ]
    rows += [
        "4 & Loss of the repair network & version 1 $\\to$ version 2 & error on the displaced "
        f"atoms removed by one step & {num['repair-removed-v1']} $\\to$ {num['repair-removed-v2']} "
        "\\\\",
        " & Steps of the repair network & 1 $\\to$ 10 steps (version 2) & glasses left with "
        f"overlapping atoms & {num['repair-overlap-1']} $\\to$ {num['repair-overlap-10']} \\\\",
        " & Repair network before minimisation & without $\\to$ with (version 2) & restored: "
        "vacancy / interstitial$^{\\dagger}$ & 51 / 34.5 $\\to$ 51.5 / 34 \\\\",
        "5 & Check of the labels & all 200 $\\to$ the "
        f"{K['frenkel']['n']} real Frenkel pairs & Frenkel pairs found / restored & 37 / 18 $\\to$ "
        f"{pct(K['frenkel']['found']['finder']['rate'])} / "
        f"{pct(K['frenkel']['restored']['finder']['rate'])} \\\\",
    ]
    (ROOT / "paper/pd_table_experiments.tex").write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{The comparisons made on the way, on the test defects (\\%). The numbers at the "
        "left are those of the experimental iterations. Some of these choices were made after "
        "test results of an earlier version had been seen, so the comparisons are a record "
        "made after the fact, not independent tests (Section~\\ref{sec:iterations}). "
        "$^{\\dagger}$Measured with the finder after 20,000 updates.}\n"
        "\\label{tab:experiments}\n\\footnotesize\n"
        "\\begin{tabularx}{\\linewidth}{@{}r>{\\raggedright\\arraybackslash}p{1.12in}"
        ">{\\raggedright\\arraybackslash}p{1.2in}>{\\raggedright\\arraybackslash}X"
        ">{\\raggedright\\arraybackslash}p{1.25in}@{}}\n\\toprule\n"
        " & Experiment & Compared & Measure & Result \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )

    # ---- the ablation runs, three seeds each: only when all are there
    table = ROOT / "paper/pd_table_ablations.tex"
    runs = abl["ablations"] if abl else {}
    seeds = json.loads(SEEDS.read_text()) if SEEDS.exists() else {"variants": {}}
    if not all(
        name in runs and len(seeds["variants"].get(name, {}).get("seeds", [])) >= 3
        for name, _, _ in ABLATION_ROWS
    ):
        table.unlink(missing_ok=True)
        return num
    order = (*KINDS, "all")

    def each(name: str) -> dict[int, float]:  # found, all defects, per seed
        return {
            s: seeds["runs"][f"{name} seed {s}"]["found"]["all"]["rate"]
            for s in seeds["variants"][name]["seeds"]
        }

    rows = []
    for name, what, base in ABLATION_ROWS:
        f = seeds["variants"][name]["found"]
        key = name.replace("_", "-")  # no underscore in the name of a number in the text
        cells = [one(f[k]["mean"]) for k in order]
        cells[-1] += f" $\\pm$ {100 * f['all']['sd']:.1f}"
        if base:
            a, b = each(name), each(base)
            diff = [a[s] - b[s] for s in a]
            mean = 100 * float(np.mean(diff))
            lower = sum(d < 0 for d in diff)
            num[f"abl-diff-{key}"] = f"{abs(mean):.1f}"
            num[f"abl-lower-{key}"] = str(lower)
            shown = "0.0" if abs(mean) < 0.05 else f"{mean:+.1f}".replace("-", "$-$")
            cells.append(f"{shown} ({lower} of {len(diff)})")
        else:
            cells.append("--")
        for k in order:
            num[f"abl-{key}-{k}"] = one(f[k]["mean"])
        num[f"abl-{key}-sd"] = f"{100 * f['all']['sd']:.1f}"
        num[f"abl-{key}-low"], num[f"abl-{key}-high"] = one(f["all"]["low"]), one(f["all"]["high"])
        # validation glasses, first run: atoms of a defect role that are given that role
        val = runs[name]["validation"]
        num[f"abl-label-{key}-vacancy"] = f"{100 * val['vacancy_recall_near_vacancy']:.0f}"
        num[f"abl-label-{key}-interstitial"] = (
            f"{100 * val['interstitial_recall_interstitial']:.0f}"
        )
        size = f"{runs[name]['parameters'] / 1e6:.2f}"
        rows.append(f"{what} & {size} & " + " & ".join(cells) + " \\\\")
        if name in ("reference", "layers1"):
            rows.append("\\addlinespace")
    num["abl-updates"] = f"{runs['reference']['updates']:,}"
    num["abl-seeds"] = "three"
    agree = seeds["reference_agreement"]
    num["seed-agree"] = f"{100 * (agree['found_by_all'] + agree['found_by_none']):.0f}"
    num["seed-by-all"], num["seed-by-none"] = (
        one(agree["found_by_all"]),
        one(agree["found_by_none"]),
    )
    num["seed-ensemble"] = one(agree["ensemble"]["all"]["rate"])
    ref = sorted(each("reference").values())
    num["seed-ref-runs"] = ", ".join(one(r) for r in ref[:-1]) + f" and {one(ref[-1])}"
    swing = seeds["variants"]["real_only"]["difference_from_reference_same_seed"]
    num["abl-real-only-behind"], num["abl-real-only-ahead"] = (
        f"{abs(100 * swing['low']):.1f}",
        f"{100 * swing['high']:.1f}",
    )
    table.write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{Ablations of the finder: defects found (\\%) on the real test defects after "
        f"{num['abl-updates']} updates, one setting changed per row. Every row is the mean of "
        "three training runs that differ only in the random seed; for all defects the spread "
        "between the three runs (standard deviation) follows the $\\pm$. Size in millions of "
        "parameters. Difference: in all defects found, from the row it is compared with, run "
        "by run with the same seed; in brackets, in how many of the three runs the row was "
        "lower. The row compared with is the reference, or, for the two rows with fewer "
        "layers, the row with random starting weights and 4 layers.}\n"
        "\\label{tab:ablations}\n\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xcccccll@{}}\n"
        "\\toprule\n"
        "Setting & Size & Vacancy & Interst. & Antisite & Frenkel & All & Difference \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )
    num.update(stability(seeds))
    num.update(finders())
    return num


OCC = RES / "defect_occupancy.json"
OCC_KEY = {"pair": "pair", "close pair": "close", "antisite-like": "anti", "complex": "complex", "none": "none"}  # fmt: skip


def occupancy(each: dict) -> dict[str, str]:
    """Vacancies by the removed species, and Frenkel pairs by what they became (definition C:
    scripts/pilots/defect_occupancy.py): rates of every method on each subset."""
    num: dict[str, str] = {}
    if not OCC.exists():
        return num
    occ = json.loads(OCC.read_text())
    rng = np.random.default_rng(2)
    methods = (*METHODS, "ceiling")

    def rates(kind: str, mask: np.ndarray, prefix: str) -> None:
        e = each[kind]
        glass = np.asarray(e["glass"])[mask]
        num[f"{prefix}-n"] = str(int(mask.sum()))
        for what in ("found", "restored"):
            for m in methods:
                if m not in e[what]:
                    continue
                hit = np.asarray(e[what][m], dtype=float)[mask]
                lo, hi = interval(hit, glass, rng)
                num[f"{prefix}-{what}-{m}"] = pct(float(hit.mean()))
                num[f"{prefix}-{what}-{m}-lo"] = f"{100 * lo:.0f}"
                num[f"{prefix}-{what}-{m}-hi"] = f"{100 * hi:.0f}"
        for what in ("found", "restored"):
            t = mcnemar(np.asarray(e[what]["finder"])[mask], np.asarray(e[what]["energy"])[mask])
            num[f"{prefix}-p-{what}-energy"] = p_text(t["p"])

    # ---- vacancies: the removed atom was A or B
    v = occ["vacancy"]
    assert v["glass"] == each["vacancy"]["glass"]  # the same defects, in the same order
    removed = np.asarray(v["removed_species"])
    for s, name in ((0, "vacA"), (1, "vacB")):
        rates("vacancy", removed == s, name)
    # ---- Frenkel pairs: what the 150 real ones are (definition C)
    f = occ["frenkel"]
    real = np.asarray(f["real"])
    cls = np.asarray(f["cls"])[real]
    assert np.asarray(f["glass"])[real].tolist() == each["frenkel"]["glass"]
    for c, key in OCC_KEY.items():
        num[f"fp-n-{key}"] = str(int((cls == c).sum()))
    num["fp-n-pair-pure"] = str(
        int(
            ((np.asarray(f["cls"]) == "pair") & (np.asarray(f["species_changed"]) == 0))[real].sum()
        )
    )
    rates("frenkel", cls == "pair", "fpair")
    rates("frenkel", cls != "pair", "fother")
    # ---- how well the rule works on defects whose answer is known (test split)
    val = occ["validation"]["test"]
    num["occ-vac"] = pct(val["vacancy"]["classes"].get("vacancy", 0) / val["vacancy"]["examples"])
    num["occ-int"] = pct(
        val["interstitial"]["classes"].get("interstitial", 0) / val["interstitial"]["examples"]
    )
    num["occ-anti"] = pct(
        val["antisite"]["classes"].get("antisite-like", 0) / val["antisite"]["examples"]
    )
    a = val["antisite"]["classes"]
    num["occ-anti-pairlike"] = pct((a.get("pair", 0) + a.get("close pair", 0)) / val["antisite"]["examples"])  # fmt: skip
    num["occ-undone"] = str(val["frenkel_undone"]["classes"].get("none", 0))
    num["occ-undone-n"] = str(val["frenkel_undone"]["examples"])
    return num


def local_pressure(finder: dict) -> dict[str, str]:
    """A fourth measure without learning, the local pressure of each atom (found only).

    From structural_signatures.json; the rule was chosen on the validation glasses. Its own
    random numbers, so that the intervals of the tables do not depend on this check.
    """
    num: dict[str, str] = {}
    path = RES / "structural_signatures.json"
    sig = json.loads(path.read_text()) if path.exists() else {}
    if "local_pressure" not in sig:
        return num
    rng = np.random.default_rng(1)
    rates = {}
    for variant, key in (("mean_area", "press"), ("own_cell", "presscell")):
        for k in KINDS:
            e = sig["local_pressure"][variant][k]
            hits, glass = finder[k]
            assert e["glass"] == glass, k  # the same defects, in the same order
            lo, hi = interval(e["found_each"], glass, rng)
            rate = float(np.mean(e["found_each"]))
            rates[(key, k)] = rate
            num[f"{key}-{k}"] = pct(rate)
            num[f"{key}-{k}-lo"], num[f"{key}-{k}-hi"] = f"{100 * lo:.0f}", f"{100 * hi:.0f}"
            num[f"{key}-{k}-rule"] = e["rule"]
            num[f"{key}-p-{k}"] = p_text(mcnemar(hits, e["found_each"])["p"])
    ref = sig["local_pressure"]["defect_free_virial"]
    num["press-ref-a"] = f"{ref['A']['mean']:.1f}\\pm{ref['A']['sd']:.1f}"
    num["press-ref-b"] = f"{ref['B']['mean']:.1f}\\pm{ref['B']['sd']:.1f}"
    on_val = sig["local_pressure"]["mean_area"]["interstitial"]["validation"]
    num["press-val-unusual"], num["press-val-highest"] = (
        pct(on_val["most unusual"]),
        pct(on_val["highest"]),
    )
    for key in ("press", "presscell"):
        values = [rates[(key, k)] for k in KINDS]
        num[f"{key}-min"], num[f"{key}-max"] = pct(min(values)), pct(max(values))
    return num


def stability(seeds: dict) -> dict[str, str]:
    """Table "stability" and its numbers: gradients and spikes in the logged training runs."""
    num: dict[str, str] = {}
    arch = json.loads(ARCH.read_text())["models"] if ARCH.exists() else {}
    short = seeds["stability"]  # "<variant> seed <s>": the 2,000-update runs
    groups = [  # key, what, start, the runs
        ("ref", "EGNN", "denoiser", [v for k, v in short.items() if k.startswith("reference ")]),
        ("now", "EGNN, no class weights", "denoiser",
         [v for k, v in short.items() if k.startswith("no_weights ")]),
        ("scr", "EGNN", "random", [v for k, v in short.items() if k.startswith("scratch ")]),
    ]  # fmt: skip
    long = (("egnn", "EGNN", "egnn_scratch"), ("tr", "Transformer", "attention"), ("cnn", "CNN", "cnn"))  # fmt: skip
    for key, what, model in long:
        if model in arch:
            groups.append((key, what, "random", list(arch[model]["stability"].values())))
    rows, total = [], 0
    for key, what, start, runs in groups:
        if not runs:
            continue
        total += len(runs)
        g = dict(
            updates=f"{runs[0]['updates']:,}",
            runs=str(len(runs)),
            median=span([r["grad_norm"]["median"] for r in runs], 1),
            largest=span([r["grad_norm"]["largest"] for r in runs], 0 if key in ("ref", "egnn") else 1),
            clipped=span([r["clipped"] for r in runs], 0, 100),
            gspikes=span([len(r["grad_spikes"]) for r in runs]),
            lspikes=span([len(r["loss_spikes"]) for r in runs]),
        )  # fmt: skip
        if key == "now":  # gradients ten times smaller: two digits
            g["median"] = span([r["grad_norm"]["median"] for r in runs], 2)
        for k, v in g.items():
            num[f"stab-{key}-{k}"] = v
        first = [min(s["step"] for s in r["grad_spikes"]) for r in runs if r["grad_spikes"]]
        if first:  # the update of the first gradient spike, earliest and latest run
            num[f"stab-{key}-first"] = f"{min(first):,} and {max(first):,}"
        rows.append(
            f"{what} & {start} & {g['updates']} & {g['runs']} & {g['median']} & {g['largest']} & "
            f"{g['clipped']}\\% & {g['gspikes']} & {g['lspikes']} \\\\"
        )
        if key == "scr":
            rows.append("\\addlinespace")
    every = list(short.values()) + [
        r for m in arch.values() for r in m.get("stability", {}).values()
    ]
    num["stab-runs"] = str(len(every))
    weighted = [r for key, _, _, runs in groups if key in ("ref", "scr", "egnn") for r in runs]
    num["stab-egnn-cut"] = span([r["clipped"] for r in weighted], 0, 100)  # EGNN, class weights
    num["stab-max-ratio"] = f"{max(s['ratio'] for r in every for s in r['grad_spikes']):.0f}"
    num["stab-not-finite"] = str(sum(r["not_finite"] for r in every))
    table = ROOT / "paper/pd_table_stability.tex"
    if len(groups) < 6:
        table.unlink(missing_ok=True)
        return num
    table.write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Training stability. Gradient: the size (norm) of the gradient of the loss "
        "in one update, before it is cut to 1; the median and the largest value of a run. "
        "Cut: the share of updates whose gradient was larger than 1 and was scaled down. "
        "Gradient spike: a gradient more than 5 times the median of the 100 updates before "
        "it. Loss spike: a loss more than 3 times the median of the last 50 updates on the "
        "same kind of batch. Where the runs differ, the lowest and the highest are given.}\n"
        "\\label{tab:stability}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xlrcccccc@{}}\n"
        "\\toprule\n"
        " & & & & \\multicolumn{2}{c}{Gradient} & & \\multicolumn{2}{c}{Spikes} \\\\\n"
        "\\cmidrule(lr){5-6}\\cmidrule(l){8-9}\n"
        "Finder & Start & Updates & Runs & median & largest & Cut & gradient & loss \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )
    return num


FINDER_ROWS = (  # model in architectures.json, key in the text, finder, start, updates
    ("attention", "tr", "Transformer", "random", "20,000"),
    ("cnn", "cnn", "CNN", "random", "20,000"),
    ("egnn_scratch", "egnn", "EGNN", "random", "20,000"),
    ("egnn_pretrained", "pre", "EGNN", "denoiser", "20,000"),
    ("egnn_pretrained_60k", "main", "EGNN, the finder of the main results", "denoiser", "60,000"),
)


def finders() -> dict[str, str]:
    """Table "finders" and its numbers: the transformer and the CNN against the EGNN."""
    num: dict[str, str] = {}
    table = ROOT / "paper/pd_table_finders.tex"
    arch = json.loads(ARCH.read_text()) if ARCH.exists() else {"models": {}}
    models = arch["models"]
    if not all(m in models for m, *_ in FINDER_ROWS):
        table.unlink(missing_ok=True)
        return num
    order = (*KINDS, "all")
    rows = []
    for model, key, what, start, updates in FINDER_ROWS:
        e = models[model]
        f, n = e["found"], len(e["seeds"])
        cells = [one(f[k]["mean"]) for k in order]
        if n > 1:
            cells[-1] += f" $\\pm$ {100 * f['all']['sd']:.1f}"
        for k in order:
            num[f"fin-{key}-{k}"] = one(f[k]["mean"])
        if n > 1:
            num[f"fin-{key}-low"], num[f"fin-{key}-high"] = (
                one(f["all"]["low"]),
                one(f["all"]["high"]),
            )
            num[f"fin-{key}-hours"] = f"{np.mean(e['hours_per_run']):.1f}"
        num[f"fin-{key}-size"] = f"{e['parameters'] / 1e6:.2f}"
        num[f"fin-{key}-shift"] = f"{100 * e['symmetry']['random shift']:.0f}"
        num[f"fin-{key}-noshift"] = f"{100 * (1 - e['symmetry']['random shift']):.0f}"
        rows.append(
            f"{what} & {start} & {updates} & {n} & {e['parameters'] / 1e6:.2f} & "
            + " & ".join(cells)
            + " \\\\"
        )
        if key == "cnn":
            rows.append("\\addlinespace")
    # against the EGNN trained the same way, per seed: the least clear of the three seeds
    for key, model in (("tr", "attention"), ("cnn", "cnn"), ("pre", "egnn_pretrained")):
        tests = models[model]["against_egnn_control"].values()
        num[f"fin-p-{key}"] = p_text(max(t["p"] for t in tests))
        num[f"fin-only-{key}"] = span([t["only_new"] for t in tests])
        num[f"fin-lost-{key}"] = span([t["only_base"] for t in tests])
    pair = arch["between"]["attention vs cnn"]  # transformer against CNN, per seed
    ps = [t["all"]["p"] for t in pair.values()]
    num["fin-p-tr-cnn"] = f"$p={min(ps):.2f}$--${max(ps):.2f}$" if min(ps) >= 0.095 else p_text(min(ps))  # fmt: skip
    for k in KINDS:
        num[f"fin-p-tr-cnn-{k}"] = f"{min(t[k]['p'] for t in pair.values()):.2f}"
    agree = arch["agreement"]
    num["fin-by-all"] = span([a["by_all"] for a in agree.values()], 0, 100)
    num["fin-by-none"] = span([a["by_none"] for a in agree.values()], 0, 100)
    num["fin-by-one"] = span([a["by_at_least_one"] for a in agree.values()], 0, 100)
    together = arch["together"]
    names = {
        "pair": "transformer + CNN, seed 0",
        "six": "transformer + CNN, 3 seeds each",
        "three": "transformer + CNN + EGNN control, seed 0",
        "report": "transformer + CNN (seed 0) + the finder of the report",
        "tr3": "transformer, 3 seeds",
        "cnn3": "CNN, 3 seeds",
        "egnn3": "EGNN control, 3 seeds",
    }
    for key, name in names.items():
        e = together[name]
        for k in order:
            num[f"fin-{key}-{k}"] = one(e["found"][k]["rate"])
        num[f"fin-{key}-lo"] = f"{100 * e['found']['all']['lo']:.0f}"
        num[f"fin-{key}-hi"] = f"{100 * e['found']['all']['hi']:.0f}"
        num[f"fin-{key}-p"] = p_text(e["against_best_member"]["p"])
    e = together[names["six"]]
    cells = [one(e["found"][k]["rate"]) for k in order]
    rows += [
        "\\addlinespace",
        "Transformer and CNN together (all six runs) & random & 20,000 & 6 & -- & "
        + " & ".join(cells)
        + " \\\\",
    ]
    table.write_text(
        "% written by make_report_numbers.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Three kinds of network as finder: defects found (\\%) on the real test "
        "defects. The first three rows were trained in exactly the same way, from random "
        "starting weights; each is the mean of three runs that differ in the random seed, "
        "with the spread between the runs after the $\\pm$. The two rows below them are the "
        "EGNN that starts from the denoiser (one run each). Last row: the six transformer "
        "and CNN runs used together, by averaging their probabilities. Size in millions of "
        "parameters.}\n"
        "\\label{tab:finders}\n\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xlrccccccl@{}}\n"
        "\\toprule\n"
        "Finder & Start & Updates & Runs & Size & Vacancy & Interst. & Antisite & Frenkel & All "
        "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )
    return num


def main() -> None:
    phys = json.loads((RES / "defect_physics.json").read_text())
    level = json.loads((RES / "glass_level.json").read_text())
    rng = np.random.default_rng(0)
    out: dict = {"kinds": {}}
    each_found: dict = {}  # per kind: the finder's result on every test defect, and its glass
    per_defect: dict = {}  # per kind: every method's result on every test defect
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
        each_found[k] = (found["finder"], glass)
        per_defect[k] = dict(found=found, restored=restored, glass=glass)
    # Holm's correction within each family of twelve comparisons (finder against the three
    # measures, four defect types): one family for "found", one for "restored"
    for what in ("found", "restored"):
        keys = [(k, f"{what}: finder vs {m}") for k in KINDS for m in BASELINES]
        adjusted = holm([out["kinds"][k]["tests"][key]["p"] for k, key in keys])
        for (k, key), p_adj in zip(keys, adjusted):
            out["kinds"][k]["tests"][key]["p_holm"] = p_adj
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
    for what in ("found", "restored"):  # after Holm's correction
        worst = max(
            K[k]["tests"][f"{what}: finder vs {m}"]["p_holm"] for k in three for m in BASELINES
        )
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
            for name, p_value in ((f"p-{what}-{other}-{k}", t["p_holm"]), (f"praw-{what}-{other}-{k}", t["p"])):  # fmt: skip
                num[name] = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    num["n-tests-family"] = str(len(KINDS) * len(BASELINES))
    num.update(experiments(K))
    num.update(occupancy(per_defect))
    num.update(local_pressure(each_found))
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
        "Brackets: the range that holds the true rate with 95\\% confidence "
        "(Section~\\ref{sec:test}). Only the atom energy uses the interatomic potential. "
        "Bold: the best method. Frenkel pair: a flag within $1.5\\sigma$ of either end of the "
        "pair counts; the finder flags the displaced atom itself in \\pd{frenkel-exact-finder}\\% "
        "of the pairs and the atom energy in \\pd{frenkel-exact-energy}\\%.}\n"
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
        "| Defect | Comparison | Only the finder | Only the baseline | p | p, Holm (12 per family) |",
        "|---|---|---|---|---|---|",
    ]
    for k in KINDS:
        for key, t in K[k]["tests"].items():
            p_raw, p_adj = ("< 0.001" if x < 0.001 else f"{x:.3f}" for x in (t["p"], t["p_holm"]))
            what, other = key.split(": finder vs ")
            md.append(
                f"| {LABEL[k]} | {what}: finder vs {HEAD[other].lower()} | {t['only_first']} | "
                f"{t['only_second']} | {p_raw} | {p_adj} |"
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
