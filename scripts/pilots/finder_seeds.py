"""Seed study of the point-defect finder: how stable is its training, and how much do its
results depend on the seed?

    bash scripts/pilots/finder_seeds.sh        # trains the runs (about 10 hours on 4 CPU cores)
    python scripts/pilots/finder_seeds.py      # this analysis

Runs: the six 2,000-update variants of the ablation table, each with seeds 0, 1 and 2. Seed 0
is the ablation run (results/point_defects_ablations_20261112/), or its logged rerun where one
exists; seeds 1 and 2 are in results/point_defects_seeds_20261114/runs/. A run that has not
finished yet is left out, so the script can be run while the study is going.

1. Training stability, from each run's steps.csv (one row per update):
     gradient norm before clipping: median, 99th percentile, largest, and the share of updates
       it was clipped (above the clip of 1.0), overall and in the last 1,000 updates;
     gradient spikes: a norm more than 5 times the median of the previous 100 updates;
     loss spikes: a loss more than 3 times the median of the previous 50 updates of the same
       batch type (each batch holds one defect type, or no defect, so the loss differs
       between batch types by design); the first 100 updates are not judged;
     updates with a loss or norm that is not a finite number.
2. Results, on the 750 real test defects (the flagged atom within 1.5 sigma of the defect,
   calibrated, as scripts/pilots/finder_ablations.py), scored with the trained weights:
     per seed and defect type, with 95% intervals from resampling test glasses;
     per variant, the mean and spread over seeds, and each variant's difference from the
       reference with the same seed (the same seed gives the same order of training batches);
     for the reference: how often the three seeds find the same defects, and the found rate
       of the three averaged (an ensemble);
     whether the logged seed-0 rerun has exactly the weights of the published seed-0 run.

Writes <out>/seeds.json, <out>/summary.md, <out>/stability.png and, for each finished run of
the study, <out>/<run>.pt (settings and trained weights), .config.yaml, .log.jsonl and
.steps.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ablations", ROOT / "scripts/pilots/finder_ablations.py"
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)
ev = fa.ev

ABLATION_DIR = ROOT / "results/point_defects_ablations_20261112"
VARIANTS = {name: what for name, what in fa.ABLATIONS.items()}
SEEDS = (0, 1, 2)
CLIP = 1.0
LOSS_SPIKE, LOSS_WINDOW, WARM = 3.0, 50, 100
GRAD_SPIKE, GRAD_WINDOW = 5.0, 100


# ------------------------------------------------------------------------- runs on disk


def finished_runs(out: Path) -> dict[tuple[str, int], dict]:
    """(variant, seed) -> the weights to score, and the step log if there is one."""
    runs: dict[tuple[str, int], dict] = {}
    for name in VARIANTS:
        published = ABLATION_DIR / f"{name}.pt"
        if published.exists():
            runs[(name, 0)] = dict(weights=published, steps=None, source="ablation run")
        for seed in SEEDS:
            found = sorted((out / "runs").glob(f"*_{name}_s{seed}"))
            if not found:
                continue
            slim = fa.slim(found[-1], out)
            if slim is None:  # still training
                continue
            trained, _, info = slim
            steps = out / f"{name}_s{seed}.steps.csv"
            shutil.copyfile(found[-1] / "steps.csv", steps)
            runs[(name, seed)] = dict(weights=trained, steps=steps, source="seed study", info=info)
    return runs


def same_weights(a: Path, b: Path) -> bool:
    wa = torch.load(a, map_location="cpu", weights_only=False)["ema"]
    wb = torch.load(b, map_location="cpu", weights_only=False)["ema"]
    return wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)


# ------------------------------------------------------------------------- stability


def read_steps(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return dict(
        step=np.array([int(r["step"]) for r in rows]),
        batch=np.array(["defect-free" if r["clean"] == "1" else r["kind"] for r in rows]),
        loss=np.array([float(r["loss"]) for r in rows]),
        grad=np.array([float(r["grad_norm"]) for r in rows]),
    )


def stability(path: Path) -> dict:
    s = read_steps(path)
    step, batch, loss, grad = s["step"], s["batch"], s["loss"], s["grad"]
    finite = np.isfinite(loss) & np.isfinite(grad)
    late = step > step.max() - 1000
    grad_spikes, loss_spikes = [], []
    for i in range(WARM, len(step)):
        if not finite[i]:
            continue
        before = grad[max(0, i - GRAD_WINDOW) : i]
        ref = np.median(before[np.isfinite(before)])
        if grad[i] > GRAD_SPIKE * ref:
            grad_spikes.append(
                dict(step=int(step[i]), batch=str(batch[i]), ratio=round(grad[i] / ref, 2))
            )
        same = np.nonzero((batch[:i] == batch[i]) & finite[:i])[0][-LOSS_WINDOW:]
        if len(same) >= 10:
            ref = np.median(loss[same])
            if loss[i] > LOSS_SPIKE * ref:
                loss_spikes.append(
                    dict(step=int(step[i]), batch=str(batch[i]), ratio=round(loss[i] / ref, 2))
                )
    g = grad[finite]
    worst = int(np.nanargmax(np.where(finite, grad, -np.inf)))
    by_batch = {
        b: round(float(np.median(loss[(batch == b) & late & finite])), 4)
        for b in sorted(set(batch))
        if ((batch == b) & late & finite).any()
    }
    return dict(
        updates=int(len(step)),
        not_finite=int((~finite).sum()),
        grad_norm=dict(
            median=round(float(np.median(g)), 3),
            p99=round(float(np.percentile(g, 99)), 3),
            largest=round(float(g.max()), 3),
            largest_at=int(step[worst]),
            largest_batch=str(batch[worst]),
            median_first_100=round(float(np.median(grad[:100])), 3),
            median_last_1000=round(float(np.median(grad[late & finite])), 3),
        ),
        clipped=round(float((g > CLIP).mean()), 3),
        clipped_last_1000=round(float((grad[late & finite] > CLIP).mean()), 3),
        grad_spikes=grad_spikes,
        loss_spikes=loss_spikes,
        loss_last_1000_by_batch=by_batch,
    )


# ------------------------------------------------------------------------- results


def scores(path: Path, data: dict) -> dict[str, np.ndarray]:
    """Calibrated probability of the role that belongs to each defect, for every atom."""
    log_prior = ev.finder_log_prior(str(path), ROOT / "data/ka2d_256_point_defects")[0]
    models = ev.Models(
        str(path), str(path), threads=4, device="cpu", log_prior=log_prior, do_heal=False
    )
    return {
        name: models.role_prob(d.pos.numpy(), d.types.numpy(), d.box.numpy())[
            ..., int(ev.REPAIR_ROLE[d.kind])
        ]
        for name, d in data.items()
    }


def hits(score: dict[str, np.ndarray], data: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, d in data.items():
        pos, box, sites = d.pos.numpy(), d.box.numpy(), d.sites.numpy()
        top = pos[np.arange(len(pos)), score[name].argmax(-1)]
        dist = np.stack(
            [np.linalg.norm(ev.wrap(top - sites[:, k], box), axis=-1) for k in range(2)]
        ).min(0)
        out[name] = dist < ev.HIT_RADIUS
    return out


def spread(values: list[float]) -> dict:
    v = np.array(values, dtype=float)
    return dict(
        mean=round(float(v.mean()), 4),
        sd=round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
        low=round(float(v.min()), 4),
        high=round(float(v.max()), 4),
        n=len(v),
    )


# ------------------------------------------------------------------------- figure


def figure(out: Path, steps: dict[tuple[str, int], Path], found: dict) -> None:
    sys.path.insert(0, str(ROOT / "paper"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import apply_publication_style, finalize_figure, panel_letter

    apply_publication_style()
    colour = {"reference": "#2A78D6", "scratch": "#EB6834", "no_weights": "#1BAF7A"}
    names = {
        "reference": "from the denoiser",
        "scratch": "random start",
        "no_weights": "no class weights",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.86, wspace=0.32)
    order = list(colour)  # draw and list the variants in this order
    for (name, seed), path in sorted(
        steps.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in colour else 9, kv[0][1])
    ):
        if name not in colour:
            continue
        s = read_steps(path)
        w = 50
        k = len(s["step"]) // w * w
        x = s["step"][:k].reshape(-1, w).mean(1)
        label = names[name] if seed == 1 else None
        axes[0].plot(
            x, np.median(s["grad"][:k].reshape(-1, w), 1), color=colour[name], lw=0.9, label=label
        )
        axes[0].plot(x, s["grad"][:k].reshape(-1, w).max(1), color=colour[name], lw=0.5, alpha=0.35)
        if name != "no_weights":  # its loss has no class weights: not on the same scale
            axes[1].plot(x, np.median(s["loss"][:k].reshape(-1, w), 1), color=colour[name], lw=0.9)
    axes[0].axhline(CLIP, color="#767676", lw=0.8, ls="--")
    axes[0].text(20, CLIP * 0.8, "clip", ha="left", va="top", fontsize=6, color="#767676")
    axes[0].set_yscale("log")
    axes[0].set_title("Gradient norm before clipping", fontsize=7.5, loc="left", pad=11)
    axes[0].text(0.0, 1.015, "median of 50 updates; faint: their largest", transform=axes[0].transAxes,
                 fontsize=5.5, color="#5A5A5A", va="bottom")  # fmt: skip
    axes[1].set_title("Training loss (class-weighted)", fontsize=7.5, loc="left", pad=11)
    for ax in axes[:2]:
        ax.set_xlabel("update", fontsize=6.5)
        ax.grid(axis="y", color="#E1E0D9", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6, frameon=False, loc="upper right")
    # found rate per seed and variant
    ax = axes[2]
    names = [n for n in VARIANTS if n in found]
    for i, name in enumerate(names):
        rates = [found[name][s] for s in sorted(found[name])]
        ax.scatter(
            [i] * len(rates),
            [100 * r for r in rates],
            s=12,
            color="#2A78D6" if name == "reference" else "#5A5A5A",
            zorder=3,
        )
        ax.plot([i - 0.25, i + 0.25], [100 * np.mean(rates)] * 2, color="#272727", lw=1.0)
    ax.set_xticks(
        range(len(names)), [n.replace("_", " ") for n in names], rotation=35, ha="right", fontsize=6
    )
    ax.set_ylabel("% of test defects found", fontsize=6.5)
    ax.set_title("Found, per seed (line: mean)", fontsize=7.5, loc="left", pad=11)
    ax.grid(axis="y", color="#E1E0D9", lw=0.5)
    ax.set_axisbelow(True)
    for a, letter in zip(axes, "abc"):
        panel_letter(fig, (a.get_position().x0 - 0.045, a.get_position().y1 + 0.1), letter)
    finalize_figure(fig, out / "stability")


# ------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results/point_defects_seeds_20261114")
    ap.add_argument("--data", default=str(ROOT / "data/ka2d_256_point_defects/test.npz"))
    ap.add_argument("--max-examples", type=int, default=None, help="per defect kind (quick checks)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    data = ev.real_defects(ev.load_point_defects(args.data))
    if args.max_examples:
        data = {k: d.subset(slice(0, args.max_examples)) for k, d in data.items()}
    glass = {k: d.glass.numpy() for k, d in data.items()}
    runs = finished_runs(args.out)
    # the logged rerun of seed 0 replaces the published run where it exists
    out: dict = dict(
        n={k: len(d) for k, d in data.items()},
        settings=dict(
            clip=CLIP,
            loss_spike=f"> {LOSS_SPIKE} x median of the previous {LOSS_WINDOW} updates of the same batch type",
            grad_spike=f"> {GRAD_SPIKE} x median of the previous {GRAD_WINDOW} updates",
            warm_up=WARM,
        ),
    )
    out["seed0_rerun_identical"] = {
        name: same_weights(runs[(name, 0)]["weights"], ABLATION_DIR / f"{name}.pt")
        for name in VARIANTS
        if runs.get((name, 0), {}).get("source") == "seed study"
    }

    # ---- stability
    out["stability"] = {
        f"{name} seed {seed}": stability(r["steps"])
        for (name, seed), r in sorted(runs.items())
        if r["steps"]
    }
    # ---- results
    score, hit = {}, {}
    for key, r in sorted(runs.items()):
        score[key] = scores(r["weights"], data)
        hit[key] = hits(score[key], data)
        print(key, {k: round(float(v.mean()), 3) for k, v in hit[key].items()}, flush=True)
    out["runs"] = {
        f"{name} seed {seed}": dict(
            source=runs[(name, seed)]["source"], found=fa.rates(hit[(name, seed)], glass, rng)
        )
        for (name, seed) in sorted(hit)
    }
    kinds = [*data, "all"]
    out["variants"], found_all = {}, {}
    for name, what in VARIANTS.items():
        seeds = sorted(s for (n, s) in hit if n == name)
        if not seeds:
            continue
        per = {s: fa.rates(hit[(name, s)], glass, rng) for s in seeds}
        found_all[name] = {s: per[s]["all"]["rate"] for s in seeds}
        v = dict(
            what=what,
            seeds=seeds,
            found={k: spread([per[s][k]["rate"] for s in seeds]) for k in kinds},
        )
        if name != "reference":
            both = [s for s in seeds if ("reference", s) in hit]
            diffs = [
                float(
                    np.concatenate(list(hit[(name, s)].values())).mean()
                    - np.concatenate(list(hit[("reference", s)].values())).mean()
                )
                for s in both
            ]
            if diffs:
                v["difference_from_reference_same_seed"] = dict(
                    **spread(diffs),
                    lower_in=sum(d < 0 for d in diffs),
                    seeds=both,
                    p_per_seed=[
                        round(fa.paired(hit[("reference", s)], hit[(name, s)])["all"]["p"], 4)
                        for s in both
                    ],
                )
        out["variants"][name] = v
    ref = [s for s in SEEDS if ("reference", s) in hit]
    if len(ref) > 1:
        each = np.stack([np.concatenate([hit[("reference", s)][k] for k in data]) for s in ref])
        n_found = each.sum(0)
        ensemble = hits(
            {k: np.mean([score[("reference", s)][k] for s in ref], 0) for k in data}, data
        )
        out["reference_agreement"] = dict(
            seeds=ref,
            found_by_all=round(float((n_found == len(ref)).mean()), 4),
            found_by_some=round(float(((n_found > 0) & (n_found < len(ref))).mean()), 4),
            found_by_none=round(float((n_found == 0).mean()), 4),
            ensemble=fa.rates(ensemble, glass, rng),
        )
    (args.out / "seeds.json").write_text(json.dumps(out, indent=1) + "\n")
    figure(args.out, {k: r["steps"] for k, r in runs.items() if r["steps"]}, found_all)
    md = summary(out, kinds)
    (args.out / "summary.md").write_text(md)
    print(md)


def summary(out: dict, kinds: list[str]) -> str:
    label = dict(
        vacancy="Vacancy",
        interstitial="Interstitial",
        frenkel="Frenkel pair",
        antisite="Antisite",
        all="All",
    )

    def pct(x):
        return f"{100 * x:.1f}"

    md = [
        "<!-- written by scripts/pilots/finder_seeds.py -->",
        "",
        "## Training stability",
        "",
        "| Run | Updates | Grad. norm: median (first 100 / last 1,000) | 99th pct | Largest (update) | Clipped: all / last 1,000 | Grad. spikes | Loss spikes | Not finite |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run, s in out["stability"].items():
        g = s["grad_norm"]
        md.append(
            f"| {run} | {s['updates']} | {g['median']} ({g['median_first_100']} / {g['median_last_1000']}) | {g['p99']} | "
            f"{g['largest']} ({g['largest_at']}, {g['largest_batch']}) | {pct(s['clipped'])}% / {pct(s['clipped_last_1000'])}% | "
            f"{len(s['grad_spikes'])} | {len(s['loss_spikes'])} | {s['not_finite']} |"
        )
    md += [
        "",
        "## Found, per variant (mean over seeds, with the lowest and highest seed)",
        "",
        "| Variant | Seeds | "
        + " | ".join(label[k] for k in kinds)
        + " | Spread of All (sd) | Difference from the reference, same seed |",
        "|---|---|" + "---|" * (len(kinds) + 2),
    ]
    for name, v in out["variants"].items():
        cells = [
            f"{pct(v['found'][k]['mean'])} [{pct(v['found'][k]['low'])}, {pct(v['found'][k]['high'])}]"
            for k in kinds
        ]
        sd = v["found"]["all"]["sd"]
        d = v.get("difference_from_reference_same_seed")
        diff = f"{100 * d['mean']:+.1f} (lower in {d['lower_in']} of {d['n']} seeds)" if d else ""
        md.append(
            f"| {name}: {v['what']} | {', '.join(map(str, v['seeds']))} | "
            + " | ".join(cells)
            + f" | {'' if sd is None else pct(sd)} | {diff} |"
        )
    md += [
        "",
        "## Every run",
        "",
        "| Run | Source | " + " | ".join(label[k] for k in kinds) + " |",
        "|---|---|" + "---|" * len(kinds),
    ]
    for run, r in out["runs"].items():
        md.append(
            f"| {run} | {r['source']} | "
            + " | ".join(
                f"{pct(r['found'][k]['rate'])} [{100 * r['found'][k]['lo']:.0f}, {100 * r['found'][k]['hi']:.0f}]"
                for k in kinds
            )
            + " |"
        )
    if "reference_agreement" in out:
        a = out["reference_agreement"]
        md += [
            "",
            f"Reference, seeds {a['seeds']}: found by all {pct(a['found_by_all'])}%, by some {pct(a['found_by_some'])}%, "
            f"by none {pct(a['found_by_none'])}% of the defects. The seeds averaged (ensemble) find "
            f"{pct(a['ensemble']['all']['rate'])}% [{100 * a['ensemble']['all']['lo']:.0f}, {100 * a['ensemble']['all']['hi']:.0f}].",
        ]
    if out["seed0_rerun_identical"]:
        md += [
            "",
            "Seed-0 reruns with the same weights as the published run: "
            + ", ".join(
                f"{k} {'yes' if v else 'no'}" for k, v in out["seed0_rerun_identical"].items()
            )
            + ".",
        ]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
