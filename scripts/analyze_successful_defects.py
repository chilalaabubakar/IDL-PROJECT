#!/usr/bin/env python3
"""Analyse retained 2D glass runs without modifying their files.

Run from the IDL-PROJECT checkout after pip install -e '.[dev]'.
python scripts/analyze_successful_defects.py --runs 'results/*/experiments/rr_*' \
    --data data/ka2d_256 --out runs/gap1 --bootstrap 1000

No MD is run. CIs resample requests/windows with a fixed natural reference;
training-seed and repeated-host dependence are not included.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glassdiff-matplotlib")
import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds
from glassdiff.data.dataset import load_split
from glassdiff.eval.metrics import QUANTILE_LEVELS, local_features, success
from glassdiff.geometry import minimum_image
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import DefectClass, Request, Structures


def ci(items, stat, nboot, seed):
    if not len(items):
        return [None, None, None]
    point = float(stat(items))
    if not np.isfinite(point):
        return [None, None, None]
    if len(items) < 2:
        return [point, None, None]
    rng = np.random.default_rng(seed)
    vals = [
        stat([items[i] for i in rng.integers(len(items), size=len(items))]) for _ in range(nboot)
    ]
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) < nboot * 0.95:
        return [point, None, None]
    return [point, *map(float, np.quantile(vals, [0.025, 0.975]))]


def pooled(items):
    return np.concatenate(items) if len(items) else np.array([])


def w1(items, reference, nboot, seed):
    """Exact request bootstrap with sorted weighted order statistics.

    Each atomic value has its request's bootstrap multiplicity. Searching cumulative
    multiplicities reproduces NumPy's linear quantiles of the expanded bootstrap
    sample without concatenating and sorting that sample on every iteration.
    """
    ref = pooled(reference)
    values = pooled(items)
    if not len(ref) or not len(values):
        return [None, None, None]
    q = np.quantile(ref, QUANTILE_LEVELS)
    order = np.argsort(values)
    values = values[order]
    owners = np.repeat(np.arange(len(items)), [len(x) for x in items])[order]

    def statistic(counts):
        cumulative = np.cumsum(counts[owners])
        if cumulative[-1] == 0:
            return float("nan")
        ranks = QUANTILE_LEVELS * (cumulative[-1] - 1)
        lower = np.floor(ranks).astype(np.int64)
        upper = np.ceil(ranks).astype(np.int64)
        a = values[np.searchsorted(cumulative, lower, side="right")]
        b = values[np.searchsorted(cumulative, upper, side="right")]
        quantiles = a + (b - a) * (ranks - lower)
        return float(np.abs(quantiles - q).mean())

    point = statistic(np.ones(len(items), dtype=np.int64))
    if len(items) < 2:
        return [point, None, None]
    rng = np.random.default_rng(seed)
    vals = np.asarray(
        [
            statistic(np.bincount(rng.integers(len(items), size=len(items)), minlength=len(items)))
            for _ in range(nboot)
        ]
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) < nboot * 0.95:
        return [point, None, None]
    return [point, *map(float, np.quantile(vals, [0.025, 0.975]))]


def features(s, centres, radius):
    out = {k: [] for k in ["energy", "angle", "pair_AA", "pair_AB", "pair_BB"]}
    for start in range(0, s.batch_size, 32):
        ix = slice(start, start + 32)
        sub = Structures(s.pos[ix], s.types[ix], s.box[ix])
        f = local_features(sub, centres[ix], r_loc=radius)
        out["energy"].extend(t.numpy() for t in f.pe_atom)
        out["angle"].extend(t.numpy() for t in f.bond_angle)
        for k, values in f.pair_dist.items():
            out["pair_" + k].extend(t.numpy() for t in values)
    return out


def radial(s, centres):
    radii, energies = [], []
    for start in range(0, s.batch_size, 32):
        ix = slice(start, start + 32)
        sub = Structures(s.pos[ix], s.types[ix], s.box[ix])
        radii.extend(minimum_image(sub.pos - centres[ix, None], sub.box).norm(dim=-1).numpy())
        energies.extend(ka_energy(sub, per_atom=True).numpy())
    return radii, energies


def natural(data, split, defect, cap, radius):
    d = load_split(data / f"{split}.npz")
    if d.labels is None:
        raise ValueError(f"{split}: reference labels missing")
    sites = torch.nonzero(d.labels == defect)[:cap]
    if not len(sites):
        raise ValueError(f"{split}: no natural {DefectClass(defect).name} defects")
    g, a = sites.T
    s = Structures(d.structures.pos[g], d.structures.types[g], d.structures.box[g])
    c = s.pos[torch.arange(len(g)), a]
    return features(s, c, radius), radial(s, c), len(torch.unique(g))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Run directories or quoted globs")
    p.add_argument("--data", type=Path, default=Path("data/ka2d_256"))
    p.add_argument("--out", type=Path, default=Path("runs/gap1"))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--r-max", type=float, default=7.0)
    p.add_argument("--reference-cap", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.bootstrap < 20 or min(args.radius, args.r_max, args.reference_cap, args.threads) <= 0:
        p.error("Use bootstrap >=20 and positive radius, r-max, reference-cap, threads")
    torch.set_num_threads(args.threads)
    paths = sorted({Path(x).resolve() for pat in args.runs for x in glob.glob(pat)})
    if not paths:
        p.error("No runs matched")
    args.out.mkdir(parents=True, exist_ok=True)
    th = DefectThresholds.from_dict(
        json.loads((args.data / "meta.json").read_text())["defects"]["thresholds"]
    )
    rows, shells, masks, skipped, cache = [], [], [], [], {}
    floor_metrics, floor_shells = {}, {}
    with torch.no_grad():
        for path in paths:
            if not all((path / f).exists() for f in ["samples.npz", "relaxed.npz"]):
                skipped.append(str(path))
                print("SKIP missing retained samples:", path, flush=True)
                continue
            with np.load(path / "samples.npz") as f:
                samp = {k: f[k] for k in f.files}
            with np.load(path / "relaxed.npz") as f:
                rel = {k: f[k] for k in f.files}
            defect = int(samp["defect"])
            split = str(samp.get("split", "test"))
            n = len(rel["pos"])
            s = Structures(
                torch.from_numpy(rel["pos"]).double(),
                torch.from_numpy(samp["types"]).long(),
                torch.from_numpy(samp["box"]).double(),
            )
            centres = torch.from_numpy(samp["target"]).double()
            req = Request(
                target=centres,
                defect=torch.full((n,), defect, dtype=torch.long),
                pin_mask=torch.from_numpy(samp["pin_mask"]).bool(),
                pin_pos=torch.from_numpy(samp["pin_pos"]).double(),
                label=torch.from_numpy(samp["label"]).long(),
            )
            ok = success(s, req, th).numpy()
            pre = Structures(torch.from_numpy(samp["pos"]).double(), s.types, s.box)
            before = success(pre, req, th).numpy()
            converged = np.asarray(rel["converged"]).astype(bool)
            for i in range(n):
                masks.append(
                    dict(
                        run=str(path),
                        sample=i,
                        success_before=int(before[i]),
                        success_after=int(ok[i]),
                        converged=int(converged[i]),
                    )
                )
            key = (split, defect)
            if key not in cache:
                cache[key] = (
                    natural(args.data, split, defect, args.reference_cap, args.radius),
                    natural(args.data, "train", defect, args.reference_cap, args.radius),
                )
            (ref, ref_rad, nhost), (floor, floor_rad, _) = cache[key]
            gen, gen_rad = features(s, centres, args.radius), radial(s, centres)
            # This measures movement of saved pinned atoms, not full original-host displacement.
            pd = minimum_image(s.pos - req.pin_pos, s.box).norm(dim=-1)
            pinned = [
                float(pd[i][req.pin_mask[i]].mean()) if req.pin_mask[i].any() else None
                for i in range(n)
            ]
            rate = ci(list(ok.astype(float)), np.mean, args.bootstrap, args.seed)
            print(f"{path.name}: {ok.sum()}/{n} successes; computing realism", flush=True)
            groups = [
                ("all", np.arange(n)),
                ("successful", np.flatnonzero(ok)),
                ("failed", np.flatnonzero(~ok)),
                ("natural_train_floor", None),
            ]
            for group, ix in groups:
                count = len(floor["energy"]) if ix is None else len(ix)
                row = dict(
                    run=str(path),
                    defect=DefectClass(defect).name,
                    split=split,
                    group=group,
                    n_requests=n,
                    n_success=int(ok.sum()),
                    n_group=count,
                    success=rate[0],
                    success_lo=rate[1],
                    success_hi=rate[2],
                    relax_converged=float(converged.mean()),
                    reference_windows=len(ref["energy"]),
                    reference_host_glasses=nhost,
                    low_count=count < 30,
                    pinned_displacement_mean=None
                    if ix is None
                    else (
                        float(np.mean([pinned[i] for i in ix if pinned[i] is not None]))
                        if any(pinned[i] is not None for i in ix)
                        else None
                    ),
                )
                f = floor if ix is None else {k: [v[i] for i in ix] for k, v in gen.items()}
                for k in ref:
                    floor_key = (key, k)
                    if ix is None and floor_key in floor_metrics:
                        v = floor_metrics[floor_key]
                    else:
                        v = w1(f[k], ref[k], args.bootstrap, args.seed)
                        if ix is None:
                            floor_metrics[floor_key] = v
                    row.update({f"w1_{k}{suffix}": z for suffix, z in zip(["", "_lo", "_hi"], v)})
                rows.append(row)
                rad, energy = (
                    floor_rad
                    if ix is None
                    else ([gen_rad[0][i] for i in ix], [gen_rad[1][i] for i in ix])
                )
                # Distances beyond half the shortest box side have periodic geometry limitations.
                for lo in np.arange(0.0, args.r_max, 1.0):
                    hi = min(lo + 1.0, args.r_max)
                    items = [e[(r >= lo) & (r < hi)] for r, e in zip(rad, energy)]
                    reference = [e[(r >= lo) & (r < hi)] for r, e in zip(*ref_rad)]
                    floor_key = (key, float(lo), float(hi))
                    if ix is None and floor_key in floor_shells:
                        v = floor_shells[floor_key]
                    else:
                        v = w1(items, reference, args.bootstrap, args.seed)
                        if ix is None:
                            floor_shells[floor_key] = v
                    shells.append(
                        dict(
                            run=str(path),
                            defect=DefectClass(defect).name,
                            group=group,
                            r_lo=float(lo),
                            r_hi=float(hi),
                            n_group=count,
                            w1=v[0],
                            lo=v[1],
                            hi=v[2],
                        )
                    )
    if not rows:
        raise SystemExit("No analysable runs: samples.npz and relaxed.npz are required.")
    write_csv(args.out / "summary.csv", rows)
    write_csv(args.out / "shells.csv", shells)
    write_csv(args.out / "sample_outcomes.csv", masks)
    (args.out / "metadata.json").write_text(
        json.dumps(
            dict(
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                skipped=skipped,
                thresholds=th.to_dict() if hasattr(th, "to_dict") else str(th),
                uncertainty="Request/window bootstrap, fixed natural reference; not host-clustered or across training seeds.",
                notes=[
                    "Groups include converged and nonconverged samples; convergence is reported separately.",
                    "Natural floor uses train defects versus the run reference split.",
                    "Reference sites use repository ordering and the requested cap.",
                    "Pinned displacement uses saved pin_pos, not all original host positions.",
                    "No finite-temperature stability test is performed.",
                    "Radial shells beyond half the shortest box side require periodic-geometry interpretation.",
                ],
            ),
            indent=2,
        )
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        if row["group"] != "successful" or row["w1_energy"] is None:
            continue
        x, y = row["w1_energy"], row["success"]
        xerr = (
            None
            if row["w1_energy_lo"] is None
            else [
                [max(0.0, x - row["w1_energy_lo"])],
                [max(0.0, row["w1_energy_hi"] - x)],
            ]
        )
        yerr = (
            None
            if row["success_lo"] is None
            else [[max(0.0, y - row["success_lo"])], [max(0.0, row["success_hi"] - y)]]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label=f"{Path(row['run']).name} (n={row['n_group']})",
        )
    ax.set(
        xlabel="Successful-only local energy W1 (lower is better)",
        ylabel="All-request relaxed success",
    )
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(args.out / "success_vs_realism.png", dpi=160)
    plt.close(fig)
    for j, path in enumerate(sorted({r["run"] for r in rows})):
        fig, ax = plt.subplots(figsize=(7, 4))
        for group in ["all", "successful", "failed", "natural_train_floor"]:
            rs = [
                r
                for r in shells
                if r["run"] == path and r["group"] == group and r["w1"] is not None
            ]
            if rs:
                ax.plot(
                    [(r["r_lo"] + r["r_hi"]) / 2 for r in rs],
                    [r["w1"] for r in rs],
                    ".-",
                    label=group,
                )
                if all(r["lo"] is not None for r in rs):
                    ax.fill_between(
                        [(r["r_lo"] + r["r_hi"]) / 2 for r in rs],
                        [r["lo"] for r in rs],
                        [r["hi"] for r in rs],
                        alpha=0.12,
                    )
        ax.set(
            title=Path(path).name,
            xlabel="Distance from requested site",
            ylabel="Shell energy W1",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / f"radial_{j:03d}.png", dpi=160)
        plt.close(fig)
    print(
        f"Done: {args.out.resolve()} ({len(rows) // 4} runs; {len(skipped)} skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main()
