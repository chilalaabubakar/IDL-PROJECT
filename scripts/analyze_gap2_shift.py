"""Gap 2: how much does placement change when the host glass changes? (interim or final)

    python scripts/analyze_gap2_shift.py \\
        --study results/kaggle_gap2_kaggle_remaining_20261101-220126Z/study [more study dirs] \\
        --out results/gap2_shift_<date>

The Gap 2 reporter (scripts/run_gap2.py) gives placement success inside each condition and
paired contrasts against hand insertion and the sham. This script adds the two comparisons
across conditions that docs/GAP2_GENERALIZATION.md asks for:

1. Host characterisation (hosts.csv). Does each condition really produce a different glass?
   Per host: mean energy per atom of the inherent structure, global hexatic order, closest
   pair, and the number of natural D1-, D1+ and D2 sites per 256 atoms, using the frozen
   training thresholds (data/ka2d_256/meta.json). Reported per condition as mean and 95%
   bootstrap interval over hosts.
2. Transfer shift (shift.csv). For every method (seed0-2, their mean, sham, hand insertion)
   and for each seed's advantage over hand insertion, the change in request-weighted success
   from the reference condition (fresh 256-atom standard cooling) to every other completed
   condition. Conditions have independent hosts, so each side is resampled over its own hosts
   (1,000 draws, seed 93 as in the reporter); within a condition, methods share hosts and
   requests, so paired quantities are resampled together.

Only conditions whose five methods all have complete.json are used. The script reads the
study folders and writes only to --out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.analysis.descriptors import psi6_global
from glassdiff.types import DefectClass, Structures

METHODS = ["seed0", "seed1", "seed2", "sham_seed0", "hand_insert"]
SEEDS = ["seed0", "seed1", "seed2"]
REFERENCE = "ka2d_256_standard"
N_BOOT, BOOT_SEED = 1000, 93


def complete_conditions(studies: list[Path]) -> dict[str, Path]:
    """condition -> folder, newest study wins when a condition appears twice."""
    found: dict[str, Path] = {}
    for study in studies:
        for folder in sorted(p for p in study.iterdir() if p.is_dir()):
            if all((folder / m / "complete.json").exists() for m in METHODS):
                found[folder.name] = folder
    return found


def load(folder: Path) -> dict:
    with np.load(folder / "requests.npz") as f:
        ids = f["host_ids"].copy()
        initially = f["original_success"].astype(bool)
    success = {}
    for m in METHODS:
        with np.load(folder / m / "samples.npz") as f:
            if not np.array_equal(f["host_ids"], ids):
                raise ValueError(f"{folder}/{m}: requests do not match")
            success[m] = f["success"].astype(float)
    success["seed_mean"] = np.mean([success[s] for s in SEEDS], axis=0)
    for s in SEEDS:
        success[f"{s}_minus_hand"] = success[s] - success["hand_insert"]
    success["seed_mean_minus_hand"] = success["seed_mean"] - success["hand_insert"]
    return {"ids": ids, "eligible": ~initially, "success": success}


def host_groups(values: np.ndarray, ids: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """[n_hosts, 2] (sum of values, number of eligible requests) per host."""
    rows = [
        (values[(ids == h) & eligible].sum(), ((ids == h) & eligible).sum())
        for h in np.unique(ids)
        if ((ids == h) & eligible).any()
    ]
    return np.asarray(rows, dtype=float)


def ratio_draws(groups: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    point = groups[:, 0].sum() / groups[:, 1].sum()
    idx = rng.integers(len(groups), size=(N_BOOT, len(groups)))
    return point, groups[idx, 0].sum(1) / groups[idx, 1].sum(1)


def shift_table(data: dict[str, dict], definition: str) -> list[dict]:
    rows = []
    ref = data[REFERENCE]
    for condition, d in sorted(data.items()):
        if condition == REFERENCE:
            continue
        for method in d["success"]:
            masks = []
            for side in (ref, d):
                eligible = (
                    side["eligible"]
                    if definition == "initially_nonD2"
                    else np.ones_like(side["eligible"])
                )
                masks.append(host_groups(side["success"][method], side["ids"], eligible))
            rng = np.random.default_rng(BOOT_SEED)
            p_ref, d_ref = ratio_draws(masks[0], rng)
            p_new, d_new = ratio_draws(masks[1], rng)
            diff = d_new - d_ref
            lo, hi = np.quantile(diff, [0.025, 0.975])
            rows.append(
                dict(
                    condition=condition,
                    reference=REFERENCE,
                    definition=definition,
                    quantity=method,
                    reference_value=round(p_ref, 4),
                    condition_value=round(p_new, 4),
                    shift=round(p_new - p_ref, 4),
                    lo=round(float(lo), 4),
                    hi=round(float(hi), 4),
                    n_hosts_reference=len(masks[0]),
                    n_hosts_condition=len(masks[1]),
                )
            )
    return rows


def host_stats(folder: Path, th: DefectThresholds) -> dict[str, np.ndarray]:
    with np.load(folder / "hosts" / "samples.npz") as f:
        pos, types, box, pe = f["pos"], f["types"].astype(np.int64), f["box"], f["pe_atom"]
    out = {k: [] for k in ("pe_atom", "psi6", "D1_MINUS", "D1_PLUS", "D2")}
    for i in range(0, len(pos), 10):
        s = Structures(
            torch.from_numpy(pos[i : i + 10]),
            torch.from_numpy(types[i : i + 10]),
            torch.from_numpy(box[i : i + 10]),
        )
        labels = detect_defects(s, th)
        per256 = 256.0 / s.n_atoms
        for c in (DefectClass.D1_MINUS, DefectClass.D1_PLUS, DefectClass.D2):
            out[c.name] += ((labels == c).sum(-1).double() * per256).tolist()
        out["psi6"] += psi6_global(s).tolist()
    out["pe_atom"] = pe.mean(-1).tolist()
    return {k: np.asarray(v) for k, v in out.items()}


def mean_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = x[rng.integers(len(x), size=(N_BOOT, len(x)))].mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--study", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, default=Path("data/ka2d_256/meta.json"))
    args = ap.parse_args()

    folders = complete_conditions(args.study)
    if REFERENCE not in folders:
        raise SystemExit(f"the reference condition {REFERENCE} is not complete in {args.study}")
    args.out.mkdir(parents=True, exist_ok=True)
    th = DefectThresholds.from_dict(
        json.loads(args.thresholds.read_text())["defects"]["thresholds"]
    )

    rng = np.random.default_rng(BOOT_SEED)
    host_rows = []
    for condition, folder in sorted(folders.items()):
        stats = host_stats(folder, th)
        row = dict(condition=condition, n_hosts=len(stats["pe_atom"]))
        for key, x in stats.items():
            m, lo, hi = mean_ci(x, rng)
            row.update({key: round(m, 4), f"{key}_lo": round(lo, 4), f"{key}_hi": round(hi, 4)})
        host_rows.append(row)
        print(f"hosts {condition}: " + ", ".join(f"{k} {row[k]}" for k in stats), flush=True)
    write_csv(args.out / "hosts.csv", host_rows)

    data = {c: load(f) for c, f in folders.items()}
    shift_rows = shift_table(data, "initially_nonD2") + shift_table(data, "all_requests")
    write_csv(args.out / "shift.csv", shift_rows)
    for r in shift_rows:
        if r["definition"] == "initially_nonD2":
            print(
                f"{r['condition']:20s} {r['quantity']:22s} {r['reference_value']:.3f} -> "
                f"{r['condition_value']:.3f}  shift {r['shift']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
            )
    meta = dict(
        conditions={c: str(f) for c, f in folders.items()},
        reference=REFERENCE,
        bootstrap=f"{N_BOOT} draws, seed {BOOT_SEED}; hosts resampled independently per condition",
        thresholds=str(args.thresholds),
        note="interim unless all six conditions are listed",
    )
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
