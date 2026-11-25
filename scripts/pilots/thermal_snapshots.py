"""Thermal snapshots of the defected validation glasses (docs/NPJ_EXTENSIONS.md, item 3).

    python scripts/pilots/thermal_snapshots.py --temperatures 0.02 0.05 0.1 0.2
        -> data/ka2d_256_point_defects_thermal/val.npz and meta.json (~3 min on 4 cores)

Every real defect of the split (Frenkel pairs that undid themselves left out) starts from its
relaxed defected glass, gets velocities at temperature T and runs at constant temperature
(Nose-Hoover, time step 0.005, 2,000 steps = 10 time units; glassdiff.md.thermal). The
positions at the end are the snapshot; neighbouring atoms move together, unlike the
independent shifts of scripts/pilots/displaced_inputs.py. The snapshot is then minimised.

A defect counts as surviving when the minimised snapshot has the same defect at the same
place as the relaxed defected glass: the same empty, doubly occupied and wrong-species sites
of the original glass (glassdiff.data.point_defects.site_occupancy, as the manuscript's
analysis). Atoms elsewhere may have moved to a nearby energy minimum; that is recorded too
(same_minimum: the minimised snapshot is the relaxed defected glass again, within 0.05 sigma).
Only surviving snapshots are scored, so the labels stay exact.

Stored per temperature T and defect kind k: k_T<T>_example (index into the split),
k_T<T>_snapshot and k_T<T>_minimised (positions [E, N, 2], float32), k_T<T>_survived and
k_T<T>_same_minimum (bool), k_T<T>_rms (per-coordinate RMS displacement of the snapshot from
the relaxed defected glass, sigma) and k_T<T>_temperature (at the end of the run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from glassdiff.data.point_defects import (
    already_original,
    load_point_defects,
    same_structure,
    site_occupancy,
    wrap,
)
from glassdiff.md.thermal import thermal_snapshots_many

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("vacancy", "interstitial", "frenkel", "antisite")


def occupancy_key(pos, types, orig, otypes, box) -> tuple:
    o = site_occupancy(np.mod(pos, box), types, orig, otypes, box)
    return o["class"], tuple(sorted(o["empty"])), tuple(sorted(o["double"])), tuple(sorted(o["wrong"]))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data/ka2d_256_point_defects/val.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ka2d_256_point_defects_thermal")
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    data = load_point_defects(args.data)
    store: dict[str, np.ndarray] = {}
    summary: dict = {}
    for ti, temp in enumerate(args.temperatures):
        summary[f"{temp:g}"] = {}
        for ki, kind in enumerate(KINDS):
            d = data[kind]
            example = np.nonzero(~already_original(d).numpy())[0]  # real defects only
            pos, types = d.pos.double().numpy(), d.types.numpy()
            box, orig, otypes = d.box.double().numpy(), d.orig_pos.double().numpy(), d.orig_types.numpy()  # fmt: skip
            jobs = [
                (pos[e], types[e], box[e], temp, args.steps,
                 args.seed + 1_000_000 * ti + 10_000 * ki + int(e))
                for e in example
            ]  # fmt: skip
            out = thermal_snapshots_many(jobs, workers=args.workers)
            snap = np.stack([o[0] for o in out])
            mini = np.stack([o[1] for o in out])
            survived, same_min, rms = [], [], []
            for j, e in enumerate(example):
                b = box[e]
                survived.append(occupancy_key(mini[j], types[e], orig[e], otypes[e], b)
                                == occupancy_key(pos[e], types[e], orig[e], otypes[e], b))  # fmt: skip
                same_min.append(same_structure(np.mod(mini[j], b), types[e], np.mod(pos[e], b), types[e], b))  # fmt: skip
                dv = wrap(snap[j] - pos[e], b)
                dv -= dv.mean(0)
                rms.append(float(np.sqrt((dv**2).mean())))
            key = f"{kind}_T{temp:g}"
            store[f"{key}_example"] = example
            store[f"{key}_snapshot"] = snap.astype(np.float32)
            store[f"{key}_minimised"] = mini.astype(np.float32)
            store[f"{key}_survived"] = np.array(survived)
            store[f"{key}_same_minimum"] = np.array(same_min)
            store[f"{key}_rms"] = np.array(rms)
            store[f"{key}_temperature"] = np.array([o[2] for o in out])
            summary[f"{temp:g}"][kind] = dict(
                defects=len(example), survived=int(np.sum(survived)),
                same_minimum=int(np.sum(same_min)),
                rms_median=round(float(np.median(rms)), 4),
                temperature_mean=round(float(np.mean(store[f"{key}_temperature"])), 4),
            )  # fmt: skip
            print(f"T={temp:g} {kind}: {json.dumps(summary[f'{temp:g}'][kind])} ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.data.stem}.npz"
    np.savez_compressed(path, **store)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # fmt: skip
    meta = dict(
        source=str(args.data.resolve().relative_to(ROOT)),
        source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        file=path.name, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        temperatures=args.temperatures, steps=args.steps, timestep=0.005, thermostat="Nose-Hoover, tdamp 0.1",
        seed=args.seed, git=git, summary=summary, seconds=round(time.time() - t0),
    )  # fmt: skip
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("->", path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
