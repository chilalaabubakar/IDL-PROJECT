"""Host-aware structural checks and finite-temperature retention of retained D2 samples.

python scripts/check_physical_credibility.py structural --out results/credibility_followup_20261018
python scripts/check_physical_credibility.py md --out results/credibility_followup_20261018 --workers 2

Inputs stay unchanged. Structural bootstrap resamples host glasses in both ensembles.
MD uses freely evolving whole boxes, dataset-consistent KA LAMMPS interactions,
T=0.01 and 0.1, dt=0.005, NVT damping 0.1, three velocity seeds, up to 50 LJ time units.
Snapshot minimization uses a separate LAMMPS instance and does not alter the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glassdiff-matplotlib")
import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.dataset import load_split
from glassdiff.eval.metrics import QUANTILE_LEVELS, nearest_atom, success
from glassdiff.geometry import minimum_image
from glassdiff.md.lammps_quench import _PAIR_COMMANDS
from glassdiff.physics.fire import fire_minimize
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import Request, Structures

_spec = importlib.util.spec_from_file_location(
    "successful_analysis", Path(__file__).with_name("analyze_successful_defects.py")
)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)
RUNS = {
    "fixed_r4": "results/kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout4_host_D2",
    "fixed_r5": "results/kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout5_host_D2",
    "fixed_r6": "results/kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout6_host_D2",
    "hand_insert": "results/kaggle_20261017-1739Z/experiments/seam_hand_insert_D2",
    "range_r5": "results/kaggle_20261017-2217Z/experiments/rr_cond_smax1_w2_rout5_host_D2",
}
DATA = Path("data/ka2d_256")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def load_run(path):
    with np.load(Path(path) / "samples.npz") as f:
        samp = {k: f[k] for k in f.files}
    with np.load(Path(path) / "relaxed.npz") as f:
        rel = {k: f[k] for k in f.files}
    s = Structures(
        torch.from_numpy(rel["pos"]).double(),
        torch.from_numpy(samp["types"]).long(),
        torch.from_numpy(samp["box"]).double(),
    )
    req = Request(
        torch.from_numpy(samp["target"]).double(),
        torch.full((s.batch_size,), 3, dtype=torch.long),
        torch.from_numpy(samp["pin_mask"]).bool(),
        torch.from_numpy(samp["pin_pos"]).double(),
        torch.from_numpy(samp["label"]).long(),
    )
    return s, req, samp


def recover_hosts(samp, hosts):
    """Validate the sequential-host convention in the original saved sampler code."""
    ids = np.arange(len(samp["pos"])) % hosts.batch_size
    h = hosts.pos[ids]
    if not torch.equal(torch.from_numpy(samp["types"]).long(), hosts.types[ids]):
        raise ValueError("Host type sequence mismatch")
    if not np.allclose(samp["box"], hosts.box[ids].numpy(), atol=2e-6, rtol=0):
        raise ValueError("Host box sequence mismatch")
    if str(samp["mode"]) == "host":
        err = (
            minimum_image(torch.from_numpy(samp["pin_pos"]).double() - h, hosts.box[ids])
            .abs()
            .max()
        )
        if err > 2e-6:
            raise ValueError(f"Original host-coordinate validation failed: {err}")
    else:
        # Hand insertion stores inserted coordinates in pin_pos. Validate the known
        # sequential-host convention against the saved original B-atom targets.
        d = minimum_image(h - torch.from_numpy(samp["target"])[:, None], hosts.box[ids]).norm(
            dim=-1
        )
        d = torch.where(hosts.types[ids] == 1, d, torch.full_like(d, float("inf")))
        if d.min(-1).values.max() > 2e-6:
            raise ValueError("Hand-insertion host target mismatch")
    return ids


def one_site_per_host(split, seed, limit=None):
    rng = np.random.default_rng(seed)
    sites = []
    for g in rng.permutation(len(split)):
        candidates = torch.nonzero(split.labels[g] == 3).flatten().numpy()
        if len(candidates):
            sites.append((int(g), int(rng.choice(candidates))))
        if limit is not None and len(sites) >= limit:
            break
    g, a = np.asarray(sites).T
    s = Structures(split.structures.pos[g], split.structures.types[g], split.structures.box[g])
    return s, s.pos[torch.arange(len(g)), a], g, a


def weighted_quantiles(items, owners):
    values = np.concatenate(items)
    order = np.argsort(values)
    values = values[order]
    _, groups = np.unique(owners, return_inverse=True)
    indices = np.repeat(groups, [len(x) for x in items])[order]

    def calc(counts):
        c = np.cumsum(counts[indices])
        if not len(c) or c[-1] == 0:
            return np.full(len(QUANTILE_LEVELS), np.nan)
        ranks = QUANTILE_LEVELS * (c[-1] - 1)
        lo = np.floor(ranks).astype(int)
        hi = np.ceil(ranks).astype(int)
        a = values[np.searchsorted(c, lo, side="right")]
        b = values[np.searchsorted(c, hi, side="right")]
        return a + (b - a) * (ranks - lo)

    return calc, len(np.unique(owners))


def cluster_w1(items, ids, reference, refids, nboot, seed):
    a, na = weighted_quantiles(items, ids)
    b, nb = weighted_quantiles(reference, refids)
    point = float(np.abs(a(np.ones(na, dtype=int)) - b(np.ones(nb, dtype=int))).mean())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        ca = np.bincount(rng.integers(na, size=na), minlength=na)
        cb = np.bincount(rng.integers(nb, size=nb), minlength=nb)
        vals.append(np.abs(a(ca) - b(cb)).mean())
    return [point, *map(float, np.quantile(vals, [0.025, 0.975]))]


def structural(out, nboot):
    torch.set_num_threads(1)
    test, train = load_split(DATA / "test.npz"), load_split(DATA / "train.npz")
    th = DefectThresholds.from_dict(
        json.loads((DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    ns, nc, nh, na = one_site_per_host(test, 71)
    ts, tc, _, _ = one_site_per_host(train, 72, 256)
    print("Computing natural structural reference", flush=True)
    ref = analysis.features(ns, nc, 3.0)
    floor = analysis.features(ts, tc, 3.0)
    ref_calcs = {k: weighted_quantiles(v, nh)[0] for k, v in ref.items()}
    floor_calcs = {k: weighted_quantiles(v, np.arange(len(v)))[0] for k, v in floor.items()}
    rows, disturb, cohort = [], [], {}
    for method, path in RUNS.items():
        s, req, samp = load_run(path)
        ids = recover_hosts(samp, test.structures)
        ok = success(s, req, th).numpy()
        ix = np.flatnonzero(ok)
        if not len(ix):
            raise ValueError(f"No successes: {method}")
        sub = Structures(s.pos[ix], s.types[ix], s.box[ix])
        feat = analysis.features(sub, req.target[ix], 3.0)
        print(method, len(ix), "successes on", len(np.unique(ids[ix])), "hosts", flush=True)
        rng = np.random.default_rng(17)
        for name, values in feat.items():
            val = cluster_w1(values, ids[ix], ref[name], nh, nboot, 17)
            natural = []
            for _ in range(nboot):
                sel = rng.choice(len(floor[name]), size=len(ix), replace=False)
                refsel = rng.integers(len(ref[name]), size=len(ref[name]))
                floor_weights = np.zeros(len(floor[name]), dtype=int)
                floor_weights[sel] = 1
                ref_weights = np.bincount(refsel, minlength=len(ref[name]))
                natural.append(
                    float(
                        np.abs(
                            floor_calcs[name](floor_weights) - ref_calcs[name](ref_weights)
                        ).mean()
                    )
                )
            rows.append(
                dict(
                    method=method,
                    metric=name,
                    n_success=len(ix),
                    n_hosts=len(np.unique(ids[ix])),
                    n_reference_hosts=len(nh),
                    w1=val[0],
                    lo=val[1],
                    hi=val[2],
                    matched_natural_median=float(np.median(natural)),
                    matched_natural_p025=float(np.quantile(natural, 0.025)),
                    matched_natural_p975=float(np.quantile(natural, 0.975)),
                )
            )
        original = test.structures.pos[ids]
        distance = minimum_image(original - req.target[:, None], s.box).norm(dim=-1)
        delta = minimum_image(s.pos - original, s.box)
        norm = delta.norm(dim=-1)
        for group, selected in [("all", np.arange(len(ids))), ("successful", ix)]:
            for lo in range(7):
                vals = []
                owner = []
                for i in selected:
                    mask = (distance[i] >= lo) & (distance[i] < lo + 1)
                    if mask.any():
                        vals.append(float(norm[i][mask].mean()))
                        owner.append(int(ids[i]))
                # Resample host means, so repeated requests do not become independent hosts.
                hostmeans = [
                    np.mean([v for v, h in zip(vals, owner) if h == g]) for g in sorted(set(owner))
                ]
                ci = analysis.ci(hostmeans, np.mean, nboot, 17)
                disturb.append(
                    dict(
                        method=method,
                        group=group,
                        r_lo=lo,
                        r_hi=lo + 1,
                        n_hosts=len(hostmeans),
                        host_balanced_mean_displacement=ci[0],
                        lo=ci[1],
                        hi=ci[2],
                    )
                )
        if method in ["fixed_r6", "hand_insert"]:
            # Uniformly select hosts first and one successful request within each host.
            rng = np.random.default_rng(73)
            unique = rng.permutation(np.unique(ids[ix]))[:32]
            chosen = np.array([rng.choice(ix[ids[ix] == h]) for h in unique])
            if len(chosen) < 32:
                raise ValueError("Not enough independent successful hosts")
            cohort[method] = dict(
                pos=s.pos[chosen].numpy(),
                types=s.types[chosen].numpy(),
                box=s.box[chosen].numpy(),
                target=req.target[chosen].numpy(),
                host_ids=ids[chosen],
                source_sample=chosen,
            )
            if method == "fixed_r6":
                cohort["unedited_host"] = dict(
                    pos=original[chosen].numpy(),
                    types=s.types[chosen].numpy(),
                    box=s.box[chosen].numpy(),
                    target=req.target[chosen].numpy(),
                    host_ids=ids[chosen],
                    source_sample=chosen,
                )
    cohort["natural"] = dict(
        pos=ns.pos[:32].numpy(),
        types=ns.types[:32].numpy(),
        box=ns.box[:32].numpy(),
        target=nc[:32].numpy(),
        host_ids=nh[:32],
        source_sample=na[:32],
    )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "host_aware_structure.csv", rows)
    write_csv(out / "host_disturbance.csv", disturb)
    np.savez_compressed(
        out / "md_cohorts.npz",
        **{f"{group}__{k}": v for group, data in cohort.items() for k, v in data.items()},
    )
    (out / "structural_protocol.json").write_text(
        json.dumps(
            dict(
                runs=RUNS,
                bootstrap=nboot,
                natural_reference="one uniformly selected D2 site per test host",
                natural_floor="one site per 256 randomly selected train hosts; repeatedly subsampled to the successful request count",
                uncertainty="host bootstrap in both ensembles; matched-floor band is subsampling variability, not a CI on a population floor",
                host_recovery="full saved host-coordinate check for model runs; sequential-host and saved target validation for hand insertion",
                cohort_selection="32 randomly selected distinct successful hosts, one request per host",
                seed=17,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Structural analysis and MD cohorts saved", flush=True)


def prepare_cohorts(out):
    torch.set_num_threads(1)
    test = load_split(DATA / "test.npz")
    th = DefectThresholds.from_dict(
        json.loads((DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    cohort = {}
    for method in ["fixed_r6", "hand_insert"]:
        s, req, samp = load_run(RUNS[method])
        ids = recover_hosts(samp, test.structures)
        ix = np.flatnonzero(success(s, req, th).numpy())
        rng = np.random.default_rng(73)
        unique = rng.permutation(np.unique(ids[ix]))[:32]
        chosen = np.array([rng.choice(ix[ids[ix] == h]) for h in unique])
        if len(chosen) != 32:
            raise ValueError("Need 32 distinct successful hosts")
        cohort[method] = dict(
            pos=s.pos[chosen].numpy(),
            types=s.types[chosen].numpy(),
            box=s.box[chosen].numpy(),
            target=req.target[chosen].numpy(),
            host_ids=ids[chosen],
            source_sample=chosen,
        )
        if method == "fixed_r6":
            cohort["unedited_host"] = dict(
                cohort[method], pos=test.structures.pos[ids[chosen]].numpy()
            )
    ns, nc, nh, na = one_site_per_host(test, 71)
    cohort["natural"] = dict(
        pos=ns.pos[:32].numpy(),
        types=ns.types[:32].numpy(),
        box=ns.box[:32].numpy(),
        target=nc[:32].numpy(),
        host_ids=nh[:32],
        source_sample=na[:32],
    )
    np.savez_compressed(
        out / "md_cohorts.npz",
        **{f"{group}__{k}": v for group, data in cohort.items() for k, v in data.items()},
    )
    print("Prepared four cohorts on 32 distinct hosts each", flush=True)


def make_lmp(pos, types, box):
    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lmp.commands_string(f"""units lj
dimension 2
atom_style atomic
boundary p p p
atom_modify map array
region box block 0 {box[0]} 0 {box[1]} -0.5 0.5
create_box 2 box
mass * 1.0
{_PAIR_COMMANDS}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
fix plane all enforce2d
thermo_style custom step temp pe fmax
thermo_modify norm no
""")
    xyz = np.zeros((len(pos), 3))
    xyz[:, :2] = np.mod(pos, box)
    lmp.create_atoms(len(pos), None, (types + 1).astype(np.int32), xyz.reshape(-1))
    lmp.command("run 0")
    return lmp


def md_worker(job):
    group, i, temperature, velseed, arrays, output = job
    torch.set_num_threads(1)
    file = Path(output) / "md_trajectories" / f"{group}_{i:02d}_T{temperature:g}_v{velseed}.json"
    fingerprint = hashlib.sha256(
        b"free-ka-2d-nvt-cg-v2"
        + _PAIR_COMMANDS.encode()
        + b"".join(np.asarray(x).tobytes() for x in arrays)
    ).hexdigest()
    if file.exists():
        saved = json.loads(file.read_text())
        if saved.get("input_fingerprint") != fingerprint:
            raise ValueError(f"MD resume input mismatch: {file}")
        return saved
    pos, types, box, target, host = arrays
    s = Structures(
        torch.tensor(pos)[None].double(),
        torch.tensor(types)[None].long(),
        torch.tensor(box)[None].double(),
    )
    th = DefectThresholds.from_dict(
        json.loads((DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    atom = int(
        nearest_atom(s, torch.tensor(target)[None].double(), torch.ones(1, dtype=torch.long))[0][0]
    )
    main = make_lmp(pos, types, box)
    probe = make_lmp(pos, types, box)
    try:
        expected = float(ka_energy(s)[0])
        actual = float(main.get_thermo("pe"))
        if abs(expected - actual) > 1e-6:
            raise ValueError(f"LAMMPS energy mismatch {expected} {actual}")
        main.commands_string(
            f"velocity all create {temperature} {velseed + 1} mom yes rot no dist gaussian\ntimestep 0.005\nfix bath all nvt temp {temperature} {temperature} 0.1"
        )
        main.command("run 0")
        probe.command("min_style cg")
        previous = 0
        rows = []
        for step in [0, 1000, 4000, 10000]:
            if step > previous:
                main.command(f"run {step - previous}")
            xyz = np.array(main.gather_atoms("x", 1, 3)).reshape(-1, 3).copy()
            thermal = Structures(torch.tensor(xyz[:, :2])[None].double(), s.types, s.box)
            thermal_label = int(detect_defects(thermal, th)[0, atom])
            data = np.ascontiguousarray(xyz.reshape(-1), dtype=np.float64)
            probe.scatter_atoms("x", 1, 3, data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            probe.commands_string("minimize 0 1e-6 100000 1000000\nrun 0")
            qxyz = np.array(probe.gather_atoms("x", 1, 3)).reshape(-1, 3).copy()
            quenched = Structures(torch.tensor(qxyz[:, :2])[None].double(), s.types, s.box)
            residual = float(probe.get_thermo("fmax"))
            minimizer = "cg"
            polish_steps = 0
            if residual > 1e-6:
                polished = fire_minimize(quenched, fmax=1e-6, max_steps=100000)
                if not bool(polished.converged[0]):
                    raise ValueError(
                        f"Snapshot refinement failed: {group} sample={i} T={temperature} step={step}"
                    )
                quenched = polished.structures
                residual = float(polished.max_force[0])
                polish_steps = int(polished.n_steps[0])
                minimizer = "cg+torch_fire"
            label = int(detect_defects(quenched, th)[0, atom])
            disp = minimum_image(
                quenched.pos[:, atom : atom + 1] - torch.tensor(target)[None, None], s.box
            ).norm()
            rows.append(
                dict(
                    step=step,
                    time=step * 0.005,
                    thermal_same_atom=int(thermal_label == 3),
                    quenched_same_atom=int(label == 3),
                    quenched_site=int(label == 3 and disp < 0.5),
                    max_force=residual,
                    minimizer=minimizer,
                    polish_steps=polish_steps,
                    temperature=float(main.get_thermo("temp")),
                )
            )
            if rows[-1]["max_force"] > 1e-6:
                raise ValueError(
                    f"Snapshot minimization did not converge: {group} sample={i} T={temperature} velocity={velseed} step={step} fmax={rows[-1]['max_force']}"
                )
            previous = step
        result = dict(
            input_fingerprint=fingerprint,
            group=group,
            sample=i,
            host=int(host),
            temperature=temperature,
            velocity_seed=velseed,
            initial_energy_error=abs(actual - expected),
            observations=rows,
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        main.close()
        probe.close()


def wilson_interval(k, n):
    """95% Wilson interval; finite uncertainty even when no losses are observed."""
    if n < 1:
        raise ValueError("positive sample count required")
    z = 1.959963984540054
    p = k / n
    scale = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / scale
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0.0, float(centre - half)), min(1.0, float(centre + half))]


def md(out, workers):
    with np.load(out / "md_cohorts.npz") as f:
        cohorts = {k: f[k] for k in f.files}
    jobs = []
    for group in ["fixed_r6", "hand_insert", "natural", "unedited_host"]:
        for i in range(len(cohorts[group + "__pos"])):
            arrays = tuple(
                cohorts[group + "__" + k][i] for k in ["pos", "types", "box", "target", "host_ids"]
            )
            for t in [0.01, 0.1]:
                for seed in [101, 202, 303]:
                    jobs.append((group, i, t, seed, arrays, str(out)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        trajectories = []
        for j, result in enumerate(pool.map(md_worker, jobs)):
            trajectories.append(result)
            if (j + 1) % 24 == 0:
                print(f"MD {j + 1}/{len(jobs)} trajectories", flush=True)
    rows = []
    for group in ["fixed_r6", "hand_insert", "natural", "unedited_host"]:
        for t in [0.01, 0.1]:
            selected = [r for r in trajectories if r["group"] == group and r["temperature"] == t]
            for ix in range(4):
                for metric in ["thermal_same_atom", "quenched_same_atom", "quenched_site"]:
                    vals = []
                    for host in sorted({r["host"] for r in selected}):
                        vals.append(
                            np.mean(
                                [
                                    r["observations"][ix][metric]
                                    for r in selected
                                    if r["host"] == host
                                ]
                            )
                        )
                    ci = analysis.ci(vals, np.mean, 1000, 19)
                    all_velocity = sum(v == 1 for v in vals)
                    bounds = wilson_interval(all_velocity, len(vals))
                    # Pointwise occupancy; not continuous survival between observations.
                    rows.append(
                        dict(
                            group=group,
                            temperature=t,
                            time=selected[0]["observations"][ix]["time"],
                            metric=metric,
                            n_hosts=len(vals),
                            n_trajectories=len(selected),
                            hosts_all_velocity_retain=all_velocity,
                            host_all_velocity_fraction=all_velocity / len(vals),
                            host_wilson_lo=bounds[0],
                            host_wilson_hi=bounds[1],
                            fraction=ci[0],
                            lo=ci[1],
                            hi=ci[2],
                        )
                    )
    write_csv(out / "md_retention.csv", rows)
    (out / "md_protocol.json").write_text(
        json.dumps(
            dict(
                temperatures=[0.01, 0.1],
                temperature_rationale="dataset final T=0.01; 10x temperature stress test is not claimed to be a calibrated material operating temperature",
                dt=0.005,
                tdamp=0.1,
                steps=[0, 1000, 4000, 10000],
                velocity_seeds=[101, 202, 303],
                ensemble="whole-box free NVT; no pins; fixed density",
                potential="repository shifted KA Lennard-Jones, species-specific cutoffs",
                n_trajectories=len(trajectories),
                interpretation="pointwise defect occupancy of the initially nearest B atom, thermal and independently minimized snapshots; not continuous lifetime or long-term stability",
                natural_controls="same-class natural defects on 32 distinct test hosts",
                unedited_controls="original hosts and sites paired with generated cohort",
                max_initial_energy_error=max(r["initial_energy_error"] for r in trajectories),
            ),
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axs, [0.01, 0.1]):
        for group in ["fixed_r6", "hand_insert", "natural", "unedited_host"]:
            rs = [
                r
                for r in rows
                if r["group"] == group
                and r["temperature"] == t
                and r["metric"] == "quenched_same_atom"
            ]
            ax.plot(
                [r["time"] for r in rs],
                [r["host_all_velocity_fraction"] for r in rs],
                ".-",
                label=group,
            )
            ax.fill_between(
                [r["time"] for r in rs],
                [r["host_wilson_lo"] for r in rs],
                [r["host_wilson_hi"] for r in rs],
                alpha=0.1,
            )
        ax.set(
            title=f"T={t:g}",
            xlabel="LJ time",
            ylabel="Hosts retaining D2 in all three velocity seeds",
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "md_retention.png", dpi=160)
    plt.close(fig)
    print("MD analysis complete", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["structural", "md", "cohorts"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.bootstrap < 20 or a.workers < 1:
        p.error("bootstrap >=20 and workers >=1 required")
    if a.mode == "structural":
        structural(a.out, a.bootstrap)
    elif a.mode == "cohorts":
        prepare_cohorts(a.out)
    else:
        md(a.out, a.workers)


if __name__ == "__main__":
    main()
