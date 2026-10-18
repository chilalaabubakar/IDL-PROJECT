"""Extended disjoint-cohort MD and all-request creation-and-retention evaluation.

python scripts/complete_gap1.py local --out results/gap1_extended --workers 4
python scripts/complete_gap1.py joint --repeats runs/full/repeats --out runs/full/joint --workers 4

Local: original 32 hosts plus 24 additional distinct hosts per method, to time 250.
Joint: only initially successful requests need MD; all failed requests remain zero
in the all-request denominator. Three velocity seeds are averaged within requests,
and whole hosts are bootstrapped. This does not treat trajectories as independent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    "physical", Path(__file__).with_name("check_physical_credibility.py")
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
STEPS = [0, 1000, 10000, 30000, 50000]
OPTIONS = dict(
    protocol="gap1-250-v1",
    steps=STEPS,
    site_definition="nearest B within 0.5 carries frozen D2 label",
)
GROUPS = ["fixed_r6", "hand_insert", "natural", "unedited_host"]


def execute_worker(job):
    return p.md_worker(job)


def execute(jobs, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(execute_worker, jobs)):
            results.append(result)
            if (i + 1) % 24 == 0:
                print(f"{i + 1}/{len(jobs)} trajectories complete", flush=True)
    return results


def disjoint_cohorts(out):
    old = Path("results/credibility_followup_20261018/md_cohorts.npz")
    with np.load(old) as f:
        original = {k: f[k] for k in f.files}
    datasets = {"A": original}
    new = {}
    test = p.load_split(p.DATA / "test.npz")
    th = p.DefectThresholds.from_dict(
        json.loads((p.DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    for method in ["fixed_r6", "hand_insert"]:
        s, req, saved = p.load_run(p.RUNS[method])
        ids = p.recover_hosts(saved, test.structures)
        good = np.flatnonzero(p.success(s, req, th).numpy())
        eligible = np.setdiff1d(np.unique(ids[good]), original[method + "__host_ids"])
        if len(eligible) < 24:
            raise ValueError(f"Need 24 disjoint additional hosts for {method}")
        rng = np.random.default_rng(417)
        hosts = rng.choice(eligible, 24, replace=False)
        ix = np.array([rng.choice(good[ids[good] == h]) for h in hosts])
        for key, value in dict(
            pos=s.pos[ix].numpy(),
            types=s.types[ix].numpy(),
            box=s.box[ix].numpy(),
            target=req.target[ix].numpy(),
            host_ids=ids[ix],
            source_sample=ix,
        ).items():
            new[method + "__" + key] = value
        if method == "fixed_r6":
            for key in ["types", "box", "target", "host_ids", "source_sample"]:
                new["unedited_host__" + key] = new[method + "__" + key]
            new["unedited_host__pos"] = test.structures.pos[ids[ix]].numpy()
    ns, nc, nh, na = p.one_site_per_host(test, 431)
    ix = np.flatnonzero(~np.isin(nh, original["natural__host_ids"]))[:24]
    for key, value in dict(
        pos=ns.pos[ix].numpy(),
        types=ns.types[ix].numpy(),
        box=ns.box[ix].numpy(),
        target=nc[ix].numpy(),
        host_ids=nh[ix],
        source_sample=na[ix],
    ).items():
        new["natural__" + key] = value
    datasets["B"] = new
    for cohort, data in datasets.items():
        target = out / cohort
        target.mkdir(parents=True, exist_ok=True)
        file = target / "md_cohorts.npz"
        if file.exists():
            with np.load(file) as f:
                if set(f.files) != set(data) or any(
                    not np.array_equal(f[k], v) for k, v in data.items()
                ):
                    raise ValueError("Cohort resume mismatch")
        else:
            np.savez_compressed(file, **data)
    for group in GROUPS:
        assert not set(datasets["A"][group + "__host_ids"]) & set(new[group + "__host_ids"])
    return datasets


def local(out, workers):
    datasets = disjoint_cohorts(out)
    jobs = []
    for cohort, data in datasets.items():
        for group in GROUPS:
            for i in range(len(data[group + "__pos"])):
                arrays = tuple(
                    data[group + "__" + key][i]
                    for key in ["pos", "types", "box", "target", "host_ids"]
                )
                for temp in [0.01, 0.1]:
                    for seed in [101, 202, 303]:
                        jobs.append((group, i, temp, seed, arrays, str(out / cohort), OPTIONS))
    results = execute(jobs, workers)
    rows = []
    for group in GROUPS:
        for temp in [0.01, 0.1]:
            selected = [r for r in results if r["group"] == group and r["temperature"] == temp]
            assert len({r["host"] for r in selected}) == 56
            for k, step in enumerate(STEPS):
                for metric in ["quenched_same_atom", "quenched_requested_site"]:
                    hostmeans = [
                        np.mean([r["observations"][k][metric] for r in selected if r["host"] == h])
                        for h in sorted({r["host"] for r in selected})
                    ]
                    ci = p.analysis.ci(hostmeans, np.mean, 1000, 91)
                    all_vel = sum(x == 1 for x in hostmeans)
                    lo, hi = p.wilson_interval(all_vel, len(hostmeans))
                    rows.append(
                        dict(
                            group=group,
                            temperature=temp,
                            time=step * 0.005,
                            n_hosts=56,
                            n_trajectories=len(selected),
                            metric=metric,
                            fraction=ci[0],
                            lo=ci[1],
                            hi=ci[2],
                            hosts_all_velocity=all_vel,
                            host_wilson_lo=lo,
                            host_wilson_hi=hi,
                        )
                    )
    p.write_csv(out / "extended_retention.csv", rows)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                options=OPTIONS,
                n_hosts_per_method=56,
                cohort_sizes=[32, 24],
                disjoint_within_method=True,
                temperatures=[0.01, 0.1],
                velocity_seeds=[101, 202, 303],
                n_trajectories=len(results),
                interpretation="pointwise occupancy, not continuous lifetime",
                all_snapshot_residuals_within_tolerance=all(
                    o["max_force"] <= 1e-6 for r in results for o in r["observations"]
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print("Extended local stability complete", flush=True)


def request_arrays(repeats):
    test = p.load_split(p.DATA / "test.npz").structures
    th = p.DefectThresholds.from_dict(
        json.loads((p.DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    methods = {}
    reference = None
    for folder in sorted(repeats.iterdir()):
        if not (folder / "samples.npz").exists():
            continue
        with np.load(folder / "samples.npz") as f:
            d = {k: f[k] for k in f.files}
        if not d["converged"].all():
            raise ValueError("Unconverged full-run samples")
        if len(d["pos"]) != 256:
            raise ValueError("Joint full evaluation requires 256 requests per model")
        ids = d["host_ids"]
        if not np.array_equal(d["host_pos"], test.pos[ids].numpy()):
            raise ValueError("Saved host IDs disagree with original host positions")
        matched = {
            key: d[key] for key in ["host_ids", "target", "types", "box", "pin_mask", "host_pos"]
        }
        if reference is None:
            reference = matched
        elif any(not np.array_equal(matched[key], reference[key]) for key in reference):
            raise ValueError("Full runs do not use identical matched requests")
        s = p.Structures(
            torch.from_numpy(d["pos"]).double(),
            torch.from_numpy(d["types"]).long(),
            torch.from_numpy(d["box"]).double(),
        )
        req = p.Request(
            torch.from_numpy(d["target"]).double(),
            torch.full((256,), 3, dtype=torch.long),
            torch.from_numpy(d["pin_mask"]),
            torch.from_numpy(d["host_pos"]).double(),
            torch.zeros(256, 256, dtype=torch.long),
        )
        ok = p.success(s, req, th).numpy()
        if not np.array_equal(ok, d["success"]):
            raise ValueError("Stored success mask disagrees with detector")
        prior = p.success(
            p.Structures(test.pos[ids], test.types[ids], test.box[ids]), req, th
        ).numpy()
        methods[folder.name] = dict(
            pos=d["pos"],
            types=d["types"],
            box=d["box"],
            target=d["target"],
            host_ids=ids,
            initial_success=ok,
            original_success=prior,
        )
    if set(methods) != {"seed0", "seed1", "seed2", "sham_seed0"}:
        raise ValueError("All four full matched runs required")
    s, req, saved = p.load_run(p.RUNS["hand_insert"])
    ids = p.recover_hosts(saved, test)
    methods["hand_insert"] = dict(
        pos=s.pos.numpy(),
        types=s.types.numpy(),
        box=s.box.numpy(),
        target=req.target.numpy(),
        host_ids=ids,
        initial_success=p.success(s, req, th).numpy(),
        original_success=p.success(
            p.Structures(test.pos[ids], test.types[ids], test.box[ids]), req, th
        ).numpy(),
    )
    return methods


def host_ratio_ci(values, ids, eligible):
    groups = []
    for host in np.unique(ids):
        mask = (ids == host) & eligible
        if mask.any():
            groups.append((float(values[mask].sum()), int(mask.sum())))
    return p.analysis.ci(groups, lambda x: sum(z[0] for z in x) / sum(z[1] for z in x), 1000, 93)


def joint(out, repeats, workers):
    methods = request_arrays(repeats)
    jobs = []
    for method, data in methods.items():
        for i in np.flatnonzero(data["initial_success"]):
            arrays = tuple(data[key][i] for key in ["pos", "types", "box", "target", "host_ids"])
            for seed in [101, 202, 303]:
                jobs.append((method, int(i), 0.01, seed, arrays, str(out), OPTIONS))
    results = execute(jobs, workers)
    rows = []
    outcomes = []
    for method, data in methods.items():
        for k, step in enumerate(STEPS):
            values = np.zeros(256)
            for i in np.flatnonzero(data["initial_success"]):
                matched = [r for r in results if r["group"] == method and r["sample"] == i]
                if len(matched) != 3:
                    raise ValueError("Missing velocity trajectories")
                values[i] = np.mean(
                    [r["observations"][k]["quenched_requested_site"] for r in matched]
                )
            for definition, eligible in [
                ("all_requests", np.ones(256, dtype=bool)),
                ("initially_nonD2", ~data["original_success"]),
            ]:
                ci = host_ratio_ci(values, data["host_ids"], eligible)
                eligible_hosts = np.unique(data["host_ids"][eligible])
                any_retention = sum(
                    bool(np.any(values[(data["host_ids"] == h) & eligible] > 0))
                    for h in eligible_hosts
                )
                host_lo, host_hi = p.wilson_interval(any_retention, len(eligible_hosts))
                rows.append(
                    dict(
                        method=method,
                        time=step * 0.005,
                        definition=definition,
                        n_requests=int(eligible.sum()),
                        initial_placements=int((data["initial_success"] & eligible).sum()),
                        joint_fraction=ci[0],
                        lo=ci[1],
                        hi=ci[2],
                        n_hosts=len(eligible_hosts),
                        hosts_with_any_joint_retention=any_retention,
                        host_incidence_wilson_lo=host_lo,
                        host_incidence_wilson_hi=host_hi,
                    )
                )
            for i in range(256):
                outcomes.append(
                    dict(
                        method=method,
                        sample=i,
                        host=int(data["host_ids"][i]),
                        time=step * 0.005,
                        initial_success=int(data["initial_success"][i]),
                        originally_D2=int(data["original_success"][i]),
                        joint_velocity_fraction=values[i],
                    )
                )
    p.write_csv(out / "joint_yield.csv", rows)
    p.write_csv(out / "joint_outcomes.csv", outcomes)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                options=OPTIONS,
                temperature=0.01,
                velocity_seeds=[101, 202, 303],
                n_trajectories=len(results),
                denominator="all 256 requests; initially failed placements count as zero; separate initially-nonD2 denominator",
                uncertainty="request-weighted ratio with whole-host bootstrap; velocity realizations averaged within requests",
                all_snapshot_residuals_within_tolerance=all(
                    o["max_force"] <= 1e-6 for r in results for o in r["observations"]
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "README.md").write_text(
        "# Full matched creation-and-retention evaluation\n\nCompleted all four 256-request runs and the retained 256-request insertion baseline. See `joint_yield.csv` for host-bootstrap intervals and `joint_outcomes.csv` for per-request outcomes. Initially failed placements contribute zero, including if a later unsolicited defect could appear; this tests successful placement *and* later retention at the requested site. Observation times are 0, 5, 50, 150 and 250 LJ units, T=0.01. These are pointwise measurements, not a continuous survival curve or proof of natural structural equivalence.\n"
    )
    print("Full joint evaluation complete", flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode", choices=["local", "joint"])
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--repeats", type=Path)
    a.add_argument("--workers", type=int, default=4)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.workers < 1:
        a.error("workers must be positive")
    if args.mode == "local":
        local(args.out, args.workers)
    else:
        if args.repeats is None:
            a.error("--repeats required for joint mode")
        joint(args.out, args.repeats, args.workers)


if __name__ == "__main__":
    main()
