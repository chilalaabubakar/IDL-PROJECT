"""Resumable zero-shot generalization: hosts, requests, model, insertion and report.

python scripts/run_gap2.py hosts --condition ka2d_1024_slow --out runs/gap2
python scripts/run_gap2.py requests --condition ka2d_1024_slow --out runs/gap2
python scripts/run_gap2.py model --condition ka2d_1024_slow --model seed0 --out runs/gap2 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from glassdiff.analysis.defects import DefectThresholds
from glassdiff.baselines import hand_insert
from glassdiff.data.patches import PatchLibrary
from glassdiff.data.requests import host_init, make_eval_request
from glassdiff.diffusion.conditioning import PinnedLabel
from glassdiff.diffusion.guidance import CFGDenoiser
from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, sample
from glassdiff.eval.metrics import success
from glassdiff.md.lammps_quench import (
    QualityGates,
    QuenchProtocol,
    make_glass,
    passes_quality_gates,
)
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize
from glassdiff.types import LabelToken, Request, Structures
from glassdiff.utils.runs import seed_everything

MANIFEST = Path("configs/eval/gap2_generalization.json")
METHODS = ["seed0", "seed1", "seed2", "sham_seed0", "hand_insert"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_npz(path, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **data)
    temp.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def guard(path, config):
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Resume configuration mismatch: {path}")
    save_json(path, config)


def candidate(job):
    config_file, folder, slot, first = job
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(config_file).read_text())
    for attempt in range(first, 10):
        seed = cfg["splits"]["test"]["seed0"] + slot + attempt * 1000
        file = Path(folder) / "candidates" / f"host_{slot:03d}_try{attempt}.npz"
        if file.exists():
            data = arrays(file)
            info = json.loads(str(data["info"]))
            if info["config_sha256"] != digest(config_file):
                raise ValueError("Host cache configuration mismatch")
        else:
            start = time.monotonic()
            protocol = QuenchProtocol(**cfg["protocol"])
            glass = make_glass(cfg["n_atoms"], seed, protocol)
            ok, stats = passes_quality_gates(glass, protocol.frac_a, QualityGates())
            info = dict(
                seed=seed,
                slot=slot,
                attempt=attempt,
                quality_pass=ok,
                stats=stats,
                seconds=time.monotonic() - start,
                config_sha256=digest(config_file),
            )
            save_npz(
                file,
                pos=glass.pos,
                types=glass.types,
                box=glass.box,
                pe_atom=glass.pe_atom,
                thermal_pos=glass.thermal_pos,
                info=np.array(json.dumps(info)),
            )
        if info["quality_pass"]:
            return str(file), info
    raise RuntimeError(
        f"Host slot {slot} failed 10 fixed attempts; inspect logs, do not silently relax gates"
    )


def hosts(args, manifest):
    cfgfile = Path("configs/data/gap2") / (args.condition + ".yaml")
    cfg = yaml.safe_load(cfgfile.read_text())
    folder = args.out / args.condition / "hosts"
    n = cfg["splits"]["test"]["n"]
    config = dict(
        config_sha256=digest(cfgfile), n_hosts=n, seed_rule="seed0+slot+1000*attempt; attempts0..9"
    )
    folder.mkdir(parents=True, exist_ok=True)
    guard(folder / "config.json", config)
    if (folder / "protocol.json").exists():
        assert len(arrays(folder / "samples.npz")["pos"]) == n
        return
    selected, next_attempt, outliers = {}, dict.fromkeys(range(n), 0), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            pending = [i for i in range(n) if i not in selected]
            jobs = [(str(cfgfile), str(folder), i, next_attempt[i]) for i in pending]
            for file, info in pool.map(candidate, jobs):
                selected[info["slot"]] = (file, info)
                print(
                    f"{args.condition}: {len(selected)}/{n} hosts provisionally accepted",
                    flush=True,
                )
            pe = np.array([selected[i][1]["stats"]["pe_mean"] for i in range(n)])
            med = np.median(pe)
            sd = 1.4826 * np.median(np.abs(pe - med)) or pe.std() or 1.0
            rejected = np.flatnonzero(np.abs(pe - med) / sd > cfg["quality"]["pe_max_z"])
            if not len(rejected):
                break
            for i in rejected:
                _, info = selected.pop(int(i))
                outliers.append(dict(seed=info["seed"], reason="within_condition_pe_outlier"))
                next_attempt[int(i)] = info["attempt"] + 1
    data = [arrays(selected[i][0]) for i in range(n)]
    save_npz(
        folder / "samples.npz",
        **{
            k: np.stack([d[k] for d in data])
            for k in ["pos", "types", "box", "pe_atom", "thermal_pos"]
        },
        seeds=np.array([selected[i][1]["seed"] for i in range(n)]),
    )
    all_info = [
        json.loads(str(arrays(f)["info"])) for f in sorted((folder / "candidates").glob("*.npz"))
    ]
    save_json(
        folder / "protocol.json",
        dict(
            config=config,
            n_atoms=cfg["n_atoms"],
            protocol=cfg["protocol"],
            quality=cfg["quality"],
            n_hosts=n,
            n_candidates=len(all_info),
            quality_rejections=[i for i in all_info if not i["quality_pass"]],
            pe_rejections=outliers,
            accepted=[selected[i][1] for i in range(n)],
        ),
    )


def structures(d, sl, key="pos", device="cpu", dtype=torch.float64):
    return Structures(
        torch.as_tensor(d[key][sl], dtype=dtype, device=device),
        torch.as_tensor(d["types"][sl], dtype=torch.long, device=device),
        torch.as_tensor(d["box"][sl], dtype=dtype, device=device),
    )


def requests(args, manifest):
    folder = args.out / args.condition
    hostfile = folder / "hosts/samples.npz"
    config = dict(
        host_sha256=digest(hostfile),
        manifest_sha256=digest(MANIFEST),
        n_requests=manifest["n_requests"],
        target_seed=937,
        target_rule="uniform spatial draws snapped to nearest B; saved once",
    )
    guard(folder / "requests_config.json", config)
    if (folder / "requests.npz").exists():
        return
    d = arrays(hostfile)
    n = manifest["n_requests"]
    ids = np.arange(n) % len(d["pos"])
    gen = torch.Generator().manual_seed(937)
    raw_target = torch.rand(n, 2, generator=gen).numpy() * d["box"][ids]
    chunks = []
    th = DefectThresholds.from_dict(manifest["thresholds"])
    for start in range(0, n, 4):
        sl = slice(start, min(start + 4, n))
        host = structures(d, ids[sl], dtype=torch.float32)
        req = make_eval_request(
            3,
            "host",
            host.types,
            host.box,
            host=host,
            target=torch.from_numpy(raw_target[sl]).float(),
            host_r_out=6,
        )
        # Score the float32 sampling input after conversion to analysis precision.
        hd = Structures(host.pos.double(), host.types, host.box.double())
        rd = Request(req.target.double(), req.defect, req.pin_mask, req.pin_pos.double(), req.label)
        chunks.append(
            dict(
                pin_pos=host.pos.numpy(),
                types=host.types.numpy(),
                box=host.box.numpy(),
                target=req.target.numpy(),
                pin_mask=req.pin_mask.numpy(),
                label=req.label.numpy(),
                original_success=success(hd, rd, th).numpy(),
                host_ids=ids[sl],
                host_seeds=d["seeds"][ids[sl]],
            )
        )
    save_npz(
        folder / "requests.npz", **{k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    )
    print(f"{args.condition}: saved {n} matched requests", flush=True)


def evaluate(args, manifest):
    folder = args.out / args.condition
    source = folder / "requests.npz"
    d = arrays(source)
    model_name = args.model if args.mode == "model" else "hand_insert"
    output = folder / model_name
    ckpt = (
        manifest["checkpoints"][model_name if model_name.startswith("seed") else "seed0"]
        if model_name != "hand_insert"
        else None
    )
    if ckpt and digest(ckpt["path"]) != ckpt["sha256"]:
        raise ValueError("Frozen checkpoint digest mismatch")
    patchfile = Path(
        manifest.get("training_patch_library", {}).get("path", "data/ka2d_256/patches_train.npz")
    )
    if not ckpt and digest(patchfile) != manifest["training_patch_library"]["sha256"]:
        raise ValueError("Frozen training patch library digest mismatch")
    config = dict(
        requests_sha256=digest(source),
        manifest_sha256=digest(MANIFEST),
        model=model_name,
        checkpoint=ckpt,
        batch_size=args.batch_size,
        device=args.device if ckpt else "cpu",
        noise_seed_rule="1000+batch_start",
        patch_library_sha256=digest(patchfile) if not ckpt else None,
        relaxation=dict(fmax=1e-6, max_steps=100000, all_atoms_free=True),
        failure_policy="nonfinite or unconverged outputs count as failed requests",
    )
    output.mkdir(parents=True, exist_ok=True)
    guard(output / "config.json", config)
    if (output / "complete.json").exists():
        return
    model = None
    if ckpt:
        model = load_denoiser(ckpt["path"], args.device)
        model.eval()
        if model_name != "sham_seed0":
            model = CFGDenoiser(model, 2)
    else:
        patches = PatchLibrary.load(patchfile)
    th = DefectThresholds.from_dict(manifest["thresholds"])
    n = len(d["pin_pos"])
    for start in range(0, n, args.batch_size):
        file = output / f"batch_{start:03d}.npz"
        if file.exists():
            arrays(file)  # CRC/decompression validation before reuse
            continue
        clock = time.monotonic()
        sl = slice(start, min(start + args.batch_size, n))
        host = structures(
            d, sl, key="pin_pos", dtype=torch.float32, device=args.device if ckpt else "cpu"
        )
        label = torch.as_tensor(d["label"][sl], dtype=torch.long, device=host.pos.device)
        if model_name == "sham_seed0":
            label = torch.where(
                label == LabelToken.UNLABELLED, label, torch.full_like(label, int(LabelToken.NULL))
            )
        req = Request(
            torch.as_tensor(d["target"][sl], device=host.pos.device),
            torch.full((host.batch_size,), 3, dtype=torch.long, device=host.pos.device),
            torch.as_tensor(d["pin_mask"][sl], device=host.pos.device),
            host.pos.clone(),
            label,
        )
        gen = seed_everything(1000 + start)
        with torch.no_grad():
            if ckpt:
                init = host_init(host, req, gen)
                result, _ = sample(
                    model,
                    init,
                    ScoreDynamicsSchedule(n_noisy=100, n_final=100, model_sigma_max=1.0),
                    PinnedLabel(),
                    req,
                    gen,
                )
            else:
                result = hand_insert(
                    host,
                    req.target,
                    [patches.sample(3, generator=gen) for _ in range(host.batch_size)],
                )
            cpu = Structures(result.pos.cpu().double(), host.types.cpu(), host.box.cpu().double())
            finite = torch.isfinite(cpu.pos).all(dim=(1, 2))
            positions = host.pos.cpu().double().clone()
            converged = torch.zeros(host.batch_size, dtype=torch.bool)
            max_force = torch.full((host.batch_size,), float("inf"), dtype=torch.float64)
            if finite.any():
                relaxed = fire_minimize(
                    Structures(cpu.pos[finite], cpu.types[finite], cpu.box[finite]),
                    fmax=1e-6,
                    max_steps=100000,
                )
                positions[finite] = relaxed.structures.pos
                converged[finite] = relaxed.converged
                max_force[finite] = relaxed.max_force
            rd = Request(
                req.target.cpu().double(),
                req.defect.cpu(),
                req.pin_mask.cpu(),
                req.pin_pos.cpu().double(),
                req.label.cpu(),
            )
            ok = success(Structures(positions, cpu.types, cpu.box), rd, th) & converged & finite
        save_npz(
            file,
            pos=positions.numpy(),
            pre_pos=cpu.pos.numpy(),
            types=cpu.types.numpy(),
            box=cpu.box.numpy(),
            target=rd.target.numpy(),
            host_pos=host.pos.cpu().numpy(),
            host_ids=d["host_ids"][sl],
            host_seeds=d["host_seeds"][sl],
            pin_mask=rd.pin_mask.numpy(),
            success=ok.numpy(),
            original_success=d["original_success"][sl],
            finite=finite.numpy(),
            converged=converged.numpy(),
            max_force=max_force.numpy(),
            seconds=np.full(host.batch_size, (time.monotonic() - clock) / host.batch_size),
        )
        print(
            f"{args.condition}/{model_name}: {sl.stop}/{n}; successes={int(ok.sum())}; batch seconds={time.monotonic() - clock:.1f}",
            flush=True,
        )
    batches = [arrays(output / f"batch_{start:03d}.npz") for start in range(0, n, args.batch_size)]
    save_npz(
        output / "samples.npz", **{k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    )
    save_json(
        output / "complete.json",
        dict(
            n_requests=n,
            successes=int(sum(b["success"].sum() for b in batches)),
            converged=int(sum(b["converged"].sum() for b in batches)),
        ),
    )


def host_bootstrap(values, ids, eligible, seed=93):
    groups = [
        (values[(ids == h) & eligible].sum(), int(((ids == h) & eligible).sum()))
        for h in np.unique(ids)
        if ((ids == h) & eligible).any()
    ]
    if not groups:
        return [None, None, None]
    g = np.asarray(groups)
    point = g[:, 0].sum() / g[:, 1].sum()
    draws = np.random.default_rng(seed).integers(len(g), size=(1000, len(g)))
    vals = g[draws, 0].sum(1) / g[draws, 1].sum(1)
    return [float(point), *map(float, np.quantile(vals, [0.025, 0.975]))]


def report(args, manifest):
    rows = []
    contrasts = []
    completed = []
    for cfg in manifest["dataset_configs"]:
        condition = Path(cfg).stem
        folder = args.out / condition
        if not all((folder / m / "complete.json").exists() for m in METHODS):
            continue
        data = {m: arrays(folder / m / "samples.npz") for m in METHODS}
        request = arrays(folder / "requests.npz")
        ids = request["host_ids"]
        for m, d in data.items():
            for key in ["host_ids", "host_seeds", "target", "original_success"]:
                if not np.array_equal(d[key], request[key]):
                    raise ValueError(f"Matched request disagreement: {condition}/{m}/{key}")
        for definition, eligible in [
            ("all_requests", np.ones(len(ids), dtype=bool)),
            ("initially_nonD2", ~request["original_success"]),
        ]:
            for method in METHODS + ["seed_mean"]:
                values = (
                    np.mean([data[m]["success"] for m in ["seed0", "seed1", "seed2"]], axis=0)
                    if method == "seed_mean"
                    else data[method]["success"].astype(float)
                )
                point, lo, hi = host_bootstrap(values, ids, eligible)
                rows.append(
                    dict(
                        condition=condition,
                        method=method,
                        definition=definition,
                        n_requests=int(eligible.sum()),
                        n_hosts=len(np.unique(ids[eligible])),
                        fraction=point,
                        lo=lo,
                        hi=hi,
                        converged=None
                        if method == "seed_mean"
                        else int(data[method]["converged"].sum()),
                        mean_seconds_per_request=None
                        if method == "seed_mean"
                        else float(data[method]["seconds"].mean()),
                    )
                )
            for method in ["seed0", "seed1", "seed2"]:
                for control in ["hand_insert", "sham_seed0"]:
                    diff = data[method]["success"].astype(float) - data[control]["success"].astype(
                        float
                    )
                    point, lo, hi = host_bootstrap(diff, ids, eligible)
                    contrasts.append(
                        dict(
                            condition=condition,
                            method=method,
                            control=control,
                            definition=definition,
                            difference=point,
                            lo=lo,
                            hi=hi,
                        )
                    )
        completed.append(condition)
    for name, content in [("placement_summary.csv", rows), ("paired_contrasts.csv", contrasts)]:
        if content:
            path = args.out / name
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(content[0]))
                writer.writeheader()
                writer.writerows(content)
            tmp.replace(path)
    save_json(
        args.out / "study_status.json",
        dict(
            complete=len(completed) == 6,
            completed_conditions=completed,
            expected_conditions=[Path(c).stem for c in manifest["dataset_configs"]],
            uncertainty="whole-host request-weighted bootstrap; seed_mean conditions on these three trained models",
            caveats=[
                "zero-event bootstrap intervals can be degenerate, not proof of zero probability",
                "seed0 sham is a same-checkpoint control only for seed0; other seeds use a cross-checkpoint control",
                "placement transfer only; structural equivalence and thermal transfer are not established",
            ],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hosts", "requests", "model", "insertion", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition")
    parser.add_argument(
        "--model", choices=["seed0", "seed1", "seed2", "sham_seed0"], default="seed0"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    choices = [Path(c).stem for c in manifest["dataset_configs"]]
    if args.mode != "report" and args.condition not in choices:
        parser.error("condition must be one of " + ", ".join(choices))
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch size must be positive")
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "hosts":
        hosts(args, manifest)
    elif args.mode == "requests":
        requests(args, manifest)
    elif args.mode in ["model", "insertion"]:
        evaluate(args, manifest)
    else:
        report(args, manifest)


if __name__ == "__main__":
    main()
