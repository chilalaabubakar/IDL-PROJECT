"""Run frozen gap-2 sampling on a GPU and its unchanged FIRE relaxation on a CPU.

python scripts/run_gap2_colab.py sample --out runs/colab_gap2 --condition ka2d_1024_standard --model seed0 --device cuda
python scripts/run_gap2_colab.py relax --out runs/colab_gap2 --condition ka2d_1024_standard --model seed0

The separation changes scheduling, not the checkpoint, request, noise, sampler,
precision, convergence threshold or failure denominator. Batch size remains four.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location("gap2_base", Path(__file__).with_name("run_gap2.py"))
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
CONDITIONS = ("ka2d_1024_standard", "ka2d_1024_fast")
MODELS = ("seed0", "seed1", "seed2", "sham_seed0")
BATCH = 4


def checkpoint(manifest, model):
    value = manifest["checkpoints"]["seed0" if model == "sham_seed0" else model]
    if base.digest(value["path"]) != value["sha256"]:
        raise ValueError("Frozen checkpoint digest mismatch")
    return value


def context(args, manifest):
    source = args.out / args.condition / "requests.npz"
    d = base.arrays(source)
    if len(d["pin_pos"]) != manifest["n_requests"]:
        raise ValueError("Unexpected request count")
    ckpt = checkpoint(manifest, args.model)
    output = args.out / args.condition / args.model
    generated = output / "generated"
    config = dict(
        requests_sha256=base.digest(source),
        manifest_sha256=base.digest(base.MANIFEST),
        checkpoint=ckpt,
        model=args.model,
        batch_size=BATCH,
        noise_seed_rule="1000+batch_start",
        sampling=dict(
            n_noisy=100,
            n_final=100,
            model_sigma_max=1.0,
            guidance_w=0 if args.model == "sham_seed0" else 2,
            radius=6,
        ),
        execution="separate-sampling-and-relaxation-v1",
    )
    return d, output, generated, config


def read_generated(file, d, start):
    g = base.arrays(file)
    end = min(start + BATCH, len(d["pin_pos"]))
    sl = slice(start, end)
    for key in ("types", "box", "target", "host_ids", "host_seeds", "pin_mask", "original_success"):
        if not np.array_equal(g[key], d[key][sl]):
            raise ValueError(f"Generated input mismatch: {file}/{key}")
    if not np.array_equal(g["host_pos"], d["pin_pos"][sl]):
        raise ValueError(f"Generated host mismatch: {file}")
    if g["pre_pos"].shape != d["pin_pos"][sl].shape:
        raise ValueError(f"Generated position shape mismatch: {file}")
    return g


def sample_phase(args, manifest):
    d, output, generated, config = context(args, manifest)
    generated.mkdir(parents=True, exist_ok=True)
    base.guard(generated / "config.json", config)
    n = len(d["pin_pos"])
    # Valid complete caches are reusable without loading a model or allocating CUDA.
    existing = []
    for start in range(0, n, BATCH):
        file = generated / f"batch_{start:03d}.npz"
        if file.exists():
            existing.append(read_generated(file, d, start))
    if len(existing) == len(range(0, n, BATCH)):
        base.save_json(generated / "complete.json", dict(n_requests=n))
        print(f"{args.condition}/{args.model}: sampling already complete", flush=True)
        return
    model = base.load_denoiser(config["checkpoint"]["path"], args.device)
    model.eval()
    if args.model != "sham_seed0":
        model = base.CFGDenoiser(model, 2)
    computed = 0
    for start in range(0, n, BATCH):
        file = generated / f"batch_{start:03d}.npz"
        if file.exists():
            continue
        sl = slice(start, min(start + BATCH, n))
        host = base.structures(d, sl, key="pin_pos", dtype=torch.float32, device=args.device)
        label = torch.as_tensor(d["label"][sl], dtype=torch.long, device=host.pos.device)
        if args.model == "sham_seed0":
            label = torch.where(
                label == base.LabelToken.UNLABELLED,
                label,
                torch.full_like(label, int(base.LabelToken.NULL)),
            )
        req = base.Request(
            torch.as_tensor(d["target"][sl], device=host.pos.device),
            torch.full((host.batch_size,), 3, dtype=torch.long, device=host.pos.device),
            torch.as_tensor(d["pin_mask"][sl], device=host.pos.device),
            host.pos.clone(),
            label,
        )
        gen = base.seed_everything(1000 + start)
        clock = time.monotonic()
        with torch.no_grad():
            init = base.host_init(host, req, gen)
            result, _ = base.sample(
                model,
                init,
                base.ScoreDynamicsSchedule(n_noisy=100, n_final=100, model_sigma_max=1.0),
                base.PinnedLabel(),
                req,
                gen,
            )
            # Match the combined runner's conversion before CPU relaxation.
            pre_pos = result.pos.cpu().double().numpy()
        elapsed = time.monotonic() - clock
        base.save_npz(
            file,
            pre_pos=pre_pos,
            types=host.types.cpu().numpy(),
            box=host.box.cpu().double().numpy(),
            target=req.target.cpu().double().numpy(),
            host_pos=host.pos.cpu().numpy(),
            host_ids=d["host_ids"][sl],
            host_seeds=d["host_seeds"][sl],
            pin_mask=req.pin_mask.cpu().numpy(),
            original_success=d["original_success"][sl],
            sampling_seconds=np.full(host.batch_size, elapsed / host.batch_size),
        )
        info = dict(
            n_requests=host.batch_size,
            seconds=elapsed,
            torch=torch.__version__,
            python=platform.python_version(),
            cuda=torch.version.cuda,
            device=args.device,
            gpu=torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else None,
            generated_sha256=base.digest(file),
        )
        base.save_json(generated / f"batch_{start:03d}.json", info)
        computed += 1
        print(
            f"{args.condition}/{args.model}: sampled through {sl.stop}/{n}; GPU/sample-stage seconds={elapsed:.1f}",
            flush=True,
        )
        if args.max_batches and computed >= args.max_batches:
            break
    count = sum(
        min(BATCH, n - s) for s in range(0, n, BATCH) if (generated / f"batch_{s:03d}.npz").exists()
    )
    base.save_json(
        generated / "progress.json", dict(n_generated=count, n_requests=n, complete=count == n)
    )
    if count == n:
        base.save_json(generated / "complete.json", dict(n_requests=n))


def relax_phase(args, manifest):
    d, output, generated, gen_config = context(args, manifest)
    if not (generated / "config.json").exists():
        raise ValueError("No generated batches: run sample phase first")
    if json.loads((generated / "config.json").read_text()) != gen_config:
        raise ValueError("Generated configuration changed")
    config = dict(
        requests_sha256=gen_config["requests_sha256"],
        manifest_sha256=gen_config["manifest_sha256"],
        model=args.model,
        checkpoint=gen_config["checkpoint"],
        batch_size=BATCH,
        device="cuda",
        noise_seed_rule="1000+batch_start",
        patch_library_sha256=None,
        relaxation=dict(fmax=1e-6, max_steps=100000, all_atoms_free=True),
        failure_policy="nonfinite or unconverged outputs count as failed requests",
        execution="separate-sampling-and-relaxation-v1",
        generation_config_sha256=base.digest(generated / "config.json"),
    )
    base.guard(output / "config.json", config)
    n = len(d["pin_pos"])
    th = base.DefectThresholds.from_dict(manifest["thresholds"])
    computed = 0
    for start in range(0, n, BATCH):
        raw = generated / f"batch_{start:03d}.npz"
        if not raw.exists():
            continue  # Partial GPU work may be relaxed, without claiming completion.
        g = read_generated(raw, d, start)
        file = output / f"batch_{start:03d}.npz"
        input_sha = base.digest(raw)
        if file.exists():
            done = base.arrays(file)
            if not np.all(done["input_generation_sha256"] == input_sha):
                raise ValueError("Generated positions changed after relaxation")
            continue
        clock = time.monotonic()
        size = len(g["pre_pos"])
        cpu = base.structures(g, slice(None), key="pre_pos")
        finite = torch.isfinite(cpu.pos).all(dim=(1, 2))
        positions = torch.from_numpy(g["host_pos"]).double().clone()
        converged = torch.zeros(size, dtype=torch.bool)
        max_force = torch.full((size,), float("inf"), dtype=torch.float64)
        with torch.no_grad():
            if finite.any():
                relaxed = base.fire_minimize(
                    base.Structures(cpu.pos[finite], cpu.types[finite], cpu.box[finite]),
                    fmax=1e-6,
                    max_steps=100000,
                )
                positions[finite] = relaxed.structures.pos
                converged[finite] = relaxed.converged
                max_force[finite] = relaxed.max_force
            sl = slice(start, start + size)
            rd = base.Request(
                torch.from_numpy(g["target"]).double(),
                torch.full((size,), 3, dtype=torch.long),
                torch.from_numpy(g["pin_mask"]),
                torch.from_numpy(g["host_pos"]).double(),
                torch.from_numpy(d["label"][sl]),
            )
            ok = (
                base.success(base.Structures(positions, cpu.types, cpu.box), rd, th)
                & finite
                & converged
            )
        seconds = np.full(size, (time.monotonic() - clock) / size)
        base.save_npz(
            file,
            pos=positions.numpy(),
            pre_pos=g["pre_pos"],
            types=g["types"],
            box=g["box"],
            target=g["target"],
            host_pos=g["host_pos"],
            host_ids=g["host_ids"],
            host_seeds=g["host_seeds"],
            pin_mask=g["pin_mask"],
            original_success=g["original_success"],
            success=ok.numpy(),
            finite=finite.numpy(),
            converged=converged.numpy(),
            max_force=max_force.numpy(),
            sampling_seconds=g["sampling_seconds"],
            relaxation_seconds=seconds,
            seconds=g["sampling_seconds"] + seconds,
            input_generation_sha256=np.full(size, input_sha),
        )
        base.save_json(
            output / f"relax_{start:03d}.json",
            dict(
                n_requests=size,
                seconds=float(seconds.sum()),
                torch=torch.__version__,
                python=platform.python_version(),
                device="cpu",
            ),
        )
        computed += 1
        print(
            f"{args.condition}/{args.model}: relaxed through {start + size}/{n}; successes={int(ok.sum())}; CPU seconds={seconds.sum():.1f}",
            flush=True,
        )
        if args.max_batches and computed >= args.max_batches:
            break
    files = [output / f"batch_{s:03d}.npz" for s in range(0, n, BATCH)]
    if all(f.exists() for f in files):
        data = [base.arrays(f) for f in files]
        final = {k: np.concatenate([b[k] for b in data]) for k in data[0]}
        if (output / "samples.npz").exists() and (output / "complete.json").exists():
            saved = base.arrays(output / "samples.npz")
            if set(saved) != set(final) or any(
                not np.array_equal(saved[key], final[key], equal_nan=final[key].dtype.kind in "fc")
                for key in final
            ):
                raise ValueError("Final aggregate differs from its checkpoint batches")
            return  # Preserve immutable completed output bytes across runtime changes.
        base.save_npz(output / "samples.npz", **final)
        base.save_json(
            output / "complete.json",
            dict(
                n_requests=n,
                successes=int(final["success"].sum()),
                converged=int(final["converged"].sum()),
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["sample", "relax"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int)
    args = p.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        p.error("max-batches must be positive")
    torch.set_num_threads(1)
    manifest = json.loads(base.MANIFEST.read_text())
    if args.phase == "sample":
        sample_phase(args, manifest)
    else:
        relax_phase(args, manifest)


if __name__ == "__main__":
    main()
