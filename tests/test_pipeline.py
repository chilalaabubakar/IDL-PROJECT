"""End-to-end pipeline test on a tiny synthetic dataset (no LAMMPS needed).

freeze_thresholds -> build_patches -> defect_stability -> train -> sample -> relax -> evaluate -> aggregate,
each through its command-line entry point, as a guard against interface drift between
scripts.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch

from glassdiff.data.dataset import GlassSplit, save_split
from glassdiff.physics.fire import fire_minimize
from glassdiff.physics.ka_potential import ka_energy

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [script, *args])
    module.main()


def _fake_dataset(data: Path, make_lattice) -> None:
    for i, (name, n) in enumerate((("train", 6), ("val", 2), ("test", 4))):
        s = make_lattice(n, 8, jitter=0.12, seed=10 + i)
        relaxed = fire_minimize(s, fmax=1e-6).structures.wrapped()
        split = GlassSplit(relaxed, ka_energy(relaxed, per_atom=True), None, torch.arange(n))
        save_split(data / f"{name}.npz", split)
    (data / "meta.json").write_text(json.dumps({"name": "tiny"}))


def test_pipeline_end_to_end(tmp_path, make_lattice, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _fake_dataset(data, make_lattice)
    defects = str(ROOT / "configs/defects/v0.yaml")
    # a jittered square lattice has many "defects" by the glass definitions: fine for a test
    _run("freeze_thresholds", "--config", defects, f"data={data}", monkeypatch=monkeypatch)
    _run("build_patches", "--config", defects, f"data={data}", monkeypatch=monkeypatch)
    meta = json.loads((data / "meta.json").read_text())
    assert "digest" in meta["defects"]
    diag = tmp_path / "diagnostics"
    _run(
        "defect_stability",
        "--config",
        str(ROOT / "configs/eval/default.yaml"),
        f"data={data}",
        f"runs_root={diag}",
        "sigmas=[0,0.05]",
        "threads=1",
        "bootstrap.n_resamples=20",
        monkeypatch=monkeypatch,
    )
    stability = json.loads(next(diag.glob("*_defect_stability/metrics.json")).read_text())
    unperturbed = [v for v in stability["sigmas"]["0"].values() if isinstance(v, dict)]
    assert set(stability["sigmas"]) == {"0", "0.05"} and unperturbed
    assert all(v["survival"][0] == 1.0 for v in unperturbed if v["survival"])  # relaxed already

    common = [f"data={data}", f"runs_root={runs}", "threads=1"]
    _run(
        "train",
        "--config",
        str(ROOT / "configs/train/cond.yaml"),
        *common,
        "model_args.hidden=16",
        "model_args.n_layers=1",
        "model_args.cutoff=2.0",
        "optim.batch_size=2",
        "optim.n_updates=3",
        "val_every=3",
        "noise.sigma_max=0.8",
        monkeypatch=monkeypatch,
    )
    ckpt = next(runs.glob("*_egnn_cond/ckpt.pt"))
    _run(
        "sample",
        "--config",
        str(ROOT / "configs/sampler/cfg.yaml"),
        *common,
        f"ckpt={ckpt}",
        "defect=D2",
        "request.mode=host",
        "request.host_r_out=2.5",
        "n_samples=2",
        "batch_size=2",
        "schedule.n_noisy=4",
        "schedule.n_final=2",
        "strategy.guidance_w=1.0",
        monkeypatch=monkeypatch,
    )
    for cfg_name, mode in (("clamp", "patch"), ("dm2", "centre")):
        _run(
            "sample",
            "--config",
            str(ROOT / f"configs/sampler/{cfg_name}.yaml"),
            *common,
            f"ckpt={ckpt}",
            "defect=D2",
            f"request.mode={mode}",
            "n_samples=2",
            "batch_size=2",
            "schedule.n_noisy=3",
            "schedule.n_final=1",
            monkeypatch=monkeypatch,
        )
    assert next(runs.glob("*_sample_clamp_patch_D2")) and next(
        runs.glob("*_sample_unconditional_centre_D2")
    )
    run = next(runs.glob("*_sample_pinned_label_host_D2"))
    # the sampler clips sigma at the checkpoint's training sigma_max unless told otherwise
    assert json.loads((run / "timing.json").read_text())["model_sigma_max"] == 0.8
    eval_cfg = str(ROOT / "configs/eval/default.yaml")
    _run("relax", "--config", eval_cfg, f"run={run}", "relax.fmax=1e-3", monkeypatch=monkeypatch)
    _run(
        "evaluate",
        "--config",
        eval_cfg,
        f"run={run}",
        f"data={data}",
        "bootstrap.n_resamples_w1=10",
        "local.r_loc=2.0",
        monkeypatch=monkeypatch,
    )
    metrics = json.loads((run / "metrics.json").read_text())
    assert len(metrics["success_after_relax"]) == 3 and "w1_pe_atom" in metrics["local"]
    for name in ("floor_D2", "floor_tiny_D2"):  # default name and a named floor
        _run(
            "evaluate",
            "--config",
            eval_cfg,
            "floor=true",
            "defect=D2",
            f"data={data}",
            f"runs_root={runs}",
            f"name={name}",
            "bootstrap.n_resamples_w1=10",
            "local.r_loc=2.0",
            monkeypatch=monkeypatch,
        )
    out = tmp_path / "summary.csv"
    _run(
        "aggregate_results",
        "--config",
        eval_cfg,
        f"runs_dir={runs}",
        f"out={out}",
        monkeypatch=monkeypatch,
    )
    with out.open() as f:
        methods = {row["method"] for row in csv.DictReader(f)}
    assert {"floor", "floor_tiny_D2", run.name.split("_", 1)[1]} <= methods


def test_training_resumes_after_interruption(tmp_path, make_lattice, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _fake_dataset(data, make_lattice)
    args = [
        "--config",
        str(ROOT / "configs/train/uncond.yaml"),
        f"data={data}",
        f"runs_root={runs}",
        "threads=1",
        "model_args.hidden=16",
        "model_args.n_layers=1",
        "model_args.cutoff=2.0",
        "optim.batch_size=2",
        "optim.n_updates=4",
        "val_every=100",
    ]
    # a session that is "disconnected" right after its first update
    _run("train", *args, "max_minutes=1e-6", monkeypatch=monkeypatch)
    run = next(runs.glob("*_egnn_uncond"))
    assert torch.load(run / "ckpt.pt", weights_only=False)["step"] == 1
    _run("train", *args, f"resume={run}", "max_minutes=0", monkeypatch=monkeypatch)
    ckpt = torch.load(run / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 4 and ckpt["cfg"]["optim"]["n_updates"] == 4
    steps = [line.split(",")[0] for line in (run / "log.csv").read_text().splitlines()[1:]]
    assert steps == ["1", "4"]  # one header, then one row per session end
    assert not (run / "ckpt.pt.tmp").exists()
