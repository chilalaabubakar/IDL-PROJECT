"""Phase splitting preserves the frozen sampling/relaxation data flow and resume guards."""

import ast
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_gap2", ROOT / "scripts/run_gap2_colab.py")
colab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab)


def fixture_data(tmp_path, monkeypatch):
    manifest = json.loads(colab.base.MANIFEST.read_text())
    manifest["n_requests"] = 8
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fixture checkpoint")
    for key in manifest["checkpoints"]:
        manifest["checkpoints"][key] = dict(
            path=str(checkpoint), sha256=colab.base.digest(checkpoint)
        )
    meta = tmp_path / "manifest.json"
    colab.base.save_json(meta, manifest)
    monkeypatch.setattr(colab.base, "MANIFEST", meta)
    out = tmp_path / "split"
    condition = "ka2d_1024_standard"
    with np.load(ROOT / "results/gap2_feasibility_20261018/gap2_ka2d_256_standard/host.npz") as d:
        # Tiny request-count fixture; source geometry is real, not a performance result.
        colab.base.save_npz(
            out / condition / "hosts/samples.npz",
            pos=np.repeat(d["pos"][None], 2, axis=0),
            types=np.repeat(d["types"][None], 2, axis=0),
            box=np.repeat(d["box"][None], 2, axis=0),
            seeds=np.array([100, 101]),
        )
    args = SimpleNamespace(
        out=out, condition=condition, model="seed0", device="cpu", max_batches=None
    )
    colab.base.requests(args, manifest)
    return args, manifest


@pytest.mark.parametrize("method", ["seed0", "sham_seed0"])
def test_split_and_combined_results_match_and_partial_work_resumes(tmp_path, monkeypatch, method):
    args, manifest = fixture_data(tmp_path, monkeypatch)
    args.model = method
    counters = dict(sample=0, relax=0)
    monkeypatch.setattr(colab.base, "load_denoiser", lambda *a: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(colab.base, "CFGDenoiser", lambda model, w: model)

    def sampling(model, init, schedule, strategy, req, gen):
        counters["sample"] += 1
        assert schedule.n_noisy == 100 and schedule.n_final == 100 and schedule.model_sigma_max == 1
        assigned = req.label[req.pin_mask & (req.label != colab.base.LabelToken.UNLABELLED)]
        if method == "sham_seed0":
            assert (assigned == colab.base.LabelToken.NULL).all()
        pos = req.pin_pos.clone()
        pos[0, 0, 0] = float("nan")
        return init.with_pos(pos), None

    def relaxing(s, **kw):
        counters["relax"] += 1
        assert kw == dict(fmax=1e-6, max_steps=100000)
        converged = torch.ones(s.batch_size, dtype=torch.bool)
        converged[-1] = False
        force = torch.where(
            converged, torch.zeros(s.batch_size), torch.full((s.batch_size,), 0.5)
        ).double()
        return SimpleNamespace(structures=s, converged=converged, max_force=force)

    monkeypatch.setattr(colab.base, "sample", sampling)
    monkeypatch.setattr(colab.base, "fire_minimize", relaxing)
    args.max_batches = 1
    colab.sample_phase(args, manifest)
    assert counters == dict(sample=1, relax=0)  # GPU phase must never run CPU FIRE.
    output = args.out / args.condition / method
    assert not (output / "generated/complete.json").exists()
    args.max_batches = None
    colab.relax_phase(args, manifest)
    assert counters == dict(sample=1, relax=1)  # CPU phase must not resample.
    assert not (output / "complete.json").exists()
    colab.sample_phase(args, manifest)
    colab.relax_phase(args, manifest)
    assert counters == dict(sample=2, relax=2)
    assert json.loads((output / "complete.json").read_text())["n_requests"] == 8
    final_hash = colab.base.digest(output / "samples.npz")
    colab.relax_phase(args, manifest)
    assert colab.base.digest(output / "samples.npz") == final_hash
    assert counters == dict(sample=2, relax=2)

    combined = tmp_path / "combined"
    target = combined / args.condition
    target.mkdir(parents=True)
    shutil.copy2(args.out / args.condition / "requests.npz", target / "requests.npz")
    colab.base.evaluate(
        SimpleNamespace(
            out=combined,
            condition=args.condition,
            mode="model",
            model=method,
            device="cpu",
            batch_size=4,
        ),
        manifest,
    )
    split = colab.base.arrays(output / "samples.npz")
    original = colab.base.arrays(target / method / "samples.npz")
    for key in [
        "pos",
        "pre_pos",
        "types",
        "box",
        "target",
        "host_pos",
        "host_ids",
        "host_seeds",
        "pin_mask",
        "success",
        "original_success",
        "finite",
        "converged",
        "max_force",
    ]:
        np.testing.assert_equal(split[key], original[key])
    assert len(split["success"]) == 8
    assert not split["finite"][[0, 4]].any()
    assert not split["success"][[0, 4]].any()
    assert not split["success"][[3, 7]].any()  # nonconverged outputs also stay failed


def test_changed_generated_positions_cannot_reuse_relaxed_batch(tmp_path, monkeypatch):
    args, manifest = fixture_data(tmp_path, monkeypatch)
    monkeypatch.setattr(colab.base, "load_denoiser", lambda *a: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(colab.base, "CFGDenoiser", lambda model, w: model)
    monkeypatch.setattr(
        colab.base,
        "sample",
        lambda model, init, schedule, strategy, req, gen: (init.with_pos(req.pin_pos), None),
    )
    monkeypatch.setattr(
        colab.base,
        "fire_minimize",
        lambda s, **kw: SimpleNamespace(
            structures=s,
            converged=torch.ones(s.batch_size, dtype=torch.bool),
            max_force=torch.zeros(s.batch_size, dtype=torch.float64),
        ),
    )
    args.max_batches = 1
    colab.sample_phase(args, manifest)
    colab.relax_phase(args, manifest)
    file = args.out / args.condition / args.model / "generated/batch_000.npz"
    d = colab.base.arrays(file)
    d["pre_pos"][0, 0, 0] += 0.1
    colab.base.save_npz(file, **d)
    with pytest.raises(ValueError, match="changed after relaxation"):
        colab.relax_phase(args, manifest)


def notebook():
    return json.loads((ROOT / "notebooks/colab_gap2_1024.ipynb").read_text())


def test_notebook_defaults_to_cpu_preparation_and_excludes_kaggle_conditions():
    n = notebook()
    scope = {}
    exec("".join(n["cells"][1]["source"]), scope)
    assert scope["PHASE"] == "prepare"
    assert scope["CONDITIONS"] == ["ka2d_1024_standard", "ka2d_1024_fast"]
    assert not scope["ALLOW_GPU_FOR_CPU_PHASE"]
    assert scope["AUTO_DISCONNECT_GPU"]
    for c in n["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
            assert c["execution_count"] is None and c["outputs"] == []


def test_backup_skips_invalid_archives_and_restores_valid_files(tmp_path):
    source = "".join(notebook()["cells"][2]["source"])
    tree = ast.parse(source)
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ["validate_saved_file", "copy_valid_tree"]
    ]
    scope = dict(Path=Path, np=np, json=json, shutil=shutil, zipfile=zipfile)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<backup>", "exec"), scope)
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    np.savez_compressed(src / "valid.npz", x=np.arange(10))
    (src / "invalid.npz").write_bytes(b"incomplete archive")
    (src / "progress.json").write_text(json.dumps(dict(status="partial")))
    (src / "batch_004.tmp").write_bytes(b"in-progress write")
    scope["copy_valid_tree"](src, dest)
    assert not (dest / "invalid.npz").exists()
    assert not (dest / "batch_004.tmp").exists()
    with np.load(dest / "valid.npz") as d:
        np.testing.assert_equal(d["x"], np.arange(10))
    assert json.loads((dest / "progress.json").read_text()) == dict(status="partial")
    with pytest.raises(ValueError, match="must be separate"):
        scope["copy_valid_tree"](src, src / "nested")


@pytest.mark.parametrize(
    "phase,gpu,allow,message",
    [
        ("prepare", True, False, "Switch the accelerator"),
        ("relax", True, False, "Switch the accelerator"),
        ("sample", False, False, "Select a GPU runtime"),
        ("sample", True, False, None),
        ("relax", False, False, None),
        ("relax", True, True, None),
    ],
)
def test_cpu_gpu_phase_guard_protects_compute_unit_budget(phase, gpu, allow, message):
    source = "".join(notebook()["cells"][2]["source"])
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.If)
        and "torch.cuda.is_available" in ast.unparse(n.test)
        and any(isinstance(statement, ast.Raise) for statement in n.body)
    ]
    assert len(nodes) == 2
    scope = dict(
        PHASE=phase,
        ALLOW_GPU_FOR_CPU_PHASE=allow,
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: gpu)),
    )
    code = compile(ast.Module(body=nodes, type_ignores=[]), "<hardware guard>", "exec")
    if message:
        with pytest.raises(RuntimeError, match=message):
            exec(code, scope)
    else:
        exec(code, scope)
