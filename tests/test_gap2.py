"""Resume integrity, host-cluster denominators and numerical failure accounting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gap2", ROOT / "scripts/run_gap2.py")
gap2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap2)


def test_host_bootstrap_preserves_requests_and_pairing():
    values = np.array([1.0, 0.0, 0.0])
    ids = np.array([7, 7, 9])
    estimate = gap2.host_bootstrap(values, ids, np.ones(3, dtype=bool))
    assert estimate[0] == pytest.approx(1 / 3)
    assert gap2.host_bootstrap(values - values, ids, np.ones(3, dtype=bool)) == [0.0, 0.0, 0.0]
    assert gap2.host_bootstrap(values, ids, np.zeros(3, dtype=bool)) == [None, None, None]


def test_resume_rejects_changed_configuration(tmp_path):
    file = tmp_path / "config.json"
    gap2.guard(file, dict(batch_size=4))
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        gap2.guard(file, dict(batch_size=8))
    assert json.loads(file.read_text()) == dict(batch_size=4)


def test_failed_generation_and_relaxation_remain_in_denominator(tmp_path, monkeypatch):
    args = SimpleNamespace(
        out=tmp_path, condition="test", mode="model", model="seed0", device="cpu", batch_size=2
    )
    folder = tmp_path / "test"
    folder.mkdir()
    with np.load(ROOT / "results/gap2_feasibility_20261018/gap2_ka2d_256_standard/host.npz") as f:
        positions = np.repeat(f["pos"][None].astype(np.float32), 2, axis=0)
        types = np.repeat(f["types"][None], 2, axis=0)
        box = np.repeat(f["box"][None].astype(np.float32), 2, axis=0)
    b = np.flatnonzero(types[0] == 1)[0]
    target = positions[:, b]
    labels = np.zeros(types.shape, dtype=np.int64)
    pin = np.zeros(types.shape, dtype=bool)
    pin[:, b] = True
    gap2.save_npz(
        folder / "requests.npz",
        pin_pos=positions,
        types=types,
        box=box,
        target=target,
        label=labels,
        pin_mask=pin,
        host_ids=np.array([0, 1]),
        host_seeds=np.array([100, 101]),
        original_success=np.zeros(2, dtype=bool),
    )
    ckpt = tmp_path / "weights"
    ckpt.write_bytes(b"test fixture")
    manifest = dict(
        checkpoints={"seed0": dict(path=str(ckpt), sha256=gap2.digest(ckpt))}, thresholds={}
    )
    manifest_file = tmp_path / "manifest.json"
    gap2.save_json(manifest_file, manifest)
    monkeypatch.setattr(gap2, "MANIFEST", manifest_file)
    monkeypatch.setattr(gap2, "load_denoiser", lambda *a: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(gap2, "CFGDenoiser", lambda model, w: model)

    def generated(model, init, schedule, strategy, req, gen):
        result = req.pin_pos.clone()
        result[1, 0, 0] = float("nan")
        return init.with_pos(result), None

    def failed_relaxation(s, **kw):
        assert s.batch_size == 1  # nonfinite generation must not enter FIRE
        return SimpleNamespace(
            structures=s, converged=torch.tensor([False]), max_force=torch.tensor([0.5])
        )

    monkeypatch.setattr(gap2, "sample", generated)
    monkeypatch.setattr(gap2, "fire_minimize", failed_relaxation)
    gap2.evaluate(args, manifest)
    data = gap2.arrays(folder / "seed0/samples.npz")
    assert len(data["success"]) == 2
    assert not data["success"].any()
    assert data["finite"].tolist() == [True, False]
    assert not data["converged"].any()
    assert json.loads((folder / "seed0/complete.json").read_text())["n_requests"] == 2


def test_partial_report_keeps_initially_nonD2_denominator(tmp_path):
    manifest = json.loads(gap2.MANIFEST.read_text())
    condition = "ka2d_256_standard"
    folder = tmp_path / condition
    request = dict(
        host_ids=np.array([0, 0, 1]),
        host_seeds=np.array([100, 100, 101]),
        target=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        original_success=np.array([True, False, False]),
    )
    gap2.save_npz(folder / "requests.npz", **request)
    for method in gap2.METHODS:
        gap2.save_npz(
            folder / method / "samples.npz",
            **request,
            success=np.array([True, True, False]),
            converged=np.ones(3, dtype=bool),
            seconds=np.ones(3),
        )
        gap2.save_json(folder / method / "complete.json", dict(n_requests=3))
    gap2.report(SimpleNamespace(out=tmp_path), manifest)
    import csv

    with (tmp_path / "placement_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["method"] == "seed0" and r["definition"] == "initially_nonD2")
    assert int(row["n_requests"]) == 2
    assert float(row["fraction"]) == 0.5
    assert not json.loads((tmp_path / "study_status.json").read_text())["complete"]
    changed = gap2.arrays(folder / "seed1/samples.npz")
    changed["target"][0, 0] += 0.1
    gap2.save_npz(folder / "seed1/samples.npz", **changed)
    with pytest.raises(ValueError, match="Matched request disagreement"):
        gap2.report(SimpleNamespace(out=tmp_path), manifest)


def test_notebook_cells_parse_without_saved_outputs():
    import ast

    notebook = json.loads((ROOT / "notebooks/kaggle_gap2_generalization.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
