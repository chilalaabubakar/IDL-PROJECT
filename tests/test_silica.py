"""3D silica pipeline: periodic graph, defect detectors, requests, model symmetries and a
training smoke run. Model tests need the optional ``.[silica]`` dependencies."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from glassdiff.silica.data import SilicaSplit, load_split, save_split
from glassdiff.silica.defects import SilicaDefect, SilicaThresholds, detect, label_glasses
from glassdiff.silica.graph import batched_radius_graph, minimum_image, radius_graph_pbc
from glassdiff.silica.requests import (
    FIRST_DEFECT_TOKEN,
    NULL,
    UNLABELLED,
    SilicaRequestConfig,
    SilicaRequestSampler,
)

ROOT = Path(__file__).resolve().parents[1]
O, SI = 0, 1  # noqa: E741


def toy_glass(n: int = 96, length: float = 12.0, seed: int = 0):
    """Random O/Si positions (1:2) in a cube; not a glass, but enough for shapes."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * length
    types = torch.zeros(n, dtype=torch.long)
    types[: n // 3] = SI
    return pos, types, torch.full((3,), length, dtype=torch.float64)


# ---------------------------------------------------------------- periodic neighbour graph


def test_radius_graph_matches_brute_force():
    pos, _, box = toy_glass(80, 11.0)
    cutoff = 4.0
    edges, vec = radius_graph_pbc(pos, box, cutoff, chunk=17)
    d = minimum_image(pos[None] - pos[:, None], box)  # d[i, j] = r_j - r_i
    r = d.norm(dim=-1)
    r.fill_diagonal_(float("inf"))
    want = set(map(tuple, torch.nonzero(r < cutoff).tolist()))
    got = set(map(tuple, edges.t().tolist()))
    assert got == want
    assert torch.allclose(vec, d[edges[0], edges[1]])


def test_batched_radius_graph_offsets():
    a, _, box = toy_glass(30, 10.0, seed=1)
    b, _, _ = toy_glass(40, 10.0, seed=2)
    pos = torch.cat([a, b])
    batch = torch.cat([torch.zeros(30), torch.ones(40)]).long()
    edges, _ = batched_radius_graph(pos, torch.stack([box, box]), batch, 4.0)
    assert (batch[edges[0]] == batch[edges[1]]).all()
    n_a = radius_graph_pbc(a, box, 4.0)[0].shape[1]
    n_b = radius_graph_pbc(b, box, 4.0)[0].shape[1]
    assert edges.shape[1] == n_a + n_b


def test_radius_graph_rejects_large_cutoff():
    pos, _, box = toy_glass(10, 8.0)
    with pytest.raises(ValueError):
        radius_graph_pbc(pos, box, 4.5)


# ---------------------------------------------------------------- defect detectors


def _tetrahedron(centre, r=1.6):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
    return np.asarray(centre) + r * v


def test_detect_coordination_defects():
    box = np.full(3, 30.0)
    si4 = np.array([5.0, 5.0, 5.0])
    si3 = np.array([15.0, 15.0, 15.0])
    pos = np.vstack([si4, _tetrahedron(si4), si3, _tetrahedron(si3)[:3]])
    types = np.array([SI, O, O, O, O, SI, O, O, O])
    labels = detect(pos, types, box)
    assert labels[0] == SilicaDefect.NONE  # four-fold Si
    assert (labels[1:5] == SilicaDefect.NBO).all()  # each O has one Si
    assert labels[5] == SilicaDefect.SI_III


def test_detect_rings_and_tricluster():
    box = np.full(3, 40.0)
    # 2-ring: two Si sharing two O
    ring2 = [[5, 5, 5], [7.5, 5, 5], [6.25, 6.0, 5], [6.25, 4.0, 5]]
    # 3-ring: three Si, one bridging O per edge, pushed outwards
    c = np.array([20.0, 20.0, 20.0])
    ang = np.deg2rad([90, 210, 330])
    si = c + 1.732 * np.stack([np.cos(ang), np.sin(ang), np.zeros(3)], 1)
    mids = [(si[i] + si[(i + 1) % 3]) / 2 for i in range(3)]
    bridges = [m + 0.9 * (m - c) / np.linalg.norm(m - c) for m in mids]
    # tricluster: one O bonded to three Si that share nothing else
    o3 = np.array([30.0, 30.0, 30.0])
    tri = o3 + 1.6 * np.array([[1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]])
    pos = np.vstack([ring2, si, bridges, [o3], tri])
    types = np.array([SI, SI, O, O, SI, SI, SI, O, O, O, O, SI, SI, SI])
    labels = detect(pos, types, box)
    assert (labels[[0, 1]] == SilicaDefect.RING2).all()
    assert (labels[[4, 5, 6]] == SilicaDefect.RING3).all()
    assert labels[10] == SilicaDefect.O_III
    assert (labels[[11, 12, 13]] != SilicaDefect.RING3).all()  # one shared O is not a ring


def test_thresholds_digest_changes_with_cutoff():
    assert SilicaThresholds().digest() != SilicaThresholds(si_o_cutoff=2.3).digest()


# ---------------------------------------------------------------- requests


def test_request_modes_and_labels():
    pos, types, box = toy_glass(90, 12.0)
    labels = torch.from_numpy(
        label_glasses(pos[None].numpy(), types[None].numpy(), box[None].numpy())
    )
    labels[0, 7] = SilicaDefect.NBO  # make sure a defect exists
    b = 6
    pos_b, box_b, lab_b = pos[None].repeat(b, 1, 1), box[None].repeat(b, 1), labels.repeat(b, 1)
    g = torch.Generator().manual_seed(0)
    cfg = SilicaRequestConfig(mode_probs={"host": 1.0}, p_uncond=0.0, r_out_range=(4.0, 5.0))
    req = SilicaRequestSampler(cfg)(pos_b, box_b, lab_b, generator=g)
    assert req.pin.shape == req.label.shape == (b, 90)
    idx = torch.arange(b)
    assert req.pin[idx, req.centre].all()
    dist = minimum_image(pos_b - pos_b[idx, req.centre][:, None], box_b[:, None]).norm(dim=-1)
    assert (req.pin == ((dist > req.r_out[:, None]) | (dist == 0))).all()
    assert (req.label[idx, req.centre] == req.defect + FIRST_DEFECT_TOKEN).all()
    assert ((req.label == UNLABELLED).sum(-1) == 89).all()
    cfg = SilicaRequestConfig(mode_probs={"centre": 1.0}, p_uncond=1.0)
    req = SilicaRequestSampler(cfg)(pos_b, box_b, lab_b, generator=g)
    assert (req.pin.sum(-1) == 1).all()
    assert (req.label[idx, req.centre] == NULL).all()


def test_split_roundtrip(tmp_path):
    pos, types, box = toy_glass(30, 10.0)
    split = SilicaSplit(
        pos[None].float(), types[None], box[None].float(), torch.tensor([10.0]), types[None]
    )
    save_split(tmp_path / "s.npz", split)
    back = load_split(tmp_path / "s.npz")
    assert torch.equal(back.pos, split.pos) and torch.equal(back.labels, split.labels)
    assert back.density().shape == (1,)


# ---------------------------------------------------------------- model (optional deps)


@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("e3nn")
    pytest.importorskip("graphite")
    from glassdiff.silica.model import ConditionalNequIP

    torch.manual_seed(0)
    model = ConditionalNequIP(hidden_irreps="8x0e + 4x1e", n_convs=2, cutoff=4.0).double()
    return model.eval()


def _cluster(n=24, seed=0):
    """Atoms in the middle of a large box, so no periodic image is within the cutoff."""
    pos, types, _ = toy_glass(n, 6.0, seed)
    box = torch.full((1, 3), 30.0, dtype=torch.float64)
    return pos + 12.0, types, box


def _call(model, pos, types, box, pin=None, label=None):
    batch = torch.zeros(len(types), dtype=torch.long)
    sigma = torch.tensor([0.3], dtype=torch.float64)
    rate = torch.tensor([10.0], dtype=torch.float64)
    return model(pos, types, box, batch, sigma, pin, label, rate)


def test_model_rotation_equivariance(tiny_model):
    pos, types, box = _cluster()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    centre = box[0] / 2
    out = _call(tiny_model, pos, types, box)
    out_rot = _call(tiny_model, (pos - centre) @ q.T + centre, types, box)
    assert torch.allclose(out_rot, out @ q.T, atol=1e-8)


def test_model_translation_and_permutation(tiny_model):
    pos, types, _ = toy_glass(60, 10.0, seed=3)
    box = torch.full((1, 3), 10.0, dtype=torch.float64)
    out = _call(tiny_model, pos, types, box)
    shifted = torch.remainder(pos + torch.tensor([3.3, -1.2, 7.7], dtype=torch.float64), box[0])
    assert torch.allclose(_call(tiny_model, shifted, types, box), out, atol=1e-8)
    perm = torch.randperm(60)
    assert torch.allclose(_call(tiny_model, pos[perm], types[perm], box), out[perm], atol=1e-8)


def test_model_pinned_atoms_get_zero(tiny_model):
    pos, types, box = _cluster()
    pin = torch.zeros(len(types), dtype=torch.bool)
    pin[:5] = True
    label = torch.full_like(types, UNLABELLED)
    label[0] = FIRST_DEFECT_TOKEN + SilicaDefect.SI_III
    out = _call(tiny_model, pos, types, box, pin, label)
    assert (out[:5] == 0).all() and out[5:].abs().sum() > 0


# ---------------------------------------------------------------- training smoke run


def test_train_script_smoke(tmp_path, monkeypatch):
    pytest.importorskip("e3nn")
    pytest.importorskip("graphite")
    data = tmp_path / "data"
    glasses = [toy_glass(48, 11.0, seed=s) for s in range(3)]
    for name, idx in (("train", [0, 1]), ("val", [2])):
        pos = torch.stack([glasses[i][0] for i in idx]).float()
        types = torch.stack([glasses[i][1] for i in idx])
        box = torch.stack([glasses[i][2] for i in idx]).float()
        labels = torch.from_numpy(label_glasses(pos.numpy(), types.numpy(), box.numpy()))
        save_split(
            data / f"{name}.npz",
            SilicaSplit(pos, types, box, torch.tensor([10.0] * len(idx)), labels),
        )
    digest = SilicaThresholds().digest()
    (data / "meta.json").write_text(json.dumps({"labels": {"digest": digest}}))

    spec = importlib.util.spec_from_file_location("silica_train", ROOT / "scripts/silica/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    args = [
        "--config", str(ROOT / "configs/silica/train_cond_dm2.yaml"),
        f"data={data}", f"run_dir={run}", "device=cpu", "n_updates=2",
        "optim.batch_size=2", "optim.accumulate=1", "val_every=2", "val_batches=1",
        "model.hidden_irreps=8x0e + 4x1e", "model.n_convs=1",
        f"defects={ROOT / 'configs/silica/defects.yaml'}",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train.py", *args])
    module.main()
    ckpt = torch.load(run / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 2
    lines = (run / "log.csv").read_text().splitlines()
    assert lines[0].startswith("step,seconds") and len(lines) == 2
    # the same command continues the run
    monkeypatch.setattr(sys, "argv", ["train.py", *args[:4], "device=cpu", "n_updates=3"])
    module.main()
    assert torch.load(run / "ckpt.pt", weights_only=False)["step"] == 3
