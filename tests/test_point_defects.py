"""Point-defect data, models, training and evaluation (docs/POINT_DEFECTS.md)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from glassdiff.data.dataset import GlassSplit, save_split
from glassdiff.data.point_defects import (
    NAMES,
    AtomRole,
    DefectConfig,
    PointDefect,
    candidate_gaps,
    finish_defect,
    first_shell,
    generate,
    heal_target,
    largest_gap,
    load_point_defects,
    make_defect,
    return_displaced_atom,
    save_point_defects,
    wrap,
)
from glassdiff.models.egnn_pbc import EGNNAtomClassifier
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize, fire_minimize_many
from glassdiff.types import Structures

ROOT = Path(__file__).resolve().parents[1]
ROT90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_relax(jobs):
    return [(np.asarray(p, dtype=np.float64), 0.0) for p, _, _ in jobs]


def fire_relax(jobs):
    return fire_minimize_many(jobs, workers=1)


@pytest.fixture
def glass(make_lattice):
    s = make_lattice(1, 8, jitter=0.05, seed=0)  # 64 atoms, box 7.3
    return s.pos[0].numpy(), s.types[0].numpy(), s.box[0].numpy()


def test_largest_gap_finds_an_empty_site(glass):
    pos, _, box = glass
    gap = largest_gap(np.delete(pos, 10, 0), box, pos[10] + 0.3, 0.0, 1.0, spacing=0.02)
    assert np.linalg.norm(wrap(gap - pos[10], box)) < 0.15
    ring = largest_gap(pos, box, pos[3], 1.2, 2.0, exclude=3)
    assert 1.2 - 1e-9 <= np.linalg.norm(ring - pos[3]) <= 2.0 + 1e-9


def test_first_shell_matches_brute_force(glass):
    pos, types, box = glass
    shell = first_shell(pos, types, box, 5, 1.4)
    sig = np.array([[1.0, 0.8], [0.8, 0.88]])
    want = [
        j
        for j in range(len(pos))
        if j != 5 and np.linalg.norm(wrap(pos[j] - pos[5], box)) < 1.4 * sig[types[5], types[j]]
    ]
    assert shell.tolist() == want and 3 <= len(want) <= 9


@pytest.mark.parametrize("kind", list(PointDefect))
def test_defect_operations_and_repair_input(kind, glass):
    """Without relaxation, the repair input is the original glass (exactly, or up to the
    grid spacing where the vacancy's atom is put back), and the roles mark the defect."""
    pos, types, box = glass
    cfg = DefectConfig()
    made = make_defect(kind, pos, types, box, np.random.default_rng(1), cfg)
    n = len(pos)
    assert len(made["pos"]) == n + {0: -1, 1: 1, 2: 0, 3: 0}[int(kind)]
    fin = finish_defect(kind, made, made["pos"], pos, types, box, cfg)
    role = fin["role"]
    shift = heal_target(
        torch.tensor(fin["heal_pos"])[None], torch.tensor(pos)[None], torch.tensor(box)[None]
    )[0]
    if kind == PointDefect.VACANCY:
        assert (role == AtomRole.NEAR_VACANCY).sum() == len(made["shell"]) > 0
        assert shift.norm(dim=-1).max() < 0.1  # atom put back in the empty site
    elif kind == PointDefect.INTERSTITIAL:
        assert role[-1] == AtomRole.INTERSTITIAL and (role[:-1] == 0).all()
        assert made["types"][-1] == cfg.interstitial_species
        assert shift.abs().max() < 1e-12
    elif kind == PointDefect.FRENKEL:
        i = made["atom"]
        hop = np.linalg.norm(wrap(made["pos"][i] - pos[i], box))
        assert cfg.frenkel_hop[0] - 1e-9 <= hop <= cfg.frenkel_hop[1] + 1e-9
        assert role[i] == AtomRole.DISPLACED and (role == AtomRole.NEAR_VACANCY).sum() > 0
        # only the moved atom must go back (less the small mean shift)
        assert shift[i].norm() > 1.0 and (np.delete(shift.numpy(), i, 0) ** 2).sum(-1).max() < 0.01
    else:
        a, b = made["atom"], made["atom2"]
        assert types[a] == 0 and types[b] == 1
        assert made["types"][a] == 1 and made["types"][b] == 0
        assert (made["types"] != types).sum() == 2
        assert set(np.nonzero(role)[0]) == {a, b}
        assert shift.abs().max() < 1e-12
    assert fin["valid"]


def test_heal_target_wraps_and_removes_the_mean():
    box = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
    orig = torch.tensor([[[0.1, 0.1], [2.0, 2.0]]], dtype=torch.float64)
    moved = orig + torch.tensor([[[5.0 - 0.2, 0.0], [0.4, 0.0]]], dtype=torch.float64)
    t = heal_target(moved, orig, box)
    assert torch.allclose(t, torch.tensor([[[-0.3, 0.0], [0.3, 0.0]]], dtype=torch.float64))


def test_candidate_gaps_are_distinct_and_most_open_first(glass):
    pos, _, box = glass
    gaps = candidate_gaps(pos, box, pos[7], 1.0, 3.0, k=6, min_sep=0.6, exclude=7)
    assert 1 <= len(gaps) <= 6
    r = np.linalg.norm(wrap(gaps - pos[7], box), axis=1)
    assert ((r >= 1.0 - 1e-9) & (r <= 3.0 + 1e-9)).all()
    sep = np.linalg.norm(wrap(gaps[:, None] - gaps[None], box), axis=-1)
    assert (sep[~np.eye(len(gaps), dtype=bool)] >= 0.6 - 1e-9).all()
    clearance = [np.linalg.norm(wrap(np.delete(pos, 7, 0) - g, box), axis=1).min() for g in gaps]
    assert clearance == sorted(clearance, reverse=True)


def _write_dataset(root: Path, make_lattice, n_glasses=2, per_glass=2, relax=identity_relax):
    """Source splits and point-defect splits as scripts/make_point_defects.py writes them."""
    cfg = DefectConfig()
    for i, split in enumerate(("train", "val", "test")):
        s = make_lattice(n_glasses, 8, jitter=0.05, seed=10 + i)
        relaxed = fire_minimize(s, fmax=1e-6).structures
        src = root / "source" / f"{split}.npz"
        g = GlassSplit(
            relaxed,
            torch.zeros(s.pos.shape[:2], dtype=torch.float64),
            None,
            torch.arange(n_glasses),
        )
        save_split(src, g)
        pos, types, box = relaxed.pos.numpy(), relaxed.types.numpy(), relaxed.box.numpy()
        examples = {}
        for kind in PointDefect:
            rng = np.random.default_rng([0, i, int(kind)])
            examples[NAMES[kind]] = generate(kind, pos, types, box, per_glass, rng, cfg, relax)
        save_point_defects(root / "pd" / f"{split}.npz", examples, src)
    return root / "pd"


@pytest.mark.slow
def test_generate_save_and_load_round_trip(tmp_path, make_lattice):
    pd = _write_dataset(tmp_path, make_lattice, relax=fire_relax)
    data = load_point_defects(pd / "test.npz")
    assert set(data) == set(NAMES.values())
    for name, d in data.items():
        n = {"vacancy": 63, "interstitial": 65}.get(name, 64)
        assert d.pos.shape == (4, n, 2) and d.heal_pos.shape == (4, 64, 2)
        assert d.role.shape == (4, n) and (d.role > 0).any(-1).all()
        assert d.valid.all()
    fr = data["frenkel"]
    back = return_displaced_atom(fr)
    i = torch.arange(len(fr))
    near_site = wrap((back.heal_pos[i, fr.atom] - fr.sites[:, 0]).numpy(), fr.box.numpy())
    assert (np.linalg.norm(near_site, axis=1) <= 1.0 + 1e-9).all()
    others = torch.ones(fr.role.shape, dtype=torch.bool)
    others[i, fr.atom] = False  # every atom but the displaced one stays put
    assert torch.equal(back.heal_pos[others], fr.pos[others])
    with pytest.raises(ValueError):
        return_displaced_atom(data["vacancy"])
    # the stored source fingerprint is checked
    other = tmp_path / "source" / "val.npz"
    with pytest.raises(ValueError):
        load_point_defects(pd / "test.npz", source=other)


def test_classifier_symmetries(make_lattice):
    torch.manual_seed(0)
    model = EGNNAtomClassifier(hidden=32, n_layers=2, k_max=48).double().eval()
    s = make_lattice(2, 8, jitter=0.15, seed=0)
    out = model(s)
    assert out.shape == (2, 64, 5) and out.std() > 0
    assert torch.allclose(model(s.with_pos(s.pos + torch.tensor([0.4, -1.3]).double())), out)
    assert torch.allclose(model(s.with_pos(s.pos @ ROT90.T)), out, atol=1e-9)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    out_p = model(Structures(s.pos[:, perm], s.types[:, perm], s.box))
    assert torch.allclose(out_p, out[:, perm], atol=1e-9)


def test_fire_minimize_many_matches_single_runs(make_lattice):
    s = make_lattice(2, 8, jitter=0.12, seed=3)
    jobs = [(s.pos[b].numpy(), s.types[b].numpy(), s.box[b].numpy()) for b in range(2)]
    for workers in (1, 2):
        out = fire_minimize_many(jobs, workers=workers)
        for b, (p, fmax) in enumerate(out):
            ref = fire_minimize(Structures(s.pos[b : b + 1], s.types[b : b + 1], s.box[b : b + 1]))
            assert np.allclose(p, ref.structures.pos[0].numpy()) and fmax < 1e-6


def test_same_structure_allows_identical_atoms_to_swap(glass):
    ev = _script("eval_point_defects")
    pos, types, box = glass
    assert ev.same_structure(pos + 0.01, types, pos, types, box)
    a = np.nonzero(types == 0)[0][:2]
    perm = np.arange(len(pos))
    perm[a] = a[::-1]
    assert ev.same_structure(pos[perm], types[perm], pos, types, box)
    moved = pos.copy()
    moved[0] += 0.5
    assert not ev.same_structure(moved, types, pos, types, box)
    b = np.nonzero(types == 1)[0][0]
    swapped = types.copy()
    swapped[a[0]], swapped[b] = 1, 0
    assert not ev.same_structure(pos, swapped, pos, types, box)


def _train(tmp_path, monkeypatch, pd, task, config=None):
    config = config or f"{task}_point_defects"
    cfg = yaml.safe_load((ROOT / f"configs/train/{config}.yaml").read_text())
    model_cfg = cfg["model"] if isinstance(cfg["model"], dict) else str(ROOT / cfg["model"])
    cfg.update(
        data=str(pd),
        model=model_cfg,
        init_from=None,
        runs_root=str(tmp_path / "runs"),
        device="cpu",
        n_val=2,
        val_every=2,
        val_sigma=[0.001],
        model_args=dict(hidden=16, n_layers=1, k_max=48),
    )
    cfg["optim"].update(n_updates=2, batch_size=2)
    path = tmp_path / f"{config}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    _script("train_point_defects").main()
    (run,) = [p for p in (tmp_path / "runs").iterdir() if p.name.endswith(cfg["name"])]
    rows = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    return run, rows


@pytest.mark.slow
def test_training_and_evaluation_smoke(tmp_path, monkeypatch, make_lattice):
    pd = _write_dataset(tmp_path, make_lattice, relax=fire_relax)
    heal_run, rows = _train(tmp_path, monkeypatch, pd, "heal")
    assert rows[-1]["step"] == 2 and "frenkel_worst_s0.001" in rows[-1]
    assert "clean_rms_s0.001" in rows[-1]
    heal2_run, rows = _train(tmp_path, monkeypatch, pd, "heal", "heal_point_defects_v2")
    assert "frenkel_defect_rms_s0.001" in rows[-1] and "frenkel_none_defect_rms_s0.001" in rows[-1]
    detect_run, rows = _train(tmp_path, monkeypatch, pd, "detect")
    assert "vacancy_hit" in rows[-1] and "clean_false_alarm" in rows[-1]
    assert load_denoiser(detect_run / "ckpt.pt").n_classes == 5

    out = tmp_path / "eval.json"
    argv = ["eval", "--data", str(pd / "test.npz"), "--healer", str(heal_run / "ckpt.pt")]
    argv += ["--detector", str(detect_run / "ckpt.pt"), "--workers", "1", "--threads", "1"]
    monkeypatch.setattr(sys, "argv", argv + ["--max-examples", "2", "--out", str(out)])
    _script("eval_point_defects").main()
    res = json.loads(out.read_text())["kinds"]
    assert set(res) == set(NAMES.values())
    for name, r in res.items():
        assert 0 <= r["find"]["model"] <= 1 and r["find"]["model_score"] == "detector"
        assert r["repair"]
    assert "flag by model, pick by model: move, heal, minimise" in res["frenkel"]["repair"]
    assert (
        "flag by energy, pick by energy: lowest of all minimised tries" in res["frenkel"]["repair"]
    )
    # without a finder the repair model's own output picks the gap
    argv = ["eval", "--data", str(pd / "test.npz"), "--healer", str(heal2_run / "ckpt.pt")]
    argv += ["--workers", "1", "--threads", "1", "--kinds", "frenkel", "--max-examples", "2"]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(out)])
    _script("eval_point_defects").main()
    fr = json.loads(out.read_text())["kinds"]["frenkel"]
    assert fr["find"]["model_score"] == "healer_output"
    assert "flag by model, pick by model: the right gap" in fr["repair"]

    # repair without a potential, the defect given
    rep = tmp_path / "repair.json"
    argv = ["repair", "--data", str(pd / "test.npz"), "--model", f"a={heal_run / 'ckpt.pt'}"]
    argv += ["--model", f"b={heal2_run / 'ckpt.pt'}", "--max-examples", "2", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv + ["--threads", "1", "--out", str(rep)])
    _script("eval_point_defect_repair").main()
    res = json.loads(rep.read_text())["kinds"]
    assert set(res) == set(NAMES.values())
    for r in res.values():
        assert {"nothing", "a, one step", "b, 10 steps", "minimise (needs the potential)"} <= set(r)
        assert 0 <= r["minimise (needs the potential)"]["restored"] <= 1
        assert r["nothing"]["overlap"] in (0.0, 0.5, 1.0)


@pytest.mark.lammps
def test_lammps_minimize_matches_fire(glass):
    pytest.importorskip("lammps")
    from glassdiff.md.minimize import lammps_minimize, minimize_many

    pos, types, box = glass
    made = make_defect(
        PointDefect.FRENKEL, pos, types, box, np.random.default_rng(2), DefectConfig()
    )
    try:
        out, fmax = lammps_minimize(made["pos"], types, box)
    except OSError as exc:  # the LAMMPS library cannot be loaded here
        pytest.skip(str(exc))
    ref = fire_minimize_many([(made["pos"], types, box)], workers=1)[0][0]
    d = wrap(out - ref, box)
    assert fmax < 1e-6 and np.linalg.norm(d - d.mean(0), axis=1).max() < 1e-3
    assert np.abs(out - made["pos"]).max() < 3.0  # returned in the input frame, not wrapped
    again = minimize_many([(made["pos"], types, box)] * 2, workers=2)
    assert np.allclose(again[0][0], out) and np.allclose(again[1][0], out)
