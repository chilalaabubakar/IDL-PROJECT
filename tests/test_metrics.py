"""Tests for Tickets E-4 (metrics) and B-2 (hand insertion)."""

import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.baselines import hand_insert
from glassdiff.data.dataset import GlassSplit
from glassdiff.data.patches import PatchLibrary
from glassdiff.eval.metrics import (
    QUANTILE_LEVELS,
    global_realism,
    local_features,
    local_realism,
    natural_windows,
    quantile_w1,
    success,
    wasserstein1,
)
from glassdiff.types import DefectClass, Request, Structures


def _vacancy(make_triangular):
    s = make_triangular(8, 8, a=1.1)
    keep = torch.tensor([k for k in range(64) if k != 4 * 8 + 3])
    return Structures(s.pos[:, keep], s.types[:, keep], s.box)


def _request(target, defect):
    return Request(
        target=target[None],
        defect=torch.tensor([int(defect)]),
        pin_mask=torch.zeros(1, 63, dtype=torch.bool),
        pin_pos=torch.zeros(1, 63, 2, dtype=torch.float64),
        label=torch.zeros(1, 63, dtype=torch.long),
    )


def test_success_requires_label_and_location(make_triangular):
    s = _vacancy(make_triangular)
    labels = detect_defects(s, DefectThresholds())[0]
    d1 = torch.nonzero(labels == DefectClass.D1_MINUS).flatten()
    plain = torch.nonzero(labels == DefectClass.NONE).flatten()
    th = DefectThresholds()
    assert success(s, _request(s.pos[0, d1[0]], DefectClass.D1_MINUS), th).item()
    assert not success(s, _request(s.pos[0, plain[0]], DefectClass.D1_MINUS), th).item()
    assert not success(s, _request(s.pos[0, d1[0]], DefectClass.D1_PLUS), th).item()


def test_local_features_and_identical_sets_have_zero_w1(make_lattice):
    s = make_lattice(4, 8, jitter=0.05, seed=0)
    centres = s.pos[:, 0]
    f = local_features(s, centres, None, r_loc=2.0)
    assert len(f) == 4 and all(len(p) > 0 for p in f.pe_atom)
    assert all(len(a) > 0 for a in f.bond_angle)
    res = local_realism(f, f, n_resamples=50)
    assert res["w1_pe_atom"][0] == 0.0 and res["w1_pair_AB"][0] == 0.0
    assert wasserstein1(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])) == 1.0
    g = global_realism(s, s)
    assert g["gr_l1_AA"] == 0.0 and g["w1_pe_atom"] == 0.0


def test_natural_windows(make_triangular):
    s = _vacancy(make_triangular)
    labels = detect_defects(s, DefectThresholds())
    split = GlassSplit(s, torch.zeros(1, 63, dtype=torch.float64), labels, torch.zeros(1))
    f = natural_windows(split, DefectClass.D1_MINUS, r_loc=2.0)
    assert len(f) == 6


def test_hand_insert_keeps_count_and_composition(make_triangular):
    s = _vacancy(make_triangular)
    labels = detect_defects(s, DefectThresholds())
    lib = PatchLibrary.build(s, labels)
    host = make_triangular(8, 8, a=1.1)
    host.types[0, ::3] = 1
    target = host.pos[:, 10].clone()
    patch = lib.patches[0]
    patch.types[:] = host.types[0, 10]  # make the patch species available in the host
    out = hand_insert(host, target, [patch])
    assert out.pos.shape == host.pos.shape and torch.equal(out.types, host.types)
    moved = (out.pos - host.pos).norm(dim=-1)[0] > 0
    assert int(moved.sum()) <= len(patch.types)
    assert torch.allclose(out.pos[0, 10], target[0])  # the centre atom lands on the target


def test_quantile_w1_matches_exact_w1():
    import numpy as np

    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 100_000)
    ref_q = np.quantile(ref, QUANTILE_LEVELS)
    for shift, n in ((0.0, 7000), (0.1, 7000), (0.5, 500)):
        x = rng.normal(shift, 1.2, n)
        exact = wasserstein1(torch.from_numpy(x), torch.from_numpy(ref))
        assert abs(quantile_w1(x, ref_q) - exact) < 0.01 * exact + 2e-3
