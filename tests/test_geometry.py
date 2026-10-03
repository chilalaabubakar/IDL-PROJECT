import pytest
import torch

from glassdiff.geometry import minimum_image, neighbor_list, pair_vectors


def _neighbor_sets(nl):
    return [
        [frozenset(nl.idx[b, i][nl.mask[b, i]].tolist()) for i in range(nl.idx.shape[1])]
        for b in range(nl.idx.shape[0])
    ]


def test_minimum_image_range(make_random):
    s = make_random(2, 50, seed=1)
    v = pair_vectors(s.pos * 3.7 - 5.0, s.box)  # positions far outside the box
    half = s.box[:, None, None, :] / 2
    assert (v.abs() <= half + 1e-9).all()


def test_distance_across_boundary():
    box = torch.tensor([[10.0, 10.0]])
    vec = torch.tensor([[[9.8, -9.7]]])
    assert torch.allclose(minimum_image(vec, box), torch.tensor([[[-0.2, 0.3]]]), atol=1e-6)


def test_neighbor_list_matches_bruteforce(make_random):
    s = make_random(2, 64, seed=2)
    nl = neighbor_list(s.pos, s.box, cutoff=2.0, k_max=40)
    d = pair_vectors(s.pos, s.box).norm(dim=-1)
    for b in range(2):
        for i in range(64):
            expect = {j for j in range(64) if j != i and d[b, i, j] <= 2.0}
            got = set(nl.idx[b, i][nl.mask[b, i]].tolist())
            assert got == expect
    assert torch.allclose(nl.dist[nl.mask], nl.vec[nl.mask].norm(dim=-1))
    assert (nl.vec[~nl.mask] == 0).all() and (nl.dist[~nl.mask] == 0).all()


def test_neighbor_list_translation_invariant(make_random):
    s = make_random(1, 64, seed=3)
    nl = neighbor_list(s.pos, s.box, cutoff=2.5, k_max=40)
    for shift in (torch.tensor([0.37, -1.9]), s.box[0] * torch.tensor([2.0, -3.0])):
        nl2 = neighbor_list(s.pos + shift, s.box, cutoff=2.5, k_max=40)
        assert _neighbor_sets(nl) == _neighbor_sets(nl2)
        assert torch.allclose(nl.dist, nl2.dist, atol=1e-9)


def test_neighbor_list_permutation_equivariant(make_random):
    s = make_random(1, 40, seed=4)
    perm = torch.randperm(40, generator=torch.Generator().manual_seed(0))
    nl = neighbor_list(s.pos, s.box, cutoff=2.0, k_max=30)
    nl_p = neighbor_list(s.pos[:, perm], s.box, cutoff=2.0, k_max=30)
    sets = _neighbor_sets(nl)[0]
    sets_p = _neighbor_sets(nl_p)[0]
    for new_i, old_i in enumerate(perm.tolist()):
        assert {int(perm[j]) for j in sets_p[new_i]} == set(sets[old_i])


def test_neighbor_list_overflow_and_cutoff_checks(make_random):
    s = make_random(1, 64, seed=5)
    with pytest.raises(ValueError, match="k_max"):
        neighbor_list(s.pos, s.box, cutoff=3.0, k_max=4)
    with pytest.raises(ValueError, match="half the smallest box side"):
        neighbor_list(s.pos, s.box, cutoff=float(s.box.min()) / 2, k_max=63)
