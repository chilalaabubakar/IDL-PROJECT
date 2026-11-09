"""Standard point defects in amorphous silica (docs/SILICA_POINT_DEFECTS.md).

The design follows the 2D study (glassdiff.data.point_defects). Each defect is made by a
known operation on a known glass (an energy-minimised DM2 glass), and the defected glass is
minimised again, so the defect-free original is the exact target for repair.

    o_vacancy       remove a bridging O (bonded to two Si). The oxygen vacancy: its two Si
                    become 3-fold (the precursor of the E' centre) and may form a Si-Si bond.
    o_interstitial  add an O at the largest gap within 3 A of a random point. It usually
                    binds into the network, often as a peroxy linkage.
    o_frenkel       move a bridging O to the largest gap 4-6 A away: an oxygen vacancy plus
                    an interstitial O, the pair made by radiation damage (closer pairs
                    recombine when the glass is relaxed).
    si_vacancy      remove a 4-fold Si: its four O become non-bridging.

Bonds are Si-O pairs closer than 2.2 A (as glassdiff.silica.defects) and peroxy O-O pairs
closer than 1.7 A. Per-atom roles reuse glassdiff.data.point_defects.AtomRole:
NEAR_VACANCY for the atoms bonded to a removed or moved atom, INTERSTITIAL for an added O,
DISPLACED for a moved O. Positions are in Angstrom, unwrapped in the frame of the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from glassdiff.data.point_defects import AtomRole
from glassdiff.silica.defects import OXYGEN, SILICON


class SilicaPointDefect(IntEnum):
    O_VACANCY = 0
    O_INTERSTITIAL = 1
    O_FRENKEL = 2
    SI_VACANCY = 3


NAMES = {
    SilicaPointDefect.O_VACANCY: "o_vacancy",
    SilicaPointDefect.O_INTERSTITIAL: "o_interstitial",
    SilicaPointDefect.O_FRENKEL: "o_frenkel",
    SilicaPointDefect.SI_VACANCY: "si_vacancy",
}
IDEAL_CN = {SILICON: 4, OXYGEN: 2}


@dataclass
class SilicaDefectConfig:
    si_o_cutoff: float = 2.2  # A, Si-O bond
    peroxy_cutoff: float = 1.7  # A, O-O bond of a peroxy linkage
    interstitial_search: float = 3.0  # A, gap search around a random point
    frenkel_hop: tuple[float, float] = (4.0, 6.0)  # A, how far the O moves (closer pairs
    # recombine when relaxed: 3 of 4 at 2-4 A, none of 4 at 4-6 A)
    recombined_tol: float = 0.6  # A, a Frenkel O back this close to its site: no defect
    grid_spacing: float = 0.15  # A, gap search grid


def wrap(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    return v - box * np.round(v / box)


def gaps(
    pos: np.ndarray,
    box: np.ndarray,
    centre: np.ndarray,
    r_min: float,
    r_max: float,
    k: int = 1,
    min_sep: float = 1.0,
    exclude: int | None = None,
    spacing: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to k gaps r_min..r_max from ``centre``: grid points farthest from every atom (but
    ``exclude``), at least ``min_sep`` apart. Returns (points [k', 3], clearance [k'])."""
    u = np.arange(-r_max, r_max + spacing / 2, spacing)
    grid = np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(grid, axis=1)
    pts = centre + grid[(r >= r_min) & (r <= r_max)]
    others = pos if exclude is None else np.delete(pos, exclude, axis=0)
    tree = cKDTree(np.mod(others, box), boxsize=box)
    clearance, _ = tree.query(np.mod(pts, box))
    chosen, values = [], []
    for j in np.argsort(-clearance):
        if all(np.linalg.norm(wrap(pts[j] - c, box)) >= min_sep for c in chosen):
            chosen.append(pts[j])
            values.append(clearance[j])
            if len(chosen) == k:
                break
    return np.stack(chosen), np.asarray(values)


def coordination(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, cfg: SilicaDefectConfig
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """(Si-O coordination number per atom, Si-O neighbours per atom, peroxy O-O bonds per atom)."""
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs = tree.query_pairs(cfg.si_o_cutoff, output_type="ndarray")
    n = len(types)
    nb: list[list[int]] = [[] for _ in range(n)]
    peroxy = np.zeros(n, dtype=np.int64)
    for a, b in pairs:
        if types[a] != types[b]:
            nb[a].append(int(b))
            nb[b].append(int(a))
        elif types[a] == OXYGEN:
            d = np.linalg.norm(wrap(pos[a] - pos[b], box))
            if d < cfg.peroxy_cutoff:
                peroxy[a] += 1
                peroxy[b] += 1
    cn = np.array([len(x) for x in nb])
    return cn, [np.array(x, dtype=np.int64) for x in nb], peroxy


def coordination_states(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, cfg: SilicaDefectConfig
) -> np.ndarray:
    """[N] code of each atom's bonding state: species, Si-O coordination, peroxy bonds."""
    cn, _, peroxy = coordination(pos, types, box, cfg)
    return 100 * types + 10 * np.minimum(cn, 9) + np.minimum(peroxy, 9)


def rarity_table(states: list[np.ndarray]) -> dict[int, float]:
    """-log(frequency) of each bonding state in defect-free glasses (add-one smoothing); a
    state never seen gets the value of a state seen once less than the rarest."""
    codes, counts = np.unique(np.concatenate(states), return_counts=True)
    total = counts.sum() + len(codes) + 1
    table = {int(c): float(-np.log((n + 1) / total)) for c, n in zip(codes, counts)}
    table[-1] = float(-np.log(1 / total))  # unseen
    return table


def coordination_score(
    pos: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    cfg: SilicaDefectConfig,
    cluster: float = 3.0,
    rarity: dict[int, float] | None = None,
) -> np.ndarray:
    """The standard coordination analysis as a per-atom score (no potential needed).

    Each atom's deviation from ideal bonding: |n_O - 4| for Si, |n_Si - 2| + peroxy bonds for
    O. An introduced defect disturbs several neighbouring atoms, while most natural defects
    in these glasses are single, so the score adds the deviations of every atom within
    ``cluster`` A (the atom included), plus half its own: the top atom is a disturbed atom in
    the most disturbed cluster.

    With ``rarity`` (rarity_table of defect-free glasses), an atom off ideal bonding counts
    by how rare its bonding state is instead of by its deviation: a 3-fold Si or a peroxy O
    outranks a non-bridging O, which these glasses contain naturally.
    """
    cn, _, peroxy = coordination(pos, types, box, cfg)
    ideal = np.where(types == SILICON, IDEAL_CN[SILICON], IDEAL_CN[OXYGEN])
    dev = np.abs(cn - ideal) + np.where(types == OXYGEN, peroxy, 0)
    if rarity is not None:
        codes = 100 * types + 10 * np.minimum(cn, 9) + np.minimum(peroxy, 9)
        value = np.array([rarity.get(int(c), rarity[-1]) for c in codes])
        dev = np.where(dev > 0, value, 0.0)
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    nbrs = tree.query_ball_point(np.mod(pos, box), cluster)
    cluster_sum = np.array([dev[np.array(js)].sum() for js in nbrs])
    return cluster_sum + 0.5 * dev  # within a cluster, the defective atom itself first


def make_defect(
    kind: SilicaPointDefect,
    pos: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    rng: np.random.Generator,
    cfg: SilicaDefectConfig,
) -> dict:
    """Unrelaxed defected glass and what was done (one site)."""
    cn, nb, _ = coordination(pos, types, box, cfg)
    n = len(pos)
    bridging = np.nonzero((types == OXYGEN) & (cn == 2))[0]
    if kind == SilicaPointDefect.O_VACANCY:
        i = int(rng.choice(bridging))
        return dict(pos=np.delete(pos, i, 0), types=np.delete(types, i), atom=i,
                    site=pos[i].copy(), shell=nb[i])  # fmt: skip
    if kind == SilicaPointDefect.SI_VACANCY:
        si4 = np.nonzero((types == SILICON) & (cn == 4))[0]
        i = int(rng.choice(si4))
        return dict(pos=np.delete(pos, i, 0), types=np.delete(types, i), atom=i,
                    site=pos[i].copy(), shell=nb[i])  # fmt: skip
    if kind == SilicaPointDefect.O_INTERSTITIAL:
        centre = rng.uniform(0, 1, 3) * box
        (gap,), _ = gaps(pos, box, centre, 0.0, cfg.interstitial_search, spacing=cfg.grid_spacing)
        return dict(pos=np.vstack([pos, gap]), types=np.append(types, OXYGEN).astype(types.dtype),
                    atom=n, site=gap, shell=np.zeros(0, dtype=np.int64))  # fmt: skip
    if kind == SilicaPointDefect.O_FRENKEL:
        i = int(rng.choice(bridging))
        lo, hi = cfg.frenkel_hop
        (gap,), _ = gaps(pos, box, pos[i], lo, hi, exclude=i, spacing=cfg.grid_spacing)
        new = pos.copy()
        new[i] = gap
        return dict(pos=new, types=types.copy(), atom=i, site=pos[i].copy(), shell=nb[i])
    raise ValueError(f"unknown defect {kind!r}")


def finish_defect(
    kind: SilicaPointDefect,
    made: dict,
    relaxed: np.ndarray,
    original: np.ndarray,
    box: np.ndarray,
    cfg: SilicaDefectConfig,
) -> dict:
    """Roles, defect sites [2, 3] and validity, from the relaxed defected glass."""
    i = made["atom"]
    role = np.zeros(len(relaxed), dtype=np.int8)
    valid = True
    if kind in (SilicaPointDefect.O_VACANCY, SilicaPointDefect.SI_VACANCY):
        shell = made["shell"] - (made["shell"] > i)
        role[shell] = AtomRole.NEAR_VACANCY
        sites = np.stack([made["site"], made["site"]])
    elif kind == SilicaPointDefect.O_INTERSTITIAL:
        role[i] = AtomRole.INTERSTITIAL
        sites = np.stack([relaxed[i], relaxed[i]])
    else:
        role[made["shell"]] = AtomRole.NEAR_VACANCY
        role[i] = AtomRole.DISPLACED
        sites = np.stack([made["site"], relaxed[i]])
        valid = float(np.linalg.norm(wrap(relaxed[i] - original[i], box))) > cfg.recombined_tol
    if not valid:
        role[:] = AtomRole.INTACT
    return dict(role=role, sites=sites, valid=valid)


def same_glass(
    pos: np.ndarray,
    types: np.ndarray,
    orig: np.ndarray,
    otypes: np.ndarray,
    box: np.ndarray,
    tol: float = 0.1,
) -> bool:
    """Is pos the original glass, allowing identical atoms to trade places? Every original
    atom must have an atom of its species within ``tol`` A, one to one, after removing a
    global shift."""
    if len(pos) != len(orig) or not np.array_equal(np.bincount(types), np.bincount(otypes)):
        return False
    shift = np.zeros(3)
    for _ in range(2):
        ok, match, vecs = True, [], []
        for sp in (OXYGEN, SILICON):
            a, b = np.nonzero(otypes == sp)[0], np.nonzero(types == sp)[0]
            tree = cKDTree(np.mod(pos[b] - shift, box), boxsize=box)
            dist, j = tree.query(np.mod(orig[a], box))
            match.append(b[j])
            vecs.append(wrap(pos[b[j]] - shift - orig[a], box))
            ok &= bool((dist < tol).all())
        shift = shift + np.concatenate(vecs).mean(0)
    match = np.concatenate(match)
    return ok and len(np.unique(match)) == len(match)
