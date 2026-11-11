"""Standard point defects in 2D Kob-Andersen glasses: vacancy, interstitial, Frenkel pair and
antisite (docs/POINT_DEFECTS.md).

Each defect is made by a known operation on a known defect-free glass (an inherent structure
of data/ka2d_256), and the defected glass is then energy-minimised, so it sits in a minimum
like a real defected sample. Because the operation is known, so is the defect-free original:
the target for repair, with no detection thresholds.

    vacancy       remove one atom (any species)
    interstitial  add one atom (B, the small species, by default) at the largest gap within
                  2 sigma of a random point
    Frenkel pair  move one atom to the largest gap 1.2-3 sigma from its site
    antisite      swap the species of an A atom and its nearest B atom

Every example stores:
    pos, types   relaxed defected glass (N = 255, 257, 256, 256)
    role         per-atom AtomRole, the target for detection
    sites        [2, 2] where the defect is (the empty site, the extra or displaced atom, the
                 two swapped atoms; the second row repeats the first for one-site defects)
    heal_pos     [256, 2] input for repair: the relaxed defected glass with the atom count and
                 species put right (vacancy: an atom of the missing species in the largest gap
                 within 1 sigma of the empty site; interstitial: the extra atom removed;
                 antisite: species swapped back; Frenkel pair: unchanged). Atom i of heal_pos
                 is atom i of the original glass, whose species it has.
Positions are unwrapped in the frame of the original glass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from glassdiff.physics.ka_potential import KA_SIGMA


class PointDefect(IntEnum):
    VACANCY = 0
    INTERSTITIAL = 1
    FRENKEL = 2
    ANTISITE = 3


NAMES = {
    PointDefect.VACANCY: "vacancy",
    PointDefect.INTERSTITIAL: "interstitial",
    PointDefect.FRENKEL: "frenkel",
    PointDefect.ANTISITE: "antisite",
}
N_CHANGE = {
    PointDefect.VACANCY: -1,
    PointDefect.INTERSTITIAL: 1,
    PointDefect.FRENKEL: 0,
    PointDefect.ANTISITE: 0,
}


class AtomRole(IntEnum):
    """Per-atom detection labels."""

    INTACT = 0
    NEAR_VACANCY = 1  # first-shell neighbour of an empty site (vacancy or Frenkel pair)
    INTERSTITIAL = 2  # the extra atom
    DISPLACED = 3  # the atom moved off its site (Frenkel pair)
    WRONG_SPECIES = 4  # one of the two swapped atoms (antisite)


N_ROLES = len(AtomRole)


@dataclass
class DefectConfig:
    sites_per_glass: dict[str, int] = field(
        default_factory=lambda: {"train": 8, "val": 2, "test": 2}
    )
    kinds: tuple[str, ...] = ("vacancy", "interstitial", "frenkel", "antisite")
    interstitial_species: int = 1  # B
    interstitial_search: float = 2.0  # radius around a random point
    frenkel_hop: tuple[float, float] = (1.2, 3.0)
    vacancy_insert_search: float = 1.0  # repair input: gap search radius around the site
    shell_factor: float = 1.4  # first shell: r < shell_factor * sigma_ab (as the CN rule)
    grid_spacing: float = 0.05
    recombined_tol: float = 0.3  # Frenkel atom back within this of its site -> no defect
    min_pair_dist: float = 0.56  # dataset quality gate (0.7 sigma_AB)
    ftol: float = 1e-8  # LAMMPS minimize force tolerance, as the dataset
    seed: int = 0


def wrap(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image vectors for a [2] box."""
    return v - box * np.round(v / box)


def largest_gap(
    pos: np.ndarray,
    box: np.ndarray,
    centre: np.ndarray,
    r_min: float,
    r_max: float,
    exclude: int | None = None,
    spacing: float = 0.05,
) -> np.ndarray:
    """The grid point r_min..r_max from ``centre`` farthest from every atom (but ``exclude``)."""
    u = np.arange(-r_max, r_max + spacing / 2, spacing)
    grid = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
    r = np.linalg.norm(grid, axis=1)
    pts = centre + grid[(r >= r_min) & (r <= r_max)]
    others = pos if exclude is None else np.delete(pos, exclude, axis=0)
    d = np.linalg.norm(wrap(pts[:, None, :] - others[None], box), axis=-1).min(1)
    return pts[int(d.argmax())]


def first_shell(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, i: int, factor: float = 1.4
) -> np.ndarray:
    """Indices of atoms within factor * sigma_ab of atom i (i itself excluded)."""
    sig = np.asarray(KA_SIGMA)[types[i], types]
    d = np.linalg.norm(wrap(pos - pos[i], box), axis=1)
    near = d < factor * sig
    near[i] = False
    return np.nonzero(near)[0]


def candidate_gaps(
    pos: np.ndarray,
    box: np.ndarray,
    centre: np.ndarray,
    r_min: float,
    r_max: float,
    k: int = 8,
    min_sep: float = 0.6,
    exclude: int | None = None,
    spacing: float = 0.05,
) -> np.ndarray:
    """Up to k distinct gaps r_min..r_max from ``centre``, the most open first: grid points
    far from every atom (but ``exclude``), at least ``min_sep`` apart. [k', 2]."""
    u = np.arange(-r_max, r_max + spacing / 2, spacing)
    grid = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
    r = np.linalg.norm(grid, axis=1)
    pts = centre + grid[(r >= r_min) & (r <= r_max)]
    others = pos if exclude is None else np.delete(pos, exclude, axis=0)
    clearance = np.linalg.norm(wrap(pts[:, None, :] - others[None], box), axis=-1).min(1)
    chosen: list[np.ndarray] = []
    for j in np.argsort(-clearance):
        if all(np.linalg.norm(wrap(pts[j] - c, box)) >= min_sep for c in chosen):
            chosen.append(pts[j])
            if len(chosen) == k:
                break
    return np.stack(chosen)


def min_pair_distance(pos: np.ndarray, box: np.ndarray) -> float:
    d = np.linalg.norm(wrap(pos[:, None] - pos[None], box), axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def make_defect(
    kind: PointDefect,
    pos: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    rng: np.random.Generator,
    cfg: DefectConfig,
) -> dict:
    """Unrelaxed defected glass and what was done (one site)."""
    n = len(pos)
    if kind == PointDefect.VACANCY:
        i = int(rng.integers(n))
        shell = first_shell(pos, types, box, i, cfg.shell_factor)
        return dict(
            pos=np.delete(pos, i, 0),
            types=np.delete(types, i),
            atom=i,
            atom2=-1,
            site=pos[i].copy(),
            shell=shell,
        )
    if kind == PointDefect.INTERSTITIAL:
        centre = rng.uniform(0, 1, 2) * box
        gap = largest_gap(pos, box, centre, 0.0, cfg.interstitial_search, spacing=cfg.grid_spacing)
        return dict(
            pos=np.vstack([pos, gap]),
            types=np.append(types, cfg.interstitial_species).astype(types.dtype),
            atom=n,
            atom2=-1,
            site=gap,
            shell=np.zeros(0, dtype=np.int64),
        )
    if kind == PointDefect.FRENKEL:
        i = int(rng.integers(n))
        lo, hi = cfg.frenkel_hop
        gap = largest_gap(pos, box, pos[i], lo, hi, exclude=i, spacing=cfg.grid_spacing)
        new = pos.copy()
        new[i] = gap
        shell = first_shell(pos, types, box, i, cfg.shell_factor)
        return dict(pos=new, types=types.copy(), atom=i, atom2=-1, site=pos[i].copy(), shell=shell)
    if kind == PointDefect.ANTISITE:
        a_atoms = np.nonzero(types == 0)[0]
        a = int(a_atoms[rng.integers(len(a_atoms))])
        d = np.linalg.norm(wrap(pos - pos[a], box), axis=1)
        d[types != 1] = np.inf
        b = int(d.argmin())
        new_types = types.copy()
        new_types[a], new_types[b] = types[b], types[a]
        return dict(
            pos=pos.copy(),
            types=new_types,
            atom=a,
            atom2=b,
            site=pos[a].copy(),
            shell=np.zeros(0, dtype=np.int64),
        )
    raise ValueError(f"unknown defect {kind!r}")


def finish_defect(
    kind: PointDefect,
    made: dict,
    relaxed: np.ndarray,
    original: np.ndarray,
    original_types: np.ndarray,
    box: np.ndarray,
    cfg: DefectConfig,
) -> dict:
    """Roles, defect sites and the repair input, from the relaxed defected glass."""
    n0 = len(original)
    role = np.zeros(len(relaxed), dtype=np.int8)
    i, j = made["atom"], made["atom2"]
    valid = True
    if kind == PointDefect.VACANCY:
        shell = made["shell"] - (made["shell"] > i)  # indices after removing atom i
        role[shell] = AtomRole.NEAR_VACANCY
        sites = np.stack([made["site"], made["site"]])
        gap = largest_gap(
            relaxed, box, made["site"], 0.0, cfg.vacancy_insert_search, spacing=cfg.grid_spacing
        )
        heal = np.insert(relaxed, i, gap, axis=0)
    elif kind == PointDefect.INTERSTITIAL:
        role[n0] = AtomRole.INTERSTITIAL
        sites = np.stack([relaxed[n0], relaxed[n0]])
        heal = relaxed[:n0].copy()
    elif kind == PointDefect.FRENKEL:
        role[made["shell"]] = AtomRole.NEAR_VACANCY
        role[i] = AtomRole.DISPLACED
        sites = np.stack([made["site"], relaxed[i]])
        heal = relaxed.copy()
        valid = float(np.linalg.norm(wrap(relaxed[i] - original[i], box))) > cfg.recombined_tol
    else:
        role[[i, j]] = AtomRole.WRONG_SPECIES
        sites = np.stack([relaxed[i], relaxed[j]])
        heal = relaxed.copy()
    if not valid:
        role[:] = AtomRole.INTACT
    return dict(role=role, sites=sites, heal_pos=heal, valid=valid)


# ------------------------------------------------------------------------- generation


def generate(
    kind: PointDefect,
    pos: np.ndarray,  # [M, N, 2] original inherent structures
    types: np.ndarray,  # [M, N]
    box: np.ndarray,  # [M, 2]
    n_per_glass: int,
    rng: np.random.Generator,
    cfg: DefectConfig,
    relax,  # callable: list of (pos, types, box) -> list of (relaxed pos, max force)
) -> dict[str, np.ndarray]:
    """All examples of one defect kind for one split."""
    made, owner = [], []
    for g in range(len(pos)):
        for _ in range(n_per_glass):
            made.append(make_defect(kind, pos[g], types[g], box[g], rng, cfg))
            owner.append(g)
    results = relax([(m["pos"], m["types"], box[g]) for m, g in zip(made, owner)])
    rows = []
    for m, g, (relaxed, fmax) in zip(made, owner, results):
        fin = finish_defect(kind, m, relaxed, pos[g], types[g], box[g], cfg)
        ok_dist = min_pair_distance(relaxed, box[g]) >= cfg.min_pair_dist
        rows.append(
            dict(m, **fin, relaxed=relaxed, fmax=fmax, glass=g, ok=fin["valid"] and ok_dist)
        )
    return dict(
        glass=np.array([r["glass"] for r in rows], dtype=np.int32),
        atom=np.array([r["atom"] for r in rows], dtype=np.int32),
        atom2=np.array([r["atom2"] for r in rows], dtype=np.int32),
        sites=np.stack([r["sites"] for r in rows]).astype(np.float64),
        pos=np.stack([r["relaxed"] for r in rows]).astype(np.float32),
        types=np.stack([r["types"] for r in rows]).astype(np.int8),
        role=np.stack([r["role"] for r in rows]).astype(np.int8),
        heal_pos=np.stack([r["heal_pos"] for r in rows]).astype(np.float32),
        fmax=np.array([r["fmax"] for r in rows], dtype=np.float64),
        valid=np.array([r["ok"] for r in rows], dtype=bool),
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_point_defects(
    path: str | Path, examples: dict[str, dict[str, np.ndarray]], source: str | Path, **extra
) -> dict:
    """Write one split: ``examples[kind name]`` as returned by generate(), made from the
    original split file ``source``; ``extra`` goes into the stored info record (which is
    returned). The file is byte-for-byte reproducible: same content, same file."""
    path, source = Path(path), Path(source)
    info = dict(source=str(source), source_sha256=file_sha256(source), **extra)
    info.setdefault("kinds", list(examples))
    arrays = {f"{name}_{k}": v for name, ex in examples.items() for k, v in ex.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, info=json.dumps(info), **arrays)
    tmp.replace(path)
    return info


# ------------------------------------------------------------------------- loading


@dataclass
class PointDefectSet:
    """All examples of one defect kind in one split (as tensors), with the originals."""

    kind: PointDefect
    glass: Tensor  # [E] long, index into the source split
    atom: Tensor  # [E] long
    atom2: Tensor  # [E] long, -1 unless antisite
    sites: Tensor  # [E, 2, 2]
    pos: Tensor  # [E, N_k, 2]
    types: Tensor  # [E, N_k] long
    role: Tensor  # [E, N_k] long
    heal_pos: Tensor  # [E, N0, 2]
    valid: Tensor  # [E] bool
    orig_pos: Tensor  # [E, N0, 2] the defect-free original of each example
    orig_types: Tensor  # [E, N0]
    box: Tensor  # [E, 2]

    def __len__(self) -> int:
        return len(self.glass)

    def subset(self, idx) -> PointDefectSet:
        return PointDefectSet(self.kind, *(getattr(self, f)[idx] for f in _TENSOR_FIELDS))


_TENSOR_FIELDS = (
    "glass", "atom", "atom2", "sites", "pos", "types", "role", "heal_pos", "valid",
    "orig_pos", "orig_types", "box",
)  # fmt: skip


def load_point_defects(
    path: str | Path, source: str | Path | None = None, valid_only: bool = True
) -> dict[str, PointDefectSet]:
    """Load one split written by scripts/make_point_defects.py.

    ``source`` is the original split (data/ka2d_256/<split>.npz); by default the path stored
    in the file. Its SHA256 must match the one recorded at generation time.
    """
    from glassdiff.data.dataset import load_split

    path = Path(path)
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    info = json.loads(str(arrays.pop("info")))
    src = Path(source) if source is not None else Path(info["source"])
    if file_sha256(src) != info["source_sha256"]:
        raise ValueError(f"{src} is not the split these defects were made from")
    split = load_split(src)
    s = split.structures
    out = {}
    for name in info["kinds"]:
        kind = next(k for k, v in NAMES.items() if v == name)
        a = {k[len(name) + 1 :]: v for k, v in arrays.items() if k.startswith(name + "_")}
        glass = torch.from_numpy(a["glass"].astype(np.int64))
        d = PointDefectSet(
            kind=kind,
            glass=glass,
            atom=torch.from_numpy(a["atom"].astype(np.int64)),
            atom2=torch.from_numpy(a["atom2"].astype(np.int64)),
            sites=torch.from_numpy(a["sites"]),
            pos=torch.from_numpy(a["pos"].astype(np.float64)),
            types=torch.from_numpy(a["types"].astype(np.int64)),
            role=torch.from_numpy(a["role"].astype(np.int64)),
            heal_pos=torch.from_numpy(a["heal_pos"].astype(np.float64)),
            valid=torch.from_numpy(a["valid"]),
            orig_pos=s.pos[glass],
            orig_types=s.types[glass],
            box=s.box[glass],
        )
        out[name] = d.subset(d.valid) if valid_only else d
    return out


def return_displaced_atom(d: PointDefectSet, search: float = 1.0) -> PointDefectSet:
    """Frenkel pairs with a different repair input: the displaced atom put back into the
    largest gap within ``search`` of its empty site, instead of the relaxed defected glass.

    Repair then means cleaning up the surroundings, not learning a jump of 1.2-3 sigma; which
    gap the atom goes back to is a separate choice (scripts/eval_point_defects.py)."""
    if d.kind != PointDefect.FRENKEL:
        raise ValueError("only Frenkel pairs have a displaced atom")
    heal = d.pos.numpy().copy()
    sites, box = d.sites.numpy(), d.box.numpy()
    for e, i in enumerate(d.atom.tolist()):
        heal[e, i] = largest_gap(heal[e], box[e], sites[e, 0], 0.0, search, exclude=i)
    return replace(d, heal_pos=torch.from_numpy(heal))


def same_structure(pos, types, orig, otypes, box, tol: float = 0.05) -> bool:
    """Is pos the original glass, allowing identical atoms to trade places? Every original
    atom must have an atom of its species within ``tol``, one to one, after a global shift
    is removed."""
    if len(pos) != len(orig) or not np.array_equal(np.bincount(types), np.bincount(otypes)):
        return False
    shift = np.zeros(2)
    for _ in range(2):  # estimate and remove a global shift, then match
        ok, match, vecs = True, [], []
        for sp in (0, 1):
            a, b = np.nonzero(otypes == sp)[0], np.nonzero(types == sp)[0]
            d = wrap(pos[b][None] - shift - orig[a][:, None], box)
            r = np.linalg.norm(d, axis=-1)
            j = r.argmin(1)
            match.append(b[j])
            vecs.append(d[np.arange(len(a)), j])
            ok &= bool((r.min(1) < tol).all())
        shift = shift + np.concatenate(vecs).mean(0)
    match = np.concatenate(match)
    return ok and len(np.unique(match)) == len(match)


def already_original(d: PointDefectSet, tol: float = 0.05) -> Tensor:
    """[E] bool: the relaxed "defected" glass is the original glass again.

    This happens to Frenkel pairs: the moved atom takes the site of a neighbour of its own
    species and that neighbour fills the empty site (recombination by exchange, as in a
    replacement sequence). The moved atom is then far from its own site, so ``valid`` (which
    only asks whether it went back there) is true, but no defect is left: the structure and
    its energy are those of the original. Such examples must not be scored as defects.
    """
    pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
    orig, otypes = d.orig_pos.numpy(), d.orig_types.numpy()
    same = [
        same_structure(pos[e], types[e], orig[e], otypes[e], box[e], tol) for e in range(len(d))
    ]
    return torch.tensor(same, dtype=torch.bool)


def real_defects(data: dict[str, PointDefectSet], tol: float = 0.05) -> dict[str, PointDefectSet]:
    """Every kind without the examples that are the original glass again (already_original)."""
    return {name: d.subset(~already_original(d, tol)) for name, d in data.items()}


def heal_target(heal_pos: Tensor, orig_pos: Tensor, box: Tensor) -> Tensor:
    """Per-atom shift heal_pos - original (minimum image), with the mean shift removed.

    Removing the mean puts the original in the frame of the input; a uniform translation of
    the whole glass is not a defect and a translation-equivariant model cannot predict it.
    """
    d = heal_pos - orig_pos
    d = d - box[:, None, :] * torch.round(d / box[:, None, :])
    return d - d.mean(1, keepdim=True)
