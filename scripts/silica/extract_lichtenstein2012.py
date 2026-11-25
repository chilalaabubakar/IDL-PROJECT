"""Read the atomic coordinates of two-dimensional silica from the supplement of
Lichtenstein, Heyde and Freund, Phys. Rev. Lett. 109, 106101 (2012) (docs/NPJ_EXTENSIONS.md, item 5c).

    pdftotext -layout Supplemental_Material.pdf si.txt
    python scripts/silica/extract_lichtenstein2012.py si.txt data/silica2d_lichtenstein2012

The supplement lists the top layer of a silica bilayer on Ru(0001), imaged by scanning
tunnelling microscopy: Table S1, the crystalline-vitreous interface of the paper's Fig. 1b;
Table S2, a vitreous region of Fig. S1b (12.2 nm x 6.6 nm). The Si positions were measured;
the O positions were computed by the authors as the midpoints of neighbouring Si atoms, and
the O height (0.052 nm) comes from density functional theory. Writes, per table, x, y, z (nm)
and the element of every atom (<out>/<table>.npz and .csv) and a summary of checks
(<out>/checks.json): atom counts, Si-Si nearest-neighbour distance (the supplement gives
0.302 +- 0.022 nm for Table S2), Si coordination, and whether every O sits at the midpoint of
two Si atoms.

Uses numpy only, so it can run isolated (python -I) on text taken from a downloaded file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROW = re.compile(r"^\s*(Si|O)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
HEAD = re.compile(r"^Table (S\d): The coordinates of the atomic model of Figure (\S+) \((\w+)\)")


def read_tables(text: str) -> dict[str, dict]:
    tables: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        head = HEAD.match(line.strip())
        if head:
            current = head.group(1)
            tables[current] = dict(figure=head.group(2), region=head.group(3), rows=[])
            continue
        row = ROW.match(line)
        if row and current:
            tables[current]["rows"].append((row.group(1), *map(float, row.group(2, 3, 4))))
    return tables


def ring_sizes(si: np.ndarray, i: np.ndarray, j: np.ndarray) -> list[int]:
    """Sizes of the rings (faces) of the planar Si network; the outer boundary is left out."""
    nbr: dict[int, list[int]] = {k: [] for k in range(len(si))}
    for a, b in zip(i.tolist(), j.tolist()):
        nbr[a].append(b)
        nbr[b].append(a)
    for a in nbr:  # neighbours in order of angle
        nbr[a].sort(key=lambda b: np.arctan2(*(si[b] - si[a])[::-1]))
    seen, sizes = set(), []
    for a in nbr:
        for b in nbr[a]:
            if (a, b) in seen:
                continue
            face, (u, v) = [], (a, b)
            while (u, v) not in seen:
                seen.add((u, v))
                face.append(u)
                k = nbr[v].index(u)
                u, v = v, nbr[v][k - 1]  # the next edge, turning the same way at every atom
            if len(set(face)) == len(face) and 3 <= len(face) <= 12:
                sizes.append(len(face))
    return sizes


def edge_distance(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(0), xy.max(0)
    return np.minimum(points - lo, hi - points).min(1)


def checks(element: np.ndarray, xyz: np.ndarray) -> dict:
    si, o = xyz[element == "Si", :2], xyz[element == "O", :2]
    d = np.linalg.norm(si[:, None] - si[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    # bonds: Si pairs whose midpoint carries an O atom
    i, j = np.nonzero(np.triu(d < 0.45, 1))
    mid = (si[i] + si[j]) / 2
    o_to_mid = np.linalg.norm(o[:, None] - mid[None], axis=-1)
    bonded = o_to_mid.min(0) < 0.01  # an O within 0.01 nm of the pair's midpoint
    coordination = np.bincount(np.concatenate([i[bonded], j[bonded]]), minlength=len(si))
    edge_si = edge_distance(si, xyz[:, :2])
    edge_o = edge_distance(o, xyz[:, :2])
    lone_o = o_to_mid.min(1) >= 0.01
    rings = ring_sizes(si, i[bonded], j[bonded])
    # rings: the largest ring of a closed network bounds the image; drop it if present
    count = np.bincount(rings, minlength=13)
    return dict(
        si=int(len(si)), o=int(len(o)),
        width_nm=round(float(np.ptp(xyz[:, 0])), 3), height_nm=round(float(np.ptp(xyz[:, 1])), 3),
        si_nn_mean_nm=round(float(nn.mean()), 4), si_nn_sd_nm=round(float(nn.std()), 4),
        bonds=int(bonded.sum()),
        bond_length_mean_nm=round(float(d[i[bonded], j[bonded]].mean()), 4),
        bond_length_sd_nm=round(float(d[i[bonded], j[bonded]].std()), 4),
        o_at_a_bond_midpoint=round(float((o_to_mid.min(1) < 0.01).mean()), 4),
        si_coordination={str(k): int(v) for k, v in enumerate(np.bincount(coordination))},
        under_coordinated_si_within_0_5nm_of_edge=round(float((edge_si[coordination < 3] < 0.5).mean()), 4),
        o_not_at_a_midpoint=int(lone_o.sum()),
        o_not_at_a_midpoint_within_0_5nm_of_edge=round(float((edge_o[lone_o] < 0.5).mean()), 4) if lone_o.any() else None,
        rings={str(k): int(v) for k, v in enumerate(count) if v},
    )  # fmt: skip


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    text = Path(sys.argv[1]).read_text(errors="replace")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, table in read_tables(text).items():
        rows = table["rows"]
        element = np.array([r[0] for r in rows])
        xyz = np.array([r[1:] for r in rows])
        stem = f"table_{name.lower()}_{table['region']}"
        np.savez(out / f"{stem}.npz", element=element, xyz_nm=xyz)
        lines = ["element,x_nm,y_nm,z_nm"] + [f"{e},{x:.3f},{y:.3f},{z:.3f}" for e, (x, y, z) in zip(element, xyz)]  # fmt: skip
        (out / f"{stem}.csv").write_text("\n".join(lines) + "\n")
        report[name] = dict(figure=table["figure"], region=table["region"], file=stem, **checks(element, xyz))  # fmt: skip
        print(name, json.dumps(report[name]))
    (out / "checks.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
