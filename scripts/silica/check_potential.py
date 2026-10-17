"""Check our LAMMPS silica potential against DM2's published glasses before making data.

    python scripts/silica/check_potential.py --config configs/silica/md.yaml \\
        [check.glasses=data/sio2_dm2/test.npz] [check.ps=20]

DM2 does not publish its LAMMPS input, so we rebuilt it from the paper (src/glassdiff/silica/
md.py). If it is the same potential, DM2's glasses must stay where they are under 300 K NPT:
density within ``check.max_density_change`` (relative) and Si four-fold coordination within
``check.max_cn4_change``. A wrong unit or parameter makes the glass expand, collapse or lose
its tetrahedra within a few ps. Writes a JSON report to ``check.out``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from glassdiff.silica.data import load_split
from glassdiff.silica.md import SilicaProtocol, npt_check, structure_stats
from glassdiff.utils.config import cli_config


def main() -> None:
    cfg = cli_config(__doc__)
    chk = cfg["check"]
    p = SilicaProtocol(**cfg["protocol"])
    split = load_split(chk["glasses"])
    rows, ok = [], True
    for i in range(min(int(chk.get("n_glasses", 2)), len(split))):
        pos, types, box = (split.pos[i].numpy(), split.types[i].numpy(), split.box[i].numpy())
        before = structure_stats(pos, types, box)
        g = npt_check(pos, types, box, p, float(chk["ps"]), seed=i + 1)
        after = structure_stats(g.pos, g.types, g.box)
        d_rho = after["density"] / before["density"] - 1
        d_cn4 = after["si_cn4"] - before["si_cn4"]
        good = abs(d_rho) <= chk["max_density_change"] and abs(d_cn4) <= chk["max_cn4_change"]
        ok &= good
        rows.append({"glass": i, "before": before, "after": after, **g.stats, "pass": good})
        print(
            f"glass {i} ({split.cooling_rate[i]:g} K/ps, {len(types)} atoms, {chk['ps']} ps NPT): "
            f"density {before['density']:.3f} -> {after['density']:.3f} ({d_rho:+.1%}), "
            f"Si 4-fold {before['si_cn4']:.3f} -> {after['si_cn4']:.3f}, "
            f"T {g.stats['temp']:.0f} K, {g.stats['seconds']:.0f} s  {'ok' if good else 'FAIL'}",
            flush=True,
        )
    out = Path(chk.get("out", "results/silica/potential_check.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": p.to_dict(), "check": chk, "glasses": rows}, indent=2))
    print(f"{'PASS' if ok else 'FAIL'} -> {out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
