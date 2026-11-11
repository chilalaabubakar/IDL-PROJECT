"""Silica point defects, stage 1: the standard defects of amorphous silica made in DM2's
glasses, and how well they are found and repaired without any learned model
(docs/SILICA_POINT_DEFECTS.md).

    python scripts/silica/point_defects_study.py --config configs/silica/point_defects.yaml \\
        out_dir=results/silica_point_defects_<date>

Every step is saved, so running the same command again continues where it stopped:

    originals.npz        the glasses of the split, energy-minimised: the defect-free originals
    defects_<kind>.npz   n_per_glass defects per glass, each relaxed (glassdiff.silica.point_defects)
    eval_<kind>.json     finding and repair
    summary.json         all kinds, and the natural coordination defects of the originals

Finding: the top-scored atom must lie within hit_radius of the defect (either site of a
Frenkel pair). Scores:
    coordination  the field's standard coordination analysis (no potential): each atom's
                  deviation from ideal bonding, summed over its cluster
    coordination, rare states first
                  the same, with each off-ideal atom weighted by how rare its bonding
                  state is in the defect-free originals (no potential)
    energy        per-atom energy, z-score within species (needs the potential)
    chance        the fraction of atoms within hit_radius of the defect
Repair: put the atom count right at the flagged atom, then minimise (all atoms free, fixed
box). Restored = the result is the original glass within restore_tol (identical atoms may
trade places).
    o_vacancy, si_vacancy  insert the missing atom in the largest gap within insert_search A of
                           the flagged atom
    o_interstitial         remove the flagged O (the composition says an O is extra)
    o_frenkel              move the flagged O to the gap 3-7 A away around which Si lack the
                           most O neighbours
"defect given" does the same at the true site (the missing atom in the largest gap within 1 A
of it): the ceiling that a perfect finder would reach. A minimisation that blows up (two atoms
too close) counts as not restored and is reported.
Minimisation: LAMMPS FIRE with the potential DM2's glasses were made with, until the largest
force is below fmax (glassdiff.silica.md.minimize_structure), in parallel processes. While a
kind is evaluated, finished minimisations are saved to eval_<kind>.partial.json about once a
minute, so a run stopped there loses at most a minute of work.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from glassdiff.silica.data import load_split
from glassdiff.silica.defects import OXYGEN, SILICON
from glassdiff.silica.md import minimize_many
from glassdiff.silica.point_defects import (
    IDEAL_CN,
    NAMES,
    SilicaDefectConfig,
    SilicaPointDefect,
    coordination,
    coordination_score,
    coordination_states,
    finish_defect,
    gaps,
    make_defect,
    rarity_table,
    same_glass,
    wrap,
)
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import git_state

SAVE_EVERY = 60  # seconds between saves of evaluation progress


def jobs_fingerprint(jobs) -> str:
    """Identifies a list of repair jobs, so saved progress is only reused for the same jobs."""
    h = hashlib.sha256()
    for pos, types, box in jobs:
        for a in (np.round(pos, 6), np.asarray(types, dtype=np.int64), np.round(box, 6)):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def frac(x) -> float:
    return round(float(np.mean(x)), 4) if len(x) else float("nan")


def save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def originals(cfg: dict, out: Path, workers) -> dict:
    path = out / "originals.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    split = load_split(cfg["data"])
    n = int(cfg.get("n_glasses") or len(split))
    pos = split.pos[:n].numpy().astype(np.float64)
    types = split.types[:n].numpy().astype(np.int64)
    box = split.box[:n].numpy().astype(np.float64)
    t0 = time.time()
    res = minimize_many(list(zip(pos, types, box)), workers=workers, fmax=float(cfg["fmax"]))
    failed = [i for i, r in enumerate(res) if r[0] is None]
    if failed:
        raise RuntimeError(f"minimising original glasses {failed} failed: {res[failed[0]][-1]}")
    o = dict(
        pos=np.stack([r[0] for r in res]),
        pe=np.stack([r[1] for r in res]),
        fmax=np.array([r[2] for r in res]),
        types=types,
        box=box,
        cooling_rate=split.cooling_rate[:n].numpy(),
    )
    save_npz(path, **o)
    print(f"originals: {n} glasses minimised in {time.time() - t0:.0f}s", flush=True)
    return o


def make_kind(kind, orig: dict, cfg: dict, dcfg: SilicaDefectConfig, out: Path, workers) -> dict:
    path = out / f"defects_{NAMES[kind]}.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    rng = np.random.default_rng([int(cfg["seed"]), int(kind)])
    made, owner = [], []
    for g in range(len(orig["pos"])):
        for _ in range(int(cfg["n_per_glass"])):
            made.append(
                make_defect(kind, orig["pos"][g], orig["types"][g], orig["box"][g], rng, dcfg)
            )
            owner.append(g)
    t0 = time.time()
    res = minimize_many(
        [(m["pos"], m["types"], orig["box"][g]) for m, g in zip(made, owner)],
        workers=workers,
        fmax=float(cfg["fmax"]),
    )
    rows = []
    for m, g, (relaxed, pe, fmax, *_) in zip(made, owner, res):
        failed = relaxed is None
        if failed:  # keep the unrelaxed glass, marked invalid
            relaxed, pe = m["pos"], np.full(len(m["pos"]), np.nan)
        fin = finish_defect(kind, m, relaxed, orig["pos"][g], orig["box"][g], dcfg)
        fin["valid"] = fin["valid"] and not failed
        rows.append(dict(m, **fin, relaxed=relaxed, pe=pe, fmax=fmax, glass=g))
    d = dict(
        glass=np.array([r["glass"] for r in rows]),
        atom=np.array([r["atom"] for r in rows]),
        site=np.stack([r["site"] for r in rows]),
        sites=np.stack([r["sites"] for r in rows]),
        pos=np.stack([r["relaxed"] for r in rows]),
        types=np.stack([r["types"] for r in rows]),
        pe=np.stack([r["pe"] for r in rows]),
        role=np.stack([r["role"] for r in rows]),
        fmax=np.array([r["fmax"] for r in rows]),
        valid=np.array([r["valid"] for r in rows]),
    )
    save_npz(path, **d)
    print(f"{NAMES[kind]}: {len(rows)} defects relaxed in {time.time() - t0:.0f}s", flush=True)
    return d


def repair(kind, pos, types, box, flagged, atom, site, cfg, dcfg):
    """The repaired glass (positions, types) for a flagged atom, or for the true site when
    ``site`` is given (the "defect given" ceiling)."""
    search = float(cfg["insert_search"])
    if kind in (SilicaPointDefect.O_VACANCY, SilicaPointDefect.SI_VACANCY):
        species = OXYGEN if kind == SilicaPointDefect.O_VACANCY else SILICON
        # the largest gap near the flagged atom, or within 1 A of the true site (neighbours
        # may have relaxed into the site itself; two atoms too close make the run blow up)
        centre, r = (pos[flagged], search) if site is None else (site, 1.0)
        (gap,), _ = gaps(pos, box, centre, 0.0, r, spacing=dcfg.grid_spacing)
        return np.insert(pos, atom, gap, axis=0), np.insert(types, atom, species)
    if kind == SilicaPointDefect.O_INTERSTITIAL:
        drop = len(pos) - 1 if site is not None else flagged
        return np.delete(pos, drop, 0), np.delete(types, drop)
    # Frenkel pair: move the O back
    new = pos.copy()
    if site is not None:
        (new[atom],), _ = gaps(pos, box, site, 0.0, 1.0, exclude=atom, spacing=dcfg.grid_spacing)
        return new, types
    cand, _ = gaps(pos, box, pos[flagged], 3.0, 7.0, k=8, min_sep=1.0, exclude=flagged,
                   spacing=dcfg.grid_spacing)  # fmt: skip
    cn, _, _ = coordination(pos, types, box, dcfg)
    lack = np.where(types == SILICON, np.maximum(0, IDEAL_CN[SILICON] - cn), 0)
    si = np.nonzero(lack > 0)[0]
    best = 0
    if len(si):
        need = [lack[si][np.linalg.norm(wrap(pos[si] - g, box), axis=1) < dcfg.si_o_cutoff + 0.3].sum()
                for g in cand]  # fmt: skip
        best = int(np.argmax(need))
    new[flagged] = cand[best]
    return new, types


def evaluate_kind(kind, orig, d, cfg, dcfg, rarity, out: Path, workers) -> dict:
    path = out / f"eval_{NAMES[kind]}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    keep = np.nonzero(d["valid"])[0]
    stats = {
        sp: (orig["pe"][orig["types"] == sp].mean(), orig["pe"][orig["types"] == sp].std())
        for sp in (OXYGEN, SILICON)
    }
    hit_r = float(cfg["hit_radius"])
    finders = ("coordination", "coordination, rare states first", "energy")
    flagged = {f: [] for f in finders}
    hits = {f: [] for f in finders}
    right_atom = {f: [] for f in finders}
    chance = []
    for e in keep:
        pos, types, g = d["pos"][e], d["types"][e], d["glass"][e]
        box = orig["box"][g]
        tie = 1e-6 * np.random.default_rng(int(e)).random(len(types))  # seeded tie-break
        scores = {
            "coordination": coordination_score(pos, types, box, dcfg) + tie,
            "coordination, rare states first": coordination_score(
                pos, types, box, dcfg, rarity=rarity
            )
            + tie,
            "energy": np.where(
                types == OXYGEN,
                (d["pe"][e] - stats[OXYGEN][0]) / stats[OXYGEN][1],
                (d["pe"][e] - stats[SILICON][0]) / stats[SILICON][1],
            ),
        }
        if kind in (SilicaPointDefect.O_INTERSTITIAL, SilicaPointDefect.O_FRENKEL):
            for f in finders:  # the extra or moved atom is an O
                scores[f] = np.where(types == OXYGEN, scores[f], -np.inf)
        dist = np.min(
            [np.linalg.norm(wrap(pos - d["sites"][e, k], box), axis=1) for k in range(2)], axis=0
        )
        chance.append(np.mean(dist < hit_r))
        for f in finders:
            top = int(np.argmax(scores[f]))
            flagged[f].append(top)
            hits[f].append(dist[top] < hit_r)
            right_atom[f].append(top == int(d["atom"][e]) if kind != SilicaPointDefect.O_VACANCY
                                 and kind != SilicaPointDefect.SI_VACANCY else np.nan)  # fmt: skip
    # finders that flag the same atom give the same repair: minimise it once
    jobs, index, labels = [], {}, []
    for f in (*finders, "defect given"):
        for n_e, e in enumerate(keep):
            flag = None if f == "defect given" else flagged[f][n_e]
            key = (int(e), "given" if flag is None else flag)
            if key not in index:
                g = d["glass"][e]
                site = None if flag is not None else d["site"][e]
                p, t = repair(kind, d["pos"][e], d["types"][e], orig["box"][g], flag,
                              int(d["atom"][e]), site, cfg, dcfg)  # fmt: skip
                index[key] = len(jobs)
                jobs.append((p, t, orig["box"][g]))
            labels.append((f, e, index[key]))
    # minimise every repaired glass, saving [restored, failed] per job as jobs finish; the
    # jobs are deterministic, so a stopped run continues from the saved ones
    tol = float(cfg["restore_tol"])
    job_example = [k[0] for k in index]
    partial = out / f"eval_{NAMES[kind]}.partial.json"
    state = dict(jobs=jobs_fingerprint(jobs), total=len(jobs), seconds=0.0, done={})
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved.get("jobs") == state["jobs"]:
            state = saved
    prior_seconds = float(state["seconds"])
    done = {int(j): v for j, v in state["done"].items()}
    todo = [j for j in range(len(jobs)) if j not in done]
    if done:
        print(f"{NAMES[kind]}: continuing, {len(done)} of {len(jobs)} minimisations saved",
              flush=True)  # fmt: skip
    last_save = time.time()

    def save():
        state["done"] = {str(j): v for j, v in sorted(done.items())}
        state["seconds"] = prior_seconds + time.time() - t0
        tmp = partial.with_name(partial.name + ".tmp")
        tmp.write_text(json.dumps(state) + "\n")
        os.replace(tmp, partial)

    def record(i, result):
        nonlocal last_save
        j = todo[i]
        p, g = result[0], d["glass"][job_example[j]]
        restored_ok = p is not None and same_glass(p, jobs[j][1], orig["pos"][g],
                                                   orig["types"][g], orig["box"][g], tol)  # fmt: skip
        done[j] = [bool(restored_ok), p is None]  # a run that blew up counts as not restored
        if time.time() - last_save > SAVE_EVERY:
            save()
            last_save = time.time()

    if todo:
        minimize_many([jobs[j] for j in todo], workers=workers, fmax=float(cfg["fmax"]),
                      on_result=record)  # fmt: skip
    ok = [done[j][0] for j in range(len(jobs))]
    failed = sum(done[j][1] for j in range(len(jobs)))
    restored = {f: [] for f in (*finders, "defect given")}
    for f, _, j in labels:
        restored[f].append(ok[j])
    result = dict(
        kind=NAMES[kind],
        examples=int(len(d["valid"])),
        valid=int(len(keep)),
        find={f: frac(hits[f]) for f in finders} | {"chance": frac(chance)},
        repair={f"flag by {f}": frac(restored[f]) for f in finders}
        | {"defect given (ceiling)": frac(restored["defect given"])},
        minimisations=len(jobs),
        failed_minimisations=int(failed),
        seconds=round(prior_seconds + time.time() - t0),
    )
    if kind in (SilicaPointDefect.O_INTERSTITIAL, SilicaPointDefect.O_FRENKEL):
        result["flagged the defect atom"] = {f: frac(right_atom[f]) for f in finders}
    path.write_text(json.dumps(result, indent=1) + "\n")
    partial.unlink(missing_ok=True)
    print(NAMES[kind], json.dumps(result), flush=True)
    return result


def main() -> None:
    cfg = cli_config(__doc__)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    workers = cfg.get("workers")
    dcfg = SilicaDefectConfig(
        **{k: cfg[k] for k in SilicaDefectConfig.__dataclass_fields__ if k in cfg}
    )
    dcfg.frenkel_hop = tuple(dcfg.frenkel_hop)
    orig = originals(cfg, out, workers)
    rarity = rarity_table(
        [
            coordination_states(p, t, b, dcfg)
            for p, t, b in zip(orig["pos"], orig["types"], orig["box"])
        ]
    )
    natural = []
    for p, t, b in zip(orig["pos"], orig["types"], orig["box"]):
        cn, _, peroxy = coordination(p, t, b, dcfg)
        ideal = np.where(t == SILICON, IDEAL_CN[SILICON], IDEAL_CN[OXYGEN])
        natural.append(int(((cn != ideal) | (peroxy > 0)).sum()))
    summary = dict(
        config=cfg,
        git=git_state(),
        originals=dict(
            glasses=len(orig["pos"]),
            cooling_rate=orig["cooling_rate"].tolist(),
            natural_coordination_defects_per_glass=natural,
            max_force=float(orig["fmax"].max()),
        ),
        kinds={},
    )
    for name in cfg["kinds"]:
        kind = next(k for k, v in NAMES.items() if v == name)
        d = make_kind(kind, orig, cfg, dcfg, out, workers)
        summary["kinds"][name] = evaluate_kind(kind, orig, d, cfg, dcfg, rarity, out, workers)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("->", out / "summary.json")


if __name__ == "__main__":
    main()
