"""A classifier trained on classical inputs, against the learned finder (docs/NPJ_EXTENSIONS.md, item 2).

The finder learns from labelled defects; the classical measures of the manuscript are fixed
rules on one atom. This script trains a classifier on classical per-atom inputs with the same
labels, flags atoms the same way as the finder, and scores both on the same defects.

    python scripts/pilots/descriptor_classifier.py run --out results/descriptor_classifier_<date>
        features of the training and validation defects, fit, score on validation (~1-2 h on
        4 cores, after the training split exists:
        python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml splits=[train])
    python scripts/pilots/descriptor_classifier.py score --out <same dir> \
        --data data/ka2d_256_point_defects_fresh/test.npz --restored
        the fresh test set, once, at the end; --restored also runs the repair and minimisation
        of the finder's evaluation (scripts/eval_point_defects.py) with the classifier's flags

Inputs per atom (z-scores within species, with mean and spread from the defect-free training
glasses; "shell" = neighbours closer than 1.4 sigma_ab, the dataset's neighbour rule):

    A  positions and species only: species; Voronoi cell area; number of neighbours; number of
       unlike neighbours; nearest distance / sigma_ab; and the mean, minimum and maximum of the
       four over the atom's shell (17 inputs). The fair match to the finder.
    B  A plus the atom energy and the per-atom virial (local pressure) and their shell mean,
       minimum and maximum (25 inputs). Needs the potential.
    C  A plus the species-resolved radial density within 2.5 sigma in 16 Gaussian bins, the
       input the transformer finder starts from (49 inputs). Positions and species only.
    D  C plus the local hexatic order |psi_6| of the shell and its shell mean, minimum and
       maximum, and the mean, minimum and maximum of the four base measures over the second
       ring (atoms beyond the shell and within 2.5 sigma) (65 inputs). Positions and species only.
    E  D plus atom energy and virial with their shell and second-ring statistics (79 inputs).
       Needs the potential.

Tuning (docs/NPJ_EXTENSIONS.md, item 2):

    python scripts/pilots/descriptor_classifier.py tune --out results/descriptor_classifier_<date>
        holds out every 8th training glass (100 glasses, about 3,100 real defects), fits the
        trees on the other 700 with each setting of a small grid, and keeps the setting that
        finds the most held-out defects (writes tuning.json); the validation glasses are not used
    python scripts/pilots/descriptor_classifier.py run --out <same dir> --params <dir>/tuning.json --tag tuned
        refits every version on all 800 training glasses, tuned settings where tuning.json has
        them, and scores once on the validation glasses (validation_tuned.json, summary_tuned.md)

Model: gradient-boosted trees (sklearn HistGradientBoostingClassifier), five roles as the
finder (intact, next to an empty site, extra atom, displaced atom, wrong species), trained on
every atom with a defect role and a random tenth of the intact atoms of the training defects
(all four kinds, Frenkel pairs that undid themselves included, as for the finder). Early
stopping uses a tenth of the training rows, never the validation glasses. A logistic
regression on A is a simpler reference. For a defect of known type the atom with the highest
score for the matching role is flagged, the score being either the probability of the role
or its log-odds against "intact"; which one is chosen on the validation glasses (the
subsampling of intact atoms shifts the probabilities but not the log-odds ranking).

Scored on real defects only (Frenkel pairs that undid themselves are left out), with the
found test of the manuscript (flagged atom within 1.5 sigma of either defect site), next to
the learned finder (calibrated, scripts/eval_point_defects.py) and the single measures of the
manuscript (cell size, unlike neighbours, atom energy, local pressure) with their rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.dataset import load_split
from glassdiff.data.point_defects import AtomRole, PointDefect, load_point_defects, real_defects
from glassdiff.physics.ka_potential import KA_SIGMA, ka_energy, ka_virial
from glassdiff.types import Structures

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_pd", ROOT / "scripts/eval_point_defects.py")
br = _load("baseline_routes", ROOT / "scripts/pilots/baseline_routes.py")
wrap = ev.wrap

DATA = ROOT / "data/ka2d_256_point_defects"
CACHE = DATA / ".cache"  # git-ignored
FINDER = ROOT / "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt"
KINDS = ("vacancy", "interstitial", "frenkel", "antisite")
VERSIONS = ("A", "B", "C", "D", "E")
POSITIONS_ONLY = ("A", "C", "D")  # no potential: the fair match to the finder
HOLD_OUT = 8  # tuning holds out the training glasses whose index is a multiple of this
SHELL, ENV_CUTOFF, N_ENV = 1.4, 2.5, 16
INTACT_FRACTION = 0.1
DEFAULT_TREES = dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=40, weighting="none")
# tuning grid: 24 settings for version D; E is fitted with the 4 best of them
GRID = [
    dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=leaf, weighting=w)
    for lr in (0.05, 0.1)
    for leaves in (31, 127, 255)
    for leaf in (20, 200)
    for w in ("none", "sqrt")
]
SIGMA = np.asarray(KA_SIGMA)
# single measures of the manuscript: the rule chosen on validation glasses for each kind
# (scripts/pilots/baseline_routes.py, scripts/pilots/structural_signatures.py)
RULES = {
    "cell size": {"vacancy": "high", "interstitial": "low", "frenkel": "abs", "antisite": "abs"},
    "unlike neighbours": dict.fromkeys(KINDS, "low"),
    "atom energy": dict.fromkeys(KINDS, "high"),
    "local pressure": {
        "vacancy": "low",
        "interstitial": "abs",
        "frenkel": "low",
        "antisite": "low",
    },
}


# ----------------------------------------------------------------------------- inputs


def reference_stats() -> dict:
    """Mean and spread per species of the base measures in the defect-free training glasses."""
    path = CACHE / "reference_stats.json"
    if path.exists():
        return {k: np.array(v) for k, v in json.loads(path.read_text()).items()}
    split = load_split(ROOT / "data/ka2d_256/train.npz")
    s = split.structures
    pos, types, box = s.pos.double().numpy(), s.types.numpy(), s.box.double().numpy()
    base = np.concatenate([geometry(p, t, b) for p, t, b in zip(pos, types, box)])
    pot = np.concatenate(list(potential_terms(pos, types, box)), axis=0).reshape(-1, 2)
    t = types.reshape(-1)
    stats = {}
    for name, vals in (("geometry", base), ("potential", pot)):
        stats[f"{name}_mu"] = np.stack([vals[t == s].mean(0) for s in (0, 1)])
        stats[f"{name}_sd"] = np.stack([vals[t == s].std(0) for s in (0, 1)])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in stats.items()}))
    return stats


def geometry(pos: np.ndarray, types: np.ndarray, box: np.ndarray) -> np.ndarray:
    """[N, 4]: cell area, neighbours, unlike neighbours, nearest distance / sigma_ab."""
    dist = np.linalg.norm(wrap(pos[:, None] - pos[None], box), axis=-1)
    np.fill_diagonal(dist, np.inf)
    sig = SIGMA[types[:, None], types[None]]
    shell = dist < SHELL * sig
    unlike = shell & (types[:, None] != types[None])
    return np.stack(
        [br.voronoi_area(pos, box), shell.sum(1), unlike.sum(1), (dist / sig).min(1)], axis=1
    ).astype(np.float64)


def potential_terms(pos, types, box, batch: int = 64):
    """[E, N, 2] per batch: atom energy and per-atom virial (the potential is needed)."""
    for i in range(0, len(pos), batch):
        s = Structures(
            torch.as_tensor(pos[i : i + batch], dtype=torch.float64),
            torch.as_tensor(types[i : i + batch]).long(),
            torch.as_tensor(box[i : i + batch], dtype=torch.float64),
        )
        yield torch.stack([ka_energy(s, per_atom=True), ka_virial(s)], -1).numpy()


def radial_density(pos, types, box) -> np.ndarray:
    """[N, 32]: neighbours of each species within 2.5 sigma in 16 Gaussian bins (as the transformer)."""
    dist = np.linalg.norm(wrap(pos[:, None] - pos[None], box), axis=-1)
    np.fill_diagonal(dist, np.inf)
    weight = np.where(dist < ENV_CUTOFF, 0.5 * (np.cos(np.pi * np.minimum(dist, ENV_CUTOFF) / ENV_CUTOFF) + 1), 0.0)  # fmt: skip
    centres, width = np.linspace(0.0, ENV_CUTOFF, N_ENV), ENV_CUTOFF / N_ENV
    g = np.exp(-(((np.where(np.isfinite(dist), dist, 1e3)[..., None] - centres) / width) ** 2))
    onehot = np.eye(2)[types]
    return np.einsum("ij,ijr,js->isr", weight, g, onehot).reshape(len(pos), -1)


def shell_stats(values: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """[N, 3K]: mean, minimum and maximum of each column over the atom's shell (own value if none)."""
    count = shell.sum(1, keepdims=True)
    mean = (shell @ values) / np.maximum(count, 1)
    big = np.where(shell[..., None], values[None], np.inf)
    small = np.where(shell[..., None], values[None], -np.inf)
    lo, hi = big.min(1), small.max(1)
    none = count[:, 0] == 0
    mean[none], lo[none], hi[none] = values[none], values[none], values[none]
    return np.concatenate([mean, lo, hi], axis=1)


def glass_inputs(args) -> dict[str, np.ndarray]:
    """Inputs A to E of every atom of one glass."""
    pos, types, box, pot, stats = args
    t = types
    base = (geometry(pos, types, box) - stats["geometry_mu"][t]) / stats["geometry_sd"][t]
    pz = (pot - stats["potential_mu"][t]) / stats["potential_sd"][t]
    vec = wrap(pos[:, None] - pos[None], box)
    dist = np.linalg.norm(vec, axis=-1)
    np.fill_diagonal(dist, np.inf)
    shell = dist < SHELL * SIGMA[types[:, None], types[None]]
    ring = ~shell & (dist < ENV_CUTOFF)  # the second ring, beyond the shell
    a = np.concatenate([types[:, None].astype(float), base, shell_stats(base, shell)], axis=1)
    b = np.concatenate([a, pz, shell_stats(pz, shell)], axis=1)
    c = np.concatenate([a, radial_density(pos, types, box)], axis=1)
    angle = np.arctan2(vec[..., 1], vec[..., 0])
    psi6 = np.abs((shell * np.exp(6j * angle)).sum(1)) / np.maximum(shell.sum(1), 1)
    psi6 = psi6[:, None]
    d = np.concatenate([c, psi6, shell_stats(psi6, shell), shell_stats(base, ring)], axis=1)
    e = np.concatenate([d, pz, shell_stats(pz, shell), shell_stats(pz, ring)], axis=1)
    return {k: v.astype(np.float32) for k, v in zip(VERSIONS, (a, b, c, d, e))}


def set_inputs(d, stats, workers: int) -> dict[str, np.ndarray]:
    """{version: [E, N, F]} for one defect kind."""
    pos, types, box = d.pos.numpy().astype(np.float64), d.types.numpy(), d.box.numpy().astype(np.float64)  # fmt: skip
    pot = np.concatenate(list(potential_terms(pos, types, box)))
    jobs = [(pos[i], types[i], box[i], pot[i], stats) for i in range(len(pos))]
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(glass_inputs, jobs, chunksize=16)
    return {v: np.stack([r[v] for r in rows]) for v in VERSIONS}


def training_rows(stats, workers: int, seed: int = 0) -> dict:
    """Inputs and roles of every defect-role atom and a random tenth of the intact atoms."""
    path = CACHE / f"descriptor_train_{''.join(VERSIONS)}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    data = load_point_defects(DATA / "train.npz")
    rng = np.random.default_rng(seed)
    parts: dict[str, list] = {v: [] for v in VERSIONS} | {"role": [], "kind": [], "glass": []}
    for k, name in enumerate(KINDS):
        d = data[name]
        t0 = time.time()
        x = set_inputs(d, stats, workers)
        role = d.role.numpy()
        keep = (role != AtomRole.INTACT) | (rng.random(role.shape) < INTACT_FRACTION)
        for v in VERSIONS:
            parts[v].append(x[v][keep])
        parts["role"].append(role[keep])
        parts["kind"].append(np.full(int(keep.sum()), k))
        parts["glass"].append(np.broadcast_to(d.glass.numpy()[:, None], role.shape)[keep])
        print(f"train {name}: {len(d)} glasses, {int(keep.sum())} rows, {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    out = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez(path, **out)
    return out


# ----------------------------------------------------------------------------- models


def fit_trees(x: np.ndarray, role: np.ndarray, cfg: dict, seed: int = 0):
    """Gradient-boosted trees with one setting; early stopping on a tenth of the rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"], max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
    )  # fmt: skip
    weight = None
    if cfg.get("weighting", "none") == "sqrt":  # as the finder: (n_c / n_intact)^-1/2
        count = np.bincount(role, minlength=5).astype(float)
        weight = (count / count[0])[role] ** -0.5
    m.fit(x, role, sample_weight=weight)
    return m


def fit_models(rows: dict, params: dict | None = None, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}
    for v in VERSIONS:
        t0 = time.time()
        cfg = (params or {}).get(v, DEFAULT_TREES)
        m = fit_trees(rows[v], rows["role"], cfg, seed)
        models[f"trees {v}"] = m
        print(f"fit trees {v} {cfg}: {m.n_iter_} iterations, {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(rows["A"], rows["role"])
    models["logistic A"] = lr
    print(f"fit logistic A: {time.time() - t0:.0f}s", flush=True)
    return models


def role_scores(model, x: np.ndarray, role: int) -> dict[str, np.ndarray]:
    """[E, N] scores for one role: probability, and log-odds against intact."""
    e, n, f = x.shape
    p = model.predict_proba(x.reshape(-1, f)).reshape(e, n, -1)
    eps = 1e-12
    return {"prob": p[..., role], "odds": np.log(p[..., role] + eps) - np.log(p[..., 0] + eps)}


class Classifier:
    """The trained classifier with the interface of eval_point_defects.Models (repair test)."""

    def __init__(self, model, version: str, how: str, stats, workers: int):
        self.model, self.version, self.how = model, version, how
        self.stats, self.workers = stats, workers
        self.detector = "descriptor classifier"  # not None: evaluate_kind uses role_prob
        self.do_heal, self.energy_search, self.empty_site_first = False, True, False
        self.energy_search_flags = {"energy"}  # as the finder's test evaluation

    def role_prob(self, pos, types, box) -> np.ndarray:
        """[E, N, 5] role probabilities, or with "odds" p_c / p_intact (same ranking as the log-odds)."""

        class _D:
            pass

        d = _D()
        d.pos, d.types, d.box = torch.as_tensor(pos), torch.as_tensor(types), torch.as_tensor(box)
        x = set_inputs(d, self.stats, self.workers)[self.version]
        e, n, f = x.shape
        p = self.model.predict_proba(x.reshape(-1, f)).reshape(e, n, -1)
        return p / np.maximum(p[..., :1], 1e-12) if self.how == "odds" else p


# ----------------------------------------------------------------------------- scoring


def site_distance(d) -> np.ndarray:
    pos, box, sites = d.pos.numpy(), d.box.numpy(), d.sites.numpy()
    return np.stack(
        [np.linalg.norm(wrap(pos - sites[:, k][:, None], box[:, None]), axis=-1) for k in range(2)]
    ).min(0)


def found_each(score: np.ndarray, d) -> np.ndarray:
    dist = site_distance(d)
    return (dist[np.arange(len(dist)), score.argmax(-1)] < ev.HIT_RADIUS).astype(int)


def single_measures(d, x: dict, kind: str) -> dict[str, np.ndarray]:
    """Scores of the manuscript's single measures, from the z-scored inputs (columns of B)."""
    b = x["B"]
    z = {"cell size": b[..., 1], "unlike neighbours": b[..., 3], "atom energy": b[..., 17],
         "local pressure": b[..., 18]}  # fmt: skip
    rng = np.random.default_rng(0)
    out = {}
    for name, rule in RULES.items():
        v = z[name] + (1e-6 * rng.random(z[name].shape) if name == "unlike neighbours" else 0.0)
        out[name] = {"high": v, "low": -v, "abs": np.abs(v)}[rule[kind]]
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import binomtest

    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)


def score(data_path: Path, models: dict, stats, workers: int, finder: Path | None, choice=None) -> dict:  # fmt: skip
    data = real_defects(load_point_defects(data_path))
    out: dict = {"data": str(data_path.relative_to(ROOT)), "kinds": {}}
    finder_models = None
    if finder is not None:
        log_prior = ev.finder_log_prior(str(finder), DATA)[0]
        finder_models = ev.Models(str(finder), str(finder), threads=4, device="cpu",
                                  log_prior=log_prior, do_heal=False)  # fmt: skip
    for name in KINDS:
        d = data[name]
        t0 = time.time()
        role = int(ev.REPAIR_ROLE[PointDefect[name.upper()]])
        x = set_inputs(d, stats, workers)
        each: dict[str, list] = {}
        for mname, model in models.items():
            version = mname.split()[-1]
            for how, sc in role_scores(model, x[version], role).items():
                each[f"{mname}, {how}"] = found_each(sc, d).tolist()
        for mname, sc in single_measures(d, x, name).items():
            each[mname] = found_each(sc, d).tolist()
        if finder_models is not None:
            p = finder_models.role_prob(d.pos.numpy(), d.types.numpy(), d.box.numpy())[..., role]
            each["finder"] = found_each(p, d).tolist()
        out["kinds"][name] = {
            "examples": len(d), "glass": d.glass.tolist(),
            "found": {k: round(float(np.mean(v)), 4) for k, v in each.items()}, "each": each,
        }  # fmt: skip
        print(f"{name}: {len(d)} defects, {time.time() - t0:.0f}s, " + json.dumps(out["kinds"][name]["found"]), flush=True)  # fmt: skip
    methods = list(out["kinds"][KINDS[0]]["each"])
    out["all"] = {
        m: round(float(np.mean(np.concatenate([out["kinds"][k]["each"][m] for k in KINDS]))), 4)
        for m in methods
    }
    if choice is None:  # validation: choose prob or odds per model on all kinds together
        choice = {}
        for mname in models:
            choice[mname] = max(("prob", "odds"), key=lambda h: out["all"][f"{mname}, {h}"])
    out["choice"] = choice
    if finder_models is not None:
        out["finder_vs"] = {}
        for mname in list(models) + list(RULES):
            key = f"{mname}, {choice[mname]}" if mname in models else mname
            out["finder_vs"][mname] = {
                k: {"finder": out["kinds"][k]["found"]["finder"],
                    "other": out["kinds"][k]["found"][key],
                    "p": round(mcnemar(np.array(out["kinds"][k]["each"]["finder"]),
                                       np.array(out["kinds"][k]["each"][key])), 4)}
                for k in KINDS
            }  # fmt: skip
    return out


def restored(data_path: Path, model, version: str, how: str, stats, workers: int, out_path: Path) -> dict:  # fmt: skip
    """Repair and minimisation with the classifier's flags, as the finder's test evaluation."""
    data = real_defects(load_point_defects(data_path))
    clf = Classifier(model, version, how, stats, workers)
    any_set = next(iter(data.values()))
    e = ka_energy(Structures(any_set.orig_pos, any_set.orig_types, any_set.box), per_atom=True)
    t = any_set.orig_types
    mu = np.array([float(e[t == s].mean()) for s in (0, 1)])
    sd = np.array([float(e[t == s].std()) for s in (0, 1)])
    res = {}
    for i, name in enumerate(KINDS):
        res[name] = ev.evaluate_kind(
            data[name], clf, mu, sd, np.random.default_rng([0, i]), workers
        )
        out_path.write_text(json.dumps(res, indent=1) + "\n")
        print(name, json.dumps(res[name]["repair"]), flush=True)
    return res


# ----------------------------------------------------------------------------- tuning


def held_out_found(model, inner: dict, version: str) -> dict:
    """Share of held-out defects found, per kind and for all, ranking by prob and by odds."""
    res: dict = {}
    for how in ("prob", "odds"):
        each = {}
        for name, (d, x) in inner.items():
            role = int(ev.REPAIR_ROLE[PointDefect[name.upper()]])
            each[name] = found_each(role_scores(model, x[version], role)[how], d)
        res[how] = {k: round(float(v.mean()), 4) for k, v in each.items()}
        res[how]["all"] = round(float(np.concatenate(list(each.values())).mean()), 4)
    return res


def tune(stats, workers: int, out: Path, seed: int = 0) -> dict:
    """Choose the tree settings on held-out training glasses (never the validation glasses)."""
    t0 = time.time()
    rows = training_rows(stats, workers)
    held = rows["glass"] % HOLD_OUT == 0
    fit_rows = {k: v[~held] for k, v in rows.items()}
    data = load_point_defects(DATA / "train.npz")
    inner = {}
    for name in KINDS:
        d = data[name]
        d = d.subset(torch.from_numpy(d.glass.numpy() % HOLD_OUT == 0))
        d = real_defects({name: d})[name]
        x = set_inputs(d, stats, workers)
        inner[name] = (d, {v: x[v] for v in ("A", "C", "D", "E")})
    sizes = {k: len(v[0]) for k, v in inner.items()}
    print(f"held out: {int(held.sum())} training rows; defects {sizes}; {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    trials = []

    def trial(version: str, cfg: dict) -> None:
        t1 = time.time()
        m = fit_trees(fit_rows[version], fit_rows["role"], cfg, seed)
        res = held_out_found(m, inner, version)
        trials.append(dict(version=version, cfg=cfg, iterations=int(m.n_iter_), found=res,
                           seconds=round(time.time() - t1)))  # fmt: skip
        best = max(res, key=lambda h: res[h]["all"])
        print(f"{version} {cfg}: {m.n_iter_} it, {time.time() - t1:.0f}s, held-out found "
              f"{res[best]['all']:.4f} ({best}) {json.dumps(res[best])}", flush=True)  # fmt: skip
        # running log in the git-ignored cache; the full list goes to out/ at the end
        (CACHE / "tuning_trials_running.json").write_text(json.dumps(trials, indent=1) + "\n")

    def best_of(version: str) -> tuple[dict, str, float]:
        cands = [(t["cfg"], h, t["found"][h]["all"]) for t in trials if t["version"] == version
                 for h in ("prob", "odds")]  # fmt: skip
        return max(cands, key=lambda c: c[2])

    for v in ("A", "C", "D"):  # the settings of the first run, for reference
        trial(v, DEFAULT_TREES)
    for cfg in GRID:
        if cfg != DEFAULT_TREES:
            trial("D", cfg)
    ranked = sorted({json.dumps(t["cfg"], sort_keys=True): max(t["found"][h]["all"] for h in ("prob", "odds"))
                     for t in trials if t["version"] == "D"}.items(), key=lambda kv: -kv[1])  # fmt: skip
    for cfg_text, _ in ranked[:4]:  # E with the 4 best settings of D
        trial("E", json.loads(cfg_text))
    for v in ("A", "C"):  # the best setting of D for the smaller versions too
        if best_of("D")[0] != DEFAULT_TREES:
            trial(v, best_of("D")[0])
    result = {"hold_out": f"training glasses with index % {HOLD_OUT} == 0", "defects": sizes,
              "params": {}, "choice": {}, "held_out_found": {}}  # fmt: skip
    for v in ("A", "C", "D", "E"):
        cfg, how, found = best_of(v)
        result["params"][v], result["choice"][f"trees {v}"] = cfg, how
        result["held_out_found"][v] = found
    result["seconds"] = round(time.time() - t0)
    (out / "tuning_trials.json").write_text(json.dumps(trials, indent=1) + "\n")
    (out / "tuning.json").write_text(json.dumps(result, indent=1) + "\n")
    print("chosen:", json.dumps(result["params"]), json.dumps(result["held_out_found"]), flush=True)
    return result


# ----------------------------------------------------------------------------- report


def summary(val: dict, out: Path, tag: str = "") -> None:
    choice = val["choice"]
    rows = [("Learned finder (EGNN, calibrated)", "finder")]
    for mname, how in choice.items():
        label = (
            f"Trees on {mname.split()[-1]}"
            if mname.startswith("trees")
            else "Logistic regression on A"
        )
        rows.append((f"{label} ({how})", f"{mname}, {how}"))
    rows += [(name, name) for name in RULES]
    lines = [
        f"# Classifier on classical inputs: validation{' (' + tag + ')' if tag else ''}",
        "",
        "Defects found (%), real validation defects, flagged atom within 1.5 sigma of the defect.",
        "A, C, D: positions and species only; B, E: also atom energy and local pressure (need the",
        "potential). See the docstring of `scripts/pilots/descriptor_classifier.py`.",
        "",
        "| Method | " + " | ".join(KINDS) + " | all |",
        "|---|" + "---|" * (len(KINDS) + 1),
    ]
    for label, key in rows:
        cells = [f"{100 * val['kinds'][k]['found'][key]:.1f}" for k in KINDS]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {100 * val['all'][key]:.1f} |")
    lines += ["", "McNemar p (exact, not corrected for multiple tests), finder against each:", ""]
    for mname, per in val.get("finder_vs", {}).items():
        lines.append(
            f"- {mname}: "
            + ", ".join(
                f"{k} {v['p']:.3g}" if v["p"] >= 0.001 else f"{k} <0.001" for k, v in per.items()
            )
        )
    (out / f"summary{'_' + tag if tag else ''}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["run", "tune", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--data", type=Path, default=DATA / "val.npz", help="score: the defects to score"
    )
    ap.add_argument("--restored", action="store_true", help="score: also repair and minimise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-finder", action="store_true", help="do not score the learned finder")
    ap.add_argument("--finder", type=Path, default=FINDER, help="finder checkpoint to compare with")
    ap.add_argument("--params", type=Path, help="run: tuning.json with the tree settings")
    ap.add_argument("--tag", default="", help="run, score: suffix of the output files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    stats = reference_stats()
    finder = None if args.no_finder else args.finder.resolve()
    sfx = f"_{args.tag}" if args.tag else ""
    if args.what == "tune":
        tune(stats, args.workers, args.out)
    elif args.what == "run":
        rows = training_rows(stats, args.workers)
        print({k: v.shape for k, v in rows.items()}, flush=True)
        tuned = json.loads(args.params.read_text()) if args.params else {}
        models = fit_models(rows, tuned.get("params"))
        with open(args.out / f"models{sfx}.pkl", "wb") as fh:
            pickle.dump(models, fh)
        # ranking (prob or odds): as chosen on the held-out training glasses when tuned
        choice = {m: tuned["choice"].get(m, "odds") for m in models} if tuned else None
        val = score(DATA / "val.npz", models, stats, args.workers, finder, choice)
        val["seconds"] = round(time.time() - t0)
        val["training_rows"] = {"rows": int(len(rows["role"])), "intact_fraction": INTACT_FRACTION,
                                "per_role": np.bincount(rows["role"], minlength=5).tolist()}  # fmt: skip
        if tuned:
            val["params"] = tuned["params"]
        (args.out / f"validation{sfx}.json").write_text(json.dumps(val, indent=1) + "\n")
        summary(val, args.out, args.tag)
    else:
        with open(args.out / f"models{sfx}.pkl", "rb") as fh:
            models = pickle.load(fh)
        choice = json.loads((args.out / f"validation{sfx}.json").read_text())["choice"]
        res = score(args.data.resolve(), models, stats, args.workers, finder, choice)
        name = args.data.parent.name + "_" + args.data.stem + sfx
        (args.out / f"score_{name}.json").write_text(json.dumps(res, indent=1) + "\n")
        if args.restored:
            # the best version that uses positions and species only
            best = max(
                (v for v in POSITIONS_ONLY if f"trees {v}" in models),
                key=lambda v: res["all"][f"trees {v}, {choice[f'trees {v}']}"],
            )
            restored(args.data.resolve(), models[f"trees {best}"], best, choice[f"trees {best}"],
                     stats, args.workers, args.out / f"restored_{name}_trees_{best}.json")  # fmt: skip
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
