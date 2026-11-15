"""Other finders: a transformer and a CNN against the EGNN, trained and tested the same way.

Used by notebooks/kaggle_attention_finder.ipynb and notebooks/kaggle_cnn_finder.ipynb; runs
anywhere (GPU or CPU). Four commands (status prints the runs not finished yet):

    python scripts/pilots/finder_architectures.py train --models attention egnn_scratch \\
        --seeds 0 1 2 --runs <dir> --minutes 600
    python scripts/pilots/finder_architectures.py export --runs <dir> --dest results/<new dir>
    python scripts/pilots/finder_architectures.py evaluate --runs <dir> --out <dir>

Models (configs/train/detect_<model>.yaml; the same data, loss, class weights, optimiser,
batch size and 20,000 updates, all from random starting weights):

    attention     AtomTransformerClassifier: attention over all atoms, distances as bias
    cnn           GridUNetClassifier: the glass drawn as an image, a periodic U-Net
    egnn_scratch  the EGNN finder from random starting weights: the control

train   runs every (model, seed) that is not finished, one per GPU at a time, each with its
        own seed. A run finished here or pushed to results/*/training/<model>_s<seed>/ema.pt
        is skipped; one stopped early (its full checkpoint pushed as ckpt.pt) continues.
        Runs stop by themselves before --minutes is up, saving a checkpoint.
export  copies each run of --runs to <dest>/training/<model>_s<seed>/: config, training log,
        per-update log (steps.csv), and the weights (ema.pt when finished, else the full
        checkpoint so that a later session can continue; --finished-only leaves those out).
        Runs pushed already and unchanged are skipped. Prints what is new, as JSON.
evaluate scores every finished run (here or pushed) and the finder of the report (from the
        denoiser, after 20,000 and 60,000 updates) on the 750 real test defects, calibrated as
        everywhere in the study, and writes <out>/architectures.json and summary.md:
          found per defect type and seed; mean and spread over seeds; each model against the
            EGNN control with the same seed (McNemar's exact test);
          symmetry: how often the flagged atom stays the same when the glass is turned by
            90 degrees, shifted by a random vector, or its atoms renumbered (1.0 = always);
          training stability from steps.csv (scripts/pilots/finder_seeds.py): gradient norm
            before clipping, share of updates clipped, gradient and loss spikes;
          finder against finder with the same seed, how far they find the same defects, and
            finders used together (the mean of their calibrated probabilities).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("seeds", ROOT / "scripts/pilots/finder_seeds.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)
fa, ev = fs.fa, fs.ev

MODELS = {
    "attention": "Transformer: attention over all atoms, distances as bias",
    "cnn": "CNN: periodic U-Net on the glass drawn as an image",
    "egnn_scratch": "EGNN (control): message passing, reach 10 sigma",
}
REFERENCE = {  # the finder of the report, started from the denoiser: weights, what it is
    "egnn_pretrained": (
        ROOT / "results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt",
        "the finder of the report after 20,000 updates (from the denoiser)",
    ),
    "egnn_pretrained_60k": (
        ROOT / "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt",
        "the finder of the report, 60,000 updates (from the denoiser)",
    ),
}
TOGETHER = {  # finders used together: the mean of their calibrated probabilities
    "transformer, 3 seeds": [("attention", s) for s in range(3)],
    "CNN, 3 seeds": [("cnn", s) for s in range(3)],
    "EGNN control, 3 seeds": [("egnn_scratch", s) for s in range(3)],
    "transformer + CNN, seed 0": [("attention", 0), ("cnn", 0)],
    "transformer + CNN + EGNN control, seed 0": [("attention", 0), ("cnn", 0), ("egnn_scratch", 0)],
    "transformer + CNN, 3 seeds each": [(m, s) for m in ("attention", "cnn") for s in range(3)],
    "transformer + CNN (seed 0) + the finder of the report": [
        ("attention", 0),
        ("cnn", 0),
        ("egnn_pretrained_60k", 0),
    ],
}
ROT90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def config(model: str) -> Path:
    return ROOT / f"configs/train/detect_{model}.yaml"


def wanted_updates(model: str) -> int:
    return int(yaml.safe_load(config(model).read_text())["optim"]["n_updates"])


def local_run(runs: Path, name: str) -> Path | None:
    found = sorted(r for r in runs.glob(f"*_{name}") if (r / "ckpt.pt").exists())
    return found[-1] if found else None


def pushed(name: str, fname: str) -> Path | None:
    found = sorted(glob.glob(str(ROOT / f"results/*/training/{name}/{fname}")))
    return Path(found[-1]) if found else None


def progress(runs: Path, name: str) -> tuple[int, int, Path] | None:
    """(updates done, updates planned, checkpoint) from this session, else the newest push."""
    for path in (local_run(runs, name), pushed(name, "ema.pt"), pushed(name, "ckpt.pt")):
        if path is None:
            continue
        ckpt = path / "ckpt.pt" if path.is_dir() else path
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(c["step"]), int(c["cfg"]["optim"]["n_updates"]), ckpt
    return None


def finished(runs: Path, name: str) -> bool:
    p = progress(runs, name)
    return p is not None and p[0] >= p[1]


# ----------------------------------------------------------------------------- train


def train(args) -> None:
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    slots = list(range(args.gpus if args.gpus is not None else max(1, torch.cuda.device_count())))
    queue = [(m, s) for s in args.seeds for m in args.models]  # every model's seed 0 first
    queue = [(m, s) for m, s in queue if not finished(runs, f"{m}_s{s}")]
    print(f"to train: {[f'{m}_s{s}' for m, s in queue]} on {len(slots)} slot(s)", flush=True)
    running: dict[int, tuple[subprocess.Popen, str, object]] = {}
    last_report = time.time()
    while queue or running:
        left = args.minutes - (time.time() - start) / 60
        for slot in [s for s in slots if s not in running]:
            if not queue or left < args.min_start:
                break
            model, seed = queue.pop(0)
            name = f"{model}_s{seed}"
            run = local_run(runs, name)
            if run is None and pushed(name, "ckpt.pt") and not pushed(name, "ema.pt"):
                run = (
                    runs / f"00000000-000000Z_{name}"
                )  # continue the run an earlier session pushed
                shutil.copytree(pushed(name, "ckpt.pt").parent, run, dirs_exist_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "scripts/train_point_defects.py"), "--config",
                   str(config(model)), f"max_minutes={max(1, int(left - 5))}", f"threads={args.threads}"]  # fmt: skip
            cmd += (
                [f"resume={run}"]
                if run
                else [f"runs_root={runs}", f"name={name}", f"seed={seed}", *args.set]
            )
            env = dict(os.environ)
            if torch.cuda.is_available():
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            log = open(runs / f"{name}.log", "a")
            running[slot] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, env=env, cwd=ROOT), name, log)  # fmt: skip
            print(f"{time.strftime('%H:%M')} started {name} on slot {slot}"
                  f"{' (continued)' if run else ''}", flush=True)  # fmt: skip
        if not running:
            break
        time.sleep(args.poll)
        for slot, (job, name, log) in list(running.items()):
            if job.poll() is not None:
                log.close()
                tail = (runs / f"{name}.log").read_text().splitlines()[-1:]
                print(f"{time.strftime('%H:%M')} {name}: exit {job.returncode} {tail}", flush=True)
                del running[slot]
        if time.time() - last_report > args.report_every * 60:
            for _, name, _ in running.values():
                rows = [json.loads(x) for x in (runs / f"{name}.log").read_text().splitlines() if x.startswith("{")]  # fmt: skip
                if rows:
                    r = rows[-1]
                    hits = ", ".join(f"{k} {r[f'{k}_hit']:.2f}" for k in ("vacancy", "interstitial", "frenkel", "antisite"))  # fmt: skip
                    print(f"{time.strftime('%H:%M')} {name} step {r['step']}: found (validation) {hits}", flush=True)  # fmt: skip
            last_report = time.time()
    if queue:
        print(f"no time left for {[f'{m}_s{s}' for m, s in queue]}: run again", flush=True)


# ----------------------------------------------------------------------------- export


def export(args) -> None:
    runs, dest = Path(args.runs), Path(args.dest)
    new = []
    for run in sorted(p for p in runs.glob("*_*") if (p / "ckpt.pt").exists()):
        name = run.name.split("_", 1)[1]
        last = pushed(name, "log.jsonl")
        if (
            last
            and (run / "log.jsonl").exists()
            and last.read_bytes() == (run / "log.jsonl").read_bytes()
        ):
            continue  # pushed already, unchanged
        ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        done = ckpt["step"] >= ckpt["cfg"]["optim"]["n_updates"]
        if args.finished_only and not done:
            continue
        out = dest / "training" / name
        out.mkdir(parents=True, exist_ok=True)
        if done:  # the moving-average weights, as every finder of the study is evaluated
            keep = ("model_cfg", "ema", "step", "cfg", "class_weights", "seconds")
            torch.save({k: ckpt[k] for k in keep if k in ckpt}, out / "ema.pt")
        else:  # stopped early: the full checkpoint, so that the next session can continue
            shutil.copy2(run / "ckpt.pt", out / "ckpt.pt")
        for f in ("config.yaml", "log.jsonl", "steps.csv", "git.txt"):
            if (run / f).exists():
                shutil.copy2(run / f, out / f)
        new.append(f"{name} " + ("finished" if done else f"step {ckpt['step']}"))
    print(json.dumps(new))


# ----------------------------------------------------------------------------- evaluate


def flagged(models, data) -> dict[str, np.ndarray]:
    """Per defect kind, the index of the atom the finder flags in each example."""
    return {
        name: models.role_prob(d.pos.numpy(), d.types.numpy(), d.box.numpy())[
            ..., int(ev.REPAIR_ROLE[d.kind])
        ].argmax(-1)  # fmt: skip
        for name, d in data.items()
    }


def symmetry(path: Path, data: dict, device: str, rng, threads: int = 2) -> dict:
    """How often the flagged atom stays the same under a turn, a shift and a renumbering."""
    log_prior = ev.finder_log_prior(str(path), ROOT / "data/ka2d_256_point_defects")[0]
    models = ev.Models(str(path), str(path), threads=threads, device=device, log_prior=log_prior, do_heal=False)  # fmt: skip
    base = flagged(models, data)
    out = {}
    for what in ("turn by 90 degrees", "random shift", "renumbering"):
        same = []
        for name, d in data.items():
            pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
            perm = np.arange(pos.shape[1])
            if what == "turn by 90 degrees":
                pos = np.mod(pos @ ROT90.T, box[:, None, :])
            elif what == "random shift":
                pos = np.mod(pos + rng.random((len(pos), 1, 2)) * box[:, None, :], box[:, None, :])
            else:
                perm = rng.permutation(pos.shape[1])
                pos, types = pos[:, perm], types[:, perm]
            score = models.role_prob(pos, types, box)[..., int(ev.REPAIR_ROLE[d.kind])]
            same.append(perm[score.argmax(-1)] == base[name])
        out[what] = round(float(np.concatenate(same).mean()), 4)
    return out


def evaluate(args) -> None:
    runs = Path(args.runs) if args.runs else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    threads = args.threads
    data = ev.real_defects(
        ev.load_point_defects(str(ROOT / "data/ka2d_256_point_defects/test.npz"))
    )
    if args.max_examples:
        data = {k: d.subset(slice(0, args.max_examples)) for k, d in data.items()}
    glass = {k: d.glass.numpy() for k, d in data.items()}
    found: dict[tuple[str, int], Path] = {}  # (model, seed) -> weights
    steps: dict[tuple[str, int], Path] = {}
    for model in MODELS:
        for seed in range(10):
            name = f"{model}_s{seed}"
            run = local_run(runs, name) if runs else None
            if run is not None and finished(runs, name):
                found[(model, seed)], steps[(model, seed)] = run / "ckpt.pt", run / "steps.csv"
            elif pushed(name, "ema.pt"):
                found[(model, seed)] = pushed(name, "ema.pt")
                steps[(model, seed)] = found[(model, seed)].parent / "steps.csv"
    for name, (path, _) in REFERENCE.items():
        if path.exists():
            found[(name, 0)] = path
    if not found:
        raise SystemExit("no finished run to evaluate")
    hits, info, scores = {}, {}, {}
    for key, path in sorted(found.items()):
        log_prior = ev.finder_log_prior(str(path), ROOT / "data/ka2d_256_point_defects")[0]
        models = ev.Models(str(path), str(path), threads=threads, device=device, log_prior=log_prior, do_heal=False)  # fmt: skip
        score = {
            name: models.role_prob(d.pos.numpy(), d.types.numpy(), d.box.numpy())[..., int(ev.REPAIR_ROLE[d.kind])]
            for name, d in data.items()
        }  # fmt: skip
        hits[key], scores[key] = fs.hits(score, data), score
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        info[key] = dict(
            parameters=sum(v.numel() for v in ckpt["ema"].values()),
            updates=int(ckpt["step"]),
            hours=round(float(ckpt.get("seconds", 0)) / 3600, 2) or None,
            weights=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        )
        print(key, {k: round(float(v.mean()), 3) for k, v in hits[key].items()}, flush=True)
    kinds = [*data, "all"]
    out: dict = dict(n={k: len(d) for k, d in data.items()}, device=device, models={})
    for model in [*MODELS, *REFERENCE]:
        seeds = sorted(s for (m, s) in hits if m == model)
        if not seeds:
            continue
        per = {s: fa.rates(hits[(model, s)], glass, rng) for s in seeds}
        entry = dict(
            what=MODELS.get(model) or REFERENCE[model][1],
            seeds=seeds,
            parameters=info[(model, seeds[0])]["parameters"],
            hours_per_run=[info[(model, s)]["hours"] for s in seeds],
            found={k: fs.spread([per[s][k]["rate"] for s in seeds]) for k in kinds},
            per_seed={s: {k: round(per[s][k]["rate"], 4) for k in kinds} for s in seeds},
        )
        if model != "egnn_scratch":
            both = [s for s in seeds if ("egnn_scratch", s) in hits]
            entry["against_egnn_control"] = {
                s: fa.paired(hits[("egnn_scratch", s)], hits[(model, s)])["all"] for s in both
            }
        entry["symmetry"] = symmetry(found[(model, seeds[0])], data, device, rng, threads)
        entry["stability"] = {
            s: fs.stability(steps[(model, s)])
            for s in seeds
            if (model, s) in steps and steps[(model, s)].exists()
        }
        out["models"][model] = entry
    # finder against finder with the same seed, and how far they find the same defects
    out["between"], out["agreement"] = {}, {}
    names = [m for m in MODELS if (m, 0) in hits]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = [s for s in range(10) if (a, s) in hits and (b, s) in hits]
            out["between"][f"{a} vs {b}"] = {
                s: {
                    k: dict(only_first=v["only_new"], only_second=v["only_base"], p=v["p"])
                    for k, v in fa.paired(hits[(b, s)], hits[(a, s)]).items()
                }
                for s in both
            }
    for s in range(10):
        if not all((m, s) in hits for m in names) or len(names) < 2:
            continue
        each = np.stack([np.concatenate(list(hits[(m, s)].values())) for m in names])
        out["agreement"][s] = dict(
            finders=names,
            by_all=round(float(each.all(0).mean()), 4),
            by_none=round(float((~each.any(0)).mean()), 4),
            by_at_least_one=round(float(each.any(0).mean()), 4),
        )
    out["together"] = {}
    for name, members in TOGETHER.items():
        if not all(key in scores for key in members):
            continue
        mean = {k: np.mean([scores[key][k] for key in members], 0) for k in data}
        joint = fs.hits(mean, data)
        best = max(members, key=lambda key: np.concatenate(list(hits[key].values())).mean())
        out["together"][name] = dict(
            members=[f"{m}_s{s}" for m, s in members],
            found={k: v for k, v in fa.rates(joint, glass, rng).items()},
            best_member=f"{best[0]}_s{best[1]}",
            against_best_member=fa.paired(hits[best], joint)["all"],
        )
    (out_dir / "architectures.json").write_text(json.dumps(out, indent=1) + "\n")
    md = summary(out, kinds)
    (out_dir / "summary.md").write_text(md)
    print(md)


def summary(out: dict, kinds: list[str]) -> str:
    label = dict(vacancy="Vacancy", interstitial="Interstitial", frenkel="Frenkel pair", antisite="Antisite", all="All")  # fmt: skip

    def pct(x):
        return f"{100 * x:.1f}"

    md = ["<!-- written by scripts/pilots/finder_architectures.py -->", "",
          f"Found on the real test defects ({', '.join(f'{label[k]} {n}' for k, n in out['n'].items())}), "
          "calibrated. Mean over seeds, lowest and highest seed in brackets.", "",
          "| Finder | Parameters | Seeds | " + " | ".join(label[k] for k in kinds) + " |",
          "|---|---|---|" + "---|" * len(kinds)]  # fmt: skip
    for model, e in out["models"].items():
        cells = [f"{pct(e['found'][k]['mean'])} [{pct(e['found'][k]['low'])}, {pct(e['found'][k]['high'])}]" for k in kinds]  # fmt: skip
        md.append(f"| {e['what']} | {e['parameters'] / 1e6:.2f} M | {len(e['seeds'])} | " + " | ".join(cells) + " |")  # fmt: skip
    md += ["", "Against the EGNN control with the same seed (all defects; defects found only by "
           "the model / only by the control, McNemar p):", ""]  # fmt: skip
    for model, e in out["models"].items():
        for s, t in e.get("against_egnn_control", {}).items():
            md.append(f"- {model}, seed {s}: {t['only_new']} / {t['only_base']}, p = {t['p']:.3g}")
    if out.get("between"):
        md += ["", "Finder against finder with the same seed (defects found only by the first / "
               "only by the second, McNemar p; all defects, then each type):", ""]  # fmt: skip
        for pair, per in out["between"].items():
            for s, t in per.items():
                cells = [f"{label[k]} {t[k]['only_first']} / {t[k]['only_second']} (p = {t[k]['p']:.2g})" for k in kinds if k != "all"]  # fmt: skip
                md.append(f"- {pair}, seed {s}: {t['all']['only_first']} / {t['all']['only_second']}, "
                          f"p = {t['all']['p']:.3g}; " + ", ".join(cells))  # fmt: skip
    if out.get("agreement"):
        md += ["", "Do the finders find the same defects (same seed)?", "",
               "| Seed | Found by all | By none | By at least one |", "|---|---|---|---|"]  # fmt: skip
        for s, a in out["agreement"].items():
            md.append(f"| {s} | {pct(a['by_all'])}% | {pct(a['by_none'])}% | {pct(a['by_at_least_one'])}% |")  # fmt: skip
    if out.get("together"):
        md += ["", "Finders used together (the mean of their calibrated probabilities; 95% interval "
               "over glasses; against its best single member: found only together / only by the "
               "member, McNemar p):", "",
               "| Together | " + " | ".join(label[k] for k in kinds) + " | Against best member |",
               "|---|" + "---|" * (len(kinds) + 1)]  # fmt: skip
        for name, e in out["together"].items():
            cells = [pct(e["found"][k]["rate"]) for k in kinds]
            cells[-1] += f" [{pct(e['found']['all']['lo'])}, {pct(e['found']['all']['hi'])}]"
            t = e["against_best_member"]
            md.append(f"| {name} | " + " | ".join(cells) + f" | {t['only_new']} / {t['only_base']}, p = {t['p']:.2g} ({e['best_member']}) |")  # fmt: skip
    md += ["", "| Finder | Same flagged atom: turn by 90 degrees | random shift | renumbering |",
           "|---|---|---|---|"]  # fmt: skip
    for model, e in out["models"].items():
        sym = e["symmetry"]
        md.append(f"| {model} | " + " | ".join(pct(sym[w]) + "%" for w in sym) + " |")
    md += ["", "| Run | Grad. norm median (last 1,000) | 99th pct | Largest | Clipped (last 1,000) | Grad. spikes | Loss spikes | Not finite |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for model, e in out["models"].items():
        for s, st in e["stability"].items():
            g = st["grad_norm"]
            md.append(f"| {model} seed {s} | {g['median']} ({g['median_last_1000']}) | {g['p99']} | {g['largest']} | "
                      f"{pct(st['clipped'])}% ({pct(st['clipped_last_1000'])}%) | {len(st['grad_spikes'])} | "
                      f"{len(st['loss_spikes'])} | {st['not_finite']} |")  # fmt: skip
    return "\n".join(md) + "\n"


def status(args) -> None:
    """Prints, as JSON, the runs that are not finished here or pushed."""
    runs = Path(args.runs)
    print(
        json.dumps(
            [
                f"{m}_s{s}"
                for m in args.models
                for s in args.seeds
                if not finished(runs, f"{m}_s{s}")
            ]
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--models", nargs="+", choices=list(MODELS), required=True)
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    t.add_argument("--runs", required=True)
    t.add_argument("--minutes", type=float, default=600, help="stop all training after this")
    t.add_argument("--min-start", type=float, default=20, help="start no run with less left")
    t.add_argument("--gpus", type=int, default=None, help="parallel runs (default: one per GPU)")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--poll", type=float, default=20)
    t.add_argument("--report-every", type=float, default=30, help="minutes")
    t.add_argument(
        "--set", nargs="*", default=[], help="training overrides, e.g. optim.n_updates=10"
    )
    x = sub.add_parser("export")
    x.add_argument("--runs", required=True)
    x.add_argument("--dest", required=True)
    x.add_argument("--finished-only", action="store_true", help="leave out runs still training")
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--max-examples", type=int, default=None, help="per defect kind (quick checks)")
    e.add_argument("--threads", type=int, default=2)
    st = sub.add_parser("status")
    st.add_argument("--models", nargs="+", choices=list(MODELS), required=True)
    st.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    st.add_argument("--runs", required=True)
    args = ap.parse_args()
    commands = {"train": train, "export": export, "evaluate": evaluate, "status": status}
    commands[args.command](args)


if __name__ == "__main__":
    main()
