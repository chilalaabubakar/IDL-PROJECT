"""Choose the guidance weight w for each trained model from validation rows.

    python scripts/select_w.py --config configs/eval/select_w.yaml \\
        runs="[runs/eval/*_wval_seed0_cond_smax1_w2_s100_host_D2, ...]" out=selection.json

Each run directory holds a metrics.json from scripts/evaluate.py, made on validation hosts
(``sample.py split=val``). Its name says which model and weight it is: the part before
``_w<number>_`` (after any timestamp) names the model, the number is w. For every model and
rule of the config, the chosen w and the validation numbers behind it are written to ``out``:

  realism  the most successful w whose local W1 <= w1_max (none: the w with the lowest W1)
  success  the smallest w whose success >= success_min (none: the most successful w)

Rules and thresholds are fixed in the config before results come in, so the test rows run at
the chosen w are a fair test of "choose w per model".
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from glassdiff.utils.config import cli_config

NAME = re.compile(r"^(?:\d{8}-\d{6}Z_)?(?P<model>.+?)_w(?P<w>\d+(?:\.\d+)?)_(?P<rest>.+)$")


def read_rows(runs: list[str]) -> dict[str, list[dict]]:
    """model -> [{w, success, w1, run}] sorted by w."""
    rows: dict[str, list[dict]] = {}
    paths = [p for pattern in runs for p in sorted(glob.glob(str(pattern)))]
    for run in map(Path, paths):
        m = NAME.match(run.name)
        if m is None or not (run / "metrics.json").exists():
            continue
        metrics = json.loads((run / "metrics.json").read_text())
        model = f"{m['model']}_{m['rest']}"
        rows.setdefault(model, []).append(
            {
                "w": float(m["w"]),
                "success": metrics["success_after_relax"][0],
                "success_ci": metrics["success_after_relax"][1:],
                "w1": metrics["local"]["w1_pe_atom"][0],
                "split": metrics.get("split", "test"),
                "run": str(run),
            }
        )
    return {k: sorted(v, key=lambda r: r["w"]) for k, v in sorted(rows.items())}


def choose_realism(rows: list[dict], w1_max: float) -> dict:
    ok = [r for r in rows if r["w1"] <= w1_max]
    if not ok:
        return min(rows, key=lambda r: r["w1"])
    return max(ok, key=lambda r: (r["success"], -r["w"]))


def choose_success(rows: list[dict], success_min: float) -> dict:
    ok = [r for r in rows if r["success"] >= success_min]
    if not ok:
        return max(rows, key=lambda r: (r["success"], -r["w"]))
    return min(ok, key=lambda r: r["w"])


def select(rows: dict[str, list[dict]], rules: dict) -> dict:
    out = {}
    for model, rs in rows.items():
        not_val = [r["run"] for r in rs if r["split"] != "val"]
        if not_val:
            raise ValueError(f"{model}: rows not made on validation hosts: {not_val}")
        chosen = {}
        if "realism" in rules:
            chosen["realism"] = choose_realism(rs, float(rules["realism"]["w1_max"]))["w"]
        if "success" in rules:
            chosen["success"] = choose_success(rs, float(rules["success"]["success_min"]))["w"]
        out[model] = {"chosen_w": chosen, "validation": rs}
    return out


def main() -> None:
    cfg = cli_config(__doc__)
    rows = read_rows(cfg["runs"])
    if not rows:
        raise SystemExit("no validation rows found")
    result = {"rules": cfg["rules"], "models": select(rows, cfg["rules"])}
    print(f"{'model':40s} {'w':>5s} {'success':>8s} {'W1':>7s}")
    for model, r in result["models"].items():
        for row in r["validation"]:
            mark = " ".join(k for k, w in r["chosen_w"].items() if w == row["w"])
            print(f"{model:40s} {row['w']:5g} {row['success']:8.3f} {row['w1']:7.3f}  {mark}")
    if cfg.get("out"):
        Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["out"]).write_text(json.dumps(result, indent=2) + "\n")
        print("->", cfg["out"])


if __name__ == "__main__":
    main()
