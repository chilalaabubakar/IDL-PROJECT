"""Task isolation, legacy resume and scheduler-safe process supervision."""

import ast
import fcntl
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("psc", ROOT / "scripts/run_gap2_psc.py")
psc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(psc)


def notebook_sources():
    nb = json.loads((ROOT / "notebooks/kaggle_gap2_generalization.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_assignments_exclude_psc_and_cover_full_study():
    scope = {}
    exec(notebook_sources()[0], scope)
    tasks = scope["TASKS"]
    remaining = set(tasks["kaggle_remaining"])
    standard = set(tasks["psc_1024_standard"])
    fast = set(tasks["psc_1024_fast"])
    assert not remaining & (standard | fast)
    assert remaining | standard | fast == set(tasks["full_study"])
    assert psc.CONDITIONS == ("ka2d_1024_standard", "ka2d_1024_fast")


def test_legacy_restore_copies_only_owned_conditions(tmp_path):
    src = next(s for s in notebook_sources() if "def matches_task" in s)
    header = src[: src.index("def push():")]
    pub = tmp_path / "publisher"
    legacy = pub / "results/kaggle_gap2_generalization_20261019"
    legacy.mkdir(parents=True)
    (legacy / "progress.json").write_text(
        json.dumps(dict(code_revision="frozen-science", batch_size=4, status="incomplete"))
    )
    for condition in [
        "ka2d_256_standard",
        "ka2d_1024_slow",
        "ka2d_1024_standard",
        "ka2d_1024_fast",
    ]:
        d = legacy / "study" / condition
        d.mkdir(parents=True)
        (d / "marker.json").write_text("{}")
    # A different task's newer checkpoint must never be selected as our resume.
    other = pub / "results/kaggle_gap2_psc_1024_fast_20990101"
    other.mkdir()
    (other / "progress.json").write_text(
        json.dumps(dict(task_id="psc_1024_fast", assigned_conditions=["ka2d_1024_fast"]))
    )
    owned = ["ka2d_256_standard", "ka2d_1024_slow", "ka2d_256_slow", "ka2d_256_fast"]
    work = tmp_path / "work"
    scope = dict(
        PUSHROOT=pub,
        WORK=work,
        ROOT=tmp_path / "code",
        REPO="unused",
        TASK_ID="kaggle_remaining",
        CONDITIONS=owned,
        CODE_SHA="current",
        run=lambda *a: "",
        Path=Path,
        json=json,
        shutil=shutil,
        datetime=datetime,
        timezone=timezone,
        time=time,
    )
    exec(header, scope)
    assert scope["CODE_SHA"] == "frozen-science"
    assert (work / "study/ka2d_1024_slow/marker.json").exists()
    assert not (work / "study/ka2d_1024_standard").exists()
    assert not (work / "study/ka2d_1024_fast").exists()
    assert scope["run_name"].startswith("kaggle_gap2_kaggle_remaining_")


def test_gpu_jobs_keep_scheduler_device_visibility():
    waves = psc.stage_jobs("evaluate", Path("/tmp/results"), "ka2d_1024_standard", 8)
    assert [x[0] for x in waves[0]] == ["seed0", "seed1", "hand_insert"]
    assert [x[0] for x in waves[1]] == ["seed2", "sham_seed0"]
    assert "--device" in waves[0][0][1] and "cuda:0" in waves[0][0][1]
    assert "cuda:1" in waves[0][1][1]
    assert "--device" not in waves[0][2][1]
    assert all(
        cmd[cmd.index("--condition") + 1] == "ka2d_1024_standard"
        for wave in waves
        for _, cmd in wave
    )
    assert all(cmd[cmd.index("--batch-size") + 1] == "4" for wave in waves for _, cmd in wave)


def test_supervisor_stops_other_workers_after_failure(tmp_path):
    start = time.monotonic()
    jobs = [
        ("fail", [sys.executable, "-c", "import sys;sys.exit(2)"]),
        ("wait", [sys.executable, "-c", "import time;time.sleep(30)"]),
    ]
    with pytest.raises(RuntimeError, match="stage failed"):
        psc.run_jobs(jobs, tmp_path)
    assert time.monotonic() - start < 10


def test_second_job_cannot_take_same_condition(tmp_path):
    folder = tmp_path / "ka2d_1024_standard"
    folder.mkdir()
    with (folder / ".psc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_gap2_psc.py"),
                "prepare",
                "--condition",
                folder.name,
                "--out",
                str(tmp_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    assert result.returncode != 0 and "Another job already owns" in result.stderr


def test_notebook_has_no_saved_outputs_and_all_cells_parse():
    nb = json.loads((ROOT / "notebooks/kaggle_gap2_generalization.ipynb").read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
            assert c["execution_count"] is None and c["outputs"] == []


def test_all_stage_prepares_then_evaluates():
    waves = psc.stage_jobs("all", Path("/tmp/psc"), "ka2d_1024_fast", 8)
    assert [w[0][0] for w in waves] == ["hosts", "requests", "seed0", "seed2"]


def test_all_stage_rejects_missing_gpus_before_generating_hosts(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gap2_psc.py", "all", "--condition", "ka2d_1024_fast", "--out", str(tmp_path)],
    )
    with pytest.raises(RuntimeError, match="two scheduler-visible GPUs"):
        psc.main()
    assert not (tmp_path / "ka2d_1024_fast/hosts").exists()


@pytest.mark.parametrize("cpu_account", [None, "CPU-123"])
def test_submitter_does_not_require_an_rm_allocation(tmp_path, cpu_account):
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    binpath = tmp_path / "bin"
    binpath.mkdir()
    log = tmp_path / "calls.jsonl"
    sbatch = binpath / "sbatch"
    sbatch.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import json,os,sys\nfrom pathlib import Path\np=Path(os.environ['GAP2_TEST_LOG'])\n"
        + "with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        + "print(1000+len(p.read_text().splitlines()))\n"
    )
    sbatch.chmod(0o700)
    env = dict(
        os.environ,
        PATH=str(binpath) + os.pathsep + os.environ["PATH"],
        GAP2_REPO=str(repo),
        GAP2_ENV=str(tmp_path / "env"),
        GAP2_OUT=str(tmp_path / "results"),
        GAP2_TEST_LOG=str(log),
    )
    env.pop("GAP2_CPU_ACCOUNT", None)
    if cpu_account:
        env["GAP2_CPU_ACCOUNT"] = cpu_account
    result = subprocess.run(
        ["bash", str(ROOT / "slurm/gap2/submit.sh"), "GPU-456"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == (2 if cpu_account else 1)
    gpu = calls[-1]
    assert gpu[gpu.index("-A") + 1] == "GPU-456"
    assert gpu[-1] == "slurm/gap2/evaluate.sbatch"
    if cpu_account:
        assert calls[0][calls[0].index("-A") + 1] == "CPU-123"
        assert "--dependency=afterok:1001" in gpu
    else:
        assert not any("--dependency" in arg for arg in gpu)


@pytest.mark.parametrize("denied", [False, True])
def test_kaggle_publisher_uses_selected_secret_and_explains_denied_write(tmp_path, denied):
    import os
    import tempfile
    import zipfile
    from types import SimpleNamespace

    import numpy as np

    source = next(s for s in notebook_sources() if "def push()" in s)
    tree = ast.parse(source)
    push = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "push")
    work = tmp_path / "work"
    work.mkdir()
    publish = tmp_path / "publish"
    publish.mkdir()
    commands = []
    token = "test-fixture-not-a-real-token"

    def local_run(cmd, *args):
        commands.append(cmd)
        return "progress.json" if "--cached" in cmd else ""

    def remote_run(cmd, **kwargs):
        commands.append(cmd)
        assert cmd[:3] == ["git", "-c", "credential.helper="]
        assert token not in " ".join(cmd)
        assert kwargs["env"]["GAP2_GIT_TOKEN"] == token
        helper = Path(kwargs["env"]["GIT_ASKPASS"])
        assert token not in helper.read_text()
        return SimpleNamespace(
            returncode=1 if denied else 0,
            stderr="remote: permission to repository denied 403 " + token if denied else "",
        )

    scope = dict(
        WORK=work,
        PUBLISH=publish,
        PUSHROOT=tmp_path,
        TOKEN=token,
        STATUS="running",
        CODE_SHA="revision",
        TASK_ID="kaggle_remaining",
        CONDITIONS=["ka2d_1024_slow"],
        run_name="test",
        run=local_run,
        subprocess=SimpleNamespace(run=remote_run),
        torch=SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test")),
        json=json,
        np=np,
        datetime=datetime,
        timezone=timezone,
        tempfile=tempfile,
        os=os,
        Path=Path,
        shutil=shutil,
        zipfile=zipfile,
    )
    exec(compile(ast.Module(body=[push], type_ignores=[]), "<publisher>", "exec"), scope)
    if denied:
        with pytest.raises(RuntimeError, match="Contents: Read and write") as error:
            scope["push"]()
        assert token not in str(error.value)
        assert "[REDACTED]" in str(error.value)
    else:
        scope["push"]()
    pushes = [cmd for cmd in commands if "push" in cmd]
    fetches = [cmd for cmd in commands if "fetch" in cmd]
    assert len(pushes) == len(fetches) == 1
    assert fetches[0][:3] == ["git", "-c", "credential.helper="]
