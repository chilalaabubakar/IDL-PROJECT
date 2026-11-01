# Run 1024-atom standard and fast cooling on PSC Bridges-2

This splits the existing frozen 2D gap-2 evaluation across Kaggle and PSC. It does not train new models. PSC owns **only** `ka2d_1024_standard` and `ka2d_1024_fast`; Kaggle owns the fresh 256 standard control, 1024 slow, and 256 slow/fast. The three checkpoints, thresholds, host preparation configs, saved-request rule, batch size 4, sampling parameters and free-atom relaxation are unchanged.

## 1. Reserve the conditions before starting PSC

The old Kaggle notebook has all six conditions in its active loop. Updating GitHub does not alter a running kernel.

1. Allow a checkpoint push, then interrupt/stop the old notebook. Preserve its saved outputs.
2. Import the updated `notebooks/kaggle_gap2_generalization.ipynb` into a fresh Kaggle session.
3. Keep `TASK_ID = "kaggle_remaining"` in the first code cell, then Run All.

The new notebook imports compatible checkpoints from the old unscoped run **for its assigned conditions only**, including the partial 1024-slow work. It restores the recorded science-code revision and saves to a new task-specific result folder. It never restores a PSC task as a Kaggle task. If you have already run standard/fast in the live old notebook, save and inspect that work before assigning a second worker to those conditions; the collector refuses conflicting completed results rather than overwriting them.

`progress.json: complete` means the assigned task is complete. The full six-condition study is complete only after collection and `study_status.json: complete`.

## 2. Login and find the allocation/storage paths

Run from your own terminal, using your PSC username and normal PSC authentication:

```bash
ssh PSC_USERNAME@bridges2.psc.edu
projects
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
sinfo -p GPU-shared,RM-shared -o '%P %G %l'
```

The activation notice confirms access but does not supply the Slurm account name. Use the **GPU allocation/account** shown for your account, not your username. `projects` also identifies the Ocean project directory; its directory ID need not be identical to the Slurm account. If `sacctmgr` is restricted, use `projects` and the PSC portal/support to identify the account. Do not send passwords, MFA codes or GitHub tokens in chat.

The default job needs a GPU allocation only: it creates hosts on the allocated GPU node's CPU cores, then evaluates the models. `RM-shared` preparation is optional and requires a corresponding CPU allocation; GPU activation alone does not establish RM access. Partition names and limits below follow the repository's existing Bridges-2 configuration; check the current `sinfo` output and the [PSC guide](https://www.psc.edu/resources/bridges-2/user-guide/), which could not be retrieved from the development environment.

## 3. One-time installation on PSC

Replace `PROJECT_ID` and `PSC_USERNAME` using the paths from `projects`. Use a dedicated checkout and keep it unchanged while jobs run.

```bash
export GAP2_REPO=/ocean/projects/PROJECT_ID/PSC_USERNAME/IDL-PROJECT-gap2
export GAP2_ENV=/ocean/projects/PROJECT_ID/PSC_USERNAME/envs/glassdiff-gap2
export GAP2_OUT=/ocean/projects/PROJECT_ID/PSC_USERNAME/gap2_psc_1024

git clone https://github.com/chilalaabubakar/IDL-PROJECT.git "$GAP2_REPO"
cd "$GAP2_REPO"
bash slurm/gap2/env.sh install
```

The install script creates a new Python 3.11 conda environment through the existing `anaconda3` module, installs PyTorch 2.5.1 with CUDA 12.1 (V100-compatible), and installs this project's `md,dev` dependencies. It refuses to modify an environment that already exists. If using an existing environment, set `GAP2_ENV` to its path and skip `install`; it must provide PyTorch with support for your assigned GPUs and the project/MPICH/LAMMPS dependencies. Conda and virtualenv activation are supported.

`env.sh` discovers `libmpi.so.12` from the MPICH wheel metadata, rather than assuming a prefix. No system MPI module is required. All worker processes use one Torch/OpenMP/MKL thread, matching the existing driver. Installation is not a GPU benchmark; CUDA kernels and LAMMPS initialization are checked in the allocated job before expensive host generation.

PSC's recommended environment differs from the recorded Kaggle Torch/CUDA version. The job records versions, GPU names, science-driver and manifest hashes, Git revision and job IDs in each condition's `psc_*.json`. Hardware/software changes can affect floating-point trajectories; report those differences when interpreting cross-platform comparisons rather than claiming bitwise-identical outputs.

## 4. Submit both reserved conditions

With the three variables above exported, run on the login node:

```bash
cd "$GAP2_REPO"
bash slurm/gap2/submit.sh YOUR_GPU_ALLOCATION
squeue -u "$USER"
```

This submits one two-element GPU job array:

| Array index | Condition | GPUs per job | Work |
|---:|---|---:|---|
| 0 | 1024 atoms, standard cooling (0.01) | 2 V100-32GB | Hosts, requests, seed0/1/2, sham and insertion |
| 1 | 1024 atoms, fast cooling (0.1) | 2 V100-32GB | Hosts, requests, seed0/1/2, sham and insertion |

Each job requests 10 CPU cores, 64 GB RAM and up to 48 hours. Both array elements can run simultaneously if allocation limits and availability permit: **four GPUs total**. To limit usage to two GPUs at once, submit directly with an array concurrency limit:

```bash
mkdir -p slurm/logs "$GAP2_OUT"
sbatch -A YOUR_GPU_ALLOCATION --array=0-1%1 --export=ALL slurm/gap2/evaluate.sbatch
```

The supervisor generates hosts and requests, then runs seed0 and seed1 on logical `cuda:0` and `cuda:1`, with CPU insertion alongside them. It next runs seed2 and the sham. It preserves Slurm's `CUDA_VISIBLE_DEVICES`; logical indices refer only to allocated GPUs. All four trained/sham evaluations use the same saved request file. A per-condition filesystem lock rejects duplicate workers on that output path.

### Optional: use an RM CPU allocation for preparation

Only if your account also has RM access:

```bash
export GAP2_CPU_ACCOUNT=YOUR_CPU_ALLOCATION
bash slurm/gap2/submit.sh YOUR_GPU_ALLOCATION
```

The submitter first queues CPU preparation on `RM-shared` (8 cores per condition) and makes the GPU array depend on its success. GPU jobs detect cached hosts/requests and skip their regeneration. Without `GAP2_CPU_ACCOUNT`, no RM job is submitted.

## 5. Monitor and resume

Slurm output goes to `slurm/logs/gap2-eval-ARRAYID_INDEX.out`. Detailed stage logs and completed atomic checkpoint batches stay under:

```text
$GAP2_OUT/
  ka2d_1024_standard/
    hosts/  requests.npz  seed0/  seed1/  seed2/  sham_seed0/  hand_insert/
    psc_all.json  hosts.log  requests.log  seed0.log ...
  ka2d_1024_fast/
    ...
```

No compute job pushes to GitHub or writes into Kaggle folders. Ocean storage retains the checkpoints between jobs. A scheduler pre-timeout signal terminates supervised child groups and marks the task incomplete. An in-progress batch can need rerunning; previously completed host candidates and batches remain reusable.

Resubmit only the unfinished index, with the **same output root, environment, two visible GPUs, and batch size**:

```bash
sbatch -A YOUR_GPU_ALLOCATION --array=0 --export=ALL slurm/gap2/evaluate.sbatch
# Use --array=1 for fast cooling instead.
```

Do not pull or edit the dedicated code checkout while jobs run. Use a separate checkout for publishing. The jobs are finite and do not automatically resubmit indefinitely. Runtime on PSC is unmeasured; Kaggle timings are not a promised PSC completion time. CPU relaxation remains an important cost even with GPU inference.

## 6. Combine completed PSC and Kaggle results

After PSC jobs finish, use the frozen job checkout and an environment containing the project:

```bash
source slurm/gap2/env.sh
python scripts/collect_gap2.py \
  --source results/kaggle_gap2_generalization_20261019-060130Z "$GAP2_OUT" \
  --out runs/gap2_combined
```

Use the latest task-specific Kaggle result folder in place of the old folder once that run is published. You may pass multiple Kaggle folders if needed. For a current copy of GitHub results, clone/pull a separate publishing checkout and pass its result directories as absolute source paths; do not update the active job checkout.

The collector imports **completed conditions only**, checks request/manifest/checkpoint fingerprints, host counts, matched requests, convergence accounting and force limits, rejects conflicting duplicates, and invokes the original host-aware reporter. Partial raw checkpoints are left in their original folders. `collection_sources.json` records origins, and `study_status.json` says whether all six conditions are present. Collecting two PSC conditions alone does not imply overall completion.

For publication, copy the validated combined folder into `results/` in your separate publishing checkout, review `git status`/diff, fetch and merge current main, commit only the intended result folder, and push without force. Each account should use its own normal GitHub authentication; tokens need not be exported into Slurm jobs.

## Validation status

The supervisor's condition routing, GPU assignment, duplicate-worker lock, failure cleanup, scoped legacy Kaggle restore and notebook syntax are tested locally. The collector has been exercised on the retained complete Kaggle control. The scripts have not been submitted to PSC from this development environment; allocation limits, module availability, CUDA compatibility and performance must be verified by the first allocated job.

## Kaggle checkpoint push fails with GitHub 403

A message `Permission to chilalaabubakar/IDL-PROJECT.git denied` at the initial `push()` check means the experiment has not started in that attempt. The debugger frozen-module warnings are unrelated.

Update the Kaggle `GITHUB_TOKEN` secret with a valid token whose repository access includes `chilalaabubakar/IDL-PROJECT` and whose **Contents permission is Read and write**. For a fine-grained token, choose the correct resource owner and explicitly select this repository. Replace an expired or revoked token. Enable access to the secret in the notebook, then start a fresh session using the latest notebook with `TASK_ID="kaggle_remaining"`. Existing published checkpoints remain available for resume. Never paste the token into code, logs, chat or the repository.

The publisher now resets Git's credential-helper list for authenticated fetch/push commands, so an older cached credential cannot override the selected Kaggle secret. It stops immediately on authentication/permission errors and shows the setup requirement with token values redacted. This cannot grant missing GitHub permissions; the token must actually have write access.
