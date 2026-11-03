# Complete 1024-atom standard/fast cooling on Google Colab

Open [`notebooks/colab_gap2_1024.ipynb`](../notebooks/colab_gap2_1024.ipynb) directly in [Colab](https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_gap2_1024.ipynb) and save a copy to your Drive.

This assigns both missing conditions to Colab **instead of PSC**. Do not run another worker on those same conditions. The four `kaggle_remaining` conditions are already complete and are not rerun.

## Your 100 compute units

Compute units buy accelerator time at a rate determined by Colab's assigned hardware and account. They are not GPU-hours. The development environment cannot inspect your live balance or rate, and no claim that 100 units guarantees completion is made.

The existing evaluation interleaves fast GPU sampling with long **CPU** FIRE relaxation while keeping a GPU allocated. This notebook separates those computations so host preparation and relaxation can use CPU-only runtimes. The scientific settings are preserved. Total CPU work remains substantial and may require multiple sessions; CPU-only Colab sessions also have availability and runtime limits.

The first saved GPU batch gives a measured sampling estimate. Optionally set `COMPUTE_UNITS_PER_GPU_HOUR` to the rate shown in your Colab Resources panel to convert the estimate to approximate units. The estimate excludes cloning, installation, backup and publishing time and is not a billing guarantee. A faster accelerator can have a higher unit rate; compare the measured rate and speed rather than assuming an A100 is the cheapest choice.

## Three phases in the same notebook and Drive folder

Add `GITHUB_TOKEN` in Colab's key/Secrets panel and enable notebook access. Its GitHub repository permission must be **Contents: Read and write** for `chilalaabubakar/IDL-PROJECT`. The Kaggle secret does not transfer automatically. Enter the token only through Colab Secrets.

| Step | Configuration | Runtime hardware | What runs |
|---|---|---|---|
| 1 | `PHASE = "prepare"` | CPU, accelerator None | 100 new hosts and 256 matched requests per condition |
| 2 | `PHASE = "sample"` | One GPU; start with T4 if available | Seed0, seed1, seed2 and seed0 null-label sham |
| 3 | `PHASE = "relax"` | CPU, accelerator None | Original FIRE minimization, matched insertion, validation and six-condition report |

For each step, use **Runtime → Change runtime type**, set `PHASE`, then **Run all**. Keep `DRIVE_FOLDER`, conditions, batch size and science settings unchanged. Authorize your own Drive mount when Colab requests it. The notebook restores checkpoints after the runtime change; it does not depend on `/content` surviving.

The notebook blocks CPU phases on an attached GPU by default to conserve units. `ALLOW_GPU_FOR_CPU_PHASE=True` is an explicit override for users who prefer to keep a GPU allocated despite CPU work. It is not required for this workflow.

CPU sessions have a 10.5-hour execution budget; GPU sessions have a four-hour budget. The GPU time cap is not a compute-unit cap. Rerun a paused phase to continue. GPU sampling automatically disconnects after its execution block, including planned pauses; the local logs and status are saved first. If automatic disconnect is unavailable, disconnect the runtime manually. You can disable automatic disconnect for debugging, then release the GPU yourself.

## Checkpoints and publication

The default Drive folder is:

```text
MyDrive/IDL-PROJECT/gap2_colab_1024_standard_fast/
```

Completed candidates, generated batches, relaxed batches, logs and status are backed up every five minutes and at stage boundaries. Copies are validated and installed using temporary files; unfinished or invalid archives are skipped. Drive is the primary persistence layer. GitHub is a second copy under:

```text
results/colab_gap2_1024_standard_fast_<UTC timestamp>/
```

The initial GitHub push checks write access before expensive work. Credentials are supplied through a temporary askpass environment, not saved in a URL, notebook, repository or log. If later publishing fails, Drive backup is retained and the phase can continue; the notebook reports the publishing error. With an empty Drive folder, the newest compatible published Colab run can be restored. Keep one active runtime for this assignment/Drive folder.

`run_config.json` pins the code revision and task. Future phases restore that source revision. Raw generated coordinates are preserved under each model's `generated/` directory; relaxation records the input batch's SHA256 and rejects changes to already-relaxed input. Finished aggregates remain immutable on resume.

## Scientific comparability

`scripts/run_gap2_colab.py` uses the same model loader, classifier-free guidance, host initialization, sampler, structures, defect detector and FIRE implementation as `scripts/run_gap2.py`. It preserves:

- The three frozen training checkpoints and original training patch library, with hashes.
- Frozen defect thresholds and the original host/quench settings for each condition.
- Batch size 4, the seed rule `1000+batch_start`, radius 6 and guidance 2; the sham uses NULL labels and no classifier-free amplification.
- 100 noisy and 100 final sampling steps, maximum model noise 1.
- Float32 sampling inputs followed by double-precision CPU relaxation, all atoms free, force tolerance 1e-6 and maximum 100,000 steps.
- Nonfinite generation and nonconverged relaxation counted as failed requests.

Generation and relaxation are separated without regenerating the random inputs between them. Each batch records its sampling or CPU environment. Hardware/library differences can change floating-point trajectories, so mixed environments are not claimed to be bitwise identical. Stored `seconds` adds sampling and CPU relaxation durations rather than measuring a single uninterrupted process; compare its definition when reporting cost.

Local tests compare separated and combined data flow for fixed fixtures, including the sham, partial resume, nonfinite outputs and convergence failures. They also test input-change rejection, backups and the CPU/GPU cost guard. A retained 1024-atom host exercises actual CPU FIRE relaxation. Full Colab GPU execution, Drive authorization and compute-unit consumption have not been validated in this CPU-only development environment.

## Read the final results

After relaxation, `scripts/collect_gap2.py` validates and combines the two new conditions with the four completed Kaggle conditions retained in the source checkout. The collector checks counts, matching requests, manifest/checkpoint fingerprints, force/convergence accounting and conflicting duplicates.

- `progress.json` records phase state and whether the **two-condition Colab task** is complete.
- `combined/study_status.json: complete=true` means the full six-condition placement matrix is complete.
- `combined/placement_summary.csv` reports each seed, their mean, insertion and sham, including the initially-nonD2 denominator and host-bootstrap intervals.
- `combined/paired_contrasts.csv` reports matched method comparisons.

GPU sampling completion alone is not completed placement evaluation. A partial CPU phase can still relax available GPU batches, preserving progress, but final completion requires all 256 outputs for all five methods in both new conditions. Placement transfer does not establish structural equivalence or thermal retention on the new conditions.
