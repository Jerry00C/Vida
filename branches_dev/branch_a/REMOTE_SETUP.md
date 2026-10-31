# VIDA Branch A: remote setup

Branch A supports a frozen encoder with a linear head and partial fine-tuning of the
final transformer blocks, final LayerNorm and head. This guide describes the transferred
runtime package and remote commands for those two methods.

## Directory layout and data

For module commands, change into the Branch A directory. In a complete checkout it is
`branches_dev/branch_a/`. A separately transferred package can have another name,
provided its contents retain this layout:

```text
branch_a/
  dev/
  tests/
  processed/
    indexes/
    real/
    generated/
  runs/
    branch_a/
      huggingface/
      splits_v1.json
  run_all_remote.sh
```

The original indexes, PNG frames, split manifests and pretrained snapshots are
transferred separately from Git. Preserve index CSV bytes and frame-relative paths. Use
`rsync -a` when transferring a cache with symlinks; the minimal ZIP package described
below stores model snapshots as ordinary files.

The orchestration script additionally requires the complete repository layout; see
[Sequential orchestration](#sequential-orchestration).

## Environment

Create a fresh Python 3.11 environment on the target machine:

```bash
python3.11 -m venv .venv-branch-a
source .venv-branch-a/bin/activate
# Install a PyTorch build appropriate for this GPU/runtime in the active environment.
python -m pip install -r dev/requirements.adaptation.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

Run all remaining module commands from this directory with the environment active.
Create the environment on the target machine rather than transferring a Mac virtual
environment.

[requirements.lock.txt](dev/requirements.lock.txt) and
[requirements.adaptation.lock.txt](dev/requirements.adaptation.lock.txt) record the
local versions. Dependency installation needs network access or a package mirror. The
target GPU still needs a compatible PyTorch build.

## Pretrained models and exact splits

If the transferred package includes all three configured pretrained snapshots:

```bash
export HF_HUB_OFFLINE=1
python -m dev.adapt inspect --mode partial
```

If weights are missing, download them first:

```bash
unset HF_HUB_OFFLINE
python -m dev download
export HF_HUB_OFFLINE=1
```

DINOv3 downloading requires approved Hugging Face access and local authentication. Use
the pinned revisions in [dev/config.json](dev/config.json).

The examples below reuse `runs/branch_a/splits_v1.json` and unchanged
`processed/indexes/` files. To prepare a new input version, run the baseline audit and
split commands from the [baseline guide](dev/README.md). Reproduction of an existing
experiment uses the original manifests.

## Fresh remote smoke checks

Use fresh output folders when changing from historical CPU results to a new GPU runtime.
Partial checkpoint identities include device, precision and library versions.

Create the small split with the frozen smoke command, then explicitly reuse that split
for partial smoke:

```bash
python -m dev --output runs/remote_smoke/frozen smoke \
  --splits runs/branch_a/splits_v1.json \
  --device cuda --frame-batch-size 8

python -m dev.adapt smoke --mode partial \
  --split runs/remote_smoke/frozen/smoke_splits.json \
  --output runs/remote_smoke/adapt --device cuda
```

The frozen command checks three backbones and writes
`runs/remote_smoke/frozen/smoke_report.json`. Inspect each recorded status because that
command aggregates failures. The partial command checks the same three backbones,
including resume, gradient and reload checks, and writes per-backbone
`smoke_verification.json` files.

For a CPU-only machine, replace `cuda` with `cpu`. Both workflows evaluate all eight
frames per video. Smoke metrics verify the pipeline.

## Full experiments with independent initialization

Extract full embeddings into a separate cache before frozen-head training:

```bash
for backbone in clip dinov2 dinov3; do
  python -m dev --output runs/remote_frozen features \
    --splits runs/branch_a/splits_v1.json \
    --backbone "$backbone" --device cuda --frame-batch-size 8
done
python -m dev --output runs/remote_frozen train-all \
  --splits runs/branch_a/splits_v1.json \
  --cache-root runs/remote_frozen/cache
```

Partial fine-tuning reads PNGs directly:

```bash
python -m dev.adapt train-all --mode partial --device cuda \
  --split runs/branch_a/splits_v1.json --output runs/remote_adapt
```

These commands initialize each backbone/round independently. Select batch size and
accumulation for the GPU before starting. Resume partial runs individually with
unchanged settings and `--resume` after confirming that their `latest.pt` exists. The
[partial guide](dev/ADAPTATION.md) lists configuration overrides.

Test evaluation is a separate step:

```bash
for backbone in clip dinov2 dinov3; do
  for round in round_1 round_2 round_3 round_4; do
    python -m dev --output runs/remote_frozen evaluate \
      --splits runs/branch_a/splits_v1.json \
      --cache "runs/remote_frozen/cache/$backbone" --round "$round" \
      --run "runs/remote_frozen/training/$backbone/$round"

    run="runs/remote_adapt/training/partial/$backbone/$round"
    python -m dev.adapt evaluate \
      --split runs/branch_a/splits_v1.json --round "$round" \
      --checkpoint "$run/best.pt" --output "$run/evaluation" --device cuda
  done
done
```

Compare these custom output directories explicitly:

```bash
python - <<'PY'
from dev.adapt_comparison import compare

compare("runs/remote_adapt", frozen_root="runs/remote_frozen")
PY
```

Comparison checks input and evaluation compatibility and writes individual round values
and descriptive means. Do not combine independently initialized runs with sequential
runs in the same result set.

## Sequential orchestration

`run_all_remote.sh` expects its location to be `branches_dev/branch_a/run_all_remote.sh`
within a full checkout. It resolves the repository root from that path and saves
source/configuration provenance. The remote runtime needs Bash, Linux `flock`,
`sha256sum` and the configured Python environment.

From `branches_dev/branch_a/` in that checkout:

```bash
PYTHON="$PWD/.venv-branch-a/bin/python" bash run_all_remote.sh
```

The default output is `runs/remote_sequential/` in the Branch A directory. The script
runs two methods × three backbones × four rounds: 24 training runs and 48 seen/unseen
evaluation outputs. It copies the pretrained cache, saves `run_manifest.json` and source
files, records logs and writes checksums plus a `COMPLETE` marker when finished.

Within each method/backbone, rounds 2–4 initialize from the previous round's best
weights. Their holdout generators can have appeared in earlier training. Report these as
sequential results; later rounds do not estimate independent unseen-generator
generalization.

Environment overrides include `PYTHON`, `DEVICE`, `WORKERS`, `FRAME_BATCH_SIZE`,
`BATCH_SIZE`, `ACCUMULATION`, `PRECISION`, `SPLITS`, `HF_CACHE` and `DATA_ROOT`. The
first positional argument selects the output directory. Resume an interrupted run with
the same command, source, configuration and environment; use a new directory when they
change.

## Historical package checks

The 2026-10-26 packaging record reported:

- All 256,000 indexed frames existed for 32,000 videos.
- Both original index hashes matched the full split manifest.
- All three pinned pretrained snapshots included configuration, processor and weights,
  with no broken cache symlinks.
- Three smoke embedding caches had shape `[75, 8, 768]`.
- The training test modules were included, and all three models/processors loaded
  offline on the source machine.

That record predates the current repository layout and minimal-package cleanup. It
checked frame existence, rather than a new byte-by-byte decode/hash audit of every PNG.
Historical report paths may contain the original Mac location; runtime inputs use the
transferred relative paths. These checks do not certify the new machine's CUDA
execution.

## Historical minimal ZIP package

The 2026-10-27 `dev-remote-minimal.zip` package contained code, configuration,
dependency files, tests, this guide, indexed PNGs, original CSV indexes, full and smoke
split manifests, audit provenance and three pinned pretrained snapshots. It omitted
original videos, old training outputs, embedding caches and Python/pytest caches.

When using that separately supplied archive, extract it into a fresh destination and
enter its `branch_a/` directory:

```bash
unzip dev-remote-minimal.zip
cd branch_a
```

The archive uses ZIP64 without compression. Model snapshots are ordinary files, so
extraction does not require symlink support. Git itself does not include this archive or
its dataset contents.
