# Run Branch A experiments on a remote Linux server

> This guide covers Branch A sequential training and transfer. Its OOF examples use the
> retained `python -m stacking.legacy_oof_cli` interface documented in
> [LEGACY_OOF.md](LEGACY_OOF.md). For the released fusion model, see
> [README.md](README.md).

The runner carries each round's **best model parameters** into the next round,
separately for each backbone and method. Optimizers, schedulers, epoch counts and early
stopping reset between rounds. An interrupted round resumes from that round's latest
checkpoint.

Later rounds are **not independent unseen-generator evaluations**: their held-out
families can have appeared in previous training rounds. OOF generation uses a separate
workflow with independent fold models.

## Environment and launch

`branches_dev/branch_a/run_all_remote.sh` runs three backbones × two methods (`frozen`
and `partial`) × four rounds: **24 training runs and 48 test evaluation outputs**. It
orchestrates commands on the current server.

Preserve `branches_dev/` and `stacking/` under the repository root. Prepare a Python
3.11 environment with
[Branch A requirements](../branches_dev/branch_a/dev/requirements.adaptation.txt) and
GPU-compatible PyTorch. Supply the frames and indexes in
`branches_dev/branch_a/processed/`, full splits in
`branches_dev/branch_a/runs/branch_a/splits_v1.json` and three pinned pretrained
snapshots in `branches_dev/branch_a/runs/branch_a/huggingface/`.

If weights are missing, run `python -m dev download` from `branches_dev/branch_a/` after
resolving model access. The runner defaults to offline loading and CUDA. See the
[remote setup guide](../branches_dev/branch_a/REMOTE_SETUP.md) for environment and smoke
commands.

On the server, start a persistent terminal session:

```bash
tmux new -s vida
cd /path/to/Vida
source /path/to/your/venv/bin/activate
bash branches_dev/branch_a/run_all_remote.sh
```

Detach with Ctrl-B, then D. Reattach with `tmux attach -t vida`. A server job scheduler
or another persistent session manager can also retain the process. The script writes
logs and stops at the first failed stage.

The default output is `branches_dev/branch_a/runs/remote_sequential/` under the
repository root. To select another output location and memory settings:

```bash
BATCH_SIZE=4 ACCUMULATION=2 FRAME_BATCH_SIZE=4 \
  bash branches_dev/branch_a/run_all_remote.sh /workspace/vida-results
```

Resume with the identical command and environment. Keep all hyperparameters, including
microbatch settings, unchanged. Use a new output directory after changing source,
dependencies or configuration.

Completed feature caches are reused; interrupted extraction restarts the unfinished
cache. Training resumes at the next completed epoch, and evaluation can rerun. Linux
`flock` prevents another copy of the runner from writing the same output directory.

## Output and model reconstruction

```text
remote_sequential/
  run_manifest.json             # Configuration, source hashes, runtime versions
  splits.json
  pip-freeze.txt
  source/                       # Reproduction snapshot
  logs/pipeline.log
  huggingface/                  # Original pretrained weights and processors
  frozen/
    cache/<backbone>/
    training/<backbone>/<round>/
      best.pt
      latest.pt
      history.json
      history.csv
      evaluation/
  adapt/
    training/partial/<backbone>/<round>/
      best.pt
      latest.pt
      history.json
      history.csv
      evaluation/
    comparison/
  SHA256SUMS
  COMPLETE
```

Per-run directories also contain plots. The `COMPLETE` marker is created after all
training and evaluations succeed.

Frozen `best.pt` contains the head and training state. Partial checkpoints contain the
trainable encoder changes and head. Both require the original pinned pretrained
backbone; retain `huggingface/` and the selected checkpoints together.

The output omits original videos and dataset PNGs. Reserve storage for full embedding
caches, checkpoints, results and the copied pretrained cache. Training runtime and peak
memory depend on the target machine.

Use validation history for model selection; test metrics report the frozen selection.
The runner does not train OOF fusion. The [legacy OOF workflow](LEGACY_OOF.md) trains
fresh fold heads from an explicitly selected full frozen-feature cache. A previously
trained final head alone cannot supply OOF training scores.

## Transfer results back locally

Run the transfer on the local machine, replacing the account, host, port and remote
path:

```bash
mkdir -p remote-results
rsync -avP -e 'ssh -p 22' \
  USER@HOST:/path/to/Vida/branches_dev/branch_a/runs/remote_sequential/ ./remote-results/
cd remote-results
shasum -a 256 -c SHA256SUMS
```

`rsync -a` preserves Hugging Face symlinks. Retain the original folder hierarchy.
Partial image evaluation still needs the indexed PNGs; frozen cached evaluation uses
matching full caches. Scoring new videos also needs the corresponding preprocessing.

After `COMPLETE` exists, an archive can be created on the server:

```bash
cd /path/to/Vida/branches_dev/branch_a/runs
tar -cf remote_sequential.tar remote_sequential
```

## Portable frozen-model export

From the local repository root, select one frozen backbone/round:

```bash
python -m stacking.legacy_oof_cli export-a \
  --cache remote-results/frozen/cache/clip \
  --checkpoint remote-results/frozen/training/clip/round_1/best.pt \
  --hf-cache remote-results/huggingface \
  --output stacking/runs/export/clip-round_1
```

`stacking.branch_a.load_frozen(path)` returns the frozen encoder, processor and head.

## Partial checkpoint evaluation after transfer

The partial evaluator reconstructs `AdaptedVision` from the checkpoint's `spec` and
loads `cp['adapted_state']`. From `branches_dev/branch_a/`:

```bash
python -m dev.adapt evaluate \
  --root processed --split runs/branch_a/splits_v1.json --round round_1 \
  --checkpoint ../../remote-results/adapt/training/partial/clip/round_1/best.pt \
  --hf-cache runs/branch_a/huggingface \
  --output ../../remote-results/local-evaluation/partial-clip-round_1 --device cpu
```

For a checkpoint saved with automatic precision, evaluation resolves precision for the
selected device. An explicitly saved precision must be supported by that device; FP16
requires CUDA. This command evaluates the model. Training resumption requires the
original experiment environment and identity.
