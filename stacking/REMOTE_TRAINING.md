# Run all Branch A experiments on a remote Linux server

The runner now carries each round's **best model parameters** into the next
round, separately for each backbone and method. Optimizers, schedulers, epoch
counts and early stopping reset between rounds. Restarting an interrupted round
resumes that round's latest checkpoint. Later rounds are **not independent
unseen-generator evaluations**: their held-out families may have been used in
previous rounds. OOF generation remains a separate, independent-fold workflow.

`run_all_remote.sh` runs 3 backbones x 3 methods x 4 rounds = 36 experiments,
sequentially. It is an orchestration script, not a remote job launcher. Copy the
script to `PR2/branch_a/run_all_remote.sh` on the server; creating it locally does
not update GitHub or the server automatically. Keep `branch_a/` beside `stacking/`.

Activate a Python 3.11 environment with `branch_a/dev/requirements.adaptation.txt`
and GPU-compatible PyTorch already installed. Supply the full dataset frames and
indexes in `branch_a/processed`, full splits in `branch_a/runs/branch_a/splits_v1.json`, and
all three pretrained snapshots in `branch_a/runs/branch_a/huggingface`. Download models
with `python -m dev download` from `branch_a/` first if missing. DINOv3 requires
approved access. The runner defaults to offline model loading and CUDA.

On the remote server, start a persistent terminal session:

```bash
tmux new -s vida
cd /path/to/PR2
source /path/to/your/venv/bin/activate
bash branch_a/run_all_remote.sh
```

Detach with Ctrl-B then D. Reattach with `tmux attach -t vida`. If tmux is not
installed, use the server's job scheduler or another persistent session manager.
The script writes logs itself and stops at the first failed stage.

The default output is `PR2/branch_a/runs/remote_sequential`. To choose another disk/location
and change memory-related settings before starting:

```bash
BATCH_SIZE=4 ACCUMULATION=2 FRAME_BATCH_SIZE=4 \
  bash branch_a/run_all_remote.sh /workspace/vida-results
```

Use the **identical command and environment** to resume after interruption. Keep
all hyperparameters unchanged, including microbatch settings. If changing them
or updating source/dependencies, use a new output directory. Completed feature
caches are reused; interrupted extraction restarts the unfinished cache, and
training resumes at the next completed epoch. Evaluation may rerun on resume.
Do not concurrently write the same output directory. Linux `flock` prevents a
second copy of this runner from doing so.

## Output

```text
remote_sequential/
  run_manifest.json             # Configuration, source hashes, runtime versions
  splits.json
  pip-freeze.txt
  source/                       # Reproduction snapshot of code/configuration
  logs/pipeline.log
  huggingface/                  # Complete original pretrained weights/processors
  frozen/
    cache/<backbone>/           # Full embeddings; useful for later OOF training
    training/<backbone>/<round>/
      best.pt                   # Best validation-AUROC head + training state
      latest.pt                 # Resume checkpoint
      history.json, history.csv, plots...
      evaluation/               # Seen/unseen predictions, metrics and plots
  adapt/
    training/<partial|lora>/<backbone>/<round>/
      best.pt, latest.pt, history..., evaluation/...
    comparison/                 # All methods/rounds, individual values and means
  SHA256SUMS
  COMPLETE                      # Created only after all evaluations succeed
```

The pretrained cache is intentionally saved: frozen-head `best.pt` contains the
head, not encoder weights. Partial/LoRA checkpoints contain trainable changes
and require the original pinned backbone as well. Keeping `huggingface/` plus
their best checkpoints preserves the complete models. No original videos or
PNG dataset are copied into this output directory. Reserve storage for full
embedding caches, checkpoints, results and the copied pretrained cache; the
runner does not estimate a GPU-independent runtime or peak storage requirement.

Selection of a backbone/method should use validation history. Test metrics are
for reporting, not for selecting the winner. No OOF stacking is trained here.
To produce leakage-safe stacking inputs, subsequently run the separate workflow
in stacking/README.md using a predeclared/nested-selected backbone and the full
cache produced here. It trains fresh OOF heads. The existing trained final heads
alone are not an OOF training dataset.

## Bring results back locally

Run this **on your local machine**, adapting user, host, port and remote path:

```bash
mkdir -p remote-results
rsync -avP -e 'ssh -p 22' \
  USER@HOST:/path/to/PR2/branch_a/runs/remote_sequential/ ./remote-results/
cd remote-results
shasum -a 256 -c SHA256SUMS
```

`rsync -a` preserves Hugging Face symlinks. Retain the original folder hierarchy.
Dataset frames are still needed for partial/LoRA image inference/evaluation and
for evaluating new videos; frozen cached evaluation only needs matching caches.
For a single archive after COMPLETE exists, run on the server:

```bash
cd /path/to/PR2/branch_a/runs
tar -cf remote_all.tar remote_all
```

## Portable frozen-model export after transfer

From local PR2, using one selected backbone/round (example: CLIP, round 1):

```bash
python -m stacking export-a \
  --cache remote-results/frozen/cache/clip \
  --checkpoint remote-results/frozen/training/clip/round_1/best.pt \
  --hf-cache remote-results/huggingface \
  --output stacking/runs/export/clip-round_1
```

Then `stacking.branch_a.load_frozen(path)` returns the frozen encoder, processor
and head. For partial/LoRA, reconstruct `AdaptedVision` with the checkpoint's
`spec` and load `cp['adapted_state']`; existing `branch_a.adapt evaluate` already
performs this reconstruction. For example, from local `branch_a/`:

```bash
python -m dev.adapt evaluate \
  --root processed --split ../remote-results/splits.json --round round_1 \
  --checkpoint ../remote-results/adapt/training/lora/clip/round_1/best.pt \
  --hf-cache ../remote-results/huggingface \
  --output ../remote-results/local-evaluation/lora-clip-round_1 --device cpu
```

This is evaluation, not resumption of GPU training on a different device.
Saved precision settings are adapted to the evaluation device by the existing
implementation. Resume requires the original experiment environment/identity.
