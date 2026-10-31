# VIDA Branch A: partial fine-tuning

Partial fine-tuning extends the frozen linear baseline by training the final transformer
blocks, final output LayerNorm and a linear classification head. CLIP, DINOv2 and DINOv3
share the same training and video-evaluation contracts.

Run commands from `branches_dev/branch_a/` with Python 3.11 in `.venv-branch-a`. This is
the directory containing `dev/`, `tests/`, `processed/` and `runs/`. The processed data,
pretrained cache and existing split manifests are transferred separately.

## Setup and supported modes

Follow the [baseline setup](README.md#setup-and-checkpoints) or the
[remote setup guide](../REMOTE_SETUP.md). Install
[requirements.adaptation.txt](requirements.adaptation.txt) for this workflow;
[requirements.adaptation.lock.txt](requirements.adaptation.lock.txt) records the local
dependency versions. Select a PyTorch build compatible with the target runtime.

Pretrained checkpoints, immutable revisions and processors are configured in
[config.json](config.json). After downloading the configured weights, `HF_HUB_OFFLINE=1`
enables offline execution. Downloading DINOv3 requires approved, locally authenticated
Hugging Face access.

| Mode      | Encoder policy                                               | Trainer                            |
| --------- | ------------------------------------------------------------ | ---------------------------------- |
| `frozen`  | All encoder parameters frozen and in evaluation mode         | Cached-feature linear-head trainer |
| `partial` | Final two blocks and final output LayerNorm train by default | Image trainer with a linear head   |

The CLI accepts these two modes. Use the original `python -m dev` commands for the
frozen baseline, including its smoke workflow. The adaptation CLI can also route frozen
training and evaluation through the existing cached-feature trainer.

## Model and sampling policies

| Backbone        | Transformer blocks            | Final normalization           | Representation                  |
| --------------- | ----------------------------- | ----------------------------- | ------------------------------- |
| CLIP ViT-B/16   | `vision_model.encoder.layers` | `vision_model.post_layernorm` | Normalized pooled vision output |
| DINOv2 Base     | `encoder.layer`               | `layernorm`                   | Normalized CLS token            |
| DINOv3 ViT-B/16 | `layer`                       | `norm`                        | Normalized CLS token            |

The configured models have 12 blocks and 768-dimensional representations. Each model
uses an external `Linear(768, 1)` head. Labels are `0=real` and `1=generated`.

In partial mode, the last two blocks, final normalization and head train.
`--unfreeze-blocks` overrides the block count. All other encoder modules stay frozen and
in evaluation mode; evaluation puts the whole model in evaluation mode. The model
reapplies this policy whenever `train()` is called.

Partial training reads RGB PNGs and uses gradient-capable encoder forwards. Each sampled
video draw chooses one of its eight frames, with balanced class and generated-source
exposure by default. Sampling is epoch-seeded; workers receive the selected video/frame
indices. The checkpoint processor runs once per image. No additional image augmentation
is applied.

Ordinary `train` and `train-all` runs start each round from the pinned pretrained
weights with a fresh, seeded head. `--init-checkpoint` explicitly initializes trainable
weights from a compatible earlier checkpoint. The sequential remote runner uses this
option between rounds; its later holdouts are therefore separate from independently
initialized unseen-generator evaluation.

## Configuration

[adaptation_config.json](adaptation_config.json) defines the defaults. Pass
`--config PATH` for a complete alternative configuration, or override individual CLI
options:

| Setting                     | Default | CLI override         |
| --------------------------- | ------- | -------------------- |
| Unfrozen transformer blocks | 2       | `--unfreeze-blocks`  |
| Encoder learning rate       | `1e-5`  | `--partial-lr`       |
| Head learning rate          | `1e-4`  | `--head-lr`          |
| Image batch size            | 8       | `--batch-size`       |
| Gradient accumulation       | 1       | `--accumulation`     |
| Evaluation frame microbatch | 8       | `--frame-microbatch` |
| Maximum epochs              | 20      | `--epochs`           |
| Random seed                 | 42      | `--seed`             |
| Precision                   | `auto`  | `--precision`        |

AdamW uses weight decay `1e-4`, except for biases and LayerNorm parameters. Only
trainable parameters enter the optimizer, once each. The default effective batch size is
`8 × 1 = 8`. Accumulation divides summed losses by the actual sample count in each
window, including a final short window.

Configuration, optimizer-group learning rates and step sample counts are recorded. These
defaults are experimental starting values.

## Evaluation and checkpoint contracts

Validation and tests use all eight frames in configurable microbatches:

```text
video_logit = mean(frame_logits)
video_score = sigmoid(video_logit)
```

Generated videos are positive, with a fixed threshold of 0.5. Metrics include video
AUROC, F1, precision, recall, confusion matrices, per-generator comparisons against
shared real negatives and false-positive rates by real source. Test evaluation is
separate from training and does not select checkpoints.

The first epoch attaining the highest validation video AUROC is the best checkpoint; a
strict increase replaces it. Early stopping separately requires an improvement greater
than `0.0001` over its anchor and stops after five insufficient epochs.
ReduceLROnPlateau uses validation AUROC, factor `0.5`, patience `2` and the same
absolute improvement threshold.

`best.pt` and `latest.pt` retain the trainable encoder weights, head weights, base
checkpoint and revision, experiment identity, optimizer, scheduler, scaler, epoch/global
step, early-stopping state, RNG and sampling state. Frozen weights are reconstructed
from the pinned pretrained model. `latest.pt` embeds the best checkpoint so resumption
can repair `best.pt`.

Resume starts at completed epoch boundaries. A mid-epoch interruption repeats that
epoch. Model, processor, method, split, training configuration, device type and library
versions must match. Keep the original epoch budget and use `--resume` only with an
existing `latest.pt`. Load only trusted local checkpoints.

## Devices and resource reporting

`--device auto` selects CUDA, then MPS, then CPU. CUDA automatic precision uses BF16
when supported, otherwise FP16 with GradScaler. CPU and MPS use FP32 in automatic mode.
`--precision fp32` provides an explicit fallback.

`cost.json` records training time, frames per second, parameter counts and CUDA peak
allocated memory when available. CPU RSS is a process-lifetime high-water mark. Validate
GPU execution and memory use on the target machine; historical CPU smoke measurements do
not measure those paths.

## Small real-data smoke run

First run the frozen smoke command from the [baseline guide](README.md) to create
`runs/branch_a/smoke_splits.json`. After weights are cached, inspect the partial policy
and run its three backbone checks:

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt inspect \
  --mode partial

HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt smoke \
  --mode partial --device cpu
```

Smoke uses two epochs, image batch size 4 and accumulation 2. The historical split has
33 training, 17 validation, 17 seen-test and 17 unseen-test rows: 75 distinct videos and
600 frames, with shared real test negatives. Both held-out generators remain present.

Each backbone runs uninterrupted training and a one-epoch interruption followed by
resumption. The workflow audits gradients and frozen weights, checks best-checkpoint
reload predictions, compares resumed weights/history and runs separate test evaluation.
Existing matching smoke checkpoints are resumed. Use a fresh output directory when
changing the runtime or configuration.

## Train, resume and evaluate one round

Train CLIP on the full split:

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train \
  --mode partial --backbones clip --round round_1 \
  --device cuda --precision auto
```

Resume the same run with unchanged settings:

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train \
  --mode partial --backbones clip --round round_1 \
  --device cuda --precision auto --resume
```

Evaluate its best checkpoint separately:

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt evaluate \
  --round round_1 \
  --checkpoint runs/branch_a_adapt/training/partial/clip/round_1/best.pt \
  --output runs/branch_a_adapt/training/partial/clip/round_1/evaluation \
  --device cuda
```

Evaluation reconstructs the backbone and policy from the checkpoint. For a smoke
checkpoint, also pass `--split runs/branch_a/smoke_splits.json`. For transferred data or
custom locations, supply matching `--root`, `--split` and `--hf-cache`.

## Train and evaluate all rounds

The following command trains three backbones across the four existing rounds, with
independent initialization for each run:

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train-all \
  --mode partial --device cuda --precision auto
```

Choose batch size and accumulation for measured GPU capacity before starting, and retain
the settings when resuming. After training, evaluate each best checkpoint:

```bash
for backbone in clip dinov2 dinov3; do
  for round in round_1 round_2 round_3 round_4; do
    run="runs/branch_a_adapt/training/partial/$backbone/$round"
    HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt evaluate \
      --round "$round" --checkpoint "$run/best.pt" \
      --output "$run/evaluation" --device cuda
  done
done
.venv-branch-a/bin/python -m dev.adapt compare
```

The [remote guide](../REMOTE_SETUP.md#sequential-orchestration) separately documents the
orchestration script that initializes later rounds from earlier best checkpoints.

## Comparison and verification

For the default frozen and partial output directories:

```bash
.venv-branch-a/bin/python -m dev.adapt compare
.venv-branch-a/bin/python -m pytest \
  tests/test_branch_a.py tests/test_adaptation.py -q
```

Comparison verifies split/round identities, pretrained revision, processor,
representation, threshold, prediction IDs, labels, sources and eight-frame aggregation.
Smoke and full splits stay separate. CSV/JSON retain individual round values and
descriptive means; plots show every round and its mean. Compare means only over matching
rounds. The rounds are not independent trials. Unavailable frozen training costs are
recorded as `null`.

The artifact verifier `python -m dev.verify_adaptation` requires the historical
`runs/branch_a_adapt/baseline_snapshot.json`, matching protected baseline files, the
default smoke split and all three partial smoke outputs. The smoke command does not
create that historical snapshot. Use the verifier only when those original provenance
files are available.

## Historical partial smoke measurements

The previous delivery recorded CPU/FP32 partial smoke checks for CLIP, DINOv2 and
DINOv3. It reported exact resumed weights/history, matching best-checkpoint reload
predictions and unchanged frozen encoder weights. These records describe that small
delivery check; this documentation revision does not rerun training.

| Backbone | Trainable parameters | Two-epoch training time | Training frames/s |
| -------- | -------------------: | ----------------------: | ----------------: |
| CLIP     |           14,178,049 |                  3.41 s |             19.37 |
| DINOv2   |           14,181,121 |                  6.23 s |             10.59 |
| DINOv3   |           14,179,585 |                  5.05 s |             13.06 |

These tiny resumed CPU runs exclude validation/reload time and ran alongside other work.
They are not controlled speed benchmarks or generalization results. Full training
records are stored separately under the repository's
[remote-results/](../../../remote-results/) directory.

Historical smoke evidence includes `verification.json`, `parameter_audit_epoch_*.json`,
`reload_verification.json`, `resume_verification.json`, `cost.json` and prediction CSVs
under `runs/branch_a_adapt/`. Those local smoke artifacts are distributed separately.
