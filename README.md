# VIDA video detection

VIDA classifies real and AI-generated videos using three complementary branches. Branch
A uses pretrained CLIP, DINOv2 and DINOv3 image encoders with a linear classification
head. Each video has eight RGB PNG frames; frame logits are averaged before sigmoid to
produce the video score.

Branch A supports two training methods:

| Method    | Trainable parameters                                          | Input to training       |
| --------- | ------------------------------------------------------------- | ----------------------- |
| `frozen`  | Linear head only                                              | Cached image embeddings |
| `partial` | Final two transformer blocks, final LayerNorm and linear head | Original RGB PNG frames |

## Final three-branch fusion

The completed fusion implementation, frozen 3–8–1 MLP, training records and final
evaluation are in [stacking/README.md](stacking/README.md). Fusion combines Partial CLIP
mean video logits, temporal LightGBM raw margins and real-reference SVM anomaly scores.

The fusion layer fits on 5,120 videos, with separate calibration and threshold sets and
10,000 final-test originals. Its calibrated model accepts `clip_id,A,B,C` scores.
Producing those scores for new videos requires the exact branch models and
preprocessing. Native and paired format-control results are documented with the fusion
release.

## Repository layout

| Path                                                                           | Contents                                                                                 |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| [branches_dev/branch_a/dev/](branches_dev/branch_a/dev/)                       | Branch A loading, policies, training, evaluation, splits, configuration and dependencies |
| [branches_dev/branch_a/tests/](branches_dev/branch_a/tests/)                   | Branch A baseline and adaptation tests                                                   |
| [branches_dev/branch_a/REMOTE_SETUP.md](branches_dev/branch_a/REMOTE_SETUP.md) | Remote environment, separate data package and training commands                          |
| [branches_dev/branch_b/](branches_dev/branch_b/)                               | Sealed temporal branch package                                                           |
| [branches_dev/branch_c/](branches_dev/branch_c/)                               | Real-only motion and temporal anomaly detection                                          |
| [preprocessing/](preprocessing/)                                               | Frame extraction and preprocessing validation                                            |
| [stacking/](stacking/)                                                         | Fusion implementation, released model and evaluation                                     |
| [remote-results/](remote-results/)                                             | Recorded Branch A training outputs and evaluation artifacts                              |

Original videos, dataset PNGs, processed indexes, pretrained encoder caches, local
environments and transfer archives are distributed separately. Branch C also includes
aggregate metrics and scientific plots without video frames.

`remote-results/` contains metrics, predictions, plots and training histories; its `.pt`
checkpoints use Git LFS. Install Git LFS and run `git lfs pull` after cloning to
retrieve those checkpoints. The pinned pretrained encoder weights remain necessary to
reconstruct complete Branch A models.

Branch B documentation retains historical paths under
`deliverables/branch_b_temporal_wavelet`. Run its relative commands from
`branches_dev/branch_b/`.

## Branch A setup

Use Python 3.11. From the repository root:

```bash
cd branches_dev/branch_a
python3.11 -m venv .venv-branch-a
source .venv-branch-a/bin/activate
# Install a PyTorch build appropriate for this machine before the requirements.
python -m pip install -r dev/requirements.adaptation.txt
python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

Run all Branch A module commands below from `branches_dev/branch_a/` with this
environment active. Keep the transferred indexes and PNG paths together under
`processed/` in that directory. Branch A training and evaluation read the PNGs. Frame
extraction uses a separate environment; see the
[preprocessing guide](preprocessing/README.md).

## Model weights and splits

Transfer the existing `runs/branch_a/huggingface/` cache and exact split manifests into
the Branch A directory. For a new input preparation, download the configured
checkpoints, audit the frames and build splits:

```bash
python -m dev download
python -m dev audit --workers 8
python -m dev splits
```

DINOv3 downloading requires approved Hugging Face access and local authentication. The
pinned revisions are in [dev/config.json](branches_dev/branch_a/dev/config.json).

To reproduce an existing experiment, retain its exact split manifests and unchanged
index CSVs from the separate data package. The code verifies their identities.

## Branch A training and evaluation

After downloading weights, run the small pipeline checks:

```bash
python -m dev smoke --device auto
python -m dev.adapt smoke --mode partial --device auto
```

The first command creates `runs/branch_a/smoke_splits.json`, which the second command
reuses. These runs verify the pipeline; smoke metrics are separate from full experiment
results.

Frozen-head training first requires full-dataset embeddings:

```bash
for backbone in clip dinov2 dinov3; do
  python -m dev features --backbone "$backbone" --device auto
done
python -m dev train-all
```

Partial fine-tuning reads PNGs directly:

```bash
python -m dev.adapt train-all --mode partial --device auto
```

Choose batch size and accumulation for available memory before starting. Training and
test evaluation are separate. See the
[frozen baseline guide](branches_dev/branch_a/dev/README.md) and
[partial fine-tuning guide](branches_dev/branch_a/dev/ADAPTATION.md) for evaluation,
resume, configuration and comparison commands.

These `train-all` commands initialize each round from the pinned base model with a fresh
head. The orchestration script uses a different protocol:

```bash
bash run_all_remote.sh
```

The script runs frozen and partial methods sequentially across four rounds, initializing
later rounds from the preceding round's best weights. Later-round holdouts can therefore
have appeared in earlier training and cannot be reported as independent unseen-generator
tests. The script requires the complete repository layout and a Linux runtime with
`flock`. See the [remote guide](branches_dev/branch_a/REMOTE_SETUP.md) for setup and
output paths. Retain the source and configuration of any active run when resuming it.

## Branch C and protocol alignment

See the [Branch C guide](branches_dev/branch_c/README.md),
[aggregate results](branches_dev/branch_c/results/README.md) and
[artifact release guide](branches_dev/branch_c/docs/ARTIFACTS_ZH.md). Trained models and
per-video OOF, validation and test scores are release assets; raw media, input manifests
and caches remain separate.

Branch C uses Python 3.12 and temporally dense video observations. Branch A's eight
sparse PNG frames do not supply the same motion and frequency input.

Branch A's historical four-round protocol and Branch C's 28-round protocol differ. Those
historical scores require common grouped manifests, validation roles and OOF folds
before they can be combined. Use the separately documented fusion release for the
completed final evaluation.
