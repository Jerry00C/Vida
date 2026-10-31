# VIDA video detection

## Final three-branch fusion

The completed fusion layer, frozen 3–8–1 MLP, training records and final evaluation
are in [stacking/README.md](stacking/README.md). It combines the existing Partial
CLIP, temporal LightGBM and real-reference SVM scores. Fusion uses 5,120 videos,
with separate calibration/threshold sets and 10,000 final-test originals.
The calibrated model can predict from `clip_id,A,B,C` scores without raw videos
or retraining branch models. Native and paired format-control results are included.


Real-versus-AI-generated video classification using pretrained CLIP, DINOv2,
and DINOv3 image encoders. Each video has eight extracted PNG frames. Evaluation
averages frame logits before sigmoid to produce a video score.

Three training methods are implemented: a frozen encoder with a linear head,
partial fine-tuning of the final two transformer blocks and final normalization,
and query/value LoRA adapters with a linear head.

## Repository layout

- `branch_a/dev/`: model loading, adaptation policies, training, evaluation,
  split generation, configuration and dependency files.
- `dev/branch_c/`: real-only motion/temporal anomaly detection, tests, Chinese
  handoff notes and aggregate experiment results.
- `dev/tests/`: Branch A training and adaptation tests.
- `branch_a/tests/`: training and adaptation tests.
- `preprocessing/`: video-to-frame extraction and preprocessing validation.
- `branch_a/REMOTE_SETUP.md`: remote runtime instructions for the separately
  transferred data package. Its historical package audit is not a claim that
  datasets or model weights are included in this Git repository.

Datasets, original videos, PNGs, model weights, embedding caches, checkpoints,
full per-video reports, local environments and transfer archives are intentionally absent.
Branch C includes named aggregate metrics and scientific plots without video frames.
The `.gitignore` allowlist prevents these from being accidentally added.
`remote-results/` contains downloaded metrics, predictions, plots and training
histories. Its `.pt` model checkpoints are stored with Git LFS; install Git LFS
and run `git lfs pull` after cloning to retrieve them. Pretrained encoder weights
are separate and still required to reconstruct the complete models.

Dataset frames, original videos, pretrained model caches, local environments and
transfer archives are excluded. Run the sequential experiment runner with
`bash branch_a/run_all_remote.sh`. Later rounds inherit the preceding round's
best weights, so their generator holdouts are not independent unseen tests.
Do not overwrite the source of an active remote run with this renamed layout;
its saved source identity belongs to the previous layout.

## Branch C

See [Branch C setup and handoff](dev/branch_c/README.md) and its
[aggregate results](dev/branch_c/results/README.md). Download its trained models and
per-video OOF/validation/test scores from the [artifact release guide](dev/branch_c/docs/ARTIFACTS_ZH.md).
These binary artifacts are release assets; raw media, input manifests and caches remain excluded.
Branch C runs in its own
Python 3.12 environment and reads temporally dense video observations. The
8 PNGs used by Branch A are not equivalent motion/frequency input.

The current Branch A four-round protocol and Branch C 28-round protocol differ.
Agree on common grouped manifests, validation roles and OOF folds before fusion;
these historical scores must not be merged directly. The six-development /
two-locked-final-test diagram has not been executed by this Branch C version.

The setup, checkpoint and training instructions below apply to Branch A.

## Setup

Use Python 3.11. From the repository root:

```bash
cd branch_a
python3.11 -m venv .venv-branch-a
source .venv-branch-a/bin/activate
# Install a PyTorch build appropriate for your GPU/runtime, then:
python -m pip install -r dev/requirements.adaptation.txt
python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

Run all model commands from `dev/`. Keep the original processed indexes and
PNG frame paths together under `dev/processed/`, transferred separately.
Original videos are not used by Branch A training or evaluation. Frame extraction and
Run all model commands from `branch_a/`. Keep the original processed indexes and
PNG frame paths together under `branch_a/processed/`, transferred separately.
Original videos are not used by training or evaluation. Frame extraction and
preprocessing validation have separate requirements; see preprocessing/README.md.

## Model weights and splits

Either transfer the existing `runs/branch_a/huggingface/` cache and split manifests
into `branch_a/`, or download the configured pretrained checkpoints and build splits:

```bash
python -m dev download
python -m dev audit --workers 8
python -m dev splits
```

DINOv3 downloading requires approved Hugging Face access and local authentication.
Never commit tokens. The pinned model revisions are in `dev/config.json`.
To reproduce an existing experiment, use its exact split manifests and unchanged
CSV indexes from the separate data package instead of generating new inputs.

## Training

For a small pipeline check with downloaded weights:

```bash
python -m dev smoke --device cuda
python -m dev.adapt smoke --mode partial --device cuda
python -m dev.adapt smoke --mode lora --device cuda
```

For full frozen-head training, first extract full-dataset embeddings:

```bash
for backbone in clip dinov2 dinov3; do
  python -m dev features --backbone "$backbone" --device cuda
done
python -m dev train-all
```

Partial fine-tuning and LoRA use images directly:

```bash
python -m dev.adapt train-all --mode partial --device cuda
python -m dev.adapt train-all --mode lora --device cuda
```

Use `--device cpu` for a CPU-only machine. Choose batch sizes for available memory.
Test evaluation is separate from training. See `branch_a/dev/README.md` and
`branch_a/dev/ADAPTATION.md` for evaluation, resume and comparison commands.
Smoke metrics verify the pipeline and are not full research results.
