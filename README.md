# VIDA video detection

Real-versus-AI-generated video classification using pretrained CLIP, DINOv2,
and DINOv3 image encoders. Each video has eight extracted PNG frames. Evaluation
averages frame logits before sigmoid to produce a video score.

Three training methods are implemented: a frozen encoder with a linear head,
partial fine-tuning of the final two transformer blocks and final normalization,
and query/value LoRA adapters with a linear head.

## Repository layout

- `dev/branch_a/`: model loading, adaptation policies, training, evaluation,
  split generation, configuration and dependency files.
- `dev/tests/`: training and adaptation tests.
- `preprocessing/`: video-to-frame extraction and preprocessing validation.
- `dev/REMOTE_SETUP.md`: remote runtime instructions for the separately
  transferred data package. Its historical package audit is not a claim that
  datasets or model weights are included in this Git repository.

Datasets, original videos, PNGs, model weights, embedding caches, checkpoints,
run reports, local environments and transfer archives are intentionally absent.
The `.gitignore` allowlist prevents these from being accidentally added.

## Setup

Use Python 3.11. From the repository root:

```bash
cd dev
python3.11 -m venv .venv-branch-a
source .venv-branch-a/bin/activate
# Install a PyTorch build appropriate for your GPU/runtime, then:
python -m pip install -r branch_a/requirements.adaptation.txt
python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

Run all model commands from `dev/`. Keep the original processed indexes and
PNG frame paths together under `dev/processed/`, transferred separately.
Original videos are not used by training or evaluation. Frame extraction and
preprocessing validation have separate requirements; see preprocessing/README.md.

## Model weights and splits

Either transfer the existing `runs/branch_a/huggingface/` cache and split manifests
into `dev/`, or download the configured pretrained checkpoints and build splits:

```bash
python -m branch_a download
python -m branch_a audit --workers 8
python -m branch_a splits
```

DINOv3 downloading requires approved Hugging Face access and local authentication.
Never commit tokens. The pinned model revisions are in `branch_a/config.json`.
To reproduce an existing experiment, use its exact split manifests and unchanged
CSV indexes from the separate data package instead of generating new inputs.

## Training

For a small pipeline check with downloaded weights:

```bash
python -m branch_a smoke --device cuda
python -m branch_a.adapt smoke --mode partial --device cuda
python -m branch_a.adapt smoke --mode lora --device cuda
```

For full frozen-head training, first extract full-dataset embeddings:

```bash
for backbone in clip dinov2 dinov3; do
  python -m branch_a features --backbone "$backbone" --device cuda
done
python -m branch_a train-all
```

Partial fine-tuning and LoRA use images directly:

```bash
python -m branch_a.adapt train-all --mode partial --device cuda
python -m branch_a.adapt train-all --mode lora --device cuda
```

Use `--device cpu` for a CPU-only machine. Choose batch sizes for available memory.
Test evaluation is separate from training. See `dev/branch_a/README.md` and
`dev/branch_a/ADAPTATION.md` for evaluation, resume and comparison commands.
Smoke metrics verify the pipeline and are not full research results.
