# VIDA Branch A adaptation

This extends the frozen linear probe with partial fine-tuning and PEFT LoRA. Outputs go to `runs/branch_a_adapt`; the existing PNGs, exact split manifests, frozen caches and frozen experiment outputs are reused without modification. Run commands from the repository root with Python 3.11 in `.venv-branch-a`.

Install `dev/requirements.adaptation.lock.txt` to reproduce the adaptation environment. It includes PEFT 0.17.1. The original frozen environment lock remains available. Pretrained revisions and processors come from `dev/config.json`. All three checkpoints are already cached locally; `HF_HUB_OFFLINE=1` avoids unnecessary network access. On a new machine, download them first with the existing baseline download command; DINOv3 needs an approved, locally authenticated Hugging Face account.

## Policies

| Backbone | Blocks | Final normalization / feature | Query and value modules |
|---|---|---|---|
| CLIP ViT-B/16 | `vision_model.encoder.layers` | `vision_model.post_layernorm`, pooled vision output | `self_attn.q_proj`, `self_attn.v_proj` |
| DINOv2 Base | `encoder.layer` | `layernorm`, normalized CLS | `attention.attention.query`, `attention.attention.value` |
| DINOv3 ViT-B/16 | `layer` | `norm`, normalized CLS | `attention.q_proj`, `attention.v_proj` |

The downloaded models have 12 blocks and 768-dimensional features. They have separate Q/V projections; combined QKV is rejected. Every model has a fresh external `Linear(768, 1)` head. Label 0 is real, label 1 is generated. Each method/backbone/round starts from the original pinned pretrained weights and a seed-recorded fresh head; no warm start from another round or frozen head occurs.

- `frozen`: the existing cached-feature trainer is preserved. The new CLI can route `--mode frozen` training/evaluation through that trainer; its optimizer defaults remain the baseline defaults.
- `partial`: default last two blocks and final output LayerNorm train, plus the head. All other encoder modules remain frozen and in evaluation mode. Override with `--unfreeze-blocks`.
- `lora`: all 24 Q/V modules receive PEFT adapters, rank 8, alpha 16, dropout 0.05, bias adaptation disabled. Original encoder parameters and original dropout remain frozen/in evaluation mode; only adapter factors/dropout and the head enter training mode. Evaluation disables every dropout. Override rank, alpha and dropout explicitly.

The train/eval policy is reapplied by the shared adapter's `train()` method. Training uses ordinary gradient-capable encoder forwards. Partial and LoRA read actual PNGs, never cached final embeddings. Each sampled video draw chooses one of eight frames, using the existing balanced class/source policy and an epoch-seeded sampler. Workers receive already-selected video/frame indices, so worker scheduling cannot change draws. The checkpoint processor runs once per image; no additional augmentation is applied. Source names are metadata only.

## Configuration and checkpoints

`adaptation_config.json` holds all defaults; use `--config PATH` for a complete configuration or CLI overrides (`--epochs`, `--batch-size`, `--accumulation`, `--frame-microbatch`, `--partial-lr`, `--lora-lr`, `--head-lr`, `--seed`, `--precision`). Default AdamW learning rates are 1e-5 for partial encoder parameters, 1e-4 for LoRA and head; weight decay is 1e-4 except biases/LayerNorm parameters. Only trainable parameters enter the optimizer, once each.

Default image batch 8 × accumulation 1 gives effective batch 8. Loss sums are divided by the actual sample count in each accumulation window, including the final short window. Gradients are cleared between optimizer steps. Configuration, actual group learning rates and step sample counts are recorded. Defaults are experimental starting values, not paper reproduction settings.

Validation and tests use all eight frames in configurable microbatches: `video_logit = mean(frame_logits)` then `video_score = sigmoid(video_logit)`. Generated is positive, with fixed threshold 0.5. Metrics, per-generator comparisons against shared real negatives and real-source false-positive rates retain baseline semantics. Test evaluation is separate from training and never drives selection.

The best checkpoint is the first epoch attaining the highest validation video AUROC (strict increase replaces it). Early stopping separately requires improvement greater than 0.0001 over its anchor and stops after five insufficient epochs. ReduceLROnPlateau uses validation AUROC, factor 0.5, patience 2 and the same absolute improvement threshold.

`best.pt` and `latest.pt` contain adapted encoder/adapter weights, head weights, exact base revision, PEFT config where applicable, experiment identity, optimizer/scheduler/scaler, epoch/global step, early-stop state, RNG and sampling state. Original frozen weights are reconstructed from the pinned base model. `latest.pt` embeds its best checkpoint so resumption can repair `best.pt`. Resume is at completed epoch boundaries, not mid-batch. Changes to model, processor, method, split, training settings, device type or library versions are rejected. Keep the original epoch budget when resuming. Only load trusted local PyTorch checkpoints.

`--device auto` selects CUDA, then MPS, then CPU. CUDA auto precision prefers BF16, otherwise FP16 with GradScaler; CPU/MPS auto uses FP32. Explicit `--precision fp32` is the fallback. CUDA execution, AMP performance and GPU memory measurements require GPU validation; CPU smoke success does not verify those paths. CPU process RSS is a lifetime high-water mark and may include earlier runs in the process. `cost.json` reports training time, training frames/second, parameter counts and CUDA peak allocated memory when available.

## Commands

Inspect exact trainable names, counts and selected modules:

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt inspect --mode partial
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt inspect --mode lora
```

Run the six real smoke workflows (three backbones per command):

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt smoke --mode partial --device cpu
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt smoke --mode lora --device cpu
```

Smoke explicitly uses two epochs, image batch 4 and accumulation 2. The exact existing smoke split has 33 training, 17 validation, 17 seen-test and 17 unseen-test rows: 75 distinct videos / 600 extracted frames, with shared real test negatives. Both held-out generators remain present. Each workflow trains uninterrupted and via a one-epoch interruption, audits gradients and all frozen parameter hashes, verifies best-checkpoint reload predictions, compares resumed weights/history, and invokes the separate evaluation implementation. A subsequent smoke command resumes existing smoke checkpoints instead of overwriting them.

One backbone and round (full data, only run when ready):

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train --mode partial --backbones clip --round round_1 --device cuda --precision auto
```

Resume with the same configuration:

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train --mode partial --backbones clip --round round_1 --device cuda --precision auto --resume
```

Separate best-checkpoint test evaluation:

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt evaluate --round round_1 --checkpoint runs/branch_a_adapt/training/partial/clip/round_1/best.pt --output runs/branch_a_adapt/training/partial/clip/round_1/evaluation --device cuda
```

For a smoke checkpoint, additionally pass `--split runs/branch_a/smoke_splits.json`. Evaluation reconstructs the method/backbone from the checkpoint.

Compare available frozen, partial and LoRA results:

```sh
.venv-branch-a/bin/python -m dev.adapt compare
.venv-branch-a/bin/python -m dev.verify_adaptation
.venv-branch-a/bin/python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

Comparisons verify split/round, checkpoint revision, processor, representation, threshold, exact prediction IDs/labels/sources and eight-frame aggregation. Smoke and full splits are separate. CSV/JSON include individual round values and descriptive means; plots show every round and its mean. Compare means only when listed rounds match. No confidence intervals treat the four rounds as independent trials. Frozen training cost is unavailable and recorded as null, not zero. Smoke scores verify plumbing and are not research conclusions.

## Full experiments (not automatically launched)

These commands train all three backbones across the exact four existing rounds. Adjust batch size/accumulation for measured GPU capacity before starting, retaining those settings for resume:

```sh
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train-all --mode partial --device cuda --precision auto
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt train-all --mode lora --device cuda --precision auto
```

After training, evaluate every best checkpoint:

```sh
for method in partial lora; do
  for backbone in clip dinov2 dinov3; do
    for round in round_1 round_2 round_3 round_4; do
      run="runs/branch_a_adapt/training/$method/$backbone/$round"
      HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -m dev.adapt evaluate --round "$round" --checkpoint "$run/best.pt" --output "$run/evaluation" --device cuda
    done
  done
done
.venv-branch-a/bin/python -m dev.adapt compare
```

## Verified delivery

All six real-data smoke workflows passed on CPU/FP32: partial and LoRA for CLIP, DINOv2 and DINOv3. Every resumed run matched uninterrupted weights and history exactly; optimizer, scheduler, scaler, RNG and sampling states also matched. Best-checkpoint reload predictions matched exactly. Gradient/hash audits confirmed intended updates and unchanged frozen weights. Test exports were independently checked against split IDs, labels, sources, all-eight-frame aggregation and recomputed metrics. All 98 protected preprocessing-index, split, cache and frozen-result files retained their original hashes.

The focused regression suite passed 24 tests, including synthetic examples for all nine backbone/mode combinations, final incomplete accumulation, checkpoint identity rejection and reproducible two-worker image loading. The sandbox initially blocked `torch_shm_manager`; the same suite passed outside that restriction. PEFT emitted three expected missing-config warnings for deliberately synthetic test models; real pretrained models loaded successfully.

| Method | Backbone | Trainable parameters | Two-epoch training seconds | Training frames/s |
|---|---|---:|---:|---:|
| Partial | CLIP | 14,178,049 | 3.41 | 19.37 |
| Partial | DINOv2 | 14,181,121 | 6.23 | 10.59 |
| Partial | DINOv3 | 14,179,585 | 5.05 | 13.06 |
| LoRA | CLIP | 295,681 | 8.70 | 7.58 |
| LoRA | DINOv2 | 295,681 | 10.47 | 6.30 |
| LoRA | DINOv3 | 295,681 | 5.74 | 11.49 |

These are tiny resumed CPU runs, exclude validation/reload time, and ran with other work present; they are not controlled speed benchmarks. CPU process RSS ranged about 1.50–2.35 GiB. No CUDA peak memory was available. Full training, GPU/AMP execution and generalization conclusions remain unverified. FP16 overflow handling allows GradScaler to skip an update and lower its scale; nonfinite FP32/BF16 gradients fail clearly.

Machine-readable evidence: `runs/branch_a_adapt/verification.json`, per-run `parameter_audit_epoch_*.json`, `reload_verification.json`, `resume_verification.json`, `cost.json`, prediction CSVs and `comparison/round_values.csv`. No full-dataset training was launched.
