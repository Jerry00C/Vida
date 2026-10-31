# VIDA Branch A: frozen encoder baseline

Frozen CLIP/DINO image embeddings → one `Linear(D, 1)` layer → raw frame logits. Eight
frame logits are averaged per video **before** sigmoid. Labels are always `0=real`,
`1=AI-generated`. The head has no hidden or activation/dropout layers. There is no extra
image augmentation, fitted feature scaling or threshold tuning.

This module reads `processed/` without changing it. All new artifacts go to
`runs/branch_a/`. The pretrained processor performs its checkpoint-specific
resize/crop/rescale/normalization once, starting from the original RGB PNG.

## Setup and checkpoints

From the repository root, enter `branches_dev/branch_a/`. All module commands in this
guide run from that directory, which contains `dev/` and `tests/`. Use Python 3.11 in a
separate environment from preprocessing:

```bash
cd branches_dev/branch_a
python3.11 -m venv .venv-branch-a
.venv-branch-a/bin/python -m pip install -r dev/requirements.txt
```

[requirements.lock.txt](requirements.lock.txt) records the local environment. Install a
compatible PyTorch build in the new environment before installing the requirements on a
remote GPU machine. CPU and available CUDA/MPS devices are supported for encoder
extraction. Linear-head training uses CPU for small, reproducible matrix operations; it
reads cached embeddings rather than images.

Checkpoints and immutable revisions are explicit in [config.json](config.json):

- CLIP: `openai/clip-vit-base-patch16`, post-LayerNorm vision pooled representation.
- DINOv2: `facebook/dinov2-base`, final LayerNorm CLS token.
- DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`, final LayerNorm CLS token.

The actual normalizer output is hooked and checked against extracted features;
dimensions are checked against the model configuration. Encoder parameters are frozen,
kept in evaluation mode, and checked for absent gradients. They never enter the head
optimizer. The default revisions were pinned when this experiment was prepared; changing
them produces a different cache identity.

```bash
.venv-branch-a/bin/python -B -m dev download
```

DINOv3 is gated: obtain approval on its official model page and authenticate locally
with `.venv-branch-a/bin/hf auth login`. Do not put tokens in code or configuration. The
pipeline never substitutes a different checkpoint. `model_access.json` records
download/access outcomes.

Official implementation references:
[CLIP](https://huggingface.co/openai/clip-vit-base-patch16),
[DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2),
[DINOv3 and access terms](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

## Audit and splits

```bash
.venv-branch-a/bin/python -B -m dev audit --workers 8
.venv-branch-a/bin/python -B -m dev splits
```

Audit checks all eight PNGs per successful clip, full RGB decoding, dimensions, indexed
hashes, labels, uniqueness and frame linkage. Exclusions and reasons are saved,
including an empty exclusions CSV when none fail. Original index hashes are captured and
checked before downstream use. PNG hashes are checked again when extracting features;
cache reuse refers to the audited immutable input snapshot. If inputs change, rerun
audit and create a new split version/cache.

The **provisional**, deterministic, alphabetically adjacent holdout pairs are:

| Round | Held-out generators     |
| ----- | ----------------------- |
| 1     | CogVideo, Mora          |
| 2     | MuseV, OpenSora         |
| 3     | Pika, SVD               |
| 4     | T2V-Zero, VideoCrafter2 |

Pairing is not performance-selected. The eight generators each appear once. The original
delivered splits/folds/validation roles are provenance only in this experiment. New
roles are `train`, `validation`, `test_seen`, `test_unseen`.

Leakage components are transitive unions of existing `group_id`, identical video
SHA-256, and identical ordered eight-frame hash signatures. Source names are not used as
individual groups. `vidprom:*` connects Pika/T2V-Zero/VideoCrafter2 prompt outputs;
`hdvg:*` connects a real video to its conditioned generated counterpart; `vript:*`
connects clips from one parent video. OpenSora only has per-file groups; unknown prompt
relationships cannot be inferred.

Whole components are stratified by their source-count vectors, deterministically
shuffled using seed 42, and assigned approximately 70/15/15. Rounding can depart
slightly from exact percentages. These base assignments, including real clips, are
reused across rounds. All eligible holdout clips go to unseen testing. **Every
non-heldout member of a holdout-connected component is excluded for that round**,
including real clips. Generator isolation overrides target ratios.

Consequently, although six generators are nominally available, round 3 excludes all
T2V-Zero/VideoCrafter2 clips linked to held-out Pika (four effective development
generators), and round 4 excludes Pika linked to held-out T2V-Zero/VideoCrafter2 (five
effective development generators). This is required by the supplied group policy, not an
implementation error or a reassignment of held-out data.

Real test IDs are shared between seen and unseen evaluations within each round. They
appear twice in split tables, once per test role. Related real/generated test components
may span these two test roles; neither intersects training or validation. Reported round
averages are descriptive, not four independent trials.

Artifacts: `audit.json`, `audit.exclusions.csv`, `splits_v1.json`, `splits_v1.csv`,
`splits_v1.exclusions.csv`. The split JSON includes identities, counts, seed, pairs,
base group assignments and exclusions. Rebuilding the same version is deterministic;
changed content requires a new output location/version.

## Small real-data smoke run

```bash
HF_HUB_OFFLINE=1 .venv-branch-a/bin/python -B -m dev smoke \
  --device cpu --frame-batch-size 8
```

Use offline mode only after download. Without it, authenticated Hugging Face access is
used. The smoke creates a small group-disjoint subset of round 1, reuses shared real
test negatives, and includes both held-out generators. Group sizes can increase the
approximate requested counts. All three backbones receive the same subset and selected
frames. For each accessible model it:

1. Reads real PNGs, extracts all eight embeddings per clip, publishes/reloads cache.
2. Trains a fresh head for two epochs and saves validation metrics/plots.
3. Trains a second fresh head for one epoch, resumes it for epoch two, and requires
   exactly equal final head parameters and epoch history to uninterrupted training.
4. Reloads the best checkpoint, evaluates seen/unseen tests, and saves predictions,
   metrics, ROC/confusion/per-generator plots and comparison plots.

All smoke plots and reports are labelled **pipeline verification, not research
results**. `smoke_report.json` records individual success/blocker status and cache
memory/throughput measurements. Errors are not replaced with synthetic results. A
blocked model does not prevent the other smoke runs from executing. Check the report,
since the smoke command intentionally aggregates failures instead of exiting on the
first unavailable model.

Focused synthetic tests are separate:

```bash
.venv-branch-a/bin/python -B -m pytest tests/test_branch_a.py -q
```

## Full experiment commands

Run from `branches_dev/branch_a/` after resolving model access. Feature extraction
caches all eight embeddings for the union of clips across all rounds:

```bash
for backbone in clip dinov2 dinov3; do
  .venv-branch-a/bin/python -B -m dev features \
    --backbone "$backbone" --device auto --frame-batch-size 32 --workers 0
done
```

For one backbone/round:

```bash
.venv-branch-a/bin/python -B -m dev train \
  --cache runs/branch_a/cache/clip --round round_1 \
  --run runs/branch_a/training/clip/round_1

# Resume from the latest completed epoch after interruption.
.venv-branch-a/bin/python -B -m dev train \
  --cache runs/branch_a/cache/clip --round round_1 \
  --run runs/branch_a/training/clip/round_1 --resume
```

For all three backbones and all four rounds, after full feature extraction:

```bash
.venv-branch-a/bin/python -B -m dev train-all --resume
```

Test evaluation is a **separate** command and always loads the best checkpoint:

```bash
for backbone in clip dinov2 dinov3; do
  for round in round_1 round_2 round_3 round_4; do
    .venv-branch-a/bin/python -B -m dev evaluate \
      --cache "runs/branch_a/cache/$backbone" --round "$round" \
      --run "runs/branch_a/training/$backbone/$round"
  done
done
.venv-branch-a/bin/python -B -m dev compare
```

Copy/edit `dev/config.json` for hyperparameters, and pass global `--config PATH`
**before** the subcommand. The baseline uses AdamW with learning rate and weight decay
`1e-4`, batch size 256, at most 30 epochs and early-stopping patience 5. Bias decay is
zero. The highest observed validation AUROC selects the checkpoint; improvement strictly
greater than `0.0001` resets early-stopping patience.

Global `--processed`, `--output`, and `--hf-cache` are configurable. Pass explicit
`--splits PATH`, `--audit PATH`, `--cache PATH`, and `--run PATH` when changing artifact
locations. Processed image paths remain relative to the processed root. Move the
processed tree and Branch A artifacts together; no Mac paths are embedded in
feature/index linkage. The global `--output` option does not automatically change the
default input paths for `--audit`, `--splits`, `--cache-root`, `--cache` or `--run`;
pass those options explicitly when using custom output directories.

## Cache, loaders, resume and reporting contracts

Caches are float32 `.npy` arrays shaped `[videos, 8, embedding_dim]`, memory-mapped for
head training. Clip order and frame order are explicit. Identity includes
checkpoint/revision, feature method, processor settings, original index hashes, selected
input hashes, dimension/dtype and library versions. Completed caches publish atomically;
interrupted extraction leaves no valid cache and restarts extraction. Incompatible or
corrupted caches are rejected. `latest.json` selects the most recently created cache
under each backbone; supply the immutable cache directory directly to pin a particular
full or smoke cache.

Each training draw samples a video with configurable class/generator balancing, then one
of its eight cached frame embeddings. Default exposure is 50/50 real vs generated in
expectation, and equal exposure across effective generated sources. Sampling is with
replacement, so an epoch need not visit every video. No feature array is loaded in full.
Epoch-specific deterministic sampler seeds also make worker count irrelevant to frame
choices. Evaluation processes all eight frames.

Checkpoints contain optimizer, scheduler, epoch, Python/NumPy/Torch RNG states,
configuration and split/cache identities. Resume restarts at the next completed epoch; a
mid-epoch interruption repeats that epoch. Latest checkpoints embed the best state so an
interrupted best-file publication can be repaired. Configuration and data identities
must match. Use a new run directory for a different config; do not concurrently write
the same run/cache from multiple processes. Load only trusted local `.pt` files, as
these include Python RNG objects.

Predictions retain clip ID, round, partition, label, source, all eight frame logits,
mean logit and sigmoid score. Metrics include AUROC, generated-positive
F1/precision/recall at 0.5, confusion matrices, generator recall and AUROC/F1 against
the shared real test set, and false-positive rate by real source. Undefined AUROC on
one-class subsets and zero-denominator metrics are JSON null.

Epoch metrics and four training plots are saved in each run. Evaluation writes
seen/unseen predictions and metrics, combined ROC, separate confusion matrices, and
generator plots. Backbone summaries retain every available round value and report how
many of the four rounds exist; no independent-trial confidence interval is computed.
Never interpret the smoke comparison as a research result.

## Historical implementation checks

The source delivery's 2026-10-17 record reported **32,000 eligible clips and zero
preprocessing exclusions**. All four rotating rounds passed leakage validation, real
base assignments were consistent and original index fingerprints were unchanged. This
section preserves that historical smoke record; the documentation cleanup does not rerun
the experiment.

Round-specific connected-group exclusions were **2,000 / 1,000 / 7,000 / 3,000
clip-round entries**. These are protocol exclusions, distinct from corrupt preprocessing
outputs.

The small real-data check used **75 distinct videos and 600 RGB frames**: 33 training,
17 validation, 17 seen-test and 17 unseen-test entries, including nine real test clips
shared by the two test roles. Both held-out generators were present.

| Encoder | Historical pretrained smoke          | Dimension |      Cache size | CPU extraction |     Throughput |
| ------- | ------------------------------------ | --------: | --------------: | -------------: | -------------: |
| CLIP    | Passed                               |       768 | 1,843,328 bytes |        24.58 s | 24.41 frames/s |
| DINOv2  | Passed                               |       768 | 1,843,328 bytes |        30.43 s | 19.72 frames/s |
| DINOv3  | Passed after approved authentication |       768 | 1,843,328 bytes |        24.63 s | 24.36 frames/s |

The shared smoke process recorded peak RSS of **1,754,152,960 bytes (about 1.63 GiB)**.
A separate DINOv3 process recorded **2,232,647,680 bytes (about 2.08 GiB)**. These are
process-lifetime CPU high-water marks. Extraction timings include local model/processor
loading and cache validation.

The record reported pretrained inference, cache reload, two-epoch head training, video
validation, best-checkpoint reload and separate test evaluation for all three backbones.
Interrupted/resumed training matched uninterrupted head parameters and history exactly,
and encoder parameters received no gradients. Prediction CSVs were independently checked
for metric recomputation, eight-frame aggregation, split/cache alignment and generator
isolation.

To rerun the baseline smoke after preparing the data and model access:

```bash
.venv-branch-a/bin/python -B -m dev download
.venv-branch-a/bin/python -B -m dev smoke --device auto
.venv-branch-a/bin/python -B -m dev.verify_run
```

The default historical report locations are under `runs/branch_a/`:
`verification_report.json`, `smoke_report.json`, `model_access.json` and
`smoke_comparison/`. These local smoke artifacts are separate from Git. Full training
outputs are recorded under the repository's [remote-results/](../../../remote-results/)
directory. Smoke measurements verify the pipeline and do not establish full-data
generalization.

## Round initialization and partial fine-tuning

The ordinary `train` and `train-all` commands initialize a fresh head for each
backbone/round unless `--init-checkpoint` is explicitly supplied. The
[sequential remote runner](../REMOTE_SETUP.md#sequential-orchestration) instead
initializes later rounds from the previous round's best weights. Later-round holdouts
may have appeared in earlier training and cannot be treated as independent
unseen-generator tests.

The second supported method, `partial`, trains the final two transformer blocks, final
output LayerNorm and linear head from original PNGs. See [ADAPTATION.md](ADAPTATION.md)
for the model policy, configuration, smoke checks, checkpoint resumption, evaluation and
full-experiment commands. Its default outputs are under `runs/branch_a_adapt/`.
