# Three-branch video stacking

Run commands from **PR2**, not `branch_a/`. This separate package imports the existing
frozen trainer from `branch_a/dev` without modifying it. It combines:

- A: mean of eight frozen appearance-head frame logits.
- B: one temporal-model video logit (GRU or shallow temporal transformer).
- C: one anomaly score, larger meaning more anomalous, from an estimator fitted
  only on real training videos.

B and C are integration interfaces, not implementations of those branches.
They must supply actual predictions before fusion training can run. No synthetic
scores are used outside tests. No full experiment is launched by setup.

## Environment and data

Use the existing Python 3.11 training environment, or install with
`python -m pip install -r stacking/requirements.txt`. Keep datasets, weights and
outputs outside tracked code. Examples put outputs in `stacking/runs/`.

Use the full processed-data split and matching full embedding cache. The old
75-video smoke cache is insufficient for a full experiment. Generate full caches
using the existing `branch_a/dev` feature-extraction commands first.

All branches share clip IDs, group-disjoint splits, labels (0 real, 1 generated)
and round. B may need a separate consecutive-frame sampling pipeline, timestamp
handling and motion descriptors; A's eight uniform samples are not necessarily
consecutive. C's descriptor extraction, scaler and anomaly estimator must obey
its real-only fitting policy. Original videos may be needed for B's preprocessing.

## 1. Freeze the experiment design and generate a fold plan

```bash
python -m stacking folds --split branch_a/runs/branch_a/splits_v1.json \
  --round round_1 --folds 5 --output stacking/runs/round_1/plan.json
```

Each outer fold contains an untouched OOF prediction set. Its remaining groups
are divided into fit and internal selection sets. Early stopping and branch
architecture selection must use only the internal selection set, never the OOF
fold. Small datasets may not support the default folds or internal two-class
split; the command fails rather than allowing leakage.

Pick the appearance backbone before this stacking run using an independent
pilot/development protocol, or implement nested selection within each OOF fold.
This package uses one explicitly selected cache; it does not automatically claim
which backbone is best. Selecting a backbone using all outer training labels
and then claiming strictly held-out OOF model selection is not valid.

The final branch models fit on the original training partition and use the
original validation partition for selection. Both test partitions remain held
out. Shared real videos in seen/unseen tests are preserved as separate role rows.

## 2. Train/export Branch A scores

```bash
python -m stacking branch-a \
  --plan stacking/runs/round_1/plan.json \
  --split branch_a/runs/branch_a/splits_v1.json \
  --cache branch_a/runs/branch_a/cache/clip \
  --output stacking/runs/round_1/A
```

This trains a fresh head per OOF fold, with independent early stopping, plus a
final head. It produces `A/scores.json`, checkpoints and training plots. Training
uses cached frozen features on CPU, as in the original implementation. Specify
`--config PATH` to change head settings and `--resume` to resume an identical run.
The encoder itself stays frozen. Existing dev checkpoints are not overwritten.

## 3. Give B/C teams their exact interfaces

```bash
python -m stacking template --plan stacking/runs/round_1/plan.json \
  --branch B --output stacking/runs/round_1/B-template.json
python -m stacking template --plan stacking/runs/round_1/plan.json \
  --branch C --output stacking/runs/round_1/C-template.json
```

Templates contain every required video and its model assignment. Fill each
`records[].score` with a finite number, and each `models[].artifact_sha256` with
the SHA-256 of the exact saved branch model bundle (including preprocessing).
Keep all identity fields, labels and declared ID lists unchanged. The template
contains null scores and cannot pass validation until filled.

A/B must return raw logits; C returns an anomaly score with larger meaning more
anomalous. For an Isolation Forest, negate a normality-oriented score before
export if necessary. Do not silently mix probabilities and logits. C's fit IDs
contain real videos only. Its selection IDs may include generated videos for
supervised validation selection; they must never be used to fit its anomaly
model or scaler. A completely unsupervised C can ignore selection labels.

All fitted preprocessing, calibration and branch architecture/parameter choices
must respect these boundaries. Metadata checks detect declared leakage, not
undisclosed misuse inside external training code. Model authors must provide
truthful provenance. Validation/test predictions come from `model_id=final`;
training predictions come from their assigned OOF model. No row-order matching.

The Python `stacking.core.aligned(plan, bundles)` function is the fusion data
loader: it joins by `(partition, clip_id)` and returns arrays in fixed A/B/C order.
It rejects missing/extra/duplicate clips, wrong labels/folds, mismatched plans,
nonfinite scores and incompatible fit/selection IDs.

## 4. Validate, fit fusion, then evaluate

Save completed external outputs as `B.json` and `C.json`:

```bash
python -m stacking validate --plan stacking/runs/round_1/plan.json \
  --a stacking/runs/round_1/A/scores.json \
  --b stacking/runs/round_1/B.json --c stacking/runs/round_1/C.json
python -m stacking fit --plan stacking/runs/round_1/plan.json \
  --a stacking/runs/round_1/A/scores.json \
  --b stacking/runs/round_1/B.json --c stacking/runs/round_1/C.json \
  --output stacking/runs/round_1/fusion.json
python -m stacking evaluate --plan stacking/runs/round_1/plan.json \
  --a stacking/runs/round_1/A/scores.json \
  --b stacking/runs/round_1/B.json --c stacking/runs/round_1/C.json \
  --model stacking/runs/round_1/fusion.json \
  --output stacking/runs/round_1/evaluation.json
```

StandardScaler and logistic regression fit only on OOF training scores.
Regularization C is selected by validation AUROC, then validation log loss.
The decision threshold stays 0.5. Fit validates the complete input contract but
never uses test values for scaling, optimization or model selection. Evaluation
is separate and requires the unchanged input bundles used during fit. Metrics
include AUROC, F1 and log loss. Repeat the process independently for every round.

The scaler and classifier are saved as portable numeric JSON (no pickle).
For new videos, obtain aligned A/B/C scores from the exported final models and
call `stacking.fusion.probabilities(model, numpy_array)` with shape `[N, 3]`,
columns A, B, C, and exactly the same score conventions. No labels are needed.

## 5. Save and reload all frozen encoder parameters

```bash
python -m stacking export-a --cache branch_a/runs/branch_a/cache/clip \
  --checkpoint stacking/runs/round_1/A/final/best.pt \
  --output stacking/runs/round_1/export/A
```

The export includes encoder weights (safetensors), model/processor configuration,
linear head weights and a checksummed provenance manifest. It uses the exact
checkpoint revision and processor from the feature cache. Only load trusted
project training checkpoints; these contain optimizer/RNG objects.

```python
from stacking.branch_a import load_frozen
encoder, processor, head = load_frozen('stacking/runs/round_1/export/A')
```

Reload explicitly restores `requires_grad=False` and evaluation mode. CLIP uses
`pooler_output`; DINO uses `last_hidden_state[:, 0, :]`. Apply the head to each
frame feature and average eight logits for A's stacking score. The standalone
A probability is sigmoid of that mean. Keep the encoder in eval mode during use.
B/C teams must supply their own model export/load code. Preserve their final
artifacts, this A export, fusion.json, and the plan for a complete deployment.

## Tests

```bash
python -m pytest stacking/tests -q
```

Tests use synthetic fixtures and tiny local models: group separation, real-only
C fitting, alignment and rejection cases, fusion save/reload, no test influence
on fitting, complete A OOF execution/resume, and frozen-weight export/reload.
