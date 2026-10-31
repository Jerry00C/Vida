# Branch B handoff to the internal stacker team

## Delivered signal and scope

Use `raw_margin` from
[`predictions_full_dataset.parquet`](../predictions/predictions_full_dataset.parquet)
(or its CSV counterpart). It is the unsquashed binary LightGBM score; larger
values favor generated video. Join on unique `clip_id`; the
[prediction schema](../predictions/SCHEMA_DESCRIPTION.md) describes all columns.
The [integration example](INTEGRATION_EXAMPLE.py) works without the repository tree.

This is the already-selected `temporal_wavelet` architecture, refitted on **all
32,000 current clips** without role partitioning, validation or early stopping.
The 600-tree booster, Platt calibrator and thresholds use that full corpus.
Predictions are **in-sample**. That is the requested internal stacking scope;
OOF generation or a holdout is **not a delivery prerequisite**. These scores alone
do not establish unseen-data performance. No improvement in generalization is
promised merely because more data were used. Historical Option-1 artifacts are a
different fit; see [provenance](PROVENANCE.md).

## Why the team chose raw_margin

It preserves score differences without the final sigmoid's compression at the
extremes and decouples the stacking interface from the standalone calibration.
It is not itself a known calibrated posterior log-odds or a likelihood ratio.
There is no universal mathematical ban on probability inputs: their utility and
scaling depend on the downstream model. Fixed-feature logistic regression remains
convex in its coefficients whether those features are probabilities or margins;
strict convexity needs additional conditions. An MLP is generally non-convex with
either input. No optimal Bayesian-combination guarantee follows from margin use.

Keep `label`, `family`, `prediction` and `decision` out of the agreed Branch B
machine feature. Consider feature scaling and missingness consistently with the
rest of the team's stacker, without changing the delivered Branch B values.

## Feature meaning and order

The exact order is [`model_config.json#/ordered_feature_names`](../model/model_config.json)
(113 columns; schema hash
`593d44101435a4f3a888ed2f588fe6698f6f19242f8fb3d4c3165a8a12750c4d`).
The view contains `tfft_v2` (17), `local_temporal_v2` (54, no timing diagnostics)
and `twavelet_v2` (42).

- FFT features summarize temporal spectral statistics.
- Local temporal features summarize estimated Farneback optical flow, warping
  residuals, kinematics and drift. Flow is an estimate, not perfect motion recovery;
  residuals can reflect decoding, scene structure, occlusions and flow errors.
- `twavelet_v2` is the implemented **1D temporal db2 DWT, level 2,
  periodization**, after linear detrending along sample index. It operates on
  gray, Sobel-edge and flow-residual patch-mean series on a **16×16 grid of 256
  cells** (8×8 pixel gray/edge cells, 6×6 residual cells). It is not a DWT on
  16×16-pixel patches and not the proposal's new `wave_gru` spatial 2D DWT grid.

## Frozen operating points

Read exact values from [`model/model_config.json`](../model/model_config.json) or
`detector.operating_thresholds`. All FPRs below describe the **training corpus**;
none guarantees 1% FPR on future populations.

| Key | Probability threshold | Margin threshold | Observed real FPR |
|---|---:|---:|---|
| `target_1pct_real_fpr` (default) | 0.025049979809292647 | -4.095739606463252 | Vript 0.01; HD-VG-130M 0.0005; unweighted pooled 0.007625 |
| `pooled_1pct_real_fpr` | 0.015457025337482144 | -4.646689782063401 | Family-weighted pooled 0.01; unweighted pooled 0.01475 |
| `macro_f1` | 0.9196089333917218 | 2.7252040699627056 | Zero real false positives on this corpus; selected by weighted training-corpus macro-F1 |

The default's config `achieved_fpr` is the maximum per-source rate, not the
unweighted pooled rate. At the secondary point Vript is 0.0195 and HD-VG-130M is
0.0005; averaging those real-source rates gives 0.01. The corpus has 12,000 Vript
and 4,000 HD-VG-130M clips, so the ordinary pooled rate differs. Measured values
are preserved in [`AUDIT_REPORT.json`](AUDIT_REPORT.json),
`/operating_points_full_fit_descriptive_only` (referenced by the receipt).

Example from the copied package root:

```python
import json
from pathlib import Path

config = json.loads(Path("model/model_config.json").read_text(encoding="utf-8"))
point = config["operating_thresholds"][config["deployed_threshold_key"]]
# For a valid row; use the probability rule to reproduce saved decisions.
def standalone_decision(row):
    return "GENERATED" if row["calibrated_prob"] >= point["threshold"] else "REAL"
# point["margin_threshold"] is the corresponding margin-space boundary.
# Prefer the detector's own result for live calls and threshold-boundary parity.
```

Do not round a threshold before comparing: even a small rounding change can
change a boundary row's decision. Display rounding is not decision logic.

## Weighting

`compute_training_weights(..., mode="mean_1")` allocates 50% total weight to real
and 50% to generated. The two present real sources receive 25% each; the eight
present generators receive 6.25% each. Within a family weight is inverse to its
clip count; normalized weights have mean 1 and sum 32,000. Thus **the ten families
do not receive equal total weight**. The same weights enter booster loss,
full-data calibration and threshold selection.

## Abstention and missingness

On decoding, QC or feature failures, `infer_video` can return `status="ABSTAIN"`,
`decision="ABSTAIN"`, null scores and `fail_closed=true`. Inspect status before
passing a value to the stacker. Follow the team's missing-feature policy (for
example an imputed constant plus `branch_b_missing=1`, learned consistently by
the stacker). If zero is chosen as that constant it is **only a convention**:
raw margin zero is not known to mean a calibrated 50% prior. Do not replace a
failure with an invented extreme score, or claim an abstention is a real decision.
