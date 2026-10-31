# Model card — Branch B temporal_wavelet full32k

## Identity and intended use

- Candidate: **`temporal_wavelet`**, already selected in Stage A; tabular LightGBM
  binary booster with **113 inputs and 600 trees**, CPU inference.
- Format: `branch_b_release_v3`; generation: `gen_933b135d79`.
- Calibration: full-data `MarginCalibrator(C=1.0)`, standardized-margin Platt
  logistic calibration. It estimates a probability on the fitted distribution;
  calibration on unseen populations is not established by this fit.
- Signals: `raw_margin` (the team's unsquashed stacking input), `calibrated_prob`,
  `prediction` (0 real / 1 generated), and `decision` (REAL / GENERATED).
  Live inference may instead abstain with null scores.
- Purpose: requested internal ensemble stacking and standalone video inference.
  This is not a statement that the artifact is validated for every deployment.

See [provenance](PROVENANCE.md) and the portable
[evidence record](PROVENANCE_EVIDENCE.json) for sources, field locations and hashes.

## Historical model-comparison scope

Stage A compared a **40-candidate configuration pool (38 eligible + 2 diagnostic
controls)**, not 40 algorithm families or 40 newly fitted full32k models. Primary
development completed 38 configurations; two remained `FAILED_FIT`/incomplete.
The frozen FPR-first seed-42 development rule selected `temporal_wavelet` over
`xgb_wavelet` by **Pika 71/256 versus 70/256**, a one-clip edge, not universal
superiority. Seed 2042 flipped the pairwise rank (`SEED_UNSTABLE`) without changing
the winner. These historical metrics are not current full32k in-sample scores,
and the 40-row API smoke fixture is not 40-model selection evidence.
See the [full selection scope and actual statuses](MODEL_SELECTION_SUMMARY.md)
and [portable evidence](MODEL_SELECTION_SUMMARY.json), including conditional
shortlist limits and source conflicts; no unbiased full nested validation is claimed.

## Corpus and full-fit directive

All **32,000 current clips** participate in booster fitting, calibration and
threshold selection, with no role partitioning, validation or early stopping.
There are 16,000 real and 16,000 generated clips, spanning **8 generator families
and 2 real-source families**:

| Kind | Family | Clips |
|---|---|---:|
| Real | Vript | 12,000 |
| Real | HD-VG-130M | 4,000 |
| Generated | Pika | 3,000 |
| Generated | VideoCrafter2 | 3,000 |
| Generated | T2V-Zero | 3,000 |
| Generated | OpenSora | 3,000 |
| Generated | SVD | 1,000 |
| Generated | MuseV | 1,000 |
| Generated | Mora | 1,000 |
| Generated | CogVideo | 1,000 |

`compute_training_weights(..., mode="mean_1")` assigns real/generated classes
50% total weight each. Within real, the two sources have 25% each of total weight;
within generated, each of eight generators has 6.25%. Weights are constant within
a family and inverse to its count, then normalized to mean 1 (sum 32,000).
They do **not** give all ten families equal total weight. These weights enter
booster fitting, calibration and threshold selection.

The latest user's all-data directive overrides the proposal's historical Option-1
role geometry. This refit is distinct from the historical 22,400-clip, 597-tree
fit. Its delivered predictions are **in-sample**, as requested for internal
stacking; OOF generation and holdout creation are not prerequisites. In-sample
rates cannot quantify future generalization, and full-data fitting does not
promise better unseen-media accuracy.

## Frozen numerical recipe

The architecture/feature view won the comparison, not an independently optimized
set of LightGBM hyperparameters. The plan explicitly reuses
`branch_b/v2/models_v2.py:ModelSpec`, frozen into
`configs/branch_b_v3_final.json`. That historical config includes the
600-iteration cap and 50-round early stopping; the full32k training script sets
early stopping to **0**, supplies no validation data, and realizes 600 trees.

| Parameter | Full32k value |
|---|---:|
| objective | binary |
| n_estimators | 600 |
| learning_rate | 0.05 |
| num_leaves | 31 |
| max_depth | 5 |
| min_child_samples | 20 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| reg_lambda | 1.0 |
| seed | 42 |
| n_jobs | 8 |

The wrapper sets `subsample_freq=1`, deterministic/column-wise LightGBM fitting
and derived bagging/feature/data seeds. `colsample_bytree` controls feature
sampling per tree, not per split. These settings constrain fitting; they do not
prevent overfitting by guarantee. The exact delivered configuration is in
[`model_config.json`](../model/model_config.json).

## Features and calibration

Feature groups: temporal FFT `tfft_v2` (17), estimated-flow/residual/kinematic
`local_temporal_v2` (54, timing diagnostics excluded), and `twavelet_v2` (42).
Exact order and schema hash live in `model_config.json#/ordered_feature_names`.

The implemented temporal transform uses a **1D db2 DWT, level 2,
periodization**, applied along sample index after linear detrending to 256
patch-mean time series arranged on a 16×16 cell grid. Gray and Sobel-edge inputs
use 8×8 pixel cells from 128×128 images; residuals use 6×6 cells from 96×96 flow
residual maps. Energy and distribution summaries produce 14 features for each
of gray, edge and residual signals. This is not the proposal's `wave_gru`
per-frame **spatial 2D DWT** grid and is not a GRU model. Farneback optical flow
is approximate; warping residuals are not proof of physical inconsistency.

The config records calibrator coefficient `10.298891625772269` and intercept
`0.06745986202738435`. The fitted standardizer is serialized in the bundle;
use the detector to apply it. No manually rounded calibration equation or gain
ranking table is maintained here, avoiding a second numerical source of truth.

## Operating points: descriptive fitting-corpus rates only

| Key | Probability threshold | Margin threshold | Real FPR on this corpus |
|---|---:|---:|---|
| `target_1pct_real_fpr` (default) | 0.025049979809292647 | -4.095739606463252 | Vript 0.01; HD-VG-130M 0.0005; unweighted pooled 0.007625 |
| `pooled_1pct_real_fpr` | 0.015457025337482144 | -4.646689782063401 | Family-weighted pooled 0.01; unweighted pooled 0.01475 |
| `macro_f1` | 0.9196089333917218 | 2.7252040699627056 | 0 unweighted pooled; selected on full-data weighted macro-F1 |

These are descriptive **training-corpus rates**, not 1% future-population FPR
guarantees. `achieved_fpr` for the default is the maximum per-source rate;
it is not the pooled unweighted rate. Load exact thresholds from the detector
or config for decisions; never use rounded display values.
See [handoff](HANDOFF_TO_STACKER.md) for a decision example and measurement links.

## Evidence boundaries and limitations

- The [receipt](../DELIVERY_RECEIPT.json) records acceptance of its timestamped
  snapshot; the [audit report](AUDIT_REPORT.json) records exact score replay from
  cached training features, CSV/Parquet parity and portable reference checks.
  Documentation changes require final integration and resealing.
- The original 32,000 videos were not supplied to that audit; full-corpus replay
  was **not** full-corpus video decoding. Synthetic video checks cover only a
  limited decoding/extraction path.
- Live extraction requires at least 8 sampled frames and must satisfy the
  implementation's QC checks. See the config's QC policy and inspect status.
- Static scenes, occlusions, scene cuts, cadence changes and lossy recompression
  can affect these temporal features. Source/generator coverage is limited to
  this corpus; new domains may behave differently.
- Stage A retains its own limitations (including a seed-2042 rank flip and
  conditional shortlist analysis); winner identity was not changed by this refit.
- Trusted joblib loading executes serialized content. SHA-256 checks detect drift
  against trusted digests, not authenticity by themselves. Runtime versions and
  installation are documented in the [quickstart](README.md).
