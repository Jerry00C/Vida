# Provenance — which proposal, winner and parameters?

## Direct answer

**Proposal:** `reports/revision/04_architecture_selection_proposal.md`, **v2
program, review revision r1**, revised 2026-10-16T13:55:58Z. This is the
architecture-selection/governance proposal, **not evidence of a LightGBM
hyperparameter sweep**. Its current source SHA-256 is
`6a1517af3c6caafd9e796916afd9f8402202db231bc2dc00567ed990f2818011`.

**Winner:** `temporal_wavelet`; **runner-up:** `xgb_wavelet`, established by
`reports/revision/program_v2/STAGE_A_AUDIT.json`,
`/stage_b_authorization/winner` and `/stage_b_authorization/runner_up`.
The primary-seed architecture comparison selected the feature view/learner;
it did not independently optimize a new set of LightGBM hyperparameters.
The audit records a seed-2042 rank flip without changing the primary winner.

**Historical comparison count:** the **40-candidate configuration pool** had
**38 eligible + 2 diagnostic controls**, not 40 algorithm families, 40 numerical
hyperparameter trials or 40 new full32k models. Actual primary development had
314 successful and 6 failed candidate-fold slots; two configurations remained
incomplete. The 40-row API regression is unrelated to that comparison count.
See the self-contained [selection summary](MODEL_SELECTION_SUMMARY.md) and
[portable status/evidence extract](MODEL_SELECTION_SUMMARY.json) for all 40 IDs,
source hashes, validation limits, the one-clip decision and `SEED_UNSTABLE` rank
flip. Conflicting historical completeness/failure claims are disclosed there,
not silently copied or repaired. No new training/search or reselection occurs.

**Where the parameters came from:**

1. `configs/branch_b_experiment_plan_v2.json`, `/candidates/3` (identified by
   `candidate_id="temporal_wavelet"`), says **"existing frozen LightGBM ModelSpec
   (models_v2.ModelSpec) unchanged"**. Its `/budget` has `max_iterations=600`
   and `early_stopping_rounds=50`.
2. Numerical defaults are defined in `branch_b/v2/models_v2.py:ModelSpec` and
   frozen into `configs/branch_b_v3_final.json#/hyperparameters`: 600 estimator
   budget, learning rate 0.05, 31 leaves, depth 5, minimum child samples 20,
   subsample 0.9, column sampling 0.9, L2 lambda 1.0, seed 42, 8 jobs;
   historical early stopping 50.
3. The **delivered full32k recipe** keeps those numerical booster settings but
   disables early stopping and supplies no validation set. See
   `scripts/train_temporal_wavelet_full.py`, `FROZEN_HYPERPARAMETERS` and
   `train_temporal_wavelet_full`, and the delivered
   [`model_config.json`](../model/model_config.json), `/hyperparameters`.

## Historical Option 1 versus this full32k refit

| Property | Historical proposal Option 1 / Stage B | Delivered full32k refit |
|---|---|---|
| Fitting clips | 22,400 | **All current 32,000** |
| Validation / stopping | 2,400 selection clips; 50-round stopping policy | None; early stopping disabled |
| Realized tree count | **597**, `best_iteration=597` | **600**, not a historical best iteration |
| Platt calibration | 2,400 held-calibration clips | Full-data weighted margins, C=1.0 |
| Threshold selection | Held calibration | Full data, family/class weights |
| Final-role geometry | 4,800 read-only clips | No role partitioning |

Historical evidence is
`experiments/gen_933b135d79/stage_b/training_manifest.json`,
`/training_execution/{fit_clips_actual,best_iteration,total_trees_built}`;
proposal §3.3 defines Option 1. Current evidence is the delivered sidecar
[`branch_b_temporal_wavelet_full.json`](../model/branch_b_temporal_wavelet_full.json),
`/provenance/{sample_count,training_partition,realized_epoch_or_iterations}`,
and config `/training_metadata`.

The **latest user's explicit directive** was to use the already-selected
`temporal_wavelet`, fit **ALL current 32,000 clips**, with no partition/leakage
concern, for the internal stacking handoff. This is an operational override of
proposal Option-1 role geometry, **not a claim that r1 originally authorized this
full-data calibration scheme**, and not execution of r1's crossfit Option 2.
The training script and delivered metadata record the implementation of that
directive; the conversation itself is not a separately archived proposal file.
Historical proposal, selection and release artifacts remain intact.

Full32k scores are **in-sample**, as intended; OOF or holdout creation is not a
prerequisite for delivery. Neither their rates nor full-data fitting guarantees
future performance. No retraining, parameter selection or predictive-state
change was performed during this documentation finalization.

## Portable evidence and source interpretation

[`PROVENANCE_EVIDENCE.json`](PROVENANCE_EVIDENCE.json) includes project-relative
source paths, JSON Pointer locations with actual extracted values, code-symbol
line ranges, and **SHA-256 hashes recomputed from actual source bytes**. It
contains enough excerpts to answer the provenance question when only this folder
is copied; repository sources themselves need not be present in that copy.
It is a documentation evidence record, not a replacement for the package seal.
Embedded historical hashes are retained as historical field values, not
substituted for actual source hashes or assumed equal across newline variants.

`twavelet_v2` semantics are supported by
`branch_b/v2/features.py:_compute_1d_twavelet_on_patches` and
`compute_twavelet_v2_features`: detrended **1D temporal db2, level-2 DWT** on a
16×16 grid of patch-mean series, not the new `wave_gru` spatial 2D DWT grid.
Weighting is supported by `branch_b/v2/protocol.py:compute_training_weights`:
real/generated total weight 50% each; two real sources 25% each; eight generators
6.25% each, mean-one loss normalization. The ten families are not equally weighted.

## Reproduction (not part of this handoff)

The existing repository training entry point is
[`scripts/train_temporal_wavelet_full.py`](../../../scripts/train_temporal_wavelet_full.py)
(repository-only link; use the project-relative path to locate it in the original
checkout). It is **not** included as a standalone trainer in this portable folder. From the original repository root, with its
training environment and data available, the frozen-seed command is:

```bash
# Set BRANCHB_SPLITS_DIR to the directory containing the original four CSVs.
# Example output paths are deliberately outside historical release locations.
python -B scripts/train_temporal_wavelet_full.py --seed 42 --n-jobs 8 --out-joblib scratch/full32k_reproduction/model.joblib --out-scores scratch/full32k_reproduction/predictions.parquet
```

Required repository/data inputs include:

- `branch_b/` source tree, `configs/branch_b_experiment_plan_v2.json`,
  `configs/branch_b_v3_final.json`, and
  `reports/revision/program_v2/CONTRACT_FREEZE.json`.
- Original `train.csv`, `validation.csv`, `in_domain_test.csv`, `cross_test.csv`
  manifests via `BRANCHB_SPLITS_DIR` (`branch_b/io.py:load_split`), with clip IDs,
  labels, families, splits and group IDs. The function may derive role metadata,
  but the full-fit script does not use those roles as fitting partitions.
- `artifacts/current/features/gen_933b135d79/tabular_377_diag6.parquet`, aligned
  to those 32,000 clip IDs, plus the same generation's `chunk_*.joblib` records
  containing native23 samples for all 16,000 real clips (local reference fitting).
- A repository training environment satisfying the script's transitive imports;
  the portable runtime pins are not a lock of every training-tool dependency.

Training consumes cached features and manifests, **not a dataset-free model**.
To regenerate those caches, the original video corpus and extraction inputs/code
are also required (media root configured by `BRANCHB_DATA_ROOT`). No raw corpus
is added to this package. The command is documented, not run in this task;
bitwise reproduction across untested platforms is not asserted. Use fresh outputs,
never overwrite historical evidence or this delivered model to run an example.
