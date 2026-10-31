# Historical model-selection scope — Branch B

## Direct answer: how many models were compared?

**有。历史 Stage A 纳入了 40 个候选配置（38 个可参选、2 个仅诊断对照），
不是 40 种独立算法，也不是本次重新训练了 40 个全量模型。**
The historical **40-candidate configuration pool** was compared on canonical
generation `gen_933b135d79`: **38 eligible to win** plus diagnostic-only
`fixed8_control` and `metadata_only`. This is **not 40 distinct algorithm families**,
**not 40 hyperparameter trials**, and **not 40 new full32k final models**.
Two eligible configurations remained incomplete; pool size is not a count of
successful models. The delivered winner is `temporal_wavelet`; runner-up was
`xgb_wavelet`.

This document and its [portable machine-readable evidence extract](MODEL_SELECTION_SUMMARY.json)
travel with the copied handoff folder. They summarize existing history only:
no new fits, hyperparameter tuning, winner reselection or runtime changes.
The **40-row API regression/smoke fixture tests one delivered detector on 40 input
rows**; it is not evidence of 40-model selection.

## Pool composition and actual primary-development disposition

- **8 pre-existing candidates**, rerun on the canonical generation:
  `corrected_B2`, `B2_no_sfft`, `B2_no_timing_metadata`, `temporal_wavelet`,
  `fixed8_control`, `metadata_only`, `compact_gru`, `compact_tcn`.
- **30 new `wave_gru` configurations**: DWT-only (3), STFT-only (3), DWT+STFT (9),
  each with transform-only or native23+transform inputs: `(3 + 3 + 9) × 2 = 30`.
  Spatial per-frame DWT tags are haar/L3, db4/L3, db4/L4; STFT tags are W16/B4,
  W32/B4, W32/B8 (B is bands, not hop; hop is 1 observed frame). STFT-only
  members are Fourier descriptors despite the common `wave_gru` prefix.
- **2 XGBoost implementation controls**: `xgb_wavelet`, `xgb_b2`.
- Total: `8 + 30 + 2 = 40`; **32 sequence + 8 tabular** (31 GRU, 1 TCN,
  6 LightGBM, 2 XGBoost). The two diagnostic controls cannot win, become
  runner-up, occupy top-6 slots or enter top-2 seed checks.

The table uses the frozen plan's `kind` and `eligible_to_win`, and persisted
`development_selection.json` fields `status`, `is_complete`, `n_expected_runs`,
`n_completed_runs`, cross-checked against stream indexes and actual fit receipts.
"Eligible" means membership in the declared selection pool, not permission for
an incomplete candidate to win. COMPLETE means all eight required primary runs
succeeded; it does not mean complete nested validation or full32k training.
Tabular indexes say `REUSED` with underlying `SUCCESS` receipts; neural receipts
say `OK` or `FAILED_FIT`. Reuse is not a newly executed fit.

| Candidate ID | Family / kind | Eligible | Actual primary-development disposition |
|---|---|---|---|
| `corrected_B2` | LightGBM / tabular_lightgbm | Yes | COMPLETE, 8/8 |
| `B2_no_sfft` | LightGBM / tabular_lightgbm | Yes | COMPLETE, 8/8 |
| `B2_no_timing_metadata` | LightGBM / tabular_lightgbm | Yes | COMPLETE, 8/8 |
| `temporal_wavelet` | LightGBM / tabular_lightgbm | Yes | COMPLETE, 8/8; frozen winner |
| `fixed8_control` | LightGBM / tabular_lightgbm | No: diagnostic | COMPLETE, 8/8; excluded |
| `metadata_only` | LightGBM / tabular_lightgbm | No: diagnostic | COMPLETE, 8/8; excluded |
| `xgb_wavelet` | XGBoost / tabular_xgboost | Yes | COMPLETE, 8/8; runner-up |
| `xgb_b2` | XGBoost / tabular_xgboost | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_haarL3` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_db4L3` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_db4L4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_stftW32B4` | GRU / sequence_gru | Yes, but incomplete | FAILED_FIT, 5/8 OK + 3 failed; cannot win |
| `wave_gru_stftW32B8` | GRU / sequence_gru | Yes, but incomplete | FAILED_FIT, 5/8 OK + 3 failed; cannot win |
| `wave_gru_dwt_stft_haarL3_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_haarL3_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_haarL3_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L3_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L3_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L3_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L4_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L4_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_dwt_stft_db4L4_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_haarL3` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_db4L3` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_db4L4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_haarL3_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_haarL3_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_haarL3_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L3_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L3_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L3_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L4_stftW16B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L4_stftW32B4` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `wave_gru_native_dwt_stft_db4L4_stftW32B8` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `compact_gru` | GRU / sequence_gru | Yes | COMPLETE, 8/8 |
| `compact_tcn` | TCN / sequence_tcn | Yes | COMPLETE, 8/8 |

Thus **38/40 configurations completed primary development**, including the two
non-eligible diagnostics; **36/38 eligible configurations completed**.
For both W32 STFT-only failures, the failed folds are `leave_Pika`,
`leave_VideoCrafter2`, `leave_T2V-Zero`. All six actual `fit_record.json`
receipts say `FAILED_FIT`, `export_sha256=null`, `export_bytes=0`,
`fit_seconds=0.0`; their `error` begins **`Insufficient class support`**, with
`fit_classes=[np.int64(0)], val_classes=[np.int64(0)]`. These are failed required
slots, not successfully trained models. The portable extract includes all six
exact errors, recipe hashes, receipt paths and actual SHA-256 values.

## Historical validation coverage: ceilings versus saved statuses

Primary seed **42** used **eight leave-one-generator-out development runs per
candidate**, group-OOF with held prompt/real mates excluded from training roles.
The bounded recipe capped fit/selection/calibration groups per family at
**256/64/75 before mate union**; these were not full32k fits.
The **28 fixed-candidate outer folds** were descriptive and never used to
reselect architecture, threshold, seed or budget.

| Historical scope | Candidate-fold slots / ceiling | Actual receipt accounting |
|---|---:|---|
| Primary development | up to 40 × 8 = 320 | **314 successful**, 6 FAILED_FIT: tabular 64 SUCCESS (index REUSED), sequence 250 OK + 6 FAILED_FIT |
| Fixed-candidate outer | up to 40 × 28 = 1,120 | **1,084 successful**, 36 FAILED_FIT: tabular 224 SUCCESS (index REUSED), sequence 860 OK + 36 FAILED_FIT |
| Top-6 shortlist inner stability | up to 6 × 6 × 8 = 288 | 204 attempted receipts: **202 SUCCESS + 2 FAILED_FIT**; 84 unsupported candidate-split slots |
| Top-2, extra seeds 1042/2042 | 2 × 2 × (8 + 28) = 144 required extra-seed runs | **144 SUCCESS**, zero failed; task matrix covers all 144 unique tuples; report records 144 reused, 0 newly executed at assembly |

The outer failures are again `wave_gru_stftW32B4` and `wave_gru_stftW32B8`:
18 failed folds each (`fold_02`..`fold_19`), 10 OK each. A CSV containing 28 rows
does not prove 28 successful fits. The portable extract retains each candidate's
outer status counts and failed fold IDs. "Successful" here means a successful
saved fit receipt with model/prediction files present, not a new fit performed by
this documentation task or an independent re-audit of every numerical prediction.

### Top-6 shortlist-conditioned stability (not full nested validation)

The frozen shortlist was:
`temporal_wavelet`, `xgb_wavelet`,
`wave_gru_native_dwt_stft_haarL3_stftW16B4`,
`wave_gru_native_dwt_stft_db4L3_stftW16B4`, `wave_gru_stftW16B4`,
`wave_gru_dwt_stft_haarL3_stftW16B4`.

- Revised evaluated folds: **`fold_03`..`fold_10`**, eight folds.
- Historical `fold_01`/`fold_02`: separate v1-pool/old-generation shortlist
  results, never pooled with the revised eight. This separation concerns the
  shortlist analysis, not the all-40 fixed-candidate outer records.
- **18 unattempted folds**, `fold_11`..`fold_28`, remain
  **`INCOMPLETE_INITIAL_BUDGET`**.
- Of 48 potential inner splits, 34 had declared support and 14 were recorded
  `UNSUPPORTED` / `MISSING_SUPPORT_SHARED_PROMPT` (84 candidate-split slots).
  The 204 attempted receipts are not 204 successful fits: two
  `wave_gru_stftW16B4` runs failed, `fold_07_inner_CogVideo` and
  `fold_08_inner_Mora`, both with `Insufficient class support` errors.
- Eight inner winners' existing outer models were `REUSED_WITH_PROOF`; no extra
  outer fits were needed. The persisted inner winner counts are 30, 24, 24, 24,
  23, 23, 30, 24 successful runs across the eight folds (202 in total).

This is **not unbiased full nested validation**. Global top-6 screening can
observe development outcomes from a generator later held out by an outer fold;
the inner analysis is conditional on that global shortlist. It neither estimates
an unbiased complete 40-candidate nested search nor changes the primary winner.

### Top-2 seed stability

The 144 additional-seed runs have 16 development and 56 outer tasks per seed,
for the fixed top two. Seed 42 is the existing primary evidence, not 72 more
additional runs. Saved matrix `status=SUCCESS` and matching run receipts verify
completion; `task_matrix_summary` records reuse rather than 144 fresh executions
at report assembly.

| Development seed | Pairwise leader | Worst Pika recall: temporal_wavelet / xgb_wavelet | Disposition |
|---|---|---|---|
| 42 | temporal_wavelet | 71/256 / 70/256 | Primary frozen choice |
| 1042 | temporal_wavelet | 70/256 / 68/256 | SEED_STABLE |
| 2042 | xgb_wavelet | 72/256 / 77/256 | **SEED_UNSTABLE**, rank flip |

The seed-2042 rank flip **does not replace the frozen seed-42 winner**. Extra
seeds are diagnostic, outer results descriptive: no favorable-seed picking,
ensembling or winner reselection.

## What decided the winner?

The frozen **FPR-first lexicographic rule**, applied to primary-development
evidence only, is: fewest development per-source FPR exceedances → highest
worst-generator recall → highest mean generator recall → highest mean
standardized AUROC → smallest serialized candidate bytes → candidate ID lexical.
Failed/incomplete required runs disqualify a candidate from winning.

Both leaders had **one source-FPR exceedance**, Vript in `leave_Pika`:
1 FP / 62 real clips (about 1.61%, above 1%). The deciding second tier was
worst-generator recall on **Pika: `temporal_wavelet` 71/256 = 0.27734375 versus
`xgb_wavelet` 70/256 = 0.2734375**. The edge is **one clip**, 0.00390625 recall
(0.390625 percentage points), not universal superiority. XGBoost's higher mean
AUROC did not override the earlier tier. Across development, both leaders had
Vript 1/512 FP (256 independent groups), HD-VG-130M 3/1,024 FP (1,024 groups).
The JSON preserves denominators, group counts, Clopper-Pearson and group-bootstrap
intervals; tiny differences are not strong population evidence.

The numerical LightGBM parameters were a **pre-fixed `ModelSpec`** used by the
architecture/feature-view winner, **not a renewed parameter sweep**. The winner
is the tabular 113-input temporal-wavelet view, not a `wave_gru` spatial-DWT
sequence model. The deferred same-channel aggregated LightGBM control also means
these comparisons cannot attribute gains specifically to GRU recurrence.

## Conflicts in historical sources: preserved, not silently resolved

1. **Failure cause:** `EXPERIMENT_RESULTS_V2.md` and `STAGE_A_AUDIT.md` describe
   CUDA memory/budget failures; `DEVELOPMENT_DECISION.md` describes the W32
   minimum-transition support check. The six actual primary failure receipts
   instead record **insufficient class support after candidate-view support
   filtering**, with only class 0 present. We report those exact recorded errors,
   not an unverified memory diagnosis.
2. **Shortlist successes:** the shortlist report/completion claims 204 fits and
   zero failures, but the 204 persisted `result_info.json` records contain
   **202 SUCCESS and 2 FAILED_FIT**. The saved per-fold successful counts already
   sum to 202. Both failure extracts are included in the portable JSON. We do
   not repair canonical reports or claim every shortlisted fit succeeded.
3. **Completeness status:** the exit audit says `STAGE_A_PASS`, while persisted
   `development_selection.json#/is_completeness_gate_passed` is **false** and
   `selection_freeze.json#/completeness_gate/passed` is **false**. The later audit
   treats the two incomplete candidates under D-05 disposition; it did not turn
   their failed runs into successes. We retain both facts, without re-authorizing
   a gate or calling the audit's **1,794** accounted fit slots 1,794 successful
   or newly executed fits. That total also includes six historical stress slots,
   not six additional candidate configurations.
4. **Ranking universe:** the plan's
   `/evaluation/fixed_evaluation_universe/common_support_ranking` calls common
   support primary. The frozen saved ranking instead records the leaders' Pika
   counts 71/256 and 70/256; `DEVELOPMENT_DECISION.json` separately records
   `candidate_details/common_support_statistics` on **1,718** development clips
   versus the **3,584**-clip development universe (32k common support: 18,195).
   Those separate common-support diagnostics are not the headline counts. This
   documentation preserves the actual frozen result and discloses the mismatch;
   it does not rerank or claim exact implementation of every plan statement.

The corpus was retrospective; role-seed discrepancy (declared 20261015 versus
materialized 42), limited support and source/acquisition confounding remain
limitations. There is no untouched-test, new-generator population guarantee or
unbiased complete-search performance claim. Pre-registration does not remove
selection optimism/winner's curse over 38 eligible configurations.

## Separate history from the current requested deliverable

These historical bounded-development metrics are **not the new full32k
in-sample training scores**. The current handoff refits **only the already-selected
`temporal_wavelet` on all 32,000 clips, 600 iterations, no validation/early stopping**,
with full-data calibration and thresholds. No other candidate is newly fitted
here. The latest full-data/internal-stacking directive supersedes the proposal's
historical Option-1 partition geometry; it does not retroactively change Stage A.
**No new OOF or holdout requirement is added.** In-sample scores are provided for
the requested internal use, not as estimates of future generalization.

## Portable evidence and repository-relative provenance

The JSON companion contains the 40 IDs/kinds/eligibility/dispositions,
per-development-fold status maps, outer status counts/failed IDs, exact failure
extracts, top-6 IDs/fold scopes, extra-seed task coverage and deciding counts.
Its `sources` entries carry **actual-byte SHA-256** values and selected
`json_pointer_values`, including contradictory fields. Paths below and all
`repository_relative_path` / `index_source` fields are **repository-relative
provenance, not package-local links**; no source checkout is needed to read the
embedded extracts. Historical embedded digests are not substituted for current
source-byte hashes (newline variants may differ).

Representative source SHA-256 references:

| Repository-relative source | SHA-256 of actual source bytes |
|---|---|
| `reports/revision/04_architecture_selection_proposal.md` (r1) | `6a1517af3c6caafd9e796916afd9f8402202db231bc2dc00567ed990f2818011` |
| `configs/branch_b_experiment_plan_v2.json` | `f48cd7a146e557837302dbb70c624b42cc70ba977f8746db87e7d64a741b559a` |
| `artifacts/current/experiments/program_v2/gen_933b135d79/development_selection.json` | `70cb96905af8585e2150502774a7d66a000e3d9e84f842f6e49e68912c4682c4` |
| `reports/revision/program_v2/STAGE_A_AUDIT.json` | `01969fab2070be72c311b90c4ee7b8dedbf2d69afbcd001d1ebc904fe54e92db` |
| `reports/revision/program_v2/SHORTLIST_STABILITY_REPORT.json` | `187619f9572c905cdbb3565340263fbf44f6d7741d53114e8b9f57d6bac543e3` |
| `reports/revision/program_v2/MULTISEED_STABILITY_REPORT.json` | `5dc3b498c15b0d9959283c588e61fc66d4f0779f9a0be6b49ffdc0b2836d45b6` |

Receipt locations in that repository:

- Development/outer tabular indexes: under
  `artifacts/current/experiments/program_v2/gen_933b135d79/{development,outer}/tabular/summary/`,
  `development_tabular_index.json` / `outer_tabular_index.json`, field `fits[]`;
  underlying `runs/r/<recipe_hash first 12>/result.json`, field `status=SUCCESS`.
- Sequence indexes: corresponding `{development,outer}/sequence/summary/index.json`,
  field `fits` keyed by candidate/fold; `runs/<recipe_hash>/fit_record.json`,
  fields `status`, `error`, `export_sha256`, `export_bytes`.
- Shortlist: `shortlist/summary/nested_selection.json`, `completed[]`, plus
  `shortlist/runs/<recipe_hash>/result_info.json`, fields `status`, `candidate`,
  `run_id`, `error`. Both failed receipts are embedded verbatim in the JSON.
- Seeds: `multiseed/summary/multiseed_144_hash_matrix.json`, rows with
  `seed`, `candidate_id`, `fold_type`, `fold_id`, `recipe_hash`, `status`,
  `model_sha256`; underlying `multiseed/runs/r/<recipe_hash first 12>/result.json`.

All locations above share the same repository generation prefix. For each stream,
`receipt_ledger_sha256` hashes UTF-8 compact JSON of sorted `[repository path,
actual receipt SHA-256]` pairs. It binds the inspected receipt set, not a model
seal; successful model/prediction files were checked for presence only, not
loaded/refitted or independently rescored. Tests in the original repository,
`tests/test_temporal_wavelet_selection_scope.py`, rebuild this evidence read-only.

This documentation addition deliberately does **not** regenerate checksums,
receipts, notebook outputs or existing seals. Downstream
`integrated_final_acceptance_and_reseal` owns integration, acceptance and resealing.
