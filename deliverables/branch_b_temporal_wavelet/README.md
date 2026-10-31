# Branch B — temporal_wavelet, full 32,000-clip delivery

**Start here:** the already-selected `temporal_wavelet` architecture, refitted on
**all current 32,000 clips**, with no role partitioning, validation set or early
stopping. The package includes a 600-tree LightGBM booster, full-data Platt
calibration, thresholds, and in-sample predictions for internal team stacking.
No OOF generation or holdout is a prerequisite for this requested handoff.

**Historical comparison scope:** Stage A compared a **40-candidate configuration
pool: 38 eligible + 2 diagnostic controls**, not 40 algorithm families or 40 new
full32k models. Primary development completed 38 configurations; two remained
`FAILED_FIT`/incomplete. `temporal_wavelet` beat runner-up `xgb_wavelet` by one
Pika clip under the frozen seed-42 development rule; the seed-2042 rank flip is
`SEED_UNSTABLE`. The 40-row API smoke fixture is not model-comparison evidence.
See the [selection scope, all 40 IDs and actual statuses](docs/MODEL_SELECTION_SUMMARY.md)
and [portable evidence extract](docs/MODEL_SELECTION_SUMMARY.json).

## Package map

- [Quickstart and environment](docs/README.md)
- [Stacker interface and operating points](docs/HANDOFF_TO_STACKER.md)
- [Model card](docs/MODEL_CARD.md)
- [Which proposal, winner and parameters?](docs/PROVENANCE.md), with portable
  [source excerpts, JSON pointers and actual source hashes](docs/PROVENANCE_EVIDENCE.json)
- [Model configuration](model/model_config.json), [bundle sidecar](model/branch_b_temporal_wavelet_full.json)
  and [trusted serialized model](model/branch_b_temporal_wavelet_full.joblib)
- [Prediction schema](predictions/SCHEMA_DESCRIPTION.md),
  [Parquet](predictions/predictions_full_dataset.parquet) and
  [CSV](predictions/predictions_full_dataset.csv)
- Preferred readable notebook: [Jupytext percent Python](notebooks/delivery_verification.py);
  [paired Jupyter notebook](notebooks/delivery_verification.ipynb)
- [Integration example](docs/INTEGRATION_EXAMPLE.py)
- [Acceptance receipt](DELIVERY_RECEIPT.json), [measured audit](docs/AUDIT_REPORT.json),
  [audit/seal procedure](docs/AUDIT_AND_SEAL.md), [checksums](CHECKSUMS.txt),
  [SHA-256 manifest](MANIFEST_SHA256.json)

## Use when only this folder is copied

Keep the subdirectories together. From inside the copied package root:

```bash
python -m venv ../branch_b_runtime
# Activate ../branch_b_runtime using your platform's activation command.
python -m pip install -r requirements-runtime.txt
python -B docs/INTEGRATION_EXAMPLE.py
python -B -m inference.cli --input-table predictions/regression_sample_features.parquet --output ../branch_b_scores.parquet
python -B -m inference.cli --video path/to/clip.mp4
```

The example resolves paths from its own location. For your own Python code,
run from this folder (or add this folder to `sys.path`):

```python
from inference import load_detector

detector = load_detector()
result = detector.infer_video("path/to/clip.mp4")
```

These commands do not require the original `deliverables/` import tree. Use `-B`
(or `PYTHONDONTWRITEBYTECODE=1`) to avoid adding caches to a sealed package. Keep
the virtual environment and generated scores outside the package, as above,
for a byte-exact inventory check.

For notebook work, install `requirements-notebook.txt`; see the
[portable notebook instructions](docs/README.md#notebook).
No raw video corpus is included. Inference on your own videos needs those videos;
repository-based retraining needs the original manifests and feature data, as
explained in [provenance](docs/PROVENANCE.md#reproduction-not-part-of-this-handoff).

## Verify before loading trusted serialized content

1. Obtain the final `MANIFEST_SHA256.json` digest via a trusted handoff channel.
   Compute its SHA-256 and compare it independently. Bundled hashes are integrity
   evidence, not signatures or proof of authenticity.
2. From the package root, run `sha256sum -c CHECKSUMS.txt` (or an equivalent
   SHA-256 utility). Any changed or added delivery file requires a new seal.
3. In the original repository, the read-only complete-inventory check is:
   `python -B scripts/audit_temporal_wavelet_delivery.py --verify-only`.
   That script is repository tooling, not included in this portable folder.
4. After trusting the package, run the integration example and the reference CLI
   command above. Full acceptance, notebook execution and resealing are a separate
   finalization step; the receipt describes only the snapshot at its timestamp.

The full-fit scores and measured false-positive rates describe this training
corpus, not future-population performance guarantees. `raw_margin` is the team's
chosen unsquashed stacking interface; internal stacking is explicitly in scope.
