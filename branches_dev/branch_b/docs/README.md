# Quickstart — Branch B temporal_wavelet

See the [package-root README](../README.md) for the complete inventory, portable
commands and verification steps. See [provenance](PROVENANCE.md) for the proposal,
selection evidence and frozen recipe; architecture selection was not an independent
LightGBM hyperparameter sweep.

Historical Stage A compared **40 candidate configurations: 38 eligible + 2
diagnostic controls**, not 40 algorithm families or new full32k models. Two
configurations remained incomplete; the **40-row API regression** is a separate
single-detector smoke check, not model-selection evidence. The
[selection summary](MODEL_SELECTION_SUMMARY.md) contains all IDs, actual statuses,
validation limits and the `SEED_UNSTABLE` disclosure, with a
[portable evidence extract](MODEL_SELECTION_SUMMARY.json).

## Scope

The requested deliverable fits the selected 113-input `temporal_wavelet` view on
all 32,000 current clips (16,000 real, 16,000 generated; **8 generator families +
2 real-source families**). The booster has 600 trees, no validation/early stopping,
and full-data Platt calibration (`C=1.0`) and threshold selection. Predictions are
**in-sample**, supplied for the requested internal stacking workflow, not a held-out
estimate of generalization. No partition or OOF artifact is required for delivery.
Using all available data increases fitting coverage; it does not guarantee better
performance on unseen media.

## Environment and installation

[Runtime requirements](../requirements-runtime.txt) pin the installed versions
used for these checks on **CPython 3.14.6, Windows AMD64**. They are a tested
reference environment, **not minimum-supported versions** or a claim that every
OS/Python combination has been tested. The final acceptance audit records its
own environment. Pins/imports and copied-folder execution were validated using
the installed host interpreter; no independent clean-venv installation is claimed. Use a compatible wheel/platform; LightGBM and OpenCV have
native components. No PyTorch, XGBoost or GPU is needed for this detector.

From the copied package root, in an activated virtual environment:

```bash
python -m pip install -r requirements-runtime.txt
python -B docs/INTEGRATION_EXAMPLE.py
```

`scikit-learn` is needed to deserialize the scaler/calibration objects inside the
joblib, even though the inference wrapper applies the fitted calibration
numerically. Load only trusted joblib files after checksum verification.
`pyarrow` supplies Parquet I/O; `opencv-python-headless` supplies `cv2` without GUI
components. Do not install multiple competing OpenCV distributions in one environment.

## Precomputed predictions for internal stacking

From the package root:

```python
import pandas as pd

preds = pd.read_parquet("predictions/predictions_full_dataset.parquet")
assert preds["clip_id"].is_unique
X_branch_b = preds.set_index("clip_id")[["raw_margin"]]
```

Join branches by unique `clip_id`, not row position. Keep labels/family metadata
separate from machine inputs. `raw_margin` preserves the booster's unsquashed
score scale and is the agreed team interface. Other score representations are not
universally mathematically forbidden; see the [handoff](HANDOFF_TO_STACKER.md).

## Python API and CLI, copied folder only

Run from the package root so `inference` is importable:

```python
from inference import load_detector

detector = load_detector()
result = detector.infer_video("path/to/video.mp4")
if result["status"] == "SUCCESS":
    print(result["raw_margin"], result["calibrated_prob"], result["decision"])
else:
    print("Branch B missing:", result)
```

```bash
python -B -m inference.cli --video path/to/video.mp4
python -B -m inference.cli --input-table path/to/features.parquet --output path/to/scores.parquet
```

A feature table must contain the 113 named columns from
[`ordered_feature_names`](../model/model_config.json), not the prediction table's
seven columns. The included `regression_sample_features.parquet` is a small
reference input for smoke checks. Decisions use exact loaded thresholds, never
rounded numbers copied from prose.

## Notebook

Install the optional [notebook requirements](../requirements-notebook.txt):

```bash
python -m pip install -r requirements-notebook.txt
```

The preferred source is [delivery_verification.py](../notebooks/delivery_verification.py),
a Jupytext percent script paired with [delivery_verification.ipynb](../notebooks/delivery_verification.ipynb).
From the copied package root, execute the readable source with:

```bash
python -B notebooks/delivery_verification.py
```

Or open the paired notebook with its kernel working directory set to the package
root or `notebooks/`. Only this package and installed requirements are needed;
final acceptance executes a fresh nbclient kernel from an isolated copied folder
with original repository imports actively blocked. The `.py` file is authoritative:
regenerate with `python -m jupytext --to ipynb --update notebooks/delivery_verification.py`
only during authorized resealing, then execute and check `--sync` zero drift.
The notebook performs no model fits, simulated branch scoring or replacement
predictions. Its plots are descriptive in-sample illustrations and its stacker
examples prepare the actual full-fit inputs, not generalization estimates.

## Decisions and evidence

The default calibrated probability threshold is **0.025049979809292647**,
corresponding to raw margin **-4.095739606463252**. On this fitting corpus its real
FPR is Vript **0.01**, HD-VG-130M **0.0005**, and unweighted pooled **0.007625**.
The secondary probability threshold **0.015457025337482144** has **0.01
family-weighted** pooled real FPR, versus **0.01475 unweighted**. These are not
future-population guarantees. Details and exact loaded decision examples are in
[HANDOFF_TO_STACKER.md](HANDOFF_TO_STACKER.md).

The final [receipt](../DELIVERY_RECEIPT.json) and [audit report](AUDIT_REPORT.json)
are timestamped evidence for the sealed snapshot. They distinguish 20 freshly
replayed historical source fixtures from full32k cached-feature scoring; the full
original video corpus was not freshly re-extracted. No OOF prerequisite is imposed.
Any future documentation/output edit requires new acceptance and resealing;
model weights, calibration, thresholds and original prediction values are unchanged.
