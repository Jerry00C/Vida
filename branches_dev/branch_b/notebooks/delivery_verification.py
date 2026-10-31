# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Branch B — temporal_wavelet delivery verification
#
# This readable notebook verifies the **already fitted** all-32,000-clip model
# for the requested internal stacker. It does not fit any model, invent other
# branch scores, replace predictions, or introduce role/OOF partitions.
# All plots and corpus rates below are **descriptive in-sample** summaries,
# not estimates of performance on unseen media. Internal full-fit use is in scope.
#
# ## 1. Identity, provenance and portable paths
#
# The selected architecture is `temporal_wavelet`: 113 tabular inputs and a
# LightGBM booster. Its numerical recipe was the fixed `ModelSpec`, **not an
# independently won hyperparameter search**. Historical Option 1 fitted 22,400
# clips and stopped at **597** trees (600 was the cap). This distinct requested
# full-data refit has **600** trees, no validation or early stopping, full-data
# Platt calibration and full-data thresholds. See `../docs/PROVENANCE.md` and
# `../docs/MODEL_SELECTION_SUMMARY.md` for bounded historical selection evidence.
#
# The corpus has 16,000 real and 16,000 generated clips: **2 real sources and
# 8 generator families**. Class-balanced weights assign 50% mass to each class,
# 25% to each real source and 6.25% to each generator, normalized to mean one.
# These weights do not guarantee equal gradients or generalization.

# %%
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Only the copied package is needed. Run from its root or notebooks directory.
def find_package() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "model/model_config.json").is_file() and (candidate / "inference").is_dir():
            return candidate
    raise FileNotFoundError("Run from the copied package root or its notebooks directory")

PACKAGE = find_package()
sys.path.insert(0, str(PACKAGE))
from inference import load_detector, DetectorIntegrityError, ORDERED_FEATURE_NAMES, SCHEMA_HASH

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.figsize": (12, 5)})
MODEL = PACKAGE / "model/branch_b_temporal_wavelet_full.joblib"
PRED = PACKAGE / "predictions"
config = json.loads((PACKAGE / "model/model_config.json").read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("Package:", PACKAGE.name, "| Python:", sys.version.split()[0])
print("Trusted model SHA-256:", sha256(MODEL))

# %% [markdown]
# ## 2. Trusted loading, schema and calibration readbacks
#
# Joblib/pickle is executable content: **trust both binary and manifest before
# loading**. SHA-256 detects drift, not authenticity. The hardened loader uses
# per-load resolution of four legacy class names to state-only carriers. It is
# not a pickle sandbox and installs no permanent `sys.modules` shims.
# Missing checksum, ordered-schema or calibration metadata fails closed; no
# guessed defaults are used. The private joblib adapter is tested with the
# versions in `../requirements-runtime.txt`, not an exhaustive version matrix.

# %%
detector = load_detector()  # verifies the required local adjacent manifest first
bundle = detector._bundle  # audit-only introspection, not a second joblib.load
sidecar = detector.sidecar
booster = detector._booster
assert sha256(MODEL) == config["bundle_sha256"] == detector.bundle_sha256
assert detector.candidate_id == config["candidate_id"] == "temporal_wavelet"
assert booster.num_trees() == config["hyperparameters"]["n_estimators"] == 600
assert booster.num_feature() == config["feature_count"] == 113
assert detector.ordered_feature_names == config["ordered_feature_names"] == list(ORDERED_FEATURE_NAMES)
assert bundle["schema_hash"] == config["schema_hash"] == sidecar["ordered_schema_sha256"] == SCHEMA_HASH
assert detector.operating_thresholds == config["operating_thresholds"] == sidecar["operating_thresholds"]
assert bundle["hyperparameters"] == config["hyperparameters"]
assert not any(k == "branch_b" or k.startswith("branch_b.") for k in sys.modules)

calibrator = bundle["calibrator"]
scaler_mean = float(calibrator.scaler_.mean_[0])
scaler_scale = float(calibrator.scaler_.scale_[0])
cal_coef = float(calibrator.coef_)
cal_intercept = float(calibrator.intercept_)
assert cal_coef == sidecar["provenance"]["calibrator_params"]["coef"]
assert cal_intercept == sidecar["provenance"]["calibrator_params"]["intercept"]
display(pd.DataFrame([{
    "candidate": detector.candidate_id, "trees": booster.num_trees(),
    "features": booster.num_feature(), "fitting_clips": sidecar["sample_counts"]["total"],
    "scaler_mean": scaler_mean, "scaler_scale": scaler_scale,
    "Platt_coef": cal_coef, "Platt_intercept": cal_intercept,
}]))

# %% [markdown]
# A real byte-tamper test uses a **temporary copy outside this package**.
# It does not alter the released binary or sidecar.

# %%
with tempfile.TemporaryDirectory(prefix="tw_notebook_") as tmp:
    tampered = Path(tmp) / "tampered.joblib"
    data = bytearray(MODEL.read_bytes())
    data[len(data) // 2] ^= 1
    tampered.write_bytes(data)
    tampered.with_suffix(".json").write_bytes(MODEL.with_suffix(".json").read_bytes())
    try:
        load_detector(tampered)
    except DetectorIntegrityError as exc:
        assert "hash mismatch" in str(exc)
        print("PASS — modified binary rejected before unpickling")
    else:
        raise AssertionError("Modified binary accepted")

# %% [markdown]
# ## 3. Feature schema and fitted importance
#
# The view contains 17 `tfft_v2`, 54 `local_temporal_v2` and 42 `twavelet_v2`
# features. Spectra use the canonical time-interpolated 128×128 grid.
# Wavelets are **1D temporal db2**, level 2, `periodization` (not a spatial 2D
# DWT), over gray/edge patch time series and flow-residual patch sequences.
# The 128×128 analysis image and 96×96 flow image are derived from resized
# uint8 frames before scaling. Importance describes this fitted booster only;
# it is not a causal explanation or independent selection result.

# %%
importance = pd.DataFrame({
    "feature": detector.ordered_feature_names,
    "gain": booster.feature_importance(importance_type="gain"),
    "splits": booster.feature_importance(importance_type="split"),
})
importance["group"] = importance.feature.str.split("__").str[0]
summary = importance.groupby("group").agg(features=("feature", "size"), gain=("gain", "sum"), splits=("splits", "sum"))
assert summary.features.to_dict() == config["feature_groups"]
summary["gain_share"] = summary.gain / summary.gain.sum()
display(summary)
display(importance.sort_values("gain", ascending=False).head(20))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
summary.gain_share.plot.barh(ax=axes[0], title="Fitted gain share by feature group")
importance.nlargest(15, "gain").sort_values("gain").plot.barh(x="feature", y="gain", ax=axes[1], legend=False, title="Top 15 fitted feature gains")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Original scores: integrity and in-sample distributions
#
# `raw_margin` is the unsquashed booster score, not guaranteed true posterior
# log-odds. Zero is a score-space reference, not an unbiased prior. The notebook
# reads the original scores; it never regenerates or overwrites them.
# This portable check does not replay the full original video corpus or require
# the repository feature cache. Fresh full-32k cached-feature replay and separate
# canonical extraction parity are recorded by the repository acceptance audit.

# %%
preds = pd.read_parquet(PRED / "predictions_full_dataset.parquet")
csv = pd.read_csv(PRED / "predictions_full_dataset.csv", float_precision="round_trip")
pd.testing.assert_frame_equal(csv, preds, check_dtype=False)
assert len(preds) == 32000 and preds.clip_id.is_unique and preds.clip_id.notna().all()
assert np.isfinite(preds[["raw_margin", "calibrated_prob"]].to_numpy()).all()
assert preds.calibrated_prob.between(0, 1).all()
assert preds.label.value_counts().to_dict() == {0: 16000, 1: 16000}
assert preds.family.value_counts().to_dict() == config["sample_counts"]["by_family"]
display(preds.groupby(["label", "family"]).raw_margin.agg(["count", "mean", "std", "min", "median", "max"]))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for label, group in preds.groupby("label"):
    axes[0].hist(group.raw_margin, bins=80, alpha=0.6, density=True, label={0: "Real", 1: "Generated"}[label])
axes[0].legend()
axes[0].set(title="In-sample full32k margin density", xlabel="Raw margin")
families = sorted(preds.family.unique())
axes[1].boxplot([preds.loc[preds.family == f, "raw_margin"] for f in families], orientation="horizontal", tick_labels=families)
axes[1].set(title="In-sample: 2 real sources + 8 generators", xlabel="Raw margin")
thresholds = detector.operating_thresholds
for ax in axes:
    ax.axvline(thresholds[detector.deployed_threshold_key]["margin_threshold"], color="black", linestyle="--")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Exact frozen thresholds and descriptive calibration
#
# Decisions use the **exact loaded thresholds**, including equality (`>=`).
# Default `target_1pct_real_fpr`: probability **0.025049979809292647**, margin
# **-4.095739606463252**. On this fitting corpus: Vript 1%, HD-VG-130M 0.05%,
# unweighted pooled 0.7625%. No unseen-media 1% FPR guarantee is implied.
# Secondary `pooled_1pct_real_fpr`: probability **0.015457025337482144**;
# real-source/family-weighted FPR is 1%, versus **1.475% unweighted**.
# `macro_f1` was selected on full-data weighted metrics, not a global optimum
# over future populations. The sigmoid below is a mapping, not a reliability
# curve proving calibration on new media.

# %%
rows = []
family_counts = preds.family.value_counts()
weights = preds.family.map({f: 1 / n for f, n in family_counts.items()})
real = preds.label == 0
for key, point in thresholds.items():
    positive = preds.calibrated_prob >= point["threshold"]
    rows.append({
        "key": key, "probability_threshold": point["threshold"], "margin_threshold": point["margin_threshold"],
        "pooled_real_fpr": float(positive[real].mean()),
        "family_weighted_real_fpr": float(np.average(positive[real], weights=weights[real])),
        "Vript_fpr": float(positive[preds.family == "Vript"].mean()),
        "HD-VG-130M_fpr": float(positive[preds.family == "HD-VG-130M"].mean()),
        "generated_recall": float(positive[~real].mean()),
    })
np.testing.assert_array_equal(preds.prediction, (preds.calibrated_prob >= thresholds[detector.deployed_threshold_key]["threshold"]).astype(int))
display(pd.DataFrame(rows).style.format(precision=17))
margin_grid = np.linspace(-16, 16, 1000)
prob_grid = 1 / (1 + np.exp(-(cal_coef * (margin_grid - scaler_mean) / scaler_scale + cal_intercept)))
fig, ax = plt.subplots()
ax.plot(margin_grid, prob_grid, label="Fitted Platt mapping")
for key, point in thresholds.items():
    ax.scatter(point["margin_threshold"], point["threshold"], label=key)
ax.set(xlabel="Raw margin", ylabel="Calibrated probability", title="Full-data Platt mapping and frozen operating points")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Portable 40-row inference and fail-closed examples
#
# The reference scores were frozen from the saved delivery, independently of
# this notebook. These checks are single-detector regression, not comparisons
# of candidate models. Invalid feature inputs raise; video QC failures abstain
# with null scores. Unknown codecs, duration outside 1–30 seconds, fewer than
# 8 observations/7 transitions, or truncated selection fail closed.

# %%
features = pd.read_parquet(PRED / "regression_sample_features.parquet")
reference = json.loads((PRED / "regression_expected.json").read_text(encoding="utf-8"))
assert reference["source_predictions_sha256"] == sha256(PRED / "predictions_full_dataset.parquet")
expected = pd.DataFrame(reference["records"]).set_index("clip_id").loc[features.clip_id]
actual = detector.infer_features(features)
assert len(actual) == 40
for col in ("raw_margin", "calibrated_prob", "prediction", "decision"):
    np.testing.assert_array_equal(actual[col].to_numpy(), expected[col].to_numpy())
single = detector.infer_features(features.iloc[0][detector.ordered_feature_names].to_dict())
assert single["raw_margin"] == actual.raw_margin.iloc[0]
display(actual.head())
for bad in ({}, np.full(113, np.nan)):
    try:
        detector.infer_features(bad)
    except ValueError:
        print("PASS — invalid feature input rejected")
    else:
        raise AssertionError("Invalid features accepted")
missing = detector.infer_video("nonexistent_notebook_clip.mp4")
assert missing["status"] == "ABSTAIN" and missing["fail_closed"]
assert missing["raw_margin"] is missing["calibrated_prob"] is missing["prediction"] is None
print("PASS — 40 frozen scores/decisions exact; missing video abstains")

# %% [markdown]
# ## 7. Internal full-fit stacker handoff (no fitting in this notebook)
#
# `raw_margin` is the team's chosen unsquashed feature; probabilities are not
# universally forbidden, and no globally optimal Bayesian stacking or MLP
# convexity is claimed. Use the original full-fit rows for the requested internal
# stacker, join branches by **unique clip_id**, and keep labels outside inputs.
# OOF generation/role partitioning is **not a delivery prerequisite**. In-sample
# stacker results would not be unbiased generalization evidence.
#
# The executable example prepares Branch B's real 32k input. Supply actual
# other-branch tables with identical IDs to `join_branch`; no fabricated signals,
# fitting, replacement predictions or pseudo-holdout evaluation are performed.

# %%
X_branch_b = preds.set_index("clip_id")[["raw_margin"]].rename(columns={"raw_margin": "branch_b_raw_margin"})
y_internal = preds.set_index("clip_id")["label"].loc[X_branch_b.index]
assert X_branch_b.shape == (32000, 1)

def join_branch(meta, branch, score_column, output_name):
    """Join a real branch score table, enforcing one-to-one full ID coverage."""
    assert branch.clip_id.notna().all() and branch.clip_id.is_unique
    other = branch.set_index("clip_id")[[score_column]].rename(columns={score_column: output_name})
    assert set(other.index) == set(meta.index)
    joined = meta.join(other, validate="one_to_one")
    assert np.isfinite(joined.to_numpy()).all()
    return joined

def branch_b_stacker_values(result):
    """Example missingness contract to agree with the internal stacker owner.

    Zero is a chosen imputation sentinel, NOT a proven neutral posterior prior.
    Preserve the missingness indicator and fit any downstream transform outside
    this verification notebook using the team's actual full-fit workflow.
    """
    if result is None or result.get("status") != "SUCCESS":
        return {"branch_b_raw_margin": 0.0, "branch_b_missing": 1.0}
    margin = float(result["raw_margin"])
    if not np.isfinite(margin):
        raise ValueError("Nonfinite successful margin")
    return {"branch_b_raw_margin": margin, "branch_b_missing": 0.0}

display(X_branch_b.head())
print("Internal stacker input:", X_branch_b.shape, "| Labels kept separately:", y_internal.shape)
print("Success:", branch_b_stacker_values(single))
print("Abstention:", branch_b_stacker_values(missing))

# %% [markdown]
# ## 8. Verification boundary
#
# This notebook checks trusted artifact readbacks, schema, thresholds,
# CSV/Parquet agreement, all 32k IDs/finite scores and exact 40-row inference.
# It performs no training. The separate acceptance report supplies hardening,
# independent canonical extraction parity, cached full32k replay and inventory
# sealing results. Only that complete acceptance process can accept the final
# snapshot. Package checksums are integrity evidence, not authenticity signatures.

# %%
print("Notebook checks complete: temporal_wavelet / full32k600 / 113 features.")
print("Descriptive in-sample scope; no model fits or replacement predictions.")
