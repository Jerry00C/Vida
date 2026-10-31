# -*- coding: utf-8 -*-
"""Thread-Safe, Fail-Closed Deployment Detector for Branch B 'temporal_wavelet'.

Authority & Responsibilities:
- Model Architecture: 'temporal_wavelet' (winning Branch B configuration).
- Predictor: Tabular LightGBM booster (600 trees) trained on full 32,000 dataset clips.
- Calibrator: Platt logistic calibration over standardized log-odds margins (MarginCalibrator C=1.0).
- Signals Provided:
    * 'raw_margin': Unbounded continuous log-odds machine signal in (-inf, +inf).
      PRIMARY signal for downstream stackers / meta-learners.
    * 'calibrated_prob': Platt-calibrated probability P(Y=1 | clip) in [0.0, 1.0].
    * 'prediction': Binary indicator {0, 1} under operating threshold.
    * 'decision': Categorical label {'REAL', 'GENERATED'}.
- Fail-Closed Integrity:
    * Validates model bundle sha256 and sidecar JSON on initialization.
    * Verifies exact 113 ordered features and schema consistency.
    * Thread-safe inference using re-entrant lock.
    * In case of decode failure, corrupted media, or nonfinite values, strictly fails closed (ABSTAIN).
"""
from __future__ import annotations

import hashlib
import json
import io
import inspect
import re
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap
import numpy as np
import pandas as pd


class _LegacyState:
    """State-only carrier for this trusted legacy bundle; never a repository class."""


# Exact pickle globals in the legacy full32k artifact, not a wildcard shim.
_LEGACY_CLASSES = {
    (module, name): type(name, (_LegacyState,), {})
    for module, name in (
        ("branch_b.v2.calibration_v2", "MarginCalibrator"),
        ("branch_b.v2.calibration_v2", "WeightedStandardScaler"),
        ("branch_b.v2.models_v2", "WeightedGBMModel"),
        ("branch_b.v2.models_v2", "ModelSpec"),
    )
}


class _BundleUnpickler(NumpyUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _LEGACY_CLASSES:
            return _LEGACY_CLASSES[module, name]
        if module == "branch_b" or module.startswith("branch_b."):
            raise DetectorIntegrityError(f"Unexpected legacy pickle global: {module}.{name}")
        return super().find_class(module, name)


def _load_trusted_bundle(payload: bytes) -> Any:
    """Scoped class resolution, including compressed joblib's numpy array format.

    No monkeypatches, module installation, or canonical class changes. Other pickle
    globals retain normal joblib semantics: THIS IS NOT AN UNTRUSTED PICKLE SANDBOX.
    Both the manifest and executable binary must come from a trusted source;
    a checksum only detects inconsistency, it does not authenticate either file.
    The private joblib adapter supports the pre/post ensure_native_byte_order API.
    """
    with io.BytesIO(payload) as stream:
        with _validate_fileobject_and_memmap(stream, "", None) as (decoded, _):
            if isinstance(decoded, str):
                raise DetectorIntegrityError("Unsupported legacy companion-file joblib format")
            kwargs = {"mmap_mode": None}
            if "ensure_native_byte_order" in inspect.signature(NumpyUnpickler).parameters:
                kwargs["ensure_native_byte_order"] = True
            return _BundleUnpickler("", decoded, **kwargs).load()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DetectorIntegrityError(f"Duplicate sidecar JSON key: {key}")
        result[key] = value
    return result


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (str, bool)):
        raise DetectorIntegrityError(f"{label} must be numeric")
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1 or not np.isfinite(arr).all():
        raise DetectorIntegrityError(f"{label} must be a finite scalar")
    return float(arr.reshape(-1)[0])

from .extract import (
    EPS,
    ORDERED_FEATURE_NAMES,
    SCHEMA_HASH,
    ExtractionError,
    decode_canonical_video,
    extract_features,
    extract_features_from_decoded,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DEFAULT_BUNDLE_PATH = DEFAULT_MODEL_DIR / "branch_b_temporal_wavelet_full.joblib"
DEFAULT_SIDECAR_PATH = DEFAULT_MODEL_DIR / "branch_b_temporal_wavelet_full.json"


class DetectorIntegrityError(RuntimeError):
    """Raised when bundle binary, schema, or configuration fails validation."""
    pass


class BranchBDetector:
    """Standalone, fail-closed detector for the Branch B winning architecture 'temporal_wavelet'.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the bundle joblib file. If None, defaults to the bundled deliverable model.
    verify_sha256 : bool, default True
        Verify the bundle checksum against its required trusted adjacent manifest.
        False is an UNSAFE legacy opt-out of the binary checksum only, never of
        schema/calibration validation. Pickle is executable; checksums are not authentication.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        verify_sha256: bool = True,
    ) -> None:
        init_start = time.perf_counter()
        self._lock = threading.RLock()

        self.model_path = Path(model_path if model_path is not None else DEFAULT_BUNDLE_PATH).resolve()
        self.sidecar_path = self.model_path.with_suffix(".json")
        try:
            if not self.model_path.is_file() or not self.sidecar_path.is_file():
                raise DetectorIntegrityError("Model and local adjacent .json sidecar are both required")
            self.sidecar = json.loads(self.sidecar_path.read_text(encoding="utf-8"),
                                      object_pairs_hook=_unique_object,
                                      parse_constant=lambda v: (_ for _ in ()).throw(
                                          DetectorIntegrityError(f"Nonfinite sidecar constant: {v}")))
            self._validate_manifest(self.sidecar)
            # Hash and unpickle the SAME bytes (no hash/load path-replacement race).
            payload = self.model_path.read_bytes()
            self.bundle_sha256 = hashlib.sha256(payload).hexdigest()
            if verify_sha256:
                if self.bundle_sha256 != self.sidecar["bundle_sha256"]:
                    raise DetectorIntegrityError("Bundle SHA-256 hash mismatch")
                if len(payload) != self.sidecar["bundle_bytes"]:
                    raise DetectorIntegrityError("Bundle byte count mismatch")
            else:
                warnings.warn("UNSAFE verify_sha256=False: trusted executable pickle only; "
                              "binary checksum verification disabled", RuntimeWarning, stacklevel=2)
            self._bundle = _load_trusted_bundle(payload)
            self._validate_bundle()
        except DetectorIntegrityError:
            raise
        except Exception as exc:
            raise DetectorIntegrityError(f"Invalid model/manifest state: {exc}") from exc
        self.initialization_time_seconds = time.perf_counter() - init_start

    @staticmethod
    def _validate_manifest(meta: dict[str, Any]) -> None:
        if not isinstance(meta, dict):
            raise DetectorIntegrityError("Sidecar must be a JSON object")
        digest = meta["bundle_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise DetectorIntegrityError("Required bundle_sha256 must be 64 lowercase hex digits")
        if type(meta["bundle_bytes"]) is not int or meta["bundle_bytes"] <= 0:
            raise DetectorIntegrityError("Invalid bundle_bytes")
        if (meta["candidate_id"] != "temporal_wavelet"
                or type(meta["feature_count"]) is not int or meta["feature_count"] != 113):
            raise DetectorIntegrityError("Candidate/feature count mismatch")
        names = meta["ordered_feature_names"]
        if names != ORDERED_FEATURE_NAMES or len(set(names)) != 113:
            raise DetectorIntegrityError("Exact ordered 113-feature schema required")
        computed = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        if computed != SCHEMA_HASH or meta["provenance"]["ordered_schema_sha256"] != computed:
            raise DetectorIntegrityError("Ordered feature schema hash mismatch")
        if "schema_hash" in meta and meta["schema_hash"] != computed:
            raise DetectorIntegrityError("Sidecar schema_hash mismatch")

    def _validate_bundle(self) -> None:
        b, s = self._bundle, self.sidecar
        if not isinstance(b, dict):
            raise DetectorIntegrityError("Bundle must be a dictionary")
        # JSON normalization handles serialized tuple/list dimensions, not missing values.
        for key in ("candidate_id", "feature_count", "ordered_feature_names", "feature_groups",
                    "extraction_config", "provenance", "calibration_mode", "operating_thresholds",
                    "deployed_threshold_key", "qc_policy", "model_architecture", "format_version"):
            if json.loads(json.dumps(b[key])) != s[key]:
                raise DetectorIntegrityError(f"Sidecar/bundle mismatch: {key}")
        if b["schema_hash"] != SCHEMA_HASH:
            raise DetectorIntegrityError("Binary schema hash mismatch")
        if (b["model_architecture"] != "tabular_lightgbm" or b["format_version"] != "branch_b_release_v3"
                or s["model_type"] != "WeightedGBMModel" or b["preprocessor"] is not None
                or s["preprocessor_type"] != "None"):
            raise DetectorIntegrityError("Unsupported model/preprocessor metadata or state")
        cfg = b["extraction_config"]
        if (cfg["generation_id"] != "gen_933b135d79"
                or cfg["schema_hash"] != "933b135d7906d48d673d322cc8a2e73c4bcc4bdf1f933a9f276082e1b8086fa9"
                or b["provenance"]["schema_hash_v2"] != cfg["schema_hash"]
                or cfg["canonical_time_policy"] != "sidecar_independent_index_fps_v1"
                or cfg["preprocessing"] != {"analysis_size": (128, 128), "flow_size": (96, 96),
                    "color_size": (64, 64), "t_max": 64, "min_sampled_frames": 8, "fps_fallback": 25.0}):
            raise DetectorIntegrityError("Unsupported extraction configuration")
        self.candidate_id = b["candidate_id"]
        self.ordered_feature_names = list(b["ordered_feature_names"])
        self._booster = self._unwrap_booster(b)
        if self._booster is None or self._booster.num_feature() != 113:
            raise DetectorIntegrityError("Booster must have exactly 113 features")
        if self._booster.num_trees() != 600 or b["model"].n_features_in_ != 113:
            raise DetectorIntegrityError("Wrong trained model tree/feature count")
        if b["model"].model_.booster_.model_to_string() != self._booster.model_to_string():
            raise DetectorIntegrityError("Conflicting bundled boosters")
        c = self._calibrator = b["calibrator"]
        if (b["calibration_mode"] != "platt_logistic" or s["calibrator_type"] != "MarginCalibrator"
                or c.standardize is not True):
            raise DetectorIntegrityError("Unsupported calibration state")
        self.cal_coef = _finite_scalar(c.coef_, "calibrator coefficient")
        self.cal_intercept = _finite_scalar(c.intercept_, "calibrator intercept")
        if self.cal_coef <= 0 or c.C <= 0 or not np.isfinite(c.C):
            raise DetectorIntegrityError("Invalid monotonic Platt calibrator")
        params = {"type": "MarginCalibrator", "C": c.C, "standardize": c.standardize,
                  "coef": self.cal_coef, "intercept": self.cal_intercept}
        if s["provenance"]["calibrator_params"] != params:
            raise DetectorIntegrityError("Calibrator metadata/state mismatch")
        if (np.asarray(c.model_.coef_).shape != (1, 1)
                or _finite_scalar(c.model_.coef_, "logistic coefficient") != self.cal_coef
                or _finite_scalar(c.model_.intercept_, "logistic intercept") != self.cal_intercept
                or not np.array_equal(c.model_.classes_, [0, 1]) or c.model_.n_features_in_ != 1):
            raise DetectorIntegrityError("Conflicting logistic state")
        scaler = c.scaler_
        if scaler.with_mean is not True or scaler.with_std is not True:
            raise DetectorIntegrityError("Unsupported scaler configuration")
        for name in ("mean_", "scale_", "var_"):
            if np.asarray(getattr(scaler, name)).shape != (1,):
                raise DetectorIntegrityError(f"Wrong scaler shape: {name}")
        self.scaler_mean = _finite_scalar(scaler.mean_, "scaler mean")
        self.scaler_scale = _finite_scalar(scaler.scale_, "scaler scale")
        variance = _finite_scalar(scaler.var_, "scaler variance")
        if self.scaler_scale <= 0 or variance <= 0 or self.scaler_scale != np.sqrt(variance):
            raise DetectorIntegrityError("Invalid scaler variance/scale")
        self.operating_thresholds = b["operating_thresholds"]
        self.deployed_threshold_key = b["deployed_threshold_key"]
        if (set(self.operating_thresholds) != {"target_1pct_real_fpr", "pooled_1pct_real_fpr", "macro_f1"}
                or self.deployed_threshold_key not in self.operating_thresholds):
            raise DetectorIntegrityError("Missing/unknown operating threshold")
        for key, info in self.operating_thresholds.items():
            prob = _finite_scalar(info["threshold"], "probability threshold")
            margin = _finite_scalar(info["margin_threshold"], "margin threshold")
            if not 0 < prob < 1 or info["score_space"] != "calibrated_probability":
                raise DetectorIntegrityError("Invalid threshold score-space/range")
            if not np.isclose(self._calibrate_margin(np.array([margin]))[0], prob, rtol=0, atol=1e-14):
                raise DetectorIntegrityError("Inconsistent margin/probability thresholds")
            if s["provenance"]["realized_thresholds"][key] != prob:
                raise DetectorIntegrityError("Inconsistent realized threshold")
        tp = s["threshold_parameters"]
        primary = self.operating_thresholds[self.deployed_threshold_key]
        if (tp["operating_thresholds"] != self.operating_thresholds
                or tp["primary_threshold_key"] != self.deployed_threshold_key
                or tp["primary_threshold"] != primary["threshold"]
                or tp["primary_margin_threshold"] != primary["margin_threshold"]
                or tp["score_space"] != primary["score_space"]):
            raise DetectorIntegrityError("Conflicting threshold metadata")
        self.qc_policy = b["qc_policy"]
        q = self.qc_policy
        if (q["fail_closed_action"] != "ABSTAIN" or q["min_frames_required"] != 8
                or q["min_transitions_required"] != 7 or q["min_duration_seconds"] != 1.0
                or q["max_duration_seconds"] != 30.0
                or q["allowed_codecs"] != ["h264", "hevc", "vp9", "av1", "mpeg4"]
                or q["source_coverage"] != {"external_inference_supported": True, "requires_source_identity": False}):
            raise DetectorIntegrityError("Unsupported QC policy")

    @staticmethod
    def _compute_file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _unwrap_booster(bundle: Any) -> Any:
        if isinstance(bundle, dict):
            if "booster" in bundle and hasattr(bundle["booster"], "predict"):
                return bundle["booster"]
            model = bundle.get("model")
            if model is not None:
                if hasattr(model, "gbm_") and hasattr(model.gbm_, "predict"):
                    return model.gbm_
                if hasattr(model, "model_"):
                    inner = model.model_
                    if hasattr(inner, "booster_"):
                        return inner.booster_
                    if hasattr(inner, "predict"):
                        return inner
                if hasattr(model, "predict"):
                    return model
        return None

    def _calibrate_margin(self, raw_margin: np.ndarray) -> np.ndarray:
        """Applies exact Platt logistic scaling to raw margins."""
        # Monotonic logistic mapping on standardized margins
        z = (raw_margin - self.scaler_mean) / self.scaler_scale
        logit = self.cal_coef * z + self.cal_intercept
        # Preserve the released float64 operation order for exact saved-score replay.
        return 1.0 / (1.0 + np.exp(-logit))

    def infer_features(
        self,
        features: pd.DataFrame | Mapping[str, float] | np.ndarray,
        threshold_key: str | None = None,
    ) -> dict[str, Any] | pd.DataFrame:
        """Infers raw margin and calibrated probability from precomputed features.

        Parameters
        ----------
        features : pd.DataFrame, dict, or np.ndarray
            Input tabular features.
            - If DataFrame: must contain the 113 ordered feature columns.
            - If dict: maps feature name -> float.
            - If np.ndarray: shape (N, 113) or (113,).
        threshold_key : str, optional
            Operating threshold key ('target_1pct_real_fpr', 'pooled_1pct_real_fpr', 'macro_f1').
            Defaults to self.deployed_threshold_key.

        Returns
        -------
        dict or pd.DataFrame
            Contains 'raw_margin', 'calibrated_prob', 'prediction', and 'decision'.
        """
        th_key = self.deployed_threshold_key if threshold_key is None else threshold_key
        if th_key not in self.operating_thresholds:
            raise ValueError(
                f"Unknown threshold_key '{th_key}'. Choose from: {list(self.operating_thresholds.keys())}"
            )

        th_info = self.operating_thresholds[th_key]
        th_val = float(th_info["threshold"])
        margin_th_val = float(th_info["margin_threshold"])

        # Convert input to float32 NumPy array of shape (N, 113)
        single_sample = False
        if isinstance(features, pd.DataFrame):
            if features.columns.has_duplicates:
                raise ValueError("Input DataFrame contains duplicate columns")
            missing = [c for c in self.ordered_feature_names if c not in features.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing {len(missing)} features: {missing[:5]}...")
            X = features[self.ordered_feature_names].to_numpy(dtype=np.float32)
        elif isinstance(features, Mapping):
            single_sample = True
            missing = [c for c in self.ordered_feature_names if c not in features]
            if missing:
                raise ValueError(f"Input dict is missing {len(missing)} features: {missing[:5]}...")
            row = [float(features[c]) for c in self.ordered_feature_names]
            X = np.asarray([row], dtype=np.float32)
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                single_sample = True
                if features.shape[0] != 113:
                    raise ValueError(f"Expected 113 features, array has shape {features.shape}")
                X = features.reshape(1, 113).astype(np.float32)
            elif features.ndim == 2:
                if features.shape[1] != 113:
                    raise ValueError(f"Expected 113 features per row, array has shape {features.shape}")
                X = features.astype(np.float32)
            else:
                raise ValueError(f"Expected 1D or 2D array, got shape {features.shape}")
        else:
            raise TypeError(f"Unsupported features type: {type(features).__name__}")

        if X.shape[0] == 0:
            raise ValueError("Input feature matrix must not be empty")
        if not np.isfinite(X).all():
            raise ValueError("Input feature matrix contains nonfinite values (NaN or Inf)")

        # Predict raw margin via LightGBM booster
        with self._lock:
            raw_margin = np.asarray(self._booster.predict(X, raw_score=True), dtype=np.float64)

        # Calibrated probability via exact Platt scaling
        cal_prob = self._calibrate_margin(raw_margin)

        if not np.isfinite(raw_margin).all() or not np.isfinite(cal_prob).all():
            raise DetectorIntegrityError("Model produced nonfinite scores")

        # Discrete prediction and decision
        preds = (cal_prob >= th_val).astype(np.int64)
        decisions = np.where(preds == 1, "GENERATED", "REAL")

        if single_sample:
            return {
                "raw_margin": float(raw_margin[0]),
                "calibrated_prob": float(cal_prob[0]),
                "prediction": int(preds[0]),
                "decision": str(decisions[0]),
                "threshold": th_val,
                "margin_threshold": margin_th_val,
                "threshold_key": th_key,
                "status": "SUCCESS",
                "fail_closed": False,
            }

        return pd.DataFrame({
            "raw_margin": raw_margin,
            "calibrated_prob": cal_prob,
            "prediction": preds,
            "decision": decisions,
        })

    def infer_video(
        self,
        video_path: str | Path,
        threshold_key: str | None = None,
    ) -> dict[str, Any]:
        """Runs end-to-end inference on a video file with fail-closed integrity validation.

        Parameters
        ----------
        video_path : str or Path
            Path to video media file.
        threshold_key : str, optional
            Operating threshold key. Defaults to self.deployed_threshold_key.

        Returns
        -------
        dict
            Structured result containing scores, decisions, QC, and runtime.
            On any failure, strictly returns fail-closed status 'ABSTAIN'.
        """
        start_time = time.perf_counter()
        v_path = Path(video_path)
        clip_id = v_path.stem
        th_key = self.deployed_threshold_key if threshold_key is None else threshold_key

        th_info = self.operating_thresholds.get(th_key)
        th_val = float(th_info["threshold"]) if th_info is not None else None
        margin_th_val = float(th_info["margin_threshold"]) if th_info is not None else None

        result: dict[str, Any] = {
            "clip_id": clip_id,
            "raw_margin": None,
            "calibrated_prob": None,
            "prediction": None,
            "decision": "ABSTAIN",
            "status": "INCOMPLETE",
            "threshold": th_val,
            "margin_threshold": margin_th_val,
            "threshold_key": th_key,
            "fail_closed": True,
            "qc": {},
            "runtime_seconds": 0.0,
        }

        try:
            if th_info is None:
                raise DetectorIntegrityError(f"Unknown threshold key: {th_key}")
            # 1. Video file validation
            if not v_path.is_file():
                raise ExtractionError(f"Video file does not exist: {v_path}")
            if v_path.stat().st_size == 0:
                raise ExtractionError(f"Video file is empty: {v_path}")

            # 2. Canonical-time video decoding
            t0 = time.perf_counter()
            decoded = decode_canonical_video(v_path)
            decode_sec = time.perf_counter() - t0

            result["qc"] = {
                "n_frames_container": decoded["n_frames"],
                "fps": decoded["fps"],
                "duration_seconds": float(decoded["times"][-1] - decoded["times"][0]) if len(decoded["times"]) > 1 else 0.0,
                "n_observations": len(decoded["times"]),
                "decode_seconds": decode_sec,
            }

            result["qc"].update({"codec": decoded["codec"], "container_duration_seconds": decoded["duration_seconds"]})
            q = self.qc_policy
            if (len(decoded["times"]) < q["min_frames_required"]
                    or len(decoded["times"]) - 1 < q["min_transitions_required"]
                    or not q["min_duration_seconds"] <= decoded["duration_seconds"] <= q["max_duration_seconds"]
                    or decoded["codec"] not in q["allowed_codecs"]
                    or decoded["decode_truncated"]):
                raise ExtractionError("Video violates declared frame/transition/duration/codec/decode QC policy")

            # 3. Exact 113 feature extraction
            t0 = time.perf_counter()
            features = extract_features_from_decoded(
                gray128=decoded["gray128"],
                gray_flow=decoded["gray_flow"],
                grid=decoded["grid"],
                times=decoded["times"],
                dt_grid=decoded["dt_grid"],
                fs_grid=decoded["fs_grid"],
            )
            result["qc"]["feature_extraction_seconds"] = time.perf_counter() - t0

            # 4. Model scoring
            t0 = time.perf_counter()
            inf_res = self.infer_features(features, threshold_key=th_key)
            result["qc"]["inference_seconds"] = time.perf_counter() - t0

            # 5. Populate success result
            result.update({
                "raw_margin": inf_res["raw_margin"],
                "calibrated_prob": inf_res["calibrated_prob"],
                "prediction": inf_res["prediction"],
                "decision": inf_res["decision"],
                "status": "SUCCESS",
                "fail_closed": False,
            })

        except Exception as exc:
            # Fail closed strictly on any error
            result.update({
                "status": "ABSTAIN",
                "decision": "ABSTAIN",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "fail_closed": True,
            })

        result["runtime_seconds"] = time.perf_counter() - start_time
        return result
