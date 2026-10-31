from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .cache import cached_features, file_sha256
from .config import Config, fingerprint, write_json
from .evaluation import bootstrap_intervals, choose_real_threshold, metrics
from .features import FEATURE_NAMES, feature_signature, extract_features
from .folds import build_fold_plan, hash_train_rows, preflight_oof
from .manifest import connected_groups, load_manifest, manifest_protocol, resolve_run_kind
from .model import RealOnlyAnomalyModel, load_bundle, runtime_versions, save_bundle
from .review import write_error_review
from .video import read_video, validate_frames


def _prepare(manifest, config, cache_dir, splits, *, manifest_rows=None, real_training_only=False,
             expected_hashes=None, workers=1):
    from .config import require_integer
    require_integer("workers", workers, lower=1, upper=32)
    all_rows = load_manifest(manifest, config) if manifest_rows is None else manifest_rows
    rows = [r for r in all_rows if r.split in splits
            and (not real_training_only or r.split != "train" or r.label == 0)]
    if not rows:
        raise ValueError(f"No clips found in {splits}")
    def prepare_one(row):
        expected = expected_hashes[row.clip_id] if expected_hashes is not None else row.metadata.get("content_sha256")
        try:
            return cached_features(row.path, config.features, cache_dir, expected_sha256=expected or None)
        except (ValueError, OSError) as error:
            raise ValueError(f"Feature preparation failed for {row.clip_id}: {error}") from error
    features, hashes = [], []
    import cv2
    previous_threads = cv2.getNumThreads()
    pool = None
    try:
        if workers > 1:
            cv2.setNumThreads(1)
            pool = ThreadPoolExecutor(max_workers=workers)
        prepared = pool.map(prepare_one, rows) if pool else map(prepare_one, rows)
        for i, (feature, digest) in enumerate(prepared):
            features.append(feature)
            hashes.append(digest)
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"Extracted {i + 1}/{len(rows)} clips (workers={workers})", file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
            cv2.setNumThreads(previous_threads)
    seen = {}
    for row, digest in zip(rows, hashes):
        partition = (row.split, row.metadata.get("validation_role", ""))
        if digest in seen and seen[digest] != partition:
            raise ValueError(f"Identical file content crosses splits: {row.clip_id}")
        seen[digest] = partition
    return rows, features, hashes


def _write_scores(path, rows, scores, score_kind, fold_ids=None, model_id=None, run_kind="auto", content_hashes=None):
    scores = np.asarray(scores)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("Exactly one finite score is required per clip")
    if fold_ids is not None and len(fold_ids) != len(rows):
        raise ValueError("One fold ID is required per score")
    if content_hashes is not None and (len(content_hashes) != len(rows) or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in content_hashes)):
        raise ValueError("One SHA256 content hash is required per score")
    run_kind = resolve_run_kind(rows, run_kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "label", "split", "group_id", "source", "generator", "branch_c_score",
              "score_kind", "fold_id", "model_id", "is_probability", "run_kind", "protocol", "content_sha256",
              "validation_role", "evaluation_subset"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow({"clip_id": row.clip_id, "label": row.label, "split": row.split,
                "group_id": row.group_id, "source": row.source, "generator": row.generator,
                "branch_c_score": float(score), "score_kind": score_kind,
                "fold_id": int(fold_ids[i]) if fold_ids is not None else "",
                "model_id": model_id or "", "is_probability": "false", "run_kind": run_kind,
                "protocol": row.metadata.get("protocol", ""),
                "content_sha256": content_hashes[i] if content_hashes is not None else "",
                "validation_role": row.metadata.get("validation_role", ""),
                "evaluation_subset": row.metadata.get("evaluation_subset", "")})


def _new_bundle(config, clip_model, window_model, rows, hashes, run_kind, manifest_rows):
    training = [(r.clip_id, h) for r, h in zip(rows, hashes) if r.split == "train"]
    return {"format_version": 1, "feature_names": FEATURE_NAMES,
            "feature_signature": feature_signature(config.features), "config": config,
            "clip_model": clip_model, "window_model": window_model, "run_kind": run_kind,
            "protocol": manifest_protocol(manifest_rows),
            "versions": runtime_versions(), "fit_manifest_rows": [asdict(r) | {"path": str(r.path)} for r in manifest_rows if r.split == "train"],
            "training_content_hashes": [h for r, h in zip(rows, hashes) if r.split == "train"],
            "model_id": fingerprint({"config": config.to_dict(), "training": training,
                                     "signature": feature_signature(config.features)})}


def fit(manifest, config: Config, output_dir, cache_dir, run_kind="auto", workers=1):
    started = time.perf_counter()
    output_dir = Path(output_dir)
    if (output_dir / "model.joblib").exists():
        raise ValueError("Output model already exists; choose a new run directory")
    if any((output_dir / name).exists() for name in ("fit_report.json", "validation_scores.csv")):
        raise ValueError("Fit output already exists; choose a new run directory")
    manifest_rows = load_manifest(manifest, config)
    run_kind = resolve_run_kind(manifest_rows, run_kind)
    if sum(r.split == "train" and r.label == 0 for r in manifest_rows) < 4:
        raise ValueError("At least 4 real training clips are required for fit")
    if not any(r.split == "val" and r.label == 0 for r in manifest_rows):
        raise ValueError("fit requires real validation clips for its fixed anomaly threshold")
    rows, features, hashes = _prepare(manifest, config, cache_dir, {"train", "val"},
                                     manifest_rows=manifest_rows, real_training_only=True, workers=workers)
    return _fit_prepared(manifest, config, output_dir, rows, features, hashes, run_kind,
                         manifest_rows, started)


def _fit_prepared(manifest, config, output_dir, rows, features, hashes, run_kind,
                  manifest_rows, started, selection_provenance=None):
    """Fit from the same prepared train/val features used by finite model selection."""
    output_dir = Path(output_dir)
    if (output_dir / "model.joblib").exists():
        raise ValueError("Output model already exists; choose a new run directory")
    train = np.array([r.split == "train" for r in rows])
    val = np.array([r.split == "val" for r in rows])
    if not train.any() or not val.any():
        raise ValueError("fit requires separate train and val rows; test is never decoded here")
    X = np.stack([f.clip for f in features])
    labels = np.array([r.label for r in rows])
    ids = np.array([r.clip_id for r in rows])
    clip_model = RealOnlyAnomalyModel(config.model).fit(X[train], labels[train], ids[train])
    # Separate window distribution avoids applying a full-clip detector to short windows.
    window_X, window_labels, window_ids = [], [], []
    for row, feature in zip(rows, features):
        if row.split == "train" and row.label == 0:
            indices = np.unique(np.linspace(0, len(feature.windows) - 1, min(4, len(feature.windows))).astype(int))
            window_X.extend(feature.windows[indices])
            window_labels.extend([0] * len(indices))
            window_ids.extend([row.clip_id] * len(indices))
    window_model = RealOnlyAnomalyModel(config.model).fit(window_X, window_labels, window_ids)
    val_scores = clip_model.score(X[val])
    from .manifest import validation_masks
    selection_mask, calibration_mask = validation_masks(rows)
    calibration_val = calibration_mask[val]
    real_val_n = int(((labels[val] == 0) & calibration_val).sum())
    if real_val_n < 20:
        warnings.warn(f"Only {real_val_n} real validation clips: threshold estimates are coarse")
    threshold = choose_real_threshold(val_scores[calibration_val], labels[val][calibration_val], config.evaluation.real_false_positive_rate)
    bundle = _new_bundle(config, clip_model, window_model, rows, hashes, run_kind, manifest_rows)
    bundle["threshold"] = threshold
    bundle["threshold_comparison"] = ">"
    bundle["validation_content_hashes"] = [h for r, h in zip(rows, hashes) if r.split == "val"]
    bundle["validation_manifest_rows"] = [asdict(r) | {"path": str(r.path)} for r in rows if r.split == "val"]
    if selection_provenance is not None:
        bundle["selection_provenance"] = selection_provenance
    save_bundle(output_dir / "model.joblib", bundle)
    report = {"run_kind": run_kind, "is_benchmark_result": False if run_kind == "synthetic_smoke" else None,
        "protocol": bundle["protocol"], "benchmark_note": "Dataset provenance must be established separately; research mode does not certify a benchmark",
        "model_id": bundle["model_id"], "config": config.to_dict(), "versions": bundle["versions"],
        "manifest_sha256": file_sha256(manifest), "feature_signature": bundle["feature_signature"],
        "clip_model": clip_model.metadata(), "window_model": window_model.metadata(),
        "threshold": threshold, "threshold_rule": "score > threshold; real calibration-role val quantile when roles exist; otherwise real val",
        "threshold_clip_ids": ids[calibration_mask & (labels == 0)].tolist(),
        "target_real_false_positive_rate": config.evaluation.real_false_positive_rate,
        "real_validation_clips": real_val_n, "validation": metrics(labels[val], val_scores, threshold),
        "fit_seconds_including_feature_extraction": time.perf_counter() - started,
        "timing_scope": "selection_and_final_fit_including_preparation" if selection_provenance else "fit_including_preparation",
        "model_size_bytes": (output_dir / "model.joblib").stat().st_size,
        "limitations": ["Raw anomaly evidence, not AI probability", "Generalisation is unverified until real benchmark evaluation",
                        "Window scores use a separate model and cannot be compared numerically with clip scores"]}
    selection_val = selection_mask[val]
    selection_threshold = choose_real_threshold(val_scores[selection_val], labels[val][selection_val],
                                               config.evaluation.real_false_positive_rate)
    report["model_selection_validation"] = metrics(labels[val][selection_val], val_scores[selection_val], selection_threshold)
    report["calibration_validation"] = metrics(labels[val][calibration_val], val_scores[calibration_val], threshold)
    if selection_provenance is not None:
        report["selection_provenance"] = selection_provenance
    _write_scores(output_dir / "validation_scores.csv", [r for r in rows if r.split == "val"], val_scores,
                  "final_model", model_id=bundle["model_id"], run_kind=run_kind,
                  content_hashes=[h for r, h in zip(rows, hashes) if r.split == "val"])
    write_json(output_dir / "fit_report.json", report)
    return report


def make_oof(manifest, config: Config, output_dir, cache_dir, n_splits=5, run_kind="auto", workers=1):
    from .config import require_integer
    require_integer("n_splits", n_splits, lower=2)
    output_dir = Path(output_dir)
    if (output_dir / "oof_scores.csv").exists():
        raise ValueError("OOF output already exists; choose a new run directory")
    if any((output_dir / name).exists() for name in ("fold_assignments.csv", "oof_audit.json")):
        raise ValueError("OOF output already exists; choose a new run directory")
    manifest_hash = file_sha256(manifest)
    manifest_rows = load_manifest(manifest, config)
    run_kind = resolve_run_kind(manifest_rows, run_kind)
    train_rows = [r for r in manifest_rows if r.split == "train"]
    preflight_oof(train_rows, n_splits)
    plan_hashes = hash_train_rows(train_rows)
    folds, groups, plan = build_fold_plan(train_rows, plan_hashes, n_splits, config.model.seed)
    rows, features, hashes = _prepare(manifest, config, cache_dir, {"train"}, manifest_rows=manifest_rows,
                                     expected_hashes={r.clip_id: h for r, h in zip(train_rows, plan_hashes)}, workers=workers)
    X = np.stack([f.clip for f in features])
    labels = np.array([r.label for r in rows])
    ids = np.array([r.clip_id for r in rows])
    scores = np.full(len(rows), np.nan)
    audits = []
    for fold in range(n_splits):
        training, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(groups[training]) & set(groups[held]):
            raise ValueError("Group leakage in OOF split")
        model = RealOnlyAnomalyModel(config.model).fit(X[training], labels[training], ids[training])
        scores[held] = model.score(X[held])
        audits.append({"fold_id": fold, "fit_clip_ids": model.fit_clip_ids,
                       "heldout_clip_ids": ids[held].tolist(), "fit_labels": [0],
                       "n_real_training": model.fit_rows})
    if not np.isfinite(scores).all():
        raise ValueError("OOF did not produce exactly one finite score per training clip")
    if file_sha256(manifest) != manifest_hash:
        raise ValueError("Manifest changed during OOF; results were not exported")
    _write_scores(output_dir / "oof_scores.csv", rows, scores, "out_of_fold", folds, run_kind=run_kind, content_hashes=hashes)
    _write_scores(output_dir / "fold_assignments.csv", rows, scores, "out_of_fold", folds, run_kind=run_kind, content_hashes=hashes)
    write_json(output_dir / "oof_audit.json", {"run_kind": run_kind, "n_splits": n_splits,
        "protocol": manifest_protocol(rows),
        "config": config.to_dict(), "manifest_sha256": manifest_hash, "fold_plan": plan,
        "feature_signature": feature_signature(config.features), "folds": audits,
        "integration": "Reuse these fold_id values for A/B; merge scores by clip_id, not row order"})
    return {"n_clips": len(rows), "n_splits": n_splits, "output_dir": str(output_dir)}


def _check_evaluation_leakage(bundle, rows, hashes=None):
    previous = bundle["fit_manifest_rows"] + bundle["validation_manifest_rows"]
    previous_ids = {r["clip_id"] for r in previous}
    previous_paths = {r["path"] for r in previous}
    previous_hashes = set(bundle["training_content_hashes"] + bundle["validation_content_hashes"])
    from .manifest import GROUP_KEYS
    previous_groups = {key: {r["metadata"].get(key, "") for r in previous} - {""} for key in GROUP_KEYS}
    for row, digest in zip(rows, hashes if hashes is not None else [None] * len(rows)):
        if row.clip_id in previous_ids or str(row.path) in previous_paths or digest in previous_hashes:
            raise ValueError(f"Test clip overlaps fitted/validation data: {row.clip_id}")
        for key in GROUP_KEYS:
            value = row.metadata.get(key, "")
            if value and value in previous_groups[key]:
                raise ValueError(f"Test {key} overlaps fitted/validation data: {row.clip_id}")


def evaluate(manifest, model_path, output_dir, cache_dir, bootstrap_repeats=None, test_subset=None, workers=1):
    output_dir = Path(output_dir)
    if (output_dir / "test_report.json").exists():
        raise ValueError("Test report already exists; avoid repeated test-guided selection")
    if any((output_dir / name).exists() for name in ("test_scores.csv", "error_review.csv")):
        raise ValueError("Evaluation output already exists; preserve scores/annotations and choose a new directory")
    bundle = load_bundle(model_path)
    config = bundle["config"]
    manifest_rows = load_manifest(manifest, config)
    stored_protocol = _bundle_protocol(bundle)
    if manifest_protocol(manifest_rows) != stored_protocol:
        raise ValueError("Evaluation protocol differs from fitted/validation protocol")
    test_rows = [r for r in manifest_rows if r.split == "test"]
    if test_subset is not None:
        test_rows = [r for r in test_rows if r.metadata.get("evaluation_subset") == test_subset]
        if not test_rows:
            raise ValueError(f"No test clips in evaluation_subset={test_subset}")
    _check_evaluation_leakage(bundle, test_rows)
    rows, features, hashes = _prepare(manifest, config, cache_dir, {"test"}, manifest_rows=test_rows, workers=workers)
    _check_evaluation_leakage(bundle, rows, hashes)
    scores = bundle["clip_model"].score(np.stack([f.clip for f in features]))
    labels = np.array([r.label for r in rows])
    threshold = bundle["threshold"]
    report = {"run_kind": bundle["run_kind"], "is_benchmark_result": False if bundle["run_kind"] == "synthetic_smoke" else None,
              "protocol": stored_protocol,
              "evaluation_subset": test_subset,
              "model_id": bundle["model_id"], "model_sha256": file_sha256(model_path),
              "manifest_sha256": file_sha256(manifest), "threshold": threshold,
              "threshold_source": "real validation; unchanged for test", "metrics": metrics(labels, scores, threshold),
              "ci_95": bootstrap_intervals(labels, scores, connected_groups(rows, hashes), threshold,
                 config.evaluation.bootstrap_repeats if bootstrap_repeats is None else bootstrap_repeats,
                 config.evaluation.seed), "per_generator": {}, "per_generator_ci_95": {}, "per_real_source": {},
              "probability_calibration_metrics": "Not applicable to raw Branch C evidence; evaluate Brier/ECE on calibrated fusion"}
    # Each generator is compared to the same real test pool; generated-only AUROC is undefined.
    for generator in sorted({r.generator for r in rows if r.label == 1}):
        selected = np.array([r.label == 0 or r.generator == generator for r in rows])
        report["per_generator"][generator] = metrics(labels[selected], scores[selected], threshold)
        selected_rows = [r for r, keep in zip(rows, selected) if keep]
        selected_hashes = [h for h, keep in zip(hashes, selected) if keep]
        report["per_generator_ci_95"][generator] = bootstrap_intervals(
            labels[selected], scores[selected], connected_groups(selected_rows, selected_hashes), threshold,
            config.evaluation.bootstrap_repeats if bootstrap_repeats is None else bootstrap_repeats,
            config.evaluation.seed)
    for source in sorted({r.source for r in rows if r.label == 0}):
        selected = np.array([r.label == 0 and r.source == source for r in rows])
        report["per_real_source"][source] = metrics(labels[selected], scores[selected], threshold)
    _write_scores(output_dir / "test_scores.csv", rows, scores, "final_model", model_id=bundle["model_id"],
                  run_kind=bundle["run_kind"], content_hashes=hashes)
    report["error_review"] = write_error_review(output_dir / "error_review.csv", rows, scores, threshold,
                                               features, model_id=bundle["model_id"], run_kind=bundle["run_kind"])
    # The JSON report is the completion record and is written after both CSVs.
    write_json(output_dir / "test_report.json", report)
    return report


def _bundle_protocol(bundle):
    if "protocol" in bundle:
        return bundle["protocol"]
    rows = bundle["fit_manifest_rows"] + bundle["validation_manifest_rows"]
    protocols = {r["metadata"].get("protocol", "") for r in rows}
    if len(protocols) != 1:
        raise ValueError("Saved manifest protocol is inconsistent")
    return next(iter(protocols))


def score_feature_result(bundle, feature, clip_id="input"):
    clip_model = bundle["clip_model"]
    score = float(clip_model.score(feature.clip[None])[0])
    window_scores = bundle["window_model"].score(feature.windows)
    timeline = [{"start_seconds": float(start), "end_seconds": float(end),
                 "anomaly_score": float(value), "is_probability": False}
                for (start, end), value in zip(feature.bounds_seconds, window_scores)]
    return {"branch": "C", "clip_id": clip_id, "model_id": bundle["model_id"],
            "run_kind": bundle["run_kind"], "anomaly_score": score,
            "protocol": _bundle_protocol(bundle),
            "score_direction": "higher_is_more_anomalous", "is_probability": False,
            "threshold": bundle["threshold"], "is_anomalous": bool(score > bundle["threshold"]),
            "timeline_model": "separate_real_only_window_model",
            "timeline": timeline, "most_anomalous_window": timeline[int(np.argmax(window_scores))],
            "descriptor_deviations": clip_model.deviations(feature.clip),
            "explanation_note": "Standardized deviations are descriptive, not causal feature importance",
            "metadata": feature.metadata}


def predict_video(model_path, video_path, output_path=None):
    total_started = time.perf_counter()
    bundle = load_bundle(model_path)
    started = time.perf_counter()
    config = bundle["config"].features
    if feature_signature(config) != bundle["feature_signature"]:
        raise ValueError("Feature preprocessing signature differs from model")
    video = read_video(video_path, config)
    decoded = time.perf_counter()
    feature = extract_features(video, config)
    extracted = time.perf_counter()
    result = score_feature_result(bundle, feature, Path(video_path).stem)
    finished = time.perf_counter()
    # Preserve the existing field, and expose its previously implicit load exclusion.
    result["inference_seconds"] = finished - started
    result["timings"] = {"model_load_seconds": started - total_started,
                         "decode_and_sample_seconds": decoded - started,
                         "feature_extraction_seconds": extracted - decoded,
                         "scoring_and_explanation_seconds": finished - extracted,
                         "total_seconds_including_model_load": finished - total_started,
                         "scope": "Excludes Python/import startup and output JSON writing; no feature cache"}
    if output_path:
        write_json(output_path, result)
    return result


def predict_frames(model_path, frames_rgb, timestamps_seconds, clip_id="input"):
    """Public interface for teammates with common, unnormalised sampled RGB frames."""
    bundle = load_bundle(model_path)
    config = bundle["config"].features
    feature = extract_features(validate_frames(frames_rgb, timestamps_seconds, config), config)
    return score_feature_result(bundle, feature, clip_id)
