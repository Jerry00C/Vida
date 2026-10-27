"""Failure-review exports, preserved input metadata and concurrent cache safety."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

import branch_c.cache as cache
from branch_c.config import FeatureConfig
from branch_c.features import extract_features, feature_signature
from branch_c.manifest import Clip, load_manifest
from branch_c.pipeline import evaluate, fit, _write_scores
from branch_c.review import REVIEW_FIELDS, write_error_review
from branch_c.video import SampledVideo, read_video

from test_iteration_guards import make_dataset


def review_inputs(tmp_path, labels):
    rows = [Clip(f"clip_{i}", tmp_path / f"clip_{i}.npz", label, "test", f"g{i}",
                 "toy_source", "toy_generator" if label else "",
                 {"protocol": "synthetic_smoke"}) for i, label in enumerate(labels)]
    features = [SimpleNamespace(metadata={"duration_seconds": 2, "input_width": 64,
                                         "input_height": 48, "sample_fps": 8,
                                         "sampled_frames": 16, "feature_image_size": 32}) for _ in rows]
    return rows, features


def test_review_matches_confusion_and_strict_threshold_without_inventing_causes(tmp_path):
    rows, features = review_inputs(tmp_path, [0, 0, 1, 1, 0, 1])
    scores = [.8, .9, .2, .5, .3, .7]
    path = tmp_path / "error_review.csv"
    summary = write_error_review(path, rows, scores, .5, features, model_id="toy-model")
    assert summary["n_cases"] == 4
    assert summary["false_positives"] == summary["false_negatives"] == 2
    assert summary["review_status"] == "pending_human_review"
    with path.open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == REVIEW_FIELDS
        records = list(reader)
    assert [r["clip_id"] for r in records] == ["clip_1", "clip_0", "clip_2", "clip_3"]
    assert [r["error_type"] for r in records] == ["FP", "FP", "FN", "FN"]
    assert records[-1]["threshold_margin"] == "0.0"  # A generated score equal to threshold is FN.
    for record in records:
        assert record["run_kind"] == "synthetic_smoke"
        assert record["is_probability"] == "false"
        assert record["review_status"] == "pending"
        assert record["failure_type"] == record["notes"] == ""
        assert record["native_fps"] == record["native_width"] == ""  # NPZ has no native-video metadata.
        assert record["input_width"] == "64" and record["feature_image_size"] == "32"


def test_zero_error_review_still_writes_a_header_and_honest_status(tmp_path):
    rows, features = review_inputs(tmp_path, [0, 1])
    path = tmp_path / "empty.csv"
    summary = write_error_review(path, rows, [.1, .9], .5, features, model_id="m")
    assert summary["n_cases"] == 0 and summary["review_status"] == "no_errors_at_this_threshold"
    with path.open() as f:
        assert list(csv.DictReader(f)) == []
    assert path.read_text().startswith("clip_id,path,")


def test_review_preserves_manual_annotations(tmp_path):
    rows, features = review_inputs(tmp_path, [0, 1])
    path = tmp_path / "review.csv"
    path.write_text("manual notes to preserve")
    with pytest.raises(ValueError, match="preserve manual annotations"):
        write_error_review(path, rows, [.9, .1], .5, features, model_id="m")
    assert path.read_text() == "manual notes to preserve"


@pytest.mark.parametrize("change", ["feature_count", "score_count", "nonfinite", "toy_provenance"])
def test_review_invalid_inputs_fail_before_writing(tmp_path, change):
    rows, features = review_inputs(tmp_path, [0, 1])
    scores, run_kind = [.9, .1], "auto"
    if change == "feature_count": features = features[:1]
    if change == "score_count": scores = scores[:1]
    if change == "nonfinite": scores[0] = float("nan")
    if change == "toy_provenance": run_kind = "research"
    with pytest.raises(ValueError):
        write_error_review(tmp_path / "bad.csv", rows, scores, .5, features, model_id="m", run_kind=run_kind)
    assert not (tmp_path / "bad.csv").exists()


def test_evaluate_exports_exact_errors_with_fixed_validation_threshold(tmp_path, monkeypatch):
    from branch_c.model import RealOnlyAnomalyModel
    manifest, config, _ = make_dataset(tmp_path)
    fit_report = fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    threshold = fit_report["threshold"]
    # Force all confusion outcomes, independently of toy detector performance.
    monkeypatch.setattr(RealOnlyAnomalyModel, "score", lambda self, X: np.array([
        threshold + .2, threshold - .2, threshold - .1, threshold + .1]))
    output = tmp_path / "test"
    report = evaluate(manifest, tmp_path / "fit/model.joblib", output, tmp_path / "cache", 0)
    assert report["metrics"]["confusion_matrix_real_generated"] == [[1, 1], [1, 1]]
    assert report["error_review"]["false_positives"] == report["error_review"]["false_negatives"] == 1
    assert report["threshold"] == threshold
    with (output / "error_review.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["clip_id"] for r in rows} == {"test_0", "test_1"}
    assert {float(r["threshold"]) for r in rows} == {threshold}
    assert json.loads((output / "test_report.json").read_text()) == report
    with (output / "test_scores.csv").open() as f:
        exported = list(csv.DictReader(f))
    input_hashes = {r.clip_id: cache.file_sha256(r.path) for r in load_manifest(manifest) if r.split == "test"}
    assert {r["clip_id"]: r["content_sha256"] for r in exported} == input_hashes


@pytest.mark.parametrize("hashes", [["a" * 64], ["a" * 64, "invalid"], ["a" * 64, "z" * 64]])
def test_score_exports_reject_misaligned_or_invalid_content_hashes(tmp_path, hashes):
    rows, _ = review_inputs(tmp_path, [0, 1])
    with pytest.raises(ValueError, match="One SHA256 content hash"):
        _write_scores(tmp_path / "bad.csv", rows, [.1, .9], "final_model", content_hashes=hashes)
    assert not (tmp_path / "bad.csv").exists()


@pytest.mark.parametrize("existing", ["test_scores.csv", "error_review.csv"])
def test_partial_evaluation_cannot_overwrite_scores_or_manual_review(tmp_path, existing):
    output = tmp_path / "partial"
    output.mkdir()
    file = output / existing
    file.write_text("preserve")
    with pytest.raises(ValueError, match="Evaluation output already exists"):
        evaluate("missing.csv", "missing.joblib", output, "unused_cache")
    assert file.read_text() == "preserve"


def test_evaluation_completion_record_is_written_after_csv_exports(tmp_path, monkeypatch):
    import branch_c.pipeline as pipeline
    manifest, config, _ = make_dataset(tmp_path)
    fit(manifest, config, tmp_path / "fit", tmp_path / "cache")

    def fail_report(path, data):
        assert Path(path).name == "test_report.json"
        assert (Path(path).parent / "test_scores.csv").is_file()
        assert (Path(path).parent / "error_review.csv").is_file()
        raise OSError("simulated report-write failure")

    monkeypatch.setattr(pipeline, "write_json", fail_report)
    with pytest.raises(OSError, match="report-write failure"):
        evaluate(manifest, tmp_path / "fit/model.joblib", tmp_path / "test", tmp_path / "cache", 0)
    assert not (tmp_path / "test/test_report.json").exists()


def write_frames(path):
    rng = np.random.default_rng(19)
    np.savez(path, frames_rgb=rng.integers(0, 255, (16, 48, 64, 3), dtype=np.uint8),
             timestamps_seconds=np.arange(16) / 8.)
    return FeatureConfig(image_size=32)


def test_original_frame_dimensions_survive_resizing_and_revalidation(tmp_path):
    path = tmp_path / "input.npz"
    config = write_frames(path)
    video = read_video(path, config)
    result = extract_features(video, config)
    assert video.frames_rgb.shape[1:3] == (32, 32)
    assert result.metadata["input_width"] == 64 and result.metadata["input_height"] == 48
    assert result.metadata["feature_image_size"] == 32
    without_metadata = extract_features(SampledVideo(video.frames_rgb, video.timestamps_seconds, {}), config)
    np.testing.assert_array_equal(result.clip, without_metadata.clip)
    np.testing.assert_array_equal(result.windows, without_metadata.windows)


def test_new_cache_format_ignores_legacy_metadata_and_preserves_correct_dimensions(tmp_path):
    path = tmp_path / "input.npz"
    config = write_frames(path)
    directory = tmp_path / "cache"
    directory.mkdir()
    digest = cache.file_sha256(path)
    legacy = directory / f"{digest}-{feature_signature(config)}.npz"
    legacy.write_bytes(b"obsolete metadata cache")
    first, _ = cache.cached_features(path, config, directory)
    second, _ = cache.cached_features(path, config, directory)
    assert first.metadata["input_width"] == second.metadata["input_width"] == 64
    assert first.metadata["input_height"] == second.metadata["input_height"] == 48
    np.testing.assert_array_equal(first.clip, second.clip)
    assert len(list(directory.glob(f"{cache.CACHE_VERSION}-*.npz"))) == 1
    assert legacy.read_bytes() == b"obsolete metadata cache"


def test_concurrent_cold_cache_writers_leave_one_complete_file(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    config = write_frames(path)
    original = cache.extract_features
    barrier = Barrier(2)

    def simultaneous_extract(*args):
        barrier.wait(timeout=10)
        return original(*args)

    monkeypatch.setattr(cache, "extract_features", simultaneous_extract)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cache.cached_features, path, config, tmp_path / "cache") for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert results[0][1] == results[1][1]
    np.testing.assert_array_equal(results[0][0].clip, results[1][0].clip)
    hit, _ = cache.cached_features(path, config, tmp_path / "cache")
    np.testing.assert_array_equal(hit.clip, results[0][0].clip)
    assert len(list((tmp_path / "cache").iterdir())) == 1


def test_failed_cache_write_leaves_no_staging_file(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    config = write_frames(path)

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cache.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk failure"):
        cache.cached_features(path, config, tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_input_edit_during_extraction_cannot_poison_cache(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    config = write_frames(path)
    original = cache.extract_features

    def changed_input(*args):
        result = original(*args)
        path.write_bytes(path.read_bytes() + b"input changed")
        return result

    monkeypatch.setattr(cache, "extract_features", changed_input)
    with pytest.raises(ValueError, match="Input changed during feature extraction"):
        cache.cached_features(path, config, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("corruption", ["metadata", "numeric_type", "missing_fields"])
def test_corrupt_cache_returns_clear_schema_error(tmp_path, corruption):
    path = tmp_path / "input.npz"
    config = write_frames(path)
    directory = tmp_path / "cache"
    cache.cached_features(path, config, directory)
    cached = next(directory.glob("*.npz"))
    with np.load(cached, allow_pickle=False) as data:
        contents = {key: data[key] for key in data.files}
    if corruption == "metadata": contents["metadata"] = "[]"
    if corruption == "numeric_type": contents["clip"] = contents["clip"].astype(str)
    if corruption == "missing_fields": contents.pop("windows")
    np.savez_compressed(cached, **contents)
    with pytest.raises(ValueError, match="Corrupt feature cache|required fields"):
        cache.cached_features(path, config, directory)
