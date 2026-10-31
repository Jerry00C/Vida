import csv
from dataclasses import replace
import json
import numpy as np
import pytest

from branch_c.cache import cached_features
from branch_c.config import Config, FeatureConfig, ModelConfig, EvaluationConfig
from branch_c.evaluation import bootstrap_intervals, choose_real_threshold, metrics
from branch_c.features import FEATURE_NAMES, _frequency, extract_features
from branch_c.manifest import connected_groups, load_manifest
from branch_c.model import RealOnlyAnomalyModel, load_bundle, save_bundle
from branch_c.pipeline import fit, make_oof, evaluate, predict_frames
from branch_c.video import validate_frames


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    fields = list(dict.fromkeys(k for record in rows for k in record))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(i, split="train", label=0, **extra):
    return dict(clip_id=f"clip{i}", path=f"clip{i}.npz", label=str(label), split=split,
                group_id=f"g{i}", source="camera", generator="toy" if label else "", **extra)


@pytest.mark.parametrize("algorithm", ["isolation_forest", "one_class_svm"])
def test_generated_features_never_fit_scaler_or_model(algorithm):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, len(FEATURE_NAMES)))
    labels = np.array([0] * 8 + [1] * 4)
    ids = [str(i) for i in range(12)]
    cfg = ModelConfig(algorithm=algorithm, n_estimators=16)
    first = RealOnlyAnomalyModel(cfg).fit(X, labels, ids)
    changed = X.copy()
    changed[labels == 1] = np.nan
    second = RealOnlyAnomalyModel(cfg).fit(changed, labels, ids)
    np.testing.assert_allclose(first.pipeline[0].mean_, X[:8].mean(axis=0))
    np.testing.assert_allclose(first.score(X), second.score(X))
    assert first.fit_clip_ids == [str(i) for i in range(8)]


def test_score_direction():
    rng = np.random.default_rng(12)
    X = rng.normal(0, .02, (100, len(FEATURE_NAMES)))
    model = RealOnlyAnomalyModel(ModelConfig(n_estimators=32)).fit(X, np.zeros(100), [str(i) for i in range(100)])
    assert model.score(np.ones((1, len(FEATURE_NAMES))) * 100)[0] > model.score(X).mean()


@pytest.mark.parametrize("key", ["group_id", "pair_id", "duplicate_id", "source_video_id"])
def test_cross_partition_groups_are_rejected(tmp_path, key):
    rows = [row(0), row(1, "test")]
    rows[0][key] = rows[1][key] = "shared"
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(write_manifest(tmp_path, rows))


@pytest.mark.parametrize("change", ["label", "split", "generator", "clip_id", "path"])
def test_bad_manifests(tmp_path, change):
    rows = [row(0), row(1, label=1)]
    if change == "label": rows[0][change] = "fake"
    if change == "split": rows[0][change] = "validation"
    if change == "generator": rows[1][change] = ""
    if change == "clip_id": rows[1][change] = rows[0][change]
    if change == "path": rows[1][change] = rows[0][change]
    with pytest.raises(ValueError): load_manifest(write_manifest(tmp_path, rows))


def test_heldout_protocol_enforced(tmp_path):
    path = write_manifest(tmp_path, [row(0), row(1, label=1)])
    with pytest.raises(ValueError, match="exclusively in test"):
        load_manifest(path, replace(Config(), heldout_generators=("toy",)))


def test_connected_groups_merge_transitive_pairs(tmp_path):
    rows = [row(0, pair_id="p"), row(1, pair_id="p"), row(2, pair_id="other")]
    rows[1]["group_id"] = rows[2]["group_id"]
    clips = load_manifest(write_manifest(tmp_path, rows))
    assert len(np.unique(connected_groups(clips))) == 1


def test_static_descriptors_are_finite():
    cfg = FeatureConfig(image_size=32)
    frames = np.full((16, 32, 32, 3), 100, np.uint8)
    feature = extract_features(validate_frames(frames, np.arange(16) / 8, cfg), cfg)
    assert feature.clip.shape == (37,)
    assert feature.windows.shape == (1, 37)
    assert np.isfinite(feature.clip).all()
    np.testing.assert_allclose(feature.clip, 0, atol=1e-6)


def test_frequency_distinguishes_flicker_from_slow_change():
    flicker = np.tile((np.arange(32) % 2)[:, None, None], (1, 24, 24)).astype(np.float32)
    slow = np.tile(np.linspace(0, 1, 32)[:, None, None], (1, 24, 24)).astype(np.float32)
    assert _frequency(flicker)[2] > _frequency(slow)[2] + .5
    assert abs(_frequency(flicker)[:3].sum() - 1) < 1e-6


@pytest.mark.parametrize("kind", ["normalised", "nonuniform", "short", "duplicate_times"])
def test_input_frame_validation(kind):
    frames = np.zeros((16, 32, 32, 3), np.uint8)
    times = np.arange(16) / 8
    if kind == "normalised": frames = frames.astype(float) / 255
    if kind == "nonuniform": times[5] += .05
    if kind == "short": frames, times = frames[:8], times[:8]
    if kind == "duplicate_times": times[5] = times[4]
    with pytest.raises(ValueError): validate_frames(frames, times, FeatureConfig())


def test_cache_tracks_content_and_config(tmp_path):
    path = tmp_path / "clip.npz"
    frames = np.zeros((16, 32, 32, 3), np.uint8)
    np.savez(path, frames_rgb=frames, timestamps_seconds=np.arange(16)/8)
    config = FeatureConfig(image_size=32)
    first, first_hash = cached_features(path, config, tmp_path / "cache")
    cached, _ = cached_features(path, config, tmp_path / "cache")
    np.testing.assert_array_equal(first.clip, cached.clip)
    frames[1::2] = 255
    np.savez(path, frames_rgb=frames, timestamps_seconds=np.arange(16)/8)
    second, second_hash = cached_features(path, config, tmp_path / "cache")
    assert first_hash != second_hash
    assert not np.array_equal(first.clip, second.clip)
    cached_features(path, replace(config, image_size=48), tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3


def test_threshold_uses_only_real_validation():
    scores = np.array([.1, .2, .3, .4, 1., 2.])
    labels = np.array([0, 0, 0, 0, 1, 1])
    assert choose_real_threshold(scores, labels, .25) == .4
    scores[labels == 1] = -999
    assert choose_real_threshold(scores, labels, .25) == .4


def test_metrics_ties_single_class_and_group_bootstrap():
    y = np.array([0, 0, 1, 1])
    scores = np.ones(4)
    result = metrics(y, scores, 1.)
    assert result["auroc"] == .5
    assert result["equal_error_rate"] == .5
    assert result["confusion_matrix_real_generated"] == [[2, 0], [2, 0]]
    assert metrics(y[:2], scores[:2], 1.)["auroc"] is None
    ci = bootstrap_intervals(y, scores, np.array([0, 0, 1, 1]), 1., repeats=30)
    assert ci["valid_repeats"] > 0 and ci["auroc"] == [.5, .5]


@pytest.fixture
def small_dataset(tmp_path):
    rng = np.random.default_rng(33)
    rows = []
    for split, count in [("train", 12), ("val", 4), ("test", 4)]:
        for i in range(count):
            label = i % 2
            record = row(f"{split}_{i}", split, label)
            frames = rng.integers(50, 200, (16, 32, 32, 3), dtype=np.uint8)
            if label: frames[1::2] = 255
            np.savez(tmp_path / record["path"], frames_rgb=frames, timestamps_seconds=np.arange(16) / 8.)
            rows.append(record)
    cfg = Config(features=FeatureConfig(image_size=32), model=ModelConfig(n_estimators=8),
                 evaluation=EvaluationConfig(bootstrap_repeats=20))
    return write_manifest(tmp_path, rows), cfg, rows


def test_fit_never_decodes_test_and_public_frames_api(small_dataset, tmp_path):
    manifest, cfg, rows = small_dataset
    for r in rows:
        if r["split"] == "test": (tmp_path / r["path"]).unlink()
    report = fit(manifest, cfg, tmp_path / "fit", tmp_path / "cache", "synthetic_smoke")
    assert report["clip_model"]["fit_labels"] == [0]
    assert len(report["clip_model"]["fit_clip_ids"]) == 6
    assert report["is_benchmark_result"] is False
    model = tmp_path / "fit/model.joblib"
    with np.load(tmp_path / rows[0]["path"]) as data:
        result = predict_frames(model, data["frames_rgb"], data["timestamps_seconds"])
    assert result["is_probability"] is False
    assert len(result["timeline"]) == 1
    assert result["is_anomalous"] == (result["anomaly_score"] > result["threshold"])


def test_oof_excludes_each_held_group(small_dataset, tmp_path):
    manifest, cfg, rows = small_dataset
    make_oof(manifest, cfg, tmp_path / "oof", tmp_path / "cache", n_splits=3)
    audit = json.loads((tmp_path / "oof/oof_audit.json").read_text())
    all_held = []
    for fold in audit["folds"]:
        assert not set(fold["fit_clip_ids"]) & set(fold["heldout_clip_ids"])
        assert fold["fit_labels"] == [0]
        all_held.extend(fold["heldout_clip_ids"])
    assert len(all_held) == len(set(all_held)) == 12


def test_evaluate_keeps_threshold_and_per_generator_pool(small_dataset, tmp_path):
    manifest, cfg, _ = small_dataset
    fit_report = fit(manifest, cfg, tmp_path / "fit", tmp_path / "cache", "synthetic_smoke")
    report = evaluate(manifest, tmp_path / "fit/model.joblib", tmp_path / "test", tmp_path / "cache")
    assert report["threshold"] == fit_report["threshold"]
    assert report["per_generator"]["toy"]["n_real"] == 2
    assert report["is_benchmark_result"] is False


def test_identical_content_train_validation_rejected(small_dataset, tmp_path):
    manifest, cfg, rows = small_dataset
    train = next(r for r in rows if r["split"] == "train")
    val = next(r for r in rows if r["split"] == "val")
    (tmp_path / val["path"]).write_bytes((tmp_path / train["path"]).read_bytes())
    with pytest.raises(ValueError, match="Identical file content"):
        fit(manifest, cfg, tmp_path / "fit", tmp_path / "cache")


def test_supplied_folds_cannot_split_connected_group(small_dataset, tmp_path):
    manifest, cfg, rows = small_dataset
    for i, record in enumerate(rows):
        if record["split"] == "train": record["fold_id"] = i % 3
    rows[0]["pair_id"] = rows[1]["pair_id"] = "shared_pair"
    manifest = write_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="crosses supplied OOF folds"):
        make_oof(manifest, cfg, tmp_path / "oof", tmp_path / "cache", 3)


def test_saved_artifact_rejects_changed_feature_signature(small_dataset, tmp_path):
    manifest, cfg, _ = small_dataset
    fit(manifest, cfg, tmp_path / "fit", tmp_path / "cache", "synthetic_smoke")
    bundle = load_bundle(tmp_path / "fit/model.joblib")
    bundle["feature_signature"] = "obsolete_implementation"
    save_bundle(tmp_path / "wrong.joblib", bundle)
    with pytest.raises(ValueError, match="differs from fitted artifact"):
        load_bundle(tmp_path / "wrong.joblib")


@pytest.mark.parametrize("overlap", ["content", "pair_id"])
def test_separate_test_manifest_cannot_bypass_fitted_data_checks(small_dataset, tmp_path, overlap):
    manifest, cfg, rows = small_dataset
    rows[0]["pair_id"] = "training_pair"
    write_manifest(tmp_path, rows)
    fit(manifest, cfg, tmp_path / "fit", tmp_path / "cache", "synthetic_smoke")
    test_rows = [r.copy() for r in rows if r["split"] == "test"]
    if overlap == "content":
        (tmp_path / test_rows[0]["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    else:
        test_rows[0]["pair_id"] = "training_pair"
    separate_manifest = write_manifest(tmp_path, test_rows)
    with pytest.raises(ValueError, match="overlaps fitted/validation"):
        evaluate(separate_manifest, tmp_path / "fit/model.joblib", tmp_path / "test", tmp_path / "cache")
