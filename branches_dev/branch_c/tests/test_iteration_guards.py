"""Regression coverage for the proposal's real-only and reproducibility invariants."""
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from branch_c.cli import main
from branch_c.config import Config, FeatureConfig, ModelConfig, EvaluationConfig, load_config
from branch_c.evaluation import metrics, bootstrap_intervals, choose_real_threshold
from branch_c.manifest import load_manifest
from branch_c.model import load_bundle
from branch_c.pipeline import fit, make_oof, evaluate, _write_scores
from branch_c.video import read_video


@pytest.mark.parametrize("section,parameters", [
    (FeatureConfig, {"image_size": 96.5}),
    (FeatureConfig, {"sample_fps": "8"}),
    (FeatureConfig, {"sample_fps": True}),
    (FeatureConfig, {"stride_seconds": .2}),
    (FeatureConfig, {"sample_fps": float("nan")}),
    (ModelConfig, {"n_estimators": 20.5}),
    (ModelConfig, {"algorithm": ["isolation_forest"]}),
    (ModelConfig, {"max_samples": True}),
    (ModelConfig, {"seed": -1}),
    (ModelConfig, {"seed": 2**32}),
    (ModelConfig, {"nu": float("nan")}),
    (EvaluationConfig, {"bootstrap_repeats": 1.5}),
    (EvaluationConfig, {"real_false_positive_rate": True}),
])
def test_invalid_config_fails_before_processing(section, parameters):
    with pytest.raises(ValueError):
        section(**parameters)


@pytest.mark.parametrize("data", [
    [], {"features": []}, {"model": {"typo": 1}},
    {"heldout_generators": "pika"}, {"heldout_real_sources": [""]},
    {"heldout_generators": ["pika", "pika"]},
])
def test_malformed_config_json(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_config(path)


def test_cli_invalid_config_has_clear_error_without_traceback(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"features":{"typo":1}}')
    assert main(["validate", "--manifest", "unused.csv", "--config", str(path)]) == 2
    error = capsys.readouterr().err
    assert "Unknown features keys" in error and "Traceback" not in error


def test_single_class_report_does_not_invent_binary_comparison():
    result = metrics([0, 0], [.1, .2], .5)
    assert result["macro_f1"] is None
    assert result["auroc"] is None
    assert result["precision_generated"] is None
    assert result["specificity"] == 1.
    assert result["binary_comparison_available"] is False
    assert metrics([1, 1], [.8, .9], .5)["macro_f1"] is None


@pytest.mark.parametrize("labels,scores,threshold", [
    ([0, .5], [.1, .2], .5), ([0, 2], [.1, .2], .5),
    ([[0, 1]], [[.1, .2]], .5), ([0, 1], [.1, .2], float("nan")),
])
def test_invalid_metric_inputs_are_rejected(labels, scores, threshold):
    with pytest.raises(ValueError):
        metrics(labels, scores, threshold)


def test_bootstrap_validates_group_alignment_and_repeat_type():
    with pytest.raises(ValueError, match="group ID"):
        bootstrap_intervals([0, 1], [.1, .9], [0], .5)
    with pytest.raises(ValueError, match="integer"):
        bootstrap_intervals([0, 1], [.1, .9], [0, 1], .5, repeats=1.5)
    with pytest.raises(ValueError):
        choose_real_threshold([.1, .2], [0, 1], 1.)


def make_dataset(tmp_path, protocol="synthetic_smoke", train_count=12):
    rng = np.random.default_rng(44)
    records = []
    for split, count in [("train", train_count), ("val", 4), ("test", 4)]:
        for i in range(count):
            label = i % 2
            clip_id = f"{split}_{i}"
            file = tmp_path / f"{clip_id}.npz"
            np.savez(file, frames_rgb=rng.integers(0, 255, (16, 32, 32, 3), dtype=np.uint8),
                     timestamps_seconds=np.arange(16) / 8.)
            records.append(dict(clip_id=clip_id, path=str(file), label=label, split=split,
                                group_id=f"group_{clip_id}", source="toy_camera",
                                generator="toy_generator" if label else "", protocol=protocol))
    path = write_manifest(tmp_path / "manifest.csv", records)
    config = Config(features=FeatureConfig(image_size=32), model=ModelConfig(n_estimators=8),
                    evaluation=EvaluationConfig(bootstrap_repeats=10))
    return path, config, records


def write_manifest(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def test_fit_never_requires_generated_train_or_any_test_files(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    for record in records:
        if record["split"] == "test" or record["split"] == "train" and record["label"] == 1:
            Path(record["path"]).unlink()
    report = fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    assert report["run_kind"] == "synthetic_smoke"
    assert report["is_benchmark_result"] is False
    bundle = load_bundle(tmp_path / "fit/model.joblib")
    assert len(bundle["fit_manifest_rows"]) == 12  # Metadata remains available for leakage checks.
    assert len(bundle["training_content_hashes"]) == 6
    assert len(bundle["clip_model"].fit_clip_ids) == 6
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 10  # Six real train, four val.


def test_synthetic_provenance_survives_default_fit_oof_and_csv(tmp_path):
    manifest, config, _ = make_dataset(tmp_path)
    fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    make_oof(manifest, config, tmp_path / "oof", tmp_path / "cache", 3)
    for path in [tmp_path / "fit/validation_scores.csv", tmp_path / "oof/oof_scores.csv"]:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert {r["run_kind"] for r in rows} == {"synthetic_smoke"}
        assert {r["protocol"] for r in rows} == {"synthetic_smoke"}
    with pytest.raises(ValueError, match="cannot be reported"):
        fit(manifest, config, tmp_path / "bad", tmp_path / "cache", "research")


def test_different_test_protocol_is_rejected_before_decode(tmp_path):
    manifest, config, records = make_dataset(tmp_path, protocol="in_domain")
    report = fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    assert report["is_benchmark_result"] is None
    test = [r.copy() for r in records if r["split"] == "test"]
    for record in test:
        record["protocol"] = "cross_generator"
        Path(record["path"]).unlink()
    path = write_manifest(tmp_path / "different.csv", test)
    with pytest.raises(ValueError, match="protocol differs"):
        evaluate(path, tmp_path / "fit/model.joblib", tmp_path / "evaluation", tmp_path / "cache")


def test_score_export_rejects_silent_row_truncation(tmp_path):
    manifest, config, _ = make_dataset(tmp_path)
    rows = load_manifest(manifest, config)
    with pytest.raises(ValueError, match="one finite score"):
        _write_scores(tmp_path / "bad.csv", rows, [0.1], "final_model")
    assert not (tmp_path / "bad.csv").exists()


def test_duplicate_csv_header_is_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("clip_id,path,label,split,group_id,source,generator,label\na,a.mp4,0,train,g,s,,1\n")
    with pytest.raises(ValueError, match="duplicate column"):
        load_manifest(path)


def test_changed_id_cannot_hide_reused_training_path(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    record = next(r.copy() for r in records if r["split"] == "train" and r["label"] == 1)
    record.update(clip_id="changed_id", group_id="changed_group", split="test")
    Path(record["path"]).unlink()  # It should be rejected by metadata before decoding.
    test_manifest = write_manifest(tmp_path / "reused.csv", [record])
    with pytest.raises(ValueError, match="overlaps fitted/validation"):
        evaluate(test_manifest, tmp_path / "fit/model.joblib", tmp_path / "test", tmp_path / "cache")


def test_explicit_research_flag_cannot_relabel_toy_run(tmp_path, capsys):
    manifest, _, _ = make_dataset(tmp_path)
    output = tmp_path / "bad_research"
    assert main(["fit", "--manifest", str(manifest), "--output-dir", str(output),
                 "--run-kind", "research"]) == 2
    assert "cannot be reported" in capsys.readouterr().err
    assert not output.exists()


def test_cli_fit_stdout_remains_parseable_json_with_progress(tmp_path, capsys):
    manifest, config, _ = make_dataset(tmp_path, train_count=50)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    assert main(["fit", "--manifest", str(manifest), "--config", str(config_path),
                 "--output-dir", str(tmp_path / "fit"), "--cache-dir", str(tmp_path / "cache")]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["run_kind"] == "synthetic_smoke"
    assert "Extracted 25/29 clips" in captured.err


def test_existing_model_is_rejected_before_reading_manifest(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    model = output / "model.joblib"
    model.write_bytes(b"existing_result")
    with pytest.raises(ValueError, match="model already exists"):
        fit(tmp_path / "missing.csv", Config(), output, tmp_path / "cache")
    assert model.read_bytes() == b"existing_result"


class FakeCapture:
    def __init__(self, frames, fps):
        self.frames, self.fps, self.index, self.released = frames, fps, 0, False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return self.fps if prop == cv2.CAP_PROP_FPS else len(self.frames)

    def read(self):
        if self.index == len(self.frames):
            return False, None
        self.index += 1
        return True, self.frames[self.index - 1]

    def release(self):
        self.released = True


@pytest.mark.parametrize("count,expected_frames", [(60, 16), (61, 16), (92, 24), (600, 160)])
def test_cfr_grid_stays_inside_decoded_duration(tmp_path, monkeypatch, count, expected_frames):
    import branch_c.video as module
    path = tmp_path / "test.mp4"
    path.touch()
    capture = FakeCapture(np.zeros((count, 32, 32, 3), np.uint8), 30.)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: capture)
    video = read_video(path, FeatureConfig(image_size=32))
    assert len(video.frames_rgb) == expected_frames
    assert video.timestamps_seconds[-1] + 1 / 8 <= count / 30 + 1e-6
    assert 0 <= video.metadata["discarded_tail_seconds"] < 1 / 8 + 1e-6
    assert capture.released


@pytest.mark.parametrize("count", [59, 601])
def test_decode_rejects_true_duration_outside_bounds(tmp_path, monkeypatch, count):
    import branch_c.video as module
    path = tmp_path / "test.mp4"
    path.touch()
    capture = FakeCapture(np.zeros((count, 32, 32, 3), np.uint8), 30.)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="duration|maximum"):
        read_video(path, FeatureConfig(image_size=32))
    assert capture.released
