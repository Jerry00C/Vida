"""Verify sealed-test model selection and honest native-memory/time measurements."""
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from branch_c.benchmark import _rss_bytes, benchmark_video
from branch_c.cli import main
from branch_c.config import fingerprint, load_config
from branch_c.model import load_bundle
from branch_c.pipeline import fit, predict_video
from branch_c.selection import predeclared_candidates, select_model

from test_iteration_guards import make_dataset, write_manifest


def test_predeclared_grid_is_small_fixed_and_seeded():
    candidates = predeclared_candidates(123)
    assert [name for name, _ in candidates] == ["if_200_256", "if_100_128", "if_100_256",
                                               "if_200_128", "svm_nu_0.01", "svm_nu_0.05"]
    assert {c.seed for _, c in candidates} == {123}
    assert len({fingerprint(c.__dict__) for _, c in candidates}) == 6


def test_selection_reuses_features_and_never_opens_generated_train_or_test(tmp_path, monkeypatch):
    import branch_c.selection as selection
    manifest, config, records = make_dataset(tmp_path)
    for row in records:
        if row["split"] == "test" or row["split"] == "train" and row["label"] == 1:
            Path(row["path"]).unlink()
    preparation_calls = []
    original = selection._prepare

    def prepare(*args, **kwargs):
        preparation_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "_prepare", prepare)
    root = tmp_path / "selection"
    report = select_model(manifest, config, root, tmp_path / "cache")
    assert len(preparation_calls) == 1
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 10
    assert report["run_kind"] == "synthetic_smoke" and report["is_benchmark_result"] is False
    assert report["selection_split"] == "val" and report["test_files_read"] == 0
    assert {(r["split"], r["label"]) for r in report["inputs_read"]} == {("train", 0), ("val", 0), ("val", 1)}
    expected_ids = sorted(r["clip_id"] for r in records if r["split"] == "train" and r["label"] == 0)
    ranked = report["ranked_candidates"]
    assert len(ranked) == 6
    for candidate in ranked:
        assert candidate["model"]["fit_clip_ids"] == expected_ids
        assert candidate["model"]["fit_labels"] == [0]
        with (root / "candidates" / f"{candidate['candidate_id']}_validation_scores.csv").open() as f:
            scores = list(csv.DictReader(f))
        assert {r["split"] for r in scores} == {"val"}
        assert {r["run_kind"] for r in scores} == {"synthetic_smoke"}
        real_scores = [float(r["branch_c_score"]) for r in scores if r["label"] == "0"]
        assert candidate["threshold"] == np.quantile(real_scores, .95, method="higher")
    keys = [(-c["validation"]["auroc"], -c["validation"]["macro_f1"], c["predeclared_order"]) for c in ranked]
    assert keys == sorted(keys)
    plan = json.loads((root / "selection_plan.json").read_text())
    plan_id = plan.pop("plan_id")
    assert fingerprint(plan) == plan_id == report["plan_id"]
    selected_config = load_config(root / "selected_config.json")
    assert selected_config.features == config.features
    assert selected_config.evaluation == config.evaluation
    assert selected_config.heldout_generators == config.heldout_generators
    bundle = load_bundle(root / "selected/model.joblib")
    assert bundle["config"] == selected_config
    assert bundle["selection_provenance"]["plan_id"] == plan_id
    selected_fit = json.loads((root / "selected/fit_report.json").read_text())
    assert selected_fit["validation"] == ranked[0]["validation"]
    repeat = select_model(manifest, config, tmp_path / "selection_repeat", tmp_path / "cache")
    assert repeat["plan_id"] == plan_id
    assert repeat["ranked_candidates"] == ranked
    assert repeat["model_id"] == report["model_id"]


def test_selection_requires_both_validation_classes_before_decoding(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    records = [r for r in records if r["split"] != "val" or r["label"] == 0]
    write_manifest(manifest, records)
    for row in records:
        Path(row["path"]).unlink()
    with pytest.raises(ValueError, match="real and generated val"):
        select_model(manifest, config, tmp_path / "selection", tmp_path / "cache")
    assert not (tmp_path / "selection").exists()


def test_selection_output_and_toy_provenance_are_protected(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "selection_report.json").write_text("keep")
    with pytest.raises(ValueError, match="output already exists"):
        select_model("missing.csv", load_config(), output, tmp_path / "cache")
    assert (output / "selection_report.json").read_text() == "keep"
    manifest, config, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="cannot be reported"):
        select_model(manifest, config, tmp_path / "bad", tmp_path / "cache", "research")
    assert not (tmp_path / "bad").exists()


def test_failed_candidate_keeps_plan_but_cannot_report_completed_selection(tmp_path, monkeypatch):
    import branch_c.selection as selection
    manifest, config, _ = make_dataset(tmp_path)

    def fail(*args, **kwargs):
        raise ValueError("candidate fit failed")

    monkeypatch.setattr(selection.RealOnlyAnomalyModel, "fit", fail)
    with pytest.raises(ValueError, match="candidate fit failed"):
        select_model(manifest, config, tmp_path / "failed", tmp_path / "cache")
    assert (tmp_path / "failed/selection_plan.json").is_file()
    assert not (tmp_path / "failed/selection_report.json").exists()
    assert not (tmp_path / "failed/selected_config.json").exists()


def test_selection_detects_manifest_edits_during_comparison(tmp_path, monkeypatch):
    import branch_c.selection as selection
    manifest, config, _ = make_dataset(tmp_path)
    original = selection.RealOnlyAnomalyModel.fit

    def edit_manifest(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        with manifest.open("a") as f:
            f.write("\n")
        return result

    monkeypatch.setattr(selection.RealOnlyAnomalyModel, "fit", edit_manifest)
    with pytest.raises(ValueError, match="Manifest changed"):
        select_model(manifest, config, tmp_path / "changed", tmp_path / "cache")
    assert not (tmp_path / "changed/selection_report.json").exists()
    assert not (tmp_path / "changed/selected/model.joblib").exists()


@pytest.mark.parametrize("value,system,expected", [(1024, "Darwin", 1024), (1024, "Linux", 1048576)])
def test_native_peak_rss_units(value, system, expected):
    assert _rss_bytes(value, system) == expected


def test_peak_rss_rejects_unsupported_platform_and_unavailable_measurement():
    with pytest.raises(ValueError, match="macOS and Linux"):
        _rss_bytes(100, "Windows")
    with pytest.raises(ValueError, match="positive peak RSS"):
        _rss_bytes(0, "Linux")


@pytest.fixture
def fitted_toy(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    return tmp_path / "fit/model.joblib", Path(records[-1]["path"])


def test_benchmark_runs_in_fresh_processes_and_reports_full_scope(tmp_path, fitted_toy):
    model, video = fitted_toy
    report = benchmark_video(model, video, tmp_path / "performance.json", repeats=2)
    assert report["run_kind"] == "synthetic_smoke"
    assert report["is_detection_benchmark_result"] is False
    assert report["model_size_bytes"] == model.stat().st_size
    assert report["input_format"] == ".npz"
    assert len({r["pid"] for r in report["runs"]} | {os.getpid()}) == 3
    expected = predict_video(model, video)
    for run in report["runs"]:
        assert run["peak_rss_bytes"] > 0
        assert run["anomaly_score"] == expected["anomaly_score"]
        assert run["inference_seconds"] > 0
        assert run["input_metadata"]["input_height"] == 32
        assert run["input_metadata"]["input_width"] == 32
        timings = run["timings"]
        stage_sum = sum(timings[k] for k in ("model_load_seconds", "decode_and_sample_seconds",
                                            "feature_extraction_seconds", "scoring_and_explanation_seconds"))
        assert timings["total_seconds_including_model_load"] == pytest.approx(stage_sum)
        assert timings["total_seconds_including_model_load"] > run["inference_seconds"]
        assert run["process_wall_seconds"] > timings["total_seconds_including_model_load"]
    assert json.loads((tmp_path / "performance.json").read_text()) == report


@pytest.mark.parametrize("repeats,timeout", [(0, 120), (31, 120), (True, 120), (1, float("nan")), (1, 0)])
def test_invalid_benchmark_budget_fails_before_opening_files(tmp_path, repeats, timeout):
    with pytest.raises(ValueError):
        benchmark_video("missing_model", "missing_video", tmp_path / "out.json", repeats, timeout)
    assert not (tmp_path / "out.json").exists()


def test_existing_benchmark_report_cannot_be_overwritten(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("keep")
    with pytest.raises(ValueError, match="output already exists"):
        benchmark_video("missing_model", "missing_video", path)
    assert path.read_text() == "keep"


def test_benchmark_timeout_does_not_emit_partial_report(tmp_path, fitted_toy, monkeypatch):
    model, video = fitted_toy

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        benchmark_video(model, video, tmp_path / "partial.json", 1, 1)
    assert not (tmp_path / "partial.json").exists()


def test_benchmark_cli_stdout_is_one_json_object(tmp_path, fitted_toy, capsys):
    capsys.readouterr()  # Clear fixture output.
    model, video = fitted_toy
    assert main(["benchmark", "--model", str(model), "--video", str(video), "--repeats", "1",
                 "--output", str(tmp_path / "cli_performance.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["repeats"] == 1 and report["run_kind"] == "synthetic_smoke"
