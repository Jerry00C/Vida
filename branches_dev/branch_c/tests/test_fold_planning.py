"""Common folds are leakage-safe, independent of video decoding and reusable by C."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from branch_c.cache import cached_features, file_sha256
from branch_c.cli import main
from branch_c.config import Config
from branch_c.folds import build_fold_plan, prepare_folds
from branch_c.manifest import connected_groups, load_manifest
from branch_c.pipeline import fit, make_oof

from test_iteration_guards import make_dataset, write_manifest


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_fold_planning_uses_only_train_hashes_and_preserves_original_manifest(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    for i, record in enumerate(records):
        path = Path(record["path"])
        if record["split"] == "train":
            path.write_bytes(f"unique non-video content {i}".encode())
        else:
            path.unlink()
        record["path"] = path.name  # Relative paths must work after export elsewhere.
        record["semantic_tag"] = f"original_{i}"
    write_manifest(manifest, records)
    before = manifest.read_bytes()
    original = load_manifest(manifest, config)
    output = tmp_path / "nested/planned"
    report = prepare_folds(manifest, config, output, 3)
    assert manifest.read_bytes() == before
    assert report["models_fitted"] == report["video_files_decoded"] == report["val_test_files_read"] == 0
    assert report["contains_predictions"] is False and report["run_kind"] == "synthetic_smoke"
    assert report["n_clips"] == 12 and report["n_splits"] == 3
    assert all(f["n_real_fit"] >= 4 for f in report["fold_summary"])
    exported = load_manifest(output / "manifest_with_folds.csv", config)
    assert [(r.clip_id, r.path, r.label, r.split, r.source, r.group_id) for r in exported] == [
        (r.clip_id, r.path, r.label, r.split, r.source, r.group_id) for r in original]
    for old, new in zip(original, exported):
        assert old.metadata["semantic_tag"] == new.metadata["semantic_tag"]
        if new.split == "train":
            assert new.metadata["content_sha256"] == file_sha256(new.path)
        else:
            assert new.metadata["fold_id"] == new.metadata["content_sha256"] == ""
    assignments = read_csv(output / "fold_assignments.csv")
    assert len(assignments) == 12 and "branch_c_score" not in assignments[0]
    train = [r for r in exported if r.split == "train"]
    hashes = [file_sha256(r.path) for r in train]
    _, _, reused = build_fold_plan(train, hashes, 3, config.model.seed)
    _, _, reordered = build_fold_plan(train[::-1], hashes[::-1], 3, config.model.seed)
    assert reused["plan_id"] == reordered["plan_id"] == report["plan_id"]
    assert reused["assignment_source"] == "manifest_fold_id" and reused["split_seed"] is None


def test_planned_manifest_reproduces_existing_oof_scores_and_assignments(tmp_path):
    manifest, config, _ = make_dataset(tmp_path)
    make_oof(manifest, config, tmp_path / "direct", tmp_path / "cache", 3)
    plan = prepare_folds(manifest, config, tmp_path / "plan", 3)
    make_oof(tmp_path / "plan/manifest_with_folds.csv", config, tmp_path / "planned", tmp_path / "cache", 3)
    direct, planned = read_csv(tmp_path / "direct/oof_scores.csv"), read_csv(tmp_path / "planned/oof_scores.csv")
    assert [(r["clip_id"], r["fold_id"]) for r in direct] == [(r["clip_id"], r["fold_id"]) for r in planned]
    np.testing.assert_array_equal([float(r["branch_c_score"]) for r in direct],
                                  [float(r["branch_c_score"]) for r in planned])
    audit = json.loads((tmp_path / "planned/oof_audit.json").read_text())
    assert audit["fold_plan"]["plan_id"] == plan["plan_id"]
    for fold in audit["folds"]:
        assert fold["n_real_training"] >= 4
        assert not set(fold["fit_clip_ids"]) & set(fold["heldout_clip_ids"])


@pytest.mark.parametrize("invalid", ["too_few_real", "missing_real_val"])
def test_fit_rejects_bad_class_metadata_before_opening_media(tmp_path, invalid):
    manifest, config, records = make_dataset(tmp_path)
    if invalid == "too_few_real":
        real_ids = [r["clip_id"] for r in records if r["split"] == "train" and r["label"] == 0]
        records = [r for r in records if r["clip_id"] not in real_ids[3:]]
    else:
        records = [r for r in records if r["split"] != "val" or r["label"] == 1]
    write_manifest(manifest, records)
    for record in records:
        Path(record["path"]).unlink()
    expected = "At least 4 real" if invalid == "too_few_real" else "real validation clips"
    with pytest.raises(ValueError, match=expected):
        fit(manifest, config, tmp_path / "fit", tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("invalid", ["too_few_real", "partial", "non_integer", "missing_fold", "group_crossing", "real_fit_shortage"])
def test_bad_fold_metadata_fails_before_opening_media(tmp_path, invalid):
    manifest, config, records = make_dataset(tmp_path)
    train = [r for r in records if r["split"] == "train"]
    if invalid == "too_few_real":
        remove = {r["clip_id"] for r in train if r["label"] == 0 and int(r["clip_id"].split('_')[-1]) >= 8}
        records = [r for r in records if r["clip_id"] not in remove]
        message = "at least 5 real"
    else:
        for i, r in enumerate(train):
            r["fold_id"] = str(i // 4)
        if invalid == "partial": train[0]["fold_id"] = ""; message = "every training row"
        if invalid == "non_integer": train[0]["fold_id"] = "1.5"; message = "integers"
        if invalid == "missing_fold":
            for r in train: r["fold_id"] = "0" if r["fold_id"] == "2" else r["fold_id"]
            message = "cover 0"
        if invalid == "group_crossing": train[4]["group_id"] = train[0]["group_id"]; message = "crosses supplied"
        if invalid == "real_fit_shortage": train[4]["fold_id"] = "0"; message = "only 3 real"
    # Ensure the optional field exists on all rows for the CSV helper.
    for r in records: r.setdefault("fold_id", "")
    write_manifest(manifest, records)
    for record in records:
        Path(record["path"]).unlink()
    with pytest.raises(ValueError, match=message):
        make_oof(manifest, config, tmp_path / "oof", tmp_path / "cache", 3)
    assert not (tmp_path / "cache").exists()


def test_content_duplicates_cannot_cross_supplied_folds_and_fail_before_features(tmp_path):
    manifest, config, records = make_dataset(tmp_path, train_count=20)
    train = [r for r in records if r["split"] == "train"]
    for i, r in enumerate(train): r["fold_id"] = str((i // 2) % 3)
    for r in records: r.setdefault("fold_id", "")
    Path(train[2]["path"]).write_bytes(Path(train[0]["path"]).read_bytes())
    write_manifest(manifest, records)
    with pytest.raises(ValueError, match="crosses supplied OOF folds"):
        make_oof(manifest, config, tmp_path / "oof", tmp_path / "cache", 3)
    assert not (tmp_path / "cache").exists()
    for r in train: r["fold_id"] = ""
    write_manifest(manifest, records)
    plan = prepare_folds(manifest, config, tmp_path / "plan", 3)
    assignments = {r["clip_id"]: r for r in read_csv(tmp_path / "plan/fold_assignments.csv")}
    assert assignments[train[0]["clip_id"]]["fold_id"] == assignments[train[2]["clip_id"]]["fold_id"]
    assert assignments[train[0]["clip_id"]]["component_id"] == assignments[train[2]["clip_id"]]["component_id"]
    assert plan["n_groups"] == 19


def test_auto_fold_with_all_real_clips_in_one_group_is_rejected_before_features(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    for record in records:
        if record["split"] == "train" and record["label"] == 0:
            record["group_id"] = "all_real_same_group"
    write_manifest(manifest, records)
    with pytest.raises(ValueError, match="only 0 real training"):
        make_oof(manifest, config, tmp_path / "oof", tmp_path / "cache", 3)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("warm", [False, True])
def test_frozen_content_mismatch_fails_before_decoding_even_with_cache(tmp_path, monkeypatch, warm):
    import branch_c.cache as cache
    manifest, config, records = make_dataset(tmp_path)
    path = Path(records[0]["path"])
    if warm:
        cached_features(path, config.features, tmp_path / "cache")
    monkeypatch.setattr(cache, "read_video", lambda *args: pytest.fail("A changed input must not be decoded"))
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        cached_features(path, config.features, tmp_path / "cache", expected_sha256="0" * 64)


def test_changed_training_file_in_exported_manifest_is_rejected_before_features(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    prepare_folds(manifest, config, tmp_path / "plan", 3)
    Path(records[0]["path"]).write_bytes(b"changed video input")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        make_oof(tmp_path / "plan/manifest_with_folds.csv", config, tmp_path / "oof", tmp_path / "cache", 3)
    assert not (tmp_path / "cache").exists()


def test_manifest_hash_schema_and_connected_group_alignment_are_validated(tmp_path):
    manifest, config, records = make_dataset(tmp_path)
    for r in records: r["content_sha256"] = ""
    records[0]["content_sha256"] = "not a digest"
    write_manifest(manifest, records)
    with pytest.raises(ValueError, match="lowercase SHA256"):
        load_manifest(manifest)
    records[0]["content_sha256"] = ""
    write_manifest(manifest, records)
    with pytest.raises(ValueError, match="One content hash"):
        connected_groups(load_manifest(manifest), ["a"])


def test_fold_cli_output_is_json_and_has_no_predictions(tmp_path, capsys):
    manifest, config, _ = make_dataset(tmp_path, train_count=50)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    assert main(["folds", "--manifest", str(manifest), "--config", str(config_path), "--folds", "3",
                 "--output-dir", str(tmp_path / "plan")]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert "Hashed 25/50" in output.err
    assert report["contains_predictions"] is False and report["val_test_files_read"] == 0
    assert {p.name for p in (tmp_path / "plan").iterdir()} == {"fold_assignments.csv", "manifest_with_folds.csv", "fold_plan.json"}


@pytest.mark.parametrize("command,file", [("fit", "fit_report.json"), ("fit", "validation_scores.csv"),
                                         ("oof", "fold_assignments.csv"), ("oof", "oof_audit.json")])
def test_existing_partial_training_outputs_are_protected_before_input_reads(tmp_path, command, file):
    output = tmp_path / "partial"
    output.mkdir()
    path = output / file
    path.write_text("preserve")
    call = fit if command == "fit" else make_oof
    with pytest.raises(ValueError, match="output already exists"):
        call("missing.csv", Config(), output, "unused")
    assert path.read_text() == "preserve"


def test_fold_output_and_synthetic_provenance_are_protected(tmp_path):
    manifest, config, _ = make_dataset(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="Fold output already exists"):
        prepare_folds("missing.csv", config, output, 3)
    with pytest.raises(ValueError, match="cannot be reported"):
        prepare_folds(manifest, config, tmp_path / "bad_kind", 3, "research")
    assert not (tmp_path / "bad_kind").exists()
