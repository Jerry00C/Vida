"""Reporting references must be declared before test and preserved on resume."""
import json
import pytest
from scripts.run_vida_guard32k import ensure_auxiliary_plans


def test_reporting_references_created_before_test_and_unchanged_on_resume(tmp_path):
    ensure_auxiliary_plans(tmp_path)
    files=list(tmp_path.glob("*_plan.json"))
    assert len(files)==3
    previous={p.name:p.read_bytes() for p in files}
    random=json.loads((tmp_path/"random_reference_plan.json").read_text())
    assert random["seed"]==42 and random["labels_used_to_generate_scores"] is False
    (tmp_path/"model_freeze.json").touch()
    ensure_auxiliary_plans(tmp_path)
    assert {p.name:p.read_bytes() for p in files}==previous


def test_missing_reference_cannot_be_retro_declared_after_sealed_test(tmp_path):
    (tmp_path/"cross_test").mkdir()
    (tmp_path/"cross_test/test_report.json").touch()
    with pytest.raises(ValueError,match="do not retro-declare"):
        ensure_auxiliary_plans(tmp_path)
    assert not list(tmp_path.glob("*_plan.json"))
