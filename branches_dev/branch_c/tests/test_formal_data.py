"""Actual handoff edge cases: native short clips, fixed roles and shared test views."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from branch_c.config import FeatureConfig
from branch_c.features import extract_features
from branch_c.manifest import load_manifest
from branch_c.pipeline import _prepare, evaluate, fit
from branch_c.selection import select_model
from branch_c.video import read_video, validate_frames
from test_iteration_guards import FakeCapture, make_dataset, write_manifest


@pytest.mark.parametrize("fps,count", [(4,8), (10,16), (12,13), (30,21), (25,155), (60,1200)])
def test_native_stride_uses_actual_distinct_frames_and_true_times(tmp_path, monkeypatch, fps, count):
    import branch_c.video as module
    cfg = FeatureConfig(image_size=32, min_duration_seconds=.5, sampling_mode="native_stride")
    frames = np.zeros((count,32,32,3),np.uint8)
    frames[:,0,0] = (np.arange(count)%256)[:,None]
    capture = FakeCapture(frames, float(fps))
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda *args: capture)
    path=tmp_path/"clip.mp4"
    path.touch()
    video=read_video(path,cfg)
    stride=video.metadata["native_stride"]
    indices=np.arange(count//stride)*stride
    np.testing.assert_allclose(video.timestamps_seconds,indices/fps)
    np.testing.assert_array_equal(video.frames_rgb[:,0,0,0],indices%256)
    assert len(video.frames_rgb)>=8 and capture.released
    result=extract_features(video,cfg)
    assert result.clip.shape==(37,) and np.isfinite(result.clip).all()
    assert result.bounds_seconds[-1,1]<=count/fps+1e-6
    if count/fps<2:
        assert len(result.windows)==1
        np.testing.assert_allclose(result.clip,result.windows[0])


def test_native_timebase_controls_flow_speed_and_rejects_nonuniform(monkeypatch):
    import branch_c.features as module
    cfg=FeatureConfig(image_size=32,min_duration_seconds=.5,sampling_mode="native_stride")
    frames=np.zeros((8,32,32,3),np.uint8)
    flow=np.zeros((32,32,2),np.float32)
    flow[:,:,0]=2
    monkeypatch.setattr(module.cv2,"calcOpticalFlowFarneback",lambda *a:flow)
    for fps in (4,8):
        feature=extract_features(validate_frames(frames,np.arange(8)/fps,cfg),cfg)
        assert feature.clip[15]==pytest.approx(2*fps/32)
    times=np.arange(8)/4
    times[3]+=.03
    with pytest.raises(ValueError,match="uniformly"):
        validate_frames(frames,times,cfg)


def test_parallel_preparation_matches_serial_without_reordering(tmp_path):
    manifest,cfg,_=make_dataset(tmp_path)
    serial=_prepare(manifest,cfg,tmp_path/"serial",{"train"})
    parallel=_prepare(manifest,cfg,tmp_path/"parallel",{"train"},workers=3)
    assert [r.clip_id for r in serial[0]]==[r.clip_id for r in parallel[0]]
    assert serial[2]==parallel[2]
    for a,b in zip(serial[1],parallel[1]):
        np.testing.assert_array_equal(a.clip,b.clip)
        np.testing.assert_array_equal(a.windows,b.windows)


def role_dataset(tmp_path):
    manifest,cfg,records=make_dataset(tmp_path)
    val=[r for r in records if r["split"]=="val"]
    for row in records:
        row["validation_role"]=""
    for i,row in enumerate(val):
        row["validation_role"]="model_selection" if i<2 else "calibration"
    write_manifest(manifest,records)
    return manifest,cfg,records


def test_calibration_never_affects_candidate_ranking_and_threshold_uses_real_calibration(tmp_path):
    manifest,cfg,records=role_dataset(tmp_path)
    first=select_model(manifest,cfg,tmp_path/"one",tmp_path/"cache")
    selection_ids={r["clip_id"] for r in records if r["validation_role"]=="model_selection"}
    for candidate in first["ranked_candidates"]:
        with (tmp_path/"one/candidates"/f"{candidate['candidate_id']}_validation_scores.csv").open() as f:
            assert {r["clip_id"] for r in csv.DictReader(f)}==selection_ids
    with (tmp_path/"one/selected/validation_scores.csv").open() as f:
        scores=list(csv.DictReader(f))
    real_calibration=[float(r["branch_c_score"]) for r in scores if r["validation_role"]=="calibration" and r["label"]=="0"]
    report=json.loads((tmp_path/"one/selected/fit_report.json").read_text())
    assert report["threshold"]==np.quantile(real_calibration,.95,method="higher")
    assert set(report["threshold_clip_ids"])=={r["clip_id"] for r in scores if r["validation_role"]=="calibration" and r["label"]=="0"}
    for r in records:
        if r["validation_role"]=="calibration":
            np.savez(r["path"],frames_rgb=np.zeros((16,32,32,3),np.uint8),timestamps_seconds=np.arange(16)/8)
    # Identical calibration bytes are within the same partition, and may be scored.
    second=select_model(manifest,cfg,tmp_path/"two",tmp_path/"cache")
    assert first["ranked_candidates"]==second["ranked_candidates"]


@pytest.mark.parametrize("bad",["mixed","group"])
def test_validation_role_leakage_rejected_before_decode(tmp_path,bad):
    manifest,cfg,records=role_dataset(tmp_path)
    val=[r for r in records if r["split"]=="val"]
    if bad=="mixed": val[-1]["validation_role"]=""
    else: val[0]["group_id"]=val[-1]["group_id"]
    write_manifest(manifest,records)
    with pytest.raises(ValueError,match="role"):
        load_manifest(manifest,cfg)


def test_test_subset_only_reads_selected_view_with_same_frozen_threshold(tmp_path):
    manifest,cfg,records=make_dataset(tmp_path)
    for row in records: row["evaluation_subset"]=""
    test=[r for r in records if r["split"]=="test"]
    for i,r in enumerate(test): r["evaluation_subset"]="in_domain_test" if i<2 else "cross_test"
    write_manifest(manifest,records)
    report=fit(manifest,cfg,tmp_path/"fit",tmp_path/"cache")
    for r in test[2:]: Path(r["path"]).unlink()
    evaluation=evaluate(manifest,tmp_path/"fit/model.joblib",tmp_path/"evaluation",tmp_path/"cache",test_subset="in_domain_test",workers=2)
    assert evaluation["threshold"]==report["threshold"]
    assert evaluation["metrics"]["n"]==2
    assert evaluation["evaluation_subset"]=="in_domain_test"
    with pytest.raises(ValueError,match="No test clips"):
        evaluate(manifest,tmp_path/"fit/model.joblib",tmp_path/"invalid",tmp_path/"cache",test_subset="absent")
