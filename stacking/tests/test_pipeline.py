import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stacking import pipeline as p
from stacking.demo import fixture

@pytest.fixture
def ready(tmp_path):return fixture(tmp_path/"fixture")

def test_full_round_trip(ready):
    r=p.train(ready)
    assert r["branch_models_retrained"] is False
    assert r["not_research_result"] is True
    model=p.load_bundle(ready)
    assert not set(model["probability_fit_ids"])&set(model["threshold_fit_ids"])
    result=p.evaluate(ready,bootstrap=0)
    assert set(result["views"])=={"in_domain","cross_generator_known_source","cross_generator_new_source",*(f+"_"+s for f in p.EXTERNAL for s in ["known_source","new_source"])}
    assert result["test_set_used_for_fitting_or_threshold_selection"] is False
    assert result["not_research_result"] is True
    assert set(result["views"]["in_domain"]["individual_branch_raw_score_auroc"])==set(p.BRANCHES)
    with pytest.raises(ValueError,match="already exists"):p.train(ready)
    with pytest.raises(ValueError,match="already exists"):p.evaluate(ready,bootstrap=0)

@pytest.mark.parametrize("key",["clip_id","group_id","sha256"])
def test_reject_original_training_overlap(ready,key):
    d=p.table(ready["manifest"]);b=p.table(ready["base_manifest"])
    d.loc[0,key]=b.loc[0,key];d.to_csv(ready["manifest"],index=False)
    with pytest.raises(ValueError,match="overlaps original"):p.manifest(ready)

def test_reject_cross_split_groups(ready):
    d=p.table(ready["manifest"])
    i=d.index[d.split=="calibration"][0];d.loc[i,"group_id"]=d.loc[0,"group_id"]
    d.to_csv(ready["manifest"],index=False)
    with pytest.raises(ValueError,match="crosses holdout"):p.manifest(ready)

def test_reject_missing_branch(ready):
    (Path(ready["predictions_dir"])/"C.csv").unlink()
    with pytest.raises(ValueError,match="Missing branch.*C"):p.train(ready)

def test_reject_in_sample_declaration(ready):
    path=Path(ready["predictions_dir"])/"B.meta.json";m=json.loads(path.read_text());m["score_kind"]="in_sample";p.save_json(path,m)
    with pytest.raises(ValueError,match="in-sample"):p.train(ready)

def test_reject_changed_scores(ready):
    path=Path(ready["predictions_dir"])/"A.csv";d=p.table(path);d.loc[0,"score"]+=1;d.to_csv(path,index=False)
    with pytest.raises(ValueError,match="changed after import"):p.train(ready)

def test_reject_synthetic_as_research(ready):
    ready["run_kind"]="research"
    with pytest.raises(ValueError,match="Synthetic/research"):p.train(ready)

def test_inference_version_must_match_training(ready):
    p.train(ready);path=Path(ready["predictions_dir"])/"C.meta.json";m=json.loads(path.read_text());m["model_id"]="different-checkpoint";p.save_json(path,m)
    with pytest.raises(ValueError,match="checkpoint/score mapping changed"):p.evaluate(ready,bootstrap=0)

def test_changed_dataset_after_freeze_rejected(ready):
    p.train(ready);d=p.table(ready["manifest"]);d=d.iloc[::-1];d.to_csv(ready["manifest"],index=False)
    with pytest.raises(ValueError,match="holdout manifest changed"):p.evaluate(ready,bootstrap=0)

def test_import_rejects_duplicate_ids_and_null_scores(ready,tmp_path):
    frame=p.table(Path(ready["_config_path"]).parent/"A_native.csv")
    with pytest.raises(ValueError,match="Duplicate predictions"):
        x=tmp_path/"duplicates.csv";pd.concat([frame,frame.iloc[:1]]).to_csv(x,index=False);p.import_scores(ready,"A",x,"new-model")
    frame.loc[0,"video_logit"]=np.nan;x=tmp_path/"null.csv";frame.to_csv(x,index=False)
    with pytest.raises(ValueError,match="nonfinite"):p.import_scores(ready,"A",x,"new-model")

def test_negative_direction_is_explicit(ready,tmp_path):
    root=Path(ready["predictions_dir"])
    (root/"A.csv").unlink();(root/"A.meta.json").unlink()
    frame=p.table(Path(ready["_config_path"]).parent/"A_native.csv")
    x=tmp_path/"real_direction.csv";frame.to_csv(x,index=False)
    p.import_scores(ready,"A",x,"real-score-model",direction="higher_is_real")
    actual=p.table(root/"A.csv")
    assert np.allclose(actual.score,-frame.video_logit)

def test_old_full32k_like_scores_cannot_be_imported(ready,tmp_path):
    x=tmp_path/"old.csv";pd.DataFrame({"clip_id":["old_1"],"raw_margin":[9.0]}).to_csv(x,index=False)
    with pytest.raises(ValueError,match="old full32k"):p.import_scores(ready,"B",x,"full32k-model")

def test_feature_schema_has_only_three_scores(ready):
    p.train(ready);b=p.load_bundle(ready)
    assert b["feature_names"]==["A","B","C"]
    assert list(b["stacker"].feature_names_in_)==["A","B","C"]


def test_hyperparameters_selected_on_grouped_training_only(ready):
    report=p.train(ready)
    training=p.table(ready['manifest']);training=training[training.split=='fusion_train']
    cv=p.table(Path(ready['output_dir'])/'fusion_train_cv.csv')
    assert set(cv.clip_id)==set(training.clip_id)
    assert set(cv.cv_fold)==set(range(5))
    assert cv.groupby('group_id').cv_fold.nunique().max()==1
    assert np.isfinite(cv.selected_model_oof_probability).all()
    selected=min(report['cross_validation'],key=lambda x:(x['mean_log_loss'],x['C']))
    assert report['hyperparameters']['selected_C']==selected['C']
    assert report['hyperparameters']['converged'] is True
    assert report['hyperparameters']['epoch'] is None
    bundle=p.load_bundle(ready)
    assert set(bundle['single_branch_baselines'])==set(p.BRANCHES)


def test_windows_utf8_bom_config_is_readable(ready):
    path=Path(ready["_config_path"])
    value=json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value),encoding="utf-8-sig")
    assert p.config(path)["protocol_id"]=="synthetic-demo-only"


def test_returned_media_hash_must_match(ready,tmp_path):
    frame=p.table(Path(ready["_config_path"]).parent/"A_native.csv")
    frame["sha256"]="wrong-bytes"
    path=tmp_path/"wrong_media.csv";frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="video hash disagrees"):
        p.import_scores(ready,"A",path,"frozen-A")


def test_mixed_checkpoint_versions_cannot_be_imported(ready,tmp_path):
    frame=p.table(Path(ready["_config_path"]).parent/"A_native.csv")
    frame["model_id"]="final-A";frame.loc[0,"model_id"]="another-round"
    path=tmp_path/"mixed_models.csv";frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="mixes multiple model versions"):
        p.import_scores(ready,"A",path,"final-A")
