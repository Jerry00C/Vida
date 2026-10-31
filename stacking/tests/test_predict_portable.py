"""Portable sealed models need neither training videos nor old machine paths."""
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stacking import pipeline as p

@pytest.mark.parametrize('version,mode',[(1,'--config'),(1,'--model-dir'),(2,'--model-dir')])
def test_predict_portable_sealed_model_ignores_labels_and_column_order(tmp_path,version,mode):
    frame=pd.DataFrame({'A':[-3.,-2.,-1.,1.,2.,3.],'B':[-2.,-1.,-.5,.5,1.,2.],'C':[0.,.1,.2,.8,.9,1.]})
    y=np.array([0,0,0,1,1,1]);stacker=make_pipeline(StandardScaler(),LogisticRegression()).fit(frame,y)
    calibrator=LogisticRegression().fit(stacker.decision_function(frame).reshape(-1,1),y)
    bundle={'schema_version':version,'stacker':stacker,'calibrator':calibrator,'threshold':.5}
    modeldir=tmp_path/'portable';modeldir.mkdir();joblib.dump(bundle,modeldir/'stacker.joblib')
    p.save_json(modeldir/'MODEL_LOCK.json',{'model_sha256':p.digest(modeldir/'stacker.joblib')})
    config={'run_kind':'research','output_dir':'portable','manifest':'does_not_exist.csv',
        'base_manifest':'also_absent.csv','predictions_dir':'absent_scores'}
    p.save_json(tmp_path/'config.json',config)
    incoming=frame.assign(clip_id=['new_'+str(i) for i in range(len(frame))],label=1-y,family='uninformative metadata')
    incoming[['clip_id','C','B','A','label','family']].to_csv(tmp_path/'input.csv',index=False)
    source=modeldir if mode=='--model-dir' else tmp_path/'config.json'
    result=subprocess.run([sys.executable,'-m','stacking.predict',mode,str(source),'--input',str(tmp_path/'input.csv'),
        '--output',str(tmp_path/'predictions.csv')],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    actual=p.table(tmp_path/'predictions.csv');expected=p.probability(bundle,frame)
    np.testing.assert_allclose(actual.generated_probability,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(actual.prediction_generated,expected>=.5)
    assert actual.clip_id.tolist()==incoming.clip_id.tolist()
