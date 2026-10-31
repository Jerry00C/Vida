from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stacking import expanded as e
from stacking import pipeline as p

def test_threshold_tie_uses_middle_operating_region():
    # Perfect validation separation must not force an unnecessarily extreme cutoff.
    assert e.choose_threshold(np.array([0,0,1,1]),np.array([.001,.01,.998,.999]))==.5

def test_meta_generator_holdouts_purge_shared_prompt_groups():
    rows=[]
    for family in p.KNOWN:
        for i in range(8):rows.append({'clip_id':family+str(i),'family':family,'label':1,
            'group_id':('paired'+str(i)) if family in ['Pika','VideoCrafter2','T2V-Zero'] else family+str(i)})
    for family,n in [('Vript',48),('HD-VG-130M',16)]:
        for i in range(n):rows.append({'clip_id':family+str(i),'family':family,'label':0,'group_id':family+str(i)})
    frame=pd.DataFrame(rows).set_index('clip_id',drop=False);folds=e.family_folds(frame)
    assert len(folds)==8
    assert Counter(c for f in folds for c in f['validation_ids'])==Counter(frame.clip_id)
    for fold in folds:
        tr=frame.loc[fold['fit_ids']];va=frame.loc[fold['validation_ids']]
        assert not set(tr.group_id)&set(va.group_id)
        assert fold['held_family'] not in set(tr.family)
        assert set(va.loc[va.label==1,'family'])=={fold['held_family']}

@pytest.fixture
def expanded_manifest(tmp_path):
    rows=[];counter=0
    for phase,total in e.COUNTS.items():
        fake=dict.fromkeys(p.KNOWN,e.FAMILY_COUNTS[phase]) if phase!='external_test' else {'ModelScope':1000}
        for family,n in {**fake,'Vript':total//2}.items():
            for _ in range(n):
                cid='expanded_'+str(counter);counter+=1
                rows.append({'clip_id':cid,'label':0 if family=='Vript' else 1,'family':family,'split':phase,
                    'group_id':'group_'+cid,'sha256':hashlib.sha256(cid.encode()).hexdigest(),'protocol_id':'test-expanded'})
    pd.DataFrame(rows).to_csv(tmp_path/'manifest.csv',index=False)
    pd.DataFrame([{'clip_id':'old_base','sha256':'base_hash'}]).to_csv(tmp_path/'base.csv',index=False)
    (tmp_path/'v1/manifests').mkdir(parents=True)
    pd.DataFrame([{'clip_id':'historical','sha256':'historical_hash','split':'in_domain_test'}]).to_csv(tmp_path/'v1/manifests/manifest.csv',index=False)
    p.save_json(tmp_path/'DATA_VERIFICATION.json',{'manifest_sha256':p.digest(tmp_path/'manifest.csv'),'all_media_SHA256_verified':True})
    return {'root':str(tmp_path),'manifest':str(tmp_path/'manifest.csv'),'base_manifest':str(tmp_path/'base.csv'),
            'v1_root':str(tmp_path/'v1'),'protocol_id':'test-expanded'}

def test_expanded_manifest_contract(expanded_manifest):
    assert len(e.manifest(expanded_manifest))==16400

def test_expanded_rejects_reused_historical_test(expanded_manifest):
    path=Path(expanded_manifest['manifest']);d=p.table(path);d.loc[0,'clip_id']='historical';d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='historical test overlap'):e.manifest(expanded_manifest)

def test_expanded_rejects_cross_calibration_role_group(expanded_manifest):
    path=Path(expanded_manifest['manifest']);d=p.table(path)
    a=d.index[d.split=='probability_calibration'][0];b=d.index[d.split=='threshold_selection'][0]
    d.loc[b,'group_id']=d.loc[a,'group_id'];d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='crosses phases'):e.manifest(expanded_manifest)

def test_expanded_rejects_changed_manifest_lock(expanded_manifest):
    path=Path(expanded_manifest['manifest']);d=p.table(path);d.loc[0,'group_id']='changed_but_still_unique';d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='verification lock'):e.manifest(expanded_manifest)
