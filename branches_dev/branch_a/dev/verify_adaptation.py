"""Read-only checks of all six real-data adaptation smoke experiments."""
import argparse
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
from .common import check,read_json,read_csv,sha,save_json
from .splits import load_split,validate_round
from .data import load_indexes
from .learning import summarize
from .adapt import verify_resume
from .image_learning import load_adapted_checkpoint


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def verify(output,root):
    output=Path(output);snapshot=read_json(output/'baseline_snapshot.json')
    for name,digest in snapshot.items():check(sha(name)==digest,f'Protected baseline changed: {name}')
    split=load_split('runs/branch_a/smoke_splits.json');videos,_=load_indexes(root)
    rid=next(iter(split['rounds']));rows=split['rounds'][rid]
    validate_round(rows,split['pairs'][0],videos)
    results=[]
    for mode in ('partial','lora'):
        for backbone in ('clip','dinov2','dinov3'):
            folder=output/'smoke'/mode/backbone;run=folder/'resumed'
            check(read_json(folder/'smoke_verification.json')['passed'],'Incomplete real smoke run')
            resume=verify_resume(folder/'uninterrupted',run)
            a=load_adapted_checkpoint(folder/'uninterrupted/latest.pt');b=load_adapted_checkpoint(run/'latest.pt')
            for key in ('optimizer','scheduler','scaler','rng','sampling'):
                check(equal_state(a[key],b[key]),f'Resume {key} differs: {mode}/{backbone}')
            check(a['epoch']==2 and all(h['optimizer_step_samples']==[8,8,8,8,1] for h in a['history']),'Incomplete accumulation window not exercised')
            del a,b
            for target,epochs in ((folder/'uninterrupted',(1,)),(run,(1,2))):
                for epoch in epochs:
                    audit=read_json(target/f'parameter_audit_epoch_{epoch}.json')
                    check(audit['frozen_unchanged'] and audit['changed_parameters'] and audit['nonzero_gradient_parameters'],'Invalid gradient/weight audit')
                check(read_json(target/'reload_verification.json')['passed'],'Reload check missing')
            for role in ('test_seen','test_unseen'):
                predictions=read_csv(run/'evaluation'/f'{role}_predictions.csv')
                expected={r['clip_id']:r for r in rows if r['role']==role}
                check(len(predictions)==len(expected) and {r['clip_id'] for r in predictions}==set(expected),'Test video coverage differs')
                for r in predictions:
                    r['label']=int(r['label']);r['frame_logits']=json.loads(r['frame_logits'])
                    r['video_logit']=float(r['video_logit']);r['video_score']=float(r['video_score'])
                    check(r['label']==expected[r['clip_id']]['label'] and r['source']==expected[r['clip_id']]['source'],'Label/source mismatch')
                    check(len(r['frame_logits'])==8 and np.isfinite(r['frame_logits']).all(),'Invalid frame logits')
                    check(abs(np.mean(r['frame_logits'])-r['video_logit'])<1e-6,'Invalid aggregation')
                    check(abs(np.exp(-np.logaddexp(0.,-r['video_logit']))-r['video_score'])<1e-6,'Invalid sigmoid')
                actual=summarize(predictions);saved=read_json(run/'evaluation'/f'{role}_metrics.json')
                for key,value in actual.items():check(value==saved[key],f'Metric mismatch: {key}')
            results.append({'mode':mode,'backbone':backbone,'passed':True,'resume':resume,'cost':read_json(run/'cost.json')})
    report={'passed':True,'protected_files_unchanged':len(snapshot),'split_identity':split['split_identity'],
            'role_counts':dict(Counter(r['role'] for r in rows)),'distinct_videos':len({r['clip_id'] for r in rows}),
            'real_data_smoke_runs':results,'gpu_verified':False}
    save_json(output/'verification.json',report);print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/branch_a_adapt');p.add_argument('--root',default='processed')
    a=p.parse_args();verify(a.output,a.root)
