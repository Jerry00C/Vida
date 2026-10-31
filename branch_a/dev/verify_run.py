"""Read-only verification of the real experiment artifacts (no training)."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .common import check,index_identity,read_csv,read_json,save_json
from .data import load_indexes
from .splits import load_split,validate_round
from .learning import metrics,scores_from_logits,load_checkpoint
from .features import FeatureCache


def verify(root,output):
    videos,_=load_indexes(root)
    full=load_split(output/'splits_v1.json')
    check(full['index_fingerprints']==index_identity(root),'Original indexes modified')
    real_roles=defaultdict(set)
    for i,(rid,rows) in enumerate(full['rounds'].items()):
        validate_round(rows,full['pairs'][i],videos)
        for r in rows:
            if r['label']==0:real_roles[r['clip_id']].add('test' if r['role'].startswith('test_') else r['role'])
        seen={r['clip_id'] for r in rows if r['role']=='test_unseen' and r['label']==1}
        expected={c for c,v in videos.items() if v['family'] in full['pairs'][i]}
        check(seen==expected,'Heldout clips missing')
    check(all(len(s)==1 for s in real_roles.values()),'Real assignments changed across rounds')
    small=load_split(output/'smoke_splits.json')
    report=read_json(output/'smoke_report.json');results={}
    for name,r in report.items():
        if r['status']!='passed':results[name]={'status':'blocked','diagnostic':r.get('model_access_diagnostic',r.get('error'))};continue
        cache=FeatureCache(r['cache'])
        check(set(cache.clips)=={x['clip_id'] for x in small['rounds']['round_1']},'Smoke cache coverage mismatch')
        run=output/'smoke'/name/'resumed'
        cp=load_checkpoint(run/'best.pt',cache,small['split_identity'],'round_1')
        check(cp['epoch'] in (1,2),'Unexpected smoke checkpoint epoch')
        counts={}
        for role in ('test_seen','test_unseen'):
            predictions=read_csv(run/'evaluation'/f'{role}_predictions.csv')
            expected={x['clip_id']:x for x in small['rounds']['round_1'] if x['role']==role}
            check(len(predictions)==len(expected) and {x['clip_id'] for x in predictions}==set(expected),'Prediction IDs mismatch')
            y=[];scores=[]
            for pred in predictions:
                row=expected[pred['clip_id']];check(int(pred['label'])==row['label'] and pred['source']==row['source'],'Prediction label/source mismatch')
                logits=np.array(json.loads(pred['frame_logits']))
                check(logits.shape==(8,) and np.isfinite(logits).all(),'Frame logits invalid')
                mean=float(logits.mean());score=float(scores_from_logits(np.array([mean]))[0])
                check(abs(mean-float(pred['video_logit']))<1e-6 and abs(score-float(pred['video_score']))<1e-6,'Aggregation mismatch')
                y.append(int(pred['label']));scores.append(float(pred['video_score']))
            saved=read_json(run/'evaluation'/f'{role}_metrics.json');computed=metrics(y,scores)
            check(all(saved[k]==v for k,v in computed.items()),'Metrics disagree with saved predictions')
            counts[role]=len(predictions)
        results[name]={'status':'passed','prediction_counts':counts,'dimension':cache.values.shape[-1],
                       'cache_bytes':cache.meta['stats']['cache_bytes']}
    result={'original_index_fingerprints_unchanged':True,'all_four_rounds_leakage_checks':'passed',
            'real_assignments_reused':True,'smoke_artifacts':results,'full_training_launched':False}
    save_json(output/'verification_report.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processed',type=Path,default=Path('processed'));p.add_argument('--output',type=Path,default=Path('runs/branch_a'))
    a=p.parse_args();verify(a.processed,a.output)
