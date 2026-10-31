"""Versioned 16,400-clip protocol, preserving the completed V1 implementation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,roc_auc_score,f1_score
from sklearn.exceptions import ConvergenceWarning
from . import pipeline as p

PHASES=('fusion_train','probability_calibration','threshold_selection','in_domain_test','external_test')
COUNTS=dict(zip(PHASES,(5120,640,640,8000,2000)))
FAMILY_COUNTS=dict(zip(PHASES[:4],(320,40,40,500)))
MODEL_IDS={'A':'12d7c3898482ada16b45f99832c9e95914388eec8c7f07bd5184ef554b86b2e1',
    'B':'66e13e1e787c12e88dd9b66d9d789d6654e117e3a9a59db77d25c83cba38a646',
    'C':'5649461f74586745570f74b2b0db77222780391c12c734feeb973ec743d6ddc6'}

def config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8-sig'));c['_config_path']=str(Path(path).resolve())
    for key in ['root','manifest','base_manifest','v1_root','predictions_dir','output_dir','neural_python']:
        c[key]=str(Path(c[key]).resolve())
    return c

def manifest(c):
    frame=p.table(c['manifest']);base=p.table(c['base_manifest'])
    p.require(frame.clip_id.is_unique and frame.sha256.is_unique,'Duplicate ID or bytes')
    p.require(set(frame.protocol_id)=={c['protocol_id']},'Protocol mismatch')
    p.require(frame.groupby('group_id').split.nunique().max()==1,'Source/prompt component crosses phases')
    p.require(frame.groupby('split').size().to_dict()==COUNTS,'Unexpected phase quotas')
    for phase,n in FAMILY_COUNTS.items():
        f=frame[(frame.split==phase)&(frame.label==1)]
        p.require(f.groupby('family').size().to_dict()==dict.fromkeys(p.KNOWN,n),'Known-family quota mismatch')
    p.require(frame[(frame.split=='external_test')&(frame.label==1)].family.eq('ModelScope').all(),'Main OOD must be ModelScope only')
    for phase in PHASES:
        part=frame[frame.split==phase];p.require(part.label.value_counts().to_dict()=={0:COUNTS[phase]//2,1:COUNTS[phase]//2},'Class balance mismatch')
    for key in ['clip_id','sha256']:p.require(not set(frame[key])&set(base[key]),'Expanded suite overlaps original32K '+key)
    old=p.table(Path(c['v1_root'])/'manifests/manifest.csv')
    historical=old[old.split.isin(['in_domain_test','external_test'])]
    for key in ['clip_id','sha256']:p.require(not set(frame[key])&set(historical[key]),'V1 historical test overlap '+key)
    verify=json.loads((Path(c['root'])/'DATA_VERIFICATION.json').read_text())
    p.require(verify['manifest_sha256']==p.digest(c['manifest']) and verify['all_media_SHA256_verified'],'Data verification lock differs')
    return frame.set_index('clip_id',drop=False)

def import_scores(c,branch,source):
    data=manifest(c);incoming=p.table(source);col=p.ALIASES[branch][0]
    if col not in incoming:col=next(k for k in p.ALIASES[branch] if k in incoming)
    p.require(incoming.clip_id.is_unique and set(incoming.clip_id)==set(data.index),'Incomplete/mixed score IDs')
    p.require(incoming.status.str.upper().isin(['OK','SUCCESS']).all(),'Abstentions cannot be imputed')
    values=pd.to_numeric(incoming[col],errors='coerce').to_numpy(dtype=float)
    p.require(np.isfinite(values).all(),'Nonfinite scores')
    if 'sha256' in incoming:p.require(np.array_equal(incoming.sha256.to_numpy(),data.loc[incoming.clip_id,'sha256'].to_numpy()),'Media hash mismatch')
    path=Path(c['predictions_dir'])/(branch+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    p.require(not path.exists(),'Refuse to overwrite imported scores')
    pd.DataFrame({'clip_id':incoming.clip_id,'score':values,'status':'OK'}).to_csv(path,index=False,encoding='utf-8-sig')
    p.save_json(path.with_suffix('.meta.json'),{'branch':branch,'model_id':MODEL_IDS[branch],'run_kind':'research',
        'protocol_id':c['protocol_id'],'manifest_sha256':p.digest(c['manifest']),'source_file_sha256':p.digest(source),
        'csv_sha256':p.digest(path),'base_manifest_sha256':p.digest(c['base_manifest']),
        'direction':'higher_is_fake','source_score_column':col,'frozen_inference_only':True})
    return {'branch':branch,'rows':len(incoming)}

def evidence(c,phases):
    frame=manifest(c);wanted=frame[frame.split.isin(phases)].copy();lineage={}
    for branch in p.BRANCHES:
        path=Path(c['predictions_dir'])/(branch+'.csv');meta=json.loads(path.with_suffix('.meta.json').read_text())
        p.require(meta['model_id']==MODEL_IDS[branch] and meta['protocol_id']==c['protocol_id'],'Branch model/protocol differs')
        p.require(meta['manifest_sha256']==p.digest(c['manifest']) and meta['csv_sha256']==p.digest(path),'Scores/dataset changed after import')
        table=p.table(path).set_index('clip_id');p.require(table.index.is_unique,'Duplicate scores')
        wanted[branch]=table.loc[wanted.index,'score'];p.require(np.isfinite(wanted[branch]).all(),'Nonfinite scores')
        lineage[branch]=meta
    return wanted,lineage

def family_folds(training):
    real=training[training.label==0]
    splitter=StratifiedGroupKFold(8,shuffle=True,random_state=42)
    real_parts=list(splitter.split(np.zeros((len(real),1)),real.family,groups=real.group_id))
    folds=[]
    for i,family in enumerate(p.KNOWN):
        validation=pd.concat([training[(training.label==1)&(training.family==family)],real.iloc[real_parts[i][1]]])
        blocked=set(validation.group_id)
        fit=training[(training.family!=family)&~training.group_id.isin(blocked)]
        p.require(not set(fit.group_id)&blocked,'Family CV group leak')
        p.require(family not in set(fit.loc[fit.label==1,'family']),'Held generator entered meta fit')
        p.require(set(fit.label)==set(validation.label)=={0,1},'Both labels required')
        folds.append({'fold':i,'held_family':family,'fit_ids':fit.clip_id.tolist(),'validation_ids':validation.clip_id.tolist(),
                      'fit_real':int((fit.label==0).sum()),'fit_generated':int((fit.label==1).sum()),
                      'validation_real':int((validation.label==0).sum()),'validation_generated':int((validation.label==1).sum())})
    p.require(Counter(cid for fold in folds for cid in fold['validation_ids'])==Counter(training.clip_id),'Family CV must validate every training video exactly once')
    return folds

def choose_threshold(y,scores):
    unique=np.unique(scores)
    candidates=np.unique(np.r_[0.,.5,1.,unique,(unique[:-1]+unique[1:])/2])
    return float(max(candidates,key=lambda cut:(f1_score(y,scores>=cut,average='macro',zero_division=0),-abs(cut-.5),-cut)))

def train(c):
    data,lineage=evidence(c,set(PHASES[:3]));dest=Path(c['output_dir'])
    p.require(not dest.exists(),'Preserve frozen model; choose new output directory')
    training=data[data.split=='fusion_train'];cal=data[data.split=='probability_calibration'];threshold=data[data.split=='threshold_selection']
    folds=family_folds(training);trials=[];oof_by_c={}
    for strength in [.001,.01,.1,1.,10.,100.]:
        results=[];oof=pd.Series(index=training.index,dtype=float)
        for fold in folds:
            fit,valid=training.loc[fold['fit_ids']],training.loc[fold['validation_ids']]
            model=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=2000,tol=1e-8,random_state=42))
            with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);model.fit(fit[list(p.BRANCHES)],fit.label)
            prob=model.predict_proba(valid[list(p.BRANCHES)])[:,1];oof.loc[valid.index]=prob
            results.append({'family':fold['held_family'],'log_loss':float(log_loss(valid.label,prob)),'auroc':float(roc_auc_score(valid.label,prob))})
        trials.append({'C':strength,'folds':results,'mean_log_loss':float(np.mean([r['log_loss'] for r in results]))});oof_by_c[strength]=oof
    best=min(trials,key=lambda x:(x['mean_log_loss'],x['C']))
    stacker=make_pipeline(StandardScaler(),LogisticRegression(C=best['C'],max_iter=2000,tol=1e-8,random_state=42))
    stacker.fit(training[list(p.BRANCHES)],training.label);kind='L2 LogisticRegression';selected_oof=oof_by_c[best['C']]
    with tempfile.TemporaryDirectory(prefix='vida-expanded-neural-') as directory:
        tmp=Path(directory);input_file=tmp/'training_only.csv';training[['clip_id','group_id','family','label',*p.BRANCHES]].to_csv(input_file,index=False)
        p.save_json(tmp/'folds.json',folds)
        subprocess.run([c['neural_python'],'-m','stacking.neural_expanded','--input',str(input_file),'--folds',str(tmp/'folds.json'),'--output',str(tmp/'output')],cwd=Path(__file__).resolve().parents[1],check=True)
        neural=json.loads((tmp/'output/report.json').read_text());candidate=joblib.load(tmp/'output/candidate.joblib')
        if neural['mean_log_loss']<best['mean_log_loss']:
            stacker=candidate;kind='MLP 3-8-1';selected_oof=pd.Series(np.load(tmp/'output/oof.npy'),index=training.index)
    calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,tol=1e-8,random_state=42))
    calibrator.fit(stacker.decision_function(cal[list(p.BRANCHES)]).reshape(-1,1),cal.label)
    bundle={'schema_version':2,'run_kind':'research','protocol_id':c['protocol_id'],'manifest_sha256':p.digest(c['manifest']),
        'feature_names':list(p.BRANCHES),'lineage':lineage,'stacker':stacker,'calibrator':calibrator,'selected_model':kind,
        'training_ids':training.clip_id.tolist(),'probability_fit_ids':cal.clip_id.tolist(),'threshold_fit_ids':threshold.clip_id.tolist(),
        'created_utc':datetime.now(timezone.utc).isoformat()}
    values=p.probability(bundle,threshold);bundle['threshold']=choose_threshold(threshold.label,values)
    bundle['single_branch_baselines']={}
    for branch in p.BRANCHES:
        platt=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,tol=1e-8,random_state=42));platt.fit(cal[[branch]],cal.label)
        scores=platt.predict_proba(threshold[[branch]])[:,1]
        bundle['single_branch_baselines'][branch]={'calibrator':platt,'threshold':choose_threshold(threshold.label,scores)}
    dest.mkdir(parents=True);joblib.dump(bundle,dest/'stacker.joblib')
    (dest/'development_candidates').mkdir();joblib.dump(candidate,dest/'development_candidates/mlp_training_only.joblib')
    p.save_json(dest/'MODEL_LOCK.json',{'model_sha256':p.digest(dest/'stacker.joblib'),'manifest_sha256':p.digest(c['manifest']),
        'protocol_id':c['protocol_id'],'branch_lineage':lineage,'threshold':bundle['threshold']})
    p.save_json(dest/'family_cv_folds.json',folds)
    training[['clip_id','label','family','group_id']].assign(selected_model_oof_probability=selected_oof).to_csv(dest/'fusion_train_cv.csv',index=False,encoding='utf-8-sig')
    summary={'run_kind':'research','selected_model':kind,'fusion_training_rows':len(training),'probability_fit_rows':len(cal),'threshold_selection_rows':len(threshold),
        'threshold':bundle['threshold'],'threshold_rule':'max macro F1; ties closest to 0.5 then lower threshold; candidate intervals represented by midpoints',
        'calibration':'StandardScaler + Platt logistic C=1, fitted on probability_calibration only',
        'LR_trials':trials,'best_LR_C':best['C'],'best_LR_log_loss':best['mean_log_loss'],'neural_candidate':neural,
        'CV':'8 held-generator folds at the meta level, source/prompt-group purging, real-source-stratified held real groups',
        'selected_epoch':neural['final_epochs'] if kind=='MLP 3-8-1' else None,
        'selected_solver_iterations':int(stacker[-1].n_iter_.max()) if kind=='L2 LogisticRegression' else None,
        'ABC_retrained':False,'final_test_used_for_any_selection':False,
        'threshold_selection_metrics_not_test':p.metrics(threshold.label,values,bundle['threshold'])}
    p.save_json(dest/'training_report.json',summary)
    return {'selected_model':kind,'epoch':summary['selected_epoch'],'threshold':bundle['threshold'],'best_LR_CV_logloss':best['mean_log_loss'],'MLP_CV_logloss':neural['mean_log_loss']}

def load(c):
    root=Path(c['output_dir']);lock=json.loads((root/'MODEL_LOCK.json').read_text())
    p.require(lock['model_sha256']==p.digest(root/'stacker.joblib'),'Frozen model changed')
    return joblib.load(root/'stacker.joblib')

def evaluate(c,bootstrap=1000):
    data,lineage=evidence(c,{'in_domain_test','external_test'});bundle=load(c)
    p.require(bundle['lineage']==lineage and bundle['manifest_sha256']==p.digest(c['manifest']),'Lineage/dataset changed after freeze')
    data['score']=p.probability(bundle,data)
    views={'in_domain':data[data.split=='in_domain_test'],'ModelScope':data[data.split=='external_test']}
    domain=views['in_domain'];real=domain[domain.label==0]
    # One common real control subset, 3:1 Vript/HDVG, used for all eight families.
    controls=pd.concat([real[real.family=='Vript'].sample(n=375,random_state=42),real[real.family=='HD-VG-130M'].sample(n=125,random_state=42)])
    for family in p.KNOWN:views['ID_'+family]=pd.concat([domain[(domain.label==1)&(domain.family==family)],controls])
    report={}
    for name,part in views.items():
        result=p.metrics(part.label,part.score,bundle['threshold']);result['auroc_group_bootstrap_95ci']=p.group_auc_ci(part,bootstrap)
        result['individual_branch_raw_score_auroc']={b:float(roc_auc_score(part.label,part[b])) for b in p.BRANCHES}
        result['individual_branch_calibrated_metrics']={b:p.metrics(part.label,baseline['calibrator'].predict_proba(part[[b]])[:,1],baseline['threshold']) for b,baseline in bundle['single_branch_baselines'].items()}
        report[name]=result
    root=Path(c['output_dir'])/'evaluation';p.require(not root.exists(),'Do not overwrite sealed results');root.mkdir()
    data[['clip_id','label','family','split','group_id','A','B','C','score']].to_csv(root/'predictions.csv',index=False,encoding='utf-8-sig')
    summary={'run_kind':'research','protocol_id':c['protocol_id'],'frozen_threshold':bundle['threshold'],'views':report,
        'ID_generator_macro_auroc':float(np.mean([report['ID_'+f]['auroc'] for f in p.KNOWN])),
        'test_used_for_training_or_selection':False,'common_ID_real_controls_shared_across_family_views':True}
    p.save_json(root/'metrics.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor');sub.add_parser('train');sub.add_parser('evaluate')
    imp=sub.add_parser('import-scores');imp.add_argument('--branch',choices=p.BRANCHES,required=True);imp.add_argument('--input',required=True)
    args=parser.parse_args();c=config(args.config)
    if args.command=='doctor':result={'clips':len(manifest(c)),'ready':all((Path(c['predictions_dir'])/(b+'.meta.json')).exists() for b in p.BRANCHES)}
    elif args.command=='import-scores':result=import_scores(c,args.branch,args.input)
    elif args.command=='train':result=train(c)
    else:
        report=evaluate(c);result={n:{k:v[k] for k in ['n','auroc','macro_f1','recall_generated','real_fpr']} for n,v in report['views'].items()}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
