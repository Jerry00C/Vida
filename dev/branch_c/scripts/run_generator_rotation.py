"""Execute all 28 frozen six-seen/two-unseen Branch C rounds using cached descriptors.

Identical real-only fits are shared with explicit input/parameter keys. Each round
still has its own selection, manifests, OOF coverage, quarantine checks and tests.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import joblib
import numpy as np

from branch_c.benchmark import benchmark_video
from branch_c.cache import CACHE_VERSION, cached_features, file_sha256
from branch_c.config import Config, FeatureConfig, fingerprint, load_config, write_json
from branch_c.evaluation import choose_real_threshold, metrics
from branch_c.features import feature_signature
from branch_c.manifest import connected_groups, load_manifest
from branch_c.model import RealOnlyAnomalyModel, load_bundle, runtime_versions, save_bundle
from branch_c.pipeline import _new_bundle, _write_scores, _check_evaluation_leakage
from branch_c.review import write_error_review
from branch_c.selection import predeclared_candidates
from generator_rotation import PROTOCOL, GENERATORS, check_round, fast_group_bootstrap, now, read_csv, write_csv


def run(args):
    root=Path(args.output_dir);splits=Path(args.splits_dir);cache=Path(args.cache_dir)
    if root.exists():raise ValueError("Run directory exists; preserve completed or partial artifacts")
    protocol=json.loads((splits/'protocol_plan.json').read_text())
    if fingerprint({k:v for k,v in protocol.items() if k!='plan_id'})!=protocol['plan_id']:
        raise ValueError("Protocol integrity failure")
    master_path=splits/'master_manifest.csv'
    if file_sha256(master_path)!=protocol['master_manifest_sha256']:raise ValueError("Master changed")
    base=replace(Config(),features=FeatureConfig(min_duration_seconds=.5,sampling_mode='native_stride'))
    master=load_manifest(master_path,base);lookup={r.clip_id:i for i,r in enumerate(master)}
    rounds={}
    for entry in protocol['rounds']:
        name=entry['round_id'];path=splits/name/'manifest.csv';cfg=load_config(splits/name/'config.json')
        if file_sha256(path)!=entry['manifest_sha256']:raise ValueError("Round manifest changed")
        records=read_csv(path);check_round(records,entry['heldout'])
        rows=load_manifest(path,cfg)
        rounds[name]={'entry':entry,'path':path,'rows':rows,'config':cfg}
    code={str(p):file_sha256(p) for p in sorted(Path('src/branch_c').glob('*.py'))}
    scripts={str(p):file_sha256(p) for p in [Path('scripts/generator_rotation.py'),Path(__file__)]}
    root.mkdir(parents=True)
    plan={'declared_at':now(),'protocol':PROTOCOL,'split_plan_id':protocol['plan_id'],'versions':runtime_versions(),
          'core_code_sha256':code,'runner_sha256':scripts,'feature_signature':feature_signature(base.features),
          'selection':'Round-specific seen-family validation only; six predeclared candidates; no test-guided changes',
          'bootstrap':'1000 group resamples; multiplicity-weighted exact equivalent validated against reference loop',
          'references':'Majority-real constant0 threshold.5; random U[0,1] seed42 over globally sorted3200 testIDs',
          'case_sampling':'After fixed tests, select lowest and highest unseen-AUROC rounds; strongestFP and lowest-scoreFN per heldout generator; eight nativeframes only',
          'reuse_policy':'Identical realIDs+SHA, preprocessing, algorithm parameters; selection labels and per-round output coverage never shared across heldout families',
          'interpretation':'Repartitioned previously evaluated cohort; no fresh blind-test claim; no independent28roundCI',
          'software_test_log_sha256':file_sha256('tmp/rotation_tests.log'),
          'old_run_model_sha256':file_sha256('artifacts/vida_guard32k/run_02/selection/selected/model.joblib')}
    plan['plan_id']=fingerprint(plan);write_json(root/'experiment_plan.json',plan)
    status={'plan_id':plan['plan_id'],'stages':{}}
    def stage(name,action):
        status['current_stage']=name;status['stages'][name]={'state':'running','started_at':now()}
        write_json(root/'status.json',status);print('Starting',name,flush=True);start=time.perf_counter()
        result=action()
        status['stages'][name]|={'state':'complete','finished_at':now(),'seconds':time.perf_counter()-start}
        write_json(root/'status.json',status);print('Completed',name,round(time.perf_counter()-start,2),'s',flush=True)
        return result
    features={};hashes={};X=np.full((len(master),37),np.nan)
    def prepare(indices):
        for n,i in enumerate(indices,1):
            r=master[i];digest=r.metadata['content_sha256']
            expected=cache/f'{CACHE_VERSION}-{digest}-{plan["feature_signature"]}.npz'
            if not expected.is_file():raise ValueError(f'Missing verified descriptor cache: {r.clip_id}')
            f,h=cached_features(r.path,base.features,cache,expected_sha256=digest)
            features[i]=f;hashes[i]=h;X[i]=f.clip
            if n%1000==0:print(f'Verified cached descriptors {n}/{len(indices)}',flush=True)
    dev=[i for i,r in enumerate(master) if r.metadata['rotation_eligible']=='true' and r.split!='test']
    stage('development_input_verification',lambda:prepare(dev))
    real_idx=np.array([i for i in dev if master[i].split=='train' and master[i].label==0])
    train_idx=np.array([i for i in dev if master[i].split=='train'])
    val_idx=np.array([i for i in dev if master[i].split=='val'])
    labels=np.array([r.label for r in master]);ids=np.array([r.clip_id for r in master])
    fold_ids=np.array([int(r.metadata['fold_id']) if r.split=='train' else -1 for r in master])
    fitkey=fingerprint({'real_inputs':[(master[i].clip_id,hashes[i]) for i in real_idx],'feature_signature':plan['feature_signature']})
    candidates={};candidate_val={};selection_results={};bundles={};oof_values={}
    def fit_candidates():
        for name,modelcfg in predeclared_candidates(42):
            model=RealOnlyAnomalyModel(modelcfg).fit(X[real_idx],labels[real_idx],ids[real_idx])
            np.testing.assert_allclose(model.pipeline[0].mean_,X[real_idx].mean(axis=0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(model.pipeline[0].var_,X[real_idx].var(axis=0),rtol=1e-12,atol=1e-12)
            candidates[name]=model;candidate_val[name]={i:s for i,s in zip(val_idx,model.score(X[val_idx]))}
            directory=root/'shared_candidates'/name;directory.mkdir(parents=True)
            joblib.dump(model,directory/'clip_model.joblib')
            write_json(directory/'fit_audit.json',{'fit_key':fitkey,'model':model.metadata(),'real_only_numeric_scaler_check':True})
            print('Fitted candidate',name,'real rows',model.fit_rows,flush=True)
    stage('six_real_only_candidates',fit_candidates)
    def select_rounds():
        for name,info in rounds.items():
            rows=info['rows'];selected=[lookup[r.clip_id] for r in rows if r.metadata.get('validation_role')=='model_selection']
            comparison=[]
            for order,(candidate,modelcfg) in enumerate(predeclared_candidates(42)):
                s=np.array([candidate_val[candidate][i] for i in selected]);y=labels[selected]
                threshold=choose_real_threshold(s,y,.05)
                comparison.append({'candidate_id':candidate,'order':order,'model_config':asdict(modelcfg),
                                   'selection_threshold':threshold,'validation':metrics(y,s,threshold)})
            comparison.sort(key=lambda c:(-c['validation']['auroc'],-c['validation']['macro_f1'],c['order']))
            result={'round_id':name,'selected_candidate_id':comparison[0]['candidate_id'],'ranked_candidates':comparison,
                    'selection_clip_ids':ids[selected].tolist(),'heldout_generators':info['entry']['heldout'],
                    'real_fit_key':fitkey,'test_metrics_read':False,'selected_at':now()}
            selection_results[name]=result;write_json(root/name/'selection_report.json',result)
            print(name,info['entry']['heldout'],'selected',comparison[0]['candidate_id'],flush=True)
    stage('28_independent_selections',select_rounds)
    used=sorted({s['selected_candidate_id'] for s in selection_results.values()})
    real_cal=[i for i in val_idx if master[i].label==0 and master[i].metadata['validation_role']=='calibration']
    real_val=[i for i in val_idx if master[i].label==0]
    def fit_selected_windows_and_oof():
        window_X=[];window_ids=[]
        for i in real_idx:
            f=features[i];take=np.unique(np.linspace(0,len(f.windows)-1,min(4,len(f.windows))).astype(int))
            window_X.extend(f.windows[take]);window_ids.extend([ids[i]]*len(take))
        window_X=np.stack(window_X)
        for candidate in used:
            model=candidates[candidate];cfg=replace(base,model=model.config)
            window=RealOnlyAnomalyModel(model.config).fit(window_X,np.zeros(len(window_X),int),window_ids)
            np.testing.assert_allclose(window.pipeline[0].mean_,window_X.mean(axis=0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(window.pipeline[0].var_,window_X.var(axis=0),rtol=1e-12,atol=1e-12)
            fit_rows=[master[i] for i in list(real_idx)+real_val]
            bundle=_new_bundle(cfg,model,window,fit_rows,[hashes[i] for i in list(real_idx)+real_val],'research',fit_rows)
            bundle['threshold']=choose_real_threshold([candidate_val[candidate][i] for i in real_cal],np.zeros(len(real_cal),int),.05)
            bundle['threshold_comparison']='>'
            bundle['validation_manifest_rows']=[asdict(master[i])|{'path':str(master[i].path)} for i in real_val]
            bundle['validation_content_hashes']=[hashes[i] for i in real_val]
            bundle['selection_provenance']={'kind':'shared_identical_real_fit','round_selections':'../../round_*/selection_report.json','real_fit_key':fitkey}
            directory=root/'shared_models'/candidate;save_bundle(directory/'model.joblib',bundle);bundles[candidate]=bundle
            write_json(directory/'fit_report.json',{'candidate_id':candidate,'model_id':bundle['model_id'],'model_sha256':file_sha256(directory/'model.joblib'),
                       'clip':model.metadata(),'window':window.metadata(),'real_fit_key':fitkey,'threshold':bundle['threshold'],
                       'threshold_clip_ids':ids[real_cal].tolist(),'versions':runtime_versions(),'mean_and_variance_verified':True})
            scores={};audits=[]
            for fold in range(5):
                fit=real_idx[fold_ids[real_idx]!=fold];held=train_idx[fold_ids[train_idx]==fold]
                estimator=RealOnlyAnomalyModel(model.config).fit(X[fit],labels[fit],ids[fit])
                np.testing.assert_allclose(estimator.pipeline[0].mean_,X[fit].mean(axis=0),rtol=1e-12,atol=1e-12)
                values=estimator.score(X[held]);scores.update(zip(held,values.tolist()))
                path=directory/'oof_models'/f'fold_{fold}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(estimator,path)
                audits.append({'fold_id':fold,'fit_clip_ids':ids[fit].tolist(),'heldout_clip_ids':ids[held].tolist(),
                               'fit_rows':len(fit),'model_sha256':file_sha256(path),
                               'fit_key':fingerprint({'real_inputs':[(ids[i],hashes[i]) for i in fit],'features':plan['feature_signature'],'model':asdict(model.config)})})
                print(candidate,'OOF fold',fold,'real fit',len(fit),'scores',len(held),flush=True)
            oof_values[candidate]=scores;write_json(directory/'oof_audit.json',{'candidate_id':candidate,'folds':audits,'real_only':True})
    stage('shared_selected_real_fits_and_oof',fit_selected_windows_and_oof)
    def export_development():
        for name,info in rounds.items():
            candidate=selection_results[name]['selected_candidate_id'];bundle=bundles[candidate];rows=info['rows']
            train=[r for r in rows if r.split=='train'];val=[r for r in rows if r.split=='val']
            if {r.clip_id for r in train if r.label==0}!=set(ids[real_idx]):raise ValueError('Shared real fit is not identical')
            for fold in range(5):
                fit=[r for r in train if r.label==0 and int(r.metadata['fold_id'])!=fold]
                held=[r for r in train if int(r.metadata['fold_id'])==fold]
                if {r.group_id for r in fit}&{r.group_id for r in held}:raise ValueError('OOF group leak')
            _write_scores(root/name/'oof_scores.csv',train,[oof_values[candidate][lookup[r.clip_id]] for r in train],
                          'out_of_fold',[int(r.metadata['fold_id']) for r in train],run_kind='research',content_hashes=[r.metadata['content_sha256'] for r in train])
            _write_scores(root/name/'validation_scores.csv',val,[candidate_val[candidate][lookup[r.clip_id]] for r in val],
                          'final_model',model_id=bundle['model_id'],run_kind='research',content_hashes=[r.metadata['content_sha256'] for r in val])
            write_json(root/name/'model_reference.json',{'round_id':name,'protocol':f'{PROTOCOL}:{name}','shared_model_protocol':PROTOCOL,
                       'model_path':f'../shared_models/{candidate}/model.joblib','model_id':bundle['model_id'],
                       'model_sha256':file_sha256(root/'shared_models'/candidate/'model.joblib'),'selected_candidate':candidate,
                       'threshold':bundle['threshold'],'real_fit_key':fitkey,'oof_shared_audit':f'../shared_models/{candidate}/oof_audit.json',
                       'oof_rows':len(train),'validation_rows':len(val),'manifest_sha256':file_sha256(info['path'])})
    stage('per_round_oof_and_validation_exports',export_development)
    freeze={'frozen_at':now(),'plan_id':plan['plan_id'],'selected_candidates':{n:s['selected_candidate_id'] for n,s in selection_results.items()},
            'selection_reports_sha256':{n:file_sha256(root/n/'selection_report.json') for n in rounds},
            'models_sha256':{c:file_sha256(root/'shared_models'/c/'model.joblib') for c in used},'test_metrics_read':False}
    write_json(root/'model_freeze.json',freeze)
    test_idx=[i for i,r in enumerate(master) if r.split=='test' and r.metadata['rotation_eligible']=='true']
    stage('test_input_verification_after_freeze',lambda:prepare(test_idx))
    test_values={c:{i:s for i,s in zip(test_idx,bundles[c]['clip_model'].score(X[test_idx]))} for c in used}
    random=dict(zip(sorted(ids[test_idx]),np.random.default_rng(42).uniform(size=len(test_idx)).tolist()))
    results=[]
    def test_rounds():
        for name,info in rounds.items():
            candidate=selection_results[name]['selected_candidate_id'];bundle=bundles[candidate];threshold=bundle['threshold']
            for view in ['seen_generators','unseen_generators']:
                rows=[r for r in info['rows'] if r.metadata.get('evaluation_subset')==view]
                _check_evaluation_leakage(bundle,rows,[r.metadata['content_sha256'] for r in rows])
                y=np.array([r.label for r in rows]);s=np.array([test_values[candidate][lookup[r.clip_id]] for r in rows])
                groups=connected_groups(rows,[r.metadata['content_sha256'] for r in rows])
                report={'round_id':name,'protocol':f'{PROTOCOL}:{name}','view':view,'model_id':bundle['model_id'],
                        'model_sha256':freeze['models_sha256'][candidate],'threshold':threshold,'evaluated_at':now(),
                        'heldout_generators':info['entry']['heldout'],'metrics':metrics(y,s,threshold),
                        'ci_95':fast_group_bootstrap(y,s,groups,threshold),'per_generator':{},'per_generator_ci_95':{},'per_real_source':{},
                        'baselines':{'majority_real':metrics(y,np.zeros(len(y)),.5),'fully_random':metrics(y,[random[r.clip_id] for r in rows],.5)}}
                for generator in sorted({r.generator for r in rows if r.label==1}):
                    selected=np.array([r.label==0 or r.generator==generator for r in rows]);sub=[r for r,k in zip(rows,selected) if k]
                    report['per_generator'][generator]=metrics(y[selected],s[selected],threshold)
                    report['per_generator_ci_95'][generator]=fast_group_bootstrap(y[selected],s[selected],connected_groups(sub,[r.metadata['content_sha256'] for r in sub]),threshold)
                for source in sorted({r.source for r in rows if r.label==0}):
                    select=np.array([r.label==0 and r.source==source for r in rows])
                    report['per_real_source'][source]=metrics(y[select],s[select],threshold)
                directory=root/name/view
                _write_scores(directory/'test_scores.csv',rows,s,'final_model',model_id=bundle['model_id'],run_kind='research',content_hashes=[r.metadata['content_sha256'] for r in rows])
                report['error_review']=write_error_review(directory/'error_review.csv',rows,s,threshold,[features[lookup[r.clip_id]] for r in rows],model_id=bundle['model_id'],run_kind='research')
                write_json(directory/'test_report.json',report)
                results.append({'round_id':name,'heldout_1':info['entry']['heldout'][0],'heldout_2':info['entry']['heldout'][1],
                                'view':view,'candidate':candidate,**{k:v for k,v in report['metrics'].items() if not isinstance(v,(list,dict))},
                                'auroc_ci_low':report['ci_95']['auroc'][0],'auroc_ci_high':report['ci_95']['auroc'][1]})
            print('Evaluated',name,'unseen AUC',round(results[-1]['auroc'],4),flush=True)
        write_csv(root/'metrics_by_round.csv',results)
    stage('all_28_fixed_model_test_rounds',test_rounds)
    def verify():
        if any(file_sha256(p)!=h for p,h in code.items()) or any(file_sha256(p)!=h for p,h in scripts.items()):raise ValueError('Frozen code changed')
        if any(file_sha256(root/n/'selection_report.json')!=h for n,h in freeze['selection_reports_sha256'].items()):raise ValueError('Selection changed')
        for candidate in used:
            bundle=load_bundle(root/'shared_models'/candidate/'model.joblib')
            if file_sha256(root/'shared_models'/candidate/'model.joblib')!=freeze['models_sha256'][candidate]:raise ValueError('Frozen model changed')
            np.testing.assert_array_equal(bundle['clip_model'].score(X[val_idx]),[candidate_val[candidate][i] for i in val_idx])
            np.testing.assert_array_equal(bundle['clip_model'].score(X[test_idx]),[test_values[candidate][i] for i in test_idx])
            if set(bundle['clip_model'].fit_clip_ids)!=set(ids[real_idx]) or set(bundle['window_model'].fit_clip_ids)!=set(ids[real_idx]):raise ValueError('Generated/incorrect final fit inputs')
            expected_threshold=choose_real_threshold([candidate_val[candidate][i] for i in real_cal],np.zeros(len(real_cal),int),.05)
            if expected_threshold!=bundle['threshold']:raise ValueError('Calibration threshold changed')
            audit=json.loads((root/'shared_models'/candidate/'oof_audit.json').read_text())
            for fold in range(5):
                fit=real_idx[fold_ids[real_idx]!=fold];held=train_idx[fold_ids[train_idx]==fold]
                model=RealOnlyAnomalyModel(candidates[candidate].config).fit(X[fit],labels[fit],ids[fit])
                np.testing.assert_array_equal(model.score(X[held]),[oof_values[candidate][i] for i in held])
                if set(audit['folds'][fold]['fit_clip_ids'])!=set(ids[fit]):raise ValueError('OOF real fit mismatch')
                saved_path=root/'shared_models'/candidate/'oof_models'/f'fold_{fold}.joblib'
                if file_sha256(saved_path)!=audit['folds'][fold]['model_sha256']:raise ValueError('Saved OOF model changed')
                saved=joblib.load(saved_path)
                np.testing.assert_array_equal(saved.score(X[held]),model.score(X[held]))
                np.testing.assert_allclose(saved.pipeline[0].var_,X[fit].var(axis=0),rtol=1e-12,atol=1e-12)
        coverage=[]
        for name,info in rounds.items():
            if file_sha256(info['path'])!=info['entry']['manifest_sha256']:raise ValueError('Round manifest changed during experiment')
            candidate=selection_results[name]['selected_candidate_id'];model=bundles[candidate]['clip_model']
            for file,expected,source in [('oof_scores.csv',[r for r in info['rows'] if r.split=='train'],oof_values[candidate]),
                                         ('validation_scores.csv',[r for r in info['rows'] if r.split=='val'],candidate_val[candidate])]:
                records=read_csv(root/name/file)
                if len(records)!=len(expected) or {r['clip_id'] for r in records}!={r.clip_id for r in expected}:raise ValueError('Incomplete export')
                np.testing.assert_array_equal([float(r['branch_c_score']) for r in records],[source[lookup[r['clip_id']]] for r in records])
                expected_rows={r.clip_id:r for r in expected}
                for record in records:
                    r=expected_rows[record['clip_id']]
                    if (int(record['label'])!=r.label or record['content_sha256']!=r.metadata['content_sha256']
                            or record['protocol']!=r.metadata['protocol'] or record['is_probability']!='false'):
                        raise ValueError('Score provenance changed')
                    if file=='oof_scores.csv' and record['fold_id']!=r.metadata['fold_id']:raise ValueError('OOF fold changed')
                    if file=='validation_scores.csv' and record['validation_role']!=r.metadata['validation_role']:raise ValueError('Validation role changed')
            for view in ['seen_generators','unseen_generators']:
                records=read_csv(root/name/view/'test_scores.csv');report=json.loads((root/name/view/'test_report.json').read_text())
                expected={r.clip_id for r in info['rows'] if r.metadata.get('evaluation_subset')==view}
                if len(records)!=len(expected) or {r['clip_id'] for r in records}!=expected:raise ValueError('Incomplete test export')
                ix=[lookup[r['clip_id']] for r in records];values=[float(r['branch_c_score']) for r in records]
                np.testing.assert_array_equal(model.score(X[ix]),values)
                if report['metrics']!=metrics(labels[ix],values,bundles[candidate]['threshold']):raise ValueError('Metric mismatch')
                if report['evaluated_at']<=freeze['frozen_at'] or report['ci_95']['valid_repeats']!=1000:raise ValueError('Test freeze or CI failure')
                for generator,point in report['per_generator'].items():
                    selected=np.array([r['label']=='0' or r['generator']==generator for r in records])
                    if point!=metrics(labels[ix][selected],np.asarray(values)[selected],bundles[candidate]['threshold']):raise ValueError('Per-generator metric mismatch')
                    if report['per_generator_ci_95'][generator]['valid_repeats']!=1000:raise ValueError('Incomplete per-generator CI')
                coverage.append({'round_id':name,'view':view,'n':len(records),'scores_sha256':file_sha256(root/name/view/'test_scores.csv')})
        if file_sha256('artifacts/vida_guard32k/run_02/selection/selected/model.joblib')!=plan['old_run_model_sha256']:raise ValueError('Old experiment changed')
        proof={'status':'passed','verified_at':now(),'plan_id':plan['plan_id'],'rounds':28,'views':56,'input_hash_and_cached_features_verified':len(features),
               'real_final_fit':len(real_idx),'real_per_oof_fit':7680,'real_threshold_calibration':len(real_cal),'shared_selected_candidates':used,
               'group_isolation':True,'6seen_2unseen_every_round':True,'all_round_scores_numerically_reproduced':True,
               'all_selected_5fold_models_independently_refitted_and_scores_exact':True,'clip_and_window_scaler_mean_variance_real_only_verified':True,
               'frozen_before_tests':True,'old_experiment_preserved':True,'coverage':coverage}
        write_json(root/'verification.json',proof)
    stage('numeric_reproduction_and_coverage_audit',verify)
    def benchmarks():
        i=min(real_idx,key=lambda i:(abs(float(master[i].metadata['duration_seconds'])-10),master[i].clip_id))
        for candidate in used:
            benchmark_video(root/'shared_models'/candidate/'model.joblib',master[i].path,root/'shared_models'/candidate/'runtime.json',3,120)
    stage('fresh_process_cpu_benchmarks',benchmarks)
    status['current_stage']='complete';status['finished_at']=now();write_json(root/'status.json',status)
    print('All28 rotations complete and numerically verified:',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits-dir',default='data/VIDA-Guard32K/rotation_6seen_2unseen_v1')
    p.add_argument('--output-dir',default='artifacts/vida_guard32k/rotation_6seen_2unseen_v1')
    p.add_argument('--cache-dir',default='artifacts/vida_guard32k/cache')
    run(p.parse_args())
