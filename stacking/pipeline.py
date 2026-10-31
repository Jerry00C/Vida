"""Only scalar branch evidence is fitted; no branch model is retrained here."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import time
import warnings
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             precision_score, recall_score, confusion_matrix, brier_score_loss, log_loss, average_precision_score)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BRANCHES = ("A", "B", "C")
ALIASES = {"A":("video_logit","video_score","score_fake"),
           "B":("raw_margin",), "C":("branch_c_score","anomaly_score")}
KNOWN = ("Pika","VideoCrafter2","T2V-Zero","OpenSora","SVD","MuseV","Mora","CogVideo")
EXTERNAL = ("ModelScope","Sora","Kling")

def require(condition, message):
    if not condition: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def table(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path,encoding="utf-8-sig",keep_default_na=False,float_precision="round_trip")

def config(path):
    p=Path(path).resolve();d=json.loads(p.read_text(encoding="utf-8-sig"));d["_config_path"]=str(p)
    for key in ("manifest","base_manifest","predictions_dir","output_dir"):
        q=Path(d[key]);d[key]=str(q if q.is_absolute() else (p.parent/q).resolve())
    require(d.get("run_kind") in {"research","synthetic_demo"},"Explicit run_kind required")
    return d

def manifest(cfg):
    data=table(cfg["manifest"]);base=table(cfg["base_manifest"])
    required={"clip_id","label","family","split","group_id","protocol_id","sha256","real_pool"}
    require(required.issubset(data.columns),f"Missing manifest fields: {required-set(data.columns)}")
    for col in ("clip_id","group_id","sha256"):
        require(data[col].astype(str).str.len().gt(0).all(),"Empty "+col)
    require(data.clip_id.is_unique,"Duplicate clip_id in holdout manifest")
    require(data.sha256.is_unique,"Byte duplicates in holdout manifest")
    require(set(data.label)=={0,1},"Both labels required")
    require(set(data.protocol_id)=={cfg["protocol_id"]},"Protocol mismatch")
    require(set(data.split)=={"fusion_train","calibration","in_domain_test","external_test"},"Unexpected/missing holdout split")
    require(data.groupby("group_id").split.nunique().max()==1,"Source group crosses holdout splits")
    for key in ("clip_id","group_id","sha256"):
        require(not set(data[key])&set(base[key]),f"Holdout overlaps original training {key}; resplitting old scores does not remove leakage")
    for split in ("fusion_train","calibration","in_domain_test"):
        part=data[data.split==split]
        require(set(part.label)=={0,1},"Both labels required in "+split)
        require(set(part.loc[part.label==1,"family"])==set(KNOWN),"Expected all eight seen families in "+split)
    ext=data[(data.split=="external_test")&(data.label==1)]
    require(set(ext.family)==set(EXTERNAL),"External test must contain ModelScope, Sora and Kling")
    return data.set_index("clip_id",drop=False),digest(cfg["manifest"]),digest(cfg["base_manifest"])

def import_scores(cfg, branch, source, model_id, score_column=None, direction="higher_is_fake"):
    require(branch in BRANCHES,"Unknown branch")
    require(bool(model_id) and not model_id.startswith("REPLACE"),"Supply the actual frozen model ID or checkpoint SHA256")
    data,holdout_sha,base_sha=manifest(cfg);incoming=table(source)
    require("clip_id" in incoming,"Predictions require clip_id")
    require(incoming.clip_id.is_unique,"Duplicate predictions; select one model/round/partition before importing")
    for identity_column in ("model_id","checkpoint_sha256"):
        if identity_column in incoming:
            require(incoming[identity_column].nunique()==1,"A branch score delivery mixes multiple model versions")
    require(set(incoming.clip_id).issubset(data.index),"Prediction IDs are not in the new independent suite; old full32k predictions are not valid holdouts")
    column=score_column or next((c for c in ALIASES[branch] if c in incoming),None)
    require(column in incoming,f"Expected score column for {branch}: {ALIASES[branch]}")
    if "label" in incoming:
        expected=data.loc[incoming.clip_id,"label"].to_numpy()
        require(np.array_equal(incoming.label.to_numpy(),expected),"Prediction labels disagree with canonical manifest")
    if "sha256" in incoming:
        require(np.array_equal(incoming.sha256.to_numpy(),data.loc[incoming.clip_id,"sha256"].to_numpy()),"Prediction video hash disagrees with manifest")
    scores=pd.to_numeric(incoming[column],errors="coerce").to_numpy(dtype=float)
    require(np.isfinite(scores).all(),"Null/nonfinite branch scores; resolve ABSTAIN explicitly, never silently fill zeros")
    if "status" in incoming:
        require(incoming.status.astype(str).str.upper().isin({"OK","SUCCESS","PASSED"}).all(),"Branch abstention/failure present")
    require(direction in {"higher_is_fake","higher_is_real"},"Explicit score direction required")
    if direction=="higher_is_real":scores=-scores
    dest=Path(cfg["predictions_dir"])/f"{branch}.csv";dest.parent.mkdir(parents=True,exist_ok=True)
    require(not dest.exists(),f"Refuse to overwrite {dest}; archive previous predictions first")
    pd.DataFrame({"clip_id":incoming.clip_id,"score":scores,"status":"OK"}).to_csv(dest,index=False,encoding="utf-8-sig")
    save_json(dest.with_suffix(".meta.json"),{"schema_version":1,"branch":branch,"model_id":model_id,
        "protocol_id":cfg["protocol_id"],"holdout_manifest_sha256":holdout_sha,"score_kind":"independent_holdout","run_kind":cfg["run_kind"],
        "base_training_manifest_sha256":base_sha,"source_score_column":column,"source_direction":direction,
        "score_direction":"higher_is_fake","source_file_sha256":digest(source),"csv_sha256":digest(dest),
        "declaration":"Frozen branch trained only on original base cohort; these holdouts were never used for fitting/selection. This declaration must be confirmed from actual training provenance."})
    return {"branch":branch,"rows":len(incoming),"output":str(dest)}

def evidence(cfg, needed_splits):
    data,manifest_sha,base_sha=manifest(cfg)
    require(data.index.is_unique,"Duplicate manifest IDs")
    wanted=data[data.split.isin(needed_splits)].copy();lineage={};missing=[]
    for branch in BRANCHES:
        path=Path(cfg["predictions_dir"])/f"{branch}.csv";meta_path=path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():missing.append(branch);continue
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        require(meta.get("branch")==branch and meta.get("score_kind")=="independent_holdout",f"{branch}: non-holdout/in-sample scores rejected")
        require(meta.get("run_kind")==cfg["run_kind"],"Synthetic/research provenance cannot be mixed")
        require(meta.get("protocol_id")==cfg["protocol_id"],"Branch protocol mismatch")
        require(meta.get("holdout_manifest_sha256")==manifest_sha,"Branch holdout manifest changed after score import")
        require(meta.get("base_training_manifest_sha256")==base_sha,"Branch base cohort identity mismatch")
        require(meta.get("csv_sha256")==digest(path),"Predictions changed after import")
        require(meta.get("score_direction")=="higher_is_fake", "Score direction mismatch")
        require(bool(meta.get("model_id")),"Missing frozen model identity")
        scores=table(path)
        require(scores.clip_id.is_unique,f"{branch}: duplicate prediction IDs")
        require(set(scores.clip_id).issubset(data.index),"Unexpected/old prediction IDs")
        require(set(wanted.index).issubset(set(scores.clip_id)),f"{branch}: missing scores for requested splits")
        scores=scores.set_index("clip_id")
        values=pd.to_numeric(scores.loc[wanted.index,"score"],errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(),f"{branch}: nonfinite scores")
        require(scores.loc[wanted.index,"status"].eq("OK").all(),"Branch failure present")
        wanted[branch]=values;lineage[branch]={k:meta[k] for k in ["model_id","base_training_manifest_sha256","source_score_column","source_direction"]}
    require(not missing,"Missing branch score delivery: "+", ".join(missing)+". Run prepare-requests and return new-suite per-video scores; no base retraining is performed here.")
    return wanted,manifest_sha,lineage

def metrics(y, scores, threshold):
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float);pred=scores>=threshold
    require(set(y)=={0,1},"Both classes required for binary evaluation")
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":len(y),"real":int((y==0).sum()),"generated":int((y==1).sum()),
        "auroc":float(roc_auc_score(y,scores)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "accuracy":float((pred==y).mean()),"f1_generated":float(f1_score(y,pred,zero_division=0)),
        "average_precision":float(average_precision_score(y,scores)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision_generated":float(precision_score(y,pred,zero_division=0)),
        "recall_generated":float(recall_score(y,pred,zero_division=0)),"real_fpr":float(fp/(tn+fp)),
        "brier":float(brier_score_loss(y,scores)),"threshold":float(threshold),
        "confusion_matrix_real_generated":[[int(tn),int(fp)],[int(fn),int(tp)]]}

def probability(bundle, frame):
    raw=bundle["stacker"].decision_function(frame[list(BRANCHES)])
    return bundle["calibrator"].predict_proba(raw.reshape(-1,1))[:,1]

def train(cfg):
    data,manifest_sha,lineage=evidence(cfg,{"fusion_train","calibration"})
    dest=Path(cfg["output_dir"])
    require(not dest.exists(),"Output already exists; preserve the frozen model, choose a new output_dir")
    training=data[data.split=="fusion_train"];cal=data[data.split=="calibration"]
    started=time.perf_counter()
    # Select regularization only inside fusion_train. Scaling is refitted in each fold.
    seed=42;max_iter=2000;tol=1e-8
    grid=[.001,.01,.1,1.,10.,100.]
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    folds=list(cv.split(training[list(BRANCHES)],training.label,groups=training.group_id))
    trials=[];oof_by_c={};fold_ids=np.full(len(training),-1,dtype=int)
    for c in grid:
        oof=np.full(len(training),np.nan);losses=[];aucs=[];iterations=[]
        for fold,(fit,valid) in enumerate(folds):
            tr,va=training.iloc[fit],training.iloc[valid]
            require(not set(tr.group_id)&set(va.group_id),"Fusion CV group leak")
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=max_iter,tol=tol,random_state=seed))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(tr[list(BRANCHES)],tr.label)
            values=model.predict_proba(va[list(BRANCHES)])[:,1]
            oof[valid]=values;fold_ids[valid]=fold
            losses.append(float(log_loss(va.label,values)))
            aucs.append(float(roc_auc_score(va.label,values)))
            iterations.append(int(model[-1].n_iter_.max()))
        require(np.isfinite(oof).all(),"Incomplete fusion CV predictions")
        trials.append({'C':c,'mean_log_loss':float(np.mean(losses)), 'fold_log_loss':losses,
                       'mean_auroc':float(np.mean(aucs)),'fold_auroc':aucs,'solver_iterations':iterations})
        oof_by_c[c]=oof
    selected=min(trials,key=lambda r:(r['mean_log_loss'],r['C']))
    stacker=make_pipeline(StandardScaler(),LogisticRegression(C=selected['C'],max_iter=max_iter,tol=tol,random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        stacker.fit(training[list(BRANCHES)],training.label)
    model_kind='L2 LogisticRegression';neural_report=None;neural_candidate=None;selected_oof=oof_by_c[selected['C']]
    if cfg.get('compare_mlp',False):
        require(Path(cfg['neural_python']).exists(),'Separate neural training Python is missing')
        with tempfile.TemporaryDirectory(prefix='vida-fusion-mlp-') as temporary:
            temporary=Path(temporary);input_file=temporary/'training_only.csv'
            training[['clip_id','group_id','label',*BRANCHES]].to_csv(input_file,index=False)
            subprocess.run([cfg['neural_python'],'-m','stacking.neural','--input',str(input_file),
                            '--output',str(temporary/'candidate')],cwd=Path(__file__).resolve().parents[1],check=True)
            neural_report=json.loads((temporary/'candidate/selection_report.json').read_text())
            neural_candidate=joblib.load(temporary/'candidate/candidate.joblib')
            if neural_report['mean_log_loss']<selected['mean_log_loss']:
                stacker=neural_candidate
                selected_oof=np.load(temporary/'candidate/oof.npy');model_kind='MLP 3-8-1'
    splitter=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=42)
    a,b=next(splitter.split(np.zeros((len(cal),1)),cal.label,groups=cal.group_id))
    probability_fit,threshold_fit=cal.iloc[a],cal.iloc[b]
    require(set(probability_fit.label)==set(threshold_fit.label)=={0,1},"Insufficient class/group diversity for independent calibration roles")
    require(not set(probability_fit.group_id)&set(threshold_fit.group_id),"Calibration group leak")
    calibrator=LogisticRegression(C=1000000.0,max_iter=1000,random_state=42)
    calibrator.fit(stacker.decision_function(probability_fit[list(BRANCHES)]).reshape(-1,1),probability_fit.label)
    require(calibrator.n_iter_.max()<1000,'Probability calibrator did not converge')
    bundle={"schema_version":1,"run_kind":cfg["run_kind"],"protocol_id":cfg["protocol_id"],"manifest_sha256":manifest_sha,
        "lineage":lineage,"stacker":stacker,"calibrator":calibrator,"feature_names":list(BRANCHES),
        "training_ids":training.clip_id.tolist(),"probability_fit_ids":probability_fit.clip_id.tolist(),
        "threshold_fit_ids":threshold_fit.clip_id.tolist(),"trained_at_utc":datetime.now(timezone.utc).isoformat()}
    # Predeclare single-branch comparisons; calibrate and set their thresholds on
    # exactly the same calibration roles, before any sealed test is evaluated.
    bundle['single_branch_baselines']={}
    for branch in BRANCHES:
        platt=make_pipeline(StandardScaler(),LogisticRegression(C=1000000.,max_iter=2000,tol=1e-8,random_state=seed))
        platt.fit(probability_fit[[branch]],probability_fit.label)
        require(platt[-1].n_iter_.max()<2000,'Single-branch calibrator did not converge')
        values=platt.predict_proba(threshold_fit[[branch]])[:,1]
        choices=np.unique(np.r_[0.,.5,1.,values])
        cut=max(choices,key=lambda t:(f1_score(threshold_fit.label,values>=t,average='macro',zero_division=0),t))
        bundle['single_branch_baselines'][branch]={'calibrator':platt,'threshold':float(cut)}
    scores=probability(bundle,threshold_fit)
    candidates=np.unique(np.r_[0.0,0.5,1.0,scores])
    threshold=max(candidates,key=lambda t:(f1_score(threshold_fit.label,scores>=t,average="macro",zero_division=0),t))
    bundle["threshold"]=float(threshold)
    dest.mkdir(parents=True)
    if neural_candidate is not None:
        (dest/'development_candidates').mkdir()
        joblib.dump(neural_candidate,dest/'development_candidates/mlp_training_only.joblib')
        save_json(dest/'development_candidates/README.json',{'purpose':'Preserved training-only MLP candidate, not a separately calibrated/tested final model; do not switch final deployment based on sealed test scores',
            'selected_for_final_fusion':model_kind=='MLP 3-8-1'})
    joblib.dump(bundle,dest/"stacker.joblib")
    save_json(dest/"MODEL_LOCK.json",{"model_sha256":digest(dest/"stacker.joblib"),"manifest_sha256":manifest_sha,
        "protocol_id":cfg["protocol_id"],"run_kind":cfg["run_kind"],"threshold":float(threshold),"lineage":lineage})
    summary={"run_kind":cfg["run_kind"],"not_research_result":cfg["run_kind"]!="research","protocol_id":cfg["protocol_id"],
        "fusion_training_rows":len(training),"probability_fit_rows":len(probability_fit),"threshold_selection_rows":len(threshold_fit),
        "threshold_selection_metrics_not_test":metrics(threshold_fit.label,scores,threshold),
        "standardized_feature_coefficients":dict(zip(BRANCHES,stacker[-1].coef_[0].tolist())) if model_kind=='L2 LogisticRegression' else None,
        "calibration_groups_disjoint":True,"test_labels_used_for_training":False,"branch_models_retrained":False,
        "hyperparameters":{'model':model_kind,'solver':'lbfgs' if model_kind=='L2 LogisticRegression' else 'AdamW',
            'selected_C':selected['C'] if model_kind=='L2 LogisticRegression' else None,'C_grid':grid,'max_iter':max_iter,'tol':tol,'random_seed':seed,
            'epoch':None if model_kind=='L2 LogisticRegression' else neural_report['final_epochs'],
            'epoch_note':'LR uses numerical solver iterations; MLP uses epoch-based AdamW with training-only validation early stopping',
            'actual_solver_iterations':int(stacker[-1].n_iter_.max()) if model_kind=='L2 LogisticRegression' else None,'converged':True,
            'neural_settings':neural_report['settings'] if model_kind=='MLP 3-8-1' else None,
            'selection':'minimum mean 5-fold StratifiedGroupKFold log loss within fusion_train only',
            'probability_calibration':'Platt logistic C=1e6 on probability_fit role only',
            'threshold_selection':'maximize macro F1 on threshold_selection role only'},
        "cross_validation":trials,"neural_candidate":neural_report,
        "selected_model_kind":model_kind,"selection_criterion":"lowest mean training-only grouped CV log loss; LR wins ties",
        "training_wall_seconds":time.perf_counter()-started}
    save_json(dest/"training_report.json",summary)
    training[['clip_id','group_id','label']].assign(cv_fold=fold_ids,
        selected_model_oof_probability=selected_oof).to_csv(dest/'fusion_train_cv.csv',index=False,encoding='utf-8-sig')
    assignment=pd.concat([probability_fit.assign(calibration_role="probability_fit"),threshold_fit.assign(calibration_role="threshold_selection")])
    assignment[["clip_id","group_id","label","calibration_role"]].to_csv(dest/"calibration_roles.csv",index=False,encoding="utf-8-sig")
    return summary

def load_bundle(cfg):
    folder=Path(cfg["output_dir"]);lock=json.loads((folder/"MODEL_LOCK.json").read_text(encoding="utf-8"))
    require(digest(folder/"stacker.joblib")==lock["model_sha256"],"Frozen stacker changed after training")
    # This file is created locally by train(); do not load third-party pickle files.
    return joblib.load(folder/"stacker.joblib")

def group_auc_ci(frame, repeats=200):
    rng=np.random.default_rng(42);groups=frame.group_id.unique();index={g:i for i,g in enumerate(groups)}
    membership=np.array([index[g] for g in frame.group_id]);results=[]
    for _ in range(repeats):
        counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));w=counts[membership]
        if (w[frame.label.to_numpy()==0].sum()>0 and w[frame.label.to_numpy()==1].sum()>0):
            results.append(float(roc_auc_score(frame.label,frame.score,sample_weight=w)))
    return np.quantile(results,[.025,.975]).tolist() if results else None

def evaluate(cfg, bootstrap=200):
    data,manifest_sha,lineage=evidence(cfg,{"in_domain_test","external_test"});bundle=load_bundle(cfg)
    require(bundle["manifest_sha256"]==manifest_sha,"Dataset changed after model freeze")
    require(bundle["lineage"]==lineage,"Branch checkpoint/score mapping changed since stacker fit")
    data["score"]=probability(bundle,data);threshold=bundle["threshold"]
    reports={};views={"in_domain":data[data.split=="in_domain_test"]}
    ext=data[data.split=="external_test"];fake=ext[ext.label==1]
    for pool in ("known_source","new_source"):
        real=ext[(ext.label==0)&(ext.real_pool==pool)]
        require(len(real)>=len(fake),"Insufficient external real controls: "+pool)
        views["cross_generator_"+pool]=pd.concat([fake,real.iloc[:len(fake)]])
        for family in EXTERNAL:
            f=fake[fake.family==family]
            shuffled=real.sample(frac=1,random_state=42).iloc[:len(f)]
            views[family+"_"+pool]=pd.concat([f,shuffled])
    for name,frame in views.items():
        result=metrics(frame.label,frame.score,threshold)
        result["auroc_group_bootstrap_95ci"]=group_auc_ci(frame,bootstrap) if bootstrap else None
        result["individual_branch_raw_score_auroc"]={b:float(roc_auc_score(frame.label,frame[b])) for b in BRANCHES}
        result['individual_branch_calibrated_metrics']={}
        for branch,baseline in bundle['single_branch_baselines'].items():
            values=baseline['calibrator'].predict_proba(frame[[branch]])[:,1]
            result['individual_branch_calibrated_metrics'][branch]=metrics(frame.label,values,baseline['threshold'])
        reports[name]=result
    summary={"run_kind":cfg["run_kind"],"not_research_result":cfg["run_kind"]!="research",
        "protocol_id":cfg["protocol_id"],"frozen_threshold":threshold,"views":reports,
        "mean_heldout_generator_auroc_known_real":float(np.mean([reports[f+"_known_source"]["auroc"] for f in EXTERNAL])),
        "external_fake_samples_shared_across_real_source_views":True,"test_set_used_for_fitting_or_threshold_selection":False}
    target=Path(cfg["output_dir"])/"evaluation"
    require(not target.exists(),"Evaluation already exists; do not overwrite sealed test results")
    target.mkdir()
    save_json(target/"metrics.json",summary)
    data[["clip_id","label","family","split","real_pool","score"]].to_csv(target/"predictions.csv",index=False,encoding="utf-8-sig")
    return summary

def prepare_requests(cfg):
    data,sha,base_sha=manifest(cfg);root=Path(cfg["predictions_dir"]);root.mkdir(parents=True,exist_ok=True)
    for branch in BRANCHES:
        folder=root/"requests"/branch;folder.mkdir(parents=True,exist_ok=True)
        data.to_csv(folder/"inference_manifest.csv",index=False,encoding="utf-8-sig")
        text=(f"# Branch {branch}: inference only\n\nProtocol: {cfg['protocol_id']}\n\n"
              "Use your frozen final checkpoint. Do not fit, fine-tune, select parameters or calibrate on these videos. "
              "Do not use labels/family as machine features. Return exactly one continuous score per clip_id. "
              f"Accepted native score fields: {', '.join(ALIASES[branch])}.\n\n"
              "Video paths are relative to the new suite data root E:/VIDA-Guard32K/stacking_v2. "
              "A may extract its own 8 sparse frames; B/C must use their checkpoint's temporal sampling. "
              "Do not pass A's sparse frames to B/C or substitute old full32k predictions. "
              "Return model identity/checkpoint SHA256 and actual training cohort provenance alongside scores. "
              "Report decoding failures as ABSTAIN with empty scores; do not invent zeros.\n")
        (folder/"EXPORT_REQUEST.md").write_text(text,encoding="utf-8")
    save_json(root/"requests/REQUEST_IDENTITY.json",{"manifest_sha256":sha,"base_manifest_sha256":base_sha,"protocol_id":cfg["protocol_id"]})
    return {"request_directory":str(root/"requests"),"clips":len(data)}

def doctor(cfg):
    result={"environment":"available","run_kind":cfg["run_kind"],"branch_models_retrained":False}
    try:
        data,sha,_=manifest(cfg);result.update(dataset="validated",clips=len(data),manifest_sha256=sha)
    except Exception as error:
        result.update(dataset="not_ready",dataset_error=str(error));return result
    try:
        evidence(cfg,{"fusion_train","calibration"});result["ready_to_train"]=True
    except Exception as error:
        result.update(ready_to_train=False,pending=str(error))
    result["branch_files"]={b:(Path(cfg["predictions_dir"])/f"{b}.csv").exists() for b in BRANCHES}
    return result
