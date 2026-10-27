"""Audit the completed real-data run, including numeric scaler/score/OOF reproduction."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import numpy as np

from branch_c.cache import cached_features, file_sha256, CACHE_VERSION
from branch_c.config import fingerprint, load_config, write_json
from branch_c.evaluation import choose_real_threshold, metrics
from branch_c.features import feature_signature
from branch_c.manifest import load_manifest, connected_groups
from branch_c.model import RealOnlyAnomalyModel, load_bundle


def require(ok, message):
    if not ok: raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify(args):
    root, cache = Path(args.run_dir), Path(args.cache_dir)
    original = load_manifest(args.manifest)
    lookup = {r.clip_id:r for r in original}
    real_parent_scopes={}
    prompt_scopes={}
    prompt_counts=Counter()
    for r in original:
        scope=(r.source,r.group_id,r.metadata.get("delivery_split",r.split),r.metadata.get("validation_role",""),
               r.metadata.get("fold_id","") if r.split=="train" else "")
        if r.label==0:
            parent=r.metadata.get("original_source_id","")
            require(bool(parent),"Missing original real-video source ID")
            require(parent not in real_parent_scopes or real_parent_scopes[parent]==scope,"Original real-video parent crosses source/group/partition/role/fold")
            real_parent_scopes[parent]=scope
        elif r.generator in {"Pika","VideoCrafter2","T2V-Zero"}:
            match=re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",r.clip_id)
            require(match is not None,"Missing shared prompt UUID")
            token=match.group()
            prompt_scope=scope[1:]
            require(token not in prompt_scopes or prompt_scopes[token]==prompt_scope,"Shared prompt crosses group/partition/role/fold")
            prompt_scopes[token]=prompt_scope
            prompt_counts[token]+=1
    require(len(real_parent_scopes)==10000 and len(prompt_scopes)==3000 and set(prompt_counts.values())=={3},"Unexpected source/prompt grouping")
    plan = json.loads((root/"protocol_plan.json").read_text())
    require(fingerprint({k:v for k,v in plan.items() if k!="plan_id"})==plan["plan_id"],"Plan integrity failure")
    frozen=plan["frozen_inputs"]
    require(file_sha256(args.manifest)==frozen["manifest_sha256"],"Manifest changed")
    require(all(file_sha256(p)==h for p,h in frozen["code_sha256"].items()),"Model source changed")
    selected=load_config(root/"selection/selected_config.json")
    bundle=load_bundle(root/"selection/selected/model.joblib")
    require(bundle["config"]==selected,"Selected config differs from bundle")
    model_hash=file_sha256(root/"selection/selected/model.joblib")
    freeze=json.loads((root/"model_freeze.json").read_text())
    require(model_hash==freeze["model_sha256"],"Frozen model changed")
    selection=json.loads((root/"selection/selection_report.json").read_text())
    require(selection["test_files_read"]==0,"Test read during selection")
    sel_ids={r.clip_id for r in original if r.metadata.get("validation_role")=="model_selection"}
    cal_ids={r.clip_id for r in original if r.metadata.get("validation_role")=="calibration" and r.label==0}
    real_ids={r.clip_id for r in original if r.split=="train" and r.label==0}
    require((len(original),len(real_ids),len(sel_ids),len(cal_ids))==(32000,8400,1800,900),"Unexpected population")
    require(set(bundle["clip_model"].fit_clip_ids)==real_ids,"Final fit used wrong clips")
    fit_report=json.loads((root/"selection/selected/fit_report.json").read_text())
    require(set(fit_report["threshold_clip_ids"])==cal_ids,"Threshold used non-calibration data")
    for candidate in selection["ranked_candidates"]:
        records=read_csv(root/"selection/candidates"/f"{candidate['candidate_id']}_validation_scores.csv")
        require({r["clip_id"] for r in records}==sel_ids and len(records)==1800,"Candidate used calibration rows")
        require(set(candidate["model"]["fit_clip_ids"])==real_ids,"Candidate fit on wrong clips")
        y=np.array([int(r["label"]) for r in records])
        s=np.array([float(r["branch_c_score"]) for r in records])
        t=choose_real_threshold(s,y,.05)
        require(t==candidate["threshold"] and metrics(y,s,t)==candidate["validation"],"Candidate metrics not reproducible")
    require(selection["selected_candidate_id"]==selection["ranked_candidates"][0]["candidate_id"],"Winner mismatch")
    # Every descriptor must already exist; this audit may not decode/re-extract media.
    signature=feature_signature(selected.features)
    features=[]
    for i,r in enumerate(original):
        path=cache/f"{CACHE_VERSION}-{r.metadata['content_sha256']}-{signature}.npz"
        require(path.is_file(),f"Missing completed cache for {r.clip_id}")
        f,_=cached_features(r.path,selected.features,cache,expected_sha256=r.metadata["content_sha256"])
        features.append(f)
        if (i+1)%1000==0: print(f"Audited input/hash/cache {i+1}/32000",flush=True)
    X=np.stack([f.clip for f in features])
    real=np.array([r.clip_id in real_ids for r in original])
    np.testing.assert_allclose(bundle["clip_model"].pipeline[0].mean_,X[real].mean(axis=0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(bundle["clip_model"].pipeline[0].var_,X[real].var(axis=0),rtol=1e-12,atol=1e-12)
    windows=[]
    for r,f in zip(original,features):
        if r.clip_id in real_ids:
            indices=np.unique(np.linspace(0,len(f.windows)-1,min(4,len(f.windows))).astype(int))
            windows.extend(f.windows[indices])
    np.testing.assert_allclose(bundle["window_model"].pipeline[0].mean_,np.stack(windows).mean(axis=0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(bundle["window_model"].pipeline[0].var_,np.stack(windows).var(axis=0),rtol=1e-12,atol=1e-12)
    require(set(bundle["window_model"].fit_clip_ids)==real_ids,"Window fit used wrong clips")
    reports={}
    score_inputs=[("validation",root/"selection/selected/validation_scores.csv",{r.clip_id for r in original if r.split=="val"})]
    for subset,n in [("in_domain_test",3600),("cross_test",8000)]:
        expected={r.clip_id for r in original if r.metadata.get("evaluation_subset")==subset}
        require(len(expected)==n,"Unexpected test population")
        score_inputs.append((subset,root/subset/"test_scores.csv",expected))
    index={r.clip_id:i for i,r in enumerate(original)}
    for name,path,expected in score_inputs:
        records=read_csv(path)
        require(len(records)==len(expected) and {r["clip_id"] for r in records}==expected,f"Incomplete {name} coverage")
        require(all(r["model_id"]==bundle["model_id"] and r["is_probability"]=="false" and r["run_kind"]=="research" for r in records),"Wrong score provenance")
        require(all(r["content_sha256"]==lookup[r["clip_id"]].metadata["content_sha256"] and int(r["label"])==lookup[r["clip_id"]].label for r in records),"Score input mismatch")
        values=np.array([float(r["branch_c_score"]) for r in records])
        np.testing.assert_array_equal(bundle["clip_model"].score(X[[index[r["clip_id"]] for r in records]]),values)
        if name=="validation":
            real_cal=np.array([r["clip_id"] in cal_ids for r in records])
            require(bundle["threshold"]==float(np.quantile(values[real_cal],.95,method="higher")),"Threshold changed")
        else:
            report=json.loads((root/name/"test_report.json").read_text())
            require(report["model_sha256"]==model_hash and report["threshold"]==bundle["threshold"],"Evaluation model/threshold changed")
            require(report["metrics"]==metrics([int(r["label"]) for r in records],values,bundle["threshold"]),"Test metrics mismatch")
            require(report["ci_95"]["valid_repeats"]==1000,"Incomplete bootstrap")
            selected_rows=[lookup[r["clip_id"]] for r in records]
            selected_hashes=[r["content_sha256"] for r in records]
            require(report["ci_95"]["n_groups"]==len(np.unique(connected_groups(selected_rows,selected_hashes))),"Wrong bootstrap groups")
            for generator,result in report["per_generator"].items():
                subset=np.array([r["label"]=="0" or r["generator"]==generator for r in records])
                y=np.array([int(r["label"]) for r in records])
                require(result==metrics(y[subset],values[subset],bundle["threshold"]),"Per-generator metrics mismatch")
                ci=report["per_generator_ci_95"][generator]
                require(ci["valid_repeats"]==1000 and ci["requested_repeats"]==1000,"Incomplete per-generator uncertainty")
            for source,result in report["per_real_source"].items():
                subset=np.array([r["label"]=="0" and r["source"]==source for r in records])
                require(result==metrics(np.zeros(int(subset.sum()),dtype=int),values[subset],bundle["threshold"]),"Real-source FPR mismatch")
            reports[name]={"n":len(records),"metrics":report["metrics"],"ci_95":report["ci_95"]}
    train=[r for r in original if r.split=="train"]
    supplied={r.clip_id:int(r.metadata["fold_id"]) for r in train}
    common=read_csv(root/"common_folds/fold_assignments.csv")
    oof=read_csv(root/"oof/oof_scores.csv")
    require({r["clip_id"]:int(r["fold_id"]) for r in common}==supplied,"Common folds changed")
    require(len(oof)==16800 and {r["clip_id"]:int(r["fold_id"]) for r in oof}==supplied,"OOF coverage/folds changed")
    require(all(r["score_kind"]=="out_of_fold" and r["run_kind"]=="research"
                and int(r["label"])==lookup[r["clip_id"]].label
                and r["content_sha256"]==lookup[r["clip_id"]].metadata["content_sha256"]
                for r in oof),"OOF score/input provenance mismatch")
    audit=json.loads((root/"oof/oof_audit.json").read_text())
    # JSON stores the configuration's tuple fields as lists; compare canonical
    # JSON fingerprints so serialization alone cannot look like a model change.
    require(fingerprint(audit["config"])==fingerprint(selected.to_dict()),"OOF config differs from selected model")
    common_plan=json.loads((root/"common_folds/fold_plan.json").read_text())
    require(audit["fold_plan"]["plan_id"]==common_plan["plan_id"],"OOF differs from common plan")
    train_X=X[[index[r.clip_id] for r in train]]
    train_y=np.array([r.label for r in train])
    train_ids=np.array([r.clip_id for r in train])
    oof_scores={r["clip_id"]:float(r["branch_c_score"]) for r in oof}
    for fold in audit["folds"]:
        k=fold["fold_id"]
        held=np.array([supplied[r.clip_id]==k for r in train])
        expected_fit={r.clip_id for r in train if r.label==0 and supplied[r.clip_id]!=k}
        require(set(fold["fit_clip_ids"])==expected_fit and len(expected_fit)==6720,"OOF fit leaked/wrong real count")
        require(set(fold["heldout_clip_ids"])==set(train_ids[held]),"OOF holdout mismatch")
        model=RealOnlyAnomalyModel(selected.model).fit(train_X[~held],train_y[~held],train_ids[~held])
        np.testing.assert_array_equal(model.score(train_X[held]),[oof_scores[i] for i in train_ids[held]])
    runtime=json.loads((root/"runtime.json").read_text())
    require(runtime["model_sha256"]==model_hash and runtime["repeats"]==3,"Runtime used wrong model")
    require(abs(runtime["runs"][0]["input_metadata"]["decoded_duration_seconds"]-10)<.05,"Benchmark is not 10 seconds")
    require(len({r["pid"] for r in runtime["runs"]})==3,"Benchmark did not use fresh processes")
    status=json.loads((root/"status.json").read_text())
    require(all(s["state"]=="complete" for s in status["stages"].values()),"Run has incomplete stages")
    completion_files={"common_folds":"common_folds/fold_plan.json","selection":"selection/selection_report.json",
                      "oof":"oof/oof_audit.json","in_domain_test":"in_domain_test/test_report.json",
                      "cross_test":"cross_test/test_report.json","benchmark":"runtime.json"}
    require(set(status["stages"])==set(completion_files),"Missing required run stage")
    require(all(file_sha256(root/completion_files[name])==s["completion_sha256"] for name,s in status["stages"].items()),"Stage completion artifact changed")
    require(all(s["started_at"]>freeze["frozen_at"] for name,s in status["stages"].items() if name in {"in_domain_test","cross_test"}),"Test ran before freeze")
    proof={"status":"passed","verified_at":datetime.now(timezone.utc).isoformat(),"plan_id":plan["plan_id"],
        "protocol":"vida_guard32k_v1","model_id":bundle["model_id"],"model_sha256":model_hash,
        "coverage":{"inputs_and_cache":32000,"real_fit":8400,"candidate_selection_val":1800,"threshold_real_calibration":900,
                    "oof":16800,"validation":3600,"in_domain_test":3600,"cross_test":8000},
        "real_only_scaler_numeric_check":True,"real_only_window_scaler_numeric_check":True,
        "all_final_scores_numerically_reproduced":True,"all_5_oof_fits_and_scores_numerically_reproduced":True,
        "common_fold_plan_id":common_plan["plan_id"],"test_started_after_freeze":True,"test_reports":reports,
        "original_real_parents_isolated":len(real_parent_scopes),"three_family_shared_prompts_isolated":len(prompt_scopes),
        "runtime_model_size_bytes":runtime["model_size_bytes"],"source_sha256":frozen["code_sha256"],
        "limitations":["A/B scores and A+B versus A+B+C paired ablation are external team dependencies",
                       "Source/group audit cannot rule out unknown semantic duplicates",
                       "This verifies this machine and dataset; it does not establish population-wide accuracy"]}
    write_json(root/"verification.json",proof)
    write_json("validation/formal_verification.json",proof|{"run_dir":str(root.resolve())})
    return proof


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",default="artifacts/vida_guard32k/run_02")
    p.add_argument("--cache-dir",default="artifacts/vida_guard32k/cache")
    p.add_argument("--manifest",default="data/VIDA-Guard32K/branch_c/manifest.csv")
    result=verify(p.parse_args())
    print(json.dumps({k:v for k,v in result.items() if k not in {"source_sha256","test_reports"}},indent=2))
