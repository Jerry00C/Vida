"""Reproducible real-data Branch C run; resume only completed, unchanged stages."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from branch_c.benchmark import benchmark_video
from branch_c.cache import file_sha256
from branch_c.config import fingerprint, load_config, write_json
from branch_c.features import feature_signature
from branch_c.folds import prepare_folds
from branch_c.manifest import load_manifest
from branch_c.model import runtime_versions
from branch_c.pipeline import evaluate, make_oof
from branch_c.selection import predeclared_candidates, select_model
from dataclasses import asdict
import json


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_auxiliary_plans(root):
    """Freeze reporting references before any test; never retro-declare missing plans."""
    plans={
        "reference_baseline_plan.json":{"baseline":"B0_majority_real","training_label_counts":{"real":8400,"generated":8400},
            "tie_rule":"real","score":0.,"threshold":.5,"fit_media_read":0,
            "purpose":"Constant majority reference on identical frozen test clips, not an anomaly-model candidate"},
        "random_reference_plan.json":{"baseline":"Fully_random","seed":42,
            "rule":"numpy.default_rng(42).uniform(0,1) once for all 11600 test clip IDs sorted globally; strict score > 0.5 predicts generated",
            "fit_media_read":0,"labels_used_to_generate_scores":False,"purpose":"Proposal PPT slide12 chance reference; not model/threshold selection"},
        "case_sampling_plan.json":{"rule":"For each sealed test view: strongest FP and median-score FP if available; lowest-score FN for each generator, or lowest-score TP reference when no FN exists",
            "purpose":"Qualitative limitations, not model/threshold selection; eight distinct native frames shown at equal indices across each video",
            "annotations":"Only direct visible-frame observations; no claimed causal explanation or human full-video review"}}
    missing=[name for name in plans if not (root/name).exists()]
    tested=(root/"model_freeze.json").exists() or any((root/s/"test_report.json").exists() for s in ("in_domain_test","cross_test"))
    if missing and tested:raise ValueError("Auxiliary reference plans are missing after model freeze; do not retro-declare them")
    for name in missing:write_json(root/name,{"declared_at":now(),**plans[name]})


def run(args):
    manifest, config_path = Path(args.manifest).resolve(), Path(args.config).resolve()
    root, cache = Path(args.output_dir), Path(args.cache_dir)
    config = load_config(config_path)
    rows = load_manifest(manifest, config)
    code = {str(p): file_sha256(p) for p in sorted(Path("src/branch_c").glob("*.py"))}
    current = {"manifest_sha256": file_sha256(manifest), "config": config.to_dict(),
               "code_sha256": code, "feature_signature": feature_signature(config.features),
               "versions": runtime_versions()}
    plan_path = root / "protocol_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["frozen_inputs"] != current:
            raise ValueError("Frozen code/config/manifest changed; do not resume an existing experiment")
    else:
        root.mkdir(parents=True, exist_ok=True)
        plan = {"created_at": now(), "frozen_inputs": current, "run_kind": "research",
                "protocol": "vida_guard32k_v1", "candidates": [{"id":n,"model":asdict(c)} for n,c in predeclared_candidates(config.model.seed)],
                "selection": "1800 model_selection val only; AUROC then macro-F1 then predeclared order",
                "candidate_threshold": "95th percentile real model_selection val, strict >",
                "final_threshold": "95th percentile of 900 real calibration val, strict >; no generated calibration influence",
                "fit": "8400 real train only; scaler and detector fitted together; separate real-only window detector",
                "oof": "Provided train fold 0..4 unchanged; real outside held fold only; all 16800 train receive scores",
                "final_tests": ["in_domain_test", "cross_test"], "test_selection_policy": "Selected model only, unchanged features/config/threshold; no test-guided refit",
                "confidence_intervals": "1000 group bootstrap replicates, seed42, 95% percentile; main + per-generator",
                "benchmark": "3 fresh CPU processes, real train video of duration closest to 10s, no descriptor cache",
                "workers": args.workers, "scope": "Branch C only; A/B/fusion paired ablation requires teammates' outputs",
                "limitations": ["Curated VIDA-Guard32K, not official untouched GenVidBench",
                    "Cross-test changes both generator and real source; generator-only causal gain cannot be isolated",
                    "Native sampling and variable duration leave residual source confounds despite time-aware descriptors",
                    "Window scores are descriptive evidence, not validated forensic localisation or calibrated probability"]}
        plan["plan_id"] = fingerprint(plan)
        write_json(plan_path, plan)
    ensure_auxiliary_plans(root)
    status_path=root/"status.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {"plan_id":plan["plan_id"],"stages":{}}
    def stage(name, completed_file, action):
        file=root/completed_file
        if file.exists():
            print(f"Reusing completed {name}",flush=True)
            return
        status["current_stage"]=name
        status["updated_at"]=now()
        status["stages"][name]={"started_at":now(),"state":"running"}
        write_json(status_path,status)
        print(f"Starting {name} at {now()}",flush=True)
        started=time.perf_counter()
        action()
        if not file.is_file(): raise ValueError(f"Missing completion record: {file}")
        status["stages"][name]|={"finished_at":now(),"state":"complete","seconds":time.perf_counter()-started,
                                 "completion_sha256":file_sha256(file)}
        write_json(status_path,status)
        print(f"Completed {name}: {status['stages'][name]['seconds']:.1f}s",flush=True)
    if args.phase in {"all","development"}:
        stage("common_folds","common_folds/fold_plan.json",lambda: prepare_folds(manifest,config,root/"common_folds",5))
        stage("selection","selection/selection_report.json",lambda: select_model(manifest,config,root/"selection",cache,workers=args.workers))
    selected_path=root/"selection/selected_config.json"
    if args.phase in {"all","oof"}:
        selected=load_config(selected_path)
        stage("oof","oof/oof_audit.json",lambda: make_oof(root/"common_folds/manifest_with_folds.csv",selected,root/"oof",cache,5,workers=args.workers))
    model=root/"selection/selected/model.joblib"
    if args.phase in {"all","test"}:
        frozen=root/"model_freeze.json"
        if not frozen.exists():
            write_json(frozen,{"frozen_at":now(),"plan_id":plan["plan_id"],"model_sha256":file_sha256(model),
                "selected_config_sha256":file_sha256(selected_path),"oof_audit_sha256":file_sha256(root/"oof/oof_audit.json"),
                "test_metrics_read":False})
        frozen_model=json.loads(frozen.read_text())["model_sha256"]
        if file_sha256(model)!=frozen_model: raise ValueError("Frozen final model changed")
        for subset in plan["final_tests"]:
            stage(subset,f"{subset}/test_report.json",lambda subset=subset: evaluate(manifest,model,root/subset,cache,test_subset=subset,workers=args.workers))
    if args.phase in {"all","benchmark"}:
        real=[r for r in rows if r.split=="train" and r.label==0]
        video=min(real,key=lambda r:(abs(float(r.metadata["duration_seconds"])-10),r.clip_id))
        write_json(root/"benchmark_input.json",{"clip_id":video.clip_id,"path":str(video.path),"split":"train","duration_seconds":float(video.metadata["duration_seconds"]),"sha256":video.metadata["content_sha256"]})
        stage("benchmark","runtime.json",lambda: benchmark_video(model,video.path,root/"runtime.json",3,120))
    status["current_stage"]="requested_phases_complete"
    status["updated_at"]=now()
    write_json(status_path,status)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",default="data/VIDA-Guard32K/branch_c/manifest.csv")
    p.add_argument("--config",default="data/VIDA-Guard32K/branch_c/config.json")
    p.add_argument("--output-dir",default="artifacts/vida_guard32k/reproduction_run")
    p.add_argument("--cache-dir",default="artifacts/vida_guard32k/cache")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--phase",choices=["all","development","oof","test","benchmark"],default="all")
    run(p.parse_args())
