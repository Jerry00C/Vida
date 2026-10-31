"""Synthetic-only integration fixture; never a research result."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from . import pipeline as p

def fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    require_empty=not (root/"config.json").exists()
    p.require(require_empty,"Demo directory already initialized")
    base=pd.DataFrame([{"clip_id":"old_"+str(i),"group_id":"oldgroup_"+str(i),"sha256":hashlib.sha256(("old"+str(i)).encode()).hexdigest()} for i in range(16)])
    base.to_csv(root/"base.csv",index=False)
    rows=[]
    def add(split,family,label,n,real_pool=""):
        for _ in range(n):
            cid="synthetic_"+str(len(rows))
            rows.append({"clip_id":cid,"group_id":"synthetic_group_"+cid,"sha256":hashlib.sha256(cid.encode()).hexdigest(),
                "label":label,"family":family,"split":split,"protocol_id":"synthetic-demo-only","real_pool":real_pool,
                "video_path":"NO_REAL_VIDEO_SYNTHETIC_ONLY"})
    for split,n in [("fusion_train",6),("calibration",6),("in_domain_test",3)]:
        for family in p.KNOWN:add(split,family,1,n)
        add(split,"Vript",0,n*8)
    for family in p.EXTERNAL:add("external_test",family,1,4)
    add("external_test","Vript",0,12,"known_source")
    add("external_test","UCF101",0,12,"new_source")
    frame=pd.DataFrame(rows);frame.to_csv(root/"manifest.csv",index=False)
    p.save_json(root/"config.json",{"run_kind":"synthetic_demo","protocol_id":"synthetic-demo-only",
        "manifest":"manifest.csv","base_manifest":"base.csv","predictions_dir":"scores","output_dir":"run"})
    cfg=p.config(root/"config.json");rng=np.random.default_rng(42)
    for branch,alias in [("A","video_logit"),("B","raw_margin"),("C","branch_c_score")]:
        incoming=frame[["clip_id","label"]].sample(frac=1,random_state=ord(branch)).copy()
        incoming[alias]=incoming.label*2-1+rng.normal(0,1,len(incoming))
        path=root/(branch+"_native.csv");incoming.to_csv(path,index=False)
        p.import_scores(cfg,branch,path,"synthetic-checkpoint-"+branch)
    return cfg

def run(root):
    cfg=fixture(root);training=p.train(cfg);evaluation=p.evaluate(cfg,bootstrap=0)
    result={"run_kind":"synthetic_demo","not_research_result":True,"workflow":"import -> fit scalar stacker -> independent calibration -> freeze -> evaluate",
        "training":training,"evaluation":evaluation}
    p.save_json(Path(root)/"DEMO_RESULT.json",result)
    return result
