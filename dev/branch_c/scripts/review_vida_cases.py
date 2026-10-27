"""Export deterministic post-test case sheets; observations must come from inspection."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

from branch_c.config import write_json
from branch_c.model import load_bundle
from branch_c.pipeline import predict_video


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def sheet(path, destination):
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    try:
        if not cap.isOpened(): raise ValueError(f"Cannot inspect {path}")
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        selected=np.unique(np.linspace(0,count-1,8).astype(int))
        canvas=np.full((2*210,4*320,3),245,np.uint8)
        index=0;output_index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if output_index<len(selected) and index==selected[output_index]:
                h,w=frame.shape[:2];factor=min(316/w,176/h)
                resized=cv2.resize(frame,(max(1,round(w*factor)),max(1,round(h*factor))),interpolation=cv2.INTER_AREA)
                y=(output_index//4)*210;x=(output_index%4)*320
                dy=y+2+(176-resized.shape[0])//2;dx=x+2+(316-resized.shape[1])//2
                canvas[dy:dy+resized.shape[0],dx:dx+resized.shape[1]]=resized
                cv2.putText(canvas,f"t={index/fps:.3f}s  native frame {index}",(x+7,y+199),cv2.FONT_HERSHEY_SIMPLEX,.43,(40,40,40),1,cv2.LINE_AA)
                output_index+=1
            index+=1
        if index!=count or output_index!=len(selected): raise ValueError("Incomplete case decoding")
        if not cv2.imwrite(str(destination),canvas): raise ValueError("Could not save case sheet")
    finally:cap.release()


def export(args):
    root=Path(args.run_dir);out=root/"report";out.mkdir(exist_ok=True)
    if (out/"cases.json").exists(): raise ValueError("Cases already exist; preserve inspection observations")
    partial=out/"cases_in_domain_test.json"
    if args.subset=="in_domain_test":
        if partial.exists():raise ValueError("In-domain cases already exist; preserve observations")
        cases=[];subsets=("in_domain_test",);destination=partial
    elif partial.exists():
        cases=json.loads(partial.read_text());subsets=("cross_test",);destination=out/"cases.json"
    else:
        cases=[];subsets=("in_domain_test","cross_test");destination=out/"cases.json"
    model=root/"selection/selected/model.joblib";bundle=load_bundle(model)
    manifest={r["clip_id"]:r for r in read_csv(args.manifest)}
    for subset in subsets:
        records=read_csv(root/subset/"test_scores.csv")
        fp=sorted([r for r in records if r["label"]=="0" and float(r["branch_c_score"])>bundle["threshold"]],key=lambda r:(-float(r["branch_c_score"]),r["clip_id"]))
        selected=[]
        if fp:
            selected.append((fp[0],"FP"))
            if len(fp)>1:selected.append((fp[len(fp)//2],"FP"))
        for generator in sorted({r["generator"] for r in records if r["label"]=="1"}):
            generated=[r for r in records if r["generator"]==generator]
            fn=[r for r in generated if float(r["branch_c_score"])<=bundle["threshold"]]
            chosen=min(fn or generated,key=lambda r:(float(r["branch_c_score"]),r["clip_id"]))
            selected.append((chosen,"FN" if fn else "TP_reference_no_FN"))
        for r,error in selected:
            case_id=f"case_{len(cases)+1:02d}"
            data=manifest[r["clip_id"]]
            image=f"{case_id}.png"
            sheet(data["path"],out/image)
            prediction=predict_video(model,data["path"],out/f"{case_id}_prediction.json")
            if not np.isclose(prediction["anomaly_score"],float(r["branch_c_score"]),rtol=0,atol=1e-10):
                raise ValueError("Post-test inference differs from frozen exported score")
            cases.append({"case_id":case_id,"subset":subset,"clip_id":r["clip_id"],"family":data["family"],
                          "error_type":error,"score":float(r["branch_c_score"]),"threshold":bundle["threshold"],
                          "image":image,"path":data["path"],"content_sha256":r["content_sha256"],
                          "duration_seconds":float(data["duration_seconds"]),"native_fps":float(data["native_fps"]),
                          "descriptor_deviations":prediction["descriptor_deviations"],
                          "review_status":"pending_visual_inspection","observation":""})
    write_json(destination,cases)
    print(f"Exported {len(cases)} cases; model/threshold unchanged")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",default="artifacts/vida_guard32k/run_02")
    p.add_argument("--manifest",default="data/VIDA-Guard32K/branch_c/manifest.csv")
    p.add_argument("--subset",choices=["all","in_domain_test"],default="all",
                   help="Inspect the completed in-domain view early; all later preserves those observations")
    export(p.parse_args())
