"""Score independent videos with the imported, frozen Branch B delivery.

No fitting, calibration or test-driven parameter changes occur here. JSONL is a
resumable inference cache, bound to the exact checkpoint and input manifest.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import uuid

from . import pipeline as p

_detector = None


def initialize(package):
    global _detector
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, package)
    import cv2
    cv2.setNumThreads(1)
    from inference import load_detector
    _detector = load_detector()


def score(task):
    clip_id, path = task
    result = _detector.infer_video(path)
    p.require(result["clip_id"] == clip_id, "Video filename differs from canonical clip_id")
    return {"clip_id": clip_id, "raw_margin": result["raw_margin"],
            "status": result["status"], "runtime_seconds": result["runtime_seconds"],
            "error": json.dumps(result.get("error", {}), ensure_ascii=False)}


def read_cache(path):
    """Recover a flushed JSONL, ignoring only an interrupted final write."""
    raw=path.read_bytes();rows=[];valid_bytes=0
    for number,line in enumerate(raw.splitlines(keepends=True)):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            p.require(number==len(raw.splitlines(keepends=True))-1 and not line.endswith(b"\n"),"Corrupt non-tail inference cache")
            break
        rows.append(row);valid_bytes+=len(line)
    if valid_bytes!=len(raw):
        with path.open("r+b") as stream:stream.truncate(valid_bytes)
    return rows


def worker(tasks_path, output_path):
    package=Path(__file__).resolve().parents[1]/"branches_dev/branch_b"
    initialize(str(package))
    tasks=json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    with Path(output_path).open("w",encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(score(task),ensure_ascii=False,allow_nan=False)+"\n");stream.flush()


def export(manifest_path, data_root, output, workers=2, limit=None):
    manifest_path=Path(manifest_path).resolve();data_root=Path(data_root).resolve();output=Path(output).resolve()
    package=Path(__file__).resolve().parents[1]/"branches_dev/branch_b"
    model=package/"model/branch_b_temporal_wavelet_full.joblib"
    p.require(model.exists(), "Branch B sealed delivery is missing")
    frame=p.table(manifest_path)
    p.require(frame.clip_id.is_unique, "Duplicate manifest IDs")
    if limit is not None:frame=frame.iloc[:limit]
    identity={"model_sha256": p.digest(model), "manifest_sha256":p.digest(manifest_path),
              "limit":limit, "score_column":"raw_margin", "branch_models_retrained":False,
              "runtime_packages":{name:importlib.metadata.version(name) for name in
                                  ("numpy","scikit-learn","joblib","lightgbm","PyWavelets","opencv-python-headless")}}
    output.parent.mkdir(parents=True,exist_ok=True)
    identity_path=output.with_suffix(".identity.json");cache=output.with_suffix(".progress.jsonl")
    if identity_path.exists():
        p.require(json.loads(identity_path.read_text(encoding="utf-8"))==identity,"Resume identity changed; choose a new output file")
    else:
        p.require(not output.exists() and not cache.exists(),"Unidentified existing export; choose a new output file")
        p.save_json(identity_path,identity)
    completed={};worker_root=output.with_suffix(".workers")
    # Independent worker subprocesses use files, not multiprocessing queues.
    # This avoids Windows semaphore handle failures in long media jobs.
    cached_files=([cache] if cache.exists() else [])+sorted(worker_root.glob("*/*.jsonl"))
    for cached in cached_files:
        for row in read_cache(cached):
            if row["clip_id"] in completed:
                p.require(completed[row["clip_id"]]==row,"Conflicting inference cache ID")
            completed[row["clip_id"]]=row
    p.require(set(completed).issubset(set(frame.clip_id)),"Cache has foreign IDs")
    tasks=[(r.clip_id,str(data_root/r.video_path)) for r in frame.itertuples() if r.clip_id not in completed]
    started=time.perf_counter()
    job=worker_root/uuid.uuid4().hex;job.mkdir(parents=True)
    processes=[];logs=[];parts=[]
    for i in range(min(workers,len(tasks))):
        taskfile=job/f"{i}.tasks.json";part=job/f"{i}.jsonl";parts.append(part)
        p.save_json(taskfile,tasks[i::workers])
        log=(job/f"{i}.log").open("w",encoding="utf-8");logs.append(log)
        env=os.environ.copy();env["OMP_NUM_THREADS"]="1";env["OPENBLAS_NUM_THREADS"]="1"
        processes.append(subprocess.Popen([sys.executable,"-m","stacking.export_b","_worker",str(taskfile),str(part)],
            cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0))
    previous=0
    try:
        while any(process.poll() is None for process in processes):
            count=len(completed)+sum(part.read_bytes().count(b"\n") for part in parts if part.exists())
            if count//100>previous//100:
                print(json.dumps({"completed":count,"total":len(frame)}),flush=True);previous=count
            time.sleep(2)
        p.require(all(process.returncode==0 for process in processes),"Inference worker failed; see worker logs and rerun to resume")
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:process.wait()
        for log in logs:log.close()
    for part in parts:
        for row in read_cache(part):
            p.require(row["clip_id"] not in completed,"Duplicate worker clip ID")
            completed[row["clip_id"]]=row
    p.require(set(completed)==set(frame.clip_id),"Missing or extra worker scores")
    rows=[completed[c] for c in frame.clip_id]
    with output.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["clip_id","raw_margin","status","runtime_seconds","error"])
        writer.writeheader();writer.writerows(rows)
    failed=[r for r in rows if r["status"]!="SUCCESS"]
    report={**identity,"clips":len(rows),"success":len(rows)-len(failed),"abstentions":failed,
            "cached_rows_reused":len(rows)-len(tasks),"rows_scored_this_resume":len(tasks),
            "wall_seconds_this_resume":time.perf_counter()-started,"workers":workers,"output":str(output),
            "limited_smoke_only":limit is not None,"not_a_hardware_comparable_latency_benchmark":True}
    p.save_json(output.with_suffix(".report.json"),report)
    return {k:v for k,v in report.items() if k!="abstentions"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="E:/VIDA-Guard32K/stacking_v2/manifests/manifest.csv")
    parser.add_argument("--data-root",default="E:/VIDA-Guard32K/stacking_v2")
    parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    p.require(args.workers>=1,"At least one worker required")
    p.require(args.limit is None or args.limit>0,"Positive smoke limit required")
    print(json.dumps(export(args.manifest,args.data_root,args.output,args.workers,args.limit),indent=2))


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="_worker":worker(sys.argv[2],sys.argv[3])
    else:main()
