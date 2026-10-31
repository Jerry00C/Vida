"""Independently verify selected bytes, metadata and documented overlap keys."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re

from . import pipeline as p


def keys(row):
    family=row["family"];source=row["source_id"];media=row["media"]
    prefix=("vidprom" if family in {"Pika","VideoCrafter2","T2V-Zero","ModelScope"}
            else "hdvg" if family in {"SVD","MuseV","Mora","CogVideo","HD-VG-130M"}
            else "vript" if family=="Vript" else family.casefold()+"_unmapped")
    result={"group:"+prefix+":"+source,"sha:"+media["sha256"]}
    if row.get("group_id"):result.add("group:"+row["group_id"])
    if family in {"Vript","HD-VG-130M","SVD","MuseV","Mora","CogVideo"}:result.add("youtube:"+source)
    prompt=re.sub(r"\s+"," ",html.unescape(row.get("prompt") or "").casefold()).strip()
    if prompt:result.add("prompt:"+hashlib.sha256(prompt.encode()).hexdigest())
    if media.get("sampled_frame_gray_std",0)>=4:
        result.add("visual:"+media["duplicate_signature"]+":"+str(round(media["duration_seconds"],1)))
    return result


def verify(cfg, original_records):
    frame,manifest_sha,base_sha=p.manifest(cfg)
    root=Path(cfg["manifest"]).resolve().parent.parent
    def one(row):
        metadata=json.loads((root/row.metadata_path).read_text(encoding="utf-8"))
        path=root/row.video_path
        p.require(path.is_file(),"Missing video: "+str(path))
        p.require(path.stat().st_size==row.file_bytes,"Size mismatch: "+row.clip_id)
        p.require(p.digest(path)==row.sha256==metadata["media"]["sha256"],"SHA mismatch: "+row.clip_id)
        p.require(metadata["clip_id"]==row.clip_id and metadata["split"]==row.split,"Metadata identity mismatch")
        p.require(metadata["media"]["qc_status"]=="passed","Failed media QC in selected set")
        return metadata
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,frame.itertuples(index=False)))
    old=json.loads(Path(original_records).read_text(encoding="utf-8"))
    old_keys=set().union(*(keys(r) for r in old))
    ownership=defaultdict(set);byte_counts=Counter();visual_counts=Counter()
    vfr=Counter();codecs=Counter();unknown=Counter()
    for r in records:
        k=keys(r)
        p.require(not k&old_keys,"Overlap with original32K known identity/prompt/visual keys: "+r["clip_id"])
        for value in k:ownership[value].add(r["split"])
        byte_counts[r["media"]["sha256"]]+=1
        for value in k:
            if value.startswith("visual:"):visual_counts[value]+=1
        vfr[r["family"]]+=int(r["media"].get("variable_frame_rate",False))
        codecs[r["media"]["codec"]]+=1
        if not r.get("prompt"):unknown[r["family"]]+=1
    p.require(all(len(s)==1 for s in ownership.values()),"A known group/prompt/visual key crosses holdout splits")
    p.require(max(byte_counts.values())==1,"Byte duplicates selected")
    p.require(not visual_counts or max(visual_counts.values())==1,"Exact heuristic visual duplicates selected")
    report={"status":"verified","clips":len(records),"video_bytes":int(frame.file_bytes.sum()),
            "video_GB_decimal":float(frame.file_bytes.sum()/1e9),"video_GiB":float(frame.file_bytes.sum()/2**30),
            "manifest_sha256":manifest_sha,"base_manifest_sha256":base_sha,
            "verified_each_selected_file_sha256":True,"media_qc":"Previously decoded in full with native timestamp checks; this verification rehashes all selected bytes and checks passed QC metadata",
            "overlap_with_original_known_ids_groups_prompts_bytes_visual_keys":0,
            "cross_split_known_key_overlap":0,"duplicate_selected_bytes_or_exact_visual_keys":0,
            "variable_frame_rate_clips_by_family":dict(vfr),"video_codecs":dict(codecs),
            "prompt_missing_by_family":dict(unknown),"all_semantic_or_pretraining_overlap_absent_proven":False,
            "new_source_means":"Unseen in project-specific branch fitting, not guaranteed unseen by generic pretrained encoders"}
    p.save_json(root/"DATA_VERIFICATION.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.local.json")))
    parser.add_argument("--original-records",default="E:/VIDA-Guard32K/metadata/final_selected_records.json")
    args=parser.parse_args();print(json.dumps(verify(p.config(args.config),args.original_records),indent=2))
