"""Group-isolated, exhaustive six-seen/two-unseen protocol for the delivered cohort.

This module never changes the original delivery or the frozen v1 detector code.
Partitions and sampling use IDs/groups only, never detector scores or quality ranks.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from branch_c.cache import file_sha256
from branch_c.config import Config, FeatureConfig, fingerprint, write_json
from branch_c.manifest import Clip, connected_groups, load_manifest

PROTOCOL = "vida_guard32k_rotation_6seen_2unseen_v1"
GENERATORS = ("CogVideo", "Mora", "MuseV", "OpenSora", "Pika", "SVD", "T2V-Zero", "VideoCrafter2")
REAL_SOURCES = ("Vript", "HD-VG-130M")
PAIRED = ("CogVideo", "Mora", "MuseV", "SVD")
SEED = 42


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records and fields is None:
        raise ValueError("Empty CSV needs explicit fields")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        w.writeheader()
        w.writerows(records)


def ordered(values, purpose):
    return sorted(values, key=lambda s: (hashlib.sha256(f"{SEED}|{purpose}|{s}".encode()).hexdigest(), s))


def partition_groups(strata):
    """60/10/10/20 group partitions, with deterministic balanced train folds."""
    result = {}
    for signature, ids in sorted(strata.items()):
        ids = ordered(ids, f"partition:{signature}")
        if len(ids) % 50:
            raise ValueError("Formal strata must support exact group partitions and five folds")
        n = len(ids)
        for i, group in enumerate(ids):
            role = "train" if i < .6*n else "model_selection" if i < .7*n else "calibration" if i < .8*n else "test"
            result[group] = (role, str(i % 5) if role == "train" else "")
    return result


def take_whole(groups, ids, n_clips, family, purpose):
    """Choose complete source groups; reject any quota that would split a parent."""
    chosen, count = set(), 0
    for group in ordered(ids, purpose):
        members = [r for r in groups[group] if r.source == family]
        if count + len(members) > n_clips:
            continue
        chosen.add(group)
        count += len(members)
        if count == n_clips:
            break
    if count != n_clips:
        raise ValueError(f"Cannot fill whole-group quota: {family} {n_clips}")
    return chosen


def round_records(master, heldout, round_id):
    heldout = set(heldout)
    records, excluded = [], []
    for r in master:
        selected = r["rotation_eligible"] == "true"
        is_held = r["generator"] in heldout
        if selected and r["split"] in {"train", "val"} and is_held:
            selected = False
        if not selected:
            excluded.append({"clip_id": r["clip_id"], "reason": "heldout_generator_quarantined_from_development" if is_held and r["split"] != "test" else "outside_predeclared_balanced_quota"})
            continue
        item = r.copy()
        item["protocol"] = f"{PROTOCOL}:{round_id}"
        if item["split"] == "test" and item["label"] == "1":
            item["evaluation_subset"] = "unseen_generators" if is_held else "seen_generators"
        records.append(item)
    return records, excluded


def check_round(records, heldout):
    heldout = set(heldout)
    seen = {r["generator"] for r in records if r["label"] == "1" and r["split"] == "train"}
    if seen != set(GENERATORS) - heldout or len(seen) != 6 or len(heldout) != 2:
        raise ValueError("Every round must contain exactly six seen and two unseen generators")
    scopes, folds = {}, {}
    for r in records:
        if r["generator"] in heldout and r["split"] != "test":
            raise ValueError("Heldout generator entered development")
        scope = (r["split"], r["validation_role"])
        if r["group_id"] in scopes and scopes[r["group_id"]] != scope:
            raise ValueError("Related group crosses partition or validation role")
        scopes[r["group_id"]] = scope
        if r["split"] == "train":
            if r["group_id"] in folds and folds[r["group_id"]] != r["fold_id"]:
                raise ValueError("Related group crosses OOF fold")
            folds[r["group_id"]] = r["fold_id"]
    expected = {("val", "model_selection", "0"): 600, ("val", "model_selection", "1"): 600,
                ("val", "calibration", "0"): 600, ("val", "calibration", "1"): 600}
    actual = Counter((r["split"], r["validation_role"], r["label"]) for r in records if r["split"] == "val")
    if actual != expected:
        raise ValueError("Incorrect independent balanced validation quotas")
    for subset, n in [("unseen_generators", 400), ("seen_generators", 1200)]:
        rows = [r for r in records if r["evaluation_subset"] == subset]
        if Counter(r["label"] for r in rows) != {"0": n, "1": n}:
            raise ValueError("Test view is not class balanced")
        if Counter(r["source"] for r in rows if r["label"] == "0") != {s: n//2 for s in REAL_SOURCES}:
            raise ValueError("Test real-source mixture changed")
        if set(Counter(r["generator"] for r in rows if r["label"] == "1").values()) != {200}:
            raise ValueError("Test generator quota changed")
    return {"seen": sorted(seen), "heldout": sorted(heldout), "counts": dict(Counter(r["split"] for r in records)),
            "real_train": sum(r["split"] == "train" and r["label"] == "0" for r in records)}


def build_protocol(manifest, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Protocol destination already exists; preserve previous experiments")
    original = load_manifest(manifest)
    if len(original) != 32000 or Counter(r.source for r in original) != {
        "Vript":12000,"HD-VG-130M":4000,"Pika":3000,"VideoCrafter2":3000,"T2V-Zero":3000,
        "OpenSora":3000,"CogVideo":1000,"Mora":1000,"MuseV":1000,"SVD":1000}:
        raise ValueError("Unexpected cohort; do not silently change this protocol")
    hashes = [r.metadata["content_sha256"] for r in original]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Byte duplicates require a revised group/quota protocol")
    components = connected_groups(original, hashes)
    groups = defaultdict(list)
    for r, component in zip(original, components):
        groups[str(component)].append(r)
    # Stable group IDs, not input row positions, drive all choices.
    groups = {"|".join(sorted({r.group_id for r in rows})): rows for rows in groups.values()}
    strata = defaultdict(list)
    for group, rows in groups.items():
        signature = "+".join(f"{s}:{n}" for s,n in sorted(Counter(r.source for r in rows).items()))
        strata[signature].append(group)
    partitions = partition_groups(strata)
    by_clip = {r.clip_id:g for g,rows in groups.items() for r in rows}
    eligible = set()
    real_view = {}
    for group, (role, _) in partitions.items():
        if role == "train":
            eligible.update(r.clip_id for r in groups[group])
    for role in ("model_selection", "calibration", "test"):
        available = {g for g,p in partitions.items() if p[0] == role}
        # Shared VidProM UUIDs use exactly the same selected group quota.
        for family in GENERATORS:
            ids = {g for g in available if any(r.source == family for r in groups[g])}
            purpose = f"quota:{role}:" + ("vidprom" if family in {"Pika","T2V-Zero","VideoCrafter2"} else family)
            selected = take_whole(groups, ids, 200 if role == "test" else 100, family, purpose)
            eligible.update(r.clip_id for g in selected for r in groups[g] if r.source == family)
        vript = {g for g in available if all(r.source == "Vript" for r in groups[g])}
        if role != "test":
            selected = take_whole(groups,vript,300,"Vript",f"real:{role}:Vript")
            eligible.update(r.clip_id for g in selected for r in groups[g])
            for family in PAIRED:
                ids = {g for g in available if any(r.source == family for r in groups[g])}
                selected = take_whole(groups,ids,75,"HD-VG-130M",f"real:{role}:{family}")
                eligible.update(r.clip_id for g in selected for r in groups[g] if r.label == 0)
        else:
            held = take_whole(groups,vript,200,"Vript","real:test:Vript:unseen")
            seen = take_whole(groups,vript-held,600,"Vript","real:test:Vript:seen")
            for subset, selected in [("unseen_generators",held),("seen_generators",seen)]:
                for g in selected:
                    for r in groups[g]: eligible.add(r.clip_id); real_view[r.clip_id]=subset
            for family in PAIRED:
                ids = {g for g in available if any(r.source == family for r in groups[g])}
                held = take_whole(groups,ids,50,"HD-VG-130M",f"real:test:{family}:unseen")
                for subset, selected in [("unseen_generators",held),("seen_generators",ids-held)]:
                    for g in selected:
                        for r in groups[g]:
                            if r.label == 0: eligible.add(r.clip_id);real_view[r.clip_id]=subset
    master = []
    for r in original:
        role, fold = partitions[by_clip[r.clip_id]]
        record = r.metadata.copy()
        record.update(original_split=r.split,original_fold_id=r.metadata.get("fold_id",""),
                      original_validation_role=r.metadata.get("validation_role",""),original_protocol=r.metadata.get("protocol",""),
                      split="train" if role=="train" else "test" if role=="test" else "val",
                      validation_role=role if role in {"model_selection","calibration"} else "",fold_id=fold,fold=fold,
                      protocol=PROTOCOL,path=str(r.path),rotation_group=by_clip[r.clip_id],base_partition=role,
                      rotation_eligible="true" if r.clip_id in eligible else "false",
                      evaluation_subset=real_view.get(r.clip_id,"") if role=="test" else "")
        master.append(record)
    destination.mkdir(parents=True)
    write_csv(destination/"master_manifest.csv",master)
    rounds=[]
    for i, pair in enumerate(combinations(GENERATORS,2),1):
        name=f"round_{i:02d}"
        records,excluded=round_records(master,pair,name)
        audit=check_round(records,pair)
        directory=destination/name
        write_csv(directory/"manifest.csv",records)
        write_csv(directory/"excluded.csv",excluded)
        config=replace(Config(),features=FeatureConfig(min_duration_seconds=.5,sampling_mode="native_stride"),heldout_generators=pair)
        write_json(directory/"config.json",config.to_dict())
        load_manifest(directory/"manifest.csv",config)
        for split in ("train","val","test"):
            write_csv(directory/f"{split}.csv",[r for r in records if r["split"]==split])
        write_csv(directory/"fold_assignments.csv",[{k:r[k] for k in ["clip_id","label","group_id","fold_id","content_sha256","protocol"]} for r in records if r["split"]=="train"])
        write_json(directory/"split_audit.json",audit)
        rounds.append({"round_id":name,**audit,"manifest_sha256":file_sha256(directory/"manifest.csv")})
    plan={"declared_at":now(),"protocol":PROTOCOL,"seed":SEED,"rounds":rounds,"n_rounds":28,
          "source_manifest_sha256":file_sha256(manifest),"master_manifest_sha256":file_sha256(destination/"master_manifest.csv"),
          "base_partition":"source/prompt connected groups, stratified by family signature; 60% train,10% model_selection,10% calibration,20% test",
          "validation":"600 real (300/source) + 100/generated-family for each of six seen families in each validation role; final threshold only600realcal",
          "test":"200 fixed clips/family; unseen2:400gen+400real, seen6:1200gen+1200real; each real pool50/50Vript/HDVG and disjoint",
          "excluded":"Unseen generator development rows quarantined; unused quota rows recorded; no video relabelled or copied",
          "selection":"Same six predeclared IF/OCSVM candidates; AUROC,macroF1,predeclared order; each round uses only its seen-family model_selection",
          "reuse":"Only identical real-fit IDs+hashes, feature signature and estimator parameters permit model reuse; generated data never fitC",
          "limitations":["Repartitioned analysis of an already evaluated cohort, not a fresh blind test",
                         "Generator families share architecture/conditioning; Mora includesSVD; two namedI2V versus sixT2V is not independently certified",
                         "Duration/fps/content/source confounds and unknown semantic duplicates remain",
                         "28 overlapping rounds are dependent; mean/range are descriptive, no iid28-round confidence interval",
                         "A/B/fusion outputs require respective teammates; no fusion gain claimed"]}
    plan["plan_id"]=fingerprint(plan)
    write_json(destination/"protocol_plan.json",plan)
    return plan


def fast_group_bootstrap(labels,scores,groups,threshold,repeats=1000,seed=42):
    """Exact multiplicity-weighted equivalent of resampling whole groups.

    Tie-aware weighted Mann–Whitney AUROC and binary confusion-based macro F1.
    Batches bound memory; RNG draws have the same order as the reference loop.
    """
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float)
    if (y.ndim!=1 or s.shape!=y.shape or np.asarray(groups).shape!=y.shape
            or set(np.unique(y))!={0,1} or not np.isfinite(s).all() or repeats<1):
        raise ValueError("Bootstrap requires aligned finite scores, both labels and positive repeats")
    unique,membership=np.unique(groups,return_inverse=True)
    order=np.argsort(s,kind="stable");starts=np.r_[0,np.flatnonzero(np.diff(s[order]))+1]
    rng=np.random.default_rng(seed);aucs=[];f1s=[]
    pred=s>threshold
    for begin in range(0,repeats,100):
        size=min(100,repeats-begin)
        draws=rng.integers(len(unique),size=(size,len(unique)))
        offsets=np.arange(size)[:,None]*len(unique)
        multiplicity=np.bincount((draws+offsets).ravel(),minlength=size*len(unique)).reshape(size,-1)
        w=multiplicity[:,membership].astype(float)
        negatives=np.add.reduceat((w*(y==0))[:,order],starts,axis=1)
        positives=np.add.reduceat((w*(y==1))[:,order],starts,axis=1)
        n0=negatives.sum(axis=1);n1=positives.sum(axis=1);valid=(n0>0)&(n1>0)
        numerator=(positives*(np.cumsum(negatives,axis=1)-.5*negatives)).sum(axis=1)
        aucs.extend((numerator[valid]/(n0[valid]*n1[valid])).tolist())
        tn=w[:,(y==0)&~pred].sum(axis=1);fp=w[:,(y==0)&pred].sum(axis=1)
        fn=w[:,(y==1)&~pred].sum(axis=1);tp=w[:,(y==1)&pred].sum(axis=1)
        f1s.extend((.5*(2*tn[valid]/(2*tn[valid]+fp[valid]+fn[valid])+2*tp[valid]/(2*tp[valid]+fp[valid]+fn[valid]))).tolist())
    return {"method":"group_bootstrap_multiplicity_weighted_exact","requested_repeats":repeats,"valid_repeats":len(aucs),
            "n_groups":len(unique),"auroc":np.quantile(aucs,[.025,.975]).tolist(),"macro_f1":np.quantile(f1s,[.025,.975]).tolist()}
