"""Serial raw-video latency on one predefined fusion-training subset, never tests."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',required=True,choices=['A','B','C'])
    parser.add_argument('--root',required=True);args=parser.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root=Path(args.root);repo=Path(__file__).resolve().parents[1]
    rows=list(csv.DictReader((root/'manifests/fusion_train.csv').open(encoding='utf-8-sig')))
    # Hash-order within each source. 3 each of 8 generators + 23 Vript + 1 HD.
    import hashlib
    selected=[]
    for family in sorted({r['family'] for r in rows}):
        count=23 if family=='Vript' else 1 if family=='HD-VG-130M' else 3
        chosen=sorted([r for r in rows if r['family']==family],key=lambda r:hashlib.sha256(('latency42:'+r['clip_id']).encode()).hexdigest())[:count]
        selected.extend(chosen)
    assert len(selected)==48
    sync=lambda:None
    if args.branch=='A':
        import av
        import torch
        from transformers import CLIPVisionModel,CLIPImageProcessor
        from branch_a.dev.adapters import AdaptedVision
        torch.set_num_threads(1)
        cp=torch.load(root/'models/branch_a/partial_clip_round4/best.pt',map_location='cpu',weights_only=False)
        spec=cp['spec'];encoder=CLIPVisionModel.from_pretrained(root/'models/branch_a/clip_base_57c2164',local_files_only=True,attn_implementation='eager')
        model=AdaptedVision(encoder,'clip','partial',spec['adaptation']);model.load_adapted_state(cp['adapted_state'])
        model.requires_grad_(False);model.to('cuda');model.eval();processor=CLIPImageProcessor(**spec['processor'])
        sync=torch.cuda.synchronize
        def infer(row):
            n=int(row['frame_count']);indices=[k*(n-1)//7 for k in range(8)];images=[]
            with av.open(str(root/row['video_path'])) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                for i,frame in enumerate(container.decode(stream)):
                    if i in indices:images.append(frame.to_image().convert('RGB'))
            assert len(images)==8
            inputs={k:v.to('cuda') for k,v in processor(images=images,return_tensors='pt').items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(inputs)
            return float(out.float().mean().cpu())
        runtime={'torch':str(torch.__version__),'device':torch.cuda.get_device_name(),'precision':'bf16','frames':8}
    elif args.branch=='B':
        import cv2
        cv2.setNumThreads(1);sys.path.insert(0,str(repo/'deliverables/branch_b_temporal_wavelet'))
        from inference import load_detector
        model=load_detector()
        def infer(row):
            out=model.infer_video(str(root/row['video_path']));assert out['status']=='SUCCESS',out
            return float(out['raw_margin'])
        runtime={'device':'CPU','frames':'native B at most 64','opencv':cv2.__version__}
    else:
        import cv2
        cv2.setNumThreads(1);sys.path.insert(0,str(repo/'dev/branch_c/src'))
        from branch_c.model import load_bundle,runtime_versions
        from branch_c.video import read_video
        from branch_c.features import extract_features
        model=load_bundle(root/'models/branch_c/extracted/branch_c_trained_artifacts/rotation_6seen_2unseen_v1/shared_models/svm_nu_0.01/model.joblib')
        def infer(row):
            video=read_video(root/row['video_path'],model['config'].features)
            feature=extract_features(video,model['config'].features)
            return float(model['clip_model'].score(feature.clip[None])[0])
        runtime=runtime_versions()|{'device':'CPU','frames':'saved native-stride configuration; no window explanations'}
    for row in selected[:3]:infer(row);sync()
    measured=[]
    for row in selected:
        sync();started=time.perf_counter();score=infer(row);sync()
        measured.append({'clip_id':row['clip_id'],'family':row['family'],'seconds':time.perf_counter()-started,'score':score})
    times=np.array([r['seconds'] for r in measured])
    report={'branch':args.branch,'rows':48,'warmup_clips':3,'batch_size_videos':1,
        'median_ms':float(np.median(times)*1000),'p95_ms':float(np.quantile(times,.95)*1000),'mean_ms':float(times.mean()*1000),
        'runtime':runtime,'scope':'Serial raw-video decode, native per-branch preprocessing and clip score. Excludes imports/model load, file hash verification, disk output, and C window explanations. A runs on GPU; B/C run on same-PC CPU. Warm disk cache; not equal-compute algorithm benchmark.',
        'cpu_thread_limit':1,'results':measured}
    folder=root/'runs/fusion_v1/report/latency';folder.mkdir(parents=True,exist_ok=True)
    (folder/(args.branch+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()
