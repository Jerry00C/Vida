"""Infer a fixed Partial CLIP checkpoint with its original eight-frame policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--base',required=True);parser.add_argument('--manifest')
    parser.add_argument('--output');parser.add_argument('--limit',type=int)
    parser.add_argument('--device',default='cuda');parser.add_argument('--frame-batch',type=int,default=16)
    parser.add_argument('--decode-workers',type=int,default=2)
    args=parser.parse_args()
    import av
    import numpy as np
    import torch
    from transformers import CLIPVisionModel,CLIPImageProcessor
    from branch_a.dev.adapters import AdaptedVision
    from branch_a.dev.common import identity
    torch.set_num_threads(4);torch.manual_seed(42)
    root=Path(args.root);checkpoint=Path(args.checkpoint);base=Path(args.base)
    manifest=Path(args.manifest) if args.manifest else root/'manifests/manifest.csv'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig',newline='')))
    if args.limit:rows=rows[:args.limit]
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert identity(cp['spec'])==cp['experiment_identity'],'Corrupt checkpoint experiment identity'
    spec=cp['spec'];assert spec['backbone']=='clip' and spec['mode']=='partial'
    assert sha(base/'pytorch_model.bin')=='ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0'
    encoder=CLIPVisionModel.from_pretrained(base,local_files_only=True,attn_implementation='eager')
    model=AdaptedVision(encoder,'clip','partial',spec['adaptation'])
    assert model.structure==spec['structure'],'Architecture differs from checkpoint'
    model.load_adapted_state(cp['adapted_state'])
    model.requires_grad_(False);model.to(args.device);model.eval()
    processor=CLIPImageProcessor(**spec['processor'])
    precision='bf16' if args.device.startswith('cuda') else 'fp32'
    if precision=='bf16':assert torch.cuda.is_bf16_supported()
    folder=Path(args.output) if args.output else root/'branch_scores/raw/A_partial_clip_round4'
    folder.mkdir(parents=True,exist_ok=True)
    state={'manifest_sha256':sha(manifest),'checkpoint_sha256':sha(checkpoint),
        'base_revision':spec['revision'],'base_sha256':sha(base/'pytorch_model.bin'),
        'device':args.device,'precision':precision,'frame_batch':args.frame_batch,
        'torch':str(torch.__version__),'pyav':av.__version__,
        'sampling':'decoded-index-floor-v1: k*(N-1)//7, k=0..7',
        'aggregation':'arithmetic mean of eight frame logits; higher is generated',
        'model_retrained':False,'limit':args.limit}
    lock=folder/'IDENTITY.json'
    if lock.exists():assert json.loads(lock.read_text())==state,'Resume identity differs'
    else:lock.write_text(json.dumps(state,indent=2)+'\n')
    log=folder/'scores.jsonl';done={}
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line);assert r['clip_id'] not in done;done[r['clip_id']]=r
    def decode(row):
        path=root/row['video_path'];assert sha(path)==row['sha256']
        n=int(row['frame_count']);assert n>=8
        indices=[k*(n-1)//7 for k in range(8)];images=[];seen=0
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            for i,frame in enumerate(container.decode(stream)):
                assert not frame.is_corrupt
                if i in indices:images.append(frame.to_image().convert('RGB'))
                seen+=1
        assert seen==n and len(images)==8,f'Frame count mismatch: {row["clip_id"]}'
        return images
    started=time.perf_counter();missing=[r for r in rows if r['clip_id'] not in done]
    with log.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(args.decode_workers) as pool,torch.inference_mode():
        for first in range(0,len(missing),args.decode_workers):
            batch=missing[first:first+args.decode_workers];t=time.perf_counter()
            clips=list(pool.map(decode,batch));images=[im for clip in clips for im in clip]
            values=[]
            for offset in range(0,len(images),args.frame_batch):
                inputs={k:v.to(args.device) for k,v in processor(images=images[offset:offset+args.frame_batch],return_tensors='pt').items()}
                if precision=='bf16':
                    with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(inputs)
                else:logits=model(inputs)
                values.extend(logits.float().cpu().tolist())
            values=np.asarray(values,dtype=np.float64).reshape(len(batch),8);assert np.isfinite(values).all()
            for row,frame_logits in zip(batch,values):
                result={'clip_id':row['clip_id'],'video_logit':float(frame_logits.mean()),
                    'frame_logits':frame_logits.tolist(),'status':'SUCCESS','sha256':row['sha256'],
                    'checkpoint_sha256':state['checkpoint_sha256']}
                stream.write(json.dumps(result)+'\n');stream.flush();done[row['clip_id']]=result
            if len(done)%100==0 or len(done)==len(rows):print(f'A: {len(done)}/{len(rows)}, {time.perf_counter()-started:.1f}s this invocation',flush=True)
    target=folder/'A_native.csv';fields=['clip_id','video_logit','status','sha256','checkpoint_sha256']
    with target.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(done[r['clip_id']] for r in rows)
    report=state|{'rows':len(rows),'cached_rows':len(rows)-len(missing),'wall_seconds_this_invocation':time.perf_counter()-started,
        'csv_sha256':sha(target),'GPU':torch.cuda.get_device_name() if args.device.startswith('cuda') else None}
    (folder/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
