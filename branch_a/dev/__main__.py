import os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','10')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import copy
from pathlib import Path
from .common import check, read_json, save_json
from .data import audit, load_indexes
from .splits import build, load_split, smoke_subset


def download(config, cache, output):
    from huggingface_hub import HfApi, snapshot_download
    statuses={}
    for name,model in config['checkpoints'].items():
        try:
            info=HfApi().model_info(model,revision=config.get('revisions',{}).get(name,config['revision']))
            files={s.rfilename for s in info.siblings}
            weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
            snapshot_download(model,revision=info.sha,cache_dir=str(cache),
                              allow_patterns=['config.json','preprocessor_config.json',weight])
            statuses[name]={'status':'available','checkpoint':model,'revision':info.sha}
        except Exception as e:
            statuses[name]={'status':'blocked','checkpoint':model,'error':str(e),'exception':type(e).__name__}
        print(name,statuses[name]['status'],flush=True)
        save_json(output,statuses)
    return statuses


def smoke(args,config):
    import torch
    from .features import extract, FeatureCache
    from .learning import train, evaluate, load_checkpoint
    from .plots import comparison
    split=load_split(args.splits)
    videos,_=load_indexes(args.processed)
    smoke_path=args.output/'smoke_splits.json'
    small=smoke_subset(split,smoke_path,videos)
    results={}
    existing=args.output/'smoke_report.json'
    if existing.exists():
        for name,result in read_json(existing).items():
            prior=result.get('split_identity',result.get('test_metrics',{}).get('test_seen',{}).get('split_identity'))
            if prior==small['split_identity']:results[name]=result
    for name in args.backbones:
        try:
            path=extract(args.processed,small,name,config,args.output/'cache',args.hf_cache,
                         args.device,args.frame_batch_size,args.workers)
            cache=FeatureCache(path)
            cfg=copy.deepcopy(config['training']);cfg.update({'epochs':2,'batch_size':8})
            run=args.output/'smoke'/name
            control=run/'uninterrupted';resumed=run/'resumed'
            train(path,small,'round_1',cfg,control,resume=(control/'latest.pt').exists(),workers=args.workers)
            if not (resumed/'latest.pt').exists():train(path,small,'round_1',cfg,resumed,workers=args.workers,stop_after=1)
            train(path,small,'round_1',cfg,resumed,resume=True,workers=args.workers)
            cp1=load_checkpoint(control/'latest.pt',cache,small['split_identity'],'round_1')
            cp2=load_checkpoint(resumed/'latest.pt',cache,small['split_identity'],'round_1')
            check(cp1['history']==cp2['history'] and all(torch.equal(cp1['model'][k],cp2['model'][k]) for k in cp1['model']),
                  'Resumed training differs from uninterrupted training')
            evaluated=evaluate(path,small,'round_1',resumed/'best.pt',resumed/'evaluation',8)
            results[name]={'status':'passed','smoke_only':True,'cache':str(path),'embedding_dimension':cache.values.shape[-1],
                           'cache_stats':cache.meta['stats'],'resume_matches_uninterrupted':True,
                           'encoder_frozen_no_gradients':True,'test_metrics':evaluated}
        except Exception as e:
            results[name]={'status':'blocked_or_failed','exception':type(e).__name__,'error':str(e),
                           'checkpoint':config['checkpoints'][name],'smoke_only':True}
            print(f'{name}: {type(e).__name__}: {e}',flush=True)
        results[name]['split_identity']=small['split_identity']
        access=args.output/'model_access.json'
        if access.exists() and read_json(access).get(name,{}).get('status')=='blocked':
            results[name]['model_access_diagnostic']=read_json(access)[name]
        save_json(args.output/'smoke_report.json',results)
    comparison(args.output/'smoke',args.output/'smoke_comparison')
    print({k:v['status'] for k,v in results.items()},flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description='VIDA Branch A frozen linear probes')
    p.add_argument('--config',type=Path,default=Path('dev/config.json'))
    p.add_argument('--processed',type=Path,default=Path('processed'))
    p.add_argument('--output',type=Path,default=Path('runs/branch_a'))
    p.add_argument('--hf-cache',type=Path,default=Path('runs/branch_a/huggingface'))
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--workers',type=int,default=8)
    sub.add_parser('download')
    s=sub.add_parser('splits');s.add_argument('--audit',type=Path,default=Path('runs/branch_a/audit.json'))
    for command in ['smoke','features','train','train-all','evaluate']:
        c=sub.add_parser(command)
        c.add_argument('--splits',type=Path,default=Path('runs/branch_a/splits_v1.json'))
        c.add_argument('--workers',type=int,default=0)
        if command in ('smoke','features'):
            c.add_argument('--device',default='auto');c.add_argument('--frame-batch-size',type=int,default=8)
        if command=='smoke':c.add_argument('--backbones',nargs='+',choices=['clip','dinov2','dinov3'],default=['clip','dinov2','dinov3'])
        if command=='features':c.add_argument('--backbone',choices=['clip','dinov2','dinov3'],required=True)
        if command in ('train','evaluate'):
            c.add_argument('--cache',type=Path,required=True);c.add_argument('--round',default='round_1');c.add_argument('--run',type=Path,required=True)
        if command in ('train','train-all'):c.add_argument('--resume',action='store_true')
        if command=='train':c.add_argument('--init-checkpoint',type=Path)
        if command=='train-all':c.add_argument('--cache-root',type=Path,default=Path('runs/branch_a/cache'))
    c=sub.add_parser('compare');c.add_argument('--runs',type=Path,default=Path('runs/branch_a/training'))
    args=p.parse_args();config=read_json(args.config)
    # All Branch A writes must remain separate from authoritative processed data.
    check(not args.output.resolve().is_relative_to(args.processed.resolve()),'Branch A output must be outside processed data')
    if hasattr(args,'run'):check(not args.run.resolve().is_relative_to(args.processed.resolve()),'Run output cannot overwrite processed data')
    if args.command=='audit':audit(args.processed,args.output/'audit.json',args.workers)
    elif args.command=='download':download(config,args.hf_cache,args.output/'model_access.json')
    elif args.command=='splits':build(args.processed,args.audit,config,args.output/'splits_v1.json')
    elif args.command=='smoke':smoke(args,config)
    elif args.command=='compare':
        from .plots import comparison
        comparison(args.runs,args.output/'comparison')
    else:
        from .features import extract
        from .learning import train,evaluate
        split=load_split(args.splits)
        if args.command=='features':extract(args.processed,split,args.backbone,config,args.output/'cache',args.hf_cache,args.device,args.frame_batch_size,args.workers)
        elif args.command=='train':train(args.cache,split,args.round,config['training'],args.run,args.resume,args.workers,init_checkpoint=args.init_checkpoint)
        elif args.command=='evaluate':evaluate(args.cache,split,args.round,args.run/'best.pt',args.run/'evaluation',config['training']['batch_size'])
        else:
            for name in config['checkpoints']:
                for rid in split['rounds']:
                    train(args.cache_root/name,split,rid,config['training'],args.output/'training'/name/rid,args.resume,args.workers)

if __name__=='__main__':main()
