"""Run image-based partial fine-tuning and LoRA without modifying frozen runs."""
import argparse
import copy
from pathlib import Path
import torch
from .common import check,read_json,save_json
from .splits import load_split
from .adapters import build_model
from .image_learning import train_images,evaluate_images,load_adapted_checkpoint


def verify_resume(first,second):
    a=load_adapted_checkpoint(Path(first)/'latest.pt');b=load_adapted_checkpoint(Path(second)/'latest.pt')
    check(a['experiment_identity']==b['experiment_identity'],'Resume experiment mismatch')
    check(a['history']==b['history'] and a['global_step']==b['global_step'],'Resume history/step mismatch')
    differences={f'{s}.{n}':float((p-b['adapted_state'][s][n]).abs().max())
                 for s,ps in a['adapted_state'].items() for n,p in ps.items()}
    check(max(differences.values())==0.,'Resume weights differ from uninterrupted training')
    result={'passed':True,'max_parameter_difference':max(differences.values()),'global_step':a['global_step'],'epochs':a['epoch']}
    save_json(Path(second).parent/'resume_verification.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inspect','smoke','train','train-all','evaluate','compare'])
    p.add_argument('--mode',choices=['frozen','partial','lora'],default='partial')
    p.add_argument('--backbones',nargs='+',choices=['clip','dinov2','dinov3'],default=['clip','dinov2','dinov3'])
    p.add_argument('--config',default=str(Path(__file__).with_name('adaptation_config.json')))
    p.add_argument('--unfreeze-blocks',type=int);p.add_argument('--rank',type=int);p.add_argument('--alpha',type=int);p.add_argument('--dropout',type=float)
    p.add_argument('--cache-root',default='runs/branch_a/cache')
    p.add_argument('--root',default='processed');p.add_argument('--output',default='runs/branch_a_adapt')
    p.add_argument('--split');p.add_argument('--round',default='round_1');p.add_argument('--checkpoint')
    p.add_argument('--hf-cache',default='runs/branch_a/huggingface');p.add_argument('--device',default='auto')
    p.add_argument('--workers',type=int,default=0);p.add_argument('--resume',action='store_true')
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'])
    for name in ('epochs','batch-size','accumulation','frame-microbatch','seed'):
        p.add_argument('--'+name,type=int)
    for name in ('partial-lr','lora-lr','head-lr'):p.add_argument('--'+name,type=float)
    a=p.parse_args();out=Path(a.output)
    for protected in (Path('runs/branch_a'),Path(a.root)):
        check(not out.resolve().is_relative_to(protected.resolve()),'Adaptation output must be outside frozen runs and processed data')
    if a.command=='compare':
        from .adapt_comparison import compare
        compare(out);return
    base=read_json(Path(__file__).with_name('config.json'))
    defaults=read_json(a.config);adaptation=defaults['adaptation'];config=copy.deepcopy(defaults['training'])
    for key in adaptation:
        value=getattr(a,key,None)
        if value is not None:adaptation[key]=value
    for key in config:
        value=getattr(a,key,None)
        if value is not None:config[key]=value
    if a.command=='inspect':
        for backbone in a.backbones:
            model,_=build_model(backbone,a.mode,base,adaptation,a.hf_cache,torch.device('cpu'))
            save_json(out/'inspection'/f'{a.mode}_{backbone}.json',model.report());print(model.report());del model
        return
    split=load_split(a.split or ('runs/branch_a/smoke_splits.json' if a.command=='smoke' else 'runs/branch_a/splits_v1.json'))
    if a.mode=='frozen':
        from .learning import train,evaluate
        check(a.command!='evaluate' or len(a.backbones)==1,'Frozen evaluation requires exactly one --backbones entry')
        frozen_config=copy.deepcopy(base['training'])
        for key in frozen_config:
            value=getattr(a,key,None)
            if value is not None:frozen_config[key]=value
        check(a.command!='smoke','Existing frozen smoke results are preserved. Use the original frozen CLI for a new baseline run.')
        for backbone in a.backbones:
            for rid in (split['rounds'] if a.command=='train-all' else [a.round]):
                run=out/'training'/'frozen'/backbone/rid
                cache=Path(a.cache_root)/backbone
                if a.command=='evaluate':
                    check(a.checkpoint,'--checkpoint required')
                    evaluate(cache,split,rid,a.checkpoint,out, frozen_config['batch_size'])
                else:train(cache,split,rid,frozen_config,run,a.resume,a.workers)
        return
    if a.command=='evaluate':
        check(a.checkpoint,'--checkpoint required')
        evaluate_images(a.root,split,a.round,a.checkpoint,a.hf_cache,out,a.device,a.workers,config['frame_microbatch']);return
    if a.command=='smoke':
        check(split['smoke'],'Smoke command requires smoke split');config.update(epochs=2,batch_size=4,accumulation=2)
        rid=next(iter(split['rounds']))
        for backbone in a.backbones:
            folder=out/'smoke'/a.mode/backbone
            def run(target,resume=False,stop=None):
                return train_images(a.root,split,rid,backbone,a.mode,base,adaptation,config,a.hf_cache,folder/target,a.device,a.workers,resume,stop,True)
            run('uninterrupted',resume=(folder/'uninterrupted/latest.pt').exists())
            if not (folder/'resumed/latest.pt').exists():run('resumed',stop=1)
            run('resumed',resume=True)
            verify_resume(folder/'uninterrupted',folder/'resumed')
            evaluate_images(a.root,split,rid,folder/'resumed/best.pt',a.hf_cache,folder/'resumed/evaluation',a.device,a.workers,config['frame_microbatch'])
            save_json(folder/'smoke_verification.json',{'passed':True,'mode':a.mode,'backbone':backbone,'split_identity':split['split_identity'],
                       'real_data':True,'resume_exact':True,'reload_equivalent':True,'frozen_weights_unchanged':True,'device':a.device})
            print(f'PASS: {a.mode}/{backbone}',flush=True)
        return
    for backbone in a.backbones:
        for rid in (split['rounds'] if a.command=='train-all' else [a.round]):
            train_images(a.root,split,rid,backbone,a.mode,base,adaptation,config,a.hf_cache,out/'training'/a.mode/backbone/rid,a.device,a.workers,a.resume,init_checkpoint=a.init_checkpoint)

if __name__=='__main__':main()
