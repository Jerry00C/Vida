"""Image training, portable adapted checkpoints and video-level evaluation."""
import contextlib
import copy
import json
import math
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from filelock import FileLock
from .common import check,identity,index_identity,save_json,save_csv,sha
from .data import FrameDataset,ProcessorCollator,load_indexes
from .learning import BalancedFrameSampler,aggregate_logits,scores_from_logits,summarize,seed_all,rng_state,restore_rng,torch_save
from .features import device_for
from .adapters import build_model,ParameterAudit
from .plots import training_plots,evaluation_plots,finish


class ImageDraws:
    def __init__(self,root,rows,videos,frames):
        self.dataset=FrameDataset(root,[r['clip_id'] for r in rows],videos,frames)
    def __len__(self):return len(self.dataset)//8
    def __getitem__(self,key):
        video,frame=key
        return self.dataset[video*8+frame]


def precision_policy(device,requested):
    if requested=='auto':
        requested=('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type=='cuda' else 'fp32'
    check(requested in ('fp32','fp16','bf16'),'Unknown precision')
    if requested=='bf16':check(device.type in ('cpu','cuda') and (device.type!='cuda' or torch.cuda.is_bf16_supported()),'BF16 unsupported on selected device')
    if requested=='fp16':check(device.type=='cuda','FP16 training/scaling is supported on CUDA only; use FP32 fallback')
    return requested


def autocast(device,precision):
    if precision=='fp32':return contextlib.nullcontext()
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16)


def optimizer_for(model,config):
    norm_ids={id(p) for m in model.modules() if isinstance(m,torch.nn.LayerNorm) for p in m.parameters(recurse=False)}
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        section='head' if name.startswith('head.') else model.mode
        lr=config['head_lr'] if section=='head' else config['partial_lr']
        decay=0. if name.endswith('.bias') or id(p) in norm_ids else config['weight_decay']
        group=groups.setdefault((section,decay),{'params':[],'lr':lr,'weight_decay':decay,
                                               'name':section+('_decay' if decay else '_no_decay'),'parameter_names':[]})
        group['params'].append(p);group['parameter_names'].append(name)
    ids=[id(p) for g in groups.values() for p in g['params']]
    check(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer membership mismatch/duplicates')
    return torch.optim.AdamW(list(groups.values()))


def window_samples(batch_index,batches,samples,batch_size,accumulation):
    first=(batch_index//accumulation)*accumulation
    last=min(first+accumulation,batches)
    return min(last*batch_size,samples)-first*batch_size


def make_loader(root,rows,videos,frames,processor,batch_size,workers,seed,epoch=None,config=None):
    generator=torch.Generator().manual_seed(seed+(epoch or 0)+100000)
    if epoch is None:
        dataset=FrameDataset(root,[r['clip_id'] for r in rows],videos,frames);sampler=None
    else:
        dataset=ImageDraws(root,rows,videos,frames)
        sampler=BalancedFrameSampler(rows,seed,epoch,config['balanced_classes'],config['equal_generators'])
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=False,num_workers=workers,
                      generator=generator,collate_fn=ProcessorCollator(processor))


@torch.inference_mode()
def image_predict(model,processor,root,rows,videos,frames,device,microbatch,workers,precision='fp32'):
    model.eval();logits=[];seen=0
    loader=make_loader(root,rows,videos,frames,processor,microbatch,workers,42)
    for batch in loader:
        with autocast(device,precision):
            values=model({k:v.to(device) for k,v in batch['inputs'].items()})
        check(torch.isfinite(values).all().item(),'Nonfinite evaluation logits')
        for item,value in zip(batch['items'],values.float().cpu().tolist()):
            row=rows[seen//8]
            check(item['clip_id']==row['clip_id'] and item['frame_position']==seen%8 and item['label']==row['label'],
                  'Evaluation frame/clip/label alignment mismatch')
            logits.append(value);seen+=1
    check(seen==len(rows)*8,'Incomplete video evaluation')
    frame_logits=np.asarray(logits,dtype=np.float64).reshape(len(rows),8)
    means=aggregate_logits(frame_logits);scores=scores_from_logits(means)
    return [dict(r,frame_logits=f.tolist(),video_logit=float(m),video_score=float(s)) for r,f,m,s in zip(rows,frame_logits,means,scores)]


def data_for(root,split):
    check(index_identity(root)==split['index_fingerprints'],'Processed indexes differ from exact split manifest')
    videos,frames=load_indexes(root)
    for rows in split['rounds'].values():
        for r in rows:
            check(r['clip_id'] in videos and videos[r['clip_id']]['label']==str(r['label']),'Split/data label mismatch')
    return videos,frames


def setup(backbone,mode,base,adaptation,config,split,rid,hf_cache,device_name):
    seed_all(config['seed'])
    if device_name.startswith('cuda'):check(torch.cuda.is_available(),'CUDA requested but unavailable')
    if device_name=='mps':check(torch.backends.mps.is_available(),'MPS requested but unavailable')
    device=device_for(device_name);precision=precision_policy(device,config['precision'])
    model,processor=build_model(backbone,mode,base,adaptation,hf_cache,device)
    import transformers
    spec={'version':'vida-adaptation-v1','backbone':backbone,'checkpoint':base['checkpoints'][backbone],
          'revision':base['revisions'][backbone],'mode':mode,'adaptation':adaptation,
          'round_id':rid,'split_identity':split['split_identity'],'index_fingerprints':split['index_fingerprints'],
          'initialization':'original_pretrained_plus_fresh_head','seed':config['seed'],
          'processor':processor.to_dict(),'representation':model.structure['representation'],
          'structure':model.structure,'training':config,'precision':precision,'device_type':device.type,
          'versions':{'torch':str(torch.__version__),'transformers':transformers.__version__}}
    return model,processor,device,precision,spec


def state_rng(device):
    state=rng_state()
    state['mps']=torch.mps.get_rng_state() if device.type=='mps' else None
    return state


def set_rng(state,device):
    restore_rng(state)
    if state.get('mps') is not None:
        check(device.type=='mps','MPS RNG requires MPS resume');torch.mps.set_rng_state(state['mps'])


def load_adapted_checkpoint(path,expected_identity=None):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    check(cp['experiment_identity']==identity(cp['spec']),'Checkpoint identity is corrupted')
    if expected_identity:check(cp['experiment_identity']==expected_identity,'Incompatible resume: model/method/split/processor/training/device identity changed')
    return cp


def group_lr_plot(history,output,smoke):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for name in history[0]['group_learning_rates']:
        ax.plot([h['epoch'] for h in history],[h['group_learning_rates'][name] for h in history],marker='o',label=name)
    ax.set(xlabel='Epoch',ylabel='Learning rate',title='SMOKE: pipeline verification, not research results' if smoke else 'Optimizer group learning rates')
    ax.legend();finish(fig,Path(output)/'group_learning_rates.png')


def train_images(root,split,rid,backbone,mode,base,adaptation,config,hf_cache,output,device_name='auto',workers=0,resume=False,stop_after=None,audit_parameters=False,init_checkpoint=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.run.lock'),timeout=0):
        return _train(root,split,rid,backbone,mode,base,adaptation,config,hf_cache,output,device_name,workers,resume,stop_after,audit_parameters,init_checkpoint)


def _train(root,split,rid,backbone,mode,base,adaptation,config,hf_cache,output,device_name,workers,resume,stop_after,audit_parameters,init_checkpoint):
    check(mode == 'partial','Image adaptation trainer requires partial mode')
    check(config['batch_size']>0 and config['accumulation']>0 and config['epochs']>0 and config['frame_microbatch']>0,'Invalid batch/epoch settings')
    videos,frames=data_for(root,split)
    rows=split['rounds'][rid];training=[r for r in rows if r['role']=='train'];validation=[r for r in rows if r['role']=='validation']
    check({r['label'] for r in training}=={0,1} and {r['label'] for r in validation}=={0,1},'Both classes required for training/validation')
    model,processor,device,precision,spec=setup(backbone,mode,base,adaptation,config,split,rid,hf_cache,device_name)
    if init_checkpoint:
        initial=load_adapted_checkpoint(init_checkpoint)
        comparable=lambda value: {k:v for k,v in value.items() if k not in ('round_id','initial_checkpoint_sha256')}
        check(comparable(initial['spec'])==comparable(spec),'Initial checkpoint model/config/split mismatch')
        model.load_adapted_state(initial['adapted_state'])
        spec['initial_checkpoint_sha256']=sha(Path(init_checkpoint))
    eid=identity(spec)
    latest=output/'latest.pt'
    check(not resume or latest.exists(),'Resume requested but latest.pt is missing')
    check(resume or not latest.exists(),'Run exists; use --resume or a fresh output')
    optimizer=optimizer_for(model,config)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=config['scheduler_factor'],patience=config['scheduler_patience'],threshold=config['min_delta'],threshold_mode='abs')
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and precision=='fp16'))
    start=1;best=-float('inf');anchor=-float('inf');bad=0;global_step=0;history=[];best_checkpoint=None
    elapsed_before=0.;compute_before=0.;samples_before=0
    if latest.exists():
        cp=load_adapted_checkpoint(latest,eid)
        model.load_adapted_state(cp['adapted_state']);optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
        start=cp['epoch']+1;best=cp['best_auroc'];anchor=cp['early_stopping_anchor'];bad=cp['bad_epochs'];global_step=cp['global_step']
        history=cp['history'];best_checkpoint=cp['best_checkpoint'];elapsed_before=cp['elapsed_seconds']
        compute_before=cp['training_seconds'];samples_before=cp['training_samples']
        set_rng(cp['rng'],device);torch_save(output/'best.pt',best_checkpoint)
    report=model.report()
    report.update({'experiment_identity':eid,'spec':spec,'effective_batch_size':config['batch_size']*config['accumulation'],
                   'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]})
    save_json(output/'trainable_parameters.json',report)
    print(f'{backbone}/{mode}: {report["trainable_parameters"]:,}/{report["total_parameters"]:,} parameters trainable',flush=True)
    print('Trainable names:',', '.join(report['trainable_names']),flush=True)
    audit=ParameterAudit(model) if audit_parameters and start<=config['epochs'] and bad<config['patience'] else None
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    started=time.monotonic();train_seconds=compute_before;train_samples=samples_before
    for epoch in range(start,config['epochs']+1):
        if bad>=config['patience']:break
        model.train(True);model.check_trainable_policy();optimizer.zero_grad(set_to_none=True)
        loader=make_loader(root,training,videos,frames,processor,config['batch_size'],workers,config['seed'],epoch,config)
        batch_count=len(loader);loss_sum=0.;seen=0;stepsizes=[];rates={g['name']:g['lr'] for g in optimizer.param_groups}
        train_start=time.monotonic()
        for batch_index,batch in enumerate(loader):
            denominator=window_samples(batch_index,batch_count,len(training),config['batch_size'],config['accumulation'])
            labels=torch.tensor([r['label'] for r in batch['items']],dtype=torch.float32,device=device)
            with autocast(device,precision):
                logits=model({k:v.to(device) for k,v in batch['inputs'].items()})
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels,reduction='sum')
            check(torch.isfinite(loss).item(),'Nonfinite training loss')
            scaler.scale(loss/denominator).backward()
            loss_sum+=float(loss.detach());seen+=len(labels)
            boundary=(batch_index+1)%config['accumulation']==0 or batch_index+1==batch_count
            if boundary:
                scaler.unscale_(optimizer)
                finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
                if not finite:
                    check(scaler.is_enabled(),'Nonfinite unscaled gradient')
                    # GradScaler skips overflowing FP16 steps and reduces its scale.
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    print(f'FP16 overflow: skipped optimizer update at epoch {epoch}, batch {batch_index}',flush=True)
                    continue
                if audit:audit.observe(model)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                global_step+=1;stepsizes.append(denominator)
        if device.type=='cuda':torch.cuda.synchronize(device)
        train_seconds+=time.monotonic()-train_start;train_samples+=seen
        predictions=image_predict(model,processor,root,validation,videos,frames,device,config['frame_microbatch'],workers,precision)
        val=summarize(predictions);auc=val['auroc'];check(auc is not None,'Undefined validation AUROC')
        improved=auc>best
        if improved:best=auc
        if auc>anchor+config['min_delta']:anchor=auc;bad=0
        else:bad+=1
        scheduler.step(auc)
        history.append({'epoch':epoch,'training_loss':loss_sum/seen,'validation_loss':val['loss'],
                        'validation_auroc':auc,'validation_f1':val['f1'],'learning_rate':rates[next(k for k in rates if k.startswith('head'))],
                        'group_learning_rates':rates,'optimizer_step_samples':stepsizes,'global_step':global_step})
        cp={'experiment_identity':eid,'spec':spec,'adapted_state':model.adapted_state(),
            'adapter_config':model.adapter_configuration,'base_checkpoint':base['checkpoints'][backbone],'base_revision':base['revisions'][backbone],
            'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),'scaler':scaler.state_dict(),
            'epoch':epoch,'global_step':global_step,'best_auroc':best,'early_stopping_anchor':anchor,'bad_epochs':bad,
            'history':copy.deepcopy(history),'rng':state_rng(device),'sampling':{'seed':config['seed'],'next_epoch':epoch+1,'policy':'balanced video then uniform frame; epoch-seeded sampler'},
            'elapsed_seconds':elapsed_before+time.monotonic()-started,'training_seconds':train_seconds,'training_samples':train_samples,
            'smoke':split['smoke']}
        if improved:best_checkpoint=copy.deepcopy(cp)
        cp['best_checkpoint']=best_checkpoint
        torch_save(latest,cp);torch_save(output/'best.pt',best_checkpoint)
        save_json(output/'history.json',history)
        save_csv(output/'history.csv',[dict(h,group_learning_rates=json.dumps(h['group_learning_rates']),optimizer_step_samples=json.dumps(h['optimizer_step_samples'])) for h in history],list(history[0]))
        print(f'{backbone}/{mode} {rid} epoch {epoch}: loss={loss_sum/seen:.5f}, val AUROC={auc:.5f}',flush=True)
        if stop_after is not None and epoch>=stop_after:break
    if audit:save_json(output/f'parameter_audit_epoch_{start}.json',audit.finish(model))
    if history:training_plots(history,output,split['smoke']);group_lr_plot(history,output,split['smoke'])
    check((output/'best.pt').exists(),'No completed checkpoint')
    cost={'elapsed_seconds':elapsed_before+time.monotonic()-started,'training_seconds':train_seconds,
          'training_samples':train_samples,'training_frames_per_second':train_samples/train_seconds if train_seconds else None,
          'trainable_parameters':report['trainable_parameters'],'total_parameters':report['total_parameters'],
          'device':str(device),'precision':precision,
          'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
          'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if os.uname().sysname=='Darwin' else 1024),
          'memory_note':'GPU peak only measured on CUDA; CPU RSS is process-lifetime high-water mark'}
    save_json(output/'cost.json',cost)
    # Compare predictions against a freshly reconstructed best-checkpoint model.
    cp=load_adapted_checkpoint(output/'best.pt',eid)
    model.load_adapted_state(cp['adapted_state'])
    before=image_predict(model,processor,root,validation,videos,frames,device,config['frame_microbatch'],workers,precision) if audit_parameters else None
    del model,optimizer,scheduler,cp,best_checkpoint
    if device.type=='cuda':torch.cuda.empty_cache()
    if before is not None:
        model2,processor2,_,_,spec2=setup(backbone,mode,base,adaptation,config,split,rid,hf_cache,device_name)
        if init_checkpoint:spec2['initial_checkpoint_sha256']=spec['initial_checkpoint_sha256']
        cp=load_adapted_checkpoint(output/'best.pt',identity(spec2));model2.load_adapted_state(cp['adapted_state'])
        after=image_predict(model2,processor2,root,validation,videos,frames,device,config['frame_microbatch'],workers,precision)
        difference=float(np.max(np.abs(np.array([r['frame_logits'] for r in before])-np.array([r['frame_logits'] for r in after]))))
        check(difference<=(1e-6 if precision=='fp32' else 1e-3),'Predictions changed after checkpoint reload')
        save_json(output/'reload_verification.json',{'max_frame_logit_difference':difference,'passed':True,'checkpoint':'best.pt'})
    return cost


def evaluate_images(root,split,rid,checkpoint,hf_cache,output,device_name='auto',workers=0,microbatch=8):
    cp=load_adapted_checkpoint(checkpoint);spec=cp['spec']
    check(spec['round_id']==rid and spec['split_identity']==split['split_identity'],'Evaluation split/round mismatch')
    videos,frames=data_for(root,split)
    base={'checkpoints':{spec['backbone']:spec['checkpoint']},'revisions':{spec['backbone']:spec['revision']}}
    model,processor,device,precision,recreated=setup(spec['backbone'],spec['mode'],base,spec['adaptation'],spec['training'],split,rid,hf_cache,device_name)
    check(recreated['processor']==spec['processor'] and recreated['representation']==spec['representation'],'Evaluation processor/feature mismatch')
    model.load_adapted_state(cp['adapted_state'])
    output=Path(output);results={};all_predictions={}
    for role in ('test_seen','test_unseen'):
        rows=[r for r in split['rounds'][rid] if r['role']==role]
        predictions=image_predict(model,processor,root,rows,videos,frames,device,microbatch,workers,precision)
        result=summarize(predictions)
        result.update({'backbone':spec['checkpoint'],'backbone_key':spec['backbone'],'adaptation_mode':spec['mode'],
                       'round_id':rid,'partition':role,'smoke':split['smoke'],'split_identity':split['split_identity'],
                       'experiment_identity':cp['experiment_identity'],'shared_real_test_examples':True,'checkpoint_epoch':cp['epoch'],
                       'evaluation_policy':'sigmoid_mean_8_logits_threshold_0.5','processor':spec['processor'],'revision':spec['revision'],
                       'representation':spec['representation'],'evaluation_precision':precision})
        save_json(output/f'{role}_metrics.json',result)
        save_csv(output/f'{role}_predictions.csv',[dict(r,frame_logits=json.dumps(r['frame_logits']),partition=role) for r in predictions],
                 ['clip_id','round_id','partition','label','source','frame_logits','video_logit','video_score'])
        results[role]=result;all_predictions[role]=predictions
    evaluation_plots(all_predictions,results,output,split['smoke'])
    return results
