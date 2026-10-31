"""A single linear logit head, deterministic frame draws and video evaluation."""
import copy
import json
import os
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Sampler
from .common import check, identity, save_csv, save_json, sha
from .features import FeatureCache


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def aggregate_logits(frame_logits):
    check(frame_logits.ndim==2 and frame_logits.shape[1]==8,'Expected eight frame logits per video')
    return frame_logits.mean(axis=1)


def scores_from_logits(logits):
    return torch.sigmoid(torch.as_tensor(logits,dtype=torch.float64)).numpy()


def metrics(labels,scores):
    y=np.asarray(labels,dtype=int); score=np.asarray(scores,dtype=float)
    check(len(y)>0 and len(y)==len(score) and np.isfinite(score).all(),'Invalid evaluation input')
    pred=(score>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'auroc':float(roc_auc_score(y,score)) if len(set(y))==2 else None,
            'auroc_undefined_reason':None if len(set(y))==2 else 'one_class_subset',
            'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None,
            'precision':float(tp/(tp+fp)) if tp+fp else None,
            'recall':float(tp/(tp+fn)) if tp+fn else None,
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],'threshold':.5,
            'undefined_policy':'null when denominator is zero; AUROC null for one-class subsets'}


class CachedDraws(Dataset):
    def __init__(self,cache_path,rows):
        self.cache_path=str(cache_path);self.rows=rows;self._cache=None
    @property
    def cache(self):
        if self._cache is None:self._cache=FeatureCache(self.cache_path,verify=False)
        return self._cache
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        i,k=key;row=self.rows[i]
        feature=np.array(self.cache.values[self.cache.lookup[row['clip_id']],k],copy=True)
        return {'features':torch.from_numpy(feature),'label':float(row['label']),
                'clip_id':row['clip_id'],'frame_position':k}


class BalancedFrameSampler(Sampler):
    def __init__(self,rows,seed,epoch,balanced=True,equal_generators=True):
        self.rows=rows;self.generator=torch.Generator().manual_seed(seed+epoch)
        class_n=Counter(r['label'] for r in rows); gen_n=Counter(r['source'] for r in rows if r['label']==1)
        weights=[]
        for r in rows:
            mass=.5 if balanced else class_n[r['label']]/len(rows)
            denom=len(gen_n)*gen_n[r['source']] if r['label']==1 and equal_generators else class_n[r['label']]
            weights.append(mass/denom)
        self.weights=torch.tensor(weights,dtype=torch.double)
    def __len__(self):return len(self.rows)
    def __iter__(self):
        ids=torch.multinomial(self.weights,len(self.rows),replacement=True,generator=self.generator)
        frames=torch.randint(8,(len(self.rows),),generator=self.generator)
        return iter(zip(ids.tolist(),frames.tolist()))


def predict(head,cache,rows,batch_size=256):
    head.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            x=torch.from_numpy(cache.rows([r['clip_id'] for r in batch]))
            logits=head(x).squeeze(-1).numpy()
            means=aggregate_logits(logits); scores=scores_from_logits(means)
            for row,frame,mean,score in zip(batch,logits,means,scores):
                predictions.append(dict(row,frame_logits=frame.tolist(),video_logit=float(mean),video_score=float(score)))
    return predictions


def summarize(predictions):
    labels=[r['label'] for r in predictions];scores=[r['video_score'] for r in predictions]
    result=metrics(labels,scores)
    result['loss']=float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([r['video_logit'] for r in predictions]),torch.tensor(labels,dtype=torch.float32)))
    real=[r for r in predictions if r['label']==0]
    result['by_generator']={}
    for source in sorted({r['source'] for r in predictions if r['label']==1}):
        generated=[r for r in predictions if r['source']==source and r['label']==1]
        combined=real+generated
        result['by_generator'][source]={'positive_count':len(generated),'shared_real_count':len(real),
            'generated_recall':sum(r['video_score']>=.5 for r in generated)/len(generated),
            'against_shared_real':metrics([r['label'] for r in combined],[r['video_score'] for r in combined])}
    result['real_false_positive_rate']={s:sum(r['video_score']>=.5 for r in real if r['source']==s)/sum(r['source']==s for r in real)
                                        for s in sorted({r['source'] for r in real})}
    return result


def torch_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def load_checkpoint(path,cache,split_id,round_id):
    # Only load local checkpoints created by this project (contains RNG Python objects).
    cp=torch.load(path,map_location='cpu',weights_only=False)
    check(cp['cache_identity']==cache.meta['cache_identity'],'Checkpoint/cache mismatch')
    check(cp['split_identity']==split_id and cp['round_id']==round_id,'Checkpoint/split mismatch')
    return cp


def train(cache_path,split,round_id,config,output,resume=False,workers=0,stop_after=None,init_checkpoint=None):
    from .plots import training_plots
    cache=FeatureCache(cache_path);output=Path(output)
    check(cache.meta['identity']['index_fingerprints']==split['index_fingerprints'],'Cache/index mismatch')
    rows=split['rounds'][round_id]
    training=[r for r in rows if r['role']=='train'];validation=[r for r in rows if r['role']=='validation']
    check(all(r['clip_id'] in cache.lookup for r in training+validation),'Required clip absent from cache')
    check({r['label'] for r in training}=={0,1} and {r['label'] for r in validation}=={0,1},'Training/validation needs both classes')
    config=copy.deepcopy(config)
    if init_checkpoint:
        config['initial_checkpoint_sha256']=sha(Path(init_checkpoint))
    seed_all(config['seed'])
    head=torch.nn.Linear(cache.values.shape[-1],1)
    if init_checkpoint:
        initial=torch.load(init_checkpoint,map_location='cpu',weights_only=False)
        check(initial['cache_identity']==cache.meta['cache_identity'] and initial['split_identity']==split['split_identity'],'Initial checkpoint/cache or split mismatch')
        head.load_state_dict(initial['model'])
    optimizer=torch.optim.AdamW([{'params':[head.weight],'weight_decay':config['weight_decay']},
                                {'params':[head.bias],'weight_decay':0.}],lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=config['scheduler_factor'],
                patience=config['scheduler_patience'],threshold=config['min_delta'],threshold_mode='abs')
    check(len(optimizer.param_groups)==2 and sum(len(g['params']) for g in optimizer.param_groups)==2,'Unexpected optimizer parameters')
    start=1;best=-float('inf');anchor=-float('inf');bad=0;history=[];best_cp=None
    latest=output/'latest.pt'
    if latest.exists():
        check(resume,'Run already exists; use --resume or a new output directory')
        cp=load_checkpoint(latest,cache,split['split_identity'],round_id)
        check(cp['config']==config,'Training configuration changed on resume')
        head.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        start=cp['epoch']+1;best=cp['best_auroc'];anchor=cp['early_stopping_anchor'];bad=cp['bad_epochs'];history=cp['history'];best_cp=cp['best_checkpoint']
        restore_rng(cp['rng'])
        torch_save(output/'best.pt',best_cp) # repair publication interrupted after latest commit
    criterion=torch.nn.BCEWithLogitsLoss()
    for epoch in range(start,config['epochs']+1):
        if bad>=config['patience']:break
        head.train()
        sampler=BalancedFrameSampler(training,config['seed'],epoch,config['balanced_classes'],config['equal_generators'])
        loader=DataLoader(CachedDraws(cache.path,training),sampler=sampler,batch_size=config['batch_size'],num_workers=workers)
        total,n=0.,0
        lr_used=optimizer.param_groups[0]['lr']
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=head(batch['features']).squeeze(-1)
            loss=criterion(logits,batch['label'].float());loss.backward();optimizer.step()
            total+=float(loss.detach())*len(logits);n+=len(logits)
        val=summarize(predict(head,cache,validation,config['batch_size']))
        auc=val['auroc'];check(auc is not None,'Validation AUROC undefined')
        # Highest AUROC checkpoint; significant improvement controls patience separately.
        improved=auc>best
        if improved:best=auc
        if auc>anchor+config['min_delta']:anchor=auc;bad=0
        else:bad+=1
        scheduler.step(auc)
        history.append({'epoch':epoch,'training_loss':total/n,'validation_loss':val['loss'],
                        'validation_auroc':auc,'validation_f1':val['f1'],'learning_rate':lr_used,
                        'next_learning_rate':optimizer.param_groups[0]['lr']})
        cp={'model':copy.deepcopy(head.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'scheduler':copy.deepcopy(scheduler.state_dict()),'epoch':epoch,'rng':rng_state(),
            'config':config,'split_identity':split['split_identity'],'cache_identity':cache.meta['cache_identity'],
            'round_id':round_id,'smoke':split['smoke'],'backbone':cache.meta['identity']['checkpoint'],
            'best_auroc':best,'early_stopping_anchor':anchor,'bad_epochs':bad,'history':copy.deepcopy(history)}
        if improved:best_cp=copy.deepcopy(cp)
        cp['best_checkpoint']=best_cp
        torch_save(latest,cp);torch_save(output/'best.pt',best_cp)
        save_json(output/'history.json',history)
        save_csv(output/'history.csv',history,list(history[0]))
        print(f'{round_id} epoch {epoch}: loss={total/n:.4f} validation AUROC={auc:.4f}',flush=True)
        if stop_after is not None and epoch>=stop_after:break
    if history:training_plots(history,output,split['smoke'])
    check((output/'best.pt').exists(),'No best checkpoint produced')
    return load_checkpoint(output/'best.pt',cache,split['split_identity'],round_id)


def evaluate(cache_path,split,round_id,checkpoint,output,batch_size=256):
    from .plots import evaluation_plots
    cache=FeatureCache(cache_path)
    check(cache.meta['identity']['index_fingerprints']==split['index_fingerprints'],'Evaluation input fingerprint mismatch')
    cp=load_checkpoint(checkpoint,cache,split['split_identity'],round_id)
    head=torch.nn.Linear(cache.values.shape[-1],1);head.load_state_dict(cp['model'])
    output=Path(output);summaries={};all_predictions={}
    for role in ('test_seen','test_unseen'):
        rows=[r for r in split['rounds'][round_id] if r['role']==role]
        predictions=predict(head,cache,rows,batch_size)
        result=summarize(predictions)
        result.update({'round_id':round_id,'partition':role,'smoke':split['smoke'],
                       'backbone':cache.meta['identity']['checkpoint'],'checkpoint_epoch':cp['epoch'],
                       'split_identity':split['split_identity'],'cache_identity':cache.meta['cache_identity'],
                       'shared_real_test_examples':True})
        save_json(output/f'{role}_metrics.json',result)
        serialized=[dict(r,frame_logits=json.dumps(r['frame_logits']),partition=role) for r in predictions]
        save_csv(output/f'{role}_predictions.csv',serialized,
                 ['clip_id','round_id','partition','label','source','frame_logits','video_logit','video_score'])
        summaries[role]=result;all_predictions[role]=predictions
    evaluation_plots(all_predictions,summaries,output,split['smoke'])
    return summaries
