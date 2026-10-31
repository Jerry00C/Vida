"""Same 41-parameter MLP; eight meta-generator holdouts, no final test input."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
from .neural import NumpyMLP

def fit(input_file,fold_file,output):
    import torch
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss,roc_auc_score
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=list(csv.DictReader(Path(input_file).open(encoding='utf-8-sig')));ids=[r['clip_id'] for r in rows];lookup={cid:i for i,cid in enumerate(ids)}
    x=np.array([[float(r[b]) for b in ['A','B','C']] for r in rows]);y=np.array([int(r['label']) for r in rows])
    folds=json.loads(Path(fold_file).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def network(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(8,1))
    def epoch(model,optimizer,tx,ty):
        model.train();order=torch.randperm(len(tx));total=0.
        for start in range(0,len(tx),64):
            part=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx[part]).reshape(-1),ty[part]);loss.backward();optimizer.step();total+=float(loss.detach())*len(part)
        return total/len(tx)
    def logits(model,values):
        model.eval()
        with torch.inference_mode():return model(torch.tensor(values,dtype=torch.float32)).reshape(-1).numpy()
    results=[];oof=np.full(len(y),np.nan)
    for fold in folds:
        tr=np.array([lookup[cid] for cid in fold['fit_ids']]);va=np.array([lookup[cid] for cid in fold['validation_ids']])
        assert not {rows[i]['group_id'] for i in tr}&{rows[i]['group_id'] for i in va}
        scaler=StandardScaler().fit(x[tr]);xx=scaler.transform(x)
        tx=torch.tensor(xx[tr],dtype=torch.float32);ty=torch.tensor(y[tr],dtype=torch.float32)
        model=network(42+fold['fold']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        best=float('inf');anchor=float('inf');bad=0;best_epoch=0;state=None;history=[]
        for n in range(1,201):
            loss=epoch(model,optimizer,tx,ty);scores=1/(1+np.exp(-np.clip(logits(model,xx[va]),-80,80)))
            val=float(log_loss(y[va],scores));history.append({'epoch':n,'training_loss':loss,'validation_loss':val})
            if val<best:best=val;best_epoch=n;state=copy.deepcopy(model.state_dict())
            if val<anchor-.0001:anchor=val;bad=0
            else:bad+=1
            if bad>=20:break
        model.load_state_dict(state);scores=1/(1+np.exp(-np.clip(logits(model,xx[va]),-80,80)));oof[va]=scores
        results.append({'held_family':fold['held_family'],'best_epoch':best_epoch,'epochs_completed':n,
            'validation_log_loss':float(log_loss(y[va],scores)),'validation_auroc':float(roc_auc_score(y[va],scores)),'history':history})
        print(f"MLP held {fold['held_family']}: best epoch {best_epoch}, epochs {n}, logloss {best:.6f}",flush=True)
    assert np.isfinite(oof).all();epochs=int(np.rint(np.median([r['best_epoch'] for r in results])))
    scaler=StandardScaler().fit(x);xx=scaler.transform(x);tx=torch.tensor(xx,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
    model=network(42);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
    for n in range(1,epochs+1):history.append({'epoch':n,'training_loss':epoch(model,optimizer,tx,ty)})
    state=model.state_dict();weights=[state[k].numpy().copy() for k in ['0.weight','0.bias','3.weight','3.bias']]
    portable=NumpyMLP(scaler.mean_,scaler.scale_,weights,epochs)
    delta=float(np.max(np.abs(portable.decision_function(x)-logits(model,xx))));assert delta<2e-5
    joblib.dump(portable,output/'candidate.joblib');np.save(output/'oof.npy',oof)
    report={'settings':{'architecture':[3,8,1],'parameters':41,'activation':'ReLU','dropout':.1,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.001,'batch_size':64,'max_epochs':200,'patience':20,'min_delta':.0001,'seed':42,'loss':'BCEWithLogitsLoss'},
        'folds':results,'mean_log_loss':float(np.mean([r['validation_log_loss'] for r in results])),
        'final_epochs':epochs,'final_training_history':history,'torch_to_numpy_max_abs_logit_delta':delta,'torch_version':str(torch.__version__),
        'model_or_threshold_selected_using_final_test':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--folds',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from stacking.neural_expanded import fit as run
    run(args.input,args.folds,args.output)

if __name__=='__main__':main()
