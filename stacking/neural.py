"""Tiny epoch-trained MLP candidate; inference object only requires NumPy."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np

class NumpyMLP:
    def __init__(self,mean,scale,weights,epochs):
        self.mean=np.asarray(mean);self.scale=np.asarray(scale);self.weights=weights
        self.feature_names_in_=np.array(['A','B','C']);self.n_iter_=np.array([epochs])
    def decision_function(self,frame):
        x=np.asarray(frame,dtype=np.float64)
        x=((x-self.mean)/self.scale).astype(np.float32)
        w1,b1,w2,b2=self.weights
        return (np.maximum(x@w1.T+b1,0)@w2.T+b2).reshape(-1).astype(np.float64)
    def __getitem__(self,key):
        if key==-1:return self
        raise IndexError(key)

def fit_candidate(input_csv,output):
    import csv
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import log_loss,roc_auc_score
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=list(csv.DictReader(Path(input_csv).open(encoding='utf-8-sig')))
    x=np.array([[float(r[b]) for b in ['A','B','C']] for r in rows]);y=np.array([int(r['label']) for r in rows])
    groups=np.array([r['group_id'] for r in rows]);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    settings={'architecture':[3,8,1],'hidden_activation':'ReLU','dropout':.1,'optimizer':'AdamW',
        'learning_rate':.001,'weight_decay':.001,'batch_size':64,'max_epochs':200,
        'patience':20,'min_delta':.0001,'seed':42,'loss':'BCEWithLogitsLoss',
        'CV':'5-fold StratifiedGroupKFold, early-stop per validation fold',
        'final_epoch_rule':'rounded median of five validation-best epochs; refit all fusion_train',
        'test_or_calibration_data_used':False,'trainable_parameters':41}
    def network(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(8,1))
    def epoch(model,optimizer,tx,ty):
        model.train();order=torch.randperm(len(tx));total=0.
        for start in range(0,len(tx),64):
            indices=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx[indices]).reshape(-1),ty[indices])
            loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        return total/len(tx)
    def infer(model,values):
        model.eval()
        with torch.inference_mode():return model(torch.tensor(values,dtype=torch.float32)).reshape(-1).numpy()
    folds=[];oof=np.full(len(y),np.nan)
    for fold,(fit,valid) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups)):
        assert not set(groups[fit])&set(groups[valid])
        scaler=StandardScaler().fit(x[fit]);xx=scaler.transform(x)
        tx=torch.tensor(xx[fit],dtype=torch.float32);ty=torch.tensor(y[fit],dtype=torch.float32)
        model=network(42+fold);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        best=float('inf');best_epoch=0;state=None;anchor=float('inf');bad=0;history=[]
        for n in range(1,201):
            training_loss=epoch(model,optimizer,tx,ty)
            logits=infer(model,xx[valid]);probs=1/(1+np.exp(-logits));validation_loss=float(log_loss(y[valid],probs))
            history.append({'epoch':n,'training_loss':training_loss,'validation_loss':validation_loss})
            if validation_loss<best:best=validation_loss;best_epoch=n;state=copy.deepcopy(model.state_dict())
            if validation_loss<anchor-.0001:anchor=validation_loss;bad=0
            else:bad+=1
            if bad>=20:break
        model.load_state_dict(state);logits=infer(model,xx[valid]);probs=1/(1+np.exp(-logits));oof[valid]=probs
        folds.append({'fold':fold,'validation_log_loss':float(log_loss(y[valid],probs)),
            'validation_auroc':float(roc_auc_score(y[valid],probs)),'best_epoch':best_epoch,
            'epochs_completed':n,'history':history})
        print(f'MLP fold {fold}: best epoch {best_epoch}, completed {n}, validation logloss {best:.5f}',flush=True)
    assert np.isfinite(oof).all()
    epochs=int(np.rint(np.median([f['best_epoch'] for f in folds])))
    scaler=StandardScaler().fit(x);xx=scaler.transform(x);tx=torch.tensor(xx,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
    model=network(42);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
    for n in range(1,epochs+1):history.append({'epoch':n,'training_loss':epoch(model,optimizer,tx,ty)})
    state=model.state_dict();weights=[state[k].numpy().copy() for k in ['0.weight','0.bias','3.weight','3.bias']]
    frozen=NumpyMLP(scaler.mean_,scaler.scale_,weights,epochs)
    maximum=float(np.max(np.abs(frozen.decision_function(x)-infer(model,xx))))
    assert maximum<1e-5,maximum
    joblib.dump(frozen,out/'candidate.joblib')
    np.save(out/'oof.npy',oof)
    report={'settings':settings,'folds':folds,'final_epochs':epochs,'final_training_history':history,
        'mean_log_loss':float(np.mean([f['validation_log_loss'] for f in folds])),
        'mean_auroc':float(np.mean([f['validation_auroc'] for f in folds])),
        'numpy_vs_torch_max_abs_logit_difference':maximum,'torch_version':str(torch.__version__)}
    (out/'selection_report.json').write_text(json.dumps(report,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    # Keep the portable object's pickle module name stable under python -m.
    from stacking.neural import fit_candidate as run
    run(args.input,args.output)

if __name__=='__main__':main()
