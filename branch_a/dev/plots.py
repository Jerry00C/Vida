import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('runs/branch_a/.matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from .common import read_json, save_json


def title(smoke): return 'SMOKE: pipeline verification, not research results' if smoke else 'VIDA Branch A'

def finish(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def training_plots(history,out,smoke):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=[r['epoch'] for r in history]
    for filename,fields in [('loss',['training_loss','validation_loss']),('validation_auroc',['validation_auroc']),
                            ('validation_f1',['validation_f1']),('learning_rate',['learning_rate'])]:
        fig,ax=plt.subplots(figsize=(7,4))
        for field in fields:ax.plot(x,[r[field] for r in history],marker='o',label=field)
        ax.set(xlabel='Epoch',ylabel=filename,title=title(smoke));ax.legend();finish(fig,out/f'{filename}.png')


def evaluation_plots(predictions,results,out,smoke):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,5))
    for role,rows in predictions.items():
        y=[r['label'] for r in rows]
        if len(set(y))==2:
            fpr,tpr,_=roc_curve(y,[r['video_score'] for r in rows]);ax.plot(fpr,tpr,label=role)
    ax.plot([0,1],[0,1],'k--',alpha=.4);ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=title(smoke));ax.legend()
    finish(fig,out/'test_roc.png')
    for role,result in results.items():
        fig,ax=plt.subplots(figsize=(5,4));cm=np.array(result['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['real','generated'],yticklabels=['real','generated'],
               xlabel='Predicted',ylabel='Actual',title=title(smoke)+'\n'+role)
        finish(fig,out/f'{role}_confusion.png')
        fig,ax=plt.subplots(figsize=(8,4));gens=result['by_generator'];names=list(gens)
        vals=[gens[n]['against_shared_real']['auroc'] for n in names]
        ax.bar(names,[np.nan if v is None else v for v in vals]);ax.set(ylim=(0,1),ylabel='AUROC vs shared real test',title=title(smoke)+'\n'+role)
        ax.tick_params(axis='x',rotation=25);finish(fig,out/f'{role}_generators.png')


def comparison(root,output):
    records=[]
    for path in sorted(Path(root).rglob('*_metrics.json')):
        r=read_json(path)
        if 'partition' in r:records.append(r)
    summaries=[]
    for key in sorted({(r['backbone'],r['partition'],r['smoke']) for r in records}):
        selected=[r for r in records if (r['backbone'],r['partition'],r['smoke'])==key]
        values={r['round_id']:r['auroc'] for r in selected}
        finite=[v for v in values.values() if v is not None]
        summaries.append({'backbone':key[0],'partition':key[1],'smoke':key[2],'round_auroc':values,
                          'mean_available_round_auroc':float(np.mean(finite)) if finite else None,
                          'available_rounds':len(values),'expected_rounds':4,
                          'note':'Individual round values retained; no independent-trial confidence interval'})
    output=Path(output);output.mkdir(parents=True,exist_ok=True);save_json(output/'backbone_comparison.json',summaries)
    fig,ax=plt.subplots(figsize=(10,5))
    for i,r in enumerate(summaries):
        for rid,v in r['round_auroc'].items():
            if v is not None:ax.scatter(i,v);ax.annotate(rid,(i,v),fontsize=7)
    ax.set(xticks=range(len(summaries)),xticklabels=[r['backbone'].split('/')[-1]+'\n'+r['partition'] for r in summaries],
           ylim=(0,1),ylabel='Video AUROC',title=title(any(r['smoke'] for r in summaries)))
    ax.tick_params(axis='x',rotation=20);finish(fig,output/'backbone_comparison.png')
