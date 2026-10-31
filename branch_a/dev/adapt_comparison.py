"""Strictly matched descriptive comparisons; rounds are not independent trials."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .common import read_json,read_csv,save_json,save_csv,check,identity
from .plots import finish


def compare(output,frozen_root='runs/branch_a'):
    output=Path(output);frozen_root=Path(frozen_root)
    caches={}
    for path in (frozen_root/'cache').glob('*/*/metadata.json'):
        meta=read_json(path);caches[meta['cache_identity']]=meta
    entries=[];references={};seen=set()
    paths=list((frozen_root/'smoke').glob('*/resumed/evaluation/*_metrics.json'))+list((frozen_root/'training').glob('**/evaluation/*_metrics.json'))
    paths+=list((output/'smoke').glob('*/*/resumed/evaluation/*_metrics.json'))+list((output/'training').glob('**/evaluation/*_metrics.json'))
    for path in paths:
        result=read_json(path);mode=result.get('adaptation_mode','frozen')
        if mode=='frozen':
            meta=caches.get(result['cache_identity']);check(meta is not None,'Missing frozen cache provenance')
            config=meta['identity'];backbone=path.parts[-4] if 'smoke' in path.parts else path.parts[-4]
            # Checkpoint ID provides a stable key independent of directory layout.
            backbone={'openai/clip-vit-base-patch16':'clip','facebook/dinov2-base':'dinov2','facebook/dinov3-vitb16-pretrain-lvd1689m':'dinov3'}[config['checkpoint']]
            signature=(config['revision'],identity(config['processor']),config['method'],result['threshold'])
        else:
            backbone=result['backbone_key']
            check(result['evaluation_policy']=='sigmoid_mean_8_logits_threshold_0.5','Incompatible evaluation policy')
            signature=(result['revision'],identity(result['processor']),result['representation'],result['threshold'])
        predictions=read_csv(path.with_name(path.name.replace('_metrics.json','_predictions.csv')))
        alignment=sorted((r['clip_id'],int(r['label']),r['source']) for r in predictions)
        check(len(alignment)==len(set(x[0] for x in alignment)),'Duplicate video predictions')
        for row in predictions:
            values=json.loads(row['frame_logits']);check(len(values)==8 and np.isfinite(values).all(),'Invalid frame logits')
            mean=float(np.mean(values));check(abs(mean-float(row['video_logit']))<1e-6,'Video aggregation mismatch')
            score=float(np.exp(-np.logaddexp(0.,-mean)))
            check(abs(score-float(row['video_score']))<1e-6,'Score aggregation mismatch')
        cohort=(bool(result['smoke']),result['split_identity'],backbone,result['round_id'],result['partition'])
        provenance=(signature,alignment)
        if cohort in references:check(references[cohort]==provenance,f'Incompatible comparison: {path}')
        references[cohort]=provenance
        unique=cohort+(mode,);check(unique not in seen,'Duplicate method/round results');seen.add(unique)
        costpath=path.parent.parent/'cost.json';cost=read_json(costpath) if costpath.exists() else {}
        entries.append(dict(smoke=result['smoke'],split_identity=result['split_identity'],backbone=backbone,mode=mode,
                    round=result['round_id'],partition=result['partition'],auroc=result['auroc'],f1=result['f1'],
                    training_seconds=cost.get('training_seconds'),trainable_parameters=cost.get('trainable_parameters',769 if mode=='frozen' else None),
                    peak_gpu_memory_bytes=cost.get('peak_gpu_memory_bytes'),metrics_path=str(path)))
    check(entries,'No evaluated runs found')
    dest=output/'comparison';save_csv(dest/'round_values.csv',entries,list(entries[0]))
    groups=defaultdict(list)
    for row in entries:groups[(row['smoke'],row['split_identity'],row['backbone'],row['mode'],row['partition'])].append(row)
    averages=[]
    for key,rows in groups.items():
        averages.append(dict(zip(['smoke','split_identity','backbone','mode','partition'],key),rounds=[r['round'] for r in rows],
                             auroc_mean=float(np.mean([r['auroc'] for r in rows])) if all(r['auroc'] is not None for r in rows) else None,
                             f1_mean=float(np.mean([r['f1'] for r in rows])) if all(r['f1'] is not None for r in rows) else None))
    save_json(dest/'comparison.json',{'round_values':entries,'averages':averages,
             'note':'Descriptive round means, no confidence intervals. Compare methods only over identical listed rounds. Smoke is pipeline verification, not research evidence. Frozen training cost unavailable; null is not zero.'})
    import matplotlib.pyplot as plt
    cohorts=sorted({(r['smoke'],r['split_identity']) for r in entries})
    for smoke,split in cohorts:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,partition in zip(axes,('test_seen','test_unseen')):
            keys=[]
            for backbone in ('clip','dinov2','dinov3'):
                for mode in ('frozen','partial','lora'):
                    rows=[r for r in entries if (r['smoke'],r['split_identity'],r['partition'],r['backbone'],r['mode'])==(smoke,split,partition,backbone,mode)]
                    if not rows:continue
                    x=len(keys);keys.append(f'{backbone}\n{mode}')
                    valid=[r for r in rows if r['auroc'] is not None]
                    if valid:
                        ax.scatter([x]*len(valid),[r['auroc'] for r in valid],s=28)
                        ax.plot([x-.18,x+.18],[np.mean([r['auroc'] for r in valid])]*2,color='black')
                        for r in valid:ax.annotate(r['round'],(x,r['auroc']),fontsize=6,xytext=(3,3),textcoords='offset points')
            ax.set(xticks=range(len(keys)),xticklabels=keys,ylim=(0,1),ylabel='Video AUROC',title=partition)
            ax.tick_params(axis='x',labelsize=7)
            ax.set_xlim(-.6,len(keys))
        fig.suptitle(('SMOKE verification only — ' if smoke else '')+'dots: individual rounds; bars: means (no confidence intervals)')
        finish(fig,dest/f'{"smoke" if smoke else "full"}_{split[:10]}_comparison.png')
    return entries
