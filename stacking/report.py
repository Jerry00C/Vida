"""Create human-readable tables and standard plots from sealed evaluations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score,log_loss
from . import pipeline as p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=p.config(args.config);root=Path(cfg['output_dir'])
    report=json.loads((root/'training_report.json').read_text())
    evaluation=json.loads((root/'evaluation/metrics.json').read_text());bundle=p.load_bundle(cfg)
    folder=root/'report';folder.mkdir(exist_ok=True)
    records=[]
    for view,result in evaluation['views'].items():
        records.append({'view':view,'method':'ABC stacking',**{k:v for k,v in result.items() if not isinstance(v,(dict,list))}})
        for branch,values in result['individual_branch_calibrated_metrics'].items():
            records.append({'view':view,'method':branch,**values,
                'raw_score_auroc':result['individual_branch_raw_score_auroc'][branch]})
    flat=pd.DataFrame(records).drop(columns=['confusion_matrix_real_generated'],errors='ignore')
    flat.to_csv(folder/'metrics_comparison.csv',index=False,encoding='utf-8-sig')
    data,_,_=p.evidence(cfg,{'in_domain_test','external_test'})
    data['score']=p.probability(bundle,data)
    data[['clip_id','label','family','split','group_id','real_pool','A','B','C','score']].to_csv(folder/'test_predictions_with_branches.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if evaluation.get('run_kind')!='research':
        plt.rcParams['figure.titlesize']=12
        original_save=plt.Figure.savefig
        def synthetic_save(fig,*a,**kw):
            fig.suptitle('SYNTHETIC DEMO - NOT A RESEARCH RESULT',color='red')
            return original_save(fig,*a,**kw)
        plt.Figure.savefig=synthetic_save
    main_views=['in_domain','cross_generator_known_source','cross_generator_new_source']
    frames=[data[data.split=='in_domain_test']]
    ext=data[data.split=='external_test'];fake=ext[ext.label==1]
    for pool in ['known_source','new_source']:frames.append(pd.concat([fake,ext[(ext.label==0)&(ext.real_pool==pool)]]))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    colors={'A':'#2563eb','B':'#ea580c','C':'#7c3aed','ABC':'#0f766e'}
    for ax,name,frame in zip(axes,main_views,frames):
        for branch in p.BRANCHES:
            fpr,tpr,_=roc_curve(frame.label,frame[branch]);auc=roc_auc_score(frame.label,frame[branch])
            ax.plot(fpr,tpr,label=f'{branch} raw: {auc:.4f}',color=colors[branch],alpha=.8)
        fpr,tpr,_=roc_curve(frame.label,frame.score)
        ax.plot(fpr,tpr,label=f'ABC: {roc_auc_score(frame.label,frame.score):.4f}',color=colors['ABC'],lw=2.5)
        ax.plot([0,1],[0,1],color='gray',ls='--',lw=.7);ax.set(xlabel='False positive rate',ylabel='True positive rate',title=name.replace('_',' '));ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(folder/'roc_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,name in zip(axes,main_views):
        matrix=np.array(evaluation['views'][name]['confusion_matrix_real_generated'])
        ax.imshow(matrix,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Generated'],yticklabels=['Real','Generated'],xlabel='Predicted',ylabel='Actual',title=name.replace('_',' '))
        for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black',fontsize=13)
    fig.tight_layout();fig.savefig(folder/'confusion_matrices.png',dpi=180);plt.close(fig)
    trials=report['cross_validation'];fig,ax=plt.subplots(figsize=(7,4))
    for fold in range(5):ax.plot([r['C'] for r in trials],[r['fold_log_loss'][fold] for r in trials],color='#94a3b8',alpha=.6,lw=.8)
    ax.plot([r['C'] for r in trials],[r['mean_log_loss'] for r in trials],marker='o',color='#0f766e',label='5-fold mean')
    if report['hyperparameters']['selected_C'] is not None:
        ax.axvline(report['hyperparameters']['selected_C'],ls='--',color='#ea580c',label='Selected LR C')
    if report.get('neural_candidate'):
        ax.axhline(report['neural_candidate']['mean_log_loss'],color='#7c3aed',ls=':',label='MLP CV mean')
    ax.set(xscale='log',xlabel='Inverse L2 regularization strength C',ylabel='Validation log loss',title='Fusion hyperparameter selection (training data only)');ax.legend()
    fig.tight_layout();fig.savefig(folder/'hyperparameter_selection.png',dpi=180);plt.close(fig)
    if report.get('neural_candidate'):
        neural=report['neural_candidate'];fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,fold in zip(axes.flat,neural['folds']):
            h=fold['history'];ax.plot([r['epoch'] for r in h],[r['training_loss'] for r in h],label='train BCE')
            ax.plot([r['epoch'] for r in h],[r['validation_loss'] for r in h],label='validation BCE')
            ax.axvline(fold['best_epoch'],ls='--',color='gray');ax.set(title=f"MLP fold {fold['fold']+1}",xlabel='Epoch',ylabel='Loss');ax.legend(fontsize=8)
        h=neural['final_training_history'];axes[1,2].plot([r['epoch'] for r in h],[r['training_loss'] for r in h]);axes[1,2].set(title=f"MLP full-train refit: {neural['final_epochs']} epochs",xlabel='Epoch',ylabel='Train BCE')
        fig.tight_layout();fig.savefig(folder/'mlp_epoch_curves.png',dpi=180);plt.close(fig)
    # This is cached-score fusion latency only; no raw-video decode/backbone cost.
    sample=data[list(p.BRANCHES)].iloc[:1]
    for _ in range(20):p.probability(bundle,sample)
    times=[]
    for _ in range(200):
        t=time.perf_counter();p.probability(bundle,sample);times.append((time.perf_counter()-t)*1000)
    latency={'scope':'cached A/B/C scalar scores -> calibrated probability; excludes video decoding, branch inference, model loading, and Python startup',
        'batch_size':1,'warmup_iterations':20,'measured_iterations':200,
        'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95))}
    p.save_json(folder/'fusion_latency.json',latency)
    lines=['# VIDA 三路融合正式实验结果','',
        '本结果来自真实三路模型的新视频推理及新训练的融合器；不是模拟分数。日期：2026-10-30。','',
        '## 训练设置','',
        f"融合输入：A video_logit、B raw_margin、C anomaly score；只使用三列连续分数。训练 {report['fusion_training_rows']:,} 条；概率校准 {report['probability_fit_rows']} 条；阈值选择 {report['threshold_selection_rows']} 条。测试不参与调参。",
        '',f"最终选定模型：{report['selected_model_kind']}。五折分组验证只使用融合训练数据，以平均 log loss 最低选定。"+
            (f"MLP 最终全训练集拟合 {report['hyperparameters']['epoch']} 个 epoch，详细神经网络超参数见下一节。作为对照的 LR 使用 lbfgs，C 网格 [0.001,0.01,0.1,1,10,100]，max_iter=2000，tol=1e-8，seed=42。" if report['selected_model_kind']=='MLP 3-8-1' else
             f"最终 LR C={report['hyperparameters']['selected_C']}，solver=lbfgs，max_iter=2000，tol=1e-8，seed=42，实际求解器迭代 {report['hyperparameters']['actual_solver_iterations']} 次。LR 不使用神经网络 epoch。"),
        '',f"概率校准为独立 Platt logistic；固定分类阈值 {evaluation['frozen_threshold']:.8f}。三路模型没有重训。",
        '', '## 融合模型最终测试','',
        '| 测试视图 | 条数 | AUROC（95% group bootstrap CI） | Macro F1 | Balanced accuracy | Real FPR |',
        '|---|---:|---:|---:|---:|---:|']
    if evaluation.get('run_kind')=='research':
        known=evaluation['views']['cross_generator_known_source']
        sr=evaluation['views']['Sora_known_source'];kr=evaluation['views']['Kling_known_source']
        observations=[
            '## 本次结果必须说明的发现','',
            f"域内 640 条的 AUROC/Macro F1 为 1.0000，但这不能外推为所有新生成器都能可靠检出。冻结阈值为 {evaluation['frozen_threshold']:.8f}，其来源是已知生成器的校准集。",
            '',f"**Sora 仅检出 {sr['confusion_matrix_real_generated'][1][1]}/{sr['generated']}，Kling 仅检出 {kr['confusion_matrix_real_generated'][1][1]}/{kr['generated']}**；对应 Macro F1={sr['macro_f1']:.4f}/{kr['macro_f1']:.4f}。ModelScope 的 600 条较容易样本使跨生成器汇总指标显得较高，必须与逐来源结果一起阅读。",
            '',f"已知真实来源下，A 单路原始 AUROC={known['individual_branch_raw_score_auroc']['A']:.4f}，融合 AUROC={known['auroc']:.4f}；不能声称融合在所有测试场景均优于 A。Sora 配 UCF101 时融合 AUROC={evaluation['views']['Sora_new_source']['auroc']:.4f}，还暴露真实来源迁移问题。",
            '', 'AUROC 衡量排序，固定阈值下的检出率衡量实际分类；较高 AUROC 不能抵消漏检。当前现象与分数分布/校准的迁移不足相符，但不能仅据此断言降低阈值就能解决。没有利用已打开的测试集调阈值或重新选模型。',
            '',f"三家外部生成器的宏平均 AUROC：已知真实来源 {np.mean([evaluation['views'][f+'_known_source']['auroc'] for f in p.EXTERNAL]):.4f}；新真实来源 {np.mean([evaluation['views'][f+'_new_source']['auroc'] for f in p.EXTERNAL]):.4f}。两组共用同一批生成视频。",'',
            '后续改进应在开发数据内设计面向生成器迁移的验证和校准，并保留新的独立最终测试；不能在本次测试结果上反复选阈值后继续将其当盲测。',
            '', '## 训练设置']
        start=lines.index('## 训练设置');lines[start:start+1]=observations
    for name,result in evaluation['views'].items():
        ci=result['auroc_group_bootstrap_95ci'];cit=f'{ci[0]:.4f}–{ci[1]:.4f}' if ci else 'n/a'
        lines.append(f"| {name} | {result['n']} | {result['auroc']:.4f} ({cit}) | {result['macro_f1']:.4f} | {result['balanced_accuracy']:.4f} | {result['real_fpr']:.4f} |")
    lines+=['',f"三家外部生成器对已知真实来源的宏平均 AUROC：{evaluation['mean_heldout_generator_auroc_known_real']:.4f}。",'',
        '## 同一测试集的单路对照','',
        '单路阈值和概率映射也只在相同 calibration 角色上拟合。Raw AUROC 不依赖阈值；单路 calibrated Macro F1 采用各自固定阈值，与同学旧报告的 0.5 阈值/F1 定义可能不同。','',
        '| 测试视图 | 方法 | Raw AUROC / fusion AUROC | Macro F1 |','|---|---|---:|---:|']
    for name in main_views:
        result=evaluation['views'][name]
        for branch in p.BRANCHES:
            lines.append(f"| {name} | {branch} | {result['individual_branch_raw_score_auroc'][branch]:.4f} | {result['individual_branch_calibrated_metrics'][branch]['macro_f1']:.4f} |")
        lines.append(f"| {name} | ABC stacking | {result['auroc']:.4f} | {result['macro_f1']:.4f} |")
    lines+=['','## 解读边界','',
        '- 原 32K 是项目开发池；本次沿用各自实际完成的 checkpoint。不能将 A/C 实际训练条数改写为 32K。',
        '- 测试集现在已经用于最终评估，不能再按其成绩调整本模型后仍称一次盲测。',
        '- 两种外部真实来源视图共用同一批 fake，不是两次独立实验。Sora 只有 44 条 fake，结论需保留小样本限制。',
        '- Bootstrap CI 反映当前样本的重采样波动；完美分类样本可能出现 [1,1] 的退化区间，不能据此保证总体零错误。',
        '- UCF101 是新增真实来源，不属于 GenVidBench 官方数据；对该来源的变化也可能受动作场景、分辨率、压缩等因素影响。',
        '- 已知 ID/source group/字节等隔离检查通过；公开数据缺少完整 prompt/parent 映射，不能证明不存在所有语义或预训练内容重叠。',
        '- A 本地原视频接口使用固定基座、权重、processor 与八帧 floor policy。5 条旧视频判断一致，但旧 CSV 连续分数最大差 2.0581 logit，未实现数值等价；缺少旧 PNG 的字节级复核，原因尚未确定。新增套件全部统一在记录的本地环境评分，不混入旧 CSV。',
        '', '## 时间和文件','',
        f"缓存分数上的融合层 batch=1：median {latency['median_ms']:.3f} ms，P95 {latency['p95_ms']:.3f} ms。此数值不含解码和三路推理，不能称端到端视频检测延迟。",
        '', '| 文件 | 用途 |','|---|---|',
        '| `../stacker.joblib` | 可加载的正式融合模型、标准化器、概率校准器、阈值和单路对照 |',
        '| `../MODEL_LOCK.json` | 模型及数据身份和三路 lineage，核验冻结后未变动 |',
        '| `../training_report.json` | 超参数、30 次分组 CV 的逐折成绩、系数和收敛情况 |',
        '| `../fusion_train_cv.csv` | 训练样本 CV fold 与被选超参数的 OOF 概率，用于开发诊断 |',
        '| `../calibration_roles.csv` | 概率校准与阈值选择的实际样本角色 |',
        '| `../evaluation/metrics.json` | 完整正式测试指标和 AUROC 分组 bootstrap CI |',
        '| `metrics_comparison.csv` | 适合导入 Excel 的各视图、各方法指标表 |',
        '| `test_predictions_with_branches.csv` | 逐条测试 ID、标签、来源、三路分数及融合概率 |',
        '| `roc_comparison.png` | 三个主要测试视图的 ROC 对照 |',
        '| `confusion_matrices.png` | 固定阈值下混淆矩阵 |',
        '| `hyperparameter_selection.png` | 训练内部 CV log loss，非 epoch loss |',
        '| `mlp_epoch_curves.png` | 若启用 MLP 比较：五折实际 train/validation BCE 曲线及全训练集 refit 曲线 |',
        '| `fusion_latency.json` | 仅融合层、缓存分数上的计时范围及统计 |','']
    (folder/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    if report.get('neural_candidate'):
        neural=report['neural_candidate'];path=folder/'RESULTS_zh.md'
        text=path.read_text(encoding='utf-8')
        text=text.replace('## 融合模型最终测试',
            '## Epoch 模型候选的实际训练\n\n'
            'MLP：3→8→1、ReLU、dropout=0.1，41 个可训练参数；AdamW lr=0.001，weight_decay=0.001，batch=64，max_epochs=200，patience=20，min_delta=0.0001，seed=42。\n\n'
            f"五折最佳 epoch：{[r['best_epoch'] for r in neural['folds']]}；全 {report['fusion_training_rows']:,} 条 refit 训练 {neural['final_epochs']} epoch。MLP CV mean log loss={neural['mean_log_loss']:.6f}；最佳 LR CV mean log loss={min(r['mean_log_loss'] for r in report['cross_validation']):.6f}。这些是选模验证成绩，不能作为最终测试性能。\n\n"
            'MLP 通过内部验证选择 epoch，这使其 CV 分数包含 epoch 选择；独立封存测试仍是最终依据。不因最终测试结果改选 LR/MLP。\n\n## 融合模型最终测试')
        path.write_text(text,encoding='utf-8')
    if evaluation.get('run_kind')!='research':
        path=folder/'RESULTS_zh.md';text=path.read_text(encoding='utf-8')
        text=text.replace('# VIDA 三路融合正式实验结果','# 合成数据接口验证：非研究结果')
        text=text.replace('本结果来自真实三路模型的新视频推理及新训练的融合器；不是模拟分数。日期：2026-10-30。',
            '**以下全部输入为人工生成的模拟分数，只用于检查程序。不是实际视频模型效果，严禁作为项目正式成绩。**')
        path.write_text(text,encoding='utf-8')
    print(str(folder/'RESULTS_zh.md'))

if __name__=='__main__':main()
