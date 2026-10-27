"""Review, report and package the completed exhaustive generator rotation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np

from branch_c.cache import file_sha256
from branch_c.config import write_json
from branch_c.pipeline import predict_video
from generator_rotation import GENERATORS, PROTOCOL, now, read_csv, write_csv
from review_vida_cases import sheet


def export_cases(root,splits):
    out=root/'report';out.mkdir(exist_ok=True)
    if (out/'cases.json').exists():raise ValueError('Preserve existing case observations')
    metrics=[r for r in read_csv(root/'metrics_by_round.csv') if r['view']=='unseen_generators']
    order=sorted(metrics,key=lambda r:(float(r['auroc']),r['round_id']))
    manifest={r['clip_id']:r for r in read_csv(splits/'master_manifest.csv')}
    cases=[]
    for round_name in dict.fromkeys([order[0]['round_id'],order[-1]['round_id']]):
        ref=json.loads((root/round_name/'model_reference.json').read_text())
        threshold=ref['threshold'];scores=read_csv(root/round_name/'unseen_generators/test_scores.csv')
        fp=sorted([r for r in scores if r['label']=='0' and float(r['branch_c_score'])>threshold],key=lambda r:(-float(r['branch_c_score']),r['clip_id']))
        chosen=[(fp[0],'FP')] if fp else []
        for g in sorted({r['generator'] for r in scores if r['label']=='1'}):
            gen=[r for r in scores if r['generator']==g]
            fn=[r for r in gen if float(r['branch_c_score'])<=threshold]
            chosen.append((min(fn or gen,key=lambda r:(float(r['branch_c_score']),r['clip_id'])),'FN' if fn else 'TP_reference_no_FN'))
        for row,error in chosen:
            case=f'case_{len(cases)+1:02d}';r=manifest[row['clip_id']]
            sheet(r['path'],out/f'{case}.png')
            prediction=predict_video(root/round_name/ref['model_path'],r['path'],out/f'{case}_prediction.json')
            if not np.isclose(prediction['anomaly_score'],float(row['branch_c_score']),atol=1e-10,rtol=0):raise ValueError('Case prediction differs from frozen test score')
            cases.append({'case_id':case,'round_id':round_name,'clip_id':row['clip_id'],'family':r['family'],
                          'error_type':error,'score':float(row['branch_c_score']),'threshold':threshold,
                          'image':f'{case}.png','content_sha256':row['content_sha256'],
                          'review_status':'pending_sampled_frame_inspection','observation':''})
    write_json(out/'cases.json',cases)
    print('Exported cases',len(cases),flush=True)


def build_report(root,splits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verification=json.loads((root/'verification.json').read_text())
    if verification['status']!='passed':raise ValueError('Numerical verification required')
    status=json.loads((root/'status.json').read_text())
    if status['current_stage']!='complete':raise ValueError('Formal run not complete')
    out=root/'report';out.mkdir(exist_ok=True)
    cases=json.loads((out/'cases.json').read_text())
    if any(not c['observation'].strip() for c in cases):raise ValueError('Complete direct frame observations before reporting')
    records=read_csv(root/'metrics_by_round.csv')
    summaries={}
    for view in ['seen_generators','unseen_generators']:
        rows=[r for r in records if r['view']==view]
        summaries[view]={k:{'mean':float(np.mean([float(r[k]) for r in rows])),
                            'median':float(np.median([float(r[k]) for r in rows])),
                            'min':min(float(r[k]) for r in rows),'max':max(float(r[k]) for r in rows)}
                         for k in ['auroc','macro_f1','balanced_accuracy','recall_generated','false_positive_rate']}
    unseen=[r for r in records if r['view']=='unseen_generators']
    for row in unseen:
        name=row['round_id']
        selected=json.loads((root/name/'selection_report.json').read_text())
        config=json.loads((splits/name/'config.json').read_text())
        config['model']=selected['ranked_candidates'][0]['model_config']
        write_json(root/name/'selected_config.json',config)
    plt.rcParams.update({'font.size':10,'savefig.facecolor':'white','figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.5),layout='constrained')
    for ax,metric,title in zip(axes,['auroc','macro_f1'],['Unseen-generator AUROC','Unseen-generator macro F1']):
        data=np.full((8,8),np.nan)
        for r in unseen:
            i=GENERATORS.index(r['heldout_1']);j=GENERATORS.index(r['heldout_2']);data[i,j]=data[j,i]=float(r[metric])
        im=ax.imshow(data,vmin=0,vmax=1,cmap='YlGnBu')
        ax.set_xticks(range(8),GENERATORS,rotation=45,ha='right',fontsize=9);ax.set_yticks(range(8),GENERATORS,fontsize=9)
        ax.set_title(title)
        for i in range(8):
            for j in range(8):
                if i!=j:ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if data[i,j]>.65 else '#172a34')
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/'holdout_pairs.png');plt.close(fig)
    gen_values=defaultdict(list);source_values=defaultdict(list)
    for r in unseen:
        report=json.loads((root/r['round_id']/'unseen_generators/test_report.json').read_text())
        for g,m in report['per_generator'].items():gen_values[g].append(m)
        for s,m in report['per_real_source'].items():source_values[s].append(m['false_positive_rate'])
    generator_summary=[]
    for g in GENERATORS:
        values=gen_values[g]
        if len(values)!=7:raise ValueError('Every generator must be held out seven times')
        generator_summary.append({'generator':g,'heldout_rounds':len(values),
            'auroc_mean':float(np.mean([v['auroc'] for v in values])),
            'auroc_min':min(v['auroc'] for v in values),'auroc_max':max(v['auroc'] for v in values),
            'recall_mean':float(np.mean([v['recall_generated'] for v in values]))})
    write_csv(out/'generator_summary.csv',generator_summary)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
    positions=np.arange(8)
    means=np.array([g['auroc_mean'] for g in generator_summary])
    lo=np.array([g['auroc_min'] for g in generator_summary]);hi=np.array([g['auroc_max'] for g in generator_summary])
    if np.any(means<lo-1e-12) or np.any(means>hi+1e-12):raise ValueError('Invalid generator summary range')
    # Identical seven values can average a few ulps outside their range.
    axes[0].errorbar(means,positions,xerr=[np.maximum(0,means-lo),np.maximum(0,hi-means)],fmt='o',capsize=3,color='#256d5a')
    axes[0].set_yticks(positions,GENERATORS);axes[0].set(xlim=(0,1),xlabel='AUROC (mean; whiskers are min–max, not CI)',title='Seven dependent holdouts per generator')
    axes[1].barh(positions,[g['recall_mean'] for g in generator_summary],color='#bc4e2a')
    axes[1].set_yticks(positions,GENERATORS);axes[1].set(xlim=(0,1),xlabel='Generated recall (mean across seven holdouts)',title='Frozen real-calibration threshold')
    fig.savefig(out/'generator_holdouts.png');plt.close(fig)
    counts=Counter(r['candidate'] for r in unseen)
    named_i2v={}
    for n in [0,1,2]:
        rows=[r for r in unseen if len({r['heldout_1'],r['heldout_2']}&{'SVD','MuseV'})==n]
        named_i2v[str(n)]={'n_rounds':len(rows),'auroc_mean':float(np.mean([float(r['auroc']) for r in rows])),
                          'macro_f1_mean':float(np.mean([float(r['macro_f1']) for r in rows]))}
    e=html.escape
    metricrows=''.join(f"<tr><td>{e(view)}</td><td>{s['auroc']['mean']:.3f}（{s['auroc']['min']:.3f}–{s['auroc']['max']:.3f}）</td><td>{s['macro_f1']['mean']:.3f}（{s['macro_f1']['min']:.3f}–{s['macro_f1']['max']:.3f}）</td><td>{s['recall_generated']['mean']:.1%}</td><td>{s['false_positive_rate']['mean']:.1%}</td></tr>" for view,s in summaries.items())
    roundrows=''.join(f"<tr><td>{r['round_id']}</td><td>{e(r['heldout_1'])} + {e(r['heldout_2'])}</td><td>{e(r['candidate'])}</td><td>{float(r['auroc']):.3f} [{float(r['auroc_ci_low']):.3f}, {float(r['auroc_ci_high']):.3f}]</td><td>{float(r['macro_f1']):.3f}</td><td><a href='../{r['round_id']}/unseen_generators/test_report.json'>原始报告</a></td></tr>" for r in unseen)
    genrows=''.join(f"<tr><td>{e(g['generator'])}</td><td>7</td><td>{g['auroc_mean']:.3f}</td><td>{g['auroc_min']:.3f}–{g['auroc_max']:.3f}</td><td>{g['recall_mean']:.1%}</td></tr>" for g in generator_summary)
    caseshtml=''.join(f"<article><h3>{c['case_id']} · {c['round_id']} · {e(c['family'])} · {e(c['error_type'])}</h3><p><code>{e(c['clip_id'])}</code>；分数 {c['score']:.5g}，阈值 {c['threshold']:.5g}。</p><img src='{c['image']}' alt='按时间排列的八个原始抽样帧'><p>{e(c['observation'])}</p></article>" for c in cases)
    resource=''
    for candidate in sorted(counts):
        r=json.loads((root/'shared_models'/candidate/'runtime.json').read_text());s=r['summary']
        resource+=f"<li>{e(candidate)}：3 个新进程，10 秒真实视频；含模型加载推理中位数 {s['total_seconds_including_model_load']['median']:.3f} 秒；进程总耗时 {s['process_wall_seconds']['median']:.3f} 秒；峰值 RSS {s['peak_rss_bytes']['median']/1048576:.1f} MiB；模型包 {r['model_size_bytes']/1048576:.2f} MiB。</li>"
    body=f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Branch C：6 已见 / 2 未见生成器轮换</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#192e35;line-height:1.75}}h1{{font-size:30px}}h2{{margin-top:38px;border-top:1px solid #dce5e5;padding-top:20px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;min-width:600px;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce5e5;text-align:left}}th{{background:#eef4f1}}.scroll{{max-width:100%;overflow-x:auto}}code{{word-break:break-all;font-size:13px}}a{{color:#216c58}}.lead{{padding:18px;background:#edf4f1}}article{{margin:28px 0}}@media print{{img,article{{break-inside:avoid}}}}</style>
<h1>Branch C：6 已见 / 2 未见生成器轮换</h1><p>Group 42 · Lang Heran · 2026 年 10 月 15 日 · VIDA-Guard32K 重新划分分析</p>
<p class='lead'>已完成全部 28 种留出配对，每轮严格六个已见、两个未见生成器。每家生成器恰好留出七次。以下是同一已评价素材池的新协议分析，不能冒称新的独立盲测。C 只拟合真实视频；是否改善融合仍需 A/B 输出做配对消融。</p>
<h2>协议与样本</h2><p>沿父视频、共享提示词与真实/生成配对的关联组，按来源组合分层，seed 42，先分成 60% train、10% model_selection、10% calibration、20% test。训练折号 0–4 固定，A/B/C 共用。每轮仅六个已见生成器的开发行进入清单，两个留出生成器的开发行隔离并记录在 excluded.csv。</p>
<p>每轮真实 train 9,600（Vript 7,200，HD-VG 2,400），生成 train 为 6,000、7,200 或 8,400，总训练数为 15,600、16,800 或 18,000；差异来自原始来源样本数不同。model_selection 与 calibration 各 1,200，均含真实 600（两来源各 300）和六家生成器各 100。最终阈值仅来自 600 个真实 calibration。</p>
<p>每家生成器固定 200 个 test。未见评价每轮 800（真实 400、生成 400），已见评价每轮 2,400（真实 1,200、生成 1,200）；两套真实池互不重叠，均为两来源等量且跨轮固定。三种 VidProM 来源共用选中提示词组；关联组没有跨 train/val/test、验证角色或 OOF 折。</p>
<h2>28 轮结果</h2><div class='scroll'><table><tr><th>测试视图</th><th>AUROC 均值（最小–最大）</th><th>macro F1 均值（最小–最大）</th><th>生成召回均值</th><th>真实误报均值</th></tr>{metricrows}</table></div><p>28 轮使用大量相同视频与共同真实参考数据，轮次相互依赖；均值、最小和最大是描述性汇总，不能按 28 个独立实验计算置信区间。每轮原始 JSON 另有固定模型、固定阈值下 1,000 次分组 bootstrap 区间，不包含重新训练/模型选择的不确定性。</p><img src='holdout_pairs.png' alt='全部留出生成器配对的AUROC与macroF1热图'>
<div class='scroll'><table><tr><th>轮次</th><th>本轮未见生成器</th><th>选定候选</th><th>未见 AUROC 与95%区间</th><th>macro F1</th><th>记录</th></tr>{roundrows}</table></div>
<h2>按留出生成器汇总</h2><div class='scroll'><table><tr><th>生成器</th><th>留出轮数</th><th>AUROC 均值</th><th>轮次范围</th><th>召回均值</th></tr>{genrows}</table></div><img src='generator_holdouts.png' alt='每家生成器七次相关留出的AUROC均值范围和召回'><p>每家生成器的同一批 200 个视频重复出现在七个留出配对中，不能当成 1,400 个独立视频。各生成器均相对于同一 400 个真实测试视频评价。误差棒是七轮最小–最大，不是置信区间。</p>
<h2>模型选择与可证明的复用</h2><p>每轮独立使用其六家已见生成器的 model_selection，按 AUROC、macro F1、预定顺序选择相同六候选。选择次数：{e(str(dict(counts)))}。所有选择与 OOF 在任何新测试指标计算前冻结。</p><p>标准化器和异常模型只拟合真实视频，所以当真实拟合 ID、内容哈希、特征签名与算法参数相同时，可复用同一数值模型。每轮仍单独导出完整 OOF、验证与测试分数和选择记录。公共五折中每个模型拟合 7,680 个真实视频，留出组不进入标准化或异常拟合；独立数值复核重新拟合全部选中候选的五折模型，并精确复现所有输出。</p><p>正式特征仍是 native_stride 和37维时间摘要，没有利用新测试结果改特征。窗口模型独立且只拟合真实窗口。分数越高越异常，不是生成概率；OOF 与最终模型的原始尺度可能不同，融合校准需核对，不能直接平均不同模型分数。</p>
<h2>输入机制与结论边界</h2><p>轮换避免某些名字永久只出现在测试，并不能自动消除输入机制、帧率、时长或内容分布的差异。SVD/MuseV 是明确的图像条件来源；交付说明也记载 Mora 包含 SVD，部分来源存在图像条件配对。不能直接把其余六家均认证为纯文生视频，也不能把八个名字当成独立底层架构。报告另保存按留出 SVD/MuseV 数量分组的描述性汇总，但不将它当作纯生成方式因果实验。</p><p>旧 run_02 使用4已见/4未见、不同真实训练来源和测试分布；新方案使用6已见/2未见、两来源训练及固定平衡测试池。二者的指标不能作为控制其他变量后的直接提升。未知语义重复、频道未完全隔离、短片频率分辨率及源格式捷径仍存在。</p>
<h2>确定性错误案例</h2><p>按预先声明的规则选择未见AUROC最低和最高轮次，各取最强真实误报与每个留出生成器最低分漏检；没有漏检时保留TP参考。以下只记录八帧可见内容，不是完整视频人工审阅，也不证明误判原因。每轮完整 FP/FN 另存 error_review.csv。</p>{caseshtml}
<h2>CPU资源与核验</h2><ul>{resource}</ul><p>资源为本机CPU测量，OS文件缓存不受控制，不能冒称GPU结果。156项软件测试通过，含新分组bootstrap与原实现的数值等价测试。核验覆盖28轮、56个测试视图、所有分数与五折数值复现；旧实验模型SHA保持一致。</p>
<h2>交给组员与复现</h2><p>每轮用同名 round_XX 的 train.csv、val.csv、test.csv 与 fold_assignments.csv，按 clip_id 合并 C 的 oof_scores.csv、validation_scores.csv 和两套 test_scores.csv。不要混用不同轮次的训练与测试。CSV 的 video_path 相对 media 根目录；其他机器按该字段定位文件，保留ID、组、折和内容SHA。</p><p>每轮 model_reference.json 指向 shared_models 中的实际模型及SHA。共享模型的protocol为公共真实参考池；每轮分数CSV保留单独的轮次protocol和选定模型ID。A/B为监督分支，需要按轮使用自己的六家已见生成器训练，不能照搬C的真实类复用规则。</p><p>源码入口：scripts/generator_rotation.py 生成新清单；scripts/run_generator_rotation.py 运行并独立核验；scripts/report_generator_rotation.py 导出案例、报告和交接包。原始视频不进入交接包。尚需全组完成A/B、融合校准和A+B对比A+B+C的配对消融。</p><p><a href='../verification.json'>数值核验</a> · <a href='../experiment_plan.json'>实验冻结计划</a> · <a href='../model_freeze.json'>全部选择冻结</a> · <a href='../metrics_by_round.csv'>全部轮次指标</a></p></html>"""
    body=body.replace("<h2>确定性错误案例</h2>","<p>案例检查额外发现：一个来源标记 real 的片段全部119个原始帧均为黑屏；另一个来源标记 real 的HD-VG片段包含动画与图解。标签继承素材来源，不能冒称逐条确认了自然摄像真实性。两者都按冻结协议保留，不在看过测试后删样本或改标签；细节见 input_case_findings.json。</p><h2>确定性错误案例</h2>")
    (out/'index.html').write_text(body,encoding='utf-8')
    summary={'status':'complete','protocol':PROTOCOL,'n_rounds':28,'metrics':summaries,'selected_candidates':dict(counts),
             'generators':generator_summary,'named_svd_musev_holdout_counts':named_i2v,'case_count':len(cases),
             'source_fpr_mean':{s:float(np.mean(v)) for s,v in source_values.items()},
             'interpretation':'Dependent repeated holdouts of previously evaluated data; no fresh blind-test, no iid28roundCI, no fusion gain claim'}
    write_json(out/'summary.json',summary)
    doc=f"""# Branch C：6 已见 / 2 未见生成器轮换交接

2026 年 10 月 15 日完成。完整结果为 `{root}/report/index.html`，新清单为 `{splits}`。共28轮，覆盖8选2全部配对，每家留出7次；原32K交付与run_02保留。

分组：来源/提示词/真实-生成配对组，60/10/10/20用于train/model_selection/calibration/test，seed42。每轮真实train9600，生成train6000/7200/8400；验证两角色各1200，真假各600；未见test800、已见test2400，真假平衡、真实两来源等量。每家固定200测试视频，未见生成器开发行隔离，所有配额外行在excluded.csv中解释。

未见28轮 AUROC 均值 {summaries['unseen_generators']['auroc']['mean']:.3f}，范围 {summaries['unseen_generators']['auroc']['min']:.3f}–{summaries['unseen_generators']['auroc']['max']:.3f}；macro F1 均值 {summaries['unseen_generators']['macro_f1']['mean']:.3f}。轮次依赖，不是28个独立数据集；每轮区间为固定模型的组bootstrap。

## 组员接入

1. 选择相同round_XX，使用该轮manifest/train/val/test CSV和公共fold_assignments.csv；video_path相对media根目录。
2. C每轮文件在实验目录round_XX：oof_scores.csv供融合训练；validation_scores.csv中calibration供融合校准；seen_generators与unseen_generators各自test_scores.csv供最终评价。
3. 按clip_id一对一合并，核对protocol、label、fold_id、content_sha256。不要混轮次，不要用训练集内最终分数代替OOF。
4. model_reference.json指向实际shared_models/candidate/model.joblib。相同真实数据与参数允许C复用；A/B监督训练按每轮六家生成器独立执行。
5. 分数不是概率，OOF和最终SVM分数尺度可能不同；校准只用允许的calibration，不能回调测试阈值。

## 复现命令

在项目根目录、锁定依赖的Python3.12.14环境运行：

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from generator_rotation import build_protocol; build_protocol('data/VIDA-Guard32K/branch_c/manifest.csv','data/VIDA-Guard32K/rotation_reproduction')"
.venv/bin/python scripts/run_generator_rotation.py --splits-dir data/VIDA-Guard32K/rotation_reproduction --output-dir artifacts/vida_guard32k/rotation_reproduction
```

先用import_delivery.py映射本机media路径，保持clipID/组/哈希。实际运行会拒绝覆盖已有目录。报告另需requirements-report.txt。

## 限制

这是已经评价过的素材池重新划分，不能称新盲测。生成器条件机制与架构存在关联；Mora含SVD，六个其他来源不能直接认证为纯T2V。来源/帧率/时长/内容捷径及未知语义重复未全部消除。旧4+4结果与新6+2改变了多项协议，不能当成受控直接提升。案例只有抽样帧观察。A/B、融合及配对消融仍属全组联调工作。
"""
    doc+="\n案例另外发现来源标real的全黑片段与动画/图解片段，说明来源标签和输入信息量存在边界；本次保留固定测试样本、不事后删样本或改标签。详见report/input_case_findings.json。\n"
    Path('docs/GENERATOR_ROTATION_ZH.md').write_text(doc,encoding='utf-8')
    print(out/'index.html')


def package(root,splits,destination):
    if destination.exists():raise ValueError('Preserve existing handoff ZIP')
    visual=json.loads((root/'report/visual_review.json').read_text())
    if visual['status']!='passed' or any(file_sha256(root/'report'/p)!=h for p,h in visual['files_sha256'].items()):raise ValueError('Current report needs visual review')
    verification=json.loads((root/'verification.json').read_text())
    if verification['status']!='passed':raise ValueError('Numeric verification required')
    files=[]
    for directory in [Path('src'),Path('scripts'),Path('tests'),Path('docs'),root,splits]:
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo','.npz','.mp4','.zip'})
    files.extend(Path(p) for p in ['README.md','pyproject.toml','requirements-lock.txt','requirements-report.txt','.gitignore'])
    files=sorted(set(files));digests={str(p):file_sha256(p) for p in files}
    manifest={'created_at':now(),'protocol':PROTOCOL,'scope':'Branch C,28 rotations;source,grouped splits,models,OOF/val/testscores,verification,report;no rawvideos',
              'members_sha256':digests,'team_dependencies':'A/B training, fusion calibration and paired A+B versus A+B+C ablation'}
    destination.parent.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p))
        z.writestr('rotation_handoff_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for name,digest in digests.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('ZIP member changed')
    write_json(destination.with_suffix('.json'),{'status':'complete','path':str(destination),'bytes':destination.stat().st_size,
               'sha256':file_sha256(destination),'members':len(files)+1,'rounds':28})
    print(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='artifacts/vida_guard32k/rotation_6seen_2unseen_v1')
    p.add_argument('--splits-dir',default='data/VIDA-Guard32K/rotation_6seen_2unseen_v1')
    p.add_argument('--phase',required=True,choices=['cases','report','package'])
    p.add_argument('--output',default='output/branch_c_generator_rotation_6seen_2unseen.zip')
    a=p.parse_args();root=Path(a.run_dir);splits=Path(a.splits_dir)
    if a.phase=='cases':export_cases(root,splits)
    elif a.phase=='report':build_report(root,splits)
    else:package(root,splits,Path(a.output))
