"""Build an offline Chinese report and scientific plots from frozen results only."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

from branch_c.config import write_json
from branch_c.evaluation import metrics


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def build(args):
    root=Path(args.run_dir)
    out=root/"report"
    out.mkdir(exist_ok=True)
    verification=json.loads((root/"verification.json").read_text())
    if verification["status"]!="passed": raise ValueError("Formal verification must pass before report")
    reports={s:json.loads((root/s/"test_report.json").read_text()) for s in ("in_domain_test","cross_test")}
    raw={s:read_csv(root/s/"test_scores.csv") for s in reports}
    manifest={r["clip_id"]:r for r in read_csv(args.manifest)}
    selection=json.loads((root/"selection/selection_report.json").read_text())
    fit=json.loads((root/"selection/selected/fit_report.json").read_text())
    runtime=json.loads((root/"runtime.json").read_text())
    baseline_plan=json.loads((root/"reference_baseline_plan.json").read_text())
    freeze=json.loads((root/"model_freeze.json").read_text())
    if baseline_plan["declared_at"]>=freeze["frozen_at"]:raise ValueError("Reference must be declared before sealed tests")
    reference={subset:metrics([int(r["label"]) for r in records],np.zeros(len(records)),.5) for subset,records in raw.items()}
    random_plan=json.loads((root/"random_reference_plan.json").read_text())
    if random_plan["declared_at"]>=freeze["frozen_at"]:raise ValueError("Random reference declared after test freeze")
    ordered=sorted(r["clip_id"] for records in raw.values() for r in records)
    random_scores=dict(zip(ordered,np.random.default_rng(42).uniform(size=len(ordered)).tolist()))
    random_metrics={subset:metrics([int(r["label"]) for r in records],[random_scores[r["clip_id"]] for r in records],.5) for subset,records in raw.items()}
    write_json(out/"reference_baseline.json",{"majority_plan":baseline_plan,"majority_test_metrics":reference,
               "random_plan":random_plan,"random_test_metrics":random_metrics})
    with (out/"random_reference_scores.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["clip_id","label","evaluation_subset","random_score","seed","is_probability"])
        w.writeheader()
        for subset,records in raw.items():
            for r in records:w.writerow({"clip_id":r["clip_id"],"label":r["label"],"evaluation_subset":subset,"random_score":random_scores[r["clip_id"]],"seed":42,"is_probability":"false"})
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":150,"savefig.facecolor":"white"})
    colors={0:"#256d5a",1:"#bc4e2a"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,(subset,records) in zip(axes,raw.items()):
        y=np.array([int(r["label"]) for r in records]); s=np.array([float(r["branch_c_score"]) for r in records])
        fpr,tpr,_=roc_curve(y,s)
        auc=reports[subset]["metrics"]["auroc"]
        lo,hi=reports[subset]["ci_95"]["auroc"]
        ax.plot(fpr,tpr,color=colors[0],lw=2,label=f"AUROC {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})")
        ax.plot([0,1],[0,1],"--",color="#aaa",lw=1)
        m=reports[subset]["metrics"]
        ax.scatter([m["false_positive_rate"]],[m["recall_generated"]],color=colors[1],s=45,zorder=3,label="Fixed real-calibration threshold")
        ax.set(xlabel="False positive rate (real)",ylabel="True positive rate (generated)",title=subset,xlim=(0,1),ylim=(0,1))
        ax.legend(loc="lower right",fontsize=8)
    fig.savefig(out/"roc.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,(subset,records) in zip(axes,raw.items()):
        values=np.array([float(r["branch_c_score"]) for r in records]); y=np.array([int(r["label"]) for r in records])
        bins=np.linspace(values.min(),values.max()+1e-10,42)
        for label,name in [(0,"Real"),(1,"Generated")]:
            ax.hist(values[y==label],bins=bins,density=True,alpha=.5,color=colors[label],label=name)
        ax.axvline(fit["threshold"],color="#303640",ls="--",label="Fixed threshold")
        ax.set(xlabel="Anomaly score (higher = more anomalous; not probability)",ylabel="Density",title=subset)
        ax.legend(fontsize=8)
    fig.savefig(out/"scores.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,(subset,report) in zip(axes,reports.items()):
        names=list(report["per_generator"])
        values=[report["per_generator"][n]["auroc"] for n in names]
        intervals=[report["per_generator_ci_95"][n]["auroc"] for n in names]
        ax.errorbar(values,np.arange(len(names)),xerr=np.array([[v-c[0] for v,c in zip(values,intervals)],
                        [c[1]-v for v,c in zip(values,intervals)]]),fmt="o",capsize=4,color=colors[0])
        ax.axvline(.5,ls="--",color="#aaa")
        ax.set(yticks=np.arange(len(names)),yticklabels=names,xlabel="AUROC, 95% group bootstrap CI",title=subset,xlim=(0,1))
    fig.savefig(out/"generators.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.4),layout="constrained")
    for ax,(subset,report) in zip(axes,reports.items()):
        cm=np.asarray(report["metrics"]["confusion_matrix_real_generated"])
        ax.imshow(cm,cmap="Blues")
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center",color="white" if v>cm.max()/2 else "#17222d",fontsize=16)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Real","Anomalous"],yticklabels=["Real","Generated"],xlabel="Fixed-threshold decision",ylabel="Ground truth",title=subset)
    fig.savefig(out/"confusion.png");plt.close(fig)
    diagnostics={}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for k,(subset,records) in enumerate(raw.items()):
        values=np.array([float(r["branch_c_score"]) for r in records]); y=np.array([int(r["label"]) for r in records])
        diagnostics[subset]={}
        for j,field in enumerate(["duration_seconds","native_fps"]):
            covariate=np.array([float(manifest[r["clip_id"]][field]) for r in records])
            ax=axes[k,j]
            diagnostics[subset][field]={}
            for label,name in [(0,"Real"),(1,"Generated")]:
                mask=y==label
                ax.scatter(covariate[mask],values[mask],s=5,alpha=.18,color=colors[label],label=name,rasterized=True)
                correlation=spearmanr(covariate[mask],values[mask]).statistic if np.ptp(covariate[mask]) else None
                diagnostics[subset][field][name]={"n":int(mask.sum()),"spearman_rho":float(correlation) if correlation is not None and np.isfinite(correlation) else None}
            ax.axhline(fit["threshold"],ls="--",color="#555",lw=1)
            ax.set(xlabel=field,ylabel="Anomaly score",title=subset)
            ax.legend(fontsize=8,markerscale=3)
    fig.savefig(out/"source_diagnostics.png");plt.close(fig)
    write_json(out/"source_diagnostics.json",{"associations":diagnostics,"interpretation":"Post-test descriptive associations only; not causal effects or model selection; duration/fps/source remain confounded"})
    oof_real=np.array([float(r["branch_c_score"]) for r in read_csv(root/"oof/oof_scores.csv") if r["label"]=="0"])
    final_real=np.array([float(r["branch_c_score"]) for r in read_csv(root/"selection/selected/validation_scores.csv")
                         if r["label"]=="0" and r["validation_role"]=="calibration"])
    scale={name:{"n":len(values),"mean":float(values.mean()),"std":float(values.std()),
                 "q01_q05_q50_q95_q99":np.quantile(values,[.01,.05,.5,.95,.99]).tolist()}
           for name,values in [("real_oof_6720_fit",oof_real),("real_calibration_8400_fit",final_real)]}
    write_json(out/"oof_scale_check.json",{"distributions":scale,"note":"Descriptive out-of-sample distributions from different real clips and fit sizes; no causal attribution or new calibration fitted. SVM raw scores are not probability-scale inputs."})
    fig,ax=plt.subplots(figsize=(9,4.2),layout="constrained")
    bins=np.linspace(min(oof_real.min(),final_real.min()),max(oof_real.max(),final_real.max())+1e-10,45)
    ax.hist(oof_real,bins=bins,density=True,alpha=.5,color="#256d5a",label="Real OOF (6,720 fit clips per fold)")
    ax.hist(final_real,bins=bins,density=True,alpha=.5,color="#bc4e2a",label="Real calibration (8,400 fit clips)")
    ax.set(xlabel="Raw anomaly score (not probability)",ylabel="Density",title="Fusion input scale check; different out-of-sample real clips")
    ax.legend(fontsize=8);fig.savefig(out/"oof_scale.png");plt.close(fig)
    summary=[]
    for subset,r in reports.items():
        summary.append({"subset":subset,**{k:r["metrics"][k] for k in ["n","n_real","n_generated","auroc","macro_f1","balanced_accuracy","precision_generated","recall_generated","specificity","false_positive_rate","equal_error_rate"]},
                        "auroc_ci_low":r["ci_95"]["auroc"][0],"auroc_ci_high":r["ci_95"]["auroc"][1],
                        "macro_f1_ci_low":r["ci_95"]["macro_f1"][0],"macro_f1_ci_high":r["ci_95"]["macro_f1"][1]})
    with (out/"metric_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    e=html.escape
    def pc(v): return "未定义" if v is None else f"{v:.3f}"
    metric_rows="".join(f"<tr><td>{e(s['subset'])}</td><td>{s['n']:,}</td><td>{s['auroc']:.3f} [{s['auroc_ci_low']:.3f}, {s['auroc_ci_high']:.3f}]</td><td>{s['macro_f1']:.3f} [{s['macro_f1_ci_low']:.3f}, {s['macro_f1_ci_high']:.3f}]</td><td>{s['recall_generated']:.3f}</td><td>{s['false_positive_rate']:.3f}</td></tr>" for s in summary)
    candidate_rows="".join(f"<tr><td>{e(c['candidate_id'])}</td><td>{c['validation']['auroc']:.4f}</td><td>{c['validation']['macro_f1']:.4f}</td><td>{c['threshold']:.6g}</td></tr>" for c in selection["ranked_candidates"])
    generator_rows="".join(f"<tr><td>{e(subset)}</td><td>{e(g)}</td><td>{m['n_generated']}</td><td>{m['auroc']:.3f}</td><td>{m['recall_generated']:.3f}</td><td>{m['macro_f1']:.3f}</td></tr>" for subset,r in reports.items() for g,m in r["per_generator"].items())
    runtime_s=runtime["summary"]
    conclusion="Branch C 单独区分真假视频的能力有限。" if any(r["metrics"]["auroc"]<.7 for r in reports.values()) else "Branch C 在当前两套测试分区中提供了可测量的异常区分证据。"
    cases_path=out/"cases.json"
    cases=json.loads(cases_path.read_text()) if cases_path.exists() else []
    timeline_html=""
    if cases:
        demo=next((c for c in cases if c["case_id"]=="case_02"),cases[0])
        prediction=json.loads((out/f"{demo['case_id']}_prediction.json").read_text())
        windows=prediction["timeline"]
        centers=[(w["start_seconds"]+w["end_seconds"])/2 for w in windows]
        values=[w["anomaly_score"] for w in windows]
        fig,ax=plt.subplots(figsize=(10,3.6),layout="constrained")
        for w in windows:ax.hlines(w["anomaly_score"],w["start_seconds"],w["end_seconds"],color="#256d5a",alpha=.35)
        ax.plot(centers,values,"o-",color="#256d5a",label="Separate real-only window model")
        highest=prediction["most_anomalous_window"]
        ax.axvspan(highest["start_seconds"],highest["end_seconds"],alpha=.12,color="#bc4e2a",label="Highest-score window within this clip")
        ax.set(xlabel="Time (seconds)",ylabel="Window anomaly score (not clip score)",title=f"{demo['case_id']}: exploratory temporal evidence")
        ax.legend(fontsize=8);fig.savefig(out/"timeline_demo.png");plt.close(fig)
        timeline_html=f"<h2>独立窗口时间线示例</h2><p>案例 {e(demo['case_id'])} 的窗口模型最高分区间为 {highest['start_seconds']:.2f}–{highest['end_seconds']:.2f} 秒。每条横线对应一个约 2 秒窗口，折线连接窗口中心；较高分只表示相对该窗口模型更异常。它与整段模型分开拟合，不能套用整段阈值，也没有经过窗口级真假标注验证。</p><img src='timeline_demo.png' alt='独立窗口模型的探索性时间线'>"
    cases_html="".join(f"<article><h3>{e(c['case_id'])} · {e(c['error_type'])} · {e(c['family'])}</h3><p><code>{e(c['clip_id'])}</code>；分数 {c['score']:.5g}，阈值 {fit['threshold']:.5g}。</p><img src='{e(c['image'])}' alt='按真实时间顺序排列的抽样帧'><p>{e(c.get('observation','抽样帧已导出；连续视频审阅与原因仍待人工确认。'))}</p><p class='muted'>抽样帧观察不等同于完整视频运动分析，不能据此证明误判原因。</p></article>" for c in cases)
    content=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Branch C 正式实验与交接</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;margin:40px auto;max-width:1080px;padding:0 24px;line-height:1.75;color:#182d35}}h1{{font-size:32px}}h2{{margin-top:42px;border-top:1px solid #dce4e6;padding-top:20px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #dae2e3;padding:10px}}th{{background:#f0f5f4}}img{{width:100%;height:auto}}code{{font-size:13px;word-break:break-all}}.muted{{color:#5e7078;font-size:13px}}.lead{{font-size:18px;background:#edf4f1;padding:20px}}a{{color:#216c58}}article{{margin:28px 0}}@media print{{body{{margin:0}}h2{{break-after:avoid}}img,table,article{{break-inside:avoid}}}}</style>
<h1>Branch C 正式实验与交接</h1><p>Group 42 · Lang Heran · VIDA-Guard32K · 2026 年 10 月 11 日</p>
<p class="lead">{conclusion}最终选定 <strong>{e(selection['selected_candidate_id'])}</strong>，只用 8,400 个真实训练视频拟合，阈值仅来自 900 个真实校准视频。已为全量训练、验证和测试样本导出融合分数。C 是否改善 A+B 融合，需要组员提供 A/B 输出后做配对消融。</p>
<h2>固定模型的测试结果</h2><table><tr><th>分区</th><th>视频数</th><th>AUROC 与 95% CI</th><th>macro F1 与 95% CI</th><th>生成召回</th><th>真实误报</th></tr>{metric_rows}</table>
<p>置信区间使用防泄漏组进行 1,000 次 bootstrap，seed 42；阈值固定为 {fit['threshold']:.8g}，严格 score &gt; threshold 判异常。macro F1、召回和误报率使用该固定阈值。EER 仅为 ROC 描述性统计，不用于测试调阈值。精度、balanced accuracy、specificity、EER 和混淆矩阵见原始 JSON 与 metric_summary.csv。</p><img src="roc.png" alt="两套测试 ROC 和固定阈值工作点"><img src="confusion.png" alt="固定阈值混淆矩阵"><img src="scores.png" alt="真实与生成视频异常分数分布">
<h2>模型选择与数据隔离</h2><p>训练 16,800 个视频，其中真实 8,400 个。验证 3,600 个，按已交付角色分为 model_selection 与 calibration，各 1,800 个。六候选只用 model_selection 的 AUROC、macro F1 和预定顺序排名；候选二值指标的阈值使用真实 model_selection 分位数。选定模型之后，最终阈值只从 calibration 的 900 个真实视频确定。生成视频没有进入标准化器、异常检测器或最终阈值的拟合。</p>
<table><tr><th>候选，按排名排列</th><th>选择集 AUROC</th><th>选择集 macro F1</th><th>选择集阈值</th></tr>{candidate_rows}</table><p>选择集指标有选择偏差，不能代替封存测试结果。公共 fold 0–4 原样保留；每折留出 3,360 个 train，C 只拟合其余折的 6,720 个真实视频。每个 train 恰好获得一个折外分数。最终模型、阈值和配置在两套测试前冻结，没有根据测试结果重新选择。</p>
<h2>实现与输入条件</h2><p>37 维摘要包括像素/亮度/边缘变化、Farneback 光流速度与方向、运动补偿残差、时间频谱和整体时序变化。统一缩放为 96×96；按整数原始帧步长接近 8 fps，必要时降低步长，至少保留 8 个实际帧，使用真实 CFR 时间戳，不复制、不插值、不重编码。光流使用图像边长/秒，像素/亮度/边缘变化按 1/8 秒计；频谱只使用共有的 0–2 Hz，分带为 0–0.5、0.5–1.2、1.2–2 Hz。短片使用整段窗口，其余约 2 秒滑窗；独立窗口模型最多从每个真实 train 取 4 个窗口。</p><p>分数为 -score_samples，越高越偏离真实训练分布；它不是 AI 生成概率。时间线来自独立窗口模型，没有窗口级真假标注或概率校准，不能套用整段阈值。标准化描述符偏离是描述性信息，不是因果特征重要性。</p>
<h2>按生成器评价与来源局限</h2><table><tr><th>分区</th><th>生成器</th><th>生成数</th><th>AUROC</th><th>召回</th><th>macro F1</th></tr>{generator_rows}</table><img src="generators.png" alt="各生成器相对同一真实测试池的 AUROC 区间">
<p>每个生成器与同一分区的全部真实视频比较。in_domain_test 真实来源为 Vript；cross_test 真实来源为 HD-VG-130M，生成器为 CogVideo、Mora、MuseV、SVD，均未进入开发。cross_test 同时改变真实来源和生成器，不能归因于仅生成器变化。数据是整理后的 VIDA-Guard32K，不是未修改的官方 GenVidBench。已继承数据负责人的来源/提示词组与重复审计，但未隔离全部频道，OpenSora 的未知提示词关系等语义重复风险仍存在。</p><img src="source_diagnostics.png" alt="分数与时长和原始帧率的描述性关系"><p>这些散点和组内 Spearman 相关仅用于测试后描述，不用于模型选择，也不能证明时长、帧率或来源导致误判。统一缩放与时间归一化不能完全消除内容、压缩、时间长度和采样差异；极短片的频率分辨率尤其有限。</p>
<h2>误报与漏检案例</h2><p>按冻结阈值导出完整 error_review.csv。FP 为真实视频高于阈值；FN 为生成视频未超过阈值。以下案例按预先固定规则抽取，保留误判与输入条件；抽样帧只支持可见内容的观察，原因需结合连续视频确认，不据此调参或删除测试样本。</p>{cases_html}
<h2>CPU 运行资源</h2><p>真实训练视频 <code>{e(Path(runtime['input_path']).stem)}</code>，原长 {runtime['runs'][0]['input_metadata']['decoded_duration_seconds']:.2f} 秒。{e(runtime['environment']['system'])} {e(runtime['environment']['machine'])}，逻辑 CPU {runtime['environment']['logical_cpu_count']}；3 次独立新进程，不使用描述符缓存。模型 {runtime['model_size_bytes']/1048576:.2f} MiB；含模型加载的推理中位数 {runtime_s['total_seconds_including_model_load']['median']:.3f} 秒（{runtime_s['total_seconds_including_model_load']['min']:.3f}–{runtime_s['total_seconds_including_model_load']['max']:.3f}）；含进程启动与导入的中位数 {runtime_s['process_wall_seconds']['median']:.3f} 秒；整个进程峰值 RSS 中位数 {runtime_s['peak_rss_bytes']['median']/1048576:.1f} MiB。OS 文件缓存不受控制；这些是本机 CPU 结果，不能冒称约定 GPU 的结果。</p>
<h2>交付与复现</h2><p>融合训练使用 <a href="../oof/oof_scores.csv">oof_scores.csv</a>，按 clip_id 与 A/B 一对一合并，核对标签、fold_id、protocol 和内容哈希。校准使用 <a href="../selection/selected/validation_scores.csv">validation_scores.csv</a> 中 calibration 行。测试使用 <a href="../in_domain_test/test_scores.csv">同分布分数</a>、<a href="../cross_test/test_scores.csv">跨来源分数</a>。模型、配置、锁定依赖与源码必须保持对应版本。原始数据不打包进代码交接包。</p><p>在项目根目录运行 scripts/import_delivery.py 导入新机器路径；scripts/run_vida_guard32k.py 执行冻结的完整流程，已有实验用新 output-dir。scripts/verify_vida_run.py 复核全部输入哈希、覆盖、真实拟合统计、验证阈值、五折数值和最终分数；scripts/report_vida_run.py 重建本报告。图表另外需要 requirements-report.txt。</p>
<p>尚需全组共同完成：A/B 对齐与融合训练、概率校准、A+B 对比 A+B+C 的同视频同组配对消融；当前结果不声称 C 已带来融合增益。新来源上的异常也可能来自正常但少见的拍摄内容，最终判断应结合来源记录、上下文和人工复核。</p><details><summary>可追溯记录</summary><p>模型 SHA256：<code>{e(verification['model_sha256'])}</code></p><p>公共折计划：<code>{e(verification['common_fold_plan_id'])}</code></p><p><a href="../verification.json">完整核验记录</a> · <a href="../protocol_plan.json">冻结实验计划</a> · <a href="../selection/selection_report.json">候选比较</a> · <a href="../runtime.json">资源原始记录</a></p></details></html>"""
    baseline_rows="".join(f"<tr><td>{name}</td><td>{e(subset)}</td><td>{m['auroc']:.3f}</td><td>{m['macro_f1']:.3f}</td><td>{m['balanced_accuracy']:.3f}</td></tr>" for name,values in [("Majority real",reference),("Fully random",random_metrics)] for subset,m in values.items())
    baseline_html=f"<h2>无学习参考</h2><p>多数类 B0：训练集真实与生成各 8,400，预先约定并列时预测真实，恒定分数 0、阈值 0.5。Fully random 对应提案 PPT 第 12 页：全部测试 clip_id 排序后用固定 seed 42 生成随机分数，阈值 0.5，生成分数时不使用标签。不读取媒体、不拟合模型，不是融合或空间分支。随机参考的实际指标如下，不能把有限样本结果强制写成恰好 0.5。</p><table><tr><th>参考</th><th>分区</th><th>AUROC</th><th>macro F1</th><th>balanced accuracy</th></tr>{baseline_rows}</table>"
    content=content.replace("<h2>模型选择与数据隔离</h2>",baseline_html+"<h2>模型选择与数据隔离</h2>")
    scale_html="<h2>融合输入尺度检查</h2><img src='oof_scale.png' alt='真实 OOF 与最终模型真实校准分数的分布'><p>OOF 与最终模型使用不同的真实拟合数量，尤其 SVM 的原始分数尺度可能变化。图中是真实 OOF 与最终模型真实 calibration 的不同视频，不能单凭分布差异归因于训练量。融合应在 calibration 上核对输入分布和校准，不能把 raw score 当作统一概率，也不能对 OOF 套用最终模型阈值。原始数值统计保存在 oof_scale_check.json；本检查不拟合任何新校准器。</p>"
    content=content.replace("<h2>交付与复现</h2>",scale_html+"<h2>交付与复现</h2>")
    content=content.replace("<h2>CPU 运行资源</h2>",timeline_html+"<h2>CPU 运行资源</h2>")
    content=content.replace("置信区间使用防泄漏组进行 1,000 次 bootstrap，seed 42；", "置信区间使用防泄漏组进行 1,000 次 bootstrap，seed 42，只反映固定模型与阈值下的测试组重采样，不包含训练、模型选择或阈值估计的不确定性；")
    # Keep wide scientific tables readable without overflowing a narrow panel.
    content=content.replace("<table>","<div class='table-scroll'><table>").replace("</table>","</table></div>")
    content=content.replace("</style>",".table-scroll{overflow-x:auto;max-width:100%}.table-scroll table{min-width:620px}</style>")
    (out/"index.html").write_text(content,encoding="utf-8")
    write_json(out/"report_summary.json",{"status":"complete","model_id":verification["model_id"],"metrics":summary,"selected_candidate":selection["selected_candidate_id"],"conclusion":conclusion,"case_count":len(cases)})
    print(out/"index.html")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",default="artifacts/vida_guard32k/run_02")
    p.add_argument("--manifest",default="data/VIDA-Guard32K/branch_c/manifest.csv")
    build(p.parse_args())
