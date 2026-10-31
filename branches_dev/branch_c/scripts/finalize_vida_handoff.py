"""Update repository handoff docs and package the verified Branch C deliverables."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from branch_c.cache import file_sha256
from branch_c.config import write_json


def finalize(args):
    root=Path(args.run_dir)
    destination=Path(args.output)
    if destination.exists():raise ValueError("Handoff ZIP already exists; choose a new file")
    verification=json.loads((root/"verification.json").read_text())
    report=json.loads((root/"report/report_summary.json").read_text())
    if report["model_id"]!=verification["model_id"]:raise ValueError("Report/model mismatch")
    cases=json.loads((root/"report/cases.json").read_text())
    if verification["status"]!="passed" or not cases or any(not c["observation"].strip() for c in cases):
        raise ValueError("Numeric verification and visual case observations must be complete")
    visual=json.loads((root/"report/visual_review.json").read_text())
    if visual["status"]!="passed" or any(file_sha256(root/"report"/p)!=h for p,h in visual["files_sha256"].items()):
        raise ValueError("Report changed since visual review")
    if any(file_sha256(p)!=h for p,h in verification["source_sha256"].items()):
        raise ValueError("Frozen model source changed since numeric verification")
    software=json.loads((root/"software_validation/record.json").read_text())
    if software["source_sha256"]!=verification["source_sha256"] or software["smoke"]["status"]!="passed":
        raise ValueError("Software evidence does not match the frozen source")
    for row in report["metrics"]:
        expected=verification["test_reports"][row["subset"]]["metrics"]
        if any(row[k]!=expected[k] for k in ["n","auroc","macro_f1","recall_generated","false_positive_rate"]):
            raise ValueError("Report metrics differ from verified results")
    selection=json.loads((root/"selection/selection_report.json").read_text())
    fit=json.loads((root/"selection/selected/fit_report.json").read_text())
    runtime=json.loads((root/"runtime.json").read_text())
    rows=report["metrics"]
    result_table="\n".join(f"| {r['subset']} | {r['n']:,} | {r['auroc']:.3f} [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}] | {r['macro_f1']:.3f} [{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}] | {r['recall_generated']:.3f} | {r['false_positive_rate']:.3f} |" for r in rows)
    candidate=selection["selected_candidate_id"]
    location=str(root)
    model_card=f"""# Branch C 模型卡

负责人：Lang Heran，Group 42。2026 年 10 月 11 日完成 VIDA-Guard32K 上的正式训练、固定阈值测试与数值核验。模型为 `{location}/selection/selected/model.joblib`，选定候选 `{candidate}`。这份结果使用整理后的 32K 数据，不能冒称未修改的官方 GenVidBench。

## 用途和方法

任务是为真实摄像视频与完全 AI 生成视频的融合检测提供异常证据。输入转换成 37 维运动、帧间变化与时间频率摘要；StandardScaler 与异常检测器只拟合 8,400 个真实 train，独立窗口模型也只使用这些真实视频，每个视频最多选择 4 个窗口。实际算法与参数见 selected_config.json。分数是 `-score_samples`，越大越偏离真实参考分布，**不是 AI 生成概率**，不能推断作者、生成器身份或给出取证结论。

输入使用 native_stride：CFR 0.5–20 秒、原始 fps 4–240、至少 8 个实际帧。整数原始帧步长接近 8 fps，短片降低步长保留观测，使用真实时间戳；不复制、不插值、不重编码。统一缩放为 96×96。光流以图像边长/秒表示，像素/亮度/边缘变化按 1/8 秒计；FFT 仅比较共有的 0–2 Hz。短片整段窗口，其他约 2 秒滑窗。旧 uniform_cfr 模式与历史 TOY 模型不是本次正式输入协议。

## 数据隔离与选择

train 16,800（真实 8,400、生成 8,400）；val 3,600，model_selection 与 calibration 各 1,800，均为真假各半。六候选只在 model_selection 按 AUROC、macro F1、预定顺序排名；最终阈值仅使用 calibration 中的 900 个真实视频，为 `{fit['threshold']:.10g}`，严格大于判异常。生成 calibration 没有参与该阈值。选择集指标有选择偏差。

公共五折保持数据负责人的分组与折号，每折 C 拟合其他折的 6,720 个真实视频，为当前留出折的 3,360 个视频评分，得到全量 16,800 行 OOF。标准化器也在每折训练侧拟合。两套测试在最终模型、配置和阈值冻结之后执行，没有根据测试结果重新选择。

## 封存测试结果

| 分区 | 视频数 | AUROC 与 95% CI | macro F1 与 95% CI | 生成召回 | 真实误报 |
| --- | ---: | --- | --- | ---: | ---: |
{result_table}

95% 区间使用防泄漏组 bootstrap 1,000 次，seed 42，只反映固定模型与阈值下的测试组重采样，不包含训练、模型选择或阈值估计的不确定性。固定阈值指标、各生成器相对同一真实池的指标及区间、EER、precision、balanced accuracy、specificity 和混淆矩阵保存在 test_report.json。EER 不用于重新确定测试阈值。完整结果和图表为 `{location}/report/index.html`。

in_domain_test 为 Vript 真实视频及 Pika、VideoCrafter2、T2V-Zero、OpenSora；cross_test 为 HD-VG-130M 真实视频及 CogVideo、Mora、MuseV、SVD。后者同时改变真实来源和生成器，不能隔离纯生成器变化的因果效果。

## 资源与复现

本机 `{runtime['environment']['system']}` `{runtime['environment']['machine']}`，逻辑 CPU {runtime['environment']['logical_cpu_count']}，真实 10 秒训练视频，3 次独立进程，无描述符缓存。含模型加载的推理中位数 {runtime['summary']['total_seconds_including_model_load']['median']:.3f} 秒；含进程启动和导入的中位数 {runtime['summary']['process_wall_seconds']['median']:.3f} 秒；全进程峰值 RSS 中位数 {runtime['summary']['peak_rss_bytes']['median']/1048576:.1f} MiB；模型 {runtime['model_size_bytes']/1048576:.2f} MiB。不是 GPU 测量；OS 文件缓存不受控制。

已存模型要求与模型卡记录的运行库完全一致，包括 Python 补丁版本 `{fit['versions']['python']}`。先安装 requirements-lock.txt，再 `pip install --no-deps -e .`；报告绘图另外安装 requirements-report.txt。换机器使用 import_delivery.py 重新映射数据根目录，在新 output-dir 训练，保留原始分区、折和内容哈希。只加载自己或团队可信的 joblib 文件。

模型 SHA256：`{verification['model_sha256']}`。公共折计划：`{verification['common_fold_plan_id']}`。verification.json 记录了 32,000 个输入/缓存的校验、整段与窗口标准化统计的真实类数值复核、900 个真实校准样本阈值、全部最终分数与五折模型/OOF 分数的数值复现。

## 局限与案例

异常也可能来自正常但少见的内容或来源差异，稳定的生成视频也可能接近真实参考分布。缩放和按时间归一化不能消除时长、编码、帧率、内容分布与来源捷径；短片频率分辨率较粗，整数抽帧没有时间低通抗混叠。来源/提示词分组和重复审计继承自数据负责人；未隔离全部频道，未知语义重复仍可能存在。

报告保存 {len(cases)} 个确定性抽样案例与可见帧观察，完整 FP/FN 列表为 error_review.csv。帧观察不等同于完整视频人工审阅，不能证明误判原因。标准化偏离不是因果特征重要性；独立窗口时间线未经窗口真假标注验证，不能套用整段阈值或称为 AI 概率。

A/B 联调、融合概率校准与 A+B 对比 A+B+C 的同视频同组配对消融是全组后续工作。当前结果不声称 C 已改善融合，不将原始异常分数用于 Brier/ECE。项目结果应结合来源记录、上下文和人工复核。
"""
    Path("docs/MODEL_CARD.md").write_text(model_card,encoding="utf-8")
    a,b=rows
    handoff=f"""# Lang Heran 的 Branch C 交接与汇报

Branch C 的正式训练、模型选择、公共五折 OOF、两套固定模型测试、CPU 资源测量、案例和数值核验已完成。数据为 VIDA-Guard32K，完整报告在 `{location}/report/index.html`，模型卡见 MODEL_CARD.md。

## 可以交给组员的文件

以下路径均相对 `{location}`：

| 文件 | 接收人和用途 |
| --- | --- |
| common_folds/fold_assignments.csv | A/B 共用原有 fold 0–4，共 16,800 行；不含预测 |
| common_folds/manifest_with_folds.csv | 本机可解析的完整清单；换机器重映射根目录 |
| oof/oof_scores.csv | 融合训练用的 C 输入，每个 train 恰好一行 |
| selection/selected/validation_scores.csv | 3,600 行 val，保留 validation_role；融合校准使用 calibration |
| in_domain_test/test_scores.csv | 3,600 行最终同分布测试输入 |
| cross_test/test_scores.csv | 8,000 行最终跨来源测试输入 |
| selection/selected/model.joblib | 整段模型、窗口模型、固定阈值和预处理配置 |
| selection/selected_config.json | 选定算法与参数，OOF 必须使用相同配置 |
| verification.json | 真实拟合、阈值、覆盖、折与分数的数值审计 |
| report/index.html | 中文结果、图表、案例与解释边界 |

所有分数按 clip_id 一对一合并，核对标签、fold_id、内容哈希、protocol；分数方向为越高越异常，is_probability=false。不要按行号合并，不要用最终模型的训练集内分数替代 OOF，不要将 candidate_validation 分数作为融合训练输入。

选择仅使用 model_selection 的 1,800 个视频；最终阈值仅来自 calibration 的 900 个真实视频。融合层可在 calibration 上校准，不能把已参与选择的 model_selection 当作未参与开发的测试集。两套最终 test 不能用于重新调参。

## 实现思路和代码结构

video.py：读 CFR 视频、按实际时间采样，短片保留至少 8 个实际观测；features.py：37 维运动与时序摘要；cache.py：按内容哈希和特征签名缓存；model.py：真实类 StandardScaler 与 IF/OCSVM；selection.py：六候选比较；folds.py：保留公共折；pipeline.py：拟合、OOF、预测和封存评价；evaluation.py：固定阈值指标与组 bootstrap；review.py：误报/漏检表；benchmark.py：独立进程延迟与峰值内存；cli.py：统一命令行。

scripts/import_delivery.py 导入同学的数据，run_vida_guard32k.py 执行完整正式流程，verify_vida_run.py 做数值复核，review_vida_cases.py 导出案例，report_vida_run.py 重建报告，finalize_vida_handoff.py 更新文档与交接包。测试仍可用无外部数据的 `python -m pytest -q` 和 smoke_test.py；TOY 只验证程序，不代表正式效果。

## 约一分钟中文讲稿

我负责 Branch C，也就是真实视频异常检测。我们从视频中提取 37 维运动和时间变化摘要，标准化器和异常模型只使用 8,400 个真实训练视频拟合。我们在固定的六个候选中选中了 {candidate}，选择只用模型选择验证集；最终异常阈值只使用另一个校准子集中的 900 个真实视频。

为避免融合泄漏，我沿用全组五折，为全部 16,800 个训练视频生成了折外分数。冻结同一模型和阈值后，同分布测试 AUROC 是 {a['auroc']:.3f}，跨来源测试是 {b['auroc']:.3f}。这些分数表示异常证据，不等于 AI 生成概率；跨来源测试也同时改变了真实视频来源，因此不能把结果全归因于未知生成器。是否改善 A+B 融合，需要大家对齐分数后做 A+B 与 A+B+C 的配对消融。

## English speaking notes

Branch C models normal video dynamics using 37 summary descriptors. Both the scaler and anomaly detector are fitted exclusively on 8,400 real training clips. A predeclared six-candidate comparison selected {candidate} using only the model-selection validation subset. A separate set of 900 real calibration clips defines the final threshold. We preserved the common five folds and exported exactly one out-of-fold score for each of the 16,800 training clips.

With the model and threshold frozen, AUROC was {a['auroc']:.3f} in-domain and {b['auroc']:.3f} on cross-test. The score is anomaly evidence, not an AI-generation probability. Cross-test changes both the real-video source and generator families. The incremental value of C still requires a paired A+B versus A+B+C evaluation using our teammates' outputs.
"""
    Path("docs/HANDOFF_ZH.md").write_text(handoff,encoding="utf-8")
    p=Path("README.md");s=p.read_text()
    s=s.replace("当前正在执行正式训练；TOY 演示与正式结果分开保存，不能混用。",f"正式训练、五折 OOF 与两套固定模型测试已完成。选定 {candidate}；同分布 AUROC {a['auroc']:.3f}，跨来源 AUROC {b['auroc']:.3f}。TOY 演示与正式结果分开保存，不能混用。")
    s=s.replace("只用真实训练视频拟合的 StandardScaler 和 Isolation Forest", "只用真实训练视频拟合的 StandardScaler 和选定的一类异常模型")
    s=s.replace("另提供 One-Class SVM 作为对照。", "六候选包含 Isolation Forest 和 One-Class SVM；正式选定 RBF One-Class SVM，nu=.01。")
    s=s.replace("正式资源结果仍需真实 10 秒视频和约定机器。",f"正式资源结果见 {location}/runtime.json，使用本机 CPU 和真实 10 秒视频；这不代表指定 GPU 工作站的测量。")
    s=s.replace("数据到位后还需正式训练、误检分析和 A+B 与 A+B+C 的配对消融。", "正式训练、固定模型测试和案例观察已完成；A+B 与 A+B+C 的配对消融仍需全组对齐分数。")
    marker="## VIDA-Guard32K 正式运行"
    if "正式结果与图表：[中文实验报告]" not in s:
        s=s.replace(marker,f"正式结果与图表：[中文实验报告]({location}/report/index.html)。交接与讲稿见 [HANDOFF_ZH.md](docs/HANDOFF_ZH.md)，限制与运行环境见 [MODEL_CARD.md](docs/MODEL_CARD.md)。\n\n"+marker,1)
    p.write_text(s,encoding="utf-8")
    log=Path("docs/ITERATION_LOG.md")
    historical=log.read_text()
    historical=historical.replace("## 仍待完成的既定工作", "## 第四轮结束时的待办（历史记录）")
    log.write_text(historical,encoding="utf-8")
    entry=f"\n\n## 2026 年 10 月 11 日 VIDA-Guard32K 正式训练\n\n完成 32,000 视频交付校验与接口映射，保留原始组、分区和折；适配原始短片、4 fps 和真实时间戳。正式运行 {location}，六候选选定 {candidate}，只用 8,400 真实 train 拟合，最终阈值来自 900 真实 calibration。导出 16,800 OOF、3,600 val、3,600 同分布 test、8,000 cross-test 分数；两套 test 前冻结模型。完成主指标/分生成器组 bootstrap、CPU 真实 10 秒视频资源测量、{len(cases)} 个帧序列案例观察，以及全量分数和五折模型数值复现。run_01 在候选拟合前停止，缓存复用于 run_02；未利用测试结果修改模型。\n"
    if "## 2026 年 10 月 11 日 VIDA-Guard32K 正式训练" not in log.read_text(): log.write_text(log.read_text()+entry)
    methods=Path("docs/METHOD_AND_EXPERIMENTS.md")
    text=methods.read_text()
    text=text.replace("当前运行位于 artifacts/vida_guard32k/run_02", "已完成的正式运行位于 artifacts/vida_guard32k/run_02")
    text=text.replace("## 数据到位后的实验顺序", "## 实验顺序与全组接口")
    if "正式选中 RBF One-Class SVM" not in text:
        text=text.replace("## 描述符", "六个预定候选中正式选中 RBF One-Class SVM（gamma=scale，nu=.01）。以下 Isolation Forest 参数说明保留为预定基线与通用入口；实际最终模型以 selected_config.json 为准。\n\n## 描述符",1)
    english="""Branch C models normal video dynamics using real training videos only. In the completed VIDA-Guard32K run, native CFR frames are sampled with an integer stride close to 8 fps, retaining at least eight actual observations and their timestamps. We extract 37 summary descriptors of inter-frame pixel and edge variation, dense optical-flow speed, motion-compensated residuals, and temporal-frequency energy within the common 0–2 Hz support. The final StandardScaler and RBF one-class SVM (gamma=scale, nu=0.01) are fitted exclusively on 8,400 real training clips. This estimator won a predeclared six-candidate comparison using only the model-selection validation subset; generated clips never enter scaler or anomaly-estimator fitting. We negate score_samples so that larger values indicate stronger anomaly evidence, not an AI-generation probability.

The supplied group-isolated five folds are preserved. Each fold fits its own scaler and estimator on 6,720 real clips, exporting exactly one out-of-fold score for each of the 16,800 training clips. A separate set of 900 real calibration clips defines the final standalone threshold. An independent real-only window model provides exploratory temporal evidence without window-level probability calibration. The frozen model achieves AUROC {in_auc:.3f} in-domain and {cross_auc:.3f} on cross-test. Cross-test changes both the generator families and the real-video source. The contribution of C to fusion still requires a paired A+B versus A+B+C evaluation with teammates' outputs.
""".format(in_auc=a['auroc'],cross_auc=b['auroc'])
    text=text.split("## English methods draft")[0]+"## English methods draft\n\n"+english
    methods.write_text(text,encoding="utf-8")
    requirements=[
        ("真实视频运动与时间摘要、真实类异常模型", "PDF p2 / PPT slide10", ["selection/selected/model.joblib","selection/selected/fit_report.json","selection/selected_config.json"]),
        ("标准化器和整段/窗口模型仅拟合真实 train", "PDF p2–3 / PPT slide10–11", ["verification.json","selection/selected/fit_report.json"]),
        ("公共组隔离五折、全量训练 OOF 和输入追溯", "PDF p3 / PPT slide11", ["common_folds/fold_plan.json","common_folds/fold_assignments.csv","oof/oof_scores.csv","oof/oof_audit.json","metadata_audit.json"]),
        ("有限候选、验证选择/真实校准阈值与测试前冻结", "PDF p3 / PPT slide11", ["protocol_plan.json","selection/selection_report.json","model_freeze.json","verification.json"]),
        ("同分布与跨来源独立评价及主次指标/组 bootstrap", "PDF p3 / PPT slide13", ["in_domain_test/test_report.json","cross_test/test_report.json","report/metric_summary.csv"]),
        ("预先声明的 Fully random 参考", "PPT slide12", ["random_reference_plan.json","reference_baseline_plan.json","report/reference_baseline.json","report/random_reference_scores.csv"]),
        ("完整错误列表和可追溯案例观察", "PDF p3–4", ["in_domain_test/error_review.csv","cross_test/error_review.csv","report/cases.json","report/visual_review.json"]),
        ("真实10秒视频的延迟、峰值RSS与模型体积", "PDF p3 / PPT slide13", ["runtime.json"]),
        ("模型卡、来源限制、复现与组员交接", "PDF p4 / user Branch C scope", ["report/index.html","verification.json"]),
        ("无正式数据也可运行的软件验证", "user request", ["software_validation/record.json","software_validation/pytest.log","software_validation/smoke.log"]),
    ]
    checked=[]
    for requirement,origin,evidence in requirements:
        if any(not (root/p).is_file() for p in evidence):raise ValueError(f"Missing requirement evidence: {requirement}")
        checked.append({"requirement":requirement,"origin":origin,"status":"verified_in_branch_c_scope",
                        "evidence_sha256":{p:file_sha256(root/p) for p in evidence}})
    write_json(root/"completion_audit.json",{
        "status":"branch_c_deliverables_verified_before_packaging","checked_at":datetime.now(timezone.utc).isoformat(),
        "model_id":verification["model_id"],"requirements":checked,
        "documentation_sha256":{p:file_sha256(p) for p in ["README.md","docs/MODEL_CARD.md","docs/HANDOFF_ZH.md","docs/METHOD_AND_EXPERIMENTS.md","docs/ITERATION_LOG.md"]},
        "reproduction_scripts_sha256":{str(p):file_sha256(p) for p in Path("scripts").glob("*.py")},
        "software_tests_passed":software["pytest"]["passed"],
        "limitations":["CPU resources measured on this Mac, not a team GPU workstation",
                       "Twelve sampled-frame reviews are not full-video human review or validated causal attribution",
                       "Unknown semantic duplicates, channels and source/fps/duration confounding remain",
                       "Single-model group bootstrap does not include training or model-selection uncertainty"],
        "external_team_dependencies":["Branch A/B outputs and fusion training/calibration",
                                      "Paired A+B versus A+B+C ablation on the same clips and group resamples",
                                      "Group presentation and any requested benchmark on a designated GPU machine"],
        "scope_note":"User assigned Branch C. Full-group fusion gain is not claimed or substituted with fabricated scores. ZIP integrity is checked separately after packaging."
    })
    source=[]
    for directory in ["src","scripts","tests","configs","docs","examples","validation"]:
        source.extend(p for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc",".pyo"})
    source.extend(Path(p) for p in ["README.md","pyproject.toml","requirements-lock.txt","requirements-report.txt",".gitignore"])
    artifacts=[p for p in root.rglob("*") if p.is_file()]
    files=sorted(set(source+artifacts))
    receipt=Path("data/VIDA-Guard32K/receipt.json")
    provenance=Path("data/VIDA-Guard32K/delivery/DATASET_README.md")
    contents={str(p):file_sha256(p) for p in files}
    contents["dataset_receipt.json"]=file_sha256(receipt)
    contents["dataset_source_readme.md"]=file_sha256(provenance)
    package={"created_at":datetime.now(timezone.utc).isoformat(),"run_dir":str(root),"model_sha256":verification["model_sha256"],
             "scope":"Branch C source, tests, locked dependencies, final model, OOF/val/test scores, audits, plots and selected frame sheets; no raw video or archives",
             "remaining_team_dependency":"A/B outputs, fusion calibration and paired A+B versus A+B+C ablation",
             "members_sha256":contents}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p))
        z.write(receipt,"dataset_receipt.json")
        z.write(provenance,"dataset_source_readme.md")
        z.writestr("handoff_manifest.json",json.dumps(package,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError("ZIP CRC failure")
        for name,digest in contents.items():
            import hashlib
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError(f"ZIP member changed: {name}")
        if any(Path(n).suffix.lower() in {".mp4",".mov",".npz",".zip"} for n in z.namelist()):
            raise ValueError("Raw media/archive must not be included in handoff")
    write_json(destination.with_suffix(".json"),{"status":"complete","zip":str(destination),"sha256":file_sha256(destination),"bytes":destination.stat().st_size,"members":len(contents)+1,"model_id":verification["model_id"]})
    print(destination)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",default="artifacts/vida_guard32k/run_02")
    p.add_argument("--output",default="output/branch_c_vida_guard32k_handoff.zip")
    finalize(p.parse_args())
