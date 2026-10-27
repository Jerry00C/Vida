# Lang Heran 的 Branch C 交接与汇报

> 本文保留 2026-10-11 原固定 4 已见 / 4 未见实验。最新 28 轮协议见 [GENERATOR_ROTATION_ZH.md](GENERATOR_ROTATION_ZH.md)，GitHub 中的汇总结果见 [results/README.md](../results/README.md)。

Branch C 的正式训练、模型选择、公共五折 OOF、两套固定模型测试、CPU 资源测量、案例和数值核验已完成。数据为 VIDA-Guard32K，完整报告在 `artifacts/vida_guard32k/run_02/report/index.html`，模型卡见 MODEL_CARD.md。

## 可以交给组员的文件

以下路径均相对 `artifacts/vida_guard32k/run_02`：

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

我负责 Branch C，也就是真实视频异常检测。我们从视频中提取 37 维运动和时间变化摘要，标准化器和异常模型只使用 8,400 个真实训练视频拟合。我们在固定的六个候选中选中了 svm_nu_0.01，选择只用模型选择验证集；最终异常阈值只使用另一个校准子集中的 900 个真实视频。

为避免融合泄漏，我沿用全组五折，为全部 16,800 个训练视频生成了折外分数。冻结同一模型和阈值后，同分布测试 AUROC 是 0.841，跨来源测试是 0.895。这些分数表示异常证据，不等于 AI 生成概率；跨来源测试也同时改变了真实视频来源，因此不能把结果全归因于未知生成器。是否改善 A+B 融合，需要大家对齐分数后做 A+B 与 A+B+C 的配对消融。

## English speaking notes

Branch C models normal video dynamics using 37 summary descriptors. Both the scaler and anomaly detector are fitted exclusively on 8,400 real training clips. A predeclared six-candidate comparison selected svm_nu_0.01 using only the model-selection validation subset. A separate set of 900 real calibration clips defines the final threshold. We preserved the common five folds and exported exactly one out-of-fold score for each of the 16,800 training clips.

With the model and threshold frozen, AUROC was 0.841 in-domain and 0.895 on cross-test. The score is anomaly evidence, not an AI-generation probability. Cross-test changes both the real-video source and generator families. The incremental value of C still requires a paired A+B versus A+B+C evaluation using our teammates' outputs.
