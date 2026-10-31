# Branch C：模型和逐视频预测分数

训练成果发布在 [GitHub Release](https://github.com/Namedx2/Vida/releases/tag/branch-c-artifacts-2026-10-29)，代码与说明通过 [小组仓库 PR #1](https://github.com/Jerry00C/Vida/pull/1) 交付。Release 位于提交账号 Namedx2 的 Fork；Jerry00C/Vida 的维护者合并 PR 后，也可从本说明直接下载。

- [下载完整模型和预测分数](https://github.com/Namedx2/Vida/releases/download/branch-c-artifacts-2026-10-29/branch_c_models_and_scores.zip)：48.39 MiB。
- [SHA256SUMS.txt](https://github.com/Namedx2/Vida/releases/download/branch-c-artifacts-2026-10-29/SHA256SUMS.txt)。
- ZIP SHA256：5b490ea09a2ddfecb6a71590207b4968cb793fadead2603251b06a950812b705。

## 包含什么

| 内容 | 数量 / 用途 |
| --- | --- |
| 最新 28 轮协议的最终模型 | 两套共享最终模型，nu=.01 与 .05；每轮按 model_reference.json 选用 |
| 最新协议的辅助 OOF 模型 | 两套参数各五折，共十个；用于核查折外分数 |
| 历史 run_02 最终模型 | 一个，固定 4 已见 / 4 未见协议，单独标为 historical_run_02 |
| OOF、验证、已见及未见测试分数 | 最新 112 个 CSV + 历史 4 个 CSV，共 116 个 |
| 分数行数 | 659,200；包含轮次复用，不代表这么多个独立视频 |
| 每轮模型引用及配置 | 28 轮各一份，包含实际模型、阈值与校验哈希 |
| 结果与校验 | 58 个测试视图的指标核对、逐文件 SHA256、导出审计及依赖锁 |

这次交付没有原始视频、抽帧图片、完整输入清单、特征缓存或候选搜索缓存。

## 直接使用

解压得到 branch_c_trained_artifacts/。对最新协议，先打开 rotation_6seen_2unseen_v1/round_XX/model_reference.json，根据 selected_candidate 和 model_path 找到实际共享模型。不要随意挑一个模型替代所有轮次。

在 dev/branch_c/ 下按 README 安装 Python 3.12.14 的锁定依赖和源码，然后使用原有命令：

    branch-c predict --model /本机路径/branch_c_trained_artifacts/rotation_6seen_2unseen_v1/shared_models/svm_nu_0.01/model.joblib --video /本机路径/new_video.mp4 --output artifacts/new_video_result.json

以上 .01 是使用示例；实际按该轮引用选择 .01 或 .05。标签为 0=real、1=generated；异常分数越高越偏离真实分布，使用 score > threshold 判异常，不能把它当生成概率。

分数按同一轮次的 clip_id 一对一合并，并核对 protocol、label、fold_id 和 content_sha256。训练融合器使用 OOF；validation_role=calibration 才用于融合校准。OOF 的 model_id 留空是因为每个样本来自对应折的模型。A/B/C 尚需统一 split，不能直接把当前 A 四轮和 C 二十八轮的分数拼起来。

## 导出核验

所有预测 CSV 保持原始字节与 SHA256，不重新训练、不改分数和阈值。最终模型保留完整的学习参数、StandardScaler、窗口模型、特征签名与 model_id，只把嵌入的本机文件路径和无关输入清单字段替换为精简的 ID/分组泄漏索引。内容哈希、ID 和已知分组仍可用于评价检查；原机器的路径匹配不再具有可移植性。

因此导出模型文件的 SHA256 与原始包不同，两者均记录在 artifact_manifest.json；每轮 model_reference.json 指向导出版本的实际 SHA256。已核对三个最终模型的估计器完全相同、探针分数逐值相同，以及 58 个测试视图的指标与原报告相同。历史原模型和原实验文件保留。

需要从自己的完整本地实验结果重新导出时，在 Branch C 根目录运行：

    python scripts/export_release_artifacts.py --project-root . --output-dir output/new_trained_artifacts_export

输出目录必须是新目录。对 joblib 文件只加载自己或团队可信的来源。
