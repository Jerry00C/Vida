# Branch C 与融合层对接

VIDA-Guard32K 的正式交付目录为 `artifacts/vida_guard32k/run_02/`。优先使用 `common_folds/fold_assignments.csv` 与 `oof/oof_scores.csv`；共有 16,800 个 train，原有五折保持不变，每折 C 仅拟合其他折的 6,720 个真实视频。验证输入为 `selection/selected/validation_scores.csv`（3,600 行），按 validation_role 区分选择与校准；最终测试输入分别为 `in_domain_test/test_scores.csv`（3,600）与 `cross_test/test_scores.csv`（8,000）。各行保留内容哈希与子集标记。

模型选择只用 model_selection；C 的独立阈值只用 calibration 中 900 个真实视频。融合校准应使用 calibration，不能再把 model_selection 当作未参与开发的评价集。正式模型为 `selection/selected/model.joblib`，需连同 selected_config、requirements-lock 和源代码交付。run_02 的两个测试集使用同一最终模型。

正式 native_stride 模式保留真实时间戳和至少 8 个实际帧；短片使用整段窗口，其余约 2 秒。调用 predict_frames 时须遵守模型内的采样策略，不能直接传入 A 的稀疏帧，也不能给低帧率视频伪造 8 fps 时间戳。

Branch C 返回方向一致的连续异常证据 `branch_c_score` / `anomaly_score`。Isolation Forest 和 One-Class SVM 均使用 `-score_samples(X)`，分数越大越异常。不同算法或不同模型之间的绝对值不能直接比较。scikit-learn 原始 `score_samples` 的方向见 [Isolation Forest 官方接口](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)。

分数 CSV 同时包含 `run_kind` 与 `protocol`。合并前应检查实验来源及协议一致，禁止把 synthetic_smoke 的结果当作正式实验输入。模型保存协议，evaluate 会在解码测试数据前检查它与测试清单一致；已有模型输出目录也会在开始运行前检查，避免覆盖结果。

分数 CSV 另含 `content_sha256`，记录实际评分文件的字节哈希。以 clip_id 合并后，若 A/B 处理同一原始文件，可核对哈希；若上游交付视频与 NPZ 两种表示，文件哈希自然不同，需由数据负责人提供表示间的稳定 ID 映射，不能把不同文件哈希直接当作标签或融合特征。

## 训练融合器

1. 在全组共用的 train 清单上约定 K 个按组隔离的折。可由数据负责人填写 fold_id，也可先运行 Branch C 的 `folds` 命令，再共同使用导出的 `manifest_with_folds.csv`。该步骤只哈希 train 文件，无需先提取特征或拟合 C。
2. A、B、C 使用相同折。第 k 折模型仅使用其余折；C 的 StandardScaler 和异常检测器还进一步只使用这些折中的 label=0。
3. 三个分支为第 k 折全部视频输出分数。合并所有留出折后，每个训练 clip 恰好得到一次 OOF 分数。
4. 使用 `clip_id` 一对一合并 A/B/C 的分数和标签，检查缺失、重复及 fold_id 一致后，训练融合器。不要按 CSV 行号合并。
5. 最终各分支重新拟合完整 train。对 val/test 使用最终模型输出；融合器的校准和阈值规则固定后，才进行最终 test。

OOF 的分组将合并 group_id、pair_id、duplicate_id、source_video_id 和相同文件哈希形成的连通组。外部传入的 fold_id 若把连通组拆到不同折，程序会拒绝。独立连通组不足 K 时可以减少 K；某折训练侧不足 4 个真实视频时，需要增加真实训练数据或在组约束内调整折方案，减少 K 反而可能进一步减少训练样本。不能通过引入验证/测试数据补足。

`folds` 导出无预测的 `fold_assignments.csv`、完整 `manifest_with_folds.csv` 和 `fold_plan.json`。后者记录每折留出数量、真实拟合数量和计划标识；C 的 `oof_audit.json` 保存实际复用的 `fold_plan.plan_id`，两者应一致。原始公共清单不会被修改，train/val/test 分区不变；新增的 train 内容哈希在解码前核对。更换输入版本后应由数据负责人重新冻结清单。

全组仍需冻结原始清单顺序。共享 fold_id 保证留出组一致，不代表随机模型在重排训练行后也一定得到相同分数。`folds` 只检查 train 字节重复与给定分组，不读取 val/test 文件，不能代替完整数据审计。

如果上层预处理或特征编码器需要学习参数，也必须在每折训练侧拟合；Branch C 当前的手工描述符没有此类参数。A/B 若使用微调编码器，C 不能复用包含当前留出折的微调特征。本实现不使用 A 的微调嵌入。

不要用最终模型对训练集生成的分数代替 OOF。当前不导出此类 `train_scores.csv`，以减少误用。

OOF 与最终模型的真实训练量分别为 6,720 与 8,400。尤其 One-Class SVM 的原始分数尺度可能随拟合样本量变化；融合应在 calibration 上核对最终分数的分布与校准，不能假定 C 已经输出统一概率尺度。

若运行了 `select`，C 的 OOF 必须使用其 `selected_config.json`；val/test 使用对应 `selected/model.joblib`。候选目录下的 CSV 标为 `score_kind=candidate_validation`，只用于模型比较，不是 OOF，也不能作为不同候选的集成分数。选择后的 val 指标存在选择偏差；全组需在正式实验前约定模型选择与融合校准的数据使用方案。

## 在线 Python 接口

```python
from branch_c.pipeline import predict_video, predict_frames

result = predict_video("artifacts/c_final/model.joblib", "new_video.mp4")

# 如果上层已有均匀采样、未归一化的 RGB uint8 帧：
result = predict_frames(
    "artifacts/c_final/model.joblib",
    frames_rgb, timestamps_seconds, clip_id="clip_0001"
)
fusion_input_c = result["anomaly_score"]
```

模型文件包含特征配置，推理会复用模型的配置，不能随意用另一套采样规则。连续多视频服务可复用已加载 bundle 和 `score_feature_result`，避免逐条重复加载；当前公开便捷接口以离线实验为主。

## JSON 语义

| 字段 | 含义 |
| --- | --- |
| anomaly_score | 整段视频异常证据，融合器的 C 输入 |
| is_probability | 恒为 false |
| threshold | 有角色时仅真实 calibration val；否则真实 val 的保守经验分位数 |
| is_anomalous | `anomaly_score > threshold`，不是 AI 分类结论 |
| timeline | 独立窗口模型的异常分数和起止秒数 |
| most_anomalous_window | 当前视频内窗口模型分数最高的区间 |
| descriptor_deviations | 相对真实训练均值的标准化偏离，便于查看异常维度 |
| model_id | 配置、特征签名和训练清单共同确定的模型标识 |
| run_kind | research 或 synthetic_smoke |
| protocol | 数据清单对应的实验协议，保持与训练/验证一致 |
| inference_seconds | 不含模型加载的单次视频推理耗时；不含 Python 启动或输出写入 |
| timings | 模型加载、解码采样、描述符、评分阶段及加载在内的总耗时 |

时间线模型只使用真实 train 的约 2 秒窗口；native_stride 模式对短片使用整段窗口。每个视频最多均匀选择 4 个窗口。它与整段模型分开拟合，窗口分数只适合在该窗口模型下查看；不能套用整段阈值、不能宣称是窗口级 AI 概率，也不是经人工标注验证的定位。长视频的窗口贡献仍可能更多，正式实验应检查视频时长分布。

descriptor_deviations 是逐维标准化偏离，不是 SHAP 或 Isolation Forest 的因果解释。融合层若显示贡献，应使用融合器自身的可解释方法。

Branch C 不执行融合概率校准、风险分档或 A/B 模型训练。融合器若使用同一个 val 同时调参、校准和选阈值，应进一步按组分子集或使用交叉验证；最终 test 始终封存。

`error_review.csv` 供人工整理 C 的误报/漏检，包含标签和最终 test 结果，只用于测试后报告与局限性分析，不作为融合训练输入。审阅原因初始为空，不能从 C 分数自动生成原因。已有 CSV 的评价目录不会被覆盖，避免丢失组员笔记。
