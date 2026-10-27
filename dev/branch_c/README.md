# Branch C 真实视频异常检测

## GitHub 交付说明

本目录是 Lang Heran 的 Branch C，可独立安装和运行。以下命令均从仓库的 `dev/branch_c/` 执行。

已上传源码、配置、测试、中文交接说明及 [汇总实验结果](results/README.md)。原始视频、视频帧、正式逐视频清单与预测、模型文件、缓存、完整案例报告和数据压缩包均需团队另行传输，未放入 Git。下文历史 `data/`、`artifacts/` 链接仅在拥有相应本地数据和实验输出时可用。

```bash
cd dev/branch_c
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/smoke_test.py --output-dir artifacts/github_smoke
```

Smoke 会本地生成明确标注 TOY 的合成素材；它验证流程，不代表正式准确率。Branch C 使用 CPU，无需预训练权重、GPU 或外部 API。

**协议尚未与 Branch A 对齐。** 当前仓库 Branch A 记录四轮留二、约 70/15/15 划分；C 的历史最新结果为 28 轮留二、60/10/10/20 划分。全组融合前必须统一清单、分组、验证角色和公共 OOF 折，不能直接拼接现有分数。上次讨论的“固定封存两家、六家内部三轮选型”也尚未执行。

C 的运动与频谱描述符需要连续视频观测或满足采样要求的帧序列，不能直接把 A 使用的每视频八张稀疏 PNG 当作等价输入。分数是异常证据，不是 AI 概率。集成说明见 [docs/INTEGRATION.md](docs/INTEGRATION.md)。



最新实验已完成 **6 已见 / 2 未见生成器轮换，共 28 轮**。新清单见 `data/VIDA-Guard32K/rotation_6seen_2unseen_v1`；[轮换报告](artifacts/vida_guard32k/rotation_6seen_2unseen_v1/report/index.html)与[轮换交接说明](docs/GENERATOR_ROTATION_ZH.md)记录每轮选择、OOF、测试和分组审计。未见生成器 AUROC 的轮次均值为 0.874，macro F1 均值为 0.652；这些轮次相互依赖，且是已评价素材池的重新划分，不能称为新盲测或直接与旧协议比较提升。

以下 run_02 为原固定 4 已见 / 4 未见协议的历史实验记录，模型和报告保留。

这是 Group 42 项目中 Lang Heran 负责的 Branch C。VIDA-Guard32K 已下载、解压并通过 32,000 个视频的 SHA256 校验。正式训练、五折 OOF 与两套固定模型测试已完成。选定 svm_nu_0.01；同分布 AUROC 0.841，跨来源 AUROC 0.895。TOY 演示与正式结果分开保存，不能混用。

输入视频 → 37 维运动与时序描述符 → 只用真实训练视频拟合的 StandardScaler 和选定的一类异常模型 → 异常分数。六候选包含 Isolation Forest 和 One-Class SVM；正式选定 RBF One-Class SVM，nu=.01。分数越大表示越偏离真实训练视频的分布，**不是 AI 生成概率**。

正式结果与图表：[中文实验报告](artifacts/vida_guard32k/run_02/report/index.html)。交接与讲稿见 [HANDOFF_ZH.md](docs/HANDOFF_ZH.md)，限制与运行环境见 [MODEL_CARD.md](docs/MODEL_CARD.md)。

## VIDA-Guard32K 正式运行

原始交付在 `data/VIDA-Guard32K/delivery/`，视频在 `data/VIDA-Guard32K/media/videos/`，校验收据为 `data/VIDA-Guard32K/receipt.json`。映射后的清单和配置在 `data/VIDA-Guard32K/branch_c/`。不要把 packages 中的 ZIP 当作视频路径。

| 分区 | 视频数 | 用途 |
| --- | ---: | --- |
| train | 16,800 | 仅真实 8,400 用于 C 拟合；全部用于公共五折 OOF |
| validation model_selection | 1,800 | 六个预定候选排名 |
| validation calibration | 1,800 | 仅其中 900 个真实视频决定最终阈值；生成样本不参与 C 的选择、拟合或阈值，可用于融合校准 |
| in_domain_test | 3,600 | 冻结模型的同分布评价 |
| cross_test | 8,000 | 冻结模型的跨来源评价 |

正式配置适配原始短视频和 4 fps 输入：`native_stride` 按整数原始帧步长接近 8 fps，必要时降低步长保留至少 8 个真实观测，使用真实时间戳，不复制或插值帧。短片采用整段窗口，其他采用约 2 秒滑窗。光流按实际时间归一化，像素/亮度/边缘变化按 1/8 秒计，频谱只使用共有的 0–2 Hz。统一缩放至 96×96。时长、编码、分辨率与来源差异仍可能影响分数。

本机已经完成导入。正式全流程和当前进度：

```bash
.venv/bin/python scripts/run_vida_guard32k.py --workers 8 --output-dir artifacts/vida_guard32k/run_02
cat artifacts/vida_guard32k/run_02/status.json
```

脚本依次准备公共折、选择模型、生成 OOF、冻结模型、分别评价两套 test、测量真实 10 秒视频。`--phase development/oof/test/benchmark` 可分阶段运行。只复用有完成记录且源代码、配置、清单未变的阶段；部分输出保留，应选择新运行目录并复用内容缓存。run_01 在候选拟合前停止，仅保留早期 4 workers 的特征准备记录；正式实验使用 run_02。

换机器先保持 delivery/media 的布局，然后执行 `scripts/import_delivery.py --delivery-root data/VIDA-Guard32K/delivery --media-root data/VIDA-Guard32K/media --output-dir data/VIDA-Guard32K/branch_c`。导入只映射接口，不改标签、分区、组或原有 fold 0..4；新实验使用新的 output-dir。

run_02 的最终模型为 `selection/selected/model.joblib`，排名为 `selection/selection_report.json`，验证分数为 `selection/selected/validation_scores.csv`，公共折为 `common_folds/fold_assignments.csv`，融合训练输入为 `oof/oof_scores.csv`。两套测试分别在 `in_domain_test/` 与 `cross_test/` 下，包含 `test_scores.csv`、`test_report.json`、`error_review.csv`；资源报告为 `runtime.json`。

两套测试复用同一模型和阈值。cross_test 同时改变生成器与真实来源，不能解释为仅生成器变化的因果效果。数据是整理后的 VIDA-Guard32K，不能冒称未修改的官方 GenVidBench。A/B 与融合增益需要对应组员的输出才能评价。

## 软件自检与 TOY 演示

当前目录已安装本地 `.venv`，可直接运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py --output-dir artifacts/my_smoke
```

第二条命令会生成 72 个明确标为 TOY 的合成片段，走完特征提取、真实类拟合、5 折 OOF、固定阈值测试和 MP4 推理。历史验证结果保存在 `artifacts/proposal_iteration_04`；新运行请换目录。TOY 的标签只代表构造方式，不能用于报告正式检测准确率。

没有训练数据时，也可以接着检查新增命令：

```bash
.venv/bin/branch-c select --manifest artifacts/my_smoke/toy_data/manifest.csv --output-dir artifacts/my_smoke/selection --cache-dir artifacts/my_smoke/cache
.venv/bin/branch-c benchmark --model artifacts/my_smoke/selection/selected/model.joblib --video artifacts/my_smoke/toy_video.mp4 --repeats 3 --output artifacts/my_smoke/runtime.json
.venv/bin/branch-c folds --manifest artifacts/my_smoke/toy_data/manifest.csv --folds 5 --output-dir artifacts/my_smoke/common_folds
.venv/bin/branch-c oof --manifest artifacts/my_smoke/common_folds/manifest_with_folds.csv --folds 5 --output-dir artifacts/my_smoke/common_oof --cache-dir artifacts/my_smoke/cache
```

这里使用的是 smoke 生成的 3 秒 TOY MP4，只验证命令和输出。最新迭代另在 10 秒 TOY MP4 上验证了测量流程；正式资源结果见 artifacts/vida_guard32k/run_02/runtime.json，使用本机 CPU 和真实 10 秒视频；这不代表指定 GPU 工作站的测量。

正式数据接入后，解码器升级为 cfr-native-stride-v3，描述符版本为 motion-frequency-time-aware-v2，特征签名包含采样模式。历史 TOY 模型应保留为记录，使用当前源码重新运行 smoke 或 fit；旧签名模型会拒绝加载。历史迭代记录见 docs/ITERATION_LOG.md。

新电脑建议 Python 3.12，安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

也可 `python -m pip install -e '.[dev]'` 安装允许范围内的依赖。复现已保存模型时使用锁定版本，模型加载会检查运行库版本。无需 GPU、PyTorch、预训练权重或外部 API。

## 其他数据的通用命令

先请数据处理同学按照 [数据接口](docs/DATA_CONTRACT.md) 交付 CSV；[examples/manifest.csv](examples/manifest.csv) 仅展示格式，不含视频。In-domain 和 cross-generator 使用不同清单和不同运行目录。在 cross-generator 配置的 `heldout_generators` 与 `heldout_real_sources` 中填写实际留出的名称；不要使用示例名称。

下面从项目根目录运行，路径中的 `data/manifest.csv` 替换成实际清单：

```bash
# 1. 检查标签和分区元数据，不读取视频。
.venv/bin/branch-c validate --manifest data/manifest.csv --config configs/baseline.json

# 2. 生成给融合用的 OOF 分数，仅处理 train。
.venv/bin/branch-c oof --manifest data/manifest.csv --config configs/baseline.json --folds 5 --output-dir artifacts/c_oof

# 3. 拟合最终分支，只用真实 train；只从真实 val 选异常阈值。
.venv/bin/branch-c fit --manifest data/manifest.csv --config configs/baseline.json --output-dir artifacts/c_final

# 4. 实验方案确定后评估 test，阈值不会重新拟合。
.venv/bin/branch-c evaluate --manifest data/manifest.csv --model artifacts/c_final/model.joblib --output-dir artifacts/c_test

# 5. 单个新视频。
.venv/bin/branch-c predict --model artifacts/c_final/model.joblib --video data/new_clip.mp4 --output artifacts/new_clip_result.json

# 6. 在约定机器上测量实际 10 秒视频，每次独立启动进程。
.venv/bin/branch-c benchmark --model artifacts/c_final/model.joblib --video data/val_10s.mp4 --repeats 3 --output artifacts/c_runtime.json
```

需要按规划比较参数时，在 OOF 和最终 test 前运行以下可选流程。`select` 固定比较 IF 的 4 个组合（树数 100/200 × max_samples 128/256）和 SVM 的 2 个 nu（.01/.05）；输入配置中的特征、评价规则、留出名称和 seed 保留，模型参数使用这份固定候选表。候选按 val AUROC、val macro-F1、预定顺序排名，精确并列时优先原始 IF 基线。阈值仍只从真实 val 确定。

```bash
.venv/bin/branch-c select --manifest data/manifest.csv --config configs/baseline.json --output-dir artifacts/c_selection
.venv/bin/branch-c oof --manifest data/manifest.csv --config artifacts/c_selection/selected_config.json --folds 5 --output-dir artifacts/c_selected_oof
# selected/model.joblib 已只在真实 train 上拟合，按其固定阈值测试。
.venv/bin/branch-c evaluate --manifest data/manifest.csv --model artifacts/c_selection/selected/model.joblib --output-dir artifacts/c_selected_test
```

选择过程只读取真实 train 和全部 val，完整清单中的 test 仅检查元数据。保存 `selection_plan.json`、全部候选的验证分数与排名、`selected_config.json` 和 `selected/model.joblib`；失败运行保留计划，不生成完成报告。模型选择要求 val 同时有真实和生成视频；只有真实 val 时继续使用上面的固定配置 fit，等数据补齐后再比较。选择后 val 指标已用于调参，不能当作独立泛化结果。

`benchmark` 报告各次运行、min/median/max 耗时、进程峰值 RSS、模型字节数、平台和库版本。它支持 macOS/Linux，峰值内存包括解释器、导入、模型加载和本地库分配；不是模型新增内存。进程总耗时包含启动，阶段耗时另列；不使用特征缓存，OS 文件缓存不受控制。NPZ 与实际视频解码的开销不同，正式报告应使用约定机器和真实 10 秒视频。已有选择目录和资源报告不会被覆盖。

可选 `extract` 命令预先缓存指定 split 的描述符；默认只提取 train。`fit` 仅解码真实 train 和全部 val，仍检查完整清单的分区元数据；生成 train 的文件可以暂缺。`oof` 必须读取全部 train，为真实和生成视频都输出折外分数。缓存按文件内容、特征和解码器版本、配置及运行库版本命名，修改预处理后不会复用旧特征。发生解码错误时直接报告错误，不静默跳过或补零。先修正清单或上游数据，再重跑。

命令默认 `--run-kind auto`：清单中 `protocol=synthetic_smoke` 会自动保留合成演示标记，并拒绝改报为 research。其他清单按 research 运行，但这不会自动认证数据属于正式基准。CSV 分数也保留 `run_kind` 和 `protocol`，避免组员合并分数时丢失实验来源。

## 交付给其他组员

训练文件齐备后，可在运行 OOF 前导出公共折：

```bash
.venv/bin/branch-c folds --manifest data/manifest.csv --config configs/baseline.json --folds 5 --output-dir artifacts/c_folds
.venv/bin/branch-c oof --manifest artifacts/c_folds/manifest_with_folds.csv --config configs/baseline.json --folds 5 --output-dir artifacts/c_common_oof
```

A/B/C 共同使用导出的 `manifest_with_folds.csv`，上述 fit/select/evaluate 命令也可将原始 manifest 路径替换为这份清单。它保留原有 train/val/test 分区、标签和来源，仅新增 train 的 fold_id、内容哈希，并将路径解析为当前机器的绝对路径。换机器时由数据负责人统一映射路径并核对哈希。原始清单不会被修改。

`c_folds/fold_assignments.csv` 没有预测分数；OOF 目录下的同名文件包含实际 C 分数。`folds` 不检查视频解码，也不读取 val/test 文件。OOF 至少需要 5 个真实 train，且每个具体训练折都必须留下至少 4 个；只有 4 个真实 train 时可以先运行固定 fit。

| 文件 | 用途 |
| --- | --- |
| `artifacts/c_folds/manifest_with_folds.csv` | A/B/C 共用的训练折和固定输入哈希 |
| `artifacts/c_folds/fold_plan.json` | 分折审计、每折真实拟合数量和 plan_id |
| `artifacts/c_oof/oof_scores.csv` | 训练融合器的 Branch C 输入，每个 train clip 一行 |
| `artifacts/c_oof/fold_assignments.csv` | A/B 复用同一 fold_id，按 clip_id 对齐 |
| `artifacts/c_oof/oof_audit.json` | 每折真实拟合 ID 和留出 ID，检查泄漏 |
| `artifacts/c_final/validation_scores.csv` | 验证集融合输入 |
| `artifacts/c_final/model.joblib` | 最终真实视频异常模型、独立窗口模型、阈值和配置 |
| `artifacts/c_test/test_scores.csv` | 最终测试融合输入 |
| `artifacts/c_test/test_report.json` | AUROC、macro-F1、EER、混淆矩阵、分生成器指标和按组 bootstrap 区间 |
| `artifacts/c_test/error_review.csv` | 固定阈值的 FP/FN、输入元数据和待人工填写的原因/笔记 |
| `artifacts/new_clip_result.json` | 视频分数、窗口时间线和描述符偏离信息 |
| `artifacts/c_selection/selection_report.json` | 有限候选的验证集排名、真实拟合 ID 和未读 test 的审计记录 |
| `artifacts/c_selection/selected_config.json` | 给 OOF 复用的选定配置 |
| `artifacts/c_runtime.json` | 耗时、峰值进程内存、模型体积和运行环境 |

参见 [融合接口](docs/INTEGRATION.md)、[方法和实验计划](docs/METHOD_AND_EXPERIMENTS.md)、[个人交接与汇报](docs/HANDOFF_ZH.md)、[开发阶段模型卡](docs/MODEL_CARD.md)。实际模型和视频被 `.gitignore` 排除，避免将数据提交进代码仓库。

## 当前验证边界

自动化测试检查真实类拟合限制、分区泄漏、OOF 留出、分数方向、缓存失效、帧输入、固定阈值和指标边界。完整 TOY 流程另测试 MP4 解码与公开接口。它们证明实现流程可用，不能证明跨生成器检测有效。正式训练、固定模型测试和案例观察已完成；A+B 与 A+B+C 的配对消融仍需全组对齐分数。

当模型清单声明了 `protocol`，最终测试清单必须使用相同值。纯真实来源子集没有真假两类对照，报告中的 AUROC、macro-F1 等比较指标使用 `null`，仍报告真实误报率和 specificity；生成类 precision 在没有任何预测阳性时也使用 `null`。

模型的 `is_anomalous` 只表示超过真实验证集确定的异常阈值。最终概率、风险等级及 Brier/ECE 应由融合和校准层提供。只加载自己或团队可信的 joblib 模型文件。

评价会同时导出 `test_scores.csv` 与 `error_review.csv`，最后写入 `test_report.json` 作为完成记录。审阅表的 FP/FN 仅用于二分类评价，人工观看后填写 `failure_type` 和 `notes`；程序不会从分数推断失效原因。已有分数或审阅文件的目录会被拒绝，以免覆盖人工笔记。各阶段分数 CSV 的 `content_sha256` 记录被评分文件的字节哈希，合并时仍以 `clip_id` 为主键。
