# Branch C 方法与正式实验计划

## 项目依据

实现对应所提供 PDF 第 2 页和 PPT 第 10 页的真实视频异常分支：从运动和时序变化的摘要学习真实视频分布，为未知生成器提供候选补充证据。这是待验证的假设，不是本次实现已证明的结论。折外输出对应 PPT 第 11 页的融合防泄漏设计。

两份材料的数值目标存在版本差异：PDF 第 3 页给出目标值，较新的 PPT 第 13 页表示暂不设数值成功门槛。代码不硬编码成功标准；正式实验前由全组确认最终报告采用的版本。

## VIDA-Guard32K 固定协议

正式数据已下载并完成交付校验。已完成的正式运行位于 artifacts/vida_guard32k/run_02，预先冻结六候选、代码与清单哈希。训练 16,800 个视频中，仅 8,400 个真实视频拟合 C；五个公共折沿用交付值，每折实际拟合 6,720 个真实视频。验证 model_selection 与 calibration 各 1,800 个，模型排名仅使用前者，最终阈值仅来自后者中的 900 个真实视频。两套封存测试分别为 3,600 与 8,000 个视频，使用同一模型。结果以该目录的原始 JSON、verification.json 与 report/index.html 为准。

正式 native_stride 模式保留至少 8 个实际观测，按整数原始帧步长接近 8 fps，不复制、不插值帧；时长允许 0.5–20 秒，短片采用整段窗口。像素/亮度/边缘变化按实际时间换算到 1/8 秒，光流以图像边长/秒表示。频谱仅保留 0–2 Hz，分带为 0–0.5、0.5–1.2、1.2–2 Hz。短片频率分辨率更粗，整数抽帧没有时间低通抗混叠；来源、编码、时长和帧率的残余差异仍可能影响分数。以下默认固定 8 fps 的说明适用于兼容保留的 uniform_cfr 模式。

六个预定候选中正式选中 RBF One-Class SVM（gamma=scale，nu=.01）。以下 Isolation Forest 参数说明保留为预定基线与通用入口；实际最终模型以 selected_config.json 为准。

## 描述符

默认每秒采样 8 帧，缩放至 96×96，灰度像素除以 255。每个视频产生 37 维描述符。

| 描述符族 | 维数 | 计算 |
| --- | --- | --- |
| 像素变化 | 12 | 相邻帧绝对差的均值、标准差、95 分位，以及平均亮度变化；各序列再取 mean/std/p90 |
| 边缘变化 | 3 | 相邻帧 Laplacian 差的平均绝对值，再取 mean/std/p90 |
| 光流 | 15 | Farneback 光流速度的均值、90 分位、空间标准差、方向一致性、前向光流重映射残差；各序列取 mean/std/p90 |
| 时间频率 | 5 | 将灰度缩至 24×24，逐像素去时间均值、Hann 加窗、rFFT 后平均能量；低/中/高频能量占比、频谱重心和归一化谱熵 |
| 整体时序变化 | 2 | 帧平均亮度的时间标准差、相邻光流平均速度变化的均值 |

光流速度乘以采样 fps 并除以图像边长，单位为“每秒移动多少个图像边长”。前向光流把当前帧重映射到上一帧坐标，排除出界像素。频率区间以 Nyquist 比例划分：`(0,.25]`、`(.25,.6]`、`(.6,1]`；默认 8 fps 下约对应 `0–1`、`1–2.4`、`2.4–4 Hz`。窗口长度会影响频率分辨率，需统一比较协议。

Farneback 参数固定为 `pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2`，参见 [OpenCV 官方跟踪接口](https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html)。这些是可复现的轻量基线选择，不代表经过项目数据调优的最优描述符。

## 拟合与阈值

对 label=0 且 split=train 的整段描述符拟合 StandardScaler，再拟合 200 棵 Isolation Forest，单树最多抽取 256 个真实样本，seed=42。`contamination='auto'` 只满足库的内部设置，最终不使用其默认二值预测阈值。对照方法是 RBF One-Class SVM，`nu=.05, gamma='scale'`，也只在真实 train 上拟合标准化器与模型。标准化的训练统计量约束见 [StandardScaler 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。

异常分数是 `s(x)=-score_samples(x)`。默认阈值取真实 val 分数的 95% 分位，使用 `method='higher'`，判异常采用严格大于。它让经验真实验证误报率不超过约 5%，但不保证新来源真实视频也有 5% 误报率。极小 val 集的阈值非常粗糙，脚本会提示。

生成视频标签可用于分层折分配、验证集模型选择及评价，但不参与异常模型、标准化器或当前阈值的拟合。默认 fit 配置固定；正式比较前冻结下述有限候选表，合成流程只用于检查代码。

## 预先限定的验证集比较

`select` 只比较 6 个候选：IF 树数 `{100,200}` × max_samples `{128,256}`，以及 RBF One-Class SVM nu `{.01,.05}`。特征和解码规则、seed、目标真实误报率、留出生成器/来源对所有候选相同。先保存候选计划及哈希，再读取真实 train 与 val；test 只做元数据检查，其文件不读取也不计算内容哈希。特征只提取一次供候选和选定模型复用。

每个候选的标准化器和异常模型只拟合真实 train。所有候选使用同一 val，分别从真实 val 分数取得阈值，再计算 val AUROC 和 macro-F1。先按 AUROC 降序，再按 macro-F1 降序，精确并列时按预定顺序；原始 IF 200/256 基线排在最前。运行时长不参与排名，以免机器负载改变选择。真实训练量少于 max_samples 时实际采样量取真实样本数，两种设定可能完全相同，应保留并列记录。

完整计划、排名、逐候选 val 分数、实际读取 ID/哈希和选定配置均保存。任一候选失败就终止，不跳过失败候选后声称选出了最优模型。选定算法再在相同真实 train 上确定性重拟合，并只拟合一个独立窗口模型；保存模型选择来源。OOF 使用选定配置和全组共用的 fold_id。

这个过程要求 val 两类都存在。只有真实 val 时可以运行固定基线 fit，不能用训练标签、测试标签或合成样本补充正式 val。选择后的 val 指标有选择偏差，不能代替最终 test。若全组需要用 val 做融合校准，应事先划分按组隔离的模型选择/校准子集或约定交叉验证方案，避免将同一 val 反复用于选择与效果宣称。

## 实验顺序与全组接口

1. 冻结公共分区、留出生成器/来源、近重复组、解码规则和 ID；检查真实/生成的分辨率、时长、帧率、来源分布。避免标签与预处理差异混淆。
2. 只用训练和验证运行 Isolation Forest 基线。检查真实来源误报与明显损坏视频，记录剔除标准，禁止为了提高 test 指标事后剔除。
3. 在相同 val 上运行上述预先声明的 6 个候选比较，冻结选定配置和阈值；记录选择理由，不对最终 test 反复调参。
4. 使用全组同一折生成 OOF，导出 C 分数。全组训练 A+B 与 A+B+C 融合，并固定校准和分类阈值。
5. 最终分别在 in-domain 与 cross-generator test 上运行一次冻结配置，保存原始视频分数、完整配置、清单哈希和拟合 ID。
6. Branch C 单独报告 AUROC、macro-F1、balanced accuracy、precision/recall、specificity、EER、混淆矩阵。每个生成器的 AUROC 使用该生成器样本与同一真实 test 池，不能对全是生成视频的子集计算 AUROC。
7. 按防泄漏连通组 bootstrap 得到 AUROC 和 macro-F1 的 95% 区间。代码提供单模型区间；A+B 与 A+B+C 的差异区间必须由融合评价脚本在相同组重采样索引上做配对比较，不能用两个单模型区间代替。
8. 抽取误报真实视频与漏检生成视频，记录快运动、镜头切换、低光、强压缩、稳定合成视频等失效类型。测量指定机器的 10 秒视频端到端延迟、峰值内存和模型体积。

原始异常分数不计算 Brier/ECE；这些指标依赖概率，应在融合校准结果上评价。

## 固定阈值的失效案例审阅

evaluate 在同一次测试评分后导出 `error_review.csv`，不重新解码或调阈值。FP 指真实视频分数高于固定阈值；FN 指生成视频分数小于或等于阈值。文件包含 clip_id、路径、组、来源/生成器、分数、阈值、阈值差和输入条件。先列 FP 再列 FN，各类按距离本模型阈值的绝对差降序，便于查看强误判；阈值差不代表概率或可跨模型比较的置信度。

人工打开视频后填写 `review_status`、`failure_type` 和 `notes`。可按观察记录快速运动、剪辑/镜头变化、低光、压缩、来源差异、稳定生成内容或无法判断等情况。这些类别是审阅提示，不是新增检测模块，也不是程序自动推断的因果标签。报告应区分直接观察与推测，保留有代表性的误报和漏检；不能据此回调最终 test 的阈值或挑选性剔除视频。

没有错误时仍写出表头，并明确该固定阈值下没有案例；合成演示中的零错误不能作为泛化证据。程序优先保护已有分数和人工审阅文件，遇到部分完成的评价目录也要求新目录。`test_report.json` 在两份 CSV 导出成功后写入。各阶段分数 CSV 保留 `content_sha256`，便于与原始文件核对。

元数据区分真实视频 `native_width/native_height/native_fps`、预采样输入的 `input_width/input_height` 和描述符工作尺寸 `feature_image_size`。NPZ 的输入尺寸不是上游原始视频尺寸，缺失的 native 字段保持空白。缓存格式版本独立于数值特征签名；本次元数据修正不改变描述符或要求重拟合已有兼容模型，但旧缓存会被新格式隔离并重新提取。

## 公共折准备与运行前检查

`folds` 将训练元数据和相同文件 SHA256 形成的连通组交给固定 seed 的 StratifiedGroupKFold，或严格复用清单中的 fold_id。它不执行视频解码或模型拟合，也不读取 val/test 文件。完整导出清单保留原来的标签、来源、分区和其他元数据，只新增 train 的折号与内容哈希，并将路径解析到当前机器。每折真实拟合数量在 `fold_plan.json` 中明确记录；少于 4 个真实训练描述符的方案直接拒绝。

OOF 使用相同分折函数，先检查元数据、读取训练文件哈希、验证重复组与真实拟合数量，再准备描述符。实际分折 plan_id 进入 OOF 审计；使用同一导出清单时应与准备阶段一致。固定输入哈希在缓存命中和冷提取前都核对，防止折计划冻结后输入被替换。fold_id 不作为模型特征，验证/测试数据不用于补足某折的训练数据。

固定 fit 也在读取视频前验证至少 4 个真实 train 和非空真实 val。已有的 fit/OOF 完成报告、分数和分折文件得到保护；拟合完成报告在模型与验证分数保存后写入。原有随机种子、算法和特征不变；同一清单顺序下，提前分折和直接 OOF 的折号与分数应精确一致。

## 运行开销的测量口径

`benchmark` 每次用当前 Python 启动新进程，载入相同模型、解码同一输入并执行完整预测（整段、窗口和描述符偏离），不复用特征缓存。保存每次测量及 min/median/max：进程墙钟耗时包含解释器启动、库导入、加载、解码、特征与预测；另列模型加载、解码采样、特征提取、评分说明阶段。普通 predict 保留 `inference_seconds`（不含模型加载），并新增 `timings` 说明加载在内的时间，避免把两种口径混淆。输出文件写入和输入哈希不计入推理延迟。

峰值内存取独立 worker 的 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，包含原生 OpenCV/NumPy 分配、解释器与模型加载，不减去基线进程内存。macOS 原值按 bytes、Linux 按 KiB 转为 bytes；当前支持这两个平台。RSS 是整个进程的驻留内存高水位，见 [Python resource 官方文档](https://docs.python.org/3/library/resource.html)，不是 Python-only tracemalloc 或 GPU 显存。每次独立进程隔离前一次运行的高水位，但不能清空 OS 文件缓存。

报告保留机器架构、系统版本、CPU 数、库版本、线程设置、视频/模型 SHA256 和模型文件字节数。模型体积包含整段模型、窗口模型、阈值与审计元数据。正式报告使用全组约定机器及真实 10 秒视频；记录原始分辨率/帧率，至少重复 3 次，避免后台负载。NPZ 演示只验证预采样路径，不能代替视频解码耗时；TOY MP4 测量只证明命令可运行。

## 结论边界

正式指标以 VIDA-Guard32K 封存测试为准，TOY 演示仍仅用于软件验证。跨来源测试同时更换真实来源和生成器，不能声称隔离了纯生成器泛化效应，也不能将整理后的子集称为未修改的官方 GenVidBench。A+B 对比 A+B+C 的配对融合增益仍依赖组员输出。异常可能来自正常但少见的摄像内容或域偏移；稳定的生成视频也可能不异常。

## English methods draft

Branch C models normal video dynamics using real training videos only. In the completed VIDA-Guard32K run, native CFR frames are sampled with an integer stride close to 8 fps, retaining at least eight actual observations and their timestamps. We extract 37 summary descriptors of inter-frame pixel and edge variation, dense optical-flow speed, motion-compensated residuals, and temporal-frequency energy within the common 0–2 Hz support. The final StandardScaler and RBF one-class SVM (gamma=scale, nu=0.01) are fitted exclusively on 8,400 real training clips. This estimator won a predeclared six-candidate comparison using only the model-selection validation subset; generated clips never enter scaler or anomaly-estimator fitting. We negate score_samples so that larger values indicate stronger anomaly evidence, not an AI-generation probability.

The supplied group-isolated five folds are preserved. Each fold fits its own scaler and estimator on 6,720 real clips, exporting exactly one out-of-fold score for each of the 16,800 training clips. A separate set of 900 real calibration clips defines the final standalone threshold. An independent real-only window model provides exploratory temporal evidence without window-level probability calibration. The frozen model achieves AUROC 0.841 in-domain and 0.895 on cross-test. Cross-test changes both the generator families and the real-video source. The contribution of C to fusion still requires a paired A+B versus A+B+C evaluation with teammates' outputs.
