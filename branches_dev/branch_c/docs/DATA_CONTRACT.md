# Branch C 数据交付接口

请提供一个 UTF-8 CSV 和可访问的视频或帧文件。每行代表一个视频。一个 CSV 对应一个评估协议；同一协议可用 evaluation_subset 标记不同测试子集，保持 clip_id 唯一。CSV 由数据处理负责人划分，Branch C 不自行改动分区。

## VIDA-Guard32K 现行接口

使用 `scripts/import_delivery.py` 导入同学的清单。`video_path` 相对 media 根目录解析；原始 validation 映射为 val，两个 test 映射为 test 并保留 evaluation_subset。family 映射到 source，生成视频另写 generator；fold 原样映射为 fold_id，sha256 映射为 content_sha256。原始字段仍保留。

`validation_role` 非空时，所有 val 必须填写 model_selection 或 calibration，连通组不得跨角色。六候选只使用 model_selection；最终阈值只用真实 calibration。分数导出保留 validation_role、evaluation_subset。evaluate 通过 `--test-subset in_domain_test/cross_test` 分别评价，同一协议、模型和阈值保持不变。

正式配置 `native_stride` 接受 0.5–20 秒 CFR、4–240 fps、至少 8 个实际帧。导入核查交付 JSON 的原始逐帧时间戳确为 CFR。采样步长为 `min(max(1, round(native_fps/8)), frame_count//8)`；只保留完整采样间隔，不补帧、不插值、不重编码，记录实际 fps 和丢弃的尾部。短于窗口的片段采用整段窗口。NPZ/API 应提供相同采样策略的 uint8 RGB 和严格等间隔真实时间戳，不能将 4 fps 帧标成 8 fps。以下 2–20 秒与固定 8 fps 条件仅适用于旧默认 uniform_cfr 模式。

## CSV 字段

| 字段 | 必需 | 定义 |
| --- | --- | --- |
| clip_id | 是 | 全清单唯一 ID，三个分支使用同一值 |
| path | 是 | 绝对路径或相对 CSV 所在目录的路径；可为 MP4/MOV 或 NPZ |
| label | 是 | `0` 为真实摄像视频，`1` 为完全 AI 生成视频 |
| split | 是 | 精确使用 `train`、`val`、`test` |
| group_id | 是 | 最终防泄漏组 ID；配对提示词、同源视频、近重复连通组归入同组 |
| source | 是 | 真实视频来源或生成视频数据池名称 |
| generator | 是 | 真实视频留空或填 `real`；生成视频填实际生成器名称 |
| pair_id | 建议 | 共享提示词或配对关系，缺失留空 |
| duplicate_id | 建议 | 近重复组，缺失留空 |
| source_video_id | 建议 | 原始视频 ID，缺失留空；不是宽泛的数据集名称 |
| protocol | 建议 | 清单内一致，如 `in_domain` 或 `cross_generator` |
| fold_id | 可选 | train 行统一使用 `0..K-1`；同一防泄漏连通组必须在同一折 |
| content_sha256 | 可选 | 交付文件的 64 位小写 SHA256；非空时解码前强制核对 |

`source` 是来源类别，可以出现在多个普通分区；`source_video_id` 是同一原始视频的身份，不能跨分区。`group_id`、`pair_id`、`duplicate_id`、`source_video_id` 使用全局唯一且稳定的标识，避免两个数据集均从数字 1 开始而错误合组。缺失可选值留空，不填统一的 `unknown` 或 `NA`，否则会被当作同一组。

程序会检查重复 ID、同一路径、组跨分区和配置中声明的留出来源/生成器。特征提取后会检查相同文件字节的 SHA256。重新编码后的同源视频和近重复无法靠字节哈希识别，仍需数据处理同学建立相关组。

cross-generator 实验将指定的生成器及真实来源全部放入 test，并填写配置的 `heldout_generators` 与 `heldout_real_sources`。没有这些设置时仅能验证普通分组隔离，不能自动证明已正确留出生成器。

训练文件齐备后，可以先运行 `folds`，只在已有 train 内确定 OOF 折，不改动数据负责人的 train/val/test 分区。导出的完整清单新增 train 的 fold_id 与内容哈希，路径变为当前机器可解析的绝对路径。val/test 文件暂缺时仍可准备折，但该步骤不会验证它们的解码或字节重复情况。原始公共清单保留不变，A/B/C 应使用同一份冻结的导出清单。

## 视频输入

基线要求 2–20 秒、恒定帧率，原始帧率至少 8 fps、最多 240 fps。MP4/MOV 是否可读取取决于当前 OpenCV 支持的编码；有问题的视频应由数据负责人统一转换。当前解码器按声明的恒定帧率顺序读取，映射到 8 fps 均匀网格，最近帧误差不超过约半个原始帧间隔。视频末尾不足一个采样间隔的零头不进入描述符，避免时间网格超出真实时长；原始解码时长和不足 1/8 秒的尾部差值记录在 metadata。时长边界按原始解码时长检查，不将不足 2 秒的视频补成 2 秒。可变帧率视频必须先按真实时间戳转为 CFR，不能直接依赖此解码器做精确时间分析。

默认缩放为 96×96；这是当前轻量基线的选择，正式实验前可统一修改。请对真实和生成视频采用相同转换流程，避免编码、帧率、分辨率或填充差异成为标签捷径。若要保留纵横比，可在上游统一处理，但须冻结规则并对所有分支保持一致。

## 公共预采样帧

如果数据处理或 Branch B 已经采样，可交付以下 NPZ，或者直接调用 Python 接口：

```python
import numpy as np
np.savez_compressed(
    "clip.npz",
    frames_rgb=frames_rgb,                  # uint8，形状 [T,H,W,3]，RGB 0..255
    timestamps_seconds=timestamps_seconds # float，形状 [T]，间距恰为 1/8 秒
)
```

提供未按视觉编码器均值和标准差归一化的 RGB 帧，不要将 BGR 或标准化后的 float 张量交给 Branch C。至少 16 帧覆盖 2 秒，最多 160 帧覆盖 20 秒，时间戳严格递增且非负。更改配置的 fps 后，帧间隔也必须同步改变。

输出元数据中的 `input_width/input_height` 保留所交付帧文件的尺寸，`feature_image_size` 表示描述符计算所用的缩放尺寸。实际视频另有 `native_width/native_height/native_fps`；NPZ 无法自动恢复上游原视频条件，未知值保持空白。各阶段分数 CSV 的 `content_sha256` 对应交付文件本身，用于核对输入版本，不用作模型特征。

Branch A 的“全视频均匀取 8–16 帧”不能直接代替这里的 8 fps 时间序列，尤其是较长视频；光流和时间频率需要较密、规则的相邻帧。建议共享 B/C 的稠密序列，A 再从中选择自己的稀疏帧。

## 最小交付

最初只要有真实 train 和真实 val 就能拟合 Branch C 和选阈值；完整 OOF、二分类评价与融合训练再加入生成视频。真实训练描述符至少 4 个才能运行，正式实验应使用具有来源和内容多样性的足量真实视频；4 个只是代码下限。阈值和置信区间在很小的验证/测试集上不稳定。

OOF 至少需要 5 个真实 train，这是留下真实视频后仍能保有 4 个拟合样本的必要条件；具体每折是否可行还取决于组大小和分配。`folds` 与 `oof` 会在特征提取前检查每折真实拟合数量，不能用生成、val 或 test 视频凑足。

如果公共清单已经列出了尚未下载的生成 train，`fit` 会保留这些行的元数据用于防泄漏检查，但不会读取其文件。所有 val 行仍需有可读取的文件；也可以先交付只有真实 train/val 的有效清单。生成 OOF 时全部 train 文件必须就绪。
