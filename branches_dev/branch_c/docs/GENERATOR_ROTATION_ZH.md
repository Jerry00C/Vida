# Branch C：6 已见 / 2 未见生成器轮换交接

2026 年 10 月 15 日完成。完整结果为 `artifacts/vida_guard32k/rotation_6seen_2unseen_v1/report/index.html`，新清单为 `data/VIDA-Guard32K/rotation_6seen_2unseen_v1`。共28轮，覆盖8选2全部配对，每家留出7次；原32K交付与run_02保留。

分组：来源/提示词/真实-生成配对组，60/10/10/20用于train/model_selection/calibration/test，seed42。每轮真实train9600，生成train6000/7200/8400；验证两角色各1200，真假各600；未见test800、已见test2400，真假平衡、真实两来源等量。每家固定200测试视频，未见生成器开发行隔离，所有配额外行在excluded.csv中解释。

未见28轮 AUROC 均值 0.874，范围 0.742–0.976；macro F1 均值 0.652。轮次依赖，不是28个独立数据集；每轮区间为固定模型的组bootstrap。

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

案例另外发现来源标real的全黑片段与动画/图解片段，说明来源标签和输入信息量存在边界；本次保留固定测试样本、不事后删样本或改标签。详见report/input_case_findings.json。
