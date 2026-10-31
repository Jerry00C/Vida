# VIDA final three-branch fusion

This directory contains the completed score-fusion implementation, its frozen
MLP model, training records, per-video final-test predictions and evaluation plots.
Branch checkpoints remain fixed while the fusion layer learns from independent
videos. Run all commands from the repository root.

## Use the trained model

Use Python 3.12 and install the recorded fusion environment:

```bash
python -m venv .venv-stacking
# Windows: .venv-stacking/Scripts/python.exe
# Linux/macOS: .venv-stacking/bin/python
python -m pip install -r stacking/requirements.lock.txt
```

Use the environment's Python for subsequent commands. Prepare a CSV with
`clip_id,A,B,C`: A is the Partial CLIP Round 4 **mean video logit**, B is the
frozen LightGBM **raw margin**, and C is the frozen real-reference **anomaly score**.
Each row represents the same video across all three branches. Larger scores
mean more evidence of generation. Hard class labels and summary metrics are
not valid inputs.

```bash
python -m stacking.predict --model-dir stacking/released_model \
  --input new_abc_scores.csv --output outputs/new_predictions.csv
```

The model folder contains the scaler, 3–8–1 MLP, Platt calibration, fixed
threshold and integrity lock. Inference needs NumPy/sklearn, not PyTorch,
original training videos or this machine's paths. Producing A/B/C scores from
new videos still requires the exact branch models and preprocessing.
Only load project model files from a trusted source because joblib uses pickle.

## Data roles

The shared branch-development collection has 32,000 videos. The independent
fusion/evaluation suite has 16,400 additional originals:

| Role | Total | Real | Generated |
|---|---:|---:|---:|
| Fusion fitting | 5,120 | 2,560 | 2,560, 320 per known family |
| Probability calibration | 640 | 320 | 320, 40 per known family |
| Threshold selection | 640 | 320 | 320, 40 per known family |
| In-domain final test | 8,000 | 4,000 | 4,000, 500 per known family |
| ModelScope final test | 2,000 | 1,000 | 1,000 ModelScope |

Known generated families are Pika, VideoCrafter2, T2V-Zero, OpenSora, SVD,
MuseV, Mora and CogVideo. Real controls use Vript and HD-VG-130M in a 3:1 ratio.
ModelScope generated videos enter only the final test. Source/prompt groups
separate the roles. Original videos and local environments are distributed
separately and are not included in Git.

## Fusion training

`expanded.py` validates the dataset roles, branch identity and score completeness.
Eight generator holdouts inside the 5,120 fusion-training videos compare six
regularized logistic-regression candidates with a small MLP. Each fold fits its
own standardization and removes source/prompt groups shared with validation.
Only mean validation log loss selects the candidate.

The selected MLP has 3 inputs, 8 ReLU hidden units and 1 output, with 41 trainable
parameters. Training uses BCEWithLogitsLoss, AdamW, learning rate 0.001, weight
decay 0.001, batch 64 and dropout 0.1. The search allows at most 200 epochs,
patience 20 and min_delta 0.0001. The median fold-selected duration gives a final
fit of **102 epochs**. A separate 640 videos fit Platt calibration. Another
640 select the fixed Macro F1 threshold **0.4495125162248878**.

Final testing happens after the entire decision pipeline is frozen. Fusion
training does not rerun or jointly backpropagate through the branch models.

## Final evaluation

| Native final-test view | AUROC | Macro F1 | TN / FP / FN / TP |
|---|---:|---:|---|
| In domain, 8,000 videos | 0.9999985 | 0.9992500 | 3998 / 2 / 4 / 3996 |
| ModelScope, 2,000 videos | 1.0000000 | 0.9995000 | 999 / 1 / 0 / 1000 |

The paired format-controlled ModelScope view uses the same originals, first
2 seconds, 8 fps, 256 × 256 and H.264 CRF18. AUROC is 0.970026 and Macro F1 is
0.390677 at the unchanged threshold. It is a sensitivity diagnostic and does
not add independent videos. Native scores alone do not establish robustness
to all generators or recording formats. The full research report also retains
the historical Sora/Kling stress results and the scope of the training audit.

## Files

| Path | Purpose |
|---|---|
| `released_model/` | Frozen MLP, scaler, probability calibration, threshold and identity lock |
| `predict.py` | Apply the released model to aligned A/B/C scores |
| `expanded.py` / `neural_expanded.py` | Eight-family holdout selection and final MLP fitting |
| `pipeline.py` / `neural.py` | Shared score checks, metrics and portable MLP implementation |
| `export_a.py` / `export_b.py` / `export_c.py` | Fixed branch inference adapters |
| `results/final_16400/` | Training history, selected folds, test predictions, metrics and plots |
| `results/format_control_ModelScope/` | Paired format-control plan, predictions and metrics |
| `results/protocol/` | Dataset separation and verification records |
| `FROZEN_PROTOCOL.md` | Experimental boundaries and model provenance |
| `TRAINING.md` | Reproduction steps and requirements |
| `tests/` | Isolation, alignment, score validation, reload and portable inference tests |

The original OOF interface remains available as `python -m stacking.legacy_oof_cli`.
`LEGACY_OOF.md` describes that interface. It is distinct from the released model.

## Tests

```bash
python -m pytest stacking/tests/test_pipeline.py stacking/tests/test_expanded.py \
  stacking/tests/test_neural.py stacking/tests/test_export_resume.py \
  stacking/tests/test_predict_portable.py -q
```

The retained legacy Branch A tests additionally require its PyTorch/Transformers
environment. `report.py` is the earlier protocol's report utility; the released
expanded-protocol results are the saved files in `results/final_16400/`.
