# Full-dataset prediction schema

[`predictions_full_dataset.parquet`](predictions_full_dataset.parquet) and
[`predictions_full_dataset.csv`](predictions_full_dataset.csv) contain the same
32,000 clip scores from the delivered **temporal_wavelet full32k refit**.
The booster, calibrator and thresholds were fitted on this full corpus: these
are **in-sample** predictions, supplied for the requested internal stacking.
OOF generation or a holdout is not a delivery prerequisite. These rows are not
held-out generalization evidence.

## Columns

| Column | Parquet type / domain | Meaning |
|---|---|---|
| `clip_id` | string, unique, nonempty | Canonical clip identifier; join branches by this key, not row position. |
| `label` | int64, {0, 1} | Ground truth: 0 real, 1 generated; 16,000 of each. Not a stacker input. |
| `family` | string, ten families | 2 real sources (Vript, HD-VG-130M) + 8 generators (Pika, VideoCrafter2, T2V-Zero, OpenSora, SVD, MuseV, Mora, CogVideo). Not a stacker input. |
| `raw_margin` | float64, finite; unbounded score space | Unsquashed binary LightGBM score before calibration; larger favors generated. The agreed Branch B stacking interface. |
| `calibrated_prob` | float64, [0, 1] | Monotonic Platt logistic estimate fitted on these same full-data margins, with C=1.0. Not proof of calibration on future media. |
| `prediction` | int64, {0, 1} | 1 iff `calibrated_prob` is at least the exact default probability threshold loaded from the config. |
| `decision` | string, REAL or GENERATED | Human-readable counterpart of `prediction`. |

No saved full-corpus row has null scores. This differs from live video inference,
which can abstain with null scores. The table contains predictions, **not** the
113 feature inputs required by `infer_features` or the CLI's feature-table path.

The measured [audit report](../docs/AUDIT_REPORT.json) records CSV/Parquet exact
float64 parity using pandas' round-trip CSV parser:

```python
import pandas as pd

preds = pd.read_csv("predictions/predictions_full_dataset.csv", float_precision="round_trip")
assert preds["clip_id"].is_unique
X_branch_b = preds.set_index("clip_id")[["raw_margin"]]
```

## Stacking semantics

`raw_margin` is the team's chosen interface because it preserves unsquashed score
differences and does not depend on the standalone Platt mapping. It is not a known
calibrated posterior log-odds or likelihood ratio. Probability inputs are not
universally mathematically prohibited; downstream modeling choices determine
which representation is useful. Margin use does not guarantee an optimal
Bayesian ensemble or convex MLP training. See the
[stacker handoff](../docs/HANDOFF_TO_STACKER.md) for missingness handling.

## Exact operating points and descriptive rates

| Key | Probability threshold | Corresponding margin threshold | Observed real FPR |
|---|---:|---:|---|
| `target_1pct_real_fpr` (default) | 0.025049979809292647 | -4.095739606463252 | Vript 0.01; HD-VG-130M 0.0005; unweighted pooled 0.007625 |
| `pooled_1pct_real_fpr` | 0.015457025337482144 | -4.646689782063401 | 0.01 family-weighted pooled versus 0.01475 unweighted pooled |
| `macro_f1` | 0.9196089333917218 | 2.7252040699627056 | Zero real false positives on this fitting corpus |

These are **training-corpus rates**, not future-population guarantees. In the
secondary pooled calculation, real-source rates receive equal weight, unlike
ordinary pooling over 12,000 Vript and 4,000 HD-VG-130M clips.

Load thresholds from [`model_config.json`](../model/model_config.json); do not
round them before decisions. From the package root:

```python
import json
from pathlib import Path
import pandas as pd

config = json.loads(Path("model/model_config.json").read_text(encoding="utf-8"))
threshold = config["operating_thresholds"][config["deployed_threshold_key"]]["threshold"]
preds = pd.read_parquet("predictions/predictions_full_dataset.parquet")
assert ((preds["calibrated_prob"] >= threshold).astype("int64") == preds["prediction"]).all()
```

The exact feature schema, sample counts and parameters are in the config;
[provenance](../docs/PROVENANCE.md) distinguishes this artifact from the historical
Option-1 release. No prediction values were changed to revise this documentation.
