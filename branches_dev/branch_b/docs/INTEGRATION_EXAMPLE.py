# -*- coding: utf-8 -*-
"""Read-only, copied-folder example for the requested full32k internal stacking.

Run: python -B docs/INTEGRATION_EXAMPLE.py
Paths resolve from this file, not the original repository or current directory.
Full-fit scores are in-sample; this example makes no generalization estimate.
"""
from pathlib import Path
import os
import sys

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
# Support long Windows workspace paths as well as short copied-package paths.
if os.name == "nt" and not str(PACKAGE_ROOT).startswith("\\\\?\\"):
    PACKAGE_ROOT = Path("\\\\?\\" + str(PACKAGE_ROOT))
from inference import load_detector  # noqa: E402


def main():
    preds = pd.read_parquet(PACKAGE_ROOT / "predictions/predictions_full_dataset.parquet")
    assert preds["clip_id"].is_unique
    # Team interface: preserve the unsquashed score; keep labels out of inputs.
    X_stacker = preds.set_index("clip_id")[["raw_margin"]]
    print("In-sample Branch B stacking inputs:", X_stacker.shape)

    # Use actual included reference features, not fabricated dummy measurements.
    detector = load_detector()
    features = pd.read_parquet(PACKAGE_ROOT / "predictions/regression_sample_features.parquet")
    scores = detector.infer_features(features)
    point = detector.operating_thresholds[detector.deployed_threshold_key]
    # Decision comparisons use the exact loaded float, never display rounding.
    decisions = (scores["calibrated_prob"] >= point["threshold"]).astype("int64")
    assert (decisions == scores["prediction"]).all()
    print("Reference feature rows scored:", len(scores))
    print("Exact probability threshold:", repr(point["threshold"]))
    print("Corresponding margin threshold:", repr(point["margin_threshold"]))
    # Live video: detector.infer_video("path/to/clip.mp4"); inspect status first.


if __name__ == "__main__":
    main()
