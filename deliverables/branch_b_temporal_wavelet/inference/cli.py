# -*- coding: utf-8 -*-
"""Command-Line Interface for Branch B 'temporal_wavelet' Detector.

Usage examples:
  1. Infer single video file:
     python -m deliverables.branch_b_temporal_wavelet.inference.cli --video path/to/clip.mp4

  2. Infer precomputed tabular features table (CSV or Parquet):
     python -m deliverables.branch_b_temporal_wavelet.inference.cli \
         --input-table path/to/features.parquet \
         --output path/to/predictions.parquet

  3. Custom threshold key:
     python -m deliverables.branch_b_temporal_wavelet.inference.cli \
         --video path/to/clip.mp4 \
         --threshold-key macro_f1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

# Support both ``python path/to/inference/cli.py`` and package/module execution.
# Resolve from this file, never from the caller's working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.detector import BranchBDetector
else:
    from .detector import BranchBDetector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Branch B 'temporal_wavelet' inference on video files or tabular feature sets."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to single video file for end-to-end decoding and inference.",
    )
    parser.add_argument(
        "--input-table",
        type=str,
        default=None,
        help="Path to CSV or Parquet file containing pre-extracted 113 features or 'video_path' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for predictions table (CSV, Parquet, or JSON).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model bundle joblib file (defaults to deliverable model).",
    )
    parser.add_argument(
        "--threshold-key",
        type=str,
        default="target_1pct_real_fpr",
        choices=["target_1pct_real_fpr", "pooled_1pct_real_fpr", "macro_f1"],
        help="Operating decision threshold to evaluate.",
    )
    parser.add_argument(
        "--no-verify-sha256",
        action="store_true",
        help="UNSAFE legacy opt-out: skip binary checksum only; trusted executable pickle and adjacent manifest still required.",
    )
    args = parser.parse_args(argv)

    if args.video is None and args.input_table is None:
        parser.print_help()
        print("\nError: Must specify either --video or --input-table.", file=sys.stderr)
        return 1

    # Load detector instance with fail-closed integrity checks
    try:
        detector = BranchBDetector(
            model_path=args.model_path,
            verify_sha256=not args.no_verify_sha256,
        )
    except Exception as exc:
        print(f"Failed to initialize BranchBDetector: {exc}", file=sys.stderr)
        return 2

    # Mode 1: Single video inference
    if args.video is not None:
        video_p = Path(args.video)
        result = detector.infer_video(video_p, threshold_key=args.threshold_key)

        formatted_json = json.dumps(result, indent=2)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(formatted_json + "\n")
            print(f"Saved inference result to {out_p}")
        else:
            print(formatted_json)

        return 0 if result["status"] == "SUCCESS" else 3

    # Mode 2: Batch tabular table inference
    if args.input_table is not None:
        table_p = Path(args.input_table)
        if not table_p.is_file():
            print(f"Input table file not found: {table_p}", file=sys.stderr)
            return 4

        print(f"Loading input table from {table_p}...")
        if table_p.suffix.lower() == ".parquet":
            df_in = pd.read_parquet(table_p)
        elif table_p.suffix.lower() in (".csv", ".txt"):
            # pandas normally mangles duplicate headers (feature -> feature.1),
            # hiding an ambiguous input schema from infer_features validation.
            with table_p.open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            if not header or len(header) != len(set(header)):
                print("Empty/duplicate CSV column names are not allowed.", file=sys.stderr)
                return 4
            df_in = pd.read_csv(table_p)
        else:
            print(f"Unsupported table format: {table_p.suffix}", file=sys.stderr)
            return 4

        # Check if table already contains the 113 features or video paths
        has_features = all(c in df_in.columns for c in detector.ordered_feature_names)

        if has_features:
            print(f"Evaluating {len(df_in)} precomputed feature rows...")
            scores_df = detector.infer_features(df_in, threshold_key=args.threshold_key)
            out_df = pd.concat([df_in.reset_index(drop=True), scores_df], axis=1)
        elif "video_path" in df_in.columns:
            print(f"Running video inference across {len(df_in)} video rows...")
            records: list[dict[str, Any]] = []
            for idx, row in df_in.iterrows():
                v_path = row["video_path"]
                res = detector.infer_video(v_path, threshold_key=args.threshold_key)
                records.append({
                    "raw_margin": res["raw_margin"],
                    "calibrated_prob": res["calibrated_prob"],
                    "prediction": res["prediction"],
                    "decision": res["decision"],
                    "status": res["status"],
                })
            scores_df = pd.DataFrame(records)
            out_df = pd.concat([df_in.reset_index(drop=True), scores_df], axis=1)
        else:
            print(
                "Input table must contain either the 113 feature columns or a 'video_path' column.",
                file=sys.stderr,
            )
            return 5

        # Output predictions
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".parquet":
                out_df.to_parquet(out_p, index=False)
            elif out_p.suffix.lower() in (".csv", ".txt"):
                out_df.to_csv(out_p, index=False)
            else:
                out_df.to_json(out_p, orient="records", indent=2)
            print(f"Exported {len(out_df)} predictions to {out_p}")
        else:
            display_columns = [c for c in ("clip_id", "raw_margin", "calibrated_prob", "prediction", "decision") if c in out_df]
            print(out_df[display_columns].head(20))

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
