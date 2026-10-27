"""Generate TOY videos and run the full pipeline; never a detection benchmark."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import time
import cv2
import numpy as np

from branch_c.config import load_config, write_json
from branch_c.pipeline import evaluate, fit, make_oof, predict_frames, predict_video


def toy_frames(rng, generated, unseen=False):
    """The label denotes the toy construction, not real camera/AI provenance."""
    texture = rng.integers(25, 210, (64, 64, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    velocity = int(rng.integers(1, 3))
    frames = []
    for t in range(24):
        frame = np.roll(texture, velocity * t, axis=1).astype(float)
        frame += rng.normal(0, 1.2, frame.shape)
        if generated:
            frame += 35 if t % 2 else -35
            if unseen:
                frame = np.roll(frame, int(rng.integers(-20, 21)), axis=0)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return np.stack(frames)


def create_toy_manifest(root, seed=42):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split, groups in [("train", 20), ("val", 8), ("test", 8)]:
        for group in range(groups):
            for label in [0, 1]:
                clip_id = f"toy_{split}_{group:03d}_{label}"
                frames = toy_frames(rng, bool(label), unseen=split == "test")
                relative = f"{clip_id}.npz"
                np.savez_compressed(root / relative, frames_rgb=frames,
                                    timestamps_seconds=np.arange(len(frames)) / 8.)
                rows.append({"clip_id": clip_id, "path": relative, "label": label, "split": split,
                    "group_id": f"toy_{split}_{group:03d}",
                    "source": "toy_heldout_camera" if split == "test" and label == 0 else "toy_camera",
                    "generator": ("toy_unseen" if split == "test" else "toy_flicker") if label else "",
                    "protocol": "synthetic_smoke"})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/smoke")
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        raise SystemExit("Choose a new --output-dir; existing smoke results are preserved")
    started = time.perf_counter()
    manifest = create_toy_manifest(root / "toy_data")
    config = replace(load_config(), heldout_generators=("toy_unseen",),
                     heldout_real_sources=("toy_heldout_camera",))
    fit_report = fit(manifest, config, root / "fit", root / "cache", "synthetic_smoke")
    make_oof(manifest, config, root / "oof", root / "cache", 5, "synthetic_smoke")
    report = evaluate(manifest, root / "fit/model.joblib", root / "evaluation", root / "cache", 100)
    model_path = root / "fit/model.joblib"
    source = root / "toy_data/toy_test_000_1.npz"
    prediction = predict_video(model_path, source, root / "prediction_npz.json")
    with np.load(source, allow_pickle=False) as data:
        frame_prediction = predict_frames(model_path, data["frames_rgb"], data["timestamps_seconds"])
        assert abs(prediction["anomaly_score"] - frame_prediction["anomaly_score"]) < 1e-12
        video_path = root / "toy_video.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable; cannot verify actual-video path")
        for frame in data["frames_rgb"]:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    video_prediction = predict_video(model_path, video_path, root / "prediction_mp4.json")
    assert prediction["is_probability"] is False
    assert len(video_prediction["timeline"]) == 2
    summary = {"run_kind": "synthetic_smoke", "is_benchmark_result": False,
        "status": "passed", "toy_clips": 72, "real_only_fit_clips": len(fit_report["clip_model"]["fit_clip_ids"]),
        "oof_folds": 5, "npz_frames_api_agree": True, "mp4_decode_predict": "passed",
        "test_threshold_unchanged": report["threshold"] == fit_report["threshold"],
        "total_seconds": time.perf_counter() - started,
        "note": "TOY data only. No claim about AI-video detection accuracy or unseen-generator generalisation."}
    write_json(root / "smoke_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
