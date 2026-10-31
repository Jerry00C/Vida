"""Map the verified VIDA-Guard32K delivery to Branch C without changing splits/folds.

Only metadata and file sizes are read here. Media hashes are independently checked
by feature preparation and OOF planning. No test video is decoded or scored.
"""
import argparse
from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from branch_c.cache import file_sha256
from branch_c.config import Config, FeatureConfig, write_json
from branch_c.manifest import load_manifest


def import_delivery(delivery_root, media_root, output_dir):
    delivery_root, media_root, output_dir = map(Path, (delivery_root, media_root, output_dir))
    if output_dir.exists():
        raise ValueError("Import output already exists; preserve frozen manifests")
    original = delivery_root / "manifests/manifest.csv"
    with original.open(encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f))
    expected = {"train": 16800, "validation": 3600, "in_domain_test": 3600, "cross_test": 8000}
    if dict(Counter(r["split"] for r in source_rows)) != expected:
        raise ValueError("Unexpected delivery population; review protocol before import")
    rows = []
    for r in source_rows:
        video, metadata = media_root / r["video_path"], media_root / r["metadata_path"]
        for path in (video, metadata):
            if not path.resolve().is_relative_to(media_root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or unsafe media path: {path}")
        m = json.loads(metadata.read_text(encoding="utf-8"))
        media = m["media"]
        if (m["clip_id"] != r["clip_id"] or str(m["label"]) != r["label"]
                or m["group_id"] != r["group_id"] or m["split"] != r["split"]
                or str(m.get("fold", "") if m.get("fold") is not None else "") != r["fold"]
                or (m.get("validation_role") or "") != r["validation_role"]
                or media["sha256"] != r["sha256"] or media["qc_status"] != "passed"
                or video.stat().st_size != int(r["file_bytes"])):
            raise ValueError(f"Manifest/metadata mismatch: {r['clip_id']}")
        fps = float(r["native_fps"])
        times = np.asarray(media["timestamp_seconds"], dtype=float)
        if (media["variable_frame_rate"] or len(times) != int(r["frame_count"])
                or not np.isfinite(times).all() or len(times) < 8 or fps < 4
                or not np.allclose(np.diff(times), 1 / fps, atol=1e-4, rtol=1e-3)):
            raise ValueError(f"CFR timestamp audit failed: {r['clip_id']}")
        rows.append(r | {"path": str(video.resolve()),
            "split": {"train": "train", "validation": "val"}.get(r["split"], "test"),
            "source": r["family"], "generator": r["family"] if r["label"] == "1" else "",
            "fold_id": r["fold"], "content_sha256": r["sha256"],
            "protocol": "vida_guard32k_v1", "delivery_split": r["split"],
            "evaluation_subset": r["split"] if r["split"].endswith("test") else ""})
    output_dir.mkdir(parents=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    config = replace(Config(), features=FeatureConfig(min_duration_seconds=.5, sampling_mode="native_stride"),
        heldout_generators=("CogVideo", "Mora", "MuseV", "SVD"), heldout_real_sources=("HD-VG-130M",))
    load_manifest(manifest, config)
    write_json(output_dir / "config.json", config.to_dict())
    report = {"status": "complete", "original_manifest_sha256": file_sha256(original),
        "branch_manifest_sha256": file_sha256(manifest), "split_counts": expected,
        "validation_roles": dict(Counter(r["validation_role"] for r in rows if r["split"] == "val")),
        "native_cfr_metadata_audited": len(rows), "test_videos_decoded": 0,
        "protocol": "vida_guard32k_v1", "fold_policy": "Unchanged supplied fold 0..4",
        "mapping": {"validation": "val", "in_domain_test": "test + evaluation_subset", "cross_test": "test + evaluation_subset"},
        "preprocessing": "Integer native stride near 8fps, reduce stride to retain >=8 distinct frames; true CFR timestamps; short clip whole-window; 96x96; common FFT support 0..2Hz; pixel/luma/edge rates per 1/8s, flow image-width/s",
        "limitations": ["CFR only; imported native timestamps checked against fps; no copied or interpolated frames",
            "Changing fps/duration/resolution distributions can still create source shortcuts",
            "Cross-test changes both real source and generator family; it is not an isolated generator effect",
            "Group labels and duplicate audit are inherited from the data owner; unknown semantic duplicates may remain"]}
    write_json(output_dir / "import_report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delivery-root", required=True)
    p.add_argument("--media-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    print(json.dumps(import_delivery(args.delivery_root, args.media_root, args.output_dir), indent=2))
