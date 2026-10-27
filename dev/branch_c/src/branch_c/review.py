"""Export fixed-threshold errors for human review; never infer a cause from a score."""
import csv
from pathlib import Path

import numpy as np

from .evaluation import metrics
from .manifest import resolve_run_kind


REVIEW_FIELDS = (
    "clip_id", "path", "group_id", "source", "generator", "label", "predicted_label", "error_type",
    "branch_c_score", "threshold", "threshold_margin", "is_probability", "model_id", "run_kind", "protocol",
    "duration_seconds", "native_fps", "native_width", "native_height", "input_width", "input_height",
    "sample_fps", "sampled_frames", "feature_image_size", "review_status", "failure_type", "notes",
)


def write_error_review(path, rows, scores, threshold, features, *, model_id, run_kind="auto"):
    path = Path(path)
    if path.exists():
        raise ValueError("Error review already exists; preserve manual annotations and choose a new directory")
    if len(features) != len(rows):
        raise ValueError("One feature metadata record is required per review clip")
    labels = np.array([r.label for r in rows])
    scores = np.asarray(scores, dtype=float)
    counts = metrics(labels, scores, threshold)  # Validate inputs with the same evaluation rules.
    run_kind = resolve_run_kind(rows, run_kind)
    predictions = (scores > threshold).astype(int)
    errors = np.flatnonzero(labels != predictions)
    # FP then FN; strongest fixed-threshold decisions first within each class.
    # Margins have meaning only within this saved model, not across estimators.
    errors = sorted(errors, key=lambda i: (int(labels[i]), -abs(scores[i] - threshold), rows[i].clip_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for i in errors:
            row, metadata = rows[i], features[i].metadata
            record = {key: metadata.get(key, "") for key in (
                "duration_seconds", "native_fps", "native_width", "native_height", "input_width", "input_height",
                "sample_fps", "sampled_frames", "feature_image_size")}
            record.update(clip_id=row.clip_id, path=str(row.path), group_id=row.group_id,
                          source=row.source, generator=row.generator, label=row.label,
                          predicted_label=int(predictions[i]), error_type="FP" if row.label == 0 else "FN",
                          branch_c_score=float(scores[i]), threshold=threshold,
                          threshold_margin=float(scores[i] - threshold), is_probability="false",
                          model_id=model_id, run_kind=run_kind, protocol=row.metadata.get("protocol", ""),
                          review_status="pending", failure_type="", notes="")
            writer.writerow(record)
    confusion = counts["confusion_matrix_real_generated"]
    return {"file": path.name, "n_cases": len(errors), "false_positives": confusion[0][1],
            "false_negatives": confusion[1][0], "threshold_rule": "score > fixed real-validation threshold",
            "review_status": "pending_human_review" if errors else "no_errors_at_this_threshold",
            "note": "FP means real clips above the anomaly threshold; FN means generated clips at or below it. Causes require human inspection; these are not calibrated AI probabilities."}
