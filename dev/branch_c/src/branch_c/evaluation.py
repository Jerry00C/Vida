"""Fixed-threshold evaluation on anomaly evidence, with group bootstrap intervals."""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import require_integer, require_number


def _validate_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or not len(labels) or scores.shape != labels.shape:
        raise ValueError("Nonempty, aligned one-dimensional labels and scores required")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1 and scores must be finite")
    return labels.astype(int), scores


def choose_real_threshold(scores, labels, false_positive_rate):
    labels, scores = _validate_scores(labels, scores)
    require_number("false_positive_rate", false_positive_rate)
    if not 0 < false_positive_rate < 1:
        raise ValueError("false_positive_rate must be in (0,1)")
    real = scores[labels == 0]
    if not len(real) or not np.isfinite(real).all():
        raise ValueError("Real validation scores are required for threshold selection")
    return float(np.quantile(real, 1 - false_positive_rate, method="higher"))


def metrics(labels, scores, threshold):
    labels, scores = _validate_scores(labels, scores)
    require_number("threshold", threshold)
    predictions = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    result = {"n": len(labels), "n_real": int((labels == 0).sum()), "n_generated": int((labels == 1).sum()),
              "auroc": float(roc_auc_score(labels, scores)) if both else None,
              "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)) if both else None,
              "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if both else None,
              "precision_generated": float(precision_score(labels, predictions, zero_division=0)) if tp + fp else None,
              "recall_generated": float(recall_score(labels, predictions, zero_division=0)) if tp + fn else None,
              "specificity": float(tn / (tn + fp)) if tn + fp else None,
              "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
              "confusion_matrix_real_generated": [[int(tn), int(fp)], [int(fn), int(tp)]],
              "binary_comparison_available": both}
    if both:
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        # Linear interpolation at the FPR/FNR crossing, including tied scores.
        difference = fpr - fnr
        right = int(np.flatnonzero(difference >= 0)[0])
        if right == 0:
            eer = fpr[0]
        else:
            left = right - 1
            weight = -difference[left] / (difference[right] - difference[left])
            eer = fpr[left] + weight * (fpr[right] - fpr[left])
        result["equal_error_rate"] = float(eer)
    else:
        result["equal_error_rate"] = None
    return result


def bootstrap_intervals(labels, scores, groups, threshold, repeats=1000, seed=42):
    require_integer("bootstrap repeats", repeats)
    require_integer("bootstrap seed", seed, upper=2**32 - 1)
    require_number("threshold", threshold)
    labels, scores = _validate_scores(labels, scores)
    groups = np.asarray(groups)
    if groups.shape != labels.shape:
        raise ValueError("Bootstrap requires one group ID per score")
    unique = np.unique(groups)
    if repeats == 0 or len(unique) < 2:
        return {"method": "group_bootstrap", "valid_repeats": 0, "auroc": None, "macro_f1": None}
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(repeats):
        indices = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        if len(np.unique(labels[indices])) < 2:
            continue
        aucs.append(roc_auc_score(labels[indices], scores[indices]))
        f1s.append(f1_score(labels[indices], scores[indices] > threshold, labels=[0, 1], average="macro", zero_division=0))
    return {"method": "group_bootstrap", "requested_repeats": repeats, "valid_repeats": len(aucs),
            "n_groups": len(unique), "auroc": np.quantile(aucs, [.025, .975]).tolist() if aucs else None,
            "macro_f1": np.quantile(f1s, [.025, .975]).tolist() if f1s else None}
