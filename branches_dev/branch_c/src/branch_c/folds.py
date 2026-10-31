"""Prepare and audit the common training folds without decoding videos or fitting models."""
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .cache import file_sha256
from .config import fingerprint, require_integer, write_json
from .manifest import GROUP_KEYS, connected_groups, load_manifest, manifest_protocol, resolve_run_kind
from .model import runtime_versions


def _check_group_folds(groups, folds):
    seen = {}
    for group, fold in zip(groups, folds):
        if group in seen and seen[group] != fold:
            raise ValueError("Connected leakage group crosses supplied OOF folds")
        seen[group] = fold


def _fold_summary(rows, groups, folds, n_splits):
    real = np.array([r.label == 0 for r in rows])
    summaries = []
    for fold in range(n_splits):
        held = folds == fold
        n_real_fit = int((real & ~held).sum())
        if n_real_fit < 4:
            raise ValueError(f"OOF fold {fold} leaves only {n_real_fit} real training clips; at least 4 are required")
        summaries.append({"fold_id": fold, "n_heldout": int(held.sum()),
                          "n_heldout_real": int((real & held).sum()),
                          "n_heldout_generated": int((~real & held).sum()),
                          "n_real_fit": n_real_fit,
                          "n_heldout_groups": len(np.unique(groups[held])),
                          "n_training_groups": len(np.unique(groups[~held]))})
    return summaries


def preflight_oof(rows, n_splits):
    """Reject metadata failures before reading training media. Auto splits need hashes later."""
    require_integer("n_splits", n_splits, lower=2)
    if not rows or any(r.split != "train" for r in rows):
        raise ValueError("OOF requires nonempty train rows only")
    if sum(r.label == 0 for r in rows) < 5:
        raise ValueError("OOF requires at least 5 real train clips to leave >=4 when a real clip is held out")
    groups = connected_groups(rows)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Need at least n_splits independent training groups and >=2 folds")
    supplied = [r.metadata.get("fold_id", "") for r in rows]
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError("fold_id must be present on every training row or none")
    if any(not str(value).isdecimal() for value in supplied):
        raise ValueError("fold_id must contain integers from 0..n_splits-1")
    folds = np.array([int(value) for value in supplied])
    if set(folds) != set(range(n_splits)):
        raise ValueError("fold_id must cover 0..n_splits-1")
    _check_group_folds(groups, folds)
    _fold_summary(rows, groups, folds, n_splits)
    return folds


def build_fold_plan(rows, hashes, n_splits, seed):
    require_integer("fold seed", seed, upper=2**32 - 1)
    folds = preflight_oof(rows, n_splits)
    groups = connected_groups(rows, hashes)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Content duplicates leave fewer than n_splits independent training groups")
    source = "manifest_fold_id" if folds is not None else "seeded_stratified_group"
    if folds is not None:
        _check_group_folds(groups, folds)
    else:
        folds = np.full(len(rows), -1, dtype=int)
        labels = np.array([r.label for r in rows])
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, held) in enumerate(splitter.split(np.zeros((len(rows), 1)), labels, groups)):
            folds[held] = fold
    if set(folds) != set(range(n_splits)):
        raise ValueError("OOF assignment must produce every nonempty fold 0..n_splits-1")
    summaries = _fold_summary(rows, groups, folds, n_splits)
    # Stable component names and identity when the same supplied folds are reused
    # from a copied/reordered manifest. Assignment itself preserves the old seed rule.
    components = {}
    for row, group in zip(rows, groups):
        components[group] = min(components.get(group, row.clip_id), row.clip_id)
    identities = [{"clip_id": r.clip_id, "label": r.label, "fold_id": int(fold),
                   "component_id": components[group], "content_sha256": digest,
                   "group_tags": {k: r.metadata.get(k, "") for k in GROUP_KEYS}}
                  for r, group, fold, digest in zip(rows, groups, folds, hashes)]
    plan = {"plan_id": fingerprint({"n_splits": n_splits,
                                    "assignments": sorted(identities, key=lambda r: r["clip_id"])}),
            "n_splits": n_splits, "n_clips": len(rows), "n_groups": len(components),
            "assignment_source": source, "split_seed": seed if source == "seeded_stratified_group" else None,
            "fold_summary": summaries,
            "component_ids": [components[group] for group in groups]}
    return folds, groups, plan


def hash_train_rows(rows):
    if any(r.split != "train" for r in rows):
        raise ValueError("Only train files may be hashed for OOF planning")
    hashes = []
    for i, row in enumerate(rows):
        digest = file_sha256(row.path)
        expected = row.metadata.get("content_sha256", "")
        if expected and digest != expected:
            raise ValueError(f"Input content differs from frozen manifest: {row.clip_id}")
        hashes.append(digest)
        if (i + 1) % 25 == 0:
            print(f"Hashed {i + 1}/{len(rows)} train files; no decoding", file=sys.stderr, flush=True)
    return hashes


def prepare_folds(manifest, config, output_dir, n_splits=5, run_kind="auto"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Fold output already exists; choose a new directory")
    manifest_hash = file_sha256(manifest)
    manifest_rows = load_manifest(manifest, config)
    run_kind = resolve_run_kind(manifest_rows, run_kind)
    rows = [r for r in manifest_rows if r.split == "train"]
    preflight_oof(rows, n_splits)
    hashes = hash_train_rows(rows)
    folds, _, plan = build_fold_plan(rows, hashes, n_splits, config.model.seed)
    if file_sha256(manifest) != manifest_hash:
        raise ValueError("Manifest changed during fold preparation")
    output_dir.mkdir(parents=True)
    by_id = {r.clip_id: (int(fold), digest) for r, fold, digest in zip(rows, folds, hashes)}
    with (output_dir / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["clip_id", "label", "split", "group_id", "fold_id", "component_id", "content_sha256", "protocol", "run_kind"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, component in zip(rows, plan["component_ids"]):
            fold, digest = by_id[row.clip_id]
            writer.writerow({"clip_id": row.clip_id, "label": row.label, "split": "train", "group_id": row.group_id,
                             "fold_id": fold, "component_id": component, "content_sha256": digest,
                             "protocol": row.metadata.get("protocol", ""), "run_kind": run_kind})
    # Rebase paths so this exported manifest is usable from its new directory.
    fields = list(dict.fromkeys([*manifest_rows[0].metadata, "fold_id", "content_sha256"]))
    with (output_dir / "manifest_with_folds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest_rows:
            record = row.metadata | {"path": str(row.path)}
            if row.split == "train":
                record |= {"fold_id": by_id[row.clip_id][0], "content_sha256": by_id[row.clip_id][1]}
            writer.writerow(record)
    report = {**plan, "run_kind": run_kind, "protocol": manifest_protocol(manifest_rows),
              "manifest_sha256": manifest_hash, "versions": runtime_versions(),
              "is_benchmark_result": False if run_kind == "synthetic_smoke" else None,
              "contains_predictions": False, "video_files_decoded": 0, "models_fitted": 0,
              "val_test_files_read": 0, "output_manifest": "manifest_with_folds.csv",
              "limitations": ["Hashes identify byte-identical files only; near-duplicate grouping comes from the data owner",
                              "No decoding/quality check is performed; every OOF fold must still fit on real train only",
                              "Use one common exported manifest for A/B/C; fold_assignments.csv here contains no branch scores",
                              "Resolved absolute paths are local to this machine; the data owner must map paths on another machine"]}
    write_json(output_dir / "fold_plan.json", report)
    return report
