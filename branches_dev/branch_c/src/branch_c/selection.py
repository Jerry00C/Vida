"""Small, predeclared validation-only comparison of the proposal's two estimators."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .cache import file_sha256
from .config import ModelConfig, fingerprint, write_json
from .evaluation import choose_real_threshold, metrics
from .features import feature_signature
from .manifest import load_manifest, manifest_protocol, resolve_run_kind, validation_masks
from .model import RealOnlyAnomalyModel, runtime_versions
from .pipeline import _fit_prepared, _prepare, _write_scores


def predeclared_candidates(seed):
    # Baseline wins exact ties. Order and budget are fixed before accessing data.
    return [
        (f"if_{trees}_{samples}", ModelConfig(n_estimators=trees, max_samples=samples, seed=seed))
        for trees, samples in [(200, 256), (100, 128), (100, 256), (200, 128)]
    ] + [
        (f"svm_nu_{nu:g}", ModelConfig(algorithm="one_class_svm", nu=nu, seed=seed))
        for nu in (.01, .05)
    ]


def select_model(manifest, config, output_dir, cache_dir, run_kind="auto", workers=1):
    started = time.perf_counter()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Selection output already exists; choose a new run directory")
    manifest_rows = load_manifest(manifest, config)
    run_kind = resolve_run_kind(manifest_rows, run_kind)
    selection_mask, _ = validation_masks(manifest_rows)
    val_rows = [r for r, selected in zip(manifest_rows, selection_mask) if selected]
    if {r.label for r in val_rows} != {0, 1}:
        raise ValueError("Model selection requires real and generated val clips; use fit for real-only val")
    if sum(r.split == "train" and r.label == 0 for r in manifest_rows) < 4:
        raise ValueError("At least 4 real training clips are required for model selection")
    candidates = predeclared_candidates(config.model.seed)
    plan = {"plan_version": 1, "run_kind": run_kind, "protocol": manifest_protocol(manifest_rows),
            "base_config": config.to_dict(), "candidate_budget": len(candidates),
            "candidates": [{"candidate_id": name, "model": asdict(model)} for name, model in candidates],
            "ranking_rule": ["validation AUROC descending", "validation macro-F1 descending",
                             "predeclared candidate order ascending (baseline first on exact ties)"],
            "threshold_rule": "Ranking uses real model_selection val quantile; winner threshold uses separate real calibration val if roles are supplied",
            "selection_clip_ids": [r.clip_id for r in val_rows],
            "manifest_sha256": file_sha256(manifest), "versions": runtime_versions(),
            "feature_signature": feature_signature(config.features),
            "test_policy": "Metadata checked; test files never read, hashed, extracted or scored"}
    plan_id = fingerprint(plan)
    # A partial run retains this frozen plan, but never a completed selection report.
    output_dir.mkdir(parents=True)
    write_json(output_dir / "selection_plan.json", plan | {"plan_id": plan_id})
    rows, features, hashes = _prepare(manifest, config, cache_dir, {"train", "val"},
                                     manifest_rows=manifest_rows, real_training_only=True, workers=workers)
    train = np.array([r.split == "train" for r in rows])
    val, _ = validation_masks(rows)
    X = np.stack([f.clip for f in features])
    labels = np.array([r.label for r in rows])
    ids = np.array([r.clip_id for r in rows])
    comparisons = []
    for order, (candidate_id, model_config) in enumerate(candidates):
        model = RealOnlyAnomalyModel(model_config).fit(X[train], labels[train], ids[train])
        scores = model.score(X[val])
        threshold = choose_real_threshold(scores, labels[val], config.evaluation.real_false_positive_rate)
        result = {"candidate_id": candidate_id, "predeclared_order": order,
                  "model": model.metadata(), "threshold": threshold,
                  "validation": metrics(labels[val], scores, threshold)}
        comparisons.append(result)
        _write_scores(output_dir / "candidates" / f"{candidate_id}_validation_scores.csv",
                      [r for r, selected in zip(rows, val) if selected], scores, "candidate_validation",
                      model_id=f"{plan_id}:{candidate_id}", run_kind=run_kind,
                      content_hashes=[h for h, selected in zip(hashes, val) if selected])
    ranked = sorted(comparisons, key=lambda c: (-c["validation"]["auroc"],
                    -c["validation"]["macro_f1"], c["predeclared_order"]))
    winner = ranked[0]
    selected_config = replace(config, model=candidates[winner["predeclared_order"]][1])
    provenance = {"plan_id": plan_id, "candidate_id": winner["candidate_id"],
                  "selection_split": "val", "candidate_budget": len(candidates),
                  "test_files_read": 0,
                  "note": "Selected validation metrics are optimistic; final evaluation uses sealed test"}
    if file_sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("Manifest changed during model selection; choose a fresh run with a frozen manifest")
    fit_report = _fit_prepared(manifest, selected_config, output_dir / "selected", rows, features,
                               hashes, run_kind, manifest_rows, started, provenance)
    if fit_report["model_selection_validation"] != winner["validation"]:
        raise ValueError("Selected model refit did not reproduce candidate validation metrics")
    if file_sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("Manifest changed during selected model refit; selection is incomplete")
    write_json(output_dir / "selected_config.json", selected_config.to_dict())
    report = {"status": "complete", "run_kind": run_kind,
              "is_benchmark_result": False if run_kind == "synthetic_smoke" else None,
              "protocol": plan["protocol"], "plan_id": plan_id,
              "ranking_rule": plan["ranking_rule"], "ranked_candidates": ranked,
              "selected_candidate_id": winner["candidate_id"], "model_id": fit_report["model_id"],
              "selected_config_path": "selected_config.json", "selected_model_path": "selected/model.joblib",
              "selection_split": "val", "test_files_read": 0,
              "inputs_read": [{"clip_id": r.clip_id, "split": r.split, "label": r.label,
                               "content_sha256": h} for r, h in zip(rows, hashes)],
              "total_seconds": time.perf_counter() - started,
              "limitations": ["Validation is used for selection and standalone threshold; it is not held-out performance",
                              "Use selected_config.json for group OOF, preserving the common A/B/C fold assignment",
                              "Do not expand the grid or repeat selection based on final test results",
                              "Scores from different estimators have different scales; never average their raw scores"]}
    write_json(output_dir / "selection_report.json", report)
    return report
