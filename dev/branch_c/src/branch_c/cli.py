import argparse
import json
import sys

from .config import load_config
from .manifest import load_manifest
from .pipeline import evaluate, fit, make_oof, predict_video


def main(argv=None):
    parser = argparse.ArgumentParser(description="Branch C real-only video anomaly evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "extract", "fit", "oof", "select", "folds"):
        p = sub.add_parser(command)
        p.add_argument("--manifest", required=True)
        p.add_argument("--config", default=None)
        if command not in {"validate", "folds"}:
            p.add_argument("--cache-dir", default="artifacts/cache")
            p.add_argument("--workers", type=int, default=1)
        if command in {"fit", "oof", "select", "folds"}:
            p.add_argument("--output-dir", required=True)
            p.add_argument("--run-kind", choices=["auto", "research", "synthetic_smoke"], default="auto",
                           help="auto preserves synthetic_smoke provenance from the manifest")
        if command == "extract":
            p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
        if command in {"oof", "folds"}:
            p.add_argument("--folds", type=int, default=5)
    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="artifacts/cache")
    p.add_argument("--bootstrap-repeats", type=int)
    p.add_argument("--test-subset", help="Preserved evaluation_subset tag; same frozen model and threshold")
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("benchmark", help="Fresh-process CPU inference latency and peak RSS (macOS/Linux)")
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        if args.command in {"validate", "extract", "fit", "oof", "select", "folds"}:
            config = load_config(args.config)
        if args.command == "validate":
            rows = load_manifest(args.manifest, config)
            result = {"clips": len(rows), "splits": {s: sum(r.split == s for r in rows) for s in ("train", "val", "test")},
                      "note": "Metadata check only; no video decoding or fitting"}
        elif args.command == "extract":
            from .pipeline import _prepare
            selected = {"train", "val", "test"} if args.split == "all" else {args.split}
            rows, _, _ = _prepare(args.manifest, config, args.cache_dir, selected, workers=args.workers)
            result = {"extracted_clips": len(rows), "cache_dir": args.cache_dir}
        elif args.command == "fit":
            result = fit(args.manifest, config, args.output_dir, args.cache_dir, args.run_kind, workers=args.workers)
        elif args.command == "oof":
            result = make_oof(args.manifest, config, args.output_dir, args.cache_dir, args.folds, args.run_kind, workers=args.workers)
        elif args.command == "folds":
            from .folds import prepare_folds
            result = prepare_folds(args.manifest, config, args.output_dir, args.folds, args.run_kind)
        elif args.command == "evaluate":
            result = evaluate(args.manifest, args.model, args.output_dir, args.cache_dir, args.bootstrap_repeats,
                              test_subset=args.test_subset, workers=args.workers)
        elif args.command == "select":
            from .selection import select_model
            result = select_model(args.manifest, config, args.output_dir, args.cache_dir, args.run_kind, workers=args.workers)
        elif args.command == "benchmark":
            from .benchmark import benchmark_video
            result = benchmark_video(args.model, args.video, args.output, args.repeats, args.timeout_seconds)
        else:
            result = predict_video(args.model, args.video, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"branch-c: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
