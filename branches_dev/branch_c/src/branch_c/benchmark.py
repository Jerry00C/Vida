"""Measure single-video CPU inference in fresh processes, including native peak RSS."""
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def _rss_bytes(value, system):
    # getrusage(2): macOS reports bytes; Linux reports KiB.
    if system not in {"Darwin", "Linux"}:
        raise ValueError("Peak RSS measurement currently supports macOS and Linux")
    if value <= 0:
        raise ValueError("Operating system did not return a positive peak RSS")
    return int(value * (1 if system == "Darwin" else 1024))


def _worker(model_path, video_path):
    started = time.perf_counter()
    import resource
    import cv2
    from .model import runtime_versions
    from .pipeline import predict_video
    imported = time.perf_counter()
    prediction = predict_video(model_path, video_path)
    peak = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform.system())
    return {"pid": os.getpid(), "imports_seconds": imported - started,
            "timings": prediction["timings"], "inference_seconds": prediction["inference_seconds"],
            "peak_rss_bytes": peak, "run_kind": prediction["run_kind"],
            "protocol": prediction["protocol"], "model_id": prediction["model_id"],
            "anomaly_score": prediction["anomaly_score"], "is_probability": False,
            "input_metadata": prediction["metadata"], "versions": runtime_versions(),
            "opencv_threads": cv2.getNumThreads()}


def _range(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def benchmark_video(model_path, video_path, output_path, repeats=3, timeout_seconds=120):
    from .cache import file_sha256
    from .config import require_integer, require_number, write_json
    require_integer("benchmark repeats", repeats, lower=1, upper=30)
    require_number("benchmark timeout_seconds", timeout_seconds, lower=1, upper=3600)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Benchmark output already exists; choose a new report path")
    if platform.system() not in {"Darwin", "Linux"}:
        raise ValueError("Peak RSS measurement currently supports macOS and Linux")
    model_path, video_path = Path(model_path).resolve(), Path(video_path).resolve()
    model_hash, video_hash = file_sha256(model_path), file_sha256(video_path)
    runs = []
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            completed = subprocess.run([sys.executable, "-m", "branch_c.benchmark", "--worker",
                                        str(model_path), str(video_path)],
                                       capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError("Benchmark worker timed out; no partial result reported") from error
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise ValueError(f"Benchmark worker failed: {completed.stderr.strip()[:1200]}")
        run = json.loads(completed.stdout)
        run["process_wall_seconds"] = elapsed
        runs.append(run)
    if file_sha256(model_path) != model_hash or file_sha256(video_path) != video_hash:
        raise ValueError("Benchmark input changed during measurement; results discarded")
    if len({(r["model_id"], r["run_kind"], r["protocol"]) for r in runs}) != 1:
        raise ValueError("Benchmark workers did not use the same model/protocol")
    reference = runs[0]
    report = {"run_kind": reference["run_kind"], "protocol": reference["protocol"],
              "is_detection_benchmark_result": False, "model_id": reference["model_id"],
              "model_sha256": model_hash, "video_sha256": video_hash,
              "model_size_bytes": model_path.stat().st_size, "input_path": str(video_path),
              "input_format": video_path.suffix.lower(), "repeats": repeats,
              "mode": "fresh_process_per_repeat_no_feature_cache",
              "summary": {"process_wall_seconds": _range([r["process_wall_seconds"] for r in runs]),
                          "inference_seconds_excluding_model_load": _range([r["inference_seconds"] for r in runs]),
                          "total_seconds_including_model_load": _range([r["timings"]["total_seconds_including_model_load"] for r in runs]),
                          "peak_rss_bytes": _range([r["peak_rss_bytes"] for r in runs])},
              "environment": {"system": platform.system(), "release": platform.release(),
                              "machine": platform.machine(), "processor": platform.processor(),
                              "logical_cpu_count": os.cpu_count(), "python_executable": sys.executable,
                              "versions": reference["versions"], "opencv_threads": reference["opencv_threads"],
                              "thread_environment": {k: os.environ.get(k) for k in
                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
              "runs": runs,
              "measurement_notes": ["Peak RSS is the whole worker process high-water mark, including Python, imports, native allocations and model load",
                                    "Peak RSS is not incremental model memory, GPU memory or Python-only tracemalloc memory",
                                    "Process wall time includes startup/imports, model load, decoding, descriptors, scoring and IPC; input hashing/report writing are excluded",
                                    "OS file cache is uncontrolled; fresh process does not imply cold disk cache",
                                    "NPZ measures the pre-sampled-frame path; use an actual 10-second video on the agreed machine for the final project measurement",
                                    "Model size includes clip/window models, threshold and audit metadata"]}
    write_json(output_path, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Internal worker: use branch-c benchmark")
    try:
        print(json.dumps(_worker(sys.argv[2], sys.argv[3]), allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
