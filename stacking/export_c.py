"""Frozen C inference on the independent suite, using C's locked environment."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'branches_dev/branch_c/src'))
from branch_c.model import load_bundle, runtime_versions
from branch_c.video import read_video
from branch_c.features import extract_features

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    root = Path(args.root); model = Path(args.model)
    manifest = root/'manifests/manifest.csv'
    rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
    bundle = load_bundle(model)  # Keep original strict runtime/feature-signature checks.
    folder = root/'branch_scores/raw/C_fixed_nu001'; folder.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': sha(manifest), 'checkpoint_sha256': sha(model),
                'model_id': bundle['model_id'], 'versions': runtime_versions(),
                'feature_signature': bundle['feature_signature'], 'model_retrained': False}
    lock = folder/'IDENTITY.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == identity, 'Resume identity differs'
    else: lock.write_text(json.dumps(identity, indent=2)+'\n')
    log = folder/'scores.jsonl'; done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            item = json.loads(line); assert item['clip_id'] not in done
            done[item['clip_id']] = item
    def infer(row):
        path = root/row['video_path']; assert sha(path) == row['sha256']
        started = time.perf_counter()
        video = read_video(path, bundle['config'].features)
        feature = extract_features(video, bundle['config'].features)
        score = float(bundle['clip_model'].score(feature.clip[None])[0])
        assert math.isfinite(score)
        return {'clip_id': row['clip_id'], 'branch_c_score': score, 'status': 'SUCCESS',
                'model_id': bundle['model_id'], 'sha256': row['sha256'],
                'seconds': time.perf_counter()-started}
    started = time.perf_counter(); missing = [r for r in rows if r['clip_id'] not in done]
    with log.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(args.workers) as pool:
        for item in pool.map(infer, missing):
            stream.write(json.dumps(item)+'\n'); stream.flush(); done[item['clip_id']] = item
            if len(done)%100 == 0: print(f'C: {len(done)}/{len(rows)}, {time.perf_counter()-started:.1f}s this invocation', flush=True)
    target = folder/'C_native.csv'
    fields = ['clip_id','branch_c_score','status','model_id','sha256','seconds']
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        writer.writerows(done[r['clip_id']] for r in rows)
    report = identity | {'rows': len(rows), 'cached_rows': len(rows)-len(missing),
                         'wall_seconds_this_invocation': time.perf_counter()-started,
                         'native_csv_sha256': sha(target),
                         'timing_note': 'Concurrent throughput; not serial per-video latency. Clip score only; window explanations omitted.'}
    (folder/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
