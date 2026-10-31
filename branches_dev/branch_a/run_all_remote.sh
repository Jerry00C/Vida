#!/usr/bin/env bash
# Run from any directory. This script lives inside branches_dev/branch_a/.
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT/branches_dev/branch_a"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-0}"
FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION="${ACCUMULATION:-1}"
PRECISION="${PRECISION:-auto}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
# Download/authenticate beforehand if weights are not already cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
SPLITS="${SPLITS:-$PROJECT_ROOT/branches_dev/branch_a/runs/branch_a/splits_v1.json}"
HF_CACHE="${HF_CACHE:-$PROJECT_ROOT/branches_dev/branch_a/runs/branch_a/huggingface}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/branches_dev/branch_a/processed}"
RUN_DIR="${1:-$PROJECT_ROOT/branches_dev/branch_a/runs/remote_sequential}"
RUN_DIR="$($PYTHON -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RUN_DIR")"
export PROJECT_ROOT RUN_DIR DEVICE WORKERS FRAME_BATCH_SIZE BATCH_SIZE ACCUMULATION PRECISION SPLITS HF_CACHE DATA_ROOT
mkdir -p "$RUN_DIR/logs"
# One writer per output directory. Available on Linux remote servers.
exec 9>"$RUN_DIR/.runner.lock"
flock -n 9 || { echo 'Another runner is using this output folder.'; exit 1; }
exec > >(tee -a "$RUN_DIR/logs/pipeline.log") 2>&1
trap 'echo "FAILED at line $LINENO. Re-run the same command/configuration to resume." >&2' ERR
printf 'Output directory: %s\n' "$RUN_DIR"

# Verify immutable settings and retain source/config/split provenance.
"$PYTHON" - <<'PY'
import os,sys,hashlib,json,shutil,subprocess
from pathlib import Path
import torch,transformers,peft,sklearn
from dev.splits import load_split
root=Path(os.environ['PROJECT_ROOT']); out=Path(os.environ['RUN_DIR'])
data=Path(os.environ['DATA_ROOT']).resolve(); hf=Path(os.environ['HF_CACHE']).resolve()
assert not out.is_relative_to(data), 'Output must not be inside processed data'
assert not out.is_relative_to(hf), 'Output must not be inside model cache'
assert hf.is_dir(), 'Missing pretrained cache: run python -m dev download first'
if os.environ['DEVICE'].startswith('cuda'):
 assert torch.cuda.is_available(), 'CUDA not available in this Python environment'
split=load_split(os.environ['SPLITS'])
assert not split['smoke'], 'Full training requires the full split, not smoke splits'
assert set(split['rounds'])=={'round_1','round_2','round_3','round_4'}
for name,value in split['index_fingerprints'].items():
 assert hashlib.sha256((data/'indexes'/name).read_bytes()).hexdigest()==value, 'Data index mismatch'
files=list((root/'branches_dev/branch_a/dev').glob('*.py'))+list((root/'branches_dev/branch_a/dev').glob('*.json'))+list((root/'branches_dev/branch_a/dev').glob('*.txt'))+list((root/'stacking').glob('*.py'))+[root/'branches_dev/branch_a/run_all_remote.sh']
spec={'round_training':'sequential_previous_best; later unseen scores are not held-out estimates','split_identity':split['split_identity'],
      'settings':{k:os.environ[k] for k in ('DEVICE','WORKERS','FRAME_BATCH_SIZE','BATCH_SIZE','ACCUMULATION','PRECISION')},
      'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'sklearn':sklearn.__version__},
      'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
p=out/'run_manifest.json'
if p.exists():
 assert json.loads(p.read_text())==spec, 'Source/config/environment changed: use a new run directory'
else:
 assert not (out/'frozen').exists() and not (out/'adapt').exists(), 'Use an empty output folder'
 p.write_text(json.dumps(spec,indent=2)+'\n')
for f in files:
 dest=out/'source'/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
shutil.copy2(os.environ['SPLITS'],out/'splits.json')
(out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Preflight passed. GPU:',torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
PY

# Copy once per invocation to retain frozen weights and repair interrupted copies.
# Preserve the cache's relative symlinks and targets. No videos/PNGs are copied.
mkdir -p "$RUN_DIR/huggingface"
cp -a "$HF_CACHE/." "$RUN_DIR/huggingface/"
HF_SAVED="$RUN_DIR/huggingface"
BASE_CONFIG="$RUN_DIR/source/branches_dev/branch_a/dev/config.json"
ADAPT_CONFIG="$RUN_DIR/source/branches_dev/branch_a/dev/adaptation_config.json"
SPLIT_SAVED="$RUN_DIR/splits.json"

for backbone in clip dinov2 dinov3; do
  "$PYTHON" -m dev --config "$BASE_CONFIG" --processed "$DATA_ROOT" \
    --hf-cache "$HF_SAVED" --output "$RUN_DIR/frozen" features \
    --splits "$SPLIT_SAVED" --backbone "$backbone" --device "$DEVICE" \
    --frame-batch-size "$FRAME_BATCH_SIZE" --workers "$WORKERS"
  previous=""
  for round in round_1 round_2 round_3 round_4; do
    train_dir="$RUN_DIR/frozen/training/$backbone/$round"
    init_args=()
    if [[ -n "$previous" ]]; then init_args=(--init-checkpoint "$previous"); fi
    "$PYTHON" -m dev --config "$BASE_CONFIG" --processed "$DATA_ROOT" \
      --output "$RUN_DIR/frozen" train --splits "$SPLIT_SAVED" \
      --cache "$RUN_DIR/frozen/cache/$backbone" --round "$round" --run "$train_dir" --resume "${init_args[@]}"
    "$PYTHON" -m dev --config "$BASE_CONFIG" --processed "$DATA_ROOT" \
      --output "$RUN_DIR/frozen" evaluate --splits "$SPLIT_SAVED" \
      --cache "$RUN_DIR/frozen/cache/$backbone" --round "$round" --run "$train_dir"
    previous="$train_dir/best.pt"
  done
done

for mode in partial lora; do
  for backbone in clip dinov2 dinov3; do
    previous=""
    for round in round_1 round_2 round_3 round_4; do
      train_dir="$RUN_DIR/adapt/training/$mode/$backbone/$round"
      init_args=()
      if [[ -n "$previous" ]]; then init_args=(--init-checkpoint "$previous"); fi
      resume_args=()
      if [[ -f "$train_dir/latest.pt" ]]; then resume_args=(--resume); fi
      "$PYTHON" -m dev.adapt train --mode "$mode" --backbones "$backbone" \
        --root "$DATA_ROOT" --split "$SPLIT_SAVED" --config "$ADAPT_CONFIG" \
        --hf-cache "$HF_SAVED" --output "$RUN_DIR/adapt" --round "$round" \
        --device "$DEVICE" --precision "$PRECISION" --workers "$WORKERS" \
        --batch-size "$BATCH_SIZE" --accumulation "$ACCUMULATION" \
        --frame-microbatch "$FRAME_BATCH_SIZE" "${resume_args[@]}" "${init_args[@]}"
      "$PYTHON" -m dev.adapt evaluate --root "$DATA_ROOT" \
        --split "$SPLIT_SAVED" --config "$ADAPT_CONFIG" --hf-cache "$HF_SAVED" \
        --round "$round" --checkpoint "$train_dir/best.pt" \
        --output "$train_dir/evaluation" --device "$DEVICE" \
        --workers "$WORKERS" --frame-microbatch "$FRAME_BATCH_SIZE"
      previous="$train_dir/best.pt"
    done
  done
done

"$PYTHON" - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
from dev.adapt_comparison import compare
root=Path(sys.argv[1])
rows=compare(root/'adapt', frozen_root=root/'frozen')
assert len(rows)==72, f'Expected 36 runs x 2 test partitions; found {len(rows)}'
print('Verified 36 experiments with seen/unseen evaluation outputs.')
PY
# Hash saved artifacts for transfer verification; excludes the actively written log.
(cd "$RUN_DIR" && find source huggingface frozen adapt -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
printf 'COMPLETE\n' > "$RUN_DIR/COMPLETE"
printf 'All training/evaluation finished. Transfer this folder: %s\n' "$RUN_DIR"
