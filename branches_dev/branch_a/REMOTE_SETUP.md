# Remote training package

Transfer this entire `branches_dev/branch_a/` directory, preserving symlinks (for example with
`rsync -a`). On the remote machine, **cd into the transferred directory** before
running the commands below. Its name does not matter; it must contain sibling
directories `dev/`, `processed/`, `runs/`, and `tests/`.

The module commands below work from the transferred folder. For the full
`run_all_remote.sh` orchestration script, preserve the repository layout
`PR2/branches_dev/branch_a/run_all_remote.sh` and invoke it from any directory.

## Environment

Create a fresh Python 3.11 environment; do not transfer the Mac virtualenv.
Install a CUDA-compatible PyTorch build for the remote machine first, then:

```sh
python3.11 -m venv .venv-branch-a
source .venv-branch-a/bin/activate
# Install the appropriate GPU PyTorch build here if not already installed.
python -m pip install -r dev/requirements.adaptation.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest tests/test_branch_a.py tests/test_adaptation.py -q
```

The requirements.lock.txt and requirements.adaptation.lock.txt files record the
tested local versions; the remote GPU runtime still needs a compatible PyTorch
build. Dependency installation requires network access or a provisioned package
mirror. Model inference can run offline using the included Hugging Face cache:

```sh
export HF_HUB_OFFLINE=1
python -m dev.adapt inspect --mode partial
python -m dev.adapt inspect --mode lora
```

If the model cache was omitted, unset HF_HUB_OFFLINE and run
`python -m dev download`. DINOv3 downloading requires approved Hugging Face
access and authentication on the remote machine.

## Fresh remote smoke checks

Use fresh output folders: previous local smoke runs were made on CPU. Adaptation
resume identities include device, precision, and library versions; do not try
to resume those CPU checkpoints as a new GPU run.

```sh
python -m dev --output runs/remote_smoke/frozen smoke --device cuda
python -m dev.adapt smoke --mode partial --device cuda --output runs/remote_smoke/adapt
python -m dev.adapt smoke --mode lora --device cuda --output runs/remote_smoke/adapt
```

These commands use the existing small split; inspect their reports for failures.
For a CPU-only remote machine, replace `cuda` with `cpu`.

## Full experiments

The old 75-video smoke embedding caches have been removed. Extract full
embeddings into a separate cache before frozen-head training:

```sh
for backbone in clip dinov2 dinov3; do
  python -m dev --output runs/remote_frozen features --backbone "$backbone" --device cuda --frame-batch-size 8
done
python -m dev --output runs/remote_frozen train-all --cache-root runs/remote_frozen/cache
```

Partial fine-tuning and LoRA read PNGs directly and do not need embedding caches:

```sh
python -m dev.adapt train-all --mode partial --device cuda --output runs/remote_adapt
python -m dev.adapt train-all --mode lora --device cuda --output runs/remote_adapt
```

Choose batch size/accumulation to fit the GPU before starting, then retain those
settings for resume. Test evaluation is separate; see dev/README.md and
dev/ADAPTATION.md, overriding run/checkpoint/output paths to match these
remote folders. Full training is not launched by packaging or verification.

## Package audit (2026-10-26)

- Moved processed/ from inside branches_dev/branch_a/ to this directory to match CLI defaults.
- All 256,000 indexed frame paths exist for 32,000 videos.
- Both original index hashes match the full split manifest.
- All three pinned pretrained snapshots include config, processor and weights;
  no broken links were found in the model cache.
- All three included embedding caches passed cache integrity checks and have
  shape [75, 8, 768]. They are smoke caches only.
- Copied the two training test modules into tests/.
- All three models and processors loaded successfully offline from this package.
- 24 training tests passed locally: 23 in the initial run, and the multi-worker
  loader test in a separate run outside the sandbox after it stalled inside.

Historical report paths may mention the original Mac location. They are
provenance, not training input paths. Keep index CSVs unchanged because the
split/cache identities use their exact hashes. This packaging check verified
frame existence, not a new byte-by-byte hash/decode of every PNG. Remote CUDA
execution must be verified on the target machine.

## Minimal ZIP package (2026-10-27)

`dev-remote-minimal.zip` includes code/configuration/dependencies, tests, this
guide, exactly the indexed PNG frames, the original CSV indexes, full and smoke
split manifests, audit provenance, and the three pinned pretrained snapshots.
Model snapshot files are stored as ordinary files so ZIP extraction does not
depend on symlink support. Original videos, preprocessing records/reports, old
training outputs, embedding caches, and Python/pytest caches are omitted.
The historical audit above describes checks before this cleanup.

Extract with `unzip dev-remote-minimal.zip`, then `cd branch_a`. Use a new destination
directory to avoid mixing this package with older run outputs. The ZIP uses
ZIP64 and no compression; PNG frames already occupy most of the package size.
