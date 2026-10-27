# VIDA video preprocessing

Python 3.9+; tested on macOS ARM64 with PyAV 15.1.0 and Pillow 11.3.0.
Run commands from the repository root. This pipeline does not train models, create
splits, resize images, or apply CLIP/DINO preprocessing.

## Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r preprocessing/requirements.txt
```

## Delivery migration

```sh
.venv/bin/python -B preprocessing/vida.py migrate --dev dev
```

The original delivery now lives in `preprocess-from Zhou/` (migrated into `dev/`, later moved to the project root), including its
34 video ZIPs, one metadata ZIP, manifests, schemas, spreadsheet, splits and
verification script. `dev/migration_inventory.json` records all original file
names, sizes and SHA-256 hashes before the move. Migration renames whole delivery
items on the same filesystem, refuses collisions, and verifies all hashes after
the move. Rerunning resumes an interrupted migration or verifies the completed
migration. Only explicitly listed delivery items move; code, environments, Git
history and `sources/` are untouched. No archives are deleted or modified.

## Extraction

```sh
# Approximately 100 development clips: 20 per source, training split only.
.venv/bin/python -B preprocessing/vida.py extract --pilot-size 100 --workers 2

# FULL RUN: all 32,000 manifest videos (resume is safe).
.venv/bin/python -B preprocessing/vida.py extract --workers 2 --resume

# Verify the most recent requested selection and its published CSV indexes.
.venv/bin/python -B preprocessing/vida.py validate
```

Paths default to `--delivery 'preprocess-from Zhou' --output processed`.
Both are configurable; output must be a separate tree from the archived delivery.
Repeat `--family NAME` or `--split NAME` to filter. `--clip-ids FILE` accepts one
exact clip ID per line, checked against the other filters. `--pilot-size N`
always uses training clips; selection is deterministic, balances sources and
cycles through resolution/frame-count buckets for format coverage. It is not
an unbiased sample for estimating compression rates.

Output layout:

```text
processed/
  real/<source>/videos/<key>.<original-extension>
  real/<source>/frames/<key>/<key>_frame_000.png ... _frame_007.png
  generated/<source>/videos/<key>.<original-extension>
  generated/<source>/frames/<key>/<key>_frame_000.png ... _frame_007.png
  indexes/{videos,frames,extraction_errors}.csv
  reports/{extraction_config,extraction_report}.json
  .records/<hash-of-clip-id>.json
```

Class and source come from the manifest (`0=real`, `1=generated`). Current source
names and clip IDs need no normalization. Unsafe IDs use a SHA-256-derived key;
case-insensitive filesystem collisions are rejected. `filesystem_key` and
`source_key` record the mapping without changing original identifiers.
All organized video and frame paths are relative to the processed root. Archive
paths are relative to the delivery root. `archive_member_path` locates the video
inside the delivery ZIP; original `source_member` retains its upstream location.
Original `video_path` and `metadata_path` retain delivery semantics, not processed
path semantics. Configuration records the machine-local delivery root as an
informational field; override `--delivery` after moving machines.

`videos.csv` preserves all manifest columns, including `group_id`,
`original_source_id`, `split`, `fold`, `validation_role`, `provenance_status`, and
source hash. `frames.csv` joins via `clip_id`; timestamps are actual decoded PTS
multiplied by time base in seconds, without shifting their origin. Missing values
are blank with `timestamp_missing=True`. Frame hashes and dimensions support
integrity checks. Errors record the clip, stage and underlying exception.

Existing splits remain metadata only. The current delivery has fixed development
and cross-source splits; future rotating-generator evaluations need separate
split tables. No appearance from held-out videos was used to choose settings.

## Sampling and resource limits

`decoded-index-floor-v1`: for N decoded frames, select
`floor(k * (N - 1) / 7)` for k=0,...,7. This includes both endpoints and gives
eight distinct indices for N>=8. It is uniform in frame index, not elapsed time.
Exactly eight-frame videos use every frame. Fewer than eight frames, invalid
sizes, corrupt frames, decode errors, non-monotonic timestamps or a decoded count
that differs from the manifest fail explicitly.

Video extraction streams one archive member per worker and verifies its SHA-256.
Two sequential decoding passes count/validate frames and then save selected
frames, avoiding whole-video decoded-frame buffers. Decoders use one thread each;
`--workers` controls parallel clips. PNGs are RGB, native decoded resolution,
lossless, compression level 6. No model preprocessing is applied.

Preflight checks archive-member safety, uniqueness, file sizes, labels, dimensions,
and source availability. It checks free disk against extracted-video bytes plus
8 native-size RGB buffers per clip with PNG overhead and a 2 GiB reserve. ZIPs
already occupy disk and are not counted again as new storage. The check is
conservative even for resumed clips. Frame compression estimates from the pilot
are resolution-weighted and uncertain for unmeasured sources; the report also
provides a conservative raw-RGB-based PNG estimate. The entire delivery's package
hashes can separately be verified with:

```sh
.venv/bin/python -B 'preprocess-from Zhou/verify_delivery.py'
```

## Recovery and validation

Resume is the default. Repeating the same extraction command validates source and
settings fingerprints, organized-video hash, all eight PNG hashes/readability,
RGB modes, dimensions, indices, timestamps and paths before skipping a clip.
Per-clip JSON records are the recovery authority; CSV indexes are rebuilt by the
single coordinator. They cover the **latest requested selection**, not every
historically extracted clip. Running the full command incorporates pilot outputs
and rebuilds indexes for all videos. Completed data from other selections remains
on disk without duplicate copies.

Videos and frame directories publish through atomic renames. A clip is complete
only after its record commits. An interruption between publishing frames and
committing a record causes reprocessing. Global CSVs are individually atomic;
if interrupted during publication, rerun extraction to rebuild a consistent set.
An OS lock prevents simultaneous processes writing the same output root.

Invalid existing frame directories are retained under `.quarantine/` before
replacement. An existing video with different bytes is never overwritten: inspect
and move it aside, then retry. `--no-resume` explicitly regenerates frames and
retains previous directories in quarantine. These retained directories and
orphan `.video-*`/`.frames-*` temporary paths after a forced termination consume
extra disk; inspect them before manually removing them. Settings/decoder changes
require a new output root to avoid mixing extraction versions.

```sh
.venv/bin/python -B -m pytest preprocessing/test_vida.py -q
```

Tests cover deterministic sampling, unsafe paths, byte-preserving extraction,
short-clip failures, mismatched video protection, readable-but-altered PNGs,
missing images, interrupted record publication, frame linkage and duplicate-free
index rebuilding. `validate` compares canonical committed records with every
index field and verifies all successful outputs against the current manifest.

## Measured pilot results (2026-10-16, Asia/Singapore)

- Delivery migration: 49 files inventoried and SHA-256 verified after moving.
- Pilot: 100 training clips, 20 each from Vript, Pika, VideoCrafter2, T2V-Zero
  and OpenSora; 100 successful, zero failures, 800 RGB PNGs.
- Initial extraction: 15.76 seconds with two workers (local cached I/O; not a
  reliable full-run timing prediction).
- Video bytes: 60,469,239; PNG bytes: 296,732,909; combined media: 357,202,148
  bytes (357.2 MB decimal), excluding index/record filesystem overhead.
- Resume: all 100 clips validated and skipped in 1.79 seconds; index counts remain
  100 videos and 800 frames. Post-resume validation passed.
- Tests: 11 passed, including partial migration recovery and preserving `sources/`.
- Visual QA: first/middle/last selected frames from two clips per source (30
  images) inspected in `processed/reports/pilot_contact_sheet.jpg`. No obvious
  channel swaps or decoding corruption observed. Native content, including
  stylized images and screen recordings, was retained without filtering.
- Full dataset estimate: 160.82 GB PNGs + 36.88 GB extracted videos = **197.70 GB
  processed media**. With the retained 36.93 GB ZIP delivery, total media is about
  **234.63 GB**. This estimate is uncertain for held-out source compression.
- Conservative full-run planning allowance: 540.89 GB PNGs + 36.88 GB videos =
  **577.76 GB additional media**, plus the 2 GiB reserve. This is a raw-pixel-based
  allocation check, not a measured full-run size.

`reports/pilot_initial_report.json` preserves the first-run measurements;
`reports/pilot_resume_report.json` preserves the resume measurements.
`reports/extraction_report.json` always describes the latest extraction command.
The initial delivery stopped after this pilot. The subsequently authorized full-run results follow below.


## Full extraction (2026-10-16, Asia/Singapore)

The authorized full run completed with eight workers: **32,000 successful videos,
256,000 PNG frames, zero failed clips**. It reused 1,229 previously completed
clips and took 3,335.26 seconds (55.59 minutes) for this resumed run.

Measured media sizes: 36,876,541,730 video bytes plus 138,076,429,002 PNG bytes,
for **174,952,970,732 bytes (174.95 GB decimal)**, excluding indexes and filesystem
overhead. Original delivery ZIPs remain archived separately.

The independent CSV audit confirmed unique video IDs and paths, unique image
paths, exactly eight linked frames per video, 16,000 videos in each class, and
zero error rows. The post-extraction test suite passed all 11 tests. Full file
validation is recorded separately in `dev/full_validation.log`.

Logs: `dev/full_preprocessing.log`, `dev/preprocessing_tests.log`, and
`dev/full_validation.log`. Full-run measurements are in
`processed/reports/extraction_report.json`; use its `png_bytes` and
`video_bytes` fields for actual storage (the generic report retains pilot-oriented
labels for its compression estimate fields).


Full validation completed successfully (confirmed 2026-10-17, Asia/Singapore):
`Validated 32000 clips and 256000 frames`. The validator exited with code 0.
This includes video hashes, all PNG hashes/readability/RGB dimensions, frame
indices/timestamps, source metadata and index-to-record consistency.
The completion summary is `processed/reports/full_run_verification.json`.
