"""Portable VIDA delivery migration and deterministic, resumable RGB extraction."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager

VERSION = 'decoded-index-floor-v1'
DELIVERY_ITEMS = ['DATASET_README.md', '数据清单.xlsx', 'dataset_statistics.json',
                  'manifest_schema.json', 'clip_metadata_schema.json', 'verify_delivery.py',
                  'manifests', 'splits', 'packages', 'checksums', 'source_metadata']

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def safe_relative(value):
    p = PurePosixPath(value)
    require(value and not p.is_absolute() and '..' not in p.parts and '\\' not in value
            and ':' not in value and '\x00' not in value, f'Unsafe path: {value!r}')
    return p

def inside(root, relative):
    safe_relative(relative)
    p = root / relative
    require(p.resolve().is_relative_to(root.resolve()), f'Path escapes root: {relative}')
    return p

def safe_key(value):
    if re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}', value) and value not in {'.', '..'}:
        return value
    return 'id_' + hashlib.sha256(value.encode()).hexdigest()

@contextmanager
def lock(root):
    # OS advisory lock releases automatically after a crash; lock file may remain.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.vida.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def migrate(dev):
    target = dev / 'preprocess-from Zhou'
    journal = dev / 'migration_inventory.json'
    with lock(dev):
        if journal.exists():
            inventory = json.loads(journal.read_text())
        else:
            inventory = {'items': [], 'files': []}
            for name in DELIVERY_ITEMS:
                source = dev / name
                require(source.exists(), f'Missing delivery item: {source}')
                require(not (target / name).exists(), f'Destination already exists: {target / name}')
                inventory['items'].append(name)
                for p in sorted(source.rglob('*')) if source.is_dir() else [source]:
                    require(not p.is_symlink(), f'Symlink in delivery: {p}')
                    if p.is_file():
                        inventory['files'].append({'path': p.relative_to(dev).as_posix(),
                                                   'bytes': p.stat().st_size, 'sha256': digest(p)})
            atomic_json(journal, inventory)
        target.mkdir(exist_ok=True)
        for name in inventory['items']:
            source, destination = dev / name, target / name
            require(not (source.exists() and destination.exists()), f'Migration collision: {name}')
            if source.exists():
                source.rename(destination)
            require(destination.exists(), f'Missing migrated item: {name}')
        for entry in inventory['files']:
            p = inside(target, entry['path'])
            require(p.stat().st_size == entry['bytes'] and digest(p) == entry['sha256'],
                    f'Migration verification failed: {p}')
        inventory['verified'] = True
        atomic_json(journal, inventory)
        print(f'Migration verified: {len(inventory["files"])} files; {target}', flush=True)

def inventory(delivery):
    schema = json.loads((delivery / 'manifest_schema.json').read_text())
    require(schema['label_mapping'] == {'real': 0, 'generated': 1}, 'Unexpected label schema')
    rows = read_csv(delivery / 'manifests/manifest.csv')
    require(rows and len({r['clip_id'] for r in rows}) == len(rows), 'Duplicate/empty clip IDs')
    members = {}
    for package in read_csv(delivery / 'checksums/packages.csv'):
        archive = 'packages/' + package['package']
        path = inside(delivery, archive)
        require(path.stat().st_size == int(package['bytes']), f'Archive size mismatch: {archive}')
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                safe_relative(info.filename)
                require((info.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink rejected')
                if info.is_dir() or not info.filename.startswith('videos/'):
                    continue
                require(info.filename not in members, f'Duplicate archived path: {info.filename}')
                members[info.filename] = (archive, info.file_size)
    keys = set()
    for r in rows:
        require(r['label'] in {'0', '1'}, 'Invalid label')
        require(int(r['width']) >= 64 and int(r['height']) >= 64, 'Invalid manifest dimensions')
        require(re.fullmatch(r'[0-9a-f]{64}', r['sha256']), 'Invalid source hash')
        safe_relative(r['video_path'])
        safe_relative(r['metadata_path'])
        require(r['video_path'] in members, f'Missing video: {r["video_path"]}')
        archive, size = members[r['video_path']]
        require(size == int(r['file_bytes']), 'Archive/manifest size mismatch')
        r['filesystem_key'] = safe_key(r['clip_id'])
        r['source_key'] = safe_key(r['family'])
        r['class'] = 'real' if r['label'] == '0' else 'generated'
        key = (r['class'], r['source_key'].casefold(), r['filesystem_key'].casefold())
        require(key not in keys, f'Filesystem key collision: {r["clip_id"]}')
        keys.add(key)
        r['archive_path'] = archive
        r['archive_member_path'] = r['video_path']
        base = f'{r["class"]}/{r["source_key"]}'
        suffix = PurePosixPath(r['video_path']).suffix
        r['organized_video_path'] = f'{base}/videos/{r["filesystem_key"]}{suffix}'
        r['frame_directory_path'] = f'{base}/frames/{r["filesystem_key"]}'
    return rows

def sample_indices(n):
    require(n >= 8, f'Only {n} decoded frames; need at least eight')
    return [k * (n - 1) // 7 for k in range(8)]

def settings():
    import av
    import PIL
    return {'sampling_version': VERSION, 'frames_per_video': 8, 'color': 'RGB',
            'resize': None, 'png_compress_level': 6, 'pyav': av.__version__,
            'pillow': PIL.__version__, 'ffmpeg_libraries': {k: list(v) for k, v in av.library_versions.items()}}

def record_path(output, row):
    return output / '.records' / (hashlib.sha256(row['clip_id'].encode()).hexdigest() + '.json')

def valid_record(output, row, config, record):
    from PIL import Image
    require(record['source_fingerprint'] == fingerprint(row), 'Source metadata changed')
    require(record['settings'] == config, 'Extraction settings changed')
    require(all(record['video'].get(k) == v for k, v in row.items()), 'Video metadata changed')
    require(record['video']['frame_extraction_status'] == 'success', 'Clip incomplete')
    video = inside(output, row['organized_video_path'])
    require(digest(video) == row['sha256'], 'Organized video checksum mismatch')
    frames = record['frames']
    require(len(frames) == 8, 'Expected eight frame records')
    require([f['original_frame_index'] for f in frames] == sample_indices(record['video']['decoded_frame_count']),
            'Incorrect selected indices')
    timestamps = [f['timestamp_seconds'] for f in frames if not f['timestamp_missing']]
    require(all(a < b for a, b in zip(timestamps, timestamps[1:])), 'Unordered timestamps')
    paths = set()
    for k, f in enumerate(frames):
        require(f['clip_id'] == row['clip_id'] and f['selected_frame_position'] == k, 'Frame linkage mismatch')
        require(f['relative_video_path'] == row['organized_video_path'], 'Video linkage mismatch')
        path = inside(output, f['relative_frame_path'])
        require(path.parent == inside(output, row['frame_directory_path']), 'Wrong frame directory')
        require(path not in paths, 'Duplicate frame path')
        paths.add(path)
        require(digest(path) == f['sha256'], 'PNG checksum mismatch')
        with Image.open(path) as im:
            im.load()
            require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (f['width'], f['height']), 'Invalid PNG')
    require(set(inside(output, row['frame_directory_path']).glob('*.png')) == paths, 'Unexpected PNG files')
    return True

def process(row, delivery, output, config, resume):
    import av
    start = time.monotonic()
    rp = record_path(output, row)
    if resume and rp.exists():
        try:
            old = json.loads(rp.read_text())
            valid_record(output, row, config, old)
            return old, True
        except (ValueError, OSError, KeyError):
            pass
    video = dict(row, original_label=row['label'], normalized_label=int(row['label']),
                 decoded_frame_count=0, extracted_frame_count=0,
                 video_extraction_status='pending', frame_extraction_status='pending')
    record = {'source_fingerprint': fingerprint(row), 'settings': config, 'video': video,
              'frames': [], 'errors': []}
    stage = 'video_extraction'
    try:
        destination = inside(output, row['organized_video_path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(digest(destination) == row['sha256'], f'Existing video differs; move it aside before retry: {destination}')
        else:
            fd, tmp = tempfile.mkstemp(prefix='.video-', dir=destination.parent)
            try:
                with os.fdopen(fd, 'wb') as dst, zipfile.ZipFile(delivery / row['archive_path']) as z:
                    with z.open(row['archive_member_path']) as src:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                require(digest(tmp) == row['sha256'], 'Extracted video hash mismatch')
                os.replace(tmp, destination)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        video['video_extraction_status'] = 'success'
        stage = 'decode_count'
        n, previous = 0, None
        with av.open(str(destination)) as container:
            require(len(container.streams.video) == 1, 'Expected one video stream')
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                require(frame.width >= 64 and frame.height >= 64 and not frame.is_corrupt, 'Invalid/corrupt decoded frame')
                ts = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
                if ts is not None:
                    require(previous is None or ts > previous, 'Non-monotonic decoded timestamps')
                    previous = ts
                n += 1
        video['decoded_frame_count'] = n
        indices = sample_indices(n)
        require(n == int(row['frame_count']), f'Decoded count {n} differs from manifest {row["frame_count"]}')
        stage = 'frame_extraction'
        frame_dir = inside(output, row['frame_directory_path'])
        frame_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.frames-', dir=frame_dir.parent) as temp:
            frames = []
            with av.open(str(destination)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                for i, frame in enumerate(container.decode(stream)):
                    if i not in indices:
                        continue
                    k = len(frames)
                    name = f'{row["filesystem_key"]}_frame_{k:03d}.png'
                    image_path = Path(temp) / name
                    im = frame.to_image().convert('RGB')
                    im.save(image_path, format='PNG', compress_level=6)
                    ts = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
                    frames.append({'clip_id': row['clip_id'], 'selected_frame_position': k,
                                   'original_frame_index': i, 'timestamp_seconds': ts,
                                   'timestamp_missing': ts is None,
                                   'relative_video_path': row['organized_video_path'],
                                   'relative_frame_path': row['frame_directory_path'] + '/' + name,
                                   'width': im.width, 'height': im.height,
                                   'sha256': digest(image_path), 'file_bytes': image_path.stat().st_size})
            require(len(frames) == 8, 'Second decode did not produce eight frames')
            # Retain old/incomplete outputs for diagnosis, never silently overwrite them.
            if frame_dir.exists():
                quarantine = output / '.quarantine'
                quarantine.mkdir(exist_ok=True)
                frame_dir.rename(quarantine / (rp.stem + '-' + str(time.time_ns())))
            os.replace(temp, frame_dir)
        record['frames'] = frames
        video['extracted_frame_count'] = 8
        video['frame_extraction_status'] = 'success'
        valid_record(output, row, config, record)
    except Exception as e:
        video['frame_extraction_status'] = 'failed'
        if video['video_extraction_status'] != 'success':
            video['video_extraction_status'] = 'failed'
        record['frames'] = []
        record['errors'] = [{'clip_id': row['clip_id'], 'stage': stage,
                             'error': f'{type(e).__name__}: {e}'}]
    record['elapsed_seconds'] = time.monotonic() - start
    atomic_json(rp, record)
    return record, False

def select(rows, args):
    chosen = [r for r in rows if (not args.split or r['split'] in args.split)
              and (not args.family or r['family'] in args.family)]
    if args.clip_ids:
        ids = set(Path(args.clip_ids).read_text().splitlines())
        require(ids <= {r['clip_id'] for r in chosen}, 'Requested clip IDs absent from selected manifest')
        chosen = [r for r in chosen if r['clip_id'] in ids]
    if args.pilot_size:
        require(not args.split or args.split == ['train'], 'Pilot uses train only')
        chosen = [r for r in chosen if r['split'] == 'train']
        groups = {}
        for r in chosen:
            groups.setdefault(r['family'], []).append(r)
        # Prefer diverse frame counts and resolutions; deterministic within each bucket.
        ordered = {}
        for family, group in groups.items():
            buckets = {}
            for r in group:
                buckets.setdefault((r['width'], r['height'], r['frame_count']), []).append(r)
            for bucket in buckets.values():
                bucket.sort(key=lambda r: fingerprint(r['clip_id']))
            ordered[family] = []
            while buckets:
                for key in sorted(list(buckets)):
                    ordered[family].append(buckets[key].pop(0))
                    if not buckets[key]:
                        del buckets[key]
        chosen = []
        while ordered and len(chosen) < args.pilot_size:
            for family in sorted(list(ordered)):
                if len(chosen) == args.pilot_size:
                    break
                chosen.append(ordered[family].pop(0))
                if not ordered[family]:
                    del ordered[family]
    require(chosen, 'Empty selection')
    return chosen

def publish(output, records):
    videos = [r['video'] for r in records]
    frames = [f for r in records for f in r['frames']]
    errors = [e for r in records for e in r['errors']]
    write_csv(output / 'indexes/videos.csv', videos, list(videos[0]))
    frame_fields = ['clip_id', 'selected_frame_position', 'original_frame_index', 'timestamp_seconds',
                    'timestamp_missing', 'relative_video_path', 'relative_frame_path', 'width', 'height', 'sha256', 'file_bytes']
    write_csv(output / 'indexes/frames.csv', frames, frame_fields)
    write_csv(output / 'indexes/extraction_errors.csv', errors, ['clip_id', 'stage', 'error'])

def run(args):
    delivery, output = args.delivery.resolve(), args.output.resolve()
    require(not output.is_relative_to(delivery) and not delivery.is_relative_to(output), 'Output and delivery must be separate trees')
    require('sources' not in output.parts, 'Do not write under sources/')
    rows = inventory(delivery)
    selected = select(rows, args)
    config = settings()
    with lock(output):
        config_path = output / 'reports/extraction_config.json'
        if config_path.exists():
            require(json.loads(config_path.read_text())['settings'] == config, 'Settings changed: use a new output root')
        request = {'settings': config, 'delivery_root': str(delivery),
                   'processed_paths_relative_to': 'processed root', 'archive_paths_relative_to': 'delivery root',
                   'requested_clip_ids': [r['clip_id'] for r in selected], 'workers': args.workers}
        atomic_json(config_path, request)
        # Conservative allocation allowance: RGB raw size plus PNG overhead and source videos.
        needed = sum(int(r['file_bytes']) + 8 * int(r['width']) * int(r['height']) * 3 * 1.02 for r in selected)
        free = shutil.disk_usage(output).free
        require(free > needed + 2 * 1024**3, f'Insufficient disk: need conservative {needed / 1e9:.2f} GB plus 2 GiB reserve; free {free / 1e9:.2f} GB')
        print(f'Preflight: {len(selected)} clips, conservative additional storage {needed / 1e9:.2f} GB', flush=True)
        records, skipped = [], 0
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, r, delivery, output, config, args.resume) for r in selected]
            for future in concurrent.futures.as_completed(futures):
                record, skip = future.result()
                records.append(record)
                skipped += skip
                if len(records) % 10 == 0:
                    print(f'{len(records)}/{len(selected)} clips processed', flush=True)
        records.sort(key=lambda r: r['video']['clip_id'])
        publish(output, records)
        successes = [r for r in records if r['video']['frame_extraction_status'] == 'success']
        png_bytes = sum(f['file_bytes'] for r in successes for f in r['frames'])
        pixels = sum(f['width'] * f['height'] for r in successes for f in r['frames'])
        ratio = png_bytes / pixels if pixels else None
        full_pixels = sum(8 * int(r['width']) * int(r['height']) for r in rows)
        full_video = sum(int(r['file_bytes']) for r in rows)
        report = {'requested': len(selected), 'successful': len(successes), 'failed': len(records)-len(successes),
                  'resumed': skipped, 'elapsed_seconds': time.monotonic()-start,
                  'png_bytes': png_bytes, 'video_bytes': sum(int(r['video']['file_bytes']) for r in records if r['video']['video_extraction_status']=='success'),
                  'pilot_png_bytes_per_pixel': ratio, 'full_video_bytes': full_video,
                  'estimated_full_png_bytes': round(full_pixels * ratio) if ratio else None,
                  'conservative_full_png_bytes': round(full_pixels * 3 * 1.02),
                  'estimate_note': 'Resolution-weighted pilot estimate; held-out source compressibility is unmeasured. Retained ZIPs already occupy disk. Reserve temporary space and do not treat estimate as an upper bound.',
                  'free_bytes_before': free, 'source_counts': {f: sum(r['family']==f for r in selected) for f in sorted({r['family'] for r in selected})}}
        atomic_json(output / 'reports/extraction_report.json', report)
        print(json.dumps(report, indent=2))
        require(not report['failed'], 'Some clips failed; inspect indexes/extraction_errors.csv')

def validate(args):
    output = args.output.resolve()
    with lock(output):
        config = json.loads((output / 'reports/extraction_config.json').read_text())
        rows = {r['clip_id']: r for r in inventory(args.delivery.resolve())}
        records = []
        for cid in config['requested_clip_ids']:
            r = rows[cid]
            record = json.loads(record_path(output, r).read_text())
            require(record['source_fingerprint'] == fingerprint(r), 'Manifest identity changed')
            if record['video']['frame_extraction_status'] == 'success':
                valid_record(output, r, config['settings'], record)
            else:
                require(record['errors'] and not record['frames'], 'Unaccounted failed clip')
            records.append(record)
        records.sort(key=lambda r: r['video']['clip_id'])
        # Compare all CSV fields to canonical committed records, including preserved metadata.
        for filename, expected in [('videos.csv', [r['video'] for r in records]),
                                   ('frames.csv', [f for r in records for f in r['frames']]),
                                   ('extraction_errors.csv', [e for r in records for e in r['errors']])]:
            actual = read_csv(output / 'indexes' / filename)
            require(len(actual) == len(expected), f'Index row count mismatch: {filename}')
            for a, e in zip(actual, expected):
                require(all(a[k] == ('' if e[k] is None else str(e[k])) for k in a), f'Index mismatch: {filename}')
        print(f'Validated {len(records)} clips and {sum(len(r["frames"]) for r in records)} frames')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('migrate')
    m.add_argument('--dev', type=Path, default=Path('dev'))
    for command in ('extract', 'validate'):
        c = sub.add_parser(command)
        c.add_argument('--delivery', type=Path, default=Path('preprocess-from Zhou'))
        c.add_argument('--output', type=Path, default=Path('processed'))
        if command == 'extract':
            c.add_argument('--workers', type=int, default=2)
            c.add_argument('--pilot-size', type=int, default=0)
            c.add_argument('--split', action='append')
            c.add_argument('--family', action='append')
            c.add_argument('--clip-ids', help='Text file: one exact clip ID per line')
            c.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.command == 'migrate':
        migrate(args.dev.resolve())
    elif args.command == 'extract':
        require(args.workers >= 1 and args.pilot_size >= 0, 'Invalid workers/pilot size')
        run(args)
    else:
        validate(args)

if __name__ == '__main__':
    main()
