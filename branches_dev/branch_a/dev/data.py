"""Read-only indexed PNG data and a persisted full eligibility audit."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from .common import check, identity, index_identity, read_csv, read_json, resolve, save_csv, save_json, sha


def load_indexes(root):
    videos = read_csv(Path(root) / 'indexes/videos.csv')
    check(len({v['clip_id'] for v in videos}) == len(videos), 'Duplicate video ID')
    frames = defaultdict(list)
    for f in read_csv(Path(root) / 'indexes/frames.csv'):
        frames[f['clip_id']].append(f)
    check(set(frames) <= {v['clip_id'] for v in videos}, 'Orphan frame rows')
    for fs in frames.values():
        fs.sort(key=lambda f: int(f['selected_frame_position']))
    return {v['clip_id']: v for v in videos}, dict(frames)


def audit(root, output, workers=8):
    videos, frames = load_indexes(root)
    paths = [f['relative_frame_path'] for fs in frames.values() for f in fs]
    check(len(paths) == len(set(paths)), 'Repeated frame paths across clips')
    def inspect(v):
        cid = v['clip_id']
        try:
            check(v['label'] in ('0', '1') and v['normalized_label'] == v['label'], 'Invalid labels')
            check(v['video_extraction_status'] == v['frame_extraction_status'] == 'success', 'Extraction unsuccessful')
            fs = frames.get(cid, [])
            check(len(fs) == 8 and [int(f['selected_frame_position']) for f in fs] == list(range(8)), 'Not eight aligned frames')
            check(len({f['original_frame_index'] for f in fs}) == 8, 'Repeated selected index')
            for f in fs:
                p = resolve(root, f['relative_frame_path'])
                check(sha(p) == f['sha256'], 'Frame hash differs from original index')
                with Image.open(p) as im:
                    im.load()
                    check(im.mode == 'RGB' and im.format == 'PNG', 'Unreadable/non-RGB PNG')
                    check(im.size == (int(f['width']), int(f['height'])), 'Dimension mismatch')
            return cid, None
        except Exception as e:
            return cid, str(e)
    eligible, excluded = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (cid, error) in enumerate(pool.map(inspect, videos.values()), 1):
            if error:
                excluded.append({'clip_id': cid, 'reason': error})
            else:
                eligible.append(cid)
            if n % 1000 == 0:
                print(f'Audited {n}/{len(videos)} videos', flush=True)
    result = {'index_fingerprints': index_identity(root), 'eligible': sorted(eligible),
              'excluded': excluded, 'policy': 'eight fully decoded RGB PNGs, indexed hashes and linkage validated'}
    result['audit_identity'] = identity(result)
    save_json(output, result)
    save_csv(Path(output).with_suffix('.exclusions.csv'), excluded, ['clip_id', 'reason'])
    print(f'Audit: {len(eligible)} eligible, {len(excluded)} excluded', flush=True)
    return result


def audited(root, path):
    result = read_json(path)
    check(result['index_fingerprints'] == index_identity(root), 'Indexes changed; rerun audit')
    expected = dict(result); expected.pop('audit_identity')
    check(identity(expected) == result['audit_identity'], 'Audit identity mismatch')
    return result


class FrameDataset:
    """One image per item; feature extraction always enumerates all eight frames."""
    def __init__(self, root, clips, videos, frames):
        self.root, self.clips, self.videos, self.frames = root, clips, videos, frames
    def __len__(self):
        return len(self.clips) * 8
    def __getitem__(self, index):
        cid = self.clips[index // 8]
        f = self.frames[cid][index % 8]
        p = resolve(self.root, f['relative_frame_path'])
        check(sha(p) == f['sha256'], f'PNG changed since audit: {cid}')
        with Image.open(p) as im:
            im.load(); check(im.mode == 'RGB', 'Expected RGB'); image = im.copy()
        return {'image': image, 'clip_id': cid, 'label': int(self.videos[cid]['label']),
                'source': self.videos[cid]['family'], 'frame_position': index % 8,
                'frame_id': f['relative_frame_path']}


class ProcessorCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, items):
        inputs = self.processor(images=[i['image'] for i in items], return_tensors='pt')
        return {'inputs': inputs, 'items': [{k:v for k,v in i.items() if k != 'image'} for i in items]}
