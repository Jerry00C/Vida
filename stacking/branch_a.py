"""Reuse dev's frozen-feature trainer without changing its implementation."""
import copy
import hashlib
from pathlib import Path
import sys
import torch
from .core import digest, read, require, seal, template, verify, write


def imports():
    # Support `python -m stacking` from PR2; dev remains independently runnable.
    dev = str(Path(__file__).resolve().parents[1] / 'branches_dev' / 'branch_a')
    if dev not in sys.path:
        sys.path.insert(0, dev)
    from dev.features import FeatureCache
    from dev.learning import predict, train
    return FeatureCache, predict, train


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def train_scores(plan, split, cache_path, config, output, resume=False):
    verify(plan, 'plan_identity')
    verify(split, 'split_identity')
    require(plan['split_identity'] == split['split_identity'], 'Plan/split mismatch')
    FeatureCache, predict, train = imports()
    cache = FeatureCache(cache_path)
    require(cache.meta['identity']['index_fingerprints'] == split['index_fingerprints'], 'Cache/index mismatch')
    require({r['clip_id'] for r in plan['rows']} <= set(cache.lookup), 'Cache lacks required videos; smoke caches are insufficient')
    output = Path(output)
    bundle = template(plan, 'A')
    provenance = dict(plan_identity=plan['plan_identity'], cache_identity=cache.meta['cache_identity'], config=config)
    if (output/'run_identity.json').exists():
        require(resume and read(output/'run_identity.json') == provenance, 'Existing output differs or --resume missing')
    else:
        require(not output.exists() or not any(output.iterdir()), 'Output must be empty')
        write(output/'run_identity.json', provenance)
    lookup = {r['clip_id']: r for r in plan['rows']}
    for mid, spec in bundle['models'].items():
        if mid == 'final':
            training_split = split
        else:
            rows = [dict(lookup[cid], role=role) for key, role in (('fit_ids', 'train'), ('selection_ids', 'validation'))
                    for cid in spec[key]]
            training_split = seal(dict(rounds={plan['round_id']: rows},
                                       index_fingerprints=split['index_fingerprints'], smoke=split['smoke'],
                                       stacking_plan=plan['plan_identity'], fold=mid), 'split_identity')
        run = output/mid
        cp = train(cache.path, training_split, plan['round_id'], copy.deepcopy(config), run,
                   resume=resume and (run/'latest.pt').exists())
        head = torch.nn.Linear(cache.values.shape[-1], 1)
        head.load_state_dict(cp['model'])
        records = [r for r in bundle['records'] if r['model_id'] == mid]
        # Test real videos may occur in both roles; preserve both records.
        rows = [dict(lookup[r['clip_id']], role=r['partition']) for r in records]
        predictions = predict(head, cache, rows)
        for record, prediction in zip(records, predictions):
            record['score'] = float(prediction['video_logit'])
        spec['artifact_sha256'] = file_hash(run/'best.pt')
    bundle['encoder'] = cache.meta['identity']
    write(output/'scores.json', bundle)
    return bundle


def export_frozen(cache_path, checkpoint, hf_cache, output):
    FeatureCache, _, _ = imports()
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
    cache = FeatureCache(cache_path)
    meta = cache.meta['identity']
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)  # Trusted project checkpoint only.
    require(cp['cache_identity'] == cache.meta['cache_identity'], 'Checkpoint/cache mismatch')
    require(cp['backbone'] == meta['checkpoint'], 'Backbone mismatch')
    output = Path(output)
    require(not output.exists(), 'Export destination already exists')
    cls = CLIPVisionModel if meta['method'] == 'vision_post_layernorm_pooled' else AutoModel
    encoder = cls.from_pretrained(meta['checkpoint'], revision=meta['revision'], cache_dir=str(hf_cache), attn_implementation='eager')
    processor = AutoImageProcessor.from_pretrained(meta['checkpoint'], revision=meta['revision'], cache_dir=str(hf_cache),
                                                   use_fast=('dinov3' in meta['checkpoint']))
    require(digest(processor.to_dict()) == digest(meta['processor']), 'Processor differs from cached feature provenance')
    encoder.requires_grad_(False)
    encoder.eval()
    output.mkdir(parents=True)
    encoder.save_pretrained(output/'encoder', safe_serialization=True)
    processor.save_pretrained(output/'encoder')
    torch.save(cp['model'], output/'head.pt')
    write(output/'manifest.json', dict(schema='vida-frozen-export-v1', encoder_class=cls.__name__,
          checkpoint=meta['checkpoint'], revision=meta['revision'], dimension=meta['dimension'],
          feature_method=meta['method'], aggregation='sigmoid(mean(eight frame logits))',
          cache_identity=cache.meta['cache_identity'], split_identity=cp['split_identity'],
          round_id=cp['round_id'], source_checkpoint_sha256=file_hash(checkpoint),
          files={str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}))


def load_frozen(path, device='cpu'):
    """Return encoder, processor, head; explicitly restore freeze and eval policy."""
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
    path = Path(path)
    meta = read(path/'manifest.json')
    for name, expected in meta['files'].items():
        require((path/name).resolve().is_relative_to(path.resolve()), 'Unsafe export path')
        require(file_hash(path/name) == expected, 'Export checksum mismatch')
    cls = CLIPVisionModel if meta['encoder_class'] == 'CLIPVisionModel' else AutoModel
    encoder = cls.from_pretrained(path/'encoder', local_files_only=True, attn_implementation='eager').to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    processor = AutoImageProcessor.from_pretrained(path/'encoder', local_files_only=True, use_fast=('dinov3' in meta['checkpoint']))
    head = torch.nn.Linear(meta['dimension'], 1).to(device)
    head.load_state_dict(torch.load(path/'head.pt', map_location=device, weights_only=True))
    head.requires_grad_(False)
    head.eval()
    return encoder, processor, head
