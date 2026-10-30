import copy
import json
import numpy as np
import pytest
from stacking.core import aligned, make_plan, model_spec, seal, template, validate_scores, write
from stacking.fusion import evaluate, fit, probabilities


def dataset():
    rows = []
    for role, count in [('train', 48), ('validation', 12), ('test_seen', 8), ('test_unseen', 8)]:
        for i in range(count):
            rows.append(dict(clip_id=f'{role}-{i}', role=role, round_id='round_1', label=i % 2,
                             component_id=f'{role}-group-{i // 2}', source='fake' if i % 2 else 'real'))
    return seal(dict(rounds={'round_1': rows}, index_fingerprints={'synthetic': 'only'}, smoke=True), 'split_identity')


def inputs():
    split = dataset()
    plan = make_plan(split, 'round_1', folds=3)
    bundles = {}
    for i, b in enumerate(('A', 'B', 'C')):
        bundle = template(plan, b)
        for spec in bundle['models'].values():
            spec['artifact_sha256'] = str(i) * 64
        for j, r in enumerate(bundle['records']):
            r['score'] = float(2*r['label'] - 1 + np.sin(j+i))
        bundles[b] = bundle
    return split, plan, bundles


def test_group_folds_and_real_only_contract():
    split, plan, bundles = inputs()
    assert plan == make_plan(split, 'round_1', folds=3)
    lookup = {r['clip_id']: r for r in plan['rows']}
    held = []
    for mid, spec in plan['folds'].items():
        sets = [{lookup[c]['component_id'] for c in spec[k]} for k in ('fit_ids', 'selection_ids', 'predict_ids')]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        assert all(lookup[c]['label'] == 0 for c in model_spec(plan, 'C', mid)['fit_ids'])
        held.extend(spec['predict_ids'])
    assert len(held) == len(set(held)) == 48
    baseline = aligned(plan, bundles)
    bundles['B']['records'].reverse()
    np.testing.assert_array_equal(baseline['train'][1], aligned(plan, bundles)['train'][1])


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'nonfinite', 'fold', 'label', 'fit_leak', 'selection_leak', 'fake_c', 'plan'])
def test_reject_invalid_inputs(problem):
    _, plan, bundles = inputs()
    b = bundles['C']
    r = next(r for r in b['records'] if r['partition'] == 'train')
    if problem == 'duplicate': b['records'].append(dict(r))
    if problem == 'missing': b['records'].pop()
    if problem == 'nonfinite': r['score'] = float('nan')
    if problem == 'fold': r['model_id'] = 'final'
    if problem == 'label': r['label'] = 1-r['label']
    if problem == 'fit_leak': b['models'][r['model_id']]['fit_ids'].append(r['clip_id'])
    if problem == 'selection_leak': b['models'][r['model_id']]['selection_ids'].append(r['clip_id'])
    if problem == 'fake_c': b['models']['final']['fit_ids'].append('train-1')
    if problem == 'plan': b['plan_identity'] = 'different'
    with pytest.raises(ValueError): validate_scores(plan, b, 'C')


def test_fusion_reload_and_test_isolation(tmp_path):
    _, plan, bundles = inputs()
    model = fit(plan, bundles, tmp_path/'fusion.json')
    loaded = json.loads((tmp_path/'fusion.json').read_text())
    x = aligned(plan, bundles)['validation'][1]
    np.testing.assert_array_equal(probabilities(model, x), probabilities(loaded, x))
    assert set(evaluate(plan, bundles, loaded, tmp_path/'evaluation.json')['partitions']) == {'test_seen', 'test_unseen'}
    altered = copy.deepcopy(bundles)
    for b in altered.values():
        for r in b['records']:
            if r['partition'].startswith('test'): r['score'] += 1000
    other = fit(plan, altered, tmp_path/'other.json')
    for k in ('coef', 'intercept', 'mean', 'scale', 'C'): assert model[k] == other[k]
    with pytest.raises(ValueError, match='Scores changed'):
        evaluate(plan, altered, model, tmp_path/'bad.json')


def test_branch_a_oof_end_to_end(tmp_path):
    from stacking.branch_a import train_scores
    from stacking.core import digest
    split, plan, _ = inputs()
    cache = tmp_path/'cache'; cache.mkdir()
    ids = sorted({r['clip_id'] for r in plan['rows']})
    values = np.random.default_rng(42).normal(size=(len(ids), 8, 4)).astype('float32')
    np.save(cache/'embeddings.npy', values)
    from stacking.branch_a import file_hash
    signature = dict(clips=ids, dimension=4, dtype='float32', checkpoint='synthetic', index_fingerprints=split['index_fingerprints'])
    write(cache/'metadata.json', dict(identity=signature, cache_identity=digest(signature), array_sha256=file_hash(cache/'embeddings.npy')))
    config = dict(lr=.001, weight_decay=.0001, batch_size=16, epochs=1, patience=5, min_delta=.0001,
                  scheduler_factor=.5, scheduler_patience=2, balanced_classes=True, equal_generators=True, seed=42)
    result = train_scores(plan, split, cache, config, tmp_path/'branch-a')
    validate_scores(plan, result, 'A')
    resumed = train_scores(plan, split, cache, config, tmp_path/'branch-a', resume=True)
    assert result == resumed


def test_frozen_export_roundtrip(tmp_path):
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor, AutoImageProcessor
    from stacking.branch_a import export_frozen, load_frozen, file_hash
    from stacking.core import digest
    source = tmp_path/'pretrained'
    encoder = CLIPVisionModel(CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                               num_attention_heads=2, image_size=32, patch_size=16))
    encoder.eval(); encoder.save_pretrained(source)
    processor = CLIPImageProcessor(size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32})
    processor.save_pretrained(source)
    processor = AutoImageProcessor.from_pretrained(source, use_fast=False)
    cache = tmp_path/'cache'; cache.mkdir()
    np.save(cache/'embeddings.npy', np.zeros((1, 8, 16), dtype='float32'))
    meta = dict(checkpoint=str(source), revision='local-test', processor=processor.to_dict(),
                method='vision_post_layernorm_pooled', dimension=16, clips=['x'], dtype='float32')
    write(cache/'metadata.json', dict(identity=meta, cache_identity=digest(meta), array_sha256=file_hash(cache/'embeddings.npy')))
    head = torch.nn.Linear(16, 1)
    checkpoint = tmp_path/'head.pt'
    torch.save(dict(model=head.state_dict(), backbone=str(source), cache_identity=digest(meta), split_identity='test', round_id='round_1'), checkpoint)
    export_frozen(cache, checkpoint, tmp_path/'hf', tmp_path/'export')
    loaded, _, restored_head = load_frozen(tmp_path/'export')
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(head(encoder(x).pooler_output), restored_head(loaded(x).pooler_output))
    with (tmp_path/'export/head.pt').open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_frozen(tmp_path/'export')
