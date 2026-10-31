"""Shared fold plan and strict cross-branch score contract."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

BRANCHES = ('A', 'B', 'C')
ROLES = ('train', 'validation', 'test_seen', 'test_unseen')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seal(value, field):
    return dict(value, **{field: digest(value)})


def verify(value, field):
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            f'Invalid {field}')


def make_plan(split, round_id, folds=5, seed=42):
    verify(split, 'split_identity')
    rows = sorted(split['rounds'][round_id], key=lambda r: (r['role'], r['clip_id']))
    require(all(r['role'] in ROLES and r['label'] in (0, 1) for r in rows), 'Invalid split rows')
    keys = [(r['role'], r['clip_id']) for r in rows]
    require(len(set(keys)) == len(keys), 'Duplicate split rows')
    partitions = {role: [r for r in rows if r['role'] == role] for role in ROLES}
    require(all(partitions.values()), 'All four partitions are required')
    groups = {role: {r['component_id'] for r in rs} for role, rs in partitions.items()}
    require(not groups['train'] & (groups['validation'] | groups['test_seen'] | groups['test_unseen']),
            'Training group leakage')
    require(not groups['validation'] & (groups['test_seen'] | groups['test_unseen']), 'Validation group leakage')
    # Real test videos may intentionally occur in both test roles.
    identities = {}
    for r in rows:
        value = (r['label'], r['component_id'], r['source'])
        require(r['clip_id'] not in identities or identities[r['clip_id']] == value, 'Conflicting clip identity')
        identities[r['clip_id']] = value
    train = partitions['train']
    y = np.array([r['label'] for r in train])
    g = np.array([r['component_id'] for r in train])
    require(folds >= 2 and len(set(g)) >= folds, 'Insufficient groups for folds')
    definitions = {}
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for fold, (development, heldout) in enumerate(splitter.split(np.zeros(len(y)), y, g)):
        require(set(y[heldout]) == {0, 1}, 'OOF fold needs both classes; reduce fold count or change seed')
        # Internal early stopping must never use the OOF prediction fold.
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed + fold + 1)
        candidate = None
        for fit, select in inner.split(np.zeros(len(development)), y[development], g[development]):
            if set(y[development[fit]]) == set(y[development[select]]) == {0, 1}:
                candidate = (development[fit], development[select])
                break
        require(candidate is not None, 'Cannot make two-class inner split; provide more training groups')
        fit, select = candidate
        definitions[str(fold)] = {
            'fit_ids': sorted(train[i]['clip_id'] for i in fit),
            'selection_ids': sorted(train[i]['clip_id'] for i in select),
            'predict_ids': sorted(train[i]['clip_id'] for i in heldout),
        }
    plan = dict(schema='vida-stacking-plan-v1', split_identity=split['split_identity'],
                round_id=round_id, seed=seed, rows=rows, folds=definitions)
    return seal(plan, 'plan_identity')


def model_spec(plan, branch, model_id):
    lookup = {r['clip_id']: r for r in plan['rows']}
    if model_id == 'final':
        fit = [r['clip_id'] for r in plan['rows'] if r['role'] == 'train']
        selection = [r['clip_id'] for r in plan['rows'] if r['role'] == 'validation']
    else:
        fold = plan['folds'][model_id]
        fit, selection = fold['fit_ids'], fold['selection_ids']
    if branch == 'C':
        fit = [cid for cid in fit if lookup[cid]['label'] == 0]
    return {'fit_ids': sorted(fit), 'selection_ids': sorted(selection)}


def template(plan, branch):
    verify(plan, 'plan_identity')
    require(branch in BRANCHES, 'Unknown branch')
    fold_for = {cid: fold for fold, spec in plan['folds'].items() for cid in spec['predict_ids']}
    models = {mid: dict(model_spec(plan, branch, mid), artifact_sha256=None)
              for mid in [*plan['folds'], 'final']}
    records = [dict(clip_id=r['clip_id'], partition=r['role'], label=r['label'],
                    model_id=fold_for[r['clip_id']] if r['role'] == 'train' else 'final', score=None)
               for r in plan['rows']]
    return dict(schema='vida-stacking-scores-v1', branch=branch,
                plan_identity=plan['plan_identity'], split_identity=plan['split_identity'],
                round_id=plan['round_id'], score_kind='anomaly' if branch == 'C' else 'logit',
                higher_is_generated=True, models=models, records=records)


def validate_scores(plan, bundle, branch):
    expected = template(plan, branch)
    for key in ('schema', 'branch', 'plan_identity', 'split_identity', 'round_id', 'score_kind', 'higher_is_generated'):
        require(bundle.get(key) == expected[key], f'{branch}: mismatched {key}')
    require(set(bundle['models']) == set(expected['models']), f'{branch}: incorrect model set')
    for mid, spec in expected['models'].items():
        actual = bundle['models'][mid]
        for key in ('fit_ids', 'selection_ids'):
            require(sorted(actual[key]) == spec[key], f'{branch}/{mid}: invalid {key} (leakage or missing provenance)')
        h = actual.get('artifact_sha256')
        require(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h),
                f'{branch}/{mid}: provide artifact SHA-256')
    records = bundle['records']
    keyed = {(r['partition'], r['clip_id']): r for r in records}
    required = {(r['partition'], r['clip_id']): r for r in expected['records']}
    require(len(keyed) == len(records), f'{branch}: duplicate predictions')
    require(set(keyed) == set(required), f'{branch}: missing or unexpected predictions')
    for key, r in keyed.items():
        e = required[key]
        require(r['label'] == e['label'] and r['model_id'] == e['model_id'], f'{branch}: label/fold mismatch')
        require(type(r['score']) in (float, int) and np.isfinite(r['score']), f'{branch}: score must be finite')
    return keyed


def aligned(plan, bundles):
    require(set(bundles) == set(BRANCHES), 'Exactly branches A, B, C required')
    tables = {b: validate_scores(plan, bundles[b], b) for b in BRANCHES}
    output = {}
    for role in ROLES:
        rows = sorted((r for r in plan['rows'] if r['role'] == role), key=lambda r: r['clip_id'])
        x = np.array([[tables[b][(role, r['clip_id'])]['score'] for b in BRANCHES] for r in rows])
        output[role] = (rows, x, np.array([r['label'] for r in rows]))
    return output
