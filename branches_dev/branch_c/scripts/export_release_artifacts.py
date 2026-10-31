"""Export trusted trained Branch C models and scores without media or private paths."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np

from branch_c.evaluation import metrics
from branch_c.manifest import GROUP_KEYS
from branch_c.model import load_bundle


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def portable_rows(rows):
    # Retain the fields used by the existing ID/group leakage guard, not input manifests.
    return [{'clip_id': row['clip_id'], 'path': 'clip-id:' + row['clip_id'],
             'metadata': {key: row['metadata'].get(key, '')
                          for key in (*GROUP_KEYS, 'protocol')}} for row in rows]


def reject_private_paths(value):
    if isinstance(value, str):
        if any(token in value for token in ('/Users/', '/var/folders/', 'wxid_', 'pan.baidu.com')):
            raise ValueError('Private local path or data-sharing link in export')
    elif isinstance(value, dict):
        for item in value.values():
            reject_private_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_private_paths(item)


def export_model(source, destination, expected_sha256):
    if digest(source) != expected_sha256:
        raise ValueError('Frozen source model changed')
    original = load_bundle(source)
    exported = dict(original)
    for key in ('fit_manifest_rows', 'validation_manifest_rows'):
        exported[key] = portable_rows(original[key])
    exported['artifact_export'] = {
        'kind': 'portable_metadata_only', 'source_model_sha256': expected_sha256,
        'note': 'Fitted clip/window estimators, scaler, threshold and model_id are unchanged. '
                'Private input paths and unused manifest fields were replaced by an ID/group '
                'leakage index; original path-only matching is not portable. IDs, known groups '
                'and training/validation content hashes remain available to the evaluation guard.'}
    reject_private_paths(exported)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(exported, destination, compress=3)
    restored = load_bundle(destination)
    for key in ('clip_model', 'window_model'):
        if joblib.hash(original[key]) != joblib.hash(restored[key]):
            raise ValueError('Fitted estimator changed during export')
        scaler = original[key].pipeline[0]
        probes = scaler.mean_[None, :] + np.linspace(-3, 3, 13)[:, None] * scaler.scale_[None, :]
        np.testing.assert_array_equal(original[key].score(probes), restored[key].score(probes))
    for key in ('model_id', 'threshold', 'feature_signature', 'versions'):
        if original[key] != restored[key]:
            raise ValueError('Prediction identity or preprocessing changed')
    return {'model_id': original['model_id'], 'threshold': original['threshold'],
            'source_model_sha256': expected_sha256, 'export_model_sha256': digest(destination),
            'bytes': destination.stat().st_size, 'estimators_identical': True,
            'probe_predictions_identical': True}


def copy_scores(source, destination, protocol, kind, model_id=None):
    ids = set()
    count = 0
    labels, scores = [], []
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if any('path' in name.lower() for name in reader.fieldnames):
            raise ValueError('Prediction CSV must not contain input paths')
        for row in reader:
            if row['clip_id'] in ids or row['protocol'] != protocol or row['score_kind'] != kind:
                raise ValueError('Duplicate ID or mixed score protocol/kind')
            ids.add(row['clip_id'])
            reject_private_paths(row)
            if row['label'] not in {'0', '1'} or row['is_probability'] != 'false':
                raise ValueError('Invalid score label or probability claim')
            score = float(row['branch_c_score'])
            if not np.isfinite(score):
                raise ValueError('Nonfinite prediction')
            if kind == 'out_of_fold' and row['fold_id'] not in {'0', '1', '2', '3', '4'}:
                raise ValueError('OOF row has no common fold')
            if model_id and row['model_id'] != model_id:
                raise ValueError('Prediction uses a different final model')
            labels.append(int(row['label']))
            scores.append(score)
            count += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise ValueError('Prediction bytes changed')
    return count, labels, scores


def export(project_root, output_dir):
    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError('Choose a new output directory; existing artifacts are preserved')
    package = output_dir / 'branch_c_trained_artifacts'
    package.mkdir(parents=True)
    latest = project_root / 'artifacts/vida_guard32k/rotation_6seen_2unseen_v1'
    old = project_root / 'artifacts/vida_guard32k/run_02'
    freeze = json.loads((latest / 'model_freeze.json').read_text())
    models, score_files, score_rows = {}, 0, 0
    prefix = Path('rotation_6seen_2unseen_v1')
    for candidate, source_sha in freeze['models_sha256'].items():
        relative = prefix / 'shared_models' / candidate / 'model.joblib'
        models[str(relative)] = export_model(latest / 'shared_models' / candidate / 'model.joblib',
                                            package / relative, source_sha)
        audit = json.loads((latest / 'shared_models' / candidate / 'oof_audit.json').read_text())
        for fold in audit['folds']:
            name = f"fold_{fold['fold_id']}.joblib"
            source = latest / 'shared_models' / candidate / 'oof_models' / name
            if digest(source) != fold['model_sha256']:
                raise ValueError('Frozen OOF estimator changed')
            destination = package / prefix / 'shared_models' / candidate / 'oof_models' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for round_number in range(1, 29):
        name = f'round_{round_number:02d}'
        reference = json.loads((latest / name / 'model_reference.json').read_text())
        chosen = reference['selected_candidate']
        model = models[str(prefix / 'shared_models' / chosen / 'model.joblib')]
        reference['source_model_sha256'] = reference['model_sha256']
        reference['model_sha256'] = model['export_model_sha256']
        reference.pop('oof_shared_audit', None)
        reference['export_note'] = 'Portable metadata export; original model_id and all predictions unchanged.'
        write_json(package / prefix / name / 'model_reference.json', reference)
        shutil.copyfile(latest / name / 'selected_config.json',
                        package / prefix / name / 'selected_config.json')
        for relative, kind in [('oof_scores.csv', 'out_of_fold'),
                               ('validation_scores.csv', 'final_model'),
                               ('seen_generators/test_scores.csv', 'final_model'),
                               ('unseen_generators/test_scores.csv', 'final_model')]:
            count, y, values = copy_scores(latest / name / relative, package / prefix / name / relative,
                                           reference['protocol'], kind,
                                           reference['model_id'] if kind == 'final_model' else None)
            score_files += 1
            score_rows += count
            if relative.endswith('test_scores.csv'):
                report = json.loads((latest / name / Path(relative).parent / 'test_report.json').read_text())
                if report['metrics'] != metrics(y, values, reference['threshold']):
                    raise ValueError('Exported test scores do not reproduce reported metrics')
                filtered = {k: report[k] for k in ('protocol', 'model_id', 'threshold', 'metrics',
                            'ci_95', 'per_generator', 'per_generator_ci_95') if k in report}
                write_json(package / prefix / name / Path(relative).parent / 'test_report.json', filtered)
    old_relative = Path('historical_run_02/model.joblib')
    old_model = export_model(old / 'selection/selected/model.joblib', package / old_relative,
                            '4d67ad3ae1d6adeec2882430569616571559329948c72389e45a50015b028730')
    models[str(old_relative)] = old_model
    for relative in ['oof/oof_scores.csv', 'selection/selected/validation_scores.csv',
                     'in_domain_test/test_scores.csv', 'cross_test/test_scores.csv']:
        kind = 'out_of_fold' if relative.startswith('oof/') else 'final_model'
        count, y, values = copy_scores(old / relative, package / 'historical_run_02' / relative,
                                       'vida_guard32k_v1', kind,
                                       old_model['model_id'] if kind == 'final_model' else None)
        score_files += 1
        score_rows += count
        if relative.endswith('test_scores.csv'):
            report = json.loads((old / Path(relative).parent / 'test_report.json').read_text())
            if report['metrics'] != metrics(y, values, old_model['threshold']):
                raise ValueError('Historical test metrics changed')
    for relative in ['requirements-lock.txt', 'requirements-report.txt']:
        shutil.copyfile(project_root / relative, package / relative)
    info = {'kind': 'trained_artifacts_without_raw_data', 'models': models,
            'final_models': 3, 'auxiliary_oof_models': 10, 'score_files': score_files,
            'score_rows': score_rows, 'all_score_files_byte_identical': True,
            'all_final_estimators_and_thresholds_identical': True,
            'test_metric_views_reproduced': 58, 'raw_media_included': False,
            'input_manifests_included': False, 'feature_caches_included': False,
            'source_models_preserved': True,
            'limitations': ['Previously evaluated cohort; not a fresh blind test',
                            'Dependent 28 rounds; do not treat score rows as independent clips',
                            'Current A/B/C split protocols must be aligned before fusion']}
    write_json(package / 'artifact_manifest.json', info)
    (package / 'README.md').write_text('''# Branch C trained models and prediction scores

This package contains trained artifacts, not original videos or extracted frames.
The completed experiments are preserved: no fitting, threshold selection or test tuning occurs during export.

## Latest 28-round experiment

Use rotation_6seen_2unseen_v1/round_XX/model_reference.json to select the actual shared final model and its fixed threshold. Two final models serve the 28 rounds because their real-only fitting inputs are identical for each selected parameter setting. Ten auxiliary fold estimators are supplied separately for OOF verification.

Each round includes oof_scores.csv, validation_scores.csv, seen_generators/test_scores.csv and unseen_generators/test_scores.csv. Join by clip_id within the same protocol/round, retain fold_id, and use validation_role=calibration for fusion calibration. OOF scores intentionally have no final model_id because they come from fold-specific estimators. Scores are anomaly evidence, not probabilities; larger means more anomalous. Apply the strict score > threshold rule.

## Historical experiment

historical_run_02 contains the earlier fixed 4-seen/4-unseen model and its OOF, validation, in-domain and cross-test scores. Do not mix these with the 28-round protocol.

## Load the final model

Install the source package from the Branch C directory of PR #1 and use its pinned Python 3.12.14 requirements. Predict with the existing branch-c predict command, pointing --model to the model.joblib selected by the round reference and --video to your own local video. Models require the matching source classes and exact runtime versions; only load trusted joblib files.

The export retains all learned estimators, scaler parameters, thresholds, feature signatures and model_id values. Embedded private file paths and unused input-manifest fields are replaced by a minimal clip-ID/group leakage index. Content-hash, ID and known-group guards remain; matching original machine paths is not portable. File SHA256 values therefore differ from the original model bundle; both original and exported hashes are recorded. Score CSV bytes are unchanged. No videos, video frames, full input manifests, feature caches or candidate-search models are included.

Check artifact_manifest.json for the numerical export audit and checksums.json for every packaged file. The rounds overlap and reuse an already evaluated cohort; this is not a new blind test. A/B/C split alignment and fusion ablation remain team work.
''', encoding='utf-8')
    files = sorted(p for p in package.rglob('*') if p.is_file())
    inventory = {}
    for p in files:
        if p.suffix not in {'.joblib', '.json', '.csv', '.txt', '.md'}:
            raise ValueError('Unexpected file type in artifact release')
        if p.suffix in {'.json', '.md', '.txt'}:
            reject_private_paths(p.read_text())
        inventory[str(p.relative_to(package))] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    write_json(package / 'checksums.json', inventory)
    archive = output_dir / 'branch_c_models_and_scores.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                handle.write(p, p.relative_to(output_dir))
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError('Corrupt release archive')
        for name, item in inventory.items():
            if hashlib.sha256(handle.read('branch_c_trained_artifacts/' + name)).hexdigest() != item['sha256']:
                raise ValueError('Archive member hash changed')
    checksum = digest(archive)
    (output_dir / 'SHA256SUMS.txt').write_text(f'{checksum}  {archive.name}\n')
    result = {'status': 'passed', 'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': checksum, 'members': len(files) + 1,
              'final_models': 3, 'oof_models': 10, 'score_files': score_files, 'score_rows': score_rows}
    write_json(output_dir / 'export_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    export(args.project_root, args.output_dir)
