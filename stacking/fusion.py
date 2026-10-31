"""Train on aligned OOF scores, select on validation, evaluate tests separately."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from .core import BRANCHES, aligned, digest, require, seal, verify, write


def probabilities(model, x):
    z = ((x - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']) + model['intercept']
    return np.exp(-np.logaddexp(0, -z))


def summary(y, p):
    return dict(n=len(y), auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                f1=float(f1_score(y, p >= .5, zero_division=0)), threshold=.5)


def fit(plan, bundles, output, penalties=(.01, .1, 1., 10.)):
    data = aligned(plan, bundles)
    _, x, y = data['train']
    _, vx, vy = data['validation']
    require(set(y) == set(vy) == {0, 1}, 'Training and validation need both classes')
    require(len(penalties) > 0 and all(np.isfinite(c) and c > 0 for c in penalties), 'Invalid C candidates')
    scaler = StandardScaler().fit(x)  # Never fit scaling on validation or tests.
    candidates = []
    best = None
    for c in penalties:
        lr = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=plan['seed']).fit(scaler.transform(x), y)
        require(lr.n_iter_[0] < 2000, 'Fusion solver did not converge')
        state = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                     coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), C=float(c))
        metrics = summary(vy, probabilities(state, vx))
        candidates.append(dict(C=float(c), validation=metrics))
        rank = (metrics['auroc'], -metrics['log_loss'])
        if best is None or rank > best[0]:
            best = (rank, state)
    model = dict(schema='vida-fusion-v1', plan_identity=plan['plan_identity'],
                 round_id=plan['round_id'], branches=list(BRANCHES),
                 branch_models={b: bundles[b]['models'] for b in BRANCHES},
                 score_kinds={b: bundles[b]['score_kind'] for b in BRANCHES},
                 input_bundle_hashes={b: digest(bundles[b]) for b in BRANCHES},
                 selection='validation AUROC, then validation log loss',
                 candidates=candidates, threshold=.5, **best[1])
    model = seal(model, 'model_identity')
    write(output, model)
    return model


def evaluate(plan, bundles, model, output):
    verify(model, 'model_identity')
    require(model['plan_identity'] == plan['plan_identity'], 'Fusion/plan mismatch')
    require(model['branches'] == list(BRANCHES), 'Feature order mismatch')
    require(model['branch_models'] == {b: bundles[b]['models'] for b in BRANCHES}, 'Branch model changed since fusion fit')
    require(model['input_bundle_hashes'] == {b: digest(bundles[b]) for b in BRANCHES}, 'Scores changed since fusion fit')
    data = aligned(plan, bundles)
    result = dict(model_identity=model['model_identity'], partitions={})
    for role in ('test_seen', 'test_unseen'):
        rows, x, y = data[role]
        p = probabilities(model, x)
        result['partitions'][role] = dict(metrics=summary(y, p), predictions=[
            dict(clip_id=r['clip_id'], label=r['label'], probability=float(score))
            for r, score in zip(rows, p)])
    write(output, result)
    return result
