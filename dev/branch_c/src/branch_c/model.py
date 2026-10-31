from dataclasses import asdict
from pathlib import Path
import platform
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import ModelConfig
from .features import FEATURE_NAMES, feature_signature


class RealOnlyAnomalyModel:
    """Both scaling and estimator fitting are restricted to real rows."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def fit(self, X, labels, clip_ids):
        X, labels, clip_ids = np.asarray(X, dtype=float), np.asarray(labels), np.asarray(clip_ids)
        if (X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES)
                or labels.shape != (len(X),) or clip_ids.shape != (len(X),)):
            raise ValueError("Training shape/ID mismatch")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be 0 or 1")
        real = labels == 0
        if real.sum() < 4:
            raise ValueError("At least 4 real training descriptors are required")
        if not np.isfinite(X[real]).all():
            raise ValueError("Real training features must be finite")
        if self.config.algorithm == "isolation_forest":
            estimator = IsolationForest(n_estimators=self.config.n_estimators,
                max_samples=min(self.config.max_samples, int(real.sum())), contamination="auto",
                random_state=self.config.seed, n_jobs=1)
        else:
            estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=self.config.nu)
        self.pipeline = make_pipeline(StandardScaler(), estimator)
        self.pipeline.fit(X[real])
        self.fit_clip_ids = sorted(set(clip_ids[real].tolist()))
        self.fit_rows = int(real.sum())
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X).all():
            raise ValueError("Scoring features must be finite [N,37]")
        # sklearn score_samples has smaller values for more abnormal observations.
        return -self.pipeline.score_samples(X)

    def deviations(self, x, top_k=5):
        z = self.pipeline[0].transform(np.asarray(x).reshape(1, -1))[0]
        indices = np.argsort(-np.abs(z), kind="stable")[:top_k]
        return [{"feature": FEATURE_NAMES[i], "standardized_deviation": float(z[i])} for i in indices]

    def metadata(self):
        return {"config": asdict(self.config), "fit_rows": self.fit_rows,
                "fit_clip_ids": self.fit_clip_ids, "fit_labels": [0],
                "score_direction": "higher_is_more_anomalous", "is_probability": False}


def runtime_versions():
    import cv2
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "opencv": cv2.__version__, "joblib": joblib.__version__}


def save_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)


def load_bundle(path):
    # joblib uses pickle. Load only your own or your team's trusted artifacts.
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1 or tuple(bundle.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Incompatible model bundle")
    if bundle["versions"] != runtime_versions():
        raise ValueError("Runtime versions differ from fitted artifact; use its requirements lock or refit")
    if bundle["feature_signature"] != feature_signature(bundle["config"].features):
        raise ValueError("Feature implementation/configuration differs from fitted artifact; refit")
    return bundle
