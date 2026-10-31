from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path


def require_number(name, value, *, lower=None, upper=None):
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside the allowed range")


def require_integer(name, value, *, lower=0, upper=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside the allowed range")


def require_sha256(name, value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def _tags(name, values):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of names, not a string")
    if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty names without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicate names")
    return tuple(values)


@dataclass(frozen=True)
class FeatureConfig:
    sample_fps: float = 8.0
    image_size: int = 96
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 20.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    sampling_mode: str = "uniform_cfr"

    def __post_init__(self):
        require_integer("features.image_size", self.image_size, lower=24)
        for name, value in asdict(self).items():
            if name not in {"image_size", "sampling_mode"}:
                require_number(f"features.{name}", value)
        if self.sampling_mode not in {"uniform_cfr", "native_stride"}:
            raise ValueError("sampling_mode must be uniform_cfr or native_stride")
        if not 0 < self.sample_fps <= 240:
            raise ValueError("sample_fps must be in (0,240]")
        if not 0 < self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("Invalid duration range")
        if self.window_seconds * self.sample_fps < 8:
            raise ValueError("Each window needs at least 8 frames for temporal frequency")
        if not 0 < self.stride_seconds <= self.window_seconds:
            raise ValueError("stride_seconds must be in (0, window_seconds]")
        if self.sampling_mode == "uniform_cfr" and self.window_seconds > self.min_duration_seconds:
            raise ValueError("window_seconds cannot exceed min_duration_seconds")
        for name in ("window_seconds", "stride_seconds"):
            frame_count = getattr(self, name) * self.sample_fps
            if not math.isclose(frame_count, round(frame_count), abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"{name} must correspond to a whole number of sampled frames")


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str = "isolation_forest"
    n_estimators: int = 200
    max_samples: int = 256
    nu: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.algorithm, str) or self.algorithm not in {"isolation_forest", "one_class_svm"}:
            raise ValueError("Unsupported algorithm")
        require_integer("model.n_estimators", self.n_estimators, lower=1)
        require_integer("model.max_samples", self.max_samples, lower=2)
        require_integer("model.seed", self.seed, upper=2**32 - 1)
        require_number("model.nu", self.nu)
        if not 0 < self.nu < 1:
            raise ValueError("model.nu must be in (0,1)")


@dataclass(frozen=True)
class EvaluationConfig:
    real_false_positive_rate: float = 0.05
    bootstrap_repeats: int = 1000
    seed: int = 42

    def __post_init__(self):
        require_number("evaluation.real_false_positive_rate", self.real_false_positive_rate)
        require_integer("evaluation.bootstrap_repeats", self.bootstrap_repeats)
        require_integer("evaluation.seed", self.seed, upper=2**32 - 1)
        if not 0 < self.real_false_positive_rate < 1:
            raise ValueError("evaluation.real_false_positive_rate must be in (0,1)")


@dataclass(frozen=True)
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    heldout_generators: tuple[str, ...] = ()
    heldout_real_sources: tuple[str, ...] = ()

    def __post_init__(self):
        for name, kind in (("features", FeatureConfig), ("model", ModelConfig), ("evaluation", EvaluationConfig)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be a {kind.__name__}")
        for name in ("heldout_generators", "heldout_real_sources"):
            object.__setattr__(self, name, _tags(name, getattr(self, name)))

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    allowed = {"features", "model", "evaluation", "heldout_generators", "heldout_real_sources"}
    if set(data) - allowed:
        raise ValueError(f"Unknown config keys: {set(data) - allowed}")
    sections = {}
    for name, kind in (("features", FeatureConfig), ("model", ModelConfig), ("evaluation", EvaluationConfig)):
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a JSON object")
        unknown = set(section) - {f.name for f in fields(kind)}
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        sections[name] = kind(**section)
    return Config(**sections, heldout_generators=_tags("heldout_generators", data.get("heldout_generators", [])),
                  heldout_real_sources=_tags("heldout_real_sources", data.get("heldout_real_sources", [])))


def fingerprint(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
