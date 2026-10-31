"""Content-addressed descriptor cache; schema and preprocessing are part of the key."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from .config import FeatureConfig, require_sha256
from .features import FEATURE_NAMES, FeatureResult, extract_features, feature_signature
from .video import read_video

CACHE_VERSION = "descriptor-cache-v2"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_features(path, config: FeatureConfig, cache_dir, *, expected_sha256=None) -> tuple[FeatureResult, str]:
    if expected_sha256 is not None:
        require_sha256("expected_sha256", expected_sha256)
    content_hash = file_sha256(path)
    if expected_sha256 is not None and content_hash != expected_sha256:
        raise ValueError(f"Input content differs from frozen manifest; no video decoded: {path}")
    signature = feature_signature(config)
    # The cache format can change independently of numerical model features.
    cache_path = Path(cache_dir) / f"{CACHE_VERSION}-{content_hash}-{signature}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            required = {"cache_version", "signature", "names", "clip", "windows", "bounds", "metadata"}
            if not required <= set(data.files):
                raise ValueError(f"Cache is missing required fields: {cache_path}")
            if (str(data["cache_version"].item()) != CACHE_VERSION
                    or str(data["signature"].item()) != signature or tuple(data["names"]) != FEATURE_NAMES):
                raise ValueError(f"Cache schema mismatch: {cache_path}")
            result = FeatureResult(data["clip"], data["windows"], data["bounds"], json.loads(str(data["metadata"].item())))
            if not isinstance(result.metadata, dict):
                raise ValueError("Corrupt feature cache metadata")
            if not all(np.issubdtype(x.dtype, np.floating) for x in
                       (result.clip, result.windows, result.bounds_seconds)):
                raise ValueError("Corrupt feature cache numeric types")
            if result.clip.shape != (len(FEATURE_NAMES),) or result.windows.ndim != 2 or result.windows.shape[1] != len(FEATURE_NAMES):
                raise ValueError("Corrupt feature cache shape")
            if len(result.windows) == 0 or result.bounds_seconds.shape != (len(result.windows), 2):
                raise ValueError("Corrupt feature cache window bounds")
            if np.any(result.bounds_seconds[:, 1] <= result.bounds_seconds[:, 0]):
                raise ValueError("Corrupt feature cache window intervals")
            if not all(np.isfinite(x).all() for x in [result.clip, result.windows, result.bounds_seconds]):
                raise ValueError("Corrupt feature cache values")
            return result, content_hash
    result = extract_features(read_video(path, config), config)
    if file_sha256(path) != content_hash:
        raise ValueError(f"Input changed during feature extraction; cache was not written: {path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Independent runs can prepare the same content concurrently. Each writer
        # gets its own staging file; readers only see a complete atomic replace.
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, clip=result.clip, windows=result.windows, bounds=result.bounds_seconds,
                                names=FEATURE_NAMES, signature=signature, cache_version=CACHE_VERSION,
                                metadata=json.dumps(result.metadata))
        temporary.replace(cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result, content_hash
