# -*- coding: utf-8 -*-
"""Branch B 'temporal_wavelet' Inference Package.

This package provides standalone, modular inference for Branch B winning architecture
'temporal_wavelet' (113 temporal wavelet features + LightGBM booster).

Public Functions & Classes:
- load_detector(model_path=None, verify_sha256=True) -> BranchBDetector
- extract_features(video_path, ...) -> dict[str, float]
- BranchBDetector: Main detector class for video or feature inference.
- ORDERED_FEATURE_NAMES: The exact 113 feature column names in frozen order.
- SCHEMA_HASH: Cryptographic schema hash of the ordered features.
"""
from __future__ import annotations

from pathlib import Path

from .detector import (
    BranchBDetector,
    DetectorIntegrityError,
)
from .extract import (
    ORDERED_FEATURE_NAMES,
    SCHEMA_HASH,
    ExtractionError,
    decode_canonical_video,
    extract_features,
    extract_features_from_decoded,
)

__version__ = "3.0.0"

__all__ = [
    "BranchBDetector",
    "DetectorIntegrityError",
    "ExtractionError",
    "ORDERED_FEATURE_NAMES",
    "SCHEMA_HASH",
    "decode_canonical_video",
    "extract_features",
    "extract_features_from_decoded",
    "load_detector",
]


def load_detector(
    model_path: str | Path | None = None,
    verify_sha256: bool = True,
) -> BranchBDetector:
    """Convenience factory function to load and validate the BranchBDetector bundle.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the bundle joblib file. If None, defaults to the deliverable model
        at 'deliverables/branch_b_temporal_wavelet/model/branch_b_temporal_wavelet_full.joblib'.
    verify_sha256 : bool, default True
        Verify the checksum against the required local adjacent trusted manifest.
        False is an UNSAFE legacy checksum-only opt-out; schema and trained state
        validation still apply. Joblib/pickle is executable content: trust BOTH
        manifest and binary. Checksums do not authenticate arbitrary pickle.

    Returns
    -------
    BranchBDetector
        Instantiated and validated detector ready for inference.
    """
    return BranchBDetector(model_path=model_path, verify_sha256=verify_sha256)
