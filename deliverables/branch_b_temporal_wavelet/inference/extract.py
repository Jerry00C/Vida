# -*- coding: utf-8 -*-
"""Standalone Canonical Feature Extractor for Branch B 'temporal_wavelet' (113 features).

Authority & Specifications:
- Architecture: 'temporal_wavelet' (winning Branch B configuration).
- Feature count: Exactly 113 numerical features across 3 groups:
    * 'tfft_v2': 17 temporal FFT features over uniform spectral grid.
    * 'local_temporal_v2': 54 motion-compensated residual, kinematics, and anchor drift features.
    * 'twavelet_v2': 42 1D temporal wavelet features on spatial patches (gray, residual, edge).
- Canonical Time Policy: 'sidecar_independent_index_fps_v1'
    * Never reads query sidecars, external annotations, or external timing caches.
    * Timestamps strictly derived from frame index and native container FPS: t_k = k / fps.
    * Fallback FPS: 25.0 if container FPS is missing or non-positive.
- Decoded resolutions:
    * 'gray128': 128x128 grayscale in float32 [0.0, 1.0].
    * 'gray_flow': 96x96 grayscale in float32 [0.0, 1.0].
    * 'grid': linearly interpolated 128x128 uniform-time spectral grid, max 128 points.
    * Up to 64 native observations selected with np.rint (not integer truncation).

Standalone port of the gen_933b135d79 training path: program_v2.video.decode_canonical
plus v2.video_v2 preprocessing/grid and v2.features summary descriptors. No optional
repository imports: extraction is identical whether the development tree exists or not.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pywt
from scipy.fft import fft2, rfft
from scipy.signal.windows import get_window

# Pinned 113 feature column names in frozen order
ORDERED_FEATURE_NAMES: list[str] = [
    # --- Group 1: tfft_v2 (17 features) ---
    "tfft_v2__ac_centroid",
    "tfft_v2__ac_entropy",
    "tfft_v2__ac_flatness",
    "tfft_v2__ac_rolloff85",
    "tfft_v2__ac_spread",
    "tfft_v2__block_high_cv",
    "tfft_v2__block_high_iqr",
    "tfft_v2__cont_high_share",
    "tfft_v2__cont_low_share",
    "tfft_v2__cont_mid_share",
    "tfft_v2__high_rel_power",
    "tfft_v2__low_rel_power",
    "tfft_v2__mid_rel_power",
    "tfft_v2__phys_band0_share",
    "tfft_v2__phys_band1_share",
    "tfft_v2__phys_band2_share",
    "tfft_v2__phys_psd_total",
    # --- Group 2: local_temporal_v2 (54 features) ---
    "local_temporal_v2__acc_mean",
    "local_temporal_v2__acc_p90",
    "local_temporal_v2__acc_std",
    "local_temporal_v2__acc_vel_ratio",
    "local_temporal_v2__anchor_edge_end_start_ratio",
    "local_temporal_v2__anchor_edge_max_dev",
    "local_temporal_v2__anchor_edge_slope",
    "local_temporal_v2__anchor_hf_end_start_ratio",
    "local_temporal_v2__anchor_hf_max_dev",
    "local_temporal_v2__anchor_hf_slope",
    "local_temporal_v2__anchor_lapvar_end_start_ratio",
    "local_temporal_v2__anchor_lapvar_max_dev",
    "local_temporal_v2__anchor_lapvar_slope",
    "local_temporal_v2__anchor_rad_max_dev",
    "local_temporal_v2__anchor_rad_slope",
    "local_temporal_v2__cum_to_net_drift_ratio",
    "local_temporal_v2__dir_coherence_mean",
    "local_temporal_v2__dir_coherence_std",
    "local_temporal_v2__fb_consistency_mean",
    "local_temporal_v2__fb_consistency_p90",
    "local_temporal_v2__fb_consistency_std",
    "local_temporal_v2__fg_bg_residual_ratio",
    "local_temporal_v2__jerk_max",
    "local_temporal_v2__jerk_mean",
    "local_temporal_v2__patch8_res_cv",
    "local_temporal_v2__patch8_res_p90_p50",
    "local_temporal_v2__patch8_res_p95_p50",
    "local_temporal_v2__patch8_res_std",
    "local_temporal_v2__reg4_res_cv",
    "local_temporal_v2__reg4_res_p90_p50",
    "local_temporal_v2__reg4_res_p95_p50",
    "local_temporal_v2__reg4_res_std",
    "local_temporal_v2__res_raw_ratio_edge",
    "local_temporal_v2__res_raw_ratio_gray",
    "local_temporal_v2__res_raw_ratio_lap",
    "local_temporal_v2__segment_res_late_minus_early",
    "local_temporal_v2__segment_res_max_min_ratio",
    "local_temporal_v2__segment_res_std",
    "local_temporal_v2__temporal_stability_mean",
    "local_temporal_v2__temporal_stability_std",
    "local_temporal_v2__vel_mean",
    "local_temporal_v2__vel_p90",
    "local_temporal_v2__vel_std",
    "local_temporal_v2__warp_res_bwd_mean",
    "local_temporal_v2__warp_res_edge_mean",
    "local_temporal_v2__warp_res_edge_std",
    "local_temporal_v2__warp_res_gray_mean",
    "local_temporal_v2__warp_res_gray_p90",
    "local_temporal_v2__warp_res_gray_std",
    "local_temporal_v2__warp_res_lap_mean",
    "local_temporal_v2__warp_res_lap_std",
    "local_temporal_v2__win4_res_iqr",
    "local_temporal_v2__win4_res_p10",
    "local_temporal_v2__win4_res_p90",
    # --- Group 3: twavelet_v2 (42 features) ---
    "twavelet_v2__edge_approx_rel",
    "twavelet_v2__edge_d1_rel",
    "twavelet_v2__edge_d2_rel",
    "twavelet_v2__edge_detail_iqr",
    "twavelet_v2__edge_detail_kurt",
    "twavelet_v2__edge_detail_p90",
    "twavelet_v2__edge_detail_p90_p10",
    "twavelet_v2__edge_detail_p99_p50",
    "twavelet_v2__edge_detail_skew",
    "twavelet_v2__edge_detail_spatial_cv",
    "twavelet_v2__edge_detail_spatial_std",
    "twavelet_v2__edge_logE_approx",
    "twavelet_v2__edge_logE_d1",
    "twavelet_v2__edge_logE_d2",
    "twavelet_v2__gray_approx_rel",
    "twavelet_v2__gray_d1_rel",
    "twavelet_v2__gray_d2_rel",
    "twavelet_v2__gray_detail_iqr",
    "twavelet_v2__gray_detail_kurt",
    "twavelet_v2__gray_detail_p90",
    "twavelet_v2__gray_detail_p90_p10",
    "twavelet_v2__gray_detail_p99_p50",
    "twavelet_v2__gray_detail_skew",
    "twavelet_v2__gray_detail_spatial_cv",
    "twavelet_v2__gray_detail_spatial_std",
    "twavelet_v2__gray_logE_approx",
    "twavelet_v2__gray_logE_d1",
    "twavelet_v2__gray_logE_d2",
    "twavelet_v2__res_approx_rel",
    "twavelet_v2__res_d1_rel",
    "twavelet_v2__res_d2_rel",
    "twavelet_v2__res_detail_iqr",
    "twavelet_v2__res_detail_kurt",
    "twavelet_v2__res_detail_p90",
    "twavelet_v2__res_detail_p90_p10",
    "twavelet_v2__res_detail_p99_p50",
    "twavelet_v2__res_detail_skew",
    "twavelet_v2__res_detail_spatial_cv",
    "twavelet_v2__res_detail_spatial_std",
    "twavelet_v2__res_logE_approx",
    "twavelet_v2__res_logE_d1",
    "twavelet_v2__res_logE_d2",
]

SCHEMA_HASH: str = "593d44101435a4f3a888ed2f588fe6698f6f19242f8fb3d4c3165a8a12750c4d"
EPS: float = 1e-12


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Ensure value is a finite Python float."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _slope_per_second(values: np.ndarray, times: np.ndarray) -> float:
    """Least-squares slope of values against times in seconds."""
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if v.size < 2 or t.size < 2:
        return 0.0
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom <= 0.0:
        return 0.0
    return _safe_float(((v - v.mean()) @ tc) / denom)


def _moments(values: np.ndarray) -> tuple[float, float]:
    """Kurtosis (Fisher) and skewness with safe degenerates."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size < 4:
        return 0.0, 0.0
    mean = flat.mean()
    std = flat.std()
    if std <= 1e-12:
        return 0.0, 0.0
    centered = (flat - mean) / std
    kurtosis = float(np.mean(centered**4) - 3.0)
    skewness = float(np.mean(centered**3))
    return _safe_float(kurtosis), _safe_float(skewness)


class ExtractionError(ValueError):
    """Fail-closed exception raised when video decoding or feature extraction fails."""
    pass


def decode_canonical_video(
    video_path: str | Path,
    t_max: int = 64,
    min_sampled_frames: int = 8,
    fps_fallback: float = 25.0,
) -> dict[str, Any]:
    """Decodes video under canonical-time policy 'sidecar_independent_index_fps_v1'.

    Parameters
    ----------
    video_path : str or Path
        Path to the media file.
    t_max : int, default 64
        Maximum number of frames to uniformly sample.
    min_sampled_frames : int, default 8
        Minimum number of valid frames required to succeed.
    fps_fallback : float, default 25.0
        Fallback frame rate if container metadata is missing or non-positive.

    Returns
    -------
    dict
        Decoded arrays and metadata:
        - 'gray128': [T, 128, 128] float32 in [0, 1]
        - 'gray_flow': [T, 96, 96] float32 in [0, 1]
        - 'grid': [G, 128, 128] float32 uniform-time grid in [0, 1]
        - 'grid_times', 'frame_indices': canonical spectral times / native indices
        - 'times': [T] float64 timestamps in seconds
        - 'dt_grid': float sampling period
        - 'fs_grid': float sampling frequency
        - 'fps': float container FPS
        - 'n_frames': int total container frames
    """
    cv2.setNumThreads(1)  # canonical decoder's deterministic OpenCV policy
    v_path = Path(video_path)
    if not v_path.is_file():
        raise ExtractionError(f"Video file not found: {v_path}")
    if v_path.stat().st_size == 0:
        raise ExtractionError(f"Video file is empty (0 bytes): {v_path}")

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise ExtractionError(f"OpenCV failed to open video file: {v_path}")

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if np.isfinite(raw_fps) and raw_fps > 0 else float(fps_fallback)
        if not np.isfinite(fps) or fps <= 0 or t_max < 1 or min_sampled_frames < 1:
            raise ExtractionError("Invalid decoding parameters")
        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        # OpenCV's demuxer-reported codec FOURCC. Unknown identifiers abstain in
        # detector QC; do not infer codec from extension or install ffprobe.
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_tag = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower()
        codec = {"h264": "h264", "avc1": "h264", "x264": "h264",
                 "hevc": "hevc", "hvc1": "hevc", "hev1": "hevc",
                 "vp90": "vp9", "av01": "av1", "mp4v": "mpeg4",
                 "fmp4": "mpeg4", "xvid": "mpeg4", "divx": "mpeg4"}.get(codec_tag, "unknown")
        if total_frames <= 0:
            total_frames = 0
            while cap.grab():
                total_frames += 1
            if total_frames <= 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise ExtractionError("Empty or unseekable video")
        selected = np.unique(np.rint(np.linspace(0, total_frames - 1, min(total_frames, t_max))).astype(np.int64))
        selected_set = set(selected.tolist())
        got, gray128_list, flow_list = [], [], []
        index = 0
        while index <= int(selected[-1]):
            if index in selected_set:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                # Training resizes uint8 BEFORE scaling, and derives flow from
                # the resized uint8 analysis frame, not directly from the source.
                gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g128_u8 = cv2.resize(gray_u8, (128, 128), interpolation=cv2.INTER_AREA)
                gflow_u8 = cv2.resize(g128_u8, (96, 96), interpolation=cv2.INTER_AREA)
                gray128_list.append(g128_u8.astype(np.float32) * np.float32(1.0 / 255.0))
                flow_list.append(gflow_u8.astype(np.float32) * np.float32(1.0 / 255.0))
                got.append(index)
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()
    if len(got) < min_sampled_frames:
        raise ExtractionError(f"Decoded only {len(got)} frames; at least {min_sampled_frames} required")
    indices = np.asarray(got, dtype=np.int64)
    times = indices.astype(np.float64) / fps
    gray128 = np.stack(gray128_list).astype(np.float32)
    gray_flow = np.stack(flow_list).astype(np.float32)
    native_fps = float(1.0 / np.median(np.diff(times))) if len(times) > 1 else fps
    grid, grid_times, dt, fs = _build_spectral_grid(gray128, times, native_fps)
    return {"gray128": gray128, "gray_flow": gray_flow, "grid": grid,
            "times": times, "grid_times": grid_times, "frame_indices": indices,
            "dt_grid": dt, "fs_grid": fs, "fps": fps, "n_frames": total_frames,
            "duration_seconds": float(total_frames) / fps,
            "decode_truncated": len(got) != len(selected), "codec": codec, "codec_tag": codec_tag}


def _build_spectral_grid(gray: np.ndarray, times: np.ndarray, native_fps: float) -> tuple:
    """Exact v2.video_v2 uniform time-grid/interpolation semantics (including round)."""
    if len(times) < 2:
        dt = 1.0 / max(native_fps, 4.0)
        return gray.copy(), times.copy(), dt, 1.0 / dt
    duration = float(times[-1] - times[0])
    target_fps = float(np.clip(max(native_fps, 4.0), 4.0, 10.0))
    n_grid = int(min(max(int(round(duration * target_fps)) + 1, 2), 128))
    grid_times = np.linspace(times[0], times[-1], n_grid)
    idx = np.searchsorted(times, grid_times, side="right") - 1
    idx = np.clip(idx, 0, gray.shape[0] - 2)
    denom = times[idx + 1] - times[idx]
    w = np.where(denom > 0, (grid_times - times[idx]) / np.where(denom > 0, denom, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    grid = ((1.0 - w)[:, None, None] * gray[idx] + w[:, None, None] * gray[idx + 1]).astype(np.float32)
    dt = float((grid_times[-1] - grid_times[0]) / (n_grid - 1))
    return grid, grid_times, dt, float(1.0 / dt)


def _compute_tfft_v2(grid: np.ndarray, dt_grid: float, fs_grid: float) -> dict[str, float]:
    """Computes the 17 tfft_v2 spectral features over uniform spectral grid."""
    out: dict[str, float] = {}
    samples = grid.shape[0]

    if samples < 4:
        for name in (
            "ac_centroid", "ac_entropy", "ac_flatness", "ac_rolloff85", "ac_spread",
            "block_high_cv", "block_high_iqr", "cont_high_share", "cont_low_share",
            "cont_mid_share", "high_rel_power", "low_rel_power", "mid_rel_power",
            "phys_band0_share", "phys_band1_share", "phys_band2_share", "phys_psd_total",
        ):
            out[name] = 0.0
        return out

    pixels = grid.reshape(samples, -1).astype(np.float64)
    time_steps = np.arange(samples, dtype=np.float64)
    t_c = time_steps - time_steps.mean()
    denom = float((t_c * t_c).sum())
    if denom > 0.0:
        slope = (pixels * t_c[:, None]).sum(axis=0) / denom
        intercept = pixels.mean(axis=0) - slope * time_steps.mean()
        residual = pixels - (intercept + slope * time_steps[:, None])
    else:
        residual = pixels - pixels.mean(axis=0, keepdims=True)

    window = get_window("hann", samples, fftbins=True).astype(np.float64)
    win_sum = float(window.sum())
    mu_w = (residual * window[:, None]).sum(axis=0, keepdims=True) / (win_sum + EPS)
    residual = residual - mu_w

    u_win = float((window * window).sum())
    fs = float(fs_grid) if fs_grid > 0 else 1.0

    spectrum = rfft(residual * window[:, None], axis=0)
    k_bins = spectrum.shape[0]

    c = np.full(k_bins, 2.0, dtype=np.float64)
    c[0] = 1.0
    if samples % 2 == 0 and k_bins > 1:
        c[-1] = 1.0
    power = (c[:, None] * (spectrum.real**2 + spectrum.imag**2)) / (u_win * fs)

    if k_bins < 2:
        for name in (
            "ac_centroid", "ac_entropy", "ac_flatness", "ac_rolloff85", "ac_spread",
            "block_high_cv", "block_high_iqr", "cont_high_share", "cont_low_share",
            "cont_mid_share", "high_rel_power", "low_rel_power", "mid_rel_power",
            "phys_band0_share", "phys_band1_share", "phys_band2_share", "phys_psd_total",
        ):
            out[name] = 0.0
        return out

    ac_power = power[1:]
    ac_mean_spectrum = ac_power.mean(axis=1)
    total_ac = float(ac_mean_spectrum.sum())
    ac_dist = ac_mean_spectrum / (total_ac + EPS)

    ac_k = np.arange(1, k_bins, dtype=np.float64)
    nyquist_bins = 0.5 * samples
    nu = ac_k / nyquist_bins

    mask_low = nu <= 0.25
    mask_mid = (nu > 0.25) & (nu <= 0.55)
    mask_high = nu > 0.55
    out["low_rel_power"] = _safe_float(ac_dist[mask_low].sum() if mask_low.any() else 0.0)
    out["mid_rel_power"] = _safe_float(ac_dist[mask_mid].sum() if mask_mid.any() else 0.0)
    out["high_rel_power"] = _safe_float(ac_dist[mask_high].sum() if mask_high.any() else 0.0)

    dense_nu = np.linspace(0.005, 1.0, 200)
    dense_dist = np.interp(dense_nu, np.concatenate(([0.0], nu)), np.concatenate(([0.0], ac_dist)))
    dense_total = float(dense_dist.sum()) + EPS
    d_low = dense_nu <= 0.25
    d_mid = (dense_nu > 0.25) & (dense_nu <= 0.55)
    d_high = dense_nu > 0.55
    out["cont_low_share"] = _safe_float(dense_dist[d_low].sum() / dense_total)
    out["cont_mid_share"] = _safe_float(dense_dist[d_mid].sum() / dense_total)
    out["cont_high_share"] = _safe_float(dense_dist[d_high].sum() / dense_total)

    centroid = float((nu * ac_dist).sum())
    spread = float(np.sqrt(np.maximum(0.0, ((nu - centroid) ** 2 * ac_dist).sum())))
    out["ac_centroid"] = _safe_float(centroid)
    out["ac_spread"] = _safe_float(spread)

    cum_dist = np.cumsum(ac_dist)
    idx_roll = np.searchsorted(cum_dist, 0.85)
    if idx_roll == 0:
        roll_val = float(nu[0])
    elif idx_roll >= len(nu):
        roll_val = float(nu[-1])
    else:
        c0, c1 = cum_dist[idx_roll - 1], cum_dist[idx_roll]
        n0, n1 = nu[idx_roll - 1], nu[idx_roll]
        frac = (0.85 - c0) / max(c1 - c0, EPS)
        roll_val = float(n0 + frac * (n1 - n0))
    out["ac_rolloff85"] = _safe_float(roll_val)

    geom = float(np.exp(np.mean(np.log(np.maximum(ac_dist, EPS)))))
    arith = float(np.mean(ac_dist))
    out["ac_flatness"] = _safe_float(geom / (arith + EPS))

    ent = -float(np.sum(np.where(ac_dist > 0, ac_dist * np.log2(np.maximum(ac_dist, EPS)), 0.0)))
    max_ent = np.log2(max(len(ac_dist), 2))
    out["ac_entropy"] = _safe_float(ent / max_ent)

    # 32x32 block high-band CV & IQR
    side = int(round(np.sqrt(pixels.shape[1])))
    if side == 128 and ac_power.shape[1] == 128 * 128:
        pixel_tot_ac = ac_power.sum(axis=0) + EPS
        pixel_high_ac = ac_power[mask_high].sum(axis=0) if mask_high.any() else np.zeros(128 * 128)
        pixel_ratio = (pixel_high_ac / pixel_tot_ac).reshape(32, 4, 32, 4).mean(axis=(1, 3))
        p_std = float(pixel_ratio.std())
        p_mean = float(pixel_ratio.mean())
        out["block_high_cv"] = _safe_float(p_std / (p_mean + EPS))
        out["block_high_iqr"] = _safe_float(
            float(np.percentile(pixel_ratio, 75) - np.percentile(pixel_ratio, 25))
        )
    else:
        out["block_high_cv"] = 0.0
        out["block_high_iqr"] = 0.0

    # Physical frequency bands: [0.25, 1.0], [1.0, 2.5], [2.5, 5.0]
    freq_hz = ac_k * (fs / samples)
    f_min = float(freq_hz[0])
    f_max = float(freq_hz[-1])

    phys_bands = [(0.25, 1.0), (1.0, 2.5), (2.5, 5.0)]
    all_phys_mask = np.zeros(len(freq_hz), dtype=bool)
    band_masks = []
    for b_idx, (b_low, b_high) in enumerate(phys_bands):
        overlap = max(0.0, min(b_high, f_max) - max(b_low, f_min))
        coverage = overlap / (b_high - b_low)
        m = (freq_hz >= b_low) & (freq_hz <= b_high)
        band_masks.append((coverage, m))
        all_phys_mask |= m

    total_phys_power = float(ac_mean_spectrum[all_phys_mask].sum()) if all_phys_mask.any() else 0.0
    out["phys_psd_total"] = _safe_float(total_phys_power)

    for b_idx, (cov, m) in enumerate(band_masks):
        if cov > 0.0 and m.any() and total_phys_power > 0.0:
            share = float(ac_mean_spectrum[m].sum()) / total_phys_power
            out[f"phys_band{b_idx}_share"] = _safe_float(share)
        else:
            out[f"phys_band{b_idx}_share"] = 0.0

    return out


def _compute_1d_twavelet_on_patches(
    patch_series: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Computes 1D DWT (db2, level=2) on detrended patch temporal signals."""
    out: dict[str, float] = {}
    samples, n_patches = patch_series.shape
    if samples < 4:
        for k in (
            "approx_rel", "d2_rel", "d1_rel",
            "logE_approx", "logE_d2", "logE_d1",
            "detail_spatial_std", "detail_spatial_cv",
            "detail_kurt", "detail_skew",
            "detail_p90", "detail_p99_p50",
            "detail_iqr", "detail_p90_p10",
        ):
            out[f"{prefix}_{k}"] = 0.0
        return out

    time_steps = np.arange(samples, dtype=np.float64)
    tc = time_steps - time_steps.mean()
    denom = float((tc * tc).sum())
    if denom > 0:
        slopes = (patch_series * tc[:, None]).sum(axis=0) / denom
        intercepts = patch_series.mean(axis=0) - slopes * time_steps.mean()
        detrended = patch_series - (intercepts + slopes * time_steps[:, None])
    else:
        detrended = patch_series - patch_series.mean(axis=0, keepdims=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coeffs = pywt.wavedec(detrended, "db2", level=2, mode="periodization", axis=0)

    ca2, cd2, cd1 = coeffs

    e_a2 = (ca2**2).sum(axis=0)
    e_d2 = (cd2**2).sum(axis=0)
    e_d1 = (cd1**2).sum(axis=0)
    e_tot = e_a2 + e_d2 + e_d1 + EPS
    e_detail = e_d2 + e_d1

    out[f"{prefix}_approx_rel"] = _safe_float(float(np.mean(e_a2 / e_tot)))
    out[f"{prefix}_d2_rel"] = _safe_float(float(np.mean(e_d2 / e_tot)))
    out[f"{prefix}_d1_rel"] = _safe_float(float(np.mean(e_d1 / e_tot)))

    out[f"{prefix}_logE_approx"] = _safe_float(float(np.log10(np.mean(ca2**2) + EPS)))
    out[f"{prefix}_logE_d2"] = _safe_float(float(np.log10(np.mean(cd2**2) + EPS)))
    out[f"{prefix}_logE_d1"] = _safe_float(float(np.log10(np.mean(cd1**2) + EPS)))

    d_std = float(e_detail.std())
    d_mean = float(e_detail.mean())
    out[f"{prefix}_detail_spatial_std"] = _safe_float(d_std)
    out[f"{prefix}_detail_spatial_cv"] = _safe_float(d_std / (d_mean + EPS))

    pooled_detail = np.concatenate([cd2.reshape(-1), cd1.reshape(-1)])
    kurt, skew = _moments(pooled_detail)
    out[f"{prefix}_detail_kurt"] = kurt
    out[f"{prefix}_detail_skew"] = skew

    abs_det = np.abs(pooled_detail)
    p50 = float(np.percentile(abs_det, 50))
    p90 = float(np.percentile(abs_det, 90))
    p99 = float(np.percentile(abs_det, 99))
    p10 = float(np.percentile(abs_det, 10))
    p25 = float(np.percentile(abs_det, 25))
    p75 = float(np.percentile(abs_det, 75))

    out[f"{prefix}_detail_p90"] = _safe_float(p90)
    out[f"{prefix}_detail_p99_p50"] = _safe_float(p99 / (p50 + EPS))
    out[f"{prefix}_detail_iqr"] = _safe_float(p75 - p25)
    out[f"{prefix}_detail_p90_p10"] = _safe_float(p90 - p10)

    return out


def _compute_twavelet_v2(gray128: np.ndarray, gray_flow_residuals: np.ndarray) -> dict[str, float]:
    """Computes the 42 twavelet_v2 features on gray, residual, and edge patches."""
    out: dict[str, float] = {}
    t_frames = gray128.shape[0]

    # 1. Gray patch series [T, 256] from 128x128 (16x16 grid of 8x8 cells)
    gray_patches = gray128.reshape(t_frames, 16, 8, 16, 8).mean(axis=(2, 4)).reshape(t_frames, 256)
    out.update(_compute_1d_twavelet_on_patches(gray_patches, "gray"))

    # 2. Residual patch series [P, 256] from 96x96 (16x16 grid of 6x6 cells)
    p_count = gray_flow_residuals.shape[0]
    if p_count >= 4 and gray_flow_residuals.shape[1] == 96:
        res_patches = gray_flow_residuals.reshape(p_count, 16, 6, 16, 6).mean(axis=(2, 4)).reshape(p_count, 256)
        out.update(_compute_1d_twavelet_on_patches(res_patches, "res"))
    else:
        out.update(_compute_1d_twavelet_on_patches(np.zeros((0, 256)), "res"))

    # 3. Edge patch series [T, 256]
    edge_frames = np.zeros_like(gray128)
    for t in range(t_frames):
        g = np.ascontiguousarray(gray128[t], dtype=np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edge_frames[t] = np.sqrt(gx * gx + gy * gy)
    edge_patches = edge_frames.reshape(t_frames, 16, 8, 16, 8).mean(axis=(2, 4)).reshape(t_frames, 256)
    out.update(_compute_1d_twavelet_on_patches(edge_patches, "edge"))

    return out


def _compute_local_temporal_v2(
    gray128: np.ndarray,
    gray_flow: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Computes the 54 local_temporal_v2 motion, residual, and anchor drift features."""
    out: dict[str, float] = {}
    t_count = gray_flow.shape[0]
    size = gray_flow.shape[1]
    pairs = max(0, t_count - 1)

    if pairs == 0:
        empty = np.zeros((0, size, size), dtype=np.float32)
        for k in (
            "warp_res_gray_mean", "warp_res_gray_std", "warp_res_gray_p90",
            "warp_res_edge_mean", "warp_res_edge_std", "warp_res_lap_mean", "warp_res_lap_std",
            "warp_res_bwd_mean", "res_raw_ratio_gray", "res_raw_ratio_edge", "res_raw_ratio_lap",
            "fb_consistency_mean", "fb_consistency_std", "fb_consistency_p90",
            "reg4_res_std", "reg4_res_cv", "reg4_res_p90_p50", "reg4_res_p95_p50",
            "patch8_res_std", "patch8_res_cv", "patch8_res_p90_p50", "patch8_res_p95_p50",
            "fg_bg_residual_ratio", "vel_mean", "vel_std", "vel_p90",
            "acc_mean", "acc_std", "acc_p90", "jerk_mean", "jerk_max", "acc_vel_ratio",
            "dir_coherence_mean", "dir_coherence_std", "temporal_stability_mean", "temporal_stability_std",
            "segment_res_std", "segment_res_max_min_ratio", "segment_res_late_minus_early",
            "win4_res_p90", "win4_res_p10", "win4_res_iqr",
            "anchor_lapvar_slope", "anchor_lapvar_end_start_ratio", "anchor_lapvar_max_dev",
            "anchor_edge_slope", "anchor_edge_end_start_ratio", "anchor_edge_max_dev",
            "anchor_hf_slope", "anchor_hf_end_start_ratio", "anchor_hf_max_dev",
            "anchor_rad_slope", "anchor_rad_max_dev", "cum_to_net_drift_ratio",
        ):
            out[k] = 0.0
        return out, empty

    grid_x, grid_y = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )

    gray_res = np.zeros((pairs, size, size), dtype=np.float32)
    gray_raw = np.zeros((pairs, size, size), dtype=np.float32)
    edge_res = np.zeros(pairs, dtype=np.float64)
    edge_raw = np.zeros(pairs, dtype=np.float64)
    lap_res = np.zeros(pairs, dtype=np.float64)
    lap_raw = np.zeros(pairs, dtype=np.float64)
    bwd_res = np.zeros(pairs, dtype=np.float64)
    fb_cons = np.zeros(pairs, dtype=np.float64)
    flow_mags = np.zeros((pairs, size, size), dtype=np.float32)
    mean_dirs = np.zeros((pairs, 2), dtype=np.float64)
    dir_coherence = np.zeros(pairs, dtype=np.float64)

    reg4_stds, reg4_cvs, reg4_p90_p50s, reg4_p95_p50s = [], [], [], []
    patch8_stds, patch8_cvs, patch8_p90_p50s, patch8_p95_p50s = [], [], [], []
    fg_bg_ratios = []

    dt = np.diff(np.asarray(times, dtype=np.float64))
    dt_safe = np.where(dt > 0, dt, 1.0 / 25.0)
    vel_fields = np.zeros((pairs, size, size, 2), dtype=np.float32)

    for t in range(pairs):
        f0 = np.ascontiguousarray(gray_flow[t], dtype=np.float32)
        f1 = np.ascontiguousarray(gray_flow[t + 1], dtype=np.float32)
        cur_dt = float(dt_safe[t])

        fwd_flow = cv2.calcOpticalFlowFarneback(
            f0 * 255.0, f1 * 255.0, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        bwd_flow = cv2.calcOpticalFlowFarneback(
            f1 * 255.0, f0 * 255.0, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(fwd_flow[..., 0] ** 2 + fwd_flow[..., 1] ** 2)
        flow_mags[t] = mag
        vel_fields[t] = fwd_flow / max(cur_dt, 1e-4)

        unit_flow = fwd_flow / (mag[..., None] + EPS)
        mean_d = unit_flow.mean(axis=(0, 1))
        mean_dirs[t] = mean_d
        dir_coherence[t] = float(np.sqrt(mean_d[0] ** 2 + mean_d[1] ** 2))

        map_x_fwd = grid_x + fwd_flow[..., 0]
        map_y_fwd = grid_y + fwd_flow[..., 1]
        warped_gray = cv2.remap(
            f1, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        map_x_bwd = grid_x + bwd_flow[..., 0]
        map_y_bwd = grid_y + bwd_flow[..., 1]
        warped_bwd = cv2.remap(
            f0, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        bwd_warped = cv2.remap(
            bwd_flow, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        fb_err = np.sqrt(
            (fwd_flow[..., 0] + bwd_warped[..., 0]) ** 2
            + (fwd_flow[..., 1] + bwd_warped[..., 1]) ** 2
        )

        in_bounds = (
            (map_x_fwd >= 0.0)
            & (map_x_fwd <= size - 1.0)
            & (map_y_fwd >= 0.0)
            & (map_y_fwd <= size - 1.0)
        )
        valid_mask = in_bounds & (fb_err <= 1.5)
        if not np.any(valid_mask):
            valid_mask = in_bounds if np.any(in_bounds) else np.ones((size, size), dtype=bool)

        fb_cons[t] = float(np.mean(fb_err[in_bounds])) if np.any(in_bounds) else float(np.mean(fb_err))

        gx0 = cv2.Sobel(f0, cv2.CV_32F, 1, 0, ksize=3)
        gy0 = cv2.Sobel(f0, cv2.CV_32F, 0, 1, ksize=3)
        e0 = np.sqrt(gx0 * gx0 + gy0 * gy0)

        gx1 = cv2.Sobel(f1, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(f1, cv2.CV_32F, 0, 1, ksize=3)
        e1 = np.sqrt(gx1 * gx1 + gy1 * gy1)

        warped_edge = cv2.remap(
            e1, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        l0 = np.abs(cv2.Laplacian(f0, cv2.CV_32F, ksize=3))
        l1 = np.abs(cv2.Laplacian(f1, cv2.CV_32F, ksize=3))
        warped_lap = cv2.remap(
            l1, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        r_gray = np.abs(f0 - warped_gray)
        d_gray = np.abs(f1 - f0)
        gray_res[t] = np.where(in_bounds, r_gray, 0.0)
        gray_raw[t] = d_gray

        edge_res[t] = float(np.mean(np.abs(e0 - warped_edge)[valid_mask]))
        edge_raw[t] = float(np.mean(np.abs(e1 - e0)[valid_mask]))
        lap_res[t] = float(np.mean(np.abs(l0 - warped_lap)[valid_mask]))
        lap_raw[t] = float(np.mean(np.abs(l1 - l0)[valid_mask]))

        in_bounds_bwd = (
            (map_x_bwd >= 0.0)
            & (map_x_bwd <= size - 1.0)
            & (map_y_bwd >= 0.0)
            & (map_y_bwd <= size - 1.0)
        )
        bwd_res[t] = float(np.mean(np.abs(f1 - warped_bwd)[in_bounds_bwd])) if np.any(in_bounds_bwd) else float(np.mean(np.abs(f1 - warped_bwd)))

        r4 = r_gray.reshape(4, 24, 4, 24).mean(axis=(1, 3)).reshape(-1)
        r4_std = float(r4.std())
        r4_mean = float(r4.mean())
        reg4_stds.append(r4_std)
        reg4_cvs.append(r4_std / (r4_mean + EPS))
        p50_4 = float(np.median(r4)) + 1e-3
        reg4_p90_p50s.append(float(np.percentile(r4, 90) / p50_4))
        reg4_p95_p50s.append(float(np.percentile(r4, 95) / p50_4))

        r8 = r_gray.reshape(8, 12, 8, 12).mean(axis=(1, 3)).reshape(-1)
        m8 = mag.reshape(8, 12, 8, 12).mean(axis=(1, 3)).reshape(-1)
        r8_std = float(r8.std())
        r8_mean = float(r8.mean())
        patch8_stds.append(r8_std)
        patch8_cvs.append(r8_std / (r8_mean + 1e-3))
        p50_8 = float(np.median(r8)) + 1e-3
        patch8_p90_p50s.append(float(np.percentile(r8, 90) / p50_8))
        patch8_p95_p50s.append(float(np.percentile(r8, 95) / p50_8))

        sort_idx = np.argsort(m8)
        bg_idx = sort_idx[:16]
        fg_idx = sort_idx[-16:]
        fg_bg_ratios.append(float(r8[fg_idx].mean() / (r8[bg_idx].mean() + 1e-3)))

    mean_r_gray = gray_res.mean(axis=(1, 2))
    out["warp_res_gray_mean"] = _safe_float(float(mean_r_gray.mean()))
    out["warp_res_gray_std"] = _safe_float(float(mean_r_gray.std()))
    out["warp_res_gray_p90"] = _safe_float(float(np.percentile(mean_r_gray, 90)))
    out["warp_res_edge_mean"] = _safe_float(float(edge_res.mean()))
    out["warp_res_edge_std"] = _safe_float(float(edge_res.std()))
    out["warp_res_lap_mean"] = _safe_float(float(lap_res.mean()))
    out["warp_res_lap_std"] = _safe_float(float(lap_res.std()))
    out["warp_res_bwd_mean"] = _safe_float(float(bwd_res.mean()))

    out["res_raw_ratio_gray"] = _safe_float(float(gray_res.mean()) / (float(gray_raw.mean()) + EPS))
    out["res_raw_ratio_edge"] = _safe_float(float(edge_res.mean()) / (float(edge_raw.mean()) + EPS))
    out["res_raw_ratio_lap"] = _safe_float(float(lap_res.mean()) / (float(lap_raw.mean()) + EPS))

    out["fb_consistency_mean"] = _safe_float(float(fb_cons.mean()))
    out["fb_consistency_std"] = _safe_float(float(fb_cons.std()))
    out["fb_consistency_p90"] = _safe_float(float(np.percentile(fb_cons, 90)))

    out["reg4_res_std"] = _safe_float(float(np.mean(reg4_stds)))
    out["reg4_res_cv"] = _safe_float(float(np.mean(reg4_cvs)))
    out["reg4_res_p90_p50"] = _safe_float(float(np.mean(reg4_p90_p50s)))
    out["reg4_res_p95_p50"] = _safe_float(float(np.mean(reg4_p95_p50s)))

    out["patch8_res_std"] = _safe_float(float(np.mean(patch8_stds)))
    out["patch8_res_cv"] = _safe_float(float(np.mean(patch8_cvs)))
    out["patch8_res_p90_p50"] = _safe_float(float(np.mean(patch8_p90_p50s)))
    out["patch8_res_p95_p50"] = _safe_float(float(np.mean(patch8_p95_p50s)))
    out["fg_bg_residual_ratio"] = _safe_float(float(np.mean(fg_bg_ratios)))

    per_pair_mag = flow_mags.mean(axis=(1, 2))
    vel = per_pair_mag / dt_safe[:pairs]
    out["vel_mean"] = _safe_float(float(vel.mean()))
    out["vel_std"] = _safe_float(float(vel.std()))
    out["vel_p90"] = _safe_float(float(np.percentile(vel, 90)))

    if pairs >= 2:
        t_arr = np.asarray(times, dtype=np.float64)
        midpoints = 0.5 * (t_arr[:pairs] + t_arr[1 : pairs + 1])
        acc_fields = []
        acc_mags_all = []

        for p_idx in range(pairs - 1):
            d_tau = float(midpoints[p_idx + 1] - midpoints[p_idx])
            if d_tau < 1e-4:
                d_tau = 0.5 * float(dt_safe[p_idx] + dt_safe[p_idx + 1])
            d_tau = max(d_tau, 1e-4)

            a_vec = (vel_fields[p_idx + 1] - vel_fields[p_idx]) / d_tau
            acc_fields.append(a_vec)
            acc_mags_all.append(np.sqrt(a_vec[..., 0] ** 2 + a_vec[..., 1] ** 2).mean())

        acc_arr = np.array(acc_mags_all, dtype=np.float64)
        out["acc_mean"] = _safe_float(float(acc_arr.mean()))
        out["acc_std"] = _safe_float(float(acc_arr.std()))
        out["acc_p90"] = _safe_float(float(np.percentile(acc_arr, 90)))
        vel_m = float(vel.mean())
        out["acc_vel_ratio"] = _safe_float(float(acc_arr.mean() / (vel_m + 0.1)) if vel_m >= 0.01 else 0.0)

        if pairs >= 3 and len(acc_fields) >= 2:
            acc_midpoints = 0.5 * (midpoints[: pairs - 1] + midpoints[1:pairs])
            jerk_mags_all = []
            for j_idx in range(len(acc_fields) - 1):
                d_mu = float(acc_midpoints[j_idx + 1] - acc_midpoints[j_idx])
                if d_mu < 1e-4:
                    d_mu = float(dt_safe[j_idx + 1])
                d_mu = max(d_mu, 1e-4)

                j_vec = (acc_fields[j_idx + 1] - acc_fields[j_idx]) / d_mu
                jerk_mags_all.append(np.sqrt(j_vec[..., 0] ** 2 + j_vec[..., 1] ** 2).mean())

            jerk_arr = np.array(jerk_mags_all, dtype=np.float64)
            out["jerk_mean"] = _safe_float(float(jerk_arr.mean()))
            out["jerk_max"] = _safe_float(float(jerk_arr.max()))
        else:
            out["jerk_mean"] = 0.0
            out["jerk_max"] = 0.0
    else:
        out["acc_mean"] = 0.0
        out["acc_std"] = 0.0
        out["acc_p90"] = 0.0
        out["acc_vel_ratio"] = 0.0
        out["jerk_mean"] = 0.0
        out["jerk_max"] = 0.0

    out["dir_coherence_mean"] = _safe_float(float(dir_coherence.mean()))
    out["dir_coherence_std"] = _safe_float(float(dir_coherence.std()))

    if pairs >= 2:
        dot = (mean_dirs[:-1] * mean_dirs[1:]).sum(axis=1)
        norm0 = np.sqrt((mean_dirs[:-1] ** 2).sum(axis=1))
        norm1 = np.sqrt((mean_dirs[1:] ** 2).sum(axis=1))
        cos_sim = dot / (norm0 * norm1 + EPS)
        out["temporal_stability_mean"] = _safe_float(float(cos_sim.mean()))
        out["temporal_stability_std"] = _safe_float(float(cos_sim.std()))
    else:
        out["temporal_stability_mean"] = 0.0
        out["temporal_stability_std"] = 0.0

    seg_size = max(1, pairs // 3)
    s_early = mean_r_gray[:seg_size].mean()
    s_mid = mean_r_gray[seg_size : 2 * seg_size].mean() if pairs >= 2 else s_early
    s_late = mean_r_gray[2 * seg_size :].mean() if pairs >= 3 else s_early
    segs = np.array([s_early, s_mid, s_late], dtype=np.float64)
    out["segment_res_std"] = _safe_float(float(segs.std()))
    out["segment_res_max_min_ratio"] = _safe_float(float(segs.max() / (segs.min() + EPS)))
    out["segment_res_late_minus_early"] = _safe_float(float(s_late - s_early))

    if pairs >= 4:
        win_means = [mean_r_gray[i : i + 4].mean() for i in range(pairs - 3)]
        out["win4_res_p90"] = _safe_float(float(np.percentile(win_means, 90)))
        out["win4_res_p10"] = _safe_float(float(np.percentile(win_means, 10)))
        out["win4_res_iqr"] = _safe_float(
            float(np.percentile(win_means, 75) - np.percentile(win_means, 25))
        )
    else:
        out["win4_res_p90"] = _safe_float(float(mean_r_gray.mean()))
        out["win4_res_p10"] = _safe_float(float(mean_r_gray.mean()))
        out["win4_res_iqr"] = 0.0

    # Anchor drift features relative to t=0
    frames = gray128.shape[0]
    lapvars = np.zeros(frames, dtype=np.float64)
    edges = np.zeros(frames, dtype=np.float64)
    hf_ratios = np.zeros(frames, dtype=np.float64)
    rad_spectra = np.zeros((frames, 16), dtype=np.float64)

    sz128 = gray128.shape[1]
    freq = np.fft.fftfreq(sz128)
    uu, vv = np.meshgrid(freq, freq, indexing="xy")
    rad = np.sqrt(uu**2 + vv**2) / 0.5
    hf_mask = rad > 0.35
    bin_idx = np.clip((rad / np.sqrt(2.0) * 16).astype(np.int64), 0, 15).reshape(-1)

    for t in range(frames):
        g = np.ascontiguousarray(gray128[t], dtype=np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        lapvars[t] = float(lap.var())

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edges[t] = float(np.sqrt(gx * gx + gy * gy).mean())

        sp = fft2(g - g.mean())
        pw = sp.real**2 + sp.imag**2
        tot_pw = float(pw.sum()) + EPS
        hf_ratios[t] = float(pw[hf_mask].sum()) / tot_pw

        r_sum = np.bincount(bin_idx, weights=pw.reshape(-1), minlength=16)
        rad_spectra[t] = r_sum / (float(r_sum.sum()) + EPS)

    rad_dist_0 = np.sum(np.abs(rad_spectra - rad_spectra[0:1]), axis=1)

    def _drift(series: np.ndarray, name: str) -> None:
        val0 = series[0]
        denom = abs(val0) + EPS
        out[f"{name}_slope"] = _slope_per_second(series, times)
        out[f"{name}_end_start_ratio"] = _safe_float(float(series[-1] / denom))
        out[f"{name}_max_dev"] = _safe_float(float(np.max(np.abs(series - val0)) / denom))

    _drift(lapvars, "anchor_lapvar")
    _drift(edges, "anchor_edge")
    _drift(hf_ratios, "anchor_hf")

    out["anchor_rad_slope"] = _slope_per_second(rad_dist_0, times)
    out["anchor_rad_max_dev"] = _safe_float(float(rad_dist_0.max()))

    consec_diffs = np.abs(gray128[1:] - gray128[:-1]).mean(axis=(1, 2))
    cum_consec = float(consec_diffs.sum())
    net_diff = float(np.abs(gray128[-1] - gray128[0]).mean())
    out["cum_to_net_drift_ratio"] = _safe_float(cum_consec / (net_diff + EPS))

    return out, gray_res


def extract_features_from_decoded(
    gray128: np.ndarray,
    gray_flow: np.ndarray,
    grid: np.ndarray,
    times: np.ndarray,
    dt_grid: float,
    fs_grid: float,
) -> dict[str, float]:
    """Computes exactly the 113 'temporal_wavelet' features from decoded tensors.

    Returns dict mapping full feature name -> float32-compatible float in frozen order.
    """
    # Validate caller-supplied tensors BEFORE canonical safe-degenerate statistics.
    times = np.asarray(times)
    if times.ndim != 1 or times.size < 8 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ExtractionError("Expected at least eight finite strictly increasing observation times")
    for label, arr, size, count in (("gray128", gray128, 128, len(times)),
                                    ("gray_flow", gray_flow, 96, len(times)),
                                    ("grid", grid, 128, None)):
        if (not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[1:] != (size, size)
                or (count is not None and len(arr) != count) or len(arr) < 2
                or not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1)):
            raise ExtractionError(f"Malformed/nonfinite decoded tensor: {label}")
    if not np.isfinite(dt_grid) or not np.isfinite(fs_grid) or dt_grid <= 0 or fs_grid <= 0:
        raise ExtractionError("Invalid spectral sampling interval/rate")
    # 1. tfft_v2 (17 features)
    tfft_dict = _compute_tfft_v2(grid, dt_grid, fs_grid)

    # 2. local_temporal_v2 (54 features)
    local_temporal_dict, flow_residuals = _compute_local_temporal_v2(gray128, gray_flow, times)

    # 3. twavelet_v2 (42 features)
    twavelet_dict = _compute_twavelet_v2(gray128, flow_residuals)

    # Merge with exact group prefixes
    full_dict: dict[str, float] = {}
    for k, v in tfft_dict.items():
        full_dict[f"tfft_v2__{k}"] = v
    for k, v in local_temporal_dict.items():
        full_dict[f"local_temporal_v2__{k}"] = v
    for k, v in twavelet_dict.items():
        full_dict[f"twavelet_v2__{k}"] = v

    # Validate against ordered schema
    ordered_out: dict[str, float] = {}
    missing: list[str] = []
    for col in ORDERED_FEATURE_NAMES:
        if col not in full_dict:
            missing.append(col)
        else:
            val = float(full_dict[col])
            if not np.isfinite(val):
                raise ExtractionError(f"Nonfinite extracted feature: {col}")
            ordered_out[col] = val

    if missing:
        raise ExtractionError(f"Missing {len(missing)} required features: {missing[:5]}...")

    return ordered_out


def extract_features(
    video_path: str | Path,
    t_max: int = 64,
    min_sampled_frames: int = 8,
    fps_fallback: float = 25.0,
) -> dict[str, float]:
    """Decodes video and extracts the 113 'temporal_wavelet' features.

    Parameters
    ----------
    video_path : str or Path
        Path to video file.
    t_max : int, default 64
        Maximum sampled frames.
    min_sampled_frames : int, default 8
        Minimum frames required.
    fps_fallback : float, default 25.0
        Fallback FPS.

    Returns
    -------
    dict
        Dictionary of 113 float features in frozen order.
    """
    decoded = decode_canonical_video(
        video_path=video_path,
        t_max=t_max,
        min_sampled_frames=min_sampled_frames,
        fps_fallback=fps_fallback,
    )
    return extract_features_from_decoded(
        gray128=decoded["gray128"],
        gray_flow=decoded["gray_flow"],
        grid=decoded["grid"],
        times=decoded["times"],
        dt_grid=decoded["dt_grid"],
        fs_grid=decoded["fs_grid"],
    )
