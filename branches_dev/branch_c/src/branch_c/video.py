"""Bounded decoding to RGB frames on a fixed time grid (CFR video contract)."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from .config import FeatureConfig

VIDEO_VERSION = "cfr-native-stride-v3"


@dataclass
class SampledVideo:
    frames_rgb: np.ndarray
    timestamps_seconds: np.ndarray
    metadata: dict


def validate_frames(frames, timestamps, config: FeatureConfig) -> SampledVideo:
    frames, timestamps = np.asarray(frames), np.asarray(timestamps, dtype=np.float64)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames_rgb must be uint8 [T,H,W,3], RGB, range 0..255")
    if min(frames.shape[1:3]) < 24 or len(frames) < 8:
        raise ValueError("Frames must be >=24 pixels per side, with >=8 frames")
    if timestamps.shape != (len(frames),) or not np.isfinite(timestamps).all():
        raise ValueError("One finite timestamp is required for every frame")
    intervals = np.diff(timestamps)
    if timestamps[0] < 0 or np.any(intervals <= 0):
        raise ValueError("Timestamps must be nonnegative and strictly increasing")
    effective_fps = 1 / float(np.median(intervals))
    if config.sampling_mode == "native_stride":
        if not 4 - 1e-3 <= effective_fps <= 240 + 1e-3:
            raise ValueError("Native-stride sampled fps must be in [4,240]")
        expected_interval = 1 / effective_fps
    else:
        expected_interval = 1 / config.sample_fps
    if not np.allclose(intervals, expected_interval, atol=1e-4, rtol=1e-3):
        raise ValueError(f"Frames must be uniformly sampled at {config.sample_fps} fps")
    duration = float(timestamps[-1] - timestamps[0] + expected_interval)
    if duration < config.min_duration_seconds - 1e-4 or duration > config.max_duration_seconds + 1e-4:
        raise ValueError(f"Duration {duration:.3f}s is outside the configured range")
    resized = np.stack([cv2.resize(frame, (config.image_size, config.image_size),
                                 interpolation=cv2.INTER_AREA) for frame in frames])
    return SampledVideo(resized, timestamps, {"duration_seconds": duration,
                        "sample_fps": effective_fps, "sampled_frames": len(frames),
                        "input_height": int(frames.shape[1]), "input_width": int(frames.shape[2])})


def read_video(path: str | Path, config: FeatureConfig) -> SampledVideo:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if not {"frames_rgb", "timestamps_seconds"} <= set(data.files):
                raise ValueError("NPZ needs frames_rgb and timestamps_seconds")
            result = validate_frames(data["frames_rgb"], data["timestamps_seconds"], config)
        result.metadata["timebase"] = "provided_uniform_timestamps"
        return result
    if config.sampling_mode == "native_stride":
        return _read_native_stride(path, config)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps < config.sample_fps or fps > 240:
            raise ValueError(f"Native fps must be in [{config.sample_fps},240]; got {fps}")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if count > 0 and count / fps > config.max_duration_seconds + 1e-6:
            raise ValueError("Video exceeds maximum duration; trim explicitly upstream")
        frames, times = [], []
        native_height = native_width = None
        index, target_index = 0, 0
        max_decode = int(np.ceil((config.max_duration_seconds + 0.25) * fps))
        # Read sequentially; never trust seek success or silently pad failed frames.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= max_decode:
                raise ValueError("Decoded video exceeds duration bound")
            if index == 0:
                native_height, native_width = frame.shape[:2]
            if index >= round(target_index * fps / config.sample_fps):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (config.image_size, config.image_size),
                                         interpolation=cv2.INTER_AREA))
                times.append(target_index / config.sample_fps)
                target_index += 1
            index += 1
        duration = index / fps
        if duration < config.min_duration_seconds - 1e-6 or duration > config.max_duration_seconds + 1e-6:
            raise ValueError(f"Decoded duration {duration:.3f}s is outside the range")
        if count > 0 and index < count - max(2, count * 0.01):
            raise ValueError("Video ended before its declared frame count; possible decode failure")
        if not frames:
            raise ValueError("No frames decoded")
        # Retain complete sampling intervals only. Never let a rounded grid extend
        # beyond EOF, which could otherwise turn a valid 20s clip into >20s.
        keep = np.asarray(times) + 1 / config.sample_fps <= duration + 1e-6
        result = validate_frames(np.stack(frames)[keep], np.asarray(times)[keep], config)
        result.metadata.update(native_fps=fps, decoded_frames=index,
                               native_height=native_height, native_width=native_width,
                               decoded_duration_seconds=duration,
                               discarded_tail_seconds=max(0.0, duration - result.metadata["duration_seconds"]),
                               decoder_version=VIDEO_VERSION,
                               timebase="CFR_nearest_frame_on_uniform_grid",
                               timestamp_error_bound_seconds=0.5 / fps)
        return result
    finally:
        cap.release()


def _read_native_stride(path, config):
    """CFR-only: retain distinct native frames at an integer stride and their true times.

    Aim for 8 fps, reducing stride if necessary to keep >=8 actual observations.
    A short clip becomes one full-clip window; no padding/interpolation/reencoding.
    The delivery adapter independently checks audited native timestamps are CFR.
    """
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or not 4 <= fps <= 240 or count < 8:
            raise ValueError("Native-stride input requires CFR fps in [4,240] and >=8 frames")
        duration = count / fps
        if not config.min_duration_seconds - 1e-4 <= duration <= config.max_duration_seconds + 1e-4:
            raise ValueError(f"Declared duration {duration:.3f}s is outside the range")
        stride = min(max(1, round(fps / config.sample_fps)), count // 8)
        retained_count = count // stride
        frames, times = [], []
        index = 0
        native_height = native_width = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= count:
                raise ValueError("Decoded more frames than declared; audit CFR input upstream")
            if index == 0:
                native_height, native_width = frame.shape[:2]
            if index % stride == 0 and len(frames) < retained_count:
                frames.append(cv2.cvtColor(cv2.resize(frame, (config.image_size, config.image_size),
                                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                times.append(index / fps)
            index += 1
        if index != count:
            raise ValueError(f"Decoded {index}/{count} frames; no partial clip accepted: {path}")
        result = validate_frames(np.stack(frames), times, config)
        result.metadata.update(native_fps=fps, decoded_frames=index, native_height=native_height,
            native_width=native_width, decoded_duration_seconds=duration, native_stride=stride,
            discarded_tail_seconds=max(0., duration - result.metadata["duration_seconds"]),
            decoder_version=VIDEO_VERSION, timebase="CFR_integer_native_stride_true_timestamps",
            timestamp_error_bound_seconds=0., sampling_mode=config.sampling_mode,
            input_height=native_height, input_width=native_width)
        return result
    finally:
        cap.release()
