"""37 deterministic descriptors; no labels, learned encoders or fitting here."""
from dataclasses import asdict, dataclass
import cv2
import numpy as np

from .config import FeatureConfig, fingerprint
from .video import SampledVideo, VIDEO_VERSION, validate_frames

FEATURE_VERSION = "motion-frequency-time-aware-v2"
TRANSITION_NAMES = (
    "pixel_delta_mae", "pixel_delta_std", "pixel_delta_p95", "luma_change",
    "edge_change", "flow_speed_mean", "flow_speed_p90", "flow_speed_std",
    "flow_direction_coherence", "flow_warp_residual",
)
FEATURE_NAMES = tuple(f"{name}_{stat}" for name in TRANSITION_NAMES for stat in ("mean", "std", "p90")) + (
    "temporal_low_energy_ratio", "temporal_mid_energy_ratio", "temporal_high_energy_ratio",
    "temporal_spectral_centroid", "temporal_spectral_entropy",
    "luma_temporal_std", "flow_speed_change_mean",
)


def feature_signature(config: FeatureConfig) -> str:
    return fingerprint({"version": FEATURE_VERSION, "names": FEATURE_NAMES, "config": asdict(config),
                        "decoder_version": VIDEO_VERSION,
                        "opencv_version": cv2.__version__, "numpy_version": np.__version__})


@dataclass
class FeatureResult:
    clip: np.ndarray
    windows: np.ndarray
    bounds_seconds: np.ndarray
    metadata: dict


def _frequency(gray, sample_fps=None):
    small = np.stack([cv2.resize(g, (24, 24), interpolation=cv2.INTER_AREA) for g in gray])
    signals = small - small.mean(axis=0, keepdims=True)
    signals *= np.hanning(len(signals))[:, None, None]
    power = np.square(np.abs(np.fft.rfft(signals, axis=0))).mean(axis=(1, 2))[1:]
    frequencies = np.fft.rfftfreq(len(gray))[1:] / 0.5
    if sample_fps is not None:
        # Common 0..2 Hz support is observable even in the 4 fps source.
        hz = np.fft.rfftfreq(len(gray), d=1 / sample_fps)[1:]
        keep = hz <= 2 + 1e-6
        power, frequencies = power[keep], hz[keep] / 2
    total = power.sum()
    if total < 1e-12:
        return np.zeros(5)
    p = power / total
    entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(max(2, len(p)))
    return np.array([p[frequencies <= 0.25].sum(),
                     p[(frequencies > 0.25) & (frequencies <= 0.6)].sum(),
                     p[frequencies > 0.6].sum(), np.sum(p * frequencies), entropy])


def _summarize(gray, transitions, sample_fps=None):
    pooled = np.stack([transitions.mean(axis=0), transitions.std(axis=0),
                       np.quantile(transitions, 0.9, axis=0)], axis=1).ravel()
    speed_change = np.abs(np.diff(transitions[:, 5])).mean() if len(transitions) > 1 else 0.0
    result = np.r_[pooled, _frequency(gray, sample_fps), gray.mean(axis=(1, 2)).std(), speed_change]
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or malformed descriptor")
    return result.astype(np.float64)


def extract_features(video: SampledVideo, config: FeatureConfig) -> FeatureResult:
    validated = validate_frames(video.frames_rgb, video.timestamps_seconds, config)
    gray8 = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in validated.frames_rgb])
    gray = gray8.astype(np.float32) / 255.0
    size = config.image_size
    gy, gx = np.mgrid[:size, :size].astype(np.float32)
    transitions = []
    effective_fps = validated.metadata["sample_fps"]
    rate_scale = effective_fps / 8 if config.sampling_mode == "native_stride" else 1.
    for prev8, curr8, prev, curr in zip(gray8[:-1], gray8[1:], gray[:-1], gray[1:]):
        delta = np.abs(curr - prev)
        flow = cv2.calcOpticalFlowFarneback(prev8, curr8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=-1)
        speed = magnitude * effective_fps / size
        coherence = np.linalg.norm(flow.mean(axis=(0, 1))) / (magnitude.mean() + 1e-8)
        warped = cv2.remap(curr, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)
        # Exclude pixels whose forward-flow coordinates leave the image.
        valid = (gx + flow[..., 0] >= 0) & (gx + flow[..., 0] < size - 1) & (gy + flow[..., 1] >= 0) & (gy + flow[..., 1] < size - 1)
        residual = float(np.abs(prev - warped)[valid].mean()) if valid.any() else 1.0
        edge_change = np.abs(cv2.Laplacian(curr, cv2.CV_32F) - cv2.Laplacian(prev, cv2.CV_32F)).mean()
        transitions.append([delta.mean() * rate_scale, delta.std() * rate_scale, np.quantile(delta, .95) * rate_scale,
                            abs(curr.mean() - prev.mean()) * rate_scale, edge_change * rate_scale,
                            speed.mean(), np.quantile(speed, .9), speed.std(), coherence, residual])
    transitions = np.asarray(transitions)
    width = max(8, round(config.window_seconds * effective_fps))
    stride = max(1, round(config.stride_seconds * effective_fps))
    if config.sampling_mode == "native_stride":
        width = min(width, len(gray))
    if len(gray) < width:
        raise ValueError("Not enough frames for configured window")
    starts = list(range(0, len(gray) - width + 1, stride))
    if starts[-1] != len(gray) - width:
        starts.append(len(gray) - width)
    spectral_fps = effective_fps if config.sampling_mode == "native_stride" else None
    windows = np.stack([_summarize(gray[s:s + width], transitions[s:s + width - 1], spectral_fps) for s in starts])
    times = validated.timestamps_seconds
    bounds = np.array([[times[s], times[s + width - 1] + 1 / effective_fps] for s in starts])
    metadata = {**video.metadata, **validated.metadata, "feature_version": FEATURE_VERSION,
                "feature_image_size": config.image_size}
    # read_video has already resized the input. Revalidation describes the
    # working frames, so retain the original pre-sampled dimensions separately.
    for key in ("input_height", "input_width"):
        if key in video.metadata:
            metadata[key] = video.metadata[key]
    metadata["window_policy"] = "whole_clip_if_short_else_sliding" if config.sampling_mode == "native_stride" else "sliding"
    return FeatureResult(_summarize(gray, transitions, spectral_fps), windows, bounds, metadata)
