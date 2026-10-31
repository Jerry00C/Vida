"""Manifest validation and connected groups for leakage-safe OOF folds."""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import Config, require_sha256

REQUIRED = {"clip_id", "path", "label", "split", "group_id", "source", "generator"}
GROUP_KEYS = ("group_id", "pair_id", "duplicate_id", "source_video_id")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: Path
    label: int
    split: str
    group_id: str
    source: str
    generator: str
    metadata: dict[str, str]


def load_manifest(path: str | Path, config: Config | None = None) -> list[Clip]:
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Manifest has duplicate column names")
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed CSV at line {line}")
            row = {key: value.strip() for key, value in raw.items()}
            if any(not row[key] for key in REQUIRED - {"generator"}):
                raise ValueError(f"Empty required field at line {line}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"label must be 0=real or 1=generated at line {line}")
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"split must be train/val/test at line {line}")
            label = int(row["label"])
            if row.get("content_sha256"):
                require_sha256(f"content_sha256 at line {line}", row["content_sha256"])
            if label == 1 and row["generator"].lower() in {"", "real"}:
                raise ValueError(f"Generated clip needs generator metadata at line {line}")
            if label == 0 and row["generator"].lower() not in {"", "real"}:
                raise ValueError(f"Real clip cannot have a generator tag at line {line}")
            clip_path = Path(row["path"]).expanduser()
            if not clip_path.is_absolute():
                clip_path = path.parent / clip_path
            rows.append(Clip(row["clip_id"], clip_path.resolve(), label, row["split"],
                             row["group_id"], row["source"], row["generator"], row))
    if not rows:
        raise ValueError("Manifest is empty")
    if len({r.clip_id for r in rows}) != len(rows):
        raise ValueError("clip_id must be unique")
    if len({r.path for r in rows}) != len(rows):
        raise ValueError("The same resolved file appears more than once")
    for key in GROUP_KEYS:
        seen = {}
        for row in rows:
            value = row.metadata.get(key, "")
            if value and value in seen and seen[value] != row.split:
                raise ValueError(f"Split leakage: {key}={value!r} crosses partitions")
            if value:
                seen[value] = row.split
    protocols = {r.metadata.get("protocol", "") for r in rows}
    if len(protocols) > 1:
        raise ValueError("Use a separate manifest for each evaluation protocol")
    config = config or Config()
    for generator in config.heldout_generators:
        selected = [r for r in rows if r.label == 1 and r.generator == generator]
        if not selected or any(r.split != "test" for r in selected):
            raise ValueError(f"Held-out generator {generator!r} must occur exclusively in test")
    for source in config.heldout_real_sources:
        selected = [r for r in rows if r.label == 0 and r.source == source]
        if not selected or any(r.split != "test" for r in selected):
            raise ValueError(f"Held-out real source {source!r} must occur exclusively in test")
    validation_masks(rows)
    return rows


def validation_masks(rows):
    """Preserve provided validation roles; legacy manifests use all val for both."""
    val = np.array([r.split == "val" for r in rows])
    roles = [r.metadata.get("validation_role", "") for r in rows]
    if any(role and r.split != "val" for r, role in zip(rows, roles)):
        raise ValueError("validation_role is only allowed on val rows")
    if not any(roles):
        return val.copy(), val.copy()
    if any(role not in {"model_selection", "calibration"} for r, role in zip(rows, roles) if r.split == "val"):
        raise ValueError("Every val row needs model_selection or calibration validation_role")
    selection = val & np.array([role == "model_selection" for role in roles])
    calibration = val & np.array([role == "calibration" for role in roles])
    if not any(s and r.label == 0 for s, r in zip(selection, rows)) or not any(c and r.label == 0 for c, r in zip(calibration, rows)):
        raise ValueError("Both validation roles require real clips")
    for key in GROUP_KEYS:
        seen = {}
        for r, role in zip(rows, roles):
            token = r.metadata.get(key, "")
            if r.split == "val" and token:
                if token in seen and seen[token] != role:
                    raise ValueError(f"Validation role leakage: {key} crosses roles")
                seen[token] = role
    return selection, calibration


def manifest_protocol(rows: list[Clip]) -> str:
    return rows[0].metadata.get("protocol", "") if rows else ""


def resolve_run_kind(rows: list[Clip], requested="auto") -> str:
    if requested not in {"auto", "research", "synthetic_smoke"}:
        raise ValueError("run_kind must be auto, research or synthetic_smoke")
    synthetic = manifest_protocol(rows) == "synthetic_smoke"
    if synthetic and requested == "research":
        raise ValueError("A synthetic_smoke manifest cannot be reported as a research run")
    return "synthetic_smoke" if synthetic or requested == "synthetic_smoke" else "research"


def connected_groups(rows: list[Clip], content_hashes=None) -> np.ndarray:
    """Merge overlapping group, prompt/pair, duplicate and source-video identifiers."""
    if content_hashes is not None and len(content_hashes) != len(rows):
        raise ValueError("One content hash is required per leakage-group row")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [(key, row.metadata.get(key, "")) for key in GROUP_KEYS]
        if content_hashes is not None:
            keys.append(("content_hash", content_hashes[i]))
        for key, value in keys:
            if not value:
                continue
            token = (key, value)
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    return np.array([find(i) for i in range(len(rows))])
