from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class Minutia:
    x: float
    y: float
    angle: float | None = None
    quality: float | None = None
    type: str | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimilarityTransform:
    query_centroid: tuple[float, float]
    template_centroid: tuple[float, float]
    rotation: float = 0.0
    scale: float = 1.0
    rotation_source: str | None = None
    scale_source: str | None = None


@dataclass(slots=True)
class PoseNormalizationConfig:
    min_minutiae: int = 3
    reliable_minutiae: int = 8
    use_quality_weighted_centroid: bool = False
    min_valid_ridge_spacing: float = 3.0
    max_valid_ridge_spacing: float = 30.0
    min_valid_scale: float = 0.5
    max_valid_scale: float = 2.0


@dataclass(slots=True)
class PoseNormalizationDiagnostics:
    query_centroid: tuple[float, float]
    template_centroid: tuple[float, float]
    rotation: float
    rotation_source: str
    scale: float
    scale_source: str
    centroid_weighted: bool
    warnings: list[str] = field(default_factory=list)
    query_orientation: float | None = None
    template_orientation: float | None = None
    query_ridge_spacing: float | None = None
    template_ridge_spacing: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_centroid": [float(self.query_centroid[0]), float(self.query_centroid[1])],
            "template_centroid": [float(self.template_centroid[0]), float(self.template_centroid[1])],
            "rotation": float(self.rotation),
            "rotation_degrees": float(math.degrees(self.rotation)),
            "rotation_source": self.rotation_source,
            "scale": float(self.scale),
            "scale_source": self.scale_source,
            "centroid_weighted": bool(self.centroid_weighted),
            "warnings": list(self.warnings),
            "query_orientation": None if self.query_orientation is None else float(self.query_orientation),
            "template_orientation": None if self.template_orientation is None else float(self.template_orientation),
            "query_ridge_spacing": None if self.query_ridge_spacing is None else float(self.query_ridge_spacing),
            "template_ridge_spacing": None if self.template_ridge_spacing is None else float(self.template_ridge_spacing),
        }


def wrap_angle_pi(angle: float) -> float:
    """Wrap a directional angle to [-pi, pi)."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped if math.isfinite(wrapped) else 0.0


def wrap_angle_2pi(angle: float) -> float:
    """Wrap a directional angle to [0, 2*pi)."""
    wrapped = float(angle) % (2.0 * math.pi)
    return wrapped if math.isfinite(wrapped) else 0.0


def wrap_orientation_pi(angle: float) -> float:
    """Wrap an axial ridge orientation to [0, pi)."""
    wrapped = float(angle) % math.pi
    return wrapped if math.isfinite(wrapped) else 0.0


def minutia_from_record(record: Mapping[str, Any]) -> Minutia:
    angle = record.get("angle", record.get("theta", record.get("direction")))
    quality = record.get("quality", record.get("score"))
    minutia_type = record.get("type")
    source = record.get("source")
    known = {"x", "y", "angle", "theta", "direction", "quality", "score", "type", "source"}
    extra = {str(key): value for key, value in record.items() if key not in known}
    if "score" in record:
        extra["score"] = record["score"]
    return Minutia(
        x=float(record["x"]),
        y=float(record["y"]),
        angle=None if angle is None else float(angle),
        quality=None if quality is None else float(quality),
        type=None if minutia_type is None else str(minutia_type),
        source=None if source is None else str(source),
        extra=extra,
    )


def minutia_to_record(minutia: Minutia) -> dict[str, Any]:
    record = dict(minutia.extra)
    record["x"] = float(minutia.x)
    record["y"] = float(minutia.y)
    if minutia.angle is not None:
        record["angle"] = float(minutia.angle)
    if minutia.quality is not None:
        record["score"] = float(minutia.quality)
    if minutia.type is not None:
        record["type"] = minutia.type
    if minutia.source is not None:
        record["source"] = minutia.source
    return record


def coerce_minutiae(records: Iterable[Minutia | Mapping[str, Any]]) -> list[Minutia]:
    minutiae: list[Minutia] = []
    for record in records:
        if isinstance(record, Minutia):
            minutiae.append(
                Minutia(
                    x=float(record.x),
                    y=float(record.y),
                    angle=None if record.angle is None else float(record.angle),
                    quality=None if record.quality is None else float(record.quality),
                    type=record.type,
                    source=record.source,
                    extra=dict(record.extra),
                )
            )
        else:
            minutiae.append(minutia_from_record(record))
    return minutiae


def compute_minutiae_centroid(
    minutiae: Sequence[Minutia | Mapping[str, Any]],
    use_quality_weights: bool = False,
) -> tuple[float, float]:
    points = coerce_minutiae(minutiae)
    if not points:
        raise ValueError("cannot compute centroid for an empty minutiae list")

    if use_quality_weights:
        weights = np.array(
            [
                max(0.0, float(point.quality))
                if point.quality is not None and math.isfinite(float(point.quality))
                else 0.0
                for point in points
            ],
            dtype=np.float64,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            xs = np.array([point.x for point in points], dtype=np.float64)
            ys = np.array([point.y for point in points], dtype=np.float64)
            return float(np.dot(weights, xs) / weight_sum), float(np.dot(weights, ys) / weight_sum)

    return (
        float(np.mean([point.x for point in points], dtype=np.float64)),
        float(np.mean([point.y for point in points], dtype=np.float64)),
    )


def translate_minutiae_to_centroid(
    minutiae: Sequence[Minutia | Mapping[str, Any]],
    centroid: tuple[float, float] | None = None,
    use_quality_weights: bool = False,
) -> tuple[list[Minutia], tuple[float, float]]:
    points = coerce_minutiae(minutiae)
    used_centroid = centroid
    if used_centroid is None:
        used_centroid = compute_minutiae_centroid(
            points,
            use_quality_weights=use_quality_weights,
        )
    cx, cy = float(used_centroid[0]), float(used_centroid[1])
    translated = [
        Minutia(
            x=float(point.x) - cx,
            y=float(point.y) - cy,
            angle=point.angle,
            quality=point.quality,
            type=point.type,
            source=point.source,
            extra=dict(point.extra),
        )
        for point in points
    ]
    return translated, (cx, cy)


def rotate_point(x: float, y: float, theta: float) -> tuple[float, float]:
    cos_t = math.cos(float(theta))
    sin_t = math.sin(float(theta))
    return (
        (cos_t * float(x)) - (sin_t * float(y)),
        (sin_t * float(x)) + (cos_t * float(y)),
    )


def estimate_global_orientation_from_field(
    orientation: np.ndarray,
    confidence: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    angles = np.asarray(orientation, dtype=np.float64)
    valid = np.isfinite(angles) & (angles != 0.0)
    if mask is not None:
        valid &= np.asarray(mask) > 0

    if confidence is None:
        weights = np.ones_like(angles, dtype=np.float64)
    else:
        weights = np.nan_to_num(np.asarray(confidence, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 0.0)
        valid &= weights > 0.0

    if not np.any(valid):
        raise ValueError("no valid orientation samples")

    doubled = 2.0 * angles[valid]
    sample_weights = weights[valid]
    c = float(np.sum(sample_weights * np.cos(doubled)))
    s = float(np.sum(sample_weights * np.sin(doubled)))
    if math.hypot(c, s) <= 1e-12:
        raise ValueError("orientation samples are ambiguous")
    return wrap_orientation_pi(0.5 * math.atan2(s, c))


def estimate_median_ridge_spacing(
    ridge_period: np.ndarray,
    min_valid_spacing: float = 3.0,
    max_valid_spacing: float = 30.0,
) -> float:
    values = np.asarray(ridge_period, dtype=np.float64)
    valid = values[
        np.isfinite(values)
        & (values >= float(min_valid_spacing))
        & (values <= float(max_valid_spacing))
    ]
    if valid.size == 0:
        raise ValueError("no valid ridge spacing samples")
    return float(np.median(valid))


def estimate_scale_from_ridge_spacing(
    query_spacing: float,
    template_spacing: float,
    min_valid_scale: float = 0.5,
    max_valid_scale: float = 2.0,
) -> float:
    if not (
        math.isfinite(float(query_spacing))
        and math.isfinite(float(template_spacing))
        and float(query_spacing) > 0.0
        and float(template_spacing) > 0.0
    ):
        raise ValueError("ridge spacing values must be finite and positive")
    scale = float(template_spacing) / float(query_spacing)
    if not (float(min_valid_scale) <= scale <= float(max_valid_scale)):
        raise ValueError(f"scale estimate out of range: {scale}")
    return scale


def apply_similarity_transform_to_minutiae(
    minutiae: Sequence[Minutia | Mapping[str, Any]],
    transform: SimilarityTransform,
) -> list[Minutia]:
    points = coerce_minutiae(minutiae)
    qx, qy = transform.query_centroid
    transformed: list[Minutia] = []
    for point in points:
        centered_x = float(point.x) - float(qx)
        centered_y = float(point.y) - float(qy)
        rotated_x, rotated_y = rotate_point(centered_x, centered_y, transform.rotation)
        angle = point.angle
        if angle is not None:
            angle = wrap_angle_pi(float(angle) + float(transform.rotation))
        transformed.append(
            Minutia(
                x=float(transform.scale) * rotated_x,
                y=float(transform.scale) * rotated_y,
                angle=angle,
                quality=point.quality,
                type=point.type,
                source=point.source,
                extra=dict(point.extra),
            )
        )
    return transformed


def load_optional_array(path: Path | str | np.ndarray | None) -> np.ndarray | None:
    if path is None:
        return None
    if isinstance(path, np.ndarray):
        return path
    candidate = Path(path)
    if not candidate.exists():
        return None
    return np.load(candidate)
