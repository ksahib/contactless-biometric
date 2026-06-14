from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from . import target_rasterization as tr


@dataclass(slots=True)
class AugmentationConfig:
    count: int = 5
    translation_typical_px: float = 16.0
    translation_strong_px: float = 32.0
    yaw_deg: float = 5.0
    pitch_roll_typical_deg: float = 15.0
    pitch_roll_strong_deg: float = 25.0
    missing_reconstruction: str = "skip"
    debug_dir: Path | None = None
    debug_limit: int = 0
    reconstruction_cache_size: int = 2
    sample_cache_size: int = 8
    group_variants: bool = False


@dataclass(slots=True)
class AugmentationParams:
    translate: bool
    yaw: bool
    pitch: bool
    roll: bool
    dx: float
    dy: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    @property
    def has_3d(self) -> bool:
        return self.pitch or self.roll


@dataclass(slots=True)
class AugmentationResult:
    image: np.ndarray
    mask: np.ndarray
    targets: dict[str, np.ndarray]
    params: AugmentationParams
    minutiae: list[dict[str, Any]]
    details: dict[str, Any]


_RECONSTRUCTION_CACHE: OrderedDict[tuple[str, str], dict[str, np.ndarray]] = OrderedDict()
_SAMPLE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def sample_augmentation_params(rng: np.random.Generator, config: AugmentationConfig) -> AugmentationParams:
    translate = bool(rng.random() < 0.80)
    yaw = bool(rng.random() < 0.45)
    pitch = bool(rng.random() < 0.35)
    roll = bool(rng.random() < 0.35)
    if not (translate or yaw or pitch or roll):
        translate = True

    dx = _sample_translation_axis(rng, config) if translate else 0.0
    dy = _sample_translation_axis(rng, config) if translate else 0.0
    yaw_deg = float(rng.uniform(-config.yaw_deg, config.yaw_deg)) if yaw else 0.0
    pitch_deg = _sample_pitch_roll(rng, config) if pitch else 0.0
    roll_deg = _sample_pitch_roll(rng, config) if roll else 0.0

    for _ in range(8):
        severity = _severity(dx, dy, yaw_deg, pitch_deg, roll_deg, config)
        if severity <= 2.45 or rng.random() < 0.08:
            break
        if abs(dx) >= 0.85 * config.translation_strong_px:
            dx *= 0.65
        if abs(dy) >= 0.85 * config.translation_strong_px:
            dy *= 0.65
        pitch_deg *= 0.75
        roll_deg *= 0.75
        yaw_deg *= 0.75

    return AugmentationParams(
        translate=translate,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        dx=float(dx),
        dy=float(dy),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
    )


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Column-vector rotation. yaw=z/in-plane, pitch=x, roll=y."""
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    cz, sz = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def has_usable_reconstruction(sample: Mapping[str, Any]) -> bool:
    reconstruction = sample.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        return False
    path = reconstruction.get("reconstruction_maps_path")
    return path is not None and Path(path).exists()


def augment_sample(
    sample: Mapping[str, Any],
    params: AugmentationParams,
    *,
    output_shape: tuple[int, int],
    debug_dir: Path | None = None,
    debug_stem: str | None = None,
    reconstruction_cache_size: int = 2,
    sample_cache_size: int = 8,
) -> AugmentationResult:
    cached_sample = _load_cached_sample(sample, max_cache_size=sample_cache_size)
    image = cached_sample["image"]
    mask = cached_sample["mask"]
    orientation = cached_sample["orientation"]
    ridge_period = cached_sample["ridge_period"]
    minutiae = cached_sample["minutiae"]
    use_3d = params.has_3d and has_usable_reconstruction(sample)
    base_targets = {} if use_3d else _cached_gradient_for_sample(sample, cached_sample, image.shape)

    if use_3d:
        role = str(sample["reconstruction"].get("role", _role_for_raw_view(sample.get("raw_view_index"))))
        rendered = render_3d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge_period,
            gradient=base_targets.get("gradient_full"),
            minutiae=minutiae,
            reconstruction=_load_reconstruction_maps(
                Path(sample["reconstruction"]["reconstruction_maps_path"]),
                role=role,
                max_cache_size=reconstruction_cache_size,
            ),
            role=role,
            params=params,
        )
    else:
        rendered = render_2d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge_period,
            gradient=base_targets.get("gradient_full"),
            minutiae=minutiae,
            params=params,
        )

    targets = tr.build_featurenet_targets(
        rendered["image"],
        rendered["mask"],
        rendered["orientation"],
        rendered["ridge_period"],
        rendered["gradient"]
        if rendered["gradient"] is not None
        else np.zeros((*rendered["image"].shape, 2), dtype=np.float32),
        rendered["minutiae"],
        output_shape=output_shape,
    )

    result = AugmentationResult(
        image=rendered["image"],
        mask=rendered["mask"],
        targets=targets,
        params=params,
        minutiae=rendered["minutiae"],
        details=rendered["details"],
    )
    if debug_dir is not None and debug_stem is not None:
        save_debug_visualization(debug_dir, debug_stem, image, mask, minutiae, result)
    return result


def sample_cache_info() -> dict[str, Any]:
    return {
        "size": len(_SAMPLE_CACHE),
        "keys": list(_SAMPLE_CACHE.keys()),
    }


def render_2d_augmented(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    orientation: np.ndarray,
    ridge_period: np.ndarray,
    gradient: np.ndarray | None,
    minutiae: list[dict[str, Any]],
    params: AugmentationParams,
) -> dict[str, Any]:
    height, width = image.shape
    matrix = _affine_matrix(width, height, params.yaw_deg, params.dx, params.dy)
    warped_image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_orientation = _warp_orientation_2d(orientation, mask, matrix, params.yaw_deg)
    warped_ridge = cv2.warpAffine(ridge_period.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_gradient = _warp_gradient_2d(gradient, matrix, params.yaw_deg, image.shape) if gradient is not None else None
    transformed_minutiae = _transform_minutiae_affine(minutiae, matrix, math.radians(params.yaw_deg), warped_mask)
    return {
        "image": _finite_u8(warped_image),
        "mask": np.where(warped_mask > 0, 255, 0).astype(np.uint8),
        "orientation": warped_orientation,
        "ridge_period": np.nan_to_num(warped_ridge, nan=0.0).astype(np.float32),
        "gradient": warped_gradient,
        "minutiae": transformed_minutiae,
        "details": {"mode": "2d_affine", "dropped_minutiae": len(minutiae) - len(transformed_minutiae)},
    }


def render_3d_augmented(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    orientation: np.ndarray,
    ridge_period: np.ndarray,
    gradient: np.ndarray | None,
    minutiae: list[dict[str, Any]],
    reconstruction: dict[str, np.ndarray],
    role: str,
    params: AugmentationParams,
) -> dict[str, Any]:
    height, width = image.shape
    base_x, base_y, depth, valid = _role_surface(reconstruction, role, image.shape)
    depth_center = float(np.nanmedian(depth[valid])) if np.any(valid) else 0.0
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    r = rotation_matrix(params.yaw_deg, params.pitch_deg, params.roll_deg)

    ys, xs = np.nonzero(valid)
    source_x = base_x[ys, xs]
    source_y = base_y[ys, xs]
    texture_image = _sample_bilinear(image.astype(np.float32), source_x, source_y)
    texture_mask = _sample_nearest(mask, source_x, source_y)
    texture_orientation = _sample_orientation(orientation, source_x, source_y)
    texture_ridge = _sample_bilinear(ridge_period.astype(np.float32), source_x, source_y)

    points = np.stack([source_x - cx, source_y - cy, depth[ys, xs] - depth_center], axis=0)
    rotated = r @ points
    proj_x = rotated[0] + cx + params.dx
    proj_y = rotated[1] + cy + params.dy
    proj_z = rotated[2] + depth_center

    rendered_image = np.zeros((height, width), dtype=np.float32)
    rendered_mask = np.zeros((height, width), dtype=np.uint8)
    rendered_depth = np.full((height, width), np.nan, dtype=np.float32)
    rendered_orientation = np.zeros((height, width), dtype=np.float32)
    rendered_ridge = np.zeros((height, width), dtype=np.float32)

    orientation_delta = _orientation_delta_for_points(texture_orientation, r)
    xp = np.rint(proj_x).astype(np.int32)
    yp = np.rint(proj_y).astype(np.int32)
    visible = (
        np.isfinite(proj_x)
        & np.isfinite(proj_y)
        & np.isfinite(proj_z)
        & (xp >= 0)
        & (xp < width)
        & (yp >= 0)
        & (yp < height)
        & (texture_mask > 0)
    )

    visible_indices = np.nonzero(visible)[0]
    if visible_indices.size:
        visible_flat = (yp[visible_indices].astype(np.int64) * int(width)) + xp[visible_indices].astype(np.int64)
        visible_z = proj_z[visible_indices].astype(np.float32)
        z_buffer_flat = np.full(height * width, -np.inf, dtype=np.float32)
        np.maximum.at(z_buffer_flat, visible_flat, visible_z)
        winner_mask = visible_z >= (z_buffer_flat[visible_flat] - 1e-6)
        winner_indices = visible_indices[winner_mask]
        winner_flat = visible_flat[winner_mask]

        image_flat = rendered_image.reshape(-1)
        mask_flat = rendered_mask.reshape(-1)
        depth_flat = rendered_depth.reshape(-1)
        orientation_flat = rendered_orientation.reshape(-1)
        ridge_flat = rendered_ridge.reshape(-1)
        image_flat[winner_flat] = texture_image[winner_indices]
        mask_flat[winner_flat] = 255
        depth_flat[winner_flat] = proj_z[winner_indices]
        orientation_flat[winner_flat] = orientation_delta[winner_indices]
        ridge_flat[winner_flat] = texture_ridge[winner_indices]
        _fill_nearest_holes(rendered_image, rendered_mask, rendered_depth, rendered_orientation, rendered_ridge)

    if not np.any(rendered_mask > 0):
        return render_2d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge_period,
            gradient=gradient,
            minutiae=minutiae,
            params=params,
        )

    rendered_depth_filled = np.where(np.isfinite(rendered_depth), rendered_depth, 0.0).astype(np.float32)
    grad_y, grad_x = np.gradient(rendered_depth_filled)
    rendered_gradient = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)
    rendered_gradient[rendered_mask <= 0] = 0.0

    transformed_minutiae = _transform_minutiae_3d(
        minutiae=minutiae,
        depth_image=_pose_depth_image(base_x, base_y, depth, valid, image.shape),
        r=r,
        cx=cx,
        cy=cy,
        depth_center=depth_center,
        dx=params.dx,
        dy=params.dy,
        rendered_mask=rendered_mask,
    )
    return {
        "image": _finite_u8(rendered_image),
        "mask": rendered_mask,
        "orientation": np.nan_to_num(rendered_orientation, nan=0.0).astype(np.float32),
        "ridge_period": np.nan_to_num(rendered_ridge, nan=0.0).astype(np.float32),
        "gradient": rendered_gradient,
        "minutiae": transformed_minutiae,
        "details": {
            "mode": "3d_orthographic_zbuffer",
            "dropped_minutiae": len(minutiae) - len(transformed_minutiae),
            "axis_convention": "yaw=z, pitch=x, roll=y",
        },
    }


def save_debug_visualization(
    debug_dir: Path,
    stem: str,
    before_image: np.ndarray,
    before_mask: np.ndarray,
    before_minutiae: list[dict[str, Any]],
    result: AugmentationResult,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{stem}_before.png"), _draw_minutiae(before_image, before_mask, before_minutiae))
    cv2.imwrite(str(debug_dir / f"{stem}_after.png"), _draw_minutiae(result.image, result.mask, result.minutiae))
    overlay = cv2.cvtColor(result.image, cv2.COLOR_GRAY2BGR)
    overlay[result.mask > 0] = (0.65 * overlay[result.mask > 0] + np.array([0, 80, 0])).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_mask_overlay.png"), overlay)
    (debug_dir / f"{stem}.json").write_text(
        json.dumps({"params": asdict(result.params), "details": result.details}, indent=2),
        encoding="utf-8",
    )


def _sample_translation_axis(rng: np.random.Generator, config: AugmentationConfig) -> float:
    high = config.translation_typical_px if rng.random() < 0.70 else config.translation_strong_px
    low = 0.0 if high == config.translation_typical_px else config.translation_typical_px
    magnitude = float(rng.uniform(low, high))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * magnitude


def _sample_pitch_roll(rng: np.random.Generator, config: AugmentationConfig) -> float:
    limit = config.pitch_roll_typical_deg if rng.random() < 0.85 else config.pitch_roll_strong_deg
    return float(rng.uniform(-limit, limit))


def _severity(dx: float, dy: float, yaw: float, pitch: float, roll: float, config: AugmentationConfig) -> float:
    t = math.hypot(dx, dy) / max(math.sqrt(2.0) * config.translation_strong_px, 1e-6)
    y = abs(yaw) / max(config.yaw_deg, 1e-6)
    p = abs(pitch) / max(config.pitch_roll_strong_deg, 1e-6)
    r = abs(roll) / max(config.pitch_roll_strong_deg, 1e-6)
    return t + y + p + r


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to read image: {path}")
    return image


def _load_or_default(path: Path, shape: tuple[int, int], dtype: Any) -> np.ndarray:
    if str(path) and path.is_file():
        value = np.load(path)
        if value.shape == shape:
            return np.nan_to_num(value.astype(dtype), nan=0.0)
        return cv2.resize(value.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _load_minutiae(path: Path) -> list[dict[str, Any]]:
    if not str(path) or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if isinstance(payload, list) else []


def _sample_cache_key(sample: Mapping[str, Any]) -> str:
    sample_id = sample.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    return str(Path(sample["masked_image"]).resolve())


def _load_cached_sample(sample: Mapping[str, Any], *, max_cache_size: int) -> dict[str, Any]:
    key = _sample_cache_key(sample)
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None:
        _SAMPLE_CACHE.move_to_end(key)
        return cached

    image = _read_gray(Path(sample["masked_image"]))
    mask = _read_gray(Path(sample["mask"]))
    cached = {
        "image": image,
        "mask": mask,
        "orientation": _load_or_default(Path(sample.get("orientation_path", "")), image.shape, dtype=np.float32),
        "ridge_period": _load_or_default(Path(sample.get("ridge_period_path", "")), image.shape, dtype=np.float32),
        "minutiae": _load_minutiae(Path(sample.get("minutiae_path", ""))),
        "gradient_full": None,
    }
    _SAMPLE_CACHE[key] = cached
    _SAMPLE_CACHE.move_to_end(key)
    while len(_SAMPLE_CACHE) > max(1, int(max_cache_size)):
        _SAMPLE_CACHE.popitem(last=False)
    return cached


def _cached_gradient_for_sample(
    sample: Mapping[str, Any],
    cached_sample: dict[str, Any],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    gradient = cached_sample.get("gradient_full")
    if gradient is None:
        loaded = _load_targets_for_gradient(sample, image_shape)
        gradient = loaded.get("gradient_full")
        cached_sample["gradient_full"] = gradient
    return {} if gradient is None else {"gradient_full": gradient}


def _load_reconstruction_maps(path: Path, *, role: str, max_cache_size: int = 2) -> dict[str, np.ndarray]:
    key = (str(path.resolve()), str(role))
    cached = _RECONSTRUCTION_CACHE.get(key)
    if cached is not None:
        _RECONSTRUCTION_CACHE.move_to_end(key)
        return cached
    depth_key = f"depth_{role}"
    required = {"support_mask", f"{role}_pose_x_map", f"{role}_pose_y_map", depth_key}
    with np.load(path) as data:
        arrays = {
            name: data[name]
            for name in required
            if name in data.files
        }
        if depth_key not in arrays and "depth_front" in data.files:
            arrays[depth_key] = data["depth_front"]
        if "support_mask" not in arrays:
            raise KeyError(f"missing support_mask in reconstruction maps: {path}")
    _RECONSTRUCTION_CACHE[key] = arrays
    _RECONSTRUCTION_CACHE.move_to_end(key)
    while len(_RECONSTRUCTION_CACHE) > max(1, int(max_cache_size)):
        _RECONSTRUCTION_CACHE.popitem(last=False)
    return arrays


def reconstruction_cache_info() -> dict[str, Any]:
    return {
        "size": len(_RECONSTRUCTION_CACHE),
        "keys": [f"{Path(path).name}:{role}" for path, role in _RECONSTRUCTION_CACHE.keys()],
    }


def _load_targets_for_gradient(sample: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    targets_path = sample.get("targets_path")
    if targets_path is None:
        return {}
    with np.load(Path(targets_path)) as data:
        if "gradient" not in data:
            return {}
        gradient = np.asarray(data["gradient"], dtype=np.float32)
    if gradient.ndim == 3 and gradient.shape[0] == 2:
        gx = cv2.resize(gradient[0], (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        gy = cv2.resize(gradient[1], (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        return {"gradient_full": np.stack([gx, gy], axis=-1).astype(np.float32)}
    return {}


def _affine_matrix(width: int, height: int, yaw_deg: float, dx: float, dy: float) -> np.ndarray:
    center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, float(yaw_deg), 1.0).astype(np.float32)
    matrix[0, 2] += float(dx)
    matrix[1, 2] += float(dy)
    return matrix


def _warp_orientation_2d(orientation: np.ndarray, mask: np.ndarray, matrix: np.ndarray, yaw_deg: float) -> np.ndarray:
    cos2 = np.cos(2.0 * orientation).astype(np.float32) * (mask > 0).astype(np.float32)
    sin2 = np.sin(2.0 * orientation).astype(np.float32) * (mask > 0).astype(np.float32)
    height, width = orientation.shape
    cos2_w = cv2.warpAffine(cos2, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sin2_w = cv2.warpAffine(sin2, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = 0.5 * np.arctan2(sin2_w, cos2_w)
    return tr.normalize_angle_pi(warped + math.radians(yaw_deg))


def _warp_gradient_2d(gradient: np.ndarray | None, matrix: np.ndarray, yaw_deg: float, shape: tuple[int, int]) -> np.ndarray | None:
    if gradient is None:
        return None
    height, width = shape
    gx = cv2.warpAffine(gradient[:, :, 0], matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gy = cv2.warpAffine(gradient[:, :, 1], matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([c * gx - s * gy, s * gx + c * gy], axis=-1).astype(np.float32)


def _transform_minutiae_affine(
    minutiae: list[dict[str, Any]],
    matrix: np.ndarray,
    theta_delta: float,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = mask.shape
    transformed: list[dict[str, Any]] = []
    for minutia in minutiae:
        x = float(minutia.get("x", float("nan")))
        y = float(minutia.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xp = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
        yp = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
        if not _point_in_mask(mask, xp, yp):
            continue
        item = {key: value for key, value in minutia.items() if key not in {"x", "y", "theta"}}
        item.update({"x": xp, "y": yp, "theta": tr.normalize_angle_2pi_scalar(float(minutia.get("theta", 0.0)) + theta_delta)})
        if 0.0 <= xp < width and 0.0 <= yp < height:
            transformed.append(item)
    return transformed


def _role_for_raw_view(raw_view_index: Any) -> str:
    return {0: "front", 1: "left", 2: "right"}.get(int(raw_view_index), "front")


def _fill_nearest_holes(
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    orientation: np.ndarray,
    ridge: np.ndarray,
) -> None:
    source_mask = mask > 0
    if not np.any(source_mask):
        return
    kernel = np.ones((3, 3), dtype=np.uint8)
    fill_mask = (cv2.dilate(source_mask.astype(np.uint8), kernel, iterations=1) > 0) & ~source_mask
    if not np.any(fill_mask):
        return

    image_d = cv2.dilate(image.astype(np.float32), kernel, iterations=1)
    finite_depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    depth_d = cv2.dilate(finite_depth, kernel, iterations=1)
    orientation_d = cv2.dilate(orientation.astype(np.float32), kernel, iterations=1)
    ridge_d = cv2.dilate(ridge.astype(np.float32), kernel, iterations=1)
    image[fill_mask] = image_d[fill_mask]
    mask[fill_mask] = 255
    depth[fill_mask] = depth_d[fill_mask]
    orientation[fill_mask] = orientation_d[fill_mask]
    ridge[fill_mask] = ridge_d[fill_mask]


def _role_surface(reconstruction: dict[str, np.ndarray], role: str, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = np.asarray(reconstruction.get("support_mask"), dtype=np.uint8) > 0
    height, width = support.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    base_x = np.asarray(reconstruction.get(f"{role}_pose_x_map", xx), dtype=np.float32)
    base_y = np.asarray(reconstruction.get(f"{role}_pose_y_map", yy), dtype=np.float32)
    depth = np.asarray(reconstruction.get(f"depth_{role}", reconstruction.get("depth_front", np.zeros_like(base_x))), dtype=np.float32)
    valid = support & np.isfinite(base_x) & np.isfinite(base_y) & np.isfinite(depth)
    if (height, width) != image_shape:
        scale_x = float(image_shape[1]) / float(max(width, 1))
        scale_y = float(image_shape[0]) / float(max(height, 1))
        base_x = cv2.resize(base_x, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR) * scale_x
        base_y = cv2.resize(base_y, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR) * scale_y
        depth = cv2.resize(depth, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return base_x.astype(np.float32), base_y.astype(np.float32), np.nan_to_num(depth, nan=0.0).astype(np.float32), valid


def _pose_depth_image(base_x: np.ndarray, base_y: np.ndarray, depth: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    zbuf = np.full((height, width), -np.inf, dtype=np.float32)
    ys, xs = np.nonzero(valid)
    xp = np.rint(base_x[ys, xs]).astype(np.int32)
    yp = np.rint(base_y[ys, xs]).astype(np.int32)
    ok = (xp >= 0) & (xp < width) & (yp >= 0) & (yp < height)
    for x, y, z in zip(xp[ok], yp[ok], depth[ys[ok], xs[ok]], strict=False):
        if z >= zbuf[y, x]:
            zbuf[y, x] = z
            out[y, x] = z
    return out


def _transform_minutiae_3d(
    *,
    minutiae: list[dict[str, Any]],
    depth_image: np.ndarray,
    r: np.ndarray,
    cx: float,
    cy: float,
    depth_center: float,
    dx: float,
    dy: float,
    rendered_mask: np.ndarray,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    delta = 3.0
    for minutia in minutiae:
        x = float(minutia.get("x", float("nan")))
        y = float(minutia.get("y", float("nan")))
        theta = float(minutia.get("theta", 0.0))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
            continue
        z = _sample_depth(depth_image, x, y)
        if z is None:
            continue
        p = _project_3d_point(x, y, z, r, cx, cy, depth_center, dx, dy)
        if not _point_in_mask(rendered_mask, p[0], p[1]):
            continue
        x2 = x + math.cos(theta) * delta
        y2 = y + math.sin(theta) * delta
        z2 = _sample_depth(depth_image, x2, y2)
        if z2 is None:
            z2 = z
        p2 = _project_3d_point(x2, y2, z2, r, cx, cy, depth_center, dx, dy)
        theta_new = math.atan2(float(p2[1] - p[1]), float(p2[0] - p[0]))
        item = {key: value for key, value in minutia.items() if key not in {"x", "y", "theta"}}
        item.update({"x": float(p[0]), "y": float(p[1]), "theta": tr.normalize_angle_2pi_scalar(theta_new)})
        transformed.append(item)
    return transformed


def _project_3d_point(x: float, y: float, z: float, r: np.ndarray, cx: float, cy: float, depth_center: float, dx: float, dy: float) -> np.ndarray:
    point = np.asarray([x - cx, y - cy, z - depth_center], dtype=np.float32)
    rotated = r @ point
    return np.asarray([rotated[0] + cx + dx, rotated[1] + cy + dy, rotated[2] + depth_center], dtype=np.float32)


def _orientation_delta_for_points(theta: np.ndarray, r: np.ndarray) -> np.ndarray:
    tangent = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=0).astype(np.float32)
    projected = r @ tangent
    return tr.normalize_angle_pi(np.arctan2(projected[1], projected[0]))


def _sample_bilinear(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = array.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    return (
        array[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + array[y0, x1] * wx * (1.0 - wy)
        + array[y1, x0] * (1.0 - wx) * wy
        + array[y1, x1] * wx * wy
    ).astype(np.float32)


def _sample_nearest(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = array.shape
    xi = np.clip(np.rint(x).astype(np.int32), 0, width - 1)
    yi = np.clip(np.rint(y).astype(np.int32), 0, height - 1)
    return array[yi, xi]


def _sample_orientation(orientation: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cos2 = _sample_bilinear(np.cos(2.0 * orientation).astype(np.float32), x, y)
    sin2 = _sample_bilinear(np.sin(2.0 * orientation).astype(np.float32), x, y)
    return tr.normalize_angle_pi(0.5 * np.arctan2(sin2, cos2))


def _sample_depth(depth: np.ndarray, x: float, y: float) -> float | None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    height, width = depth.shape
    if x < 0.0 or y < 0.0 or x >= width or y >= height:
        return None
    value = float(_sample_bilinear(depth, np.asarray([x], dtype=np.float32), np.asarray([y], dtype=np.float32))[0])
    return value if math.isfinite(value) else None


def _point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    height, width = mask.shape
    cx = int(round(x))
    cy = int(round(y))
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return False
    y0 = max(0, cy - 1)
    y1 = min(height, cy + 2)
    x0 = max(0, cx - 1)
    x1 = min(width, cx + 2)
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _finite_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def _draw_minutiae(image: np.ndarray, mask: np.ndarray, minutiae: list[dict[str, Any]]) -> np.ndarray:
    canvas = cv2.cvtColor(_finite_u8(image), cv2.COLOR_GRAY2BGR)
    canvas[mask <= 0] = (canvas[mask <= 0] * 0.35).astype(np.uint8)
    for minutia in minutiae:
        x = float(minutia.get("x", float("nan")))
        y = float(minutia.get("y", float("nan")))
        theta = float(minutia.get("theta", 0.0))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        p0 = (int(round(x)), int(round(y)))
        p1 = (int(round(x + math.cos(theta) * 8.0)), int(round(y + math.sin(theta) * 8.0)))
        cv2.arrowedLine(canvas, p0, p1, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    return canvas
