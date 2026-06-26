from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np

from . import target_rasterization as tr


@dataclass(slots=True)
class AugmentationConfig:
    count: int = 5
    translation_typical_px: float = 16.0
    translation_strong_px: float = 32.0
    yaw_deg: float = 15.0
    pitch_roll_typical_deg: float = 15.0
    pitch_roll_strong_deg: float = 25.0
    missing_reconstruction: str = "skip"
    debug_dir: Path | None = None
    debug_limit: int = 0
    reconstruction_cache_size: int = 2
    sample_cache_size: int = 8
    group_variants: bool = False
    # Photometric + uniform-scale augmentation (applied to synthetic variants only).
    photometric: bool = True
    yaw_prob: float = 0.70
    scale_jitter_min: float = 0.90
    scale_jitter_max: float = 1.10
    brightness_max: float = 30.0
    contrast_delta: float = 0.25
    gamma_min: float = 0.70
    gamma_max: float = 1.50
    blur_sigma_max: float = 1.60
    blur_prob: float = 0.40
    noise_std_max: float = 10.0
    noise_prob: float = 0.50
    jpeg_quality_min: int = 35
    jpeg_prob: float = 0.40
    illum_strength_max: float = 0.35
    illum_prob: float = 0.40
    glare_prob: float = 0.20
    glare_intensity_max: float = 120.0


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
    # Uniform in-plane scale (2D-affine path only); 1.0 is identity.
    scale: float = 1.0
    # Photometric parameters; neutral defaults are a no-op.
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    noise_seed: int = 0
    jpeg_quality: int = 0
    illum_strength: float = 0.0
    illum_angle: float = 0.0
    glare_intensity: float = 0.0
    glare_cx: float = 0.5
    glare_cy: float = 0.5
    glare_radius: float = 0.2

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
    yaw = bool(rng.random() < config.yaw_prob)
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

    photometric = _sample_photometric_params(rng, config)

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
        **photometric,
    )


def _sample_photometric_params(rng: np.random.Generator, config: AugmentationConfig) -> dict[str, Any]:
    """Sample appearance + scale jitter. Neutral when photometric is disabled."""
    neutral = {
        "scale": 1.0,
        "brightness": 0.0,
        "contrast": 1.0,
        "gamma": 1.0,
        "blur_sigma": 0.0,
        "noise_std": 0.0,
        "noise_seed": 0,
        "jpeg_quality": 0,
        "illum_strength": 0.0,
        "illum_angle": 0.0,
        "glare_intensity": 0.0,
        "glare_cx": 0.5,
        "glare_cy": 0.5,
        "glare_radius": 0.2,
    }
    if not config.photometric:
        return neutral

    values = dict(neutral)
    values["scale"] = float(rng.uniform(config.scale_jitter_min, config.scale_jitter_max))
    if rng.random() < 0.70:
        values["brightness"] = float(rng.uniform(-config.brightness_max, config.brightness_max))
    if rng.random() < 0.70:
        values["contrast"] = float(rng.uniform(1.0 - config.contrast_delta, 1.0 + config.contrast_delta))
    if rng.random() < 0.60:
        values["gamma"] = float(rng.uniform(config.gamma_min, config.gamma_max))
    if rng.random() < config.blur_prob:
        values["blur_sigma"] = float(rng.uniform(0.4, config.blur_sigma_max))
    if rng.random() < config.noise_prob:
        values["noise_std"] = float(rng.uniform(0.3 * config.noise_std_max, config.noise_std_max))
        values["noise_seed"] = int(rng.integers(0, 2**31 - 1))
    if rng.random() < config.jpeg_prob:
        values["jpeg_quality"] = int(rng.integers(config.jpeg_quality_min, 96))
    if rng.random() < config.illum_prob:
        values["illum_strength"] = float(rng.uniform(0.1, config.illum_strength_max))
        values["illum_angle"] = float(rng.uniform(0.0, 2.0 * math.pi))
    if rng.random() < config.glare_prob:
        values["glare_intensity"] = float(rng.uniform(40.0, config.glare_intensity_max))
        values["glare_cx"] = float(rng.uniform(0.2, 0.8))
        values["glare_cy"] = float(rng.uniform(0.2, 0.8))
        values["glare_radius"] = float(rng.uniform(0.08, 0.20))
    return values


def apply_photometric(image: np.ndarray, mask: np.ndarray, params: AugmentationParams) -> np.ndarray:
    """Label-preserving appearance jitter on a masked grayscale uint8 image.

    Geometry is untouched, so FeatureNet targets remain valid. Background stays 0.
    """
    foreground = mask > 0
    if not np.any(foreground):
        return image

    img = image.astype(np.float32)
    if params.contrast != 1.0 or params.brightness != 0.0:
        mean = float(img[foreground].mean())
        img = (img - mean) * float(params.contrast) + mean + float(params.brightness)
    if params.gamma != 1.0:
        norm = np.clip(img, 0.0, 255.0) / 255.0
        img = np.power(norm, float(params.gamma)) * 255.0
    if params.illum_strength > 0.0:
        height, width = img.shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / max(width - 1, 1)) - 0.5
        ny = (yy / max(height - 1, 1)) - 0.5
        ramp = math.cos(params.illum_angle) * nx + math.sin(params.illum_angle) * ny
        img = img * (1.0 + float(params.illum_strength) * ramp)
    if params.glare_intensity > 0.0:
        height, width = img.shape
        cx = float(params.glare_cx) * width
        cy = float(params.glare_cy) * height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        sigma = max(float(params.glare_radius) * max(height, width), 1.0)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        img = img + float(params.glare_intensity) * blob
    img = np.clip(img, 0.0, 255.0)
    if params.blur_sigma > 0.0:
        img = cv2.GaussianBlur(img, (0, 0), float(params.blur_sigma))
    if params.noise_std > 0.0:
        noise_rng = np.random.default_rng(int(params.noise_seed))
        img = img + noise_rng.normal(0.0, float(params.noise_std), size=img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 255.0)
    out = np.clip(img, 0.0, 255.0).astype(np.uint8)
    if params.jpeg_quality > 0:
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(params.jpeg_quality)])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                out = decoded.astype(np.uint8)
    out[~foreground] = 0
    return out


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
    single_source_candidates = cached_sample["single_source_candidates"]
    use_3d = params.has_3d and has_usable_reconstruction(sample)
    base_targets = _cached_gradient_for_sample(sample, cached_sample, image.shape)
    gradient_full = base_targets.get("gradient_full")

    if use_3d:
        role = str(sample["reconstruction"].get("role", _role_for_raw_view(sample.get("raw_view_index"))))
        rendered = render_3d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge_period,
            gradient=gradient_full,
            minutiae=minutiae,
            single_source_candidates=single_source_candidates,
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
            gradient=gradient_full,
            minutiae=minutiae,
            single_source_candidates=single_source_candidates,
            params=params,
        )

    rendered["image"] = apply_photometric(rendered["image"], rendered["mask"], params)

    extractor_config = _sample_target_config(sample)
    targets = tr.build_featurenet_targets(
        rendered["image"],
        rendered["mask"],
        rendered["orientation"],
        rendered["ridge_period"],
        rendered["gradient"]
        if rendered["gradient"] is not None
        else np.zeros((*rendered["image"].shape, 2), dtype=np.float32),
        rendered["minutiae"],
        single_source_candidates=rendered.get("single_source_candidates"),
        extractor_config=extractor_config,
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
    single_source_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    height, width = image.shape
    scale = float(params.scale)
    matrix = _affine_matrix(width, height, params.yaw_deg, params.dx, params.dy, scale)
    warped_image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_orientation = _warp_orientation_2d(orientation, mask, matrix, params.yaw_deg)
    # Uniform zoom scales the apparent ridge period linearly.
    warped_ridge = cv2.warpAffine(ridge_period.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * scale
    warped_gradient = _warp_gradient_2d(gradient, matrix, params.yaw_deg, image.shape) if gradient is not None else None
    if warped_gradient is not None and scale != 1.0:
        # Surface slope measured in screen space shrinks as the image is enlarged.
        warped_gradient = (warped_gradient / scale).astype(np.float32)
    transformed_minutiae = _transform_minutiae_affine(minutiae, matrix, math.radians(params.yaw_deg), warped_mask)
    transformed_single_source = _transform_minutiae_affine(
        single_source_candidates or [], matrix, math.radians(params.yaw_deg), warped_mask
    )
    return {
        "image": _finite_u8(warped_image),
        "mask": np.where(warped_mask > 0, 255, 0).astype(np.uint8),
        "orientation": warped_orientation,
        "ridge_period": np.nan_to_num(warped_ridge, nan=0.0).astype(np.float32),
        "gradient": warped_gradient,
        "minutiae": transformed_minutiae,
        "single_source_candidates": transformed_single_source,
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
    single_source_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    height, width = image.shape
    base_x, base_y, depth, valid = _role_surface(reconstruction, role, image.shape)
    depth_center = float(np.nanmedian(depth[valid])) if np.any(valid) else 0.0
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    r = rotation_matrix(params.yaw_deg, params.pitch_deg, params.roll_deg)

    # Forward-project every grid vertex of the role surface (height field z=depth)
    # into screen space; invalid vertices are flagged with NaN so the rasterizer
    # never builds a triangle that touches them.
    points = np.stack([base_x - cx, base_y - cy, depth - depth_center], axis=0).astype(np.float32)
    rotated = np.einsum("ij,jhw->ihw", r, points).astype(np.float32)
    proj_x = rotated[0] + cx + float(params.dx)
    proj_y = rotated[1] + cy + float(params.dy)
    proj_z = rotated[2] + depth_center
    invalid = ~valid
    proj_x[invalid] = np.nan
    proj_y[invalid] = np.nan
    proj_z[invalid] = np.nan

    # Z-buffered mesh rasterization yields, per output pixel, the interpolated
    # source coordinate (sx_map, sy_map) -- the dense inverse map -- plus coverage.
    source_map_x, source_map_y, coverage, _ = _rasterize_source_map(
        proj_x, proj_y, proj_z, base_x, base_y, valid, height, width
    )

    if not np.any(coverage):
        return render_2d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge_period,
            gradient=gradient,
            minutiae=minutiae,
            single_source_candidates=single_source_candidates,
            params=params,
        )

    map_x = source_map_x.astype(np.float32)
    map_y = source_map_y.astype(np.float32)
    remap_kwargs = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rendered_image = cv2.remap(image.astype(np.float32), map_x, map_y, **remap_kwargs)
    cos2 = cv2.remap(np.cos(2.0 * orientation).astype(np.float32), map_x, map_y, **remap_kwargs)
    sin2 = cv2.remap(np.sin(2.0 * orientation).astype(np.float32), map_x, map_y, **remap_kwargs)
    theta_src = (0.5 * np.arctan2(sin2, cos2)).astype(np.float32)
    ridge_src = cv2.remap(ridge_period.astype(np.float32), map_x, map_y, **remap_kwargs)
    if gradient is not None:
        gx_src = cv2.remap(np.ascontiguousarray(gradient[:, :, 0], dtype=np.float32), map_x, map_y, **remap_kwargs)
        gy_src = cv2.remap(np.ascontiguousarray(gradient[:, :, 1], dtype=np.float32), map_x, map_y, **remap_kwargs)
    else:
        gx_src = np.zeros((height, width), dtype=np.float32)
        gy_src = np.zeros((height, width), dtype=np.float32)

    # Closed-form geometric transforms of the dense labels under the pose change.
    rendered_orientation = _transform_orientation_field(theta_src, gx_src, gy_src, r)
    rendered_gradient = _transform_gradient_field(gx_src, gy_src, r)
    rendered_ridge = _foreshorten_ridge_field(ridge_src, theta_src, gx_src, gy_src, r)

    # Erode coverage by 1px so partially-covered silhouette cells are not supervised.
    coverage_u8 = coverage.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    rendered_mask = np.where(cv2.erode(coverage_u8, kernel, iterations=1) > 0, 255, 0).astype(np.uint8)

    # Keep finite values everywhere inside the rendered surface; the eroded mask
    # passed to build_featurenet_targets is what actually gates supervision.
    rendered_image = np.where(coverage, rendered_image, 0.0).astype(np.float32)
    rendered_orientation = np.where(coverage, rendered_orientation, 0.0).astype(np.float32)
    rendered_ridge = np.where(coverage, rendered_ridge, 0.0).astype(np.float32)
    rendered_gradient[~coverage] = 0.0

    pose_depth = _pose_depth_image(base_x, base_y, depth, valid, image.shape)
    transformed_minutiae = _transform_minutiae_3d(
        minutiae=minutiae,
        depth_image=pose_depth,
        r=r,
        cx=cx,
        cy=cy,
        depth_center=depth_center,
        dx=params.dx,
        dy=params.dy,
        rendered_mask=rendered_mask,
        gradient=gradient,
    )
    transformed_single_source = _transform_minutiae_3d(
        minutiae=single_source_candidates or [],
        depth_image=pose_depth,
        r=r,
        cx=cx,
        cy=cy,
        depth_center=depth_center,
        dx=params.dx,
        dy=params.dy,
        rendered_mask=rendered_mask,
        gradient=gradient,
    )
    return {
        "image": _finite_u8(rendered_image),
        "mask": rendered_mask,
        "orientation": np.nan_to_num(rendered_orientation, nan=0.0).astype(np.float32),
        "ridge_period": np.nan_to_num(rendered_ridge, nan=0.0).astype(np.float32),
        "gradient": np.nan_to_num(rendered_gradient, nan=0.0).astype(np.float32),
        "minutiae": transformed_minutiae,
        "single_source_candidates": transformed_single_source,
        "details": {
            "mode": "3d_orthographic_mesh",
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
        "single_source_candidates": _load_minutiae(Path(sample.get("single_source_candidates_path", ""))),
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


def _affine_matrix(width: int, height: int, yaw_deg: float, dx: float, dy: float, scale: float = 1.0) -> np.ndarray:
    center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, float(yaw_deg), float(scale)).astype(np.float32)
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


def _rasterize_source_map(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    proj_z: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    valid: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffered rasterization of the grid mesh.

    Returns dense maps ``(sx, sy)`` giving, per output pixel, the interpolated
    source coordinate to sample from, a boolean coverage mask, and the projected
    depth of the winning (nearest) triangle. Out-of-coverage source coordinates
    are set to ``-1`` so a subsequent ``cv2.remap`` reads the border value.
    """
    sx_map = np.full((height, width), -1.0, dtype=np.float32)
    sy_map = np.full((height, width), -1.0, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    depth_map = np.full((height, width), np.nan, dtype=np.float32)

    v00 = valid[:-1, :-1]
    v01 = valid[:-1, 1:]
    v10 = valid[1:, :-1]
    v11 = valid[1:, 1:]
    quad = v00 & v01 & v10 & v11
    if not np.any(quad):
        return sx_map, sy_map, coverage, depth_map

    top_left = (slice(0, -1), slice(0, -1))
    top_right = (slice(0, -1), slice(1, None))
    bottom_left = (slice(1, None), slice(0, -1))
    bottom_right = (slice(1, None), slice(1, None))

    def gather(arr: np.ndarray, idx: tuple[slice, slice]) -> np.ndarray:
        return arr[idx][quad].astype(np.float32)

    # Each valid quad splits into triangle1 = (TL, TR, BR) and triangle2 = (TL, BR, BL).
    ax = np.concatenate([gather(proj_x, top_left), gather(proj_x, top_left)])
    ay = np.concatenate([gather(proj_y, top_left), gather(proj_y, top_left)])
    az = np.concatenate([gather(proj_z, top_left), gather(proj_z, top_left)])
    asx = np.concatenate([gather(src_x, top_left), gather(src_x, top_left)])
    asy = np.concatenate([gather(src_y, top_left), gather(src_y, top_left)])
    bx = np.concatenate([gather(proj_x, top_right), gather(proj_x, bottom_right)])
    by = np.concatenate([gather(proj_y, top_right), gather(proj_y, bottom_right)])
    bz = np.concatenate([gather(proj_z, top_right), gather(proj_z, bottom_right)])
    bsx = np.concatenate([gather(src_x, top_right), gather(src_x, bottom_right)])
    bsy = np.concatenate([gather(src_y, top_right), gather(src_y, bottom_right)])
    cx = np.concatenate([gather(proj_x, bottom_right), gather(proj_x, bottom_left)])
    cy = np.concatenate([gather(proj_y, bottom_right), gather(proj_y, bottom_left)])
    cz = np.concatenate([gather(proj_z, bottom_right), gather(proj_z, bottom_left)])
    csx = np.concatenate([gather(src_x, bottom_right), gather(src_x, bottom_left)])
    csy = np.concatenate([gather(src_y, bottom_right), gather(src_y, bottom_left)])

    _raster_triangles_into(
        sx_map, sy_map, coverage, depth_map,
        ax, ay, az, asx, asy,
        bx, by, bz, bsx, bsy,
        cx, cy, cz, csx, csy,
        height, width,
    )
    return sx_map, sy_map, coverage, depth_map


def _raster_triangles_into(
    sx_map: np.ndarray,
    sy_map: np.ndarray,
    coverage: np.ndarray,
    depth_map: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray, asx: np.ndarray, asy: np.ndarray,
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray, bsx: np.ndarray, bsy: np.ndarray,
    cx: np.ndarray, cy: np.ndarray, cz: np.ndarray, csx: np.ndarray, csy: np.ndarray,
    height: int,
    width: int,
) -> None:
    finite = (
        np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
        & np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz)
        & np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
    )
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    keep = finite & (np.abs(denom) > 1e-8)
    if not np.any(keep):
        return
    ax, ay, az, asx, asy = ax[keep], ay[keep], az[keep], asx[keep], asy[keep]
    bx, by, bz, bsx, bsy = bx[keep], by[keep], bz[keep], bsx[keep], bsy[keep]
    cx, cy, cz, csx, csy = cx[keep], cy[keep], cz[keep], csx[keep], csy[keep]
    denom = denom[keep]

    x0 = np.clip(np.floor(np.minimum(np.minimum(ax, bx), cx)), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(np.maximum(np.maximum(ax, bx), cx)), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(np.minimum(np.minimum(ay, by), cy)), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.ceil(np.maximum(np.maximum(ay, by), cy)), 0, height - 1).astype(np.int64)
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    area = (box_w * box_h).astype(np.int64)
    total = int(area.sum())
    if total <= 0:
        return

    tri = np.repeat(np.arange(area.size, dtype=np.int64), area)
    starts = np.zeros(area.size, dtype=np.int64)
    if area.size > 1:
        np.cumsum(area[:-1], out=starts[1:])
    local = np.arange(total, dtype=np.int64) - starts[tri]
    box_w_rep = box_w[tri]
    px = x0[tri] + (local % box_w_rep)
    py = y0[tri] + (local // box_w_rep)
    pxf = px.astype(np.float32)
    pyf = py.astype(np.float32)

    axt, ayt = ax[tri], ay[tri]
    bxt, byt = bx[tri], by[tri]
    cxt, cyt = cx[tri], cy[tri]
    dent = denom[tri]
    l1 = ((byt - cyt) * (pxf - cxt) + (cxt - bxt) * (pyf - cyt)) / dent
    l2 = ((cyt - ayt) * (pxf - cxt) + (axt - cxt) * (pyf - cyt)) / dent
    l3 = 1.0 - l1 - l2
    edge_eps = -1e-4
    inside = (l1 >= edge_eps) & (l2 >= edge_eps) & (l3 >= edge_eps)
    if not np.any(inside):
        return
    tri_i = tri[inside]
    px = px[inside]
    py = py[inside]
    l1 = l1[inside].astype(np.float32)
    l2 = l2[inside].astype(np.float32)
    l3 = l3[inside].astype(np.float32)

    zc = l1 * az[tri_i] + l2 * bz[tri_i] + l3 * cz[tri_i]
    sxc = l1 * asx[tri_i] + l2 * bsx[tri_i] + l3 * csx[tri_i]
    syc = l1 * asy[tri_i] + l2 * bsy[tri_i] + l3 * csy[tri_i]
    flat = py.astype(np.int64) * int(width) + px.astype(np.int64)

    zbuf = np.full(height * width, -np.inf, dtype=np.float32)
    np.maximum.at(zbuf, flat, zc.astype(np.float32))
    winner = zc >= (zbuf[flat] - 1e-4)
    flat_w = flat[winner]

    sx_map.reshape(-1)[flat_w] = sxc[winner]
    sy_map.reshape(-1)[flat_w] = syc[winner]
    coverage.reshape(-1)[flat_w] = True
    depth_map.reshape(-1)[flat_w] = zc[winner]


def _transform_orientation_field(theta: np.ndarray, gx: np.ndarray, gy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Lift the undirected ridge tangent onto the surface, rotate by R, reproject."""
    ct = np.cos(theta)
    st = np.sin(theta)
    tz = gx * ct + gy * st
    tx = r[0, 0] * ct + r[0, 1] * st + r[0, 2] * tz
    ty = r[1, 0] * ct + r[1, 1] * st + r[1, 2] * tz
    return tr.normalize_angle_pi(np.arctan2(ty, tx)).astype(np.float32)


def _transform_gradient_field(gx: np.ndarray, gy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Transform the surface gradient via the rotated surface normal n=(-gx,-gy,1)."""
    nx = -gx
    ny = -gy
    nz = np.ones_like(gx)
    n_x = r[0, 0] * nx + r[0, 1] * ny + r[0, 2] * nz
    n_y = r[1, 0] * nx + r[1, 1] * ny + r[1, 2] * nz
    n_z = r[2, 0] * nx + r[2, 1] * ny + r[2, 2] * nz
    eps = 1e-3
    n_z_safe = np.where(np.abs(n_z) < eps, eps, n_z)
    new_gx = -n_x / n_z_safe
    new_gy = -n_y / n_z_safe
    return np.stack([new_gx, new_gy], axis=-1).astype(np.float32)


def _foreshorten_ridge_field(
    ridge: np.ndarray, theta: np.ndarray, gx: np.ndarray, gy: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Scale ridge period by the screen/source Jacobian along the ridge-normal.

    The screen-vs-source Jacobian of the orthographic projection of the height
    field is J = R[:2,:2] + R[:2,2] (x) (gx, gy). Period is measured along the
    ridge-normal n_hat=(-sin, cos); the apparent period scales by ||J @ n_hat||.
    """
    j00 = r[0, 0] + r[0, 2] * gx
    j01 = r[0, 1] + r[0, 2] * gy
    j10 = r[1, 0] + r[1, 2] * gx
    j11 = r[1, 1] + r[1, 2] * gy
    nhx = -np.sin(theta)
    nhy = np.cos(theta)
    vx = j00 * nhx + j01 * nhy
    vy = j10 * nhx + j11 * nhy
    factor = np.sqrt(vx * vx + vy * vy)
    return (ridge * factor).astype(np.float32)


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
    gradient: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
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
        # Lift the directional minutia tangent onto the surface using the local
        # gradient, rotate by R, then read the screen-space direction.
        gx, gy = _sample_gradient(gradient, x, y)
        ct = math.cos(theta)
        st = math.sin(theta)
        tz = gx * ct + gy * st
        tangent = r @ np.asarray([ct, st, tz], dtype=np.float32)
        theta_new = math.atan2(float(tangent[1]), float(tangent[0]))
        item = {key: value for key, value in minutia.items() if key not in {"x", "y", "theta"}}
        item.update({"x": float(p[0]), "y": float(p[1]), "theta": tr.normalize_angle_2pi_scalar(theta_new)})
        transformed.append(item)
    return transformed


def _project_3d_point(x: float, y: float, z: float, r: np.ndarray, cx: float, cy: float, depth_center: float, dx: float, dy: float) -> np.ndarray:
    point = np.asarray([x - cx, y - cy, z - depth_center], dtype=np.float32)
    rotated = r @ point
    return np.asarray([rotated[0] + cx + dx, rotated[1] + cy + dy, rotated[2] + depth_center], dtype=np.float32)


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


def _sample_gradient(gradient: np.ndarray | None, x: float, y: float) -> tuple[float, float]:
    if gradient is None or not (math.isfinite(x) and math.isfinite(y)):
        return 0.0, 0.0
    coords_x = np.asarray([x], dtype=np.float32)
    coords_y = np.asarray([y], dtype=np.float32)
    gx = float(_sample_bilinear(np.ascontiguousarray(gradient[:, :, 0], dtype=np.float32), coords_x, coords_y)[0])
    gy = float(_sample_bilinear(np.ascontiguousarray(gradient[:, :, 1], dtype=np.float32), coords_x, coords_y)[0])
    if not (math.isfinite(gx) and math.isfinite(gy)):
        return 0.0, 0.0
    return gx, gy


def _sample_target_config(sample: Mapping[str, Any]) -> Any | None:
    config = sample.get("minutiae_target_config")
    if not isinstance(config, Mapping) or not config:
        return None
    return tr.target_config_from_object(SimpleNamespace(**dict(config)))


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
