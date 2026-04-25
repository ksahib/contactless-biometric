from __future__ import annotations

import argparse
import csv
import json
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest


def ensure_stdlib_copy_module() -> None:
    existing = sys.modules.get("copy")
    repo_root = Path(__file__).resolve().parent
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None:
            return
        if Path(existing_path).resolve().parent != repo_root:
            return
        del sys.modules["copy"]

    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pyfing
from center_unwarping import run_center_unwarping

try:
    from rembg import new_session, remove

    _HAS_REMBG = True
except Exception:  # pragma: no cover - optional at import time
    new_session = None
    remove = None
    _HAS_REMBG = False

try:
    import fingerprint_enhancer
except Exception:  # pragma: no cover - optional at import time
    fingerprint_enhancer = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "DS1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ground_truth" / "DS1"
DEFAULT_FINGERFLOW_MODEL_DIR = REPO_ROOT / ".fingerflow_models"
VARIANT_SUFFIXES = ("HT1", "HT2", "HT4", "HT6", "R414")
DEFAULT_DPI = 500
MINUTIA_SCORE_THRESHOLD = 0.15
MIN_RASTERIZED_MINUTIAE_FOR_RECONSTRUCTION = 1
REMBG_MODEL = os.environ.get("FINGER_REMBG_MODEL", "u2netp")
FINGERFLOW_MODEL_SOURCES = {
    "coarse": {
        "filename": "CoarseNet.h5",
        "urls": (
            "https://www.dropbox.com/s/gppil4wybdjcihy/CoarseNet.h5?dl=1",
            "https://drive.google.com/uc?export=download&id=1alvw_kAyY4sxdzAkGABQR7waux-rgJKm",
        ),
    },
    "fine": {
        "filename": "FineNet.h5",
        "urls": (
            "https://www.dropbox.com/s/k7q2vs9255jf2dh/FineNet.h5?dl=1",
            "https://drive.google.com/uc?export=download&id=1wdGZKNNDAyN-fajjVKJoiyDtXAvl-4zq",
        ),
    },
    "classify": {
        "filename": "ClassifyNet.h5",
        "urls": (
            "https://drive.usercontent.google.com/download?id=1dfQDW8yxjmFPVu0Ddui2voxdngOrU3rc&export=download&confirm=t",
            "https://drive.google.com/uc?export=download&id=1dfQDW8yxjmFPVu0Ddui2voxdngOrU3rc",
        ),
    },
    "core": {
        "filename": "CoreNet.h5",
        "urls": (
            "https://drive.usercontent.google.com/download?id=1v091s0eY4_VOLU9BqDXVSaZcFnA9qJPl&export=download&confirm=t",
            "https://drive.google.com/uc?export=download&id=1v091s0eY4_VOLU9BqDXVSaZcFnA9qJPl",
        ),
    },
}

_REMBG_SESSION = None
_EXTRACTOR_CACHE: dict[tuple[str, str, str, str], Any] = {}
_FINGERFLOW_MODEL_PATHS = None
_FINGERFLOW_EXTRACTOR_CLASS = None
_GENERATOR_RUNTIME: "GeneratorRuntimeConfig | None" = None
_SELECTED_REMBG_PROVIDERS: list[str] | None = None
_SELECTED_FINGERFLOW_DEVICE: str = "cpu"
_RECONSTRUCTION_MINUTIAE_CACHE: dict[str, dict[str, Any]] = {}
_PIPELINE_STAGE_TOTALS: dict[str, float] = {
    "load_input": 0.0,
    "rembg": 0.0,
    "cpu_preprocess": 0.0,
    "fingerflow": 0.0,
    "pyfing_fallback": 0.0,
    "postprocess": 0.0,
    "write_bundle": 0.0,
}


@dataclass(slots=True)
class FingerflowBackendConfig:
    backend: str
    wsl_distro: str
    wsl_activate: str


@dataclass(slots=True)
class RawViewSample:
    sample_id: str
    subject_id: int
    subject_index: int
    finger_id: int
    acquisition_id: int
    finger_class_id: int
    raw_image_path: str
    raw_view_index: int
    sire_path: str | None
    raw_view_paths: list[str]
    variant_paths: dict[str, str]
    is_extra_acquisition: bool


@dataclass(slots=True)
class PreprocessedContactlessImage:
    raw_gray: np.ndarray
    normalized_gray: np.ndarray
    pose_normalized_gray: np.ndarray
    pose_normalized_mask: np.ndarray
    preprocessed_gray: np.ndarray
    final_mask: np.ndarray
    mask_source: str
    pose_rotation_degrees: float
    ridge_scale_factor: float


@dataclass(slots=True)
class ReconstructionViewGeometry:
    role: str
    raw_image_path: str
    image_shape: tuple[int, int]
    x_left: np.ndarray
    x_right: np.ndarray
    widths: np.ndarray
    centers: np.ndarray
    valid_rows: np.ndarray


@dataclass(slots=True)
class AcquisitionReconstructionResult:
    acquisition_id: str
    reconstruction_dir: str
    depth_front_path: str
    depth_left_path: str
    depth_right_path: str
    depth_gradient_labels_path: str
    reconstruction_maps_path: str
    support_mask_path: str
    row_measurements_path: str
    meta_path: str
    preview_path: str
    center_unwarp_maps_path: str
    center_unwarped_image_path: str
    center_unwarped_mask_path: str
    surface_front_3d_html_path: str
    surface_front_3d_png_path: str
    surface_all_branches_3d_html_path: str
    surface_all_branches_3d_png_path: str
    reprojection_report_path: str
    reprojection_preview_path: str
    valid_row_count: int
    support_pixel_count: int
    input_view_paths: dict[str, str]
    debug_view_paths: dict[str, dict[str, str]]


@dataclass(slots=True)
class PatchDatasetConfig:
    patch_size: int
    minimum_mask_ratio: float


@dataclass(slots=True)
class GeneratorRuntimeConfig:
    execution_target: str
    gpu_only: bool
    gpu_batch_size: int
    cpu_workers: int
    prefetch_samples: int
    skip_existing: bool


@dataclass(slots=True)
class LoadedSampleInput:
    sample: RawViewSample
    image_path: Path
    full_bgr: np.ndarray
    visualize: bool
    bundle_dir: Path
    reconstruction: AcquisitionReconstructionResult | None


@dataclass(slots=True)
class PreparedBundleArtifacts:
    sample: RawViewSample
    bundle_dir: Path
    image_path: Path
    preprocessed: PreprocessedContactlessImage
    gray_image: np.ndarray
    mask: np.ndarray
    orientation: np.ndarray
    ridge_period: np.ndarray
    visualization_gradient: np.ndarray
    reconstruction_gradient: np.ndarray | None
    masked_image: np.ndarray
    enhanced_image: np.ndarray
    visualize: bool
    reconstruction: AcquisitionReconstructionResult | None


@dataclass(slots=True)
class BundleWritePayload:
    bundle_dir: Path
    preprocessed: PreprocessedContactlessImage
    gray_image: np.ndarray
    mask: np.ndarray
    orientation: np.ndarray
    ridge_period: np.ndarray
    visualization_gradient: np.ndarray
    masked_image: np.ndarray
    enhanced_image: np.ndarray
    minutiae: list[dict[str, Any]]
    featurenet_targets: dict[str, np.ndarray]
    meta: dict[str, Any]
    visualize: bool


def _require_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def _record_stage_time(stage: str, seconds: float) -> None:
    _PIPELINE_STAGE_TOTALS[stage] = _PIPELINE_STAGE_TOTALS.get(stage, 0.0) + max(0.0, float(seconds))


def _current_runtime_config() -> GeneratorRuntimeConfig:
    if _GENERATOR_RUNTIME is None:
        return GeneratorRuntimeConfig(
            execution_target="local",
            gpu_only=False,
            gpu_batch_size=1,
            cpu_workers=1,
            prefetch_samples=1,
            skip_existing=False,
        )
    return _GENERATOR_RUNTIME


def _bundle_outputs_exist(bundle_dir: Path) -> bool:
    required_paths = (
        bundle_dir / "meta.json",
        bundle_dir / "mask.png",
        bundle_dir / "masked_image.png",
        bundle_dir / "orientation.npy",
        bundle_dir / "ridge_period.npy",
        bundle_dir / "gradient_visualization.npy",
        bundle_dir / "minutiae.json",
        bundle_dir / "featurenet_targets.npz",
    )
    return all(path.exists() for path in required_paths)


def _available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def _resolve_rembg_providers(config: GeneratorRuntimeConfig) -> list[str] | None:
    providers = _available_onnx_providers()
    if config.execution_target == "kaggle":
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if config.gpu_only:
            raise RuntimeError(
                "Kaggle GPU run requested, but onnxruntime does not expose CUDAExecutionProvider for rembg"
            )
    return None


def _torch_cuda_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    summary: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        summary["device_name"] = torch.cuda.get_device_name(0)
    return summary


def _tensorflow_gpu_summary(require_gpu: bool = False) -> dict[str, Any]:
    global _SELECTED_FINGERFLOW_DEVICE
    try:
        import tensorflow as tf
    except Exception as exc:
        if require_gpu:
            raise RuntimeError(f"TensorFlow import failed for FingerFlow GPU path: {exc}") from exc
        _SELECTED_FINGERFLOW_DEVICE = "cpu"
        return {"available": False, "visible_gpus": [], "error": str(exc)}

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    visible = [getattr(device, "name", str(device)) for device in gpus]
    if require_gpu and not visible:
        raise RuntimeError("TensorFlow GPU is required for FingerFlow, but no visible GPU was found")
    _SELECTED_FINGERFLOW_DEVICE = "gpu" if visible else "cpu"
    return {"available": bool(visible), "visible_gpus": visible}


def _configure_runtime_environment(config: GeneratorRuntimeConfig) -> dict[str, Any]:
    global _GENERATOR_RUNTIME
    _GENERATOR_RUNTIME = config

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    tf_summary = _tensorflow_gpu_summary(require_gpu=config.gpu_only and config.execution_target == "kaggle")
    onnx_providers = _available_onnx_providers()
    if config.execution_target == "kaggle" and config.gpu_only and "CUDAExecutionProvider" not in onnx_providers:
        raise RuntimeError("Kaggle GPU run requested, but onnxruntime-gpu is not available for rembg")

    return {
        "execution_target": config.execution_target,
        "gpu_only": config.gpu_only,
        "onnx_available_providers": onnx_providers,
        "tensorflow": tf_summary,
        "torch": _torch_cuda_summary(),
    }


def _ensure_fingerflow_models(model_dir: Path):
    global _FINGERFLOW_MODEL_PATHS
    if _FINGERFLOW_MODEL_PATHS is None:
        _FINGERFLOW_MODEL_PATHS = ensure_fingerflow_models(model_dir)
    return _FINGERFLOW_MODEL_PATHS


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _bash_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _resolve_fingerflow_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if os.name == "nt" and shutil.which("wsl") is not None:
        return "wsl"
    return "local"


def _normalize_angle_pi(angle: np.ndarray) -> np.ndarray:
    normalized = np.mod(angle.astype(np.float32), np.float32(math.pi))
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def _normalize_angle_2pi_scalar(angle: float) -> float:
    wrapped = float(angle) % (2.0 * math.pi)
    return wrapped if math.isfinite(wrapped) else 0.0


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _save_uint8_png(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), _to_uint8_image(image))


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 1.0, 0.0).astype(np.float32)


def _downsample_mask_for_points(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Downsample a binary mask with max-pooling over each target cell footprint."""
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")

    source_h, source_w = mask.shape
    target_h, target_w = target_shape
    target_h = max(1, int(target_h))
    target_w = max(1, int(target_w))
    mask_bool = mask > 0
    out = np.zeros((target_h, target_w), dtype=np.float32)

    for cy in range(target_h):
        y0 = int(math.floor(cy * source_h / target_h))
        y1 = int(math.ceil((cy + 1) * source_h / target_h))
        y0 = max(0, min(source_h, y0))
        y1 = max(y0 + 1, min(source_h, y1))
        for cx in range(target_w):
            x0 = int(math.floor(cx * source_w / target_w))
            x1 = int(math.ceil((cx + 1) * source_w / target_w))
            x0 = max(0, min(source_w, x0))
            x1 = max(x0 + 1, min(source_w, x1))
            if np.any(mask_bool[y0:y1, x0:x1]):
                out[cy, cx] = 1.0

    return out.astype(np.float32)


def _compute_output_shape(height: int, width: int) -> tuple[int, int]:
    return max(1, height // 8), max(1, width // 8)


def get_rembg_session():
    global _REMBG_SESSION, _SELECTED_REMBG_PROVIDERS
    if not _HAS_REMBG:
        raise RuntimeError("rembg is not installed")
    if _REMBG_SESSION is None:
        providers = _resolve_rembg_providers(_current_runtime_config())
        kwargs: dict[str, Any] = {}
        if providers is not None:
            kwargs["providers"] = providers
            _SELECTED_REMBG_PROVIDERS = list(providers)
        _REMBG_SESSION = new_session(REMBG_MODEL, **kwargs)
        if _SELECTED_REMBG_PROVIDERS is None:
            _SELECTED_REMBG_PROVIDERS = ["auto"]
    return _REMBG_SESSION


def _fallback_mask_from_bgr(bgr_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _largest_ratio(mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        largest_area = max(cv2.contourArea(c) for c in contours)
        return float(largest_area) / float(mask.size)

    ratio_bright = _largest_ratio(threshold_bright)
    ratio_dark = _largest_ratio(threshold_dark)
    best_mask = threshold_bright if ratio_bright >= ratio_dark else threshold_dark
    return largest_component_mask(best_mask)


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no foreground contour found")
    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
    return refined


def validate_foreground_area(mask: np.ndarray, minimum_ratio: float) -> None:
    area_ratio = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
    if area_ratio < minimum_ratio:
        raise RuntimeError("foreground mask is too small to represent a finger")


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def load_image_unchanged(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def load_gray_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image = load_image_unchanged(path)
    if image.ndim == 2:
        gray = image
        alpha = None
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = image[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        raise ValueError(f"unsupported image shape for masked grayscale loading: {image.shape}")
    return gray, mask, alpha


def rembg_mask_from_bgr(bgr_image: np.ndarray) -> tuple[np.ndarray, str]:
    if not _HAS_REMBG:
        return _fallback_mask_from_bgr(bgr_image), "fallback_threshold"

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mask = remove(
        rgb_image,
        session=get_rembg_session(),
        only_mask=True,
        post_process_mask=True,
    )
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return largest_component_mask(mask), "rembg"


def estimate_finger_axis(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    edge_a = box[1] - box[0]
    edge_b = box[2] - box[1]
    len_a = float(np.linalg.norm(edge_a))
    len_b = float(np.linalg.norm(edge_b))
    if len_a >= len_b:
        long_edge, short_edge, length_l, width_w = edge_a, edge_b, len_a, len_b
    else:
        long_edge, short_edge, length_l, width_w = edge_b, edge_a, len_b, len_a
    if length_l <= 0 or width_w <= 0:
        raise RuntimeError("failed to estimate finger orientation")
    axis_u = long_edge / length_l
    axis_v = short_edge / width_w
    return axis_u.astype(np.float32), axis_v.astype(np.float32), length_l, width_w


def select_fingertip_end(
    contour: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    length_l: float,
    mask_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = contour.reshape(-1, 2).astype(np.float32)
    proj_u = points @ axis_u
    proj_v = points @ axis_v
    min_u = float(proj_u.min())
    max_u = float(proj_u.max())
    band = max(2.0, 0.14 * length_l)
    height, width = mask_shape
    min_selector = proj_u <= (min_u + band)
    max_selector = proj_u >= (max_u - band)

    def stats(selector: np.ndarray) -> tuple[float, float, float, bool]:
        chosen = points[selector]
        if chosen.shape[0] < 6:
            raise RuntimeError("insufficient contour support to localize fingertip")
        local_v = proj_v[selector]
        border_margin = max(6.0, 0.02 * max(width, height))
        min_border_distance = min(
            float(np.min(chosen[:, 0])),
            float(np.min(chosen[:, 1])),
            float((width - 1) - np.max(chosen[:, 0])),
            float((height - 1) - np.max(chosen[:, 1])),
        )
        return (
            float(local_v.max() - local_v.min()),
            float(np.mean(local_v)),
            float(np.mean(chosen[:, 1])),
            min_border_distance <= border_margin,
        )

    min_width, min_center_v, min_mean_y, min_border_attached = stats(min_selector)
    max_width, max_center_v, max_mean_y, max_border_attached = stats(max_selector)
    if min_border_attached != max_border_attached:
        choose_min = not min_border_attached
    else:
        width_delta = abs(min_width - max_width) / max(min_width, max_width, 1.0)
        choose_min = min_width < max_width if width_delta >= 0.08 else min_mean_y <= max_mean_y
    if choose_min:
        tip_proj = min_u
        center_v = min_center_v
        tip_to_base = axis_u
    else:
        tip_proj = max_u
        center_v = max_center_v
        tip_to_base = -axis_u
    tip_center = (axis_u * tip_proj) + (axis_v * center_v)
    return tip_center.astype(np.float32), tip_to_base.astype(np.float32)


def local_width_from_mask(
    mask: np.ndarray,
    tip_center: np.ndarray,
    tip_to_base: np.ndarray,
    axis_v: np.ndarray,
    length_l: float,
    band_fraction: float,
) -> float:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise RuntimeError("mask is empty")
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - tip_center
    long_coord = relative @ tip_to_base
    cross_coord = relative @ axis_v
    selector = (long_coord >= (-0.04 * length_l)) & (long_coord <= (band_fraction * length_l))
    if not np.any(selector):
        raise RuntimeError("failed to measure local width")
    local_cross = cross_coord[selector]
    return float(np.percentile(local_cross, 95) - np.percentile(local_cross, 5))


def normalise_brightness_array(gray_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground = mask > 0
    ys, xs = np.where(foreground)
    if ys.size == 0:
        raise RuntimeError("foreground mask is empty")
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    cropped_grey = gray_image[y_min:y_max, x_min:x_max]
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    cropped_clahe = clahe.apply(cropped_grey)
    clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    cropped_clahe = clahe2.apply(cropped_clahe)
    masked_grey = np.zeros_like(gray_image)
    masked_grey[y_min:y_max, x_min:x_max] = cropped_clahe
    masked_grey[~foreground] = 0
    return masked_grey


def _rotation_matrix_for_center(image_shape: tuple[int, int], angle_degrees: float) -> np.ndarray:
    height, width = image_shape[:2]
    center = (width / 2.0, height / 2.0)
    return cv2.getRotationMatrix2D(center, angle_degrees, 1.0)


def _rotate_array(array: np.ndarray, angle_degrees: float, interpolation: int, border_value: int = 0) -> np.ndarray:
    matrix = _rotation_matrix_for_center(array.shape, angle_degrees)
    return cv2.warpAffine(
        array,
        matrix,
        (array.shape[1], array.shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _estimate_pose_rotation(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contour available for pose normalization")
    contour = max(contours, key=cv2.contourArea)
    axis_u, _, _, _ = estimate_finger_axis(contour)
    angle = math.degrees(math.atan2(float(axis_u[1]), float(axis_u[0])))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return -90.0 - angle


def _estimate_central_ridge_spacing(gray_image: np.ndarray, mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    center_x = int(round(float(np.mean(xs))))
    center_y = int(round(float(np.mean(ys))))
    half = 64
    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(gray_image.shape[1], center_x + half)
    y1 = min(gray_image.shape[0], center_y + half)
    patch = gray_image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1]
    if patch.size == 0 or float(np.count_nonzero(patch_mask)) / float(patch_mask.size) < 0.35:
        return None
    patch_float = patch.astype(np.float32)
    patch_float[patch_mask <= 0] = float(np.mean(patch_float[patch_mask > 0])) if np.any(patch_mask > 0) else 0.0
    patch_fft = np.fft.fft2(patch_float)
    patch_fft_shifted = np.fft.fftshift(patch_fft)
    magnitude = np.abs(patch_fft_shifted)
    center_py = patch.shape[0] // 2
    center_px = patch.shape[1] // 2
    magnitude[max(0, center_py - 2):min(patch.shape[0], center_py + 3), max(0, center_px - 2):min(patch.shape[1], center_px + 3)] = 0.0
    peak = int(np.argmax(magnitude))
    x_peak, y_peak = np.unravel_index(peak, magnitude.shape)
    dx_freq = (x_peak - center_px) / max(patch.shape[1], 1)
    dy_freq = (y_peak - center_py) / max(patch.shape[0], 1)
    freq = float(np.sqrt(dx_freq**2 + dy_freq**2))
    if not math.isfinite(freq) or freq <= 1e-6:
        return None
    wavelength = 1.0 / freq
    if not math.isfinite(wavelength):
        return None
    return float(np.clip(wavelength, 3.0, 25.0))


def _normalize_ridge_frequency(gray_image: np.ndarray, mask: np.ndarray, target_spacing: float = 10.0) -> tuple[np.ndarray, np.ndarray, float]:
    spacing = _estimate_central_ridge_spacing(gray_image, mask)
    if spacing is None:
        return gray_image.copy(), mask.copy(), 1.0
    scale = float(np.clip(target_spacing / spacing, 0.5, 2.5))
    if abs(scale - 1.0) < 0.02:
        return gray_image.copy(), mask.copy(), 1.0
    resized_image = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    resized_mask = np.where(resized_mask > 0, 255, 0).astype(np.uint8)
    return resized_image, resized_mask, scale


def _preprocess_contactless_bgr(full_bgr: np.ndarray) -> PreprocessedContactlessImage:
    raw_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    rembg_started_at = time.perf_counter()
    initial_mask, mask_source = rembg_mask_from_bgr(full_bgr)
    _record_stage_time("rembg", time.perf_counter() - rembg_started_at)
    validate_foreground_area(initial_mask, minimum_ratio=0.03)
    cpu_started_at = time.perf_counter()
    normalized_gray = normalise_brightness_array(raw_gray, initial_mask)
    pose_rotation_degrees = _estimate_pose_rotation(initial_mask)
    pose_normalized_gray = _rotate_array(normalized_gray, pose_rotation_degrees, cv2.INTER_LINEAR, border_value=0)
    pose_normalized_mask = _rotate_array(initial_mask, pose_rotation_degrees, cv2.INTER_NEAREST, border_value=0)
    pose_normalized_mask = np.where(pose_normalized_mask > 0, 255, 0).astype(np.uint8)
    validate_foreground_area(pose_normalized_mask, minimum_ratio=0.02)
    preprocessed_gray, final_mask, ridge_scale_factor = _normalize_ridge_frequency(
        pose_normalized_gray,
        pose_normalized_mask,
        target_spacing=10.0,
    )
    _record_stage_time("cpu_preprocess", time.perf_counter() - cpu_started_at)
    return PreprocessedContactlessImage(
        raw_gray=raw_gray,
        normalized_gray=normalized_gray,
        pose_normalized_gray=pose_normalized_gray,
        pose_normalized_mask=pose_normalized_mask,
        preprocessed_gray=preprocessed_gray,
        final_mask=final_mask,
        mask_source=mask_source,
        pose_rotation_degrees=pose_rotation_degrees,
        ridge_scale_factor=ridge_scale_factor,
    )


def _preprocess_contactless_raw(raw_image_path: Path) -> PreprocessedContactlessImage:
    return _preprocess_contactless_bgr(load_bgr_image(raw_image_path))


def _acquisition_key(subject_id: int, finger_id: int, acquisition_id: int) -> tuple[int, int, int]:
    return subject_id, finger_id, acquisition_id


def _acquisition_name(subject_id: int, finger_id: int, acquisition_id: int) -> str:
    return f"s{subject_id:02d}_f{finger_id:02d}_a{acquisition_id:02d}"


def _parse_raw_view_suffix(raw_path: str | Path) -> int:
    return int(Path(raw_path).stem.split("_")[-1])


def _resolve_reconstruction_triplet(raw_view_paths: Iterable[str]) -> dict[str, Path] | None:
    mapping: dict[int, Path] = {}
    for raw_view_path in raw_view_paths:
        path = Path(raw_view_path)
        mapping[_parse_raw_view_suffix(path)] = path
    if not {0, 1, 2}.issubset(mapping):
        return None
    return {
        "front": mapping[0],
        "left": mapping[1],
        "right": mapping[2],
    }


def _extract_reconstruction_view_geometry(role: str, raw_image_path: Path) -> tuple[PreprocessedContactlessImage, ReconstructionViewGeometry]:
    preprocessed = _preprocess_contactless_raw(raw_image_path)
    mask = preprocessed.pose_normalized_mask
    height, width = mask.shape
    x_left = np.full(height, -1.0, dtype=np.float32)
    x_right = np.full(height, -1.0, dtype=np.float32)
    widths = np.zeros(height, dtype=np.float32)
    centers = np.zeros(height, dtype=np.float32)
    valid_rows = np.zeros(height, dtype=bool)

    for y in range(height):
        cols = np.flatnonzero(mask[y] > 0)
        if cols.size == 0:
            continue
        left = float(cols[0])
        right = float(cols[-1])
        x_left[y] = left
        x_right[y] = right
        widths[y] = right - left + 1.0
        centers[y] = 0.5 * (left + right)
        valid_rows[y] = True

    return preprocessed, ReconstructionViewGeometry(
        role=role,
        raw_image_path=str(raw_image_path.resolve()),
        image_shape=(height, width),
        x_left=x_left,
        x_right=x_right,
        widths=widths,
        centers=centers,
        valid_rows=valid_rows,
    )


def _rotate_depth_branch(x_coords: np.ndarray, depth: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_rot = (x_coords * cos_theta) + (depth * sin_theta)
    z_rot = (-x_coords * sin_theta) + (depth * cos_theta)
    return x_rot.astype(np.float32), z_rot.astype(np.float32)


def _depth_to_color_panel(depth: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    masked_depth = np.where(support_mask > 0, depth, 0.0).astype(np.float32)
    if np.count_nonzero(support_mask) == 0:
        return cv2.cvtColor(_to_uint8_mask(support_mask), cv2.COLOR_GRAY2BGR)
    valid_values = masked_depth[support_mask > 0]
    minimum = float(valid_values.min(initial=0.0))
    maximum = float(valid_values.max(initial=0.0))
    if maximum <= minimum + 1e-6:
        normalized = np.zeros_like(masked_depth, dtype=np.uint8)
    else:
        normalized = np.clip((masked_depth - minimum) / (maximum - minimum), 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    panel = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    panel[support_mask <= 0] = 0
    return panel


def _sample_surface_points(
    depth: np.ndarray,
    support_mask: np.ndarray,
    max_points: int = 25000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask_bool = support_mask > 0
    valid_count = int(np.count_nonzero(mask_bool))
    if valid_count == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, np.zeros((0, 3), dtype=np.uint8)

    stride = max(1, int(math.ceil(math.sqrt(valid_count / max(max_points, 1)))))
    sampled_mask = np.zeros_like(mask_bool, dtype=bool)
    sampled_mask[::stride, ::stride] = mask_bool[::stride, ::stride]
    if not np.any(sampled_mask):
        sampled_mask = mask_bool

    ys, xs = np.where(sampled_mask)
    zs = depth[sampled_mask].astype(np.float32)
    valid_values = zs[np.isfinite(zs)]
    if valid_values.size == 0:
        colors = np.zeros((zs.shape[0], 3), dtype=np.uint8)
        return xs.astype(np.float32), ys.astype(np.float32), zs, colors

    minimum = float(valid_values.min(initial=0.0))
    maximum = float(valid_values.max(initial=0.0))
    if maximum <= minimum + 1e-6:
        normalized = np.zeros_like(zs, dtype=np.uint8)
    else:
        normalized = np.clip((zs - minimum) / (maximum - minimum), 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(normalized[:, None], cv2.COLORMAP_TURBO)[:, 0, :]
    colors_rgb = colors_bgr[:, ::-1].copy()
    return xs.astype(np.float32), ys.astype(np.float32), zs, colors_rgb


def _write_front_surface_visualizations(
    reconstruction_dir: Path,
    depth_front: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[Path, Path]:
    html_path = reconstruction_dir / "surface_front_3d.html"
    png_path = reconstruction_dir / "surface_front_3d.png"

    xs, ys, zs, colors_rgb = _sample_surface_points(depth_front, support_mask)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:  # pragma: no cover - optional visualization dependency
        plt = None

    if plt is not None:
        fig = plt.figure(figsize=(8, 6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        if xs.size > 0:
            ax.scatter(xs, -ys, zs, c=colors_rgb.astype(np.float32) / 255.0, s=1.0, depthshade=False)
        ax.set_title("Front Reconstructed Surface")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.view_init(elev=28, azim=-58)
        try:
            ax.set_box_aspect((float(depth_front.shape[1]), float(depth_front.shape[0]), max(float(np.ptp(zs)) if zs.size > 0 else 1.0, 1.0)))
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    points = np.stack([xs, -ys, zs], axis=1).astype(np.float32) if xs.size > 0 else np.zeros((0, 3), dtype=np.float32)
    colors = colors_rgb.astype(np.uint8)
    bounds = {
        "x": [float(xs.min(initial=0.0)), float(xs.max(initial=1.0))],
        "y": [float((-ys).min(initial=0.0)), float((-ys).max(initial=1.0))],
        "z": [float(zs.min(initial=0.0)), float(zs.max(initial=1.0))],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Front Reconstructed Surface</title>
  <style>
    body {{ margin: 0; background: #0b0e14; color: #f3f4f6; font: 14px/1.4 system-ui, sans-serif; }}
    .wrap {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 20px; background: #131722; border-right: 1px solid #232838; }}
    .canvas-wrap {{ position: relative; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100vh; background:
      radial-gradient(circle at top, rgba(52,73,94,.35), transparent 38%),
      linear-gradient(180deg, #0c1018 0%, #06080d 100%); }}
    h1 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; color: #cbd5e1; }}
    code {{ color: #fde68a; }}
    .legend {{ margin-top: 16px; padding-top: 16px; border-top: 1px solid #232838; }}
    .hint {{ color: #94a3b8; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="sidebar">
      <h1>Front 3D Surface</h1>
      <p>This viewer shows the <code>depth_front</code> reconstruction inside the saved boundary mask.</p>
      <p>The colors match the bottom-left preview panel: they are normalized depth values, not confidence or gradient heatmaps.</p>
      <div class="legend">
        <p><strong>Controls</strong></p>
        <p class="hint">Drag to rotate</p>
        <p class="hint">Mouse wheel to zoom</p>
        <p class="hint">Double-click to reset view</p>
      </div>
    </div>
    <div class="canvas-wrap">
      <canvas id="surface"></canvas>
    </div>
  </div>
  <script>
    const points = {json.dumps(points.tolist(), separators=(",", ":"))};
    const colors = {json.dumps(colors.tolist(), separators=(",", ":"))};
    const bounds = {json.dumps(bounds, separators=(",", ":"))};
    const canvas = document.getElementById('surface');
    const ctx = canvas.getContext('2d');
    let width = 0, height = 0;
    let yaw = -0.85, pitch = 0.7, zoom = 1.55;
    let dragging = false, lastX = 0, lastY = 0;
    const center = [
      (bounds.x[0] + bounds.x[1]) / 2,
      (bounds.y[0] + bounds.y[1]) / 2,
      (bounds.z[0] + bounds.z[1]) / 2
    ];
    const span = Math.max(bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0], 1);

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function project(px, py, pz) {{
      const x0 = (px - center[0]) / span;
      const y0 = (py - center[1]) / span;
      const z0 = (pz - center[2]) / span;

      const cosy = Math.cos(yaw), siny = Math.sin(yaw);
      const cosp = Math.cos(pitch), sinp = Math.sin(pitch);
      const x1 = x0 * cosy - z0 * siny;
      const z1 = x0 * siny + z0 * cosy;
      const y1 = y0 * cosp - z1 * sinp;
      const z2 = y0 * sinp + z1 * cosp;
      const camera = 3.0 / zoom;
      const perspective = camera / Math.max(camera - z2, 0.2);
      return {{
        x: width * 0.5 + x1 * perspective * width * 0.85,
        y: height * 0.54 - y1 * perspective * width * 0.85,
        z: z2,
        size: Math.max(1.2, perspective * 2.2)
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      const projected = [];
      for (let i = 0; i < points.length; i++) {{
        const p = project(points[i][0], points[i][1], points[i][2]);
        projected.push([p.z, p.x, p.y, p.size, colors[i]]);
      }}
      projected.sort((a, b) => a[0] - b[0]);
      for (const item of projected) {{
        const color = item[4];
        ctx.fillStyle = `rgb(${{color[0]}},${{color[1]}},${{color[2]}})`;
        ctx.beginPath();
        ctx.arc(item[1], item[2], item[3], 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01));
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('pointerleave', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom = Math.max(0.5, Math.min(4.0, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }}, {{ passive: false }});
    canvas.addEventListener('dblclick', () => {{
      yaw = -0.85;
      pitch = 0.7;
      zoom = 1.55;
      draw();
    }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, png_path


def _sample_mask_for_points(support_mask: np.ndarray, max_points: int = 30000) -> np.ndarray:
    mask_bool = support_mask > 0
    valid_count = int(np.count_nonzero(mask_bool))
    if valid_count == 0:
        return mask_bool
    stride = max(1, int(math.ceil(math.sqrt(valid_count / max(max_points, 1)))))
    sampled_mask = np.zeros_like(mask_bool, dtype=bool)
    sampled_mask[::stride, ::stride] = mask_bool[::stride, ::stride]
    if not np.any(sampled_mask):
        sampled_mask = mask_bool
    return sampled_mask


def _compute_branch_point_clouds(
    x_relative: np.ndarray,
    z_front: np.ndarray,
    stitched_left: np.ndarray,
    stitched_right: np.ndarray,
    support_mask: np.ndarray,
    max_points_per_branch: int = 18000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_mask = _sample_mask_for_points(support_mask, max_points=max_points_per_branch)
    if not np.any(sampled_mask):
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty

    ys, xs = np.where(sampled_mask)
    y_world = (-ys).astype(np.float32)
    x_world = x_relative[sampled_mask].astype(np.float32)

    front_points = np.stack([x_world, y_world, z_front[sampled_mask].astype(np.float32)], axis=1).astype(np.float32)

    x_left_rot, z_left_rot = _rotate_depth_branch(x_relative, stitched_left, -45.0)
    left_points = np.stack(
        [
            x_left_rot[sampled_mask].astype(np.float32),
            y_world,
            z_left_rot[sampled_mask].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    x_right_rot, z_right_rot = _rotate_depth_branch(x_relative, stitched_right, 45.0)
    right_points = np.stack(
        [
            x_right_rot[sampled_mask].astype(np.float32),
            y_world,
            z_right_rot[sampled_mask].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return front_points, left_points, right_points


def _write_all_branch_surface_visualizations(
    reconstruction_dir: Path,
    front_points: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> tuple[Path, Path]:
    html_path = reconstruction_dir / "surface_all_branches_3d.html"
    png_path = reconstruction_dir / "surface_all_branches_3d.png"

    branch_points = {
        "front": front_points.astype(np.float32),
        "left": left_points.astype(np.float32),
        "right": right_points.astype(np.float32),
    }
    branch_colors = {
        "front": [242, 96, 58],
        "left": [44, 181, 232],
        "right": [195, 235, 70],
    }

    nonempty = [points for points in branch_points.values() if points.size > 0]
    if nonempty:
        all_points = np.concatenate(nonempty, axis=0).astype(np.float32)
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:  # pragma: no cover - optional visualization dependency
        plt = None

    if plt is not None:
        fig = plt.figure(figsize=(8.5, 6.5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        for name in ("front", "left", "right"):
            points = branch_points[name]
            if points.size == 0:
                continue
            color = np.array(branch_colors[name], dtype=np.float32) / 255.0
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], s=1.0, alpha=0.75, depthshade=False, label=name)
        ax.set_title("Raw Reconstructed Branch Point Clouds")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.view_init(elev=24, azim=-58)
        if all_points.size > 0:
            mins = all_points.min(axis=0)
            maxs = all_points.max(axis=0)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            try:
                spans = np.maximum(maxs - mins, 1.0)
                ax.set_box_aspect((float(spans[0]), float(spans[1]), float(spans[2])))
            except Exception:
                pass
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    bounds = {
        "x": [float(all_points[:, 0].min(initial=0.0)), float(all_points[:, 0].max(initial=1.0))],
        "y": [float(all_points[:, 1].min(initial=0.0)), float(all_points[:, 1].max(initial=1.0))],
        "z": [float(all_points[:, 2].min(initial=0.0)), float(all_points[:, 2].max(initial=1.0))],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>All Reconstructed Branches</title>
  <style>
    body {{ margin: 0; background: #0b0e14; color: #f3f4f6; font: 14px/1.4 system-ui, sans-serif; }}
    .wrap {{ display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 20px; background: #131722; border-right: 1px solid #232838; }}
    .canvas-wrap {{ position: relative; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100vh; background:
      radial-gradient(circle at top, rgba(52,73,94,.35), transparent 38%),
      linear-gradient(180deg, #0c1018 0%, #06080d 100%); }}
    h1 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; color: #cbd5e1; }}
    code {{ color: #fde68a; }}
    .controls {{ margin: 16px 0; padding: 14px; border: 1px solid #232838; border-radius: 12px; background: rgba(255,255,255,0.02); }}
    .controls label {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
    button {{ margin-right: 8px; margin-top: 8px; padding: 8px 10px; border-radius: 10px; border: 1px solid #2c3446; background: #1a2130; color: #e5e7eb; cursor: pointer; }}
    .hint {{ color: #94a3b8; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="sidebar">
      <h1>Raw Branch 3D View</h1>
      <p>This shows the raw <code>front</code>, <code>left</code>, and <code>right</code> reconstructed branch point clouds in one shared 3D frame.</p>
      <p>It is not a fused solid mesh. The earlier sharp spikes came from convex-hull closure, not necessarily from the reconstruction itself.</p>
      <p>The bottom row of <code>preview.png</code> is false-color depth, not confidence or quality heatmaps.</p>
      <div class="controls">
        <label><input id="toggle-front" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["front"][0]}, {branch_colors["front"][1]}, {branch_colors["front"][2]})"></span>Front</label>
        <label><input id="toggle-left" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["left"][0]}, {branch_colors["left"][1]}, {branch_colors["left"][2]})"></span>Left</label>
        <label><input id="toggle-right" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["right"][0]}, {branch_colors["right"][1]}, {branch_colors["right"][2]})"></span>Right</label>
        <div>
          <button id="all-on" type="button">All On</button>
          <button id="reset-view" type="button">Reset View</button>
        </div>
      </div>
      <p class="hint">Controls: drag to rotate, mouse wheel to zoom.</p>
    </div>
    <div class="canvas-wrap">
      <canvas id="surface"></canvas>
    </div>
  </div>
  <script>
    const branchData = {{
      front: {{ points: {json.dumps(branch_points["front"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["front"])} }},
      left: {{ points: {json.dumps(branch_points["left"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["left"])} }},
      right: {{ points: {json.dumps(branch_points["right"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["right"])} }}
    }};
    const bounds = {json.dumps(bounds, separators=(",", ":"))};
    const toggles = {{
      front: document.getElementById('toggle-front'),
      left: document.getElementById('toggle-left'),
      right: document.getElementById('toggle-right')
    }};
    const canvas = document.getElementById('surface');
    const ctx = canvas.getContext('2d');
    let width = 0, height = 0;
    let yaw = -0.9, pitch = 0.65, zoom = 1.5;
    let dragging = false, lastX = 0, lastY = 0;
    const center = [
      (bounds.x[0] + bounds.x[1]) / 2,
      (bounds.y[0] + bounds.y[1]) / 2,
      (bounds.z[0] + bounds.z[1]) / 2
    ];
    const span = Math.max(bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0], 1);

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function project(px, py, pz) {{
      const x0 = (px - center[0]) / span;
      const y0 = (py - center[1]) / span;
      const z0 = (pz - center[2]) / span;
      const cosy = Math.cos(yaw), siny = Math.sin(yaw);
      const cosp = Math.cos(pitch), sinp = Math.sin(pitch);
      const x1 = x0 * cosy - z0 * siny;
      const z1 = x0 * siny + z0 * cosy;
      const y1 = y0 * cosp - z1 * sinp;
      const z2 = y0 * sinp + z1 * cosp;
      const camera = 3.1 / zoom;
      const perspective = camera / Math.max(camera - z2, 0.2);
      return {{
        x: width * 0.5 + x1 * perspective * width * 0.88,
        y: height * 0.55 - y1 * perspective * width * 0.88,
        z: z2,
        size: Math.max(1.0, perspective * 2.0)
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      const projected = [];
      for (const name of ['front', 'left', 'right']) {{
        if (!toggles[name].checked) continue;
        const branch = branchData[name];
        for (const point of branch.points) {{
          const p = project(point[0], point[1], point[2]);
          projected.push([p.z, p.x, p.y, p.size, branch.color]);
        }}
      }}
      projected.sort((a, b) => a[0] - b[0]);
      for (const item of projected) {{
        const color = item[4];
        ctx.fillStyle = `rgb(${{color[0]}},${{color[1]}},${{color[2]}})`;
        ctx.beginPath();
        ctx.arc(item[1], item[2], item[3], 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    for (const toggle of Object.values(toggles)) {{
      toggle.addEventListener('change', draw);
    }}
    document.getElementById('all-on').addEventListener('click', () => {{
      toggles.front.checked = true;
      toggles.left.checked = true;
      toggles.right.checked = true;
      draw();
    }});
    document.getElementById('reset-view').addEventListener('click', () => {{
      yaw = -0.9;
      pitch = 0.65;
      zoom = 1.5;
      draw();
    }});
    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01));
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('pointerleave', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom = Math.max(0.45, Math.min(4.0, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }}, {{ passive: false }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, png_path


def _extract_mask_row_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = mask.shape[0]
    x_left = np.full(height, -1.0, dtype=np.float32)
    x_right = np.full(height, -1.0, dtype=np.float32)
    widths = np.zeros(height, dtype=np.float32)
    centers = np.zeros(height, dtype=np.float32)
    valid_rows = np.zeros(height, dtype=bool)
    for y in range(height):
        cols = np.flatnonzero(mask[y] > 0)
        if cols.size == 0:
            continue
        left = float(cols[0])
        right = float(cols[-1])
        x_left[y] = left
        x_right[y] = right
        widths[y] = right - left + 1.0
        centers[y] = 0.5 * (left + right)
        valid_rows[y] = True
    return x_left, x_right, widths, centers, valid_rows


def _interpolate_row_centers(geometry: ReconstructionViewGeometry, fallback: np.ndarray) -> np.ndarray:
    centers = geometry.centers.astype(np.float32).copy()
    valid_rows = geometry.valid_rows
    if np.any(valid_rows):
        row_indices = np.arange(valid_rows.shape[0], dtype=np.float32)
        missing = ~valid_rows
        if np.any(missing):
            centers[missing] = np.interp(
                row_indices[missing],
                row_indices[valid_rows],
                centers[valid_rows],
            ).astype(np.float32)
        return centers
    return fallback.astype(np.float32).copy()


def _label_mask_panel(panel: np.ndarray, title: str) -> np.ndarray:
    labeled = panel.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1] - 1, 30), (12, 16, 24), thickness=cv2.FILLED)
    cv2.putText(labeled, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 243, 248), 1, cv2.LINE_AA)
    return labeled


def _build_reprojection_overlay(observed_mask: np.ndarray, projected_mask: np.ndarray) -> np.ndarray:
    observed = observed_mask > 0
    projected = projected_mask > 0
    overlay = np.zeros((observed_mask.shape[0], observed_mask.shape[1], 3), dtype=np.uint8)
    overlay[observed & projected] = (255, 255, 255)
    overlay[observed & ~projected] = (64, 96, 255)
    overlay[projected & ~observed] = (72, 235, 232)
    return overlay


def _write_reprojection_diagnostics(
    reconstruction_dir: Path,
    front_geometry: ReconstructionViewGeometry,
    left_geometry: ReconstructionViewGeometry,
    right_geometry: ReconstructionViewGeometry,
    x_relative: np.ndarray,
    x_left_rot: np.ndarray,
    x_right_rot: np.ndarray,
    depth_front: np.ndarray,
    stitched_left: np.ndarray,
    stitched_right: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[Path, Path, dict[str, Any]]:
    report_path = reconstruction_dir / "reprojection_report.json"
    preview_path = reconstruction_dir / "reprojection_preview.png"

    observed_masks = {
        "front": np.where(
            np.arange(front_geometry.image_shape[1], dtype=np.float32)[None, :]
            >= front_geometry.x_left[:, None],
            255,
            0,
        ).astype(np.uint8),
        "left": np.where(
            np.arange(left_geometry.image_shape[1], dtype=np.float32)[None, :]
            >= left_geometry.x_left[:, None],
            255,
            0,
        ).astype(np.uint8),
        "right": np.where(
            np.arange(right_geometry.image_shape[1], dtype=np.float32)[None, :]
            >= right_geometry.x_left[:, None],
            255,
            0,
        ).astype(np.uint8),
    }
    for role, geometry in (("front", front_geometry), ("left", left_geometry), ("right", right_geometry)):
        mask = np.zeros(geometry.image_shape, dtype=np.uint8)
        for y in range(geometry.image_shape[0]):
            if not geometry.valid_rows[y]:
                continue
            x0 = int(round(float(geometry.x_left[y])))
            x1 = int(round(float(geometry.x_right[y])))
            mask[y, x0 : x1 + 1] = 255
        observed_masks[role] = mask

    front_centers = _interpolate_row_centers(front_geometry, front_geometry.centers)
    view_definitions = {
        "front": {
            "geometry": front_geometry,
            "centers": front_centers,
            "projected_x": x_relative.astype(np.float32),
            "description": "front branch compared against the front pose-normalized silhouette on the shared row support grid",
        },
        "left": {
            "geometry": left_geometry,
            "centers": _interpolate_row_centers(left_geometry, front_centers),
            "projected_x": x_left_rot.astype(np.float32),
            "description": "left branch compared against the left pose-normalized silhouette using the branch's shared row support shifted to the left-view row centers",
        },
        "right": {
            "geometry": right_geometry,
            "centers": _interpolate_row_centers(right_geometry, front_centers),
            "projected_x": x_right_rot.astype(np.float32),
            "description": "right branch compared against the right pose-normalized silhouette using the branch's shared row support shifted to the right-view row centers",
        },
    }

    projected_masks: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    overlay_panels: list[np.ndarray] = []

    for role in ("front", "left", "right"):
        geometry = view_definitions[role]["geometry"]
        centers = view_definitions[role]["centers"]
        projected_x = view_definitions[role]["projected_x"]
        projected_mask = np.zeros(geometry.image_shape, dtype=np.uint8)

        for y_front in range(front_geometry.image_shape[0]):
            row_mask = support_mask[y_front] > 0
            if not np.any(row_mask):
                continue

            if role == "front":
                y_target = float(y_front)
            else:
                y_target = _map_row_between_views(front_geometry.valid_rows, geometry.valid_rows, float(y_front))
                if y_target is None:
                    continue

            target_y = int(round(y_target))
            if target_y < 0 or target_y >= geometry.image_shape[0]:
                continue

            row_projected_x = projected_x[y_front, row_mask].astype(np.float32)
            if row_projected_x.size == 0:
                continue
            center = float(centers[target_y])
            x0 = max(0, int(math.floor(float(np.min(row_projected_x) + center))))
            x1 = min(geometry.image_shape[1] - 1, int(math.ceil(float(np.max(row_projected_x) + center))))
            if x1 >= x0:
                projected_mask[target_y, x0 : x1 + 1] = 255

        projected_masks[role] = projected_mask
        observed_mask = observed_masks[role]
        observed_bool = observed_mask > 0
        projected_bool = projected_mask > 0
        intersection = int(np.count_nonzero(observed_bool & projected_bool))
        union = int(np.count_nonzero(observed_bool | projected_bool))
        target_pixels = int(np.count_nonzero(observed_bool))
        projected_pixels = int(np.count_nonzero(projected_bool))

        _, _, projected_widths, projected_centers, projected_valid_rows = _extract_mask_row_geometry(projected_mask)
        overlap_rows = geometry.valid_rows & projected_valid_rows
        metrics[role] = {
            "iou": float(intersection / union) if union else 1.0,
            "precision": float(intersection / projected_pixels) if projected_pixels else 1.0,
            "recall": float(intersection / target_pixels) if target_pixels else 1.0,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "target_pixels": target_pixels,
            "projected_pixels": projected_pixels,
            "false_positive_pixels": int(np.count_nonzero(projected_bool & ~observed_bool)),
            "false_negative_pixels": int(np.count_nonzero(observed_bool & ~projected_bool)),
            "target_rows": int(np.count_nonzero(geometry.valid_rows)),
            "projected_rows": int(np.count_nonzero(projected_valid_rows)),
            "overlap_rows": int(np.count_nonzero(overlap_rows)),
            "row_width_mae": float(np.mean(np.abs(projected_widths[overlap_rows] - geometry.widths[overlap_rows]))) if np.any(overlap_rows) else None,
            "row_center_mae": float(np.mean(np.abs(projected_centers[overlap_rows] - geometry.centers[overlap_rows]))) if np.any(overlap_rows) else None,
        }

        observed_panel = _label_mask_panel(cv2.cvtColor(observed_mask, cv2.COLOR_GRAY2BGR), f"{role} observed")
        projected_panel = _label_mask_panel(cv2.cvtColor(projected_mask, cv2.COLOR_GRAY2BGR), f"{role} projected")
        overlay_panel = _label_mask_panel(_build_reprojection_overlay(observed_mask, projected_mask), f"{role} overlay")
        overlay_panels.extend([observed_panel, projected_panel, overlay_panel])

    reprojection_preview = np.vstack(
        [
            np.hstack(overlay_panels[0:3]),
            np.hstack(overlay_panels[3:6]),
            np.hstack(overlay_panels[6:9]),
        ]
    )
    cv2.imwrite(str(preview_path), reprojection_preview)

    report = {
        "projection_model": {
            "type": "branch_support_rowwise_silhouette",
            "description": "compare each reconstructed branch against its matching pose-normalized view by reusing the branch's shared row support and aligning it to that view's row center; this checks whether the branch occupancy matches the observed silhouette without fusing branches across views",
            "view_checks": {role: view_definitions[role]["description"] for role in ("front", "left", "right")},
        },
        "views": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, preview_path, metrics


def _resample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        return np.zeros((num_samples, 3), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32), num_samples, axis=0)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1).astype(np.float32)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(deltas, dtype=np.float32)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(points[:1].astype(np.float32), num_samples, axis=0)
    targets = np.linspace(0.0, total, num_samples, dtype=np.float32)
    xs = np.interp(targets, cumulative, points[:, 0].astype(np.float32))
    ys = np.interp(targets, cumulative, points[:, 1].astype(np.float32))
    zs = np.interp(targets, cumulative, points[:, 2].astype(np.float32))
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def _smooth_patch_rows(patch: np.ndarray) -> np.ndarray:
    if patch.shape[0] < 3:
        return patch.astype(np.float32)
    smoothed = patch.astype(np.float32).copy()
    smoothed[1:-1] = (
        (0.25 * patch[:-2].astype(np.float32))
        + (0.5 * patch[1:-1].astype(np.float32))
        + (0.25 * patch[2:].astype(np.float32))
    ).astype(np.float32)
    return smoothed


def _append_patch_mesh(
    vertices: list[np.ndarray],
    faces: list[np.ndarray],
    face_colors: list[np.ndarray],
    patch: np.ndarray,
    color_rgb: tuple[int, int, int],
) -> None:
    if patch.ndim != 3 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return
    base_index = int(sum(block.shape[0] for block in vertices))
    rows, cols, _ = patch.shape
    vertices.append(patch.reshape(-1, 3).astype(np.float32))

    tris: list[list[int]] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = base_index + (row * cols) + col
            b = a + 1
            c = base_index + ((row + 1) * cols) + col
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    if not tris:
        return
    tri_array = np.asarray(tris, dtype=np.int32)
    faces.append(tri_array)
    color_array = np.repeat(np.asarray(color_rgb, dtype=np.uint8)[None, :], tri_array.shape[0], axis=0)
    face_colors.append(color_array)


def _write_obj_mesh(mesh_path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# stitched reconstruction mesh"]
    lines.extend(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}" for v in vertices)
    lines.extend(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}" for face in faces)
    mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_stitched_surface_mesh(
    x_relative: np.ndarray,
    depth_front: np.ndarray,
    stitched_left: np.ndarray,
    stitched_right: np.ndarray,
    center_depth: np.ndarray,
    x_crit: np.ndarray,
    support_mask: np.ndarray,
    front_samples: int = 64,
    side_samples: int = 28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_left_rot, z_left_rot = _rotate_depth_branch(x_relative, stitched_left, -45.0)
    x_right_rot, z_right_rot = _rotate_depth_branch(x_relative, stitched_right, 45.0)

    front_rows: list[np.ndarray] = []
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    seam_left_rows: list[np.ndarray] = []
    seam_right_rows: list[np.ndarray] = []

    for y in range(support_mask.shape[0]):
        row_mask = support_mask[y] > 0
        if not np.any(row_mask):
            continue

        x_vals = x_relative[y, row_mask].astype(np.float32)
        y_vals = np.full((x_vals.shape[0],), -float(y), dtype=np.float32)
        seam_left = -float(x_crit[y])
        seam_right = float(x_crit[y])

        front_points = np.stack(
            [x_vals, y_vals, depth_front[y, row_mask].astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        left_points = np.stack(
            [x_left_rot[y, row_mask].astype(np.float32), y_vals, z_left_rot[y, row_mask].astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        right_points = np.stack(
            [x_right_rot[y, row_mask].astype(np.float32), y_vals, z_right_rot[y, row_mask].astype(np.float32)],
            axis=1,
        ).astype(np.float32)

        front_mask = (x_vals >= seam_left) & (x_vals <= seam_right)
        left_mask = x_vals <= seam_left
        right_mask = x_vals >= seam_right

        front_segment = front_points[front_mask]
        left_segment = left_points[left_mask]
        right_segment = right_points[right_mask]

        if front_segment.shape[0] >= 2:
            front_rows.append(_resample_polyline(front_segment, front_samples))
        if left_segment.shape[0] >= 2:
            left_rows.append(_resample_polyline(left_segment, side_samples))
        if right_segment.shape[0] >= 2:
            right_rows.append(_resample_polyline(right_segment, side_samples))

        left_seam_front = np.stack(
            [
                np.array([seam_left, -float(y), float(np.interp(seam_left, x_vals, front_points[:, 2]))], dtype=np.float32),
                np.array(
                    [
                        float(np.interp(seam_left, x_vals, left_points[:, 0])),
                        -float(y),
                        float(np.interp(seam_left, x_vals, left_points[:, 2])),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)
        seam_left_rows.append(left_seam_front)

        right_seam_front = np.stack(
            [
                np.array([seam_right, -float(y), float(np.interp(seam_right, x_vals, front_points[:, 2]))], dtype=np.float32),
                np.array(
                    [
                        float(np.interp(seam_right, x_vals, right_points[:, 0])),
                        -float(y),
                        float(np.interp(seam_right, x_vals, right_points[:, 2])),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)
        seam_right_rows.append(right_seam_front)

    vertices_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    patch_colors = {
        "front": (242, 96, 58),
        "left": (44, 181, 232),
        "right": (195, 235, 70),
        "seam_left": (166, 126, 240),
        "seam_right": (245, 188, 88),
    }

    if front_rows:
        _append_patch_mesh(vertices_blocks, face_blocks, color_blocks, _smooth_patch_rows(np.stack(front_rows, axis=0)), patch_colors["front"])
    if left_rows:
        _append_patch_mesh(vertices_blocks, face_blocks, color_blocks, _smooth_patch_rows(np.stack(left_rows, axis=0)), patch_colors["left"])
    if right_rows:
        _append_patch_mesh(vertices_blocks, face_blocks, color_blocks, _smooth_patch_rows(np.stack(right_rows, axis=0)), patch_colors["right"])
    if len(seam_left_rows) >= 2:
        _append_patch_mesh(vertices_blocks, face_blocks, color_blocks, _smooth_patch_rows(np.stack(seam_left_rows, axis=0)), patch_colors["seam_left"])
    if len(seam_right_rows) >= 2:
        _append_patch_mesh(vertices_blocks, face_blocks, color_blocks, _smooth_patch_rows(np.stack(seam_right_rows, axis=0)), patch_colors["seam_right"])

    if not vertices_blocks or not face_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)

    vertices = np.concatenate(vertices_blocks, axis=0).astype(np.float32)
    faces = np.concatenate(face_blocks, axis=0).astype(np.int32)
    face_colors = np.concatenate(color_blocks, axis=0).astype(np.uint8)
    return vertices, faces, face_colors


def _write_stitched_surface_visualizations(
    reconstruction_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> tuple[Path, Path, Path]:
    html_path = reconstruction_dir / "surface_stitched_3d.html"
    png_path = reconstruction_dir / "surface_stitched_3d.png"
    mesh_path = reconstruction_dir / "surface_stitched_mesh.obj"

    _write_obj_mesh(mesh_path, vertices, faces)

    if vertices.size == 0 or faces.size == 0:
        html_path.write_text("<!doctype html><html><body><p>No stitched surface geometry available.</p></body></html>", encoding="utf-8")
        png_path.write_bytes(b"")
        return html_path, png_path, mesh_path

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:  # pragma: no cover - optional visualization dependency
        plt = None
        Poly3DCollection = None

    if plt is not None and Poly3DCollection is not None:
        fig = plt.figure(figsize=(8.5, 6.5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        polys = vertices[faces]
        collection = Poly3DCollection(
            polys,
            facecolors=(face_colors.astype(np.float32) / 255.0),
            linewidths=0.05,
            edgecolors=(0.05, 0.07, 0.12, 0.10),
        )
        ax.add_collection3d(collection)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.set_title("Stitched Reconstructed Skin Surface")
        ax.view_init(elev=24, azim=-58)
        try:
            spans = np.maximum(maxs - mins, 1.0)
            ax.set_box_aspect((float(spans[0]), float(spans[1]), float(spans[2])))
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    bounds = {
        "x": [float(vertices[:, 0].min(initial=0.0)), float(vertices[:, 0].max(initial=1.0))],
        "y": [float(vertices[:, 1].min(initial=0.0)), float(vertices[:, 1].max(initial=1.0))],
        "z": [float(vertices[:, 2].min(initial=0.0)), float(vertices[:, 2].max(initial=1.0))],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stitched Reconstructed Surface</title>
  <style>
    body {{ margin: 0; background: #0b0e14; color: #f3f4f6; font: 14px/1.4 system-ui, sans-serif; }}
    .wrap {{ display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 20px; background: #131722; border-right: 1px solid #232838; }}
    .canvas-wrap {{ position: relative; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100vh; background:
      radial-gradient(circle at top, rgba(52,73,94,.35), transparent 38%),
      linear-gradient(180deg, #0c1018 0%, #06080d 100%); }}
    h1 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; color: #cbd5e1; }}
    code {{ color: #fde68a; }}
    .hint {{ color: #94a3b8; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="sidebar">
      <h1>Stitched Skin 3D View</h1>
      <p>This surface stitches the algorithm-computed front, left, and right branches into one continuous open skin.</p>
      <p>It is intentionally not a fully invented watertight solid. Unobserved back-side regions remain open where the model lacks direct support.</p>
      <p>The colors indicate source contribution: front, left, right, and seam-bridge strips.</p>
      <p class="hint">Controls: drag to rotate, mouse wheel to zoom, double-click to reset.</p>
    </div>
    <div class="canvas-wrap">
      <canvas id="surface"></canvas>
    </div>
  </div>
  <script>
    const vertices = {json.dumps(vertices.tolist(), separators=(",", ":"))};
    const faces = {json.dumps(faces.tolist(), separators=(",", ":"))};
    const faceColors = {json.dumps(face_colors.tolist(), separators=(",", ":"))};
    const bounds = {json.dumps(bounds, separators=(",", ":"))};
    const canvas = document.getElementById('surface');
    const ctx = canvas.getContext('2d');
    let width = 0, height = 0;
    let yaw = -0.9, pitch = 0.65, zoom = 1.45;
    let dragging = false, lastX = 0, lastY = 0;
    const center = [
      (bounds.x[0] + bounds.x[1]) / 2,
      (bounds.y[0] + bounds.y[1]) / 2,
      (bounds.z[0] + bounds.z[1]) / 2
    ];
    const span = Math.max(bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0], 1);

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function project(v) {{
      const x0 = (v[0] - center[0]) / span;
      const y0 = (v[1] - center[1]) / span;
      const z0 = (v[2] - center[2]) / span;
      const cosy = Math.cos(yaw), siny = Math.sin(yaw);
      const cosp = Math.cos(pitch), sinp = Math.sin(pitch);
      const x1 = x0 * cosy - z0 * siny;
      const z1 = x0 * siny + z0 * cosy;
      const y1 = y0 * cosp - z1 * sinp;
      const z2 = y0 * sinp + z1 * cosp;
      const camera = 3.2 / zoom;
      const perspective = camera / Math.max(camera - z2, 0.22);
      return {{
        x: width * 0.5 + x1 * perspective * width * 0.9,
        y: height * 0.55 - y1 * perspective * width * 0.9,
        z: z2
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      const projected = vertices.map(project);
      const tris = [];
      for (let i = 0; i < faces.length; i++) {{
        const face = faces[i];
        const a = projected[face[0]], b = projected[face[1]], c = projected[face[2]];
        const cross = ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
        if (cross >= 0) continue;
        const depth = (a.z + b.z + c.z) / 3;
        tris.push([depth, a, b, c, faceColors[i]]);
      }}
      tris.sort((lhs, rhs) => lhs[0] - rhs[0]);
      for (const tri of tris) {{
        const color = tri[4];
        ctx.fillStyle = `rgb(${{color[0]}},${{color[1]}},${{color[2]}})`;
        ctx.strokeStyle = 'rgba(10, 12, 18, 0.18)';
        ctx.beginPath();
        ctx.moveTo(tri[1].x, tri[1].y);
        ctx.lineTo(tri[2].x, tri[2].y);
        ctx.lineTo(tri[3].x, tri[3].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }}
    }}

    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01));
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('pointerleave', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom = Math.max(0.45, Math.min(4.0, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }}, {{ passive: false }});
    canvas.addEventListener('dblclick', () => {{
      yaw = -0.9;
      pitch = 0.65;
      zoom = 1.45;
      draw();
    }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, png_path, mesh_path


def _compute_mls_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    y_offset, x_offset = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x_offset.astype(np.float32), y_offset.astype(np.float32)


def _solve_weighted_quadratic_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    if z_values.size < 6:
        return None
    design = np.stack(
        [
            np.ones_like(x_values, dtype=np.float32),
            x_values,
            y_values,
            x_values**2,
            x_values * y_values,
            y_values**2,
        ],
        axis=1,
    ).astype(np.float32)
    weighted_design = design * weights[:, None]
    weighted_target = z_values * weights
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 4:
        return None
    return coeffs.astype(np.float32)


def _smooth_depth_with_quadratic_mls(
    depth: np.ndarray,
    support_mask: np.ndarray,
    radius: int = 5,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_bool = support_mask > 0
    smoothed = np.zeros_like(depth, dtype=np.float32)
    grad_x = np.zeros_like(depth, dtype=np.float32)
    grad_y = np.zeros_like(depth, dtype=np.float32)
    if not np.any(mask_bool):
        return smoothed, grad_x, grad_y

    x_offsets, y_offsets = _compute_mls_offsets(radius)
    distance_sq = (x_offsets**2 + y_offsets**2).astype(np.float32)
    gaussian_weights = np.exp(-distance_sq / max(2.0 * sigma * sigma, 1e-6)).astype(np.float32)

    height, width = depth.shape
    for y, x in np.argwhere(mask_bool):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)

        local_mask = mask_bool[y0:y1, x0:x1]
        if np.count_nonzero(local_mask) < 6:
            smoothed[y, x] = depth[y, x]
            continue

        ky0 = radius - (y - y0)
        ky1 = ky0 + (y1 - y0)
        kx0 = radius - (x - x0)
        kx1 = kx0 + (x1 - x0)
        local_weights = gaussian_weights[ky0:ky1, kx0:kx1]

        y_coords, x_coords = np.nonzero(local_mask)
        x_values = (x_coords + x0 - x).astype(np.float32)
        y_values = (y_coords + y0 - y).astype(np.float32)
        z_values = depth[y0:y1, x0:x1][local_mask].astype(np.float32)
        weights = local_weights[local_mask].astype(np.float32)
        if weights.size < 6 or np.all(weights <= 0.0):
            smoothed[y, x] = depth[y, x]
            continue

        coeffs = _solve_weighted_quadratic_surface(x_values, y_values, z_values, weights)
        if coeffs is None:
            smoothed[y, x] = depth[y, x]
            continue

        smoothed[y, x] = float(coeffs[0])
        grad_x[y, x] = float(coeffs[1])
        grad_y[y, x] = float(coeffs[2])

    smoothed[~mask_bool] = 0.0
    grad_x[~mask_bool] = 0.0
    grad_y[~mask_bool] = 0.0
    return smoothed.astype(np.float32), grad_x.astype(np.float32), grad_y.astype(np.float32)


def _build_depth_gradient_labels(
    depth_front: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    front_smooth, front_grad_x, front_grad_y = _smooth_depth_with_quadratic_mls(depth_front, support_mask)
    left_smooth, left_grad_x, left_grad_y = _smooth_depth_with_quadratic_mls(depth_left, support_mask)
    right_smooth, right_grad_x, right_grad_y = _smooth_depth_with_quadratic_mls(depth_right, support_mask)
    return {
        "depth_front_smooth": front_smooth.astype(np.float32),
        "depth_left_smooth": left_smooth.astype(np.float32),
        "depth_right_smooth": right_smooth.astype(np.float32),
        "gradient_front": np.stack([front_grad_x, front_grad_y], axis=0).astype(np.float32),
        "gradient_left": np.stack([left_grad_x, left_grad_y], axis=0).astype(np.float32),
        "gradient_right": np.stack([right_grad_x, right_grad_y], axis=0).astype(np.float32),
        "support_mask": (support_mask[np.newaxis, ...] > 0).astype(np.float32),
    }


def _load_reconstruction_gradient_for_sample(
    sample: RawViewSample,
    reconstruction: AcquisitionReconstructionResult | None,
) -> np.ndarray | None:
    if reconstruction is None or sample.raw_view_index not in {0, 1, 2}:
        return None
    role_key = {0: "gradient_front", 1: "gradient_left", 2: "gradient_right"}[sample.raw_view_index]
    with np.load(reconstruction.depth_gradient_labels_path) as data:
        if role_key not in data:
            raise KeyError(f"missing {role_key} in {reconstruction.depth_gradient_labels_path}")
        gradient = data[role_key].astype(np.float32)
    if gradient.ndim != 3 or gradient.shape[0] != 2:
        raise ValueError(f"expected gradient with shape (2, H, W), got {gradient.shape}")
    return np.transpose(gradient, (1, 2, 0)).astype(np.float32)


def _view_role_for_sample(sample: RawViewSample) -> str | None:
    return {0: "front", 1: "left", 2: "right"}.get(sample.raw_view_index)


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _build_inverse_unwarp_maps(
    x_out: np.ndarray,
    y_out: np.ndarray,
    valid_mask: np.ndarray,
    output_shape: tuple[int, int] | list[int],
) -> dict[str, np.ndarray]:
    height_out, width_out = int(output_shape[0]), int(output_shape[1])
    source_x_map = np.full((height_out, width_out), np.nan, dtype=np.float32)
    source_y_map = np.full((height_out, width_out), np.nan, dtype=np.float32)
    best_dist = np.full((height_out, width_out), np.inf, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(x_out) & np.isfinite(y_out)
    src_y, src_x = np.nonzero(valid)
    sample_x = x_out[valid].astype(np.float32)
    sample_y = y_out[valid].astype(np.float32)
    base_x = np.floor(sample_x).astype(np.int32)
    base_y = np.floor(sample_y).astype(np.int32)

    for off_x in (0, 1):
        for off_y in (0, 1):
            tx = base_x + off_x
            ty = base_y + off_y
            inside = (tx >= 0) & (tx < width_out) & (ty >= 0) & (ty < height_out)
            if not np.any(inside):
                continue
            tx_inside = tx[inside]
            ty_inside = ty[inside]
            dist = ((sample_x[inside] - tx_inside.astype(np.float32)) ** 2) + (
                (sample_y[inside] - ty_inside.astype(np.float32)) ** 2
            )
            replace = dist < best_dist[ty_inside, tx_inside]
            if not np.any(replace):
                continue
            best_ty = ty_inside[replace]
            best_tx = tx_inside[replace]
            best_dist[best_ty, best_tx] = dist[replace]
            source_x_map[best_ty, best_tx] = src_x[inside][replace].astype(np.float32)
            source_y_map[best_ty, best_tx] = src_y[inside][replace].astype(np.float32)

    valid_inverse = np.isfinite(source_x_map) & np.isfinite(source_y_map)
    return {
        "source_x_map": source_x_map.astype(np.float32),
        "source_y_map": source_y_map.astype(np.float32),
        "source_valid_mask": valid_inverse.astype(np.uint8),
    }


def _sample_1d_linear(values: np.ndarray, position: float) -> float | None:
    if values.ndim != 1 or values.size == 0 or not math.isfinite(position):
        return None
    if position < 0.0 or position > float(values.size - 1):
        return None
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        value = float(values[low])
        return value if math.isfinite(value) else None
    weight = float(position - low)
    v0 = float(values[low])
    v1 = float(values[high])
    if not (math.isfinite(v0) and math.isfinite(v1)):
        return None
    return ((1.0 - weight) * v0) + (weight * v1)


def _sample_2d_bilinear(array: np.ndarray, x: float, y: float, valid_mask: np.ndarray | None = None) -> float | None:
    if array.ndim != 2 or not math.isfinite(x) or not math.isfinite(y):
        return None
    height, width = array.shape
    if x < 0.0 or y < 0.0 or x > float(width - 1) or y > float(height - 1):
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    samples: list[tuple[float, float]] = []
    for yy in (y0, y1):
        for xx in (x0, x1):
            if valid_mask is not None and valid_mask[yy, xx] <= 0:
                continue
            value = float(array[yy, xx])
            if not math.isfinite(value):
                continue
            weight_x = 1.0 - abs(float(xx) - x)
            weight_y = 1.0 - abs(float(yy) - y)
            weight = max(weight_x, 0.0) * max(weight_y, 0.0)
            if weight > 0.0:
                samples.append((weight, value))
    if not samples:
        return None
    weight_sum = sum(weight for weight, _ in samples)
    if weight_sum <= 1e-8:
        return None
    return sum(weight * value for weight, value in samples) / weight_sum


def _valid_row_bounds(valid_rows: np.ndarray) -> tuple[float, float] | None:
    rows = np.flatnonzero(valid_rows.astype(bool))
    if rows.size < 2:
        return None
    return float(rows[0]), float(rows[-1])


def _map_row_between_views(
    source_valid_rows: np.ndarray,
    target_valid_rows: np.ndarray,
    y_source: float,
) -> float | None:
    """Map a row by normalized tip-to-base position between pose-normalized views."""
    if not math.isfinite(float(y_source)):
        return None
    source_bounds = _valid_row_bounds(source_valid_rows)
    target_bounds = _valid_row_bounds(target_valid_rows)
    if source_bounds is None or target_bounds is None:
        return None
    source_start, source_end = source_bounds
    target_start, target_end = target_bounds
    denom = max(source_end - source_start, 1e-6)
    t = (float(y_source) - source_start) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return float(target_start + t * (target_end - target_start))


def _role_valid_rows(reconstruction_maps: dict[str, np.ndarray], role: str) -> np.ndarray | None:
    value = reconstruction_maps.get(f"{role}_valid_rows")
    if value is not None:
        return value.astype(bool)
    if role == "front" and "support_mask" in reconstruction_maps:
        return np.any(reconstruction_maps["support_mask"] > 0, axis=1)
    return None


def _map_front_row_to_role(
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    y_front: float,
) -> float | None:
    if not math.isfinite(float(y_front)):
        return None
    if role == "front":
        return float(y_front)
    front_rows = _role_valid_rows(reconstruction_maps, "front")
    target_rows = _role_valid_rows(reconstruction_maps, role)
    if front_rows is None or target_rows is None:
        return float(y_front)
    mapped = _map_row_between_views(front_rows, target_rows, y_front)
    return float(y_front) if mapped is None else mapped


def _map_unwarped_to_front_source(unwarp_maps: dict[str, np.ndarray], x: float, y: float) -> tuple[float, float] | None:
    valid_mask = unwarp_maps["source_valid_mask"]
    source_x = _sample_2d_bilinear(unwarp_maps["source_x_map"], x, y, valid_mask=valid_mask)
    source_y = _sample_2d_bilinear(unwarp_maps["source_y_map"], x, y, valid_mask=valid_mask)
    if source_x is None or source_y is None:
        return None
    return float(source_x), float(source_y)


def _canonical_minutiae_with_front_sources(
    canonical_minutiae: list[dict[str, Any]],
    unwarp_maps: dict[str, np.ndarray],
) -> tuple[list[tuple[dict[str, Any], float, float]], dict[str, int]]:
    """Filter canonical minutiae and attach inverse-unwarped front source points."""
    unwarped_mask = unwarp_maps.get("unwarped_mask")
    source_valid_mask = unwarp_maps.get("source_valid_mask")
    if unwarped_mask is None:
        unwarped_mask = source_valid_mask
    if unwarped_mask is None:
        raise KeyError("center unwarp maps must contain unwarped_mask or source_valid_mask")

    height, width = unwarped_mask.shape
    kept: list[tuple[dict[str, Any], float, float]] = []
    counters = {
        "canonical_total": len(canonical_minutiae),
        "dropped_nonfinite_canonical_xy": 0,
        "dropped_outside_unwarped_bounds": 0,
        "dropped_outside_unwarped_mask": 0,
        "dropped_no_inverse_source": 0,
        "inverse_mapped_count": 0,
    }

    for minutia in canonical_minutiae:
        x = float(minutia.get("x", float("nan")))
        y = float(minutia.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            counters["dropped_nonfinite_canonical_xy"] += 1
            continue
        if x < 0.0 or y < 0.0 or x >= float(width) or y >= float(height):
            counters["dropped_outside_unwarped_bounds"] += 1
            continue
        if not _point_has_mask_support(unwarped_mask, x, y, radius=1):
            counters["dropped_outside_unwarped_mask"] += 1
            continue
        source_point = _map_unwarped_to_front_source(unwarp_maps, x, y)
        if source_point is None:
            counters["dropped_no_inverse_source"] += 1
            continue
        kept.append((minutia, float(source_point[0]), float(source_point[1])))

    counters["inverse_mapped_count"] = len(kept)
    return kept, counters


def _sample_side_rotated_x(
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    x_front: float,
    y_front: float,
    support_mask: np.ndarray,
) -> float | None:
    key = "x_left_rot" if role == "left" else "x_right_rot"
    if key in reconstruction_maps:
        return _sample_2d_bilinear(reconstruction_maps[key], x_front, y_front, valid_mask=support_mask)

    x_relative = _sample_2d_bilinear(reconstruction_maps["x_relative"], x_front, y_front, valid_mask=support_mask)
    if x_relative is None:
        return None
    stitched_key = "stitched_left" if role == "left" else "stitched_right"
    if stitched_key in reconstruction_maps:
        branch_depth = _sample_2d_bilinear(reconstruction_maps[stitched_key], x_front, y_front, valid_mask=support_mask)
        if branch_depth is None:
            return None
        angle = -45.0 if role == "left" else 45.0
        theta = math.radians(angle)
        return float((x_relative * math.cos(theta)) + (branch_depth * math.sin(theta)))

    depth_front = _sample_2d_bilinear(reconstruction_maps["depth_front"], x_front, y_front, valid_mask=support_mask)
    if depth_front is None:
        return None
    angle = -45.0 if role == "left" else 45.0
    theta = math.radians(angle)
    return float((x_relative * math.cos(theta)) + (depth_front * math.sin(theta)))


def _training_frame_scale_from_preprocessed(
    preprocessed: PreprocessedContactlessImage,
) -> tuple[float, float]:
    pose_h, pose_w = preprocessed.pose_normalized_gray.shape[:2]
    train_h, train_w = preprocessed.preprocessed_gray.shape[:2]
    scale_x = float(train_w) / float(max(pose_w, 1))
    scale_y = float(train_h) / float(max(pose_h, 1))

    fallback = float(preprocessed.ridge_scale_factor)
    if not math.isfinite(scale_x) or scale_x <= 0.0:
        scale_x = fallback if math.isfinite(fallback) and fallback > 0.0 else 1.0
    if not math.isfinite(scale_y) or scale_y <= 0.0:
        scale_y = fallback if math.isfinite(fallback) and fallback > 0.0 else 1.0
    return scale_x, scale_y


def _project_front_source_to_pose_frame(
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    x_front: float,
    y_front: float,
) -> tuple[float, float] | None:
    support_mask = reconstruction_maps["support_mask"]
    x_relative = _sample_2d_bilinear(reconstruction_maps["x_relative"], x_front, y_front, valid_mask=support_mask)
    if x_relative is None:
        return None
    y_pose = _map_front_row_to_role(reconstruction_maps, role, y_front)
    if y_pose is None:
        return None

    if role == "front":
        center = _sample_1d_linear(reconstruction_maps["front_centers"], y_pose)
        if center is None:
            return None
        return float(center + x_relative), float(y_pose)

    if role == "left":
        centers_key = "left_centers"
    elif role == "right":
        centers_key = "right_centers"
    else:
        return None

    x_rot = _sample_side_rotated_x(reconstruction_maps, role, x_front, y_front, support_mask)
    if x_rot is None:
        return None
    center = _sample_1d_linear(reconstruction_maps[centers_key], y_pose)
    if center is None:
        return None
    return float(center + x_rot), float(y_pose)


def _project_front_source_to_training_frame(
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    x_front: float,
    y_front: float,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float] | None:
    pose_point = _project_front_source_to_pose_frame(reconstruction_maps, role, x_front, y_front)
    if pose_point is None:
        return None
    sx = float(scale_x) if math.isfinite(float(scale_x)) and float(scale_x) > 0.0 else 1.0
    sy = float(scale_y) if math.isfinite(float(scale_y)) and float(scale_y) > 0.0 else 1.0
    return float(pose_point[0] * sx), float(pose_point[1] * sy)


def _point_has_mask_support(mask: np.ndarray, x: float, y: float, radius: int = 2) -> bool:
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")
    height, width = mask.shape
    if x < 0.0 or y < 0.0 or x >= float(width) or y >= float(height):
        return False
    cx = int(round(x))
    cy = int(round(y))
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _remap_unwarped_minutiae_to_sample(
    canonical_minutiae: list[dict[str, Any]],
    unwarp_maps: dict[str, np.ndarray],
    reconstruction_maps: dict[str, np.ndarray],
    sample: RawViewSample,
    preprocessed: PreprocessedContactlessImage,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    role = _view_role_for_sample(sample)
    if role is None:
        return [], {"view_role": None, "reason": "unsupported_raw_view_index"}

    remapped: list[dict[str, Any]] = []
    height, width = preprocessed.preprocessed_gray.shape
    scale_x, scale_y = _training_frame_scale_from_preprocessed(preprocessed)
    source_minutiae, filter_counters = _canonical_minutiae_with_front_sources(canonical_minutiae, unwarp_maps)
    remap_details: dict[str, Any] = {
        "view_role": role,
        "scale_x": scale_x,
        "scale_y": scale_y,
        **filter_counters,
        "dropped_projection_failed": 0,
        "dropped_out_of_training_bounds": 0,
        "dropped_no_final_mask_support": 0,
        "reprojected_minutiae_count": 0,
    }
    delta = 4.0
    for minutia, source_x, source_y in source_minutiae:
        destination = _project_front_source_to_training_frame(
            reconstruction_maps,
            role,
            source_x,
            source_y,
            scale_x,
            scale_y,
        )
        if destination is None:
            remap_details["dropped_projection_failed"] += 1
            continue
        x_dst, y_dst = destination
        if x_dst < 0.0 or y_dst < 0.0 or x_dst >= float(width) or y_dst >= float(height):
            remap_details["dropped_out_of_training_bounds"] += 1
            continue
        if not _point_has_mask_support(preprocessed.final_mask, x_dst, y_dst):
            remap_details["dropped_no_final_mask_support"] += 1
            continue

        theta = float(minutia["theta"])
        dx = math.cos(theta) * delta
        dy = math.sin(theta) * delta
        forward = _map_unwarped_to_front_source(unwarp_maps, float(minutia["x"]) + dx, float(minutia["y"]) + dy)
        backward = _map_unwarped_to_front_source(unwarp_maps, float(minutia["x"]) - dx, float(minutia["y"]) - dy)

        projected_points: list[tuple[float, float]] = []
        for candidate in (forward, backward):
            if candidate is None:
                continue
            projected = _project_front_source_to_training_frame(
                reconstruction_maps,
                role,
                candidate[0],
                candidate[1],
                scale_x,
                scale_y,
            )
            if projected is not None:
                projected_points.append(projected)

        if len(projected_points) == 2:
            theta_dst = math.atan2(
                float(projected_points[0][1] - projected_points[1][1]),
                float(projected_points[0][0] - projected_points[1][0]),
            )
        elif len(projected_points) == 1:
            theta_dst = math.atan2(
                float(projected_points[0][1] - y_dst),
                float(projected_points[0][0] - x_dst),
            )
        else:
            theta_dst = theta

        remapped.append(
            {
                "x": float(x_dst),
                "y": float(y_dst),
                "theta": _normalize_angle_2pi_scalar(theta_dst),
                "score": minutia.get("score"),
                "type": minutia.get("type"),
                "source": str(minutia.get("source", "reconstruction_unwarp")),
            }
        )
    remap_details["reprojected_minutiae_count"] = len(remapped)
    return remapped, remap_details


def _write_minutiae_json(path: Path, minutiae: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(minutiae, indent=2), encoding="utf-8")


def _load_or_extract_canonical_reconstruction_minutiae(
    reconstruction: AcquisitionReconstructionResult,
    fingerflow_model_dir: Path,
    fingerflow_backend: FingerflowBackendConfig,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    cache_key = reconstruction.acquisition_id
    cached = _RECONSTRUCTION_MINUTIAE_CACHE.get(cache_key)
    if cached is not None:
        return (
            list(cached["minutiae"]),
            str(cached["source"]),
            dict(cached["details"]),
        )

    reconstruction_dir = Path(reconstruction.reconstruction_dir)
    unwarped_path = Path(reconstruction.center_unwarped_image_path)
    enhanced_path = reconstruction_dir / "center_unwarped_enhanced.png"
    canonical_json_path = reconstruction_dir / "canonical_unwarped_minutiae.json"
    canonical_gray = _require_grayscale(unwarped_path)
    enhanced = _enhance_for_minutiae(canonical_gray)
    _save_uint8_png(enhanced_path, enhanced)

    minutiae_source = f"fingerflow_{fingerflow_backend.backend}_reconstruction_unwarp"
    try:
        if fingerflow_backend.backend == "wsl":
            minutiae = _extract_fingerflow_minutiae_wsl(
                unwarped_path,
                enhanced_path,
                reconstruction_dir,
                fingerflow_model_dir,
                backend=fingerflow_backend,
            )
        else:
            minutiae = _extract_fingerflow_minutiae(
                unwarped_path,
                enhanced_path,
                reconstruction_dir,
                fingerflow_model_dir,
            )
    except Exception:
        minutiae_source = "pyfing_fallback_reconstruction_unwarp"
        fallback_started_at = time.perf_counter()
        minutiae = _extract_pyfing_minutiae(enhanced)
        _record_stage_time("pyfing_fallback", time.perf_counter() - fallback_started_at)

    canonical_minutiae = [
        {
            **item,
            "source": minutiae_source,
        }
        for item in minutiae
    ]
    _write_minutiae_json(canonical_json_path, canonical_minutiae)
    details = {
        "canonical_unwarped_minutiae_path": str(canonical_json_path.resolve()),
        "center_unwarped_enhanced_path": str(enhanced_path.resolve()),
        "canonical_minutiae_count": len(canonical_minutiae),
    }
    _RECONSTRUCTION_MINUTIAE_CACHE[cache_key] = {
        "minutiae": canonical_minutiae,
        "source": minutiae_source,
        "details": details,
    }
    return canonical_minutiae, minutiae_source, details


def _write_reconstruction_preview(
    reconstruction_dir: Path,
    view_geometries: dict[str, ReconstructionViewGeometry],
    support_mask: np.ndarray,
    depth_front: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
) -> None:
    silhouette_panels = []
    for role in ("front", "left", "right"):
        panel = np.zeros(view_geometries[role].image_shape, dtype=np.uint8)
        for y in range(view_geometries[role].image_shape[0]):
            if not view_geometries[role].valid_rows[y]:
                continue
            x0 = int(round(float(view_geometries[role].x_left[y])))
            x1 = int(round(float(view_geometries[role].x_right[y])))
            panel[y, x0 : x1 + 1] = 255
        silhouette_panels.append(cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR))

    depth_panels = [
        _depth_to_color_panel(depth_front, support_mask),
        _depth_to_color_panel(depth_left, support_mask),
        _depth_to_color_panel(depth_right, support_mask),
    ]
    preview = np.vstack([np.hstack(silhouette_panels), np.hstack(depth_panels)])
    cv2.imwrite(str(reconstruction_dir / "preview.png"), preview)


def _write_row_measurements(
    reconstruction_dir: Path,
    valid_rows: np.ndarray,
    front_geometry: ReconstructionViewGeometry,
    left_geometry: ReconstructionViewGeometry,
    right_geometry: ReconstructionViewGeometry,
    semi_major: np.ndarray,
    semi_minor: np.ndarray,
    translation_left: np.ndarray,
    translation_right: np.ndarray,
    center_depth: np.ndarray,
    theta: np.ndarray,
    x_crit: np.ndarray,
) -> None:
    rows: list[dict[str, Any]] = []
    for y in range(front_geometry.image_shape[0]):
        rows.append(
            {
                "row_index": y,
                "valid": bool(valid_rows[y]),
                "front_width": float(front_geometry.widths[y]),
                "left_width": float(left_geometry.widths[y]),
                "right_width": float(right_geometry.widths[y]),
                "front_center": float(front_geometry.centers[y]),
                "left_center": float(left_geometry.centers[y]),
                "right_center": float(right_geometry.centers[y]),
                "semi_major": float(semi_major[y]),
                "semi_minor": float(semi_minor[y]),
                "translation_left": float(translation_left[y]),
                "translation_right": float(translation_right[y]),
                "center_depth": float(center_depth[y]),
                "theta": float(theta[y]),
                "x_crit": float(x_crit[y]),
            }
        )
    (reconstruction_dir / "row_measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _reconstruct_multiview_acquisition(sample: RawViewSample, output_root: Path) -> AcquisitionReconstructionResult:
    triplet_paths = _resolve_reconstruction_triplet(sample.raw_view_paths)
    if triplet_paths is None:
        raise RuntimeError("missing one or more required reconstruction views 0/1/2")

    preprocessed_views: dict[str, PreprocessedContactlessImage] = {}
    view_geometries: dict[str, ReconstructionViewGeometry] = {}
    for role, raw_view_path in triplet_paths.items():
        preprocessed, geometry = _extract_reconstruction_view_geometry(role, raw_view_path)
        preprocessed_views[role] = preprocessed
        view_geometries[role] = geometry

    front_geometry = view_geometries["front"]
    left_geometry = view_geometries["left"]
    right_geometry = view_geometries["right"]
    if left_geometry.image_shape != front_geometry.image_shape or right_geometry.image_shape != front_geometry.image_shape:
        raise RuntimeError("pose-normalized masks do not share a common image shape")

    height, width = front_geometry.image_shape
    valid_rows = front_geometry.valid_rows & left_geometry.valid_rows & right_geometry.valid_rows

    semi_major = 0.5 * front_geometry.widths
    d_right = 0.5 * right_geometry.widths
    d_left = 0.5 * left_geometry.widths
    # Paper Eq. from Algorithm 1 section:
    # d = sqrt((a^2 + b^2) / 2)  =>  b = sqrt(max(2*d^2 - a^2, 0))
    right_term = np.maximum(2.0 * d_right**2 - semi_major**2, 0.0)
    left_term = np.maximum(2.0 * d_left**2 - semi_major**2, 0.0)
    semi_minor_right = np.sqrt(right_term, dtype=np.float32)
    semi_minor_left = np.sqrt(left_term, dtype=np.float32)
    semi_minor = 0.5 * (semi_minor_right + semi_minor_left)
    center_front = front_geometry.centers.astype(np.float32)
    center_left = left_geometry.centers.astype(np.float32)
    center_right = right_geometry.centers.astype(np.float32)
    translation_left = center_left
    translation_right = center_right
    center_depth_right = (np.float32(math.sqrt(2.0) * 0.5) * (center_front + center_right)).astype(np.float32)
    center_depth_left = (np.float32(math.sqrt(2.0) * 0.5) * (center_front + center_left)).astype(np.float32)
    center_depth = (0.5 * (center_depth_right + center_depth_left)).astype(np.float32)
    valid_rows &= semi_major > 1e-3

    x_coords = np.arange(width, dtype=np.float32)[None, :]
    row_centers = front_geometry.centers[:, None].astype(np.float32)
    x_relative = x_coords - row_centers
    support_mask = np.where(preprocessed_views["front"].pose_normalized_mask > 0, 255, 0).astype(np.uint8)

    radicand = np.maximum(semi_major[:, None] ** 2 - x_relative**2, 0.0)
    row_scale = np.zeros((height, 1), dtype=np.float32)
    nonzero_rows = semi_major > 1e-6
    row_scale[nonzero_rows, 0] = (semi_minor[nonzero_rows] / semi_major[nonzero_rows]).astype(np.float32)
    arc_component = row_scale * np.sqrt(radicand, dtype=np.float32)
    z_up = arc_component + center_depth[:, None]
    z_down = (-arc_component) + center_depth[:, None]
    z_up[support_mask <= 0] = 0.0
    z_down[support_mask <= 0] = 0.0

    theta = np.zeros(height, dtype=np.float32)
    theta[valid_rows] = np.arctan((semi_minor[valid_rows] ** 2) / np.maximum(semi_major[valid_rows] ** 2, 1e-6)).astype(np.float32)
    x_crit = np.zeros(height, dtype=np.float32)
    x_crit[valid_rows] = (semi_major[valid_rows] * np.cos(theta[valid_rows])).astype(np.float32)

    x_relative_masked = np.where(support_mask > 0, x_relative, 0.0).astype(np.float32)
    right_stitch_mask = x_relative < x_crit[:, None]
    left_stitch_mask = x_relative <= (-x_crit[:, None])
    stitched_right = np.where(right_stitch_mask, z_down, z_up).astype(np.float32)
    stitched_left = np.where(left_stitch_mask, z_up, z_down).astype(np.float32)
    stitched_right[support_mask <= 0] = 0.0
    stitched_left[support_mask <= 0] = 0.0

    x_right_rot, depth_right = _rotate_depth_branch(x_relative_masked, stitched_right, 45.0)
    x_left_rot, depth_left = _rotate_depth_branch(x_relative_masked, stitched_left, -45.0)
    depth_front = z_up.astype(np.float32)
    depth_front[support_mask <= 0] = 0.0
    depth_left[support_mask <= 0] = 0.0
    depth_right[support_mask <= 0] = 0.0
    x_right_rot[support_mask <= 0] = 0.0
    x_left_rot[support_mask <= 0] = 0.0
    depth_front = np.nan_to_num(depth_front, nan=0.0, posinf=0.0, neginf=0.0)
    depth_left = np.nan_to_num(depth_left, nan=0.0, posinf=0.0, neginf=0.0)
    depth_right = np.nan_to_num(depth_right, nan=0.0, posinf=0.0, neginf=0.0)
    x_right_rot = np.nan_to_num(x_right_rot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_left_rot = np.nan_to_num(x_left_rot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth_gradient_labels = _build_depth_gradient_labels(
        depth_front=depth_front,
        depth_left=depth_left,
        depth_right=depth_right,
        support_mask=support_mask,
    )

    acquisition_id = _acquisition_name(sample.subject_id, sample.finger_id, sample.acquisition_id)
    reconstruction_dir = output_root / "reconstructions" / acquisition_id
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "surface_fused_mesh.obj",
        "surface_fused_3d.html",
        "surface_fused_3d.png",
        "surface_stitched_mesh.obj",
        "surface_stitched_3d.html",
        "surface_stitched_3d.png",
    ):
        stale_path = reconstruction_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    debug_views_dir = reconstruction_dir / "debug_views"
    debug_views_dir.mkdir(parents=True, exist_ok=True)

    front_centers = _interpolate_row_centers(front_geometry, front_geometry.centers.astype(np.float32))
    left_centers = _interpolate_row_centers(left_geometry, front_centers)
    right_centers = _interpolate_row_centers(right_geometry, front_centers)

    debug_view_paths: dict[str, dict[str, str]] = {}
    for role, raw_path in triplet_paths.items():
        # The reconstruction geometry is derived from the pose-normalized mask,
        # so keep the raw view, pose-normalized image, and pose mask together.
        preprocessed = preprocessed_views[role]
        role_paths = {
            "raw_input": debug_views_dir / f"{role}_raw_input.png",
            "pose_normalized": debug_views_dir / f"{role}_pose_normalized.png",
            "pose_mask": debug_views_dir / f"{role}_pose_mask.png",
            "preprocessed_input": debug_views_dir / f"{role}_preprocessed_input.png",
        }
        cv2.imwrite(str(role_paths["raw_input"]), load_bgr_image(raw_path))
        cv2.imwrite(str(role_paths["pose_normalized"]), preprocessed.pose_normalized_gray)
        cv2.imwrite(str(role_paths["pose_mask"]), preprocessed.pose_normalized_mask)
        cv2.imwrite(str(role_paths["preprocessed_input"]), preprocessed.preprocessed_gray)
        debug_view_paths[role] = {name: str(path.resolve()) for name, path in role_paths.items()}

    np.save(reconstruction_dir / "depth_front.npy", depth_front)
    np.save(reconstruction_dir / "depth_left.npy", depth_left)
    np.save(reconstruction_dir / "depth_right.npy", depth_right)
    _save_npz(reconstruction_dir / "depth_gradient_labels.npz", depth_gradient_labels)
    _save_npz(
        reconstruction_dir / "reconstruction_maps.npz",
        {
            "support_mask": (support_mask > 0).astype(np.uint8),
            "x_relative": x_relative_masked.astype(np.float32),
            "depth_front": depth_front.astype(np.float32),
            "depth_left": depth_left.astype(np.float32),
            "depth_right": depth_right.astype(np.float32),
            "x_left_rot": x_left_rot.astype(np.float32),
            "x_right_rot": x_right_rot.astype(np.float32),
            "stitched_left": stitched_left.astype(np.float32),
            "stitched_right": stitched_right.astype(np.float32),
            "front_centers": front_centers.astype(np.float32),
            "left_centers": left_centers.astype(np.float32),
            "right_centers": right_centers.astype(np.float32),
            "front_valid_rows": front_geometry.valid_rows.astype(np.uint8),
            "left_valid_rows": left_geometry.valid_rows.astype(np.uint8),
            "right_valid_rows": right_geometry.valid_rows.astype(np.uint8),
            "x_crit": x_crit.astype(np.float32),
        },
    )
    cv2.imwrite(str(reconstruction_dir / "support_mask.png"), support_mask)
    front_gradient = depth_gradient_labels["gradient_front"]
    unwarp = run_center_unwarping(
        image=preprocessed_views["front"].pose_normalized_gray,
        mask=support_mask,
        gradient_x=np.nan_to_num(front_gradient[0], nan=0.0),
        gradient_y=np.nan_to_num(front_gradient[1], nan=0.0),
    )
    center_unwarped_image_path = reconstruction_dir / "center_unwarped.png"
    center_unwarped_mask_path = reconstruction_dir / "center_unwarped_mask.png"
    center_unwarp_maps_path = reconstruction_dir / "center_unwarp_maps.npz"
    _save_uint8_png(center_unwarped_image_path, unwarp["unwarped_image"])
    cv2.imwrite(str(center_unwarped_mask_path), _to_uint8_mask(unwarp["unwarped_mask"]))
    inverse_unwarp = _build_inverse_unwarp_maps(
        unwarp["x_out"],
        unwarp["y_out"],
        unwarp["valid_mask"],
        unwarp["output_shape"],
    )
    _save_npz(
        center_unwarp_maps_path,
        {
            "u": unwarp["u"].astype(np.float32),
            "v": unwarp["v"].astype(np.float32),
            "x_new": unwarp["x_new"].astype(np.float32),
            "y_new": unwarp["y_new"].astype(np.float32),
            "x_out": unwarp["x_out"].astype(np.float32),
            "y_out": unwarp["y_out"].astype(np.float32),
            "center_point": np.asarray(unwarp["center_point"], dtype=np.int32),
            "valid_mask": unwarp["valid_mask"].astype(np.uint8),
            "unwarped_mask": unwarp["unwarped_mask"].astype(np.uint8),
            "output_offset_x": np.asarray(unwarp["output_offset_x"], dtype=np.int32),
            "output_offset_y": np.asarray(unwarp["output_offset_y"], dtype=np.int32),
            **inverse_unwarp,
        },
    )
    _write_row_measurements(
        reconstruction_dir,
        valid_rows=valid_rows,
        front_geometry=front_geometry,
        left_geometry=left_geometry,
        right_geometry=right_geometry,
        semi_major=semi_major,
        semi_minor=semi_minor,
        translation_left=translation_left,
        translation_right=translation_right,
        center_depth=center_depth,
        theta=theta,
        x_crit=x_crit,
    )
    _write_reconstruction_preview(
        reconstruction_dir,
        view_geometries=view_geometries,
        support_mask=support_mask,
        depth_front=depth_front,
        depth_left=depth_left,
        depth_right=depth_right,
    )
    surface_front_3d_html_path, surface_front_3d_png_path = _write_front_surface_visualizations(
        reconstruction_dir,
        depth_front=depth_front,
        support_mask=support_mask,
    )
    front_points, left_points, right_points = _compute_branch_point_clouds(
        x_relative=x_relative_masked,
        z_front=depth_front,
        stitched_left=stitched_left,
        stitched_right=stitched_right,
        support_mask=support_mask,
    )
    surface_all_branches_3d_html_path, surface_all_branches_3d_png_path = _write_all_branch_surface_visualizations(
        reconstruction_dir,
        front_points=front_points,
        left_points=left_points,
        right_points=right_points,
    )
    reprojection_report_path, reprojection_preview_path, reprojection_metrics = _write_reprojection_diagnostics(
        reconstruction_dir,
        front_geometry=front_geometry,
        left_geometry=left_geometry,
        right_geometry=right_geometry,
        x_relative=x_relative_masked,
        x_left_rot=x_left_rot,
        x_right_rot=x_right_rot,
        depth_front=depth_front,
        stitched_left=stitched_left,
        stitched_right=stitched_right,
        support_mask=support_mask,
    )

    meta = {
        "acquisition_id": acquisition_id,
        "subject_id": sample.subject_id,
        "finger_id": sample.finger_id,
        "acquisition_index": sample.acquisition_id,
        "input_views": {role: str(path.resolve()) for role, path in triplet_paths.items()},
        "view_roles": {"0": "front", "1": "left", "2": "right"},
        "geometry_space": "pose_normalized_pre_scale_mask",
        "algorithm": {
            "name": "multiview_ellipse_reconstruction",
            "front_formula": "z_up = (b/a)*sqrt(max(a^2 - x^2, 0)) + c_z",
            "lower_formula": "z_down = -(b/a)*sqrt(max(a^2 - x^2, 0)) + c_z",
            "semi_major": "a = d_front / 2",
            "semi_minor": (
                "d_right = right_width / 2, d_left = left_width / 2, "
                "b_right = sqrt(max(2*d_right^2 - a^2, 0)), "
                "b_left = sqrt(max(2*d_left^2 - a^2, 0)), "
                "b = 0.5 * (b_right + b_left)"
            ),
            "center_depth": "c_z = 0.5 * (sqrt(2)/2 * (l_f + l_r) + sqrt(2)/2 * (l_f + l_l))",
            "vanishing_angle": "theta = arctan(b^2 / a^2)",
            "critical_point": "x_crit = a * cos(theta)",
            "side_rotation_degrees": {"left": -45.0, "right": 45.0},
            "side_stitching": "piecewise stitch before rotation: Z_R uses z_down for x<x_crit else z_up; Z_L uses z_up for x<=-x_crit else z_down",
            "rotated_side_x_maps": {
                "left": "x_left_rot in reconstruction_maps.npz",
                "right": "x_right_rot in reconstruction_maps.npz",
            },
        },
        "outputs": {
            "depth_front": "depth_front.npy",
            "depth_left": "depth_left.npy",
            "depth_right": "depth_right.npy",
            "depth_gradient_labels": "depth_gradient_labels.npz",
            "reconstruction_maps": "reconstruction_maps.npz",
            "support_mask": "support_mask.png",
            "row_measurements": "row_measurements.json",
            "preview": "preview.png",
            "center_unwarped": "center_unwarped.png",
            "center_unwarped_mask": "center_unwarped_mask.png",
            "center_unwarp_maps": "center_unwarp_maps.npz",
            "surface_front_3d_html": "surface_front_3d.html",
            "surface_front_3d_png": "surface_front_3d.png",
            "surface_all_branches_3d_html": "surface_all_branches_3d.html",
            "surface_all_branches_3d_png": "surface_all_branches_3d.png",
            "reprojection_report": "reprojection_report.json",
            "reprojection_preview": "reprojection_preview.png",
            "debug_views_dir": "debug_views",
            "debug_views": {
                role: {
                    name: str(Path(path).relative_to(reconstruction_dir))
                    for name, path in paths.items()
                }
                for role, paths in debug_view_paths.items()
            },
        },
        "labels": {
            "depth_smoothing": "quadratic_2d_mls_on_reconstructed_depth",
            "depth_gradients": "partial derivatives gx and gy computed from smoothed depth surfaces inside the front-silhouette boundary mask",
            "support_mask": "post-reconstruction boundary mask derived from the front pose-normalized silhouette and used to zero depth/gradients outside the model footprint",
            "preview_depth_panels": "bottom-row preview panels are false-color depth visualizations of depth_front, depth_left, and depth_right normalized inside the boundary mask",
            "surface_front_3d": "interactive front-branch 3D viewer and fixed-angle PNG snapshot for debugging reconstruction shape",
            "surface_all_branches_3d": "interactive raw front/left/right branch point-cloud viewer and fixed-angle PNG snapshot in the shared reconstruction coordinate frame",
            "reprojection_check": "branch-matched row-wise silhouette check for front/left/right depth maps against their corresponding pose-normalized view masks, reported as IoU/precision/recall plus an observed-vs-projected overlay preview",
        },
        "reprojection": reprojection_metrics,
        "counts": {
            "valid_rows": int(np.count_nonzero(valid_rows)),
            "support_pixels": int(np.count_nonzero(support_mask)),
        },
        "shape": list(depth_front.shape),
        "center_unwarping": {
            "center_point": [int(unwarp["center_point"][0]), int(unwarp["center_point"][1])],
            "output_shape": list(unwarp["output_shape"]),
            "warnings": list(unwarp["warnings"]),
        },
        "version": 2,
    }
    (reconstruction_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return AcquisitionReconstructionResult(
        acquisition_id=acquisition_id,
        reconstruction_dir=str(reconstruction_dir.resolve()),
        depth_front_path=str((reconstruction_dir / "depth_front.npy").resolve()),
        depth_left_path=str((reconstruction_dir / "depth_left.npy").resolve()),
        depth_right_path=str((reconstruction_dir / "depth_right.npy").resolve()),
        depth_gradient_labels_path=str((reconstruction_dir / "depth_gradient_labels.npz").resolve()),
        reconstruction_maps_path=str((reconstruction_dir / "reconstruction_maps.npz").resolve()),
        support_mask_path=str((reconstruction_dir / "support_mask.png").resolve()),
        row_measurements_path=str((reconstruction_dir / "row_measurements.json").resolve()),
        meta_path=str((reconstruction_dir / "meta.json").resolve()),
        preview_path=str((reconstruction_dir / "preview.png").resolve()),
        center_unwarp_maps_path=str(center_unwarp_maps_path.resolve()),
        center_unwarped_image_path=str(center_unwarped_image_path.resolve()),
        center_unwarped_mask_path=str(center_unwarped_mask_path.resolve()),
        surface_front_3d_html_path=str(surface_front_3d_html_path.resolve()),
        surface_front_3d_png_path=str(surface_front_3d_png_path.resolve()),
        surface_all_branches_3d_html_path=str(surface_all_branches_3d_html_path.resolve()),
        surface_all_branches_3d_png_path=str(surface_all_branches_3d_png_path.resolve()),
        reprojection_report_path=str(reprojection_report_path.resolve()),
        reprojection_preview_path=str(reprojection_preview_path.resolve()),
        valid_row_count=int(np.count_nonzero(valid_rows)),
        support_pixel_count=int(np.count_nonzero(support_mask)),
        input_view_paths={role: str(path.resolve()) for role, path in triplet_paths.items()},
        debug_view_paths=debug_view_paths,
    )


def _build_manifest(dataset_root: Path) -> list[RawViewSample]:
    subject_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: int(path.name))
    samples: list[RawViewSample] = []

    for subject_index, subject_dir in enumerate(subject_dirs):
        subject_id = int(subject_dir.name)
        raw_dir = subject_dir / "raw"
        acquisitions: dict[tuple[int, int], list[Path]] = {}
        if raw_dir.exists():
            for raw_path in sorted(raw_dir.glob(f"{subject_id}_*_*.jpg")):
                stem_parts = raw_path.stem.split("_")
                if len(stem_parts) != 4:
                    continue
                _, finger_text, acquisition_text, _ = stem_parts
                finger_id = int(finger_text)
                acquisition_id = int(acquisition_text)
                acquisitions.setdefault((finger_id, acquisition_id), []).append(raw_path)

        for finger_id in range(1, 11):
            acquisition_ids = sorted(acq for found_finger_id, acq in acquisitions if found_finger_id == finger_id)
            for acquisition_id in acquisition_ids:
                raw_views = sorted(
                    acquisitions[(finger_id, acquisition_id)],
                    key=lambda path: int(path.stem.split("_")[-1]),
                )
                sire_path = subject_dir / f"SIRE-{subject_id}_{finger_id}_{acquisition_id}.bmp"
                variant_paths = {
                    suffix: str((subject_dir / f"SIRE-{subject_id}_{finger_id}_{acquisition_id}_{suffix}.bmp").resolve())
                    for suffix in VARIANT_SUFFIXES
                    if (subject_dir / f"SIRE-{subject_id}_{finger_id}_{acquisition_id}_{suffix}.bmp").exists()
                }
                sire_value: str | None = str(sire_path.resolve()) if sire_path.exists() else None
                resolved_raw_views = [str(path.resolve()) for path in raw_views]
                for raw_view_index, raw_view_path in enumerate(raw_views):
                    samples.append(
                        RawViewSample(
                            sample_id=f"s{subject_id:02d}_f{finger_id:02d}_a{acquisition_id:02d}_v{raw_view_index:02d}",
                            subject_id=subject_id,
                            subject_index=subject_index,
                            finger_id=finger_id,
                            acquisition_id=acquisition_id,
                            finger_class_id=subject_index * 10 + (finger_id - 1),
                            raw_image_path=str(raw_view_path.resolve()),
                            raw_view_index=raw_view_index,
                            sire_path=sire_value,
                            raw_view_paths=resolved_raw_views,
                            variant_paths=variant_paths,
                            is_extra_acquisition=acquisition_id > 2,
                        )
                    )

    return samples


def _write_manifest(samples: list[RawViewSample], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_json_path = output_root / "manifest.json"
    manifest_csv_path = output_root / "manifest.csv"

    payload = [asdict(sample) for sample in samples]
    manifest_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "sample_id",
        "subject_id",
        "subject_index",
        "finger_id",
        "acquisition_id",
        "finger_class_id",
        "raw_image_path",
        "raw_view_index",
        "sire_path",
        "raw_view_paths",
        "variant_paths",
        "is_extra_acquisition",
    ]
    with manifest_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in payload:
            row = dict(sample)
            row["raw_view_paths"] = json.dumps(row["raw_view_paths"])
            row["variant_paths"] = json.dumps(row["variant_paths"], sort_keys=True)
            writer.writerow(row)


def _collect_reconstruction_candidates(samples: Iterable[RawViewSample]) -> dict[tuple[int, int, int], RawViewSample]:
    candidates: dict[tuple[int, int, int], RawViewSample] = {}
    for sample in samples:
        if _resolve_reconstruction_triplet(sample.raw_view_paths) is None:
            continue
        key = _acquisition_key(sample.subject_id, sample.finger_id, sample.acquisition_id)
        existing = candidates.get(key)
        if existing is None or sample.raw_view_index < existing.raw_view_index:
            candidates[key] = sample
    return candidates


def _compute_gradient(gray_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = gray_image.astype(np.float32) / 255.0
    normalized *= (mask > 0).astype(np.float32)
    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)
    gradient[mask <= 0] = 0.0
    return gradient


def _build_orientation_one_hot(orientation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(orientation, 0.0, math.pi - 1e-6) * (180.0 / math.pi)).astype(np.int64)
    bins = np.clip(bins, 0, 179)
    one_hot = np.eye(180, dtype=np.float32)[bins]
    one_hot *= mask[..., None].astype(np.float32)
    return np.transpose(one_hot, (2, 0, 1))


def _resize_orientation_for_model(orientation: np.ndarray, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    cos2 = np.cos(2.0 * orientation).astype(np.float32) * mask.astype(np.float32)
    sin2 = np.sin(2.0 * orientation).astype(np.float32) * mask.astype(np.float32)
    cos2_small = _resize_float(cos2, shape)
    sin2_small = _resize_float(sin2, shape)
    orientation_small = 0.5 * np.arctan2(sin2_small, cos2_small)
    orientation_small = _normalize_angle_pi(orientation_small)
    return orientation_small


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-6:
        return None
    return rounded


def _standardize_minutiae(records: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    minutiae: list[dict[str, Any]] = []
    for record in records:
        x = float(record.get("x", 0.0))
        y = float(record.get("y", 0.0))
        theta = record.get("theta")
        if theta is None:
            theta = record.get("angle")
        if theta is None:
            theta = record.get("direction", 0.0)
        score = record.get("score")
        minutia_type = record.get("type")
        if minutia_type is None:
            minutia_type = _coerce_optional_int(record.get("class"))
        minutiae.append(
            {
                "x": x,
                "y": y,
                "theta": _normalize_angle_2pi_scalar(float(theta)),
                "score": float(score) if score is not None else None,
                "type": minutia_type,
                "source": source,
            }
        )
    return minutiae


def _is_downloadable_model(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _download_file_atomic(url: str, destination: Path, timeout_seconds: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    try:
        request = urlrequest.Request(url, headers={"User-Agent": "fingerflow-bootstrap/1.0"})
        with urlrequest.urlopen(request, timeout=timeout_seconds) as response, temp_path.open("wb") as file_obj:
            content_type = response.headers.get_content_type()
            if content_type and content_type.startswith("text/"):
                raise RuntimeError(f"unexpected content-type '{content_type}' while downloading model from {url}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
        if temp_path.stat().st_size <= 0:
            raise RuntimeError(f"downloaded empty file from {url}")
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _ensure_model_file(model_name: str, model_dir: Path, filename: str, urls: tuple[str, ...]) -> Path:
    target = model_dir / filename
    if _is_downloadable_model(target):
        return target
    model_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            _download_file_atomic(url, target)
            if _is_downloadable_model(target):
                return target
            errors.append(f"download from {url} produced an invalid file")
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        f"failed to download FingerFlow model '{model_name}' to {target}. Tried URLs: {'; '.join(errors)}"
    )


def ensure_fingerflow_models(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    model_dir = model_dir.expanduser().resolve()
    coarse = _ensure_model_file(
        "coarse",
        model_dir,
        FINGERFLOW_MODEL_SOURCES["coarse"]["filename"],
        FINGERFLOW_MODEL_SOURCES["coarse"]["urls"],
    )
    fine = _ensure_model_file(
        "fine",
        model_dir,
        FINGERFLOW_MODEL_SOURCES["fine"]["filename"],
        FINGERFLOW_MODEL_SOURCES["fine"]["urls"],
    )
    classify = _ensure_model_file(
        "classify",
        model_dir,
        FINGERFLOW_MODEL_SOURCES["classify"]["filename"],
        FINGERFLOW_MODEL_SOURCES["classify"]["urls"],
    )
    core = _ensure_model_file(
        "core",
        model_dir,
        FINGERFLOW_MODEL_SOURCES["core"]["filename"],
        FINGERFLOW_MODEL_SOURCES["core"]["urls"],
    )
    return coarse, fine, classify, core


def load_bgr_for_fingerflow(path: Path) -> np.ndarray:
    image = load_image_unchanged(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"unsupported image shape for FingerFlow extraction: {image.shape}")


def _df_to_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return list(data.to_dict(orient="records"))
    return list(data)


def _install_core_net_compat_fallback() -> None:
    try:
        import fingerflow.extractor.extractor as extractor_module
    except Exception:
        return
    if getattr(extractor_module, "_biometric_core_fallback_enabled", False):
        return

    class _NoCoreNet:
        def __init__(self, core_net_path: str):
            self._core_net_path = core_net_path

        def detect_fingerprint_core(self, raw_image_data: np.ndarray):
            return np.empty((0,))

    extractor_module.CoreNet = _NoCoreNet
    extractor_module._biometric_core_fallback_enabled = True


def _get_fingerflow_extractor_class():
    global _FINGERFLOW_EXTRACTOR_CLASS
    if _FINGERFLOW_EXTRACTOR_CLASS is None:
        _tensorflow_gpu_summary(require_gpu=_current_runtime_config().gpu_only and _current_runtime_config().execution_target == "kaggle")
        from fingerflow.extractor import Extractor

        _FINGERFLOW_EXTRACTOR_CLASS = Extractor
    return _FINGERFLOW_EXTRACTOR_CLASS


def extract_minutiae_with_fingerflow(
    source_image_path: Path,
    enhanced_image_path: Path,
    model_paths: tuple[Path, Path, Path, Path],
    minutiae_json_path: Path,
    minutiae_csv_path: Path,
    core_csv_path: Path,
) -> tuple[int, int]:
    coarse_path, fine_path, classify_path, core_path = model_paths
    image = load_bgr_for_fingerflow(enhanced_image_path)
    cache_key = (str(coarse_path), str(fine_path), str(classify_path), str(core_path))
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor_class = _get_fingerflow_extractor_class()
        try:
            extractor = extractor_class(str(coarse_path), str(fine_path), str(classify_path), str(core_path))
            _EXTRACTOR_CACHE[cache_key] = extractor
        except Exception as exc:
            if "A KerasTensor cannot be used as input to a TensorFlow function" in str(exc):
                _install_core_net_compat_fallback()
                extractor = extractor_class(str(coarse_path), str(fine_path), str(classify_path), str(core_path))
                _EXTRACTOR_CACHE[cache_key] = extractor
            else:
                raise

    extracted = extractor.extract_minutiae(image)
    minutiae_df = extracted.get("minutiae")
    core_df = extracted.get("core")
    minutiae_json_path.parent.mkdir(parents=True, exist_ok=True)
    minutiae_json_path.write_text(
        json.dumps(
            {
                "source_image": str(source_image_path),
                "enhanced_image": str(enhanced_image_path),
                "minutiae": _df_to_records(minutiae_df),
                "core": _df_to_records(core_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if hasattr(minutiae_df, "to_csv"):
        minutiae_df.to_csv(minutiae_csv_path, index=False)
    if hasattr(core_df, "to_csv"):
        core_df.to_csv(core_csv_path, index=False)
    minutiae_count = int(getattr(minutiae_df, "shape", [0])[0]) if minutiae_df is not None else 0
    core_count = int(getattr(core_df, "shape", [0])[0]) if core_df is not None else 0
    return minutiae_count, core_count


def _extract_fingerflow_minutiae(
    source_image_path: Path,
    enhanced_image_path: Path,
    bundle_dir: Path,
    fingerflow_model_dir: Path,
) -> list[dict[str, Any]]:
    model_paths = _ensure_fingerflow_models(fingerflow_model_dir)
    raw_json_path = bundle_dir / "_fingerflow_raw.json"
    raw_csv_path = bundle_dir / "_fingerflow_raw.csv"
    raw_core_path = bundle_dir / "_fingerflow_core.csv"
    try:
        started_at = time.perf_counter()
        extract_minutiae_with_fingerflow(
            source_image_path,
            enhanced_image_path,
            model_paths,
            raw_json_path,
            raw_csv_path,
            raw_core_path,
        )
        _record_stage_time("fingerflow", time.perf_counter() - started_at)
        payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
        return _standardize_minutiae(payload.get("minutiae", []), source="fingerflow")
    finally:
        for path in (raw_json_path, raw_csv_path, raw_core_path):
            if path.exists():
                path.unlink()


def _extract_fingerflow_minutiae_wsl(
    source_image_path: Path,
    enhanced_image_path: Path,
    bundle_dir: Path,
    fingerflow_model_dir: Path,
    backend: FingerflowBackendConfig,
) -> list[dict[str, Any]]:
    raw_json_path = bundle_dir / "_fingerflow_raw.json"
    raw_csv_path = bundle_dir / "_fingerflow_raw.csv"
    raw_core_path = bundle_dir / "_fingerflow_core.csv"
    bridge_path = REPO_ROOT / "fingerflow_bridge.py"
    command = "\n".join(
        [
            "set -eu",
            backend.wsl_activate,
            f"cd {_bash_quote(_to_wsl_path(REPO_ROOT))}",
            "PYTHON_BIN=python",
            "GPU_LIB_PATHS=\"$($PYTHON_BIN - <<'PY'\n"
            "import site\n"
            "from pathlib import Path\n"
            "\n"
            "paths = []\n"
            "for base in site.getsitepackages():\n"
            "    paths.extend(sorted(str(path) for path in Path(base).glob('nvidia/*/lib')))\n"
            "print(':'.join(paths))\n"
            "PY\n"
            ")\"",
            f"export FINGERFLOW_ALLOW_CPU={'0' if _current_runtime_config().gpu_only else '${FINGERFLOW_ALLOW_CPU:-0}'}",
            "export LD_LIBRARY_PATH=\"/usr/lib/wsl/lib${GPU_LIB_PATHS:+:$GPU_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"",
            "exec \"$PYTHON_BIN\" "
            + _bash_quote(_to_wsl_path(bridge_path))
            + " --source-image "
            + _bash_quote(_to_wsl_path(source_image_path))
            + " --enhanced-image "
            + _bash_quote(_to_wsl_path(enhanced_image_path))
            + " --model-dir "
            + _bash_quote(_to_wsl_path(fingerflow_model_dir))
            + " --minutiae-json "
            + _bash_quote(_to_wsl_path(raw_json_path))
            + " --minutiae-csv "
            + _bash_quote(_to_wsl_path(raw_csv_path))
            + " --core-csv "
            + _bash_quote(_to_wsl_path(raw_core_path)),
        ]
    )
    try:
        started_at = time.perf_counter()
        completed = subprocess.run(
            ["wsl", "-d", backend.wsl_distro, "bash", "-lc", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown wsl fingerflow failure"
            raise RuntimeError(stderr)
        _record_stage_time("fingerflow", time.perf_counter() - started_at)
        payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
        return _standardize_minutiae(payload.get("minutiae", []), source="fingerflow_wsl")
    finally:
        for path in (raw_json_path, raw_csv_path, raw_core_path):
            if path.exists():
                path.unlink()


def _extract_pyfing_minutiae(gray_image: np.ndarray) -> list[dict[str, Any]]:
    raw_minutiae = pyfing.minutiae_extraction(gray_image, dpi=DEFAULT_DPI)
    rows = []
    for minutia in raw_minutiae:
        rows.append(
            {
                "x": getattr(minutia, "x"),
                "y": getattr(minutia, "y"),
                "direction": getattr(minutia, "direction"),
                "quality": getattr(minutia, "quality", None),
                "type": getattr(minutia, "type", None),
            }
        )
    standardized = _standardize_minutiae(rows, source="pyfing")
    for item, row in zip(standardized, rows):
        if row.get("quality") is not None:
            item["score"] = float(row["quality"])
    return standardized


def _enhance_for_minutiae(gray_image: np.ndarray) -> np.ndarray:
    if fingerprint_enhancer is None:
        return gray_image.copy()
    enhanced = fingerprint_enhancer.enhance_fingerprint(gray_image)
    return _to_uint8_image(enhanced)


def _extract_direct_sample_minutiae(
    prepared: PreparedBundleArtifacts,
    minutiae_enhanced_path: Path,
    fingerflow_model_dir: Path,
    fingerflow_backend: FingerflowBackendConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Extract fallback minutiae directly from this sample's enhanced 2D image."""
    minutiae_source = f"fingerflow_{fingerflow_backend.backend}"
    try:
        if fingerflow_backend.backend == "wsl":
            minutiae = _extract_fingerflow_minutiae_wsl(
                prepared.image_path,
                minutiae_enhanced_path,
                prepared.bundle_dir,
                fingerflow_model_dir,
                backend=fingerflow_backend,
            )
        else:
            minutiae = _extract_fingerflow_minutiae(
                prepared.image_path,
                minutiae_enhanced_path,
                prepared.bundle_dir,
                fingerflow_model_dir,
            )
        return minutiae, minutiae_source
    except Exception:
        fallback_started_at = time.perf_counter()
        minutiae = _extract_pyfing_minutiae(prepared.enhanced_image)
        _record_stage_time("pyfing_fallback", time.perf_counter() - fallback_started_at)
        return minutiae, "pyfing_fallback"


def _rasterize_minutiae(
    minutiae: list[dict[str, Any]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    mask_small: np.ndarray,
) -> dict[str, np.ndarray]:
    target_height, target_width = target_shape
    source_height, source_width = source_shape
    score_map = np.zeros((target_height, target_width), dtype=np.float32)
    valid_mask = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_x = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_y = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_x_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_y_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_orientation = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_orientation_vec = np.zeros((2, target_height, target_width), dtype=np.float32)
    ownership = np.full((target_height, target_width), -1.0, dtype=np.float32)
    center_distance = np.full((target_height, target_width), np.inf, dtype=np.float32)
    cell_width = float(source_width) / float(max(target_width, 1))
    cell_height = float(source_height) / float(max(target_height, 1))

    for minutia in minutiae:
        x = float(minutia["x"])
        y = float(minutia["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        cell_x = int(np.clip(math.floor(x * target_width / max(source_width, 1)), 0, target_width - 1))
        cell_y = int(np.clip(math.floor(y * target_height / max(source_height, 1)), 0, target_height - 1))
        if mask_small[cell_y, cell_x] <= 0:
            continue
        score = minutia.get("score")
        score_value = float(score) if score is not None and math.isfinite(float(score)) else 1.0
        score_value = float(np.clip(score_value, 0.0, 1.0))
        cell_origin_x = cell_x * cell_width
        cell_origin_y = cell_y * cell_height
        local_x = (x - cell_origin_x) / max(cell_width, 1e-6)
        local_y = (y - cell_origin_y) / max(cell_height, 1e-6)
        local_x = float(np.clip(local_x, 0.0, 1.0 - 1e-6))
        local_y = float(np.clip(local_y, 0.0, 1.0 - 1e-6))
        distance = float((local_x - 0.5) ** 2 + (local_y - 0.5) ** 2)
        replace = False
        if score_value > ownership[cell_y, cell_x]:
            replace = True
        elif abs(score_value - ownership[cell_y, cell_x]) <= 1e-6 and distance < center_distance[cell_y, cell_x]:
            replace = True
        if not replace:
            continue
        ownership[cell_y, cell_x] = score_value
        center_distance[cell_y, cell_x] = distance
        score_map[cell_y, cell_x] = 1.0
        valid_mask[cell_y, cell_x] = 1.0
        minutia_x_bin = int(np.clip(math.floor(local_x * 8.0), 0, 7))
        minutia_y_bin = int(np.clip(math.floor(local_y * 8.0), 0, 7))
        minutia_theta = _normalize_angle_2pi_scalar(float(minutia["theta"]))
        minutia_orientation_bin = int(np.clip(math.floor(minutia_theta * (360.0 / (2.0 * math.pi))), 0, 359))
        minutia_x[cell_y, cell_x] = minutia_x_bin
        minutia_y[cell_y, cell_x] = minutia_y_bin
        minutia_x_offset[cell_y, cell_x] = local_x
        minutia_y_offset[cell_y, cell_x] = local_y
        minutia_orientation[cell_y, cell_x] = minutia_orientation_bin
        minutia_orientation_vec[0, cell_y, cell_x] = float(math.cos(minutia_theta))
        minutia_orientation_vec[1, cell_y, cell_x] = float(math.sin(minutia_theta))

    valid_mask *= mask_small
    score_map *= mask_small
    score_map = np.where(score_map > 0.0, 1.0, 0.0).astype(np.float32)
    valid_mask = np.where(valid_mask > 0.0, 1.0, 0.0).astype(np.float32)
    return {
        "minutia_score": score_map[np.newaxis, ...].astype(np.float32),
        "minutia_valid_mask": valid_mask[np.newaxis, ...].astype(np.float32),
        "minutia_x": minutia_x.astype(np.int64),
        "minutia_y": minutia_y.astype(np.int64),
        "minutia_x_offset": minutia_x_offset[np.newaxis, ...].astype(np.float32),
        "minutia_y_offset": minutia_y_offset[np.newaxis, ...].astype(np.float32),
        "minutia_orientation": minutia_orientation.astype(np.int64),
        "minutia_orientation_vec": minutia_orientation_vec.astype(np.float32),
    }


def _build_featurenet_targets(
    gray_image: np.ndarray,
    mask: np.ndarray,
    orientation: np.ndarray,
    ridge_period: np.ndarray,
    gradient: np.ndarray | None,
    minutiae: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    output_shape = _compute_output_shape(*gray_image.shape)
    mask_small_dense = _resize_mask(mask, output_shape)
    mask_small_points = _downsample_mask_for_points(mask, output_shape)

    orientation_small = _resize_orientation_for_model(orientation, mask.astype(np.float32) / 255.0, output_shape)
    orientation_one_hot = _build_orientation_one_hot(orientation_small, mask_small_dense)

    ridge_small = _resize_float(ridge_period, output_shape) * mask_small_dense
    ridge_max = float(ridge_small.max(initial=0.0))
    if ridge_max > 0.0:
        ridge_small = ridge_small / ridge_max

    minutia_targets = _rasterize_minutiae(minutiae, gray_image.shape, output_shape, mask_small_points)
    targets = {
        "orientation": orientation_one_hot.astype(np.float32),
        "ridge_period": ridge_small[np.newaxis, ...].astype(np.float32),
        "minutia_score": minutia_targets["minutia_score"],
        "minutia_valid_mask": minutia_targets["minutia_valid_mask"],
        "minutia_x": minutia_targets["minutia_x"],
        "minutia_y": minutia_targets["minutia_y"],
        "minutia_x_offset": minutia_targets["minutia_x_offset"],
        "minutia_y_offset": minutia_targets["minutia_y_offset"],
        "minutia_orientation": minutia_targets["minutia_orientation"],
        "minutia_orientation_vec": minutia_targets["minutia_orientation_vec"],
        "output_mask": mask_small_dense[np.newaxis, ...].astype(np.float32),
    }
    if gradient is not None:
        grad_x_small = _resize_float(gradient[:, :, 0], output_shape) * mask_small_dense
        grad_y_small = _resize_float(gradient[:, :, 1], output_shape) * mask_small_dense
        targets["gradient"] = np.stack([grad_x_small, grad_y_small], axis=0).astype(np.float32)
    return targets


def _build_targets_and_count_rasterized_minutiae(
    prepared: PreparedBundleArtifacts,
    minutiae: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], int]:
    targets = _build_featurenet_targets(
        gray_image=prepared.gray_image,
        mask=prepared.mask,
        orientation=prepared.orientation,
        ridge_period=prepared.ridge_period,
        gradient=prepared.reconstruction_gradient,
        minutiae=minutiae,
    )
    valid_count = int(np.count_nonzero(targets["minutia_valid_mask"]))
    return targets, valid_count


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _write_visualization(
    bundle_dir: Path,
    gray_image: np.ndarray,
    mask: np.ndarray,
    orientation: np.ndarray,
    ridge_period: np.ndarray,
    gradient: np.ndarray,
    minutiae: list[dict[str, Any]],
) -> None:
    base_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    mask_panel = cv2.cvtColor(_to_uint8_mask(mask), cv2.COLOR_GRAY2BGR)
    ridge_panel = cv2.applyColorMap(_to_uint8_image(np.where(mask > 0, ridge_period, 0.0)), cv2.COLORMAP_TURBO)
    grad_mag = np.linalg.norm(gradient, axis=2)
    grad_panel = cv2.applyColorMap(_to_uint8_image(grad_mag), cv2.COLORMAP_VIRIDIS)

    orientation_overlay = base_bgr.copy()
    step = max(16, min(gray_image.shape) // 24)
    line_len = max(6, step // 2)
    for y in range(step // 2, gray_image.shape[0], step):
        for x in range(step // 2, gray_image.shape[1], step):
            if mask[y, x] <= 0:
                continue
            theta = float(orientation[y, x])
            dx = int(round(math.cos(theta) * line_len))
            dy = int(round(math.sin(theta) * line_len))
            cv2.line(orientation_overlay, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 0), 1, cv2.LINE_AA)

    minutiae_overlay = base_bgr.copy()
    for minutia in minutiae:
        x = int(round(float(minutia["x"])))
        y = int(round(float(minutia["y"])))
        theta = float(minutia["theta"])
        cv2.circle(minutiae_overlay, (x, y), 3, (0, 0, 255), 1, cv2.LINE_AA)
        dx = int(round(math.cos(theta) * 10))
        dy = int(round(math.sin(theta) * 10))
        cv2.line(minutiae_overlay, (x, y), (x + dx, y + dy), (255, 255, 0), 1, cv2.LINE_AA)

    top = np.hstack([base_bgr, mask_panel, ridge_panel])
    bottom = np.hstack([grad_panel, orientation_overlay, minutiae_overlay])
    preview = np.vstack([top, bottom])
    cv2.imwrite(str(bundle_dir / "preview.png"), preview)


def _load_sample_input(
    sample: RawViewSample,
    output_root: Path,
    visualize: bool,
    reconstruction: AcquisitionReconstructionResult | None,
) -> LoadedSampleInput:
    started_at = time.perf_counter()
    image_path = Path(sample.raw_image_path)
    loaded = LoadedSampleInput(
        sample=sample,
        image_path=image_path,
        full_bgr=load_bgr_image(image_path),
        visualize=visualize,
        bundle_dir=output_root / "samples" / sample.sample_id,
        reconstruction=reconstruction,
    )
    _record_stage_time("load_input", time.perf_counter() - started_at)
    return loaded


def _prepare_bundle_from_loaded(loaded: LoadedSampleInput, dpi: int) -> PreparedBundleArtifacts:
    preprocessed = _preprocess_contactless_bgr(loaded.full_bgr)
    gray_image = preprocessed.preprocessed_gray
    mask = preprocessed.final_mask.copy()
    cpu_started_at = time.perf_counter()
    orientation = pyfing.orientation_field_estimation(gray_image, mask, dpi=dpi, method="SNFOE")
    orientation = _normalize_angle_pi(orientation)
    orientation[mask <= 0] = 0.0

    ridge_period = pyfing.frequency_estimation(gray_image, orientation, mask, dpi=dpi, method="SNFFE")
    ridge_period = np.nan_to_num(ridge_period.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ridge_period = np.clip(ridge_period, 0.0, None)
    ridge_period[mask <= 0] = 0.0

    visualization_gradient = _compute_gradient(gray_image, mask)
    reconstruction_gradient = _load_reconstruction_gradient_for_sample(loaded.sample, loaded.reconstruction)
    masked_image = gray_image.copy()
    masked_image[mask <= 0] = 0
    enhanced_image = _enhance_for_minutiae(masked_image)
    _record_stage_time("cpu_preprocess", time.perf_counter() - cpu_started_at)

    return PreparedBundleArtifacts(
        sample=loaded.sample,
        bundle_dir=loaded.bundle_dir,
        image_path=loaded.image_path,
        preprocessed=preprocessed,
        gray_image=gray_image,
        mask=mask,
        orientation=orientation,
        ridge_period=ridge_period,
        visualization_gradient=visualization_gradient,
        reconstruction_gradient=reconstruction_gradient,
        masked_image=masked_image,
        enhanced_image=enhanced_image,
        visualize=loaded.visualize,
        reconstruction=loaded.reconstruction,
    )


def _build_bundle_meta(
    prepared: PreparedBundleArtifacts,
    featurenet_targets: dict[str, np.ndarray],
    minutiae: list[dict[str, Any]],
    minutiae_source: str,
    minutiae_ground_truth_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = prepared.sample
    actual_scale_x, actual_scale_y = _training_frame_scale_from_preprocessed(prepared.preprocessed)
    output_shape = featurenet_targets["output_mask"].shape[-2:]
    point_mask = _downsample_mask_for_points(prepared.mask, output_shape)
    rasterized_minutiae_count = int(np.count_nonzero(featurenet_targets["minutia_valid_mask"]))
    meta = {
        "sample_id": sample.sample_id,
        "subject_id": sample.subject_id,
        "finger_id": sample.finger_id,
        "acquisition_id": sample.acquisition_id,
        "raw_view_index": sample.raw_view_index,
        "finger_class_id": sample.finger_class_id,
        "raw_image_path": sample.raw_image_path,
        "sire_path": sample.sire_path,
        "raw_view_paths": sample.raw_view_paths,
        "variant_paths": sample.variant_paths,
        "approximate": True,
        "input_domain": "raw_contactless_preprocessed",
        "execution": {
            "execution_target": _current_runtime_config().execution_target,
            "gpu_only": _current_runtime_config().gpu_only,
            "stage_devices": {
                "rembg": "gpu" if _SELECTED_REMBG_PROVIDERS and "CUDAExecutionProvider" in _SELECTED_REMBG_PROVIDERS else "cpu",
                "fingerflow": _SELECTED_FINGERFLOW_DEVICE,
                "classical_preprocess": "cpu",
                "bundle_write": "cpu",
            },
        },
        "preprocessing": {
            "masking": prepared.preprocessed.mask_source,
            "contrast": "clahe_clip_4_tile_4_then_clip_1_tile_10",
            "pose_normalization": {
                "method": "contour_centerline_rotated_to_vertical",
                "rotation_degrees": prepared.preprocessed.pose_rotation_degrees,
            },
            "ridge_frequency_normalization": {
                "method": "central_region_spacing_rescaled_to_10_pixels",
                "target_spacing_pixels": 10.0,
                "scale_factor": prepared.preprocessed.ridge_scale_factor,
                "actual_scale_x": actual_scale_x,
                "actual_scale_y": actual_scale_y,
            },
            "saved_artifacts": [
                "raw_input.png",
                "preprocess_normalized.png",
                "preprocess_pose_normalized.png",
                "preprocess_pose_mask.png",
                "preprocessed_input.png",
                "preprocess_mask.png",
            ],
        },
        "methods": {
            "mask": "copied_main_segmentation_used as primary preprocessing and label mask",
            "geometry_mask": "pose_normalized_pre_scale_mask retained for multiview reconstruction",
            "orientation": "pyfing.orientation_field_estimation(method='SNFOE')",
            "ridge_period": "pyfing.frequency_estimation(method='SNFFE')",
            "gradient": "MLS-smoothed reconstruction depth partial derivatives resized to the FeatureNet output grid",
            "gradient_visualization": "cv2.Sobel on masked grayscale preprocessed raw image for preview/debug only",
            "enhancement": "fingerprint_enhancer.enhance_fingerprint",
            "minutiae": minutiae_source,
            "minutiae_ground_truth_pipeline": "reconstruction_unwarp_reproject" if minutiae_ground_truth_details and minutiae_ground_truth_details.get("mode") == "reconstruction_backed" else "direct_per_sample_extraction",
            "featurenet_adapter": "dense orientation/ridge/gradient resize plus paper-style minutiae heatmap and precise subcell labels",
            "minutiae_targets": "binary score heatmap with continuous x/y offsets plus legacy 8-bin x/y labels, legacy 360-bin orientation labels, and optional cos/sin orientation vectors",
        },
        "shapes": {
            "input_image": list(prepared.gray_image.shape),
            "featurenet_output": list(featurenet_targets["ridge_period"].shape[-2:]),
        },
        "counts": {
            "raw_views": len(sample.raw_view_paths),
            "variants": len(sample.variant_paths),
            "minutiae": len(minutiae),
            "minutia_support_pixels": rasterized_minutiae_count,
            "rasterized_minutiae_count": rasterized_minutiae_count,
            "output_mask_pixels": int(np.count_nonzero(featurenet_targets["output_mask"])),
            "point_mask_pixels": int(np.count_nonzero(point_mask)),
            "dense_output_mask_pixels": int(np.count_nonzero(featurenet_targets["output_mask"])),
        },
        "version": 2,
    }
    if prepared.reconstruction is not None and sample.raw_view_index in {0, 1, 2}:
        role = {0: "front", 1: "left", 2: "right"}[sample.raw_view_index]
        meta["multiview_reconstruction"] = {
            "role": role,
            "acquisition_id": prepared.reconstruction.acquisition_id,
            "reconstruction_dir": prepared.reconstruction.reconstruction_dir,
            "depth_front_path": prepared.reconstruction.depth_front_path,
            "depth_left_path": prepared.reconstruction.depth_left_path,
            "depth_right_path": prepared.reconstruction.depth_right_path,
            "depth_gradient_labels_path": prepared.reconstruction.depth_gradient_labels_path,
            "reconstruction_maps_path": prepared.reconstruction.reconstruction_maps_path,
            "support_mask_path": prepared.reconstruction.support_mask_path,
            "preview_path": prepared.reconstruction.preview_path,
            "center_unwarp_maps_path": prepared.reconstruction.center_unwarp_maps_path,
            "center_unwarped_image_path": prepared.reconstruction.center_unwarped_image_path,
            "center_unwarped_mask_path": prepared.reconstruction.center_unwarped_mask_path,
            "surface_front_3d_html_path": prepared.reconstruction.surface_front_3d_html_path,
            "surface_front_3d_png_path": prepared.reconstruction.surface_front_3d_png_path,
            "surface_all_branches_3d_html_path": prepared.reconstruction.surface_all_branches_3d_html_path,
            "surface_all_branches_3d_png_path": prepared.reconstruction.surface_all_branches_3d_png_path,
            "reprojection_report_path": prepared.reconstruction.reprojection_report_path,
            "reprojection_preview_path": prepared.reconstruction.reprojection_preview_path,
        }
    if minutiae_ground_truth_details is not None:
        meta["minutiae_ground_truth"] = minutiae_ground_truth_details
    return meta


def _persist_bundle(payload: BundleWritePayload) -> None:
    started_at = time.perf_counter()
    bundle_dir = payload.bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bundle_dir / "minutiae_enhanced.png"), payload.enhanced_image)
    cv2.imwrite(str(bundle_dir / "raw_input.png"), payload.preprocessed.raw_gray)
    cv2.imwrite(str(bundle_dir / "preprocess_normalized.png"), payload.preprocessed.normalized_gray)
    cv2.imwrite(str(bundle_dir / "preprocess_pose_normalized.png"), payload.preprocessed.pose_normalized_gray)
    cv2.imwrite(str(bundle_dir / "preprocess_pose_mask.png"), payload.preprocessed.pose_normalized_mask)
    cv2.imwrite(str(bundle_dir / "preprocessed_input.png"), payload.preprocessed.preprocessed_gray)
    cv2.imwrite(str(bundle_dir / "preprocess_mask.png"), payload.preprocessed.final_mask)
    cv2.imwrite(str(bundle_dir / "mask.png"), payload.mask)
    cv2.imwrite(str(bundle_dir / "masked_image.png"), payload.masked_image)
    np.save(bundle_dir / "orientation.npy", payload.orientation.astype(np.float32))
    np.save(bundle_dir / "ridge_period.npy", payload.ridge_period.astype(np.float32))
    np.save(bundle_dir / "gradient_visualization.npy", payload.visualization_gradient.astype(np.float32))
    (bundle_dir / "minutiae.json").write_text(json.dumps(payload.minutiae, indent=2), encoding="utf-8")
    _save_npz(bundle_dir / "featurenet_targets.npz", payload.featurenet_targets)
    (bundle_dir / "meta.json").write_text(json.dumps(payload.meta, indent=2), encoding="utf-8")
    if payload.visualize:
        _write_visualization(
            bundle_dir,
            payload.gray_image,
            payload.mask,
            payload.orientation,
            payload.ridge_period,
            payload.visualization_gradient,
            payload.minutiae,
        )
    _record_stage_time("write_bundle", time.perf_counter() - started_at)


def _generate_bundle_from_loaded(
    loaded: LoadedSampleInput,
    fingerflow_model_dir: Path,
    dpi: int,
    fingerflow_backend: FingerflowBackendConfig,
) -> tuple[dict[str, Any], BundleWritePayload]:
    prepared = _prepare_bundle_from_loaded(loaded, dpi=dpi)
    minutiae_enhanced_path = prepared.bundle_dir / "minutiae_enhanced.png"
    prepared.bundle_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(minutiae_enhanced_path), prepared.enhanced_image)

    minutiae_ground_truth_details: dict[str, Any] | None = None
    minutiae_source = f"fingerflow_{fingerflow_backend.backend}"
    minutiae: list[dict[str, Any]]
    featurenet_targets: dict[str, np.ndarray]
    rasterized_count = 0
    reconstruction = prepared.reconstruction
    role = _view_role_for_sample(prepared.sample)

    postprocess_started_at = time.perf_counter()
    if reconstruction is not None and role is not None:
        try:
            canonical_minutiae, canonical_source, canonical_details = _load_or_extract_canonical_reconstruction_minutiae(
                reconstruction,
                fingerflow_model_dir,
                fingerflow_backend,
            )
            unwarp_maps = _load_npz_arrays(Path(reconstruction.center_unwarp_maps_path))
            reconstruction_maps = _load_npz_arrays(Path(reconstruction.reconstruction_maps_path))
            reprojected_minutiae, remap_details = _remap_unwarped_minutiae_to_sample(
                canonical_minutiae,
                unwarp_maps,
                reconstruction_maps,
                prepared.sample,
                prepared.preprocessed,
            )
            reconstruction_targets, reconstruction_rasterized_count = _build_targets_and_count_rasterized_minutiae(
                prepared,
                reprojected_minutiae,
            )
            if reconstruction_rasterized_count >= MIN_RASTERIZED_MINUTIAE_FOR_RECONSTRUCTION:
                minutiae = reprojected_minutiae
                minutiae_source = f"{canonical_source}_reprojected_{role}"
                featurenet_targets = reconstruction_targets
                rasterized_count = reconstruction_rasterized_count
                minutiae_ground_truth_details = {
                    "mode": "reconstruction_backed",
                    "canonical_source": canonical_source,
                    "view_role": role,
                    "canonical_minutiae_count": len(canonical_minutiae),
                    "reprojected_minutiae_count": len(minutiae),
                    "rasterized_minutiae_count": rasterized_count,
                    "center_unwarped_image_path": reconstruction.center_unwarped_image_path,
                    "center_unwarped_mask_path": reconstruction.center_unwarped_mask_path,
                    "center_unwarp_maps_path": reconstruction.center_unwarp_maps_path,
                    "reconstruction_maps_path": reconstruction.reconstruction_maps_path,
                    **canonical_details,
                    **remap_details,
                }
            else:
                fallback_minutiae, fallback_source = _extract_direct_sample_minutiae(
                    prepared,
                    minutiae_enhanced_path,
                    fingerflow_model_dir,
                    fingerflow_backend,
                )
                fallback_targets, fallback_rasterized_count = _build_targets_and_count_rasterized_minutiae(
                    prepared,
                    fallback_minutiae,
                )
                fallback_reason = (
                    "empty_reprojection"
                    if not reprojected_minutiae
                    else "zero_rasterized_minutiae_after_reprojection"
                )
                minutiae = fallback_minutiae
                minutiae_source = fallback_source
                featurenet_targets = fallback_targets
                rasterized_count = fallback_rasterized_count
                minutiae_ground_truth_details = {
                    "mode": (
                        "direct_fallback_after_empty_reprojection"
                        if not reprojected_minutiae
                        else "direct_fallback_after_zero_rasterized_reprojection"
                    ),
                    "view_role": role,
                    "fallback_reason": fallback_reason,
                    "canonical_source": canonical_source,
                    "canonical_minutiae_count": len(canonical_minutiae),
                    "reprojected_minutiae_count_before_fallback": len(reprojected_minutiae),
                    "rasterized_minutiae_count_before_fallback": reconstruction_rasterized_count,
                    "fallback_source": fallback_source,
                    "fallback_minutiae_count": len(fallback_minutiae),
                    "fallback_rasterized_minutiae_count": fallback_rasterized_count,
                    "center_unwarped_image_path": reconstruction.center_unwarped_image_path,
                    "center_unwarped_mask_path": reconstruction.center_unwarped_mask_path,
                    "center_unwarp_maps_path": reconstruction.center_unwarp_maps_path,
                    "reconstruction_maps_path": reconstruction.reconstruction_maps_path,
                    **canonical_details,
                    **remap_details,
                }
        except Exception as exc:
            fallback_minutiae, fallback_source = _extract_direct_sample_minutiae(
                prepared,
                minutiae_enhanced_path,
                fingerflow_model_dir,
                fingerflow_backend,
            )
            featurenet_targets, rasterized_count = _build_targets_and_count_rasterized_minutiae(
                prepared,
                fallback_minutiae,
            )
            minutiae = fallback_minutiae
            minutiae_source = fallback_source
            minutiae_ground_truth_details = {
                "mode": "direct_fallback_after_reconstruction_failure",
                "view_role": role,
                "fallback_reason": "reconstruction_exception",
                "reason": str(exc),
                "fallback_source": fallback_source,
                "fallback_minutiae_count": len(fallback_minutiae),
                "fallback_rasterized_minutiae_count": rasterized_count,
            }
    else:
        minutiae, minutiae_source = _extract_direct_sample_minutiae(
            prepared,
            minutiae_enhanced_path,
            fingerflow_model_dir,
            fingerflow_backend,
        )
        featurenet_targets, rasterized_count = _build_targets_and_count_rasterized_minutiae(prepared, minutiae)
        minutiae_ground_truth_details = {
            "mode": "direct_per_sample",
            "source": minutiae_source,
            "rasterized_minutiae_count": rasterized_count,
        }

    meta = _build_bundle_meta(
        prepared,
        featurenet_targets,
        minutiae,
        minutiae_source,
        minutiae_ground_truth_details=minutiae_ground_truth_details,
    )
    write_payload = BundleWritePayload(
        bundle_dir=prepared.bundle_dir,
        preprocessed=prepared.preprocessed,
        gray_image=prepared.gray_image,
        mask=prepared.mask,
        orientation=prepared.orientation,
        ridge_period=prepared.ridge_period,
        visualization_gradient=prepared.visualization_gradient,
        masked_image=prepared.masked_image,
        enhanced_image=prepared.enhanced_image,
        minutiae=minutiae,
        featurenet_targets=featurenet_targets,
        meta=meta,
        visualize=prepared.visualize,
    )
    _record_stage_time("postprocess", time.perf_counter() - postprocess_started_at)

    return {
        "bundle_dir": str(prepared.bundle_dir.resolve()),
        "sample_id": prepared.sample.sample_id,
        "raw_view_index": prepared.sample.raw_view_index,
        "view_role": role,
        "minutiae_count": len(minutiae),
        "rasterized_minutiae_count": rasterized_count,
        "minutiae_source": minutiae_source,
        "minutiae_gt_mode": minutiae_ground_truth_details.get("mode") if minutiae_ground_truth_details else None,
        "reconstruction_available": reconstruction is not None,
        "used_reconstruction_backed_final_labels": (
            minutiae_ground_truth_details is not None
            and minutiae_ground_truth_details.get("mode") == "reconstruction_backed"
        ),
        "used_direct_fallback": (
            minutiae_ground_truth_details is not None
            and str(minutiae_ground_truth_details.get("mode", "")).startswith("direct_fallback")
        ),
        "mask_source": prepared.preprocessed.mask_source,
        "pose_rotation_degrees": prepared.preprocessed.pose_rotation_degrees,
        "ridge_scale_factor": prepared.preprocessed.ridge_scale_factor,
    }, write_payload


def _load_featurenet_samples(output_root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for row in manifest:
        bundle_dir = output_root / "samples" / row["sample_id"]
        npz_path = bundle_dir / "featurenet_targets.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            if "gradient" not in data:
                continue
            if "minutia_x_offset" in data and "minutia_y_offset" in data:
                minutia_x_offset = data["minutia_x_offset"]
                minutia_y_offset = data["minutia_y_offset"]
            else:
                minutia_x_offset = ((data["minutia_x"].astype(np.float32) + 0.5) / 8.0)[np.newaxis, ...]
                minutia_y_offset = ((data["minutia_y"].astype(np.float32) + 0.5) / 8.0)[np.newaxis, ...]
            targets = {
                "orientation": data["orientation"],
                "ridge_period": data["ridge_period"],
                "gradient": data["gradient"],
                "minutia_score": data["minutia_score"],
                "minutia_valid_mask": data["minutia_valid_mask"],
                "minutia_x": data["minutia_x"],
                "minutia_y": data["minutia_y"],
                "minutia_x_offset": minutia_x_offset,
                "minutia_y_offset": minutia_y_offset,
                "minutia_orientation": data["minutia_orientation"],
            }
            if "minutia_orientation_vec" in data:
                targets["minutia_orientation_vec"] = data["minutia_orientation_vec"]
        samples.append(
            {
                "masked_image": bundle_dir / "masked_image.png",
                "mask": bundle_dir / "mask.png",
                "targets": targets,
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _load_bundle_targets(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def _iter_patch_source_bundles(source_root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples_root = source_root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"missing sample directory: {samples_root}")

    bundle_dirs = sorted(path for path in samples_root.iterdir() if path.is_dir())
    source_rows: list[dict[str, Any]] = []
    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "meta.json"
        masked_image_path = bundle_dir / "masked_image.png"
        mask_path = bundle_dir / "mask.png"
        targets_path = bundle_dir / "featurenet_targets.npz"
        if not (meta_path.exists() and masked_image_path.exists() and mask_path.exists() and targets_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        targets = _load_bundle_targets(targets_path)
        if meta.get("raw_view_index") not in {0, 1, 2}:
            continue
        if "gradient" not in targets:
            continue
        source_rows.append(
            {
                "bundle_dir": bundle_dir,
                "meta": meta,
                "masked_image_path": masked_image_path,
                "mask_path": mask_path,
                "targets": targets,
            }
        )
        if limit is not None and len(source_rows) >= limit:
            break
    return source_rows


def _compute_patch_mask_ratios(mask: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_size > mask.shape[0] or patch_size > mask.shape[1]:
        return np.zeros((0, 0), dtype=np.float32)
    binary = (mask > 0).astype(np.float32)
    integral = np.pad(binary, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[patch_size:, patch_size:]
        - integral[:-patch_size, patch_size:]
        - integral[patch_size:, :-patch_size]
        + integral[:-patch_size, :-patch_size]
    ) / float(patch_size * patch_size)


def _select_patch_windows(mask: np.ndarray, patch_size: int, minimum_mask_ratio: float) -> list[tuple[int, int, float]]:
    ratios = _compute_patch_mask_ratios(mask, patch_size)
    if ratios.size == 0:
        return []
    valid_y, valid_x = np.where(ratios >= minimum_mask_ratio)
    max_x = max(0, ((mask.shape[1] // 8) - (patch_size // 8)) * 8)
    max_y = max(0, ((mask.shape[0] // 8) - (patch_size // 8)) * 8)
    within_target_bounds = (valid_x <= max_x) & (valid_y <= max_y)
    valid_y = valid_y[within_target_bounds]
    valid_x = valid_x[within_target_bounds]
    alignment = 8
    aligned = (valid_x % alignment == 0) & (valid_y % alignment == 0)
    valid_y = valid_y[aligned]
    valid_x = valid_x[aligned]
    if valid_y.size < 2:
        return []

    centers = np.stack([valid_x + (patch_size / 2.0), valid_y + (patch_size / 2.0)], axis=1).astype(np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        axis_u, _, _, _ = estimate_finger_axis(contour)
    else:
        axis_u = np.array([1.0, 0.0], dtype=np.float32)
    projections = centers @ axis_u
    first_idx = int(np.argmin(projections))
    second_idx = int(np.argmax(projections))
    if first_idx == second_idx:
        return []

    picks = []
    for idx in (first_idx, second_idx):
        x = int(valid_x[idx])
        y = int(valid_y[idx])
        picks.append((x, y, float(ratios[y, x])))
    if picks[0][:2] == picks[1][:2]:
        return []
    return picks


def _crop_target_array(
    array: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    source_shape: tuple[int, int],
) -> np.ndarray:
    spatial_height = array.shape[-2]
    spatial_width = array.shape[-1]
    if spatial_height == source_shape[0] and spatial_width == source_shape[1]:
        crop_x = x
        crop_y = y
        crop_w = patch_size
        crop_h = patch_size
    else:
        crop_x = x // 8
        crop_y = y // 8
        crop_w = patch_size // 8
        crop_h = patch_size // 8
    if array.ndim == 3:
        return array[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()
    if array.ndim == 2:
        return array[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()
    raise ValueError(f"unsupported target array ndim for patch cropping: {array.ndim}")


def _write_patch_dataset(
    source_root: Path,
    output_root: Path,
    limit: int | None,
    config: PatchDatasetConfig,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    source_rows = _iter_patch_source_bundles(source_root, limit=limit)
    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    mask_ratios: list[float] = []

    for row in source_rows:
        bundle_dir = Path(row["bundle_dir"])
        meta = dict(row["meta"])
        masked_image = _require_grayscale(Path(row["masked_image_path"]))
        mask = _require_grayscale(Path(row["mask_path"]))
        targets = row["targets"]
        patch_windows = _select_patch_windows(mask, config.patch_size, config.minimum_mask_ratio)
        if len(patch_windows) != 2:
            skipped.append(
                {
                    "sample_id": meta["sample_id"],
                    "reason": "insufficient_valid_patch_windows",
                    "image_shape": list(mask.shape),
                }
            )
            continue

        for patch_index, (x, y, mask_ratio) in enumerate(patch_windows):
            patch_id = f"{meta['sample_id']}_p{patch_index:02d}"
            patch_dir = samples_root / patch_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            patch_masked_image = masked_image[y:y + config.patch_size, x:x + config.patch_size].copy()
            patch_mask = mask[y:y + config.patch_size, x:x + config.patch_size].copy()
            patch_targets = {
                key: _crop_target_array(value, x, y, config.patch_size, source_shape=mask.shape)
                for key, value in targets.items()
            }
            cv2.imwrite(str(patch_dir / "masked_image.png"), patch_masked_image)
            cv2.imwrite(str(patch_dir / "mask.png"), patch_mask)
            _save_npz(patch_dir / "featurenet_targets.npz", patch_targets)
            patch_meta = {
                **meta,
                "sample_id": patch_id,
                "parent_sample_id": meta["sample_id"],
                "patch_index": patch_index,
                "patch_size": config.patch_size,
                "crop_window": {
                    "x": x,
                    "y": y,
                    "width": config.patch_size,
                    "height": config.patch_size,
                },
                "patch_mask_ratio": mask_ratio,
                "patch_source_root": str(source_root.resolve()),
            }
            (patch_dir / "meta.json").write_text(json.dumps(patch_meta, indent=2), encoding="utf-8")
            manifest_rows.append(
                {
                    "sample_id": patch_id,
                    "patch_id": patch_id,
                    "parent_sample_id": meta["sample_id"],
                    "patch_index": patch_index,
                    "subject_id": meta.get("subject_id"),
                    "finger_id": meta.get("finger_id"),
                    "acquisition_id": meta.get("acquisition_id"),
                    "raw_view_index": meta.get("raw_view_index"),
                    "finger_class_id": meta.get("finger_class_id"),
                    "crop_x": x,
                    "crop_y": y,
                    "patch_size": config.patch_size,
                    "patch_mask_ratio": mask_ratio,
                    "source_bundle_dir": str(bundle_dir.resolve()),
                }
            )
            mask_ratios.append(mask_ratio)

    (output_root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    fieldnames = [
        "sample_id",
        "patch_id",
        "parent_sample_id",
        "patch_index",
        "subject_id",
        "finger_id",
        "acquisition_id",
        "raw_view_index",
        "finger_class_id",
        "crop_x",
        "crop_y",
        "patch_size",
        "patch_mask_ratio",
        "source_bundle_dir",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    skip_reason_counts: dict[str, int] = {}
    for item in skipped:
        skip_reason_counts[item["reason"]] = skip_reason_counts.get(item["reason"], 0) + 1

    summary = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "attempted_source_sample_count": len(source_rows),
        "generated_patch_count": len(manifest_rows),
        "generated_parent_sample_count": len({row["parent_sample_id"] for row in manifest_rows}),
        "skipped_source_sample_count": len(skipped),
        "skip_reason_counts": skip_reason_counts,
        "skipped_samples": skipped,
        "patch_size": config.patch_size,
        "minimum_mask_ratio": config.minimum_mask_ratio,
        "patch_mask_ratio_stats": {
            "min": min(mask_ratios) if mask_ratios else None,
            "mean": (sum(mask_ratios) / len(mask_ratios)) if mask_ratios else None,
            "max": max(mask_ratios) if mask_ratios else None,
        },
    }
    _write_summary(output_root, summary)
    return summary


def _verify_manifest(samples: list[RawViewSample]) -> dict[str, Any]:
    finger_groups: dict[int, list[RawViewSample]] = {}
    for sample in samples:
        finger_groups.setdefault(sample.finger_class_id, []).append(sample)

    total_pairs = len(samples) * (len(samples) - 1) // 2
    positive_pairs = sum(len(group) * (len(group) - 1) // 2 for group in finger_groups.values())
    negative_pairs = total_pairs - positive_pairs
    raw_view_issues = [
        sample.sample_id for sample in samples if len(sample.raw_view_paths) != 6 or not all(Path(path).exists() for path in sample.raw_view_paths)
    ]
    missing_raw_paths = [sample.sample_id for sample in samples if not Path(sample.raw_image_path).exists()]

    return {
        "raw_view_sample_count": len(samples),
        "finger_identity_count": len(finger_groups),
        "extra_acquisition_count": sum(1 for sample in samples if sample.is_extra_acquisition),
        "strict_sire_sample_count": sum(1 for sample in samples if sample.sire_path is not None),
        "positive_pair_count": positive_pairs,
        "negative_pair_count": negative_pairs,
        "raw_view_issues": raw_view_issues,
        "missing_raw_paths": missing_raw_paths,
    }


def _run_smoke_test(output_root: Path, sample_count: int) -> dict[str, Any] | None:
    if sample_count <= 0:
        return None
    from featurenet.models.train import load_bundle_samples, run_smoke_test

    samples = load_bundle_samples(output_root, limit=sample_count)
    if not samples:
        return None
    report = run_smoke_test(samples=samples, epochs=1, batch_size=min(2, len(samples)), lr=1e-3)
    return {
        "device": report["device"],
        "num_samples": report["num_samples"],
        "output_shapes": report["output_shapes"],
        "first_total_loss": report["first_total_loss"],
        "last_total_loss": report["last_total_loss"],
        "loss_delta": report["loss_delta"],
    }


def _summarize_minutiae_generation_results(generation_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate final minutiae supervision quality by view role and mode."""
    summary: dict[str, Any] = {
        "total_samples": 0,
        "zero_rasterized_minutiae_samples": 0,
        "by_view_role": {},
        "by_raw_view_index": {},
        "by_mode": {},
    }

    def ensure_bucket(container: dict[str, Any], key: str) -> dict[str, Any]:
        if key not in container:
            container[key] = {
                "sample_count": 0,
                "zero_rasterized_minutiae_samples": 0,
                "minutiae_count_sum": 0,
                "rasterized_minutiae_count_sum": 0,
                "rasterized_minutiae_count_min": None,
                "rasterized_minutiae_count_max": None,
                "used_reconstruction_backed_final_labels": 0,
                "used_direct_fallback": 0,
            }
        return container[key]

    def update_bucket(bucket: dict[str, Any], result: dict[str, Any], rasterized: int) -> None:
        bucket["sample_count"] += 1
        bucket["minutiae_count_sum"] += int(result.get("minutiae_count", 0) or 0)
        bucket["rasterized_minutiae_count_sum"] += rasterized
        if rasterized == 0:
            bucket["zero_rasterized_minutiae_samples"] += 1
        current_min = bucket["rasterized_minutiae_count_min"]
        current_max = bucket["rasterized_minutiae_count_max"]
        bucket["rasterized_minutiae_count_min"] = rasterized if current_min is None else min(current_min, rasterized)
        bucket["rasterized_minutiae_count_max"] = rasterized if current_max is None else max(current_max, rasterized)
        if result.get("used_reconstruction_backed_final_labels"):
            bucket["used_reconstruction_backed_final_labels"] += 1
        if result.get("used_direct_fallback"):
            bucket["used_direct_fallback"] += 1

    for result in generation_results.values():
        summary["total_samples"] += 1
        rasterized = int(result.get("rasterized_minutiae_count", 0) or 0)
        if rasterized == 0:
            summary["zero_rasterized_minutiae_samples"] += 1

        role = str(result.get("view_role") or "none")
        raw_view = str(result.get("raw_view_index"))
        mode = str(result.get("minutiae_gt_mode") or "unknown")

        update_bucket(ensure_bucket(summary["by_view_role"], role), result, rasterized)
        update_bucket(ensure_bucket(summary["by_raw_view_index"], raw_view), result, rasterized)
        update_bucket(ensure_bucket(summary["by_mode"], mode), result, rasterized)

    for container_name in ("by_view_role", "by_raw_view_index", "by_mode"):
        for bucket in summary[container_name].values():
            count = max(int(bucket["sample_count"]), 1)
            bucket["mean_minutiae_count"] = float(bucket["minutiae_count_sum"] / count)
            bucket["mean_rasterized_minutiae_count"] = float(bucket["rasterized_minutiae_count_sum"] / count)
            bucket["zero_rasterized_minutiae_fraction"] = float(bucket["zero_rasterized_minutiae_samples"] / count)

    total = max(int(summary["total_samples"]), 1)
    summary["zero_rasterized_minutiae_fraction"] = float(summary["zero_rasterized_minutiae_samples"] / total)
    return summary


def _write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _load_manifest_samples(manifest_json_path: Path) -> list[RawViewSample]:
    payload = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    return [RawViewSample(**item) for item in payload]


def _group_samples_by_acquisition(samples: list[RawViewSample]) -> list[list[RawViewSample]]:
    groups: list[list[RawViewSample]] = []
    current_group: list[RawViewSample] = []
    current_key: tuple[int, int, int] | None = None
    for sample in samples:
        acquisition_key = _acquisition_key(sample.subject_id, sample.finger_id, sample.acquisition_id)
        if current_group and acquisition_key != current_key:
            groups.append(current_group)
            current_group = []
        current_group.append(sample)
        current_key = acquisition_key
    if current_group:
        groups.append(current_group)
    return groups


def _build_shard_spans(samples: list[RawViewSample], shard_count: int) -> list[tuple[int, int]]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    groups = _group_samples_by_acquisition(samples)
    spans: list[tuple[int, int]] = []
    group_index = 0
    sample_index = 0
    for shard_index in range(shard_count):
        start_index = sample_index
        if group_index >= len(groups):
            spans.append((start_index, start_index))
            continue

        remaining_shards = shard_count - shard_index
        remaining_samples = len(samples) - sample_index
        target_size = max(1, int(math.ceil(remaining_samples / max(remaining_shards, 1))))
        shard_size = 0

        while group_index < len(groups):
            next_group = groups[group_index]
            next_group_size = len(next_group)
            groups_left_after = len(groups) - group_index - 1
            must_leave_groups = groups_left_after >= remaining_shards - 1
            if shard_size > 0 and shard_size + next_group_size > target_size and must_leave_groups:
                break
            shard_size += next_group_size
            sample_index += next_group_size
            group_index += 1
            if shard_size >= target_size and must_leave_groups:
                break

        spans.append((start_index, sample_index))
    return spans


def _resolve_shard_configuration(
    samples: list[RawViewSample],
    args: argparse.Namespace,
    base_output_root: Path,
) -> tuple[list[RawViewSample], Path, dict[str, Any]]:
    requested_samples = samples[: args.limit] if args.limit is not None else list(samples)
    shard_mode = args.shard_mode
    shard_index = int(args.shard_index)

    if shard_mode == "off":
        shard_meta = {
            "mode": "off",
            "boundary_alignment": "acquisition",
            "base_output_root": str(base_output_root),
            "actual_output_root": str(base_output_root),
            "requested_sample_count": len(requested_samples),
            "shard_count": 1,
            "shard_index": 0,
            "start_index": 0,
            "end_index_exclusive": len(requested_samples),
            "sample_count": len(requested_samples),
            "target_shard_size": None,
        }
        return requested_samples, base_output_root, shard_meta

    if shard_mode == "manual":
        if args.shard_count is None:
            raise ValueError("--shard-count is required when --shard-mode manual")
        shard_count = max(1, int(args.shard_count))
        target_shard_size = None
    else:
        if args.shard_count is not None:
            shard_count = max(1, int(args.shard_count))
            target_shard_size = None
        else:
            target_shard_size = max(1, int(args.target_shard_size or 500))
            shard_count = max(1, int(math.ceil(len(requested_samples) / float(target_shard_size)))) if requested_samples else 1

    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count - 1}]")

    shard_spans = _build_shard_spans(requested_samples, shard_count)
    start_index, end_index = shard_spans[shard_index]
    shard_name = f"shard_{shard_index:03d}"
    actual_output_root = base_output_root / shard_name
    selected_samples = requested_samples[start_index:end_index]
    shard_meta = {
        "mode": shard_mode,
        "boundary_alignment": "acquisition",
        "base_output_root": str(base_output_root),
        "actual_output_root": str(actual_output_root),
        "requested_sample_count": len(requested_samples),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "sample_count": len(selected_samples),
        "target_shard_size": target_shard_size,
    }
    return selected_samples, actual_output_root, shard_meta


def _merge_shard_outputs(merge_shards_root: Path, output_root: Path) -> dict[str, Any]:
    shard_roots = sorted(
        [
            child
            for child in merge_shards_root.iterdir()
            if child.is_dir() and (child / "manifest.json").exists() and (child / "summary.json").exists()
        ],
        key=lambda path: path.name,
    )
    if not shard_roots:
        raise RuntimeError(f"no shard outputs found in {merge_shards_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to merge into non-empty output root {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    merged_samples_dir = output_root / "samples"
    merged_recon_dir = output_root / "reconstructions"
    merged_samples_dir.mkdir(parents=True, exist_ok=True)
    merged_recon_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[RawViewSample] = []
    seen_sample_ids: set[str] = set()
    seen_reconstruction_ids: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []

    for shard_root in shard_roots:
        shard_samples = _load_manifest_samples(shard_root / "manifest.json")
        shard_summary = json.loads((shard_root / "summary.json").read_text(encoding="utf-8"))
        shard_summaries.append(
            {
                "shard_root": str(shard_root.resolve()),
                "sample_count": len(shard_samples),
                "generated_bundle_count": shard_summary.get("generated_bundle_count"),
                "skipped_existing_bundle_count": shard_summary.get("skipped_existing_bundle_count"),
                "error_count": len(shard_summary.get("errors", [])),
                "shard": shard_summary.get("shard"),
            }
        )

        for sample in shard_samples:
            if sample.sample_id in seen_sample_ids:
                raise RuntimeError(f"duplicate sample_id during merge: {sample.sample_id}")
            seen_sample_ids.add(sample.sample_id)
            merged_samples.append(sample)

        shard_samples_dir = shard_root / "samples"
        if shard_samples_dir.exists():
            for sample_dir in sorted([path for path in shard_samples_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
                destination = merged_samples_dir / sample_dir.name
                if destination.exists():
                    raise RuntimeError(f"duplicate sample bundle directory during merge: {sample_dir.name}")
                shutil.copytree(sample_dir, destination)

        shard_recon_dir = shard_root / "reconstructions"
        if shard_recon_dir.exists():
            for reconstruction_dir in sorted([path for path in shard_recon_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
                if reconstruction_dir.name in seen_reconstruction_ids:
                    raise RuntimeError(f"duplicate reconstruction directory during merge: {reconstruction_dir.name}")
                seen_reconstruction_ids.add(reconstruction_dir.name)
                shutil.copytree(reconstruction_dir, merged_recon_dir / reconstruction_dir.name)

    _write_manifest(merged_samples, output_root)
    summary = {
        "merge_source_root": str(merge_shards_root.resolve()),
        "output_root": str(output_root.resolve()),
        "merged_bundle_count": len(merged_samples),
        "merged_reconstruction_count": len(seen_reconstruction_ids),
        "verification": _verify_manifest(merged_samples),
        "shards": shard_summaries,
        "version": 1,
    }
    _write_summary(output_root, summary)
    return summary


def _slugify_merge_label(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", label.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"could not derive a usable merge label from {label!r}")
    return slug


def _parse_merge_generated_roots(values: list[str] | None) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for raw_value in values or []:
        if "=" in raw_value:
            raw_label, raw_path = raw_value.split("=", 1)
            label = raw_label.strip() or Path(raw_path).name
        else:
            raw_path = raw_value
            label = Path(raw_path).name
        source_root = Path(raw_path).expanduser().resolve()
        slug = _slugify_merge_label(label)
        if slug in seen_labels:
            raise ValueError(f"duplicate merge label {slug!r}; use unique labels for each --merge-generated-root")
        seen_labels.add(slug)
        parsed.append((slug, source_root))
    if not parsed:
        raise ValueError("at least one --merge-generated-root must be provided")
    return parsed


def _merged_sample_id(source_label: str, sample: RawViewSample) -> str:
    return f"{source_label}_{sample.sample_id}"


def _merged_reconstruction_id(source_label: str, sample: RawViewSample) -> str:
    return f"{source_label}_{_acquisition_name(sample.subject_id, sample.finger_id, sample.acquisition_id)}"


def _rewrite_merged_reconstruction_meta(
    meta_path: Path,
    *,
    merged_acquisition_id: str,
    merged_subject_id: int,
    source_label: str,
    source_subject_id: int,
) -> None:
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["acquisition_id"] = merged_acquisition_id
    meta["subject_id"] = merged_subject_id
    meta["merge_source"] = {
        "label": source_label,
        "subject_id": source_subject_id,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _rewrite_merged_bundle_meta(
    meta_path: Path,
    *,
    merged_sample_id: str,
    merged_subject_id: int,
    merged_subject_index: int,
    merged_finger_class_id: int,
    merged_reconstruction_id: str | None,
    source_label: str,
    source_root: Path,
    source_sample: RawViewSample,
    output_root: Path,
) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sample_id"] = merged_sample_id
    meta["subject_id"] = merged_subject_id
    meta["subject_index"] = merged_subject_index
    meta["finger_class_id"] = merged_finger_class_id
    meta["merge_source"] = {
        "label": source_label,
        "root": str(source_root.resolve()),
        "sample_id": source_sample.sample_id,
        "subject_id": source_sample.subject_id,
        "subject_index": source_sample.subject_index,
        "finger_class_id": source_sample.finger_class_id,
    }

    if merged_reconstruction_id is not None and "multiview_reconstruction" in meta:
        reconstruction = dict(meta["multiview_reconstruction"])
        reconstruction_root = output_root / "reconstructions" / merged_reconstruction_id
        reconstruction["acquisition_id"] = merged_reconstruction_id
        reconstruction["reconstruction_dir"] = str(reconstruction_root.resolve())
        reconstruction["depth_front_path"] = str((reconstruction_root / "depth_front.npy").resolve())
        reconstruction["depth_left_path"] = str((reconstruction_root / "depth_left.npy").resolve())
        reconstruction["depth_right_path"] = str((reconstruction_root / "depth_right.npy").resolve())
        reconstruction["depth_gradient_labels_path"] = str((reconstruction_root / "depth_gradient_labels.npz").resolve())
        reconstruction["reconstruction_maps_path"] = str((reconstruction_root / "reconstruction_maps.npz").resolve())
        reconstruction["support_mask_path"] = str((reconstruction_root / "support_mask.png").resolve())
        reconstruction["preview_path"] = str((reconstruction_root / "preview.png").resolve())
        reconstruction["center_unwarp_maps_path"] = str((reconstruction_root / "center_unwarp_maps.npz").resolve())
        reconstruction["center_unwarped_image_path"] = str((reconstruction_root / "center_unwarped.png").resolve())
        reconstruction["center_unwarped_mask_path"] = str((reconstruction_root / "center_unwarped_mask.png").resolve())
        reconstruction["surface_front_3d_html_path"] = str((reconstruction_root / "surface_front_3d.html").resolve())
        reconstruction["surface_front_3d_png_path"] = str((reconstruction_root / "surface_front_3d.png").resolve())
        reconstruction["surface_all_branches_3d_html_path"] = str((reconstruction_root / "surface_all_branches_3d.html").resolve())
        reconstruction["surface_all_branches_3d_png_path"] = str((reconstruction_root / "surface_all_branches_3d.png").resolve())
        reconstruction["reprojection_report_path"] = str((reconstruction_root / "reprojection_report.json").resolve())
        reconstruction["reprojection_preview_path"] = str((reconstruction_root / "reprojection_preview.png").resolve())
        meta["multiview_reconstruction"] = reconstruction
        if "minutiae_ground_truth" in meta and isinstance(meta["minutiae_ground_truth"], dict):
            minutiae_gt = dict(meta["minutiae_ground_truth"])
            for key, filename in (
                ("center_unwarped_image_path", "center_unwarped.png"),
                ("center_unwarped_mask_path", "center_unwarped_mask.png"),
                ("center_unwarp_maps_path", "center_unwarp_maps.npz"),
                ("reconstruction_maps_path", "reconstruction_maps.npz"),
                ("canonical_unwarped_minutiae_path", "canonical_unwarped_minutiae.json"),
                ("center_unwarped_enhanced_path", "center_unwarped_enhanced.png"),
            ):
                if key in minutiae_gt:
                    minutiae_gt[key] = str((reconstruction_root / filename).resolve())
            meta["minutiae_ground_truth"] = minutiae_gt

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _merge_generated_ground_truth_roots(merge_sources: list[tuple[str, Path]], output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to merge into non-empty output root {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    merged_samples_dir = output_root / "samples"
    merged_recon_dir = output_root / "reconstructions"
    merged_samples_dir.mkdir(parents=True, exist_ok=True)
    merged_recon_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[RawViewSample] = []
    merged_source_summaries: list[dict[str, Any]] = []
    subject_map: dict[tuple[str, int], int] = {}
    copied_reconstruction_ids: set[str] = set()

    for source_label, source_root in merge_sources:
        manifest_path = source_root / "manifest.json"
        summary_path = source_root / "summary.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest.json under {source_root}")
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary.json under {source_root}")

        source_samples = _load_manifest_samples(manifest_path)
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_subject_ids: set[int] = set()
        copied_bundle_count = 0
        missing_bundle_count = 0
        copied_reconstruction_count = 0

        for sample in source_samples:
            source_subject_ids.add(sample.subject_id)
            subject_key = (source_label, sample.subject_id)
            if subject_key not in subject_map:
                subject_map[subject_key] = len(subject_map)
            merged_subject_index = subject_map[subject_key]
            merged_subject_id = merged_subject_index + 1
            merged_finger_class_id = merged_subject_index * 10 + (sample.finger_id - 1)
            merged_sample_id = _merged_sample_id(source_label, sample)
            merged_reconstruction_id = _merged_reconstruction_id(source_label, sample)

            merged_samples.append(
                RawViewSample(
                    sample_id=merged_sample_id,
                    subject_id=merged_subject_id,
                    subject_index=merged_subject_index,
                    finger_id=sample.finger_id,
                    acquisition_id=sample.acquisition_id,
                    finger_class_id=merged_finger_class_id,
                    raw_image_path=sample.raw_image_path,
                    raw_view_index=sample.raw_view_index,
                    sire_path=sample.sire_path,
                    raw_view_paths=list(sample.raw_view_paths),
                    variant_paths=dict(sample.variant_paths),
                    is_extra_acquisition=sample.is_extra_acquisition,
                )
            )

            source_sample_dir = source_root / "samples" / sample.sample_id
            if not source_sample_dir.exists():
                missing_bundle_count += 1
                continue

            destination_sample_dir = merged_samples_dir / merged_sample_id
            if destination_sample_dir.exists():
                raise RuntimeError(f"duplicate merged sample bundle directory: {destination_sample_dir.name}")
            shutil.copytree(source_sample_dir, destination_sample_dir)
            copied_bundle_count += 1

            updated_meta = _rewrite_merged_bundle_meta(
                destination_sample_dir / "meta.json",
                merged_sample_id=merged_sample_id,
                merged_subject_id=merged_subject_id,
                merged_subject_index=merged_subject_index,
                merged_finger_class_id=merged_finger_class_id,
                merged_reconstruction_id=merged_reconstruction_id,
                source_label=source_label,
                source_root=source_root,
                source_sample=sample,
                output_root=output_root,
            )

            if updated_meta is None or "multiview_reconstruction" not in updated_meta:
                continue

            if merged_reconstruction_id in copied_reconstruction_ids:
                continue

            source_reconstruction_dir = source_root / "reconstructions" / _acquisition_name(
                sample.subject_id,
                sample.finger_id,
                sample.acquisition_id,
            )
            if not source_reconstruction_dir.exists():
                continue

            destination_reconstruction_dir = merged_recon_dir / merged_reconstruction_id
            shutil.copytree(source_reconstruction_dir, destination_reconstruction_dir)
            _rewrite_merged_reconstruction_meta(
                destination_reconstruction_dir / "meta.json",
                merged_acquisition_id=merged_reconstruction_id,
                merged_subject_id=merged_subject_id,
                source_label=source_label,
                source_subject_id=sample.subject_id,
            )
            copied_reconstruction_ids.add(merged_reconstruction_id)
            copied_reconstruction_count += 1

        merged_source_summaries.append(
            {
                "label": source_label,
                "source_root": str(source_root.resolve()),
                "manifest_sample_count": len(source_samples),
                "source_subject_count": len(source_subject_ids),
                "copied_bundle_count": copied_bundle_count,
                "missing_bundle_count": missing_bundle_count,
                "copied_reconstruction_count": copied_reconstruction_count,
                "source_summary": {
                    "generated_bundle_count": source_summary.get("generated_bundle_count"),
                    "skipped_existing_bundle_count": source_summary.get("skipped_existing_bundle_count"),
                    "error_count": len(source_summary.get("errors", [])),
                    "dataset_root": source_summary.get("dataset_root"),
                    "output_root": source_summary.get("output_root"),
                },
            }
        )

    _write_manifest(merged_samples, output_root)
    summary = {
        "merge_strategy": "merge_generated_roots_with_global_identity_remap",
        "output_root": str(output_root.resolve()),
        "merged_bundle_count": sum(item["copied_bundle_count"] for item in merged_source_summaries),
        "merged_reconstruction_count": len(copied_reconstruction_ids),
        "merged_subject_count": len(subject_map),
        "verification": _verify_manifest(merged_samples),
        "sources": merged_source_summaries,
        "version": 2,
    }
    _write_summary(output_root, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style approximate ground truth bundles for DS1.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--merge-shards-root", type=Path, default=None, help="Merge shard output roots from this directory into --output-root and exit.")
    parser.add_argument(
        "--merge-generated-root",
        action="append",
        default=None,
        help="Merge separately generated ground-truth roots into --output-root. Pass LABEL=PATH, or PATH to infer LABEL from the folder name.",
    )
    parser.add_argument("--execution-target", choices=("local", "kaggle"), default="local")
    parser.add_argument("--gpu-only", action="store_true", help="Require GPU-backed providers for model-backed stages.")
    parser.add_argument("--gpu-batch-size", type=int, default=1, help="Reserved GPU batch size knob for Kaggle runs.")
    parser.add_argument("--cpu-workers", type=int, default=max(1, min((os.cpu_count() or 1), 4)))
    parser.add_argument("--prefetch-samples", type=int, default=2, help="Number of samples to preload ahead of GPU-backed work.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose bundle files already exist.")
    parser.add_argument("--shard-mode", choices=("off", "auto", "manual"), default="off")
    parser.add_argument("--shard-count", type=int, default=None, help="Total number of shards to define when shard mode is active.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to run when shard mode is active.")
    parser.add_argument("--target-shard-size", type=int, default=None, help="Approximate number of samples per shard in auto mode (default: 500).")
    parser.add_argument("--patch-source-root", type=Path, default=None, help="Read an existing full-image ground-truth root and write an offline patch dataset.")
    parser.add_argument("--patch-output-root", type=Path, default=None, help="Output root for offline patch dataset generation.")
    parser.add_argument("--patch-limit-samples", type=int, default=None, help="Only consider the first N eligible source samples when building a patch dataset.")
    parser.add_argument("--patch-size", type=int, default=512, help="Square patch size for offline patch dataset generation.")
    parser.add_argument("--patch-min-mask-ratio", type=float, default=0.30, help="Minimum foreground mask ratio required for a valid training patch.")
    parser.add_argument("--patch-smoke-samples", type=int, default=0, help="Run featurenet smoke test on the first N generated patches.")
    parser.add_argument("--fingerflow-model-dir", type=Path, default=DEFAULT_FINGERFLOW_MODEL_DIR)
    parser.add_argument("--fingerflow-backend", choices=("auto", "local", "wsl"), default="auto")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--wsl-activate", default="source ~/.venvs/contactless-biometric/bin/activate")
    parser.add_argument("--limit", type=int, default=None, help="Only generate bundle files for the first N manifest rows.")
    parser.add_argument("--visualize-count", type=int, default=0, help="Generate preview overlays for the first N generated bundles.")
    parser.add_argument("--smoke-samples", type=int, default=0, help="Run featurenet smoke test on the first N generated bundles.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_shards_root is not None and args.merge_generated_root:
        raise ValueError("use either --merge-shards-root or --merge-generated-root, not both")
    if args.merge_generated_root:
        merged_summary = _merge_generated_ground_truth_roots(
            _parse_merge_generated_roots(args.merge_generated_root),
            args.output_root.resolve(),
        )
        print(json.dumps(merged_summary, indent=2))
        return 0
    if args.merge_shards_root is not None:
        merged_summary = _merge_shard_outputs(args.merge_shards_root.resolve(), args.output_root.resolve())
        print(json.dumps(merged_summary, indent=2))
        return 0

    runtime_config = GeneratorRuntimeConfig(
        execution_target=args.execution_target,
        gpu_only=bool(args.gpu_only),
        gpu_batch_size=max(1, int(args.gpu_batch_size)),
        cpu_workers=max(1, int(args.cpu_workers)),
        prefetch_samples=max(1, int(args.prefetch_samples)),
        skip_existing=bool(args.skip_existing),
    )
    patch_config = PatchDatasetConfig(
        patch_size=args.patch_size,
        minimum_mask_ratio=args.patch_min_mask_ratio,
    )

    if args.patch_source_root is not None:
        patch_source_root = args.patch_source_root.resolve()
        patch_output_root = (args.patch_output_root or (patch_source_root.parent / f"{patch_source_root.name}_patches")).resolve()
        patch_summary = _write_patch_dataset(
            source_root=patch_source_root,
            output_root=patch_output_root,
            limit=args.patch_limit_samples,
            config=patch_config,
        )
        patch_summary["smoke_test"] = _run_smoke_test(patch_output_root, args.patch_smoke_samples)
        _write_summary(patch_output_root, patch_summary)
        print(json.dumps(patch_summary, indent=2))
        return 0

    dataset_root = args.dataset_root.resolve()
    base_output_root = args.output_root.resolve()
    runtime_summary = _configure_runtime_environment(runtime_config)
    fingerflow_backend = FingerflowBackendConfig(
        backend=_resolve_fingerflow_backend(args.fingerflow_backend),
        wsl_distro=args.wsl_distro,
        wsl_activate=args.wsl_activate,
    )

    all_samples = _build_manifest(dataset_root)
    samples, output_root, shard_meta = _resolve_shard_configuration(all_samples, args, base_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(samples, output_root)
    print(
        json.dumps(
            {
                "shard": shard_meta,
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
            },
            indent=2,
        ),
        flush=True,
    )

    verification = _verify_manifest(samples)
    reconstruction_candidates = _collect_reconstruction_candidates(samples)
    total_requested = len(samples)
    generation_results: dict[str, Any] = {}
    generated = 0
    skipped_existing = 0
    errors: list[dict[str, str]] = []
    reconstruction_results: dict[tuple[int, int, int], AcquisitionReconstructionResult] = {}
    reconstruction_errors: list[dict[str, str]] = []
    load_executor = ThreadPoolExecutor(max_workers=runtime_config.cpu_workers, thread_name_prefix="gt-load")
    write_executor = ThreadPoolExecutor(max_workers=max(1, min(runtime_config.cpu_workers, 2)), thread_name_prefix="gt-write")
    pending_inputs: deque[tuple[int, RawViewSample, Future[LoadedSampleInput]]] = deque()
    pending_writes: list[Future[None]] = []

    def emit_progress(status: str, sample_id: str | None = None) -> None:
        finalized = generated + skipped_existing + len(errors)
        left = max(total_requested - finalized, 0)
        sample_suffix = f" sample={sample_id}" if sample_id else ""
        print(
            f"[progress] status={status}{sample_suffix} completed={finalized}/{total_requested} "
            f"left={left} generated={generated} skipped={skipped_existing} errors={len(errors)}",
            flush=True,
        )

    def submit_sample(index: int, sample: RawViewSample) -> None:
        nonlocal skipped_existing
        bundle_dir = output_root / "samples" / sample.sample_id
        if runtime_config.skip_existing and _bundle_outputs_exist(bundle_dir):
            skipped_existing += 1
            emit_progress("skipped_existing", sample.sample_id)
            return

        reconstruction: AcquisitionReconstructionResult | None = None
        acquisition_key = _acquisition_key(sample.subject_id, sample.finger_id, sample.acquisition_id)
        if acquisition_key in reconstruction_candidates:
            if acquisition_key not in reconstruction_results and not any(
                item["acquisition_id"] == _acquisition_name(*acquisition_key) for item in reconstruction_errors
            ):
                try:
                    reconstruction_results[acquisition_key] = _reconstruct_multiview_acquisition(sample, output_root)
                except Exception as exc:
                    reconstruction_errors.append(
                        {
                            "acquisition_id": _acquisition_name(*acquisition_key),
                            "sample_id": sample.sample_id,
                            "error": str(exc),
                        }
                    )
            reconstruction = reconstruction_results.get(acquisition_key)
        future = load_executor.submit(
            _load_sample_input,
            sample,
            output_root,
            index < args.visualize_count,
            reconstruction,
        )
        pending_inputs.append((index, sample, future))

    try:
        scheduled = 0
        print(f"[progress] starting total={total_requested}", flush=True)
        while scheduled < len(samples) and (args.limit is None or scheduled < args.limit) and len(pending_inputs) < runtime_config.prefetch_samples:
            submit_sample(scheduled, samples[scheduled])
            scheduled += 1

        while pending_inputs:
            index, sample, future = pending_inputs.popleft()
            try:
                loaded = future.result()
                generation_results[sample.sample_id], write_payload = _generate_bundle_from_loaded(
                    loaded,
                    fingerflow_model_dir=args.fingerflow_model_dir.resolve(),
                    dpi=args.dpi,
                    fingerflow_backend=fingerflow_backend,
                )
                pending_writes.append(write_executor.submit(_persist_bundle, write_payload))
                generated += 1
                emit_progress("generated", sample.sample_id)
            except Exception as exc:
                errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                emit_progress("error", sample.sample_id)

            while scheduled < len(samples) and (args.limit is None or scheduled < args.limit) and len(pending_inputs) < runtime_config.prefetch_samples:
                submit_sample(scheduled, samples[scheduled])
                scheduled += 1

        for future in pending_writes:
            future.result()
    finally:
        load_executor.shutdown(wait=True)
        write_executor.shutdown(wait=True)

    minutiae_audit = _summarize_minutiae_generation_results(generation_results)
    (output_root / "minutiae_audit.json").write_text(json.dumps(minutiae_audit, indent=2), encoding="utf-8")
    smoke = _run_smoke_test(output_root, args.smoke_samples)
    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "base_output_root": str(base_output_root),
        "shard": shard_meta,
        "verification": verification,
        "generated_bundle_count": generated,
        "skipped_existing_bundle_count": skipped_existing,
        "requested_limit": args.limit,
        "visualize_count": args.visualize_count,
        "runtime": runtime_summary,
        "selected_rembg_providers": _SELECTED_REMBG_PROVIDERS,
        "selected_fingerflow_device": _SELECTED_FINGERFLOW_DEVICE,
        "pipeline_stage_seconds": {key: round(value, 3) for key, value in _PIPELINE_STAGE_TOTALS.items()},
        "fingerflow_backend": asdict(fingerflow_backend),
        "minutiae_audit": minutiae_audit,
        "reconstruction": {
            "eligible_acquisition_count": len(reconstruction_candidates),
            "generated_acquisition_count": len(reconstruction_results),
            "errors": reconstruction_errors,
        },
        "smoke_test": smoke,
        "errors": errors,
    }
    _write_summary(output_root, summary)

    if args.patch_output_root is not None:
        patch_output_root = args.patch_output_root.resolve()
        if shard_meta["mode"] != "off":
            patch_output_root = patch_output_root / f"shard_{int(shard_meta['shard_index']):03d}"
        patch_summary = _write_patch_dataset(
            source_root=output_root,
            output_root=patch_output_root,
            limit=args.patch_limit_samples,
            config=patch_config,
        )
        patch_summary["smoke_test"] = _run_smoke_test(patch_output_root, args.patch_smoke_samples)
        _write_summary(patch_output_root, patch_summary)
        summary["patch_dataset"] = {
            "output_root": str(patch_output_root),
            "generated_patch_count": patch_summary["generated_patch_count"],
            "generated_parent_sample_count": patch_summary["generated_parent_sample_count"],
            "skipped_source_sample_count": patch_summary["skipped_source_sample_count"],
        }
        _write_summary(output_root, summary)

    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
