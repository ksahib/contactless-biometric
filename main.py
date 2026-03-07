from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
import cv2
import numpy as np

try:
    from rembg import new_session, remove
    _HAS_REMBG = True
except Exception:
    new_session = None
    remove = None
    _HAS_REMBG = False


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_NOBG_DIR = REPO_ROOT / "output_nobg"
OUTPUT_CROPPED_DIR = REPO_ROOT / "output_cropped"
REMBG_MODEL = os.environ.get("FINGER_REMBG_MODEL", "u2netp")
DEFAULT_ENHANCED_IMAGE = REPO_ROOT / "filtered_cropped.png"
DEFAULT_MINUTIAE_JSON = REPO_ROOT / "fingerflow_minutiae.json"
DEFAULT_MINUTIAE_CSV = REPO_ROOT / "fingerflow_minutiae.csv"
DEFAULT_CORE_CSV = REPO_ROOT / "fingerflow_core.csv"
DEFAULT_FINGERFLOW_MODEL_DIR = REPO_ROOT / ".fingerflow_models"

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


def _install_numpy_lib_pad_compat() -> None:
    try:
        import numpy as np
    except Exception:
        return

    if not hasattr(np, "pad"):
        return

    lib = getattr(np, "lib", None)
    if lib is None:
        return

    if not hasattr(lib, "pad"):
        setattr(lib, "pad", getattr(np, "pad"))


def _install_numpy_scalar_aliases_compat() -> None:
    try:
        import numpy as np
    except Exception:
        return

    if not hasattr(np, "int"):
        setattr(np, "int", int)


def _install_skimage_gaussian_compat() -> None:
    try:
        import skimage.filters as sk_filters
    except Exception:
        return

    if getattr(sk_filters.gaussian, "_legacy_multichannel_compat", False):
        return

    orig_gaussian = sk_filters.gaussian

    def _gaussian(image, *args, **kwargs):
        if "multichannel" in kwargs:
            multichannel = kwargs.pop("multichannel")
            kwargs.setdefault("channel_axis", -1 if bool(multichannel) else None)
        return orig_gaussian(image, *args, **kwargs)

    sk_filters.gaussian = _gaussian  # type: ignore[assignment]
    sk_filters.gaussian._legacy_multichannel_compat = True  # type: ignore[attr-defined]


def _install_scipy_signal_gaussian_compat() -> None:
    try:
        from scipy import signal
        from scipy.signal import windows
    except Exception:
        return

    if hasattr(signal, "gaussian"):
        return

    signal.gaussian = windows.gaussian  # type: ignore[attr-defined]


def _disable_model_compile_on_load() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    load_model = tf.keras.models.load_model
    if getattr(load_model, "_legacy_compile_compat", False):
        return

    def _load_model(*args, **kwargs):
        kwargs.setdefault("compile", False)
        return load_model(*args, **kwargs)

    tf.keras.models.load_model = _load_model  # type: ignore[assignment]
    tf.keras.models.load_model._legacy_compile_compat = True  # type: ignore[attr-defined]


def _force_cpu_mode() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _normalize_optimizer_learning_rate_compat() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    Optimizer = tf.keras.optimizers.Optimizer
    if getattr(Optimizer, "_legacy_learning_rate_compat", False):
        return

    orig_from_config = Optimizer.from_config

    def _to_float(value):
        if isinstance(value, int):
            return float(value)
        if isinstance(value, dict):
            return {k: _to_float(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [_to_float(v) for v in value]
            return type(value)(converted)
        return value

    def _from_config(cls, config, custom_objects=None):
        cfg = dict(config)
        for key in ("learning_rate", "lr"):
            if key in cfg:
                cfg[key] = _to_float(cfg[key])
        try:
            return orig_from_config(cls, cfg, custom_objects=custom_objects)
        except TypeError:
            return orig_from_config(cls, cfg)

    Optimizer.from_config = classmethod(_from_config)  # type: ignore[assignment]
    Optimizer._legacy_learning_rate_compat = True


def _normalize_optimizer_constructor_args_compat() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    Adam = getattr(tf.keras.optimizers, "Adam", None)
    if Adam is None:
        return

    if getattr(Adam, "_legacy_lr_ctor_compat", False):
        return

    orig_init = Adam.__init__

    def _coerce_learning_rate(value):
        if isinstance(value, int):
            return float(value)
        return value

    def _init(self, *args, **kwargs):
        if args:
            coerced_first = _coerce_learning_rate(args[0])
            args = (coerced_first, *args[1:])

        if "lr" in kwargs and "learning_rate" not in kwargs:
            kwargs["learning_rate"] = _coerce_learning_rate(kwargs.pop("lr"))
        if "learning_rate" in kwargs:
            kwargs["learning_rate"] = _coerce_learning_rate(kwargs["learning_rate"])

        return orig_init(self, *args, **kwargs)

    Adam.__init__ = _init  # type: ignore[assignment]
    Adam._legacy_lr_ctor_compat = True  # type: ignore[attr-defined]


def _install_conv2d_legacy_weights_compat() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    Conv2D = getattr(tf.keras.layers, "Conv2D", None)
    if Conv2D is None:
        return

    if getattr(Conv2D, "_legacy_weights_compat", False):
        return

    orig_init = Conv2D.__init__
    orig_build = Conv2D.build

    def _init(self, *args, weights=None, **kwargs):
        self._legacy_initial_weights = weights
        return orig_init(self, *args, **kwargs)

    def _build(self, input_shape):
        result = orig_build(self, input_shape)
        weights = getattr(self, "_legacy_initial_weights", None)
        if weights is not None:
            self.set_weights(weights)
            self._legacy_initial_weights = None
        return result

    Conv2D.__init__ = _init  # type: ignore[assignment]
    Conv2D.build = _build  # type: ignore[assignment]
    Conv2D._legacy_weights_compat = True


def _install_lambda_output_shape_compat() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    Lambda = getattr(tf.keras.layers, "Lambda", None)
    if Lambda is None:
        return

    if getattr(Lambda, "_legacy_output_shape_compat", False):
        return

    orig_compute_output_shape = Lambda.compute_output_shape

    def _compute_output_shape(self, input_shape):
        configured_output = getattr(self, "output_shape", None) or getattr(
            self, "_output_shape", None
        )
        if configured_output is None:
            try:
                return orig_compute_output_shape(self, input_shape)
            except Exception:
                return input_shape
        return orig_compute_output_shape(self, input_shape)

    Lambda.compute_output_shape = _compute_output_shape  # type: ignore[assignment]
    Lambda._legacy_output_shape_compat = True


def _install_keras_tensor_math_compat() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        from tensorflow.keras import ops as keras_ops
    except Exception:
        return

    if getattr(tf.math.tanh, "_keras_tensor_compat", False):
        return

    def _is_keras_tensor(value):
        return type(value).__name__ == "KerasTensor"

    orig_tanh = tf.math.tanh
    orig_softplus = tf.math.softplus

    def _tanh(value, *args, **kwargs):
        if _is_keras_tensor(value):
            return keras_ops.tanh(value)
        try:
            return orig_tanh(value, *args, **kwargs)
        except TypeError:
            return keras_ops.tanh(value)

    def _softplus(value, *args, **kwargs):
        if _is_keras_tensor(value):
            return keras_ops.softplus(value)
        try:
            return orig_softplus(value, *args, **kwargs)
        except TypeError:
            return keras_ops.softplus(value)

    tf.math.tanh = _tanh  # type: ignore[assignment]
    tf.math.softplus = _softplus  # type: ignore[assignment]
    tf.math.tanh._keras_tensor_compat = True  # type: ignore[attr-defined]
    tf.math.softplus._keras_tensor_compat = True  # type: ignore[attr-defined]


_install_conv2d_legacy_weights_compat()
_install_lambda_output_shape_compat()
_force_cpu_mode()
_disable_model_compile_on_load()
_normalize_optimizer_learning_rate_compat()
_normalize_optimizer_constructor_args_compat()
_install_keras_tensor_math_compat()
_install_numpy_lib_pad_compat()
_install_numpy_scalar_aliases_compat()
_install_skimage_gaussian_compat()
_install_scipy_signal_gaussian_compat()

from fingerflow.extractor import Extractor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a finger to the distal phalanx and remove its background."
    )
    parser.add_argument("input_image", type=Path, help="Path to the source finger image")
    parser.add_argument(
        "--fingerflow-model-dir",
        type=Path,
        default=DEFAULT_FINGERFLOW_MODEL_DIR,
        help="Directory to cache FingerFlow model files.",
    )
    parser.add_argument(
        "--minutiae-json",
        type=Path,
        default=DEFAULT_MINUTIAE_JSON,
        help="Output JSON path for extracted minutiae/core results.",
    )
    parser.add_argument(
        "--minutiae-csv",
        type=Path,
        default=DEFAULT_MINUTIAE_CSV,
        help="Output CSV path for extracted minutiae rows.",
    )
    parser.add_argument(
        "--core-csv",
        type=Path,
        default=DEFAULT_CORE_CSV,
        help="Output CSV path for detected core rows.",
    )
    return parser.parse_args()


def get_rembg_session():
    global _REMBG_SESSION
    if not _HAS_REMBG:
        raise RuntimeError(
            "rembg is not installed. Install rembg for background masking or"
            " use the fallback path in this script."
        )
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session(REMBG_MODEL)
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
        total_pixels = mask.size
        largest_area = max(cv2.contourArea(c) for c in contours)
        return float(largest_area) / float(total_pixels)

    ratio_bright = _largest_ratio(threshold_bright)
    ratio_dark = _largest_ratio(threshold_dark)
    best_mask = threshold_bright if ratio_bright >= ratio_dark else threshold_dark

    return largest_component_mask(best_mask)


def load_bgr_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"input image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def load_gray_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"input image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def load_image_unchanged(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"input image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
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


def build_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        OUTPUT_NOBG_DIR / f"{stem}_nobg.png",
        OUTPUT_CROPPED_DIR / f"{stem}_cropped.png",
    )


def rembg_mask_from_bgr(bgr_image: np.ndarray) -> np.ndarray:
    if not _HAS_REMBG:
        return _fallback_mask_from_bgr(bgr_image)

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

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1
    )
    return largest_component_mask(mask)


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


def estimate_finger_axis(
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    edge_a = box[1] - box[0]
    edge_b = box[2] - box[1]
    len_a = float(np.linalg.norm(edge_a))
    len_b = float(np.linalg.norm(edge_b))

    if len_a >= len_b:
        long_edge = edge_a
        short_edge = edge_b
        length_l = len_a
        width_w = len_b
    else:
        long_edge = edge_b
        short_edge = edge_a
        length_l = len_b
        width_w = len_a

    if length_l <= 0 or width_w <= 0:
        raise RuntimeError("failed to estimate finger orientation")

    axis_u = long_edge / length_l
    axis_v = short_edge / width_w
    return axis_u.astype(np.float32), axis_v.astype(np.float32), length_l, width_w


def select_fingertip_end(
    contour: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray, length_l: float
) -> tuple[np.ndarray, np.ndarray]:
    points = contour.reshape(-1, 2).astype(np.float32)
    proj_u = points @ axis_u
    proj_v = points @ axis_v
    min_u = float(proj_u.min())
    max_u = float(proj_u.max())
    band = max(2.0, 0.14 * length_l)

    min_selector = proj_u <= (min_u + band)
    max_selector = proj_u >= (max_u - band)

    def stats(selector: np.ndarray) -> tuple[float, float, float]:
        chosen = points[selector]
        if chosen.shape[0] < 6:
            raise RuntimeError("insufficient contour support to localize fingertip")
        local_v = proj_v[selector]
        return (
            float(local_v.max() - local_v.min()),
            float(np.mean(local_v)),
            float(np.mean(chosen[:, 1])),
        )

    min_width, min_center_v, min_mean_y = stats(min_selector)
    max_width, max_center_v, max_mean_y = stats(max_selector)

    width_delta = abs(min_width - max_width) / max(min_width, max_width, 1.0)
    if width_delta >= 0.08:
        choose_min = min_width < max_width
    else:
        choose_min = min_mean_y <= max_mean_y

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


def compute_distal_crop_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contour available for crop")
    contour = max(contours, key=cv2.contourArea)

    axis_u, axis_v, length_l, _ = estimate_finger_axis(contour)
    tip_center, tip_to_base = select_fingertip_end(contour, axis_u, axis_v, length_l)

    width_near_tip = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.20)
    width_stable = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.34)
    distal_width = max(width_near_tip, 0.92 * width_stable)

    inward_extension = min(0.66 * length_l, max(2.35 * distal_width, 0.50 * length_l))
    outward_extension = 0.10 * length_l
    half_width = 0.82 * distal_width
    margin_px = max(20, int(round(0.10 * max(length_l, distal_width))))

    outer_center = tip_center - (tip_to_base * outward_extension)
    inner_center = tip_center + (tip_to_base * inward_extension)

    crop_quad = np.array(
        [
            outer_center - (axis_v * half_width),
            outer_center + (axis_v * half_width),
            inner_center + (axis_v * half_width),
            inner_center - (axis_v * half_width),
        ],
        dtype=np.float32,
    )

    x_min = int(np.floor(np.min(crop_quad[:, 0]))) - margin_px
    y_min = int(np.floor(np.min(crop_quad[:, 1]))) - margin_px
    x_max = int(np.ceil(np.max(crop_quad[:, 0]))) + margin_px
    y_max = int(np.ceil(np.max(crop_quad[:, 1]))) + margin_px

    height, width = mask.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    if (x_max - x_min) < 32 or (y_max - y_min) < 32:
        raise RuntimeError("computed distal phalanx crop is too small")

    return x_min, y_min, x_max, y_max


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max].copy()


def compose_rgba(bgr_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    return bgra


def save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save image: {path}")


def process_image(input_path: Path) -> tuple[Path, Path]:
    full_bgr = load_bgr_image(input_path)

    coarse_mask = rembg_mask_from_bgr(full_bgr)
    validate_foreground_area(coarse_mask, minimum_ratio=0.03)

    crop_bbox = compute_distal_crop_bbox(coarse_mask)
    cropped_bgr = crop_image(full_bgr, crop_bbox)

    cropped_mask = rembg_mask_from_bgr(cropped_bgr)
    validate_foreground_area(cropped_mask, minimum_ratio=0.08)
    cropped_rgba = compose_rgba(cropped_bgr, cropped_mask)

    nobg_path, cropped_path = build_output_paths(input_path)
    OUTPUT_NOBG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    save_png(cropped_path, cropped_bgr)
    save_png(nobg_path, cropped_rgba)
    return nobg_path, cropped_path

def normalise_brightness(input_path: Path, output_path: Path) -> None:
    image = load_image_unchanged(input_path)
    if image.ndim == 2:
        img_grey = image
        alpha = None
    elif image.ndim == 3 and image.shape[2] == 4:
        img_grey = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = image[:, :, 3]
    elif image.ndim == 3 and image.shape[2] == 3:
        img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None
    else:
        raise ValueError(f"unsupported image shape for brightness normalization: {image.shape}")

    if alpha is not None:
        foreground = alpha > 0
        ys, xs = np.where(foreground)
        if ys.size == 0:
            raise RuntimeError("foreground mask is empty")

        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        cropped_grey = img_grey[y_min:y_max, x_min:x_max]

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        cropped_clahe = clahe.apply(cropped_grey)
        clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
        cropped_clahe = clahe2.apply(cropped_clahe)

        masked_grey = np.zeros_like(img_grey)
        masked_grey[y_min:y_max, x_min:x_max] = cropped_clahe
        masked_grey[~foreground] = 0

        output = cv2.cvtColor(masked_grey, cv2.COLOR_GRAY2BGRA)
        output[:, :, 3] = alpha
        cv2.imwrite(str(output_path), output)
        return

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    img_clahe = clahe.apply(img_grey)
    clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    img_clahe = clahe2.apply(img_clahe)
    cv2.imwrite(str(output_path), img_clahe)

def filter(input_path: Path, output_path: Path) -> None:
    image, mask, alpha = load_gray_and_mask(input_path)
    if not np.any(mask):
        raise RuntimeError("foreground mask is empty")

    blockh, blockw = 32, 32
    strideh, stridew = 16, 16
    pad = 8
    detail_gain = 24.0
    default_wavelength = 8.0
    dc_suppression_radius = 2
    image_padded = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    mask_padded = cv2.copyMakeBorder(
        mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    height, width = image.shape[:2]

    accum = np.zeros_like(image_padded, dtype=np.float32)
    count = np.zeros_like(image_padded, dtype=np.float32)

    for y in range(0, height - blockh + 1, strideh):
        for x in range(0, width - blockw + 1, stridew):
            block = image_padded[y:y+blockh, x:x+blockw]
            block_mask = mask_padded[y:y+blockh, x:x+blockw]
            block_float = block.astype(np.float32)
            foreground = block_mask == 255
            accum_view = accum[y:y+blockh, x:x+blockw]
            count_view = count[y:y+blockh, x:x+blockw]

            if not np.all(foreground):
                accum_view[foreground] += block_float[foreground]
                count_view[foreground] += 1.0
                continue

            block_fft = np.fft.fft2(block_float)
            block_fft_shifted = np.fft.fftshift(block_fft)
            magnitude = np.abs(block_fft_shifted)
            magnitude_for_peak = magnitude.copy()
            center_y = blockh // 2
            center_x = blockw // 2
            y0 = max(0, center_y - dc_suppression_radius)
            y1 = min(blockh, center_y + dc_suppression_radius + 1)
            x0 = max(0, center_x - dc_suppression_radius)
            x1 = min(blockw, center_x + dc_suppression_radius + 1)
            magnitude_for_peak[y0:y1, x0:x1] = 0.0
            peak = np.argmax(magnitude_for_peak)
            x_peak, y_peak = np.unravel_index(peak, magnitude.shape)
            peak_value = float(magnitude_for_peak[y_peak, x_peak])
            mean_energy = float(np.mean(magnitude_for_peak))
            std_energy = float(np.std(magnitude_for_peak))
            peak_confidence = (peak_value - mean_energy) / (std_energy + 1e-6)
            frequency_confidence = float(np.clip((peak_confidence - 1.5) / 3.0, 0.0, 1.0))
            dx_freq = (x_peak - center_x) / blockw
            dy_freq = (y_peak - center_y) / blockh
            freq = np.sqrt(dx_freq**2 + dy_freq**2)
            lambda_ = 1.0 / freq if freq > 1e-6 else default_wavelength
            lambda_ = float(np.clip(lambda_, 5.0, 12.0))
            sigma = max(2.0, 0.5 * lambda_)
            dx = cv2.Sobel(block_float, cv2.CV_64F, dx=1, dy=0, ksize=3)
            dy = cv2.Sobel(block_float, cv2.CV_64F, dx=0, dy=1, ksize=3)
            Gxx = np.sum(dx ** 2)
            Gyy = np.sum(dy ** 2)
            Gxy = np.sum(dx * dy)
            coherence_num = np.sqrt((2.0 * Gxy) ** 2 + (Gxx - Gyy) ** 2)
            coherence_den = Gxx + Gyy + 1e-6
            coherence = float(np.clip(coherence_num / coherence_den, 0.0, 1.0))
            reliability = float(np.clip((0.65 * coherence) + (0.35 * frequency_confidence), 0.0, 1.0))
            if reliability < 0.5:
                accum_view[foreground] += block_float[foreground]
                count_view[foreground] += 1.0
                continue
            theta = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
            ksize = int(np.round(3 * sigma))
            if ksize % 2 == 0:
                ksize += 1
            if ksize < 1:
                ksize = 3
            psi = 0
            gamma = 0.5
            gabor_kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F
            )
            block_mean = float(np.mean(block_float))
            block_std = float(np.std(block_float))
            if block_std < 1e-3:
                accum_view[foreground] += block_float[foreground]
                count_view[foreground] += 1.0
                continue

            normalized_block = (block_float - block_mean) / block_std
            response = cv2.filter2D(normalized_block, cv2.CV_32F, gabor_kernel)
            effective_gain = detail_gain * reliability
            enhanced_block = np.clip(
                block_float + (effective_gain * np.tanh(response)),
                0.0,
                255.0,
            )

            accum_view[foreground] += enhanced_block[foreground]
            count_view[foreground] += 1.0

    stitched = np.zeros_like(image, dtype=np.float32)
    cropped_accum = accum[pad:pad+height, pad:pad+width]
    cropped_count = count[pad:pad+height, pad:pad+width]
    foreground = mask == 255
    valid = foreground & (cropped_count > 0)
    stitched[valid] = cropped_accum[valid] / cropped_count[valid]
    stitched[foreground & ~valid] = image[foreground & ~valid].astype(np.float32)
    stitched = np.clip(stitched, 0.0, 255.0).astype(np.uint8)
    stitched[~foreground] = 0

    if alpha is not None:
        output = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGRA)
        output[:, :, 3] = alpha
        output[alpha == 0, :3] = 0
        cv2.imwrite(str(output_path), output)
        return

    cv2.imwrite(str(output_path), stitched)
    

def _is_downloadable_model(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _download_file_atomic(url: str, destination: Path, timeout_seconds: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        request = urlrequest.Request(url, headers={"User-Agent": "fingerflow-bootstrap/1.0"})
        with urlrequest.urlopen(request, timeout=timeout_seconds) as response, temp_path.open(
            "wb"
        ) as file_obj:
            content_type = response.headers.get_content_type()
            if content_type and content_type.startswith("text/"):
                raise RuntimeError(
                    f"unexpected content-type '{content_type}' while downloading model from {url}"
                )

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

    joined_errors = "; ".join(errors)
    raise RuntimeError(
        f"failed to download FingerFlow model '{model_name}' to {target}. "
        f"Tried URLs: {joined_errors}"
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


def _df_to_records(data) -> list[dict]:
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
        def __init__(self, core_net_path):
            self._core_net_path = core_net_path

        def detect_fingerprint_core(self, raw_image_data):
            return np.empty((0,))

    extractor_module.CoreNet = _NoCoreNet
    extractor_module._biometric_core_fallback_enabled = True


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

    try:
        extractor = Extractor(
            str(coarse_path), str(fine_path), str(classify_path), str(core_path)
        )
    except Exception as exc:
        if "A KerasTensor cannot be used as input to a TensorFlow function" in str(exc):
            _install_core_net_compat_fallback()
            extractor = Extractor(
                str(coarse_path), str(fine_path), str(classify_path), str(core_path)
            )
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
        minutiae_csv_path.parent.mkdir(parents=True, exist_ok=True)
        minutiae_df.to_csv(minutiae_csv_path, index=False)
    if hasattr(core_df, "to_csv"):
        core_csv_path.parent.mkdir(parents=True, exist_ok=True)
        core_df.to_csv(core_csv_path, index=False)

    minutiae_count = int(getattr(minutiae_df, "shape", [0])[0]) if minutiae_df is not None else 0
    core_count = int(getattr(core_df, "shape", [0])[0]) if core_df is not None else 0
    return minutiae_count, core_count

def main() -> int:
    args = parse_args()
    try:
        nobg_path, cropped_path = process_image(args.input_image)
        normalised_path = REPO_ROOT / "normalised_cropped.png"
        enhanced_path = DEFAULT_ENHANCED_IMAGE
        normalise_brightness(nobg_path, normalised_path)
        filter(normalised_path, enhanced_path)
        if not enhanced_path.exists() or enhanced_path.stat().st_size == 0:
            raise RuntimeError(f"enhanced image was not created: {enhanced_path}")

        model_paths = ensure_fingerflow_models(args.fingerflow_model_dir)
        minutiae_count, core_count = extract_minutiae_with_fingerflow(
            args.input_image.resolve(),
            enhanced_path.resolve(),
            model_paths,
            args.minutiae_json.resolve(),
            args.minutiae_csv.resolve(),
            args.core_csv.resolve(),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved background-removed image: {nobg_path}")
    print(f"Saved cropped fingerprint image: {cropped_path}")
    print(f"Saved enhanced fingerprint image: {enhanced_path.resolve()}")
    print(f"FingerFlow model cache directory: {args.fingerflow_model_dir.resolve()}")
    print(f"Saved FingerFlow minutiae JSON: {args.minutiae_json.resolve()}")
    print(f"Saved FingerFlow minutiae CSV: {args.minutiae_csv.resolve()}")
    print(f"Saved FingerFlow core CSV: {args.core_csv.resolve()}")
    print(f"FingerFlow extracted minutiae: {minutiae_count}, cores: {core_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
