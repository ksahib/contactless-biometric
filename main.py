from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
import sys
import sysconfig
import time
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

import pandas as pd
import cv2
import numpy as np
import math
from descriptor import MCCCell, MCCCylinder, MCCOverlapContext
import pose_normalization as pose_norm

try:
    from rembg import new_session, remove
    _HAS_REMBG = True
except Exception:
    new_session = None
    remove = None
    _HAS_REMBG = False


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_NOBG_DIR = REPO_ROOT / "output_nobg"
OUTPUT_CROPPED_DIR = REPO_ROOT / "output_cropped"
MATCH_OUTPUTS_DIR = REPO_ROOT / "match_outputs"
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
_PIPELINE_TIMINGS: dict[str, float] = {}
_PIPELINE_START = time.perf_counter()
_EXTRACTOR_CACHE: dict[tuple[str, str, str, str], "Extractor"] = {}
_MODEL_PATH_CACHE: dict[Path, tuple[Path, Path, Path, Path]] = {}


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _directml_diagnosis_message(tf_version: str) -> str:
    if not sys.platform.startswith("win"):
        return ""

    plugin_version = _installed_version("tensorflow-directml-plugin")
    if plugin_version:
        return (
            "Windows native TensorFlow>=2.11 is CPU-only; a DirectML plugin is installed "
            f"(tensorflow-directml-plugin=={plugin_version}), but no TF GPU is visible. "
            f"Verify plugin compatibility with TensorFlow {tf_version} in this same venv."
        )

    return (
        "Windows native TensorFlow>=2.11 is CPU-only. Install TensorFlow-DirectML plugin or run under WSL2.\n"
        "For DirectML, run:\n"
        "  pip install tensorflow-directml-plugin\n"
        "Then rerun:\n"
        "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"\n"
        "If no GPU is shown after install, install a DirectML-compatible TensorFlow build/version."
    )


def _log_tensorflow_environment() -> None:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TensorFlow import error while checking GPU diagnostics: {exc}", file=sys.stderr)
        return

    print(f"TensorFlow version: {getattr(tf, '__version__', 'unknown')}")
    try:
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    except Exception:
        pass
    try:
        print(f"TensorFlow built with GPU support: {tf.test.is_built_with_gpu_support()}")
    except Exception:
        pass
    try:
        print(f"TensorFlow build info: {tf.sysconfig.get_build_info()}")
    except Exception:
        pass

    try:
        gpus = tf.config.list_physical_devices("GPU")
        logical = tf.config.list_logical_devices("GPU")
        print(f"Visible TF physical GPU devices: {[str(device) for device in gpus]}")
        print(f"Visible TF logical GPU devices: {[str(device) for device in logical]}")
    except Exception as exc:
        print(f"Error reading TensorFlow devices: {exc}", file=sys.stderr)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")


def _print_runtime_context() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    try:
        import site

        print(f"site-packages: {', '.join(site.getsitepackages())}")
        print(f"user site-packages: {site.getusersitepackages()}")
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        print(f"venv active: {in_venv}")
        if not in_venv:
            print(
                "Warning: venv is not active. GPU packages installed in .venv may be ignored."
                " Activate venv with: .\\.venv\\Scripts\\Activate.ps1",
                file=sys.stderr,
            )
    except Exception:
        pass


def _ensure_tensorflow_gpu_available() -> None:
    if os.environ.get("FINGERFLOW_ALLOW_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(f"TensorFlow import failed: {exc}") from exc

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        _log_tensorflow_environment()
        tf_version = getattr(tf, "__version__", "unknown")
        platform_hint = []
        if sys.platform.startswith("win"):
            platform_hint.append(
                "Windows TensorFlow wheels may be CPU-only for this TensorFlow version."
            )

        platform_hint_text = " ".join(platform_hint)

        raise RuntimeError(
            "No TensorFlow-visible GPU found. This run is configured for GPU-only."
            f"{' ' + platform_hint_text if platform_hint_text else ''}"
            f" {_directml_diagnosis_message(tf_version)}"
            " Install/configure CUDA-enabled TensorFlow + NVIDIA drivers, and set"
            " CUDA_VISIBLE_DEVICES to an available GPU, or set FINGERFLOW_ALLOW_CPU=1 to allow fallback."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _record_timing(stage: str, start_time: float) -> None:
    _PIPELINE_TIMINGS[stage] = time.perf_counter() - start_time


def _format_timing_summary() -> str:
    ordered_stages = (
        "gpu_check",
        "process_image",
        "normalise_brightness",
        "filter",
        "model_load",
        "extract_minutiae",
        "fingerflow_total",
        "total",
    )
    chunks = []
    for stage in ordered_stages:
        value = _PIPELINE_TIMINGS.get(stage)
        if value is not None:
            chunks.append(f"{stage}={value:.3f}s")
    return ", ".join(chunks)


def _configure_inference_toggles() -> None:
    if getattr(_configure_inference_toggles, "_configured", False):
        return
    _configure_inference_toggles._configured = True
    if os.environ.get("FINGERFLOW_DISABLE_TF_ONEDNN", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    enable_xla = os.environ.get("FINGERFLOW_ENABLE_XLA", "").lower()
    if enable_xla not in {"1", "true", "yes", "on"}:
        return

    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        return


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


def _install_fingerflow_distance_compat() -> None:
    try:
        from fingerflow.extractor.MinutiaeNet.CoarseNet import minutiae_net_utils
        from scipy.spatial import distance as spatial_distance
    except Exception:
        return

    if getattr(minutiae_net_utils.distance, "_biometric_scalar_compat", False):
        return

    def _distance(y_true, y_pred, max_d=16, max_o=np.pi / 6):
        d = spatial_distance.cdist(y_true[:, :2], y_pred[:, :2], "euclidean")
        y_true_angles = np.reshape(y_true[:, 2], [-1, 1])
        y_pred_angles = np.reshape(y_pred[:, 2], [1, -1])
        o = np.abs(y_true_angles - y_pred_angles)
        o = np.minimum(o, (2 * np.pi) - o)
        return (d <= max_d) * (o <= max_o)

    _distance._biometric_scalar_compat = True  # type: ignore[attr-defined]
    minutiae_net_utils.distance = _distance  # type: ignore[assignment]


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
_disable_model_compile_on_load()
_normalize_optimizer_learning_rate_compat()
_normalize_optimizer_constructor_args_compat()
_install_keras_tensor_math_compat()
_install_numpy_lib_pad_compat()
_install_numpy_scalar_aliases_compat()
_install_skimage_gaussian_compat()
_install_scipy_signal_gaussian_compat()
_configure_inference_toggles()
_install_fingerflow_distance_compat()

from fingerflow.extractor import Extractor

def parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] == "match":
        parser = argparse.ArgumentParser(
            description="Match two fingerprint images with the MCC pipeline."
        )
        parser.add_argument("command", choices=["match"])
        parser.add_argument("image_a", type=Path, help="Path to the first fingerprint image")
        parser.add_argument("image_b", type=Path, help="Path to the second fingerprint image")
        parser.add_argument(
            "--method",
            default="LSA-R",
            choices=(
                "LSS",
                "LSA",
                "LSS-R",
                "LSA-R",
                "LSA-OVERLAP",
                "LSA-R-OVERLAP",
                "LSA-CANONICAL",
                "LSA-R-CANONICAL",
                "LSA-CANONICAL-OVERLAP",
                "LSA-R-CANONICAL-OVERLAP",
                "LSA-CENTROID",
                "LSA-R-CENTROID",
            ),
            help="MCC consolidation method.",
        )
        parser.add_argument(
            "--fingerflow-model-dir",
            type=Path,
            default=DEFAULT_FINGERFLOW_MODEL_DIR,
            help="Directory to cache FingerFlow model files.",
        )
        return parser.parse_args()

    parser = argparse.ArgumentParser(
        description="Crop a finger to the distal phalanx and remove its background."
    )
    parser.set_defaults(command="extract")
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
    tip_center, tip_to_base = select_fingertip_end(
        contour,
        axis_u,
        axis_v,
        length_l,
        mask.shape[:2],
    )

    width_near_tip = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.16)
    width_stable = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.28)
    distal_width = max(width_near_tip, 0.88 * width_stable)

    inward_extension = min(0.58 * length_l, max(1.95 * distal_width, 0.44 * length_l))
    outward_extension = min(0.07 * length_l, 0.32 * distal_width)
    half_width = 0.82 * distal_width
    margin_px = max(12, int(round(0.05 * max(length_l, distal_width))))

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
    start = time.perf_counter()
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
    _record_timing("process_image", start)
    return nobg_path, cropped_path

def normalise_brightness(input_path: Path, output_path: Path) -> None:
    start = time.perf_counter()
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
        _record_timing("normalise_brightness", start)
        return

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    img_clahe = clahe.apply(img_grey)
    clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    img_clahe = clahe2.apply(img_clahe)
    cv2.imwrite(str(output_path), img_clahe)
    _record_timing("normalise_brightness", start)

def filter(input_path: Path, output_path: Path) -> None:
    start = time.perf_counter()
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
        _record_timing("filter", start)
        return

    cv2.imwrite(str(output_path), stitched)
    _record_timing("filter", start)
    

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
    cached_paths = _MODEL_PATH_CACHE.get(model_dir)
    if cached_paths is not None:
        if all(_is_downloadable_model(path) for path in cached_paths):
            return cached_paths

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
    _MODEL_PATH_CACHE[model_dir] = (coarse, fine, classify, core)
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
    cache_key = (
        str(coarse_path),
        str(fine_path),
        str(classify_path),
        str(core_path),
    )

    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        try:
            extractor = Extractor(
                str(coarse_path), str(fine_path), str(classify_path), str(core_path)
            )
            _EXTRACTOR_CACHE[cache_key] = extractor
        except Exception as exc:
            if "A KerasTensor cannot be used as input to a TensorFlow function" in str(exc):
                _install_core_net_compat_fallback()
                extractor = Extractor(
                    str(coarse_path), str(fine_path), str(classify_path), str(core_path)
                )
                _EXTRACTOR_CACHE[cache_key] = extractor
            else:
                raise

    extract_start = time.perf_counter()
    extracted = extractor.extract_minutiae(image)
    _record_timing("extract_minutiae", extract_start)

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

MCC_RADIUS = 70.0
MCC_NS = 16
MCC_ND = 6
MCC_SIGMA_S = 28.0 / 3.0
MCC_SIGMA_D = 2.0 * math.pi / 9.0
MCC_MU_PSI = 1.0 / 100.0
MCC_TAU_PSI = 400.0
MCC_HULL_OFFSET = 50.0
MCC_MIN_M = 2
MCC_SPARSE_MINUTIAE_COUNT = 12
MCC_DELTA_THETA = math.pi / 2.0
MCC_MIN_NP = 4
MCC_MAX_NP = 12
MCC_MU_P = 20.0
MCC_TAU_P = 2.0 / 5.0
MCC_W_R = 0.5
MCC_NREL = 5
MCC_RHO_PARAMS = (
    (5.0, -8.0 / 5.0),
    (math.pi / 12.0, -30.0),
    (math.pi / 12.0, -30.0),
)
MCC_OVERLAP_MIN_RATIO = 0.18
MCC_OVERLAP_MIN_MINUTIAE = 4
MCC_OVERLAP_MIN_SCALE = 0.80
MCC_OVERLAP_MAX_SCALE = 1.25
MCC_CANONICAL_FRAME_WIDTH = 1024
MCC_CANONICAL_FRAME_HEIGHT = 1280
MCC_CANONICAL_TIP_X = MCC_CANONICAL_FRAME_WIDTH / 2.0
MCC_CANONICAL_TIP_Y = 120.0
MCC_CANONICAL_TARGET_LENGTH = 850.0
MCC_CANONICAL_TARGET_WIDTH = 550.0


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _sigmoid(value: float, mu: float, tau: float) -> float:
    exponent = -tau * (value - mu)
    if exponent > 60.0:
        return 0.0
    if exponent < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def _gaussian(distance: float, sigma: float) -> float:
    return math.exp(-(distance * distance) / (2.0 * sigma * sigma)) / (
        sigma * math.sqrt(2.0 * math.pi)
    )


def _gaussian_area(alpha: float, delta: float, sigma: float) -> float:
    lower = (alpha - (delta / 2.0)) / (sigma * math.sqrt(2.0))
    upper = (alpha + (delta / 2.0)) / (sigma * math.sqrt(2.0))
    return 0.5 * (math.erf(upper) - math.erf(lower))


def _max_valid_cells(ns: int, nd: int, delta_s: float, radius: float) -> int:
    count = 0
    center = (ns + 1) / 2.0
    for i in range(ns):
        for j in range(ns):
            dx = delta_s * ((i + 1) - center)
            dy = delta_s * ((j + 1) - center)
            if (dx * dx + dy * dy) <= (radius * radius):
                count += nd
    return count


def _build_convex_hull(minutiae: list[dict]) -> np.ndarray | None:
    if len(minutiae) < 3:
        return None
    points = np.array(
        [[float(minutia["x"]), float(minutia["y"])] for minutia in minutiae],
        dtype=np.float32,
    )
    return cv2.convexHull(points)


def _point_in_enlarged_hull(
    hull: np.ndarray | None,
    x: float,
    y: float,
    offset: float,
) -> bool:
    if hull is None:
        return True
    distance = cv2.pointPolygonTest(hull, (float(x), float(y)), True)
    return distance >= -offset


def _load_validity_mask(mask_path: Path | None) -> np.ndarray | None:
    if mask_path is None:
        return None
    if isinstance(mask_path, np.ndarray):
        mask = mask_path.copy()
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"validity mask not found: {mask_path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _point_in_validity_mask(mask: np.ndarray | None, x: float, y: float) -> bool:
    if mask is None:
        return False
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _build_overlap_context(validity_mask: np.ndarray | None) -> MCCOverlapContext | None:
    if validity_mask is None:
        return None
    return MCCOverlapContext(mask=validity_mask)


def _descriptor_has_overlap_context(descriptor: MCCCylinder) -> bool:
    return descriptor.overlap_context is not None and descriptor.overlap_context.mask is not None


def _transform_point(
    x: float,
    y: float,
    rotation: float,
    translation_x: float,
    translation_y: float,
) -> tuple[float, float]:
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    projected_x = (cos_r * float(x)) - (sin_r * float(y)) + float(translation_x)
    projected_y = (sin_r * float(x)) + (cos_r * float(y)) + float(translation_y)
    return projected_x, projected_y


def _descriptor_pair_alignment(
    cylinder_a: MCCCylinder,
    cylinder_b: MCCCylinder,
) -> tuple[float, float, float]:
    rotation = wrap_angle(float(cylinder_a.center_theta) - float(cylinder_b.center_theta))
    projected_center_x, projected_center_y = _transform_point(
        cylinder_b.center_x,
        cylinder_b.center_y,
        rotation,
        0.0,
        0.0,
    )
    translation_x = float(cylinder_a.center_x) - projected_center_x
    translation_y = float(cylinder_a.center_y) - projected_center_y
    return rotation, translation_x, translation_y


def _overlap_valid_cell_mask(
    cylinder_a: MCCCylinder,
    cylinder_b: MCCCylinder,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
) -> np.ndarray | None:
    if not _descriptor_has_overlap_context(cylinder_a) or not _descriptor_has_overlap_context(cylinder_b):
        return None

    mask_a = cylinder_a.overlap_context.mask
    mask_b = cylinder_b.overlap_context.mask
    rotation, translation_x, translation_y = _descriptor_pair_alignment(cylinder_a, cylinder_b)
    overlap_valid = np.zeros(valid_a.shape, dtype=bool)

    for index, cell_b in enumerate(cylinder_b.cells):
        if valid_a[index] != 1 or valid_b[index] != 1:
            continue
        projected_x, projected_y = _transform_point(
            cell_b.x,
            cell_b.y,
            rotation,
            translation_x,
            translation_y,
        )
        if not _point_in_validity_mask(mask_a, projected_x, projected_y):
            continue
        if not _point_in_validity_mask(mask_b, cell_b.x, cell_b.y):
            continue
        overlap_valid[index] = True

    return overlap_valid


def _rotation_angle(vector: np.ndarray) -> float:
    return float(math.atan2(float(vector[1]), float(vector[0])))


def _similarity_affine_matrix(
    scale: float,
    rotation: float,
    translation_x: float,
    translation_y: float,
) -> np.ndarray:
    cos_r = math.cos(rotation) * float(scale)
    sin_r = math.sin(rotation) * float(scale)
    return np.array(
        [
            [cos_r, -sin_r, float(translation_x)],
            [sin_r, cos_r, float(translation_y)],
        ],
        dtype=np.float32,
    )


def _inverse_similarity_transform(
    scale: float,
    rotation: float,
    translation_x: float,
    translation_y: float,
) -> tuple[float, float, float, float]:
    inv_scale = 1.0 / float(scale)
    inv_rotation = -float(rotation)
    cos_r = math.cos(inv_rotation) * inv_scale
    sin_r = math.sin(inv_rotation) * inv_scale
    inv_translation_x = -((cos_r * float(translation_x)) - (sin_r * float(translation_y)))
    inv_translation_y = -((sin_r * float(translation_x)) + (cos_r * float(translation_y)))
    return inv_scale, inv_rotation, inv_translation_x, inv_translation_y


def _warp_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
    scale: float,
    rotation: float,
    translation_x: float,
    translation_y: float,
) -> np.ndarray:
    warped = cv2.warpAffine(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        _similarity_affine_matrix(scale, rotation, translation_x, translation_y),
        (int(shape[1]), int(shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(warped > 0, 255, 0).astype(np.uint8)


def _axis_stats_from_mask(mask: np.ndarray) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contour available for overlap estimation")
    contour = max(contours, key=cv2.contourArea)
    axis_u, axis_v, length_l, width_w = estimate_finger_axis(contour)
    tip_center, tip_to_base = select_fingertip_end(
        contour,
        axis_u,
        axis_v,
        length_l,
        mask.shape[:2],
    )
    center = np.array(
        [float(np.mean(contour[:, 0, 0])), float(np.mean(contour[:, 0, 1]))],
        dtype=np.float32,
    )
    return {
        "contour": contour,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "length": float(length_l),
        "width": float(width_w),
        "tip_center": tip_center.astype(np.float32),
        "tip_to_base": tip_to_base.astype(np.float32),
        "center": center,
    }


def _estimate_overlap_region(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict:
    stats_a = _axis_stats_from_mask(mask_a)
    stats_b = _axis_stats_from_mask(mask_b)
    length_scale = stats_a["length"] / max(stats_b["length"], 1e-6)
    width_scale = stats_a["width"] / max(stats_b["width"], 1e-6)
    scale = float((length_scale + width_scale) / 2.0)

    details = {
        "estimated_transform": None,
        "overlap_area": 0,
        "overlap_ratio": 0.0,
        "fallback_reason": None,
        "common_mask_a": None,
        "common_mask_b": None,
        "warped_mask_b_area": 0,
    }

    if not (MCC_OVERLAP_MIN_SCALE <= scale <= MCC_OVERLAP_MAX_SCALE):
        details["fallback_reason"] = "scale_out_of_range"
        return details

    rotation = wrap_angle(
        _rotation_angle(stats_a["tip_to_base"]) - _rotation_angle(stats_b["tip_to_base"])
    )
    scaled_tip_b_x = float(stats_b["tip_center"][0]) * scale
    scaled_tip_b_y = float(stats_b["tip_center"][1]) * scale
    rotated_tip_b_x, rotated_tip_b_y = _transform_point(
        scaled_tip_b_x,
        scaled_tip_b_y,
        rotation,
        0.0,
        0.0,
    )
    translation_x = float(stats_a["tip_center"][0]) - rotated_tip_b_x
    translation_y = float(stats_a["tip_center"][1]) - rotated_tip_b_y

    warped_mask_b = _warp_mask(
        mask_b,
        mask_a.shape[:2],
        scale,
        rotation,
        translation_x,
        translation_y,
    )
    common_mask_a = np.where((mask_a > 0) & (warped_mask_b > 0), 255, 0).astype(np.uint8)
    smaller_area = max(1, min(int(np.count_nonzero(mask_a)), int(np.count_nonzero(mask_b))))
    overlap_area = int(np.count_nonzero(common_mask_a))
    overlap_ratio = float(overlap_area / smaller_area)

    if overlap_ratio < MCC_OVERLAP_MIN_RATIO:
        details["fallback_reason"] = "overlap_ratio_too_small"
        details["overlap_area"] = overlap_area
        details["overlap_ratio"] = overlap_ratio
        details["warped_mask_b_area"] = int(np.count_nonzero(warped_mask_b))
        details["estimated_transform"] = {
            "scale": scale,
            "rotation": rotation,
            "rotation_degrees": float(math.degrees(rotation)),
            "translation_x": translation_x,
            "translation_y": translation_y,
        }
        return details

    inv_scale, inv_rotation, inv_tx, inv_ty = _inverse_similarity_transform(
        scale,
        rotation,
        translation_x,
        translation_y,
    )
    warped_mask_a = _warp_mask(
        mask_a,
        mask_b.shape[:2],
        inv_scale,
        inv_rotation,
        inv_tx,
        inv_ty,
    )
    common_mask_b = np.where((mask_b > 0) & (warped_mask_a > 0), 255, 0).astype(np.uint8)

    details["estimated_transform"] = {
        "scale": scale,
        "rotation": rotation,
        "rotation_degrees": float(math.degrees(rotation)),
        "translation_x": translation_x,
        "translation_y": translation_y,
    }
    details["overlap_area"] = overlap_area
    details["overlap_ratio"] = overlap_ratio
    details["common_mask_a"] = common_mask_a
    details["common_mask_b"] = common_mask_b
    details["warped_mask_b_area"] = int(np.count_nonzero(warped_mask_b))
    return details


def _estimate_canonical_transform(mask: np.ndarray) -> dict:
    stats = _axis_stats_from_mask(mask)
    target_rotation = math.pi / 2.0
    rotation = wrap_angle(target_rotation - _rotation_angle(stats["tip_to_base"]))
    length_scale = MCC_CANONICAL_TARGET_LENGTH / max(stats["length"], 1e-6)
    width_scale = MCC_CANONICAL_TARGET_WIDTH / max(stats["width"], 1e-6)
    scale = float((length_scale + width_scale) / 2.0)

    details = {
        "estimated_transform": None,
        "warped_mask": None,
        "mask_area": 0,
        "fallback_reason": None,
    }
    if not (MCC_OVERLAP_MIN_SCALE <= scale <= MCC_OVERLAP_MAX_SCALE):
        details["fallback_reason"] = "scale_out_of_range"
        details["estimated_transform"] = {
            "scale": scale,
            "rotation": rotation,
            "rotation_degrees": float(math.degrees(rotation)),
            "translation_x": None,
            "translation_y": None,
            "source_length": float(stats["length"]),
            "source_width": float(stats["width"]),
        }
        return details

    scaled_tip_x = float(stats["tip_center"][0]) * scale
    scaled_tip_y = float(stats["tip_center"][1]) * scale
    rotated_tip_x, rotated_tip_y = _transform_point(
        scaled_tip_x,
        scaled_tip_y,
        rotation,
        0.0,
        0.0,
    )
    translation_x = float(MCC_CANONICAL_TIP_X) - rotated_tip_x
    translation_y = float(MCC_CANONICAL_TIP_Y) - rotated_tip_y
    warped_mask = _warp_mask(
        mask,
        (MCC_CANONICAL_FRAME_HEIGHT, MCC_CANONICAL_FRAME_WIDTH),
        scale,
        rotation,
        translation_x,
        translation_y,
    )
    details["estimated_transform"] = {
        "scale": scale,
        "rotation": rotation,
        "rotation_degrees": float(math.degrees(rotation)),
        "translation_x": translation_x,
        "translation_y": translation_y,
        "source_length": float(stats["length"]),
        "source_width": float(stats["width"]),
        "canonical_tip_x": float(MCC_CANONICAL_TIP_X),
        "canonical_tip_y": float(MCC_CANONICAL_TIP_Y),
        "canonical_frame_width": int(MCC_CANONICAL_FRAME_WIDTH),
        "canonical_frame_height": int(MCC_CANONICAL_FRAME_HEIGHT),
    }
    details["warped_mask"] = warped_mask
    details["mask_area"] = int(np.count_nonzero(warped_mask))
    return details


def _estimate_canonical_overlap_region(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict:
    canonical_a = _estimate_canonical_transform(mask_a)
    canonical_b = _estimate_canonical_transform(mask_b)
    details = {
        "estimated_transform": {
            "strategy": "canonical",
            "left": canonical_a["estimated_transform"],
            "right": canonical_b["estimated_transform"],
        },
        "overlap_area": 0,
        "overlap_ratio": 0.0,
        "fallback_reason": None,
        "common_mask_a": None,
        "common_mask_b": None,
        "warped_mask_a_area": int(canonical_a["mask_area"]),
        "warped_mask_b_area": int(canonical_b["mask_area"]),
    }

    if canonical_a["fallback_reason"] is not None:
        details["fallback_reason"] = f"left_{canonical_a['fallback_reason']}"
        return details
    if canonical_b["fallback_reason"] is not None:
        details["fallback_reason"] = f"right_{canonical_b['fallback_reason']}"
        return details

    warped_mask_a = canonical_a["warped_mask"]
    warped_mask_b = canonical_b["warped_mask"]
    common_mask = np.where((warped_mask_a > 0) & (warped_mask_b > 0), 255, 0).astype(np.uint8)
    smaller_area = max(1, min(int(np.count_nonzero(warped_mask_a)), int(np.count_nonzero(warped_mask_b))))
    overlap_area = int(np.count_nonzero(common_mask))
    overlap_ratio = float(overlap_area / smaller_area)
    details["overlap_area"] = overlap_area
    details["overlap_ratio"] = overlap_ratio
    if overlap_ratio < MCC_OVERLAP_MIN_RATIO:
        details["fallback_reason"] = "overlap_ratio_too_small"
        return details

    details["common_mask_a"] = common_mask
    details["common_mask_b"] = common_mask.copy()
    return details


def _filter_minutiae_by_mask(
    minutiae_source: Path | pd.DataFrame,
    mask: np.ndarray,
) -> pd.DataFrame:
    if isinstance(minutiae_source, pd.DataFrame):
        frame = minutiae_source.copy()
    else:
        frame = pd.read_csv(minutiae_source)
    frame = frame.dropna(subset=["x", "y", "angle"])
    if frame.empty:
        return frame

    keep = frame.apply(
        lambda row: _point_in_validity_mask(mask, float(row["x"]), float(row["y"])),
        axis=1,
    )
    return frame.loc[keep].reset_index(drop=True)


def _transform_minutiae_frame(
    minutiae_source: Path | pd.DataFrame,
    scale: float,
    rotation: float,
    translation_x: float,
    translation_y: float,
) -> pd.DataFrame:
    if isinstance(minutiae_source, pd.DataFrame):
        frame = minutiae_source.copy()
    else:
        frame = pd.read_csv(minutiae_source)
    frame = frame.dropna(subset=["x", "y", "angle"]).reset_index(drop=True)
    if frame.empty:
        return frame

    transformed = frame.copy()
    coords = frame[["x", "y"]].to_numpy(dtype=np.float32)
    cos_r = math.cos(rotation) * float(scale)
    sin_r = math.sin(rotation) * float(scale)
    matrix = np.array(
        [
            [cos_r, -sin_r],
            [sin_r, cos_r],
        ],
        dtype=np.float32,
    )
    transformed_coords = coords @ matrix.T
    transformed_coords[:, 0] += float(translation_x)
    transformed_coords[:, 1] += float(translation_y)
    transformed.loc[:, "x"] = transformed_coords[:, 0]
    transformed.loc[:, "y"] = transformed_coords[:, 1]
    transformed.loc[:, "angle"] = transformed["angle"].apply(
        lambda value: wrap_angle(float(value) + float(rotation))
    )
    return transformed.reset_index(drop=True)


def _base_method_for_overlap(method: str) -> str:
    normalized = method.upper()
    if normalized == "LSA-OVERLAP":
        return "LSA"
    if normalized == "LSA-R-OVERLAP":
        return "LSA-R"
    raise ValueError(f"unsupported overlap-aware method: {method}")


def _base_method_for_legacy(method: str) -> str:
    normalized = method.upper()
    if normalized in {"LSA", "LSA-R", "LSS", "LSS-R"}:
        return normalized
    if normalized == "LSA-LEGACY":
        return "LSA"
    if normalized == "LSA-R-LEGACY":
        return "LSA-R"
    raise ValueError(f"unsupported legacy MCC method: {method}")


def _base_method_for_canonical(method: str) -> str:
    normalized = method.upper()
    if normalized in {"LSA-CANONICAL", "LSA-CANONICAL-OVERLAP"}:
        return "LSA"
    if normalized in {"LSA-R-CANONICAL", "LSA-R-CANONICAL-OVERLAP"}:
        return "LSA-R"
    raise ValueError(f"unsupported canonical MCC method: {method}")


def _base_method_for_centroid(method: str) -> str:
    normalized = method.upper()
    if normalized == "LSA-CENTROID":
        return "LSA"
    if normalized == "LSA-R-CENTROID":
        return "LSA-R"
    raise ValueError(f"unsupported centroid MCC method: {method}")


def _canonical_method_uses_overlap(method: str) -> bool:
    return method.upper().endswith("-OVERLAP")


def _load_minutiae_frame_for_centroid(path: Path | pd.DataFrame) -> pd.DataFrame:
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    if "angle" not in frame.columns and "theta" in frame.columns:
        frame = frame.copy()
        frame["angle"] = frame["theta"]
    return frame.dropna(subset=["x", "y", "angle"]).reset_index(drop=True)


def _centroid_sidecar_path(
    minutiae_source: Path | pd.DataFrame,
    explicit_path: Path | np.ndarray | None,
    filename: str,
) -> Path | np.ndarray | None:
    if explicit_path is not None:
        return explicit_path
    if isinstance(minutiae_source, pd.DataFrame):
        return None
    candidate = Path(minutiae_source).parent / filename
    return candidate if candidate.exists() else None


def _load_centroid_sidecar_array(
    source: Path | np.ndarray | None,
    label: str,
    warnings: list[str],
) -> np.ndarray | None:
    if source is None:
        warnings.append(f"{label}_missing")
        return None
    try:
        return pose_norm.load_optional_array(source)
    except Exception as exc:
        warnings.append(f"{label}_load_failed:{exc}")
        return None


def _frame_from_minutiae(minutiae: list[pose_norm.Minutia]) -> pd.DataFrame:
    frame = pd.DataFrame([pose_norm.minutia_to_record(minutia) for minutia in minutiae])
    if "angle" not in frame.columns:
        frame["angle"] = np.nan
    return frame.reset_index(drop=True)


def _select_report_pairs(
    descriptors_a: list[MCCCylinder],
    descriptors_b: list[MCCCylinder],
    sim_matrix: np.ndarray,
    base_method: str,
) -> tuple[list[tuple[int, int, float]], list[float], dict | None]:
    n_pairs = _compute_n_pairs(len(descriptors_a), len(descriptors_b))
    if n_pairs == 0:
        return [], [], None
    if base_method == "LSA":
        pairs = _select_lsa_pairs(sim_matrix, n_pairs)
        return pairs, [], None

    rel_pairs = _select_lsa_pairs(
        sim_matrix,
        min(len(descriptors_a), len(descriptors_b)),
    )
    relaxed, efficiency, relaxation_details = _relax_pairs_with_details(
        descriptors_a,
        descriptors_b,
        rel_pairs,
    )
    top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(rel_pairs))]
    selected_pairs = [rel_pairs[index] for index in top_indices]
    relaxed_top_scores = [float(relaxed[index]) for index in top_indices]
    return selected_pairs, relaxed_top_scores, relaxation_details


def match_minutiae_csv_centroid_details(
    path_a: Path | pd.DataFrame,
    path_b: Path | pd.DataFrame,
    method: str = "LSA-R-CENTROID",
    orientation_path_a: Path | np.ndarray | None = None,
    orientation_path_b: Path | np.ndarray | None = None,
    ridge_period_path_a: Path | np.ndarray | None = None,
    ridge_period_path_b: Path | np.ndarray | None = None,
    config: pose_norm.PoseNormalizationConfig | None = None,
) -> tuple[float, np.ndarray, dict]:
    base_method = _base_method_for_centroid(method)
    cfg = config or pose_norm.PoseNormalizationConfig()
    warnings: list[str] = []

    raw_frame_a = _load_minutiae_frame_for_centroid(path_a)
    raw_frame_b = _load_minutiae_frame_for_centroid(path_b)
    details = {
        "fallback_to_legacy": False,
        "fallback_reason": None,
        "method": method.upper(),
        "base_method": base_method,
        "left_raw_minutiae_count": int(len(raw_frame_a)),
        "right_raw_minutiae_count": int(len(raw_frame_b)),
        "left_descriptor_count_after": 0,
        "right_descriptor_count_after": 0,
        "selected_pair_count": 0,
        "selected_pairs": [],
        "selected_pair_scores": [],
        "relaxed_top_scores": [],
        "relaxation_details": None,
        "transform": None,
        "pose_normalization": None,
        "warnings": warnings,
    }

    if len(raw_frame_a) < int(cfg.min_minutiae) or len(raw_frame_b) < int(cfg.min_minutiae):
        details["fallback_reason"] = "too_few_minutiae_for_centroid"
        details["warnings"].append("too_few_minutiae_for_centroid")
        return 0.0, np.zeros((0, 0), dtype=np.float32), details
    if len(raw_frame_a) < int(cfg.reliable_minutiae):
        warnings.append("left_centroid_low_confidence")
    if len(raw_frame_b) < int(cfg.reliable_minutiae):
        warnings.append("right_centroid_low_confidence")

    minutiae_a = pose_norm.coerce_minutiae(raw_frame_a.to_dict(orient="records"))
    minutiae_b = pose_norm.coerce_minutiae(raw_frame_b.to_dict(orient="records"))
    centroid_a = pose_norm.compute_minutiae_centroid(
        minutiae_a,
        use_quality_weights=cfg.use_quality_weighted_centroid,
    )
    centroid_b = pose_norm.compute_minutiae_centroid(
        minutiae_b,
        use_quality_weights=cfg.use_quality_weighted_centroid,
    )

    rotation = 0.0
    rotation_source = "none"
    theta_a = None
    theta_b = None
    resolved_orientation_a = _centroid_sidecar_path(path_a, orientation_path_a, "orientation.npy")
    resolved_orientation_b = _centroid_sidecar_path(path_b, orientation_path_b, "orientation.npy")
    orientation_a = _load_centroid_sidecar_array(resolved_orientation_a, "left_orientation", warnings)
    orientation_b = _load_centroid_sidecar_array(resolved_orientation_b, "right_orientation", warnings)
    if orientation_a is not None and orientation_b is not None:
        try:
            theta_a = pose_norm.estimate_global_orientation_from_field(orientation_a)
            theta_b = pose_norm.estimate_global_orientation_from_field(orientation_b)
            rotation = pose_norm.wrap_angle_pi(float(theta_a) - float(theta_b))
            rotation_source = "ridge_orientation"
        except Exception as exc:
            warnings.append(f"rotation_estimate_failed:{exc}")

    scale = 1.0
    scale_source = "none"
    spacing_a = None
    spacing_b = None
    resolved_ridge_a = _centroid_sidecar_path(path_a, ridge_period_path_a, "ridge_period.npy")
    resolved_ridge_b = _centroid_sidecar_path(path_b, ridge_period_path_b, "ridge_period.npy")
    ridge_a = _load_centroid_sidecar_array(resolved_ridge_a, "left_ridge_period", warnings)
    ridge_b = _load_centroid_sidecar_array(resolved_ridge_b, "right_ridge_period", warnings)
    if ridge_a is not None and ridge_b is not None:
        try:
            spacing_a = pose_norm.estimate_median_ridge_spacing(
                ridge_a,
                min_valid_spacing=cfg.min_valid_ridge_spacing,
                max_valid_spacing=cfg.max_valid_ridge_spacing,
            )
            spacing_b = pose_norm.estimate_median_ridge_spacing(
                ridge_b,
                min_valid_spacing=cfg.min_valid_ridge_spacing,
                max_valid_spacing=cfg.max_valid_ridge_spacing,
            )
            scale = pose_norm.estimate_scale_from_ridge_spacing(
                query_spacing=spacing_b,
                template_spacing=spacing_a,
                min_valid_scale=cfg.min_valid_scale,
                max_valid_scale=cfg.max_valid_scale,
            )
            scale_source = "ridge_spacing"
        except Exception as exc:
            warnings.append(f"scale_estimate_failed:{exc}")

    transform = pose_norm.SimilarityTransform(
        query_centroid=centroid_b,
        template_centroid=centroid_a,
        rotation=rotation,
        scale=scale,
        rotation_source=rotation_source,
        scale_source=scale_source,
    )
    normalized_a, _ = pose_norm.translate_minutiae_to_centroid(minutiae_a, centroid_a)
    normalized_b = pose_norm.apply_similarity_transform_to_minutiae(minutiae_b, transform)
    normalized_frame_a = _frame_from_minutiae(normalized_a)
    normalized_frame_b = _frame_from_minutiae(normalized_b)

    descriptors_a = build_descriptors(normalized_frame_a, validity_mask_path=None, validity_mode="auto")
    descriptors_b = build_descriptors(normalized_frame_b, validity_mask_path=None, validity_mode="auto")
    details["left_descriptor_count_after"] = int(len(descriptors_a))
    details["right_descriptor_count_after"] = int(len(descriptors_b))
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        details["fallback_reason"] = "no_descriptors_after_centroid_normalization"
        details["warnings"].append("no_descriptors_after_centroid_normalization")
        return 0.0, np.zeros((len(descriptors_a), len(descriptors_b)), dtype=np.float32), details

    score, sim_matrix = match_descriptors(
        descriptors_a,
        descriptors_b,
        method=base_method,
        overlap_mode="off",
    )
    selected_pairs, relaxed_top_scores, relaxation_details = _select_report_pairs(
        descriptors_a,
        descriptors_b,
        sim_matrix,
        base_method,
    )
    diagnostics = pose_norm.PoseNormalizationDiagnostics(
        query_centroid=centroid_b,
        template_centroid=centroid_a,
        rotation=rotation,
        rotation_source=rotation_source,
        scale=scale,
        scale_source=scale_source,
        centroid_weighted=cfg.use_quality_weighted_centroid,
        warnings=warnings,
        query_orientation=theta_b,
        template_orientation=theta_a,
        query_ridge_spacing=spacing_b,
        template_ridge_spacing=spacing_a,
    )
    details["transform"] = {
        "query_centroid": [float(centroid_b[0]), float(centroid_b[1])],
        "template_centroid": [float(centroid_a[0]), float(centroid_a[1])],
        "rotation": float(rotation),
        "rotation_degrees": float(math.degrees(rotation)),
        "scale": float(scale),
        "rotation_source": rotation_source,
        "scale_source": scale_source,
    }
    details["pose_normalization"] = diagnostics.to_dict()
    details["selected_pair_count"] = int(len(selected_pairs))
    details["selected_pairs"] = [
        {"row": int(row), "col": int(col), "score": float(pair_score)}
        for row, col, pair_score in selected_pairs
    ]
    details["selected_pair_scores"] = [float(pair_score) for _, _, pair_score in selected_pairs]
    details["relaxed_top_scores"] = relaxed_top_scores
    details["relaxation_details"] = relaxation_details
    details["final_score"] = float(score)
    return score, sim_matrix, details


def match_minutiae_csv_legacy(
    path_a: Path | pd.DataFrame,
    path_b: Path | pd.DataFrame,
    method: str = "LSA-R",
    mask_path_a: Path | np.ndarray | None = None,
    mask_path_b: Path | np.ndarray | None = None,
    overlap_mode: str = "auto",
) -> tuple[float, np.ndarray]:
    base_method = _base_method_for_legacy(method)
    descriptors_a = build_descriptors(path_a, validity_mask_path=mask_path_a)
    descriptors_b = build_descriptors(path_b, validity_mask_path=mask_path_b)
    return match_descriptors(
        descriptors_a,
        descriptors_b,
        method=base_method,
        overlap_mode=overlap_mode,
    )


def match_minutiae_csv_pose_normalized_details(
    path_a: Path | pd.DataFrame,
    path_b: Path | pd.DataFrame,
    mask_path_a: Path | np.ndarray | None,
    mask_path_b: Path | np.ndarray | None,
    method: str,
    strategy: str = "relative",
    use_common_region_filter: bool = True,
    overlap_mode: str = "auto",
) -> tuple[float, np.ndarray, dict]:
    base_method = _base_method_for_legacy(method)
    normalized_strategy = strategy.lower()
    if normalized_strategy not in {"relative", "canonical"}:
        raise ValueError(f"unsupported pose normalization strategy: {strategy}")
    legacy_score, legacy_sim_matrix = match_minutiae_csv_legacy(
        path_a if isinstance(path_a, Path) else path_a.copy(),
        path_b if isinstance(path_b, Path) else path_b.copy(),
        method=base_method,
        mask_path_a=mask_path_a if mask_path_a is None or isinstance(mask_path_a, Path) else mask_path_a.copy(),
        mask_path_b=mask_path_b if mask_path_b is None or isinstance(mask_path_b, Path) else mask_path_b.copy(),
        overlap_mode=overlap_mode,
    )

    details = {
        "fallback_to_legacy": False,
        "fallback_reason": None,
        "estimated_transform": None,
        "overlap_area": 0,
        "overlap_ratio": 0.0,
        "left_raw_minutiae_count": 0,
        "right_raw_minutiae_count": 0,
        "left_overlap_minutiae_count": 0,
        "right_overlap_minutiae_count": 0,
        "left_descriptor_count_before": 0,
        "right_descriptor_count_before": 0,
        "left_descriptor_count_after": 0,
        "right_descriptor_count_after": 0,
        "selected_pair_count": 0,
        "selected_pairs": [],
        "selected_pair_scores": [],
        "relaxed_top_scores": [],
        "relaxation_details": None,
        "legacy_score": float(legacy_score),
        "legacy_method": f"{base_method}-LEGACY",
        "strategy": normalized_strategy,
        "use_common_region_filter": bool(use_common_region_filter),
    }

    if mask_path_a is None or mask_path_b is None:
        details["fallback_to_legacy"] = True
        details["fallback_reason"] = "missing_masks"
        return legacy_score, legacy_sim_matrix, details

    mask_a = _load_validity_mask(mask_path_a)
    mask_b = _load_validity_mask(mask_path_b)
    if mask_a is None or mask_b is None:
        details["fallback_to_legacy"] = True
        details["fallback_reason"] = "missing_masks"
        return legacy_score, legacy_sim_matrix, details

    raw_frame_a = path_a.copy() if isinstance(path_a, pd.DataFrame) else pd.read_csv(path_a)
    raw_frame_b = path_b.copy() if isinstance(path_b, pd.DataFrame) else pd.read_csv(path_b)
    raw_frame_a = raw_frame_a.dropna(subset=["x", "y", "angle"]).reset_index(drop=True)
    raw_frame_b = raw_frame_b.dropna(subset=["x", "y", "angle"]).reset_index(drop=True)
    details["left_raw_minutiae_count"] = int(len(raw_frame_a))
    details["right_raw_minutiae_count"] = int(len(raw_frame_b))

    if normalized_strategy == "relative":
        overlap_details = _estimate_overlap_region(mask_a, mask_b)
    else:
        overlap_details = _estimate_canonical_overlap_region(mask_a, mask_b)
    details["estimated_transform"] = overlap_details["estimated_transform"]
    details["overlap_area"] = int(overlap_details["overlap_area"])
    details["overlap_ratio"] = float(overlap_details["overlap_ratio"])
    if overlap_details["fallback_reason"] is not None:
        details["fallback_to_legacy"] = True
        details["fallback_reason"] = overlap_details["fallback_reason"]
        return legacy_score, legacy_sim_matrix, details

    if normalized_strategy == "relative":
        transform = overlap_details["estimated_transform"]
        normalized_a = raw_frame_a.copy()
        normalized_b = _transform_minutiae_frame(
            raw_frame_b,
            scale=float(transform["scale"]),
            rotation=float(transform["rotation"]),
            translation_x=float(transform["translation_x"]),
            translation_y=float(transform["translation_y"]),
        )
        common_mask_a = overlap_details["common_mask_a"]
        common_mask_b = overlap_details["common_mask_b"]
        descriptor_mask_a = common_mask_a
        descriptor_mask_b = common_mask_b
    else:
        transform_left = overlap_details["estimated_transform"]["left"]
        transform_right = overlap_details["estimated_transform"]["right"]
        normalized_a = _transform_minutiae_frame(
            raw_frame_a,
            scale=float(transform_left["scale"]),
            rotation=float(transform_left["rotation"]),
            translation_x=float(transform_left["translation_x"]),
            translation_y=float(transform_left["translation_y"]),
        )
        normalized_b = _transform_minutiae_frame(
            raw_frame_b,
            scale=float(transform_right["scale"]),
            rotation=float(transform_right["rotation"]),
            translation_x=float(transform_right["translation_x"]),
            translation_y=float(transform_right["translation_y"]),
        )
        left_canonical = _estimate_canonical_transform(mask_a)
        right_canonical = _estimate_canonical_transform(mask_b)
        descriptor_mask_a = left_canonical["warped_mask"]
        descriptor_mask_b = right_canonical["warped_mask"]
        common_mask_a = overlap_details["common_mask_a"]
        common_mask_b = overlap_details["common_mask_b"]

    if use_common_region_filter:
        filtered_a = _filter_minutiae_by_mask(normalized_a, common_mask_a)
        filtered_b = _filter_minutiae_by_mask(normalized_b, common_mask_b)
        active_mask_a = common_mask_a
        active_mask_b = common_mask_b
    else:
        filtered_a = normalized_a.reset_index(drop=True)
        filtered_b = normalized_b.reset_index(drop=True)
        active_mask_a = descriptor_mask_a
        active_mask_b = descriptor_mask_b

    details["left_overlap_minutiae_count"] = int(len(filtered_a))
    details["right_overlap_minutiae_count"] = int(len(filtered_b))
    if len(filtered_a) < MCC_OVERLAP_MIN_MINUTIAE or len(filtered_b) < MCC_OVERLAP_MIN_MINUTIAE:
        details["fallback_to_legacy"] = True
        details["fallback_reason"] = "too_few_overlap_minutiae"
        return legacy_score, legacy_sim_matrix, details

    descriptors_before_a = build_descriptors(raw_frame_a, validity_mask_path=mask_a, validity_mode="mask")
    descriptors_before_b = build_descriptors(raw_frame_b, validity_mask_path=mask_b, validity_mode="mask")
    descriptors_after_a = build_descriptors(
        filtered_a,
        validity_mask_path=active_mask_a,
        validity_mode="mask",
    )
    descriptors_after_b = build_descriptors(
        filtered_b,
        validity_mask_path=active_mask_b,
        validity_mode="mask",
    )
    details["left_descriptor_count_before"] = int(len(descriptors_before_a))
    details["right_descriptor_count_before"] = int(len(descriptors_before_b))
    details["left_descriptor_count_after"] = int(len(descriptors_after_a))
    details["right_descriptor_count_after"] = int(len(descriptors_after_b))
    if len(descriptors_after_a) == 0 or len(descriptors_after_b) == 0:
        details["fallback_to_legacy"] = True
        details["fallback_reason"] = "no_descriptors_after_normalization"
        return legacy_score, legacy_sim_matrix, details

    score, sim_matrix = match_descriptors(
        descriptors_after_a,
        descriptors_after_b,
        method=base_method,
        overlap_mode=overlap_mode,
    )
    n_pairs = _compute_n_pairs(len(descriptors_after_a), len(descriptors_after_b))
    if base_method == "LSA":
        selected_pairs = _select_lsa_pairs(sim_matrix, n_pairs)
    else:
        rel_pairs = _select_lsa_pairs(
            sim_matrix,
            min(len(descriptors_after_a), len(descriptors_after_b)),
        )
        relaxed, efficiency, relaxation_details = _relax_pairs_with_details(
            descriptors_after_a,
            descriptors_after_b,
            rel_pairs,
        )
        top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(rel_pairs))]
        selected_pairs = [rel_pairs[index] for index in top_indices]
        details["relaxed_top_scores"] = [float(relaxed[index]) for index in top_indices]
        details["relaxation_details"] = relaxation_details

    details["selected_pair_count"] = int(len(selected_pairs))
    details["selected_pairs"] = [
        {"row": int(row), "col": int(col), "score": float(pair_score)}
        for row, col, pair_score in selected_pairs
    ]
    details["selected_pair_scores"] = [float(pair_score) for _, _, pair_score in selected_pairs]
    return score, sim_matrix, details


def match_minutiae_csv_overlap_details(
    path_a: Path | pd.DataFrame,
    path_b: Path | pd.DataFrame,
    mask_path_a: Path | np.ndarray | None,
    mask_path_b: Path | np.ndarray | None,
    method: str,
    overlap_mode: str = "auto",
) -> tuple[float, np.ndarray, dict]:
    base_method = _base_method_for_overlap(method)
    base_score, base_sim_matrix = match_minutiae_csv(
        path_a if isinstance(path_a, Path) else path_a.copy(),
        path_b if isinstance(path_b, Path) else path_b.copy(),
        method=base_method,
        mask_path_a=mask_path_a if mask_path_a is None or isinstance(mask_path_a, Path) else mask_path_a.copy(),
        mask_path_b=mask_path_b if mask_path_b is None or isinstance(mask_path_b, Path) else mask_path_b.copy(),
        overlap_mode=overlap_mode,
    )

    details = {
        "fallback_to_base": False,
        "fallback_reason": None,
        "estimated_transform": None,
        "overlap_area": 0,
        "overlap_ratio": 0.0,
        "left_raw_minutiae_count": 0,
        "right_raw_minutiae_count": 0,
        "left_overlap_minutiae_count": 0,
        "right_overlap_minutiae_count": 0,
        "left_descriptor_count_before": 0,
        "right_descriptor_count_before": 0,
        "left_descriptor_count_after": 0,
        "right_descriptor_count_after": 0,
        "selected_pair_count": 0,
    }

    if mask_path_a is None or mask_path_b is None:
        details["fallback_to_base"] = True
        details["fallback_reason"] = "missing_masks"
        return base_score, base_sim_matrix, details

    mask_a = _load_validity_mask(mask_path_a)
    mask_b = _load_validity_mask(mask_path_b)
    if mask_a is None or mask_b is None:
        details["fallback_to_base"] = True
        details["fallback_reason"] = "missing_masks"
        return base_score, base_sim_matrix, details

    raw_frame_a = path_a.copy() if isinstance(path_a, pd.DataFrame) else pd.read_csv(path_a)
    raw_frame_b = path_b.copy() if isinstance(path_b, pd.DataFrame) else pd.read_csv(path_b)
    raw_frame_a = raw_frame_a.dropna(subset=["x", "y", "angle"])
    raw_frame_b = raw_frame_b.dropna(subset=["x", "y", "angle"])
    details["left_raw_minutiae_count"] = int(len(raw_frame_a))
    details["right_raw_minutiae_count"] = int(len(raw_frame_b))

    overlap_details = _estimate_overlap_region(mask_a, mask_b)
    details["estimated_transform"] = overlap_details["estimated_transform"]
    details["overlap_area"] = int(overlap_details["overlap_area"])
    details["overlap_ratio"] = float(overlap_details["overlap_ratio"])
    if overlap_details["fallback_reason"] is not None:
        details["fallback_to_base"] = True
        details["fallback_reason"] = overlap_details["fallback_reason"]
        return base_score, base_sim_matrix, details

    filtered_a = _filter_minutiae_by_mask(raw_frame_a, overlap_details["common_mask_a"])
    filtered_b = _filter_minutiae_by_mask(raw_frame_b, overlap_details["common_mask_b"])
    details["left_overlap_minutiae_count"] = int(len(filtered_a))
    details["right_overlap_minutiae_count"] = int(len(filtered_b))
    if len(filtered_a) < MCC_OVERLAP_MIN_MINUTIAE or len(filtered_b) < MCC_OVERLAP_MIN_MINUTIAE:
        details["fallback_to_base"] = True
        details["fallback_reason"] = "too_few_overlap_minutiae"
        return base_score, base_sim_matrix, details

    descriptors_before_a = build_descriptors(raw_frame_a, validity_mask_path=mask_a, validity_mode="mask")
    descriptors_before_b = build_descriptors(raw_frame_b, validity_mask_path=mask_b, validity_mode="mask")
    descriptors_after_a = build_descriptors(
        filtered_a,
        validity_mask_path=overlap_details["common_mask_a"],
        validity_mode="mask",
    )
    descriptors_after_b = build_descriptors(
        filtered_b,
        validity_mask_path=overlap_details["common_mask_b"],
        validity_mode="mask",
    )
    details["left_descriptor_count_before"] = int(len(descriptors_before_a))
    details["right_descriptor_count_before"] = int(len(descriptors_before_b))
    details["left_descriptor_count_after"] = int(len(descriptors_after_a))
    details["right_descriptor_count_after"] = int(len(descriptors_after_b))
    if len(descriptors_after_a) == 0 or len(descriptors_after_b) == 0:
        details["fallback_to_base"] = True
        details["fallback_reason"] = "no_descriptors_after_overlap_filter"
        return base_score, base_sim_matrix, details

    score, sim_matrix = match_descriptors(
        descriptors_after_a,
        descriptors_after_b,
        method=base_method,
        overlap_mode=overlap_mode,
    )
    n_pairs = _compute_n_pairs(len(descriptors_after_a), len(descriptors_after_b))
    if base_method == "LSA":
        selected_pairs = [pair for pair in _select_lsa_pairs(sim_matrix, n_pairs) if pair[2] > 0.0]
    else:
        rel_pairs = _select_lsa_pairs(sim_matrix, min(len(descriptors_after_a), len(descriptors_after_b)))
        relaxed, efficiency = _relax_pairs(descriptors_after_a, descriptors_after_b, rel_pairs)
        top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(rel_pairs))]
        selected_pairs = [rel_pairs[index] for index in top_indices if rel_pairs[index][2] > 0.0]
    details["selected_pair_count"] = int(len(selected_pairs))
    return score, sim_matrix, details


def _resolve_validity_mode(
    validity_mode: str,
    validity_mask: np.ndarray | None,
) -> str:
    if validity_mode == "auto":
        return "mask" if validity_mask is not None else "hull"
    if validity_mode not in {"mask", "hull", "hybrid"}:
        raise ValueError(f"unsupported MCC validity mode: {validity_mode}")
    if validity_mode in {"mask", "hybrid"} and validity_mask is None:
        raise ValueError(
            f"validity mode '{validity_mode}' requires a validity mask"
        )
    return validity_mode


def _cell_is_valid(
    dx: float,
    dy: float,
    x_cell: float,
    y_cell: float,
    radius: float,
    hull: np.ndarray | None,
    validity_mask: np.ndarray | None,
    validity_mode: str,
) -> bool:
    if (dx * dx + dy * dy) > (radius * radius):
        return False

    if validity_mode == "hull":
        return _point_in_enlarged_hull(hull, x_cell, y_cell, MCC_HULL_OFFSET)

    in_mask = _point_in_validity_mask(validity_mask, x_cell, y_cell)
    if validity_mode == "mask":
        return in_mask

    return in_mask or _point_in_enlarged_hull(
        hull, x_cell, y_cell, MCC_HULL_OFFSET
    )


def _required_neighbor_count(
    raw_minutiae_count: int,
    adaptive_neighbor_support: bool = True,
) -> int:
    if not adaptive_neighbor_support:
        return MCC_MIN_M
    if raw_minutiae_count <= MCC_SPARSE_MINUTIAE_COUNT:
        return 1
    return MCC_MIN_M


def _descriptor_to_vectors(descriptor: MCCCylinder) -> tuple[np.ndarray, np.ndarray]:
    n_cells = descriptor.ns * descriptor.ns * descriptor.nd
    values = np.zeros(n_cells, dtype=np.float32)
    validities = np.zeros(n_cells, dtype=np.uint8)

    for cell in descriptor.cells:
        index = (cell.k * descriptor.ns * descriptor.ns) + (cell.j * descriptor.ns) + cell.i
        values[index] = float(cell.contribution)
        validities[index] = 1 if cell.valid else 0

    return values, validities


def _flatten_descriptors(
    descriptors: list[MCCCylinder],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not descriptors:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    n_cells = descriptors[0].ns * descriptors[0].ns * descriptors[0].nd
    values = np.zeros((len(descriptors), n_cells), dtype=np.float32)
    validities = np.zeros((len(descriptors), n_cells), dtype=np.uint8)
    angles = np.zeros((len(descriptors),), dtype=np.float32)

    for index, descriptor in enumerate(descriptors):
        values[index], validities[index] = _descriptor_to_vectors(descriptor)
        angles[index] = float(descriptor.center_theta)

    return values, validities, angles


def build_descriptors(
    path: Path | pd.DataFrame,
    validity_mask_path: Path | np.ndarray | None = None,
    validity_mode: str = "auto",
    adaptive_neighbor_support: bool = True,
) -> list[MCCCylinder]:
    if isinstance(path, pd.DataFrame):
        minutiae_df = path.copy()
    else:
        minutiae_df = pd.read_csv(path)
    minutiae_df = minutiae_df.dropna(subset=["x", "y", "angle"])
    minutiae = minutiae_df.to_dict(orient="records")
    if not minutiae:
        return []

    radius = MCC_RADIUS
    angle_height = 2.0 * math.pi
    ns = MCC_NS
    nd = MCC_ND
    delta_s = (2.0 * radius) / ns
    delta_d = angle_height / nd
    sigma_s = MCC_SIGMA_S
    sigma_d = MCC_SIGMA_D
    min_valid_cells = math.ceil(0.75 * _max_valid_cells(ns, nd, delta_s, radius))
    hull = _build_convex_hull(minutiae)
    validity_mask = _load_validity_mask(validity_mask_path)
    overlap_context = _build_overlap_context(validity_mask)
    resolved_validity_mode = _resolve_validity_mode(validity_mode, validity_mask)
    required_neighbor_count = _required_neighbor_count(
        len(minutiae),
        adaptive_neighbor_support=adaptive_neighbor_support,
    )
    descriptors: list[MCCCylinder] = []
    center = (ns + 1) / 2.0

    for c_i, minutia in enumerate(minutiae):
        x_m = float(minutia["x"])
        y_m = float(minutia["y"])
        theta_m = float(minutia["angle"])
        cos_t = math.cos(theta_m)
        sin_t = math.sin(theta_m)

        contributing_minutiae = 0
        for t, neighbor in enumerate(minutiae):
            if t == c_i:
                continue
            x_t = float(neighbor["x"])
            y_t = float(neighbor["y"])
            if math.hypot(x_t - x_m, y_t - y_m) <= (radius + (3.0 * sigma_s)):
                contributing_minutiae += 1

        descriptor = MCCCylinder(
            center_index=c_i,
            center_x=x_m,
            center_y=y_m,
            center_theta=theta_m,
            radius=radius,
            angle_height=angle_height,
            ns=ns,
            nd=nd,
            delta_s=delta_s,
            delta_d=delta_d,
            sigma_s=sigma_s,
            sigma_d=sigma_d,
            cells=[],
            overlap_context=overlap_context,
        )

        valid_cell_count = 0
        for i in range(ns):
            for j in range(ns):
                dx = delta_s * ((i + 1) - center)
                dy = delta_s * ((j + 1) - center)
                x_cell = x_m + (dx * cos_t) - (dy * sin_t)
                y_cell = y_m + (dx * sin_t) + (dy * cos_t)
                is_valid = _cell_is_valid(
                    dx=dx,
                    dy=dy,
                    x_cell=x_cell,
                    y_cell=y_cell,
                    radius=radius,
                    hull=hull,
                    validity_mask=validity_mask,
                    validity_mode=resolved_validity_mode,
                )

                if is_valid:
                    valid_cell_count += nd

                for k in range(nd):
                    angle_center = -math.pi + ((k + 0.5) * delta_d)
                    contribution = 0.0

                    if is_valid:
                        accumulator = 0.0
                        for t, neighbor in enumerate(minutiae):
                            if t == c_i:
                                continue

                            x_t = float(neighbor["x"])
                            y_t = float(neighbor["y"])
                            theta_t = float(neighbor["angle"])
                            distance = math.hypot(x_t - x_cell, y_t - y_cell)
                            if distance > (3.0 * sigma_s):
                                continue

                            spatial = _gaussian(distance, sigma_s)
                            relative_angle = wrap_angle(theta_m - theta_t)
                            alpha = wrap_angle(angle_center - relative_angle)
                            directional = _gaussian_area(alpha, delta_d, sigma_d)
                            accumulator += spatial * directional

                        contribution = _sigmoid(accumulator, MCC_MU_PSI, MCC_TAU_PSI)

                    descriptor.cells.append(
                        MCCCell(
                            i=i,
                            j=j,
                            k=k,
                            x=x_cell,
                            y=y_cell,
                            angle_center=angle_center,
                            contribution=contribution,
                            valid=is_valid,
                        )
                    )

        if (
            valid_cell_count < min_valid_cells
            or contributing_minutiae < required_neighbor_count
        ):
            continue

        descriptors.append(descriptor)

    return descriptors


def _cylinder_similarity_from_vectors(
    value_a: np.ndarray,
    value_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    theta_a: float,
    theta_b: float,
    cylinder_a: MCCCylinder | None = None,
    cylinder_b: MCCCylinder | None = None,
    overlap_mode: str = "auto",
) -> float:
    max_matchable = _max_valid_cells(
        MCC_NS,
        MCC_ND,
        (2.0 * MCC_RADIUS) / MCC_NS,
        MCC_RADIUS,
    )
    min_matchable = math.ceil(0.60 * max_matchable)

    if abs(wrap_angle(theta_a - theta_b)) > MCC_DELTA_THETA:
        return 0.0

    matchable_mask = (valid_a == 1) & (valid_b == 1)
    if overlap_mode == "auto" and cylinder_a is not None and cylinder_b is not None:
        overlap_mask = _overlap_valid_cell_mask(cylinder_a, cylinder_b, valid_a, valid_b)
        if overlap_mask is not None:
            matchable_mask = matchable_mask & overlap_mask
    elif overlap_mode != "off":
        raise ValueError(f"unsupported overlap mode: {overlap_mode}")

    if int(np.count_nonzero(matchable_mask)) < min_matchable:
        return 0.0

    a_masked = value_a[matchable_mask]
    b_masked = value_b[matchable_mask]
    denominator = np.linalg.norm(a_masked + b_masked)
    if denominator == 0.0:
        return 0.0

    score = 1.0 - (np.linalg.norm(a_masked - b_masked) / denominator)
    return float(np.clip(score, 0.0, 1.0))


def cylinder_similarity(cylinder_a: MCCCylinder, cylinder_b: MCCCylinder) -> float:
    value_a, valid_a = _descriptor_to_vectors(cylinder_a)
    value_b, valid_b = _descriptor_to_vectors(cylinder_b)
    return _cylinder_similarity_from_vectors(
        value_a=value_a,
        value_b=value_b,
        valid_a=valid_a,
        valid_b=valid_b,
        theta_a=cylinder_a.center_theta,
        theta_b=cylinder_b.center_theta,
        cylinder_a=cylinder_a,
        cylinder_b=cylinder_b,
        overlap_mode="auto",
    )


def _compute_n_pairs(n_a: int, n_b: int) -> int:
    if n_a == 0 or n_b == 0:
        return 0
    scaled = _sigmoid(float(min(n_a, n_b)), MCC_MU_P, MCC_TAU_P)
    return int(round(MCC_MIN_NP + (scaled * (MCC_MAX_NP - MCC_MIN_NP))))


def _select_lss_pairs(sim_matrix: np.ndarray, n_pairs: int) -> list[tuple[int, int, float]]:
    if sim_matrix.size == 0 or n_pairs <= 0:
        return []

    rows, cols = sim_matrix.shape
    total = rows * cols
    n_pairs = min(n_pairs, total)
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1][:n_pairs]
    return [
        (int(index // cols), int(index % cols), float(sim_matrix.flat[index]))
        for index in flat_indices
    ]


class _FlowEdge:
    __slots__ = ("to", "rev", "capacity", "cost")

    def __init__(self, to: int, rev: int, capacity: int, cost: float):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: float,
) -> None:
    graph[source].append(_FlowEdge(target, len(graph[target]), capacity, cost))
    graph[target].append(_FlowEdge(source, len(graph[source]) - 1, 0, -cost))


def _select_lsa_pairs(sim_matrix: np.ndarray, n_pairs: int) -> list[tuple[int, int, float]]:
    rows, cols = sim_matrix.shape
    if rows == 0 or cols == 0 or n_pairs <= 0:
        return []

    source = 0
    row_offset = 1
    col_offset = row_offset + rows
    sink = col_offset + cols
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    for row in range(rows):
        _add_flow_edge(graph, source, row_offset + row, 1, 0.0)
    for row in range(rows):
        for col in range(cols):
            _add_flow_edge(graph, row_offset + row, col_offset + col, 1, -float(sim_matrix[row, col]))
    for col in range(cols):
        _add_flow_edge(graph, col_offset + col, sink, 1, 0.0)

    target_flow = min(n_pairs, rows, cols)
    flow = 0
    while flow < target_flow:
        distances = [float("inf")] * len(graph)
        in_queue = [False] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)

        distances[source] = 0.0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                next_cost = distances[node] + edge.cost
                if next_cost + 1e-12 < distances[edge.to]:
                    distances[edge.to] = next_cost
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    if not in_queue[edge.to]:
                        queue.append(edge.to)
                        in_queue[edge.to] = True

        if previous_node[sink] == -1:
            break

        node = sink
        while node != source:
            parent = previous_node[node]
            edge = graph[parent][previous_edge[node]]
            edge.capacity -= 1
            graph[node][edge.rev].capacity += 1
            node = parent

        flow += 1

    pairs: list[tuple[int, int, float]] = []
    for row in range(rows):
        for edge in graph[row_offset + row]:
            if col_offset <= edge.to < sink and edge.capacity == 0:
                col = edge.to - col_offset
                pairs.append((row, col, float(sim_matrix[row, col])))

    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs[:target_flow]


def _pair_distance(a: MCCCylinder, b: MCCCylinder) -> float:
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


def _pair_direction_difference(a: MCCCylinder, b: MCCCylinder) -> float:
    return wrap_angle(a.center_theta - b.center_theta)


def _pair_radial_angle(a: MCCCylinder, b: MCCCylinder) -> float:
    return wrap_angle(
        a.center_theta - math.atan2(a.center_y - b.center_y, b.center_x - a.center_x)
    )


def _pair_compatibility(
    a_t: MCCCylinder,
    a_k: MCCCylinder,
    b_t: MCCCylinder,
    b_k: MCCCylinder,
) -> float:
    d1 = abs(_pair_distance(a_t, a_k) - _pair_distance(b_t, b_k))
    d2 = abs(
        wrap_angle(
            _pair_direction_difference(a_t, a_k)
            - _pair_direction_difference(b_t, b_k)
        )
    )
    d3 = abs(
        wrap_angle(_pair_radial_angle(a_t, a_k) - _pair_radial_angle(b_t, b_k))
    )

    compatibility = 1.0
    for value, (mu, tau) in zip((d1, d2, d3), MCC_RHO_PARAMS):
        compatibility *= _sigmoid(value, mu, tau)
    return compatibility


def _relax_pairs_with_details(
    descriptors_a: list[MCCCylinder],
    descriptors_b: list[MCCCylinder],
    pairs: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not pairs:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, {
            "compatibility_summary": {
                "min": 0.0,
                "median": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "nonzero_ratio": 0.0,
            },
            "iteration_summaries": [],
            "support_summaries": [],
        }

    relaxed = np.array([pair[2] for pair in pairs], dtype=np.float32)
    initial = relaxed.copy()
    compatibility_matrix = np.zeros((len(pairs), len(pairs)), dtype=np.float32)

    for t, (row_t, col_t, _) in enumerate(pairs):
        for k, (row_k, col_k, _) in enumerate(pairs):
            if k == t:
                continue
            compatibility_matrix[t, k] = _pair_compatibility(
                descriptors_a[row_t],
                descriptors_a[row_k],
                descriptors_b[col_t],
                descriptors_b[col_k],
            )

    compatibility_values = compatibility_matrix[compatibility_matrix > 0]
    if compatibility_values.size == 0:
        compatibility_summary = {
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "nonzero_ratio": 0.0,
        }
    else:
        compatibility_summary = {
            "min": float(np.min(compatibility_values)),
            "median": float(np.median(compatibility_values)),
            "max": float(np.max(compatibility_values)),
            "mean": float(np.mean(compatibility_values)),
            "nonzero_ratio": float(
                np.count_nonzero(compatibility_matrix > 0) / max(1, compatibility_matrix.size - len(pairs))
            ),
        }

    if len(pairs) == 1:
        return relaxed, np.ones((1,), dtype=np.float32), {
            "compatibility_summary": compatibility_summary,
            "iteration_summaries": [
                {
                    "iteration": 0,
                    "min": float(relaxed[0]),
                    "median": float(relaxed[0]),
                    "max": float(relaxed[0]),
                    "mean": float(relaxed[0]),
                }
            ],
            "support_summaries": [
                {"iteration": 0, "support_mean": 1.0, "support_median": 1.0}
            ],
        }

    iteration_summaries: list[dict] = []
    support_summaries: list[dict] = []

    for iteration in range(MCC_NREL):
        previous = relaxed.copy()
        support_values = []
        for t, (row_t, col_t, _) in enumerate(pairs):
            weights = compatibility_matrix[t]
            weight_sum = float(np.sum(weights))
            support_values.append(weight_sum)
            if weight_sum > 1e-12:
                compatible_average = float(np.dot(weights, previous) / weight_sum)
            else:
                compatible_average = float(previous[t])

            relaxed[t] = (MCC_W_R * previous[t]) + (
                (1.0 - MCC_W_R) * compatible_average
            )

        iteration_summaries.append(
            {
                "iteration": iteration + 1,
                "min": float(np.min(relaxed)),
                "median": float(np.median(relaxed)),
                "max": float(np.max(relaxed)),
                "mean": float(np.mean(relaxed)),
            }
        )
        support_summaries.append(
            {
                "iteration": iteration + 1,
                "support_mean": float(np.mean(support_values)),
                "support_median": float(np.median(support_values)),
                "support_min": float(np.min(support_values)),
                "support_max": float(np.max(support_values)),
            }
        )

    efficiency = np.zeros_like(relaxed)
    nonzero = initial > 0
    efficiency[nonzero] = relaxed[nonzero] / initial[nonzero]
    return relaxed, efficiency, {
        "compatibility_summary": compatibility_summary,
        "iteration_summaries": iteration_summaries,
        "support_summaries": support_summaries,
    }


def _relax_pairs(
    descriptors_a: list[MCCCylinder],
    descriptors_b: list[MCCCylinder],
    pairs: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    relaxed, efficiency, _ = _relax_pairs_with_details(
        descriptors_a,
        descriptors_b,
        pairs,
    )
    return relaxed, efficiency


def match_descriptors(
    descriptors_a: list[MCCCylinder],
    descriptors_b: list[MCCCylinder],
    method: str = "LSA-R",
    overlap_mode: str = "auto",
) -> tuple[float, np.ndarray]:
    values_a, validities_a, angles_a = _flatten_descriptors(descriptors_a)
    values_b, validities_b, angles_b = _flatten_descriptors(descriptors_b)
    sim_matrix = np.zeros((len(descriptors_a), len(descriptors_b)), dtype=np.float32)

    for i in range(len(descriptors_a)):
        for j in range(len(descriptors_b)):
            sim_matrix[i, j] = _cylinder_similarity_from_vectors(
                value_a=values_a[i],
                value_b=values_b[j],
                valid_a=validities_a[i],
                valid_b=validities_b[j],
                theta_a=float(angles_a[i]),
                theta_b=float(angles_b[j]),
                cylinder_a=descriptors_a[i],
                cylinder_b=descriptors_b[j],
                overlap_mode=overlap_mode,
            )

    n_pairs = _compute_n_pairs(len(descriptors_a), len(descriptors_b))
    if n_pairs == 0:
        return 0.0, sim_matrix

    normalized_method = method.upper()
    if normalized_method in {
        "LSA-OVERLAP",
        "LSA-R-OVERLAP",
        "LSA-CANONICAL",
        "LSA-R-CANONICAL",
        "LSA-CANONICAL-OVERLAP",
        "LSA-R-CANONICAL-OVERLAP",
        "LSA-CENTROID",
        "LSA-R-CENTROID",
    }:
        raise ValueError(
            f"{normalized_method} requires minutiae CSV inputs, not prebuilt descriptors"
        )
    if normalized_method == "LSS":
        pairs = _select_lss_pairs(sim_matrix, n_pairs)
        return (
            float(np.mean([pair[2] for pair in pairs])) if pairs else 0.0,
            sim_matrix,
        )

    if normalized_method == "LSA":
        pairs = _select_lsa_pairs(sim_matrix, n_pairs)
        return (
            float(np.mean([pair[2] for pair in pairs])) if pairs else 0.0,
            sim_matrix,
        )

    n_rel_pairs = min(len(descriptors_a), len(descriptors_b))
    if normalized_method == "LSS-R":
        pairs = _select_lss_pairs(sim_matrix, n_rel_pairs)
    elif normalized_method == "LSA-R":
        pairs = _select_lsa_pairs(sim_matrix, n_rel_pairs)
    else:
        raise ValueError(f"unsupported MCC matching method: {method}")

    relaxed, efficiency = _relax_pairs(descriptors_a, descriptors_b, pairs)
    if relaxed.size == 0:
        return 0.0, sim_matrix

    top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(pairs))]
    score = float(np.mean(relaxed[top_indices])) if len(top_indices) > 0 else 0.0
    return score, sim_matrix


def match_minutiae_csv(
    path_a: Path,
    path_b: Path,
    method: str = "LSA-R",
    mask_path_a: Path | None = None,
    mask_path_b: Path | None = None,
    orientation_path_a: Path | np.ndarray | None = None,
    orientation_path_b: Path | np.ndarray | None = None,
    ridge_period_path_a: Path | np.ndarray | None = None,
    ridge_period_path_b: Path | np.ndarray | None = None,
    overlap_mode: str = "auto",
) -> tuple[float, np.ndarray]:
    normalized_method = method.upper()
    if normalized_method in {"LSA-CENTROID", "LSA-R-CENTROID"}:
        score, sim_matrix, _ = match_minutiae_csv_centroid_details(
            path_a,
            path_b,
            method=method,
            orientation_path_a=orientation_path_a,
            orientation_path_b=orientation_path_b,
            ridge_period_path_a=ridge_period_path_a,
            ridge_period_path_b=ridge_period_path_b,
        )
        return score, sim_matrix
    if normalized_method in {"LSA-OVERLAP", "LSA-R-OVERLAP"}:
        score, sim_matrix, _ = match_minutiae_csv_overlap_details(
            path_a,
            path_b,
            mask_path_a,
            mask_path_b,
            method=method,
            overlap_mode=overlap_mode,
        )
        return score, sim_matrix
    if normalized_method in {
        "LSA-CANONICAL",
        "LSA-R-CANONICAL",
        "LSA-CANONICAL-OVERLAP",
        "LSA-R-CANONICAL-OVERLAP",
    }:
        canonical_use_overlap = _canonical_method_uses_overlap(method)
        score, sim_matrix, _ = match_minutiae_csv_pose_normalized_details(
            path_a,
            path_b,
            mask_path_a,
            mask_path_b,
            method=_base_method_for_canonical(method),
            strategy="canonical",
            use_common_region_filter=canonical_use_overlap,
            overlap_mode=overlap_mode if canonical_use_overlap else "off",
        )
        return score, sim_matrix
    if normalized_method in {"LSA", "LSA-R"} and mask_path_a is not None and mask_path_b is not None:
        score, sim_matrix, _ = match_minutiae_csv_pose_normalized_details(
            path_a,
            path_b,
            mask_path_a,
            mask_path_b,
            method=method,
            overlap_mode=overlap_mode,
        )
        return score, sim_matrix
    return match_minutiae_csv_legacy(
        path_a,
        path_b,
        method=method,
        mask_path_a=mask_path_a,
        mask_path_b=mask_path_b,
        overlap_mode=overlap_mode,
    )


def _extract_minutiae_csv_from_image(
    image_path: Path,
    workspace_dir: Path,
    model_paths: tuple[Path, Path, Path, Path],
) -> tuple[Path, Path]:
    image_root = workspace_dir / image_path.stem
    image_root.mkdir(parents=True, exist_ok=True)

    nobg_path = image_root / "nobg.png"
    cropped_path = image_root / "cropped.png"
    cropped_mask_path = image_root / "cropped_mask.png"
    normalised_path = image_root / "normalised.png"
    enhanced_path = image_root / "enhanced.png"
    minutiae_json_path = image_root / "minutiae.json"
    minutiae_csv_path = image_root / "minutiae.csv"
    core_csv_path = image_root / "core.csv"

    full_bgr = load_bgr_image(image_path)
    coarse_mask = rembg_mask_from_bgr(full_bgr)
    validate_foreground_area(coarse_mask, minimum_ratio=0.03)

    crop_bbox = compute_distal_crop_bbox(coarse_mask)
    cropped_bgr = crop_image(full_bgr, crop_bbox)

    cropped_mask = rembg_mask_from_bgr(cropped_bgr)
    validate_foreground_area(cropped_mask, minimum_ratio=0.08)
    cropped_rgba = compose_rgba(cropped_bgr, cropped_mask)

    save_png(cropped_path, cropped_bgr)
    save_png(nobg_path, cropped_rgba)
    save_png(cropped_mask_path, cropped_mask)
    normalise_brightness(nobg_path, normalised_path)
    filter(normalised_path, enhanced_path)

    if not enhanced_path.exists() or enhanced_path.stat().st_size == 0:
        raise RuntimeError(f"enhanced image was not created: {enhanced_path}")

    extract_minutiae_with_fingerflow(
        image_path.resolve(),
        enhanced_path.resolve(),
        model_paths,
        minutiae_json_path.resolve(),
        minutiae_csv_path.resolve(),
        core_csv_path.resolve(),
    )
    return minutiae_csv_path, cropped_mask_path


def match_fingerprint_images(
    image_a: Path,
    image_b: Path,
    method: str = "LSA-R",
    fingerflow_model_dir: Path = DEFAULT_FINGERFLOW_MODEL_DIR,
) -> tuple[float, np.ndarray]:
    image_a = Path(image_a)
    image_b = Path(image_b)
    if not image_a.exists():
        raise FileNotFoundError(f"image not found: {image_a}")
    if not image_b.exists():
        raise FileNotFoundError(f"image not found: {image_b}")

    score, sim_matrix, _ = _match_fingerprint_images_with_workspace(
        image_a=image_a,
        image_b=image_b,
        method=method,
        fingerflow_model_dir=fingerflow_model_dir,
    )
    return score, sim_matrix


def _match_fingerprint_images_with_workspace(
    image_a: Path,
    image_b: Path,
    method: str = "LSA-R",
    fingerflow_model_dir: Path = DEFAULT_FINGERFLOW_MODEL_DIR,
) -> tuple[float, np.ndarray, Path]:
    image_a = Path(image_a)
    image_b = Path(image_b)
    if not image_a.exists():
        raise FileNotFoundError(f"image not found: {image_a}")
    if not image_b.exists():
        raise FileNotFoundError(f"image not found: {image_b}")

    model_paths = ensure_fingerflow_models(fingerflow_model_dir)
    run_dir = MATCH_OUTPUTS_DIR / f"match_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_a, mask_a = _extract_minutiae_csv_from_image(image_a, run_dir / "a", model_paths)
    csv_b, mask_b = _extract_minutiae_csv_from_image(image_b, run_dir / "b", model_paths)
    if method.upper() in {
        "LSA",
        "LSA-R",
        "LSA-OVERLAP",
        "LSA-R-OVERLAP",
        "LSA-CANONICAL",
        "LSA-R-CANONICAL",
        "LSA-CANONICAL-OVERLAP",
        "LSA-R-CANONICAL-OVERLAP",
        "LSA-CENTROID",
        "LSA-R-CENTROID",
    }:
        score, sim_matrix = match_minutiae_csv(
            csv_a,
            csv_b,
            method=method,
            mask_path_a=mask_a,
            mask_path_b=mask_b,
            overlap_mode="auto",
        )
        return score, sim_matrix, run_dir
    descriptors_a = build_descriptors(
        csv_a,
        validity_mask_path=mask_a,
        validity_mode="mask",
    )
    descriptors_b = build_descriptors(
        csv_b,
        validity_mask_path=mask_b,
        validity_mode="mask",
    )
    score, sim_matrix = match_descriptors(descriptors_a, descriptors_b, method=method)
    return score, sim_matrix, run_dir
    


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    _print_runtime_context()
    try:
        if args.command == "match":
            gpu_check_start = time.perf_counter()
            _ensure_tensorflow_gpu_available()
            _record_timing("gpu_check", gpu_check_start)
            score, sim_matrix, run_dir = _match_fingerprint_images_with_workspace(
                args.image_a.resolve(),
                args.image_b.resolve(),
                method=args.method,
                fingerflow_model_dir=args.fingerflow_model_dir.resolve(),
            )
            _record_timing("total", start)
            print(f"MCC match score ({args.method}): {score:.6f}")
            print(f"Similarity matrix shape: {sim_matrix.shape}")
            print(f"Saved match outputs: {run_dir}")
            print(f"Stage timings: {_format_timing_summary()}")
            return 0

        gpu_check_start = time.perf_counter()
        _ensure_tensorflow_gpu_available()
        _record_timing("gpu_check", gpu_check_start)
        try:
            import tensorflow as tf

            print(f"TensorFlow GPU devices visible: {[gpu.name for gpu in tf.config.list_physical_devices('GPU')]}")
        except Exception:
            pass
        nobg_path, cropped_path = process_image(args.input_image)
        normalised_path = REPO_ROOT / "normalised_cropped.png"
        enhanced_path = DEFAULT_ENHANCED_IMAGE
        normalise_brightness(nobg_path, normalised_path)
        filter(normalised_path, enhanced_path)
        if not enhanced_path.exists() or enhanced_path.stat().st_size == 0:
            raise RuntimeError(f"enhanced image was not created: {enhanced_path}")

        model_start = time.perf_counter()
        model_paths = ensure_fingerflow_models(args.fingerflow_model_dir)
        _record_timing("model_load", model_start)
        minutiae_count, core_count = extract_minutiae_with_fingerflow(
            args.input_image.resolve(),
            enhanced_path.resolve(),
            model_paths,
            args.minutiae_json.resolve(),
            args.minutiae_csv.resolve(),
            args.core_csv.resolve(),
        )
        _record_timing("fingerflow_total", model_start)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"Saved background-removed image: {nobg_path}")
    print(f"Saved cropped fingerprint image: {cropped_path}")
    print(f"Saved enhanced fingerprint image: {enhanced_path.resolve()}")
    print(f"FingerFlow model cache directory: {args.fingerflow_model_dir.resolve()}")
    print(f"Saved FingerFlow minutiae JSON: {args.minutiae_json.resolve()}")
    print(f"Saved FingerFlow minutiae CSV: {args.minutiae_csv.resolve()}")
    print(f"Saved FingerFlow core CSV: {args.core_csv.resolve()}")
    print(f"FingerFlow extracted minutiae: {minutiae_count}, cores: {core_count}")
    _record_timing("total", start)
    print(f"Stage timings: {_format_timing_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
