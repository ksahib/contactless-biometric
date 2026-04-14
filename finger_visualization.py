from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import sysconfig
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from center_unwarping import run_center_unwarping


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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


IMAGE_CANDIDATES = (
    "preprocess_pose_normalized.png",
    "preprocess_normalized.png",
    "masked_image.png",
    "preprocessed_input.png",
)
MASK_CANDIDATES = (
    "preprocess_pose_mask.png",
    "preprocess_mask.png",
    "mask.png",
)
DEFAULT_REPORT_NAME = "pointcloud_report.json"
DEFAULT_PIPELINE = "center_depth_completion"
LEGACY_PIPELINE = "center_dense_surface"
UNWARP_PIPELINE = "center_unwarp"
NORMALIZATION_CHOICES = ("none", "min", "p01", "p05", "median")
GRADIENT_METHODS = ("masked_central_difference",)
COMPLETION_METHODS = ("anchored_poisson_relaxation",)
CENTER_POINT_MODES = ("min_gradient",)
RESAMPLING_METHODS = ("bilinear_forward",)


class ResolvedInputs(NamedTuple):
    center_image: Path
    center_mask: Path
    center_depth: Path | None


class RowGeometry(NamedTuple):
    rows: np.ndarray
    lefts: np.ndarray
    rights: np.ndarray
    centers: np.ndarray
    half_widths: np.ndarray


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(x / float(sigma)))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or values.size == 0:
        return values.astype(np.float32, copy=True)
    kernel = gaussian_kernel1d(sigma)
    pad = kernel.size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def convolve_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(arr, dtype=np.float32)
        for col in range(arr.shape[1]):
            out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
        return out
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)
    for row in range(arr.shape[0]):
        out[row, :] = np.convolve(padded[row, :], kernel, mode="valid")
    return out


def normalized_gaussian_fill(
    values: np.ndarray,
    support_mask: np.ndarray,
    target_mask: np.ndarray,
    sigma: float,
    *,
    preserve_support: bool,
    fill_value: float,
    outside_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    support_mask = support_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    base = np.full(values.shape, outside_value, dtype=np.float32)
    if not np.any(target_mask):
        return base, np.zeros_like(target_mask, dtype=bool)

    if sigma <= 0 or not np.any(support_mask):
        base[target_mask] = float(fill_value)
        if preserve_support and np.any(support_mask):
            base[support_mask] = values[support_mask].astype(np.float32)
        return base, support_mask & target_mask

    kernel = gaussian_kernel1d(float(sigma))
    support_f = support_mask.astype(np.float32)
    weighted = np.where(support_mask, values.astype(np.float32), 0.0)
    smooth_values = convolve_axis(convolve_axis(weighted, kernel, axis=1), kernel, axis=0)
    smooth_weights = convolve_axis(convolve_axis(support_f, kernel, axis=1), kernel, axis=0)

    stable = smooth_weights > 1e-6
    base[target_mask] = float(fill_value)
    stable_target = target_mask & stable
    base[stable_target] = (smooth_values[stable_target] / smooth_weights[stable_target]).astype(np.float32)
    if preserve_support:
        base[support_mask] = values[support_mask].astype(np.float32)
    return base, stable_target


def masked_gaussian_smooth(values: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    out, stable = normalized_gaussian_fill(
        values=values,
        support_mask=valid_mask,
        target_mask=valid_mask,
        sigma=sigma,
        preserve_support=False,
        fill_value=0.0,
        outside_value=np.nan,
    )
    fallback = np.where(valid_mask, values.astype(np.float32), np.nan).astype(np.float32)
    replace = valid_mask & stable
    fallback[replace] = out[replace]
    return fallback.astype(np.float32)


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        return np.asarray(image, dtype=np.uint8) > 0


def load_depth(path: Path) -> np.ndarray:
    try:
        depth = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Depth array could not be loaded as a numeric .npy file: {path}") from exc
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError(f"Depth array must be numeric: {path}")
    return np.asarray(depth, dtype=np.float32)


def find_first_existing(base_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def resolve_inputs(args: argparse.Namespace) -> tuple[ResolvedInputs, list[str]]:
    warnings: list[str] = []

    center_image = Path(args.center_image) if args.center_image else None
    center_mask = Path(args.center_mask) if args.center_mask else None
    center_depth = Path(args.center_depth) if args.center_depth else None

    if args.front_sample_dir:
        sample_dir = Path(args.front_sample_dir)
        if not sample_dir.exists():
            raise FileNotFoundError(f"Front sample directory does not exist: {sample_dir}")
        if center_image is None:
            center_image = find_first_existing(sample_dir, IMAGE_CANDIDATES)
        if center_mask is None:
            center_mask = find_first_existing(sample_dir, MASK_CANDIDATES)

    if args.reconstruction_dir and center_depth is None:
        reconstruction_dir = Path(args.reconstruction_dir)
        if not reconstruction_dir.exists():
            raise FileNotFoundError(f"Reconstruction directory does not exist: {reconstruction_dir}")
        depth_candidate = reconstruction_dir / "depth_front.npy"
        if depth_candidate.exists():
            center_depth = depth_candidate

    if center_image is None:
        raise ValueError("Center image was not resolved. Provide --center-image or --front-sample-dir.")
    if center_mask is None:
        raise ValueError("Center mask was not resolved. Provide --center-mask or --front-sample-dir.")

    for label, path in (
        ("center image", center_image),
        ("center mask", center_mask),
        ("center depth", center_depth),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} does not exist: {path}")

    ignored_inputs = {
        "left_image": args.left_image,
        "left_mask": args.left_mask,
        "left_depth": args.left_depth,
        "left_sample_dir": args.left_sample_dir,
        "right_image": args.right_image,
        "right_mask": args.right_mask,
        "right_depth": args.right_depth,
        "right_sample_dir": args.right_sample_dir,
    }
    ignored_keys = sorted(key for key, value in ignored_inputs.items() if value)
    if ignored_keys:
        warnings.append(
            "Left/right inputs were provided but ignored by the center-view pipelines: "
            + ", ".join(ignored_keys)
        )

    return ResolvedInputs(center_image=center_image, center_mask=center_mask, center_depth=center_depth), warnings


def compute_sparse_valid_mask(mask: np.ndarray, depth: np.ndarray | None, depth_epsilon: float) -> np.ndarray:
    if depth is None:
        return np.zeros_like(mask, dtype=bool)
    return mask & np.isfinite(depth) & (depth > float(depth_epsilon))


def summarize_depth(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "max": None,
            "span": None,
        }
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "span": float(np.max(values) - np.min(values)),
    }


def normalize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, float, dict[str, float | None]]:
    valid_values = depth[valid_mask].astype(np.float32)
    if valid_values.size == 0:
        raise ValueError("No valid recovered depth exists after masking.")

    if mode == "none":
        reference = 0.0
    elif mode == "min":
        reference = float(np.min(valid_values))
    elif mode == "p01":
        reference = float(np.percentile(valid_values, 1))
    elif mode == "p05":
        reference = float(np.percentile(valid_values, 5))
    elif mode == "median":
        reference = float(np.median(valid_values))
    else:
        raise ValueError(f"Unsupported depth normalization mode: {mode}")

    normalized = np.full(depth.shape, np.nan, dtype=np.float32)
    normalized[valid_mask] = (valid_values - float(reference)).astype(np.float32)
    return normalized, float(reference), summarize_depth(normalized[valid_mask])


def shift2d(values: np.ndarray, axis: int, step: int) -> np.ndarray:
    out = np.zeros_like(values)
    if axis == 1:
        if step > 0:
            out[:, step:] = values[:, :-step]
        elif step < 0:
            out[:, :step] = values[:, -step:]
        else:
            out[:] = values
    else:
        if step > 0:
            out[step:, :] = values[:-step, :]
        elif step < 0:
            out[:step, :] = values[-step:, :]
        else:
            out[:] = values
    return out


def compute_axis_gradient(depth: np.ndarray, valid_mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    depth = depth.astype(np.float32)
    valid_mask = valid_mask.astype(bool)

    if axis == 1:
        neg_valid = shift2d(valid_mask, axis=1, step=1)
        pos_valid = shift2d(valid_mask, axis=1, step=-1)
        neg_values = shift2d(depth, axis=1, step=1)
        pos_values = shift2d(depth, axis=1, step=-1)
    else:
        neg_valid = shift2d(valid_mask, axis=0, step=1)
        pos_valid = shift2d(valid_mask, axis=0, step=-1)
        neg_values = shift2d(depth, axis=0, step=1)
        pos_values = shift2d(depth, axis=0, step=-1)

    grad = np.full(depth.shape, np.nan, dtype=np.float32)
    central = valid_mask & neg_valid & pos_valid
    forward = valid_mask & (~neg_valid) & pos_valid
    backward = valid_mask & neg_valid & (~pos_valid)

    grad[central] = 0.5 * (pos_values[central] - neg_values[central])
    grad[forward] = pos_values[forward] - depth[forward]
    grad[backward] = depth[backward] - neg_values[backward]
    support = central | forward | backward
    return grad, support


def compute_masked_gradients(
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad_x, support_x = compute_axis_gradient(depth=depth, valid_mask=valid_mask, axis=1)
    grad_y, support_y = compute_axis_gradient(depth=depth, valid_mask=valid_mask, axis=0)
    return grad_x, grad_y, support_x, support_y


def compute_divergence(grad_x: np.ndarray, grad_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grad_x = np.where(mask, np.nan_to_num(grad_x, nan=0.0), 0.0).astype(np.float32)
    grad_y = np.where(mask, np.nan_to_num(grad_y, nan=0.0), 0.0).astype(np.float32)

    dgrad_x = np.zeros_like(grad_x, dtype=np.float32)
    dgrad_y = np.zeros_like(grad_y, dtype=np.float32)

    dgrad_x[:, 1:-1] = 0.5 * (grad_x[:, 2:] - grad_x[:, :-2])
    dgrad_x[:, 0] = grad_x[:, 1] - grad_x[:, 0]
    dgrad_x[:, -1] = grad_x[:, -1] - grad_x[:, -2]

    dgrad_y[1:-1, :] = 0.5 * (grad_y[2:, :] - grad_y[:-2, :])
    dgrad_y[0, :] = grad_y[1, :] - grad_y[0, :]
    dgrad_y[-1, :] = grad_y[-1, :] - grad_y[-2, :]

    divergence = dgrad_x + dgrad_y
    divergence[~mask] = 0.0
    return divergence.astype(np.float32)


def build_neighbor_sum_and_count(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighbor_sum = np.zeros_like(values, dtype=np.float32)
    neighbor_count = np.zeros_like(values, dtype=np.float32)

    horizontal_pairs = mask[:, 1:] & mask[:, :-1]
    vertical_pairs = mask[1:, :] & mask[:-1, :]

    left_values = np.where(horizontal_pairs, values[:, :-1], 0.0)
    right_values = np.where(horizontal_pairs, values[:, 1:], 0.0)
    up_values = np.where(vertical_pairs, values[:-1, :], 0.0)
    down_values = np.where(vertical_pairs, values[1:, :], 0.0)

    neighbor_sum[:, 1:] += left_values
    neighbor_sum[:, :-1] += right_values
    neighbor_sum[1:, :] += up_values
    neighbor_sum[:-1, :] += down_values

    neighbor_count[:, 1:] += horizontal_pairs.astype(np.float32)
    neighbor_count[:, :-1] += horizontal_pairs.astype(np.float32)
    neighbor_count[1:, :] += vertical_pairs.astype(np.float32)
    neighbor_count[:-1, :] += vertical_pairs.astype(np.float32)
    return neighbor_sum.astype(np.float32), neighbor_count.astype(np.float32)


def anchored_poisson_relaxation(
    initial_depth: np.ndarray,
    anchor_depth: np.ndarray,
    anchor_mask: np.ndarray,
    mask: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    mask = mask.astype(bool)
    anchor_mask = anchor_mask.astype(bool)
    free_mask = mask & (~anchor_mask)

    depth = np.where(mask, initial_depth.astype(np.float32), np.nan).astype(np.float32)
    depth[anchor_mask] = anchor_depth[anchor_mask].astype(np.float32)
    if np.any(mask & ~np.isfinite(depth)):
        fill_value = float(np.mean(anchor_depth[anchor_mask])) if np.any(anchor_mask) else 0.0
        depth[mask & ~np.isfinite(depth)] = fill_value

    divergence = compute_divergence(grad_x=grad_x, grad_y=grad_y, mask=mask)
    residual = 0.0
    iteration_count = 0
    relaxation = 0.85
    smooth_blend = 0.25

    for iteration in range(max(1, int(max_iterations))):
        iteration_count = iteration + 1
        neighbor_sum, neighbor_count = build_neighbor_sum_and_count(values=depth, mask=mask)
        neighbor_count_safe = np.where(neighbor_count > 0, neighbor_count, 1.0)
        neighbor_avg = neighbor_sum / neighbor_count_safe
        poisson_estimate = (neighbor_sum - divergence) / neighbor_count_safe
        proposal = (1.0 - smooth_blend) * poisson_estimate + smooth_blend * neighbor_avg

        next_depth = depth.copy()
        updatable = free_mask & (neighbor_count > 0)
        next_depth[updatable] = (
            (1.0 - relaxation) * depth[updatable] + relaxation * proposal[updatable]
        ).astype(np.float32)
        next_depth[anchor_mask] = anchor_depth[anchor_mask].astype(np.float32)

        residual = float(np.max(np.abs(next_depth[updatable] - depth[updatable]))) if np.any(updatable) else 0.0
        depth = next_depth
        if residual <= float(tolerance):
            break

    depth[anchor_mask] = anchor_depth[anchor_mask].astype(np.float32)
    depth[~mask] = np.nan
    return depth.astype(np.float32), {
        "iterations": int(iteration_count),
        "residual": float(residual),
    }


def dense_surface_points(image: np.ndarray, dense_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(dense_depth)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("Dense surface generation produced no finite surface points.")
    height, width = image.shape
    x = cols.astype(np.float32) - (width / 2.0)
    y = -(rows.astype(np.float32) - (height / 2.0))
    z = dense_depth[rows, cols].astype(np.float32)
    points = np.column_stack((x, y, z)).astype(np.float32)
    intensities = image[rows, cols].astype(np.float32)
    return points, intensities


def bounds_from_points(points: np.ndarray) -> dict[str, list[float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return {
        "x": [float(mins[0]), float(maxs[0])],
        "y": [float(mins[1]), float(maxs[1])],
        "z": [float(mins[2]), float(maxs[2])],
    }


def set_box_aspect(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    ax.set_box_aspect(spans.tolist())


def crop_to_mask(mask: np.ndarray) -> tuple[slice, slice]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Mask is empty.")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def render_surface_plot(
    output_path: Path,
    image: np.ndarray,
    dense_depth: np.ndarray,
    points_for_aspect: np.ndarray,
    title: str,
    plot_max_points: int,
    elev: float,
    azim: float,
) -> None:
    row_slice, col_slice = crop_to_mask(np.isfinite(dense_depth))
    z_crop = np.ma.masked_invalid(dense_depth[row_slice, col_slice])
    image_crop = image[row_slice, col_slice]
    height = dense_depth.shape[0]
    width = dense_depth.shape[1]
    row_indices = np.arange(row_slice.start, row_slice.stop, dtype=np.float32)
    col_indices = np.arange(col_slice.start, col_slice.stop, dtype=np.float32)
    x_grid = col_indices[None, :] - (width / 2.0)
    y_grid = -(row_indices[:, None] - (height / 2.0))
    x_grid = np.broadcast_to(x_grid, z_crop.shape)
    y_grid = np.broadcast_to(y_grid, z_crop.shape)
    facecolors = plt.cm.gray(np.clip(image_crop, 0.0, 1.0))
    if np.ma.isMaskedArray(z_crop):
        facecolors[np.ma.getmaskarray(z_crop)] = (0.0, 0.0, 0.0, 0.0)
    total = int(z_crop.shape[0] * z_crop.shape[1])
    stride = max(1, int(math.ceil(math.sqrt(total / max(plot_max_points, 1)))))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.plot_surface(
            x_grid,
            y_grid,
            z_crop,
            rstride=stride,
            cstride=stride,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
    except Exception:
        valid = np.isfinite(dense_depth)
        rows, cols = np.nonzero(valid)
        sample_stride = max(1, int(math.ceil(rows.size / max(plot_max_points, 1))))
        rows = rows[::sample_stride]
        cols = cols[::sample_stride]
        x = cols.astype(np.float32) - (width / 2.0)
        y = -(rows.astype(np.float32) - (height / 2.0))
        z = dense_depth[rows, cols]
        ax.scatter(x, y, z, c=image[rows, cols], cmap="gray", s=1.0, linewidths=0)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    set_box_aspect(ax, points_for_aspect)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_image(path: Path, array: np.ndarray, cmap: str = "gray", title: str | None = None) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plot_array = array.astype(np.float32)
    im = ax.imshow(plot_array, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=80, color="#555555", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_mask_image(path: Path, mask: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_side_by_side_depth(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
) -> None:
    finite_values = np.concatenate(
        [
            left[np.isfinite(left)].astype(np.float32),
            right[np.isfinite(right)].astype(np.float32),
        ]
    )
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, image, title in (
        (axes[0], left, left_title),
        (axes[1], right, right_title),
    ):
        im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.03)
    fig.subplots_adjust(left=0.03, right=0.92, top=0.90, bottom=0.05, wspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_normal_map(path: Path, normal_map: np.ndarray, mask: np.ndarray) -> None:
    rgb = np.zeros_like(normal_map, dtype=np.float32)
    rgb[mask] = 0.5 * (normal_map[mask] + 1.0)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(np.clip(rgb, 0.0, 1.0))
    ax.set_title("Center Surface Normal Map")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_center_point_debug(path: Path, image: np.ndarray, mask: np.ndarray, center_point: tuple[int, int]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.contour(mask.astype(np.float32), levels=[0.5], colors=["#2b8cbe"], linewidths=0.8)
    ax.scatter([center_point[0]], [center_point[1]], s=60, c="#d94841", marker="x", linewidths=2.0)
    ax.set_title("Center Unwarp Zero Point")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_unwarp_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    overlay_samples: dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.contour(mask.astype(np.float32), levels=[0.5], colors=["#31a354"], linewidths=0.8)
    src_x = overlay_samples["source_x"]
    src_y = overlay_samples["source_y"]
    tgt_x = overlay_samples["target_x"]
    tgt_y = overlay_samples["target_y"]
    if src_x.size:
        ax.quiver(
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            tgt_x.astype(np.float32) - src_x.astype(np.float32),
            tgt_y.astype(np.float32) - src_y.astype(np.float32),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#d94841",
            width=0.002,
        )
    ax.set_title("Center Unwarp Displacement Overlay")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def build_normal_map(dense_depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_valid = mask & np.isfinite(dense_depth)
    dense_input = np.where(dense_valid, dense_depth.astype(np.float32), 0.0)
    grad_x, grad_y, _, _ = compute_masked_gradients(depth=dense_input, valid_mask=dense_valid)
    grad_x = np.where(dense_valid, np.nan_to_num(grad_x, nan=0.0), 0.0).astype(np.float32)
    grad_y = np.where(dense_valid, np.nan_to_num(grad_y, nan=0.0), 0.0).astype(np.float32)

    normals = np.zeros(dense_depth.shape + (3,), dtype=np.float32)
    normals[..., 0] = -grad_x
    normals[..., 1] = -grad_y
    normals[..., 2] = 1.0
    magnitude = np.linalg.norm(normals, axis=2, keepdims=True)
    magnitude = np.maximum(magnitude, 1e-6)
    normals = normals / magnitude
    normals[~mask] = 0.0
    return normals.astype(np.float32), grad_x, grad_y


def prepare_center_depth_and_gradients(
    args: argparse.Namespace,
    mask: np.ndarray,
    depth: np.ndarray | None,
) -> dict[str, Any]:
    if depth is None:
        raise ValueError("Center pipeline requires a center recovered depth array.")

    valid_mask = compute_sparse_valid_mask(mask, depth, args.depth_epsilon)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        raise ValueError("Center pipeline requires at least one valid recovered depth pixel.")

    normalized_sparse_depth, normalization_reference, normalized_stats = normalize_depth(
        depth=depth,
        valid_mask=valid_mask,
        mode=args.depth_normalization,
    )
    prepared_depth = normalized_sparse_depth.astype(np.float32, copy=True)
    if args.smooth_depth:
        prepared_depth = masked_gaussian_smooth(
            values=normalized_sparse_depth,
            valid_mask=valid_mask,
            sigma=float(args.smooth_sigma),
        )
    prepared_depth_stats = summarize_depth(prepared_depth[valid_mask])

    sparse_depth_input = np.where(valid_mask, prepared_depth, 0.0).astype(np.float32)
    gradient_x_sparse, gradient_y_sparse, gradient_x_support, gradient_y_support = compute_masked_gradients(
        depth=sparse_depth_input,
        valid_mask=valid_mask,
    )
    gradient_x = normalized_gaussian_fill(
        values=np.nan_to_num(gradient_x_sparse, nan=0.0).astype(np.float32),
        support_mask=gradient_x_support,
        target_mask=mask,
        sigma=float(args.gradient_fill_sigma),
        preserve_support=True,
        fill_value=0.0,
        outside_value=np.nan,
    )[0]
    gradient_y = normalized_gaussian_fill(
        values=np.nan_to_num(gradient_y_sparse, nan=0.0).astype(np.float32),
        support_mask=gradient_y_support,
        target_mask=mask,
        sigma=float(args.gradient_fill_sigma),
        preserve_support=True,
        fill_value=0.0,
        outside_value=np.nan,
    )[0]
    gradient_x = np.where(mask, gradient_x, np.nan).astype(np.float32)
    gradient_y = np.where(mask, gradient_y, np.nan).astype(np.float32)
    gradient_magnitude = np.where(
        mask,
        np.sqrt(np.square(np.nan_to_num(gradient_x, nan=0.0)) + np.square(np.nan_to_num(gradient_y, nan=0.0))),
        np.nan,
    ).astype(np.float32)

    return {
        "valid_mask": valid_mask,
        "valid_count": valid_count,
        "normalized_sparse_depth": normalized_sparse_depth.astype(np.float32),
        "prepared_depth": prepared_depth.astype(np.float32),
        "normalization_reference": float(normalization_reference),
        "normalized_sparse_depth_stats": normalized_stats,
        "prepared_depth_stats": prepared_depth_stats,
        "gradient_x": gradient_x.astype(np.float32),
        "gradient_y": gradient_y.astype(np.float32),
        "gradient_magnitude": gradient_magnitude.astype(np.float32),
    }


def extract_row_geometry(mask: np.ndarray, min_width_px: int = 4) -> RowGeometry:
    rows: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    centers: list[float] = []
    half_widths: list[float] = []

    for row_index in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[row_index])
        if cols.size < min_width_px:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        width = right - left + 1
        if width < min_width_px:
            continue
        center = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        if half_width <= 0:
            continue
        rows.append(int(row_index))
        lefts.append(left)
        rights.append(right)
        centers.append(float(center))
        half_widths.append(float(half_width))

    if not rows:
        raise ValueError("Mask does not contain any rows wide enough to build a dense surface.")

    return RowGeometry(
        rows=np.asarray(rows, dtype=np.int32),
        lefts=np.asarray(lefts, dtype=np.int32),
        rights=np.asarray(rights, dtype=np.int32),
        centers=np.asarray(centers, dtype=np.float32),
        half_widths=np.asarray(half_widths, dtype=np.float32),
    )


def silhouette_ratio_profile(count: int, sigma: float, thickness_scale: float) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    bulge = np.clip(np.sin(np.pi * t), 0.0, None)
    ratio = 0.16 + 0.40 * np.power(bulge, 0.9) + 0.16 * t - 0.14 * np.square(t)
    ratio = np.clip(ratio, 0.10, 0.72)
    ratio *= float(thickness_scale)
    ratio = smooth_1d(ratio.astype(np.float32), sigma=float(sigma))
    return np.clip(ratio, 0.04, None).astype(np.float32)


def compute_sparse_guidance(
    row_geometry: RowGeometry,
    sparse_valid_mask: np.ndarray,
    sparse_depth: np.ndarray | None,
    silhouette_ratio: np.ndarray,
    row_smoothing_sigma: float,
) -> tuple[np.ndarray | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if sparse_depth is None or not np.any(sparse_valid_mask):
        return None, {"status": "missing"}, warnings

    support_rows: list[int] = []
    support_local_indices: list[int] = []
    support_ratios: list[float] = []
    support_counts: list[int] = []

    row_to_local_index = {int(row): idx for idx, row in enumerate(row_geometry.rows)}
    for row_index in np.flatnonzero(sparse_valid_mask.any(axis=1)):
        local_index = row_to_local_index.get(int(row_index))
        if local_index is None:
            continue
        row_mask = sparse_valid_mask[row_index]
        row_values = sparse_depth[row_index, row_mask]
        if row_values.size < 4:
            continue
        half_width = float(row_geometry.half_widths[local_index])
        if half_width <= 0:
            continue
        span = float(np.max(row_values) - np.min(row_values))
        ratio = span / half_width
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        support_rows.append(int(row_index))
        support_local_indices.append(local_index)
        support_ratios.append(ratio)
        support_counts.append(int(row_values.size))

    if len(support_rows) < 12:
        warnings.append("Sparse center depth did not cover enough rows for a stable guidance profile.")
        return None, {"status": "insufficient_rows", "supported_rows": len(support_rows)}, warnings

    ratios = np.asarray(support_ratios, dtype=np.float32)
    p10, p90 = np.percentile(ratios, [10, 90])
    clipped = np.clip(ratios, p10, p90).astype(np.float32)
    median_ratio = float(np.median(clipped))
    lower = max(float(np.percentile(clipped, 10)), 1e-4)
    upper = float(np.percentile(clipped, 90))
    stability_ratio = upper / lower
    stable = median_ratio > 0.01 and stability_ratio <= 50.0
    if not stable:
        warnings.append(
            "Sparse center depth produced an unstable thickness signal after robust clipping; silhouette mode is safer."
        )
        return None, {
            "status": "unstable",
            "supported_rows": len(support_rows),
            "median_ratio": median_ratio,
            "stability_ratio": stability_ratio,
        }, warnings

    all_local_indices = np.arange(row_geometry.rows.size, dtype=np.int32)
    interpolated = np.interp(all_local_indices, support_local_indices, clipped).astype(np.float32)

    first_local = support_local_indices[0]
    last_local = support_local_indices[-1]
    transition = max(24.0, 0.10 * row_geometry.rows.size)
    alpha = np.ones_like(interpolated, dtype=np.float32)
    if first_local > 0:
        distance = np.arange(first_local, 0, -1, dtype=np.float32)
        alpha[:first_local] = np.clip(np.exp(-np.square(distance / transition)), 0.0, 1.0)
    if last_local < row_geometry.rows.size - 1:
        distance = np.arange(1, row_geometry.rows.size - last_local, dtype=np.float32)
        alpha[last_local + 1 :] = np.clip(np.exp(-np.square(distance / transition)), 0.0, 1.0)
    alpha[first_local : last_local + 1] = 1.0

    blended = silhouette_ratio * (1.0 - alpha) + interpolated * alpha
    blended = smooth_1d(blended.astype(np.float32), sigma=float(row_smoothing_sigma))
    blended = np.clip(blended, 0.04, None).astype(np.float32)

    info = {
        "status": "ok",
        "supported_rows": len(support_rows),
        "row_range": [int(support_rows[0]), int(support_rows[-1])],
        "guidance_statistic": "row_depth_span_over_half_width",
        "support_rows": support_rows,
        "support_counts": support_counts,
        "raw_ratios": ratios.tolist(),
        "clipped_ratios": clipped.tolist(),
        "median_ratio": median_ratio,
        "stability_ratio": stability_ratio,
        "smoothing_interpolation_strategy": (
            "Robust percentile clipping, linear interpolation over mask rows, "
            "Gaussian row smoothing, and silhouette blending outside sparse support."
        ),
    }
    return blended, info, warnings


def choose_thickness_profile(
    requested_mode: str,
    row_geometry: RowGeometry,
    sparse_valid_mask: np.ndarray,
    sparse_depth: np.ndarray | None,
    row_smoothing_sigma: float,
    thickness_scale: float,
) -> tuple[np.ndarray, str, bool, dict[str, Any], list[str]]:
    warnings: list[str] = []
    silhouette_ratio = silhouette_ratio_profile(
        count=row_geometry.rows.size,
        sigma=row_smoothing_sigma,
        thickness_scale=thickness_scale,
    )

    if requested_mode == "silhouette":
        return silhouette_ratio, "silhouette", False, {"status": "not_requested"}, warnings

    sparse_ratio, sparse_info, sparse_warnings = compute_sparse_guidance(
        row_geometry=row_geometry,
        sparse_valid_mask=sparse_valid_mask,
        sparse_depth=sparse_depth,
        silhouette_ratio=silhouette_ratio,
        row_smoothing_sigma=row_smoothing_sigma,
    )
    warnings.extend(sparse_warnings)

    if requested_mode == "sparse_guided":
        if sparse_ratio is None:
            status = sparse_info.get("status", "unavailable")
            raise ValueError(f"Sparse-guided thickness was requested, but sparse depth guidance is unavailable ({status}).")
        return sparse_ratio, "sparse_guided", True, sparse_info, warnings

    if sparse_ratio is not None:
        return sparse_ratio, "sparse_guided", True, sparse_info, warnings

    warnings.append("Auto thickness mode fell back to silhouette guidance.")
    return silhouette_ratio, "silhouette", False, sparse_info, warnings


def build_legacy_dense_depth(
    mask: np.ndarray,
    row_geometry: RowGeometry,
    row_thickness: np.ndarray,
    z_scale: float,
) -> np.ndarray:
    dense_depth = np.full(mask.shape, np.nan, dtype=np.float32)
    for index, row_index in enumerate(row_geometry.rows):
        cols = np.flatnonzero(mask[row_index])
        if cols.size == 0:
            continue
        center = float(row_geometry.centers[index])
        half_width = float(row_geometry.half_widths[index])
        thickness = float(row_thickness[index])
        normalized = (cols.astype(np.float32) - center) / max(half_width, 1e-6)
        inside = np.maximum(1.0 - np.square(normalized), 0.0)
        z_rel = thickness * np.sqrt(inside)
        dense_depth[row_index, cols] = z_rel.astype(np.float32) * float(z_scale)
    return dense_depth.astype(np.float32)


def save_mask_boundaries_debug(path: Path, mask: np.ndarray, row_geometry: RowGeometry) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.imshow(mask, cmap="gray")
    ax.plot(row_geometry.lefts, row_geometry.rows, color="#d94841", linewidth=1.0, label="Left boundary")
    ax.plot(row_geometry.rights, row_geometry.rows, color="#2b8cbe", linewidth=1.0, label="Right boundary")
    ax.invert_yaxis()
    ax.set_title("Center Mask Boundaries")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_row_thickness_debug(
    path: Path,
    row_geometry: RowGeometry,
    half_widths: np.ndarray,
    row_thickness: np.ndarray,
    ratio_profile: np.ndarray,
    effective_mode: str,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(row_geometry.rows, half_widths, label="Half-width a(y)", color="#2b8cbe", linewidth=1.5)
    ax.plot(row_geometry.rows, row_thickness, label="Thickness b(y)", color="#d94841", linewidth=1.5)
    ax.plot(row_geometry.rows, ratio_profile, label="Thickness ratio b(y)/a(y)", color="#31a354", linewidth=1.2)
    ax.set_title(f"Row Thickness Profile ({effective_mode})")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Pixels")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_sparse_guidance_debug(path: Path, sparse_info: dict[str, Any], row_geometry: RowGeometry, final_ratio: np.ndarray) -> None:
    support_rows = np.asarray(sparse_info.get("support_rows", []), dtype=np.int32)
    raw_ratios = np.asarray(sparse_info.get("raw_ratios", []), dtype=np.float32)
    clipped_ratios = np.asarray(sparse_info.get("clipped_ratios", []), dtype=np.float32)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    if support_rows.size:
        ax.scatter(support_rows, raw_ratios, label="Sparse row ratios", color="#756bb1", s=18)
        ax.scatter(support_rows, clipped_ratios, label="Robust-clipped ratios", color="#238b45", s=18)
    ax.plot(row_geometry.rows, final_ratio, label="Final ratio profile", color="#d94841", linewidth=1.5)
    ax.set_title("Sparse Depth Guidance")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Thickness ratio")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_thickness_ratio_samples(path: Path, row_geometry: RowGeometry, final_ratio: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(row_geometry.rows, final_ratio, color="#d94841", linewidth=1.6)
    ax.set_title("Thickness Ratio Samples")
    ax.set_xlabel("Row index")
    ax.set_ylabel("b(y) / a(y)")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_center_depth_completion(
    args: argparse.Namespace,
    output_dir: Path,
    debug_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray | None,
    resolved: ResolvedInputs,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if depth is None or resolved.center_depth is None:
        raise ValueError("Center depth completion requires a center recovered depth array.")

    prepared = prepare_center_depth_and_gradients(args=args, mask=mask, depth=depth)
    valid_mask = prepared["valid_mask"]
    valid_count = prepared["valid_count"]
    prepared_depth = prepared["prepared_depth"]
    gradient_x = prepared["gradient_x"]
    gradient_y = prepared["gradient_y"]
    gradient_magnitude = prepared["gradient_magnitude"]
    normalization_reference = prepared["normalization_reference"]
    normalized_stats = prepared["normalized_sparse_depth_stats"]
    prepared_depth_stats = prepared["prepared_depth_stats"]

    mean_depth = float(np.mean(prepared_depth[valid_mask]))
    initial_dense_depth = normalized_gaussian_fill(
        values=np.where(valid_mask, prepared_depth, 0.0).astype(np.float32),
        support_mask=valid_mask,
        target_mask=mask,
        sigma=float(args.initial_fill_sigma),
        preserve_support=True,
        fill_value=mean_depth,
        outside_value=np.nan,
    )[0]
    initial_dense_depth = np.where(mask, initial_dense_depth, np.nan).astype(np.float32)

    completed_depth, completion_info = anchored_poisson_relaxation(
        initial_depth=initial_dense_depth,
        anchor_depth=prepared_depth,
        anchor_mask=valid_mask,
        mask=mask,
        grad_x=np.where(mask, np.nan_to_num(gradient_x, nan=0.0), 0.0),
        grad_y=np.where(mask, np.nan_to_num(gradient_y, nan=0.0), 0.0),
        max_iterations=int(args.completion_iterations),
        tolerance=float(args.completion_tol),
    )
    dense_depth = np.where(mask, completed_depth * float(args.z_scale), np.nan).astype(np.float32)
    dense_depth[valid_mask] = (prepared_depth[valid_mask] * float(args.z_scale)).astype(np.float32)

    points, intensities = dense_surface_points(image=image, dense_depth=dense_depth)
    dense_depth_values = dense_depth[np.isfinite(dense_depth)]
    bounds = bounds_from_points(points)
    normal_map, _, _ = build_normal_map(dense_depth=dense_depth, mask=mask)

    output_png = output_dir / "center_depth_completion.png"
    output_npz = output_dir / "center_depth_completion.npz"
    render_surface_plot(
        output_path=output_png,
        image=image,
        dense_depth=dense_depth,
        points_for_aspect=points,
        title="Center Depth Completion Surface",
        plot_max_points=max(args.plot_max_points, 1),
        elev=22.0,
        azim=-64.0,
    )
    np.savez_compressed(
        output_npz,
        points=points.astype(np.float32),
        intensities=intensities.astype(np.float32),
        dense_depth=dense_depth.astype(np.float32),
        normalized_sparse_depth=prepared_depth.astype(np.float32),
        valid_mask=valid_mask.astype(np.uint8),
        mask=mask.astype(np.uint8),
        image=image.astype(np.float32),
        gradient_x=gradient_x.astype(np.float32),
        gradient_y=gradient_y.astype(np.float32),
        normal_map=normal_map.astype(np.float32),
    )

    outputs: dict[str, Any] = {
        "center_depth_completion_png": str(output_png.resolve()),
        "center_depth_completion_npz": str(output_npz.resolve()),
    }

    if args.debug:
        valid_mask_png = debug_dir / "center_valid_mask.png"
        sparse_depth_png = debug_dir / "center_normalized_sparse_depth.png"
        gradient_x_png = debug_dir / "center_gradient_x.png"
        gradient_y_png = debug_dir / "center_gradient_y.png"
        gradient_magnitude_png = debug_dir / "center_gradient_magnitude.png"
        completed_depth_png = debug_dir / "center_completed_depth.png"
        completed_hist_png = debug_dir / "center_completed_depth_histogram.png"
        sparse_vs_completed_png = debug_dir / "center_sparse_vs_completed_depth.png"
        completion_debug_png = debug_dir / "center_depth_completion_debug.png"
        normal_map_png = debug_dir / "center_normal_map.png"

        save_mask_image(valid_mask_png, valid_mask, "Center Valid Recovered-Depth Mask")
        save_image(
            sparse_depth_png,
            np.where(valid_mask, prepared_depth, np.nan),
            cmap="viridis",
            title="Center Normalized Sparse Depth",
        )
        save_image(gradient_x_png, gradient_x, cmap="coolwarm", title="Center Gradient X")
        save_image(gradient_y_png, gradient_y, cmap="coolwarm", title="Center Gradient Y")
        save_image(gradient_magnitude_png, gradient_magnitude, cmap="magma", title="Center Gradient Magnitude")
        save_image(completed_depth_png, dense_depth, cmap="viridis", title="Center Completed Dense Depth")
        save_histogram(completed_hist_png, dense_depth_values, "Completed Dense Depth Histogram", "Dense depth")
        save_side_by_side_depth(
            sparse_vs_completed_png,
            np.where(valid_mask, prepared_depth * float(args.z_scale), np.nan),
            dense_depth,
            "Sparse Normalized Depth",
            "Completed Dense Depth",
        )
        render_surface_plot(
            output_path=completion_debug_png,
            image=image,
            dense_depth=dense_depth,
            points_for_aspect=points,
            title="Center Depth Completion Debug View",
            plot_max_points=max(args.plot_max_points, 1),
            elev=28.0,
            azim=-48.0,
        )
        save_normal_map(normal_map_png, normal_map=normal_map, mask=mask)
        outputs["debug"] = {
            "center_valid_mask": str(valid_mask_png.resolve()),
            "center_normalized_sparse_depth": str(sparse_depth_png.resolve()),
            "center_gradient_x": str(gradient_x_png.resolve()),
            "center_gradient_y": str(gradient_y_png.resolve()),
            "center_gradient_magnitude": str(gradient_magnitude_png.resolve()),
            "center_completed_depth": str(completed_depth_png.resolve()),
            "center_completed_depth_histogram": str(completed_hist_png.resolve()),
            "center_sparse_vs_completed_depth": str(sparse_vs_completed_png.resolve()),
            "center_depth_completion_debug": str(completion_debug_png.resolve()),
            "center_normal_map": str(normal_map_png.resolve()),
        }

    section = {
        "status": "ok",
        "reason": "Built a dense center-view depth completion surface successfully.",
        "pipeline_name": DEFAULT_PIPELINE,
        "input_image_path": str(resolved.center_image.resolve()),
        "input_mask_path": str(resolved.center_mask.resolve()),
        "input_sparse_depth_path": str(resolved.center_depth.resolve()),
        "normalization_mode": args.depth_normalization,
        "normalization_reference": float(normalization_reference),
        "smoothing_used": bool(args.smooth_depth),
        "smoothing_sigma": float(args.smooth_sigma),
        "gradient_method": args.gradient_method,
        "completion_method": args.completion_method,
        "completion_iterations": int(completion_info["iterations"]),
        "completion_residual": float(completion_info["residual"]),
        "valid_sparse_point_count": int(valid_count),
        "valid_sparse_fraction_of_mask": float(valid_count / max(int(np.count_nonzero(mask)), 1)),
        "valid_sparse_row_coverage": int(np.count_nonzero(valid_mask.any(axis=1))),
        "valid_sparse_column_coverage": int(np.count_nonzero(valid_mask.any(axis=0))),
        "dense_point_count": int(points.shape[0]),
        "bounds": bounds,
        "normalized_sparse_depth_stats": prepared_depth_stats,
        "normalized_sparse_depth_stats_before_smoothing": normalized_stats,
        "completed_dense_depth_stats": summarize_depth(dense_depth_values),
        "warnings": warnings.copy(),
        "outputs": outputs,
        "geometry_description": (
            "A dense surface completed from sparse recovered depth, guided by gradients and smoothness constraints. "
            "It is more faithful than the silhouette-only model while still including inferred regions where no "
            "direct depth existed."
        ),
    }
    return section, outputs


def run_center_unwarp(
    args: argparse.Namespace,
    output_dir: Path,
    debug_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray | None,
    resolved: ResolvedInputs,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if depth is None or resolved.center_depth is None:
        raise ValueError("Center unwarp requires a center recovered depth array.")

    prepared = prepare_center_depth_and_gradients(args=args, mask=mask, depth=depth)
    gradient_x = prepared["gradient_x"]
    gradient_y = prepared["gradient_y"]
    unwarp = run_center_unwarping(
        image=image,
        mask=mask,
        gradient_x=np.nan_to_num(gradient_x, nan=0.0),
        gradient_y=np.nan_to_num(gradient_y, nan=0.0),
        center_point_mode=args.center_point_mode,
        resampling_method=args.resampling_method,
    )
    warnings.extend(unwarp["warnings"])

    center_unwarped_png = output_dir / "center_unwarped.png"
    center_unwarped_mask_png = output_dir / "center_unwarped_mask.png"
    center_unwarp_maps_npz = output_dir / "center_unwarp_maps.npz"
    save_image(center_unwarped_png, unwarp["unwarped_image"], cmap="gray", title="Center Unwarped Fingerprint")
    save_mask_image(center_unwarped_mask_png, unwarp["unwarped_mask"] > 0, "Center Unwarped Mask")
    np.savez_compressed(
        center_unwarp_maps_npz,
        u=unwarp["u"].astype(np.float32),
        v=unwarp["v"].astype(np.float32),
        x_new=unwarp["x_new"].astype(np.float32),
        y_new=unwarp["y_new"].astype(np.float32),
        x_out=unwarp["x_out"].astype(np.float32),
        y_out=unwarp["y_out"].astype(np.float32),
        center_point=np.asarray(unwarp["center_point"], dtype=np.int32),
        mask=mask.astype(np.uint8),
        gradient_x=gradient_x.astype(np.float32),
        gradient_y=gradient_y.astype(np.float32),
        output_offset_x=np.asarray(unwarp["output_offset_x"], dtype=np.int32),
        output_offset_y=np.asarray(unwarp["output_offset_y"], dtype=np.int32),
    )

    outputs: dict[str, Any] = {
        "center_unwarped_png": str(center_unwarped_png.resolve()),
        "center_unwarped_mask_png": str(center_unwarped_mask_png.resolve()),
        "center_unwarp_maps_npz": str(center_unwarp_maps_npz.resolve()),
    }

    if args.debug:
        center_u_png = debug_dir / "center_unwarp_u.png"
        center_v_png = debug_dir / "center_unwarp_v.png"
        center_x_new_png = debug_dir / "center_unwarp_x_new.png"
        center_y_new_png = debug_dir / "center_unwarp_y_new.png"
        center_point_png = debug_dir / "center_unwarp_center_point.png"
        center_unwarped_debug_png = debug_dir / "center_unwarped_debug.png"
        center_unwarped_overlay_png = debug_dir / "center_unwarped_overlay.png"

        valid_mask = mask & np.isfinite(unwarp["u"]) & np.isfinite(unwarp["v"])
        save_image(center_u_png, np.where(valid_mask, unwarp["u"], np.nan), cmap="viridis", title="Center Unwarp U")
        save_image(center_v_png, np.where(valid_mask, unwarp["v"], np.nan), cmap="viridis", title="Center Unwarp V")
        save_image(
            center_x_new_png,
            np.where(valid_mask, unwarp["x_new"], np.nan),
            cmap="viridis",
            title="Center Unwarp X New",
        )
        save_image(
            center_y_new_png,
            np.where(valid_mask, unwarp["y_new"], np.nan),
            cmap="viridis",
            title="Center Unwarp Y New",
        )
        save_center_point_debug(
            center_point_png,
            image=image,
            mask=mask,
            center_point=(int(unwarp["center_point"][0]), int(unwarp["center_point"][1])),
        )
        save_image(
            center_unwarped_debug_png,
            unwarp["unwarped_image"],
            cmap="gray",
            title="Center Unwarped Fingerprint",
        )
        save_unwarp_overlay(
            center_unwarped_overlay_png,
            image=image,
            mask=mask,
            overlay_samples=unwarp["overlay_samples"],
        )
        outputs["debug"] = {
            "center_unwarp_u": str(center_u_png.resolve()),
            "center_unwarp_v": str(center_v_png.resolve()),
            "center_unwarp_x_new": str(center_x_new_png.resolve()),
            "center_unwarp_y_new": str(center_y_new_png.resolve()),
            "center_unwarp_center_point": str(center_point_png.resolve()),
            "center_unwarped_debug": str(center_unwarped_debug_png.resolve()),
            "center_unwarped_overlay": str(center_unwarped_overlay_png.resolve()),
        }

    section = {
        "status": "ok",
        "reason": "Built a center unwarped fingerprint image successfully.",
        "algorithm_name": "algorithm3_arc_length_unwarping",
        "input_image_path": str(resolved.center_image.resolve()),
        "input_mask_path": str(resolved.center_mask.resolve()),
        "input_gradient_source": "masked gradients derived from recovered center depth",
        "center_point": [int(unwarp["center_point"][0]), int(unwarp["center_point"][1])],
        "output_bounds": unwarp["output_bounds"],
        "output_shape": unwarp["output_shape"],
        "interpolation_method": args.resampling_method,
        "u_stats": unwarp["u_stats"],
        "v_stats": unwarp["v_stats"],
        "x_new_stats": unwarp["x_new_stats"],
        "y_new_stats": unwarp["y_new_stats"],
        "warnings": warnings.copy(),
        "outputs": outputs,
        "description": (
            "Algorithm 3 style center unwarping using the minimum-gradient zero point, arc length from gradients, "
            "and bilinear interpolation transform on the original center image."
        ),
    }
    return section, outputs


def run_center_dense_surface(
    args: argparse.Namespace,
    output_dir: Path,
    debug_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray | None,
    resolved: ResolvedInputs,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sparse_valid_mask = compute_sparse_valid_mask(mask, depth, args.depth_epsilon)
    working_depth = depth
    if working_depth is not None and args.smooth_depth and np.any(sparse_valid_mask):
        working_depth = masked_gaussian_smooth(working_depth, sparse_valid_mask, sigma=float(args.smooth_sigma))
        sparse_valid_mask = compute_sparse_valid_mask(mask, working_depth, args.depth_epsilon)

    row_geometry = extract_row_geometry(mask)
    ratio_profile, effective_mode, sparse_guidance_used, sparse_info, thickness_warnings = choose_thickness_profile(
        requested_mode=args.thickness_mode,
        row_geometry=row_geometry,
        sparse_valid_mask=sparse_valid_mask,
        sparse_depth=working_depth,
        row_smoothing_sigma=float(args.row_smoothing_sigma),
        thickness_scale=float(args.thickness_scale),
    )
    warnings.extend(thickness_warnings)

    row_thickness = (ratio_profile * row_geometry.half_widths).astype(np.float32)
    dense_depth = build_legacy_dense_depth(
        mask=mask,
        row_geometry=row_geometry,
        row_thickness=row_thickness,
        z_scale=float(args.z_scale),
    )
    points, intensities = dense_surface_points(image=image, dense_depth=dense_depth)
    dense_depth_values = dense_depth[np.isfinite(dense_depth)]
    bounds = bounds_from_points(points)

    surface_png = output_dir / "center_dense_surface.png"
    surface_npz = output_dir / "center_dense_surface.npz"
    render_surface_plot(
        output_path=surface_png,
        image=image,
        dense_depth=dense_depth,
        points_for_aspect=points,
        title="Center Dense Finger Surface",
        plot_max_points=max(args.plot_max_points, 1),
        elev=22.0,
        azim=-64.0,
    )
    np.savez_compressed(
        surface_npz,
        points=points.astype(np.float32),
        intensities=intensities.astype(np.float32),
        dense_depth=dense_depth.astype(np.float32),
        mask=mask.astype(np.uint8),
        image=image.astype(np.float32),
    )

    outputs: dict[str, Any] = {
        "center_dense_surface_png": str(surface_png.resolve()),
        "center_dense_surface_npz": str(surface_npz.resolve()),
    }

    if args.debug:
        dense_depth_debug = debug_dir / "center_dense_depth.png"
        dense_hist_debug = debug_dir / "center_dense_depth_histogram.png"
        row_thickness_debug = debug_dir / "center_row_thickness.png"
        mask_boundaries_debug = debug_dir / "center_mask_boundaries.png"
        surface_debug = debug_dir / "center_dense_surface_debug.png"

        save_image(dense_depth_debug, dense_depth, cmap="viridis", title="Dense Surface Depth")
        save_histogram(dense_hist_debug, dense_depth_values, "Dense Surface Depth Histogram", "Dense depth")
        save_row_thickness_debug(
            row_thickness_debug,
            row_geometry=row_geometry,
            half_widths=row_geometry.half_widths,
            row_thickness=row_thickness,
            ratio_profile=ratio_profile,
            effective_mode=effective_mode,
        )
        save_mask_boundaries_debug(mask_boundaries_debug, mask=mask, row_geometry=row_geometry)
        render_surface_plot(
            output_path=surface_debug,
            image=image,
            dense_depth=dense_depth,
            points_for_aspect=points,
            title="Center Dense Surface Debug View",
            plot_max_points=max(args.plot_max_points, 1),
            elev=28.0,
            azim=-48.0,
        )
        debug_outputs = {
            "center_dense_depth": str(dense_depth_debug.resolve()),
            "center_dense_depth_histogram": str(dense_hist_debug.resolve()),
            "center_row_thickness": str(row_thickness_debug.resolve()),
            "center_mask_boundaries": str(mask_boundaries_debug.resolve()),
            "center_dense_surface_debug": str(surface_debug.resolve()),
        }
        if sparse_guidance_used:
            sparse_guidance_debug = debug_dir / "center_sparse_depth_guidance.png"
            ratio_samples_debug = debug_dir / "center_thickness_ratio_samples.png"
            save_sparse_guidance_debug(
                sparse_guidance_debug,
                sparse_info=sparse_info,
                row_geometry=row_geometry,
                final_ratio=ratio_profile,
            )
            save_thickness_ratio_samples(ratio_samples_debug, row_geometry=row_geometry, final_ratio=ratio_profile)
            debug_outputs["center_sparse_depth_guidance"] = str(sparse_guidance_debug.resolve())
            debug_outputs["center_thickness_ratio_samples"] = str(ratio_samples_debug.resolve())
        outputs["debug"] = debug_outputs

    section = {
        "status": "ok",
        "reason": "Built a legacy silhouette-derived center surface successfully.",
        "pipeline_name": LEGACY_PIPELINE,
        "input_image_path": str(resolved.center_image.resolve()),
        "input_mask_path": str(resolved.center_mask.resolve()),
        "input_sparse_depth_path": str(resolved.center_depth.resolve()) if resolved.center_depth is not None else None,
        "thickness_mode": effective_mode,
        "requested_thickness_mode": args.thickness_mode,
        "row_count_used": int(row_geometry.rows.size),
        "dense_point_count": int(points.shape[0]),
        "dense_fraction_of_mask": float(points.shape[0] / max(int(np.count_nonzero(mask)), 1)),
        "bounds": bounds,
        "dense_depth_stats": summarize_depth(dense_depth_values),
        "sparse_depth_guidance_used": bool(sparse_guidance_used),
        "warnings": warnings.copy(),
        "outputs": outputs,
        "geometry_description": (
            "A legacy dense center-view finger surface synthesized row-by-row from the center silhouette. "
            "Sparse center depth, when available and stable, only guides the thickness profile."
        ),
    }
    if sparse_guidance_used:
        section["sparse_guidance"] = {
            "sparse_supported_rows": int(sparse_info["supported_rows"]),
            "guidance_statistic": sparse_info["guidance_statistic"],
            "guidance_row_range": sparse_info["row_range"],
            "smoothing_interpolation_strategy": sparse_info["smoothing_interpolation_strategy"],
            "median_ratio": float(sparse_info["median_ratio"]),
            "stability_ratio": float(sparse_info["stability_ratio"]),
        }
    else:
        section["sparse_guidance"] = {
            "status": sparse_info.get("status", "unused"),
        }
    return section, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense center-view finger surface for visualization.")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE, choices=[DEFAULT_PIPELINE, UNWARP_PIPELINE, LEGACY_PIPELINE])
    parser.add_argument("--center-image")
    parser.add_argument("--center-mask")
    parser.add_argument("--center-depth")
    parser.add_argument("--front-sample-dir")
    parser.add_argument("--reconstruction-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth-normalization", default="p05", choices=NORMALIZATION_CHOICES)
    parser.add_argument("--gradient-method", default=GRADIENT_METHODS[0], choices=GRADIENT_METHODS)
    parser.add_argument("--completion-method", default=COMPLETION_METHODS[0], choices=COMPLETION_METHODS)
    parser.add_argument("--center-point-mode", default=CENTER_POINT_MODES[0], choices=CENTER_POINT_MODES)
    parser.add_argument("--resampling-method", default=RESAMPLING_METHODS[0], choices=RESAMPLING_METHODS)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--depth-epsilon", type=float, default=1e-6)
    parser.add_argument("--smooth-depth", dest="smooth_depth", action="store_true")
    parser.add_argument("--no-smooth-depth", dest="smooth_depth", action="store_false")
    parser.set_defaults(smooth_depth=True)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--completion-iterations", type=int, default=400)
    parser.add_argument("--completion-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-fill-sigma", type=float, default=6.0)
    parser.add_argument("--initial-fill-sigma", type=float, default=10.0)
    parser.add_argument("--thickness-mode", default="auto", choices=["auto", "silhouette", "sparse_guided"])
    parser.add_argument("--row-smoothing-sigma", type=float, default=5.0)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    parser.add_argument("--plot-max-points", type=int, default=40000)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--left-image", help=argparse.SUPPRESS)
    parser.add_argument("--left-mask", help=argparse.SUPPRESS)
    parser.add_argument("--left-depth", help=argparse.SUPPRESS)
    parser.add_argument("--right-image", help=argparse.SUPPRESS)
    parser.add_argument("--right-mask", help=argparse.SUPPRESS)
    parser.add_argument("--right-depth", help=argparse.SUPPRESS)
    parser.add_argument("--left-sample-dir", help=argparse.SUPPRESS)
    parser.add_argument("--right-sample-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / DEFAULT_REPORT_NAME
    warnings: list[str] = []
    report: dict[str, Any] = {
        "status": "failed",
        "report_type": "center_depth_visualization",
        "description": (
            "Center-view finger visualization utilities, including depth-completion geometry reconstruction and "
            "Algorithm 3 style gradient-based unwarping."
        ),
        "requested_pipeline": args.pipeline,
        "warnings": warnings,
        "outputs": {},
        "coordinate_convention": {
            "x": "col - W / 2",
            "y": "-(row - H / 2)",
            "z": "completed_dense_depth * z_scale",
            "interpretation": "Orthographic pixel-grid visualization coordinates, not camera unprojection.",
        },
        "center_depth_completion": {
            "status": "not_run",
            "pipeline_name": DEFAULT_PIPELINE,
        },
        "center_unwarping": {
            "status": "not_run",
            "pipeline_name": UNWARP_PIPELINE,
        },
        "center_dense_surface": {
            "status": "not_run",
            "pipeline_name": LEGACY_PIPELINE,
        },
    }

    try:
        resolved, resolution_warnings = resolve_inputs(args)
        warnings.extend(resolution_warnings)

        image = load_grayscale_image(resolved.center_image)
        mask = load_mask(resolved.center_mask)
        depth = load_depth(resolved.center_depth) if resolved.center_depth is not None else None

        if image.shape != mask.shape:
            raise ValueError(f"Center image and mask shapes must match, got {image.shape} and {mask.shape}.")
        if depth is not None and depth.shape != mask.shape:
            raise ValueError(f"Center depth and mask shapes must match, got {depth.shape} and {mask.shape}.")
        if not np.any(mask):
            raise ValueError("Center mask is empty.")

        sparse_valid_mask = compute_sparse_valid_mask(mask, depth, args.depth_epsilon)
        mask_pixel_count = int(np.count_nonzero(mask))
        report["inputs"] = {
            "center_image": str(resolved.center_image.resolve()),
            "center_mask": str(resolved.center_mask.resolve()),
            "center_depth": str(resolved.center_depth.resolve()) if resolved.center_depth is not None else None,
        }
        report["shape_checks"] = {
            "image_shape": list(image.shape),
            "mask_shape": list(mask.shape),
            "depth_shape": list(depth.shape) if depth is not None else None,
            "image_mask_match": bool(image.shape == mask.shape),
            "depth_mask_match": bool(depth is None or depth.shape == mask.shape),
        }
        report["mask_stats"] = {
            "mask_pixel_count": mask_pixel_count,
            "mask_fraction": float(mask_pixel_count / mask.size),
            "row_count_with_mask_support": int(np.count_nonzero(mask.any(axis=1))),
            "col_count_with_mask_support": int(np.count_nonzero(mask.any(axis=0))),
        }
        report["sparse_depth_stats"] = {
            "input_sparse_depth_path": str(resolved.center_depth.resolve()) if resolved.center_depth is not None else None,
            "sparse_valid_pixel_count": int(np.count_nonzero(sparse_valid_mask)),
            "sparse_valid_fraction_of_mask": float(np.count_nonzero(sparse_valid_mask) / max(mask_pixel_count, 1)),
            "sparse_supported_rows": int(np.count_nonzero(sparse_valid_mask.any(axis=1))),
            "sparse_supported_cols": int(np.count_nonzero(sparse_valid_mask.any(axis=0))),
            "stats_over_valid_pixels": summarize_depth(depth[sparse_valid_mask]) if depth is not None else summarize_depth(np.array([], dtype=np.float32)),
        }

        if args.pipeline == DEFAULT_PIPELINE:
            section, outputs = run_center_depth_completion(
                args=args,
                output_dir=output_dir,
                debug_dir=debug_dir,
                image=image,
                mask=mask,
                depth=depth,
                resolved=resolved,
                warnings=warnings,
            )
            report["center_depth_completion"] = section
        elif args.pipeline == UNWARP_PIPELINE:
            section, outputs = run_center_unwarp(
                args=args,
                output_dir=output_dir,
                debug_dir=debug_dir,
                image=image,
                mask=mask,
                depth=depth,
                resolved=resolved,
                warnings=warnings,
            )
            report["center_unwarping"] = section
        else:
            section, outputs = run_center_dense_surface(
                args=args,
                output_dir=output_dir,
                debug_dir=debug_dir,
                image=image,
                mask=mask,
                depth=depth,
                resolved=resolved,
                warnings=warnings,
            )
            report["center_dense_surface"] = section

        outputs["report"] = str(report_path.resolve())
        report["outputs"] = outputs
        if args.pipeline == DEFAULT_PIPELINE:
            report["center_depth_completion"]["outputs"] = outputs
        elif args.pipeline == UNWARP_PIPELINE:
            report["center_unwarping"]["outputs"] = outputs
        else:
            report["center_dense_surface"]["outputs"] = outputs

        report["status"] = "ok"
        save_json(report_path, report)
        return 0
    except Exception as exc:
        warnings.append(str(exc))
        report["status"] = "failed"
        if args.pipeline == DEFAULT_PIPELINE:
            report["center_depth_completion"] = {
                "status": "failed",
                "reason": str(exc),
                "pipeline_name": DEFAULT_PIPELINE,
                "warnings": warnings.copy(),
                "outputs": report.get("outputs", {}),
            }
        elif args.pipeline == UNWARP_PIPELINE:
            report["center_unwarping"] = {
                "status": "failed",
                "reason": str(exc),
                "pipeline_name": UNWARP_PIPELINE,
                "warnings": warnings.copy(),
                "outputs": report.get("outputs", {}),
            }
        else:
            report["center_dense_surface"] = {
                "status": "failed",
                "reason": str(exc),
                "pipeline_name": LEGACY_PIPELINE,
                "warnings": warnings.copy(),
                "outputs": report.get("outputs", {}),
            }
        save_json(report_path, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
