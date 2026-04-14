from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional dependency fallback
    gaussian_filter = None


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
DEFAULT_PIPELINE = "center_dense_surface"


@dataclass
class ResolvedInputs:
    center_image: Path
    center_mask: Path
    center_depth: Path | None


@dataclass
class RowGeometry:
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
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
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


def masked_gaussian_smooth(values: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(np.float32, copy=True)

    valid_mask_f = valid_mask.astype(np.float32)
    weighted = np.where(valid_mask, values.astype(np.float32), 0.0)
    if gaussian_filter is not None:
        smooth_values = gaussian_filter(weighted, sigma=sigma, mode="nearest")
        smooth_weights = gaussian_filter(valid_mask_f, sigma=sigma, mode="nearest")
    else:  # pragma: no cover - only used if scipy is unavailable
        kernel = gaussian_kernel1d(sigma)
        smooth_values = convolve_axis(convolve_axis(weighted, kernel, axis=1), kernel, axis=0)
        smooth_weights = convolve_axis(convolve_axis(valid_mask_f, kernel, axis=1), kernel, axis=0)

    out = values.astype(np.float32, copy=True)
    stable = smooth_weights > 1e-6
    out[valid_mask & stable] = (smooth_values[valid_mask & stable] / smooth_weights[valid_mask & stable]).astype(
        np.float32
    )
    return out


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
            "Left/right inputs were provided but ignored by the center_dense_surface pipeline: "
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


def build_dense_depth(mask: np.ndarray, row_geometry: RowGeometry, row_thickness: np.ndarray, z_scale: float) -> np.ndarray:
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
    return dense_depth


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
    im = ax.imshow(array, cmap=cmap)
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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense center-view finger surface for visualization.")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE, choices=[DEFAULT_PIPELINE])
    parser.add_argument("--center-image")
    parser.add_argument("--center-mask")
    parser.add_argument("--center-depth")
    parser.add_argument("--front-sample-dir")
    parser.add_argument("--reconstruction-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thickness-mode", default="auto", choices=["auto", "silhouette", "sparse_guided"])
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--depth-epsilon", type=float, default=1e-6)
    parser.add_argument("--smooth-depth", dest="smooth_depth", action="store_true")
    parser.add_argument("--no-smooth-depth", dest="smooth_depth", action="store_false")
    parser.set_defaults(smooth_depth=True)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--row-smoothing-sigma", type=float, default=5.0)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    parser.add_argument("--plot-max-points", type=int, default=40000)
    parser.add_argument("--debug", action="store_true")

    # Compatibility-only inputs: accepted, then ignored.
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
        "report_type": "center_dense_surface_visualization",
        "description": (
            "Dense silhouette-derived finger surface built from the center image and center mask, "
            "optionally guided by sparse center depth. This is intended for visualization and is not "
            "a calibrated metric 3D reconstruction."
        ),
        "warnings": warnings,
        "outputs": {},
        "coordinate_convention": {
            "x": "col - W / 2",
            "y": "-(row - H / 2)",
            "z": "dense_row_ellipse_depth * z_scale",
            "interpretation": "Orthographic pixel-grid visualization coordinates, not camera unprojection.",
        },
        "center_dense_surface": {
            "status": "failed",
            "pipeline_name": DEFAULT_PIPELINE,
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
        initial_sparse_valid_count = int(np.count_nonzero(sparse_valid_mask))
        if depth is not None and args.smooth_depth and initial_sparse_valid_count > 0:
            depth = masked_gaussian_smooth(depth, sparse_valid_mask, sigma=float(args.smooth_sigma))
            sparse_valid_mask = compute_sparse_valid_mask(mask, depth, args.depth_epsilon)

        row_geometry = extract_row_geometry(mask)
        ratio_profile, effective_mode, sparse_guidance_used, sparse_info, thickness_warnings = choose_thickness_profile(
            requested_mode=args.thickness_mode,
            row_geometry=row_geometry,
            sparse_valid_mask=sparse_valid_mask,
            sparse_depth=depth,
            row_smoothing_sigma=float(args.row_smoothing_sigma),
            thickness_scale=float(args.thickness_scale),
        )
        warnings.extend(thickness_warnings)

        row_thickness = (ratio_profile * row_geometry.half_widths).astype(np.float32)
        dense_depth = build_dense_depth(
            mask=mask,
            row_geometry=row_geometry,
            row_thickness=row_thickness,
            z_scale=float(args.z_scale),
        )
        points, intensities = dense_surface_points(image=image, dense_depth=dense_depth)
        dense_valid = np.isfinite(dense_depth)
        dense_depth_values = dense_depth[dense_valid]
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

        outputs = {
            "center_dense_surface_png": str(surface_png.resolve()),
            "center_dense_surface_npz": str(surface_npz.resolve()),
            "report": str(report_path.resolve()),
        }
        report["outputs"] = outputs

        if args.debug:
            dense_depth_debug = debug_dir / "center_dense_depth.png"
            dense_hist_debug = debug_dir / "center_dense_depth_histogram.png"
            row_thickness_debug = debug_dir / "center_row_thickness.png"
            mask_boundaries_debug = debug_dir / "center_mask_boundaries.png"
            surface_debug = debug_dir / "center_dense_surface_debug.png"

            depth_preview = np.where(np.isfinite(dense_depth), dense_depth, np.nan)
            save_image(dense_depth_debug, depth_preview, cmap="viridis", title="Dense Surface Depth")
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
            "stats_over_valid_pixels": summarize_depth(depth[sparse_valid_mask]) if depth is not None else summarize_depth(np.array([], dtype=np.float32)),
        }

        center_dense_surface = {
            "status": "ok",
            "reason": "Built a dense silhouette-derived center surface successfully.",
            "pipeline_name": DEFAULT_PIPELINE,
            "input_image_path": str(resolved.center_image.resolve()),
            "input_mask_path": str(resolved.center_mask.resolve()),
            "input_sparse_depth_path": str(resolved.center_depth.resolve()) if resolved.center_depth is not None else None,
            "thickness_mode": effective_mode,
            "requested_thickness_mode": args.thickness_mode,
            "row_count_used": int(row_geometry.rows.size),
            "dense_point_count": int(points.shape[0]),
            "dense_fraction_of_mask": float(points.shape[0] / max(mask_pixel_count, 1)),
            "bounds": bounds,
            "dense_depth_stats": summarize_depth(dense_depth_values),
            "sparse_depth_guidance_used": bool(sparse_guidance_used),
            "warnings": warnings.copy(),
            "outputs": outputs,
            "geometry_description": (
                "A dense center-view finger surface synthesized row-by-row from the center silhouette. "
                "Sparse center depth, when available and stable, only guides the thickness profile."
            ),
        }
        if sparse_guidance_used:
            center_dense_surface["sparse_guidance"] = {
                "sparse_supported_rows": int(sparse_info["supported_rows"]),
                "guidance_statistic": sparse_info["guidance_statistic"],
                "guidance_row_range": sparse_info["row_range"],
                "smoothing_interpolation_strategy": sparse_info["smoothing_interpolation_strategy"],
                "median_ratio": float(sparse_info["median_ratio"]),
                "stability_ratio": float(sparse_info["stability_ratio"]),
            }
        else:
            center_dense_surface["sparse_guidance"] = {
                "status": sparse_info.get("status", "unused"),
            }

        report["center_dense_surface"] = center_dense_surface
        report["status"] = "ok"
        save_json(report_path, report)
        return 0
    except Exception as exc:
        warnings.append(str(exc))
        report["status"] = "failed"
        report["center_dense_surface"] = {
            "status": "failed",
            "reason": str(exc),
            "pipeline_name": DEFAULT_PIPELINE,
            "warnings": warnings.copy(),
            "outputs": report.get("outputs", {}),
        }
        save_json(report_path, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
