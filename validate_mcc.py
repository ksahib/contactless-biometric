from __future__ import annotations

import argparse
import math
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import main as mcc


DEFAULT_METHODS = ("LSS", "LSA", "LSS-R", "LSA-R")
SUPPORTED_METHODS = DEFAULT_METHODS + (
    "LSA-LEGACY",
    "LSA-R-LEGACY",
    "LSA-OVERLAP",
    "LSA-R-OVERLAP",
)
DEFAULT_COMPARE_RUN_DIRS = (
    Path("match_outputs/match_1773388243"),
    Path("match_outputs/match_1773387992"),
    Path("match_outputs/match_1773386602"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test and benchmark the MCC matcher."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run synthetic MCC smoke checks")
    smoke.add_argument(
        "--csv",
        type=Path,
        default=Path("fingerflow_minutiae.csv"),
        help="Source minutiae CSV used for the smoke checks.",
    )
    smoke.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Matcher methods to run.",
    )
    smoke.add_argument("--seed", type=int, default=7, help="Random seed.")
    smoke.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    verify = subparsers.add_parser("verify", help="Run verification on a manifest")
    verify.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV manifest with labels and image/minutiae paths.",
    )
    verify.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Matcher methods to benchmark.",
    )
    verify.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".mcc_cache"),
        help="Directory used to cache extracted minutiae for image inputs.",
    )
    verify.add_argument(
        "--fingerflow-model-dir",
        type=Path,
        default=mcc.DEFAULT_FINGERFLOW_MODEL_DIR,
        help="FingerFlow model cache directory for image-based extraction.",
    )
    verify.add_argument(
        "--max-genuine-pairs",
        type=int,
        default=1000,
        help="Maximum number of genuine pairs to score.",
    )
    verify.add_argument(
        "--max-impostor-pairs",
        type=int,
        default=3000,
        help="Maximum number of impostor pairs to score.",
    )
    verify.add_argument("--seed", type=int, default=7, help="Random seed.")
    verify.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Diagnose a saved image-vs-image match run from match_outputs.",
    )
    diagnose.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a saved match run directory containing a/ and b/ subdirectories.",
    )
    diagnose.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Matcher methods to inspect.",
    )
    diagnose.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top local or selected pairs to report.",
    )
    diagnose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    compare_cached = subparsers.add_parser(
        "compare-cached",
        help="Compare legacy, relative, and canonical pose normalization on cached runs.",
    )
    compare_cached.add_argument(
        "--run-dirs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_COMPARE_RUN_DIRS),
        help="Saved match run directories to compare.",
    )
    compare_cached.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top local or selected pairs to report in the embedded diagnose output.",
    )
    compare_cached.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        name = method.upper()
        if name not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported MCC method: {method}")
        normalized.append(name)
    return normalized


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _smoke_perturbation(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    perturbed = frame.copy()
    perturbed["x"] = perturbed["x"].astype(float) + rng.normal(0.0, 3.0, len(perturbed))
    perturbed["y"] = perturbed["y"].astype(float) + rng.normal(0.0, 3.0, len(perturbed))
    perturbed["angle"] = (
        perturbed["angle"].astype(float) + rng.normal(0.0, 0.08, len(perturbed))
    )
    perturbed["angle"] = perturbed["angle"].apply(mcc.wrap_angle)
    return perturbed


def _smoke_randomized(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    randomized = frame.copy()
    x_values = frame["x"].astype(float).to_numpy()
    y_values = frame["y"].astype(float).to_numpy()
    x_pad = max(20.0, 0.1 * (x_values.max() - x_values.min()))
    y_pad = max(20.0, 0.1 * (y_values.max() - y_values.min()))
    randomized["x"] = rng.uniform(x_values.min() - x_pad, x_values.max() + x_pad, len(frame))
    randomized["y"] = rng.uniform(y_values.min() - y_pad, y_values.max() + y_pad, len(frame))
    randomized["angle"] = rng.uniform(-np.pi, np.pi, len(frame))
    return randomized


def run_smoke(csv_path: Path, methods: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base_frame = pd.read_csv(csv_path)
    perturbed_frame = _smoke_perturbation(base_frame, rng)
    randomized_frame = _smoke_randomized(base_frame, rng)

    with tempfile.TemporaryDirectory(prefix="mcc-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        perturbed_path = temp_root / "perturbed.csv"
        randomized_path = temp_root / "randomized.csv"
        _write_csv(perturbed_path, perturbed_frame)
        _write_csv(randomized_path, randomized_frame)

        base_descriptors = mcc.build_descriptors(csv_path)
        perturbed_descriptors = mcc.build_descriptors(perturbed_path)
        randomized_descriptors = mcc.build_descriptors(randomized_path)

        results: dict[str, dict] = {}
        for method in methods:
            if method in {"LSA-OVERLAP", "LSA-R-OVERLAP"}:
                self_score, self_matrix = mcc.match_minutiae_csv(csv_path, csv_path, method=method)
                perturb_score, _ = mcc.match_minutiae_csv(csv_path, perturbed_path, method=method)
                perturb_reverse_score, _ = mcc.match_minutiae_csv(perturbed_path, csv_path, method=method)
                randomized_score, _ = mcc.match_minutiae_csv(csv_path, randomized_path, method=method)
            else:
                descriptor_method = method.replace("-LEGACY", "")
                self_score, self_matrix = mcc.match_descriptors(
                    base_descriptors, base_descriptors, method=descriptor_method
                )
                perturb_score, _ = mcc.match_descriptors(
                    base_descriptors, perturbed_descriptors, method=descriptor_method
                )
                perturb_reverse_score, _ = mcc.match_descriptors(
                    perturbed_descriptors, base_descriptors, method=descriptor_method
                )
                randomized_score, _ = mcc.match_descriptors(
                    base_descriptors, randomized_descriptors, method=descriptor_method
                )

            results[method] = {
                "descriptor_count": len(base_descriptors),
                "self_score": float(self_score),
                "self_matrix_shape": list(self_matrix.shape),
                "perturbed_score": float(perturb_score),
                "perturbed_reverse_score": float(perturb_reverse_score),
                "randomized_score": float(randomized_score),
                "symmetry_gap": float(abs(perturb_score - perturb_reverse_score)),
            }

        return {
            "mode": "smoke",
            "source_csv": str(csv_path.resolve()),
            "methods": methods,
            "results": results,
        }


def _resolve_manifest_path(row: pd.Series, manifest_path: Path) -> Path:
    candidates = ("minutiae_csv", "image_path", "path")
    for field in candidates:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        return path
    raise ValueError(
        "manifest row must contain one of: minutiae_csv, image_path, path"
    )


def _manifest_identity(row: pd.Series) -> tuple[str, str, str]:
    subject = str(row["subject_id"])
    finger = str(row["finger_id"])
    sample = str(row.get("sample_id", row.get("path", row.get("image_path", row.get("minutiae_csv", "")))))
    return subject, finger, sample


def _cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _extract_minutiae_csv_for_image(
    image_path: Path,
    cache_dir: Path,
    model_dir: Path,
) -> Path:
    cache_root = cache_dir / _cache_key(image_path)
    minutiae_csv = cache_root / "minutiae.csv"
    if minutiae_csv.exists():
        return minutiae_csv

    cache_root.mkdir(parents=True, exist_ok=True)
    nobg_path = cache_root / "nobg.png"
    cropped_path = cache_root / "cropped.png"
    cropped_mask_path = cache_root / "cropped_mask.png"
    normalised_path = cache_root / "normalised.png"
    enhanced_path = cache_root / "enhanced.png"
    minutiae_json = cache_root / "minutiae.json"
    core_csv = cache_root / "core.csv"

    full_bgr = mcc.load_bgr_image(image_path)
    coarse_mask = mcc.rembg_mask_from_bgr(full_bgr)
    mcc.validate_foreground_area(coarse_mask, minimum_ratio=0.03)
    crop_bbox = mcc.compute_distal_crop_bbox(coarse_mask)
    cropped_bgr = mcc.crop_image(full_bgr, crop_bbox)
    cropped_mask = mcc.rembg_mask_from_bgr(cropped_bgr)
    mcc.validate_foreground_area(cropped_mask, minimum_ratio=0.08)
    cropped_rgba = mcc.compose_rgba(cropped_bgr, cropped_mask)
    mcc.save_png(cropped_path, cropped_bgr)
    mcc.save_png(nobg_path, cropped_rgba)
    mcc.save_png(cropped_mask_path, cropped_mask)
    mcc.normalise_brightness(nobg_path, normalised_path)
    mcc.filter(normalised_path, enhanced_path)
    model_paths = mcc.ensure_fingerflow_models(model_dir)
    mcc.extract_minutiae_with_fingerflow(
        image_path.resolve(),
        enhanced_path.resolve(),
        model_paths,
        minutiae_json.resolve(),
        minutiae_csv.resolve(),
        core_csv.resolve(),
    )
    return minutiae_csv


def _resolve_minutiae_csv(
    source_path: Path,
    cache_dir: Path,
    model_dir: Path,
) -> Path:
    if source_path.suffix.lower() == ".csv":
        return source_path
    return _extract_minutiae_csv_for_image(source_path, cache_dir, model_dir)


def _build_pair_lists(
    manifest_frame: pd.DataFrame,
    rng: np.random.Generator,
    max_genuine_pairs: int,
    max_impostor_pairs: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    identities = [_manifest_identity(row) for _, row in manifest_frame.iterrows()]
    genuine_pairs: list[tuple[int, int]] = []
    impostor_pairs: list[tuple[int, int]] = []

    for i in range(len(identities)):
        subject_i, finger_i, _ = identities[i]
        for j in range(i + 1, len(identities)):
            subject_j, finger_j, _ = identities[j]
            if subject_i == subject_j and finger_i == finger_j:
                genuine_pairs.append((i, j))
            else:
                impostor_pairs.append((i, j))

    rng.shuffle(genuine_pairs)
    rng.shuffle(impostor_pairs)
    return genuine_pairs[:max_genuine_pairs], impostor_pairs[:max_impostor_pairs]


def _roc_metrics(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> dict:
    thresholds = np.unique(np.concatenate((genuine_scores, impostor_scores)))
    thresholds = np.concatenate(
        (
            np.array([-np.inf], dtype=np.float64),
            thresholds,
            np.array([np.inf], dtype=np.float64),
        )
    )

    roc_points = []
    for threshold in thresholds:
        tar = float(np.mean(genuine_scores >= threshold))
        far = float(np.mean(impostor_scores >= threshold))
        roc_points.append((far, tar, float(threshold)))

    roc_points.sort(key=lambda item: item[0])
    fars = np.array([point[0] for point in roc_points], dtype=np.float64)
    tars = np.array([point[1] for point in roc_points], dtype=np.float64)
    frrs = 1.0 - tars
    best_eer_index = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[best_eer_index] + frrs[best_eer_index]) / 2.0)

    tar_at_far_1pct = 0.0
    valid_far = np.where(fars <= 0.01)[0]
    if valid_far.size > 0:
        tar_at_far_1pct = float(np.max(tars[valid_far]))

    comparisons = genuine_scores[:, None] - impostor_scores[None, :]
    auc = (
        np.count_nonzero(comparisons > 0)
        + (0.5 * np.count_nonzero(comparisons == 0))
    ) / comparisons.size

    return {
        "eer": eer,
        "tar_at_far_1pct": tar_at_far_1pct,
        "roc_auc": float(auc),
        "genuine_median": float(np.median(genuine_scores)),
        "impostor_median": float(np.median(impostor_scores)),
        "genuine_histogram": np.histogram(genuine_scores, bins=10, range=(0.0, 1.0))[0].tolist(),
        "impostor_histogram": np.histogram(impostor_scores, bins=10, range=(0.0, 1.0))[0].tolist(),
    }


def run_verification(
    manifest_path: Path,
    methods: list[str],
    cache_dir: Path,
    model_dir: Path,
    max_genuine_pairs: int,
    max_impostor_pairs: int,
    seed: int,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "finger_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    resolved_csvs: list[Path] = []
    for _, row in manifest.iterrows():
        source_path = _resolve_manifest_path(row, manifest_path)
        resolved_csvs.append(_resolve_minutiae_csv(source_path, cache_dir, model_dir))

    descriptor_cache: dict[Path, list[mcc.MCCCylinder]] = {}

    def get_descriptors(csv_path: Path) -> list[mcc.MCCCylinder]:
        cached = descriptor_cache.get(csv_path)
        if cached is None:
            cached = mcc.build_descriptors(csv_path)
            descriptor_cache[csv_path] = cached
        return cached

    rng = np.random.default_rng(seed)
    genuine_pairs, impostor_pairs = _build_pair_lists(
        manifest,
        rng,
        max_genuine_pairs=max_genuine_pairs,
        max_impostor_pairs=max_impostor_pairs,
    )

    if not genuine_pairs or not impostor_pairs:
        raise RuntimeError(
            "verification requires at least one genuine pair and one impostor pair"
        )

    results: dict[str, dict] = {}
    for method in methods:
        genuine_scores = []
        impostor_scores = []

        def _pair_score(left_csv: Path, right_csv: Path) -> float:
            left_mask = left_csv.with_name("cropped_mask.png")
            right_mask = right_csv.with_name("cropped_mask.png")
            has_masks = left_mask.exists() and right_mask.exists()

            if method in {"LSA-OVERLAP", "LSA-R-OVERLAP"}:
                score, _ = mcc.match_minutiae_csv(
                    left_csv,
                    right_csv,
                    method=method,
                    mask_path_a=left_mask if has_masks else None,
                    mask_path_b=right_mask if has_masks else None,
                )
                return float(score)

            if method in {"LSA", "LSA-R"} and has_masks:
                score, _ = mcc.match_minutiae_csv(
                    left_csv,
                    right_csv,
                    method=method,
                    mask_path_a=left_mask,
                    mask_path_b=right_mask,
                )
                return float(score)

            if method in {"LSA-LEGACY", "LSA-R-LEGACY"} and has_masks:
                score, _ = mcc.match_minutiae_csv_legacy(
                    left_csv,
                    right_csv,
                    method=method,
                    mask_path_a=left_mask,
                    mask_path_b=right_mask,
                )
                return float(score)

            descriptor_method = method.replace("-LEGACY", "")
            score, _ = mcc.match_descriptors(
                get_descriptors(left_csv),
                get_descriptors(right_csv),
                method=descriptor_method,
            )
            return float(score)

        for left, right in genuine_pairs:
            genuine_scores.append(_pair_score(resolved_csvs[left], resolved_csvs[right]))

        for left, right in impostor_pairs:
            impostor_scores.append(_pair_score(resolved_csvs[left], resolved_csvs[right]))

        genuine_array = np.array(genuine_scores, dtype=np.float64)
        impostor_array = np.array(impostor_scores, dtype=np.float64)
        results[method] = {
            "genuine_pair_count": len(genuine_pairs),
            "impostor_pair_count": len(impostor_pairs),
            **_roc_metrics(genuine_array, impostor_array),
        }

    return {
        "mode": "verify",
        "manifest": str(manifest_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "methods": methods,
        "results": results,
    }


def _latest_match_run(match_root: Path) -> Path:
    candidates = [path for path in match_root.glob("match_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no saved match runs found under: {match_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def _image_stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"exists": True, "readable": False}

    stats = {
        "exists": True,
        "readable": True,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }

    if image.ndim == 2:
        nonzero_mask = image > 0
    elif image.ndim == 3 and image.shape[2] == 4:
        nonzero_mask = image[:, :, 3] > 0
        stats["alpha_foreground_ratio"] = float(np.mean(nonzero_mask))
    else:
        nonzero_mask = np.any(image > 0, axis=2)

    coords = np.argwhere(nonzero_mask)
    stats["nonzero_ratio"] = float(np.mean(nonzero_mask))
    stats["mean_intensity"] = float(np.mean(image))
    stats["std_intensity"] = float(np.std(image))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)[:2]
        y_max, x_max = coords.max(axis=0)[:2]
        stats["nonzero_bbox"] = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
        }
    else:
        stats["nonzero_bbox"] = None
    return stats


def _minutiae_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"exists": False}

    frame = pd.read_csv(csv_path)
    summary = {
        "exists": True,
        "count": int(len(frame)),
        "columns": list(frame.columns),
    }
    if frame.empty:
        return summary

    for axis in ("x", "y", "angle", "score"):
        if axis in frame.columns:
            series = frame[axis].astype(float)
            summary[f"{axis}_stats"] = {
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }

    if {"x", "y"}.issubset(frame.columns):
        x = frame["x"].astype(float)
        y = frame["y"].astype(float)
        summary["spatial_spread"] = {
            "x_range": float(x.max() - x.min()),
            "y_range": float(y.max() - y.min()),
            "x_median": float(x.median()),
            "y_median": float(y.median()),
        }

    if "class" in frame.columns:
        counts = frame["class"].value_counts(dropna=False).sort_index()
        summary["class_counts"] = {
            str(index): int(value) for index, value in counts.items()
        }
    return summary


def _resolve_validity_mode(
    validity_mask_path: Path | None,
    validity_mode: str = "auto",
) -> str:
    if validity_mode != "auto":
        return validity_mode
    return "mask" if validity_mask_path is not None and validity_mask_path.exists() else "hull"


def _mask_coverage(validity_mask_path: Path | None) -> dict | None:
    if validity_mask_path is None or not validity_mask_path.exists():
        return None
    mask = cv2.imread(str(validity_mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = mask > 0
    return {
        "path": str(validity_mask_path.resolve()),
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "foreground_ratio": float(np.count_nonzero(binary) / binary.size) if binary.size else 0.0,
    }


def _descriptor_invalidation_stats(
    csv_path: Path,
    validity_mask_path: Path | None = None,
    validity_mode: str = "auto",
    adaptive_neighbor_support: bool = True,
) -> dict:
    frame = pd.read_csv(csv_path).dropna(subset=["x", "y", "angle"])
    minutiae = frame.to_dict(orient="records")
    resolved_mode = _resolve_validity_mode(validity_mask_path, validity_mode)

    if not minutiae:
        return {
            "validity_mode": resolved_mode,
            "base_neighbor_threshold": int(mcc.MCC_MIN_M),
            "resolved_neighbor_threshold": int(mcc.MCC_MIN_M),
            "adaptive_neighbor_support": bool(adaptive_neighbor_support),
            "min_valid_cells_threshold": 0,
            "too_few_valid_cells": 0,
            "too_few_neighbors": 0,
            "dropped_descriptors": 0,
            "support_region_invalidated_cells": 0,
            "in_radius_cells": 0,
        }

    radius = mcc.MCC_RADIUS
    ns = mcc.MCC_NS
    nd = mcc.MCC_ND
    delta_s = (2.0 * radius) / ns
    sigma_s = mcc.MCC_SIGMA_S
    center = (ns + 1) / 2.0
    min_valid_cells = int(math.ceil(0.75 * mcc._max_valid_cells(ns, nd, delta_s, radius)))
    hull = mcc._build_convex_hull(minutiae)
    validity_mask = mcc._load_validity_mask(validity_mask_path) if validity_mask_path and validity_mask_path.exists() else None
    resolved_mode = mcc._resolve_validity_mode(resolved_mode, validity_mask)
    resolved_neighbor_threshold = mcc._required_neighbor_count(
        len(minutiae),
        adaptive_neighbor_support=adaptive_neighbor_support,
    )

    too_few_valid_cells = 0
    too_few_neighbors = 0
    dropped_descriptors = 0
    support_region_invalidated_cells = 0
    in_radius_cells = 0

    for minutia in minutiae:
        x_m = float(minutia["x"])
        y_m = float(minutia["y"])
        theta_m = float(minutia["angle"])
        cos_t = math.cos(theta_m)
        sin_t = math.sin(theta_m)

        contributing_minutiae = 0
        for neighbor in minutiae:
            if neighbor is minutia:
                continue
            x_t = float(neighbor["x"])
            y_t = float(neighbor["y"])
            if math.hypot(x_t - x_m, y_t - y_m) <= (radius + (3.0 * sigma_s)):
                contributing_minutiae += 1

        valid_cell_count = 0
        for i in range(ns):
            for j in range(ns):
                dx = delta_s * ((i + 1) - center)
                dy = delta_s * ((j + 1) - center)
                if (dx * dx + dy * dy) > (radius * radius):
                    continue
                in_radius_cells += nd
                x_cell = x_m + (dx * cos_t) - (dy * sin_t)
                y_cell = y_m + (dx * sin_t) + (dy * cos_t)
                is_valid = mcc._cell_is_valid(
                    dx=dx,
                    dy=dy,
                    x_cell=x_cell,
                    y_cell=y_cell,
                    radius=radius,
                    hull=hull,
                    validity_mask=validity_mask,
                    validity_mode=resolved_mode,
                )
                if is_valid:
                    valid_cell_count += nd
                else:
                    support_region_invalidated_cells += nd

        dropped = False
        if valid_cell_count < min_valid_cells:
            too_few_valid_cells += 1
            dropped = True
        if contributing_minutiae < resolved_neighbor_threshold:
            too_few_neighbors += 1
            dropped = True
        if dropped:
            dropped_descriptors += 1

    return {
        "validity_mode": resolved_mode,
        "base_neighbor_threshold": int(mcc.MCC_MIN_M),
        "resolved_neighbor_threshold": int(resolved_neighbor_threshold),
        "adaptive_neighbor_support": bool(adaptive_neighbor_support),
        "min_valid_cells_threshold": min_valid_cells,
        "too_few_valid_cells": too_few_valid_cells,
        "too_few_neighbors": too_few_neighbors,
        "dropped_descriptors": dropped_descriptors,
        "support_region_invalidated_cells": support_region_invalidated_cells,
        "in_radius_cells": in_radius_cells,
    }


def _descriptor_summary(
    csv_path: Path,
    validity_mask_path: Path | None = None,
    validity_mode: str = "auto",
    adaptive_neighbor_support: bool = True,
) -> tuple[list[mcc.MCCCylinder], dict]:
    raw_frame = pd.read_csv(csv_path)
    raw_count = int(len(raw_frame))
    resolved_mode = _resolve_validity_mode(validity_mask_path, validity_mode)
    build_mask_path = validity_mask_path if resolved_mode in {"mask", "hybrid"} else None
    descriptors = mcc.build_descriptors(
        csv_path,
        validity_mask_path=build_mask_path,
        validity_mode=resolved_mode,
        adaptive_neighbor_support=adaptive_neighbor_support,
    )
    descriptor_count = len(descriptors)
    invalidation = _descriptor_invalidation_stats(
        csv_path,
        validity_mask_path=validity_mask_path,
        validity_mode=resolved_mode,
        adaptive_neighbor_support=adaptive_neighbor_support,
    )
    return descriptors, {
        "raw_minutiae_count": raw_count,
        "surviving_descriptor_count": descriptor_count,
        "descriptor_survival_ratio": (
            float(descriptor_count / raw_count) if raw_count > 0 else 0.0
        ),
        "validity_mode": resolved_mode,
        "adaptive_neighbor_support": bool(adaptive_neighbor_support),
        "mask_coverage": _mask_coverage(validity_mask_path),
        "invalidation": invalidation,
    }


def _pair_gating_stats(
    descriptors_a: list[mcc.MCCCylinder],
    descriptors_b: list[mcc.MCCCylinder],
    sim_matrix: np.ndarray,
) -> dict:
    if not descriptors_a or not descriptors_b:
        return {
            "total_pairs": 0,
            "angle_rejected_pairs": 0,
            "matchable_rejected_pairs": 0,
            "comparable_pairs": 0,
            "nonzero_similarity_pairs": 0,
            "min_matchable_cells": 0,
        }

    values_a, validities_a, angles_a = mcc._flatten_descriptors(descriptors_a)
    values_b, validities_b, angles_b = mcc._flatten_descriptors(descriptors_b)
    min_matchable = int(
        np.ceil(
            0.60
            * mcc._max_valid_cells(
                mcc.MCC_NS,
                mcc.MCC_ND,
                (2.0 * mcc.MCC_RADIUS) / mcc.MCC_NS,
                mcc.MCC_RADIUS,
            )
        )
    )

    angle_rejected = 0
    matchable_rejected = 0
    comparable_pairs = 0
    for i in range(len(descriptors_a)):
        for j in range(len(descriptors_b)):
            if abs(mcc.wrap_angle(float(angles_a[i]) - float(angles_b[j]))) > mcc.MCC_DELTA_THETA:
                angle_rejected += 1
                continue

            matchable_mask = (validities_a[i] == 1) & (validities_b[j] == 1)
            if int(np.count_nonzero(matchable_mask)) < min_matchable:
                matchable_rejected += 1
                continue
            comparable_pairs += 1

    return {
        "total_pairs": int(len(descriptors_a) * len(descriptors_b)),
        "angle_rejected_pairs": angle_rejected,
        "matchable_rejected_pairs": matchable_rejected,
        "comparable_pairs": comparable_pairs,
        "nonzero_similarity_pairs": int(np.count_nonzero(sim_matrix > 0)),
        "min_matchable_cells": min_matchable,
    }


def _top_local_pairs(sim_matrix: np.ndarray, limit: int) -> list[dict]:
    if sim_matrix.size == 0 or limit <= 0:
        return []
    rows, cols = sim_matrix.shape
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1][:limit]
    pairs = []
    for flat_index in flat_indices:
        score = float(sim_matrix.flat[flat_index])
        pairs.append(
            {
                "row": int(flat_index // cols),
                "col": int(flat_index % cols),
                "score": score,
            }
        )
    return pairs


def _method_diagnostics(
    descriptors_a: list[mcc.MCCCylinder],
    descriptors_b: list[mcc.MCCCylinder],
    method: str,
    top_k: int,
    csv_a: Path | None = None,
    csv_b: Path | None = None,
    mask_a: Path | None = None,
    mask_b: Path | None = None,
) -> dict:
    overlap_details = None
    pose_details = None
    canonical_pose_details = None
    descriptor_method = method.replace("-LEGACY", "")
    diagnostic_descriptors_a = descriptors_a
    diagnostic_descriptors_b = descriptors_b

    if method in {"LSA-OVERLAP", "LSA-R-OVERLAP"}:
        if csv_a is None or csv_b is None:
            raise ValueError(f"{method} diagnostics require minutiae CSV paths")
        score, sim_matrix, overlap_details = mcc.match_minutiae_csv_overlap_details(
            csv_a,
            csv_b,
            mask_a,
            mask_b,
            method=method,
        )
    elif method in {"LSA", "LSA-R"} and csv_a is not None and csv_b is not None and mask_a is not None and mask_b is not None:
        score, sim_matrix, pose_details = mcc.match_minutiae_csv_pose_normalized_details(
            csv_a,
            csv_b,
            mask_a,
            mask_b,
            method=method,
            strategy="relative",
        )
        canonical_score, canonical_sim_matrix, canonical_pose_details = mcc.match_minutiae_csv_pose_normalized_details(
            csv_a,
            csv_b,
            mask_a,
            mask_b,
            method=method,
            strategy="canonical",
        )
    else:
        if method in {"LSA-LEGACY", "LSA-R-LEGACY"} and csv_a is not None and csv_b is not None:
            score, sim_matrix = mcc.match_minutiae_csv_legacy(
                csv_a,
                csv_b,
                method=method,
                mask_path_a=mask_a,
                mask_path_b=mask_b,
            )
        else:
            score, sim_matrix = mcc.match_descriptors(descriptors_a, descriptors_b, method=descriptor_method)
    n_pairs = mcc._compute_n_pairs(sim_matrix.shape[0], sim_matrix.shape[1])
    selected_pairs: list[tuple[int, int, float]]
    relaxed_scores: list[float] = []
    selected_scores: list[float] = []

    if pose_details is not None:
        selected_pairs = [
            (int(pair["row"]), int(pair["col"]), float(pair["score"]))
            for pair in pose_details["selected_pairs"]
        ]
        selected_scores = [float(score_value) for score_value in pose_details["selected_pair_scores"]]
        relaxed_scores = [float(score_value) for score_value in pose_details["relaxed_top_scores"]]
        relaxation_details = pose_details["relaxation_details"]
    elif method == "LSS":
        selected_pairs = mcc._select_lss_pairs(sim_matrix, n_pairs)
    elif method in {"LSA", "LSA-LEGACY", "LSA-OVERLAP"}:
        selected_pairs = mcc._select_lsa_pairs(sim_matrix, n_pairs)
    elif method == "LSS-R":
        selected_pairs = mcc._select_lss_pairs(
            sim_matrix,
            min(len(descriptors_a), len(descriptors_b)),
        )
        relaxed, efficiency, relaxation_details = mcc._relax_pairs_with_details(
            descriptors_a,
            descriptors_b,
            selected_pairs,
        )
        top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(selected_pairs))]
        relaxed_scores = [float(relaxed[index]) for index in top_indices]
    else:
        selected_pairs = mcc._select_lsa_pairs(
            sim_matrix,
            min(len(descriptors_a), len(descriptors_b)),
        )
        relaxed, efficiency, relaxation_details = mcc._relax_pairs_with_details(
            descriptors_a,
            descriptors_b,
            selected_pairs,
        )
        top_indices = np.argsort(efficiency)[::-1][: min(n_pairs, len(selected_pairs))]
        relaxed_scores = [float(relaxed[index]) for index in top_indices]

    if method not in {"LSS-R", "LSA-R", "LSA-R-LEGACY", "LSA-R-OVERLAP"}:
        relaxation_details = None

    selected_scores = [float(pair[2]) for pair in selected_pairs[:top_k]]
    selected_pairs_report = [
        {"row": int(row), "col": int(col), "score": float(score)}
        for row, col, score in selected_pairs[:top_k]
    ]
    if overlap_details is None and pose_details is None:
        gating_stats = _pair_gating_stats(diagnostic_descriptors_a, diagnostic_descriptors_b, sim_matrix)
    else:
        gating_stats = {
            "total_pairs": int(sim_matrix.shape[0] * sim_matrix.shape[1]),
            "angle_rejected_pairs": None,
            "matchable_rejected_pairs": None,
            "comparable_pairs": None,
            "nonzero_similarity_pairs": int(np.count_nonzero(sim_matrix > 0)),
            "min_matchable_cells": None,
        }

    report = {
        "final_score": float(score),
        "similarity_matrix_shape": list(sim_matrix.shape),
        "top_local_pairs": _top_local_pairs(sim_matrix, top_k),
        "selected_pair_count": len(selected_pairs),
        "selected_pairs": selected_pairs_report,
        "selected_pair_scores": selected_scores,
        "relaxed_top_scores": relaxed_scores[:top_k],
        **gating_stats,
    }
    if overlap_details is not None:
        report["overlap_details"] = {
            "estimated_transform": overlap_details["estimated_transform"],
            "overlap_area": int(overlap_details["overlap_area"]),
            "overlap_ratio": float(overlap_details["overlap_ratio"]),
            "fallback_to_base": bool(overlap_details["fallback_to_base"]),
            "fallback_reason": overlap_details["fallback_reason"],
            "raw_minutiae_count": {
                "left": int(overlap_details["left_raw_minutiae_count"]),
                "right": int(overlap_details["right_raw_minutiae_count"]),
            },
            "overlap_minutiae_count": {
                "left": int(overlap_details["left_overlap_minutiae_count"]),
                "right": int(overlap_details["right_overlap_minutiae_count"]),
            },
            "descriptor_count_before": {
                "left": int(overlap_details["left_descriptor_count_before"]),
                "right": int(overlap_details["right_descriptor_count_before"]),
            },
            "descriptor_count_after": {
                "left": int(overlap_details["left_descriptor_count_after"]),
                "right": int(overlap_details["right_descriptor_count_after"]),
            },
            "final_selected_pair_count": int(overlap_details["selected_pair_count"]),
        }
    if pose_details is not None:
        report["pose_details"] = {
            "strategy": pose_details["strategy"],
            "estimated_transform": pose_details["estimated_transform"],
            "overlap_area": int(pose_details["overlap_area"]),
            "overlap_ratio": float(pose_details["overlap_ratio"]),
            "fallback_to_legacy": bool(pose_details["fallback_to_legacy"]),
            "fallback_reason": pose_details["fallback_reason"],
            "raw_minutiae_count": {
                "left": int(pose_details["left_raw_minutiae_count"]),
                "right": int(pose_details["right_raw_minutiae_count"]),
            },
            "normalized_overlap_minutiae_count": {
                "left": int(pose_details["left_overlap_minutiae_count"]),
                "right": int(pose_details["right_overlap_minutiae_count"]),
            },
            "descriptor_count_before": {
                "left": int(pose_details["left_descriptor_count_before"]),
                "right": int(pose_details["right_descriptor_count_before"]),
            },
            "descriptor_count_after": {
                "left": int(pose_details["left_descriptor_count_after"]),
                "right": int(pose_details["right_descriptor_count_after"]),
            },
            "final_selected_pair_count": int(pose_details["selected_pair_count"]),
            "legacy_method": pose_details["legacy_method"],
            "legacy_score": float(pose_details["legacy_score"]),
        }
        if canonical_pose_details is not None:
            report["pose_comparison"] = {
                "active_strategy": "relative",
                "relative": {
                    "final_score": float(score),
                    "estimated_transform": pose_details["estimated_transform"],
                    "overlap_area": int(pose_details["overlap_area"]),
                    "overlap_ratio": float(pose_details["overlap_ratio"]),
                    "fallback_to_legacy": bool(pose_details["fallback_to_legacy"]),
                    "fallback_reason": pose_details["fallback_reason"],
                    "descriptor_count_after": {
                        "left": int(pose_details["left_descriptor_count_after"]),
                        "right": int(pose_details["right_descriptor_count_after"]),
                    },
                    "final_selected_pair_count": int(pose_details["selected_pair_count"]),
                },
                "canonical": {
                    "final_score": float(canonical_score),
                    "estimated_transform": canonical_pose_details["estimated_transform"],
                    "overlap_area": int(canonical_pose_details["overlap_area"]),
                    "overlap_ratio": float(canonical_pose_details["overlap_ratio"]),
                    "fallback_to_legacy": bool(canonical_pose_details["fallback_to_legacy"]),
                    "fallback_reason": canonical_pose_details["fallback_reason"],
                    "descriptor_count_after": {
                        "left": int(canonical_pose_details["left_descriptor_count_after"]),
                        "right": int(canonical_pose_details["right_descriptor_count_after"]),
                    },
                    "final_selected_pair_count": int(canonical_pose_details["selected_pair_count"]),
                },
                "legacy": {
                    "method": pose_details["legacy_method"],
                    "final_score": float(pose_details["legacy_score"]),
                },
                "deltas": {
                    "relative_minus_legacy": float(score - pose_details["legacy_score"]),
                    "canonical_minus_legacy": float(canonical_score - canonical_pose_details["legacy_score"]),
                    "canonical_minus_relative": float(canonical_score - score),
                },
            }
    if method in {"LSS-R", "LSA-R", "LSA-R-LEGACY", "LSA-R-OVERLAP"}:
        strongest_local = report["top_local_pairs"][0]["score"] if report["top_local_pairs"] else 0.0
        report["relaxation_collapse_ratio"] = (
            float(score / strongest_local) if strongest_local > 0 else 0.0
        )
        report["non_relaxed_reference_score"] = float(
            np.mean(selected_scores[: min(n_pairs, len(selected_scores))])
        ) if selected_scores else 0.0
        if pose_details is not None:
            report["relaxation_details"] = {
                "compatibility_summary": relaxation_details["compatibility_summary"] if relaxation_details else None,
                "iteration_summaries": relaxation_details["iteration_summaries"] if relaxation_details else [],
                "support_summaries": relaxation_details["support_summaries"] if relaxation_details else [],
                "final_relaxed_ranking": [
                    {
                        "row": int(pair["row"]),
                        "col": int(pair["col"]),
                        "initial_score": float(pair["score"]),
                        "relaxed_score": float(relaxed_score),
                    }
                    for pair, relaxed_score in zip(selected_pairs_report, report["relaxed_top_scores"])
                ],
            }
        else:
            report["relaxation_details"] = {
                "compatibility_summary": relaxation_details["compatibility_summary"],
                "iteration_summaries": relaxation_details["iteration_summaries"],
                "support_summaries": relaxation_details["support_summaries"],
                "final_relaxed_ranking": [
                    {
                        "row": int(selected_pairs[index][0]),
                        "col": int(selected_pairs[index][1]),
                        "initial_score": float(selected_pairs[index][2]),
                        "relaxed_score": float(relaxed[index]),
                        "efficiency": float(efficiency[index]),
                    }
                    for index in np.argsort(efficiency)[::-1][:top_k]
                ],
            }
    return report


def _stage_file_summary(sample_dir: Path) -> dict:
    return {
        "cropped": _image_stats(sample_dir / "cropped.png"),
        "cropped_mask": _image_stats(sample_dir / "cropped_mask.png"),
        "nobg": _image_stats(sample_dir / "nobg.png"),
        "normalised": _image_stats(sample_dir / "normalised.png"),
        "enhanced": _image_stats(sample_dir / "enhanced.png"),
        "minutiae_csv": str((sample_dir / "minutiae.csv").resolve()) if (sample_dir / "minutiae.csv").exists() else None,
        "minutiae_json": str((sample_dir / "minutiae.json").resolve()) if (sample_dir / "minutiae.json").exists() else None,
        "core_csv": str((sample_dir / "core.csv").resolve()) if (sample_dir / "core.csv").exists() else None,
    }


def _resolve_sample_dir(side_dir: Path) -> Path:
    direct_csv = side_dir / "minutiae.csv"
    if direct_csv.exists():
        return side_dir

    children = [path for path in side_dir.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / "minutiae.csv").exists():
        return children[0]

    raise FileNotFoundError(
        f"could not resolve sample directory under: {side_dir}"
    )


def _dominant_bottleneck(report: dict) -> str:
    left = report["left"]
    right = report["right"]
    enhanced_left = left["stages"]["enhanced"]
    enhanced_right = right["stages"]["enhanced"]

    if enhanced_left.get("exists") and enhanced_right.get("exists"):
        left_shape = enhanced_left.get("shape")
        right_shape = enhanced_right.get("shape")
        left_nonzero = enhanced_left.get("nonzero_ratio", 0.0)
        right_nonzero = enhanced_right.get("nonzero_ratio", 0.0)
        if left_shape and right_shape:
            height_ratio = max(left_shape[0], right_shape[0]) / max(1, min(left_shape[0], right_shape[0]))
            width_ratio = max(left_shape[1], right_shape[1]) / max(1, min(left_shape[1], right_shape[1]))
            if height_ratio > 1.5 or width_ratio > 1.5 or abs(left_nonzero - right_nonzero) > 0.25:
                return "crop/enhancement"

    left_raw = left["descriptors"]["raw_minutiae_count"]
    right_raw = right["descriptors"]["raw_minutiae_count"]
    if min(left_raw, right_raw) == 0:
        return "minutiae extraction"
    if max(left_raw, right_raw) / max(1, min(left_raw, right_raw)) > 1.8:
        return "minutiae extraction"

    left_survival = left["descriptors"]["descriptor_survival_ratio"]
    right_survival = right["descriptors"]["descriptor_survival_ratio"]
    if left_survival < 0.5 or right_survival < 0.5:
        left_adaptive = left["descriptors"]["invalidation"].get("adaptive_neighbor_support", False)
        right_adaptive = right["descriptors"]["invalidation"].get("adaptive_neighbor_support", False)
        if left_adaptive or right_adaptive:
            return "descriptor filtering after adaptive neighbor fallback"
        return "descriptor filtering"

    lss = report["methods"].get("LSS", {})
    lsa = report["methods"].get("LSA", {})
    lsa_r = report["methods"].get("LSA-R", {})
    lss_r = report["methods"].get("LSS-R", {})
    best_non_relaxed = max(lss.get("final_score", 0.0), lsa.get("final_score", 0.0))
    best_relaxed = max(lss_r.get("final_score", 0.0), lsa_r.get("final_score", 0.0))
    strongest_local = 0.0
    if lsa_r.get("top_local_pairs"):
        strongest_local = lsa_r["top_local_pairs"][0]["score"]
    elif lss.get("top_local_pairs"):
        strongest_local = lss["top_local_pairs"][0]["score"]

    if best_non_relaxed >= 0.25 and best_relaxed <= (0.15 * best_non_relaxed) and strongest_local >= 0.35:
        return "MCC pair gating"

    comparable_pairs = lsa_r.get("comparable_pairs")
    total_pairs = lsa_r.get("total_pairs", 0)
    if comparable_pairs is not None and total_pairs > 0 and comparable_pairs / total_pairs < 0.05:
        return "MCC pair gating"

    pose_comparison = lsa_r.get("pose_comparison")
    if pose_comparison:
        relative = pose_comparison.get("relative", {})
        canonical = pose_comparison.get("canonical", {})
        deltas = pose_comparison.get("deltas", {})
        if (
            relative.get("overlap_ratio", 0.0) >= 0.8
            and canonical.get("overlap_ratio", 0.0) >= 0.8
            and deltas.get("relative_minus_legacy", 0.0) < 0.03
            and deltas.get("canonical_minus_legacy", 0.0) < 0.03
        ):
            return "extractor repeatability / local deformation"

    if lsa_r.get("top_local_pairs"):
        top_score = lsa_r["top_local_pairs"][0]["score"]
        if top_score < 0.2:
            return "local similarity weakness"

    return "no dominant bottleneck identified"


def run_diagnose(run_dir: Path | None, methods: list[str], top_k: int) -> dict:
    resolved_run_dir = _latest_match_run(mcc.MATCH_OUTPUTS_DIR) if run_dir is None else run_dir.resolve()
    left_root = resolved_run_dir / "a"
    right_root = resolved_run_dir / "b"
    if not left_root.exists() or not right_root.exists():
        raise FileNotFoundError(f"run directory must contain 'a' and 'b' subdirectories: {resolved_run_dir}")
    left_dir = _resolve_sample_dir(left_root)
    right_dir = _resolve_sample_dir(right_root)

    left_csv = left_dir / "minutiae.csv"
    right_csv = right_dir / "minutiae.csv"
    left_mask = left_dir / "cropped_mask.png"
    right_mask = right_dir / "cropped_mask.png"
    left_descriptors, left_descriptor_summary = _descriptor_summary(left_csv, left_mask)
    right_descriptors, right_descriptor_summary = _descriptor_summary(right_csv, right_mask)

    report = {
        "mode": "diagnose",
        "run_dir": str(resolved_run_dir),
        "left": {
            "path": str(left_dir),
            "stages": _stage_file_summary(left_dir),
            "minutiae": _minutiae_summary(left_csv),
            "descriptors": left_descriptor_summary,
        },
        "right": {
            "path": str(right_dir),
            "stages": _stage_file_summary(right_dir),
            "minutiae": _minutiae_summary(right_csv),
            "descriptors": right_descriptor_summary,
        },
        "methods": {},
    }

    if left_mask.exists() and right_mask.exists():
        _, left_hull_summary = _descriptor_summary(left_csv, None, validity_mode="hull")
        _, right_hull_summary = _descriptor_summary(right_csv, None, validity_mode="hull")
        report["descriptor_mode_comparison"] = {
            "mask": {
                "left_surviving_descriptor_count": left_descriptor_summary["surviving_descriptor_count"],
                "right_surviving_descriptor_count": right_descriptor_summary["surviving_descriptor_count"],
            },
            "hull": {
                "left_surviving_descriptor_count": left_hull_summary["surviving_descriptor_count"],
                "right_surviving_descriptor_count": right_hull_summary["surviving_descriptor_count"],
            },
        }

    _, left_fixed_summary = _descriptor_summary(
        left_csv,
        left_mask if left_mask.exists() else None,
        adaptive_neighbor_support=False,
    )
    _, right_fixed_summary = _descriptor_summary(
        right_csv,
        right_mask if right_mask.exists() else None,
        adaptive_neighbor_support=False,
    )
    report["adaptive_neighbor_comparison"] = {
        "adaptive_on": {
            "left_surviving_descriptor_count": left_descriptor_summary["surviving_descriptor_count"],
            "right_surviving_descriptor_count": right_descriptor_summary["surviving_descriptor_count"],
            "left_resolved_neighbor_threshold": left_descriptor_summary["invalidation"]["resolved_neighbor_threshold"],
            "right_resolved_neighbor_threshold": right_descriptor_summary["invalidation"]["resolved_neighbor_threshold"],
        },
        "adaptive_off": {
            "left_surviving_descriptor_count": left_fixed_summary["surviving_descriptor_count"],
            "right_surviving_descriptor_count": right_fixed_summary["surviving_descriptor_count"],
            "left_resolved_neighbor_threshold": left_fixed_summary["invalidation"]["resolved_neighbor_threshold"],
            "right_resolved_neighbor_threshold": right_fixed_summary["invalidation"]["resolved_neighbor_threshold"],
        },
    }

    for method in methods:
        report["methods"][method] = _method_diagnostics(
            left_descriptors,
            right_descriptors,
            method,
            top_k,
            csv_a=left_csv,
            csv_b=right_csv,
            mask_a=left_mask if left_mask.exists() else None,
            mask_b=right_mask if right_mask.exists() else None,
        )

    report["dominant_bottleneck"] = _dominant_bottleneck(report)
    return report


def run_compare_cached(run_dirs: list[Path], top_k: int) -> dict:
    methods = ["LSA", "LSA-LEGACY", "LSA-R", "LSA-R-LEGACY"]
    comparisons = []
    for run_dir in run_dirs:
        diagnose_report = run_diagnose(run_dir, methods, top_k=top_k)
        method_comparisons = {}
        for method in ("LSA", "LSA-R"):
            method_report = diagnose_report["methods"][method]
            pose_comparison = method_report.get("pose_comparison", {})
            method_comparisons[method] = {
                "active_score": float(method_report["final_score"]),
                "legacy_score": float(diagnose_report["methods"][f"{method}-LEGACY"]["final_score"]),
                "relative_score": float(pose_comparison.get("relative", {}).get("final_score", method_report["final_score"])),
                "canonical_score": float(pose_comparison.get("canonical", {}).get("final_score", method_report["final_score"])),
                "relative_fallback": bool(pose_comparison.get("relative", {}).get("fallback_to_legacy", False)),
                "canonical_fallback": bool(pose_comparison.get("canonical", {}).get("fallback_to_legacy", False)),
                "relative_fallback_reason": pose_comparison.get("relative", {}).get("fallback_reason"),
                "canonical_fallback_reason": pose_comparison.get("canonical", {}).get("fallback_reason"),
                "deltas": pose_comparison.get(
                    "deltas",
                    {
                        "relative_minus_legacy": 0.0,
                        "canonical_minus_legacy": 0.0,
                        "canonical_minus_relative": 0.0,
                    },
                ),
                "best_strategy": max(
                    (
                        ("legacy", float(diagnose_report["methods"][f"{method}-LEGACY"]["final_score"])),
                        ("relative", float(pose_comparison.get("relative", {}).get("final_score", method_report["final_score"]))),
                        ("canonical", float(pose_comparison.get("canonical", {}).get("final_score", method_report["final_score"]))),
                    ),
                    key=lambda item: item[1],
                )[0],
            }
        comparisons.append(
            {
                "run_dir": diagnose_report["run_dir"],
                "dominant_bottleneck": diagnose_report["dominant_bottleneck"],
                "methods": method_comparisons,
                "diagnose": diagnose_report,
            }
        )

    return {
        "mode": "compare-cached",
        "run_dirs": [str(Path(run_dir).resolve()) for run_dir in run_dirs],
        "comparisons": comparisons,
    }


def _print_summary(report: dict) -> None:
    print(json.dumps(report, indent=2))


def _maybe_write_output(report: dict, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    methods = _normalize_methods(list(args.methods)) if hasattr(args, "methods") else []

    if args.command == "smoke":
        report = run_smoke(args.csv, methods, seed=args.seed)
        _print_summary(report)
        _maybe_write_output(report, args.output)
        return 0

    if args.command == "diagnose":
        report = run_diagnose(args.run_dir, methods, top_k=args.top_k)
        _print_summary(report)
        _maybe_write_output(report, args.output)
        return 0

    if args.command == "compare-cached":
        report = run_compare_cached(list(args.run_dirs), top_k=args.top_k)
        _print_summary(report)
        _maybe_write_output(report, args.output)
        return 0

    report = run_verification(
        manifest_path=args.manifest,
        methods=methods,
        cache_dir=args.cache_dir,
        model_dir=args.fingerflow_model_dir,
        max_genuine_pairs=args.max_genuine_pairs,
        max_impostor_pairs=args.max_impostor_pairs,
        seed=args.seed,
    )
    _print_summary(report)
    _maybe_write_output(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
