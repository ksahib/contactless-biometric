from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Iterable


def ensure_stdlib_copy_module() -> None:
    """Avoid importing this repository's copy.py when libraries need stdlib copy."""
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()


def prepend_workspace_site_packages() -> None:
    site_packages = Path(__file__).resolve().parent / ".venv" / "lib" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))


prepend_workspace_site_packages()

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from featurenet.models.infer import (
    decode_minutiae_rows,
    load_checkpoint_model,
    preprocess_input_bgr,
    run_inference,
    save_minutiae_csv,
    save_pose_sidecars,
    _resolve_device,
)
from featurenet.models.match_infer import (
    _crop_distal_phalanx_with_main,
    _save_mask_png,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOTS = (REPO_ROOT / "dataset" / "DS1", REPO_ROOT / "dataset" / "DS2", REPO_ROOT / "dataset" / "DS3")
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "weights" / "best.pt"
DEFAULT_MATCH_OUTPUTS_DIR = REPO_ROOT / "match_outputs"
FAR_TARGETS = (0.10, 0.05, 0.01, 0.001)


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    dataset_root: str
    subject_id: int
    finger_id: int
    acquisition_id: int
    view_index: int
    image_path: str

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha1(self.image_path.encode("utf-8")).hexdigest()[:10]
        return (
            f"{_slug(self.dataset)}"
            f"/s{self.subject_id:03d}"
            f"/f{self.finger_id:02d}"
            f"/a{self.acquisition_id:02d}"
            f"/v{self.view_index:02d}_{digest}"
        )


@dataclass(frozen=True)
class PairSpec:
    label: str
    a: ImageRecord
    b: ImageRecord

    @property
    def same_acquisition(self) -> bool:
        return self.a.acquisition_id == self.b.acquisition_id

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return self.label, self.a.image_path, self.b.image_path


@dataclass
class ExtractedImage:
    record: ImageRecord
    minutiae_csv: Path
    mask_png: Path
    orientation_npy: Path
    ridge_period_npy: Path
    metadata_json: Path
    minutiae_count: int


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "dataset"


def _default_output_dir() -> Path:
    return DEFAULT_MATCH_OUTPUTS_DIR / f"threshold_estimation_{time.strftime('%Y%m%d_%H%M%S')}"


def _parse_raw_image_path(dataset_root: Path, raw_path: Path) -> ImageRecord | None:
    if raw_path.suffix.lower() != ".jpg":
        return None
    parts = raw_path.stem.split("_")
    if len(parts) != 4:
        return None
    try:
        subject_id, finger_id, acquisition_id, view_index = [int(part) for part in parts]
    except ValueError:
        return None
    return ImageRecord(
        dataset=dataset_root.name,
        dataset_root=str(dataset_root.resolve()),
        subject_id=subject_id,
        finger_id=finger_id,
        acquisition_id=acquisition_id,
        view_index=view_index,
        image_path=str(raw_path.resolve()),
    )


def discover_raw_images(dataset_roots: Iterable[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for dataset_root in dataset_roots:
        root = dataset_root.resolve()
        if not root.exists():
            print(f"[warn] dataset root does not exist, skipping: {root}", file=sys.stderr)
            continue
        subject_dirs = [path for path in root.iterdir() if path.is_dir()]
        for subject_dir in sorted(
            subject_dirs,
            key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name),
        ):
            raw_dir = subject_dir / "raw"
            if not raw_dir.exists():
                continue
            for raw_path in sorted(raw_dir.glob("*.jpg")):
                record = _parse_raw_image_path(root, raw_path)
                if record is not None:
                    records.append(record)
    return records


def _group_by_identity(records: Iterable[ImageRecord]) -> dict[tuple[str, int, int], list[ImageRecord]]:
    groups: dict[tuple[str, int, int], list[ImageRecord]] = {}
    for record in records:
        groups.setdefault((record.dataset, record.subject_id, record.finger_id), []).append(record)
    return groups


def build_genuine_pairs(
    records: list[ImageRecord],
    side_views: set[int],
    rng: np.random.Generator,
    max_pairs: int | None,
) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    for group_records in _group_by_identity(records).values():
        fronts = [record for record in group_records if record.view_index == 0]
        sides = [record for record in group_records if record.view_index in side_views]
        for front in fronts:
            for side in sides:
                pairs.append(PairSpec(label="genuine", a=front, b=side))
    pairs.sort(key=lambda pair: (pair.a.dataset, pair.a.subject_id, pair.a.finger_id, pair.a.acquisition_id, pair.b.acquisition_id, pair.b.view_index))
    if max_pairs is not None and len(pairs) > max_pairs:
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(index)] for index in sorted(indices)]
    return pairs


def _build_impostor_candidate_groups(
    records: Iterable[ImageRecord],
    side_views: set[int],
) -> list[tuple[list[ImageRecord], list[ImageRecord]]]:
    grouped: dict[tuple[str, int], dict[str, list[ImageRecord]]] = {}
    for record in records:
        bucket = grouped.setdefault((record.dataset, record.finger_id), {"fronts": [], "sides": []})
        if record.view_index == 0:
            bucket["fronts"].append(record)
        elif record.view_index in side_views:
            bucket["sides"].append(record)

    candidate_groups: list[tuple[list[ImageRecord], list[ImageRecord]]] = []
    for bucket in grouped.values():
        fronts = bucket["fronts"]
        sides = bucket["sides"]
        if fronts and sides and len({record.subject_id for record in fronts + sides}) >= 2:
            candidate_groups.append((fronts, sides))
    return candidate_groups


def build_impostor_pairs(
    records: list[ImageRecord],
    side_views: set[int],
    rng: np.random.Generator,
    max_pairs: int,
) -> list[PairSpec]:
    if max_pairs <= 0:
        return []

    candidate_groups = _build_impostor_candidate_groups(records, side_views)
    if not candidate_groups:
        return []

    pairs: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(1000, max_pairs * 200)
    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        fronts, sides = candidate_groups[int(rng.integers(0, len(candidate_groups)))]
        front = fronts[int(rng.integers(0, len(fronts)))]
        side = sides[int(rng.integers(0, len(sides)))]
        if front.subject_id == side.subject_id:
            continue
        pair = PairSpec(label="impostor", a=front, b=side)
        if pair.pair_key in seen:
            continue
        seen.add(pair.pair_key)
        pairs.append(pair)
    return pairs


def _count_minutiae_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "minutiae_csv": cache_dir / "minutiae.csv",
        "mask_png": cache_dir / "mask.png",
        "orientation_npy": cache_dir / "orientation.npy",
        "ridge_period_npy": cache_dir / "ridge_period.npy",
        "metadata_json": cache_dir / "metadata.json",
    }


def _cached_extraction_is_complete(files: dict[str, Path]) -> bool:
    required = ("minutiae_csv", "mask_png", "orientation_npy", "ridge_period_npy", "metadata_json")
    return all(files[name].exists() and files[name].stat().st_size > 0 for name in required)


def extract_image(
    record: ImageRecord,
    *,
    cache_root: Path,
    model: Any,
    device: Any,
    score_threshold: float,
    reuse_cache: bool,
) -> ExtractedImage:
    cache_dir = cache_root / record.cache_key
    files = _cache_files(cache_dir)
    if reuse_cache and _cached_extraction_is_complete(files):
        try:
            metadata = json.loads(files["metadata_json"].read_text(encoding="utf-8"))
            minutiae_count = int(metadata.get("minutiae_count", _count_minutiae_rows(files["minutiae_csv"])))
        except Exception:
            minutiae_count = _count_minutiae_rows(files["minutiae_csv"])
        return ExtractedImage(
            record=record,
            minutiae_csv=files["minutiae_csv"],
            mask_png=files["mask_png"],
            orientation_npy=files["orientation_npy"],
            ridge_period_npy=files["ridge_period_npy"],
            metadata_json=files["metadata_json"],
            minutiae_count=minutiae_count,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(record.image_path)
    crop_result = _crop_distal_phalanx_with_main(image_path=image_path, crop_output_dir=cache_dir / "crop")
    image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
        full_bgr=crop_result["inference_bgr"],
        save_preprocess_dir=cache_dir / "preprocess",
    )
    outputs = run_inference(model=model, image_tensor=image_tensor, mask_tensor=mask_tensor, device=device)
    minutiae_rows = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=score_threshold,
        apply_nms=True,
    )

    save_minutiae_csv(minutiae_rows, files["minutiae_csv"])
    orientation_npy, ridge_period_npy = save_pose_sidecars(outputs, cache_dir)
    _save_mask_png(mask_tensor, files["mask_png"])

    metadata = {
        "record": asdict(record),
        "minutiae_count": len(minutiae_rows),
        "minutia_score_threshold": float(score_threshold),
        "crop": {
            "crop_bbox_xyxy": list(crop_result["crop_bbox"]),
            "crop_mode": crop_result["crop_mode"],
            "fallback_reason": crop_result["fallback_reason"],
            "original_shape_hw": [int(crop_result["full_bgr"].shape[0]), int(crop_result["full_bgr"].shape[1])],
            "cropped_shape_hw": [int(crop_result["cropped_bgr"].shape[0]), int(crop_result["cropped_bgr"].shape[1])],
            "coarse_mask_pixels": int(crop_result["coarse_mask_pixels"]),
            "distal_mask_pixels": int(crop_result["distal_mask_pixels"]),
        },
        "artifacts": {
            "minutiae_csv": str(files["minutiae_csv"].resolve()),
            "mask_png": str(files["mask_png"].resolve()),
            "orientation_npy": str(orientation_npy.resolve()),
            "ridge_period_npy": str(ridge_period_npy.resolve()),
        },
    }
    files["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExtractedImage(
        record=record,
        minutiae_csv=files["minutiae_csv"],
        mask_png=files["mask_png"],
        orientation_npy=orientation_npy,
        ridge_period_npy=ridge_period_npy,
        metadata_json=files["metadata_json"],
        minutiae_count=len(minutiae_rows),
    )


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def describe_scores(scores: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "histogram_20": [],
            "histogram_20_bin_edges": np.linspace(0.0, 1.0, 21).tolist(),
        }
    percentiles = np.percentile(values, [1, 5, 25, 75, 95, 99])
    hist, edges = np.histogram(values, bins=20, range=(0.0, 1.0))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "histogram_20": hist.astype(int).tolist(),
        "histogram_20_bin_edges": edges.astype(float).tolist(),
    }


def threshold_metrics(genuine_scores: Iterable[float], impostor_scores: Iterable[float]) -> dict[str, Any]:
    genuine = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        return {
            "eer": None,
            "eer_threshold": None,
            "far_at_eer_threshold": None,
            "frr_at_eer_threshold": None,
            "tar_at_far": {},
            "recommended_threshold": None,
        }

    thresholds = np.unique(np.concatenate((genuine, impostor)))
    thresholds = np.concatenate((thresholds, np.array([np.inf], dtype=np.float64)))
    points: list[dict[str, float]] = []
    for threshold in thresholds:
        tar = float(np.mean(genuine >= threshold))
        far = float(np.mean(impostor >= threshold))
        frr = float(1.0 - tar)
        points.append({"threshold": float(threshold), "tar": tar, "far": far, "frr": frr})

    best = min(points, key=lambda point: (abs(point["far"] - point["frr"]), point["threshold"]))
    tar_at_far: dict[str, Any] = {}
    for target in FAR_TARGETS:
        valid = [point for point in points if point["far"] <= target]
        if not valid:
            tar_at_far[f"{target:.3g}"] = {"tar": None, "threshold": None, "far": None}
            continue
        best_target = max(valid, key=lambda point: (point["tar"], -point["threshold"]))
        tar_at_far[f"{target:.3g}"] = {
            "tar": float(best_target["tar"]),
            "threshold": _finite_or_none(best_target["threshold"]),
            "far": float(best_target["far"]),
        }

    eer_threshold = _finite_or_none(best["threshold"])
    return {
        "eer": float((best["far"] + best["frr"]) / 2.0),
        "eer_threshold": eer_threshold,
        "far_at_eer_threshold": float(best["far"]),
        "frr_at_eer_threshold": float(best["frr"]),
        "tar_at_far": tar_at_far,
        "recommended_threshold": eer_threshold,
    }


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    genuine = [float(row["score"]) for row in rows if row.get("label") == "genuine" and row.get("status") == "ok"]
    impostor = [float(row["score"]) for row in rows if row.get("label") == "impostor" and row.get("status") == "ok"]
    return {
        "genuine": describe_scores(genuine),
        "impostor": describe_scores(impostor),
        "thresholds": threshold_metrics(genuine, impostor),
    }


def summarize_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(row[field]) for row in rows if row.get("status") == "ok" and row.get(field) not in {None, ""}})
    return {value: summarize_scores([row for row in rows if str(row.get(field)) == value]) for value in values}


def _pair_row_base(pair: PairSpec) -> dict[str, Any]:
    return {
        "label": pair.label,
        "dataset": pair.a.dataset,
        "finger_id": pair.a.finger_id,
        "same_acquisition": str(pair.same_acquisition).lower(),
        "a_dataset": pair.a.dataset,
        "a_subject_id": pair.a.subject_id,
        "a_finger_id": pair.a.finger_id,
        "a_acquisition_id": pair.a.acquisition_id,
        "a_view_index": pair.a.view_index,
        "a_image_path": pair.a.image_path,
        "b_dataset": pair.b.dataset,
        "b_subject_id": pair.b.subject_id,
        "b_finger_id": pair.b.finger_id,
        "b_acquisition_id": pair.b.acquisition_id,
        "b_view_index": pair.b.view_index,
        "b_image_path": pair.b.image_path,
    }


def score_pairs(
    pairs: list[PairSpec],
    *,
    cache_root: Path,
    model: Any,
    device: Any,
    score_threshold: float,
    reuse_cache: bool,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import main as mcc_main

    extraction_cache: dict[str, ExtractedImage] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def get_extracted(record: ImageRecord) -> ExtractedImage:
        cached = extraction_cache.get(record.image_path)
        if cached is None:
            cached = extract_image(
                record,
                cache_root=cache_root,
                model=model,
                device=device,
                score_threshold=score_threshold,
                reuse_cache=reuse_cache,
            )
            extraction_cache[record.image_path] = cached
        return cached

    for index, pair in enumerate(pairs, start=1):
        row = _pair_row_base(pair)
        try:
            extracted_a = get_extracted(pair.a)
            extracted_b = get_extracted(pair.b)
            score, sim_matrix = mcc_main.match_minutiae_csv(
                path_a=extracted_a.minutiae_csv,
                path_b=extracted_b.minutiae_csv,
                method=method,
                mask_path_a=extracted_a.mask_png,
                mask_path_b=extracted_b.mask_png,
                orientation_path_a=extracted_a.orientation_npy,
                orientation_path_b=extracted_b.orientation_npy,
                ridge_period_path_a=extracted_a.ridge_period_npy,
                ridge_period_path_b=extracted_b.ridge_period_npy,
                overlap_mode="auto",
            )
            row.update(
                {
                    "status": "ok",
                    "score": float(score),
                    "method": method,
                    "similarity_matrix_shape": "x".join(str(dim) for dim in np.asarray(sim_matrix).shape),
                    "a_minutiae_count": extracted_a.minutiae_count,
                    "b_minutiae_count": extracted_b.minutiae_count,
                    "a_minutiae_csv": str(extracted_a.minutiae_csv.resolve()),
                    "b_minutiae_csv": str(extracted_b.minutiae_csv.resolve()),
                    "a_mask_png": str(extracted_a.mask_png.resolve()),
                    "b_mask_png": str(extracted_b.mask_png.resolve()),
                    "error": "",
                }
            )
        except Exception as exc:
            error = {
                "pair_index": index,
                "label": pair.label,
                "a_image_path": pair.a.image_path,
                "b_image_path": pair.b.image_path,
                "error": str(exc),
            }
            errors.append(error)
            row.update(
                {
                    "status": "error",
                    "score": "",
                    "method": method,
                    "similarity_matrix_shape": "",
                    "a_minutiae_count": "",
                    "b_minutiae_count": "",
                    "a_minutiae_csv": "",
                    "b_minutiae_csv": "",
                    "a_mask_png": "",
                    "b_mask_png": "",
                    "error": str(exc),
                }
            )
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(pairs):
            ok_count = sum(1 for item in rows if item["status"] == "ok")
            print(f"[progress] scored {index}/{len(pairs)} pairs ok={ok_count} errors={len(errors)}")
    return rows, errors


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_score_csv(path: Path, rows: list[dict[str, Any]], label: str) -> None:
    score_rows = [
        {
            "score": row["score"],
            "dataset": row["dataset"],
            "finger_id": row["finger_id"],
            "same_acquisition": row["same_acquisition"],
            "a_subject_id": row["a_subject_id"],
            "b_subject_id": row["b_subject_id"],
            "a_acquisition_id": row["a_acquisition_id"],
            "b_acquisition_id": row["b_acquisition_id"],
            "b_view_index": row["b_view_index"],
        }
        for row in rows
        if row.get("label") == label and row.get("status") == "ok"
    ]
    _write_csv_rows(path, score_rows)


def _write_threshold_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    thresholds = overall["thresholds"]
    genuine = overall["genuine"]
    impostor = overall["impostor"]
    lines = [
        "Matching Threshold Estimation Report",
        "",
        f"Recommended threshold (EER): {thresholds.get('recommended_threshold')}",
        f"EER: {thresholds.get('eer')}",
        f"FAR at threshold: {thresholds.get('far_at_eer_threshold')}",
        f"FRR at threshold: {thresholds.get('frr_at_eer_threshold')}",
        "",
        f"Genuine pairs scored: {genuine.get('count')}",
        f"Genuine median/mean: {genuine.get('median')} / {genuine.get('mean')}",
        f"Impostor pairs scored: {impostor.get('count')}",
        f"Impostor median/mean: {impostor.get('median')} / {impostor.get('mean')}",
        "",
        "TAR at FAR targets:",
    ]
    for target, item in thresholds.get("tar_at_far", {}).items():
        lines.append(f"- FAR <= {target}: TAR={item.get('tar')} threshold={item.get('threshold')} observed_far={item.get('far')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate an MCC matching threshold by matching front contactless images against side images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Dataset root to include. Repeatable. Defaults to dataset/DS1, dataset/DS2, and dataset/DS3.",
    )
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH, help="FeatureNet checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", type=str, default="LSA-R", help="MCC method passed to main.match_minutiae_csv.")
    parser.add_argument("--side-views", type=int, nargs="+", default=[1, 2], help="Side view indices to match against view 0.")
    parser.add_argument("--minutia-score-threshold", type=float, default=0.6, help="FeatureNet minutia score threshold.")
    parser.add_argument("--max-genuine-pairs", type=int, default=None, help="Maximum genuine pairs to score. Default: unlimited.")
    parser.add_argument("--max-impostor-pairs", type=int, default=50000, help="Maximum impostor pairs to sample and score.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for pair sampling.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: match_outputs/threshold_estimation_<timestamp>.")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True, help="Reuse complete cached image extractions.")
    parser.add_argument("--save-pair-csv", action=argparse.BooleanOptionalAction, default=True, help="Write pairs.csv with per-pair details.")
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()

    dataset_roots = [path.resolve() for path in (args.dataset_root or list(DEFAULT_DATASET_ROOTS))]
    weights_path = args.weights_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir().resolve()
    cache_root = output_dir / "cache"
    side_views = {int(view) for view in args.side_views}
    rng = np.random.default_rng(int(args.seed))

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if args.max_genuine_pairs is not None and args.max_genuine_pairs < 0:
        raise ValueError("--max-genuine-pairs must be non-negative")
    if args.max_impostor_pairs < 0:
        raise ValueError("--max-impostor-pairs must be non-negative")
    if 0 in side_views:
        raise ValueError("--side-views must not include front view 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    records = discover_raw_images(dataset_roots)
    if not records:
        raise RuntimeError(f"no raw .jpg images discovered under: {[str(path) for path in dataset_roots]}")

    genuine_pairs = build_genuine_pairs(records, side_views, rng, args.max_genuine_pairs)
    impostor_pairs = build_impostor_pairs(records, side_views, rng, args.max_impostor_pairs)
    if not genuine_pairs:
        raise RuntimeError("no genuine front-vs-side pairs were found")
    if not impostor_pairs:
        raise RuntimeError("no impostor front-vs-side pairs were found")

    print(f"Discovered raw images: {len(records)}")
    print(f"Genuine pairs to score: {len(genuine_pairs)}")
    print(f"Impostor pairs to score: {len(impostor_pairs)}")
    print(f"Output directory: {output_dir}")

    device = _resolve_device(args.device)
    model = load_checkpoint_model(weights_path, device)

    all_pairs = genuine_pairs + impostor_pairs
    pair_rows, errors = score_pairs(
        all_pairs,
        cache_root=cache_root,
        model=model,
        device=device,
        score_threshold=float(args.minutia_score_threshold),
        reuse_cache=bool(args.reuse_cache),
        method=str(args.method),
    )

    if args.save_pair_csv:
        _write_csv_rows(output_dir / "pairs.csv", pair_rows)
    _write_score_csv(output_dir / "scores_genuine.csv", pair_rows, "genuine")
    _write_score_csv(output_dir / "scores_impostor.csv", pair_rows, "impostor")

    ok_rows = [row for row in pair_rows if row.get("status") == "ok"]
    summary = {
        "config": {
            "dataset_roots": [str(path) for path in dataset_roots],
            "weights_path": str(weights_path),
            "device": str(device),
            "method": str(args.method),
            "side_views": sorted(side_views),
            "minutia_score_threshold": float(args.minutia_score_threshold),
            "max_genuine_pairs": args.max_genuine_pairs,
            "max_impostor_pairs": int(args.max_impostor_pairs),
            "seed": int(args.seed),
            "reuse_cache": bool(args.reuse_cache),
            "save_pair_csv": bool(args.save_pair_csv),
        },
        "counts": {
            "raw_image_count": len(records),
            "planned_genuine_pair_count": len(genuine_pairs),
            "planned_impostor_pair_count": len(impostor_pairs),
            "scored_pair_count": len(ok_rows),
            "error_pair_count": len(errors),
            "scored_genuine_pair_count": sum(1 for row in ok_rows if row["label"] == "genuine"),
            "scored_impostor_pair_count": sum(1 for row in ok_rows if row["label"] == "impostor"),
        },
        "overall": summarize_scores(pair_rows),
        "by_finger_id": summarize_by_field(pair_rows, "finger_id"),
        "by_dataset": summarize_by_field(pair_rows, "dataset"),
        "errors": errors[:200],
        "error_count_total": len(errors),
        "outputs": {
            "output_dir": str(output_dir),
            "cache_dir": str(cache_root),
            "pairs_csv": str(output_dir / "pairs.csv") if args.save_pair_csv else None,
            "scores_genuine_csv": str(output_dir / "scores_genuine.csv"),
            "scores_impostor_csv": str(output_dir / "scores_impostor.csv"),
            "summary_json": str(output_dir / "summary.json"),
            "threshold_report_txt": str(output_dir / "threshold_report.txt"),
        },
        "wall_seconds": round(time.time() - started_at, 3),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_threshold_report(output_dir / "threshold_report.txt", summary)

    threshold = summary["overall"]["thresholds"]["recommended_threshold"]
    eer = summary["overall"]["thresholds"]["eer"]
    print(f"Recommended EER threshold: {threshold}")
    print(f"EER: {eer}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(f"Saved report: {output_dir / 'threshold_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
