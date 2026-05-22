#!/usr/bin/env python
"""Compare pyfing and FingerFlow minutiae quality on normal vs unwarped images."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = REPO_ROOT / "ground_truth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "minutiae_unwarped_quality_compare"
DEFAULT_FINGERFLOW_MODEL_DIR = REPO_ROOT / ".fingerflow_models"
DEFAULT_DPI = 500
VARIANTS = ("normal", "unwarped")
EXTRACTORS = ("pyfing", "fingerflow")


def _path_from_sys_path_entry(entry: str) -> Path:
    return Path(entry or ".").resolve()


def _hide_repo_root_for_third_party_imports() -> None:
    """Prevent top-level repo modules such as copy.py from shadowing stdlib."""
    sys.path = [entry for entry in sys.path if _path_from_sys_path_entry(entry) != REPO_ROOT]


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_hide_repo_root_for_third_party_imports()
_ensure_stdlib_copy_module()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    gt_run_root: Path
    acquisition_id: str
    sample_id: str
    normal_image: Path
    unwarped_image: Path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_report_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _safe_pair_id(gt_run_root: Path, acquisition_id: str) -> str:
    return f"{gt_run_root.name}_{acquisition_id}"


def discover_pairs(gt_root: Path, limit: int | None = None) -> list[PairSpec]:
    """Find reconstruction outputs with matching front-view preprocessed samples."""
    gt_root = gt_root.resolve()
    if not gt_root.exists():
        raise FileNotFoundError(f"ground-truth root not found: {gt_root}")

    pairs: list[PairSpec] = []
    for unwarped_image in sorted(gt_root.glob("*/reconstructions/*/center_unwarped_enhanced.png")):
        reconstruction_dir = unwarped_image.parent
        gt_run_root = reconstruction_dir.parents[1]
        acquisition_id = reconstruction_dir.name
        sample_id = f"{acquisition_id}_v00"
        normal_image = gt_run_root / "samples" / sample_id / "preprocessed_input.png"
        if not normal_image.exists():
            continue
        pairs.append(
            PairSpec(
                pair_id=_safe_pair_id(gt_run_root, acquisition_id),
                gt_run_root=gt_run_root,
                acquisition_id=acquisition_id,
                sample_id=sample_id,
                normal_image=normal_image,
                unwarped_image=unwarped_image,
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise RuntimeError(
            "no existing normal/unwarped pairs found. Expected files like "
            "ground_truth/<run>/reconstructions/<sample>/center_unwarped_enhanced.png "
            "and ground_truth/<run>/samples/<sample>_v00/preprocessed_input.png"
        )
    return pairs


def _parse_extractors(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(EXTRACTORS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown extractor(s): {', '.join(invalid)}. Expected comma-separated subset of {', '.join(EXTRACTORS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one extractor must be selected")
    return selected


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _summarize_scores(scores: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return {
            "mean_score": 0.0,
            "median_score": 0.0,
            "score_p25": 0.0,
            "score_p75": 0.0,
        }
    return {
        "mean_score": float(np.mean(values)),
        "median_score": float(np.median(values)),
        "score_p25": float(np.percentile(values, 25)),
        "score_p75": float(np.percentile(values, 75)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_minutiae_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    minutiae = payload.get("minutiae", [])
    return [row for row in minutiae if isinstance(row, dict)]


def _load_minutiae_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    return rows


def _standardize_pyfing_minutiae(raw_minutiae: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_minutiae:
        quality = getattr(item, "quality", None)
        rows.append(
            {
                "x": _float_or_default(getattr(item, "x", 0.0)),
                "y": _float_or_default(getattr(item, "y", 0.0)),
                "angle": _float_or_default(getattr(item, "direction", getattr(item, "angle", 0.0))),
                "score": _float_or_default(quality, 1.0),
                "type": str(getattr(item, "type", "")),
            }
        )
    return rows


def _extract_pyfing(image_path: Path, case_dir: Path) -> list[dict[str, Any]]:
    minutiae_json = case_dir / "minutiae.json"
    minutiae_csv = case_dir / "minutiae.csv"
    if minutiae_json.exists() and minutiae_csv.exists():
        return _load_minutiae_json(minutiae_json)

    try:
        import pyfing
    except Exception as exc:
        raise RuntimeError(
            "pyfing import failed. In this repo's current Windows venv this may be the Keras/optree issue; "
            "repair dependencies first, for example by running `.venv\\Scripts\\python.exe -m pip install optree`, "
            "then rerun this script."
        ) from exc

    rows = _standardize_pyfing_minutiae(pyfing.minutiae_extraction(_read_gray(image_path), dpi=DEFAULT_DPI))
    _write_json(minutiae_json, {"image": str(image_path.resolve()), "minutiae": rows})
    _write_csv(minutiae_csv, ["x", "y", "angle", "score", "type"], rows)
    return rows


def _import_main_module():
    _ensure_stdlib_copy_module()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import main as mcc_main

    return mcc_main


def _extract_fingerflow(
    image_path: Path,
    case_dir: Path,
    model_paths: tuple[Path, Path, Path, Path],
    mcc_main: Any,
) -> list[dict[str, Any]]:
    minutiae_json = case_dir / "minutiae.json"
    minutiae_csv = case_dir / "minutiae.csv"
    core_csv = case_dir / "core.csv"
    if minutiae_json.exists() and minutiae_csv.exists():
        return _load_minutiae_csv(minutiae_csv)

    case_dir.mkdir(parents=True, exist_ok=True)
    mcc_main.extract_minutiae_with_fingerflow(
        image_path.resolve(),
        image_path.resolve(),
        model_paths,
        minutiae_json.resolve(),
        minutiae_csv.resolve(),
        core_csv.resolve(),
    )
    return _load_minutiae_csv(minutiae_csv)


def _preflight(selected_extractors: list[str], fingerflow_model_dir: Path) -> dict[str, Any]:
    context: dict[str, Any] = {"main": None, "fingerflow_model_paths": None}
    if "pyfing" in selected_extractors:
        try:
            import pyfing  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "pyfing import failed during preflight. If this is the known Keras/optree dependency error, "
                "run `.venv\\Scripts\\python.exe -m pip install optree` inside this repo venv and try again."
            ) from exc

    if "fingerflow" in selected_extractors:
        try:
            mcc_main = _import_main_module()
            context["main"] = mcc_main
            context["fingerflow_model_paths"] = mcc_main.ensure_fingerflow_models(fingerflow_model_dir)
        except Exception as exc:
            raise RuntimeError(f"FingerFlow preflight failed: {exc}") from exc
    return context


def _score_values(minutiae: list[dict[str, Any]]) -> list[float]:
    return [_float_or_default(row.get("score"), 0.0) for row in minutiae]


def _image_result_row(
    pair: PairSpec,
    extractor: str,
    variant: str,
    image_path: Path,
    case_dir: Path,
    minutiae: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = _score_values(minutiae)
    summary = _summarize_scores(scores)
    row: dict[str, Any] = {
        "pair_id": pair.pair_id,
        "gt_run": pair.gt_run_root.name,
        "acquisition_id": pair.acquisition_id,
        "sample_id": pair.sample_id,
        "extractor": extractor,
        "variant": variant,
        "image_path": _repo_relative(image_path),
        "case_dir": _repo_relative(case_dir),
        "minutiae_count": int(len(minutiae)),
    }
    row.update(summary)
    return row


def _group_summary(rows: list[dict[str, Any]], extractor: str, variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["extractor"] == extractor and row["variant"] == variant]
    counts = [int(row["minutiae_count"]) for row in selected]
    per_image_means = [float(row["mean_score"]) for row in selected]
    pooled_scores: list[float] = []
    for row in selected:
        minutiae_path = _resolve_report_path(str(row["case_dir"])) / "minutiae.csv"
        if not minutiae_path.exists():
            minutiae_path = _resolve_report_path(str(row["case_dir"])) / "minutiae.json"
        minutiae = _load_minutiae_csv(minutiae_path) if minutiae_path.suffix == ".csv" else _load_minutiae_json(minutiae_path)
        pooled_scores.extend(_score_values(minutiae))

    return {
        "extractor": extractor,
        "variant": variant,
        "image_count": int(len(selected)),
        "mean_minutiae_count": float(statistics.fmean(counts)) if counts else 0.0,
        "median_minutiae_count": float(statistics.median(counts)) if counts else 0.0,
        "pooled_mean_score": float(statistics.fmean(pooled_scores)) if pooled_scores else 0.0,
        "mean_per_image_score": float(statistics.fmean(per_image_means)) if per_image_means else 0.0,
        "median_per_image_score": float(statistics.median(per_image_means)) if per_image_means else 0.0,
    }


def _comparison_deltas(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_key = {(row["extractor"], row["variant"]): row for row in summary_rows}
    deltas: dict[str, dict[str, float]] = {}
    for extractor in sorted({row["extractor"] for row in summary_rows}):
        normal = by_key.get((extractor, "normal"))
        unwarped = by_key.get((extractor, "unwarped"))
        if normal is None or unwarped is None:
            continue
        normal_count = float(normal["mean_minutiae_count"])
        unwarped_count = float(unwarped["mean_minutiae_count"])
        deltas[extractor] = {
            "unwarped_minus_normal_mean_count": unwarped_count - normal_count,
            "unwarped_minus_normal_mean_score": float(unwarped["mean_per_image_score"]) - float(normal["mean_per_image_score"]),
            "percent_count_change": ((unwarped_count - normal_count) / normal_count * 100.0) if normal_count else 0.0,
        }
    return deltas


def run(
    gt_root: Path,
    output_dir: Path,
    selected_extractors: list[str],
    fingerflow_model_dir: Path,
    limit: int | None,
    discovery_only: bool,
) -> dict[str, Any]:
    pairs = discover_pairs(gt_root, limit=limit)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if discovery_only:
        payload = {
            "gt_root": str(gt_root.resolve()),
            "pair_count": len(pairs),
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "normal_image": str(pair.normal_image.resolve()),
                    "unwarped_image": str(pair.unwarped_image.resolve()),
                }
                for pair in pairs
            ],
        }
        _write_json(output_dir / "discovery.json", payload)
        return payload

    preflight = _preflight(selected_extractors, fingerflow_model_dir.resolve())
    mcc_main = preflight["main"]
    model_paths = preflight["fingerflow_model_paths"]

    per_image_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for variant, image_path in (("normal", pair.normal_image), ("unwarped", pair.unwarped_image)):
            for extractor in selected_extractors:
                case_dir = output_dir / pair.pair_id / variant / extractor
                if extractor == "pyfing":
                    minutiae = _extract_pyfing(image_path, case_dir)
                elif extractor == "fingerflow":
                    minutiae = _extract_fingerflow(image_path, case_dir, model_paths, mcc_main)
                else:
                    raise ValueError(f"unsupported extractor: {extractor}")
                per_image_rows.append(_image_result_row(pair, extractor, variant, image_path, case_dir, minutiae))

    per_image_fields = [
        "pair_id",
        "gt_run",
        "acquisition_id",
        "sample_id",
        "extractor",
        "variant",
        "image_path",
        "case_dir",
        "minutiae_count",
        "mean_score",
        "median_score",
        "score_p25",
        "score_p75",
    ]
    _write_csv(output_dir / "per_image_results.csv", per_image_fields, per_image_rows)

    summary_rows = [
        _group_summary(per_image_rows, extractor, variant)
        for extractor in selected_extractors
        for variant in VARIANTS
    ]
    summary_fields = [
        "extractor",
        "variant",
        "image_count",
        "mean_minutiae_count",
        "median_minutiae_count",
        "pooled_mean_score",
        "mean_per_image_score",
        "median_per_image_score",
    ]
    _write_csv(output_dir / "summary.csv", summary_fields, summary_rows)

    report = {
        "gt_root": str(gt_root.resolve()),
        "output_dir": str(output_dir),
        "pair_count": len(pairs),
        "extractors": selected_extractors,
        "variants": list(VARIANTS),
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "gt_run": pair.gt_run_root.name,
                "acquisition_id": pair.acquisition_id,
                "sample_id": pair.sample_id,
                "normal_image": str(pair.normal_image.resolve()),
                "unwarped_image": str(pair.unwarped_image.resolve()),
            }
            for pair in pairs
        ],
        "summary": summary_rows,
        "deltas": _comparison_deltas(summary_rows),
        "artifacts": {
            "per_image_results_csv": str((output_dir / "per_image_results.csv").resolve()),
            "summary_csv": str((output_dir / "summary.csv").resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
        },
    }
    _write_json(output_dir / "summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fingerflow-model-dir", type=Path, default=DEFAULT_FINGERFLOW_MODEL_DIR)
    parser.add_argument("--extractors", type=_parse_extractors, default=list(EXTRACTORS))
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="Only discover normal/unwarped pairs and write discovery.json; do not run dependency preflight or extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(
        gt_root=args.gt_root,
        output_dir=args.output_dir,
        selected_extractors=args.extractors,
        fingerflow_model_dir=args.fingerflow_model_dir,
        limit=args.limit,
        discovery_only=args.discovery_only,
    )
    if args.discovery_only:
        print(json.dumps({"pair_count": report["pair_count"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    else:
        print(
            json.dumps(
                {
                    "pair_count": report["pair_count"],
                    "summary_json": report["artifacts"]["summary_json"],
                    "summary_csv": report["artifacts"]["summary_csv"],
                    "deltas": report["deltas"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
