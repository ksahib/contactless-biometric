from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import preprocess


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20, 0.25)
VIEW_ROLES = {0: "front", 1: "side_left", 2: "side_right"}
FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<finger>\d+)_(?P<acquisition>\d+)_(?P<view>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Sample:
    path: Path
    subject: int
    finger: int
    acquisition: int
    view: int
    role: str


def _parse_sample(path: Path) -> Sample | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    view = int(match.group("view"))
    role = VIEW_ROLES.get(view)
    if role is None:
        return None
    return Sample(
        path=path,
        subject=int(match.group("subject")),
        finger=int(match.group("finger")),
        acquisition=int(match.group("acquisition")),
        view=view,
        role=role,
    )


def _iter_dataset_samples(dataset_root: Path) -> Iterable[Sample]:
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        sample = _parse_sample(path)
        if sample is not None:
            yield sample


def select_samples(dataset_root: Path) -> list[Sample]:
    selected: list[Sample] = []
    role_limits = {"front": 3, "side_left": 2, "side_right": 1}
    role_counts = {role: 0 for role in role_limits}

    for sample in _iter_dataset_samples(dataset_root):
        if role_counts[sample.role] >= role_limits[sample.role]:
            continue
        selected.append(sample)
        role_counts[sample.role] += 1
        if all(role_counts[role] >= limit for role, limit in role_limits.items()):
            break

    missing = {
        role: limit - role_counts[role]
        for role, limit in role_limits.items()
        if role_counts[role] < limit
    }
    if missing:
        raise RuntimeError(f"not enough DS1 samples for mask sweep: {missing}")
    return selected


def _sample_stem(sample: Sample) -> str:
    return f"{sample.role}_{sample.path.stem}"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    overlay = np.where(mask_u8[:, :, None] > 0, cv2.addWeighted(overlay, 0.65, color, 0.35, 0), overlay)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=3)
    return overlay


def _threshold_dir(output_root: Path, threshold: float) -> Path:
    return output_root / f"threshold_{threshold:.2f}"


def run_sweep(
    *,
    dataset_root: Path,
    output_root: Path,
    thresholds: tuple[float, ...],
    device: str | None,
    model_config: Path | None,
    checkpoint: Path | None,
) -> dict[str, object]:
    samples = select_samples(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)

    overall: dict[str, object] = {
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "thresholds": [float(threshold) for threshold in thresholds],
        "samples": [
            {
                **asdict(sample),
                "path": str(sample.path.resolve()),
            }
            for sample in samples
        ],
    }

    for threshold in thresholds:
        threshold_root = _threshold_dir(output_root, threshold)
        threshold_root.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, object]] = []

        for sample in samples:
            image_bgr = _load_bgr(sample.path)
            sample_root = threshold_root / _sample_stem(sample)
            sample_root.mkdir(parents=True, exist_ok=True)
            _write_image(sample_root / "original.png", image_bgr)

            record: dict[str, object] = {
                "sample_id": sample.path.stem,
                "image_path": str(sample.path.resolve()),
                "view": int(sample.view),
                "view_role": sample.role,
                "threshold": float(threshold),
                "success": False,
            }

            try:
                enhanced, mask = preprocess.segment_then_clahe(
                    image_bgr,
                    score_thr=float(threshold),
                    device=device,
                    model_config=model_config,
                    checkpoint=checkpoint,
                )
                mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
                masked_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                overlay = _mask_overlay(image_bgr, mask_u8)

                _write_image(sample_root / "mask.png", mask_u8)
                _write_image(sample_root / "masked_enhanced.png", masked_bgr)
                _write_image(sample_root / "overlay.png", overlay)

                record.update(
                    {
                        "success": True,
                        "mask_area_ratio": float(np.count_nonzero(mask_u8) / mask_u8.size),
                        "bbox_xyxy": _mask_bbox(mask_u8),
                    }
                )
            except Exception as exc:
                record["error"] = str(exc)

            results.append(record)

        summary = {
            "threshold": float(threshold),
            "success_count": int(sum(bool(result["success"]) for result in results)),
            "failure_count": int(sum(not bool(result["success"]) for result in results)),
            "results": results,
        }
        (threshold_root / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        overall[f"threshold_{threshold:.2f}"] = {
            "summary_path": str((threshold_root / "summary.json").resolve()),
            "success_count": summary["success_count"],
            "failure_count": summary["failure_count"],
        }

    (output_root / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return overall


def _parse_thresholds(values: list[str] | None) -> tuple[float, ...]:
    if not values:
        return DEFAULT_THRESHOLDS
    thresholds = tuple(float(value) for value in values)
    if any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError("thresholds must be positive")
    return thresholds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep SOLOv2 mask thresholds on a small DS1 front/side sample set.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "DS1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "solov2_mask_sweep")
    parser.add_argument("--device", default=None, help="SOLOv2 device, e.g. cpu or cuda:0.")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", action="append", default=None, help="Threshold to test. Repeat to override defaults.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_sweep(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        thresholds=_parse_thresholds(args.threshold),
        device=args.device,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
