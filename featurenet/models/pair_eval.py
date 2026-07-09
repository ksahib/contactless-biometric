"""Held-out-fingers pair AUC/EER + matcher-free repeatability evaluation.

This module builds genuine/impostor pairs from the held-out validation samples
(which are identity-disjoint from training because ``split_samples`` groups by
``finger_class_id``), extracts + Route-A-unwarps minutiae per sample, scores each
pair with the deployed MCC matcher, and reports:

* ``pair_auc``  - Mann-Whitney AUC of genuine vs impostor match scores.
* ``pair_eer``  - equal error rate from the FAR/FRR crossing.
* ``repeatability_rate`` - matcher-free symmetric minutia re-detection rate over
  genuine pairs (centroid-aligned canonical minutiae within distance/angle tol).

The exact same ``unwarp_minutiae_rows`` helper used by ``match_infer.py`` is used
here, so the selection metric reflects the matcher that is actually deployed.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .infer import (
    decode_minutiae_rows,
    preprocess_saved_masked_input,
    run_inference,
    save_minutiae_csv,
    unwarp_minutiae_rows,
)

try:  # pragma: no cover - exercised only when OpenCV is installed
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def _identity_key(sample: Mapping[str, Any]) -> Any:
    """Disjoint identity key, mirroring train._split_group_key fallbacks."""
    finger_class_id = sample.get("finger_class_id")
    if finger_class_id is not None:
        return ("finger_class_id", finger_class_id)
    subject_id = sample.get("subject_id")
    finger_id = sample.get("finger_id")
    if subject_id is not None and finger_id is not None:
        return ("subject_finger", subject_id, finger_id)
    parent_sample_id = sample.get("parent_sample_id")
    if parent_sample_id:
        return ("parent_sample_id", parent_sample_id)
    return ("sample_id", sample.get("sample_id"))


def _view_index(sample: Mapping[str, Any]) -> int:
    try:
        return int(sample.get("raw_view_index", -1))
    except (TypeError, ValueError):
        return -1


def _save_mask_array_png(mask_array: np.ndarray, path: Path) -> None:
    if cv2 is None:
        raise RuntimeError("pair_eval mask saving requires opencv-python")
    mask_img = (np.asarray(mask_array) > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask_img):
        raise RuntimeError(f"failed to write mask image to {path}")


def build_pairs(
    samples: Sequence[Mapping[str, Any]],
    *,
    max_genuine: int,
    max_impostor: int,
    seed: int,
    side_views: Sequence[int] = (1, 2),
    front_view: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (genuine_pairs, impostor_pairs) as index pairs into ``samples``.

    Genuine = same identity, front x side and side x side. Impostor = different
    identity. Both are capped and sampled with a fixed seed so the evaluated set
    is stable across epochs.
    """
    side_set = {int(view) for view in side_views}
    by_identity: dict[Any, list[int]] = {}
    for index, sample in enumerate(samples):
        by_identity.setdefault(_identity_key(sample), []).append(index)

    genuine_candidates: list[tuple[int, int]] = []
    for indices in by_identity.values():
        if len(indices) < 2:
            continue
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                a = indices[i_pos]
                b = indices[j_pos]
                view_a = _view_index(samples[a])
                view_b = _view_index(samples[b])
                front_side = (
                    (view_a == front_view and view_b in side_set)
                    or (view_b == front_view and view_a in side_set)
                )
                side_side = view_a in side_set and view_b in side_set
                if front_side or side_side:
                    genuine_candidates.append((a, b))

    identity_keys = list(by_identity.keys())
    rng = random.Random(seed)

    impostor_candidates: list[tuple[int, int]] = []
    if len(identity_keys) >= 2:
        seen: set[tuple[int, int]] = set()
        max_attempts = max(1000, max_impostor * 500)
        attempts = 0
        target_impostor = max_impostor if max_impostor > 0 else 0
        while len(impostor_candidates) < target_impostor and attempts < max_attempts:
            attempts += 1
            key_a, key_b = rng.sample(identity_keys, 2)
            a = rng.choice(by_identity[key_a])
            b = rng.choice(by_identity[key_b])
            pair = (a, b) if a < b else (b, a)
            if pair in seen:
                continue
            seen.add(pair)
            impostor_candidates.append(pair)

    rng.shuffle(genuine_candidates)
    if max_genuine > 0:
        genuine_candidates = genuine_candidates[:max_genuine]
    return genuine_candidates, impostor_candidates


@torch.no_grad()
def _extract_sample(
    model: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    apply_nms: bool,
    unwarp: str,
    cache_dir: Path,
) -> dict[str, Any]:
    """Run forward + decode (+ Route A unwarp) and cache CSV/mask for the sample."""
    masked_image_path = Path(sample["masked_image"])
    mask_path = Path(sample["mask"])
    image_tensor, mask_tensor, input_shape_hw = preprocess_saved_masked_input(
        masked_image_path=masked_image_path,
        mask_path=mask_path,
    )
    outputs = run_inference(model, image_tensor, mask_tensor, device)
    rows = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=score_threshold,
        apply_nms=apply_nms,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    minutiae_csv = cache_dir / "minutiae.csv"
    mask_png = cache_dir / "mask.png"

    unwarp_status = "not_requested"
    if unwarp == "gradient":
        gray_full = image_tensor.detach().cpu().numpy()[0, 0]
        status_out: dict[str, str] = {}
        warped_rows, unwarped_mask = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=outputs["gradient"],
            mask_tensor=mask_tensor,
            input_shape_hw=input_shape_hw,
            gray_image=gray_full,
            status_out=status_out,
        )
        unwarp_status = status_out.get("status", "unknown")
        if unwarp_status == "ok":
            rows = warped_rows
            save_minutiae_csv(rows, minutiae_csv)
            _save_mask_array_png(unwarped_mask, mask_png)
        else:
            save_minutiae_csv(rows, minutiae_csv)
            _save_mask_array_png(mask_tensor.detach().cpu().numpy()[0, 0], mask_png)
    else:
        save_minutiae_csv(rows, minutiae_csv)
        _save_mask_array_png(mask_tensor.detach().cpu().numpy()[0, 0], mask_png)

    return {
        "minutiae_csv": minutiae_csv,
        "mask_png": mask_png,
        "rows": rows,
        "unwarp_status": unwarp_status,
    }


def _mcc_score(
    extract_a: Mapping[str, Any],
    extract_b: Mapping[str, Any],
    *,
    method: str,
) -> float | None:
    import main as mcc_main

    # An unwarped frame must never be matched against a raw frame.
    status_a = str(extract_a.get("unwarp_status", "not_requested"))
    status_b = str(extract_b.get("unwarp_status", "not_requested"))
    if (status_a == "ok") != (status_b == "ok"):
        return None

    try:
        score, _ = mcc_main.match_minutiae_csv(
            path_a=Path(extract_a["minutiae_csv"]),
            path_b=Path(extract_b["minutiae_csv"]),
            method=method,
            mask_path_a=Path(extract_a["mask_png"]),
            mask_path_b=Path(extract_b["mask_png"]),
            overlap_mode="auto",
        )
    except Exception:
        return None
    value = float(score)
    if not math.isfinite(value):
        return None
    return value


def compute_auc(genuine: Sequence[float], impostor: Sequence[float]) -> float:
    """Mann-Whitney AUC: P(genuine > impostor) with 0.5 credit for ties."""
    if not genuine or not impostor:
        return float("nan")
    impostor_sorted = sorted(impostor)
    total = float(len(genuine) * len(impostor))
    wins = 0.0
    for g in genuine:
        lo = _bisect_left(impostor_sorted, g)
        hi = _bisect_right(impostor_sorted, g)
        wins += float(lo) + 0.5 * float(hi - lo)
    return wins / total


def _bisect_left(values: Sequence[float], target: float) -> int:
    low, high = 0, len(values)
    while low < high:
        mid = (low + high) // 2
        if values[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def _bisect_right(values: Sequence[float], target: float) -> int:
    low, high = 0, len(values)
    while low < high:
        mid = (low + high) // 2
        if values[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low


def compute_eer(genuine: Sequence[float], impostor: Sequence[float]) -> float:
    """Equal error rate from the FAR/FRR crossing over candidate thresholds."""
    if not genuine or not impostor:
        return float("nan")
    genuine_arr = np.asarray(sorted(genuine), dtype=np.float64)
    impostor_arr = np.asarray(sorted(impostor), dtype=np.float64)
    thresholds = np.unique(np.concatenate([genuine_arr, impostor_arr]))
    n_gen = float(genuine_arr.size)
    n_imp = float(impostor_arr.size)
    best_eer = 1.0
    best_gap = float("inf")
    for threshold in thresholds:
        far = float(np.count_nonzero(impostor_arr >= threshold)) / n_imp
        frr = float(np.count_nonzero(genuine_arr < threshold)) / n_gen
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2.0
    return float(best_eer)


def _angle_diff(a: float, b: float) -> float:
    diff = abs((a - b) % (2.0 * math.pi))
    return min(diff, 2.0 * math.pi - diff)


def _pair_redetection_rate(
    rows_a: Sequence[Mapping[str, float]],
    rows_b: Sequence[Mapping[str, float]],
    *,
    dist_px: float,
    angle_rad: float,
) -> float | None:
    if not rows_a or not rows_b:
        return None

    pts_a = np.array([[float(r["x"]), float(r["y"])] for r in rows_a], dtype=np.float64)
    pts_b = np.array([[float(r["x"]), float(r["y"])] for r in rows_b], dtype=np.float64)
    ang_a = np.array([float(r.get("angle", 0.0)) for r in rows_a], dtype=np.float64)
    ang_b = np.array([float(r.get("angle", 0.0)) for r in rows_b], dtype=np.float64)

    pts_a = pts_a - pts_a.mean(axis=0, keepdims=True)
    pts_b = pts_b - pts_b.mean(axis=0, keepdims=True)

    def _redetected(source_pts, source_ang, target_pts, target_ang) -> float:
        hits = 0
        for index in range(source_pts.shape[0]):
            deltas = target_pts - source_pts[index]
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            within = np.nonzero(distances <= dist_px)[0]
            matched = False
            for cand in within:
                if _angle_diff(float(source_ang[index]), float(target_ang[cand])) <= angle_rad:
                    matched = True
                    break
            if matched:
                hits += 1
        return float(hits) / float(source_pts.shape[0])

    forward = _redetected(pts_a, ang_a, pts_b, ang_b)
    backward = _redetected(pts_b, ang_b, pts_a, ang_a)
    return (forward + backward) / 2.0


def evaluate_pairs(
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device | str,
    *,
    method: str = "LSA",
    score_threshold: float = 0.5,
    apply_nms: bool = True,
    unwarp: str = "gradient",
    max_genuine: int = 200,
    max_impostor: int = 200,
    seed: int = 13,
    side_views: Sequence[int] = (1, 2),
    repeat_dist_px: float = 25.0,
    repeat_angle_deg: float = 30.0,
) -> dict[str, Any] | None:
    """Evaluate held-out pair AUC/EER + repeatability. Returns ``None`` if no
    genuine pairs can be formed (caller should fall back to another metric)."""
    resolved_device = torch.device(device)
    genuine_pairs, impostor_pairs = build_pairs(
        samples,
        max_genuine=max_genuine,
        max_impostor=max_impostor,
        seed=seed,
        side_views=side_views,
    )
    if not genuine_pairs:
        return None

    was_training = bool(getattr(model, "training", False))
    model.eval()

    extracted: dict[int, dict[str, Any]] = {}

    def _get(index: int, cache_root: Path) -> dict[str, Any] | None:
        if index in extracted:
            return extracted[index]
        try:
            result = _extract_sample(
                model,
                samples[index],
                resolved_device,
                score_threshold=score_threshold,
                apply_nms=apply_nms,
                unwarp=unwarp,
                cache_dir=cache_root / f"sample_{index:05d}",
            )
        except Exception:
            result = None
        extracted[index] = result  # type: ignore[assignment]
        return result

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    repeat_rates: list[float] = []
    angle_rad = math.radians(float(repeat_angle_deg))

    try:
        with tempfile.TemporaryDirectory(prefix="pair_eval_") as tmp_dir:
            cache_root = Path(tmp_dir)
            for a, b in genuine_pairs:
                ex_a = _get(a, cache_root)
                ex_b = _get(b, cache_root)
                if ex_a is None or ex_b is None:
                    continue
                score = _mcc_score(ex_a, ex_b, method=method)
                if score is not None:
                    genuine_scores.append(score)
                rate = _pair_redetection_rate(
                    ex_a["rows"],
                    ex_b["rows"],
                    dist_px=float(repeat_dist_px),
                    angle_rad=angle_rad,
                )
                if rate is not None:
                    repeat_rates.append(rate)
            for a, b in impostor_pairs:
                ex_a = _get(a, cache_root)
                ex_b = _get(b, cache_root)
                if ex_a is None or ex_b is None:
                    continue
                score = _mcc_score(ex_a, ex_b, method=method)
                if score is not None:
                    impostor_scores.append(score)
    finally:
        if was_training:
            model.train()

    pair_auc = compute_auc(genuine_scores, impostor_scores)
    pair_eer = compute_eer(genuine_scores, impostor_scores)
    repeatability_rate = float(np.mean(repeat_rates)) if repeat_rates else float("nan")

    return {
        "pair_auc": float(pair_auc),
        "pair_eer": float(pair_eer),
        "repeatability_rate": repeatability_rate,
        "n_genuine": len(genuine_pairs),
        "n_impostor": len(impostor_pairs),
        "n_genuine_scored": len(genuine_scores),
        "n_impostor_scored": len(impostor_scores),
        "n_repeat_pairs": len(repeat_rates),
        "method": method,
        "unwarp": unwarp,
    }
