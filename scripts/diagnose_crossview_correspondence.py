"""Diagnose why cross-view minutia repeatability is stuck (~0.11).

For genuine cross-view pairs from the held-out validation split, this script
measures the minutia counterpart rate (same tolerance as pair_eval's
``repeatability_rate``) under progressively stronger alignments, for both the
MODEL's decoded minutiae and the GT labels transported through the same path:

  frame      alignment            question answered
  ---------  -------------------  ------------------------------------------
  unwarped   centroid (current)   reproduces the ~0.11 pair_eval baseline
  unwarped   ridge similarity     does the deployed LSA-CENTROID transform help?
  unwarped   RANSAC similarity    best achievable with ANY global similarity
  raw        centroid / RANSAC    is the gradient unwarp helping or hurting?
  (GT rows through the same four cells = label-consistency ceiling)

Decision gate:
  * model RANSAC >> model centroid  -> transport/alignment problem (fix matcher
    alignment; no retraining needed).
  * model RANSAC still ~0.1 AND GT ceiling also low -> labels are not
    view-consistent -> rebuild GT with cross-view consensus, then retrain.
  * GT ceiling high but model RANSAC low -> detection itself is view-dependent
    -> training-side fix (cross-view supervision).

Rates are additionally split by same-vs-cross acquisition and by view combo
(front-side vs side-side), because within-acquisition GT shares a consensus
reconstruction while cross-acquisition GT does not.

IMPORTANT: the same cells are also computed for IMPOSTOR pairs (different
fingers). At 25 px / 30 deg tolerance and typical minutiae densities the
CHANCE-LEVEL counterpart rate is roughly 0.1 even between unrelated point sets
(verified synthetically), i.e. the observed 0.11 repeatability may carry no
signal at all. Only the genuine-minus-impostor gap per cell is meaningful.

Example (GPU box):
  python scripts/diagnose_crossview_correspondence.py \
    --ground-truth-root ground_truth/merged_DS123 \
    --weights-path runs/featurenet_v5_supervised_consistency/best.pt \
    --max-pairs 100 --output-dir match_outputs/crossview_diag_v5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from featurenet.models.infer import (  # noqa: E402  (imports ensure_stdlib_copy shim transitively)
    decode_minutiae_rows,
    load_checkpoint_model,
    preprocess_saved_masked_input,
    run_inference,
    unwarp_minutiae_rows,
    _resolve_device,
)
from featurenet.models.pair_eval import build_pairs, _angle_diff  # noqa: E402
from featurenet.models.train import load_bundle_samples, split_samples  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pose_normalization as pose_norm  # noqa: E402


# ---------------------------------------------------------------------------
# Alignment primitives
# ---------------------------------------------------------------------------


def _rows_to_arrays(rows: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    ang = np.array([float(r.get("angle", r.get("theta", 0.0))) for r in rows], dtype=np.float64)
    return pts, ang


def _counterpart_rate(
    pts_a: np.ndarray,
    ang_a: np.ndarray,
    pts_b: np.ndarray,
    ang_b: np.ndarray,
    *,
    dist_px: float,
    angle_rad: float,
) -> float | None:
    """Symmetric counterpart rate; assumes the two sets are already aligned."""
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return None

    def _one_way(src_pts, src_ang, dst_pts, dst_ang) -> float:
        hits = 0
        for index in range(src_pts.shape[0]):
            deltas = dst_pts - src_pts[index]
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            for cand in np.nonzero(distances <= dist_px)[0]:
                if _angle_diff(float(src_ang[index]), float(dst_ang[cand])) <= angle_rad:
                    hits += 1
                    break
        return float(hits) / float(src_pts.shape[0])

    return (_one_way(pts_a, ang_a, pts_b, ang_b) + _one_way(pts_b, ang_b, pts_a, ang_a)) / 2.0


def _apply_similarity(
    pts: np.ndarray,
    ang: np.ndarray,
    *,
    scale: float,
    rotation: float,
    tx: float,
    ty: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    x = scale * (cos_r * pts[:, 0] - sin_r * pts[:, 1]) + tx
    y = scale * (sin_r * pts[:, 0] + cos_r * pts[:, 1]) + ty
    return np.stack([x, y], axis=1), ang + rotation


def align_centroid(
    pts_a: np.ndarray, ang_a: np.ndarray, pts_b: np.ndarray, ang_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Translate B's centroid onto A's centroid (what pair_eval does today)."""
    shift = pts_a.mean(axis=0) - pts_b.mean(axis=0)
    return pts_b + shift, ang_b, {"scale": 1.0, "rotation_deg": 0.0}


def align_ridge_similarity(
    pts_a: np.ndarray,
    ang_a: np.ndarray,
    pts_b: np.ndarray,
    ang_b: np.ndarray,
    *,
    orientation_a: np.ndarray | None,
    orientation_b: np.ndarray | None,
    ridge_a: np.ndarray | None,
    ridge_b: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rotation from ridge-orientation sidecars + scale from ridge spacing,
    applied about the centroids — the LSA-CENTROID transform recipe."""
    rotation = 0.0
    if orientation_a is not None and orientation_b is not None:
        try:
            theta_a = pose_norm.estimate_global_orientation_from_field(orientation_a)
            theta_b = pose_norm.estimate_global_orientation_from_field(orientation_b)
            rotation = pose_norm.wrap_angle_pi(float(theta_a) - float(theta_b))
        except Exception:
            rotation = 0.0
    scale = 1.0
    if ridge_a is not None and ridge_b is not None:
        try:
            spacing_a = pose_norm.estimate_median_ridge_spacing(ridge_a)
            spacing_b = pose_norm.estimate_median_ridge_spacing(ridge_b)
            scale = pose_norm.estimate_scale_from_ridge_spacing(
                query_spacing=spacing_b, template_spacing=spacing_a
            )
        except Exception:
            scale = 1.0

    centroid_a = pts_a.mean(axis=0)
    centroid_b = pts_b.mean(axis=0)
    centered = pts_b - centroid_b
    rotated, ang_out = _apply_similarity(
        centered, ang_b, scale=scale, rotation=rotation, tx=0.0, ty=0.0
    )
    return rotated + centroid_a, ang_out, {"scale": float(scale), "rotation_deg": math.degrees(rotation)}


def align_ransac_similarity(
    pts_a: np.ndarray,
    ang_a: np.ndarray,
    pts_b: np.ndarray,
    ang_b: np.ndarray,
    *,
    dist_px: float,
    angle_rad: float,
    iterations: int = 3000,
    min_segment_px: float = 30.0,
    scale_range: tuple[float, float] = (0.6, 1.6),
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Best global similarity found by RANSAC over segment correspondences.

    Hypotheses: pick two points in A and two in B with compatible segment
    lengths; the segment pair fully determines a similarity transform. The
    inlier count is the number of A-minutiae with a transformed-B counterpart
    within (dist_px, angle_rad). This upper-bounds what ANY global-similarity
    prealignment (centroid, ridge, canonical) could achieve.
    """
    n_a, n_b = pts_a.shape[0], pts_b.shape[0]
    if n_a < 2 or n_b < 2:
        aligned, ang_out, _ = align_centroid(pts_a, ang_a, pts_b, ang_b)
        return aligned, ang_out, {"scale": 1.0, "rotation_deg": 0.0, "inliers": 0}

    rng = random.Random(seed)
    best_inliers = -1
    best_params = (1.0, 0.0, 0.0, 0.0)  # scale, rotation, tx, ty

    def _count_inliers(params: tuple[float, float, float, float]) -> int:
        scale, rotation, tx, ty = params
        moved, moved_ang = _apply_similarity(pts_b, ang_b, scale=scale, rotation=rotation, tx=tx, ty=ty)
        inliers = 0
        for index in range(n_a):
            deltas = moved - pts_a[index]
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            for cand in np.nonzero(distances <= dist_px)[0]:
                if _angle_diff(float(ang_a[index]), float(moved_ang[cand])) <= angle_rad:
                    inliers += 1
                    break
        return inliers

    for _ in range(iterations):
        i, j = rng.sample(range(n_a), 2)
        k, l = rng.sample(range(n_b), 2)
        seg_a = pts_a[j] - pts_a[i]
        seg_b = pts_b[l] - pts_b[k]
        len_a = float(np.hypot(seg_a[0], seg_a[1]))
        len_b = float(np.hypot(seg_b[0], seg_b[1]))
        if len_a < min_segment_px or len_b < min_segment_px:
            continue
        scale = len_a / len_b
        if not (scale_range[0] <= scale <= scale_range[1]):
            continue
        rotation = math.atan2(seg_a[1], seg_a[0]) - math.atan2(seg_b[1], seg_b[0])
        # Minutia angles must roughly agree with the segment rotation.
        if _angle_diff(float(ang_b[k]) + rotation, float(ang_a[i])) > angle_rad:
            continue
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        tx = pts_a[i, 0] - scale * (cos_r * pts_b[k, 0] - sin_r * pts_b[k, 1])
        ty = pts_a[i, 1] - scale * (sin_r * pts_b[k, 0] + cos_r * pts_b[k, 1])
        params = (scale, rotation, tx, ty)
        inliers = _count_inliers(params)
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = params

    scale, rotation, tx, ty = best_params
    moved, moved_ang = _apply_similarity(pts_b, ang_b, scale=scale, rotation=rotation, tx=tx, ty=ty)
    return moved, moved_ang, {
        "scale": float(scale),
        "rotation_deg": math.degrees(rotation),
        "inliers": int(max(best_inliers, 0)),
    }


# ---------------------------------------------------------------------------
# Per-sample extraction (model rows + GT rows, raw and unwarped frames)
# ---------------------------------------------------------------------------


def _load_gt_rows(minutiae_path: Path) -> list[dict[str, float]]:
    if not minutiae_path.is_file():
        return []
    payload = json.loads(minutiae_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, float]] = []
    for item in payload:
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        angle = item.get("theta", item.get("angle", 0.0))
        try:
            angle = float(angle)
        except (TypeError, ValueError):
            angle = 0.0
        rows.append({"x": x, "y": y, "angle": angle, "score": float(item.get("score", 1.0))})
    return rows


def _load_optional_npy(path: Any) -> np.ndarray | None:
    try:
        path = Path(path)
    except TypeError:
        return None
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


@torch.no_grad()
def extract_sample_views(
    model: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    top_k: int | None,
) -> dict[str, Any] | None:
    try:
        image_tensor, mask_tensor, input_shape_hw = preprocess_saved_masked_input(
            masked_image_path=Path(sample["masked_image"]),
            mask_path=Path(sample["mask"]),
        )
        outputs = run_inference(model, image_tensor, mask_tensor, device)
    except Exception as exc:
        print(f"[extract] {sample.get('sample_id')}: inference failed: {exc}", flush=True)
        return None

    model_rows_raw = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=score_threshold,
        apply_nms=True,
        top_k=top_k,
    )
    gt_rows_raw = _load_gt_rows(Path(sample["minutiae_path"]))

    gray_full = image_tensor.detach().cpu().numpy()[0, 0]

    def _unwarp(rows: list[dict[str, float]]) -> tuple[list[dict[str, float]], str]:
        status: dict[str, str] = {}
        warped, _ = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=outputs["gradient"],
            mask_tensor=mask_tensor,
            input_shape_hw=input_shape_hw,
            gray_image=gray_full,
            status_out=status,
        )
        return warped, status.get("status", "unknown")

    model_rows_unwarped, unwarp_status = _unwarp(model_rows_raw)
    gt_rows_unwarped, gt_unwarp_status = _unwarp(gt_rows_raw)

    return {
        "sample_id": sample.get("sample_id"),
        "raw_view_index": int(sample.get("raw_view_index", -1)),
        "acquisition_id": sample.get("acquisition_id"),
        "model_raw": model_rows_raw,
        "model_unwarped": model_rows_unwarped if unwarp_status == "ok" else None,
        "gt_raw": gt_rows_raw,
        "gt_unwarped": gt_rows_unwarped if gt_unwarp_status == "ok" else None,
        "unwarp_status": unwarp_status,
        "orientation": _load_optional_npy(sample.get("orientation_path")),
        "ridge_period": _load_optional_npy(sample.get("ridge_period_path")),
    }


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

CELLS = (
    ("model", "raw", "centroid"),
    ("model", "raw", "ridge"),
    ("model", "raw", "ransac"),
    ("model", "unwarped", "centroid"),
    ("model", "unwarped", "ridge"),
    ("model", "unwarped", "ransac"),
    ("gt", "raw", "centroid"),
    ("gt", "raw", "ransac"),
    ("gt", "unwarped", "centroid"),
    ("gt", "unwarped", "ransac"),
)


def evaluate_pair_cells(
    ex_a: Mapping[str, Any],
    ex_b: Mapping[str, Any],
    *,
    dist_px: float,
    angle_rad: float,
    ransac_iterations: int,
    seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for source, frame, alignment in CELLS:
        key = f"{source}_{frame}_{alignment}"
        rows_a = ex_a.get(f"{source}_{frame}")
        rows_b = ex_b.get(f"{source}_{frame}")
        if rows_a is None or rows_b is None or not rows_a or not rows_b:
            row[key] = ""
            continue
        pts_a, ang_a = _rows_to_arrays(rows_a)
        pts_b, ang_b = _rows_to_arrays(rows_b)
        extras: dict[str, float] = {}
        if alignment == "centroid":
            aligned_b, aligned_ang_b, extras = align_centroid(pts_a, ang_a, pts_b, ang_b)
        elif alignment == "ridge":
            aligned_b, aligned_ang_b, extras = align_ridge_similarity(
                pts_a,
                ang_a,
                pts_b,
                ang_b,
                orientation_a=ex_a.get("orientation"),
                orientation_b=ex_b.get("orientation"),
                ridge_a=ex_a.get("ridge_period"),
                ridge_b=ex_b.get("ridge_period"),
            )
            # ridge transform is rotation/scale about centroids; recenter onto A
        else:
            aligned_b, aligned_ang_b, extras = align_ransac_similarity(
                pts_a,
                ang_a,
                pts_b,
                ang_b,
                dist_px=dist_px,
                angle_rad=angle_rad,
                iterations=ransac_iterations,
                seed=seed,
            )
        rate = _counterpart_rate(
            pts_a, ang_a, aligned_b, aligned_ang_b, dist_px=dist_px, angle_rad=angle_rad
        )
        row[key] = "" if rate is None else round(float(rate), 4)
        if alignment == "ransac":
            row[f"{key}_rotation_deg"] = round(float(extras.get("rotation_deg", 0.0)), 2)
            row[f"{key}_scale"] = round(float(extras.get("scale", 1.0)), 4)
    return row


def _summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"group": label, "pairs": len(rows)}
    for source, frame, alignment in CELLS:
        key = f"{source}_{frame}_{alignment}"
        values = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        summary[key] = round(float(np.mean(values)), 4) if values else ""
    return summary


def build_same_view_pairs(
    samples: Sequence[Mapping[str, Any]],
    *,
    max_genuine: int,
    max_impostor: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Same-pose pairs: genuine = same identity, SAME view index, different
    acquisition; impostor = different identity, same view index. This is the
    case the cross-view pair builder never produces (it has no front x front),
    and it isolates capture-to-capture detection repeatability from cross-view
    geometry."""
    from featurenet.models.pair_eval import _identity_key, _view_index

    by_identity: dict[Any, list[int]] = {}
    for index, sample in enumerate(samples):
        by_identity.setdefault(_identity_key(sample), []).append(index)

    genuine: list[tuple[int, int]] = []
    for indices in by_identity.values():
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                a, b = indices[i_pos], indices[j_pos]
                if _view_index(samples[a]) != _view_index(samples[b]):
                    continue
                if samples[a].get("acquisition_id") == samples[b].get("acquisition_id"):
                    continue
                genuine.append((a, b))

    rng = random.Random(seed)
    rng.shuffle(genuine)
    if max_genuine > 0:
        genuine = genuine[:max_genuine]

    by_view: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_view.setdefault(_view_index(sample), []).append(index)
    impostor: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(impostor) < max_impostor and attempts < max(1000, max_impostor * 500):
        attempts += 1
        view = rng.choice(sorted(by_view.keys()))
        candidates = by_view[view]
        if len(candidates) < 2:
            continue
        a, b = rng.sample(candidates, 2)
        if _identity_key(samples[a]) == _identity_key(samples[b]):
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            continue
        seen.add(pair)
        impostor.append(pair)
    return genuine, impostor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-pairs", type=int, default=100, help="Genuine cross-view pairs to evaluate.")
    parser.add_argument(
        "--pair-mode",
        choices=("cross-view", "same-view", "all"),
        default="cross-view",
        help=(
            "cross-view: front x side / side x side (pair_eval's genuine set). "
            "same-view: same view index across acquisitions (front x front etc.) — "
            "isolates capture repeatability from cross-view geometry. all: both."
        ),
    )
    parser.add_argument("--seed", type=int, default=13, help="Must match training seed to reproduce the val split.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Must match training val fraction. Pass 0 to use ALL samples as the pair pool (ad-hoc diagnostics).",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minutia decode threshold (pair_eval default).")
    parser.add_argument("--minutia-top-k", type=int, default=None)
    parser.add_argument("--dist-px", type=float, default=25.0, help="Counterpart distance tolerance (pair_eval default).")
    parser.add_argument("--angle-deg", type=float, default=30.0, help="Counterpart angle tolerance (pair_eval default).")
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model = load_checkpoint_model(args.weights_path.resolve(), device)

    samples = load_bundle_samples(args.ground_truth_root)
    if float(args.val_fraction) > 0.0:
        _, val_samples = split_samples(samples, val_fraction=float(args.val_fraction), seed=int(args.seed))
    else:
        val_samples = samples
    print(f"[diag] loaded {len(samples)} samples, {len(val_samples)} in the pair pool", flush=True)

    genuine_pairs: list[tuple[int, int]] = []
    impostor_pairs: list[tuple[int, int]] = []
    if args.pair_mode in ("cross-view", "all"):
        cross_genuine, cross_impostor = build_pairs(
            val_samples,
            max_genuine=int(args.max_pairs),
            max_impostor=int(args.max_pairs),
            seed=int(args.seed),
        )
        genuine_pairs.extend(cross_genuine)
        impostor_pairs.extend(cross_impostor)
    if args.pair_mode in ("same-view", "all"):
        same_genuine, same_impostor = build_same_view_pairs(
            val_samples,
            max_genuine=int(args.max_pairs),
            max_impostor=int(args.max_pairs),
            seed=int(args.seed) + 1,
        )
        genuine_pairs.extend(same_genuine)
        impostor_pairs.extend(same_impostor)
    if not genuine_pairs:
        raise SystemExit(f"no genuine pairs could be formed from the pool (pair mode {args.pair_mode})")
    print(
        f"[diag] evaluating {len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor pairs "
        f"(mode {args.pair_mode})",
        flush=True,
    )

    angle_rad = math.radians(float(args.angle_deg))
    extracted: dict[int, dict[str, Any] | None] = {}

    def _get(index: int) -> dict[str, Any] | None:
        if index not in extracted:
            extracted[index] = extract_sample_views(
                model,
                val_samples[index],
                device,
                score_threshold=float(args.score_threshold),
                top_k=args.minutia_top_k,
            )
        return extracted[index]

    pair_rows: list[dict[str, Any]] = []
    all_pairs = [("genuine", pair) for pair in genuine_pairs] + [
        ("impostor", pair) for pair in impostor_pairs
    ]
    for pair_index, (label, (a, b)) in enumerate(all_pairs, start=1):
        ex_a = _get(a)
        ex_b = _get(b)
        if ex_a is None or ex_b is None:
            continue
        view_a, view_b = ex_a["raw_view_index"], ex_b["raw_view_index"]
        if view_a == view_b:
            view_combo = "same_view"
        elif 0 in (view_a, view_b):
            view_combo = "front_side"
        else:
            view_combo = "side_side"
        view_pair = f"v{min(view_a, view_b)}v{max(view_a, view_b)}"
        same_acquisition = (
            label == "genuine"
            and ex_a.get("acquisition_id") is not None
            and ex_a.get("acquisition_id") == ex_b.get("acquisition_id")
        )
        row: dict[str, Any] = {
            "pair_id": f"pair_{pair_index:05d}",
            "label": label,
            "a_sample_id": ex_a["sample_id"],
            "b_sample_id": ex_b["sample_id"],
            "a_view": view_a,
            "b_view": view_b,
            "view_combo": view_combo,
            "view_pair": view_pair,
            "same_acquisition": str(same_acquisition).lower(),
            "a_model_count": len(ex_a["model_raw"]),
            "b_model_count": len(ex_b["model_raw"]),
            "a_gt_count": len(ex_a["gt_raw"]),
            "b_gt_count": len(ex_b["gt_raw"]),
            "a_unwarp_status": ex_a["unwarp_status"],
            "b_unwarp_status": ex_b["unwarp_status"],
        }
        row.update(
            evaluate_pair_cells(
                ex_a,
                ex_b,
                dist_px=float(args.dist_px),
                angle_rad=angle_rad,
                ransac_iterations=int(args.ransac_iterations),
                seed=int(args.seed),
            )
        )
        pair_rows.append(row)
        if pair_index == 1 or pair_index % 10 == 0 or pair_index == len(all_pairs):
            print(f"[diag] processed {pair_index}/{len(all_pairs)} pairs", flush=True)

    if not pair_rows:
        raise SystemExit("no pairs could be evaluated (all extractions failed)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = output_dir / "crossview_pair_rates.csv"
    fieldnames: list[str] = []
    for row in pair_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with pairs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        label = row["label"]
        prefix = "gen" if label == "genuine" else "imp"
        groups[label].append(row)
        groups[f"{prefix}_{row['view_combo']}"].append(row)
        groups[f"{prefix}_{row['view_pair']}"].append(row)
        if label == "genuine":
            groups["gen_same_acq" if row["same_acquisition"] == "true" else "gen_cross_acq"].append(row)

    ordered_labels = [key for key in ("genuine", "impostor") if key in groups]
    ordered_labels += sorted(key for key in groups if key not in ("genuine", "impostor"))
    summaries = [_summarize(groups[label], label) for label in ordered_labels]
    summary_csv = output_dir / "crossview_summary.csv"
    summary_fields = ["group", "pairs"] + [f"{s}_{f}_{a}" for s, f, a in CELLS]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in summary_fields})

    print("\n=== Mean counterpart rates (tolerance "
          f"{args.dist_px:.0f}px / {args.angle_deg:.0f}deg) ===")
    header = f"{'group':<16}{'pairs':>6}  " + "".join(f"{s[:2]}_{f[:3]}_{a[:4]:<8}" for s, f, a in CELLS)
    print(header)
    for summary in summaries:
        cells = "".join(
            f"{summary.get(f'{s}_{f}_{a}', ''):<12}" if summary.get(f"{s}_{f}_{a}", "") != "" else f"{'--':<12}"
            for s, f, a in CELLS
        )
        print(f"{summary['group']:<16}{summary['pairs']:>6}  {cells}")

    print(f"\nSaved per-pair rates: {pairs_csv}")
    print(f"Saved summary:        {summary_csv}")
    print(
        "\nHow to read this:\n"
        "  Compare every 'genuine' cell against the matching 'impostor' cell --\n"
        "  the impostor row is the CHANCE FLOOR at this tolerance (~0.1 for RANSAC\n"
        "  on unrelated sets); only the genuine-minus-impostor gap is signal.\n"
        "  model_unwarped_centroid  ~ the flat 0.11 repeatability_rate baseline\n"
        "  model_unwarped_ransac    = ceiling under ANY global similarity alignment\n"
        "  gt_*_ransac              = label-consistency ceiling (model cannot beat it)\n"
        "  ransac >> centroid (genuine only) -> alignment/transport problem (fix matcher)\n"
        "  gt ceiling ~= impostor floor      -> labels not view-consistent (rebuild GT + retrain)\n"
        "  gt high, model low                -> detections view-dependent (training-side fix)"
    )


if __name__ == "__main__":
    main()
