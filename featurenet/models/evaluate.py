from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .feature_extractor import FeatureExtractor
from .losses import FeatureNetLoss
from .train import (
    LOSS_KEYS,
    _move_targets_to_device,
    _resolve_device,
    create_dataloader,
    evaluate,
    load_bundle_samples,
    split_samples,
)


PAPER_DISTANCE_THRESHOLD_PIXELS = 8.0
PAPER_ANGLE_THRESHOLD_DEGREES = 15.0
MINUTIA_SUBCELL_BINS = 8
MINUTIA_ORIENTATION_BINS = 360
POSE_SPLITS = ("all", "front", "side")


def _init_score_stats(score_thresholds: list[float]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        f"{threshold:.2f}": {
            split: {"tp": 0, "fp": 0, "fn": 0}
            for split in POSE_SPLITS
        }
        for threshold in score_thresholds
    }


def _resolve_hw_for_sample(
    hw_tensor: torch.Tensor | None,
    sample_index: int,
    fallback_hw: tuple[int, int],
) -> tuple[int, int]:
    if hw_tensor is None:
        return fallback_hw
    try:
        if hw_tensor.dim() == 1 and hw_tensor.numel() >= 2:
            values = hw_tensor[:2]
        elif hw_tensor.dim() >= 2 and hw_tensor.shape[0] > sample_index and hw_tensor.shape[1] >= 2:
            values = hw_tensor[sample_index, :2]
        else:
            return fallback_hw
        height = int(values[0].item())
        width = int(values[1].item())
        if height <= 0 or width <= 0:
            return fallback_hw
        return height, width
    except (TypeError, ValueError, RuntimeError):
        return fallback_hw


def _resolve_raw_view_index(raw_view_tensor: torch.Tensor | None, sample_index: int) -> int:
    if raw_view_tensor is None:
        return -1
    try:
        if raw_view_tensor.dim() == 0:
            return int(raw_view_tensor.item())
        if raw_view_tensor.dim() >= 1 and raw_view_tensor.shape[0] > sample_index:
            return int(raw_view_tensor[sample_index].item())
    except (TypeError, ValueError, RuntimeError):
        return -1
    return -1


def _split_keys_for_raw_view(raw_view_index: int) -> tuple[str, ...]:
    if raw_view_index == 0:
        return ("all", "front")
    if raw_view_index in {1, 2}:
        return ("all", "side")
    return ("all",)


def _decode_minutia_points(
    score_map: torch.Tensor,
    x_bins: torch.Tensor,
    y_bins: torch.Tensor,
    orientation_bins: torch.Tensor,
    eval_mask: torch.Tensor,
    threshold: float,
    apply_nms: bool,
    input_shape_hw: tuple[int, int],
    output_shape_hw: tuple[int, int],
    inclusive_threshold: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inclusive_threshold:
        active = (score_map >= threshold) & eval_mask
    else:
        active = (score_map > threshold) & eval_mask
    if apply_nms:
        pooled = torch.nn.functional.max_pool2d(
            score_map.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(0).squeeze(0)
        active = active & (score_map >= (pooled - 1e-8))

    indices = torch.nonzero(active, as_tuple=False)
    if indices.numel() == 0:
        return score_map.new_zeros((0, 2)), score_map.new_zeros((0,))

    rows = indices[:, 0]
    cols = indices[:, 1]
    x_labels = x_bins.long().clamp(0, MINUTIA_SUBCELL_BINS - 1)
    y_labels = y_bins.long().clamp(0, MINUTIA_SUBCELL_BINS - 1)
    ori_labels = orientation_bins.long().clamp(0, MINUTIA_ORIENTATION_BINS - 1)

    sub_x = (x_labels[rows, cols].float() + 0.5) / float(MINUTIA_SUBCELL_BINS)
    sub_y = (y_labels[rows, cols].float() + 0.5) / float(MINUTIA_SUBCELL_BINS)
    x_out = cols.float() + sub_x
    y_out = rows.float() + sub_y

    scale_x = float(input_shape_hw[1]) / max(float(output_shape_hw[1]), 1.0)
    scale_y = float(input_shape_hw[0]) / max(float(output_shape_hw[0]), 1.0)
    x_in = x_out * scale_x
    y_in = y_out * scale_y
    points = torch.stack((x_in, y_in), dim=1)

    angles = (ori_labels[rows, cols].float() + 0.5) * (360.0 / float(MINUTIA_ORIENTATION_BINS))
    angles = torch.remainder(angles, 360.0)
    return points, angles


def _max_bipartite_match_count(adjacency: list[list[int]], right_size: int) -> int:
    if not adjacency or right_size <= 0:
        return 0
    match_right = [-1] * right_size

    def _try_match(left_index: int, seen: list[bool]) -> bool:
        for right_index in adjacency[left_index]:
            if seen[right_index]:
                continue
            seen[right_index] = True
            if match_right[right_index] == -1 or _try_match(match_right[right_index], seen):
                match_right[right_index] = left_index
                return True
        return False

    matches = 0
    for left_index in range(len(adjacency)):
        seen = [False] * right_size
        if _try_match(left_index, seen):
            matches += 1
    return matches


def _count_valid_matches(
    pred_points: torch.Tensor,
    pred_angles: torch.Tensor,
    gt_points: torch.Tensor,
    gt_angles: torch.Tensor,
) -> int:
    if pred_points.numel() == 0 or gt_points.numel() == 0:
        return 0

    distances = torch.cdist(pred_points, gt_points, p=2)
    angular_diff = torch.abs(pred_angles[:, None] - gt_angles[None, :])
    angular_diff = torch.minimum(angular_diff, 360.0 - angular_diff)
    valid = (distances < PAPER_DISTANCE_THRESHOLD_PIXELS) & (angular_diff < PAPER_ANGLE_THRESHOLD_DEGREES)

    adjacency: list[list[int]] = [[] for _ in range(valid.shape[0])]
    for pred_index, gt_index in torch.nonzero(valid, as_tuple=False).tolist():
        adjacency[pred_index].append(gt_index)

    return _max_bipartite_match_count(adjacency, right_size=valid.shape[1])


def _stats_with_rates(stats: dict[str, int]) -> dict[str, float | int]:
    tp = int(stats["tp"])
    fp = int(stats["fp"])
    fn = int(stats["fn"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def compute_validation_metrics(
    model: FeatureExtractor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresholds: list[float],
    target_threshold: float,
    amp: bool = False,
    channels_last: bool = False,
) -> dict[str, Any]:
    score_stats = _init_score_stats(score_thresholds)
    argmax_stats = {
        "minutia_x": {"correct": 0, "total": 0},
        "minutia_y": {"correct": 0, "total": 0},
        "minutia_orientation": {"correct": 0, "total": 0},
    }

    model.eval()
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        if channels_last and inputs.dim() == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        targets = _move_targets_to_device(targets, device)

        image = inputs[:, :1]
        mask = inputs[:, 1:2]
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
        with autocast_ctx:
            outputs = model(image, mask=mask)

        eval_mask = targets["mask"] > 0.5
        if eval_mask.dim() == 4:
            eval_mask = eval_mask.squeeze(1)
        gt_score = targets["minutia_score"]
        if gt_score.dim() == 4:
            gt_score = gt_score.squeeze(1)
        pred_score = torch.sigmoid(outputs["minutia_score"])
        if pred_score.dim() == 4:
            pred_score = pred_score.squeeze(1)

        pred_x_bins = outputs["minutia_x"].argmax(dim=1)
        pred_y_bins = outputs["minutia_y"].argmax(dim=1)
        pred_ori_bins = outputs["minutia_orientation"].argmax(dim=1)
        gt_x_bins = targets["minutia_x"]
        gt_y_bins = targets["minutia_y"]
        gt_ori_bins = targets["minutia_orientation"]

        raw_view_tensor = targets.get("raw_view_index")
        input_shape_hw_tensor = targets.get("input_shape_hw")
        output_shape_hw_tensor = targets.get("output_shape_hw")

        batch_size = pred_score.shape[0]
        for sample_index in range(batch_size):
            sample_eval_mask = eval_mask[sample_index]
            fallback_output_shape = (int(sample_eval_mask.shape[0]), int(sample_eval_mask.shape[1]))
            sample_output_shape_hw = _resolve_hw_for_sample(
                output_shape_hw_tensor,
                sample_index,
                fallback_output_shape,
            )
            sample_input_shape_hw = _resolve_hw_for_sample(
                input_shape_hw_tensor,
                sample_index,
                sample_output_shape_hw,
            )
            sample_splits = _split_keys_for_raw_view(_resolve_raw_view_index(raw_view_tensor, sample_index))

            gt_points, gt_angles = _decode_minutia_points(
                score_map=gt_score[sample_index].float(),
                x_bins=gt_x_bins[sample_index],
                y_bins=gt_y_bins[sample_index],
                orientation_bins=gt_ori_bins[sample_index],
                eval_mask=sample_eval_mask,
                threshold=target_threshold,
                apply_nms=False,
                input_shape_hw=sample_input_shape_hw,
                output_shape_hw=sample_output_shape_hw,
                inclusive_threshold=False,
            )

            for threshold in score_thresholds:
                key = f"{threshold:.2f}"
                pred_points, pred_angles = _decode_minutia_points(
                    score_map=pred_score[sample_index],
                    x_bins=pred_x_bins[sample_index],
                    y_bins=pred_y_bins[sample_index],
                    orientation_bins=pred_ori_bins[sample_index],
                    eval_mask=sample_eval_mask,
                    threshold=threshold,
                    apply_nms=True,
                    input_shape_hw=sample_input_shape_hw,
                    output_shape_hw=sample_output_shape_hw,
                    inclusive_threshold=True,
                )
                tp = _count_valid_matches(pred_points, pred_angles, gt_points, gt_angles)
                fp = int(pred_points.shape[0]) - tp
                fn = int(gt_points.shape[0]) - tp
                for split in sample_splits:
                    stats = score_stats[key][split]
                    stats["tp"] += tp
                    stats["fp"] += fp
                    stats["fn"] += fn

        positive_mask = targets["minutia_valid_mask"] > 0.5
        for head_name in ("minutia_x", "minutia_y", "minutia_orientation"):
            pred_label = outputs[head_name].argmax(dim=1)
            target_label = targets[head_name]
            active = positive_mask.squeeze(1) if positive_mask.dim() == 4 else positive_mask
            stats = argmax_stats[head_name]
            stats["correct"] += int(((pred_label == target_label) & active).sum().item())
            stats["total"] += int(active.sum().item())

    score_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    best_threshold = None
    best_f1 = -1.0
    for key, per_split in score_stats.items():
        split_payload: dict[str, dict[str, float | int]] = {}
        for split_name in POSE_SPLITS:
            split_payload[split_name] = _stats_with_rates(per_split[split_name])
        score_metrics[key] = split_payload
        all_f1 = float(split_payload["all"]["f1"])
        if all_f1 > best_f1:
            best_f1 = all_f1
            best_threshold = key

    label_metrics = {}
    for head_name, stats in argmax_stats.items():
        total = stats["total"]
        label_metrics[head_name] = {
            "correct": stats["correct"],
            "total": total,
            "accuracy": (stats["correct"] / total) if total > 0 else 0.0,
        }

    return {
        "score_thresholds": score_metrics,
        "best_score_threshold": best_threshold,
        "best_score_f1": best_f1,
        "label_accuracy": label_metrics,
    }


def _load_checkpoint(path: Path, device: torch.device) -> tuple[FeatureExtractor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FeatureExtractor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _default_thresholds() -> list[float]:
    return [round(value, 2) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FeatureNet checkpoint on DS1 validation data.")
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-threshold", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    model, checkpoint = _load_checkpoint(args.checkpoint_path, device)

    checkpoint_args = checkpoint.get("args", {})
    val_fraction = args.val_fraction if args.val_fraction is not None else checkpoint_args.get("val_fraction", 0.2)
    seed = args.seed if args.seed is not None else checkpoint_args.get("seed", 13)
    ground_truth_root = args.ground_truth_root

    samples = load_bundle_samples(ground_truth_root, limit=args.limit)
    if val_fraction > 0.0:
        try:
            _, val_samples = split_samples(samples, val_fraction=val_fraction, seed=seed)
        except ValueError:
            val_samples = samples
    else:
        val_samples = samples
    if not val_samples:
        val_samples = samples
    dataloader = create_dataloader(
        samples=val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    criterion = FeatureNetLoss().to(device)
    use_amp = bool(args.amp) and device.type == "cuda"
    loss_metrics = evaluate(model, dataloader, criterion, device, amp=use_amp, channels_last=args.channels_last)
    metric_payload = compute_validation_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        score_thresholds=_default_thresholds(),
        target_threshold=args.target_threshold,
        amp=use_amp,
        channels_last=args.channels_last,
    )

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(device),
        "ground_truth_root": str(ground_truth_root),
        "amp": use_amp,
        "pin_memory": bool(args.pin_memory) and device.type == "cuda",
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "sample_count": len(samples),
        "val_sample_count": len(val_samples),
        "seed": seed,
        "val_fraction": val_fraction,
        "loss": {key: loss_metrics[key] for key in LOSS_KEYS},
        "metrics": metric_payload,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
