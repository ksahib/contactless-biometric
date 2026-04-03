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
    score_stats = {
        f"{threshold:.2f}": {"tp": 0, "fp": 0, "fn": 0}
        for threshold in score_thresholds
    }
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
        gt_positive = (targets["minutia_score"] > target_threshold) & eval_mask
        pred_score = torch.sigmoid(outputs["minutia_score"])

        for threshold in score_thresholds:
            key = f"{threshold:.2f}"
            pred_positive = (pred_score >= threshold) & eval_mask
            stats = score_stats[key]
            stats["tp"] += int((pred_positive & gt_positive).sum().item())
            stats["fp"] += int((pred_positive & (~gt_positive) & eval_mask).sum().item())
            stats["fn"] += int(((~pred_positive) & gt_positive).sum().item())

        positive_mask = targets["minutia_valid_mask"] > 0.5
        for head_name in ("minutia_x", "minutia_y", "minutia_orientation"):
            pred_label = outputs[head_name].argmax(dim=1)
            target_label = targets[head_name]
            active = positive_mask.squeeze(1) if positive_mask.dim() == 4 else positive_mask
            stats = argmax_stats[head_name]
            stats["correct"] += int(((pred_label == target_label) & active).sum().item())
            stats["total"] += int(active.sum().item())

    score_metrics: dict[str, dict[str, float | int]] = {}
    best_threshold = None
    best_f1 = -1.0
    for key, stats in score_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        score_metrics[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if f1 > best_f1:
            best_f1 = f1
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
