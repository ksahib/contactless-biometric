from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

import torch

from .evaluate import (
    _count_valid_matches,
    _decode_minutia_points,
    _default_thresholds,
    _resolve_hw_for_sample,
    _resolve_target_offsets,
    _resolve_target_orientation_degrees,
    _stats_with_rates,
    _orientation_vectors_to_degrees,
    compute_validation_metrics,
)
from .infer import _torch_load_checkpoint
from .feature_extractor import FeatureExtractor
from .losses import FeatureNetLoss
from .train import _resolve_device, create_dataloader, evaluate, load_bundle_samples, split_samples, _move_targets_to_device


def _decode_count_diagnostics(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresholds: list[float],
    target_threshold: float,
    amp: bool = False,
    channels_last: bool = False,
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = {
        f"{threshold:.2f}": {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pred_count_sum": 0.0,
            "gt_count_sum": 0.0,
            "count_abs_error_sum": 0.0,
            "sample_count": 0.0,
        }
        for threshold in score_thresholds
    }

    model.eval()
    with torch.no_grad():
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

            pred_x_offsets = torch.sigmoid(outputs["minutia_x"])
            pred_y_offsets = torch.sigmoid(outputs["minutia_y"])
            if pred_x_offsets.dim() == 4 and pred_x_offsets.shape[1] == 1:
                pred_x_offsets = pred_x_offsets.squeeze(1)
            if pred_y_offsets.dim() == 4 and pred_y_offsets.shape[1] == 1:
                pred_y_offsets = pred_y_offsets.squeeze(1)
            pred_ori_degrees = _orientation_vectors_to_degrees(outputs["minutia_orientation"])
            gt_x_offsets, gt_y_offsets = _resolve_target_offsets(targets)
            gt_ori_degrees = _resolve_target_orientation_degrees(targets)

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

                gt_points, gt_angles = _decode_minutia_points(
                    score_map=gt_score[sample_index].float(),
                    x_offsets=gt_x_offsets[sample_index],
                    y_offsets=gt_y_offsets[sample_index],
                    orientation_degrees_map=gt_ori_degrees[sample_index],
                    eval_mask=sample_eval_mask,
                    threshold=target_threshold,
                    apply_nms=False,
                    input_shape_hw=sample_input_shape_hw,
                    output_shape_hw=sample_output_shape_hw,
                    offsets_are_logits=False,
                    inclusive_threshold=False,
                )
                gt_count = int(gt_points.shape[0])

                for threshold in score_thresholds:
                    key = f"{threshold:.2f}"
                    pred_points, pred_angles = _decode_minutia_points(
                        score_map=pred_score[sample_index],
                        x_offsets=pred_x_offsets[sample_index],
                        y_offsets=pred_y_offsets[sample_index],
                        orientation_degrees_map=pred_ori_degrees[sample_index],
                        eval_mask=sample_eval_mask,
                        threshold=threshold,
                        apply_nms=True,
                        input_shape_hw=sample_input_shape_hw,
                        output_shape_hw=sample_output_shape_hw,
                        offsets_are_logits=False,
                        inclusive_threshold=True,
                    )
                    pred_count = int(pred_points.shape[0])
                    tp = _count_valid_matches(pred_points, pred_angles, gt_points, gt_angles)
                    fp = pred_count - tp
                    fn = gt_count - tp

                    bucket = aggregate[key]
                    bucket["tp"] += float(tp)
                    bucket["fp"] += float(fp)
                    bucket["fn"] += float(fn)
                    bucket["pred_count_sum"] += float(pred_count)
                    bucket["gt_count_sum"] += float(gt_count)
                    bucket["count_abs_error_sum"] += abs(float(pred_count - gt_count))
                    bucket["sample_count"] += 1.0

    payload: dict[str, Any] = {}
    for key, stats in aggregate.items():
        sample_count = max(int(stats["sample_count"]), 1)
        rates = _stats_with_rates(
            {
                "tp": int(stats["tp"]),
                "fp": int(stats["fp"]),
                "fn": int(stats["fn"]),
            }
        )
        payload[key] = {
            **rates,
            "mean_pred_count": stats["pred_count_sum"] / sample_count,
            "mean_gt_count": stats["gt_count_sum"] / sample_count,
            "count_mae": stats["count_abs_error_sum"] / sample_count,
            "count_bias": (stats["pred_count_sum"] - stats["gt_count_sum"]) / sample_count,
            "samples": int(stats["sample_count"]),
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why decoded FeatureNet minutiae underperform in matching.")
    parser.add_argument("--ground-truth-root", type=Path, required=True, help="Path to generated ground truth root.")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("weights") / "best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
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
    checkpoint = _torch_load_checkpoint(args.checkpoint_path, device)
    model = FeatureExtractor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    checkpoint_args = checkpoint.get("args", {})
    val_fraction = args.val_fraction if args.val_fraction is not None else checkpoint_args.get("val_fraction", 0.2)
    seed = args.seed if args.seed is not None else checkpoint_args.get("seed", 13)

    samples = load_bundle_samples(args.ground_truth_root, limit=args.limit)
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

    try:
        loss_metrics: dict[str, Any] = evaluate(
            model,
            dataloader,
            criterion,
            device,
            amp=use_amp,
            channels_last=args.channels_last,
        )
        loss_metrics_status: dict[str, Any] = {"available": True, "reason": None}
    except KeyError as exc:
        loss_metrics = {}
        loss_metrics_status = {
            "available": False,
            "reason": f"missing target required for loss evaluation: {exc}",
        }
    validation_metrics = compute_validation_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        score_thresholds=_default_thresholds(),
        target_threshold=args.target_threshold,
        amp=use_amp,
        channels_last=args.channels_last,
    )
    decode_diagnostics = _decode_count_diagnostics(
        model=model,
        dataloader=dataloader,
        device=device,
        score_thresholds=_default_thresholds(),
        target_threshold=args.target_threshold,
        amp=use_amp,
        channels_last=args.channels_last,
    )

    best_threshold = validation_metrics["best_score_threshold"]
    best_threshold_payload = decode_diagnostics.get(str(best_threshold), None)

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "ground_truth_root": str(args.ground_truth_root),
        "device": str(device),
        "amp": use_amp,
        "sample_count": len(samples),
        "val_sample_count": len(val_samples),
        "seed": seed,
        "val_fraction": val_fraction,
        "loss": loss_metrics,
        "loss_status": loss_metrics_status,
        "validation_metrics": validation_metrics,
        "decode_diagnostics": decode_diagnostics,
        "best_threshold_decode_summary": best_threshold_payload,
        "interpretation_hints": {
            "high_score_f1_but_high_count_mae": "decoded point count is unstable even when matches exist",
            "good_offset_mae_px_but_low_score_f1": "score head / thresholding is the bottleneck",
            "good_score_f1_but_high_orientation_mae_deg": "orientation head is likely hurting MCC consistency",
            "good_head_metrics_but_weak_matching": "matching may be limited by cross-image overlap/preprocessing drift rather than single-image head quality",
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
