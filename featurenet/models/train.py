from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in lighter environments
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow may also be unavailable
    Image = None

from .feature_extractor import FeatureExtractor
from .losses import FeatureNetLoss


FLOAT_TARGET_KEYS = {
    "mask",
    "orientation",
    "ridge_period",
    "gradient",
    "minutia_score",
    "minutia_valid_mask",
    "minutia_x_offset",
    "minutia_y_offset",
    "minutia_orientation_vec",
}
LONG_TARGET_KEYS = {"minutia_x", "minutia_y", "minutia_orientation"}
LOSS_KEYS = ("total", "orientation", "ridge", "gradient", "minutia", "m1", "m2", "m3", "m4")
EARLY_STOPPING_METRICS = (
    "val_total",
    "best_score_f1",
    "minutia_x_accuracy",
    "minutia_y_accuracy",
    "minutia_orientation_accuracy",
)


def _read_grayscale_image(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        array = source
    else:
        if cv2 is not None:
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"unable to load image from {source}")
            array = image
        elif Image is not None:
            with Image.open(source) as image:
                array = np.array(image)
        else:
            raise RuntimeError("image loading requires either opencv-python or Pillow to be installed")

    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 4:
        if cv2 is not None:
            return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        rgb = array[:, :, :3].astype(np.float32)
        return np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(array.dtype)
    if array.ndim == 3 and array.shape[2] == 3:
        if cv2 is not None:
            return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        rgb = array.astype(np.float32)
        return np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(array.dtype)
    raise ValueError(f"unsupported image shape: {array.shape}")


def _to_image_tensor(image: str | Path | np.ndarray) -> torch.Tensor:
    image_array = _read_grayscale_image(image).astype(np.float32)
    if image_array.max() > 1.0:
        image_array /= 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def _to_mask_tensor(mask: str | Path | np.ndarray) -> torch.Tensor:
    mask_array = _read_grayscale_image(mask)
    mask_array = (mask_array > 0).astype(np.float32)
    return torch.from_numpy(mask_array).unsqueeze(0)


def build_input_tensor(
    masked_image: str | Path | np.ndarray,
    mask: str | Path | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = _to_image_tensor(masked_image)
    mask_tensor = _to_mask_tensor(mask)

    if image_tensor.shape[-2:] != mask_tensor.shape[-2:]:
        raise ValueError(
            "masked image and mask must have the same spatial shape, "
            f"got {tuple(image_tensor.shape[-2:])} and {tuple(mask_tensor.shape[-2:])}"
        )

    return torch.cat([image_tensor, mask_tensor], dim=0), mask_tensor


def _to_float_target_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] not in {1, 2, 180, 360} and tensor.shape[-1] in {1, 2, 180, 360}:
        tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous()


def _to_long_target_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.contiguous()


def prepare_targets(targets: Mapping[str, Any], mask_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {"mask": mask_tensor}

    for key, value in targets.items():
        if key in FLOAT_TARGET_KEYS:
            prepared[key] = _to_float_target_tensor(value)
        elif key in LONG_TARGET_KEYS:
            prepared[key] = _to_long_target_tensor(value)
        else:
            prepared[key] = torch.as_tensor(value)

    return prepared


class FeatureNetDataset(Dataset):
    def __init__(self, samples: list[Mapping[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[index]
        input_tensor, mask_tensor = build_input_tensor(
            sample["masked_image"],
            sample["mask"],
        )

        targets = prepare_targets(sample.get("targets", {}), mask_tensor)
        return input_tensor, targets


def _pad_tensor_to_shape(tensor: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    target_height, target_width = spatial_shape
    pad_height = target_height - tensor.shape[-2]
    pad_width = target_width - tensor.shape[-1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError(f"cannot pad tensor with shape {tuple(tensor.shape)} to smaller shape {spatial_shape}")
    if pad_height == 0 and pad_width == 0:
        return tensor
    return F.pad(tensor, (0, pad_width, 0, pad_height))


def _collate_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    inputs, targets = zip(*batch)
    max_height = max(tensor.shape[-2] for tensor in inputs)
    max_width = max(tensor.shape[-1] for tensor in inputs)
    padded_inputs = [_pad_tensor_to_shape(tensor, (max_height, max_width)) for tensor in inputs]
    collated_inputs = torch.stack(padded_inputs, dim=0)

    collated_targets: dict[str, torch.Tensor] = {}
    for key in targets[0]:
        values = [target[key] for target in targets]
        if values[0].dim() >= 2:
            max_height = max(value.shape[-2] for value in values)
            max_width = max(value.shape[-1] for value in values)
            values = [_pad_tensor_to_shape(value, (max_height, max_width)) for value in values]
        collated_targets[key] = torch.stack(values, dim=0)

    return collated_inputs, collated_targets


def create_dataloader(
    samples: list[Mapping[str, Any]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    dataset = FeatureNetDataset(samples)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_batch,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        **dataloader_kwargs,
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _maybe_channels_last(model: FeatureExtractor, enabled: bool) -> FeatureExtractor:
    if enabled:
        model = model.to(memory_format=torch.channels_last)
    return model


def _prepare_inputs_for_model(inputs: torch.Tensor, channels_last: bool) -> torch.Tensor:
    if channels_last and inputs.dim() == 4:
        return inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def _maybe_compile_model(model: FeatureExtractor, enabled: bool) -> FeatureExtractor:
    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _resize_like(tensor: torch.Tensor, spatial_shape: tuple[int, int], mode: str) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(tensor, size=spatial_shape, mode=mode, align_corners=False)
    return F.interpolate(tensor, size=spatial_shape, mode=mode)


def _compute_spatial_shape(output_shapes: Mapping[str, torch.Size | tuple[int, ...]]) -> tuple[int, int]:
    spatial_shapes = {
        key: tuple(shape[-2:])
        for key, shape in output_shapes.items()
    }
    unique_shapes = set(spatial_shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"expected all heads to share one spatial shape, got {spatial_shapes}")
    return next(iter(unique_shapes))


def _sobel_gradients(image_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image_batch.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = image_batch.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    grad_x = F.conv2d(image_batch, kernel_x, padding=1)
    grad_y = F.conv2d(image_batch, kernel_y, padding=1)
    return grad_x, grad_y


def _one_hot_from_bins(
    bins: torch.Tensor,
    num_bins: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    one_hot = F.one_hot(bins.long().clamp(0, num_bins - 1), num_classes=num_bins)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot * mask


def _build_spatial_bin_targets(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_coords = torch.arange(height, device=device).view(1, height, 1).expand(batch_size, height, width)
    x_coords = torch.arange(width, device=device).view(1, 1, width).expand(batch_size, height, width)

    x_bins = torch.div(x_coords * 8, max(width, 1), rounding_mode="floor").clamp(0, 7)
    y_bins = torch.div(y_coords * 8, max(height, 1), rounding_mode="floor").clamp(0, 7)
    return x_bins.long(), y_bins.long()


def build_pseudo_targets(
    image_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    output_shapes: Mapping[str, torch.Size | tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    spatial_shape = _compute_spatial_shape(output_shapes)
    resized_image = _resize_like(image_batch.float(), spatial_shape, mode="bilinear")
    resized_mask = _resize_like(mask_batch.float(), spatial_shape, mode="nearest")
    resized_mask = (resized_mask > 0.5).float()
    masked_image = resized_image * resized_mask

    grad_x, grad_y = _sobel_gradients(masked_image)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8) * resized_mask

    angle = torch.atan2(grad_y, grad_x + 1e-8)
    angle = torch.remainder(angle, 2 * torch.pi).squeeze(1)

    orientation_bins = torch.floor(angle * 180.0 / (2 * torch.pi)).clamp(0, 179)
    orientation = _one_hot_from_bins(orientation_bins, 180, resized_mask)

    minutia_orientation = torch.floor(angle * 360.0 / (2 * torch.pi)).clamp(0, 359).long()
    minutia_orientation = minutia_orientation * resized_mask.squeeze(1).long()

    ridge_period = F.avg_pool2d(masked_image, kernel_size=5, stride=1, padding=2)
    ridge_period = ridge_period * resized_mask
    ridge_period = ridge_period / ridge_period.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    local_mean = F.avg_pool2d(masked_image, kernel_size=9, stride=1, padding=4)
    contrast = (masked_image - local_mean).abs() * resized_mask
    response = grad_mag + contrast
    threshold = response.flatten(2).mean(dim=-1, keepdim=True).view(-1, 1, 1, 1)
    minutia_score = (response > threshold).float() * resized_mask

    batch_size, _, height, width = resized_mask.shape
    minutia_x, minutia_y = _build_spatial_bin_targets(batch_size, height, width, resized_mask.device)
    mask_labels = resized_mask.squeeze(1).long()
    minutia_x = minutia_x * mask_labels
    minutia_y = minutia_y * mask_labels
    minutia_x_offset = ((minutia_x.float() + 0.5) / 8.0).unsqueeze(1) * resized_mask
    minutia_y_offset = ((minutia_y.float() + 0.5) / 8.0).unsqueeze(1) * resized_mask

    return {
        "mask": resized_mask,
        "orientation": orientation,
        "ridge_period": ridge_period,
        "minutia_score": minutia_score,
        "minutia_valid_mask": minutia_score,
        "minutia_x": minutia_x,
        "minutia_y": minutia_y,
        "minutia_x_offset": minutia_x_offset,
        "minutia_y_offset": minutia_y_offset,
        "minutia_orientation": minutia_orientation,
    }


def _move_targets_to_device(targets: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _merge_targets(
    explicit_targets: Mapping[str, torch.Tensor],
    pseudo_targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    merged = dict(pseudo_targets)
    merged.update(explicit_targets)
    return merged


def _loss_dict_to_scalars(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in losses.items() if key in LOSS_KEYS}


def _run_model_step(
    model: FeatureExtractor,
    criterion: FeatureNetLoss,
    inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    image = inputs[:, :1]
    mask = inputs[:, 1:2]
    outputs = model(image, mask=mask)
    pseudo_targets = build_pseudo_targets(image, mask, {key: value.shape for key, value in outputs.items()})
    merged_targets = _merge_targets(targets, pseudo_targets)
    if "gradient" not in merged_targets:
        raise KeyError("missing explicit reconstruction-derived gradient target for gradient supervision")
    return criterion(outputs, merged_targets)


def train_one_epoch(
    model: FeatureExtractor,
    dataloader: DataLoader,
    criterion: FeatureNetLoss,
    optimizer: optim.Optimizer,
    device: str | torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    channels_last: bool = False,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    model.train()
    resolved_device = torch.device(device)
    running = {key: 0.0 for key in LOSS_KEYS}
    num_batches = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(resolved_device, non_blocking=True)
        inputs = _prepare_inputs_for_model(inputs, channels_last)
        targets = _move_targets_to_device(targets, resolved_device)
        with _autocast_context(resolved_device, amp):
            losses = _run_model_step(model, criterion, inputs, targets)
            total_loss = losses["total"]
        if not torch.isfinite(total_loss):
            raise RuntimeError("encountered non-finite total loss during training")

        scaled_loss = total_loss / grad_accum_steps
        if scaler is not None and amp and resolved_device.type == "cuda":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if batch_index % grad_accum_steps == 0:
            if scaler is not None and amp and resolved_device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scalar_losses = _loss_dict_to_scalars(losses)
        for key in LOSS_KEYS:
            running[key] += scalar_losses[key]
        num_batches += 1

    if num_batches % grad_accum_steps != 0:
        if scaler is not None and amp and resolved_device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if num_batches == 0:
        raise ValueError("training dataloader is empty")

    return {key: value / num_batches for key, value in running.items()}


@torch.no_grad()
def evaluate(
    model: FeatureExtractor,
    dataloader: DataLoader,
    criterion: FeatureNetLoss,
    device: str | torch.device,
    amp: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    resolved_device = torch.device(device)
    running = {key: 0.0 for key in LOSS_KEYS}
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(resolved_device, non_blocking=True)
        inputs = _prepare_inputs_for_model(inputs, channels_last)
        targets = _move_targets_to_device(targets, resolved_device)
        with _autocast_context(resolved_device, amp):
            losses = _run_model_step(model, criterion, inputs, targets)

        scalar_losses = _loss_dict_to_scalars(losses)
        for key in LOSS_KEYS:
            running[key] += scalar_losses[key]
        num_batches += 1

    if num_batches == 0:
        raise ValueError("validation dataloader is empty")

    return {key: value / num_batches for key, value in running.items()}


def fit_debug(
    samples: list[Mapping[str, Any]],
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    amp: bool = False,
    channels_last: bool = False,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("fit_debug requires at least one sample")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    resolved_device = _resolve_device(device)
    dataloader = create_dataloader(
        samples=samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    model = _maybe_channels_last(FeatureExtractor().to(resolved_device), channels_last)
    criterion = FeatureNetLoss().to(resolved_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = _make_grad_scaler(amp and resolved_device.type == "cuda")

    history: list[dict[str, float]] = []
    stopped_early = False

    for epoch in range(epochs):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
            scaler=scaler,
            amp=amp,
            channels_last=channels_last,
        )
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)
        if not np.isfinite(metrics["total"]):
            stopped_early = True
            break

    return {
        "device": str(resolved_device),
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "batch_size": batch_size,
        "lr": lr,
        "stopped_early": stopped_early,
        "history": history,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
    }


def _collect_output_shapes(
    model: FeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, list[int]]:
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    image = inputs[:, :1]
    mask = inputs[:, 1:2]

    model.eval()
    with torch.no_grad():
        outputs = model(image, mask=mask)
    model.train()
    return {key: list(value.shape) for key, value in outputs.items()}


def run_smoke_test(
    samples: list[Mapping[str, Any]],
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    amp: bool = False,
    channels_last: bool = False,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("run_smoke_test requires at least one sample")

    resolved_device = _resolve_device(device)
    dataloader = create_dataloader(samples=samples, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _maybe_channels_last(FeatureExtractor().to(resolved_device), channels_last)
    output_shapes = _collect_output_shapes(model, dataloader, resolved_device)

    result = fit_debug(
        samples=samples,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=resolved_device,
        shuffle=False,
        num_workers=0,
        amp=amp,
        channels_last=channels_last,
    )

    history = result["history"]
    first_total = history[0]["total"]
    last_total = history[-1]["total"]
    return {
        "device": result["device"],
        "epochs_ran": result["epochs_ran"],
        "batch_size": batch_size,
        "num_samples": len(samples),
        "output_shapes": output_shapes,
        "first_total_loss": first_total,
        "last_total_loss": last_total,
        "loss_delta": last_total - first_total,
        "history": history,
    }


def _load_npz_targets(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path) as data:
        targets = {key: data[key] for key in data.files}
    if "output_mask" in targets and "mask" not in targets:
        targets["mask"] = targets.pop("output_mask")
    return targets


def _coerce_hw_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        height = int(value[0])
        width = int(value[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _infer_output_shape_from_targets(targets: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("mask", "minutia_score", "minutia_valid_mask"):
        value = targets.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 2:
            height = int(array.shape[-2])
            width = int(array.shape[-1])
            if height > 0 and width > 0:
                return height, width
    return None


def _resolve_output_mask_from_targets(targets: Mapping[str, Any], output_shape_hw: tuple[int, int]) -> np.ndarray:
    mask_value = targets.get("mask", targets.get("minutia_valid_mask"))
    if mask_value is None:
        return np.ones(output_shape_hw, dtype=np.float32)
    mask_array = np.asarray(mask_value, dtype=np.float32)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    elif mask_array.ndim != 2:
        mask_array = np.asarray(mask_array).reshape(output_shape_hw)
    return np.where(mask_array > 0.0, 1.0, 0.0).astype(np.float32)


def _recover_offset_maps_from_minutiae(
    minutiae_path: Path,
    input_shape_hw: tuple[int, int],
    output_shape_hw: tuple[int, int],
    output_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not minutiae_path.exists():
        raise FileNotFoundError(f"missing minutiae.json required for offset recovery: {minutiae_path}")
    rows = json.loads(minutiae_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"invalid minutiae payload in {minutiae_path}: expected a list")

    target_height, target_width = int(output_shape_hw[0]), int(output_shape_hw[1])
    source_height, source_width = int(input_shape_hw[0]), int(input_shape_hw[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"invalid output shape for offset recovery: {output_shape_hw}")
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"invalid input shape for offset recovery: {input_shape_hw}")

    x_offset = np.zeros((target_height, target_width), dtype=np.float32)
    y_offset = np.zeros((target_height, target_width), dtype=np.float32)
    ownership = np.full((target_height, target_width), -1.0, dtype=np.float32)
    center_distance = np.full((target_height, target_width), np.inf, dtype=np.float32)
    cell_width = float(source_width) / float(max(target_width, 1))
    cell_height = float(source_height) / float(max(target_height, 1))

    for row in rows:
        if not isinstance(row, dict):
            continue
        x_value = row.get("x")
        y_value = row.get("y")
        if x_value is None or y_value is None:
            continue
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        cell_x = int(np.clip(math.floor(x * target_width / max(source_width, 1)), 0, target_width - 1))
        cell_y = int(np.clip(math.floor(y * target_height / max(source_height, 1)), 0, target_height - 1))
        if output_mask[cell_y, cell_x] <= 0.0:
            continue
        score = row.get("score")
        score_value = float(score) if score is not None and math.isfinite(float(score)) else 1.0
        score_value = float(np.clip(score_value, 0.0, 1.0))
        cell_origin_x = cell_x * cell_width
        cell_origin_y = cell_y * cell_height
        local_x = (x - cell_origin_x) / max(cell_width, 1e-6)
        local_y = (y - cell_origin_y) / max(cell_height, 1e-6)
        local_x = float(np.clip(local_x, 0.0, 1.0 - 1e-6))
        local_y = float(np.clip(local_y, 0.0, 1.0 - 1e-6))
        distance = float((local_x - 0.5) ** 2 + (local_y - 0.5) ** 2)
        replace = False
        if score_value > ownership[cell_y, cell_x]:
            replace = True
        elif abs(score_value - ownership[cell_y, cell_x]) <= 1e-6 and distance < center_distance[cell_y, cell_x]:
            replace = True
        if not replace:
            continue
        ownership[cell_y, cell_x] = score_value
        center_distance[cell_y, cell_x] = distance
        x_offset[cell_y, cell_x] = local_x
        y_offset[cell_y, cell_x] = local_y

    x_offset *= output_mask
    y_offset *= output_mask
    return x_offset[np.newaxis, ...].astype(np.float32), y_offset[np.newaxis, ...].astype(np.float32)


def load_bundle_samples(
    ground_truth_root: str | Path,
    limit: int | None = None,
    strict_gradient_targets: bool = False,
) -> list[dict[str, Any]]:
    root = Path(ground_truth_root)
    samples_root = root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"missing sample directory: {samples_root}")

    sample_dirs = sorted(path for path in samples_root.iterdir() if path.is_dir())
    if limit is not None:
        sample_dirs = sample_dirs[:limit]

    samples: list[dict[str, Any]] = []
    missing_gradient_paths: list[Path] = []
    recovered_offset_paths: list[Path] = []
    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        masked_image_path = sample_dir / "masked_image.png"
        mask_path = sample_dir / "mask.png"
        targets_path = sample_dir / "featurenet_targets.npz"
        if not (meta_path.exists() and masked_image_path.exists() and mask_path.exists() and targets_path.exists()):
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("raw_view_index") not in {0, 1, 2}:
            continue
        targets = _load_npz_targets(targets_path)
        if "gradient" not in targets:
            if strict_gradient_targets:
                raise ValueError(f"missing reconstruction-derived gradient target in {targets_path}")
            missing_gradient_paths.append(targets_path)
            continue
        shapes = meta.get("shapes", {})
        input_shape_hw = _coerce_hw_shape(shapes.get("input_image")) if isinstance(shapes, dict) else None
        output_shape_hw = _coerce_hw_shape(shapes.get("featurenet_output")) if isinstance(shapes, dict) else None
        if output_shape_hw is None:
            output_shape_hw = _infer_output_shape_from_targets(targets)
        if output_shape_hw is None:
            continue
        if input_shape_hw is None:
            image_height, image_width = _read_grayscale_image(masked_image_path).shape[-2:]
            input_shape_hw = (int(image_height), int(image_width))
        if "minutia_x_offset" not in targets or "minutia_y_offset" not in targets:
            output_mask = _resolve_output_mask_from_targets(targets, output_shape_hw)
            minutiae_path = sample_dir / "minutiae.json"
            try:
                recovered_x, recovered_y = _recover_offset_maps_from_minutiae(
                    minutiae_path=minutiae_path,
                    input_shape_hw=input_shape_hw,
                    output_shape_hw=output_shape_hw,
                    output_mask=output_mask,
                )
            except Exception as exc:
                raise ValueError(
                    f"missing continuous offset targets in {targets_path} and failed recovery from {minutiae_path}: {exc}"
                ) from exc
            targets["minutia_x_offset"] = recovered_x
            targets["minutia_y_offset"] = recovered_y
            recovered_offset_paths.append(minutiae_path)
        targets["raw_view_index"] = np.int64(meta.get("raw_view_index", -1))
        targets["input_shape_hw"] = np.asarray(input_shape_hw, dtype=np.int64)
        targets["output_shape_hw"] = np.asarray(output_shape_hw, dtype=np.int64)
        samples.append(
            {
                "sample_id": meta["sample_id"],
                "finger_class_id": meta.get("finger_class_id"),
                "subject_id": meta.get("subject_id"),
                "finger_id": meta.get("finger_id"),
                "acquisition_id": meta.get("acquisition_id"),
                "masked_image": masked_image_path,
                "mask": mask_path,
                "targets": targets,
            }
        )

    if missing_gradient_paths:
        preview_items = [str(path) for path in missing_gradient_paths[:3]]
        suffix = " ..." if len(missing_gradient_paths) > 3 else ""
        preview = ", ".join(preview_items)
        print(
            "[load_bundle_samples] skipped "
            f"{len(missing_gradient_paths)} samples missing reconstruction-derived gradient target "
            f"(strict mode off). Examples: {preview}{suffix}",
            flush=True,
        )
    if recovered_offset_paths:
        preview_items = [str(path) for path in recovered_offset_paths[:3]]
        suffix = " ..." if len(recovered_offset_paths) > 3 else ""
        preview = ", ".join(preview_items)
        print(
            "[load_bundle_samples] recovered "
            f"{len(recovered_offset_paths)} samples missing continuous x/y offsets from minutiae.json. "
            f"Examples: {preview}{suffix}",
            flush=True,
        )

    if not samples:
        raise ValueError(f"no usable samples found under {samples_root}")
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")
    if val_fraction == 0.0 or len(samples) < 2:
        return samples, []

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        group_key = sample.get("finger_class_id")
        if group_key is None:
            group_key = sample["sample_id"]
        groups[group_key].append(sample)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    val_group_count = max(1, int(round(len(group_keys) * val_fraction)))
    val_keys = set(group_keys[:val_group_count])

    train_samples = [sample for key, group in groups.items() if key not in val_keys for sample in group]
    val_samples = [sample for key, group in groups.items() if key in val_keys for sample in group]

    if not train_samples or not val_samples:
        raise ValueError("split produced an empty train or validation set; reduce val_fraction or add more samples")
    return train_samples, val_samples


def _checkpoint_payload(
    model: FeatureExtractor,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": dict(metrics),
        "args": vars(args),
    }


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_extended_validation_metrics(
    model: FeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
    amp: bool,
    channels_last: bool,
) -> dict[str, Any]:
    from .evaluate import _default_thresholds, compute_validation_metrics

    return compute_validation_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        score_thresholds=_default_thresholds(),
        target_threshold=0.0,
        amp=amp,
        channels_last=channels_last,
    )


def _select_monitored_metric(
    metric_name: str,
    val_losses: Mapping[str, float],
    val_metrics: Mapping[str, Any] | None,
) -> tuple[float, str]:
    if metric_name == "val_total":
        return float(val_losses["total"]), "min"

    if val_metrics is None:
        raise ValueError(f"validation metrics are required to monitor {metric_name}")

    if metric_name == "best_score_f1":
        return float(val_metrics["best_score_f1"]), "max"

    if metric_name == "minutia_x_accuracy":
        return float(val_metrics["label_accuracy"]["minutia_x"]["accuracy"]), "max"

    if metric_name == "minutia_y_accuracy":
        return float(val_metrics["label_accuracy"]["minutia_y"]["accuracy"]), "max"

    if metric_name == "minutia_orientation_accuracy":
        return float(val_metrics["label_accuracy"]["minutia_orientation"]["accuracy"]), "max"

    raise ValueError(f"unsupported early stopping metric: {metric_name}")


def _is_metric_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"unsupported monitor mode: {mode}")


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_bundle_samples(
        args.ground_truth_root,
        limit=args.limit,
        strict_gradient_targets=args.strict_gradient_targets,
    )
    train_samples, val_samples = split_samples(samples, val_fraction=args.val_fraction, seed=args.seed)
    resolved_device = _resolve_device(args.device)
    use_amp = bool(args.amp) and resolved_device.type == "cuda"
    use_pin_memory = bool(args.pin_memory) and resolved_device.type == "cuda"
    if args.cudnn_benchmark and resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = create_dataloader(
        samples=train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = None
    if val_samples:
        val_loader = create_dataloader(
            samples=val_samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    model = _maybe_channels_last(FeatureExtractor().to(resolved_device), args.channels_last)
    model = _maybe_compile_model(model, args.compile and resolved_device.type == "cuda")
    criterion = FeatureNetLoss(
        mu_score=args.mu_score,
        mu_x=args.mu_x,
        mu_y=args.mu_y,
        mu_ori=args.mu_ori,
        m1_focal_gamma=args.m1_focal_gamma,
        m1_pos_weight_max=args.m1_pos_weight_max,
        m1_hard_neg_enable=args.m1_hard_neg_enable,
        m1_hard_neg_ratio=args.m1_hard_neg_ratio,
        m1_hard_neg_min=args.m1_hard_neg_min,
        m1_hard_neg_fraction=args.m1_hard_neg_fraction,
    ).to(resolved_device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    scaler = _make_grad_scaler(use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_name = args.early_stopping_metric
    best_monitor_value = float("inf") if monitor_name == "val_total" else float("-inf")
    monitor_mode = "min" if monitor_name == "val_total" else "max"
    history: list[dict[str, Any]] = []
    started_at = time.time()
    validation_checks = 0
    patience_counter = 0
    stop_reason = "max_epochs"
    effective_early_stopping = bool(args.early_stopping)
    early_stopping_disabled_reason: str | None = None
    if val_loader is None and effective_early_stopping:
        effective_early_stopping = False
        early_stopping_disabled_reason = "no_validation_data"

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            resolved_device,
            scaler=scaler,
            amp=use_amp,
            channels_last=args.channels_last,
            grad_accum_steps=args.grad_accum_steps,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "seconds": round(time.time() - epoch_started_at, 3),
            "validation_ran": False,
            "early_stopping": {
                "enabled": effective_early_stopping,
                "configured": bool(args.early_stopping),
                "metric": monitor_name,
                "mode": monitor_mode,
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "validate_every": args.validate_every,
                "checks_run": validation_checks,
                "patience_counter": patience_counter,
                "disabled_reason": early_stopping_disabled_reason,
            },
        }
        if not np.isfinite(train_metrics["total"]):
            stop_reason = "non_finite_loss"
            history.append(record)
            print(json.dumps(record), flush=True)
            break

        should_validate = val_loader is not None and (epoch % args.validate_every == 0 or epoch == args.epochs)
        if should_validate:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                resolved_device,
                amp=use_amp,
                channels_last=args.channels_last,
            )
            record["val"] = val_metrics
            record["validation_ran"] = True
            validation_checks += 1
            extended_val_metrics: dict[str, Any] | None = None
            if monitor_name != "val_total":
                extended_val_metrics = _compute_extended_validation_metrics(
                    model=model,
                    dataloader=val_loader,
                    device=resolved_device,
                    amp=use_amp,
                    channels_last=args.channels_last,
                )
                record["val_metrics"] = extended_val_metrics

            current_monitor_value, monitor_mode = _select_monitored_metric(
                monitor_name,
                val_metrics,
                extended_val_metrics,
            )
            improved = _is_metric_improved(
                current=current_monitor_value,
                best=best_monitor_value,
                mode=monitor_mode,
                min_delta=args.early_stopping_min_delta,
            )
            if improved:
                best_monitor_value = current_monitor_value
                patience_counter = 0
                torch.save(
                    _checkpoint_payload(model, optimizer, epoch, record, args),
                    output_dir / "best.pt",
                )
            else:
                patience_counter += 1

            record["monitor"] = {
                "name": monitor_name,
                "mode": monitor_mode,
                "value": current_monitor_value,
                "best_value": best_monitor_value,
                "improved": improved,
            }
            record["early_stopping"].update(
                {
                    "checks_run": validation_checks,
                    "patience_counter": patience_counter,
                }
            )
            if effective_early_stopping and patience_counter >= args.early_stopping_patience:
                stop_reason = "early_stopping_plateau"
                history.append(record)
                print(json.dumps(record), flush=True)
                break

        history.append(record)
        print(json.dumps(record), flush=True)

    last_record = history[-1]
    torch.save(
        _checkpoint_payload(model, optimizer, int(last_record["epoch"]), last_record, args),
        output_dir / "last.pt",
    )

    summary = {
        "device": str(resolved_device),
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "epochs": args.epochs,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "amp": use_amp,
        "channels_last": bool(args.channels_last),
        "compile": bool(args.compile and resolved_device.type == "cuda"),
        "grad_accum_steps": args.grad_accum_steps,
        "pin_memory": use_pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "seed": args.seed,
        "strict_gradient_targets": bool(args.strict_gradient_targets),
        "mu_score": args.mu_score,
        "mu_x": args.mu_x,
        "mu_y": args.mu_y,
        "mu_ori": args.mu_ori,
        "m1_focal_gamma": args.m1_focal_gamma,
        "m1_pos_weight_max": args.m1_pos_weight_max,
        "m1_hard_neg_enable": bool(args.m1_hard_neg_enable),
        "m1_hard_neg_ratio": args.m1_hard_neg_ratio,
        "m1_hard_neg_min": args.m1_hard_neg_min,
        "m1_hard_neg_fraction": args.m1_hard_neg_fraction,
        "validate_every": args.validate_every,
        "early_stopping": {
            "configured": bool(args.early_stopping),
            "enabled": effective_early_stopping,
            "metric": monitor_name,
            "mode": monitor_mode,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "validation_checks": validation_checks,
            "best_value": None if validation_checks == 0 else best_monitor_value,
            "disabled_reason": early_stopping_disabled_reason,
        },
        "stop_reason": stop_reason,
        "wall_seconds": round(time.time() - started_at, 3),
        "history": history,
    }
    _save_json(output_dir / "history.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FeatureNet against generated DS1 ground-truth bundles.")
    parser.add_argument("--ground-truth-root", type=Path, required=True, help="Path to a generated ground_truth/DS1 folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and training history.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=max(1, min((os.cpu_count() or 1), 4)))
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on supported CUDA runtimes.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--early-stopping", action="store_true", help="Stop training when the monitored validation metric plateaus.")
    parser.add_argument("--early-stopping-metric", choices=EARLY_STOPPING_METRICS, default="val_total")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--mu-score", type=float, default=120.0)
    parser.add_argument("--mu-x", type=float, default=20.0)
    parser.add_argument("--mu-y", type=float, default=20.0)
    parser.add_argument("--mu-ori", type=float, default=5.0)
    parser.add_argument("--m1-focal-gamma", type=float, default=2.0)
    parser.add_argument("--m1-pos-weight-max", type=float, default=100.0)
    parser.add_argument("--m1-hard-neg-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m1-hard-neg-ratio", type=float, default=20.0)
    parser.add_argument("--m1-hard-neg-min", type=int, default=2000)
    parser.add_argument("--m1-hard-neg-fraction", type=float, default=0.05)
    parser.add_argument(
        "--strict-gradient-targets",
        action="store_true",
        help="Fail fast if any selected sample is missing reconstruction-derived gradient targets.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.validate_every < 1:
        parser.error("--validate-every must be at least 1")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")
    if args.early_stopping_min_delta < 0.0:
        parser.error("--early-stopping-min-delta must be non-negative")
    if args.mu_score <= 0.0 or args.mu_x <= 0.0 or args.mu_y <= 0.0 or args.mu_ori <= 0.0:
        parser.error("--mu-score, --mu-x, --mu-y, and --mu-ori must be positive")
    if args.m1_focal_gamma < 0.0:
        parser.error("--m1-focal-gamma must be non-negative")
    if args.m1_pos_weight_max < 1.0:
        parser.error("--m1-pos-weight-max must be at least 1.0")
    if args.m1_hard_neg_ratio < 0.0:
        parser.error("--m1-hard-neg-ratio must be non-negative")
    if args.m1_hard_neg_min < 0:
        parser.error("--m1-hard-neg-min must be non-negative")
    if not 0.0 <= args.m1_hard_neg_fraction <= 1.0:
        parser.error("--m1-hard-neg-fraction must be in [0.0, 1.0]")
    return args


def main() -> None:
    args = parse_args()
    summary = train_model(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
