from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import sysconfig
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

# stdlib `dataclasses` imports `copy`, so it must come after the shim above
# (the repo-level copy.py would otherwise shadow the stdlib module).
import dataclasses

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in lighter environments
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow may also be unavailable
    Image = None

from .feature_extractor import FeatureExtractor
from .losses import (
    ConsistencyConfig,
    FeatureNetLoss,
    apply_gpu_photometric,
    make_affine_theta,
    masked_mse,
    orientation_equivariance_loss,
    soft_bce_logits_loss,
    warp_tensor,
)
from .augmentation import (
    AugmentationConfig,
    augment_sample,
    has_usable_reconstruction,
    sample_augmentation_params,
)


FLOAT_TARGET_KEYS = {
    "mask",
    "orientation",
    "ridge_period",
    "gradient",
    "minutia_score",
    "minutia_score_weight_map",
    "minutia_score_ignore_mask",
    "minutia_score_center_map",
    "minutia_valid_mask",
    "minutia_x_offset",
    "minutia_y_offset",
    "minutia_orientation_vec",
}
LONG_TARGET_KEYS = {"minutia_x", "minutia_y", "minutia_orientation"}
LOSS_KEYS = ("total", "orientation", "ridge", "gradient", "minutia", "m1", "m1_candidate", "m2", "m3", "m4", "consistency")
TARGET_FINITE_KEYS = frozenset(FLOAT_TARGET_KEYS | LONG_TARGET_KEYS)
EARLY_STOPPING_METRICS = (
    "val_total",
    "best_score_f1",
    "minutia_x_accuracy",
    "minutia_y_accuracy",
    "minutia_orientation_accuracy",
    "pair_auc",
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


def _materialize_sample_targets(sample: Mapping[str, Any]) -> dict[str, Any]:
    if "targets" in sample:
        targets = dict(sample.get("targets", {}))
    else:
        targets_path = sample.get("targets_path")
        if targets_path is None:
            targets = {}
        else:
            targets = _load_npz_targets(Path(targets_path))

    if "raw_view_index" in sample:
        targets["raw_view_index"] = np.int64(sample["raw_view_index"])
    if "input_shape_hw" in sample:
        targets["input_shape_hw"] = np.asarray(sample["input_shape_hw"], dtype=np.int64)
    if "output_shape_hw" in sample:
        targets["output_shape_hw"] = np.asarray(sample["output_shape_hw"], dtype=np.int64)
    return targets


class FeatureNetDataset(Dataset):
    def __init__(
        self,
        samples: list[Mapping[str, Any]],
        augmentation_config: AugmentationConfig | None = None,
        seed: int | None = None,
    ):
        self.samples = samples
        self.augmentation_config = augmentation_config
        self.seed = int(seed if seed is not None else 0)
        self._access_counts: dict[tuple[int, int], int] = defaultdict(int)

    def __len__(self) -> int:
        if self.augmentation_config is None:
            return len(self.samples)
        return len(self.samples) * (1 + max(0, int(self.augmentation_config.count)))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.augmentation_config is None:
            base_index = index
            variant_index = 0
        else:
            variants_per_sample = 1 + max(0, int(self.augmentation_config.count))
            base_index = index // variants_per_sample
            variant_index = index % variants_per_sample

        sample = self.samples[base_index]
        augmentation_mode = 0
        if self.augmentation_config is None or variant_index == 0:
            input_tensor, mask_tensor = build_input_tensor(
                sample["masked_image"],
                sample["mask"],
            )
            materialized_targets = _materialize_sample_targets(sample)
        else:
            access_key = (base_index, variant_index)
            access_count = self._access_counts[access_key]
            self._access_counts[access_key] = access_count + 1
            rng = np.random.default_rng(
                self.seed
                + (base_index + 1) * 1_000_003
                + variant_index * 10_007
                + access_count * 97_409
            )
            params = sample_augmentation_params(rng, self.augmentation_config)
            output_shape = tuple(int(value) for value in sample["output_shape_hw"])
            debug_dir = None
            debug_stem = None
            if self.augmentation_config.debug_dir is not None and access_count < self.augmentation_config.debug_limit:
                debug_dir = self.augmentation_config.debug_dir
                debug_stem = f"{sample.get('sample_id', base_index)}_v{variant_index}_n{access_count}"
            augmented = augment_sample(
                sample,
                params,
                output_shape=output_shape,
                debug_dir=debug_dir,
                debug_stem=debug_stem,
                reconstruction_cache_size=self.augmentation_config.reconstruction_cache_size,
                sample_cache_size=self.augmentation_config.sample_cache_size,
            )
            input_tensor, mask_tensor = build_input_tensor(augmented.image, augmented.mask)
            materialized_targets = dict(augmented.targets)
            for metadata_key in ("raw_view_index", "input_shape_hw", "output_shape_hw"):
                if metadata_key in sample:
                    if metadata_key.endswith("_shape_hw"):
                        materialized_targets[metadata_key] = np.asarray(sample[metadata_key], dtype=np.int64)
                    else:
                        materialized_targets[metadata_key] = np.int64(sample[metadata_key])
            if "output_mask" in materialized_targets and "mask" not in materialized_targets:
                materialized_targets["mask"] = materialized_targets.pop("output_mask")
            augmentation_mode = 2 if str(augmented.details.get("mode", "")).startswith("3d") else 1

        targets = prepare_targets(materialized_targets, mask_tensor)
        targets["_sample_index"] = torch.tensor(base_index, dtype=torch.long)
        targets["_augmentation_mode"] = torch.tensor(augmentation_mode, dtype=torch.long)
        targets["_augmentation_variant_index"] = torch.tensor(variant_index, dtype=torch.long)
        return input_tensor, targets


class FeatureNetGroupedVariantSampler(Sampler[int]):
    def __init__(self, sample_count: int, variants_per_sample: int, seed: int | None = None):
        self.sample_count = int(sample_count)
        self.variants_per_sample = int(variants_per_sample)
        self.seed = int(seed if seed is not None else 0)
        self.epoch = 0

    def __iter__(self):
        order = list(range(self.sample_count))
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        rng.shuffle(order)
        self.epoch += 1
        for base_index in order:
            start = base_index * self.variants_per_sample
            for variant_index in range(self.variants_per_sample):
                yield start + variant_index

    def __len__(self) -> int:
        return self.sample_count * self.variants_per_sample


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
    train_augmentations: bool = False,
    augmentation_config: AugmentationConfig | None = None,
    seed: int | None = None,
) -> DataLoader:
    dataset = FeatureNetDataset(
        samples,
        augmentation_config=augmentation_config if train_augmentations else None,
        seed=seed,
    )
    sampler = None
    if train_augmentations and augmentation_config is not None and augmentation_config.group_variants:
        sampler = FeatureNetGroupedVariantSampler(
            sample_count=len(samples),
            variants_per_sample=1 + max(0, int(augmentation_config.count)),
            seed=seed,
        )
        shuffle = False
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
    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
        dataloader_kwargs.pop("shuffle", None)
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


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported AMP dtype: {name}")


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype = torch.float16):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def _move_targets_to_device(targets: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _loss_dict_to_scalars(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in losses.items() if key in LOSS_KEYS}


def _np_target_non_finite_issues(targets: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in sorted(TARGET_FINITE_KEYS & set(targets.keys())):
        array = np.asarray(targets[key])
        if not np.issubdtype(array.dtype, np.number):
            continue
        finite = np.isfinite(array)
        if bool(finite.all()):
            continue
        total = int(array.size)
        finite_count = int(finite.sum())
        issues.append(f"{key}: non_finite={total - finite_count}/{total}")
    return issues


def _minutia_support_count(targets: Mapping[str, Any]) -> int:
    valid_mask = targets.get("minutia_valid_mask")
    if valid_mask is None:
        return 0
    return int(np.count_nonzero(np.asarray(valid_mask) > 0.5))


def _meta_minutiae_count(meta: Mapping[str, Any]) -> int:
    counts = meta.get("counts")
    if isinstance(counts, Mapping) and counts.get("minutiae") is not None:
        try:
            return int(counts["minutiae"])
        except (TypeError, ValueError):
            return 0
    return 0


def _float_outputs(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in outputs.items()
    }


def _loss_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _tensor_non_finite_issues(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    issues: list[str] = []
    for key, value in tensors.items():
        if key.startswith("_") or not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        finite = torch.isfinite(value.detach())
        if bool(finite.all().item()):
            continue
        total = int(value.numel())
        finite_count = int(finite.sum().item())
        issues.append(f"{key}: non_finite={total - finite_count}/{total}")
    return issues


def _sample_ids_for_batch(dataloader: DataLoader, targets: Mapping[str, torch.Tensor]) -> list[str]:
    sample_indices = targets.get("_sample_index")
    dataset = getattr(dataloader, "dataset", None)
    samples = getattr(dataset, "samples", None)
    if sample_indices is None or samples is None:
        return []
    ids: list[str] = []
    for sample_index in sample_indices.detach().cpu().flatten().tolist():
        try:
            sample = samples[int(sample_index)]
            ids.append(str(sample.get("sample_id", sample_index)))
        except (IndexError, TypeError, ValueError):
            ids.append(str(sample_index))
    return ids


def _non_finite_loss_message(
    losses: Mapping[str, torch.Tensor],
    batch_index: int,
    sample_ids: list[str],
    target_issues: list[str] | None = None,
    output_issues: list[str] | None = None,
) -> str:
    parts = []
    for key in LOSS_KEYS:
        value = losses.get(key)
        if value is None:
            continue
        scalar = value.detach()
        if scalar.numel() == 1:
            try:
                parts.append(f"{key}={float(scalar.item()):.6g}")
            except (RuntimeError, ValueError):
                parts.append(f"{key}=<unavailable>")
    sample_text = ", ".join(sample_ids[:8])
    if len(sample_ids) > 8:
        sample_text += ", ..."
    if sample_text:
        sample_text = f"; samples=[{sample_text}]"
    issue_parts = []
    if target_issues:
        issue_parts.append(f"target_issues=[{'; '.join(target_issues)}]")
    if output_issues:
        issue_parts.append(f"output_issues=[{'; '.join(output_issues)}]")
    issue_text = f"; {'; '.join(issue_parts)}" if issue_parts else ""
    return (
        f"encountered non-finite total loss during training at batch {batch_index}: "
        f"{', '.join(parts)}{sample_text}{issue_text}"
    )


@torch.no_grad()
def _diagnose_non_finite_step(
    model: FeatureExtractor,
    inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    amp: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[list[str], list[str]]:
    image = inputs[:, :1]
    mask = inputs[:, 1:2]
    with _autocast_context(inputs.device, amp, amp_dtype):
        outputs = model(image, mask=mask)
    output_issues = _tensor_non_finite_issues(outputs)
    if amp and output_issues:
        with _loss_context(inputs.device):
            fp32_outputs = model(image, mask=mask)
        fp32_output_issues = _tensor_non_finite_issues(fp32_outputs)
        if fp32_output_issues:
            output_issues = [f"amp {issue}" for issue in output_issues]
            output_issues.extend(f"fp32 {issue}" for issue in fp32_output_issues)
        else:
            output_issues = [f"amp {issue}; fp32 retry finite" for issue in output_issues]
    return _tensor_non_finite_issues(targets), output_issues


def _optimizer_step(
    model: FeatureExtractor,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    device: torch.device,
    max_grad_norm: float | None,
) -> tuple[bool, float | None]:
    grad_norm: float | None = None
    use_scaler = scaler is not None and amp and device.type == "cuda"
    should_clip = max_grad_norm is not None and max_grad_norm > 0.0

    if use_scaler and should_clip:
        scaler.unscale_(optimizer)

    if should_clip:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(max_grad_norm),
            error_if_nonfinite=False,
        )
        grad_norm = float(grad_norm_tensor.detach().cpu().item())
        if not torch.isfinite(grad_norm_tensor):
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.update()
            return False, grad_norm

    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return True, grad_norm


def _run_model_step(
    model: FeatureExtractor,
    criterion: FeatureNetLoss,
    inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    consistency: ConsistencyConfig | None = None,
) -> dict[str, torch.Tensor]:
    image = inputs[:, :1]
    mask = inputs[:, 1:2]
    device = inputs.device
    with _autocast_context(device, amp, amp_dtype):
        outputs = model(image, mask=mask)
    if amp:
        output_issues = _tensor_non_finite_issues(outputs)
        if output_issues:
            with _loss_context(device):
                fp32_outputs = model(image, mask=mask)
            if not _tensor_non_finite_issues(fp32_outputs):
                print(
                    "[train] recovered non-finite AMP model outputs with a float32 retry: "
                    + "; ".join(output_issues),
                    flush=True,
                )
                outputs = fp32_outputs
    if "gradient" not in targets:
        raise KeyError("missing explicit reconstruction-derived gradient target for gradient supervision")
    with _loss_context(device):
        losses = criterion(_float_outputs(outputs), targets)
    if consistency is not None and consistency.weight > 0.0:
        _augment_losses_with_consistency(
            model, image, mask, outputs, targets, losses, consistency, amp=amp, amp_dtype=amp_dtype
        )
    elif "consistency" not in losses:
        losses["consistency"] = outputs["minutia_score"].new_tensor(0.0)
    return losses


def _augment_losses_with_consistency(
    model: FeatureExtractor,
    image: torch.Tensor,
    mask: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    cfg: ConsistencyConfig,
    amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    """Affine + photometric equivariance on a second, warped forward pass.

    ``supervised`` mode (default): the warped view's score head is supervised
    against the GT score map warped with the same theta, plus a small
    peak-weighted self-consistency term and a minutia-orientation equivariance
    term. Pure self-consistency (``self`` mode) can be satisfied by lowering
    all scores — that collapse produced the sparser extraction observed with
    the v4_consistency checkpoint — so supervised mode anchors the scale to GT.
    """
    device = image.device
    batch = image.shape[0]
    image_f = image.float()
    mask_f = mask.float()
    theta = make_affine_theta(
        batch, cfg.max_rot_deg, cfg.max_shift_frac, cfg.max_scale_delta, device, dtype=torch.float32
    )
    warped_image = warp_tensor(image_f, theta, mode="bilinear")
    warped_mask = (warp_tensor(mask_f, theta, mode="nearest") > 0.5).float()
    warped_image = apply_gpu_photometric(warped_image, warped_mask, cfg)

    with _autocast_context(device, amp, amp_dtype):
        outputs2 = model(warped_image.to(image.dtype), mask=warped_mask.to(image.dtype))

    with _loss_context(device):
        prob1 = torch.sigmoid(outputs["minutia_score"].float())
        prob2 = torch.sigmoid(outputs2["minutia_score"].float())
        reference_prob = warp_tensor(prob1, theta, mode="bilinear").detach()
        score_mask = F.interpolate(mask_f, size=prob1.shape[-2:], mode="nearest")
        warped_score_mask = warp_tensor(score_mask, theta, mode="nearest")

        if cfg.mode == "self":
            consistency_loss = masked_mse(prob2, reference_prob, warped_score_mask)
        else:
            target_score = targets["minutia_score"].float()
            if target_score.dim() == 3:
                target_score = target_score.unsqueeze(1)
            warped_target = warp_tensor(target_score, theta, mode="bilinear").clamp(0.0, 1.0)

            sup_weight = warped_score_mask
            weight_map = targets.get("minutia_score_weight_map")
            if weight_map is not None:
                weight_map = weight_map.float()
                if weight_map.dim() == 3:
                    weight_map = weight_map.unsqueeze(1)
                sup_weight = sup_weight * warp_tensor(weight_map, theta, mode="nearest")
            if cfg.pos_boost > 0.0:
                # Positive cells are sparse; boost them so suppressing every
                # score (the collapse shortcut) stays expensive.
                sup_weight = sup_weight * (1.0 + cfg.pos_boost * warped_target)

            consistency_loss = soft_bce_logits_loss(
                outputs2["minutia_score"].float(),
                warped_target,
                sup_weight,
            )

            if cfg.self_weight > 0.0:
                # Peak-weighted so background agreement cannot dominate and
                # jointly lowering both peaks does not reduce the loss.
                peak_weight = (torch.maximum(reference_prob, prob2) * warped_score_mask).detach()
                self_term = ((prob2 - reference_prob) ** 2 * peak_weight).sum() / peak_weight.sum().clamp_min(1e-6)
                consistency_loss = consistency_loss + cfg.self_weight * self_term

            if cfg.orientation_weight > 0.0:
                ori_term = orientation_equivariance_loss(
                    outputs["minutia_orientation"],
                    outputs2["minutia_orientation"],
                    theta,
                    weight=warped_target * warped_score_mask,
                )
                consistency_loss = consistency_loss + cfg.orientation_weight * ori_term

    losses["consistency"] = consistency_loss
    losses["total"] = losses["total"] + cfg.weight * consistency_loss


def train_one_epoch(
    model: FeatureExtractor,
    dataloader: DataLoader,
    criterion: FeatureNetLoss,
    optimizer: optim.Optimizer,
    device: str | torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
    skip_non_finite_target_batches: bool = True,
    consistency: ConsistencyConfig | None = None,
) -> dict[str, float]:
    model.train()
    resolved_device = torch.device(device)
    running = {key: 0.0 for key in LOSS_KEYS}
    num_batches = 0
    accum_batches = 0
    skipped_batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    data_wait_seconds = 0.0
    host_to_device_seconds = 0.0
    forward_loss_seconds = 0.0
    backward_optimizer_seconds = 0.0
    augmentation_mode_counts = {"original": 0, "2d": 0, "3d": 0, "unknown": 0}

    batch_index = 0
    iterator = iter(dataloader)
    next_fetch_started = time.perf_counter()
    while True:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            break
        fetched_at = time.perf_counter()
        data_wait_seconds += fetched_at - next_fetch_started
        batch_index += 1

        mode_values = targets.get("_augmentation_mode")
        if mode_values is not None:
            for mode_value in mode_values.detach().cpu().flatten().tolist():
                if int(mode_value) == 0:
                    augmentation_mode_counts["original"] += 1
                elif int(mode_value) == 1:
                    augmentation_mode_counts["2d"] += 1
                elif int(mode_value) == 2:
                    augmentation_mode_counts["3d"] += 1
                else:
                    augmentation_mode_counts["unknown"] += 1

        host_to_device_started = time.perf_counter()
        inputs = inputs.to(resolved_device, non_blocking=True)
        inputs = _prepare_inputs_for_model(inputs, channels_last)
        targets = _move_targets_to_device(targets, resolved_device)
        host_to_device_seconds += time.perf_counter() - host_to_device_started

        forward_loss_started = time.perf_counter()
        losses = _run_model_step(
            model, criterion, inputs, targets, amp=amp, amp_dtype=amp_dtype, consistency=consistency
        )
        forward_loss_seconds += time.perf_counter() - forward_loss_started
        total_loss = losses["total"]
        if not torch.isfinite(total_loss):
            sample_ids = _sample_ids_for_batch(dataloader, targets)
            target_issues, output_issues = _diagnose_non_finite_step(model, inputs, targets, amp=amp, amp_dtype=amp_dtype)
            if skip_non_finite_target_batches and target_issues and not output_issues:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                accum_batches = 0
                print(
                    "[train_one_epoch] skipped non-finite target batch: "
                    + _non_finite_loss_message(losses, batch_index, sample_ids, target_issues, output_issues),
                    flush=True,
                )
                next_fetch_started = time.perf_counter()
                continue
            raise RuntimeError(
                _non_finite_loss_message(losses, batch_index, sample_ids, target_issues, output_issues)
            )

        backward_started = time.perf_counter()
        scaled_loss = total_loss / grad_accum_steps
        if scaler is not None and amp and resolved_device.type == "cuda":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        accum_batches += 1

        if accum_batches >= grad_accum_steps:
            stepped, grad_norm = _optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp,
                device=resolved_device,
                max_grad_norm=max_grad_norm,
            )
            if stepped:
                optimizer_steps += 1
            else:
                skipped_optimizer_steps += 1
            if grad_norm is not None and math.isfinite(grad_norm):
                grad_norm_sum += grad_norm
                grad_norm_count += 1
            accum_batches = 0
        backward_optimizer_seconds += time.perf_counter() - backward_started

        scalar_losses = _loss_dict_to_scalars(losses)
        for key in LOSS_KEYS:
            running[key] += scalar_losses[key]
        num_batches += 1
        next_fetch_started = time.perf_counter()

    if accum_batches > 0:
        stepped, grad_norm = _optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            device=resolved_device,
            max_grad_norm=max_grad_norm,
        )
        if stepped:
            optimizer_steps += 1
        else:
            skipped_optimizer_steps += 1
        if grad_norm is not None and math.isfinite(grad_norm):
            grad_norm_sum += grad_norm
            grad_norm_count += 1

    if num_batches == 0:
        raise ValueError("training dataloader is empty")

    metrics = {key: value / num_batches for key, value in running.items()}
    metrics["skipped_batches"] = float(skipped_batches)
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
    if grad_norm_count > 0:
        metrics["grad_norm"] = grad_norm_sum / grad_norm_count
    metrics["data_wait_seconds"] = data_wait_seconds
    metrics["host_to_device_seconds"] = host_to_device_seconds
    metrics["forward_loss_seconds"] = forward_loss_seconds
    metrics["backward_optimizer_seconds"] = backward_optimizer_seconds
    metrics["data_wait_seconds_per_batch"] = data_wait_seconds / num_batches
    metrics["host_to_device_seconds_per_batch"] = host_to_device_seconds / num_batches
    metrics["forward_loss_seconds_per_batch"] = forward_loss_seconds / num_batches
    metrics["backward_optimizer_seconds_per_batch"] = backward_optimizer_seconds / num_batches
    metrics["augmentation_original_batches"] = float(augmentation_mode_counts["original"])
    metrics["augmentation_2d_batches"] = float(augmentation_mode_counts["2d"])
    metrics["augmentation_3d_batches"] = float(augmentation_mode_counts["3d"])
    metrics["augmentation_unknown_batches"] = float(augmentation_mode_counts["unknown"])
    return metrics


@torch.no_grad()
def evaluate(
    model: FeatureExtractor,
    dataloader: DataLoader,
    criterion: FeatureNetLoss,
    device: str | torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
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
        losses = _run_model_step(model, criterion, inputs, targets, amp=amp, amp_dtype=amp_dtype)

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
    amp_dtype: torch.dtype = torch.float16,
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
            amp_dtype=amp_dtype,
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


def load_bundle_samples(
    ground_truth_root: str | Path,
    limit: int | None = None,
    strict_gradient_targets: bool = False,
    strict_finite_targets: bool = False,
    skip_empty_minutia_support_with_minutiae: bool = True,
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
    non_finite_target_paths: list[tuple[str, Path, list[str]]] = []
    empty_minutia_support_paths: list[tuple[str, Path, int]] = []
    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        masked_image_path = sample_dir / "masked_image.png"
        mask_path = sample_dir / "mask.png"
        targets_path = sample_dir / "featurenet_targets.npz"
        minutiae_path = sample_dir / "minutiae.json"
        single_source_candidates_path = sample_dir / "minutiae_single_source_candidates.json"
        orientation_path = sample_dir / "orientation.npy"
        ridge_period_path = sample_dir / "ridge_period.npy"
        if not (meta_path.exists() and masked_image_path.exists() and mask_path.exists() and targets_path.exists()):
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            raw_view_index = int(meta.get("raw_view_index", -1))
        except (TypeError, ValueError):
            continue
        if raw_view_index not in {0, 1, 2}:
            continue
        targets = _load_npz_targets(targets_path)
        if "gradient" not in targets:
            if strict_gradient_targets:
                raise ValueError(f"missing reconstruction-derived gradient target in {targets_path}")
            missing_gradient_paths.append(targets_path)
            continue
        non_finite_issues = _np_target_non_finite_issues(targets)
        if non_finite_issues:
            if strict_finite_targets:
                raise ValueError(
                    f"non-finite target values in {targets_path}: {'; '.join(non_finite_issues)}"
                )
            non_finite_target_paths.append((str(meta.get("sample_id", sample_dir.name)), targets_path, non_finite_issues))
            continue
        minutiae_count = _meta_minutiae_count(meta)
        minutia_support_count = _minutia_support_count(targets)
        if skip_empty_minutia_support_with_minutiae and minutiae_count > 0 and minutia_support_count == 0:
            empty_minutia_support_paths.append((str(meta.get("sample_id", sample_dir.name)), targets_path, minutiae_count))
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
        sample_record = {
            "sample_id": meta["sample_id"],
            "parent_sample_id": meta.get("parent_sample_id"),
            "finger_class_id": meta.get("finger_class_id"),
            "subject_id": meta.get("subject_id"),
            "finger_id": meta.get("finger_id"),
            "acquisition_id": meta.get("acquisition_id"),
            "meta_path": meta_path,
            "masked_image": masked_image_path,
            "mask": mask_path,
            "targets_path": targets_path,
            "minutiae_path": minutiae_path,
            "single_source_candidates_path": single_source_candidates_path,
            "orientation_path": orientation_path,
            "ridge_period_path": ridge_period_path,
            "raw_view_index": raw_view_index,
            "input_shape_hw": tuple(int(value) for value in input_shape_hw),
            "output_shape_hw": tuple(int(value) for value in output_shape_hw),
            "minutiae_target_config": meta.get("minutiae_ground_truth")
            if isinstance(meta.get("minutiae_ground_truth"), Mapping)
            else None,
        }
        reconstruction = meta.get("multiview_reconstruction")
        if isinstance(reconstruction, Mapping):
            sample_record["reconstruction"] = {
                key: value
                for key, value in reconstruction.items()
                if key
                in {
                    "role",
                    "reconstruction_dir",
                    "reconstruction_maps_path",
                    "depth_front_path",
                    "depth_left_path",
                    "depth_right_path",
                    "depth_gradient_labels_path",
                }
            }
        samples.append(sample_record)

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

    if non_finite_target_paths:
        preview_items = [
            f"{sample_id} ({path}: {'; '.join(issues)})"
            for sample_id, path, issues in non_finite_target_paths[:3]
        ]
        suffix = " ..." if len(non_finite_target_paths) > 3 else ""
        preview = ", ".join(preview_items)
        print(
            "[load_bundle_samples] skipped "
            f"{len(non_finite_target_paths)} samples with non-finite target values. "
            f"Examples: {preview}{suffix}",
            flush=True,
        )

    if empty_minutia_support_paths:
        preview_items = [
            f"{sample_id} ({path}: minutiae={minutiae_count}, support=0)"
            for sample_id, path, minutiae_count in empty_minutia_support_paths[:3]
        ]
        suffix = " ..." if len(empty_minutia_support_paths) > 3 else ""
        preview = ", ".join(preview_items)
        print(
            "[load_bundle_samples] skipped "
            f"{len(empty_minutia_support_paths)} samples with minutiae but zero active minutia target cells. "
            f"Examples: {preview}{suffix}",
            flush=True,
        )

    if not samples:
        raise ValueError(f"no usable samples found under {samples_root}")
    return samples


def _raw_view_counts(samples: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        raw_view_index = str(sample.get("raw_view_index", "unknown"))
        counts[raw_view_index] = counts.get(raw_view_index, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _filter_samples_for_augmentation(
    samples: list[dict[str, Any]],
    augmentation_config: AugmentationConfig,
) -> list[dict[str, Any]]:
    if augmentation_config.missing_reconstruction != "skip":
        return samples
    kept = [sample for sample in samples if has_usable_reconstruction(sample)]
    skipped = len(samples) - len(kept)
    if skipped:
        print(
            "[train_model] skipped "
            f"{skipped} samples without usable reconstruction metadata for train-time 3D augmentation",
            flush=True,
        )
    if not kept:
        raise ValueError("no usable samples remain after augmentation reconstruction filtering")
    return kept


def _split_group_key(sample: Mapping[str, Any]) -> Any:
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

    return ("sample_id", sample["sample_id"])


def _assert_disjoint_split(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
) -> None:
    train_sample_ids = {str(sample["sample_id"]) for sample in train_samples}
    val_sample_ids = {str(sample["sample_id"]) for sample in val_samples}
    overlapping_sample_ids = train_sample_ids & val_sample_ids
    if overlapping_sample_ids:
        preview = ", ".join(sorted(overlapping_sample_ids)[:3])
        raise ValueError(f"train/validation split leaked sample_ids across both sets: {preview}")

    train_group_keys = {_split_group_key(sample) for sample in train_samples}
    val_group_keys = {_split_group_key(sample) for sample in val_samples}
    overlapping_group_keys = train_group_keys & val_group_keys
    if overlapping_group_keys:
        preview = ", ".join(str(key) for key in sorted(overlapping_group_keys, key=str)[:3])
        raise ValueError(f"train/validation split leaked grouping identities across both sets: {preview}")


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
        groups[_split_group_key(sample)].append(sample)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    val_group_count = max(1, int(round(len(group_keys) * val_fraction)))
    val_keys = set(group_keys[:val_group_count])

    train_samples = [sample for key, group in groups.items() if key not in val_keys for sample in group]
    val_samples = [sample for key, group in groups.items() if key in val_keys for sample in group]

    if not train_samples or not val_samples:
        raise ValueError("split produced an empty train or validation set; reduce val_fraction or add more samples")
    _assert_disjoint_split(train_samples, val_samples)
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


def _load_resume_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(checkpoint).__name__}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"missing model_state_dict in checkpoint: {path}")
    return checkpoint


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_extended_validation_metrics(
    model: FeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
) -> dict[str, Any]:
    from .evaluate import _default_thresholds, compute_validation_metrics

    return compute_validation_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        score_thresholds=_default_thresholds(),
        target_threshold=0.5,
        amp=amp,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )


def _select_monitored_metric(
    metric_name: str,
    val_losses: Mapping[str, float],
    val_metrics: Mapping[str, Any] | None,
    pair_metrics: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    if metric_name == "val_total":
        return float(val_losses["total"]), "min"

    if metric_name == "pair_auc":
        if pair_metrics is None or pair_metrics.get("pair_auc") is None:
            raise ValueError("pair metrics are required to monitor pair_auc")
        return float(pair_metrics["pair_auc"]), "max"

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


def _make_augmentation_config(args: argparse.Namespace) -> AugmentationConfig:
    translation_values = tuple(float(value) for value in args.translation_jitter_px)
    pitch_roll_values = tuple(float(value) for value in args.pitch_roll_jitter_deg)
    translation_typical, translation_strong = sorted(translation_values)
    _, pitch_roll_strong = sorted(pitch_roll_values)
    scale_min, scale_max = sorted(float(value) for value in args.scale_jitter)
    return AugmentationConfig(
        count=args.augmentation_count,
        translation_typical_px=translation_typical,
        translation_strong_px=translation_strong,
        yaw_deg=float(args.yaw_jitter_deg),
        pitch_roll_typical_deg=pitch_roll_strong,
        pitch_roll_strong_deg=max(25.0, pitch_roll_strong),
        missing_reconstruction=args.augmentation_missing_reconstruction,
        debug_dir=args.augmentation_debug_dir,
        debug_limit=args.augmentation_debug_limit,
        reconstruction_cache_size=args.augmentation_reconstruction_cache_size,
        sample_cache_size=args.augmentation_sample_cache_size,
        group_variants=bool(args.augmentation_group_variants),
        photometric=bool(args.augmentation_photometric),
        scale_jitter_min=scale_min,
        scale_jitter_max=scale_max,
    )


def run_augmentation_probe(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_bundle_samples(
        args.ground_truth_root,
        limit=args.limit,
        strict_gradient_targets=args.strict_gradient_targets,
        strict_finite_targets=args.strict_finite_targets,
        skip_empty_minutia_support_with_minutiae=args.skip_empty_minutia_support_with_minutiae,
    )
    config = _make_augmentation_config(args)
    samples = _filter_samples_for_augmentation(samples, config)
    sample = samples[0]
    image_shape = tuple(_read_grayscale_image(sample["masked_image"]).shape)
    output_shape = tuple(int(value) for value in sample["output_shape_hw"])

    from .augmentation import AugmentationParams, augment_sample, reconstruction_cache_info, sample_cache_info

    probes = [
        AugmentationParams(True, True, False, False, 8.0, -8.0, min(3.0, config.yaw_deg), 0.0, 0.0),
        AugmentationParams(True, False, True, True, 4.0, -4.0, 0.0, 8.0, -8.0),
    ]
    records: list[dict[str, Any]] = []
    for params in probes:
        started = time.perf_counter()
        result = augment_sample(
            sample,
            params,
            output_shape=output_shape,
            debug_dir=args.augmentation_debug_dir,
            debug_stem=f"probe_{len(records)}" if args.augmentation_debug_dir is not None else None,
            reconstruction_cache_size=config.reconstruction_cache_size,
            sample_cache_size=config.sample_cache_size,
        )
        records.append(
            {
                "params": {
                    "dx": params.dx,
                    "dy": params.dy,
                    "yaw_deg": params.yaw_deg,
                    "pitch_deg": params.pitch_deg,
                    "roll_deg": params.roll_deg,
                },
                "seconds": round(time.perf_counter() - started, 3),
                "image_shape": list(result.image.shape),
                "mask_pixels": int(np.count_nonzero(result.mask)),
                "minutiae": len(result.minutiae),
                "mode": result.details.get("mode"),
            }
        )
    summary = {
        "sample_id": sample.get("sample_id"),
        "input_shape": list(image_shape),
        "output_shape": list(output_shape),
        "records": records,
        "reconstruction_cache": reconstruction_cache_info(),
        "sample_cache": sample_cache_info(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


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
        strict_finite_targets=args.strict_finite_targets,
        skip_empty_minutia_support_with_minutiae=args.skip_empty_minutia_support_with_minutiae,
    )
    augmentation_config = None
    if args.train_augmentations:
        augmentation_config = _make_augmentation_config(args)
        samples = _filter_samples_for_augmentation(samples, augmentation_config)
        first_image_shape = tuple(_read_grayscale_image(samples[0]["masked_image"]).shape)
        print(
            "[train_model] train augmentations enabled: "
            f"count={augmentation_config.count}, image_shape={first_image_shape}, "
            f"translation_typical/strong=({augmentation_config.translation_typical_px}, "
            f"{augmentation_config.translation_strong_px}), yaw_deg={augmentation_config.yaw_deg}, "
            f"pitch_roll_typical/strong=({augmentation_config.pitch_roll_typical_deg}, "
            f"{augmentation_config.pitch_roll_strong_deg}), "
            f"reconstruction_cache_size={augmentation_config.reconstruction_cache_size}, "
            f"sample_cache_size={augmentation_config.sample_cache_size}, "
            f"group_variants={augmentation_config.group_variants}",
            flush=True,
        )
        h, w = first_image_shape
        cache_mb = (float(h) * float(w) * 18.0 * float(augmentation_config.sample_cache_size)) / (1024.0 * 1024.0)
        print(
            "[train_model] estimated maximum sample-cache memory per worker: "
            f"{cache_mb:.1f} MiB (includes lazy full-resolution gradient when used)",
            flush=True,
        )
    train_samples, val_samples = split_samples(samples, val_fraction=args.val_fraction, seed=args.seed)
    resolved_device = _resolve_device(args.device)
    use_amp = bool(args.amp) and resolved_device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    use_pin_memory = bool(args.pin_memory) and resolved_device.type == "cuda"
    if args.cudnn_benchmark and resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_prefetch_factor = args.prefetch_factor
    if args.train_augmentations and args.num_workers > 0:
        if train_prefetch_factor is None or train_prefetch_factor > 1:
            print(
                "[train_model] reducing train DataLoader prefetch_factor to 1 for full-resolution augmentation",
                flush=True,
            )
        train_prefetch_factor = 1

    train_loader = create_dataloader(
        samples=train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=train_prefetch_factor,
        train_augmentations=bool(args.train_augmentations),
        augmentation_config=augmentation_config,
        seed=args.seed,
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
            train_augmentations=False,
        )

    model = _maybe_channels_last(FeatureExtractor().to(resolved_device), args.channels_last)
    criterion = FeatureNetLoss(
        orientation_weight=args.orientation_weight,
        ridge_weight=args.ridge_weight,
        gradient_weight=args.gradient_weight,
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
        m1_neg_weight=args.m1_neg_weight,
        m1_side_pos_weight=args.m1_side_pos_weight,
        m1_candidate_loss_weight=float(args.m1_candidate_weight),
        m1_candidate_topk=int(args.m1_candidate_topk),
        m1_candidate_match_radius=int(args.m1_candidate_match_radius),
        m1_candidate_focal_gamma=float(args.m1_candidate_focal_gamma),
    ).to(resolved_device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
    )
    scaler = _make_grad_scaler(use_amp and amp_dtype == torch.float16)

    consistency_config: ConsistencyConfig | None = None
    if args.consistency:
        consistency_config = ConsistencyConfig(
            weight=float(args.consistency_weight),
            max_rot_deg=float(args.consistency_max_rot_deg),
            max_shift_frac=float(args.consistency_max_shift_frac),
            max_scale_delta=float(args.consistency_max_scale),
            mode=str(args.consistency_mode),
            self_weight=float(args.consistency_self_weight),
            orientation_weight=float(args.consistency_orientation_weight),
            warmup_epochs=int(args.consistency_warmup_epochs),
            pos_boost=float(args.consistency_pos_boost),
        )
        print(
            "[train_model] in-step consistency loss enabled: "
            f"mode={consistency_config.mode}, weight={consistency_config.weight}, "
            f"self_weight={consistency_config.self_weight}, "
            f"orientation_weight={consistency_config.orientation_weight}, "
            f"warmup_epochs={consistency_config.warmup_epochs}, "
            f"max_rot_deg={consistency_config.max_rot_deg}, "
            f"max_shift_frac={consistency_config.max_shift_frac}, max_scale_delta={consistency_config.max_scale_delta}",
            flush=True,
        )

    resume_epoch = 0
    resume_metrics: Mapping[str, Any] = {}
    if args.resume_checkpoint is not None:
        resume_checkpoint = _load_resume_checkpoint(args.resume_checkpoint, resolved_device)
        try:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "resume checkpoint is incompatible with the current FeatureExtractor architecture"
            ) from exc
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        resume_metrics = resume_checkpoint.get("metrics", {})
        if resume_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {resume_epoch}; "
                f"increase --epochs above {resume_epoch} to continue"
            )

    model = _maybe_compile_model(model, args.compile and resolved_device.type == "cuda")

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
    if resume_metrics:
        monitor = resume_metrics.get("monitor")
        if isinstance(monitor, Mapping) and monitor.get("name") == monitor_name:
            best_monitor_value = float(monitor.get("best_value", best_monitor_value))
            monitor_mode = str(monitor.get("mode", monitor_mode))
        early_stopping_state = resume_metrics.get("early_stopping")
        if isinstance(early_stopping_state, Mapping):
            validation_checks = int(early_stopping_state.get("checks_run", validation_checks))
            patience_counter = int(early_stopping_state.get("patience_counter", patience_counter))

    for epoch in range(resume_epoch + 1, args.epochs + 1):
        epoch_started_at = time.time()

        if str(args.lr_schedule) == "cosine":
            min_lr = float(args.lr) * float(args.lr_min_factor)
            progress = (epoch - 1) / max(1, args.epochs - 1)
            epoch_lr = min_lr + 0.5 * (float(args.lr) - min_lr) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr

        effective_consistency = consistency_config
        if consistency_config is not None and consistency_config.warmup_epochs > 0:
            warmup_scale = min(1.0, epoch / float(consistency_config.warmup_epochs))
            if warmup_scale < 1.0:
                effective_consistency = dataclasses.replace(
                    consistency_config, weight=consistency_config.weight * warmup_scale
                )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            resolved_device,
            scaler=scaler,
            amp=use_amp,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            skip_non_finite_target_batches=args.skip_non_finite_target_batches,
            consistency=effective_consistency,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
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
                amp_dtype=amp_dtype,
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
                    amp_dtype=amp_dtype,
                    channels_last=args.channels_last,
                )
                record["val_metrics"] = extended_val_metrics

            pair_metrics: dict[str, Any] | None = None
            want_pair_for_selection = monitor_name == "pair_auc"
            pair_eval_enabled = bool(args.pair_eval) or want_pair_for_selection
            run_pair_this_epoch = bool(
                pair_eval_enabled
                and val_samples
                and (
                    want_pair_for_selection
                    or epoch % max(1, int(args.pair_eval_every)) == 0
                    or epoch == args.epochs
                )
            )
            if run_pair_this_epoch:
                from .pair_eval import evaluate_pairs

                pair_metrics = evaluate_pairs(
                    model,
                    val_samples,
                    resolved_device,
                    method=str(args.pair_eval_method),
                    score_threshold=float(args.pair_eval_score_threshold),
                    apply_nms=True,
                    unwarp=str(args.pair_eval_unwarp),
                    max_genuine=int(args.pair_eval_max_genuine),
                    max_impostor=int(args.pair_eval_max_impostor),
                    seed=int(args.seed),
                    repeat_dist_px=float(args.pair_eval_dist_px),
                    repeat_angle_deg=float(args.pair_eval_angle_deg),
                )
                if pair_metrics is not None:
                    record["pair_metrics"] = pair_metrics
                else:
                    record["pair_metrics_warning"] = "no_genuine_pairs"
                    print(
                        "[train_model] pair evaluation skipped: no genuine pairs could be formed "
                        "from the held-out validation fingers",
                        flush=True,
                    )

            monitor_available = True
            if monitor_name == "pair_auc":
                pair_auc_value = None if pair_metrics is None else pair_metrics.get("pair_auc")
                monitor_available = pair_auc_value is not None and bool(np.isfinite(float(pair_auc_value)))

            if monitor_available:
                current_monitor_value, monitor_mode = _select_monitored_metric(
                    monitor_name,
                    val_metrics,
                    extended_val_metrics,
                    pair_metrics,
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
            else:
                current_monitor_value = float("nan")
                improved = False
                patience_counter += 1
                print(
                    f"[train_model] monitored metric {monitor_name} unavailable this epoch; "
                    "treating as non-improvement",
                    flush=True,
                )

            record["monitor"] = {
                "name": monitor_name,
                "mode": monitor_mode,
                "value": current_monitor_value,
                "best_value": best_monitor_value,
                "improved": improved,
                "available": monitor_available,
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
        "sample_count_by_raw_view_index": _raw_view_counts(samples),
        "train_sample_count": len(train_samples),
        "train_sample_count_by_raw_view_index": _raw_view_counts(train_samples),
        "val_sample_count": len(val_samples),
        "val_sample_count_by_raw_view_index": _raw_view_counts(val_samples),
        "epochs": args.epochs,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "start_epoch": resume_epoch + 1,
        "resume_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "amp": use_amp,
        "amp_dtype": args.amp_dtype,
        "channels_last": bool(args.channels_last),
        "compile": bool(args.compile and resolved_device.type == "cuda"),
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "skip_non_finite_target_batches": bool(args.skip_non_finite_target_batches),
        "skip_empty_minutia_support_with_minutiae": bool(args.skip_empty_minutia_support_with_minutiae),
        "pin_memory": use_pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": train_prefetch_factor,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": float(args.weight_decay),
        "consistency_enabled": bool(args.consistency),
        "consistency_weight": float(args.consistency_weight) if args.consistency else 0.0,
        "consistency_mode": str(args.consistency_mode) if args.consistency else None,
        "consistency_self_weight": float(args.consistency_self_weight) if args.consistency else 0.0,
        "consistency_orientation_weight": float(args.consistency_orientation_weight) if args.consistency else 0.0,
        "consistency_warmup_epochs": int(args.consistency_warmup_epochs) if args.consistency else 0,
        "m1_candidate_weight": float(args.m1_candidate_weight),
        "m1_candidate_topk": int(args.m1_candidate_topk),
        "lr_schedule": str(args.lr_schedule),
        "lr_min_factor": float(args.lr_min_factor),
        "augmentation_photometric": bool(args.augmentation_photometric) if args.train_augmentations else False,
        "scale_jitter": list(args.scale_jitter) if args.train_augmentations else None,
        "seed": args.seed,
        "strict_gradient_targets": bool(args.strict_gradient_targets),
        "strict_finite_targets": bool(args.strict_finite_targets),
        "train_augmentations": bool(args.train_augmentations),
        "augmentation_count": int(args.augmentation_count) if args.train_augmentations else 0,
        "effective_train_dataset_length": len(train_loader.dataset),
        "augmentation_reconstruction_cache_size": args.augmentation_reconstruction_cache_size,
        "augmentation_sample_cache_size": args.augmentation_sample_cache_size,
        "augmentation_group_variants": bool(args.augmentation_group_variants),
        "orientation_weight": args.orientation_weight,
        "ridge_weight": args.ridge_weight,
        "gradient_weight": args.gradient_weight,
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
        "m1_neg_weight": args.m1_neg_weight,
        "m1_side_pos_weight": args.m1_side_pos_weight,
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
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume model and optimizer state from a training checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=max(1, min((os.cpu_count() or 1), 4)))
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Floating dtype to use inside CUDA autocast when --amp is enabled.",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on supported CUDA runtimes.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--m1-neg-weight", type=float, default=2.0)
    parser.add_argument("--m1-side-pos-weight", type=float, default=2.0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Clip gradient norm before optimizer steps; set to 0 to disable.",
    )
    parser.add_argument(
        "--skip-non-finite-target-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a training batch when non-finite loss is caused by non-finite target values.",
    )
    parser.add_argument(
        "--skip-empty-minutia-support-with-minutiae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip bundles whose metadata has minutiae but whose rasterized minutia target support is empty.",
    )
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--early-stopping", action="store_true", help="Stop training when the monitored validation metric plateaus.")
    parser.add_argument("--early-stopping-metric", choices=EARLY_STOPPING_METRICS, default="val_total")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--pair-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute held-out pair AUC/EER + repeatability during validation (auto-enabled when monitoring pair_auc).",
    )
    parser.add_argument("--pair-eval-every", type=int, default=1, help="Run pair evaluation every N validation epochs (logging only; selection on pair_auc always runs).")
    parser.add_argument("--pair-eval-method", type=str, default="LSA", help="MCC method for pair scoring.")
    parser.add_argument(
        "--pair-eval-unwarp",
        choices=("none", "gradient"),
        default="gradient",
        help="Route A unwarp mode applied to decoded minutiae before MCC during pair evaluation.",
    )
    parser.add_argument("--pair-eval-score-threshold", type=float, default=0.5, help="Minutia score threshold for pair-eval decoding.")
    parser.add_argument("--pair-eval-max-genuine", type=int, default=200, help="Cap on genuine pairs sampled for pair evaluation.")
    parser.add_argument("--pair-eval-max-impostor", type=int, default=200, help="Cap on impostor pairs sampled for pair evaluation.")
    parser.add_argument("--pair-eval-dist-px", type=float, default=25.0, help="Distance tolerance (px) for matcher-free repeatability.")
    parser.add_argument("--pair-eval-angle-deg", type=float, default=30.0, help="Angle tolerance (deg) for matcher-free repeatability.")
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--ridge-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--mu-score", type=float, default=120.0)
    parser.add_argument("--mu-x", type=float, default=20.0)
    parser.add_argument("--mu-y", type=float, default=20.0)
    parser.add_argument("--mu-ori", type=float, default=15.0)
    parser.add_argument("--m1-focal-gamma", type=float, default=2.0)
    parser.add_argument("--m1-pos-weight-max", type=float, default=100.0)
    parser.add_argument("--m1-hard-neg-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m1-hard-neg-ratio", type=float, default=20.0)
    parser.add_argument("--m1-hard-neg-min", type=int, default=2000)
    parser.add_argument("--m1-hard-neg-fraction", type=float, default=0.00)
    parser.add_argument(
        "--strict-gradient-targets",
        action="store_true",
        help="Fail fast if any selected sample is missing reconstruction-derived gradient targets.",
    )
    parser.add_argument(
        "--strict-finite-targets",
        action="store_true",
        help="Fail fast if any selected sample has NaN or Inf target values instead of skipping it.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-augmentations", action="store_true", help="Enable train-only on-the-fly supervised geometry augmentations.")
    parser.add_argument("--augmentation-count", type=int, default=5, help="Number of synthetic variants per eligible training sample.")
    parser.add_argument(
        "--translation-jitter-px",
        type=float,
        nargs=2,
        default=(16.0, 32.0),
        metavar=("TYPICAL", "STRONG"),
        help=(
            "Translation jitter typical/strong pixel ranges. Sampling includes small shifts: "
            "70%% in [0,TYPICAL] and 30%% in [TYPICAL,STRONG], with random sign per axis."
        ),
    )
    parser.add_argument(
        "--yaw-jitter-deg",
        type=float,
        default=15.0,
        help="Maximum residual in-plane yaw degrees sampled uniformly from [-value,+value].",
    )
    parser.add_argument(
        "--pitch-roll-jitter-deg",
        type=float,
        nargs=2,
        default=(5.0, 15.0),
        metavar=("TYPICAL_MIN", "TYPICAL_MAX"),
        help=(
            "Pitch/roll CLI compatibility range. The upper value is used as the typical "
            "mild limit, so small pitch/roll values are included; rare strong samples may reach +/-25 deg."
        ),
    )
    parser.add_argument(
        "--augmentation-missing-reconstruction",
        choices=("skip", "allow"),
        default="skip",
        help="How to handle samples without reconstruction metadata when train augmentations are enabled.",
    )
    parser.add_argument("--augmentation-debug-dir", type=Path, default=None, help="Optional directory for before/after augmentation debug PNGs.")
    parser.add_argument("--augmentation-debug-limit", type=int, default=0, help="Maximum debug examples per synthetic variant access counter.")
    parser.add_argument(
        "--augmentation-reconstruction-cache-size",
        type=int,
        default=2,
        help="Maximum reconstruction map entries cached inside each DataLoader worker.",
    )
    parser.add_argument(
        "--augmentation-sample-cache-size",
        type=int,
        default=8,
        help="Maximum full-resolution sample bundles cached inside each DataLoader worker.",
    )
    parser.add_argument(
        "--augmentation-group-variants",
        action="store_true",
        help="Shuffle base samples, then emit original plus all augmentation variants together for better worker cache hits.",
    )
    parser.add_argument(
        "--augmentation-photometric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply photometric jitter (brightness/contrast/gamma/blur/noise/JPEG/illumination/glare) to synthetic variants.",
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        nargs=2,
        default=(0.9, 1.1),
        metavar=("MIN", "MAX"),
        help="Uniform in-plane scale jitter range applied to synthetic 2D-affine variants.",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Enable the in-step affine+photometric equivariance loss on the minutia score head.",
    )
    parser.add_argument("--consistency-weight", type=float, default=25.0, help="Weight of the consistency term added to the total loss.")
    parser.add_argument("--consistency-max-rot-deg", type=float, default=15.0, help="Max in-plane rotation (deg) of the consistency partner view.")
    parser.add_argument("--consistency-max-shift-frac", type=float, default=0.06, help="Max translation (fraction of half-size) of the consistency partner view.")
    parser.add_argument("--consistency-max-scale", type=float, default=0.10, help="Max scale delta of the consistency partner view.")
    parser.add_argument(
        "--consistency-mode",
        choices=("supervised", "self"),
        default="supervised",
        help=(
            "supervised: warped view supervised against warped GT (no score-collapse shortcut), "
            "plus peak-weighted self-consistency and orientation equivariance. "
            "self: legacy masked-MSE between the two score maps."
        ),
    )
    parser.add_argument(
        "--consistency-self-weight",
        type=float,
        default=0.2,
        help="Relative weight of the peak-weighted self-consistency term inside supervised mode.",
    )
    parser.add_argument(
        "--consistency-orientation-weight",
        type=float,
        default=1.0,
        help="Relative weight of the minutia-orientation equivariance term inside supervised mode.",
    )
    parser.add_argument(
        "--consistency-warmup-epochs",
        type=int,
        default=5,
        help="Linearly ramp the consistency weight from 0 to full over this many epochs (0 disables warmup).",
    )
    parser.add_argument(
        "--consistency-pos-boost",
        type=float,
        default=20.0,
        help="Up-weight of warped positive cells in the supervised consistency term (keeps score collapse expensive).",
    )
    parser.add_argument(
        "--m1-candidate-weight",
        type=float,
        default=10.0,
        help="Absolute weight of the candidate-level focal FP-suppression loss (0 disables it).",
    )
    parser.add_argument("--m1-candidate-topk", type=int, default=512, help="Top-K model-scored cells fed to the candidate loss.")
    parser.add_argument("--m1-candidate-match-radius", type=int, default=1, help="Cell radius around GT positives counted as candidate positives.")
    parser.add_argument("--m1-candidate-focal-gamma", type=float, default=2.0, help="Focal gamma of the candidate loss.")
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="Learning-rate schedule over epochs (cosine anneals from --lr to --lr * --lr-min-factor).",
    )
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=0.05,
        help="Final LR as a fraction of --lr when --lr-schedule cosine is used.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay; 0 disables decay (equivalent to Adam).")
    parser.add_argument(
        "--augmentation-probe",
        action="store_true",
        help="Load one eligible sample, run forced 2D/3D augmentations, print timing/shape diagnostics, and exit.",
    )
    args = parser.parse_args()
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        parser.error(f"--resume-checkpoint does not exist: {args.resume_checkpoint}")
    if args.validate_every < 1:
        parser.error("--validate-every must be at least 1")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")
    if args.early_stopping_min_delta < 0.0:
        parser.error("--early-stopping-min-delta must be non-negative")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be at least 1")
    if args.augmentation_count < 0:
        parser.error("--augmentation-count must be non-negative")
    if any(float(value) < 0.0 for value in args.translation_jitter_px):
        parser.error("--translation-jitter-px values must be non-negative")
    if max(args.translation_jitter_px) <= 0.0 and args.train_augmentations:
        parser.error("--translation-jitter-px must include a positive value when augmentations are enabled")
    if args.yaw_jitter_deg < 0.0:
        parser.error("--yaw-jitter-deg must be non-negative")
    if any(float(value) < 0.0 for value in args.pitch_roll_jitter_deg):
        parser.error("--pitch-roll-jitter-deg values must be non-negative")
    if args.augmentation_debug_limit < 0:
        parser.error("--augmentation-debug-limit must be non-negative")
    if args.augmentation_reconstruction_cache_size < 1:
        parser.error("--augmentation-reconstruction-cache-size must be at least 1")
    if args.augmentation_sample_cache_size < 1:
        parser.error("--augmentation-sample-cache-size must be at least 1")
    if args.max_grad_norm < 0.0:
        parser.error("--max-grad-norm must be non-negative")
    if args.amp and args.amp_dtype == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        parser.error("--amp-dtype bf16 requested, but this CUDA device does not report bfloat16 support")
    if args.orientation_weight < 0.0 or args.ridge_weight < 0.0 or args.gradient_weight < 0.0:
        parser.error("--orientation-weight, --ridge-weight, and --gradient-weight must be non-negative")
    if args.mu_score < 0.0 or args.mu_x < 0.0 or args.mu_y < 0.0 or args.mu_ori < 0.0:
        parser.error("--mu-score, --mu-x, --mu-y, and --mu-ori must be non-negative")
    if args.m1_focal_gamma < 0.0:
        parser.error("--m1-focal-gamma must be non-negative")
    if args.m1_pos_weight_max < 1.0:
        parser.error("--m1-pos-weight-max must be at least 1.0")
    if args.m1_neg_weight <= 0.0:
        parser.error("--m1-neg-weight must be positive")
    if args.m1_side_pos_weight <= 0.0:
        parser.error("--m1-side-pos-weight must be positive")
    if args.m1_hard_neg_ratio < 0.0:
        parser.error("--m1-hard-neg-ratio must be non-negative")
    if args.m1_hard_neg_min < 0:
        parser.error("--m1-hard-neg-min must be non-negative")
    if not 0.0 <= args.m1_hard_neg_fraction <= 1.0:
        parser.error("--m1-hard-neg-fraction must be in [0.0, 1.0]")
    if any(float(value) <= 0.0 for value in args.scale_jitter):
        parser.error("--scale-jitter values must be positive")
    if args.consistency_weight < 0.0:
        parser.error("--consistency-weight must be non-negative")
    if args.consistency_max_rot_deg < 0.0:
        parser.error("--consistency-max-rot-deg must be non-negative")
    if args.consistency_max_shift_frac < 0.0:
        parser.error("--consistency-max-shift-frac must be non-negative")
    if args.consistency_max_scale < 0.0:
        parser.error("--consistency-max-scale must be non-negative")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.augmentation_probe:
        run_augmentation_probe(args)
        return
    summary = train_model(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
