from __future__ import annotations

import importlib.util
import json
import math
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()
sys.path.append(str(Path(__file__).resolve().parents[1]))

from featurenet.models.augmentation import (  # noqa: E402
    AugmentationConfig,
    AugmentationParams,
    _RECONSTRUCTION_CACHE,
    _SAMPLE_CACHE,
    _load_reconstruction_maps,
    _project_3d_point,
    _transform_minutiae_3d,
    augment_sample,
    render_3d_augmented,
    rotation_matrix,
    sample_augmentation_params,
)
from featurenet.models.feature_extractor import FeatureExtractor  # noqa: E402
from featurenet.models.losses import FeatureNetLoss  # noqa: E402
from featurenet.models.target_rasterization import build_featurenet_targets  # noqa: E402
from featurenet.models.train import (  # noqa: E402
    FeatureNetDataset,
    FeatureNetGroupedVariantSampler,
    _run_model_step,
    create_dataloader,
    load_bundle_samples,
    train_one_epoch,
)


def _write_bundle(root: Path, sample_id: str = "sample_001") -> dict[str, Path]:
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    height = width = 16
    image = np.zeros((height, width), dtype=np.uint8)
    image[4:12, 4:12] = np.arange(64, dtype=np.uint8).reshape(8, 8) + 80
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[3:13, 3:13] = 255
    orientation = np.full((height, width), 0.25, dtype=np.float32)
    ridge = np.ones((height, width), dtype=np.float32) * 10.0
    gradient = np.zeros((height, width, 2), dtype=np.float32)
    minutiae = [{"x": 8.0, "y": 8.0, "theta": 0.25, "score": 1.0, "type": "ending"}]
    targets = build_featurenet_targets(image, mask, orientation, ridge, gradient, minutiae, output_shape=(2, 2))

    cv2.imwrite(str(sample_dir / "masked_image.png"), image)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)
    np.save(sample_dir / "orientation.npy", orientation)
    np.save(sample_dir / "ridge_period.npy", ridge)
    np.savez_compressed(sample_dir / "featurenet_targets.npz", **targets)
    (sample_dir / "minutiae.json").write_text(json.dumps(minutiae), encoding="utf-8")

    yy, xx = np.indices((height, width), dtype=np.float32)
    reconstruction_dir = root / "reconstructions" / f"acq_{sample_id}"
    reconstruction_dir.mkdir(parents=True)
    reconstruction_maps_path = reconstruction_dir / "reconstruction_maps.npz"
    np.savez_compressed(
        reconstruction_maps_path,
        support_mask=(mask > 0).astype(np.uint8),
        front_pose_x_map=xx,
        front_pose_y_map=yy,
        depth_front=np.zeros((height, width), dtype=np.float32),
    )
    meta = {
        "sample_id": sample_id,
        "subject_id": 1,
        "finger_id": 1,
        "acquisition_id": 1,
        "finger_class_id": 0,
        "raw_view_index": 0,
        "counts": {"minutiae": 1},
        "shapes": {"input_image": [height, width], "featurenet_output": [2, 2]},
        "multiview_reconstruction": {
            "role": "front",
            "reconstruction_dir": str(reconstruction_dir),
            "reconstruction_maps_path": str(reconstruction_maps_path),
        },
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"sample_dir": sample_dir, "reconstruction_maps": reconstruction_maps_path}


class FeatureNetAugmentationTests(unittest.TestCase):
    def test_dataset_expansion_variant_zero_and_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=2, missing_reconstruction="skip")
            dataset = FeatureNetDataset(samples, augmentation_config=config, seed=7)
            original_dataset = FeatureNetDataset(samples)

            self.assertEqual(len(dataset), 3)
            original_input, original_targets = original_dataset[0]
            variant_zero_input, variant_zero_targets = dataset[0]
            self.assertTrue(torch.equal(original_input, variant_zero_input))
            self.assertTrue(torch.equal(original_targets["minutia_valid_mask"], variant_zero_targets["minutia_valid_mask"]))

            first_aug, _ = dataset[1]
            second_aug, _ = dataset[1]
            self.assertFalse(torch.equal(first_aug, second_aug))

    def test_rotation_axes_follow_camera_convention(self) -> None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        np.testing.assert_allclose(rotation_matrix(90, 0, 0) @ x_axis, y_axis, atol=1e-5)
        np.testing.assert_allclose(rotation_matrix(0, 90, 0) @ y_axis, z_axis, atol=1e-5)
        np.testing.assert_allclose(rotation_matrix(0, 0, 90) @ z_axis, x_axis, atol=1e-5)

    def test_sampling_includes_small_translation_and_mild_pitch_roll(self) -> None:
        rng = np.random.default_rng(123)
        config = AugmentationConfig(count=1, translation_typical_px=16.0, translation_strong_px=32.0)
        params = [sample_augmentation_params(rng, config) for _ in range(400)]

        translations = [abs(p.dx) for p in params if p.translate]
        pitch_roll = [abs(value) for p in params for value in (p.pitch_deg, p.roll_deg) if abs(value) > 0.0]

        self.assertTrue(any(value < 2.0 for value in translations))
        self.assertTrue(any(value < 5.0 for value in pitch_roll))

    def test_translation_rerasterizes_sparse_minutiae_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            sample = load_bundle_samples(root, strict_gradient_targets=True)[0]
            params = AugmentationParams(
                translate=True,
                yaw=False,
                pitch=False,
                roll=False,
                dx=4.0,
                dy=0.0,
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=0.0,
            )
            result = augment_sample(sample, params, output_shape=(2, 2))

            self.assertAlmostEqual(result.minutiae[0]["x"], 12.0)
            self.assertAlmostEqual(result.minutiae[0]["y"], 8.0)
            self.assertAlmostEqual(result.minutiae[0]["theta"], 0.25)
            self.assertEqual(float(result.targets["minutia_valid_mask"][0, 1, 1]), 1.0)
            self.assertEqual(int(result.targets["minutia_x"][1, 1]), 4)
            self.assertEqual(int(result.targets["minutia_y"][1, 1]), 0)
            self.assertAlmostEqual(float(result.targets["minutia_x_offset"][0, 1, 1]), 0.5)
            self.assertAlmostEqual(float(result.targets["minutia_y_offset"][0, 1, 1]), 0.0)

    def test_zero_rotation_render_reproduces_flat_projection_layout(self) -> None:
        image = np.full((8, 8), 120, dtype=np.uint8)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        orientation = np.zeros((8, 8), dtype=np.float32)
        ridge = np.ones((8, 8), dtype=np.float32)
        yy, xx = np.indices((8, 8), dtype=np.float32)
        reconstruction = {
            "support_mask": np.ones((8, 8), dtype=np.uint8),
            "front_pose_x_map": xx,
            "front_pose_y_map": yy,
            "depth_front": np.zeros((8, 8), dtype=np.float32),
        }
        params = AugmentationParams(False, False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0)

        rendered = render_3d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge,
            gradient=None,
            minutiae=[],
            reconstruction=reconstruction,
            role="front",
            params=params,
        )

        self.assertGreater(np.count_nonzero(rendered["mask"]), 0)
        self.assertLess(float(np.mean(np.abs(rendered["image"].astype(np.float32) - image.astype(np.float32)))), 1.0)

    def test_3d_orientation_update_uses_projected_tangent(self) -> None:
        depth = np.zeros((16, 16), dtype=np.float32)
        r = rotation_matrix(35.0, 10.0, 0.0)
        minutiae = [{"x": 8.0, "y": 8.0, "theta": 0.0, "score": 1.0}]
        transformed = _transform_minutiae_3d(
            minutiae=minutiae,
            depth_image=depth,
            r=r,
            cx=7.5,
            cy=7.5,
            depth_center=0.0,
            dx=0.0,
            dy=0.0,
            rendered_mask=np.full((16, 16), 255, dtype=np.uint8),
        )
        p0 = _project_3d_point(8.0, 8.0, 0.0, r, 7.5, 7.5, 0.0, 0.0, 0.0)
        p1 = _project_3d_point(11.0, 8.0, 0.0, r, 7.5, 7.5, 0.0, 0.0, 0.0)
        expected = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
        self.assertAlmostEqual(transformed[0]["theta"], expected, places=5)

    def test_determinism_for_same_seed_and_access_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=1, missing_reconstruction="skip")
            left = FeatureNetDataset(samples, augmentation_config=config, seed=99)
            right = FeatureNetDataset(samples, augmentation_config=config, seed=99)

            left_input, left_targets = left[1]
            right_input, right_targets = right[1]

            self.assertTrue(torch.equal(left_input, right_input))
            self.assertTrue(torch.equal(left_targets["minutia_score"], right_targets["minutia_score"]))

    def test_reconstruction_cache_loads_role_keys_and_evicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _RECONSTRUCTION_CACHE.clear()
            root = Path(directory)
            yy, xx = np.indices((4, 4), dtype=np.float32)
            for index in range(3):
                np.savez_compressed(
                    root / f"maps_{index}.npz",
                    support_mask=np.ones((4, 4), dtype=np.uint8),
                    front_pose_x_map=xx,
                    front_pose_y_map=yy,
                    depth_front=np.ones((4, 4), dtype=np.float32) * index,
                    unused_large=np.ones((4, 4), dtype=np.float32) * 99,
                )

            first = _load_reconstruction_maps(root / "maps_0.npz", role="front", max_cache_size=2)
            self.assertEqual(set(first), {"support_mask", "front_pose_x_map", "front_pose_y_map", "depth_front"})
            _load_reconstruction_maps(root / "maps_1.npz", role="front", max_cache_size=2)
            _load_reconstruction_maps(root / "maps_2.npz", role="front", max_cache_size=2)

            self.assertEqual(len(_RECONSTRUCTION_CACHE), 2)
            self.assertFalse(any("maps_0.npz" in key[0] for key in _RECONSTRUCTION_CACHE))

    def test_sample_cache_evicts_and_reuses_loaded_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _SAMPLE_CACHE.clear()
            root = Path(directory)
            _write_bundle(root, "sample_a")
            _write_bundle(root, "sample_b")
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=1, sample_cache_size=1)
            params = AugmentationParams(True, False, False, False, 1.0, 0.0, 0.0, 0.0, 0.0)

            augment_sample(samples[0], params, output_shape=(2, 2), sample_cache_size=config.sample_cache_size)
            self.assertEqual(list(_SAMPLE_CACHE.keys()), ["sample_a"])
            augment_sample(samples[1], params, output_shape=(2, 2), sample_cache_size=config.sample_cache_size)
            self.assertEqual(list(_SAMPLE_CACHE.keys()), ["sample_b"])

    def test_grouped_variant_sampler_keeps_variants_adjacent(self) -> None:
        sampler = FeatureNetGroupedVariantSampler(sample_count=3, variants_per_sample=3, seed=4)
        values = list(iter(sampler))
        self.assertEqual(len(values), 9)
        for offset in range(0, len(values), 3):
            group = values[offset : offset + 3]
            base_indices = {value // 3 for value in group}
            variant_indices = [value % 3 for value in group]
            self.assertEqual(len(base_indices), 1)
            self.assertEqual(variant_indices, [0, 1, 2])

    def test_augmented_batch_collates_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=1, missing_reconstruction="skip", group_variants=True)
            loader = create_dataloader(
                samples,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                train_augmentations=True,
                augmentation_config=config,
                seed=9,
            )

            _, targets = next(iter(loader))

            self.assertIn("raw_view_index", targets)
            self.assertIn("input_shape_hw", targets)
            self.assertIn("output_shape_hw", targets)
            self.assertEqual(tuple(targets["raw_view_index"].shape), (2,))
            self.assertEqual(tuple(targets["input_shape_hw"].shape), (2, 2))
            self.assertEqual(tuple(targets["output_shape_hw"].shape), (2, 2))
            self.assertTrue(torch.equal(targets["raw_view_index"], torch.zeros(2, dtype=torch.long)))
            self.assertTrue(torch.equal(targets["_augmentation_variant_index"], torch.tensor([0, 1])))

    def test_real_sized_synthetic_3d_variant_completes(self) -> None:
        height, width = 384, 416
        yy, xx = np.indices((height, width), dtype=np.float32)
        image = np.where((xx > 40) & (xx < width - 40) & (yy > 40) & (yy < height - 40), 120, 0).astype(np.uint8)
        mask = np.where(image > 0, 255, 0).astype(np.uint8)
        orientation = np.zeros((height, width), dtype=np.float32)
        ridge = np.ones((height, width), dtype=np.float32)
        reconstruction = {
            "support_mask": (mask > 0).astype(np.uint8),
            "front_pose_x_map": xx,
            "front_pose_y_map": yy,
            "depth_front": np.sin(xx / 30.0).astype(np.float32) * 4.0,
        }
        params = AugmentationParams(True, False, True, True, 3.0, -2.0, 0.0, 6.0, -6.0)

        rendered = render_3d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge,
            gradient=None,
            minutiae=[{"x": width / 2, "y": height / 2, "theta": 0.0, "score": 1.0}],
            reconstruction=reconstruction,
            role="front",
            params=params,
        )

        self.assertEqual(rendered["image"].shape, (height, width))
        self.assertGreater(np.count_nonzero(rendered["mask"]), 1000)

    def test_augmented_outputs_are_finite_and_training_step_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=1, missing_reconstruction="skip")
            loader = create_dataloader(
                samples,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                train_augmentations=True,
                augmentation_config=config,
                seed=5,
            )
            batch_iter = iter(loader)
            next(batch_iter)
            inputs, targets = next(batch_iter)

            self.assertTrue(torch.isfinite(inputs).all().item())
            for key, value in targets.items():
                if torch.is_floating_point(value):
                    self.assertTrue(torch.isfinite(value).all().item(), key)

            model = FeatureExtractor()
            model.eval()
            criterion = FeatureNetLoss()
            losses = _run_model_step(model, criterion, inputs, targets, amp=False)
            self.assertTrue(torch.isfinite(losses["total"]).item())

    def test_train_one_epoch_reports_timing_and_augmentation_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)
            config = AugmentationConfig(count=1, missing_reconstruction="skip")
            loader = create_dataloader(
                samples,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                train_augmentations=True,
                augmentation_config=config,
                seed=11,
            )
            model = FeatureExtractor()
            criterion = FeatureNetLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            metrics = train_one_epoch(
                model=model,
                dataloader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device="cpu",
                grad_accum_steps=1,
            )

            self.assertIn("data_wait_seconds_per_batch", metrics)
            self.assertIn("forward_loss_seconds_per_batch", metrics)
            self.assertEqual(metrics["augmentation_original_batches"], 1.0)
            self.assertEqual(
                metrics["augmentation_original_batches"]
                + metrics["augmentation_2d_batches"]
                + metrics["augmentation_3d_batches"]
                + metrics["augmentation_unknown_batches"],
                2.0,
            )


if __name__ == "__main__":
    unittest.main()
