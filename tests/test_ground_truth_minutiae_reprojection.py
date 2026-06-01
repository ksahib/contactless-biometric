from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import generate_ground_truth as gt


class GroundTruthMinutiaeReprojectionTests(unittest.TestCase):
    def _make_reconstruction(self, tmp_path: Path) -> gt.AcquisitionReconstructionResult:
        return gt.AcquisitionReconstructionResult(
            acquisition_id="s01_f01_a01",
            reconstruction_dir=str(tmp_path),
            depth_front_path="",
            depth_left_path="",
            depth_right_path="",
            depth_gradient_labels_path="",
            reconstruction_maps_path=str(tmp_path / "reconstruction_maps.npz"),
            support_mask_path="",
            row_measurements_path="",
            meta_path="",
            preview_path="",
            algorithm1_depth_then_algorithm3_unwarp_dir=str(tmp_path / "algorithm1_depth_then_algorithm3_unwarp"),
            center_unwarp_maps_path=str(tmp_path / "center_unwarp_maps.npz"),
            center_unwarped_image_path="center_unwarped.png",
            center_unwarped_mask_path="center_unwarped_mask.png",
            surface_front_3d_html_path="",
            surface_front_3d_png_path="",
            surface_all_branches_3d_html_path="",
            surface_all_branches_3d_png_path="",
            reprojection_report_path="",
            reprojection_preview_path="",
            valid_row_count=1,
            support_pixel_count=1,
            input_view_paths={},
            debug_view_paths={},
        )

    def _make_sample(self, tmp_path: Path, *, raw_view_index: int = 0) -> gt.RawViewSample:
        return gt.RawViewSample(
            sample_id=f"s01_f01_a01_v{raw_view_index:02d}",
            subject_id=1,
            subject_index=0,
            finger_id=1,
            acquisition_id=1,
            finger_class_id=0,
            raw_image_path=str(tmp_path / f"raw_{raw_view_index}.png"),
            raw_view_index=raw_view_index,
            sire_path=None,
            raw_view_paths=[str(tmp_path / f"raw_{idx}.png") for idx in range(3)],
            variant_paths={},
            is_extra_acquisition=False,
        )

    def _make_prepared(
        self,
        tmp_path: Path,
        *,
        raw_view_index: int = 0,
        reconstruction: gt.AcquisitionReconstructionResult | None = None,
    ) -> gt.PreparedBundleArtifacts:
        sample = self._make_sample(tmp_path, raw_view_index=raw_view_index)
        return gt.PreparedBundleArtifacts(
            sample=sample,
            bundle_dir=tmp_path / "bundle",
            image_path=Path(sample.raw_image_path),
            preprocessed=gt.PreprocessedContactlessImage(
                raw_gray=np.zeros((16, 16), dtype=np.uint8),
                normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                pose_normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                pose_normalized_mask=np.ones((16, 16), dtype=np.uint8),
                preprocessed_gray=np.zeros((16, 16), dtype=np.uint8),
                final_mask=np.ones((16, 16), dtype=np.uint8),
                mask_source="test",
                pose_rotation_degrees=0.0,
                ridge_scale_factor=1.0,
            ),
            gray_image=np.zeros((16, 16), dtype=np.uint8),
            mask=np.ones((16, 16), dtype=np.uint8),
            orientation=np.zeros((16, 16), dtype=np.float32),
            ridge_period=np.zeros((16, 16), dtype=np.float32),
            visualization_gradient=np.zeros((16, 16, 2), dtype=np.float32),
            reconstruction_gradient=None,
            masked_image=np.zeros((16, 16), dtype=np.uint8),
            enhanced_image=np.zeros((16, 16), dtype=np.uint8),
            visualize=False,
            reconstruction=reconstruction,
        )

    def test_downsample_mask_for_points_keeps_any_foreground_pixel(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[1, 1] = 255

        point_mask = gt._downsample_mask_for_points(mask, (2, 2))

        self.assertEqual(point_mask[0, 0], 1.0)
        self.assertEqual(int(np.count_nonzero(gt._downsample_mask_for_points(np.zeros_like(mask), (2, 2)))), 0)

    def test_rasterize_minutia_survives_point_mask_cell(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[1, 1] = 255
        output_shape = (2, 2)
        point_mask = gt._downsample_mask_for_points(mask, output_shape)
        minutiae = [{"x": 1.0, "y": 1.0, "theta": 0.0, "score": 1.0}]

        targets = gt._rasterize_minutiae(minutiae, mask.shape, output_shape, point_mask)

        self.assertEqual(int(np.count_nonzero(targets["minutia_valid_mask"])), 1)

    def test_side_projection_uses_saved_rotated_x_map(self):
        support = np.ones((3, 3), dtype=np.uint8)
        maps = {
            "support_mask": support,
            "x_relative": np.full((3, 3), 10.0, dtype=np.float32),
            "depth_front": np.full((3, 3), 999.0, dtype=np.float32),
            "x_left_rot": np.full((3, 3), -7.0, dtype=np.float32),
            "x_right_rot": np.full((3, 3), 8.0, dtype=np.float32),
            "front_centers": np.zeros(3, dtype=np.float32),
            "left_centers": np.full(3, 100.0, dtype=np.float32),
            "right_centers": np.full(3, 200.0, dtype=np.float32),
            "front_valid_rows": np.ones(3, dtype=np.uint8),
            "left_valid_rows": np.ones(3, dtype=np.uint8),
            "right_valid_rows": np.ones(3, dtype=np.uint8),
        }

        self.assertEqual(gt._project_front_source_to_pose_frame(maps, "left", 1.0, 1.0), (93.0, 1.0))
        self.assertEqual(gt._project_front_source_to_pose_frame(maps, "right", 1.0, 1.0), (208.0, 1.0))

    def test_map_row_between_views_uses_normalized_position(self):
        front_rows = np.zeros(140, dtype=np.uint8)
        left_rows = np.zeros(160, dtype=np.uint8)
        front_rows[10:111] = 1
        left_rows[20:121] = 1

        self.assertEqual(gt._map_row_between_views(front_rows, left_rows, 10.0), 20.0)
        self.assertEqual(gt._map_row_between_views(front_rows, left_rows, 110.0), 120.0)
        self.assertLess(abs(gt._map_row_between_views(front_rows, left_rows, 60.0) - 70.0), 1e-6)

    def test_training_frame_scale_uses_actual_array_shapes(self):
        pp = gt.PreprocessedContactlessImage(
            raw_gray=np.zeros((1, 1), dtype=np.uint8),
            normalized_gray=np.zeros((1, 1), dtype=np.uint8),
            pose_normalized_gray=np.zeros((100, 200), dtype=np.uint8),
            pose_normalized_mask=np.zeros((100, 200), dtype=np.uint8),
            preprocessed_gray=np.zeros((151, 301), dtype=np.uint8),
            final_mask=np.zeros((151, 301), dtype=np.uint8),
            mask_source="test",
            pose_rotation_degrees=0.0,
            ridge_scale_factor=1.5,
        )

        sx, sy = gt._training_frame_scale_from_preprocessed(pp)

        self.assertLess(abs(sx - 301 / 200), 1e-6)
        self.assertLess(abs(sy - 151 / 100), 1e-6)

    def test_canonical_filter_counts_mask_and_inverse_failures(self):
        unwarped_mask = np.zeros((5, 5), dtype=np.uint8)
        unwarped_mask[2, 2] = 1
        source_x = np.full((5, 5), np.nan, dtype=np.float32)
        source_y = np.full((5, 5), np.nan, dtype=np.float32)
        source_x[2, 2] = 1.0
        source_y[2, 2] = 1.0
        valid = np.zeros((5, 5), dtype=np.uint8)
        valid[2, 2] = 1
        maps = {
            "unwarped_mask": unwarped_mask,
            "source_valid_mask": valid,
            "source_x_map": source_x,
            "source_y_map": source_y,
        }
        minutiae = [
            {"x": 2.0, "y": 2.0, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 100.0, "y": 100.0, "theta": 0.0},
        ]

        kept, counters = gt._canonical_minutiae_with_front_sources(minutiae, maps)

        self.assertEqual(len(kept), 1)
        self.assertEqual(counters["canonical_total"], 3)
        self.assertEqual(counters["dropped_outside_unwarped_mask"], 1)
        self.assertEqual(counters["dropped_outside_unwarped_bounds"], 1)

    def test_cpu_segment_preprocessing_does_not_call_rembg(self):
        raw = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:56, 16:48] = 255
        final_image = np.full((40, 32), 90, dtype=np.uint8)
        final_mask = np.ones((40, 32), dtype=np.uint8) * 255

        fake_preprocess = gt.SimpleNamespace(
            _masked_clahe=mock.Mock(return_value=np.full_like(raw, 80)),
            circular_mask=mock.Mock(return_value=np.where(mask > 0, 255, 0).astype(np.uint8)),
            scale_to_paper_ridge_period=mock.Mock(
                return_value=(
                    np.full((40, 32), 85, dtype=np.uint8),
                    final_mask.copy(),
                    12.5,
                    0.8,
                )
            ),
            rotate_to_vertical_centerline=mock.Mock(return_value=(final_image, final_mask, -6.0)),
        )

        with mock.patch.object(gt, "rembg_mask_from_bgr", side_effect=AssertionError("rembg should stay main-process")):
            with (
                mock.patch.object(gt, "solov2_preprocess", fake_preprocess),
                mock.patch.object(gt, "normalise_brightness_array", side_effect=AssertionError("old brightness preprocessing should not run")),
                mock.patch.object(gt, "_estimate_pose_rotation", side_effect=AssertionError("old pose preprocessing should not run")),
                mock.patch.object(gt, "_normalize_ridge_frequency", side_effect=AssertionError("old ridge preprocessing should not run")),
            ):
                preprocessed = gt._preprocess_contactless_segment_cpu(raw, mask, "test_mask")

        self.assertEqual(preprocessed.mask_source, "test_mask_canonical_preprocess")
        self.assertTrue(np.array_equal(preprocessed.normalized_gray, np.full_like(raw, 80)))
        self.assertTrue(np.array_equal(preprocessed.preprocessed_gray, final_image))
        self.assertTrue(np.array_equal(preprocessed.pose_normalized_gray, final_image))
        self.assertTrue(np.array_equal(preprocessed.final_mask, final_mask))
        self.assertTrue(np.array_equal(preprocessed.pose_normalized_mask, final_mask))
        self.assertAlmostEqual(preprocessed.pose_rotation_degrees, -6.0)
        self.assertAlmostEqual(preprocessed.ridge_scale_factor, 0.8)

    def test_bundle_write_uses_canonical_final_image_for_training_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory) / "bundle"
            final_image = np.arange(16, dtype=np.uint8).reshape(4, 4)
            final_mask = np.zeros((4, 4), dtype=np.uint8)
            final_mask[1:3, 1:3] = 255
            masked_image = final_image.copy()
            masked_image[final_mask <= 0] = 0
            payload = gt.BundleWritePayload(
                bundle_dir=bundle_dir,
                preprocessed=gt.PreprocessedContactlessImage(
                    raw_gray=np.zeros((4, 4), dtype=np.uint8),
                    normalized_gray=np.full((4, 4), 10, dtype=np.uint8),
                    pose_normalized_gray=final_image,
                    pose_normalized_mask=final_mask,
                    preprocessed_gray=final_image,
                    final_mask=final_mask,
                    mask_source="solov2_canonical_preprocess",
                    pose_rotation_degrees=3.0,
                    ridge_scale_factor=1.2,
                ),
                gray_image=final_image,
                mask=final_mask,
                orientation=np.zeros((4, 4), dtype=np.float32),
                ridge_period=np.zeros((4, 4), dtype=np.float32),
                visualization_gradient=np.zeros((4, 4, 2), dtype=np.float32),
                masked_image=masked_image,
                enhanced_image=np.full((4, 4), 20, dtype=np.uint8),
                minutiae=[],
                featurenet_targets={"output_mask": np.ones((1, 1), dtype=np.float32)},
                meta={"sample_id": "test"},
                visualize=False,
            )

            gt._persist_bundle(payload)

            self.assertTrue(np.array_equal(gt._require_grayscale(bundle_dir / "preprocessed_input.png"), final_image))
            self.assertTrue(np.array_equal(gt._require_grayscale(bundle_dir / "mask.png"), final_mask))
            self.assertTrue(np.array_equal(gt._require_grayscale(bundle_dir / "masked_image.png"), masked_image))

    def test_main_segmentation_uses_solov2_by_default(self):
        full_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 8:24] = 255

        old_runtime = gt._GENERATOR_RUNTIME
        gt._GENERATOR_RUNTIME = gt.GeneratorRuntimeConfig(
            execution_target="local",
            gpu_only=False,
            gpu_batch_size=1,
            cpu_workers=1,
            prefetch_samples=1,
            skip_existing=False,
        )
        try:
            with (
                mock.patch.object(gt, "solov2_mask_from_bgr", return_value=(mask, "solov2")) as solov2_mock,
                mock.patch.object(gt, "rembg_mask_from_bgr", side_effect=AssertionError("rembg should not run")),
            ):
                segmented = gt._segment_contactless_bgr_main(full_bgr, Path("sample.jpg"))
        finally:
            gt._GENERATOR_RUNTIME = old_runtime

        solov2_mock.assert_called_once()
        self.assertEqual(segmented.mask_source, "solov2")
        self.assertEqual(int(np.count_nonzero(segmented.initial_mask)), int(np.count_nonzero(mask)))

    def test_main_segmentation_can_explicitly_use_rembg(self):
        full_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 8:24] = 255

        old_runtime = gt._GENERATOR_RUNTIME
        gt._GENERATOR_RUNTIME = gt.GeneratorRuntimeConfig(
            execution_target="local",
            gpu_only=False,
            gpu_batch_size=1,
            cpu_workers=1,
            prefetch_samples=1,
            skip_existing=False,
            mask_extractor="rembg",
        )
        try:
            with (
                mock.patch.object(gt, "rembg_mask_from_bgr", return_value=(mask, "rembg")) as rembg_mock,
                mock.patch.object(gt, "solov2_mask_from_bgr", side_effect=AssertionError("solov2 should not run")),
            ):
                segmented = gt._segment_contactless_bgr_main(full_bgr, Path("sample.jpg"))
        finally:
            gt._GENERATOR_RUNTIME = old_runtime

        rembg_mock.assert_called_once()
        self.assertEqual(segmented.mask_source, "rembg")

    def test_reconstruction_geometry_can_use_presegmented_view(self):
        segmented = gt.SegmentedContactlessInput(
            raw_image_path=str(Path("view_0.jpg").resolve()),
            raw_gray=np.zeros((8, 8), dtype=np.uint8),
            initial_mask=np.ones((8, 8), dtype=np.uint8) * 255,
            mask_source="test",
        )
        preprocessed = gt.PreprocessedContactlessImage(
            raw_gray=np.zeros((8, 8), dtype=np.uint8),
            normalized_gray=np.zeros((8, 8), dtype=np.uint8),
            pose_normalized_gray=np.zeros((8, 8), dtype=np.uint8),
            pose_normalized_mask=np.ones((8, 8), dtype=np.uint8) * 255,
            preprocessed_gray=np.zeros((8, 8), dtype=np.uint8),
            final_mask=np.ones((8, 8), dtype=np.uint8) * 255,
            mask_source="test",
            pose_rotation_degrees=0.0,
            ridge_scale_factor=1.0,
        )

        with (
            mock.patch.object(gt, "_preprocess_contactless_segment_cpu", return_value=preprocessed) as cpu_preprocess,
            mock.patch.object(gt, "_preprocess_contactless_raw", side_effect=AssertionError("raw rembg path should not run")),
        ):
            _, geometry = gt._extract_reconstruction_view_geometry_from_segment("front", segmented)

        cpu_preprocess.assert_called_once()
        self.assertEqual(geometry.image_shape, (8, 8))
        self.assertTrue(np.all(geometry.valid_rows))

    def test_triplet_preprocessing_uses_front_scale_and_shared_canvas(self):
        triplet_paths = {
            "front": Path("front.jpg"),
            "left": Path("left.jpg"),
            "right": Path("right.jpg"),
        }
        segmented = {
            "front": gt.SegmentedContactlessInput(
                raw_image_path=str(triplet_paths["front"].resolve()),
                raw_gray=np.full((20, 12), 20, dtype=np.uint8),
                initial_mask=np.ones((20, 12), dtype=np.uint8) * 255,
                mask_source="front_mask",
            ),
            "left": gt.SegmentedContactlessInput(
                raw_image_path=str(triplet_paths["left"].resolve()),
                raw_gray=np.full((16, 10), 40, dtype=np.uint8),
                initial_mask=np.ones((16, 10), dtype=np.uint8) * 255,
                mask_source="left_mask",
            ),
            "right": gt.SegmentedContactlessInput(
                raw_image_path=str(triplet_paths["right"].resolve()),
                raw_gray=np.full((24, 14), 60, dtype=np.uint8),
                initial_mask=np.ones((24, 14), dtype=np.uint8) * 255,
                mask_source="right_mask",
            ),
        }

        def fake_scale(image, mask, **_kwargs):
            return image.copy(), mask.copy(), 8.0, 1.25

        fake_preprocess = gt.SimpleNamespace(
            _masked_clahe=mock.Mock(side_effect=lambda image, _mask: image.copy()),
            circular_mask=mock.Mock(side_effect=lambda mask: mask.copy()),
            scale_to_paper_ridge_period=mock.Mock(side_effect=fake_scale),
            rotate_to_vertical_centerline=mock.Mock(side_effect=lambda image, mask: (image.copy(), mask.copy(), 0.0)),
        )

        with mock.patch.object(gt, "solov2_preprocess", fake_preprocess):
            preprocessed, geometries, meta = gt._preprocess_reconstruction_triplet_from_segments(
                triplet_paths,
                segmented_views=segmented,
            )

        shapes = {role: item.pose_normalized_gray.shape for role, item in preprocessed.items()}
        self.assertEqual(len(set(shapes.values())), 1)
        height, width = next(iter(shapes.values()))
        self.assertEqual(height % 8, 0)
        self.assertEqual(width % 8, 0)
        self.assertEqual(meta["front_ridge_period"], 8.0)
        self.assertEqual(meta["shared_scale"], 1.25)
        for role in ("front", "left", "right"):
            self.assertEqual(preprocessed[role].ridge_scale_factor, 1.25)
            self.assertEqual(geometries[role].image_shape, (height, width))
            self.assertTrue(np.any(geometries[role].valid_rows))

    def test_reconstructed_bundle_uses_shared_triplet_training_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            shared_image = np.arange(64, dtype=np.uint8).reshape(8, 8)
            shared_mask = np.zeros((8, 8), dtype=np.uint8)
            shared_mask[2:6, 2:6] = 255
            normalized = np.full((4, 4), 33, dtype=np.uint8)
            pose_path = tmp_path / "front_preprocessed_input.png"
            mask_path = tmp_path / "front_pose_mask.png"
            normalized_path = tmp_path / "front_normalized_input.png"
            gt._write_image(pose_path, shared_image)
            gt._write_image(mask_path, shared_mask)
            gt._write_image(normalized_path, normalized)
            gradient_path = tmp_path / "depth_gradient_labels.npz"
            np.savez(
                gradient_path,
                gradient_front=np.zeros((2, 8, 8), dtype=np.float32),
                gradient_left=np.zeros((2, 8, 8), dtype=np.float32),
                gradient_right=np.zeros((2, 8, 8), dtype=np.float32),
            )
            reconstruction = self._make_reconstruction(tmp_path)
            reconstruction.depth_gradient_labels_path = str(gradient_path)
            reconstruction.debug_view_paths = {
                "front": {
                    "preprocessed_input": str(pose_path),
                    "pose_mask": str(mask_path),
                    "normalized_input": str(normalized_path),
                }
            }
            reconstruction.triplet_preprocess = {
                "shared_scale": 1.25,
                "roles": {"front": {"yaw_angle": -3.0, "mask_source": "solov2_triplet_shared_preprocess"}},
            }
            sample = gt.RawViewSample(
                sample_id="s01_f01_a01_v00",
                subject_id=1,
                subject_index=0,
                finger_id=1,
                acquisition_id=1,
                finger_class_id=0,
                raw_image_path=str(tmp_path / "raw.png"),
                raw_view_index=0,
                sire_path=None,
                raw_view_paths=[str(tmp_path / f"raw_{idx}.png") for idx in range(3)],
                variant_paths={},
                is_extra_acquisition=False,
            )

            with (
                mock.patch.object(
                    gt,
                    "_preprocess_contactless_segment_cpu",
                    side_effect=AssertionError("direct preprocessing should not run"),
                ),
                mock.patch.object(gt.pyfing, "orientation_field_estimation", return_value=np.zeros((8, 8), dtype=np.float32)),
                mock.patch.object(gt.pyfing, "frequency_estimation", return_value=np.ones((8, 8), dtype=np.float32)),
                mock.patch.object(gt, "_enhance_for_minutiae", side_effect=lambda image: image.copy()),
            ):
                prepared = gt._prepare_bundle_from_segmented(
                    sample=sample,
                    image_path=tmp_path / "raw.png",
                    raw_gray=np.zeros((4, 4), dtype=np.uint8),
                    initial_mask=np.ones((4, 4), dtype=np.uint8) * 255,
                    mask_source="solov2",
                    visualize=False,
                    bundle_dir=tmp_path / "bundle",
                    reconstruction=reconstruction,
                    dpi=gt.DEFAULT_DPI,
                )

        self.assertTrue(np.array_equal(prepared.gray_image, shared_image))
        self.assertTrue(np.array_equal(prepared.mask, shared_mask))
        self.assertEqual(prepared.preprocessed.ridge_scale_factor, 1.25)
        self.assertEqual(prepared.preprocessed.pose_rotation_degrees, -3.0)

    def test_collect_reconstruction_candidates_keeps_one_sample_per_acquisition(self):
        raw_views = [str(Path(f"1_1_1_{idx}.jpg").resolve()) for idx in range(3)]
        samples = [
            gt.RawViewSample(
                sample_id=f"s01_f01_a01_v0{idx}",
                subject_id=1,
                subject_index=0,
                finger_id=1,
                acquisition_id=1,
                finger_class_id=0,
                raw_image_path=raw_views[idx],
                raw_view_index=idx,
                sire_path=None,
                raw_view_paths=raw_views,
                variant_paths={},
                is_extra_acquisition=False,
            )
            for idx in range(3)
        ]

        candidates = gt._collect_reconstruction_candidates(samples)

        self.assertEqual(list(candidates.keys()), [(1, 1, 1)])
        self.assertEqual(candidates[(1, 1, 1)].raw_view_index, 0)

    def test_quality_skip_marker_removes_load_bearing_bundle_files(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            reconstruction = self._make_reconstruction(tmp_path)
            prepared = self._make_prepared(tmp_path, raw_view_index=1, reconstruction=reconstruction)
            prepared.bundle_dir.mkdir(parents=True)
            for name in gt.QUALITY_SKIP_LOAD_BEARING_FILES:
                (prepared.bundle_dir / name).write_text("stale", encoding="utf-8")
            built_targets = gt.BuiltBundleTargets(
                minutiae=[],
                minutiae_source="pyfing_nbis_fingerflow_consensus_reprojected_left",
                featurenet_targets={},
                rasterized_count=0,
                minutiae_ground_truth_details={
                    "mode": "reconstruction_backed",
                    "canonical_unwrapped_minutiae_count": 0,
                    "reprojected_minutiae_count": 0,
                    "single_source_candidate_count": 85,
                    "consensus_counts": {"pyfing_count": 0, "mindtct_count": 85, "fingerflow_count": 0},
                },
                stage_seconds={},
            )

            record = gt._persist_quality_skip_marker(prepared, built_targets)
            marker = json.loads((prepared.bundle_dir / "quality_skip.json").read_text(encoding="utf-8"))

            for name in gt.QUALITY_SKIP_LOAD_BEARING_FILES:
                self.assertFalse((prepared.bundle_dir / name).exists(), name)
            self.assertEqual(record["sample_id"], "s01_f01_a01_v01")
            self.assertEqual(marker["reason"], "insufficient_reconstruction_minutiae")
            self.assertEqual(marker["view_role"], "left")
            self.assertEqual(marker["minimum_rasterized_minutiae"], 1)
            self.assertEqual(marker["rasterized_minutiae_count"], 0)
            self.assertEqual(marker["single_source_candidate_count"], 85)

    def test_quality_skip_is_reported_without_direct_fallback_or_error(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            prepared = self._make_prepared(
                tmp_path,
                raw_view_index=1,
                reconstruction=self._make_reconstruction(tmp_path),
            )
            built_targets = gt.BuiltBundleTargets(
                minutiae=[],
                minutiae_source="pyfing_nbis_fingerflow_consensus_reprojected_left",
                featurenet_targets={},
                rasterized_count=0,
                minutiae_ground_truth_details={"mode": "reconstruction_backed"},
                stage_seconds={},
            )
            quality_skips: list[dict[str, object]] = []
            errors: list[dict[str, str]] = []

            with mock.patch.object(gt, "_extract_direct_sample_minutiae") as direct_mock:
                gt._record_quality_skip(prepared, built_targets, quality_skips)

            direct_mock.assert_not_called()
            self.assertEqual(errors, [])
            self.assertEqual(len(quality_skips), 1)
            audit = gt._summarize_minutiae_generation_results({}, quality_skips)
            self.assertEqual(audit["quality_skipped_samples"], 1)
            self.assertEqual(audit["by_view_role"]["left"]["quality_skipped_samples"], 1)
            self.assertEqual(audit["by_raw_view_index"]["1"]["quality_skipped_samples"], 1)
            self.assertEqual(audit["by_quality_skip_reason"]["insufficient_reconstruction_minutiae"], 1)

    def test_post_rasterization_fallback_after_zero_reconstruction_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sample = gt.RawViewSample(
                sample_id="s01_f01_a01_v00",
                subject_id=1,
                subject_index=0,
                finger_id=1,
                acquisition_id=1,
                finger_class_id=0,
                raw_image_path=str(tmp_path / "raw.png"),
                raw_view_index=0,
                sire_path=None,
                raw_view_paths=[str(tmp_path / f"raw_{idx}.png") for idx in range(3)],
                variant_paths={},
                is_extra_acquisition=False,
            )
            reconstruction_maps_path = tmp_path / "reconstruction_maps.npz"
            unwarp_maps_path = tmp_path / "center_unwarp_maps.npz"
            np.savez(reconstruction_maps_path, dummy=np.zeros((1, 1), dtype=np.float32))
            np.savez(unwarp_maps_path, dummy=np.zeros((1, 1), dtype=np.float32))
            reconstruction = gt.AcquisitionReconstructionResult(
                acquisition_id="s01_f01_a01",
                reconstruction_dir=str(tmp_path),
                depth_front_path="",
                depth_left_path="",
                depth_right_path="",
                depth_gradient_labels_path="",
                reconstruction_maps_path=str(reconstruction_maps_path),
                support_mask_path="",
                row_measurements_path="",
                meta_path="",
                preview_path="",
                algorithm1_depth_then_algorithm3_unwarp_dir=str(tmp_path / "algorithm1_depth_then_algorithm3_unwarp"),
                center_unwarp_maps_path=str(unwarp_maps_path),
                center_unwarped_image_path="center_unwarped.png",
                center_unwarped_mask_path="center_unwarped_mask.png",
                surface_front_3d_html_path="",
                surface_front_3d_png_path="",
                surface_all_branches_3d_html_path="",
                surface_all_branches_3d_png_path="",
                reprojection_report_path="",
                reprojection_preview_path="",
                valid_row_count=1,
                support_pixel_count=1,
                input_view_paths={},
                debug_view_paths={},
            )
            prepared = gt.PreparedBundleArtifacts(
                sample=sample,
                bundle_dir=tmp_path / "bundle",
                image_path=tmp_path / "raw.png",
                preprocessed=gt.PreprocessedContactlessImage(
                    raw_gray=np.zeros((16, 16), dtype=np.uint8),
                    normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                    pose_normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                    pose_normalized_mask=np.ones((16, 16), dtype=np.uint8),
                    preprocessed_gray=np.zeros((16, 16), dtype=np.uint8),
                    final_mask=np.ones((16, 16), dtype=np.uint8),
                    mask_source="test",
                    pose_rotation_degrees=0.0,
                    ridge_scale_factor=1.0,
                ),
                gray_image=np.zeros((16, 16), dtype=np.uint8),
                mask=np.ones((16, 16), dtype=np.uint8),
                orientation=np.zeros((16, 16), dtype=np.float32),
                ridge_period=np.zeros((16, 16), dtype=np.float32),
                visualization_gradient=np.zeros((16, 16, 2), dtype=np.float32),
                reconstruction_gradient=None,
                masked_image=np.zeros((16, 16), dtype=np.uint8),
                enhanced_image=np.zeros((16, 16), dtype=np.uint8),
                visualize=False,
                reconstruction=reconstruction,
            )
            loaded = gt.LoadedSampleInput(
                sample=sample,
                image_path=tmp_path / "raw.png",
                full_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
                visualize=False,
                bundle_dir=prepared.bundle_dir,
                reconstruction=reconstruction,
            )

            def fake_targets(*, minutiae, **_kwargs):
                valid_count = 1 if minutiae and minutiae[0].get("source") == "direct" else 0
                valid = np.zeros((1, 2, 2), dtype=np.float32)
                valid[0, 0, 0] = float(valid_count)
                return {
                    "orientation": np.zeros((180, 2, 2), dtype=np.float32),
                    "ridge_period": np.zeros((1, 2, 2), dtype=np.float32),
                    "minutia_score": valid.copy(),
                    "minutia_valid_mask": valid,
                    "minutia_x": np.zeros((2, 2), dtype=np.int64),
                    "minutia_y": np.zeros((2, 2), dtype=np.int64),
                    "minutia_x_offset": np.zeros((1, 2, 2), dtype=np.float32),
                    "minutia_y_offset": np.zeros((1, 2, 2), dtype=np.float32),
                    "minutia_orientation": np.zeros((2, 2), dtype=np.int64),
                    "minutia_orientation_vec": np.zeros((2, 2, 2), dtype=np.float32),
                    "output_mask": np.ones((1, 2, 2), dtype=np.float32),
                }

            with (
                mock.patch.object(gt, "_prepare_bundle_from_loaded", return_value=prepared),
                mock.patch.object(
                    gt,
                    "_load_or_extract_canonical_reconstruction_minutiae",
                    return_value=([{"x": 1.0, "y": 1.0, "theta": 0.0}], "canonical_test", {}),
                ),
                mock.patch.object(
                    gt,
                    "_build_reprojected_targets_cpu",
                    return_value=gt.BuiltBundleTargets(
                        minutiae=[{"x": 1.0, "y": 1.0, "theta": 0.0, "source": "reprojected"}],
                        minutiae_source="canonical_test_reprojected_front",
                        featurenet_targets=fake_targets(minutiae=[]),
                        rasterized_count=0,
                        minutiae_ground_truth_details={
                            "mode": "reconstruction_backed",
                            "reprojected_minutiae_count": 1,
                            "rasterized_minutiae_count": 0,
                        },
                        stage_seconds={},
                    ),
                ),
                mock.patch.object(
                    gt,
                    "_extract_direct_sample_minutiae",
                    return_value=([{"x": 1.0, "y": 1.0, "theta": 0.0, "source": "direct"}], "direct_test"),
                ) as direct_mock,
                mock.patch.object(gt, "_build_featurenet_targets", side_effect=fake_targets),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "consensus reconstruction-backed minutiae produced fewer than 1 rasterized minutiae",
                ):
                    gt._generate_bundle_from_loaded(
                        loaded,
                        fingerflow_model_dir=tmp_path,
                        dpi=gt.DEFAULT_DPI,
                        fingerflow_backend=gt.FingerflowBackendConfig("local", "Ubuntu", ""),
                        minutiae_extractor_config=gt.MinutiaeExtractorConfig(),
                    )

            direct_mock.assert_not_called()

    def test_default_minutiae_extractor_args_use_r20_consensus(self):
        with mock.patch.object(sys, "argv", ["generate_ground_truth.py"]):
            args = gt.parse_args()

        self.assertEqual(args.minutiae_extractor, "consensus")
        self.assertEqual(args.minutiae_score_target, "gaussian")
        self.assertEqual(args.mindtct_bin, "mindtct")
        self.assertEqual(args.fingerflow_bin, "fingerflow")
        self.assertEqual(args.consensus_overlap_radius_px, 20.0)
        self.assertEqual(args.min_consensus_sources, 2)
        self.assertEqual(args.mask_extractor, "solov2")
        self.assertEqual(args.solov2_score_thr, 0.3)

    def test_load_consensus_role_minutiae_reads_consensus_json(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            reconstruction = self._make_reconstruction(tmp_path)
            consensus_path = tmp_path / "algorithm1_depth_then_algorithm3_unwarp" / "front" / "consensus_minutiae.json"
            consensus_path.parent.mkdir(parents=True, exist_ok=True)
            consensus_path.write_text(
                json.dumps(
                    {
                        "angle_units": "degrees",
                        "minutiae": [
                            {
                                "x": 7.0,
                                "y": 9.0,
                                "theta": 90.0,
                                "score": 0.9,
                                "type": "E",
                                "matched_nbis_x": 8.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report_path = tmp_path / "algorithm1_depth_then_algorithm3_unwarp" / "consensus_minutiae_r20" / "consensus_minutiae_report.json"
            report = {
                "report_path": str(report_path),
                "output_dir": str(report_path.parent),
                "overlap_radius_px": 20.0,
                "roles": {
                    "front": {
                        "pyfing_count": 4,
                        "nbis_count": 10,
                        "consensus_survivor_count": 1,
                        "pyfing_discarded_count": 3,
                        "nbis_unmatched_count": 9,
                        "artifacts": {"consensus_minutiae_json": str(consensus_path)},
                    }
                },
            }

            gt._RECONSTRUCTION_CONSENSUS_CACHE.clear()
            with mock.patch.object(gt, "_run_or_load_reconstruction_consensus", return_value=report):
                minutiae, source, details = gt._load_consensus_role_minutiae(
                    reconstruction,
                    "front",
                    gt.MinutiaeExtractorConfig(consensus_overlap_radius_px=20.0),
                )

        self.assertEqual(source, "pyfing_nbis_fingerflow_consensus")
        self.assertEqual(len(minutiae), 1)
        self.assertAlmostEqual(minutiae[0]["theta"], np.pi / 2.0)
        self.assertEqual(minutiae[0]["matched_nbis_x"], 8.0)
        self.assertEqual(details["label_minutiae_extractor"], "pyfing_nbis_fingerflow_consensus")
        self.assertEqual(details["consensus_counts"]["consensus_survivor_count"], 1)
        self.assertEqual(details["consensus_overlap_radius_px"], 20.0)

    def test_consensus_canonical_extraction_writes_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            reconstruction = self._make_reconstruction(tmp_path)
            with mock.patch.object(
                gt,
                "_load_consensus_role_minutiae",
                return_value=(
                    [{"x": 1.0, "y": 2.0, "theta": 0.0, "source": "pyfing_nbis_fingerflow_consensus", "matched_nbis_x": 1.5}],
                    "pyfing_nbis_fingerflow_consensus",
                    {"label_minutiae_count": 1},
                ),
            ):
                minutiae, source, details = gt._load_or_extract_canonical_reconstruction_minutiae(
                    reconstruction,
                    tmp_path,
                    gt.FingerflowBackendConfig("local", "Ubuntu", ""),
                    gt.MinutiaeExtractorConfig(),
                )

            written = json.loads((tmp_path / "canonical_unwarped_minutiae.json").read_text(encoding="utf-8"))

        self.assertEqual(source, "pyfing_nbis_fingerflow_consensus")
        self.assertEqual(len(minutiae), 1)
        self.assertEqual(written[0]["matched_nbis_x"], 1.5)
        self.assertEqual(details["canonical_minutiae_count"], 1)

    def test_consensus_reconstruction_failure_propagates_without_direct_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sample = gt.RawViewSample(
                sample_id="s01_f01_a01_v00",
                subject_id=1,
                subject_index=0,
                finger_id=1,
                acquisition_id=1,
                finger_class_id=0,
                raw_image_path=str(tmp_path / "raw.png"),
                raw_view_index=0,
                sire_path=None,
                raw_view_paths=[str(tmp_path / f"raw_{idx}.png") for idx in range(3)],
                variant_paths={},
                is_extra_acquisition=False,
            )
            reconstruction = self._make_reconstruction(tmp_path)
            prepared = gt.PreparedBundleArtifacts(
                sample=sample,
                bundle_dir=tmp_path / "bundle",
                image_path=tmp_path / "raw.png",
                preprocessed=gt.PreprocessedContactlessImage(
                    raw_gray=np.zeros((16, 16), dtype=np.uint8),
                    normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                    pose_normalized_gray=np.zeros((16, 16), dtype=np.uint8),
                    pose_normalized_mask=np.ones((16, 16), dtype=np.uint8),
                    preprocessed_gray=np.zeros((16, 16), dtype=np.uint8),
                    final_mask=np.ones((16, 16), dtype=np.uint8),
                    mask_source="test",
                    pose_rotation_degrees=0.0,
                    ridge_scale_factor=1.0,
                ),
                gray_image=np.zeros((16, 16), dtype=np.uint8),
                mask=np.ones((16, 16), dtype=np.uint8),
                orientation=np.zeros((16, 16), dtype=np.float32),
                ridge_period=np.zeros((16, 16), dtype=np.float32),
                visualization_gradient=np.zeros((16, 16, 2), dtype=np.float32),
                reconstruction_gradient=None,
                masked_image=np.zeros((16, 16), dtype=np.uint8),
                enhanced_image=np.zeros((16, 16), dtype=np.uint8),
                visualize=False,
                reconstruction=reconstruction,
            )
            loaded = gt.LoadedSampleInput(
                sample=sample,
                image_path=tmp_path / "raw.png",
                full_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
                visualize=False,
                bundle_dir=prepared.bundle_dir,
                reconstruction=reconstruction,
            )

            with (
                mock.patch.object(gt, "_prepare_bundle_from_loaded", return_value=prepared),
                mock.patch.object(
                    gt,
                    "_load_or_extract_canonical_reconstruction_minutiae",
                    side_effect=RuntimeError("mindtct missing"),
                ),
                mock.patch.object(gt, "_extract_direct_sample_minutiae") as direct_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "mindtct missing"):
                    gt._generate_bundle_from_loaded(
                        loaded,
                        fingerflow_model_dir=tmp_path,
                        dpi=gt.DEFAULT_DPI,
                        fingerflow_backend=gt.FingerflowBackendConfig("local", "Ubuntu", ""),
                        minutiae_extractor_config=gt.MinutiaeExtractorConfig(),
                    )

            direct_mock.assert_not_called()

    def test_generated_root_symlink_merge_links_payload_and_rewrites_meta(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_root = tmp_path / "DSX"
            output_root = tmp_path / "merged"
            sample = self._make_sample(tmp_path)

            source_sample_dir = source_root / "samples" / sample.sample_id
            source_sample_dir.mkdir(parents=True)
            (source_sample_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "subject_index": sample.subject_index,
                        "finger_class_id": sample.finger_class_id,
                        "multiview_reconstruction": {"acquisition_id": sample.acquisition_id},
                    }
                ),
                encoding="utf-8",
            )
            (source_sample_dir / "featurenet_targets.npz").write_bytes(b"payload")

            source_reconstruction_dir = source_root / "reconstructions" / "s01_f01_a01"
            source_reconstruction_dir.mkdir(parents=True)
            (source_reconstruction_dir / "meta.json").write_text(
                json.dumps({"acquisition_id": "s01_f01_a01", "subject_id": sample.subject_id}),
                encoding="utf-8",
            )
            (source_reconstruction_dir / "depth_front.npy").write_bytes(b"depth")

            gt._write_manifest([sample], source_root)
            gt._write_summary(source_root, {"generated_bundle_count": 1, "errors": []})

            summary = gt._merge_generated_ground_truth_roots(
                [("dsx", source_root)],
                output_root,
                link_mode="symlink",
            )

            merged_sample_dir = output_root / "samples" / "dsx_s01_f01_a01_v00"
            merged_reconstruction_dir = output_root / "reconstructions" / "dsx_s01_f01_a01"
            self.assertEqual(summary["merge_link_mode"], "symlink")
            self.assertFalse((merged_sample_dir / "meta.json").is_symlink())
            self.assertTrue((merged_sample_dir / "featurenet_targets.npz").is_symlink())
            self.assertFalse((merged_reconstruction_dir / "meta.json").is_symlink())
            self.assertTrue((merged_reconstruction_dir / "depth_front.npy").is_symlink())

            merged_meta = json.loads((merged_sample_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(merged_meta["sample_id"], "dsx_s01_f01_a01_v00")
            self.assertEqual(merged_meta["multiview_reconstruction"]["acquisition_id"], "dsx_s01_f01_a01")


if __name__ == "__main__":
    unittest.main()
