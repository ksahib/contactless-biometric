from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import generate_ground_truth as gt


class GroundTruthMinutiaeReprojectionTests(unittest.TestCase):
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
                    "_remap_unwarped_minutiae_to_sample",
                    return_value=(
                        [{"x": 1.0, "y": 1.0, "theta": 0.0, "source": "reprojected"}],
                        {"view_role": "front", "scale_x": 1.0, "scale_y": 1.0, "reprojected_minutiae_count": 1},
                    ),
                ),
                mock.patch.object(
                    gt,
                    "_extract_direct_sample_minutiae",
                    return_value=([{"x": 1.0, "y": 1.0, "theta": 0.0, "source": "direct"}], "direct_test"),
                ),
                mock.patch.object(gt, "_build_featurenet_targets", side_effect=fake_targets),
            ):
                result, payload = gt._generate_bundle_from_loaded(
                    loaded,
                    fingerflow_model_dir=tmp_path,
                    dpi=gt.DEFAULT_DPI,
                    fingerflow_backend=gt.FingerflowBackendConfig("local", "Ubuntu", ""),
                )

            self.assertTrue(result["used_direct_fallback"])
            self.assertEqual(result["rasterized_minutiae_count"], 1)
            self.assertEqual(
                payload.meta["minutiae_ground_truth"]["fallback_reason"],
                "zero_rasterized_minutiae_after_reprojection",
            )


if __name__ == "__main__":
    unittest.main()
