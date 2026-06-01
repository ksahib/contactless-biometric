from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import sysconfig
import tempfile
import unittest

import numpy as np


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

import generate_ground_truth as gt


class CrossingNumberPlausibleRegionTests(unittest.TestCase):
    def test_straight_segment_has_two_endpoints(self):
        gray = np.full((9, 9), 255, dtype=np.uint8)
        gray[4, 2:7] = 0
        mask = np.full((9, 9), 255, dtype=np.uint8)

        plausible_mask, details = gt._build_crossing_number_plausible_mask(
            gray,
            mask,
            polarity="dark",
            dilation_radius=0,
        )

        self.assertEqual(details["cn_endpoint_pixels"], 2)
        self.assertEqual(details["cn_bifurcation_pixels"], 0)
        self.assertEqual(int(np.count_nonzero(plausible_mask)), 2)

    def test_t_junction_has_bifurcation(self):
        gray = np.full((9, 9), 255, dtype=np.uint8)
        gray[4, 2:7] = 0
        gray[2:7, 4] = 0
        mask = np.full((9, 9), 255, dtype=np.uint8)

        plausible_mask, details = gt._build_crossing_number_plausible_mask(
            gray,
            mask,
            polarity="dark",
            dilation_radius=0,
        )

        self.assertGreaterEqual(details["cn_bifurcation_pixels"], 1)
        self.assertGreaterEqual(int(np.count_nonzero(plausible_mask)), 1)

    def test_flat_image_yields_no_candidates(self):
        gray = np.full((9, 9), 255, dtype=np.uint8)
        mask = np.full((9, 9), 255, dtype=np.uint8)

        plausible_mask, details = gt._build_crossing_number_plausible_mask(
            gray,
            mask,
            polarity="dark",
            dilation_radius=0,
        )

        self.assertEqual(details["candidate_pixels"], 0)
        self.assertEqual(int(np.count_nonzero(plausible_mask)), 0)

    def test_reprojected_plausible_mask_lands_in_expected_output_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            reconstruction_dir = tmp / "recon"
            reconstruction_dir.mkdir(parents=True, exist_ok=True)

            unwarped_gray = np.full((16, 16), 255, dtype=np.uint8)
            unwarped_gray[8, 2:14] = 0
            unwarped_mask = np.full((16, 16), 255, dtype=np.uint8)
            cv2.imwrite(str(reconstruction_dir / "center_unwarped.png"), unwarped_gray)
            cv2.imwrite(str(reconstruction_dir / "center_unwarped_mask.png"), unwarped_mask)

            source_x_map = np.tile(np.arange(16, dtype=np.float32), (16, 1))
            source_y_map = np.tile(np.arange(16, dtype=np.float32)[:, None], (1, 16))
            np.savez(
                reconstruction_dir / "center_unwarp_maps.npz",
                source_x_map=source_x_map,
                source_y_map=source_y_map,
                source_valid_mask=np.ones((16, 16), dtype=np.uint8),
                unwarped_mask=np.ones((16, 16), dtype=np.uint8),
            )

            reconstruction_maps = {
                "support_mask": np.ones((16, 16), dtype=np.uint8),
                "front_pose_x_map": source_x_map,
                "front_pose_y_map": source_y_map,
            }
            np.savez(reconstruction_dir / "reconstruction_maps.npz", **reconstruction_maps)

            sample = gt.RawViewSample(
                sample_id="s01_f01_a01_v00",
                subject_id=1,
                subject_index=0,
                finger_id=1,
                acquisition_id=1,
                finger_class_id=0,
                raw_image_path=str(tmp / "raw.png"),
                raw_view_index=0,
                sire_path=None,
                raw_view_paths=[str(tmp / "raw_0.png"), str(tmp / "raw_1.png"), str(tmp / "raw_2.png")],
                variant_paths={},
                is_extra_acquisition=False,
            )
            preprocessed = gt.PreprocessedContactlessImage(
                raw_gray=unwarped_gray,
                normalized_gray=unwarped_gray,
                pose_normalized_gray=unwarped_gray,
                pose_normalized_mask=unwarped_mask,
                preprocessed_gray=unwarped_gray,
                final_mask=unwarped_mask,
                mask_source="test",
                pose_rotation_degrees=0.0,
                ridge_scale_factor=1.0,
            )
            reconstruction = gt.AcquisitionReconstructionResult(
                acquisition_id="s01_f01_a01",
                reconstruction_dir=str(reconstruction_dir),
                depth_front_path="",
                depth_left_path="",
                depth_right_path="",
                depth_gradient_labels_path="",
                reconstruction_maps_path=str(reconstruction_dir / "reconstruction_maps.npz"),
                support_mask_path="",
                row_measurements_path="",
                meta_path="",
                preview_path="",
                center_unwarp_maps_path=str(reconstruction_dir / "center_unwarp_maps.npz"),
                center_unwarped_image_path=str(reconstruction_dir / "center_unwarped.png"),
                center_unwarped_mask_path=str(reconstruction_dir / "center_unwarped_mask.png"),
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
                bundle_dir=tmp,
                image_path=Path(sample.raw_image_path),
                preprocessed=preprocessed,
                gray_image=unwarped_gray,
                mask=unwarped_mask,
                orientation=np.zeros((16, 16), dtype=np.float32),
                ridge_period=np.zeros((16, 16), dtype=np.float32),
                visualization_gradient=np.zeros((16, 16, 2), dtype=np.float32),
                reconstruction_gradient=None,
                masked_image=unwarped_gray,
                enhanced_image=unwarped_gray,
                visualize=False,
                reconstruction=reconstruction,
            )

            aligned_mask, details = gt._build_crossing_number_plausible_reprojected_mask(
                prepared,
                polarity="dark",
                dilation_radius=0,
            )

            self.assertIsNotNone(aligned_mask)
            self.assertEqual(aligned_mask.shape, (1, 2, 2))
            self.assertGreaterEqual(int(aligned_mask[0, 1, 0]), 1)
            self.assertGreaterEqual(int(aligned_mask[0, 1, 1]), 1)
            self.assertEqual(details["reprojected_plausible_pixels"], 2)
