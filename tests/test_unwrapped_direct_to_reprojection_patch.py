from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import generate_ground_truth as gt
from scripts import patch_unwrapped_direct_to_reprojection_ground_truth as patcher


class UnwrappedDirectToReprojectionPatchTests(unittest.TestCase):
    def test_build_candidates_selects_only_unwrapped_direct_reconstruction_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reconstructions" / "s01_f01_a01").mkdir(parents=True)
            manifest = []
            for idx, mode in ((0, "reconstruction_unwrapped_direct"), (1, "reconstruction_backed"), (3, "reconstruction_unwrapped_direct")):
                sample_id = f"s01_f01_a01_v{idx:02d}"
                sample_dir = root / "samples" / sample_id
                sample_dir.mkdir(parents=True)
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": 1,
                        "subject_index": 0,
                        "finger_id": 1,
                        "acquisition_id": 1,
                        "finger_class_id": 0,
                        "raw_image_path": "raw.png",
                        "raw_view_index": idx,
                        "sire_path": None,
                        "raw_view_paths": ["0.png", "1.png", "2.png"],
                        "variant_paths": {},
                        "is_extra_acquisition": False,
                    }
                )
                (sample_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "minutiae_ground_truth": {"mode": mode},
                            "multiview_reconstruction": {"acquisition_id": "s01_f01_a01"},
                        }
                    ),
                    encoding="utf-8",
                )
                for name in (
                    "minutiae.json",
                    "featurenet_targets.npz",
                    "preprocessed_input.png",
                    "mask.png",
                    "preprocess_pose_normalized.png",
                    "preprocess_pose_mask.png",
                ):
                    (sample_dir / name).write_bytes(b"x")

            candidates, skipped = patcher._build_candidates(root, manifest, limit=None)

            self.assertEqual([candidate.sample_id for candidate in candidates], ["s01_f01_a01_v00"])
            self.assertEqual(len(skipped), 2)

    def test_side_v4_reprojection_maps_chart_to_training_frame(self):
        sample = gt.RawViewSample(
            sample_id="s01_f01_a01_v01",
            subject_id=1,
            subject_index=0,
            finger_id=1,
            acquisition_id=1,
            finger_class_id=0,
            raw_image_path="raw.png",
            raw_view_index=1,
            sire_path=None,
            raw_view_paths=["0.png", "1.png", "2.png"],
            variant_paths={},
            is_extra_acquisition=False,
        )
        preprocessed = gt.PreprocessedContactlessImage(
            raw_gray=np.zeros((4, 4), dtype=np.uint8),
            normalized_gray=np.zeros((4, 4), dtype=np.uint8),
            pose_normalized_gray=np.zeros((13, 13), dtype=np.uint8),
            pose_normalized_mask=np.ones((13, 13), dtype=np.uint8) * 255,
            preprocessed_gray=np.zeros((26, 39), dtype=np.uint8),
            final_mask=np.ones((26, 39), dtype=np.uint8) * 255,
            mask_source="test",
            pose_rotation_degrees=0.0,
            ridge_scale_factor=1.0,
        )
        source_x = np.tile(np.arange(13, dtype=np.float32), (13, 1))
        source_y = np.tile(np.arange(13, dtype=np.float32)[:, None], (1, 13))
        side_maps = {
            "source_x_map": source_x,
            "source_y_map": source_y,
            "unwrapped_mask": np.ones((13, 13), dtype=np.uint8),
        }
        minutiae = [{"x": 6.0, "y": 6.0, "theta": 0.0, "score": 1.0}]

        remapped, details = gt._remap_side_v4_unwrapped_minutiae_to_sample(
            minutiae,
            side_maps,
            np.ones((13, 13), dtype=bool),
            sample,
            preprocessed,
        )

        self.assertEqual(len(remapped), 1)
        self.assertAlmostEqual(remapped[0]["x"], 18.0)
        self.assertAlmostEqual(remapped[0]["y"], 12.0)
        self.assertEqual(details["orientation_projected_count"], 1)

    def test_reconstruction_gradient_cache_patches_side_roles_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recon = root / "reconstructions" / "s01_f01_a01"
            (recon / "side_depth_unwrap_v4" / "left").mkdir(parents=True)
            (recon / "side_depth_unwrap_v4" / "right").mkdir(parents=True)
            front = np.ones((2, 2, 2), dtype=np.float32)
            old_side = np.zeros((2, 2, 2), dtype=np.float32)
            np.savez_compressed(
                recon / "depth_gradient_labels.npz",
                gradient_front=front,
                gradient_left=old_side,
                gradient_right=old_side,
            )
            np.save(recon / "side_depth_unwrap_v4" / "left" / "left_gradient.npy", np.ones((2, 2, 2), dtype=np.float32) * 3)
            np.save(recon / "side_depth_unwrap_v4" / "right" / "right_gradient.npy", np.ones((2, 2, 2), dtype=np.float32) * 4)

            report = patcher._patch_reconstruction_gradient_cache(recon, dry_run=False)

            with np.load(recon / "depth_gradient_labels.npz") as data:
                self.assertTrue(np.array_equal(data["gradient_front"], front))
                self.assertEqual(float(data["gradient_left"][0, 0, 0]), 3.0)
                self.assertEqual(float(data["gradient_right"][0, 0, 0]), 4.0)
            self.assertEqual(report["status"], "patched")


if __name__ == "__main__":
    unittest.main()
