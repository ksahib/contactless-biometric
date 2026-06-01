from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

from featurenet.models import plausible_minutiae as pm
from featurenet.models.train import load_bundle_samples


BUILD_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_plausible_minutiae_ground_truth.py"
BUILD_SPEC = importlib.util.spec_from_file_location("build_plausible_minutiae_ground_truth", BUILD_SCRIPT_PATH)
build = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = build
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build)


class PlausibleMinutiaeSidecarTests(unittest.TestCase):
    def test_rle_round_trip(self) -> None:
        mask = np.zeros((1, 4, 5), dtype=np.float32)
        mask[0, 0, 1] = 1.0
        mask[0, 2, 3:5] = 1.0

        rle = pm.encode_binary_mask_rle(mask)
        decoded = pm.decode_binary_mask_rle(rle, (4, 5))

        self.assertEqual(decoded.shape, (1, 4, 5))
        np.testing.assert_array_equal(decoded, mask)

    def test_loader_attaches_plausible_minutia_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_root = root / "samples"
            sample_dir = samples_root / "sample_001"
            sample_dir.mkdir(parents=True, exist_ok=True)

            gray = np.full((4, 4), 255, dtype=np.uint8)
            gray[1:3, 1:3] = 0
            cv2.imwrite(str(sample_dir / "masked_image.png"), gray)
            cv2.imwrite(str(sample_dir / "mask.png"), np.full((4, 4), 255, dtype=np.uint8))
            cv2.imwrite(str(sample_dir / "raw_input.png"), gray)
            np.savez(
                sample_dir / "featurenet_targets.npz",
                mask=np.ones((1, 2, 2), dtype=np.float32),
                minutia_score=np.zeros((1, 2, 2), dtype=np.float32),
                minutia_valid_mask=np.ones((1, 2, 2), dtype=np.float32),
                gradient=np.zeros((2, 2, 2), dtype=np.float32),
            )
            meta = {
                "sample_id": "sample_001",
                "subject_id": 1,
                "subject_index": 0,
                "finger_id": 1,
                "acquisition_id": 1,
                "finger_class_id": 0,
                "raw_image_path": str(sample_dir / "raw_input.png"),
                "raw_view_index": 0,
                "sire_path": None,
                "raw_view_paths": [str(sample_dir / "raw_input.png")],
                "variant_paths": {},
                "is_extra_acquisition": False,
                "counts": {"minutiae": 0},
                "shapes": {"input_image": [4, 4], "featurenet_output": [2, 2]},
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            sidecar_row = {
                "sample_id": "sample_001",
                "view_role": "front",
                "source_frame": "center_unwrapped",
                "source_image_path": "reconstructions/sample_001/center_unwarped.png",
                "source_mask_path": "reconstructions/sample_001/center_unwarped_mask.png",
                "output_shape": [2, 2],
                "mask_rle": [[1, 1]],
                "plausible_pixel_count": 1,
            }
            (root / "plausible_minutiae.jsonl").write_text(json.dumps(sidecar_row) + "\n", encoding="utf-8")

            samples = load_bundle_samples(root, strict_gradient_targets=True)

            self.assertEqual(len(samples), 1)
            plausible_mask = samples[0]["targets"]["plausible_minutia_mask"]
            self.assertEqual(tuple(plausible_mask.shape), (1, 2, 2))
            self.assertEqual(float(plausible_mask[0, 0, 1].item()), 1.0)
            self.assertEqual(int(plausible_mask.sum().item()), 1)

    def test_parallel_chunking_and_merge_keep_every_sample_once(self) -> None:
        entries = [{"sample_id": f"sample_{index:03d}"} for index in range(7)]

        with patch.object(build.os, "cpu_count", return_value=8):
            self.assertEqual(build._resolve_worker_count(None, len(entries)), len(entries))

        chunks = build._chunk_entries(entries, 3)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(entries))
        self.assertEqual([entry["sample_id"] for chunk in chunks for entry in chunk], [entry["sample_id"] for entry in entries])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths: list[Path] = []

            def fake_process_sample(sample_dir: Path, root: Path, **_: object) -> dict[str, object]:
                return {"sample_id": sample_dir.name, "view_role": "front", "source_frame": "center_unwrapped", "output_shape": [2, 2], "mask_rle": [], "plausible_pixel_count": 0}

            with patch.object(build, "_process_sample", side_effect=fake_process_sample):
                for index, chunk in enumerate(chunks):
                    shard_path = root / f"shard_{index:03d}.jsonl"
                    build._write_shard(
                        shard_path,
                        chunk,
                        root,
                        polarity="dark",
                        border_margin=10,
                        min_branch_length=1,
                        source_suppression_radius=8,
                        projection_suppression_radius=1,
                    )
                    shard_paths.append(shard_path)

            merged = root / "merged.jsonl"
            build._merge_shards(shard_paths, merged)
            merged_ids = [json.loads(line)["sample_id"] for line in merged.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(merged_ids, [entry["sample_id"] for entry in entries])


if __name__ == "__main__":
    unittest.main()
