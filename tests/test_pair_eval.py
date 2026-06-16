from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from featurenet.models import pair_eval


class AucEerTest(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        genuine = [0.8, 0.9, 0.95, 1.0]
        impostor = [0.0, 0.1, 0.2, 0.3]
        self.assertAlmostEqual(pair_eval.compute_auc(genuine, impostor), 1.0, places=6)
        self.assertAlmostEqual(pair_eval.compute_eer(genuine, impostor), 0.0, places=6)

    def test_ties_score_half(self) -> None:
        genuine = [0.5, 0.5]
        impostor = [0.5, 0.5]
        self.assertAlmostEqual(pair_eval.compute_auc(genuine, impostor), 0.5, places=6)

    def test_partial_overlap_auc(self) -> None:
        genuine = [0.4, 0.6]
        impostor = [0.5, 0.7]
        # pairs: (0.4 vs .5,.7)->0 wins; (0.6 vs .5)->1, (0.6 vs .7)->0 => 1/4
        self.assertAlmostEqual(pair_eval.compute_auc(genuine, impostor), 0.25, places=6)

    def test_empty_inputs_are_nan(self) -> None:
        self.assertTrue(math.isnan(pair_eval.compute_auc([], [1.0])))
        self.assertTrue(math.isnan(pair_eval.compute_eer([1.0], [])))


class RepeatabilityTest(unittest.TestCase):
    def test_identical_sets_are_fully_repeatable(self) -> None:
        rows = [
            {"x": 10.0, "y": 10.0, "angle": 0.0},
            {"x": 30.0, "y": 50.0, "angle": 1.0},
            {"x": 70.0, "y": 20.0, "angle": -0.5},
        ]
        rate = pair_eval._pair_redetection_rate(rows, rows, dist_px=5.0, angle_rad=math.radians(15.0))
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_translation_invariance(self) -> None:
        rows_a = [
            {"x": 10.0, "y": 10.0, "angle": 0.0},
            {"x": 30.0, "y": 50.0, "angle": 1.0},
        ]
        rows_b = [{"x": r["x"] + 100.0, "y": r["y"] - 40.0, "angle": r["angle"]} for r in rows_a]
        rate = pair_eval._pair_redetection_rate(rows_a, rows_b, dist_px=3.0, angle_rad=math.radians(10.0))
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_partial_perturbation_reduces_rate(self) -> None:
        rows_a = [
            {"x": 10.0, "y": 10.0, "angle": 0.0},
            {"x": 30.0, "y": 50.0, "angle": 1.0},
            {"x": 70.0, "y": 20.0, "angle": -0.5},
            {"x": 90.0, "y": 90.0, "angle": 2.0},
        ]
        rows_b = [dict(r) for r in rows_a]
        # Centroid-preserving perturbation: shift two points in opposite
        # directions so the centroid alignment stays fixed and exactly two of
        # the four points fall outside the spatial tolerance.
        rows_b[0]["x"] += 40.0
        rows_b[0]["y"] += 40.0
        rows_b[1]["x"] -= 40.0
        rows_b[1]["y"] -= 40.0
        rate = pair_eval._pair_redetection_rate(rows_a, rows_b, dist_px=5.0, angle_rad=math.radians(15.0))
        self.assertLess(rate, 1.0)
        self.assertGreater(rate, 0.0)
        self.assertAlmostEqual(rate, 0.5, places=6)


class BuildPairsTest(unittest.TestCase):
    def _samples(self) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for finger in range(4):
            for view in (0, 1, 2):
                samples.append(
                    {
                        "sample_id": f"f{finger}_v{view}",
                        "finger_class_id": finger,
                        "raw_view_index": view,
                        "masked_image": f"/tmp/f{finger}_v{view}_img.png",
                        "mask": f"/tmp/f{finger}_v{view}_mask.png",
                    }
                )
        return samples

    def test_genuine_same_identity_impostor_different(self) -> None:
        samples = self._samples()
        genuine, impostor = pair_eval.build_pairs(
            samples, max_genuine=50, max_impostor=50, seed=7
        )
        self.assertGreater(len(genuine), 0)
        self.assertGreater(len(impostor), 0)
        for a, b in genuine:
            self.assertEqual(samples[a]["finger_class_id"], samples[b]["finger_class_id"])
        for a, b in impostor:
            self.assertNotEqual(samples[a]["finger_class_id"], samples[b]["finger_class_id"])

    def test_caps_are_respected(self) -> None:
        samples = self._samples()
        genuine, impostor = pair_eval.build_pairs(
            samples, max_genuine=2, max_impostor=3, seed=1
        )
        self.assertLessEqual(len(genuine), 2)
        self.assertLessEqual(len(impostor), 3)

    def test_no_genuine_pairs_returns_empty(self) -> None:
        # All distinct identities -> no genuine pairs possible.
        samples = [
            {"sample_id": f"s{i}", "finger_class_id": i, "raw_view_index": 0}
            for i in range(5)
        ]
        genuine, _ = pair_eval.build_pairs(samples, max_genuine=10, max_impostor=10, seed=3)
        self.assertEqual(genuine, [])


class EvaluatePairsGuardTest(unittest.TestCase):
    def test_returns_none_when_no_genuine_pairs(self) -> None:
        samples = [
            {"sample_id": f"s{i}", "finger_class_id": i, "raw_view_index": 0}
            for i in range(3)
        ]
        result = pair_eval.evaluate_pairs(
            model=None,
            samples=samples,
            device=torch.device("cpu"),
            max_genuine=5,
            max_impostor=5,
        )
        self.assertIsNone(result)


class SelectionWiringTest(unittest.TestCase):
    def test_select_monitored_metric_pair_auc(self) -> None:
        from featurenet.models.train import _select_monitored_metric

        value, mode = _select_monitored_metric(
            "pair_auc",
            {"total": 1.0},
            None,
            {"pair_auc": 0.83},
        )
        self.assertAlmostEqual(value, 0.83, places=6)
        self.assertEqual(mode, "max")

    def test_pair_auc_in_early_stopping_metrics(self) -> None:
        from featurenet.models.train import EARLY_STOPPING_METRICS

        self.assertIn("pair_auc", EARLY_STOPPING_METRICS)


if __name__ == "__main__":
    unittest.main()
