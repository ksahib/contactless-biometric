from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pose_normalization as pose


class PoseNormalizationTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected, places: int = 6):
        self.assertAlmostEqual(actual[0], expected[0], places=places)
        self.assertAlmostEqual(actual[1], expected[1], places=places)

    def test_centroid_calculation_example(self):
        minutiae = [
            pose.Minutia(260.0, 270.0),
            pose.Minutia(280.0, 250.0),
            pose.Minutia(230.0, 290.0),
        ]
        centroid = pose.compute_minutiae_centroid(minutiae)
        self.assertPointAlmostEqual(centroid, (256.6666667, 270.0))

        translated, used_centroid = pose.translate_minutiae_to_centroid(minutiae)
        self.assertPointAlmostEqual(used_centroid, centroid)
        expected = [(3.3333333, 0.0), (23.3333333, -20.0), (-26.6666667, 20.0)]
        for actual, expected_point in zip(translated, expected):
            self.assertPointAlmostEqual((actual.x, actual.y), expected_point)

    def test_pure_translation_invariance(self):
        query = [
            pose.Minutia(10.0, 10.0),
            pose.Minutia(20.0, 10.0),
            pose.Minutia(10.0, 20.0),
        ]
        shifted = [
            pose.Minutia(110.0, 60.0),
            pose.Minutia(120.0, 60.0),
            pose.Minutia(110.0, 70.0),
        ]
        query_centered, _ = pose.translate_minutiae_to_centroid(query)
        shifted_centered, _ = pose.translate_minutiae_to_centroid(shifted)
        for left, right in zip(query_centered, shifted_centered):
            self.assertPointAlmostEqual((left.x, left.y), (right.x, right.y))

    def test_metadata_preservation_and_no_input_mutation(self):
        original = [
            pose.Minutia(
                10.0,
                20.0,
                angle=0.25,
                quality=0.9,
                type="B",
                source="unit",
                extra={"id": "m1"},
            )
        ]
        translated, _ = pose.translate_minutiae_to_centroid(original)
        self.assertEqual((original[0].x, original[0].y), (10.0, 20.0))
        self.assertEqual((translated[0].x, translated[0].y), (0.0, 0.0))
        self.assertEqual(translated[0].angle, 0.25)
        self.assertEqual(translated[0].quality, 0.9)
        self.assertEqual(translated[0].type, "B")
        self.assertEqual(translated[0].source, "unit")
        self.assertEqual(translated[0].extra, {"id": "m1"})

    def test_quality_weighted_centroid_and_invalid_weight_fallback(self):
        minutiae = [
            pose.Minutia(0.0, 0.0, quality=1.0),
            pose.Minutia(10.0, 0.0, quality=3.0),
        ]
        self.assertPointAlmostEqual(
            pose.compute_minutiae_centroid(minutiae, use_quality_weights=True),
            (7.5, 0.0),
        )

        invalid_weights = [
            pose.Minutia(0.0, 0.0, quality=-1.0),
            pose.Minutia(10.0, 0.0, quality=None),
        ]
        self.assertPointAlmostEqual(
            pose.compute_minutiae_centroid(invalid_weights, use_quality_weights=True),
            (5.0, 0.0),
        )

    def test_rotation_and_scale_helpers(self):
        self.assertPointAlmostEqual(pose.rotate_point(1.0, 0.0, math.pi / 2.0), (0.0, 1.0))

        transform = pose.SimilarityTransform(
            query_centroid=(0.0, 0.0),
            template_centroid=(0.0, 0.0),
            rotation=0.0,
            scale=0.5,
        )
        scaled = pose.apply_similarity_transform_to_minutiae([pose.Minutia(10.0, 20.0)], transform)
        self.assertPointAlmostEqual((scaled[0].x, scaled[0].y), (5.0, 10.0))

    def test_doubled_angle_global_orientation_average(self):
        orientation = np.array([[0.1, 0.1 + math.pi]], dtype=np.float32)
        estimated = pose.estimate_global_orientation_from_field(orientation)
        self.assertAlmostEqual(estimated, 0.1, places=5)

    def test_ridge_period_scale_estimation_and_invalid_scale(self):
        query_spacing = pose.estimate_median_ridge_spacing(np.array([0.0, 8.0, 10.0, 40.0]))
        template_spacing = pose.estimate_median_ridge_spacing(np.array([0.0, 4.0, 6.0, 99.0]))
        self.assertAlmostEqual(query_spacing, 9.0)
        self.assertAlmostEqual(template_spacing, 5.0)
        self.assertAlmostEqual(
            pose.estimate_scale_from_ridge_spacing(query_spacing, template_spacing),
            5.0 / 9.0,
        )
        with self.assertRaises(ValueError):
            pose.estimate_scale_from_ridge_spacing(1.0, 10.0)

    def test_synthetic_full_transform_alignment(self):
        query = [
            pose.Minutia(10.0, 20.0, angle=0.1),
            pose.Minutia(20.0, 20.0, angle=0.2),
            pose.Minutia(10.0, 30.0, angle=0.3),
        ]
        query_centroid = pose.compute_minutiae_centroid(query)
        template_centroid = (100.0, 80.0)
        rotation = math.pi / 6.0
        scale = 1.25

        template = []
        for minutia in query:
            centered_x = minutia.x - query_centroid[0]
            centered_y = minutia.y - query_centroid[1]
            rx, ry = pose.rotate_point(centered_x, centered_y, rotation)
            template.append(
                pose.Minutia(
                    template_centroid[0] + (scale * rx),
                    template_centroid[1] + (scale * ry),
                    angle=pose.wrap_angle_pi(float(minutia.angle) + rotation),
                )
            )
        template_centered, _ = pose.translate_minutiae_to_centroid(template, template_centroid)
        transform = pose.SimilarityTransform(
            query_centroid=query_centroid,
            template_centroid=template_centroid,
            rotation=rotation,
            scale=scale,
        )
        transformed = pose.apply_similarity_transform_to_minutiae(query, transform)
        for actual, expected in zip(transformed, template_centered):
            self.assertPointAlmostEqual((actual.x, actual.y), (expected.x, expected.y))
            self.assertAlmostEqual(actual.angle, expected.angle)


if __name__ == "__main__":
    unittest.main()
