from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_binary_roc_curve.py"
SPEC = importlib.util.spec_from_file_location("plot_binary_roc_curve", MODULE_PATH)
plotter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotter
assert SPEC.loader is not None
SPEC.loader.exec_module(plotter)


def _write_roc_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_score_threshold", "matching_threshold", "TPR", "FPR"],
        )
        writer.writeheader()
        for feature_threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            writer.writerow(
                {
                    "feature_score_threshold": f"{feature_threshold:.2f}",
                    "matching_threshold": "0.10",
                    "TPR": feature_threshold,
                    "FPR": "0.00",
                }
            )
            writer.writerow(
                {
                    "feature_score_threshold": f"{feature_threshold:.2f}",
                    "matching_threshold": "0.20",
                    "TPR": "1.00",
                    "FPR": "1.00",
                }
            )


def _write_small_roc_csv(path: Path, feature_thresholds: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_score_threshold", "matching_threshold", "TPR", "FPR"],
        )
        writer.writeheader()
        for feature_threshold in feature_thresholds:
            writer.writerow(
                {
                    "feature_score_threshold": f"{feature_threshold:.2f}",
                    "matching_threshold": "0.10",
                    "TPR": "0.50",
                    "FPR": "0.00",
                }
            )
            writer.writerow(
                {
                    "feature_score_threshold": f"{feature_threshold:.2f}",
                    "matching_threshold": "0.20",
                    "TPR": "1.00",
                    "FPR": "1.00",
                }
            )


def _write_combined_method_roc_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "feature_score_threshold", "matching_threshold", "TPR", "FPR"],
        )
        writer.writeheader()
        for method in ("LSA", "LSA-R", "LSA-CENTROID"):
            for feature_threshold in (0.8, 0.9):
                writer.writerow(
                    {
                        "method": method,
                        "feature_score_threshold": f"{feature_threshold:.2f}",
                        "matching_threshold": "0.10",
                        "TPR": "0.50",
                        "FPR": "0.00",
                    }
                )
                writer.writerow(
                    {
                        "method": method,
                        "feature_score_threshold": f"{feature_threshold:.2f}",
                        "matching_threshold": "0.20",
                        "TPR": "1.00",
                        "FPR": "1.00",
                    }
                )


class PlotBinaryRocCurveTest(unittest.TestCase):
    def test_plot_roc_curves_uses_all_feature_score_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            roc_csv = tmp_path / "roc_counts.csv"
            output_png = tmp_path / "roc_curve.png"
            output_svg = tmp_path / "roc_curve.svg"
            _write_roc_csv(roc_csv)

            auc_by_threshold = plotter.plot_roc_curves(roc_csv, output_png, output_svg)

            self.assertEqual(list(auc_by_threshold), ["0.50", "0.60", "0.70", "0.80", "0.90"])
            self.assertTrue(output_png.exists())
            self.assertTrue(output_svg.exists())

    def test_plot_roc_curve_sets_overlays_labeled_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            lsa_r_csv = tmp_path / "lsa_r" / "roc_counts.csv"
            centroid_csv = tmp_path / "centroid" / "roc_counts.csv"
            output_png = tmp_path / "roc_comparison.png"
            _write_small_roc_csv(lsa_r_csv, (0.8, 0.9))
            _write_small_roc_csv(centroid_csv, (0.8, 0.9))

            auc_by_source = plotter.plot_roc_curve_sets(
                [("MCC LSA-R", lsa_r_csv), ("MCC Centroid", centroid_csv)],
                output_png,
            )

            self.assertEqual(set(auc_by_source), {"MCC LSA-R", "MCC Centroid"})
            self.assertEqual(list(auc_by_source["MCC LSA-R"]), ["0.80", "0.90"])
            self.assertEqual(list(auc_by_source["MCC Centroid"]), ["0.80", "0.90"])
            self.assertTrue(output_png.exists())

    def test_plot_roc_curve_sets_splits_combined_method_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            roc_csv = tmp_path / "roc_counts.csv"
            output_png = tmp_path / "roc_combined.png"
            _write_combined_method_roc_csv(roc_csv)

            auc_by_source = plotter.plot_roc_curve_sets([(None, roc_csv)], output_png)

            self.assertEqual(set(auc_by_source), {"LSA", "LSA-R", "LSA-CENTROID"})
            self.assertEqual(list(auc_by_source["LSA"]), ["0.80", "0.90"])
            self.assertEqual(list(auc_by_source["LSA-R"]), ["0.80", "0.90"])
            self.assertEqual(list(auc_by_source["LSA-CENTROID"]), ["0.80", "0.90"])
            self.assertTrue(output_png.exists())


if __name__ == "__main__":
    unittest.main()
