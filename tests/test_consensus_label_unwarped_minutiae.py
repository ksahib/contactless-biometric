from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "consensus_label_unwarped_minutiae.py"
SPEC = importlib.util.spec_from_file_location("consensus_label_unwarped_minutiae", MODULE_PATH)
consensus_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = consensus_script
SPEC.loader.exec_module(consensus_script)


def _record(source: str, x: float, y: float, theta: float = 0.0, score: float | None = 0.8, index: int = 0):
    return {
        "x": x,
        "y": y,
        "theta": theta,
        "score": score,
        "type": "ending",
        "source": source,
        "source_index": index,
        "raw": {},
    }


class ConsensusLabelUnwarpedMinutiaeTests(unittest.TestCase):
    def test_parse_nbis_xyt_normalizes_degrees_and_quality(self):
        rows, skipped = consensus_script._parse_nbis_xyt_lines(
            [
                "",
                "# comment",
                "10 20 450 75",
                "bad 1 2 3",
                "1 2",
                "30 40 -180",
            ]
        )

        self.assertEqual(skipped, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "mindtct")
        self.assertEqual(rows[0]["theta"], 90.0)
        self.assertAlmostEqual(rows[0]["score"], 0.75)
        self.assertEqual(rows[1]["theta"], 180.0)
        self.assertIsNone(rows[1]["score"])

    def test_two_source_pyfing_mindtct_consensus_uses_pyfing_representative(self):
        clustered = consensus_script.cluster_consensus_minutiae(
            {
                "pyfing": [_record("pyfing", 100, 50, theta=10, score=0.9)],
                "mindtct": [_record("mindtct", 112, 55, theta=14, score=0.7)],
                "fingerflow": [],
            },
            consensus_overlap_radius_px=20.0,
            min_consensus_sources=2,
        )

        self.assertEqual(len(clustered["consensus"]), 1)
        row = clustered["consensus"][0]
        self.assertEqual(row["source_count"], 2)
        self.assertEqual(row["representative_source"], "pyfing")
        self.assertEqual(row["x"], 100.0)
        self.assertEqual(row["y"], 50.0)
        self.assertEqual(row["amplitude"], 0.85)
        self.assertEqual(row["sigma_cells"], 1.25)

    def test_fingerflow_mindtct_consensus_without_pyfing_uses_fingerflow(self):
        clustered = consensus_script.cluster_consensus_minutiae(
            {
                "pyfing": [],
                "mindtct": [_record("mindtct", 200, 80, theta=90, score=0.6)],
                "fingerflow": [_record("fingerflow", 195, 79, theta=92, score=0.8)],
            },
            consensus_overlap_radius_px=20.0,
            min_consensus_sources=2,
        )

        self.assertEqual(len(clustered["consensus"]), 1)
        row = clustered["consensus"][0]
        self.assertEqual(row["representative_source"], "fingerflow")
        self.assertEqual(row["x"], 195.0)
        self.assertEqual(row["y"], 79.0)

    def test_three_source_consensus_gets_tier_and_policy(self):
        clustered = consensus_script.cluster_consensus_minutiae(
            {
                "pyfing": [_record("pyfing", 10, 10)],
                "mindtct": [_record("mindtct", 11, 11)],
                "fingerflow": [_record("fingerflow", 9, 10)],
            },
            consensus_overlap_radius_px=20.0,
            min_consensus_sources=2,
        )

        row = clustered["consensus"][0]
        self.assertEqual(row["source_count"], 3)
        self.assertEqual(row["label_tier"], "consensus_3src")
        self.assertEqual(row["amplitude"], 1.0)
        self.assertEqual(row["sigma_cells"], 1.0)

    def test_single_source_candidate_is_not_positive(self):
        clustered = consensus_script.cluster_consensus_minutiae(
            {"pyfing": [_record("pyfing", 10, 10)], "mindtct": [], "fingerflow": []},
            consensus_overlap_radius_px=20.0,
            min_consensus_sources=2,
        )

        self.assertEqual(clustered["consensus"], [])
        self.assertEqual(len(clustered["single_source"]), 1)
        self.assertEqual(clustered["single_source"][0]["label_tier"], "single_source_ignore")

    def test_same_source_detections_do_not_create_consensus(self):
        clustered = consensus_script.cluster_consensus_minutiae(
            {
                "pyfing": [
                    _record("pyfing", 10, 10, index=0),
                    _record("pyfing", 11, 11, index=1),
                ],
                "mindtct": [],
                "fingerflow": [],
            },
            consensus_overlap_radius_px=20.0,
            min_consensus_sources=2,
        )

        self.assertEqual(clustered["consensus"], [])
        self.assertGreaterEqual(len(clustered["single_source"]), 1)

    def test_missing_mindtct_message_is_clear(self):
        with self.assertRaisesRegex(FileNotFoundError, "NBIS mindtct executable was not found"):
            consensus_script._resolve_mindtct_bin("definitely_missing_mindtct_for_consensus_test")

    def test_missing_custom_fingerflow_message_is_clear(self):
        with self.assertRaisesRegex(FileNotFoundError, "FingerFlow executable was not found"):
            consensus_script.extract_fingerflow_minutiae(
                Path("dummy.png"),
                "definitely_missing_fingerflow_for_consensus_test",
            )

    def test_windows_resolve_accepts_explicit_wsl_mindtct_path(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(consensus_script.os, "name", "nt"), mock.patch.object(
            consensus_script.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            resolved = consensus_script._resolve_mindtct_bin(
                "/home/kazi/opt/nbis/bin/mindtct",
                wsl_distro="Ubuntu",
            )

        self.assertEqual(resolved, "wsl:/home/kazi/opt/nbis/bin/mindtct")
        run_mock.assert_called_once_with(
            ["wsl", "-d", "Ubuntu", "test", "-x", "/home/kazi/opt/nbis/bin/mindtct"],
            capture_output=True,
            text=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
