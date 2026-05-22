from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCH_OUTPUTS_DIR = REPO_ROOT / "match_outputs"


def ensure_stdlib_copy_module() -> None:
    """Avoid importing this repository's copy.py when matplotlib imports dataclasses."""
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

Path("/tmp/contactless_matplotlib_config").mkdir(parents=True, exist_ok=True)
Path("/tmp/contactless_matplotlib_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/contactless_matplotlib_config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/contactless_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_roc_csv(match_outputs_dir: Path = DEFAULT_MATCH_OUTPUTS_DIR) -> Path:
    candidates = sorted(
        match_outputs_dir.glob("binary_roc_sweep_*/roc_counts.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no roc_counts.csv found under {match_outputs_dir}/binary_roc_sweep_*")
    return candidates[0]


def _float_value(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float("nan")
    return float(value)


def read_roc_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"feature_score_threshold", "matching_threshold", "TPR", "FPR"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            feature_threshold = f"{_float_value(row, 'feature_score_threshold'):.2f}"
            tpr = _float_value(row, "TPR")
            fpr = _float_value(row, "FPR")
            if not (math.isfinite(tpr) and math.isfinite(fpr)):
                continue
            grouped[feature_threshold].append(
                {
                    "matching_threshold": _float_value(row, "matching_threshold"),
                    "tpr": tpr,
                    "fpr": fpr,
                }
            )
    return dict(grouped)


def trapezoid_auc(points: list[dict[str, Any]]) -> float:
    unique: dict[float, float] = {}
    for point in points:
        fpr = float(point["fpr"])
        tpr = float(point["tpr"])
        unique[fpr] = max(tpr, unique.get(fpr, 0.0))
    ordered = sorted(unique.items())
    if not ordered:
        return 0.0
    if ordered[0][0] > 0.0:
        ordered.insert(0, (0.0, ordered[0][1]))
    if ordered[-1][0] < 1.0:
        ordered.append((1.0, ordered[-1][1]))
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        auc += (x1 - x0) * ((y0 + y1) / 2.0)
    return float(auc)


def plot_roc_curves(roc_csv: Path, output_png: Path, output_svg: Path | None = None) -> dict[str, float]:
    grouped = read_roc_rows(roc_csv)
    wanted = ["0.80", "0.90"]
    missing = [threshold for threshold in wanted if threshold not in grouped]
    if missing:
        raise ValueError(f"{roc_csv} does not contain feature score thresholds: {missing}")

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    colors = {"0.80": "#1f77b4", "0.90": "#d62728"}
    auc_by_threshold: dict[str, float] = {}

    for threshold in wanted:
        points = sorted(grouped[threshold], key=lambda point: (float(point["fpr"]), float(point["tpr"])))
        fpr = [float(point["fpr"]) for point in points]
        tpr = [float(point["tpr"]) for point in points]
        auc = trapezoid_auc(points)
        auc_by_threshold[threshold] = auc
        ax.plot(
            fpr,
            tpr,
            marker="o",
            markersize=3.2,
            linewidth=2.0,
            color=colors[threshold],
            label=f"Feature score {threshold} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777777", label="Random")
    ax.set_title("MCC Binary Matching ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)
    return auc_by_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ROC curves from binary_roc_sweep roc_counts.csv.")
    parser.add_argument("--roc-csv", type=Path, default=None, help="Path to roc_counts.csv. Defaults to latest sweep run.")
    parser.add_argument("--output-png", type=Path, default=None, help="Output PNG path. Defaults beside roc_counts.csv.")
    parser.add_argument("--output-svg", type=Path, default=None, help="Optional SVG output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roc_csv = args.roc_csv.resolve() if args.roc_csv is not None else latest_roc_csv().resolve()
    output_png = args.output_png.resolve() if args.output_png is not None else roc_csv.with_name("roc_curve.png")
    output_svg = args.output_svg.resolve() if args.output_svg is not None else roc_csv.with_name("roc_curve.svg")

    auc_by_threshold = plot_roc_curves(roc_csv, output_png, output_svg)
    print(f"ROC CSV: {roc_csv}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved SVG: {output_svg}")
    for threshold, auc in sorted(auc_by_threshold.items()):
        print(f"Feature score {threshold} AUC: {auc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
