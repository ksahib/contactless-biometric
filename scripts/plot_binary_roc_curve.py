from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
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

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

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


def read_roc_row_sets(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped_by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
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
            source_label = str(row.get("method", "")).strip()
            grouped_by_source[source_label][feature_threshold].append(
                {
                    "matching_threshold": _float_value(row, "matching_threshold"),
                    "tpr": tpr,
                    "fpr": fpr,
                }
            )
    return {source: dict(grouped) for source, grouped in grouped_by_source.items()}


def read_roc_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    row_sets = read_roc_row_sets(path)
    if len(row_sets) == 1:
        return next(iter(row_sets.values()))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_grouped in row_sets.values():
        for threshold, points in source_grouped.items():
            grouped[threshold].extend(points)
    return dict(grouped)


def _sorted_thresholds(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(grouped, key=lambda threshold: float(threshold))


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
        ordered.insert(0, (0.0, 0.0))

    if ordered[-1][0] < 1.0:
        ordered.append((1.0, 1.0))

    auc = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        auc += (x1 - x0) * ((y0 + y1) / 2.0)

    return float(auc)


def _unique_label(label: str, seen: set[str]) -> str:
    if label not in seen:
        seen.add(label)
        return label
    index = 2
    while f"{label} ({index})" in seen:
        index += 1
    unique = f"{label} ({index})"
    seen.add(unique)
    return unique


def _parse_roc_csv_arg(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, path = value.split("=", 1)
    label = label.strip() or None
    path = path.strip()
    if not path:
        raise ValueError(f"--roc-csv value has an empty path: {value!r}")
    return label, Path(path)


def plot_roc_curve_sets(
    roc_inputs: list[tuple[str | None, Path]],
    output_png: Path,
    output_svg: Path | None = None,
) -> dict[str, dict[str, float]]:
    if not roc_inputs:
        raise ValueError("at least one ROC CSV is required")

    datasets: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for explicit_label, roc_csv in roc_inputs:
        row_sets = read_roc_row_sets(roc_csv)
        if not row_sets:
            raise ValueError(f"{roc_csv} does not contain any finite ROC rows")
        for source_label, grouped in row_sets.items():
            thresholds = _sorted_thresholds(grouped)
            if not thresholds:
                continue
            if explicit_label is not None and source_label:
                label = f"{explicit_label} {source_label}"
            else:
                label = explicit_label or source_label or roc_csv.parent.name
            datasets.append(
                {
                    "label": _unique_label(label, seen_labels),
                    "explicit_label": explicit_label is not None or bool(source_label),
                    "path": roc_csv,
                    "grouped": grouped,
                    "thresholds": thresholds,
                }
            )

    show_source_labels = len(datasets) > 1 or any(dataset["explicit_label"] for dataset in datasets)
    all_thresholds = _sorted_thresholds(
        {
            threshold: []
            for dataset in datasets
            for threshold in dataset["thresholds"]
        }
    )
    threshold_index = {threshold: index for index, threshold in enumerate(all_thresholds)}

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    method_cmap = plt.get_cmap("tab10")
    single_cmap = plt.get_cmap("tab10" if len(all_thresholds) <= 10 else "viridis")
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    auc_by_source: dict[str, dict[str, float]] = {}

    for dataset_index, dataset in enumerate(datasets):
        label = str(dataset["label"])
        grouped = dataset["grouped"]
        auc_by_threshold: dict[str, float] = {}
        auc_by_source[label] = auc_by_threshold

        for threshold in dataset["thresholds"]:
            style_index = threshold_index[threshold]
            points = sorted(grouped[threshold], key=lambda point: (float(point["fpr"]), float(point["tpr"])))
            fpr = [float(point["fpr"]) for point in points]
            tpr = [float(point["tpr"]) for point in points]
            auc = trapezoid_auc(points)
            auc_by_threshold[threshold] = auc
            if len(datasets) == 1:
                color = (
                    single_cmap(style_index % single_cmap.N)
                    if len(all_thresholds) <= 10
                    else single_cmap(style_index / max(len(all_thresholds) - 1, 1))
                )
                linestyle = "-"
            else:
                color = method_cmap(dataset_index % method_cmap.N)
                linestyle = line_styles[style_index % len(line_styles)]
            legend_label = (
                f"{label} score {threshold} (AUC={auc:.3f})"
                if show_source_labels
                else f"Feature score {threshold} (AUC={auc:.3f})"
            )
            ax.plot(
                fpr,
                tpr,
                marker=markers[style_index % len(markers)],
                markersize=3.2,
                linewidth=2.0,
                linestyle=linestyle,
                color=color,
                label=legend_label,
            )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777777", label="Random")
    ax.set_title("MCC Binary Matching ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)
    return auc_by_source


def plot_roc_curves(roc_csv: Path, output_png: Path, output_svg: Path | None = None) -> dict[str, float]:
    auc_by_source = plot_roc_curve_sets([(None, roc_csv)], output_png, output_svg)
    return next(iter(auc_by_source.values()))


def plot_raw_score_roc_curves(
    score_csvs: list[Path],
    output_png: Path,
    output_svg: Path | None = None,
) -> dict[str, float]:
    if not score_csvs:
        raise ValueError("at least one raw score CSV is required")

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    auc_by_threshold: dict[str, float] = {}

    cmap = plt.get_cmap("tab10")

    for index, score_csv in enumerate(sorted(_expand_raw_score_csvs(score_csvs))):
        threshold = _raw_score_threshold_label(score_csv)

        df = pd.read_csv(score_csv)

        # Keep only successful comparisons, if the status column exists.
        if "status" in df.columns:
            df = df[df["status"].eq("ok")].copy()

        df = df.dropna(subset=["score", "label"])

        y_true = (df["label"] == "genuine").astype(int)
        y_score = df["score"].astype(float)
        method = str(df["method"].dropna().iloc[0]) if "method" in df.columns and not df["method"].dropna().empty else ""
        curve_label = f"{method} score {threshold}" if method else f"Feature score {threshold}"

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        auc_by_threshold[curve_label] = float(auc)

        ax.plot(
            fpr,
            tpr,
            marker="o",
            markersize=2.0,
            linewidth=2.0,
            color=cmap(index % cmap.N),
            label=f"{curve_label} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777777", label="Random")
    ax.set_title("MCC Binary Matching ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)

    if output_svg is not None:
        fig.savefig(output_svg)

    plt.close(fig)
    return auc_by_threshold


def _raw_score_threshold_label(score_csv: Path) -> str:
    match = re.search(r"(?:^|_)score_([0-9]+(?:\.[0-9]+)?)$", score_csv.stem)
    if match:
        return match.group(1)
    return score_csv.stem.replace("mcc_scores_score_", "")


def _expand_raw_score_csvs(score_csvs: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for score_csv in score_csvs:
        threshold = _raw_score_threshold_label(score_csv)
        method_specific = sorted(score_csv.parent.glob(f"mcc_scores_*_score_{threshold}.csv"))
        candidates = method_specific or [score_csv]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            expanded.append(resolved)
    return expanded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ROC curves from binary_roc_sweep roc_counts.csv.")
    parser.add_argument(
        "--roc-csv",
        action="append",
        default=None,
        help="Path to roc_counts.csv, optionally as LABEL=PATH. Repeat to overlay multiple runs. Defaults to latest sweep run.",
    )
    parser.add_argument("--output-png", type=Path, default=None, help="Output PNG path. Defaults beside roc_counts.csv.")
    parser.add_argument("--output-svg", type=Path, default=None, help="Optional SVG output path.")
    parser.add_argument(
        "--raw-score-csv",
        action="append",
        default=None,
        help="Path to raw mcc_scores_score_*.csv. Repeat for multiple feature thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.raw_score_csv is not None:
        score_csvs = [Path(path).resolve() for path in args.raw_score_csv]

        if args.output_png is not None:
            output_png = args.output_png.resolve()
        else:
            output_png = score_csvs[0].parent / "roc_curve_raw_scores.png"

        if args.output_svg is not None:
            output_svg = args.output_svg.resolve()
        else:
            output_svg = score_csvs[0].parent / "roc_curve_raw_scores.svg"

        auc_by_threshold = plot_raw_score_roc_curves(score_csvs, output_png, output_svg)

        print(f"Saved PNG: {output_png}")
        print(f"Saved SVG: {output_svg}")

        for label, auc in sorted(auc_by_threshold.items()):
            print(f"{label} AUC: {auc:.6f}")

        return 0
    roc_inputs = (
        [_parse_roc_csv_arg(value) for value in args.roc_csv]
        if args.roc_csv is not None
        else [(None, latest_roc_csv())]
    )
    roc_inputs = [(label, path.resolve()) for label, path in roc_inputs]
    if args.output_png is not None:
        output_png = args.output_png.resolve()
    elif len(roc_inputs) == 1:
        output_png = roc_inputs[0][1].with_name("roc_curve.png")
    else:
        output_png = DEFAULT_MATCH_OUTPUTS_DIR / "roc_curve_comparison.png"
    if args.output_svg is not None:
        output_svg = args.output_svg.resolve()
    elif len(roc_inputs) == 1:
        output_svg = roc_inputs[0][1].with_name("roc_curve.svg")
    else:
        output_svg = DEFAULT_MATCH_OUTPUTS_DIR / "roc_curve_comparison.svg"

    auc_by_source = plot_roc_curve_sets(roc_inputs, output_png, output_svg)
    for label, roc_csv in roc_inputs:
        if len(roc_inputs) > 1 or label is not None:
            print(f"ROC CSV [{label or roc_csv.parent.name}]: {roc_csv}")
        else:
            print(f"ROC CSV: {roc_csv}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved SVG: {output_svg}")
    for source_label, auc_by_threshold in auc_by_source.items():
        for threshold, auc in sorted(auc_by_threshold.items()):
            if len(auc_by_source) == 1 and roc_inputs[0][0] is None:
                print(f"Feature score {threshold} AUC: {auc:.6f}")
            else:
                print(f"{source_label} feature score {threshold} AUC: {auc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
