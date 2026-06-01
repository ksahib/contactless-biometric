#!/usr/bin/env python
"""Consensus-label unwrapped minutiae with pyfing, NBIS MINDTCT, and FingerFlow."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DPI = 500
DEFAULT_OVERLAP_RADIUS_PX = 20.0
DEFAULT_ROLES = ("front", "left", "right")
DEFAULT_FINGERFLOW_MODEL_DIR = REPO_ROOT / ".fingerflow_models"

VALID_SOURCES = {"pyfing", "mindtct", "fingerflow"}
REPRESENTATIVE_SOURCE_PRIORITY = ("pyfing", "fingerflow", "mindtct")

ROLE_INPUTS: dict[str, tuple[str, str]] = {
    "front": ("front_algorithm3_unwrapped.png", "front_algorithm3_filled_output.png"),
    "left": ("left_canonical_chart_unwrapped.png", "left_canonical_chart_filled_output.png"),
    "right": ("right_canonical_chart_unwrapped.png", "right_canonical_chart_filled_output.png"),
}

CONSENSUS_CSV_FIELDS = [
    "x",
    "y",
    "theta",
    "score",
    "type",
    "sources",
    "source_count",
    "representative_source",
    "mean_score",
    "max_score",
    "spatial_spread_px",
    "orientation_spread_deg",
    "amplitude",
    "sigma_cells",
    "label_tier",
]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _is_repo_root_path(path_value: str) -> bool:
    try:
        return Path(path_value or ".").resolve() == REPO_ROOT
    except OSError:
        return False


def _import_module_without_repo_root(module_name: str) -> Any:
    old_path = list(sys.path)
    sys.path[:] = [path for path in sys.path if not _is_repo_root_path(path)]
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path[:] = old_path


def _load_cv2() -> Any:
    return _import_module_without_repo_root("cv2")


def _load_numpy() -> Any:
    import numpy as np

    return np


def normalize_theta(theta: Any) -> float | None:
    if theta is None:
        return None
    value = float(theta)
    if not math.isfinite(value):
        return None
    return value % 360.0


def _angle_to_degrees(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if abs(numeric) <= (2.0 * math.pi + 1e-6):
        numeric = math.degrees(numeric)
    return normalize_theta(numeric)


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs((float(a) - float(b)) % 360.0)
    return min(diff, 360.0 - diff)


def _score_to_unit(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric < 0.0:
        return 0.0
    if numeric <= 1.0:
        return float(numeric)
    return float(max(0.0, min(numeric / 100.0, 1.0)))


def _normal_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"ending", "ridge ending", "end", "e", "1"}:
        return "ending"
    if text in {"bifurcation", "bif", "b", "2"}:
        return "bifurcation"
    return "unknown"


def _read_gray(path: Path) -> Any:
    cv2 = _load_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _role_input_paths(unwarp_output_dir: Path, role: str) -> tuple[Path, Path]:
    if role not in ROLE_INPUTS:
        raise ValueError(f"unsupported role {role!r}; expected one of {sorted(ROLE_INPUTS)}")
    image_name, mask_name = ROLE_INPUTS[role]
    role_dir = unwarp_output_dir / role
    return role_dir / image_name, role_dir / mask_name


def _validate_role_inputs(unwarp_output_dir: Path, roles: Iterable[str]) -> dict[str, dict[str, Path]]:
    inputs: dict[str, dict[str, Path]] = {}
    for role in roles:
        image_path, mask_path = _role_input_paths(unwarp_output_dir, role)
        missing = [str(path) for path in (image_path, mask_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing {role} input artifact(s): {', '.join(missing)}")
        inputs[role] = {"image": image_path, "mask": mask_path}
    return inputs


def _normalized_record(
    *,
    x: Any,
    y: Any,
    theta: Any,
    score: Any,
    minutia_type: Any,
    source: str,
    source_index: int,
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    x_float = float(x)
    y_float = float(y)
    if not (math.isfinite(x_float) and math.isfinite(y_float)):
        return None
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid source: {source}")
    return {
        "x": x_float,
        "y": y_float,
        "theta": _angle_to_degrees(theta),
        "score": _score_to_unit(score),
        "type": _normal_type(minutia_type),
        "source": source,
        "source_index": int(source_index),
        "raw": dict(raw),
    }


def extract_pyfing_minutiae(image_path: Path, dpi: int = DEFAULT_DPI) -> list[dict[str, Any]]:
    image = _read_gray(image_path)
    pyfing = _import_module_without_repo_root("pyfing")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(pyfing.minutiae_extraction(image, dpi=int(dpi))):
        raw = {
            "x": getattr(item, "x", None),
            "y": getattr(item, "y", None),
            "direction": getattr(item, "direction", getattr(item, "angle", None)),
            "quality": getattr(item, "quality", None),
            "type": getattr(item, "type", None),
        }
        record = _normalized_record(
            x=raw["x"],
            y=raw["y"],
            theta=raw["direction"],
            score=raw["quality"],
            minutia_type=raw["type"],
            source="pyfing",
            source_index=index,
            raw=raw,
        )
        if record is not None:
            rows.append(record)
    return rows


def _parse_nbis_xyt_lines(lines: Iterable[str], source: str = "mindtct") -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            skipped += 1
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            theta_degrees = float(parts[2])
            quality = float(parts[3]) if len(parts) >= 4 else None
        except ValueError:
            skipped += 1
            continue
        raw = {
            "line_number": line_number,
            "line": stripped,
            "x": x,
            "y": y,
            "theta_degrees": theta_degrees,
            "quality": quality,
        }
        record = _normalized_record(
            x=x,
            y=y,
            theta=theta_degrees,
            score=quality,
            minutia_type=None,
            source=source,
            source_index=len(rows),
            raw=raw,
        )
        if record is None:
            skipped += 1
            continue
        rows.append(record)
    return rows, skipped


def _parse_nbis_xyt(path: Path) -> tuple[list[dict[str, Any]], int]:
    with path.open(encoding="utf-8") as handle:
        return _parse_nbis_xyt_lines(handle)


def _looks_like_wsl_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("wsl:")


def _strip_wsl_prefix(value: str) -> str:
    return value[4:] if value.startswith("wsl:") else value


def _wsl_command(args: list[str], distro: str | None = None) -> list[str]:
    command = ["wsl"]
    if distro:
        command.extend(["-d", distro])
    command.extend(args)
    return command


def _windows_path_to_wsl(path: Path, distro: str | None = None) -> str:
    completed = subprocess.run(
        _wsl_command(["wslpath", "-a", str(path.resolve())], distro=distro),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "wslpath failed"
        distro_hint = f" in distro {distro!r}" if distro else ""
        raise RuntimeError(f"could not convert Windows path to WSL path{distro_hint} for {path}: {stderr}")
    converted = completed.stdout.strip()
    if not converted:
        raise RuntimeError(f"wslpath returned an empty path for {path}")
    return converted


def _resolve_mindtct_bin(mindtct_bin: str, wsl_distro: str | None = None) -> str:
    if os.name == "nt" and _looks_like_wsl_path(mindtct_bin):
        wsl_path = _strip_wsl_prefix(mindtct_bin)
        completed = subprocess.run(
            _wsl_command(["test", "-x", wsl_path], distro=wsl_distro),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            distro_hint = f" in distro {wsl_distro!r}" if wsl_distro else ""
            raise FileNotFoundError(f"NBIS mindtct executable not found in WSL{distro_hint}: {wsl_path}")
        return f"wsl:{wsl_path}"

    candidate = Path(mindtct_bin)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"NBIS mindtct executable not found: {mindtct_bin}")
    resolved = shutil.which(mindtct_bin)
    if resolved is None:
        raise FileNotFoundError(
            "NBIS mindtct executable was not found on PATH. Install NBIS or pass --mindtct-bin "
            "with the full path to the mindtct executable."
        )
    return resolved


def _extract_mindtct_minutiae_with_details(
    image_path: Path,
    mindtct_bin: str,
    *,
    output_root: Path | None = None,
    wsl_distro: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_root = output_root or (image_path.parent / f"{image_path.stem}_mindtct")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    executable = _resolve_mindtct_bin(mindtct_bin, wsl_distro=wsl_distro)

    if executable.startswith("wsl:"):
        wsl_executable = _strip_wsl_prefix(executable)
        command = _wsl_command(
            [
                wsl_executable,
                _windows_path_to_wsl(image_path, distro=wsl_distro),
                _windows_path_to_wsl(output_root, distro=wsl_distro),
            ],
            distro=wsl_distro,
        )
        display_executable = wsl_executable
    else:
        command = [executable, str(image_path), str(output_root)]
        display_executable = executable

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    details: dict[str, Any] = {
        "mindtct_bin": display_executable,
        "mindtct_invocation": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "xyt_path": str(output_root.with_suffix(".xyt")),
        "skipped_xyt_lines": 0,
    }
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown NBIS mindtct failure"
        raise RuntimeError(f"NBIS mindtct failed for {image_path}: {stderr}")

    xyt_path = output_root.with_suffix(".xyt")
    if not xyt_path.exists():
        raise FileNotFoundError(f"NBIS mindtct did not produce expected .xyt file: {xyt_path}")
    rows, skipped = _parse_nbis_xyt(xyt_path)
    details["skipped_xyt_lines"] = int(skipped)
    return rows, details


def extract_mindtct_minutiae(image_path: Path, mindtct_bin: str) -> list[dict[str, Any]]:
    rows, _details = _extract_mindtct_minutiae_with_details(image_path, mindtct_bin)
    return rows


def _resolve_bridge_command(
    fingerflow_bin: str,
    image_path: Path,
    model_dir: Path,
    minutiae_json: Path,
    minutiae_csv: Path,
    core_csv: Path,
) -> list[str]:
    if fingerflow_bin == "fingerflow":
        bridge_path = REPO_ROOT / "fingerflow_bridge.py"
        if not bridge_path.exists():
            raise FileNotFoundError(f"FingerFlow bridge was not found: {bridge_path}")
        return [
            sys.executable,
            str(bridge_path),
            "--source-image",
            str(image_path.resolve()),
            "--enhanced-image",
            str(image_path.resolve()),
            "--model-dir",
            str(model_dir.resolve()),
            "--minutiae-json",
            str(minutiae_json.resolve()),
            "--minutiae-csv",
            str(minutiae_csv.resolve()),
            "--core-csv",
            str(core_csv.resolve()),
        ]

    candidate = Path(fingerflow_bin)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"FingerFlow executable was not found: {fingerflow_bin}")
        executable = str(candidate)
    else:
        resolved = shutil.which(fingerflow_bin)
        if resolved is None:
            raise FileNotFoundError(
                "FingerFlow executable was not found on PATH. Use the default 'fingerflow' "
                "bridge or pass --fingerflow-bin with a bridge-compatible executable."
            )
        executable = resolved
    return [
        executable,
        "--source-image",
        str(image_path.resolve()),
        "--enhanced-image",
        str(image_path.resolve()),
        "--model-dir",
        str(model_dir.resolve()),
        "--minutiae-json",
        str(minutiae_json.resolve()),
        "--minutiae-csv",
        str(minutiae_csv.resolve()),
        "--core-csv",
        str(core_csv.resolve()),
    ]


def _fingerflow_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    use_gpu = env.get("FINGERFLOW_USE_GPU", "").lower() in {"1", "true", "yes", "on"}
    if use_gpu:
        env.setdefault("FINGERFLOW_ALLOW_CPU", "0")
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["NVIDIA_VISIBLE_DEVICES"] = ""
        env["FINGERFLOW_ALLOW_CPU"] = "1"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    return env


def _extract_fingerflow_minutiae_with_details(
    image_path: Path,
    fingerflow_bin: str,
    *,
    model_dir: Path = DEFAULT_FINGERFLOW_MODEL_DIR,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = output_dir or (image_path.parent / "fingerflow_raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json = output_dir / "fingerflow_minutiae.json"
    raw_csv = output_dir / "fingerflow_minutiae.csv"
    raw_core_csv = output_dir / "fingerflow_core.csv"
    command = _resolve_bridge_command(fingerflow_bin, image_path, model_dir, raw_json, raw_csv, raw_core_csv)
    env = _fingerflow_subprocess_env()
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    details = {
        "fingerflow_bin": fingerflow_bin,
        "fingerflow_invocation": command,
        "fingerflow_env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "FINGERFLOW_ALLOW_CPU": env.get("FINGERFLOW_ALLOW_CPU"),
            "FINGERFLOW_USE_GPU": env.get("FINGERFLOW_USE_GPU"),
            "TF_FORCE_GPU_ALLOW_GROWTH": env.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "TF_GPU_ALLOCATOR": env.get("TF_GPU_ALLOCATOR"),
        },
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "raw_json_path": str(raw_json),
    }
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown FingerFlow failure"
        raise RuntimeError(f"FingerFlow failed for {image_path}: {stderr}")
    if not raw_json.exists():
        raise FileNotFoundError(f"FingerFlow did not produce expected JSON file: {raw_json}")

    try:
        payload = json.loads(raw_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FingerFlow output could not be parsed: {raw_json}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("minutiae", [])):
        if not isinstance(raw, dict):
            continue
        theta = raw.get("theta", raw.get("angle", raw.get("direction")))
        score = raw.get("score", raw.get("quality", raw.get("confidence")))
        record = _normalized_record(
            x=raw.get("x"),
            y=raw.get("y"),
            theta=theta,
            score=score,
            minutia_type=raw.get("type", raw.get("class")),
            source="fingerflow",
            source_index=index,
            raw=raw,
        )
        if record is not None:
            rows.append(record)
    return rows, details


def extract_fingerflow_minutiae(image_path: Path, fingerflow_bin: str) -> list[dict[str, Any]]:
    rows, _details = _extract_fingerflow_minutiae_with_details(image_path, fingerflow_bin)
    return rows


def _point_inside_mask(mask: Any, x: float, y: float) -> bool:
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return False
    h, w = mask.shape[:2]
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    return bool(0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0)


def _filter_minutiae_by_mask(rows: list[dict[str, Any]], mask: Any) -> tuple[list[dict[str, Any]], int]:
    kept = [
        row
        for row in rows
        if _point_inside_mask(mask, float(row.get("x", float("nan"))), float(row.get("y", float("nan"))))
    ]
    return kept, int(len(rows) - len(kept))


def _score_or_minus_one(row: dict[str, Any]) -> float:
    score = row.get("score")
    if score is None:
        return -1.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        return -1.0
    return value if math.isfinite(value) else -1.0


def _choose_best_per_source(component: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cx = sum(float(row["x"]) for row in component) / max(len(component), 1)
    cy = sum(float(row["y"]) for row in component) / max(len(component), 1)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component:
        by_source[str(row["source"])].append(row)

    chosen: dict[str, dict[str, Any]] = {}
    for source, rows in by_source.items():
        chosen[source] = sorted(
            rows,
            key=lambda row: (
                -_score_or_minus_one(row),
                math.hypot(float(row["x"]) - cx, float(row["y"]) - cy),
                int(row.get("source_index", 0)),
            ),
        )[0]
    return chosen


def _representative_source(sources: set[str]) -> str:
    for source in REPRESENTATIVE_SOURCE_PRIORITY:
        if source in sources:
            return source
    raise ValueError(f"no representative source available for {sorted(sources)}")


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _max_or_none(values: list[float]) -> float | None:
    return None if not values else float(max(values))


def _orientation_spread_or_none(records: list[dict[str, Any]]) -> float | None:
    angles = [float(row["theta"]) for row in records if row.get("theta") is not None]
    if len(angles) < 2:
        return None
    return float(max(angle_diff_deg(left, right) for i, left in enumerate(angles) for right in angles[i + 1 :]))


def cluster_consensus_minutiae(
    detections_by_source: dict[str, list[dict[str, Any]]],
    *,
    consensus_overlap_radius_px: float,
    min_consensus_sources: int = 2,
    gaussian_amp_3src: float = 1.0,
    gaussian_amp_2src: float = 0.85,
    gaussian_sigma_3src_cells: float = 1.0,
    gaussian_sigma_2src_cells: float = 1.25,
    single_source_ignore_sigma_cells: float = 1.25,
) -> dict[str, list[dict[str, Any]]]:
    radius = float(consensus_overlap_radius_px)
    if radius < 0.0:
        raise ValueError("consensus_overlap_radius_px must be non-negative")
    if int(min_consensus_sources) < 1:
        raise ValueError("min_consensus_sources must be at least 1")

    detections: list[dict[str, Any]] = []
    for source in sorted(detections_by_source):
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid source: {source}")
        for fallback_index, record in enumerate(detections_by_source[source]):
            normalized = dict(record)
            normalized["source"] = source
            normalized["source_index"] = int(normalized.get("source_index", fallback_index))
            normalized["x"] = float(normalized["x"])
            normalized["y"] = float(normalized["y"])
            normalized["theta"] = normalize_theta(normalized.get("theta"))
            if math.isfinite(normalized["x"]) and math.isfinite(normalized["y"]):
                detections.append(normalized)

    uf = UnionFind(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if detections[i]["source"] == detections[j]["source"]:
                continue
            distance = math.hypot(detections[i]["x"] - detections[j]["x"], detections[i]["y"] - detections[j]["y"])
            if distance <= radius:
                uf.union(i, j)

    components_by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, detection in enumerate(detections):
        components_by_root[uf.find(index)].append(detection)

    all_clusters: list[dict[str, Any]] = []
    for component in components_by_root.values():
        chosen_by_source = _choose_best_per_source(component)
        sources = set(chosen_by_source)
        source_count = len(sources)
        rep_source = _representative_source(sources)
        rep = chosen_by_source[rep_source]
        chosen_records = list(chosen_by_source.values())
        scores = [_score_or_minus_one(row) for row in chosen_records if _score_or_minus_one(row) >= 0]
        xs = [float(row["x"]) for row in chosen_records]
        ys = [float(row["y"]) for row in chosen_records]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        spatial_spread_px = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))

        if source_count >= 3:
            amplitude = float(gaussian_amp_3src)
            sigma_cells = float(gaussian_sigma_3src_cells)
            label_tier = "consensus_3src"
        elif source_count == 2:
            amplitude = float(gaussian_amp_2src)
            sigma_cells = float(gaussian_sigma_2src_cells)
            label_tier = "consensus_2src"
        else:
            amplitude = 0.0
            sigma_cells = float(single_source_ignore_sigma_cells)
            label_tier = "single_source_ignore"

        cluster = {
            "x": float(rep["x"]),
            "y": float(rep["y"]),
            "theta": normalize_theta(rep.get("theta")),
            "score": rep.get("score"),
            "type": rep.get("type"),
            "sources": sorted(sources),
            "source_count": int(source_count),
            "representative_source": rep_source,
            "mean_score": _mean_or_none(scores),
            "max_score": _max_or_none(scores),
            "spatial_spread_px": float(spatial_spread_px),
            "orientation_spread_deg": _orientation_spread_or_none(chosen_records),
            "amplitude": float(max(0.0, min(amplitude, 1.0))),
            "sigma_cells": float(sigma_cells),
            "label_tier": label_tier,
            "source_records": chosen_by_source,
        }
        all_clusters.append(cluster)

    all_clusters.sort(key=lambda row: (float(row["y"]), float(row["x"]), -int(row["source_count"])))
    consensus = [row for row in all_clusters if int(row["source_count"]) >= int(min_consensus_sources)]
    single_source = [row for row in all_clusters if int(row["source_count"]) == 1]
    return {
        "consensus": consensus,
        "single_source": single_source,
        "all_clusters": all_clusters,
    }


def consensus_match_minutiae(
    pyfing_rows: list[dict[str, Any]],
    nbis_rows: list[dict[str, Any]],
    overlap_radius_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clustered = cluster_consensus_minutiae(
        {"pyfing": pyfing_rows, "mindtct": nbis_rows, "fingerflow": []},
        consensus_overlap_radius_px=overlap_radius_px,
        min_consensus_sources=2,
    )
    details = {
        "pyfing_count": int(len(pyfing_rows)),
        "nbis_count": int(len(nbis_rows)),
        "consensus_count": int(len(clustered["consensus"])),
        "pyfing_discarded_count": max(0, int(len(pyfing_rows)) - int(len(clustered["consensus"]))),
        "nbis_unmatched_count": max(0, int(len(nbis_rows)) - int(len(clustered["consensus"]))),
        "overlap_radius_px": float(overlap_radius_px),
    }
    return clustered["consensus"], details


def _draw_consensus_overlay(image: Any, mask: Any, clusters: list[dict[str, Any]]) -> Any:
    cv2 = _load_cv2()
    np = _load_numpy()
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if mask.shape == canvas.shape[:2]:
        tint = np.zeros_like(canvas)
        tint[:, :, 1] = np.where(mask > 0, 45, 0).astype(np.uint8)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.45, 0)
    h, w = canvas.shape[:2]
    colors = {
        "consensus_3src": (0, 255, 0),
        "consensus_2src": (0, 220, 255),
        "single_source_ignore": (160, 160, 160),
    }
    for row in clusters:
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        color = colors.get(str(row.get("label_tier")), (255, 255, 255))
        cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
        theta = row.get("theta")
        if theta is not None:
            theta_rad = math.radians(float(theta))
            x2 = int(round(x + 16.0 * math.cos(theta_rad)))
            y2 = int(round(y + 16.0 * math.sin(theta_rad)))
            cv2.line(canvas, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
    return canvas


def _json_payload(
    image_path: Path,
    mask_path: Path,
    overlap_radius_px: float,
    min_consensus_sources: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "image": str(image_path.resolve()),
        "mask": str(mask_path.resolve()),
        "angle_units": "degrees",
        "overlap_radius_px": float(overlap_radius_px),
        "min_consensus_sources": int(min_consensus_sources),
        "count": int(len(rows)),
        "minutiae": rows,
    }


def _process_role(
    role: str,
    image_path: Path,
    mask_path: Path,
    role_output_dir: Path,
    *,
    dpi: int,
    overlap_radius_px: float,
    mindtct_bin: str,
    fingerflow_bin: str,
    fingerflow_model_dir: Path,
    min_consensus_sources: int,
    gaussian_amp_3src: float,
    gaussian_amp_2src: float,
    gaussian_sigma_3src_cells: float,
    gaussian_sigma_2src_cells: float,
    single_source_ignore_sigma_cells: float,
    wsl_distro: str | None = None,
) -> dict[str, Any]:
    cv2 = _load_cv2()
    image = _read_gray(image_path)
    mask = _read_gray(mask_path)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"{role} image/mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}")

    role_output_dir.mkdir(parents=True, exist_ok=True)

    pyfing_raw = extract_pyfing_minutiae(image_path, dpi=dpi)
    pyfing_rows, pyfing_dropped_mask = _filter_minutiae_by_mask(pyfing_raw, mask)

    mindtct_raw, mindtct_details = _extract_mindtct_minutiae_with_details(
        image_path=image_path,
        output_root=role_output_dir / "mindtct_raw" / f"{role}_mindtct",
        mindtct_bin=mindtct_bin,
        wsl_distro=wsl_distro,
    )
    mindtct_rows, mindtct_dropped_mask = _filter_minutiae_by_mask(mindtct_raw, mask)

    fingerflow_raw, fingerflow_details = _extract_fingerflow_minutiae_with_details(
        image_path=image_path,
        fingerflow_bin=fingerflow_bin,
        model_dir=fingerflow_model_dir,
        output_dir=role_output_dir / "fingerflow_raw",
    )
    fingerflow_rows, fingerflow_dropped_mask = _filter_minutiae_by_mask(fingerflow_raw, mask)

    clustered = cluster_consensus_minutiae(
        {
            "pyfing": pyfing_rows,
            "mindtct": mindtct_rows,
            "fingerflow": fingerflow_rows,
        },
        consensus_overlap_radius_px=float(overlap_radius_px),
        min_consensus_sources=int(min_consensus_sources),
        gaussian_amp_3src=gaussian_amp_3src,
        gaussian_amp_2src=gaussian_amp_2src,
        gaussian_sigma_3src_cells=gaussian_sigma_3src_cells,
        gaussian_sigma_2src_cells=gaussian_sigma_2src_cells,
        single_source_ignore_sigma_cells=single_source_ignore_sigma_cells,
    )
    consensus_rows = clustered["consensus"]
    single_source_rows = clustered["single_source"]
    all_clusters = clustered["all_clusters"]

    pyfing_json = role_output_dir / "pyfing_minutiae.json"
    mindtct_json = role_output_dir / "mindtct_minutiae.json"
    fingerflow_json = role_output_dir / "fingerflow_minutiae.json"
    consensus_json = role_output_dir / "consensus_minutiae.json"
    consensus_csv = role_output_dir / "consensus_minutiae.csv"
    all_clusters_json = role_output_dir / "all_minutiae_clusters.json"
    single_source_json = role_output_dir / "single_source_candidates.json"
    role_report_path = role_output_dir / "consensus_report.json"
    overlay_path = role_output_dir / "consensus_overlay.png"

    _write_json(pyfing_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, pyfing_rows))
    _write_json(mindtct_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, mindtct_rows) | {"mindtct": mindtct_details})
    _write_json(fingerflow_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, fingerflow_rows) | {"fingerflow": fingerflow_details})
    _write_json(consensus_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, consensus_rows))
    _write_json(all_clusters_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, all_clusters))
    _write_json(single_source_json, _json_payload(image_path, mask_path, overlap_radius_px, min_consensus_sources, single_source_rows))
    _write_csv(consensus_rows, consensus_csv, CONSENSUS_CSV_FIELDS)

    overlay = _draw_consensus_overlay(image, mask, all_clusters)
    cv2.imwrite(str(overlay_path), overlay)

    counts = {
        "pyfing": int(len(pyfing_rows)),
        "mindtct": int(len(mindtct_rows)),
        "fingerflow": int(len(fingerflow_rows)),
        "consensus_positive": int(len(consensus_rows)),
        "single_source_candidates": int(len(single_source_rows)),
        "all_clusters": int(len(all_clusters)),
    }
    role_report = {
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "role": role,
        "extractors": ["pyfing", "mindtct", "fingerflow"],
        "consensus_overlap_radius_px": float(overlap_radius_px),
        "min_consensus_sources": int(min_consensus_sources),
        "counts": counts,
        "amplitude_policy": {"3src": float(gaussian_amp_3src), "2src": float(gaussian_amp_2src)},
        "sigma_policy_cells": {
            "3src": float(gaussian_sigma_3src_cells),
            "2src": float(gaussian_sigma_2src_cells),
            "single_source_ignore": float(single_source_ignore_sigma_cells),
        },
    }
    _write_json(role_report_path, role_report)

    return {
        "role": role,
        "image": str(image_path.resolve()),
        "mask": str(mask_path.resolve()),
        "pyfing_count": int(len(pyfing_rows)),
        "pyfing_raw_count": int(len(pyfing_raw)),
        "pyfing_dropped_outside_mask": int(pyfing_dropped_mask),
        "mindtct_count": int(len(mindtct_rows)),
        "nbis_count": int(len(mindtct_rows)),
        "mindtct_raw_count": int(len(mindtct_raw)),
        "mindtct_dropped_outside_mask": int(mindtct_dropped_mask),
        "mindtct_skipped_xyt_lines": int(mindtct_details.get("skipped_xyt_lines", 0)),
        "fingerflow_count": int(len(fingerflow_rows)),
        "fingerflow_raw_count": int(len(fingerflow_raw)),
        "fingerflow_dropped_outside_mask": int(fingerflow_dropped_mask),
        "consensus_positive_count": int(len(consensus_rows)),
        "consensus_survivor_count": int(len(consensus_rows)),
        "single_source_candidate_count": int(len(single_source_rows)),
        "all_cluster_count": int(len(all_clusters)),
        "pyfing_discarded_count": max(0, int(len(pyfing_rows)) - int(len(consensus_rows))),
        "nbis_unmatched_count": 0,
        "overlap_radius_px": float(overlap_radius_px),
        "artifacts": {
            "pyfing_minutiae_json": str(pyfing_json.resolve()),
            "mindtct_minutiae_json": str(mindtct_json.resolve()),
            "nbis_minutiae_json": str(mindtct_json.resolve()),
            "fingerflow_minutiae_json": str(fingerflow_json.resolve()),
            "consensus_minutiae_json": str(consensus_json.resolve()),
            "consensus_minutiae_csv": str(consensus_csv.resolve()),
            "all_minutiae_clusters_json": str(all_clusters_json.resolve()),
            "single_source_candidates_json": str(single_source_json.resolve()),
            "consensus_report_json": str(role_report_path.resolve()),
            "consensus_overlay": str(overlay_path.resolve()),
        },
        "report": role_report,
    }


def run(
    unwarp_output_dir: Path,
    output_dir: Path | None,
    roles: Iterable[str],
    overlap_radius_px: float,
    dpi: int,
    mindtct_bin: str,
    *,
    fingerflow_bin: str = "fingerflow",
    fingerflow_model_dir: Path = DEFAULT_FINGERFLOW_MODEL_DIR,
    min_consensus_sources: int = 2,
    gaussian_amp_3src: float = 1.0,
    gaussian_amp_2src: float = 0.85,
    gaussian_sigma_3src_cells: float = 1.0,
    gaussian_sigma_2src_cells: float = 1.25,
    single_source_ignore_sigma_cells: float = 1.25,
    wsl_distro: str | None = None,
) -> dict[str, Any]:
    unwarp_output_dir = unwarp_output_dir.resolve()
    if output_dir is None:
        output_dir = unwarp_output_dir / "consensus_minutiae"
    output_dir = output_dir.resolve()
    role_list = tuple(roles)
    role_inputs = _validate_role_inputs(unwarp_output_dir, role_list)
    resolved_mindtct_bin = _resolve_mindtct_bin(mindtct_bin, wsl_distro=wsl_distro)

    output_dir.mkdir(parents=True, exist_ok=True)
    role_reports: dict[str, Any] = {}
    for role in role_list:
        role_reports[role] = _process_role(
            role=role,
            image_path=role_inputs[role]["image"],
            mask_path=role_inputs[role]["mask"],
            role_output_dir=output_dir / role,
            dpi=int(dpi),
            overlap_radius_px=float(overlap_radius_px),
            mindtct_bin=resolved_mindtct_bin,
            fingerflow_bin=fingerflow_bin,
            fingerflow_model_dir=fingerflow_model_dir,
            min_consensus_sources=int(min_consensus_sources),
            gaussian_amp_3src=float(gaussian_amp_3src),
            gaussian_amp_2src=float(gaussian_amp_2src),
            gaussian_sigma_3src_cells=float(gaussian_sigma_3src_cells),
            gaussian_sigma_2src_cells=float(gaussian_sigma_2src_cells),
            single_source_ignore_sigma_cells=float(single_source_ignore_sigma_cells),
            wsl_distro=wsl_distro,
        )

    totals = {
        "pyfing_count": int(sum(report["pyfing_count"] for report in role_reports.values())),
        "mindtct_count": int(sum(report["mindtct_count"] for report in role_reports.values())),
        "nbis_count": int(sum(report["mindtct_count"] for report in role_reports.values())),
        "fingerflow_count": int(sum(report["fingerflow_count"] for report in role_reports.values())),
        "consensus_positive_count": int(sum(report["consensus_positive_count"] for report in role_reports.values())),
        "consensus_survivor_count": int(sum(report["consensus_survivor_count"] for report in role_reports.values())),
        "single_source_candidate_count": int(sum(report["single_source_candidate_count"] for report in role_reports.values())),
        "all_cluster_count": int(sum(report["all_cluster_count"] for report in role_reports.values())),
    }
    report = {
        "unwarp_output_dir": str(unwarp_output_dir),
        "output_dir": str(output_dir),
        "roles": role_reports,
        "totals": totals,
        "overlap_radius_px": float(overlap_radius_px),
        "consensus_overlap_radius_px": float(overlap_radius_px),
        "min_consensus_sources": int(min_consensus_sources),
        "dpi": int(dpi),
        "mindtct_bin": resolved_mindtct_bin,
        "fingerflow_bin": fingerflow_bin,
        "fingerflow_model_dir": str(fingerflow_model_dir.resolve()),
        "wsl_distro": wsl_distro,
    }
    report_path = output_dir / "consensus_minutiae_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unwarp-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--roles", nargs="+", choices=DEFAULT_ROLES, default=list(DEFAULT_ROLES))
    parser.add_argument("--overlap-radius-px", type=float, default=DEFAULT_OVERLAP_RADIUS_PX)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--mindtct-bin", default="mindtct")
    parser.add_argument("--fingerflow-bin", default="fingerflow")
    parser.add_argument("--fingerflow-model-dir", type=Path, default=DEFAULT_FINGERFLOW_MODEL_DIR)
    parser.add_argument("--min-consensus-sources", type=int, default=2)
    parser.add_argument("--gaussian-amp-3src", type=float, default=1.0)
    parser.add_argument("--gaussian-amp-2src", type=float, default=0.85)
    parser.add_argument("--gaussian-sigma-3src-cells", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma-2src-cells", type=float, default=1.25)
    parser.add_argument("--single-source-ignore-sigma-cells", type=float, default=1.25)
    parser.add_argument("--wsl-distro", default=None, help="Optional WSL distro name when --mindtct-bin points to a WSL path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(
            unwarp_output_dir=args.unwarp_output_dir,
            output_dir=args.output_dir,
            roles=args.roles,
            overlap_radius_px=args.overlap_radius_px,
            dpi=args.dpi,
            mindtct_bin=args.mindtct_bin,
            fingerflow_bin=args.fingerflow_bin,
            fingerflow_model_dir=args.fingerflow_model_dir,
            min_consensus_sources=args.min_consensus_sources,
            gaussian_amp_3src=args.gaussian_amp_3src,
            gaussian_amp_2src=args.gaussian_amp_2src,
            gaussian_sigma_3src_cells=args.gaussian_sigma_3src_cells,
            gaussian_sigma_2src_cells=args.gaussian_sigma_2src_cells,
            single_source_ignore_sigma_cells=args.single_source_ignore_sigma_cells,
            wsl_distro=args.wsl_distro,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "output_dir": report["output_dir"],
                "overlap_radius_px": report["overlap_radius_px"],
                "totals": report["totals"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
