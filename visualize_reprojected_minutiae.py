from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import sysconfig
from pathlib import Path
from typing import Any


def ensure_stdlib_copy_module() -> None:
    existing = sys.modules.get("copy")
    repo_root = Path(__file__).resolve().parent
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None:
            return
        if Path(existing_path).resolve().parent != repo_root:
            return
        del sys.modules["copy"]

    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

import cv2
import numpy as np

import generate_ground_truth as gt


def _require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return path


def _load_reconstruction_inputs(reconstruction_dir: Path) -> dict[str, Any]:
    reconstruction_dir = reconstruction_dir.resolve()
    required = {
        "center_unwarped_image": reconstruction_dir / "center_unwarped.png",
        "canonical_minutiae": reconstruction_dir / "canonical_unwarped_minutiae.json",
        "center_unwarp_maps": reconstruction_dir / "center_unwarp_maps.npz",
        "reconstruction_maps": reconstruction_dir / "reconstruction_maps.npz",
        "support_mask": reconstruction_dir / "support_mask.png",
        "depth_front": reconstruction_dir / "depth_front.npy",
        "depth_left": reconstruction_dir / "depth_left.npy",
        "depth_right": reconstruction_dir / "depth_right.npy",
    }
    for path in required.values():
        _require_path(path)

    unwarped_image = gt._require_grayscale(required["center_unwarped_image"])
    minutiae = json.loads(required["canonical_minutiae"].read_text(encoding="utf-8"))
    if not isinstance(minutiae, list):
        raise ValueError(f"expected minutiae list in {required['canonical_minutiae']}")

    support_mask = gt._to_uint8_mask(gt._require_grayscale(required["support_mask"]))
    reconstruction_maps = gt._load_npz_arrays(required["reconstruction_maps"])
    unwarp_maps = gt._load_npz_arrays(required["center_unwarp_maps"])
    depth_front = np.load(required["depth_front"]).astype(np.float32)
    depth_left = np.load(required["depth_left"]).astype(np.float32)
    depth_right = np.load(required["depth_right"]).astype(np.float32)

    meta_path = reconstruction_dir / "meta.json"
    acquisition_id = reconstruction_dir.name
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            acquisition_id = str(meta.get("acquisition_id", acquisition_id))
        except Exception:
            pass

    return {
        "reconstruction_dir": reconstruction_dir,
        "acquisition_id": acquisition_id,
        "unwarped_image": unwarped_image,
        "canonical_minutiae": gt._standardize_minutiae(minutiae, source="canonical_unwarped"),
        "unwarp_maps": unwarp_maps,
        "reconstruction_maps": reconstruction_maps,
        "support_mask": support_mask,
        "depth_front": depth_front,
        "depth_left": depth_left,
        "depth_right": depth_right,
        "required_paths": {key: str(path.resolve()) for key, path in required.items()},
    }


def _draw_minutiae_overlay(image: np.ndarray, minutiae: list[dict[str, Any]], marker_scale: float) -> np.ndarray:
    base = cv2.cvtColor(gt._to_uint8_image(image), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    segment_length = max(float(marker_scale), 1.0)
    for minutia in minutiae:
        x = int(round(float(minutia["x"])))
        y = int(round(float(minutia["y"])))
        theta = float(minutia["theta"])
        if x < 0 or y < 0 or x >= overlay.shape[1] or y >= overlay.shape[0]:
            continue
        dx = int(round(math.cos(theta) * segment_length))
        dy = int(round(math.sin(theta) * segment_length))
        cv2.circle(overlay, (x, y), 3, (30, 245, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (x, y), (x + dx, y + dy), (255, 210, 40), 1, cv2.LINE_AA)
    return overlay


def _lift_front_source_to_3d(reconstruction_maps: dict[str, np.ndarray], x_front: float, y_front: float) -> tuple[float, float, float] | None:
    support_mask = reconstruction_maps["support_mask"]
    x_relative = gt._sample_2d_bilinear(reconstruction_maps["x_relative"], x_front, y_front, valid_mask=support_mask)
    depth_front = gt._sample_2d_bilinear(reconstruction_maps["depth_front"], x_front, y_front, valid_mask=support_mask)
    if x_relative is None or depth_front is None:
        return None
    return float(x_relative), float(-y_front), float(depth_front)


def _build_reprojected_minutiae_geometry(
    canonical_minutiae: list[dict[str, Any]],
    unwarp_maps: dict[str, np.ndarray],
    reconstruction_maps: dict[str, np.ndarray],
    marker_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points_3d: list[dict[str, Any]] = []
    dropped = 0
    half_step = max(float(marker_scale) * 0.5, 2.0)

    for minutia in canonical_minutiae:
        x = float(minutia["x"])
        y = float(minutia["y"])
        theta = float(minutia["theta"])
        source_point = gt._map_unwarped_to_front_source(unwarp_maps, x, y)
        if source_point is None:
            dropped += 1
            continue
        center_3d = _lift_front_source_to_3d(reconstruction_maps, source_point[0], source_point[1])
        if center_3d is None:
            dropped += 1
            continue

        forward_src = gt._map_unwarped_to_front_source(
            unwarp_maps,
            x + (math.cos(theta) * half_step),
            y + (math.sin(theta) * half_step),
        )
        backward_src = gt._map_unwarped_to_front_source(
            unwarp_maps,
            x - (math.cos(theta) * half_step),
            y - (math.sin(theta) * half_step),
        )
        forward_3d = _lift_front_source_to_3d(reconstruction_maps, forward_src[0], forward_src[1]) if forward_src is not None else None
        backward_3d = _lift_front_source_to_3d(reconstruction_maps, backward_src[0], backward_src[1]) if backward_src is not None else None

        if forward_3d is not None and backward_3d is not None:
            direction_start = backward_3d
            direction_end = forward_3d
        elif forward_3d is not None:
            direction_start = center_3d
            direction_end = forward_3d
        elif backward_3d is not None:
            direction_start = backward_3d
            direction_end = center_3d
        else:
            direction_start = center_3d
            direction_end = center_3d

        points_3d.append(
            {
                "x": float(center_3d[0]),
                "y": float(center_3d[1]),
                "z": float(center_3d[2]),
                "theta": theta,
                "score": minutia.get("score"),
                "type": minutia.get("type"),
                "direction_start": [float(direction_start[0]), float(direction_start[1]), float(direction_start[2])],
                "direction_end": [float(direction_end[0]), float(direction_end[1]), float(direction_end[2])],
            }
        )

    summary = {
        "canonical_minutiae_count": len(canonical_minutiae),
        "reprojected_minutiae_count": len(points_3d),
        "dropped_minutiae_count": dropped,
    }
    return points_3d, summary


def _write_overlay_summary(path: Path, acquisition_id: str, details: dict[str, Any], source_paths: dict[str, str]) -> None:
    payload = {
        "acquisition_id": acquisition_id,
        **details,
        "source_artifacts": source_paths,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_3d_png(
    output_path: Path,
    front_points: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    minutiae_points: list[dict[str, Any]],
    point_size: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(9.0, 6.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    branches = {
        "front": (front_points, np.array([242, 96, 58], dtype=np.float32) / 255.0),
        "left": (left_points, np.array([44, 181, 232], dtype=np.float32) / 255.0),
        "right": (right_points, np.array([195, 235, 70], dtype=np.float32) / 255.0),
    }
    for points, color in branches.values():
        if points.size == 0:
            continue
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], s=1.0, alpha=0.22, depthshade=False)

    if minutiae_points:
        mins = np.array([[item["x"], item["y"], item["z"]] for item in minutiae_points], dtype=np.float32)
        ax.scatter(mins[:, 0], mins[:, 1], mins[:, 2], c=[[1.0, 0.95, 0.2]], s=max(point_size, 1.0) * 8.0, depthshade=False)
        for item in minutiae_points:
            start = np.asarray(item["direction_start"], dtype=np.float32)
            end = np.asarray(item["direction_end"], dtype=np.float32)
            ax.plot(
                [float(start[0]), float(end[0])],
                [float(start[1]), float(end[1])],
                [float(start[2]), float(end[2])],
                color=(0.27, 0.98, 0.84),
                linewidth=1.0,
                alpha=0.95,
            )

    all_points = [points for points in (front_points, left_points, right_points) if points.size > 0]
    if all_points:
        cloud = np.concatenate(all_points, axis=0).astype(np.float32)
        mins = cloud.min(axis=0)
        maxs = cloud.max(axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        try:
            spans = np.maximum(maxs - mins, 1.0)
            ax.set_box_aspect((float(spans[0]), float(spans[1]), float(spans[2])))
        except Exception:
            pass
    ax.set_title("Reprojected Minutiae on Shared 3D Branch Model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("depth")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_3d_html(
    output_path: Path,
    acquisition_id: str,
    front_points: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    minutiae_points: list[dict[str, Any]],
    details: dict[str, Any],
    source_paths: dict[str, str],
    point_size: float,
) -> None:
    branch_points = {
        "front": front_points.astype(np.float32),
        "left": left_points.astype(np.float32),
        "right": right_points.astype(np.float32),
    }
    branch_colors = {
        "front": [242, 96, 58],
        "left": [44, 181, 232],
        "right": [195, 235, 70],
    }
    all_cloud = [points for points in branch_points.values() if points.size > 0]
    if all_cloud:
        all_points = np.concatenate(all_cloud, axis=0).astype(np.float32)
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)
    if minutiae_points:
        minutiae_xyz = np.asarray([[item["x"], item["y"], item["z"]] for item in minutiae_points], dtype=np.float32)
        if all_points.size > 0:
            all_points = np.concatenate([all_points, minutiae_xyz], axis=0)
        else:
            all_points = minutiae_xyz

    bounds = {
        "x": [float(all_points[:, 0].min(initial=0.0)), float(all_points[:, 0].max(initial=1.0))],
        "y": [float(all_points[:, 1].min(initial=0.0)), float(all_points[:, 1].max(initial=1.0))],
        "z": [float(all_points[:, 2].min(initial=0.0)), float(all_points[:, 2].max(initial=1.0))],
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reprojected Minutiae on Shared 3D Branches</title>
  <style>
    body {{ margin: 0; background: #0b0e14; color: #f3f4f6; font: 14px/1.4 system-ui, sans-serif; }}
    .wrap {{ display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 20px; background: #131722; border-right: 1px solid #232838; overflow: auto; }}
    .canvas-wrap {{ position: relative; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100vh; background:
      radial-gradient(circle at top, rgba(52,73,94,.35), transparent 38%),
      linear-gradient(180deg, #0c1018 0%, #06080d 100%); }}
    h1 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; color: #cbd5e1; }}
    code {{ color: #fde68a; word-break: break-all; }}
    .controls {{ margin: 16px 0; padding: 14px; border: 1px solid #232838; border-radius: 12px; background: rgba(255,255,255,0.02); }}
    .controls label {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
    button {{ margin-right: 8px; margin-top: 8px; padding: 8px 10px; border-radius: 10px; border: 1px solid #2c3446; background: #1a2130; color: #e5e7eb; cursor: pointer; }}
    .hint {{ color: #94a3b8; font-size: 13px; }}
    .stats {{ margin-top: 18px; padding-top: 18px; border-top: 1px solid #232838; }}
    ul {{ padding-left: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="sidebar">
      <h1>Reprojected Minutiae 3D View</h1>
      <p>Acquisition: <code>{acquisition_id}</code></p>
      <p>This shows the shared reconstructed branch clouds with the canonical unwarped minutiae reprojected onto the front-surface 3D geometry.</p>
      <div class="controls">
        <label><input id="toggle-front" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["front"][0]}, {branch_colors["front"][1]}, {branch_colors["front"][2]})"></span>Front cloud</label>
        <label><input id="toggle-left" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["left"][0]}, {branch_colors["left"][1]}, {branch_colors["left"][2]})"></span>Left cloud</label>
        <label><input id="toggle-right" type="checkbox" checked><span class="swatch" style="background: rgb({branch_colors["right"][0]}, {branch_colors["right"][1]}, {branch_colors["right"][2]})"></span>Right cloud</label>
        <label><input id="toggle-minutiae" type="checkbox" checked><span class="swatch" style="background: rgb(255, 240, 50)"></span>Minutiae</label>
        <div>
          <button id="all-on" type="button">All On</button>
          <button id="reset-view" type="button">Reset View</button>
        </div>
      </div>
      <p class="hint">Controls: drag to rotate, mouse wheel to zoom.</p>
      <div class="stats">
        <p><strong>Counts</strong></p>
        <ul>
          <li>Canonical minutiae: {details["canonical_minutiae_count"]}</li>
          <li>Reprojected minutiae: {details["reprojected_minutiae_count"]}</li>
          <li>Dropped minutiae: {details["dropped_minutiae_count"]}</li>
        </ul>
        <p><strong>Source Artifacts</strong></p>
        <ul>
          <li><code>{source_paths["center_unwarped_image"]}</code></li>
          <li><code>{source_paths["canonical_minutiae"]}</code></li>
          <li><code>{source_paths["center_unwarp_maps"]}</code></li>
          <li><code>{source_paths["reconstruction_maps"]}</code></li>
        </ul>
      </div>
    </div>
    <div class="canvas-wrap">
      <canvas id="surface"></canvas>
    </div>
  </div>
  <script>
    const branchData = {{
      front: {{ points: {json.dumps(branch_points["front"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["front"])} }},
      left: {{ points: {json.dumps(branch_points["left"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["left"])} }},
      right: {{ points: {json.dumps(branch_points["right"].tolist(), separators=(",", ":"))}, color: {json.dumps(branch_colors["right"])} }}
    }};
    const minutiae = {json.dumps(minutiae_points, separators=(",", ":"))};
    const bounds = {json.dumps(bounds, separators=(",", ":"))};
    const pointSize = {float(max(point_size, 1.0))};
    const toggles = {{
      front: document.getElementById('toggle-front'),
      left: document.getElementById('toggle-left'),
      right: document.getElementById('toggle-right'),
      minutiae: document.getElementById('toggle-minutiae')
    }};
    const canvas = document.getElementById('surface');
    const ctx = canvas.getContext('2d');
    let width = 0, height = 0;
    let yaw = -0.9, pitch = 0.65, zoom = 1.5;
    let dragging = false, lastX = 0, lastY = 0;
    const center = [
      (bounds.x[0] + bounds.x[1]) / 2,
      (bounds.y[0] + bounds.y[1]) / 2,
      (bounds.z[0] + bounds.z[1]) / 2
    ];
    const span = Math.max(bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0], 1);

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function project(px, py, pz) {{
      const x0 = (px - center[0]) / span;
      const y0 = (py - center[1]) / span;
      const z0 = (pz - center[2]) / span;
      const cosy = Math.cos(yaw), siny = Math.sin(yaw);
      const cosp = Math.cos(pitch), sinp = Math.sin(pitch);
      const x1 = x0 * cosy - z0 * siny;
      const z1 = x0 * siny + z0 * cosy;
      const y1 = y0 * cosp - z1 * sinp;
      const z2 = y0 * sinp + z1 * cosp;
      const camera = 3.1 / zoom;
      const perspective = camera / Math.max(camera - z2, 0.2);
      return {{
        x: width * 0.5 + x1 * perspective * width * 0.88,
        y: height * 0.55 - y1 * perspective * width * 0.88,
        z: z2,
        size: Math.max(1.0, perspective * pointSize)
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      const projected = [];
      for (const name of ['front', 'left', 'right']) {{
        if (!toggles[name].checked) continue;
        const branch = branchData[name];
        for (const point of branch.points) {{
          const p = project(point[0], point[1], point[2]);
          projected.push({{ kind: 'cloud', z: p.z, x: p.x, y: p.y, size: p.size, color: branch.color }});
        }}
      }}
      if (toggles.minutiae.checked) {{
        for (const item of minutiae) {{
          const p = project(item.x, item.y, item.z);
          const s = project(item.direction_start[0], item.direction_start[1], item.direction_start[2]);
          const e = project(item.direction_end[0], item.direction_end[1], item.direction_end[2]);
          projected.push({{
            kind: 'minutia',
            z: p.z,
            x: p.x,
            y: p.y,
            size: Math.max(p.size * 1.35, 2.2),
            lineStart: s,
            lineEnd: e
          }});
        }}
      }}
      projected.sort((a, b) => a.z - b.z);
      for (const item of projected) {{
        if (item.kind === 'cloud') {{
          const color = item.color;
          ctx.fillStyle = `rgba(${{color[0]}},${{color[1]}},${{color[2]}},0.28)`;
          ctx.beginPath();
          ctx.arc(item.x, item.y, item.size, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }}
        ctx.strokeStyle = 'rgba(70, 255, 215, 0.95)';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(item.lineStart.x, item.lineStart.y);
        ctx.lineTo(item.lineEnd.x, item.lineEnd.y);
        ctx.stroke();
        ctx.fillStyle = 'rgb(255, 240, 50)';
        ctx.beginPath();
        ctx.arc(item.x, item.y, item.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(12, 14, 20, 0.9)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }}
    }}

    for (const toggle of Object.values(toggles)) {{
      toggle.addEventListener('change', draw);
    }}
    document.getElementById('all-on').addEventListener('click', () => {{
      toggles.front.checked = true;
      toggles.left.checked = true;
      toggles.right.checked = true;
      toggles.minutiae.checked = true;
      draw();
    }});
    document.getElementById('reset-view').addEventListener('click', () => {{
      yaw = -0.9;
      pitch = 0.65;
      zoom = 1.5;
      draw();
    }});
    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01));
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('pointerleave', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom = Math.max(0.45, Math.min(4.0, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }}, {{ passive: false }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize canonical unwarped minutiae and their 3D reprojection.")
    parser.add_argument("--reconstruction-dir", type=Path, required=True, help="Path to an existing reconstructions/<acquisition_id> directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write visualization outputs (default: reconstruction dir).")
    parser.add_argument("--marker-scale", type=float, default=10.0, help="Marker/orientation segment length in pixels for 2D and source-space steps for 3D.")
    parser.add_argument("--point-size", type=float, default=2.4, help="Relative 3D minutia marker size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = _load_reconstruction_inputs(args.reconstruction_dir)
    reconstruction_dir = inputs["reconstruction_dir"]
    output_dir = args.output_dir.resolve() if args.output_dir is not None else reconstruction_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_minutiae = inputs["canonical_minutiae"]
    overlay = _draw_minutiae_overlay(inputs["unwarped_image"], canonical_minutiae, marker_scale=float(args.marker_scale))
    overlay_png = output_dir / "canonical_unwarped_minutiae_overlay.png"
    cv2.imwrite(str(overlay_png), overlay)

    minutiae_3d, details = _build_reprojected_minutiae_geometry(
        canonical_minutiae,
        inputs["unwarp_maps"],
        inputs["reconstruction_maps"],
        marker_scale=float(args.marker_scale),
    )
    overlay_json = output_dir / "canonical_unwarped_minutiae_overlay.json"
    _write_overlay_summary(overlay_json, inputs["acquisition_id"], details, inputs["required_paths"])

    front_points, left_points, right_points = gt._compute_branch_point_clouds(
        x_relative=inputs["reconstruction_maps"]["x_relative"],
        z_front=inputs["depth_front"],
        stitched_left=inputs["reconstruction_maps"]["stitched_left"],
        stitched_right=inputs["reconstruction_maps"]["stitched_right"],
        support_mask=inputs["support_mask"],
    )

    html_path = output_dir / "reprojected_minutiae_all_branches.html"
    png_path = output_dir / "reprojected_minutiae_all_branches.png"
    _write_3d_html(
        html_path,
        inputs["acquisition_id"],
        front_points,
        left_points,
        right_points,
        minutiae_3d,
        details,
        inputs["required_paths"],
        point_size=float(args.point_size),
    )
    _write_3d_png(
        png_path,
        front_points,
        left_points,
        right_points,
        minutiae_3d,
        point_size=float(args.point_size),
    )

    print(f"Saved 2D overlay PNG: {overlay_png.resolve()}")
    print(f"Saved 2D overlay summary JSON: {overlay_json.resolve()}")
    print(f"Saved 3D HTML: {html_path.resolve()}")
    print(f"Saved 3D PNG: {png_path.resolve()}")
    print(f"Canonical minutiae: {details['canonical_minutiae_count']}")
    print(f"Reprojected minutiae: {details['reprojected_minutiae_count']}")
    print(f"Dropped minutiae: {details['dropped_minutiae_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
