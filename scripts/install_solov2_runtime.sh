#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this installer is for Linux/WSL only."
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the target Python environment first."
  echo "This script installs SOLOv2 runtime packages into the current env in-place."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.7}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMDET_VERSION="${MMDET_VERSION:-3.3.0}"
MMCV_FIND_LINK="${MMCV_FIND_LINK:-}"
PIP_CMD=("$PYTHON_BIN" -Im pip)

echo "[1/7] Active environment"
echo "python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "prefix: $($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
echo "version: $($PYTHON_BIN -c 'import sys; print(sys.version.split()[0])')"

echo "[2/7] Upgrading base packaging tools"
"${PIP_CMD[@]}" install --upgrade pip wheel "setuptools==70.2.0" "packaging==24.2"

echo "[3/7] Installing PyTorch runtime"
"${PIP_CMD[@]}" install --upgrade \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$TORCH_INDEX_URL"

echo "[4/7] Installing compatibility pins"
"${PIP_CMD[@]}" install --upgrade \
  "numpy<2" \
  "ninja" \
  "setuptools==70.2.0" \
  "packaging==24.2" \
  "addict" \
  "Pillow" \
  "pyyaml" \
  "yapf" \
  "matplotlib" \
  "rich==13.4.2" \
  "termcolor" \
  "opencv-python<4.12"

echo "[5/7] Refreshing OpenMMLab runtime packages"
"${PIP_CMD[@]}" uninstall -y mmcv mmcv-lite mmcv-full mmengine mmdet >/dev/null 2>&1 || true
"${PIP_CMD[@]}" install --upgrade --no-deps "mmengine==$MMENGINE_VERSION"
MMCV_INSTALL_ARGS=(--upgrade --force-reinstall --no-cache-dir --no-deps --no-build-isolation)
if [[ -n "$MMCV_FIND_LINK" ]]; then
  MMCV_INSTALL_ARGS+=(-f "$MMCV_FIND_LINK")
fi
MMCV_WITH_OPS=1 FORCE_CUDA=1 "${PIP_CMD[@]}" install "${MMCV_INSTALL_ARGS[@]}" "mmcv==$MMCV_VERSION"
"${PIP_CMD[@]}" install --upgrade "numpy<2" "setuptools==70.2.0" "pycocotools" "scipy" "shapely" "terminaltables"
"${PIP_CMD[@]}" install --upgrade --no-deps "mmdet==$MMDET_VERSION"

echo "[6/7] Verifying runtime imports"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
import sysconfig
from pathlib import Path

repo_root = Path.cwd().resolve()
sys.path = [
    entry
    for entry in sys.path
    if entry not in ("", str(repo_root))
]
stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
if spec is not None and spec.loader is not None:
    copy_module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = copy_module
    spec.loader.exec_module(copy_module)

import torch
import mmcv
import mmengine
import mmdet

print(f"python: {sys.executable}")
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmengine: {mmengine.__version__}")
print(f"mmdet: {mmdet.__version__}")
PY

echo "[7/7] Repo asset check"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

repo_root = Path.cwd().resolve()
config_path = repo_root / "solov2_assets" / "solov2_distal_phalanx_infer.py"
checkpoint_path = repo_root / "weights" / "solov2_distal_phalanx_best.pth"

print(f"config exists: {config_path.exists()} -> {config_path}")
print(f"checkpoint exists: {checkpoint_path.exists()} -> {checkpoint_path}")
if not config_path.exists() or not checkpoint_path.exists():
    raise SystemExit(
        "Missing repo-local SOLOv2 assets. Make sure the repo checkout includes "
        "solov2_assets/solov2_distal_phalanx_infer.py and weights/solov2_distal_phalanx_best.pth."
    )
PY

echo
echo "SOLOv2 runtime install complete."
echo "You can now call preprocess.segment_then_clahe(...) from this active environment."
