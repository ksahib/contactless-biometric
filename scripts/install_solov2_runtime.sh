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
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.7}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMDET_VERSION="${MMDET_VERSION:-3.3.0}"

echo "[1/7] Active environment"
echo "python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "prefix: $($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
echo "version: $($PYTHON_BIN -c 'import sys; print(sys.version.split()[0])')"

echo "[2/7] Upgrading base packaging tools"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel packaging

echo "[3/7] Installing PyTorch runtime"
"$PYTHON_BIN" -m pip install --upgrade \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$TORCH_INDEX_URL"

echo "[4/7] Installing compatibility pins"
"$PYTHON_BIN" -m pip install --upgrade "numpy<2" "openmim>=0.3.9"

echo "[5/7] Refreshing OpenMMLab runtime packages"
"$PYTHON_BIN" -m pip uninstall -y mmcv mmcv-lite mmcv-full mmengine mmdet >/dev/null 2>&1 || true
"$PYTHON_BIN" -m pip install --upgrade "mmengine==$MMENGINE_VERSION"
"$PYTHON_BIN" -m mim install "mmcv==$MMCV_VERSION"
"$PYTHON_BIN" -m pip install --upgrade "mmdet==$MMDET_VERSION"

echo "[6/7] Verifying runtime imports"
"$PYTHON_BIN" - <<'PY'
import sys
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
