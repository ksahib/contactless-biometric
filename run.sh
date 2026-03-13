#!/usr/bin/env sh
set -eu

if [ "${PYTHON_BIN:-}" ]; then
  :
elif [ "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -x "${HOME:-}/.venvs/contactless-biometric/bin/python" ]; then
  PYTHON_BIN="${HOME}/.venvs/contactless-biometric/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

IMAGE_A="${1:-sahibind4.jpeg}"
IMAGE_B="${2:-sahibind5.jpeg}"
METHOD="${3:-${METHOD:-LSA-R}}"

GPU_LIB_PATHS="$("$PYTHON_BIN" - <<'PY'
import site
from pathlib import Path

paths = []
for base in site.getsitepackages():
    paths.extend(sorted(str(path) for path in Path(base).glob('nvidia/*/lib')))
print(':'.join(paths))
PY
)"

export FINGERFLOW_ALLOW_CPU="${FINGERFLOW_ALLOW_CPU:-0}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${GPU_LIB_PATHS:+:$GPU_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Running fresh full pipeline:"
echo "  Python: $PYTHON_BIN"
echo "  A: $IMAGE_A"
echo "  B: $IMAGE_B"
echo "  Method: $METHOD"

exec "$PYTHON_BIN" main.py match "$IMAGE_A" "$IMAGE_B" --method "$METHOD"
