#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ground-truth-root> [output-root] [extra train args...]" >&2
  echo "Example: $0 ground_truth/DS1 weights/minutia_loss_ablation --epochs 20 --batch-size 8" >&2
  exit 2
fi

GROUND_TRUTH_ROOT="$1"
shift

OUTPUT_ROOT="${1:-weights/minutia_loss_ablation}"
if [ "$#" -gt 0 ]; then
  shift
fi

if [ "${PYTHON_BIN:-}" ]; then
  :
elif [ "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
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

run_config() {
  name="$1"
  mu_score="$2"
  mu_x="$3"
  mu_y="$4"
  mu_ori="$5"
  shift 5

  output_dir="${OUTPUT_ROOT}/${name}"
  echo "Running ${name}: score=${mu_score}, x=${mu_x}, y=${mu_y}, orientation=${mu_ori}"

  "$PYTHON_BIN" -m featurenet.models.train \
    --ground-truth-root "$GROUND_TRUTH_ROOT" \
    --output-dir "$output_dir" \
    --orientation-weight 0 \
    --ridge-weight 0 \
    --gradient-weight 0 \
    --mu-score "$mu_score" \
    --mu-x "$mu_x" \
    --mu-y "$mu_y" \
    --mu-ori "$mu_ori" \
    "$@"
}

run_config score 1 0 0 0 "$@"
run_config x 0 1 0 0 "$@"
run_config y 0 0 1 0 "$@"
run_config orientation 0 0 0 1 "$@"
