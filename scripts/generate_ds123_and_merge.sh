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

DATASET_ROOT="${DATASET_ROOT:-dataset}"
GROUND_TRUTH_ROOT="${GROUND_TRUTH_ROOT:-ground_truth}"
MERGED_ROOT="${MERGED_ROOT:-${GROUND_TRUTH_ROOT}/DS123_merged}"

DS1_ROOT="${DS1_ROOT:-${GROUND_TRUTH_ROOT}/DS1}"
DS2_ROOT="${DS2_ROOT:-${GROUND_TRUTH_ROOT}/DS2}"
DS3_ROOT="${DS3_ROOT:-${GROUND_TRUTH_ROOT}/DS3}"

GENERATOR_FLAGS="${GENERATOR_FLAGS:-}"

for dataset_name in DS1 DS2 DS3; do
  if [ ! -d "${DATASET_ROOT}/${dataset_name}" ]; then
    echo "Missing dataset root: ${DATASET_ROOT}/${dataset_name}" >&2
    exit 1
  fi
done

run_generate() {
  dataset_name="$1"
  output_root="$2"

  echo "Generating ground truth for ${dataset_name}"
  echo "  dataset root: ${DATASET_ROOT}/${dataset_name}"
  echo "  output root:  ${output_root}"

  # shellcheck disable=SC2086
  exec_cmd="\"${PYTHON_BIN}\" generate_ground_truth.py --dataset-root \"${DATASET_ROOT}/${dataset_name}\" --output-root \"${output_root}\" ${GENERATOR_FLAGS}"
  # shellcheck disable=SC2086
  eval "${exec_cmd}"
}

echo "Using Python: ${PYTHON_BIN}"
echo "Base dataset root: ${DATASET_ROOT}"
echo "Base ground truth root: ${GROUND_TRUTH_ROOT}"
echo "Merged output root: ${MERGED_ROOT}"
if [ -n "${GENERATOR_FLAGS}" ]; then
  echo "Extra generator flags: ${GENERATOR_FLAGS}"
fi

run_generate "DS1" "${DS1_ROOT}"
run_generate "DS2" "${DS2_ROOT}"
run_generate "DS3" "${DS3_ROOT}"

echo "Merging generated roots into ${MERGED_ROOT}"
"${PYTHON_BIN}" generate_ground_truth.py \
  --merge-generated-root "ds1=${DS1_ROOT}" \
  --merge-generated-root "ds2=${DS2_ROOT}" \
  --merge-generated-root "ds3=${DS3_ROOT}" \
  --output-root "${MERGED_ROOT}"

echo "Done."
echo "  DS1: ${DS1_ROOT}"
echo "  DS2: ${DS2_ROOT}"
echo "  DS3: ${DS3_ROOT}"
echo "  Merged: ${MERGED_ROOT}"
