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

DATASET_ROOT="${DATASET_ROOT:-archive}"
GROUND_TRUTH_ROOT="${GROUND_TRUTH_ROOT:-ground_truth}"
MERGED_ROOT="${MERGED_ROOT:-${GROUND_TRUTH_ROOT}/DS123_merged_v2}"

DS1_ROOT="${DS1_ROOT:-${GROUND_TRUTH_ROOT}/DS1_v2}"
DS2_ROOT="${DS2_ROOT:-${GROUND_TRUTH_ROOT}/DS2_v2}"
DS3_ROOT="${DS3_ROOT:-${GROUND_TRUTH_ROOT}/DS3_v2}"

DEFAULT_GENERATOR_FLAGS="--execution-target kaggle --gpu-only --cpu-workers 10 --prefetch-samples 10 --fingerflow-backend local"
GENERATOR_FLAGS="${GENERATOR_FLAGS:-${DEFAULT_GENERATOR_FLAGS}}"

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

cleanup_merge_root() {
  target_root="$1"

  if [ ! -d "${target_root}" ]; then
    return 0
  fi

  if [ -z "$(ls -A "${target_root}" 2>/dev/null)" ]; then
    return 0
  fi

  case "${target_root}" in
    "${GROUND_TRUTH_ROOT}"/*)
      ;;
    *)
      echo "Refusing to clear merge root outside ${GROUND_TRUTH_ROOT}: ${target_root}" >&2
      exit 1
      ;;
  esac

  echo "Clearing existing merged output root: ${target_root}"
  rm -rf "${target_root}"
}

echo "Using Python: ${PYTHON_BIN}"
echo "Base dataset root: ${DATASET_ROOT}"
echo "Base ground truth root: ${GROUND_TRUTH_ROOT}"
echo "Merged output root: ${MERGED_ROOT}"
echo "Generator flags: ${GENERATOR_FLAGS}"

run_generate "DS1" "${DS1_ROOT}"
run_generate "DS2" "${DS2_ROOT}"
run_generate "DS3" "${DS3_ROOT}"

echo "Merging generated roots into ${MERGED_ROOT}"
cleanup_merge_root "${MERGED_ROOT}"
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
