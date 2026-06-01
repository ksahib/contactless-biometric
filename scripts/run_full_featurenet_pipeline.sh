#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the target Python virtualenv or Conda environment first." >&2
  echo "scripts/install_solov2_runtime.sh installs packages into the active environment." >&2
  exit 1
fi

resolve_mindtct_bin() {
  if [[ -n "${MINDTCT_BIN:-}" ]]; then
    if [[ "$MINDTCT_BIN" == */* ]]; then
      printf '%s\n' "$MINDTCT_BIN"
      return
    fi
    command -v "$MINDTCT_BIN" || true
    return
  fi

  if command -v mindtct >/dev/null 2>&1; then
    command -v mindtct
    return
  fi

  if [[ -x "$HOME/opt/nbis/bin/mindtct" ]]; then
    printf '%s\n' "$HOME/opt/nbis/bin/mindtct"
    return
  fi
}

torch_cuda_available() {
  python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}

MINDTCT_BIN="$(resolve_mindtct_bin)"
if [[ -z "$MINDTCT_BIN" || ! -x "$MINDTCT_BIN" ]]; then
  echo "ERROR: NBIS mindtct executable was not found." >&2
  echo "Install it with: scripts/install_nbis_mindtct_ubuntu.sh" >&2
  echo "Or set MINDTCT_BIN=/full/path/to/mindtct before running this script." >&2
  exit 1
fi
export PATH="$(dirname "$MINDTCT_BIN"):$PATH"

if [[ -z "${SOLOV2_DEVICE:-}" ]]; then
  if torch_cuda_available; then
    SOLOV2_DEVICE="cuda:0"
  else
    SOLOV2_DEVICE="cpu"
  fi
fi

if [[ -z "${TRAIN_DEVICE:-}" ]]; then
  if torch_cuda_available; then
    TRAIN_DEVICE="cuda"
  else
    TRAIN_DEVICE="cpu"
  fi
fi

CPU_WORKERS="${CPU_WORKERS:-24}"
PREFETCH_SAMPLES="${PREFETCH_SAMPLES:-24}"
SOLOV2_SCORE_THR="${SOLOV2_SCORE_THR:-0.15}"
START_DATASET="${START_DATASET:-DS1}"
GT_SKIP_EXISTING="${GT_SKIP_EXISTING:-1}"
RESET_MERGE_OUTPUT="${RESET_MERGE_OUTPUT:-0}"
MERGE_LINK_MODE="${MERGE_LINK_MODE:-symlink}"
DS1_OUTPUT_ROOT="${DS1_OUTPUT_ROOT:-/media/milab-5/430b66f7-67f1-44f8-a7bd-1299b3b1c6df/DS1}"
DS2_OUTPUT_ROOT="${DS2_OUTPUT_ROOT:-$REPO_ROOT/ground_truth/source_roots/DS2}"
DS3_OUTPUT_ROOT="${DS3_OUTPUT_ROOT:-$REPO_ROOT/ground_truth/source_roots/DS3}"
MERGED_OUTPUT_ROOT="${MERGED_OUTPUT_ROOT:-ground_truth/merged_DS123}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-runs/featurenet_v4}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-1}"
TRAIN_AMP_DTYPE="${TRAIN_AMP_DTYPE:-bf16}"
GT_FAILED_DATASETS=()
GT_COMMON_ARGS=()
if [[ "$GT_SKIP_EXISTING" == "1" || "$GT_SKIP_EXISTING" == "true" || "$GT_SKIP_EXISTING" == "yes" ]]; then
  GT_COMMON_ARGS+=(--skip-existing)
fi

echo "Using NBIS mindtct: $MINDTCT_BIN"
echo "Using SOLOv2 device: $SOLOV2_DEVICE"
echo "Using FeatureNet train device: $TRAIN_DEVICE"
echo "Starting from dataset: $START_DATASET"
echo "DS1 output root: $DS1_OUTPUT_ROOT"
echo "DS2 output root: $DS2_OUTPUT_ROOT"
echo "DS3 output root: $DS3_OUTPUT_ROOT"
echo "Merged output root: $MERGED_OUTPUT_ROOT"
echo "Merge link mode: $MERGE_LINK_MODE"
echo "Skip existing bundles: $GT_SKIP_EXISTING"
echo "FeatureNet output dir: $TRAIN_OUTPUT_DIR"
echo "FeatureNet batch size: $TRAIN_BATCH_SIZE"
echo "FeatureNet grad accumulation steps: $TRAIN_GRAD_ACCUM_STEPS"
echo "FeatureNet AMP dtype: $TRAIN_AMP_DTYPE"

dataset_enabled() {
  case "$START_DATASET:$1" in
    DS1:DS1|DS1:DS2|DS1:DS3|DS1:merge|DS1:train) return 0 ;;
    DS2:DS2|DS2:DS3|DS2:merge|DS2:train) return 0 ;;
    DS3:DS3|DS3:merge|DS3:train) return 0 ;;
    merge:merge|merge:train|MERGE:merge|MERGE:train) return 0 ;;
    train:train|TRAIN:train) return 0 ;;
    *)
      if [[ "$START_DATASET" == "DS1" || "$START_DATASET" == "DS2" || "$START_DATASET" == "DS3" || "$START_DATASET" == "merge" || "$START_DATASET" == "MERGE" || "$START_DATASET" == "train" || "$START_DATASET" == "TRAIN" ]]; then
        return 1
      fi
      echo "ERROR: START_DATASET must be one of DS1, DS2, DS3, merge, train; got $START_DATASET" >&2
      exit 1
      ;;
  esac
}

run_ground_truth() {
  local label="$1"
  shift
  echo "Generating $label ground truth"
  set +e
  python generate_ground_truth.py "$@"
  local status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    GT_FAILED_DATASETS+=("$label:$status")
    echo "WARNING: $label ground truth exited with status $status; continuing with generated usable bundles." >&2
  fi
}

require_completed_root() {
  local label="$1"
  local root="$2"
  if [[ ! -f "$root/summary.json" ]]; then
    echo "ERROR: $label output root is not complete: missing $root/summary.json" >&2
    echo "Run or resume $label generation before merging." >&2
    exit 1
  fi
  if [[ ! -f "$root/manifest.json" ]]; then
    echo "ERROR: $label output root is not complete: missing $root/manifest.json" >&2
    echo "Run or resume $label generation before merging." >&2
    exit 1
  fi
}

# echo "[1/7] Preparing installer scripts"
# chmod +x scripts/install_nbis_mindtct_ubuntu.sh scripts/install_solov2_runtime.sh

# echo "[2/7] Installing NBIS mindtct"
# ./scripts/install_nbis_mindtct_ubuntu.sh
# export PATH="$HOME/opt/nbis/bin:$PATH"

# echo "[3/7] Installing SOLOv2 runtime"
# ./scripts/install_solov2_runtime.sh

if dataset_enabled DS1; then
  echo "[4/7] Generating DS1 ground truth"
  run_ground_truth DS1 "${GT_COMMON_ARGS[@]}" --dataset-root /media/milab-5/82002d9e-66a9-4739-925b-e2b789ec5641/archive/DS1 --output-root "$DS1_OUTPUT_ROOT" --mask-extractor solov2 --solov2-device "$SOLOV2_DEVICE" --solov2-score-thr "$SOLOV2_SCORE_THR" --cpu-workers "$CPU_WORKERS" --prefetch-samples "$PREFETCH_SAMPLES" --mindtct-bin "$MINDTCT_BIN"
else
  echo "[4/7] Skipping DS1 ground truth"
fi

if dataset_enabled DS2; then
  echo "[5/7] Generating DS2 ground truth"
  run_ground_truth DS2 "${GT_COMMON_ARGS[@]}" --dataset-root /media/milab-5/82002d9e-66a9-4739-925b-e2b789ec5641/archive/DS2 --output-root "$DS2_OUTPUT_ROOT" --mask-extractor solov2 --solov2-device "$SOLOV2_DEVICE" --solov2-score-thr "$SOLOV2_SCORE_THR" --cpu-workers "$CPU_WORKERS" --prefetch-samples "$PREFETCH_SAMPLES" --mindtct-bin "$MINDTCT_BIN"
else
  echo "[5/7] Skipping DS2 ground truth"
fi

if dataset_enabled DS3; then
  echo "[6/7] Generating DS3 ground truth"
  run_ground_truth DS3 "${GT_COMMON_ARGS[@]}" --dataset-root /media/milab-5/82002d9e-66a9-4739-925b-e2b789ec5641/archive/DS3 --output-root "$DS3_OUTPUT_ROOT" --mask-extractor solov2 --solov2-device "$SOLOV2_DEVICE" --solov2-score-thr "$SOLOV2_SCORE_THR" --cpu-workers "$CPU_WORKERS" --prefetch-samples "$PREFETCH_SAMPLES" --mindtct-bin "$MINDTCT_BIN"
else
  echo "[6/7] Skipping DS3 ground truth"
fi

if dataset_enabled merge; then
  echo "[6/7] Merging DS1/DS2/DS3 outputs"
  require_completed_root DS1 "$DS1_OUTPUT_ROOT"
  require_completed_root DS2 "$DS2_OUTPUT_ROOT"
  require_completed_root DS3 "$DS3_OUTPUT_ROOT"
  if [[ -e "$MERGED_OUTPUT_ROOT" ]]; then
    if [[ "$RESET_MERGE_OUTPUT" == "1" || "$RESET_MERGE_OUTPUT" == "true" || "$RESET_MERGE_OUTPUT" == "yes" ]]; then
      echo "Removing existing merge output root: $MERGED_OUTPUT_ROOT"
      rm -rf -- "$MERGED_OUTPUT_ROOT"
    else
      echo "ERROR: merge output root already exists: $MERGED_OUTPUT_ROOT" >&2
      echo "Set RESET_MERGE_OUTPUT=1 to delete it before merging." >&2
      exit 1
    fi
  fi
  python generate_ground_truth.py --output-root "$MERGED_OUTPUT_ROOT" --merge-link-mode "$MERGE_LINK_MODE" --merge-generated-root DS1="$DS1_OUTPUT_ROOT" --merge-generated-root DS2="$DS2_OUTPUT_ROOT" --merge-generated-root DS3="$DS3_OUTPUT_ROOT"
else
  echo "[6/7] Skipping merge"
fi

if dataset_enabled train; then
  echo "[7/7] Training FeatureNet"
  python -m featurenet.models.train --ground-truth-root "$MERGED_OUTPUT_ROOT" --output-dir "$TRAIN_OUTPUT_DIR" --device "$TRAIN_DEVICE" --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH_SIZE" --grad-accum-steps "$TRAIN_GRAD_ACCUM_STEPS" --max-grad-norm 5.0 --num-workers "$TRAIN_NUM_WORKERS" --amp --channels-last --early-stopping --early-stopping-metric best_score_f1 --early-stopping-patience 15 --mu-score 80 --mu-x 40 --mu-y 40 --mu-ori 30 --amp-dtype "$TRAIN_AMP_DTYPE"
else
  echo "[7/7] Skipping training"
fi

if [[ ${#GT_FAILED_DATASETS[@]} -gt 0 ]]; then
  echo "Ground-truth generation had nonzero exits but pipeline continued: ${GT_FAILED_DATASETS[*]}" >&2
fi
