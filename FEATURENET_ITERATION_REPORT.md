# FeatureNet Iteration Report

Date: 2026-04-26  
Scope: End-to-end experiment history for FeatureNet training/evaluation and DS123 ground-truth generation in this repo.

## 1) Initial State and Early Failures

- Ground-truth generation initially failed repeatedly on GPU with TensorFlow/cuDNN runtime mismatch.
- Early merge attempts failed when shard roots did not exist yet (`FileNotFoundError` on merge root).
- Early training runs also hit environment issues:
  - PyTorch/CUDA library mismatch (`libcusparse ... __nvJitLink...`).
  - Unsupported CUDA kernel image for RTX 5090 with incompatible PyTorch build.

## 2) Dataset Strategy and Merge Decisions

- DS1, DS2, DS3 were generated separately and merged at the bundle level (not raw image overwrite).
- Sample ID namespace collisions were handled by prefixed IDs in merged outputs.
- Ground-truth generation and merge workflows were standardized:
  - shard generation per dataset
  - shard merge per dataset
  - final DS1+DS2+DS3 merged root for training/eval
- Later, regenerated roots (`*_v2`, `*_v3`) were used to align targets with newer model expectations.

## 3) Training Loop and Monitoring Refactors

- Added configurable validation cadence and early stopping controls:
  - `--validate-every`
  - `--early-stopping`
  - `--early-stopping-metric` (`val_total`, `best_score_f1`, label metrics)
  - patience/min-delta controls
- Training records/history now capture validation cadence, monitor values, patience state, and stop reason.
- `best.pt` and `last.pt` behavior preserved with monitored-metric alignment.

## 4) Ground-Truth Target Evolution

- Ground-truth bundles were extended to include continuous supervision fields:
  - `minutia_x_offset`, `minutia_y_offset` in `[0,1)` within output cell
  - `minutia_orientation_vec = [cos(theta), sin(theta)]`
- Legacy fields were retained for compatibility:
  - `minutia_x`, `minutia_y`, `minutia_orientation` bins
- Loader fallback was added:
  - If offsets are missing, recover from `minutiae.json` with same cell ownership/tie-break policy.
- Important observed issue in latest eval:
  - large number of samples skipped for "minutiae present but zero active minutia target cells"
  - this materially changes effective sample pool and comparability across runs

## 5) Loss and Objective Iterations

### 5.1 Baseline Behavior

- Minutia detection quality was poor under earlier setup: very low precision, near-flat threshold response, low F1.

### 5.2 Score-Head Supervision Changes (M1)

- Implemented focal BCE with class imbalance handling.
- Introduced dynamic `pos_weight` logic and later strengthened adaptive behavior.
- Switched M1 supervision to stronger negative/background learning in valid fingerprint regions.
- Added hard-negative mining controls (enable/disable, ratio/min/fraction, hardest-by-loss selection).

### 5.3 X/Y Supervision Changes

- Migrated x/y from discrete bins to continuous offset regression.
- M2/M3 use masked SmoothL1 on positive minutia cells.
- Gaussian soft-target CE phase for ordered-bin x/y was also tested in earlier stage.

### 5.4 Orientation Supervision Changes (M4)

- Replaced 360-bin minutia-orientation classification with continuous vector regression:
  - model predicts 2 channels (`cos`, `sin`)
  - M4 loss uses vector-space regression on minutia-positive cells
  - eval recovers degrees with `atan2`
- Added `orientation_mae_deg` metric.
- `label_accuracy.minutia_orientation` is interpreted as within-15deg accuracy in the current evaluator path.

## 6) Architecture Iterations

### 6.1 Raw-Logit Head Refactor

- Prediction heads moved to raw terminal logits where required to align with BCE/CE assumptions.

### 6.2 Continuous X/Y Head Refactor

- X/Y heads changed from 8-bin logits to 1-channel continuous offset logits.
- Decode remained anchored to selected score cell:
  - `x = col + sigmoid(x_logit)`
  - `y = row + sigmoid(y_logit)`

### 6.3 Patch-Aware X/Y Refactor (Latest Architecture Step)

- Replaced old `/4 -> stride-2 compress -> /8` x/y descriptor path with:
  - fused `/4` localization map
  - bottom/right pad-to-even policy
  - `pixel_unshuffle(..., 2)` to preserve 2x2 `/4` local geometry per `/8` cell
  - patch refinement blocks on regrouped `/8` descriptor
  - crop back to exact score-grid `/8` shape for strict alignment
- Score and orientation paths remained unchanged.

## 7) Evaluator Refactors

- Replaced old overlap-style detection accounting with paper-style one-to-one matching:
  - distance threshold `< 8 px`
  - angular threshold `< 15deg`
  - one-to-one bipartite matching
  - threshold sweep and NMS retained
- Added split reporting: `all`, `front`, `side`.
- Added regression diagnostics:
  - `minutia_x_mae_cell`, `minutia_y_mae_cell`
  - `minutia_x_mae_px`, `minutia_y_mae_px`
  - `orientation_mae_deg`

## 8) Iteration-by-Iteration Results (Available Logs)

Notes:
- `Regime=legacy-overlap` and `Regime=paper-match` are not directly comparable.
- Some runs used different effective sample pools due skipping/filtering.
- `N/A` means the metric was not present in the captured log for that run.

| Iter | Regime | Main change set | best_score_f1 | best thr | Precision@best | Recall@best | x acc | y acc | ori acc | ori mae (deg) | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| I0 | legacy-overlap | Early baseline eval | 0.043954 | 0.10 | 0.022471 | 1.000000 | 0.1182 | 0.1204 | 0.0889 | N/A | Flat threshold behavior, very low precision |
| I1 | legacy-overlap | Tuned baseline eval checkpoint | 0.045173 | 0.90 | 0.023109 | 0.999244 | 0.1182 | 0.1184 | 0.0736 | N/A | Slight F1 bump only |
| I2 | paper-match | First paper-style evaluator run | 0.032628 | 0.10 | 0.022009 | 0.063048 | 0.1326 | 0.1394 | 0.0130 | N/A | Strict matching exposed low recall/precision |
| I3 | paper-match | Stronger score supervision + hard negatives (early) | 0.061491 | 0.10 | 0.031844 | 0.891745 | 0.1326 | 0.1170 | 0.0981 | N/A | Big recall jump, precision still low |
| I4 | paper-match | Continuous orientation phase, stabilized run | 0.426165 | 0.60 | 0.469603 | 0.390082 | 0.1848 | 0.1817 | 0.7746 | 18.71 | Major detection jump |
| I5 | paper-match | Follow-up tuned run | 0.452410 | 0.60 | 0.527599 | 0.395978 | 0.2046 | 0.1979 | 0.8022 | 17.80 | Better precision and label accuracy |
| I6 | paper-match | Next tuned run (peak logged F1 in chat) | 0.484694 | 0.60 | 0.511152 | 0.460841 | 0.1990 | 0.1920 | 0.8199 | 17.42 | Best logged F1 in this thread |
| I7 | paper-match | Another strong run before latest regressions | 0.476653 | 0.60 | 0.558125 | 0.415936 | 0.1893 | 0.1793 | 0.8657 | 15.09 | Best logged orientation quality |
| I8 | paper-match | Later run with changed data/target mix | 0.442435 | 0.60 | 0.495778 | 0.399456 | 0.3701 | 0.3379 | 0.7744 | 18.54 | X/Y acc increased sharply, F1 dropped |
| I9 | paper-match | Latest shared eval on `DS123_merged_v3` | 0.420428 | 0.60 | 0.497547 | 0.364008 | 0.4300 | 0.4276 | 0.7436 | 27.29 | Effective sample pool reduced by skip filters |

Observed pattern:

- X/Y metrics improved significantly in later runs.
- Orientation quality was sensitive to objective/data changes.
- Detection F1 peaked mid-iteration (I6) and regressed in later runs with changed data composition.
- Direct run-to-run comparison is sometimes non-apples-to-apples because evaluator semantics and filtered sample sets changed.

## 9) Current Snapshot (Latest Shared Eval)

- Ground truth root: `ground_truth/DS123_merged_v3`
- Effective sample count after loader filtering: `3539`
- Validation sample count: `725`
- Best detection threshold: `0.60`
- `best_score_f1`: `0.4204`
- Precision/Recall at best threshold:
  - precision: `0.4975`
  - recall: `0.3640`
- Label accuracy:
  - x: `0.4300`
  - y: `0.4276`
  - orientation: `0.7436`
- MAE:
  - x cell: `0.1858`
  - y cell: `0.1891`
  - x px: `1.4917`
  - y px: `1.5171`
  - orientation deg: `27.29`

## 10) Main Lessons So Far

- Objective alignment matters more than raw training loss magnitude.
- Score head required stronger negative supervision and hard-negative pressure to improve precision/F1.
- Paper-style detection evaluation exposed quality gaps hidden by earlier metric style.
- Continuous orientation and continuous x/y formulations enabled better geometric diagnostics.
- Ground-truth consistency is critical; regenerated targets are necessary to avoid fallback/recovery artifacts in training/eval.

## 11) Immediate Action Items

- Fully regenerate DS1/DS2/DS3 targets in current format (offset + orientation vector) to eliminate on-the-fly recovery.
- Re-merge and run a clean apples-to-apples evaluation set (same filtering rules across compared runs).
- Re-run the strongest known configuration and compare against current patch-aware x/y path.
- Investigate root cause of "minutiae present but zero active target cells" in skipped samples.

