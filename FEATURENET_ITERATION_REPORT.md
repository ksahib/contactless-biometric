# FeatureNet Iteration Report

Date: 2026-04-26  
Scope: Detailed documentation of model-facing iterations (data targets, losses, heads, evaluator, training objective/monitoring) and observed results.

## 1) Experiment Tracking Conventions

- Detection metric of record: `best_score_f1` (from evaluator threshold sweep).
- Detection summaries are from paper-style matching runs unless marked otherwise.
- Label metrics:
  - `minutia_x` / `minutia_y` accuracy: tolerance-based offset accuracy.
  - `minutia_orientation` accuracy: within-15deg angular accuracy.
- Some iterations are not perfectly apples-to-apples due to target/regime/dataset updates; this is noted inline.

## 2) Iteration Chronology (What Changed + Results)

## I0 - Initial baseline (legacy evaluator regime)

### What was tried
- Baseline training/eval before paper-style matching refactor.
- Early objective centered around loss minimization and initial checkpoint selection.

### Result (available)
- `best_score_f1 = 0.043954` (legacy metric path baseline reference).
- Precision extremely low and threshold behavior mostly flat.

---

## I1 - Validation cadence + early stopping controls

### What was tried
- Added configurable validation cadence and early-stopping behavior:
  - `--validate-every`
  - `--early-stopping`
  - metric choice (`val_total`, `best_score_f1`, label metrics)
  - patience/min-delta tracking
- Checkpointing aligned with monitored metric.

### Result (available)
- Enabled controlled F1-first experiments and reproducible stop behavior.
- No standalone metric delta isolated for this iteration alone (infrastructure iteration).

---

## I2 - DS1/DS2/DS3 merged training corpus strategy

### What was tried
- Merged DS1+DS2+DS3 at generated bundle level with dataset-prefix sample IDs.
- Standardized train/val split flow on merged bundle root.

### Result (available)
- Created a stable combined dataset path for all subsequent model iterations.
- No single isolated metric snapshot tied only to this change.

---

## I3 - F1-first objective alignment

### What was tried
- Switched experiment focus from `val_total` to detection quality (`best_score_f1`) for model selection and early stopping in F1-centric runs.

### Result (available)
- Made improvements visible that were previously masked by loss-only monitoring.
- No isolated number without accompanying model/loss changes.

---

## I4 - Minutia score loss reweighting (first major score-head intervention)

### What was tried
- Introduced class-balancing and focal behavior for M1 (score supervision).
- Rebalanced minutia-component weights (`mu_score`, `mu_x`, `mu_y`, `mu_ori`) to reduce score dominance and support localization heads.

### Result (available)
- Improved calibration behavior versus earlier flat-threshold baseline.
- Representative progression in this phase moved from ~`0.04395` to ~`0.04517`, then into stronger gains in later score-focused iterations.

---

## I5 - Raw-logit prediction head refactor

### What was tried
- Refactored prediction-head terminals to true logits (no final activation-style distortion in terminal prediction conv).
- Architecture treated as new-run branch.

### Result (available)
- Improved x/y label metrics in that branch, but detection F1 remained limited before further objective fixes.
- Representative run in this phase (paper-style evaluator): `best_score_f1 = 0.032628`.

---

## I6 - Paper-style minutia evaluator implementation

### What was tried
- Replaced old overlap-like detection metric with paper-style one-to-one matching:
  - distance `< 8 px`
  - angle `< 15deg`
  - one-to-one bipartite matching
  - threshold sweep + NMS
  - split reporting (`all/front/side`)

### Result (available)
- Detection metric became stricter and more realistic.
- Early paper-style reference values:
  - `best_score_f1 = 0.032628` (first strict run)
  - later this pipeline captured improvement to `0.061491` after score-objective updates.

---

## I7 - Full-negative score supervision + hard-negative mining

### What was tried
- Changed score supervision to learn positives and background negatives strongly.
- Added hard-negative mining controls and stronger imbalance handling.
- Kept x/y/orientation objective structure intact in this step.

### Result (available)
- Major detection jump:
  - from early paper-style values (`0.0326` / `0.0615`) to strong runs around:
    - `0.426165`
    - `0.452410`
    - `0.484694` (best logged in this thread)

---

## I8 - Continuous minutia orientation (cos/sin) replacing 360-bin orientation head

### What was tried
- Replaced minutia orientation classification with continuous 2D vector prediction (`cos`, `sin`).
- Loss moved to vector regression; eval decodes via `atan2`.
- Added `orientation_mae_deg`.

### Result (available)
- Orientation quality improved significantly in strong runs:
  - Orientation accuracy up to `0.8657`
  - Orientation MAE down to `15.09 deg`
- These orientation improvements coexisted with high-F1 runs in the `0.45-0.48` range.

---

## I9 - Continuous x/y offsets (head + targets + loss)

### What was tried
- Migrated x/y from 8-bin classification heads to 1-channel continuous offset logits.
- Supervision shifted to `minutia_x_offset` / `minutia_y_offset` with masked SmoothL1.
- Ground-truth generation writes offsets; loader supports compatibility recovery when offsets absent.

### Result (available)
- X/Y localization metrics improved materially in later runs:
  - x/y accuracy climbed from ~`0.18-0.20` range to ~`0.37-0.43` range in later reports.
- Tradeoff observed in some runs: detection F1 and/or orientation MAE could regress depending on data composition and target quality.

---

## I10 - Adaptive hard-negative/imbalance strengthening

### What was tried
- Increased score-loss pressure with stronger hard-negative settings and adaptive balancing:
  - higher hard-negative ratio/min
  - stronger positive-weight cap strategy

### Result (available)
- Produced one of the largest precision/F1 gains in paper-style detection.
- Representative strong checkpoint behavior:
  - `best_score_f1` in the `0.45-0.48` band
  - better precision at practical thresholds.

---

## I11 - Gaussian soft-target CE phase for ordered x/y bins (intermediate experiment)

### What was tried
- Tested Gaussian soft-target CE for x/y ordered bins (`sigma=1.0`) in the bin-based phase.
- Intended to smooth class boundaries and improve sub-bin localization learning.

### Result (available)
- Helped stabilize x/y behavior in that phase, but was superseded by continuous offset formulation.
- Not the final x/y design in current branch.

---

## I12 - Patch-aware x/y localization path (`pixel_unshuffle` 2x2 preservation)

### What was tried
- Replaced early `/4 -> stride-2 compress -> /8` x/y descriptor collapse with patch-aware regrouping:
  - fused `/4` localization map
  - pad-to-even (bottom/right)
  - `pixel_unshuffle(..., 2)` to preserve per-cell 2x2 `/4` structure at `/8`
  - patch refinement at `/8`
  - crop to exact score-grid shape for alignment
- Score and orientation paths unchanged.

### Result (available)
- Further x/y quality increase observed in later reports.
- Depending on dataset/target version, detection F1 ranged from strong to regressed:
  - strong prior references around `0.476653`
  - later run with changed data composition at `0.420428`

---

## I13 - Ground-truth regeneration alignment work (`*_v2`, `*_v3`)

### What was tried
- Regenerating ground truth to match current model target contract directly:
  - continuous x/y offsets
  - orientation vectors
- Goal: remove dependence on on-the-fly compatibility reconstruction.

### Result (available)
- Latest shared eval on `DS123_merged_v3`:
  - `best_score_f1 = 0.420428`
  - best threshold `0.60`
  - precision `0.497547`, recall `0.364008`
  - x acc `0.42996`, y acc `0.42758`, orientation acc `0.74363`
  - x MAE cell `0.18584`, y MAE cell `0.18906`
  - orientation MAE `27.29 deg`
- Important context from same run:
  - large sample skipping due to zero active minutia support affected effective pool and comparability.

## 3) Consolidated Results Table (Available Metrics)

| Iter | Main change focus | best_score_f1 | best thr | Precision@best | Recall@best | x acc | y acc | ori acc | ori mae (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0 | Initial baseline (legacy regime) | 0.043954 | 0.10 | 0.022471 | 1.000000 | 0.1182 | 0.1204 | 0.0889 | N/A |
| I5/I6 early | Raw-logit + first paper-style strict eval | 0.032628 | 0.10 | 0.022009 | 0.063048 | 0.1326 | 0.1394 | 0.0130 | N/A |
| I7 early | Full-negative score + hard negatives (early) | 0.061491 | 0.10 | 0.031844 | 0.891745 | 0.1326 | 0.1170 | 0.0981 | N/A |
| I8/I9 | Continuous orientation/x-y era strong run | 0.426165 | 0.60 | 0.469603 | 0.390082 | 0.1848 | 0.1817 | 0.7746 | 18.71 |
| I8/I9 | Tuned follow-up | 0.452410 | 0.60 | 0.527599 | 0.395978 | 0.2046 | 0.1979 | 0.8022 | 17.80 |
| I10 peak | Strongest logged F1 run | 0.484694 | 0.60 | 0.511152 | 0.460841 | 0.1990 | 0.1920 | 0.8199 | 17.42 |
| I10 alt | High-precision/strong orientation run | 0.476653 | 0.60 | 0.558125 | 0.415936 | 0.1893 | 0.1793 | 0.8657 | 15.09 |
| I12/I13 | Later data/target mix run | 0.442435 | 0.60 | 0.495778 | 0.399456 | 0.3701 | 0.3379 | 0.7744 | 18.54 |
| I13 latest | `DS123_merged_v3` latest shared eval | 0.420428 | 0.60 | 0.497547 | 0.364008 | 0.4300 | 0.4276 | 0.7436 | 27.29 |

## 4) Current Understanding

- Biggest improvement lever so far was score-head objective alignment (full-negative learning + hard negatives).
- Continuous orientation and continuous offsets gave much better geometric interpretability and strong gains in several runs.
- Patch-aware x/y representation improved localization behavior, but global detection quality remained sensitive to data-target consistency.
- The most recent regression signals are likely confounded by target-support/skipping behavior in latest dataset variant.

## 5) Next Iteration Logging Template (Recommended)

For each next run, log this exact block for strict comparability:

- Data root + exact sample_count/val_sample_count after loader filtering
- Training config hash (loss weights, hard-neg params, monitor metric, seeds)
- `best_score_f1`, best threshold, precision/recall at best threshold
- x/y/orientation accuracy
- x/y cell MAE and orientation MAE
- skip statistics (`missing gradient`, `zero active minutia target cells`)

