# FeatureNet Implementation Reference (Code-Accurate)

This document describes the current implementation in this repository, including:
- model architecture and heads
- exact model inputs and preprocessing
- ground-truth bundle format
- loss functions and weighting
- training loop and early stopping behavior
- evaluation (paper-style minutia matching)

Scope is based on:
- `featurenet/models/blocks.py`
- `featurenet/models/feature_extractor.py`
- `featurenet/models/losses.py`
- `featurenet/models/train.py`
- `featurenet/models/evaluate.py`
- `generate_ground_truth.py`

## 1) End-to-End Data Flow

1. Raw contactless RGB images are converted into generated bundles by `generate_ground_truth.py`.
2. Each bundle stores:
   - preprocessed model input (`masked_image.png`, `mask.png`)
   - per-task supervision (`featurenet_targets.npz`)
   - metadata (`meta.json`)
3. Training (`python -m featurenet.models.train`) loads these bundle files, builds tensors, and trains `FeatureExtractor` using `FeatureNetLoss`.
4. Validation/evaluation uses the same bundle format, with:
   - loss metrics (`total`, `orientation`, `ridge`, `gradient`, `minutia`, `m1..m4`)
   - paper-style minutia detection metrics (distance + angle one-to-one matching)

## 2) Raw Dataset and Generated Ground-Truth Layout

## 2.1 Raw Dataset Assumptions

`generate_ground_truth.py` expects subject folders under `dataset_root`:
- each subject directory name is numeric (`1`, `2`, ...)
- raw contactless views in `subject/raw/`
- raw filename pattern: `{subject}_{finger}_{acquisition}_{view}.jpg`

Manifest sample IDs are created as:
- `s{subject:02d}_f{finger:02d}_a{acquisition:02d}_v{raw_view_index:02d}`

Identity fields:
- `subject_index`: 0-based subject order in sorted directory listing
- `finger_class_id = subject_index * 10 + (finger_id - 1)`
- `raw_view_index` is expected to be 0/1/2 for front/side split logic

## 2.2 Bundle Files Per Sample

`generate_ground_truth.py` writes `ground_truth/.../samples/<sample_id>/` containing:
- `meta.json`
- `mask.png`
- `masked_image.png`
- `orientation.npy`
- `ridge_period.npy`
- `gradient_visualization.npy` (debug gradient from Sobel on masked grayscale)
- `minutiae.json`
- `featurenet_targets.npz`
- plus additional preprocessing debug images (`raw_input.png`, `preprocess_*`, etc.)

Training/eval loader (`load_bundle_samples`) requires:
- `meta.json`
- `masked_image.png`
- `mask.png`
- `featurenet_targets.npz`

And keeps only samples where:
- `meta.raw_view_index in {0,1,2}`
- `gradient` exists in `featurenet_targets.npz` (unless strict mode disabled; then sample is skipped)

## 3) Input Preprocessing (Before Model Sees Data)

## 3.1 Generation-Time Preprocessing (`generate_ground_truth.py`)

For each raw BGR image:

1. **Mask estimation**
- primary: `rembg` (`remove(..., only_mask=True, post_process_mask=True)`)
- fallback: Otsu threshold-based foreground extraction
- morphological close/open then largest connected component
- foreground area checks:
  - initial mask ratio >= `0.03`
  - pose-normalized mask ratio >= `0.02`

2. **Brightness normalization**
- CLAHE pass 1: `clipLimit=4.0`, `tileGridSize=(4,4)`
- CLAHE pass 2: `clipLimit=1.0`, `tileGridSize=(10,10)`
- only inside foreground bbox; outside mask set to zero

3. **Pose normalization**
- estimate dominant finger axis from largest contour (`cv2.minAreaRect`)
- rotate to vertical alignment (`_estimate_pose_rotation`)
- rotate grayscale with bilinear interpolation
- rotate mask with nearest-neighbor

4. **Ridge-frequency normalization**
- estimate central ridge spacing via FFT on local patch
- rescale image/mask so estimated spacing maps to target `10` pixels
- scale is clamped to `[0.5, 2.5]`; if near 1 (`abs(scale-1)<0.02`), no resize

5. **Classical fingerprint maps**
- orientation: `pyfing.orientation_field_estimation(..., method="SNFOE")`
- ridge period: `pyfing.frequency_estimation(..., method="SNFFE")`
- reconstruction gradient target: loaded from multiview reconstruction labels when available (`gradient_front/left/right`)
- visualization gradient: Sobel on normalized masked grayscale (debug only)

6. **Masked image construction**
- `masked_image = preprocessed_gray`
- pixels outside mask set to 0

7. **Minutiae source**
- primary: FingerFlow extractor (local or WSL backend)
- fallback: `pyfing.minutiae_extraction`
- standardized minutia record fields: `x`, `y`, `theta`, `score`, `type`, `source`

## 3.2 Ground-Truth Target Tensor Generation (`featurenet_targets.npz`)

Output grid shape:
- `output_height = max(1, input_height // 8)`
- `output_width = max(1, input_width // 8)`

Keys in `featurenet_targets.npz`:
- `orientation`: `float32`, shape `[180, Ho, Wo]`
  - one-hot bins from resized orientation field in `[0, pi)` domain
- `ridge_period`: `float32`, shape `[1, Ho, Wo]`
  - resized, masked, normalized by max
- `gradient`: `float32`, shape `[2, Ho, Wo]` (if reconstruction gradient exists)
- `minutia_score`: `float32`, shape `[1, Ho, Wo]`
  - binary occupancy map of minutia-positive cells
- `minutia_valid_mask`: `float32`, shape `[1, Ho, Wo]`
  - binary mask marking cells with minutia labels
- `minutia_x`: `int64`, shape `[Ho, Wo]`
  - 8-bin x-subcell label (`0..7`) for active minutia cells
- `minutia_y`: `int64`, shape `[Ho, Wo]`
  - 8-bin y-subcell label (`0..7`) for active minutia cells
- `minutia_x_offset`: `float32`, shape `[1, Ho, Wo]`
  - continuous x offset inside the selected `/8` output cell in `[0, 1)`
- `minutia_y_offset`: `float32`, shape `[1, Ho, Wo]`
  - continuous y offset inside the selected `/8` output cell in `[0, 1)`
- `minutia_orientation`: `int64`, shape `[Ho, Wo]`
  - legacy 360-bin orientation label (`0..359`) for active minutia cells
- `minutia_orientation_vec`: `float32`, shape `[2, Ho, Wo]`
  - continuous orientation target vector `[cos(theta), sin(theta)]` for active minutia cells
- `output_mask`: `float32`, shape `[1, Ho, Wo]`
  - downsampled binary mask

Rasterization details:
- minutiae are assigned to output cells by floor mapping from input pixel coordinates
- if multiple minutiae land in one cell, tie-break prefers higher score, then center proximity
- subcell bins use local coordinate in cell: `floor(local * 8)`
- continuous offsets use the same local coordinate before binning (`local_x`, `local_y`)
- orientation bin: `floor(theta * 360 / (2*pi))`
- orientation vector: `[cos(theta), sin(theta)]`

## 3.3 Train/Eval-Time Input Tensor Build (`train.py`)

`build_input_tensor(masked_image, mask)`:
- reads grayscale image and mask
- image converted to `float32`; if max > 1 then divide by 255
- mask converted to binary float (`>0 -> 1.0`)
- returns:
  - `input_tensor`: shape `[2, H, W]` = concatenation of `[masked_image, mask]`
  - `mask_tensor`: shape `[1, H, W]`

In model step:
- `image = inputs[:, :1]`
- `mask = inputs[:, 1:2]`
- `outputs = model(image, mask=mask)`

So model receives:
- `x`: single-channel grayscale tensor `[B,1,H,W]`
- `mask`: binary mask tensor `[B,1,H,W]`

Inside `FeatureExtractor.forward`:
- it multiplies `image = x * mask` again
- then concatenates `[image, mask]` into 2 channels

## 4) Model Architecture (`feature_extractor.py`)

## 4.1 Building Block

`ConvBlock(in_ch, out_ch, k=3, s=1, p=1)`:
- `Conv2d`
- `BatchNorm2d`
- `ReLU(inplace=True)`

## 4.2 Backbone and Branches

There are two main feature paths from the 2-channel input:

### Branch 1 (orientation/ridge/gradient trunk)
- `ConvBlock(2,64,3,1,1)`
- `ConvBlock(64,64,3,1,1)`
- `MaxPool2d(2,2)`   -> `/2`
- `ConvBlock(64,128,3,1,1)`
- `ConvBlock(128,128,3,1,1)`
- `MaxPool2d(2,2)`   -> `/4`
- `ConvBlock(128,256,3,1,1)`
- `ConvBlock(256,256,3,1,1)`
- `MaxPool2d(2,2)`   -> `/8`

Two stems from branch1 output:
- `branch_stem_ridge`: 3 x `ConvBlock(256,256,3,1,1)`
- `branch_stem_orient`: 3 x `ConvBlock(256,256,3,1,1)`

Head projections from stems:
- `ridge_conv`: `Conv2d(256,1,k=1)`
- `gradient_conv`:
  - `ConvBlock(256,256,3,1,1)` x3
  - `Conv2d(256,2,k=1)`
- `orientation_conv`: `Conv2d(256,180,k=1)`

### Branch 2 (minutiae trunk)
- `ConvBlock(2,64,k=9,p=4)`
- `MaxPool2d(2,2)`   -> `/2`
- `ConvBlock(64,128,k=5,p=2)`
- `MaxPool2d(2,2)`   -> `/4`
- `ConvBlock(128,256,k=3,p=1)`
- `MaxPool2d(2,2)`   -> `/8`

For x/y localization, branch2 now exposes both:
- shallow `branch2_feat_4x` (`/4`)
- deep `branch2_feat_8x` (`/8`)

X/Y fusion path:
- upsample `branch2_feat_8x` to `/4` using bilinear interpolation
- concatenate with `branch2_feat_4x` (`384` channels total)
- two `ConvBlock` refinement layers at `/4`
- stride-2 `ConvBlock` to compress back to `/8`
- x/y heads read this fused `/8` descriptor

Minutia heads:
- `minutiae_score_head`:
  - `ConvBlock(256,256,k=1,p=0)`
  - `Conv2d(256,1,k=1)`  (raw logit)
- `minutia_head_x`:
  - `ConvBlock(256,256,k=1,p=0)`
  - `Conv2d(256,1,k=1)`  (raw x-offset logits)
- `minutia_head_y`:
  - `ConvBlock(256,256,k=1,p=0)`
  - `Conv2d(256,1,k=1)`  (raw y-offset logits)
- `minuiae_orient_head`:
  - input is concatenation of `[branch2_features, orient_stem_features]` => 512 channels
  - `ConvBlock(512,256,k=1,p=0)`
  - `Conv2d(256,2,k=1)` (raw orientation vector channels: cos/sin)

All heads output at the same `/8` spatial grid.

## 4.3 Model Output Dictionary

`forward(...)` returns:
- `orientation`: `[B,180,Ho,Wo]`
- `ridge_period`: `[B,1,Ho,Wo]`
- `gradient`: `[B,2,Ho,Wo]`
- `minutia_orientation`: `[B,2,Ho,Wo]`
- `minutia_score`: `[B,1,Ho,Wo]`
- `minutia_x`: `[B,1,Ho,Wo]` (raw x-offset logits)
- `minutia_y`: `[B,1,Ho,Wo]` (raw y-offset logits)

## 5) Losses (`losses.py`)

`FeatureNetLoss` outputs:
- `total`, `orientation`, `ridge`, `gradient`, `minutia`, `m1`, `m2`, `m3`, `m4`

Defaults:
- `alpha=1.0` (orientation coherence term)
- `beta=60.0` (ridge smoothness)
- `gamma=300.0` (gradient smoothness)
- `sigma=0.5` (gradient weighting)
- minutia weights: `mu_score=120`, `mu_x=20`, `mu_y=20`, `mu_ori=5`
- M1 focal params:
  - `m1_focal_gamma=2.0`
  - `m1_pos_weight_max=100.0`
  - hard negatives:
    - `m1_hard_neg_enable=True`
    - `m1_hard_neg_ratio=20.0`
    - `m1_hard_neg_min=2000`
    - `m1_hard_neg_fraction=0.05`

## 5.1 Mask Resolution Rules

- `score_mask` for M1:
  - from `targets["mask"]`
  - if empty, replaced by all-ones mask
- `minutia_mask` for M2/M3/M4:
  - from `targets["minutia_valid_mask"]` if present, else `score_mask`
  - multiplied by `score_mask`
  - if empty, M2/M3/M4 are effectively zero (positive-only supervision)

## 5.2 Orientation Loss (`OrientationLoss`)

Inputs:
- prediction logits over 180 bins
- one-hot orientation target
- mask

Loss = masked binwise BCE-like term + coherence regularizer:
- `pred_prob = softmax(pred, dim=1)`
- CE-style term:
  - `-(target*log(pred_prob+eps) + (1-target)*log(1-pred_prob+eps))`
  - summed over bins, masked average over pixels
- coherence term:
  - compute orientation vector field from `cos(2*theta), sin(2*theta)` weighted by predicted distribution
  - smooth with fixed `3x3` average kernel
  - magnitude term `sqrt(dcos^2 + dsin^2 + eps)`
  - masked mean of `(magnitude - 1.0)`
- final: `L_ori = ce + alpha * coh`

## 5.3 Ridge Period Loss (`RidgePeriodLoss`)

- masked MSE between prediction and target
- plus gradient smoothness:
  - squared finite differences in x and y on prediction
  - `grad_loss = mean(dx^2) + mean(dy^2)`
- final: `L_ridge = masked_mse + beta * grad_loss`

## 5.4 Gradient Loss (`GradientLoss`)

- per-pixel weighted vector MSE with mask:
  - target gradient magnitude `mag = ||target||`
  - weight `w = exp(-mag / sigma) * mask`
- plus masked smoothness on prediction gradients (`dx`, `dy`)
- final:
  - `L_grad = weighted_mse + gamma * (dx_loss + dy_loss)`

## 5.5 Minutia Loss Components

### M1 (minutia score, binary with focal + hard negatives)

1. Full score supervision region = `score_mask` (positives + negatives).
2. Dynamic class balancing:
- `pos = count(target_score>0.5 within valid score mask)`
- `neg = count(target_score<=0.5 within valid score mask)`
- `pos_weight = clamp(sqrt(neg/(pos+eps)), 1.0, m1_pos_weight_max)`
3. Base map:
   - `bce_map = BCEWithLogits(logits, target, pos_weight, reduction='none')`
   - `pt = exp(-bce_map)`
   - `focal_map = (1-pt)^m1_focal_gamma * bce_map`
4. Hard-negative mining (per sample):
- always keep all positive cells
- among negatives, keep hardest by focal value
- `k = max(int(pos_count * m1_hard_neg_ratio), m1_hard_neg_min, int(valid_negative_count * m1_hard_neg_fraction))`
- cap `k` by available negatives
- if selection becomes empty, fallback to full valid mask
5. Normalize by selected-cell count.

### M2/M3/M4 (x/y/orientation)

- `M2` and `M3` are masked SmoothL1 regression on continuous offsets:
  - predictions: `sigmoid(outputs["minutia_x"])`, `sigmoid(outputs["minutia_y"])`
  - targets: `minutia_x_offset`, `minutia_y_offset`
  - each loss map is masked by `minutia_mask` and normalized by `minutia_mask.sum()`
  - x/y offsets are validated on active cells to be in `[0,1]`
- `M4` is continuous orientation regression:
  - prediction is normalized per-cell to unit vector
  - target vector priority:
    - `minutia_orientation_vec` if present
    - else derived from legacy `minutia_orientation` bin centers
  - per-cell loss: `(cos_pred - cos_gt)^2 + (sin_pred - sin_gt)^2`
- each term is masked by `minutia_mask` and normalized by `minutia_mask.sum()`.

### Combined Minutia + Total

- `L_minu = mu_score*L_m1 + mu_x*L_m2 + mu_y*L_m3 + mu_ori*L_m4`
- `L_total = L_ori + L_ridge + L_grad + L_minu`

## 6) Training Pipeline (`train.py`)

## 6.1 Data Loader and Collation

- `FeatureNetDataset` returns `(input_tensor, targets)`
- variable-size tensors are padded per batch by `_collate_batch`
  - both inputs and each target key are padded to max H/W in that batch
  - padding uses zeros (`torch.nn.functional.pad`)

Target key typing:
- float keys: `mask`, `orientation`, `ridge_period`, `gradient`, `minutia_score`, `minutia_valid_mask`, `minutia_x_offset`, `minutia_y_offset`, `minutia_orientation_vec`
- long keys: `minutia_x`, `minutia_y`, `minutia_orientation`
- extra keys (`raw_view_index`, `input_shape_hw`, `output_shape_hw`) are kept as tensors

## 6.2 Pseudo Targets + Explicit Targets Merge

Each step builds pseudo targets from input image/mask and output shapes:
- orientation one-hot from Sobel angle
- ridge from local avg pooling
- minutia score from response heuristic
- x/y bins from grid coordinates
- x/y offsets from bin centers (`(bin + 0.5)/8`) as pseudo fallback
- minutia orientation from Sobel angle (legacy bin pseudo-target)

Then:
- `merged = pseudo_targets`
- overwrite with explicit bundle targets (`merged.update(explicit_targets)`)
- explicit `gradient` is required; missing gradient raises `KeyError`
- if `minutia_x_offset` / `minutia_y_offset` are missing in a bundle, loader recovers exact offsets on the fly from `minutiae.json` using the same cell assignment/tie-break policy

So, in practice, generated bundle targets drive supervision.

## 6.3 Optimization

- optimizer: `Adam(lr, betas=(0.9,0.999), weight_decay=0.0)`
- optional:
  - AMP autocast fp16 on CUDA
  - GradScaler
  - channels-last memory format
  - `torch.compile` (CUDA only, if available)
  - gradient accumulation (`grad_accum_steps`)
  - cudnn benchmark

## 6.4 Split Logic

`split_samples(samples, val_fraction, seed)`:
- grouped by `finger_class_id` (fallback `sample_id` if missing)
- shuffle group keys by seed
- select validation groups by `round(len(groups)*val_fraction)` with minimum 1 group
- prevents same finger class leaking into train+val

## 6.5 Validation Cadence + Early Stopping

Validation runs when:
- `epoch % validate_every == 0` OR `epoch == final_epoch`

Monitored metric options:
- `val_total` (mode `min`)
- `best_score_f1` (mode `max`)
- `minutia_x_accuracy` (mode `max`)
- `minutia_y_accuracy` (mode `max`)
- `minutia_orientation_accuracy` (mode `max`, now angular within-15deg)

Improvement test:
- `min` mode: `current < best - min_delta`
- `max` mode: `current > best + min_delta`

Patience behavior:
- increments only on validation checks
- non-validation epochs do not change patience counter

No-validation safety:
- if no validation set and early stopping requested, it is auto-disabled with `disabled_reason="no_validation_data"`

Checkpoints:
- `best.pt` saved when monitored metric improves
- `last.pt` always saved after loop ends

History:
- per-epoch JSON records include train loss, optional val loss, optional extended val metrics, monitor state, early-stopping state, and `validation_ran`
- full summary written to `output_dir/history.json`

## 6.6 Train CLI (all args)

Required:
- `--ground-truth-root`
- `--output-dir`

Core:
- `--epochs` (default `5`)
- `--batch-size` (default `4`)
- `--lr` (default `1e-3`)
- `--num-workers` (default `max(1,min(cpu_count,4))`)
- `--device` (default auto: cuda if available else cpu)

Performance/runtime:
- `--amp/--no-amp` (default `True`)
- `--channels-last` (default `False`)
- `--compile` (default `False`)
- `--persistent-workers/--no-persistent-workers` (default `True`)
- `--prefetch-factor` (default `2`)
- `--pin-memory/--no-pin-memory` (default `True`)
- `--grad-accum-steps` (default `1`)
- `--cudnn-benchmark/--no-cudnn-benchmark` (default `True`)

Validation/early stopping:
- `--val-fraction` (default `0.2`)
- `--validate-every` (default `1`, must be >=1)
- `--early-stopping` (default disabled)
- `--early-stopping-metric` (choices above, default `val_total`)
- `--early-stopping-patience` (default `5`, must be >=1)
- `--early-stopping-min-delta` (default `1e-4`, must be >=0)

Loss tuning:
- `--mu-score` (default `120.0`, must be >0)
- `--mu-x` (default `20.0`, must be >0)
- `--mu-y` (default `20.0`, must be >0)
- `--mu-ori` (default `5.0`, must be >0)
- `--m1-focal-gamma` (default `2.0`, must be >=0)
- `--m1-pos-weight-max` (default `100.0`, must be >=1.0)
- `--m1-hard-neg-enable/--no-m1-hard-neg-enable` (default `True`)
- `--m1-hard-neg-ratio` (default `20.0`, must be >=0)
- `--m1-hard-neg-min` (default `2000`, must be >=0)
- `--m1-hard-neg-fraction` (default `0.05`, must be in `[0,1]`)

Data selection:
- `--strict-gradient-targets` (default `False`)
- `--seed` (default `13`)
- `--limit` (default `None`)

## 7) Evaluation (`evaluate.py`)

## 7.1 What `evaluate.py` Reports

1. Loss metrics using `FeatureNetLoss`:
- `total`, `orientation`, `ridge`, `gradient`, `minutia`, `m1`, `m2`, `m3`, `m4`

2. Paper-style minutia detection metrics:
- score thresholds `0.10 ... 0.90`
- splits: `all`, `front`, `side`
- stats: `tp`, `fp`, `fn`, `precision`, `recall`, `f1`
- best chosen by highest `all.f1`:
  - `best_score_threshold`
  - `best_score_f1`

3. Label accuracies on minutia-positive cells (`minutia_valid_mask`):
- `minutia_x`
- `minutia_y`
- `minutia_orientation` (within-15deg angular accuracy)
- `minutia_x` and `minutia_y` are tolerance-based regression accuracies (`abs(error_cell) < 0.125`), not class exact-match
- plus regression quality metrics:
  - `minutia_x_mae_cell`, `minutia_y_mae_cell`
  - `minutia_x_mae_px`, `minutia_y_mae_px`
- plus `orientation_mae_deg` (mean absolute angular error in degrees)

## 7.2 Paper-Style Minutia Matching Details

Pred decode:
- `pred_score = sigmoid(minutia_score_logit)`
- thresholding:
  - prediction uses `>= threshold`
  - GT uses `> target_threshold` (default `0.0`)
- eval region: `targets["mask"] > 0.5`
- local NMS: 3x3 max-pool; keep cells where `score >= pooled - 1e-8`
- subcell decode:
  - `x = col + sigmoid(x_logit)`
  - `y = row + sigmoid(y_logit)`
- coordinate conversion to input pixels:
  - scale by `input_shape_hw / output_shape_hw` from targets
- orientation decode:
  - normalize predicted orientation vector `[cos,sin]`
  - `theta = atan2(sin, cos)`
  - `theta_deg = (theta * 180/pi) % 360`
  - GT angle uses `minutia_orientation_vec` when present, otherwise bin-center decode from legacy labels

Matching rule (paper-style):
- spatial distance `< 8` pixels (strict)
- angular difference `< 15` degrees (strict), circular wrap
- one-to-one assignment by maximum-cardinality bipartite matching

Split mapping:
- `raw_view_index == 0` -> `all` + `front`
- `raw_view_index in {1,2}` -> `all` + `side`
- otherwise -> `all` only

## 7.3 Evaluation Data Selection

`evaluate.py`:
- loads checkpoint args
- uses checkpoint `val_fraction` and `seed` unless overridden
- uses same `load_bundle_samples` and `split_samples` logic as train
- if split fails or empty, falls back to all samples

Important implementation detail:
- standalone `evaluate.py` instantiates `FeatureNetLoss()` with default weights, not necessarily the custom training weights in checkpoint args
- detection metrics (`best_score_f1` etc.) are independent of loss weights
- old 360-bin orientation checkpoints are rejected with a clear compatibility error because current model expects 2-channel minutia orientation output
- checkpoints with 8-channel x/y heads are also rejected because current model expects 1-channel continuous x/y offsets

## 7.4 Evaluate CLI (all args)

Required:
- `--ground-truth-root`
- `--checkpoint-path`

Other args:
- `--device` (default auto)
- `--batch-size` (default `1`)
- `--num-workers` (default `2`)
- `--amp/--no-amp` (default `True`)
- `--pin-memory/--no-pin-memory` (default `True`)
- `--persistent-workers/--no-persistent-workers` (default `True`)
- `--prefetch-factor` (default `2`)
- `--channels-last` (default `False`)
- `--seed` (default `None`, fallback checkpoint seed)
- `--val-fraction` (default `None`, fallback checkpoint val_fraction)
- `--limit` (default `None`)
- `--target-threshold` (default `0.0`, used for GT point decoding)
- `--output-json` (default `None`)

## 8) Ground-Truth Generation CLI and Merge/Shard Behavior

`generate_ground_truth.py` supports:
- generation
- shard generation
- shard merge
- multi-root merge with identity remapping
- offline patch dataset generation

CLI args:
- `--dataset-root`
- `--output-root`
- `--merge-shards-root`
- `--merge-generated-root` (repeatable `LABEL=PATH` or `PATH`)
- `--execution-target {local,kaggle}`
- `--gpu-only`
- `--gpu-batch-size`
- `--cpu-workers`
- `--prefetch-samples`
- `--skip-existing`
- `--shard-mode {off,auto,manual}`
- `--shard-count`
- `--shard-index`
- `--target-shard-size`
- `--patch-source-root`
- `--patch-output-root`
- `--patch-limit-samples`
- `--patch-size`
- `--patch-min-mask-ratio`
- `--patch-smoke-samples`
- `--fingerflow-model-dir`
- `--fingerflow-backend {auto,local,wsl}`
- `--wsl-distro`
- `--wsl-activate`
- `--limit`
- `--visualize-count`
- `--smoke-samples`
- `--dpi`

Shard behavior:
- acquisition-boundary aligned sharding (keeps views of an acquisition together)
- auto mode picks shard count from `target_shard_size` default 500 unless explicit `shard_count`
- output shard folder name: `shard_{index:03d}`

Multi-root merge behavior (`--merge-generated-root`):
- safe merge into empty output root only
- rewrites IDs to prevent clashes:
  - sample IDs prefixed with source label
  - reconstruction IDs prefixed with source label
  - global subject remap `(source_label, source_subject_id) -> merged_subject_index`
  - recomputes `finger_class_id`
- rewrites bundle/reconstruction metadata paths after copy

## 9) What the Model Actually Consumes at Train/Val/Inference Time

At runtime, the network input is not raw RGB.

It consumes:
- `image` channel: preprocessed grayscale (`masked_image.png`) normalized to `[0,1]`
- `mask` channel: binary mask (`mask.png`) in `{0,1}`

Forward contract:
- call `model(image, mask=mask)` where each is `[B,1,H,W]`
- model internally computes `image * mask` and concatenates mask to create 2-channel tensor

So segmentation/masking and normalization are already baked into the generated bundle inputs, and the model always receives the mask during both training and evaluation.

## 10) Practical Notes and Edge Cases

- If any selected sample lacks reconstruction `gradient` target:
  - strict mode on: training/eval fail fast
  - strict mode off: sample skipped with warning
- Validation metrics can be computed every N epochs; patience counts only validation checks.
- `best.pt` tracks whichever metric is configured for monitoring.
- `last.pt` is always written at training end (including early stop).
- This branch changes `minutia_orientation` head shape from `360` to `2` and `minutia_x/minutia_y` from `8` to `1`; old checkpoints are intentionally incompatible and require a new training run.
