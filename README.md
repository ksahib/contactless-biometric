# contactless-biometric

## Kaggle GPU Path

Use the Kaggle/Linux bootstrap first:

```bash
bash scripts/kaggle_gpu_bootstrap.sh
```

Run the full GPU-first pipeline with:

```bash
python scripts/run_kaggle_pipeline.py
```

The generator now supports Kaggle-oriented controls:

```bash
python generate_ground_truth.py \
  --execution-target kaggle \
  --gpu-only \
  --cpu-workers 4 \
  --prefetch-samples 2 \
  --skip-existing
```

To merge shard outputs from a single dataset:

```bash
python generate_ground_truth.py \
  --merge-shards-root ground_truth/DS1_shards \
  --output-root ground_truth/DS1
```

To safely merge separately generated dataset roots such as `DS1`, `DS2`, and `DS3`, use one labeled source per root. This rewrites copied bundle metadata so `sample_id`, `subject_id`, `finger_class_id`, and reconstruction references stay unique in the merged output:

```bash
python generate_ground_truth.py \
  --merge-generated-root ds1=ground_truth/DS1 \
  --merge-generated-root ds2=ground_truth/DS2 \
  --merge-generated-root ds3=ground_truth/DS3 \
  --output-root ground_truth/DS123_merged
```

FeatureNet train/eval now expose CUDA throughput knobs such as `--amp`, `--channels-last`, `--compile`, `--persistent-workers`, `--prefetch-factor`, `--pin-memory`, `--grad-accum-steps`, and `--cudnn-benchmark`.

If you adopt the raw-logit head refactor (final head layers changed from `ConvBlock` to plain `nn.Conv2d`), treat it as a new architecture run. Do not resume or evaluate with older checkpoints from the pre-refactor head design.

If you adopt the continuous minutia-orientation refactor (`minutia_orientation` head changed from `360` bins to `2` channels `[cos,sin]`), this is also a new architecture run. Older checkpoints with 360-bin minutia orientation heads are intentionally incompatible.

For paper-style minutia optimization with full-negative score supervision and hard-negative mining, use F1-aligned validation:

```bash
python -m featurenet.models.train \
  --ground-truth-root ground_truth/DS123_merged \
  --output-dir runs/featurenet_ds123_f1_hardneg \
  --device cuda \
  --epochs 80 \
  --batch-size 8 \
  --num-workers 4 \
  --val-fraction 0.2 \
  --validate-every 1 \
  --early-stopping \
  --early-stopping-metric best_score_f1 \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 1e-4 \
  --xy-soft-target-sigma 1.0 \
  --m1-hard-neg-enable \
  --m1-hard-neg-ratio 3.0 \
  --m1-hard-neg-min 128 \
  --amp \
  --channels-last
```

The evaluator keeps the paper-style threshold + NMS protocol (no top-K post-cap in default behavior).
Minutia x/y supervision now uses Gaussian soft-target cross-entropy over ordered 8 bins (always enabled), controlled by `--xy-soft-target-sigma` (default `1.0`).
