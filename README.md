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
