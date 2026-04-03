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

FeatureNet train/eval now expose CUDA throughput knobs such as `--amp`, `--channels-last`, `--compile`, `--persistent-workers`, `--prefetch-factor`, `--pin-memory`, `--grad-accum-steps`, and `--cudnn-benchmark`.
