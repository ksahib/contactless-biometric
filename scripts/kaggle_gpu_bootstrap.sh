#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-kaggle-gpu.txt

python - <<'PY'
import json

summary = {}

try:
    import tensorflow as tf
    summary["tensorflow_gpus"] = [getattr(device, "name", str(device)) for device in tf.config.list_physical_devices("GPU")]
except Exception as exc:
    summary["tensorflow_error"] = str(exc)

try:
    import onnxruntime as ort
    summary["onnx_providers"] = list(ort.get_available_providers())
except Exception as exc:
    summary["onnxruntime_error"] = str(exc)

try:
    import torch
    summary["torch_cuda_available"] = bool(torch.cuda.is_available())
    summary["torch_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        summary["torch_device_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    summary["torch_error"] = str(exc)

print(json.dumps(summary, indent=2))
PY
