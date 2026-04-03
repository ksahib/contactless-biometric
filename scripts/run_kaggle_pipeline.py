from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _provider_summary() -> dict[str, object]:
    summary: dict[str, object] = {}

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

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kaggle-first GPU pipeline end-to-end.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "DS1")
    parser.add_argument("--ground-truth-root", type=Path, default=REPO_ROOT / "ground_truth" / "DS1_kaggle")
    parser.add_argument("--patch-output-root", type=Path, default=None)
    parser.add_argument("--train-output-dir", type=Path, default=REPO_ROOT / "match_outputs" / "featurenet_kaggle")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--generate-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = _provider_summary()
    print(json.dumps(summary, indent=2), flush=True)

    if not summary.get("tensorflow_gpus"):
        raise RuntimeError("TensorFlow GPU is not visible")
    if "CUDAExecutionProvider" not in list(summary.get("onnx_providers", [])):
        raise RuntimeError("onnxruntime-gpu is not exposing CUDAExecutionProvider")
    if not summary.get("torch_cuda_available"):
        raise RuntimeError("PyTorch CUDA is not available")

    if not args.skip_generate:
        command = [
            sys.executable,
            "generate_ground_truth.py",
            "--execution-target",
            "kaggle",
            "--gpu-only",
            "--skip-existing",
            "--dataset-root",
            str(args.dataset_root),
            "--output-root",
            str(args.ground_truth_root),
            "--cpu-workers",
            str(args.num_workers),
            "--prefetch-samples",
            "2",
            "--fingerflow-backend",
            "local",
        ]
        if args.generate_limit is not None:
            command.extend(["--limit", str(args.generate_limit)])
        if args.patch_output_root is not None:
            command.extend(["--patch-output-root", str(args.patch_output_root)])
        _run(command)

    checkpoint_path = args.checkpoint_path or (args.train_output_dir / "best.pt")
    if not args.skip_train:
        _run(
            [
                sys.executable,
                "-m",
                "featurenet.models.train",
                "--ground-truth-root",
                str(args.ground_truth_root),
                "--output-dir",
                str(args.train_output_dir),
                "--device",
                "cuda",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--amp",
                "--pin-memory",
                "--persistent-workers",
                "--prefetch-factor",
                "2",
                "--cudnn-benchmark",
            ]
            + (["--limit", str(args.train_limit)] if args.train_limit is not None else [])
        )

    if not args.skip_eval:
        _run(
            [
                sys.executable,
                "-m",
                "featurenet.models.evaluate",
                "--ground-truth-root",
                str(args.ground_truth_root),
                "--checkpoint-path",
                str(checkpoint_path),
                "--device",
                "cuda",
                "--batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--amp",
                "--pin-memory",
                "--persistent-workers",
                "--prefetch-factor",
                "2",
                "--output-json",
                str(args.train_output_dir / "eval_metrics.json"),
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
