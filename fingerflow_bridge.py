from __future__ import annotations

import argparse
from pathlib import Path

import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge FingerFlow extraction through main.py helpers.")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--enhanced-image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--minutiae-json", type=Path, required=True)
    parser.add_argument("--minutiae-csv", type=Path, required=True)
    parser.add_argument("--core-csv", type=Path, required=True)
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
    model_paths = main.ensure_fingerflow_models(args.model_dir)
    main.extract_minutiae_with_fingerflow(
        args.source_image.resolve(),
        args.enhanced_image.resolve(),
        model_paths,
        args.minutiae_json.resolve(),
        args.minutiae_csv.resolve(),
        args.core_csv.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
