from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ridge-aware glare suppression for contactless fingerprint images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input fingerprint image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for glare-suppressed output image (default: <input_dir>/glare_suppressed.png).",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=None,
        help="Path for binary glare mask output (default: <input_dir>/glare_mask.png).",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=None,
        help="Optional foreground finger mask path. If omitted, mask is estimated.",
    )
    return parser.parse_args()


def load_image_unchanged(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to decode image: {path}")
    return image


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
    return refined


def estimate_foreground_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def largest_ratio(mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        largest_area = max(cv2.contourArea(c) for c in contours)
        return float(largest_area) / float(mask.size)

    mask = threshold_bright if largest_ratio(threshold_bright) >= largest_ratio(threshold_dark) else threshold_dark
    mask = largest_component_mask(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def resolve_gray_and_mask(image: np.ndarray, foreground_mask_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim == 2:
        gray = image
        alpha = None
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = image[:, :, 3]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")

    if foreground_mask_path is not None:
        provided = load_image_unchanged(foreground_mask_path)
        if provided.ndim == 3:
            if provided.shape[2] == 4:
                provided = provided[:, :, 3]
            else:
                provided = cv2.cvtColor(provided, cv2.COLOR_BGR2GRAY)
        if provided.shape != gray.shape:
            raise ValueError(
                f"foreground mask shape mismatch: expected {gray.shape}, got {provided.shape}"
            )
        return gray, np.where(provided > 0, 255, 0).astype(np.uint8)

    if alpha is not None and np.any(alpha > 0):
        return gray, np.where(alpha > 0, 255, 0).astype(np.uint8)
    return gray, estimate_foreground_mask(gray)


def gaussian_blur_sigma(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def suppress_glare(gray: np.ndarray, foreground_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)
    foreground = foreground_mask > 0

    if int(np.count_nonzero(foreground)) == 0:
        raise RuntimeError("foreground mask is empty")

    illumination = gaussian_blur_sigma(gray_f, sigma=15.0)
    ridge_base = gaussian_blur_sigma(gray_f, sigma=2.2)
    ridge_detail = gray_f - ridge_base

    ridge_energy = gaussian_blur_sigma(np.abs(ridge_detail), sigma=3.0)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = gaussian_blur_sigma(np.sqrt((grad_x * grad_x) + (grad_y * grad_y)), sigma=3.0)

    specular_residual = np.maximum(0.0, gray_f - illumination)

    fg_intensity = gray_f[foreground]
    fg_residual = specular_residual[foreground]
    fg_ridge = ridge_energy[foreground]
    fg_grad = grad_energy[foreground]

    intensity_t = float(np.percentile(fg_intensity, 85))
    residual_t = float(np.percentile(fg_residual, 80))
    ridge_t = float(np.percentile(fg_ridge, 45))
    grad_t = float(np.percentile(fg_grad, 45))

    glare_mask = (
        foreground
        & (gray_f >= intensity_t)
        & (specular_residual >= residual_t)
        & (ridge_energy <= ridge_t)
        & (grad_energy <= grad_t)
    )
    glare_mask_u8 = np.where(glare_mask, 255, 0).astype(np.uint8)
    glare_mask_u8 = cv2.morphologyEx(
        glare_mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    glare_mask_u8 = cv2.morphologyEx(
        glare_mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1
    )
    glare_mask_u8 = cv2.dilate(glare_mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    glare_mask_u8 = np.where(foreground, glare_mask_u8, 0).astype(np.uint8)

    in_glare = glare_mask_u8 > 0
    if int(np.count_nonzero(in_glare)) == 0:
        return gray.copy(), glare_mask_u8

    cap = float(np.percentile(specular_residual[in_glare], 90))
    cap = max(cap, 6.0)
    compressed_residual = cap * np.tanh(specular_residual / cap)

    recovered = illumination + compressed_residual + ridge_detail
    recovered = np.clip(recovered, 0.0, 255.0)

    soft = gaussian_blur_sigma(glare_mask_u8.astype(np.float32) / 255.0, sigma=2.0)
    soft = np.clip(soft, 0.0, 1.0)
    soft *= foreground.astype(np.float32)

    blended = (gray_f * (1.0 - soft)) + (recovered * soft)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    blended[~foreground] = gray[~foreground]
    return blended, glare_mask_u8


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    image = load_image_unchanged(input_path)
    gray, foreground_mask = resolve_gray_and_mask(image, args.foreground_mask)

    suppressed, glare_mask = suppress_glare(gray, foreground_mask)

    output_path = args.output or (input_path.parent / "glare_suppressed.png")
    mask_output_path = args.mask_output or (input_path.parent / "glare_mask.png")
    output_path = output_path.resolve()
    mask_output_path = mask_output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), suppressed):
        raise RuntimeError(f"failed to save glare-suppressed image: {output_path}")
    if not cv2.imwrite(str(mask_output_path), glare_mask):
        raise RuntimeError(f"failed to save glare mask image: {mask_output_path}")

    print(f"Saved glare-suppressed image: {output_path}")
    print(f"Saved glare mask: {mask_output_path}")
    print(f"Glare mask foreground ratio: {float(np.mean(glare_mask > 0)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
