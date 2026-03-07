# from __future__ import annotations

# import argparse
# import os
# import sys
# from pathlib import Path

# import cv2
# import numpy as np
# from rembg import new_session, remove


# REPO_ROOT = Path(__file__).resolve().parent
# OUTPUT_NOBG_DIR = REPO_ROOT / "output_nobg"
# OUTPUT_CROPPED_DIR = REPO_ROOT / "output_cropped"
# REMBG_MODEL = os.environ.get("FINGER_REMBG_MODEL", "u2netp")

# _REMBG_SESSION = None


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Crop a finger to the distal phalanx and remove its background."
#     )
#     parser.add_argument("input_image", type=Path, help="Path to the source finger image")
#     return parser.parse_args()


# def get_rembg_session():
#     global _REMBG_SESSION
#     if _REMBG_SESSION is None:
#         _REMBG_SESSION = new_session(REMBG_MODEL)
#     return _REMBG_SESSION


# def load_bgr_image(path: Path) -> np.ndarray:
#     if not path.exists():
#         raise FileNotFoundError(f"input image does not exist: {path}")
#     image = cv2.imread(str(path), cv2.IMREAD_COLOR)
#     if image is None:
#         raise ValueError(f"failed to decode image: {path}")
#     return image


# def load_gray_image(path: Path) -> np.ndarray:
#     if not path.exists():
#         raise FileNotFoundError(f"input image does not exist: {path}")
#     image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"failed to decode image: {path}")
#     return image


# def load_image_unchanged(path: Path) -> np.ndarray:
#     if not path.exists():
#         raise FileNotFoundError(f"input image does not exist: {path}")
#     image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#     if image is None:
#         raise ValueError(f"failed to decode image: {path}")
#     return image


# def load_gray_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
#     image = load_image_unchanged(path)
#     if image.ndim == 2:
#         gray = image
#         alpha = None
#         mask = np.full(gray.shape, 255, dtype=np.uint8)
#     elif image.ndim == 3 and image.shape[2] == 4:
#         gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
#         alpha = image[:, :, 3]
#         mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
#     elif image.ndim == 3 and image.shape[2] == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         alpha = None
#         mask = np.full(gray.shape, 255, dtype=np.uint8)
#     else:
#         raise ValueError(f"unsupported image shape for masked grayscale loading: {image.shape}")

#     return gray, mask, alpha


# def build_output_paths(input_path: Path) -> tuple[Path, Path]:
#     stem = input_path.stem
#     return (
#         OUTPUT_NOBG_DIR / f"{stem}_nobg.png",
#         OUTPUT_CROPPED_DIR / f"{stem}_cropped.png",
#     )


# def rembg_mask_from_bgr(bgr_image: np.ndarray) -> np.ndarray:
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     mask = remove(
#         rgb_image,
#         session=get_rembg_session(),
#         only_mask=True,
#         post_process_mask=True,
#     )
#     if mask.ndim == 3:
#         mask = mask[:, :, 0]
#     mask = np.where(mask > 127, 255, 0).astype(np.uint8)

#     mask = cv2.morphologyEx(
#         mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2
#     )
#     mask = cv2.morphologyEx(
#         mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1
#     )
#     return largest_component_mask(mask)


# def largest_component_mask(mask: np.ndarray) -> np.ndarray:
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         raise RuntimeError("no foreground contour found")
#     largest = max(contours, key=cv2.contourArea)
#     refined = np.zeros_like(mask)
#     cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
#     return refined


# def validate_foreground_area(mask: np.ndarray, minimum_ratio: float) -> None:
#     area_ratio = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
#     if area_ratio < minimum_ratio:
#         raise RuntimeError("foreground mask is too small to represent a finger")


# def estimate_finger_axis(
#     contour: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, float, float]:
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)

#     edge_a = box[1] - box[0]
#     edge_b = box[2] - box[1]
#     len_a = float(np.linalg.norm(edge_a))
#     len_b = float(np.linalg.norm(edge_b))

#     if len_a >= len_b:
#         long_edge = edge_a
#         short_edge = edge_b
#         length_l = len_a
#         width_w = len_b
#     else:
#         long_edge = edge_b
#         short_edge = edge_a
#         length_l = len_b
#         width_w = len_a

#     if length_l <= 0 or width_w <= 0:
#         raise RuntimeError("failed to estimate finger orientation")

#     axis_u = long_edge / length_l
#     axis_v = short_edge / width_w
#     return axis_u.astype(np.float32), axis_v.astype(np.float32), length_l, width_w


# def select_fingertip_end(
#     contour: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray, length_l: float
# ) -> tuple[np.ndarray, np.ndarray]:
#     points = contour.reshape(-1, 2).astype(np.float32)
#     proj_u = points @ axis_u
#     proj_v = points @ axis_v
#     min_u = float(proj_u.min())
#     max_u = float(proj_u.max())
#     band = max(2.0, 0.14 * length_l)

#     min_selector = proj_u <= (min_u + band)
#     max_selector = proj_u >= (max_u - band)

#     def stats(selector: np.ndarray) -> tuple[float, float, float]:
#         chosen = points[selector]
#         if chosen.shape[0] < 6:
#             raise RuntimeError("insufficient contour support to localize fingertip")
#         local_v = proj_v[selector]
#         return (
#             float(local_v.max() - local_v.min()),
#             float(np.mean(local_v)),
#             float(np.mean(chosen[:, 1])),
#         )

#     min_width, min_center_v, min_mean_y = stats(min_selector)
#     max_width, max_center_v, max_mean_y = stats(max_selector)

#     width_delta = abs(min_width - max_width) / max(min_width, max_width, 1.0)
#     if width_delta >= 0.08:
#         choose_min = min_width < max_width
#     else:
#         choose_min = min_mean_y <= max_mean_y

#     if choose_min:
#         tip_proj = min_u
#         center_v = min_center_v
#         tip_to_base = axis_u
#     else:
#         tip_proj = max_u
#         center_v = max_center_v
#         tip_to_base = -axis_u

#     tip_center = (axis_u * tip_proj) + (axis_v * center_v)
#     return tip_center.astype(np.float32), tip_to_base.astype(np.float32)


# def local_width_from_mask(
#     mask: np.ndarray,
#     tip_center: np.ndarray,
#     tip_to_base: np.ndarray,
#     axis_v: np.ndarray,
#     length_l: float,
#     band_fraction: float,
# ) -> float:
#     ys, xs = np.where(mask > 0)
#     if ys.size == 0:
#         raise RuntimeError("mask is empty")

#     points = np.stack([xs, ys], axis=1).astype(np.float32)
#     relative = points - tip_center
#     long_coord = relative @ tip_to_base
#     cross_coord = relative @ axis_v

#     selector = (long_coord >= (-0.04 * length_l)) & (long_coord <= (band_fraction * length_l))
#     if not np.any(selector):
#         raise RuntimeError("failed to measure local width")

#     local_cross = cross_coord[selector]
#     return float(np.percentile(local_cross, 95) - np.percentile(local_cross, 5))


# def compute_distal_crop_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         raise RuntimeError("no contour available for crop")
#     contour = max(contours, key=cv2.contourArea)

#     axis_u, axis_v, length_l, _ = estimate_finger_axis(contour)
#     tip_center, tip_to_base = select_fingertip_end(contour, axis_u, axis_v, length_l)

#     width_near_tip = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.20)
#     width_stable = local_width_from_mask(mask, tip_center, tip_to_base, axis_v, length_l, 0.34)
#     distal_width = max(width_near_tip, 0.92 * width_stable)

#     inward_extension = min(0.66 * length_l, max(2.35 * distal_width, 0.50 * length_l))
#     outward_extension = 0.10 * length_l
#     half_width = 0.82 * distal_width
#     margin_px = max(20, int(round(0.10 * max(length_l, distal_width))))

#     outer_center = tip_center - (tip_to_base * outward_extension)
#     inner_center = tip_center + (tip_to_base * inward_extension)

#     crop_quad = np.array(
#         [
#             outer_center - (axis_v * half_width),
#             outer_center + (axis_v * half_width),
#             inner_center + (axis_v * half_width),
#             inner_center - (axis_v * half_width),
#         ],
#         dtype=np.float32,
#     )

#     x_min = int(np.floor(np.min(crop_quad[:, 0]))) - margin_px
#     y_min = int(np.floor(np.min(crop_quad[:, 1]))) - margin_px
#     x_max = int(np.ceil(np.max(crop_quad[:, 0]))) + margin_px
#     y_max = int(np.ceil(np.max(crop_quad[:, 1]))) + margin_px

#     height, width = mask.shape[:2]
#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(width, x_max)
#     y_max = min(height, y_max)

#     if (x_max - x_min) < 32 or (y_max - y_min) < 32:
#         raise RuntimeError("computed distal phalanx crop is too small")

#     return x_min, y_min, x_max, y_max


# def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
#     x_min, y_min, x_max, y_max = bbox
#     return image[y_min:y_max, x_min:x_max].copy()


# def compose_rgba(bgr_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
#     bgra[:, :, 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
#     return bgra


# def save_png(path: Path, image: np.ndarray) -> None:
#     if not cv2.imwrite(str(path), image):
#         raise RuntimeError(f"failed to save image: {path}")


# def process_image(input_path: Path) -> tuple[Path, Path]:
#     full_bgr = load_bgr_image(input_path)

#     coarse_mask = rembg_mask_from_bgr(full_bgr)
#     validate_foreground_area(coarse_mask, minimum_ratio=0.03)

#     crop_bbox = compute_distal_crop_bbox(coarse_mask)
#     cropped_bgr = crop_image(full_bgr, crop_bbox)

#     cropped_mask = rembg_mask_from_bgr(cropped_bgr)
#     validate_foreground_area(cropped_mask, minimum_ratio=0.08)
#     cropped_rgba = compose_rgba(cropped_bgr, cropped_mask)

#     nobg_path, cropped_path = build_output_paths(input_path)
#     OUTPUT_NOBG_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

#     save_png(cropped_path, cropped_bgr)
#     save_png(nobg_path, cropped_rgba)
#     return nobg_path, cropped_path

# def normalise_brightness(input_path: Path, output_path: Path) -> None:
#     image = load_image_unchanged(input_path)
#     if image.ndim == 2:
#         img_grey = image
#         alpha = None
#     elif image.ndim == 3 and image.shape[2] == 4:
#         img_grey = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
#         alpha = image[:, :, 3]
#     elif image.ndim == 3 and image.shape[2] == 3:
#         img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         alpha = None
#     else:
#         raise ValueError(f"unsupported image shape for brightness normalization: {image.shape}")

#     if alpha is not None:
#         foreground = alpha > 0
#         ys, xs = np.where(foreground)
#         if ys.size == 0:
#             raise RuntimeError("foreground mask is empty")

#         y_min, y_max = int(ys.min()), int(ys.max()) + 1
#         x_min, x_max = int(xs.min()), int(xs.max()) + 1
#         cropped_grey = img_grey[y_min:y_max, x_min:x_max]

#         clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
#         cropped_clahe = clahe.apply(cropped_grey)
#         clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
#         cropped_clahe = clahe2.apply(cropped_clahe)

#         masked_grey = np.zeros_like(img_grey)
#         masked_grey[y_min:y_max, x_min:x_max] = cropped_clahe
#         masked_grey[~foreground] = 0

#         output = cv2.cvtColor(masked_grey, cv2.COLOR_GRAY2BGRA)
#         output[:, :, 3] = alpha
#         cv2.imwrite(str(output_path), output)
#         return

#     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
#     img_clahe = clahe.apply(img_grey)
#     clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
#     img_clahe = clahe2.apply(img_clahe)
#     cv2.imwrite(str(output_path), img_clahe)

# def filter(input_path: Path, output_path: Path) -> None:
#     image, mask, alpha = load_gray_and_mask(input_path)
#     if not np.any(mask):
#         raise RuntimeError("foreground mask is empty")

#     blockh, blockw = 32, 32
#     strideh, stridew = 4, 4
#     pad = 0
#     detail_gain = 24.0
#     image_padded = cv2.copyMakeBorder(
#         image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
#     )
#     mask_padded = cv2.copyMakeBorder(
#         mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
#     )
#     height, width = image.shape[:2]

#     accum = np.zeros_like(image_padded, dtype=np.float32)
#     count = np.zeros_like(image_padded, dtype=np.float32)

#     for y in range(0, height - blockh + 1, strideh):
#         for x in range(0, width - blockw + 1, stridew):
#             block = image_padded[y:y+blockh, x:x+blockw]
#             block_mask = mask_padded[y:y+blockh, x:x+blockw]
#             block_float = block.astype(np.float32)
#             foreground = block_mask == 255
#             accum_view = accum[y:y+blockh, x:x+blockw]
#             count_view = count[y:y+blockh, x:x+blockw]

#             if not np.all(foreground):
#                 accum_view[foreground] += block_float[foreground]
#                 count_view[foreground] += 1.0
#                 continue

#             block_fft = np.fft.fft2(block_float)
#             block_fft_shifted = np.fft.fftshift(block_fft)
#             magnitude = np.abs(block_fft_shifted)
#             peak = np.argmax(magnitude)
#             x_peak, y_peak = np.unravel_index(peak, magnitude.shape)
#             sp_x = x_peak * (1.0 / blockw)
#             sp_y = y_peak * (1.0 / blockh)
#             freq = np.sqrt(sp_x**2 + sp_y**2)
#             lambda_ = 1.0 / freq if freq > 0 else 0.0
#             sigma = 0.5 * lambda_
#             dx = cv2.Sobel(block_float, cv2.CV_64F, dx=1, dy=0, ksize=3)
#             dy = cv2.Sobel(block_float, cv2.CV_64F, dx=0, dy=1, ksize=3)
#             Gxx = np.sum(dx ** 2)
#             Gyy = np.sum(dy ** 2)
#             Gxy = np.sum(dx * dy)
#             theta = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
#             ksize = int(np.round(3 * sigma))
#             if ksize % 2 == 0:
#                 ksize += 1
#             if ksize < 1:
#                 ksize = 3
#             psi = 0
#             gamma = 0.5
#             gabor_kernel = cv2.getGaborKernel(
#                 (ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F
#             )
#             block_mean = float(np.mean(block_float))
#             block_std = float(np.std(block_float))
#             if block_std < 1e-3:
#                 accum_view[foreground] += block_float[foreground]
#                 count_view[foreground] += 1.0
#                 continue

#             normalized_block = (block_float - block_mean) / block_std
#             response = cv2.filter2D(normalized_block, cv2.CV_32F, gabor_kernel)
#             enhanced_block = np.clip(
#                 block_float + (detail_gain * np.tanh(response)),
#                 0.0,
#                 255.0,
#             )

#             accum_view[foreground] += enhanced_block[foreground]
#             count_view[foreground] += 1.0

#     stitched = np.zeros_like(image, dtype=np.float32)
#     cropped_accum = accum[pad:pad+height, pad:pad+width]
#     cropped_count = count[pad:pad+height, pad:pad+width]
#     foreground = mask == 255
#     valid = foreground & (cropped_count > 0)
#     stitched[valid] = cropped_accum[valid] / cropped_count[valid]
#     stitched[foreground & ~valid] = image[foreground & ~valid].astype(np.float32)
#     stitched = np.clip(stitched, 0.0, 255.0).astype(np.uint8)
#     stitched[~foreground] = 0

#     if alpha is not None:
#         output = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGRA)
#         output[:, :, 3] = alpha
#         output[alpha == 0, :3] = 0
#         cv2.imwrite(str(output_path), output)
#         return

#     cv2.imwrite(str(output_path), stitched)
    

# def main() -> int:
#     args = parse_args()
#     try:
#         nobg_path, cropped_path = process_image(args.input_image)
#         normalise_brightness(nobg_path, Path("normalised_cropped.png"))
#         filter(Path("normalised_cropped.png"), Path("filtered_cropped.png"))
#     except Exception as exc:
#         print(f"Error: {exc}", file=sys.stderr)
#         return 1

#     print(f"Saved background-removed image: {nobg_path}")
#     print(f"Saved cropped fingerprint image: {cropped_path}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
