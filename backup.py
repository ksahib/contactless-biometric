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
#         description="Crop a finger to the distal phalanx, remove its background, and enhance the print."
#     )
#     parser.add_argument("input_image", type=Path, help="Path to the source finger image")
#     parser.add_argument(
#         "--skip-library",
#         action="store_true",
#         help="Skip the optional library-based comparison output.",
#     )
#     parser.add_argument(
#         "--custom-only",
#         action="store_true",
#         help="Alias for --skip-library.",
#     )
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
#     image = load_bgr_image(input_path)
#     img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
#     img_clahe = clahe.apply(img_grey)
#     clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
#     img_clahe = clahe2.apply(img_clahe)
#     # clahe3 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
#     # img_clahe = clahe3.apply(img_clahe)
#     cv2.imwrite(str(output_path), img_clahe)

# def sliding_starts(length: int, block_size: int, stride: int) -> list[int]:
#     if block_size >= length:
#         return [0]
#     starts = list(range(0, length - block_size + 1, stride))
#     tail = length - block_size
#     if starts[-1] != tail:
#         starts.append(tail)
#     return starts


# def build_foreground_mask(image: np.ndarray) -> np.ndarray:
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     foreground = largest_component_mask(mask)
#     foreground = cv2.morphologyEx(
#         foreground, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1
#     )
#     foreground = cv2.morphologyEx(
#         foreground, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1
#     )
#     return foreground


# def estimate_orientation_components(
#     image: np.ndarray, block_size: int
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     image_float = image.astype(np.float32)
#     dx = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
#     dy = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)

#     gxx = cv2.boxFilter(dx * dx, cv2.CV_32F, (block_size, block_size), normalize=False)
#     gyy = cv2.boxFilter(dy * dy, cv2.CV_32F, (block_size, block_size), normalize=False)
#     gxy = cv2.boxFilter(dx * dy, cv2.CV_32F, (block_size, block_size), normalize=False)

#     vx = 2.0 * gxy
#     vy = gxx - gyy
#     coherence = np.sqrt((vx * vx) + (vy * vy)) / (gxx + gyy + 1e-6)
#     return vx, vy, coherence.astype(np.float32)


# def estimate_orientation_field(
#     image: np.ndarray, block_size: int, smooth_ksize: int
# ) -> np.ndarray:
#     vx, vy, _ = estimate_orientation_components(image, block_size)
#     magnitude = np.sqrt((vx * vx) + (vy * vy)) + 1e-6
#     sin2theta = vx / magnitude
#     cos2theta = vy / magnitude

#     sin2theta = cv2.GaussianBlur(sin2theta, (smooth_ksize, smooth_ksize), 0)
#     cos2theta = cv2.GaussianBlur(cos2theta, (smooth_ksize, smooth_ksize), 0)
#     return (0.5 * np.arctan2(sin2theta, cos2theta)).astype(np.float32)


# def extract_centered_block(
#     image: np.ndarray, center_y: int, center_x: int, block_size: int
# ) -> np.ndarray:
#     radius = block_size // 2
#     return image[center_y - radius:center_y + radius, center_x - radius:center_x + radius]


# def estimate_block_wavelength(
#     block: np.ndarray, theta: float, min_wavelength: int, max_wavelength: int
# ) -> tuple[float, bool]:
#     size = block.shape[0]
#     center = (size / 2.0, size / 2.0)
#     rotation = cv2.getRotationMatrix2D(center, -np.degrees(theta), 1.0)
#     rotated = cv2.warpAffine(
#         block,
#         rotation,
#         (size, size),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT,
#     )

#     band_margin = max(2, size // 4)
#     band = rotated[:, band_margin:size - band_margin]
#     profile = band.mean(axis=1).astype(np.float32)
#     profile = cv2.GaussianBlur(profile[:, None], (1, 5), 0)[:, 0]
#     profile -= float(np.mean(profile))

#     energy = float(np.std(profile))
#     if energy < 1.5:
#         return 8.0, False

#     autocorr = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
#     zero_lag = float(autocorr[0])
#     if zero_lag <= 1e-6:
#         return 8.0, False

#     search = autocorr[min_wavelength:max_wavelength + 1]
#     if search.size == 0:
#         return 8.0, False

#     best_offset = int(np.argmax(search))
#     wavelength = float(min_wavelength + best_offset)
#     confidence = float(search[best_offset] / zero_lag)
#     valid = confidence > 0.12
#     return wavelength, valid


# def fill_invalid_blocks(
#     block_values: np.ndarray, valid_mask: np.ndarray, default_value: float
# ) -> np.ndarray:
#     filled = np.where(valid_mask, block_values, 0.0).astype(np.float32)
#     weights = valid_mask.astype(np.float32)

#     for _ in range(4):
#         smooth_values = cv2.GaussianBlur(filled, (3, 3), 0)
#         smooth_weights = cv2.GaussianBlur(weights, (3, 3), 0)
#         candidate = np.where(
#             smooth_weights > 1e-6, smooth_values / (smooth_weights + 1e-6), default_value
#         )
#         filled = np.where(valid_mask, block_values, candidate)
#         weights = np.where(valid_mask, 1.0, smooth_weights)

#     return np.where(weights > 1e-6, filled, default_value).astype(np.float32)


# def estimate_frequency_and_confidence_field(
#     image: np.ndarray,
#     orientation: np.ndarray,
#     block_size: int,
#     min_wavelength: int = 5,
#     max_wavelength: int = 12,
#     default_wavelength: float = 8.0,
# ) -> tuple[np.ndarray, np.ndarray]:
#     height, width = image.shape[:2]
#     grid_h = max(1, int(np.ceil(height / block_size)))
#     grid_w = max(1, int(np.ceil(width / block_size)))
#     frequencies = np.full((grid_h, grid_w), default_wavelength, dtype=np.float32)
#     valid = np.zeros((grid_h, grid_w), dtype=bool)
#     confidence = np.zeros((grid_h, grid_w), dtype=np.float32)

#     radius = block_size // 2
#     image_padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT)
#     orientation_padded = cv2.copyMakeBorder(
#         orientation, radius, radius, radius, radius, cv2.BORDER_REFLECT
#     )

#     for gy in range(grid_h):
#         center_y = min(gy * block_size + (block_size // 2), height - 1)
#         for gx in range(grid_w):
#             center_x = min(gx * block_size + (block_size // 2), width - 1)
#             padded_y = center_y + radius
#             padded_x = center_x + radius
#             block = extract_centered_block(image_padded, padded_y, padded_x, block_size)
#             theta = float(orientation_padded[padded_y, padded_x])
#             wavelength, is_valid = estimate_block_wavelength(
#                 block, theta, min_wavelength, max_wavelength
#             )
#             frequencies[gy, gx] = wavelength
#             valid[gy, gx] = is_valid
#             confidence[gy, gx] = 1.0 if is_valid else 0.0

#     filled_blocks = fill_invalid_blocks(frequencies, valid, default_wavelength)
#     frequency_field = cv2.resize(filled_blocks, (width, height), interpolation=cv2.INTER_LINEAR)
#     confidence_field = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
#     return frequency_field.astype(np.float32), confidence_field.astype(np.float32)


# def estimate_frequency_field(
#     image: np.ndarray, orientation: np.ndarray, block_size: int
# ) -> np.ndarray:
#     frequency_field, _ = estimate_frequency_and_confidence_field(
#         image, orientation, block_size
#     )
#     return frequency_field


# def enhance_custom(image: np.ndarray) -> np.ndarray:
#     orientation_block = 16
#     filter_block = 32
#     stride = 8
#     inner_margin = filter_block // 4
#     default_wavelength = 8.0
#     blend_alpha = 0.4
#     detail_gain =100.0

#     mask = build_foreground_mask(image)
#     orientation = estimate_orientation_field(image, orientation_block, smooth_ksize=5)
#     _, _, coherence = estimate_orientation_components(image, orientation_block)
#     frequency, frequency_confidence = estimate_frequency_and_confidence_field(
#         image,
#         orientation,
#         orientation_block,
#         default_wavelength=default_wavelength,
#     )

#     image_float = image.astype(np.float32)
#     mask_float = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
#     coherence = coherence.clip(0.0, 1.0)
#     reliability = (0.65 * coherence) + (0.35 * frequency_confidence)
#     reliability *= mask_float

#     pad = inner_margin
#     image_padded = cv2.copyMakeBorder(
#         image_float, pad, pad, pad, pad, cv2.BORDER_REFLECT
#     )
#     orientation_padded = cv2.copyMakeBorder(
#         orientation, pad, pad, pad, pad, cv2.BORDER_REFLECT
#     )
#     frequency_padded = cv2.copyMakeBorder(
#         frequency, pad, pad, pad, pad, cv2.BORDER_REFLECT
#     )
#     reliability_padded = cv2.copyMakeBorder(
#         reliability, pad, pad, pad, pad, cv2.BORDER_REFLECT
#     )
#     mask_padded = cv2.copyMakeBorder(mask_float, pad, pad, pad, pad, cv2.BORDER_REFLECT)

#     accum = np.zeros_like(image_padded, dtype=np.float32)
#     count = np.zeros_like(image_padded, dtype=np.float32)

#     inner_size = filter_block - (2 * inner_margin)
#     weight = np.outer(np.hanning(inner_size), np.hanning(inner_size)).astype(np.float32)
#     weight += 1e-4

#     starts_y = sliding_starts(image_padded.shape[0], filter_block, stride)
#     starts_x = sliding_starts(image_padded.shape[1], filter_block, stride)

#     for y in starts_y:
#         for x in starts_x:
#             block = image_padded[y:y + filter_block, x:x + filter_block]
#             center_y = y + (filter_block // 2)
#             center_x = x + (filter_block // 2)
#             theta = float(orientation_padded[center_y, center_x])
#             wavelength = float(np.clip(frequency_padded[center_y, center_x], 5.0, 12.0))
#             block_reliability = float(reliability_padded[center_y, center_x])

#             inner_y0 = y + inner_margin
#             inner_y1 = y + filter_block - inner_margin
#             inner_x0 = x + inner_margin
#             inner_x1 = x + filter_block - inner_margin
#             source_center = image_padded[inner_y0:inner_y1, inner_x0:inner_x1]
#             center_mask = mask_padded[inner_y0:inner_y1, inner_x0:inner_x1]

#             if block_reliability < 0.08:
#                 accum[inner_y0:inner_y1, inner_x0:inner_x1] += source_center * weight
#                 count[inner_y0:inner_y1, inner_x0:inner_x1] += weight
#                 continue

#             block_mean = float(np.mean(block))
#             block_std = float(np.std(block))
#             if block_std < 1e-3:
#                 accum[inner_y0:inner_y1, inner_x0:inner_x1] += source_center * weight
#                 count[inner_y0:inner_y1, inner_x0:inner_x1] += weight
#                 continue

#             normalized_block = (block - block_mean) / block_std
#             sigma = max(2.0, 0.5 * wavelength)
#             ksize = int(np.ceil(6.0 * sigma))
#             if ksize % 2 == 0:
#                 ksize += 1
#             ksize = max(7, ksize)

#             kernel = cv2.getGaborKernel(
#                 (ksize, ksize),
#                 sigma,
#                 theta,
#                 wavelength,
#                 0.5,
#                 0,
#                 ktype=cv2.CV_32F,
#             )
#             kernel -= float(np.mean(kernel))
#             kernel_norm = float(np.sum(np.abs(kernel)))
#             if kernel_norm > 1e-6:
#                 kernel /= kernel_norm

#             response = cv2.filter2D(
#                 normalized_block, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT
#             )
#             center_response = response[
#                 inner_margin:filter_block - inner_margin,
#                 inner_margin:filter_block - inner_margin,
#             ]
#             enhanced_center = np.clip(
#                 source_center + (detail_gain * np.tanh(center_response)),
#                 0.0,
#                 255.0,
#             )

#             alpha_map = (blend_alpha * block_reliability * center_mask).astype(np.float32)
#             blended_center = (
#                 (1.0 - alpha_map) * source_center
#                 + alpha_map * enhanced_center
#             )
#             accum[inner_y0:inner_y1, inner_x0:inner_x1] += blended_center * weight
#             count[inner_y0:inner_y1, inner_x0:inner_x1] += weight

#     count = np.where(count > 0, count, 1.0)
#     enhanced = accum / count
#     enhanced = enhanced[pad:pad + image.shape[0], pad:pad + image.shape[1]]

#     background = mask_float < 0.5
#     enhanced[background] = image_float[background]
#     return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)


# def to_uint8_image(image: np.ndarray) -> np.ndarray:
#     if image.dtype == np.bool_:
#         return (image.astype(np.uint8) * 255)

#     image_float = image.astype(np.float32)
#     max_value = float(np.max(image_float)) if image_float.size else 0.0
#     min_value = float(np.min(image_float)) if image_float.size else 0.0

#     if 0.0 <= min_value and max_value <= 1.0:
#         image_float *= 255.0
#     elif min_value < 0.0 or max_value > 255.0:
#         image_float = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)

#     return np.clip(image_float, 0.0, 255.0).astype(np.uint8)


# def enhance_with_library(image: np.ndarray) -> np.ndarray | None:
#     try:
#         import fingerprint_enhancer

#         enhanced = fingerprint_enhancer.enhance_fingerprint(image)
#         return to_uint8_image(enhanced)
#     except ImportError:
#         pass
#     except Exception as exc:
#         print(f"Library enhancement via fingerprint_enhancer failed: {exc}", file=sys.stderr)

#     try:
#         import pyfing as pf

#         segmentation_mask = pf.fingerprint_segmentation(image)
#         orientations = pf.orientation_field_estimation(image, segmentation_mask)
#         frequencies = pf.frequency_estimation(image, orientations, segmentation_mask)
#         enhanced = pf.fingerprint_enhancement(
#             image,
#             orientations,
#             frequencies,
#             segmentation_mask,
#             method="GBFEN",
#         )
#         return to_uint8_image(enhanced)
#     except ImportError:
#         return None
#     except Exception as exc:
#         print(f"Library enhancement via pyfing failed: {exc}", file=sys.stderr)
#         return None


# def filter(input_path: Path, output_path: Path) -> None:
#     image = load_gray_image(input_path)
#     save_png(output_path, enhance_custom(image))


# def main() -> int:
#     args = parse_args()
#     try:
#         nobg_path, cropped_path = process_image(args.input_image)
#         normalised_path = Path("normalised_cropped.png")
#         custom_path = Path("filtered_cropped_custom.png")
#         library_path = Path("filtered_cropped_library.png")

#         normalise_brightness(nobg_path, normalised_path)
#         normalised_image = load_gray_image(normalised_path)

#         custom_image = enhance_custom(normalised_image)
#         save_png(custom_path, custom_image)

#         library_written = False
#         if not (args.skip_library or args.custom_only):
#             library_image = enhance_with_library(normalised_image)
#             if library_image is not None:
#                 save_png(library_path, library_image)
#                 library_written = True
#     except Exception as exc:
#         print(f"Error: {exc}", file=sys.stderr)
#         return 1

#     print(f"Saved background-removed image: {nobg_path}")
#     print(f"Saved cropped fingerprint image: {cropped_path}")
#     print(f"Saved normalised fingerprint image: {normalised_path}")
#     print(f"Saved custom enhanced fingerprint image: {custom_path}")
#     if args.skip_library or args.custom_only:
#         print("Skipped library-based enhancement output.")
#     elif library_written:
#         print(f"Saved library enhanced fingerprint image: {library_path}")
#     else:
#         print("Skipped library-based enhancement output because no optional package is installed.")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
