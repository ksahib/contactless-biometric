# Ground Truth Generation: Code, Theory, Reconstruction, Unwarping, and Reprojection

This document explains exactly what the current ground-truth generation pipeline does in this repository. The main implementation is in `generate_ground_truth.py`; the center-line surface unwarping math lives in `center_unwarping.py`; DS1/DS2/DS3 batch generation and merge orchestration lives in `scripts/generate_ds123_and_merge.sh`; and optional minutiae-on-3D visualization support lives in `visualize_reprojected_minutiae.py`.

The short version is:

1. Read the raw contactless finger dataset and build a manifest of samples.
2. Preprocess each raw contactless image into a normalized grayscale image, a foreground mask, and a pose-normalized frame.
3. When a front/left/right triplet exists for the same subject, finger, and acquisition, reconstruct an approximate 3D finger surface using per-row silhouette widths.
4. Smooth reconstructed depth maps with moving least-squares quadratic fits and save depth-gradient labels.
5. Center-unwarp the reconstructed front surface into a flatter canonical image using arc-length coordinates derived from the reconstructed surface gradient.
6. Extract minutiae once on that canonical unwarped image.
7. Map those canonical minutiae back through the inverse unwarp to the front reconstruction, then project them into the front, left, or right sample view.
8. If reconstruction-backed minutiae fail or no triplet exists, fall back to direct per-sample minutiae extraction.
9. Build FeatureNet targets: orientation, ridge period, reconstruction gradient, minutia score, sub-cell offsets, orientation labels, and masks.
10. Save each sample bundle plus reconstruction artifacts and metadata.

The pipeline is approximate. It does not claim to produce physical ground truth from a calibrated camera rig. It produces consistent, geometry-aware supervision from contactless image triplets using silhouette reconstruction, surface flattening, and reprojection.

## Important Files

- `generate_ground_truth.py`
  - Builds the dataset manifest.
  - Preprocesses raw images.
  - Reconstructs acquisition-level front/left/right surface labels.
  - Extracts or reprojects minutiae.
  - Builds and writes FeatureNet target tensors.
  - Merges shards and DS1/DS2/DS3 roots.
- `center_unwarping.py`
  - Selects a low-gradient center point.
  - Integrates surface arc-length coordinates.
  - Forward-splats pixels into a canonical unwarped image.
  - Builds the maps used later for inverse unwarping.
- `visualize_reprojected_minutiae.py`
  - Loads reconstruction artifacts.
  - Maps canonical minutiae back to the 3D reconstructed surface.
  - Writes overlay/visual diagnostics.
- `scripts/generate_ds123_and_merge.sh`
  - Runs `generate_ground_truth.py` separately for `DS1`, `DS2`, and `DS3`.
  - Merges the generated roots into one unique identity space.

## Output Layout

A generated dataset root looks like:

```text
ground_truth/<DATASET>/
  manifest.json
  manifest.csv
  summary.json
  samples/
    <sample_id>/
      raw_input.png
      preprocess_normalized.png
      preprocess_pose_normalized.png
      preprocess_pose_mask.png
      preprocessed_input.png
      preprocess_mask.png
      mask.png
      masked_image.png
      minutiae_enhanced.png
      orientation.npy
      ridge_period.npy
      gradient_visualization.npy
      minutiae.json
      featurenet_targets.npz
      meta.json
      preview.png               # only for requested visualization samples
  reconstructions/
    <acquisition_id>/
      depth_front.npy
      depth_left.npy
      depth_right.npy
      depth_gradient_labels.npz
      reconstruction_maps.npz
      support_mask.png
      row_measurements.json
      preview.png
      center_unwarped.png
      center_unwarped_mask.png
      center_unwarp_maps.npz
      canonical_unwarped_minutiae.json
      center_unwarped_enhanced.png
      surface_front_3d.html
      surface_front_3d.png
      surface_all_branches_3d.html
      surface_all_branches_3d.png
      reprojection_report.json
      reprojection_preview.png
      meta.json
      debug_views/
```

`samples/<sample_id>/featurenet_targets.npz` is the actual training-label bundle. `reconstructions/<acquisition_id>/...` stores shared acquisition-level geometry used to create those labels.

## Dataset Manifest

The generator starts in `main()` by calling `_build_manifest(dataset_root)`.

The dataset is assumed to be organized by numeric subject folders. For every raw view image found, the manifest records:

- `sample_id`: unique sample name.
- `subject_id`: original numeric subject ID.
- `subject_index`: zero-based subject index in sorted subject folders.
- `finger_id`: finger identifier.
- `acquisition_id`: acquisition identifier.
- `finger_class_id`: class ID used by training.
- `raw_image_path`: path to this exact raw image.
- `raw_view_index`: view suffix parsed from the image name.
- `raw_view_paths`: all raw view paths for the same acquisition.
- `variant_paths`: paths for known variant suffixes such as `HT1`, `HT2`, `HT4`, `HT6`, and `R414`.
- `sire_path`: optional paired SIRE path.
- `is_extra_acquisition`: whether acquisition index is greater than 2.

The manifest is written by `_write_manifest()` to both JSON and CSV. All later stages use this manifest rather than rediscovering files ad hoc.

## Runtime and Model Setup

`generate_ground_truth.py` supports both local and Kaggle-oriented execution.

The CLI options include:

- `--dataset-root`
- `--output-root`
- `--execution-target local|kaggle`
- `--gpu-only`
- `--cpu-workers`
- `--prefetch-samples`
- `--skip-existing`
- `--fingerflow-backend auto|local|wsl`
- `--visualize-count`
- `--smoke-samples`
- sharding options
- patch dataset options
- merge options

`_configure_runtime_environment()` sets TensorFlow and ONNX runtime behavior:

- TensorFlow memory growth is enabled when possible.
- TensorFlow GPU visibility is checked for FingerFlow.
- ONNX providers are checked for `rembg`.
- In Kaggle GPU mode with `--gpu-only`, missing GPU providers are treated as errors.

FingerFlow model files are managed by `ensure_fingerflow_models()`. If they are absent, `_ensure_model_file()` downloads them atomically into `.fingerflow_models`.

On Windows, `--fingerflow-backend auto` chooses WSL when `wsl` is available. Otherwise it uses the local backend.

## Per-Image Preprocessing

The core preprocessing function is `_preprocess_contactless_bgr(full_bgr)`.

It returns a `PreprocessedContactlessImage` containing:

- `raw_gray`
- `normalized_gray`
- `pose_normalized_gray`
- `pose_normalized_mask`
- `preprocessed_gray`
- `final_mask`
- `mask_source`
- `pose_rotation_degrees`
- `ridge_scale_factor`

### Step 1: Convert to Grayscale

The raw BGR image is converted to grayscale:

```python
raw_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
```

This is the intensity image used for contrast normalization, pose normalization, orientation estimation, ridge frequency estimation, enhancement, and visualization.

### Step 2: Foreground Mask

`rembg_mask_from_bgr()` builds a binary finger foreground mask.

If `rembg` is installed:

1. Convert BGR to RGB.
2. Run `rembg.remove(..., only_mask=True, post_process_mask=True)`.
3. Threshold the mask at 127.
4. Morphologically close with a `7 x 7` kernel for two iterations.
5. Morphologically open with a `5 x 5` kernel for one iteration.
6. Keep only the largest connected contour.

If `rembg` is unavailable, `_fallback_mask_from_bgr()` is used:

1. Convert to grayscale.
2. Apply a `5 x 5` Gaussian blur.
3. Build both Otsu bright and Otsu dark binary masks.
4. Choose the mask whose largest contour occupies the larger image ratio.
5. Keep only the largest contour.

The code then calls `validate_foreground_area(initial_mask, minimum_ratio=0.03)`. If the mask covers less than 3 percent of the image, generation fails for that sample because the segmentation is too small to plausibly represent a finger.

### Step 3: Brightness Normalization

`normalise_brightness_array(gray_image, mask)` applies contrast normalization only inside the foreground bounding box:

1. Find the foreground bounding rectangle from mask coordinates.
2. Crop the grayscale image to that rectangle.
3. Apply CLAHE with `clipLimit=4.0`, `tileGridSize=(4, 4)`.
4. Apply a second CLAHE with `clipLimit=1.0`, `tileGridSize=(10, 10)`.
5. Place the normalized crop back into a full-size image.
6. Set all background pixels to zero.

This produces `normalized_gray`.

The theory is local contrast equalization: fingerprint ridges are high-frequency local structures, and CLAHE improves local contrast without globally over-amplifying all intensity differences. The second gentler CLAHE pass stabilizes contrast after the stronger first pass.

### Step 4: Pose Normalization

`_estimate_pose_rotation(mask)` estimates how much to rotate the finger so the long axis becomes vertical.

The code:

1. Finds the largest foreground contour.
2. Calls `estimate_finger_axis(contour)`.
3. `estimate_finger_axis()` uses `cv2.minAreaRect()` and `cv2.boxPoints()` to get the minimum-area rotated rectangle around the finger.
4. It chooses the longer rectangle edge as the finger axis.
5. It converts that axis to an angle with `atan2`.
6. It wraps the angle into `[-90, 90]`.
7. It returns `-90 - angle`, so the long axis rotates toward vertical.

`_rotate_array()` then applies `cv2.warpAffine()` around the image center:

- grayscale uses bilinear interpolation,
- mask uses nearest-neighbor interpolation,
- outside pixels are filled with zero.

The pose-normalized mask is re-binarized to `0` or `255`, then validated with a 2 percent minimum area check.

This pose-normalized mask is especially important for reconstruction. The reconstruction assumes rows are comparable cross-sections along the finger length.

### Step 5: Ridge Frequency Normalization

`_normalize_ridge_frequency()` estimates central ridge spacing and rescales the image so the approximate ridge wavelength becomes 10 pixels.

`_estimate_central_ridge_spacing()`:

1. Finds the mask centroid.
2. Takes a central `128 x 128` patch where possible.
3. Requires at least 35 percent foreground in that patch.
4. Replaces background patch pixels with the foreground mean.
5. Computes a 2D FFT.
6. Uses `fftshift()` and suppresses the central low-frequency component.
7. Finds the largest non-DC frequency peak.
8. Converts frequency magnitude to wavelength: `wavelength = 1 / freq`.
9. Clips the wavelength to `[3, 25]` pixels.

The scale factor is:

```text
scale = target_spacing / measured_spacing
```

then clipped to `[0.5, 2.5]`. If the scale is within 2 percent of 1.0, no resizing is performed. Otherwise the image is resized bilinearly and the mask by nearest neighbor.

The outputs are:

- `preprocessed_gray`
- `final_mask`
- `ridge_scale_factor`

Important distinction:

- reconstruction uses the `pose_normalized_*` image and mask before ridge-frequency scaling;
- FeatureNet sample labels use the final `preprocessed_gray` and `final_mask` after ridge-frequency scaling.

This is why reprojection later multiplies projected coordinates by `preprocessed.ridge_scale_factor`.

## Orientation, Ridge Period, and Debug Gradients

After a sample is loaded, `_prepare_bundle_from_loaded()` computes dense per-image labels.

### Orientation

The code calls:

```python
pyfing.orientation_field_estimation(gray_image, mask, dpi=dpi, method="SNFOE")
```

The result is normalized into `[0, pi)` by `_normalize_angle_pi()`, and background pixels are set to zero.

Fingerprint orientation is undirected: an angle `theta` and `theta + pi` represent the same ridge flow. That is why the orientation target lives in `[0, pi)`, not `[0, 2pi)`.

When the orientation is resized for FeatureNet, `_resize_orientation_for_model()` does not directly average angles. It converts orientation into a doubled-angle vector:

```text
cos2 = cos(2 theta)
sin2 = sin(2 theta)
```

Then it resizes `cos2` and `sin2` and reconstructs:

```text
theta_small = 0.5 * atan2(sin2_small, cos2_small)
```

This avoids the wrap-around error where angles near 0 and pi would average incorrectly.

Finally `_build_orientation_one_hot()` quantizes the resized orientation into 180 bins:

```text
bin = floor(theta * 180 / pi)
```

The final orientation target shape is `[180, Ho, Wo]`.

### Ridge Period

The code calls:

```python
pyfing.frequency_estimation(gray_image, orientation, mask, dpi=dpi, method="SNFFE")
```

Invalid values are replaced with zero, negative periods are clipped to zero, and background is zeroed.

For FeatureNet, the ridge period is resized to the output grid, multiplied by the small mask, and normalized by its maximum positive value in the image. The target shape is `[1, Ho, Wo]`.

### Visualization Gradient

`_compute_gradient(gray_image, mask)` computes a Sobel image-intensity gradient:

```text
grad_x = Sobel(I, dx=1, dy=0)
grad_y = Sobel(I, dx=0, dy=1)
```

This is saved as `gradient_visualization.npy` and used in preview images. It is not the reconstruction depth-gradient target. The actual training gradient target, when available, comes from the multiview reconstruction and is stored in `featurenet_targets.npz` under `gradient`.

## Acquisition-Level Multiview Reconstruction

The reconstruction stage is run per acquisition, not per sample, because a single subject/finger/acquisition can have multiple views.

`_collect_reconstruction_candidates(samples)` finds acquisitions whose `raw_view_paths` include view suffixes `0`, `1`, and `2`.

`_resolve_reconstruction_triplet(raw_view_paths)` maps:

- raw view `0` to `front`,
- raw view `1` to `left`,
- raw view `2` to `right`.

`_reconstruct_multiview_acquisition(sample, output_root)` then builds the shared reconstruction.

### Reconstruction Input Space

Each of the three views is preprocessed by `_extract_reconstruction_view_geometry(role, raw_image_path)`.

This function calls `_preprocess_contactless_raw()`, but it uses the `pose_normalized_mask`, not the ridge-scaled final mask.

For each image row `y`, it computes:

```text
x_left[y]  = first foreground column
x_right[y] = last foreground column
width[y]   = x_right[y] - x_left[y] + 1
center[y]  = (x_left[y] + x_right[y]) / 2
valid[y]   = row has at least one foreground pixel
```

Rows are assumed to be cross-sections of the finger. This is why pose normalization matters: without a vertical finger axis, row widths would not correspond cleanly to cross-sectional widths.

The reconstruction requires the front, left, and right pose-normalized masks to share the same image shape. If shapes differ, the acquisition reconstruction fails.

### Elliptical Cross-Section Model

The reconstruction treats each row as an approximate elliptical cross-section.

For a row `y`, the front-view half-width is:

```text
a[y] = front_width[y] / 2
```

The side-view half-widths are:

```text
d_right[y] = right_width[y] / 2
d_left[y]  = left_width[y] / 2
```

The code uses the formula:

```text
d = sqrt((a^2 + b^2) / 2)
```

Solving for the unknown semi-minor axis `b` gives:

```text
b = sqrt(max(2 d^2 - a^2, 0))
```

So the code computes:

```text
b_right = sqrt(max(2 d_right^2 - a^2, 0))
b_left  = sqrt(max(2 d_left^2  - a^2, 0))
b       = 0.5 * (b_right + b_left)
```

The `max(..., 0)` prevents numerical or silhouette inconsistencies from producing negative square roots.

Theory: If the side views are approximately 45-degree views of an elliptical finger cross-section, the apparent half-width in the side view is a combination of the front horizontal radius `a` and depth radius `b`. Averaging the left-derived and right-derived `b` stabilizes asymmetric segmentation noise.

### Row Coordinates

For every pixel column `x_pixel` in the front pose-normalized frame:

```text
x_relative[y, x] = x_pixel - front_center[y]
```

This puts each row into a local coordinate system centered on the front silhouette center.

The support mask is the front pose-normalized foreground mask:

```python
support_mask = preprocessed_views["front"].pose_normalized_mask > 0
```

Pixels outside this support are forced to zero in all depth maps.

### Front Surface Depth

For each row, the upper ellipse branch is:

```text
z_up[y, x] = (b[y] / a[y]) * sqrt(max(a[y]^2 - x_relative[y, x]^2, 0)) + c_z[y]
```

The lower branch is:

```text
z_down[y, x] = -(b[y] / a[y]) * sqrt(max(a[y]^2 - x_relative[y, x]^2, 0)) + c_z[y]
```

The front depth label saved as `depth_front.npy` is `z_up`.

`c_z[y]` is a row-wise depth translation. The code estimates it from the front and side row centers:

```text
c_z_right = (sqrt(2) / 2) * (front_center + right_center)
c_z_left  = (sqrt(2) / 2) * (front_center + left_center)
c_z       = 0.5 * (c_z_right + c_z_left)
```

This centers the reconstructed cross-section consistently in the shared coordinate frame using the 45-degree side-view assumption.

### Critical Point and Side Branch Stitching

The side branches need a piecewise choice between the upper and lower ellipse branch. The code computes:

```text
theta  = atan(b^2 / a^2)
x_crit = a * cos(theta)
```

Then it builds stitched branches before rotation:

```text
stitched_right = z_down when x_relative <  x_crit, otherwise z_up
stitched_left  = z_up   when x_relative <= -x_crit, otherwise z_down
```

These piecewise definitions choose the branch that should be visible from each side direction and avoid using one continuous front-only surface for side views.

### Rotating Side Branches

`_rotate_depth_branch(x_coords, depth, angle_degrees)` rotates points in the `(x, z)` plane:

```text
x_rot = x cos(theta) + z sin(theta)
z_rot = -x sin(theta) + z cos(theta)
```

The code applies:

```text
depth_right = rotate(stitched_right, +45 degrees).z_rot
depth_left  = rotate(stitched_left,  -45 degrees).z_rot
```

Only the rotated `z` component is saved as `depth_right.npy` and `depth_left.npy`. The rotated x coordinate is used in visualization and projection logic where needed.

### Saved Reconstruction Maps

`reconstruction_maps.npz` contains:

- `support_mask`: binary support mask.
- `x_relative`: front-centered x coordinate per pixel.
- `depth_front`: front branch depth.
- `depth_left`: left branch depth.
- `depth_right`: right branch depth.
- `stitched_left`: pre-rotation left branch depth.
- `stitched_right`: pre-rotation right branch depth.
- `front_centers`: interpolated row centers for the front view.
- `left_centers`: interpolated row centers for the left view.
- `right_centers`: interpolated row centers for the right view.
- `x_crit`: row-wise critical x value.

These maps are used later to project canonical minutiae into a target sample view.

## Depth Smoothing and Reconstruction Gradient Labels

The network gradient target is not computed from raw image intensity. It is computed from reconstructed depth.

`_build_depth_gradient_labels()` runs `_smooth_depth_with_quadratic_mls()` separately on:

- `depth_front`
- `depth_left`
- `depth_right`

### Moving Least-Squares Surface Fit

For each foreground pixel `(x0, y0)`, `_smooth_depth_with_quadratic_mls()` considers a local window with:

- `radius = 5`
- `sigma = 3.0`

It fits a weighted quadratic surface:

```text
z(x, y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2
```

where `x` and `y` are local offsets relative to the center pixel.

The design matrix row for a neighbor is:

```text
[1, x, y, x^2, x y, y^2]
```

The Gaussian weight for an offset is:

```text
w = exp(-(x^2 + y^2) / (2 sigma^2))
```

The code solves a weighted least-squares problem:

```text
min_c sum_i w_i^2 * (A_i c - z_i)^2
```

implemented as:

```python
weighted_design = design * weights[:, None]
weighted_target = z_values * weights
coeffs = np.linalg.lstsq(weighted_design, weighted_target)
```

At the center pixel, local coordinates are `(0, 0)`, so:

```text
smoothed_depth = c0
grad_x = partial z / partial x at center = c1
grad_y = partial z / partial y at center = c2
```

If there are fewer than six local support points, or the least-squares rank is too low, the code keeps the original depth and leaves gradients at zero for that pixel.

The saved `depth_gradient_labels.npz` contains:

- `depth_front_smooth`
- `depth_left_smooth`
- `depth_right_smooth`
- `gradient_front` with shape `[2, H, W]`
- `gradient_left` with shape `[2, H, W]`
- `gradient_right` with shape `[2, H, W]`
- `support_mask` with shape `[1, H, W]`

For each sample, `_load_reconstruction_gradient_for_sample()` selects the gradient by raw view:

```text
raw_view_index 0 -> gradient_front
raw_view_index 1 -> gradient_left
raw_view_index 2 -> gradient_right
```

It transposes the gradient to `[H, W, 2]` for internal use. Later `_build_featurenet_targets()` resizes it to the FeatureNet output grid and saves it as `[2, Ho, Wo]`.

## Center Unwarping

The unwarping implementation is in `center_unwarping.py`, with the public entry point:

```python
run_center_unwarping(image, mask, gradient_x, gradient_y)
```

The generator calls it with:

- `image = front pose-normalized grayscale image`
- `mask = front reconstruction support mask`
- `gradient_x = gradient_front[0]`
- `gradient_y = gradient_front[1]`

The goal is to produce `center_unwarped.png`: a canonical flattened front image where distances follow approximate surface arc length instead of raw image-plane distances.

### Why Arc-Length Unwarping?

The reconstructed front surface can be thought of as:

```text
S(x, y) = (x, y, z(x, y))
```

If we move horizontally by a small image-plane amount `dx`, the corresponding surface displacement is:

```text
dS = (dx, 0, z_x dx)
```

Its length is:

```text
|dS| = sqrt(dx^2 + (z_x dx)^2)
     = sqrt(1 + z_x^2) |dx|
```

Similarly, vertical movement has approximate surface length:

```text
sqrt(1 + z_y^2) |dy|
```

The code uses these local metric factors:

```text
s(x, y) = sqrt(1 + gradient_x(x, y)^2)
t(x, y) = sqrt(1 + gradient_y(x, y)^2)
```

Then it integrates them along rows and columns to build flattened coordinates.

This is an approximation because it ignores the full mixed metric tensor term caused by both gradients together. A fully general surface parameterization would use:

```text
ds^2 = (1 + z_x^2) dx^2 + 2 z_x z_y dx dy + (1 + z_y^2) dy^2
```

The implemented method deliberately uses separable row/column integration because it is simple, stable, and produces a usable canonical image for minutiae extraction.

### Center Point Selection

`compute_center_point_from_gradients()` computes:

```text
gradient_magnitude = sqrt(gradient_x^2 + gradient_y^2)
```

inside the mask and chooses the pixel with the smallest gradient magnitude.

This point is used as the coordinate origin for unwarping:

```text
u(center) = 0
v(center) = 0
```

Theory: a low-gradient point is locally flattest and therefore a stable anchor. Centering the unwarp there reduces accumulated distortion around the least-curved region.

### Row Integration

`compute_arc_length_maps()` creates:

```text
s = sqrt(1 + gradient_x^2)
t = sqrt(1 + gradient_y^2)
```

`_integrate_rows(s, mask, center_x)` computes `u[y, x]`.

For each row:

1. Find foreground columns.
2. Use `center_x` as the seed if it lies in that row's mask.
3. Otherwise choose the valid column nearest `center_x` and record that row as a fallback row.
4. Set the seed value to zero.
5. Integrate to the right with the trapezoidal rule:

```text
u[y, x+1] = u[y, x] + 0.5 * (s[y, x] + s[y, x+1])
```

6. Integrate to the left:

```text
u[y, x-1] = u[y, x] - 0.5 * (s[y, x] + s[y, x-1])
```

7. Stop integration if either neighboring pixel leaves the mask.

The trapezoidal rule approximates:

```text
u(x, y) = integral from x_seed to x of sqrt(1 + z_x(xi, y)^2) dxi
```

### Column Integration

`_integrate_cols(t, mask, center_y)` computes `v[y, x]`.

For each column:

1. Find foreground rows.
2. Use `center_y` as the seed if it lies in that column's mask.
3. Otherwise choose the valid row nearest `center_y` and record that column as a fallback column.
4. Set the seed to zero.
5. Integrate downward:

```text
v[y+1, x] = v[y, x] + 0.5 * (t[y, x] + t[y+1, x])
```

6. Integrate upward:

```text
v[y-1, x] = v[y, x] - 0.5 * (t[y, x] + t[y-1, x])
```

This approximates:

```text
v(x, y) = integral from y_seed to y of sqrt(1 + z_y(x, eta)^2) deta
```

### New Coordinates

`build_unwarp_coordinates()` converts arc-length coordinates into a new coordinate frame:

```text
x_new = center_x + u
y_new = center_y + v
```

It computes the bounding box of valid `(x_new, y_new)` points:

```text
x0 = floor(min(x_new))
x1 = ceil(max(x_new))
y0 = floor(min(y_new))
y1 = ceil(max(y_new))
```

Then it creates output-image coordinates:

```text
x_out = x_new - x0
y_out = y_new - y0
```

The output image shape is:

```text
height_out = y1 - y0 + 1
width_out  = x1 - x0 + 1
```

The unwarp map therefore contains both absolute flattened coordinates (`x_new`, `y_new`) and image-local output coordinates (`x_out`, `y_out`).

### Forward Bilinear Splatting

`resample_unwarped_image()` performs forward mapping, also called splatting.

For every valid source pixel:

1. Read its target location `(sample_x, sample_y) = (x_out, y_out)`.
2. Compute:

```text
x_floor = floor(sample_x)
y_floor = floor(sample_y)
dx = sample_x - x_floor
dy = sample_y - y_floor
```

3. Distribute the source intensity to the four neighboring target pixels:

```text
(x0, y0): (1 - dx)(1 - dy)
(x1, y0): dx(1 - dy)
(x0, y1): (1 - dx)dy
(x1, y1): dx dy
```

4. Accumulate weighted intensity and accumulated weight using `np.add.at`.
5. Divide accumulated intensity by accumulated weight wherever weight is nonzero.

This is the forward equivalent of bilinear interpolation. It avoids holes caused by directly assigning a source pixel to only one nearest output pixel.

Some holes can still occur. `_fill_small_holes()` fills small isolated holes for at most two iterations. A hole is filled only when at least two 4-connected neighbors are valid. The fill value is the mean of those valid neighbors.

The final outputs are:

- `center_unwarped.png`
- `center_unwarped_mask.png`
- `center_unwarp_maps.npz`

### Inverse Unwarp Maps

The reconstruction code calls `_build_inverse_unwarp_maps(x_out, y_out, valid_mask, output_shape)`.

This builds:

- `source_x_map`
- `source_y_map`
- `source_valid_mask`

For each source pixel that was forward-splatted:

1. Consider the four integer output pixels around its floating target coordinate.
2. Compute squared distance from the floating coordinate to each candidate integer output pixel.
3. For each output pixel, keep the source pixel with the smallest squared distance.

This creates a nearest-source inverse map from unwarped coordinates back to front-source coordinates.

Later, `_map_unwarped_to_front_source(unwarp_maps, x, y)` samples `source_x_map` and `source_y_map` with bilinear interpolation. This lets a minutia at non-integer unwarped coordinates map back to subpixel front coordinates.

## Canonical Unwarped Minutiae Extraction

The pipeline extracts minutiae once per reconstruction on the unwarped front image.

`_load_or_extract_canonical_reconstruction_minutiae()`:

1. Loads `center_unwarped.png`.
2. Enhances it with `_enhance_for_minutiae()`.
3. Saves `center_unwarped_enhanced.png`.
4. Runs FingerFlow on the enhanced canonical image if possible.
5. Falls back to `pyfing.minutiae_extraction()` if FingerFlow fails.
6. Standardizes all minutiae into:

```json
{
  "x": 0.0,
  "y": 0.0,
  "theta": 0.0,
  "score": null,
  "type": null,
  "source": "..."
}
```

7. Saves them to `canonical_unwarped_minutiae.json`.
8. Caches them in `_RECONSTRUCTION_MINUTIAE_CACHE` by acquisition ID.

This is a key design choice. Instead of extracting minutiae separately in each distorted contactless view, the pipeline extracts them from a single flattened canonical representation and then reprojects them consistently into each view.

## Reprojection of Canonical Minutiae into Sample Views

The reprojection implementation is `_remap_unwarped_minutiae_to_sample()`.

It maps each canonical minutia through this chain:

```text
canonical unwarped coordinate
  -> front pose-normalized source coordinate
  -> reconstructed local front coordinate/depth
  -> requested view pose-normalized coordinate
  -> requested view training coordinate after ridge scaling
```

### Step 1: Canonical Unwarped to Front Source

For a minutia `(x_u, y_u)` in the unwarped image:

```python
source_point = _map_unwarped_to_front_source(unwarp_maps, x_u, y_u)
```

This samples:

```text
x_front = bilinear(source_x_map, x_u, y_u)
y_front = bilinear(source_y_map, x_u, y_u)
```

The source maps were produced from the inverse of the forward unwarp. If sampling fails or falls outside `source_valid_mask`, the minutia is dropped.

### Step 2: Front Source to Requested Pose Frame

`_project_front_source_to_pose_frame(reconstruction_maps, role, x_front, y_front)` does the view projection.

It first samples:

```text
x_relative = bilinear(reconstruction_maps["x_relative"], x_front, y_front)
```

For the front view:

```text
x_pose = front_center(y_front) + x_relative
y_pose = y_front
```

`front_center(y_front)` is obtained by `_sample_1d_linear()` from `front_centers`.

For left and right views, it also samples:

```text
depth_front = bilinear(reconstruction_maps["depth_front"], x_front, y_front)
```

Then it rotates the `(x_relative, depth_front)` point by the side view angle:

```text
angle = -45 degrees for left
angle = +45 degrees for right

x_rot = x_relative cos(angle) + depth_front sin(angle)
```

Then it shifts by the side-view row center:

```text
x_pose = side_center(y_front) + x_rot
y_pose = y_front
```

This is effectively a weak orthographic projection of the reconstructed front surface into the side view's pose-normalized image frame.

### Step 3: Pose Frame to Training Frame

The sample's training image may have been ridge-frequency scaled after pose normalization. `_project_front_source_to_training_frame()` applies:

```text
x_train = x_pose * ridge_scale_factor
y_train = y_pose * ridge_scale_factor
```

This aligns the reprojected point with `preprocessed_gray` and `final_mask`, which are the arrays saved in the sample bundle and used for FeatureNet labels.

### Step 4: Bounds and Mask Support Checks

A reprojected minutia is kept only if:

- the projected coordinate is finite,
- it lies inside the preprocessed image bounds,
- `_point_has_mask_support(final_mask, x, y, radius=2)` finds some foreground support nearby.

The radius check prevents labels from landing just outside the final mask because of interpolation, scaling, or segmentation differences.

### Step 5: Reprojecting Minutia Orientation

The canonical minutia has an orientation `theta`.

To transform orientation through the nonlinear unwarp/projection chain, the code does not simply reuse `theta`. It samples two nearby points along the canonical orientation direction:

```text
delta = 4 pixels
forward_u  = (x_u + cos(theta) delta, y_u + sin(theta) delta)
backward_u = (x_u - cos(theta) delta, y_u - sin(theta) delta)
```

Each neighbor is mapped through the same chain:

```text
unwarped -> front source -> requested training frame
```

If both projected neighbor points are valid, the new orientation is:

```text
theta_dst = atan2(y_forward - y_backward, x_forward - x_backward)
```

If only one projected neighbor is valid:

```text
theta_dst = atan2(y_neighbor - y_center, x_neighbor - x_center)
```

If neither neighbor is valid, it keeps the original canonical `theta`.

Finally the angle is normalized into `[0, 2pi)` by `_normalize_angle_2pi_scalar()`.

This finite-difference orientation transport is important because unwarping and reprojection can rotate local direction vectors.

### Reprojection Output

For every kept minutia, the code writes:

```json
{
  "x": x_train,
  "y": y_train,
  "theta": theta_dst_wrapped_to_2pi,
  "score": original_score,
  "type": original_type,
  "source": original_source
}
```

The sample's `meta.json` records:

- `minutiae_ground_truth.mode = "reconstruction_backed"`
- canonical source
- view role
- canonical and reprojected minutiae counts
- paths to the unwarp and reconstruction maps

If reconstruction-backed minutiae produce no usable points, the generator falls back to direct per-sample extraction.

## Direct Per-Sample Minutiae Fallback

If there is no reconstruction for a sample, or the reconstruction-backed path fails, the code extracts minutiae directly from that sample.

The path is:

1. Save `minutiae_enhanced.png`.
2. Try FingerFlow through `_extract_fingerflow_minutiae()` or `_extract_fingerflow_minutiae_wsl()`.
3. If FingerFlow fails, call `_extract_pyfing_minutiae(enhanced_image)`.
4. Standardize the output fields with `_standardize_minutiae()`.

The fallback mode is recorded in `meta.json`.

## FeatureNet Target Tensor Generation

`_build_featurenet_targets()` creates the final training tensors saved in `featurenet_targets.npz`.

The output grid is:

```text
Ho = floor(input_height / 8), minimum 1
Wo = floor(input_width / 8), minimum 1
```

This matches the FeatureNet output stride.

### Output Mask

The foreground mask is resized by nearest neighbor:

```text
output_mask shape = [1, Ho, Wo]
```

All dense and minutia targets are masked by this small mask.

### Orientation Target

As described earlier:

1. Resize orientation with doubled-angle vectors.
2. Quantize into 180 bins.
3. One-hot encode.
4. Apply output mask.

Saved as:

```text
orientation shape = [180, Ho, Wo]
```

### Ridge Period Target

The ridge-period map is resized linearly, masked, and normalized by its image maximum.

Saved as:

```text
ridge_period shape = [1, Ho, Wo]
```

### Reconstruction Gradient Target

If reconstruction gradient is available for the sample's raw view:

1. Resize `gradient[:, :, 0]` linearly to `[Ho, Wo]`.
2. Resize `gradient[:, :, 1]` linearly to `[Ho, Wo]`.
3. Multiply each by `output_mask`.
4. Stack into `[2, Ho, Wo]`.

Saved as:

```text
gradient shape = [2, Ho, Wo]
```

This is the MLS-smoothed depth derivative target, not the Sobel intensity gradient.

### Minutiae Rasterization

`_rasterize_minutiae()` converts a list of minutiae into grid labels.

For each minutia `(x, y, theta)` in the full-resolution preprocessed image:

```text
cell_x = floor(x * Wo / input_width)
cell_y = floor(y * Ho / input_height)
```

The cell indices are clipped to valid bounds.

If the corresponding small mask cell is background, the minutia is ignored.

The source image footprint of one output cell is:

```text
cell_width  = input_width  / Wo
cell_height = input_height / Ho
```

The local continuous offset inside that cell is:

```text
local_x = (x - cell_x * cell_width) / cell_width
local_y = (y - cell_y * cell_height) / cell_height
```

Both are clipped to `[0, 1)`.

If multiple minutiae land in the same output cell, the code keeps one using:

1. higher score wins;
2. if scores tie, the minutia closer to the cell center wins.

The center distance is:

```text
(local_x - 0.5)^2 + (local_y - 0.5)^2
```

The rasterized outputs are:

- `minutia_score`: binary score heatmap, shape `[1, Ho, Wo]`.
- `minutia_valid_mask`: where a minutia label exists, shape `[1, Ho, Wo]`.
- `minutia_x`: legacy 8-bin x offset class, shape `[Ho, Wo]`.
- `minutia_y`: legacy 8-bin y offset class, shape `[Ho, Wo]`.
- `minutia_x_offset`: continuous x offset, shape `[1, Ho, Wo]`.
- `minutia_y_offset`: continuous y offset, shape `[1, Ho, Wo]`.
- `minutia_orientation`: legacy 360-bin orientation class, shape `[Ho, Wo]`.
- `minutia_orientation_vec`: continuous `[cos(theta), sin(theta)]`, shape `[2, Ho, Wo]`.

The legacy subcell bins are:

```text
minutia_x_bin = floor(local_x * 8)
minutia_y_bin = floor(local_y * 8)
```

The legacy orientation bin is:

```text
orientation_bin = floor(theta * 360 / (2 pi))
```

The continuous orientation target is:

```text
[cos(theta), sin(theta)]
```

Unlike ridge orientation, minutia direction is directional, so it uses `[0, 2pi)`.

## Bundle Persistence

`_persist_bundle()` writes all sample artifacts:

- raw and preprocessed images,
- masks,
- enhanced image,
- `orientation.npy`,
- `ridge_period.npy`,
- `gradient_visualization.npy`,
- `minutiae.json`,
- `featurenet_targets.npz`,
- `meta.json`,
- optional `preview.png`.

The metadata explicitly records that labels are approximate:

```json
"approximate": true,
"input_domain": "raw_contactless_preprocessed"
```

It also records the preprocessing methods, reconstruction paths, target shapes, minutia counts, and selected execution devices.

## Reprojection Diagnostics

`_write_reprojection_diagnostics()` creates `reprojection_report.json` and `reprojection_preview.png`.

This is a silhouette-level diagnostic. It does not re-render full textured images. Instead, for each branch:

1. Build the observed mask from the pose-normalized foreground silhouette.
2. Build a projected row-wise support mask by aligning reconstructed row support to that view's row centers.
3. Compare observed and projected masks.

The metrics include:

- IoU
- precision
- recall
- intersection pixels
- union pixels
- false positives
- false negatives
- row width MAE
- row center MAE

The preview uses colors:

- white: observed and projected overlap,
- red/blue-ish: observed but not projected,
- cyan/yellow-ish: projected but not observed.

This diagnostic checks whether the reconstructed branch occupancy is consistent with the original view silhouettes.

## 3D Minutiae Visualization

`visualize_reprojected_minutiae.py` is a diagnostic script for viewing canonical minutiae on the shared reconstructed surface.

It loads:

- `center_unwarped.png`
- `canonical_unwarped_minutiae.json`
- `center_unwarp_maps.npz`
- `reconstruction_maps.npz`
- `support_mask.png`
- depth maps

For each canonical minutia:

1. `_map_unwarped_to_front_source()` maps the unwarped coordinate to front source coordinates.
2. `_lift_front_source_to_3d()` samples:

```text
x_relative = bilinear(reconstruction_maps["x_relative"])
z = bilinear(reconstruction_maps["depth_front"])
```

and creates:

```text
(X, Y, Z) = (x_relative, -y_front, depth_front)
```

The negative `Y` is used so the plotted vertical direction has a natural visual orientation.

3. It maps forward/backward orientation samples to 3D as well, producing a small 3D line segment for minutia direction.
4. It writes overlay summaries and optional PNG/HTML views.

This script does not create training labels. It verifies and visualizes how canonical minutiae sit on the reconstructed 3D branch model.

## DS1/DS2/DS3 Generation and Merge

`scripts/generate_ds123_and_merge.sh` automates the multi-dataset path.

It resolves a Python interpreter in this order:

1. `PYTHON_BIN` environment variable.
2. active virtual environment.
3. `~/.venvs/contactless-biometric/bin/python`.
4. `.venv/bin/python`.
5. `.venv/Scripts/python.exe`.
6. `python3`.
7. `python`.

Default environment variables:

```sh
DATASET_ROOT=archive
GROUND_TRUTH_ROOT=ground_truth
DS1_ROOT=ground_truth/DS1_v2
DS2_ROOT=ground_truth/DS2_v2
DS3_ROOT=ground_truth/DS3_v2
MERGED_ROOT=ground_truth/DS123_merged_v2
```

Default generator flags:

```sh
--execution-target kaggle --gpu-only --cpu-workers 10 --prefetch-samples 10 --fingerflow-backend local
```

The script runs:

```sh
python generate_ground_truth.py --dataset-root archive/DS1 --output-root ground_truth/DS1_v2 ...
python generate_ground_truth.py --dataset-root archive/DS2 --output-root ground_truth/DS2_v2 ...
python generate_ground_truth.py --dataset-root archive/DS3 --output-root ground_truth/DS3_v2 ...
```

Then it merges:

```sh
python generate_ground_truth.py \
  --merge-generated-root ds1=ground_truth/DS1_v2 \
  --merge-generated-root ds2=ground_truth/DS2_v2 \
  --merge-generated-root ds3=ground_truth/DS3_v2 \
  --output-root ground_truth/DS123_merged_v2
```

The merge path is `_merge_generated_ground_truth_roots()`.

Important merge behavior:

- It refuses to merge into a non-empty output root.
- It prefixes sample IDs with the source label.
- It prefixes reconstruction IDs with the source label.
- It creates a global subject remap from `(source_label, source_subject_id)` to a new merged subject index.
- It recomputes:
  - `subject_id`
  - `subject_index`
  - `finger_class_id`
- It copies sample bundles.
- It copies reconstruction directories only once per merged reconstruction ID.
- It rewrites `meta.json` paths inside copied bundles and reconstructions so they point at the merged output root.

This prevents collisions such as `s01_f01_a01` from DS1 and DS2 referring to different people or different reconstruction folders.

## Sharding and Patch Generation

The generator can also:

- split work into shards,
- merge shard outputs,
- build offline patch datasets.

Shard merging is `_merge_shard_outputs()`. It copies sample bundles and reconstruction directories into an empty output root and fails on duplicate sample or reconstruction IDs.

Patch generation reads existing full-image bundles and crops images, masks, and target arrays into square patches. `_select_patch_windows()` chooses windows whose foreground mask ratio exceeds the configured threshold. `_crop_target_array()` crops arrays according to whether they are image-like, channel-first, or grid-like target arrays.

Patch generation is downstream of the main ground-truth generation and does not change the reconstruction/unwarp/reprojection logic.

## Why This Pipeline Exists

Direct labels from contactless images are unstable:

- contactless views are curved and perspective-distorted,
- left/right views do not share the same local geometry,
- direct minutiae extraction can produce inconsistent point locations across views,
- training needs dense labels on a fixed output grid.

The reconstruction-backed path creates a common canonical source:

1. Estimate an approximate surface from the acquisition triplet.
2. Flatten the front surface by approximate arc length.
3. Extract minutiae once in that flattened space.
4. Reproject those minutiae to every view using the same geometry.

That makes view labels more consistent than independent extraction from each raw view.

## Known Approximations and Limitations

The code intentionally makes practical approximations:

- The reconstruction uses silhouette widths, not calibrated stereo or true multi-view photogrammetry.
- The cross-section is modeled as an ellipse per row.
- Side views are assumed to correspond approximately to `+45` and `-45` degree rotations.
- The center-depth formula is heuristic and based on row centers.
- Center unwarping integrates separable row and column arc lengths rather than solving a full surface parameterization.
- Inverse unwarp maps are nearest-source maps with bilinear sampling, not exact analytic inverses.
- Minutia reprojection uses weak orthographic row-wise projection.
- The fallback path may use direct per-sample extraction if reconstruction-backed labels fail.
- Metadata marks the output as approximate.

These approximations are acceptable for the intended purpose: producing consistent training supervision from contactless biometric images when true sensor-calibrated ground truth is unavailable.

## End-to-End Function Map

The major call flow is:

```text
main()
  _configure_runtime_environment()
  _build_manifest()
  _write_manifest()
  _collect_reconstruction_candidates()
  submit_sample()
    _reconstruct_multiview_acquisition()         # once per eligible acquisition
      _resolve_reconstruction_triplet()
      _extract_reconstruction_view_geometry()
        _preprocess_contactless_raw()
          _preprocess_contactless_bgr()
      ellipse row reconstruction
      _build_depth_gradient_labels()
        _smooth_depth_with_quadratic_mls()
      run_center_unwarping()
        compute_center_point_from_gradients()
        compute_arc_length_maps()
        build_unwarp_coordinates()
        resample_unwarped_image()
      _build_inverse_unwarp_maps()
      write reconstruction artifacts
    _load_sample_input()
  _generate_bundle_from_loaded()
    _prepare_bundle_from_loaded()
      _preprocess_contactless_bgr()
      pyfing.orientation_field_estimation()
      pyfing.frequency_estimation()
      _load_reconstruction_gradient_for_sample()
      _enhance_for_minutiae()
    _load_or_extract_canonical_reconstruction_minutiae()
      FingerFlow or pyfing fallback on center_unwarped.png
    _remap_unwarped_minutiae_to_sample()
      _map_unwarped_to_front_source()
      _project_front_source_to_training_frame()
    direct minutiae fallback if needed
    _build_featurenet_targets()
      _resize_orientation_for_model()
      _build_orientation_one_hot()
      _rasterize_minutiae()
    _build_bundle_meta()
    _persist_bundle()
  _write_summary()
```

## Practical Interpretation of the Labels

When training FeatureNet on these bundles:

- `orientation` teaches ridge-flow direction in a 180-bin undirected representation.
- `ridge_period` teaches normalized local ridge spacing.
- `gradient`, when present, teaches the network reconstructed surface slope, not image intensity slope.
- `minutia_score` teaches which output cells contain minutiae.
- `minutia_x_offset` and `minutia_y_offset` teach precise sub-cell localization.
- `minutia_orientation_vec` teaches directional minutia angle.
- `output_mask` and `minutia_valid_mask` define where supervision is valid.

The strongest form of minutiae label is the reconstruction-backed label because it comes from canonical extraction plus geometry reprojection. The direct fallback is useful coverage when reconstruction is unavailable or fails.

