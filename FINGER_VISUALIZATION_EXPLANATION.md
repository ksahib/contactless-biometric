# Finger Visualization: Two-Pipeline Rewrite

## Overview

I rewrote `finger_visualization.py` so it no longer treats the old sparse three-view point-cloud fusion as the main output.

The script now has two explicit runtime paths:

- `center_only`
- `dense_ellipse`
- `both` as the default

This change was necessary because the repo contains two different kinds of geometry evidence:

- sparse valid-depth ribbons in `depth_front.npy`, `depth_left.npy`, and `depth_right.npy`
- a much richer analytic reconstruction in `row_measurements.json` plus `support_mask.png` and `meta.json`

Those are not the same thing, and the new script now treats them differently.

## What Changed

### 1. Added two real pipelines

`center_only` now does exactly one job:

- build a strict center-view point cloud from valid center depth only
- save `center_only_pointcloud.png`
- save `center_only_pointcloud.npz`
- save `center_only_surface.png` when the valid ROI spans at least 2 rows and 2 columns

It explicitly reports that no multiview fusion is performed.

`dense_ellipse` now does the dense reconstruction job:

- load `row_measurements.json` as the primary parameter source
- interpolate the valid rows onto a dense row grid
- sample each ellipse densely across x
- build a dense canonical upper/front surface
- build dense left and right stitched branches using `x_crit` and the metadata side rotations
- stitch those patches into an open skin surface
- save `dense_ellipse_canonical.png`
- save `dense_ellipse_canonical.npz`
- save `dense_ellipse_fused.png`
- save `dense_ellipse_fused.npz`

This pipeline does not use centroid registration, manual translations, or the old sparse fusion output as its geometry source.

### 2. Kept the shared sparse diagnostics

I kept the earlier diagnostic layer at the top level of `fusion_report.json`:

- valid-depth filtering
- per-view sparse/ribbon detection
- reconstruction metadata summary
- geometry interpretation
- raw-vs-normalized sparse comparison when center, left, and right sparse views are available

That shared diagnostic report is still important, because it explains why the dense analytic pipeline exists in the first place.

For the current DS1 sample, the sparse depth arrays are still honestly classified as:

- `too_sparse_or_partial`

### 3. Added pipeline-specific validation and outputs

The CLI now supports:

```text
--pipeline {center_only,dense_ellipse,both}
--dense-sampling-rows
--dense-sampling-cols
--row-measurements
--support-mask
--reconstruction-meta
```

Validation is now pipeline-aware:

- `center_only` needs center image, mask, and depth
- `dense_ellipse` needs reconstruction artifacts
- `both` runs them independently and can finish with `run_status: "partial"` if only one succeeds

### 4. Added dense-pipeline debug outputs

With `--debug`, the dense pipeline now writes:

- `debug/dense_ellipse_canonical.png`
- `debug/dense_ellipse_center.png`
- `debug/dense_ellipse_left.png`
- `debug/dense_ellipse_right.png`
- `debug/dense_ellipse_overlap.png`
- `debug/dense_ellipse_parameter_summary.json`

The center-only pipeline now also writes:

- `debug/center_only_local.png`
- `debug/center_only_surface.png`

## Why I Implemented It This Way

### Why split the script into two pipelines

The sparse `depth_*.npy` files in this repo only contain valid depth on a narrow row band.

For `s01_f01_a01`, each sparse view still has just `42` unique valid rows, so treating those arrays as if they were a dense fused 3D model would be misleading.

At the same time, the reconstruction directory already stores the denser analytic model we actually want:

- `row_measurements.json`
- `support_mask.png`
- `meta.json`

That made a split architecture the honest choice:

- one pipeline for the sparse center view
- one pipeline for the dense analytic reconstruction

### Why `row_measurements.json` became the main dense source

`row_measurements.json` already stores the per-row ellipse parameters needed for dense reconstruction:

- `semi_major`
- `semi_minor`
- `center_depth`
- `theta`
- `x_crit`
- row validity
- left/right translations

Those are better geometry primitives than reverse-engineering a dense surface from the sparse saved depth bands.

### Why the dense pipeline ignores `--depth-normalization`

The dense analytic surface is reconstructed from persisted ellipse parameters and metadata formulas, not from the sparse cross-view depth alignment path.

So I left `--depth-normalization` in place for the sparse diagnostics and `center_only`, but I explicitly ignore it for `dense_ellipse` and report that choice in the pipeline warnings.

## What I Verified

### Shortcut smoke tests

I ran:

- `--pipeline center_only` with `--front-sample-dir` and `--reconstruction-dir`
- `--pipeline dense_ellipse` with `--reconstruction-dir`
- `--pipeline both` with front/left/right sample dirs plus reconstruction dir

Outputs:

- [center report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_center\fusion_report.json>)
- [dense report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_dense\fusion_report.json>)
- [both report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_both\fusion_report.json>)

Key results for `s01_f01_a01`:

- shared sparse classification stayed `too_sparse_or_partial`
- `center_only` succeeded with `valid_point_count: 7440`
- `dense_ellipse` succeeded with `parameter_source: "row_measurements"`
- dense canonical samples: `65,536`
- dense fused mesh: `110,080` vertices and `216,750` faces

### Explicit-path smoke tests

I also ran:

- explicit center-only inputs
- explicit dense reconstruction-artifact inputs
- explicit `both`

Outputs:

- [center explicit report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_center_explicit\fusion_report.json>)
- [dense explicit report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_dense_explicit\fusion_report.json>)
- [both explicit report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_both_explicit\fusion_report.json>)

Those all completed successfully.

### Partial-success cases

I checked the independent pipeline behavior too.

Dense succeeds while center fails:

- [partial dense-only report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_partial_dense_only\fusion_report.json>)
- `run_status: "partial"`
- `center_only.status: "failed"`
- `dense_ellipse.status: "ok"`

Center succeeds while dense fails:

- [partial center-only report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_partial_center_only\fusion_report.json>)
- `run_status: "partial"`
- `center_only.status: "ok"`
- `dense_ellipse.status: "failed"`
- dense failure reason: missing `meta.json`

### Failure cases

I re-ran the error paths and confirmed the script now writes a report before failing:

- missing depth file
- shape mismatch
- empty mask
- non-numeric depth
- all-invalid depth

Example reports:

- [all-invalid report](</d:\contactless biome\contactless-biometric\tmp\finger_visualization_two_pipeline_fail_all_invalid\fusion_report.json>)

## Final State

`finger_visualization.py` is now organized around the two geometry stories that actually exist in this repo:

- sparse center-view visualization
- dense analytic ellipse reconstruction

The old sparse three-view fusion still exists only as shared diagnostics, not as the main deliverable.

That makes the outputs much more honest:

- `center_only` is clearly single-view and sparse
- `dense_ellipse` is clearly reconstruction-driven and dense
- `fusion_report.json` is now the source of truth for which pipeline ran, what succeeded, and why
