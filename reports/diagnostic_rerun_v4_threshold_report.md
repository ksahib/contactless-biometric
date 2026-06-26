# FeatureNet v4 Diagnostic Rerun Report

## Scope

This report summarizes a new diagnostic pass run after the original matching-pipeline diagnostic report. The goal was to check whether the previously identified stability issue was solved, and then to test whether lowering the FeatureNet minutia score threshold from `0.6` to `0.5` improves or regresses the pipeline.

The original report identified the main failure as unstable FeatureNet minutia extraction across real captures of the same finger. In particular, the old index-pair waterfall showed many missing score responses in the second capture:

- `model_score_missing`: `31 / 50`
- counterparts within `16 px`: `5 / 50`
- counterparts within `32 px`: `16 / 50`
- median location error: `45.98 px`
- median angle error: `21.00 deg`

## Rerun Settings

Unless otherwise stated, the rerun used:

| Setting | Value |
|---|---|
| Checkpoint | `runs/featurenet_v4_front_only_aug_workers4/best.pt` |
| Matcher | `LSA-CENTROID` |
| Segmentation mode | `finger-pad-auto` |
| NMS | on |
| Device | CPU for rerun diagnostics |
| Thresholds tested | `0.6`, then `0.5` |

CUDA was not visible from the tool session used for the rerun, so diagnostics were run on CPU. The diagnostic logic and outputs are the same; only runtime differs.

## Tests Run

The following diagnostics were rerun:

1. Real same-finger waterfall for the index pair:
   - `amit_right_ind1.jpg` vs `amit_right_ind2.jpg`
   - script: `scripts/diagnose_real_pair_failure.py`

2. Real same-finger waterfall for the middle pair:
   - `amit_right_mid1.jpg` vs `amit_right_mid2.jpg`
   - script: `scripts/diagnose_real_pair_failure.py`

3. Four-image genuine/impostor cross-check:
   - `amit_right_ind1.jpg`
   - `amit_right_ind2.jpg`
   - `amit_right_mid1.jpg`
   - `amit_right_mid2.jpg`

4. Same-capture synthetic transform MCC sweep:
   - base: `amit_right_ind1.jpg`
   - shifts: `5, 10, 15, 20 px`
   - rotations: `5, 10, 15, 20 deg`
   - script: `scripts/sweep_fixed_period_transform_scores.py`

5. FeatureNet minutia extraction instability diagnostic:
   - same copy
   - shift `x5`
   - rotate `+5 deg`
   - script: `scripts/diagnose_featurenet_minutia_instability.py`

Output roots:

- `match_outputs/diag_rerun_v4/`
- `match_outputs/diag_rerun_v4_thr05/`

## Executive Summary

The v4 checkpoint improves the original stability issue substantially for the index pair. It finds many more corresponding minutia responses across captures, reduces missing score responses, and improves localization and orientation errors.

However, the issue is not fully solved. The middle-finger genuine pair still fails mostly at FeatureNet extraction repeatability. Lowering the score threshold to `0.5` recovers more candidates, but it also adds noisy minutiae, reduces genuine scores, and worsens impostor separation compared with `0.6`.

Best current diagnostic operating point from this pass: `0.6`, not `0.5`.

## Index Pair Waterfall

Pair:

```text
amit_right_ind1.jpg vs amit_right_ind2.jpg
```

| Metric | Original PDF | v4 @ 0.6 | v4 @ 0.5 | v4 @ 0.5 vs Original |
|---|---:|---:|---:|---:|
| Final MCC score | `0.6624` | `0.6645` | `0.6494` | `-0.0130` |
| A minutiae | `50` | `47` | `55` | `+5` |
| B minutiae | `54` | `46` | `63` | `+9` |
| Counterparts within `16 px` | `5 / 50` | `15 / 47` | `19 / 55` | `+14` count, `+24.5 pp` |
| Counterparts within `32 px` | `16 / 50` | `33 / 47` | `37 / 55` | `+21` count, `+35.3 pp` |
| Model-score-missing | `31 / 50` | `11 / 47` | `14 / 55` | `-17` count, improved |
| Score-map responses found | `17 / 50` | `35 / 47` | `41 / 55` | `+24` count, improved |
| Survived NMS | `16 / 50` | `33 / 47` | `37 / 55` | `+21` count, improved |
| Median location error | `45.98 px` | `24.43 px` | `23.25 px` | `-22.73 px`, improved |
| Median angle error | `21.00 deg` | `5.31 deg` | `7.60 deg` | `-13.40 deg`, improved |
| Descriptor pairs available | not listed | `1048` | `1762` | increased vs v4 @ 0.6 |
| LSA selected pairs | `12` | `12` | `12` | unchanged |

### What Improved

Compared with the original diagnostic, FeatureNet repeatability improved strongly for this pair:

- `model_score_missing` dropped from `31 / 50` to `14 / 55`.
- Nearby counterparts within `32 px` increased from `16 / 50` to `37 / 55`.
- Median location error dropped by `22.73 px`.
- Median angle error dropped by `13.40 deg`.

### What Regressed

Lowering threshold from `0.6` to `0.5` did not improve final matching:

- MCC score decreased from `0.6645` to `0.6494`, a drop of `0.0151`.
- `model_score_missing` increased from `11` to `14`.
- `descriptor_score_low` increased from `4` to `7`.
- `nms_suppressed` increased from `2` to `4`.
- `matched` in the per-minutia waterfall dropped from `9` to `5`.

Interpretation: `0.5` recovers more candidates, but some of the added candidates are weaker or less stable for MCC assignment.

Artifacts:

- `match_outputs/diag_rerun_v4/current_index_waterfall/report.json`
- `match_outputs/diag_rerun_v4_thr05/current_index_waterfall/report.json`

## Middle Pair Waterfall

Pair:

```text
amit_right_mid1.jpg vs amit_right_mid2.jpg
```

| Metric | v4 @ 0.6 | v4 @ 0.5 | Change |
|---|---:|---:|---:|
| Final MCC score | `0.5276` | `0.4856` | `-0.0420`, worse |
| A minutiae | `33` | `45` | `+12` |
| B minutiae | `24` | `45` | `+21` |
| Counterparts within `16 px` | `3 / 33` | `5 / 45` | `+2` count, but weak proportion |
| Counterparts within `32 px` | `8 / 33` | `13 / 45` | `+5` count |
| Model-score-missing | `24 / 33` | `29 / 45` | `+5` count, still dominant |
| Score-map responses found | `8 / 33` | `15 / 45` | `+7` count |
| Survived NMS | `8 / 33` | `13 / 45` | `+5` count |
| Median location error | `46.97 px` | `42.33 px` | `-4.63 px`, improved |
| Median angle error | `41.46 deg` | `41.46 deg` | unchanged |
| Descriptor pairs available | `239` | `809` | `+570` |
| LSA selected pairs | `5` | `12` | `+7` |

### What Improved

At `0.5`, the middle pair produces more minutiae and more descriptor opportunities:

- A/B minutiae increased from `33 / 24` to `45 / 45`.
- Score-map responses found increased from `8` to `15`.
- Descriptor pairs available increased from `239` to `809`.
- LSA selected pairs increased from `5` to `12`.
- Median location error improved by `4.63 px`.

### What Regressed

The final match got worse:

- MCC score dropped from `0.5276` to `0.4856`.
- `model_score_missing` increased from `24` to `29`.
- `offset_moved` increased from `5` to `8`.
- The dominant conclusion remains FeatureNet extraction/image-quality robustness failure.

Interpretation: `0.5` increases recall, but the extra points do not form stable or discriminative correspondences. This pair still reproduces the original stability problem.

Artifacts:

- `match_outputs/diag_rerun_v4/current_middle_waterfall/report.json`
- `match_outputs/diag_rerun_v4_thr05/current_middle_waterfall/report.json`

## Genuine vs Impostor Cross-Check

The cross-check used:

```text
ind1 = amit_right_ind1.jpg
ind2 = amit_right_ind2.jpg
mid1 = amit_right_mid1.jpg
mid2 = amit_right_mid2.jpg
```

| Bucket | Original PDF | v4 @ 0.6 | v4 @ 0.5 |
|---|---:|---:|---:|
| Self controls | `1.0000` | `1.0000` | `1.0000` |
| Genuine index | `0.6624` | `0.6645` | `0.6494` |
| Genuine middle | `0.5257` | `0.5276` | `0.4856` |
| Impostor range | `0.5055-0.5484` | `0.4314-0.4709` | `0.4379-0.5102` |

### Improvements vs Original PDF

At v4 @ `0.6`:

- Max impostor dropped from `0.5484` to `0.4709`, an improvement of `0.0775`.
- Genuine middle moved above the impostor range:
  - old: genuine middle `0.5257` was inside impostor range `0.5055-0.5484`
  - v4 @ `0.6`: genuine middle `0.5276` is above max impostor `0.4709`
- Genuine index stayed stable:
  - old: `0.6624`
  - v4 @ `0.6`: `0.6645`

### Regressions at Threshold 0.5

Compared with v4 @ `0.6`:

- Genuine index dropped from `0.6645` to `0.6494`, down `0.0151`.
- Genuine middle dropped from `0.5276` to `0.4856`, down `0.0420`.
- Max impostor increased from `0.4709` to `0.5102`, up `0.0393`.
- Genuine middle again falls below the max impostor:
  - genuine middle: `0.4856`
  - max impostor: `0.5102`

Interpretation: v4 @ `0.6` improves separation over the PDF, but lowering to `0.5` regresses separation.

Artifacts:

- `match_outputs/diag_rerun_v4/current_crosscheck/crosscheck_scores.csv`
- `match_outputs/diag_rerun_v4_thr05/current_crosscheck/crosscheck_scores.csv`

## Same-Capture Extraction Stability

This diagnostic checks whether FeatureNet extraction is stable under a same copy, a small shift, and a small rotation.

### Threshold 0.6

| Comparison | Within 16 px | Within 32 px | Score-map correlation | Main categories |
|---|---:|---:|---:|---|
| same copy | `47 / 47` | `47 / 47` | `1.0000` | all matched |
| shift x5 | `37 / 47` | `40 / 47` | `0.9634` | `37` matched, `7` threshold drop |
| rotate +5 | `25 / 47` | `38 / 47` | `0.6171` | `25` matched, `10` offset moved |

### Threshold 0.5

| Comparison | Within 16 px | Within 32 px | Score-map correlation | Main categories |
|---|---:|---:|---:|---|
| same copy | `55 / 55` | `55 / 55` | `1.0000` | all matched |
| shift x5 | `51 / 55` | `53 / 55` | `0.9634` | `51` matched |
| rotate +5 | `35 / 55` | `51 / 55` | `0.6171` | `35` matched, `11` offset moved |

### Improvements at 0.5

Lowering to `0.5` improves extraction recall under controlled transforms:

- Base decoded minutiae increased from `47` to `55`.
- Shift x5 within `16 px` improved from `37 / 47` to `51 / 55`.
- Rotate +5 within `16 px` improved from `25 / 47` to `35 / 55`.
- Rotate +5 within `32 px` improved from `38 / 47` to `51 / 55`.

### Remaining Stability Issue

The score-map correlation for rotate +5 remains `0.6171` at both thresholds. Lowering the threshold recovers more points, but it does not make the underlying score map more stable. Rotation still changes the response structure substantially.

Artifacts:

- `match_outputs/diag_rerun_v4/current_minutia_instability/diagnosis_report.json`
- `match_outputs/diag_rerun_v4_thr05/current_minutia_instability/diagnosis_report.json`

## Same-Capture Synthetic MCC Robustness

The final MCC matcher remains robust under synthetic transforms of the same capture.

| Transform | v4 @ 0.6 | v4 @ 0.5 | Change |
|---|---:|---:|---:|
| same copy | `1.0000` | `1.0000` | unchanged |
| shift x5 | `0.8222` | `0.8145` | `-0.0078` |
| shift x10 | `0.7922` | `0.8539` | `+0.0617` |
| shift x15 | `0.7404` | `0.7981` | `+0.0577` |
| shift x20 | `0.9601` | `0.9439` | `-0.0162` |
| rotate +5 | `0.8493` | `0.8409` | `-0.0084` |
| rotate +10 | `0.7748` | `0.8013` | `+0.0264` |
| rotate +15 | `0.9451` | `0.9650` | `+0.0199` |
| rotate +20 | `0.8588` | `0.8608` | `+0.0021` |

### Interpretation

MCC robustness is still not the main failure. Synthetic same-capture scores remain high at both thresholds. The real-pair failures are still mostly caused upstream by unstable extraction/localization across real captures.

Artifacts:

- `match_outputs/diag_rerun_v4/current_transform_sweep/mcc_transform_scores.csv`
- `match_outputs/diag_rerun_v4_thr05/current_transform_sweep/mcc_transform_scores.csv`

## Overall Findings

| Area | Improved? | Evidence |
|---|---|---|
| Index-pair extraction repeatability | Yes | `model_score_missing` improved from `31 / 50` to `14 / 55`; counterparts within `32 px` improved from `16 / 50` to `37 / 55` |
| Index-pair localization | Yes | median location error improved from `45.98 px` to `23.25 px` |
| Index-pair orientation | Yes | median angle error improved from `21.00 deg` to `7.60 deg` |
| Impostor suppression at v4 @ 0.6 | Yes | max impostor dropped from `0.5484` to `0.4709` |
| Middle-pair extraction | Not solved | `model_score_missing` remains high: `29 / 45` at threshold `0.5` |
| Threshold 0.5 candidate recovery | Yes | more minutiae and more nearby candidates in both pairs |
| Threshold 0.5 match quality | Regressed | genuine index and middle scores both dropped; max impostor increased |
| Synthetic MCC transform robustness | Still good | most same-capture transform scores remain around `0.8-0.96` |
| Rotation score-map stability | Not solved | rotate +5 score-map correlation remains `0.6171` |

## Final Conclusion

The new diagnostic pass shows that the v4 checkpoint improved the original extraction stability problem, especially for the index pair. The clearest improvements are fewer missing FeatureNet score responses, more nearby counterparts after alignment, lower location error, lower angle error, and lower impostor scores at threshold `0.6`.

However, the stability issue is not fully solved. The middle-finger genuine pair still fails mostly through `model_score_missing`, and the small-rotation diagnostic still shows unstable FeatureNet score maps.

Lowering the score threshold to `0.5` improves candidate recall, but it regresses matching quality and impostor separation. It is therefore not a better operating threshold for this diagnostic set. The current evidence favors keeping `0.6` for matching while continuing to improve real-capture FeatureNet extraction consistency.
