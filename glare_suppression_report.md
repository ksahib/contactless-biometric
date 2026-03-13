# Glare Suppression Report

## Summary

This report describes the preprocessing changes made to improve same-finger MCC matching scores on glare-affected contactless fingerprint images.

The key point is:

- I did **not** improve scores by changing MCC descriptor math or matcher logic in this phase.
- I improved scores by changing the **image preprocessing before minutiae extraction**.
- The main target was **specular glare suppression with ridge preservation**.

## What Was Wrong Before

The earlier glare suppression method was a generic brightness clamp:

- detect bright, low-local-variance regions
- clamp those pixels toward a local mean

That reduces highlight intensity, but it is not fingerprint-specific.

For fingerprints, the real goal is not to make the image look less shiny. The goal is:

- preserve ridge/valley contrast
- avoid flattening glared ridge structure
- improve minutiae extraction quality

The original approach was too blunt because it treated glare mainly as a brightness problem instead of a ridge-recovery problem.

## What Changed

### 1. Match output hygiene

I changed saved match run directories from:

- `match_<seconds>`

to:

- `match_<time_ns>_<pid>`

This fixed collisions where multiple runs in the same second could overwrite each other. Without this, score comparisons were not trustworthy.

### 2. Ridge-aware glare detection

I replaced the old glare-mask logic with a stricter fingerprint-oriented mask.

Now a region is considered glare only when it is:

- inside the finger foreground mask
- bright
- above a smooth illumination field
- low in ridge-detail energy
- low in gradient energy

This matters because bright regions are not always useless. Some bright regions still contain valid ridge structure and should not be aggressively suppressed.

### 3. Illumination-field-based correction

Instead of clamping bright pixels to a local mean, the new method estimates a low-frequency illumination field and separates:

- low-frequency lighting
- high-frequency ridge detail
- positive specular residual

The corrected image is built by:

- compressing only the positive glare residual
- preserving ridge-detail content
- blending the correction only inside the glare mask

This is more appropriate for fingerprints because it tries to remove the glare component without erasing ridge information.

### 4. Soft blending instead of hard replacement

The glare mask is blurred into a soft blend map.

- outside glare: keep the original image
- inside glare: blend in the corrected image conservatively

This avoids hard edges and reduces the risk of creating artificial structures.

### 5. Debug artifacts

I added saved intermediate outputs:

- `glare_mask.png`
- `glare_suppressed.png`

These make it possible to inspect:

- where glare was detected
- whether suppression is localized
- whether ridge structure is being preserved

## Technical Description of the Final Glare Suppression

For a grayscale fingerprint image with a foreground mask:

1. Compute a smooth illumination field:

```python
illumination = GaussianBlur(gray, sigma ~ 15)
```

2. Compute a smoother ridge base and residual ridge detail:

```python
ridge_base = GaussianBlur(gray, sigma ~ 2.2)
ridge_detail = gray - ridge_base
```

3. Compute ridge-energy and gradient-energy maps:

```python
ridge_energy = GaussianBlur(abs(ridge_detail), sigma ~ 3)
grad_x, grad_y = Sobel(gray)
grad_energy = GaussianBlur(magnitude(grad_x, grad_y), sigma ~ 3)
```

4. Compute positive specular residual:

```python
specular_residual = max(0, gray - illumination)
```

5. Build the glare mask using percentile thresholds within the fingerprint foreground:

- high intensity
- high residual above illumination
- low ridge energy
- low gradient energy

6. Clean the glare mask with small morphology and dilation.

7. Compress only the glare residual:

```python
compressed_residual = residual_cap * tanh(specular_residual / residual_cap)
```

8. Reconstruct a corrected image:

```python
recovered = illumination + compressed_residual + ridge_detail
```

9. Blend the corrected image with the original using a soft glare mask.

## What Stayed Unchanged

The following were not changed in this phase:

- crop logic
- MCC descriptor construction
- overlap-aware matching
- mask-based validity logic
- adaptive neighbor support
- MCC local similarity and global scoring

So any score change from this work came from preprocessing and extraction quality, not from MCC formula changes.

## Why Scores Improved

The score improvement came from making a few surviving local correspondences stronger.

This is important:

- the weak glare-heavy pair did **not** improve because many more minutiae were extracted
- it improved because the few surviving descriptors became more matchable

So the effect was:

- stronger local MCC similarities for the pairs that remained
- higher final same-finger score
- but still unstable weak-image minutiae extraction

## Observed Results

### Weak glare-heavy pair

Pre-glare baseline:

- run: `match_1773107529`
- raw minutiae: `6 / 6`
- surviving descriptors: `4 / 5`
- `LSA-R = 0.1025`

After the glare redesign:

- run: `match_1773110022611522500_36464`
- raw minutiae: `3 / 7`
- surviving descriptors: `2 / 6`
- `LSA-R = 0.3024`
- `LSS-R = 0.4345`

Interpretation:

- match score improved a lot
- weak-side minutiae extraction got worse
- the score increase came from stronger surviving correspondences, not higher minutiae yield

### Healthier same-finger pair

Earlier healthier baseline:

- run: `match_1773098684`
- raw minutiae: `17 / 13`
- surviving descriptors: `12 / 10`
- `LSA-R = 0.1967`

After the glare redesign:

- run: `match_1773110022610516500_21644`
- raw minutiae: `18 / 13`
- surviving descriptors: `16 / 10`
- `LSA-R = 0.3065`

Interpretation:

- healthier pair improved clearly
- no obvious regression on that stronger case

## Current Limitation

The current glare suppression improved matching score, but it did **not** solve the main weak-image extraction problem.

The weak pair still suffers from:

- too few raw minutiae on one side
- sparse surviving descriptors
- extraction instability in glare-heavy regions

So the current state is:

- score improvement: yes
- extractor robustness in weak glared images: still insufficient

## Conclusion

The implemented preprocessing changes improved same-finger matching scores by making glare suppression more fingerprint-specific and ridge-aware.

The main improvement mechanism was:

- preserve useful ridge structure in glared regions
- feed a better image into normalization, enhancement, and FingerFlow
- strengthen the few surviving descriptor correspondences

However, the current method still does not reliably increase minutiae yield on the weakest images. That remains the main bottleneck.
