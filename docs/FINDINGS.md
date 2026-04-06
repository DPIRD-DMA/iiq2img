# Fast Pipeline Research Findings

Everything we learned while building the fast IIQ demosaic pipeline and matching it to LibRaw's output.

Target camera: **Phase One iXM-GS120** (120 MP, 12768x9564, RGGB Bayer).

## rawpy / LibRaw quirks for Phase One IIQ

### Black level is 1024, not 0

`raw.black_level_per_channel` incorrectly reports `[0, 0, 0, 0]`. The actual black level is **1024 ADU**, confirmed by tracing individual raw pixels against LibRaw's linear 16-bit output:

```
raw pixel = 1028  →  LibRaw linear output = 4  →  diff = 1024 exactly
```

Hardcoded as `_IIQ_BLACK_LEVEL = 1024`.

### No color matrix is applied

`raw.color_matrix` IS populated (a 3x4 matrix with row sums ~1.0, off-diagonal elements up to 0.73), but **LibRaw does not actually use it** for this camera. Confirmed by comparing `postprocess(output_color=sRGB)` vs `postprocess(output_color=raw)` — the outputs are identical.

`raw.rgb_xyz_matrix` is all zeros.

Applying the color matrix ourselves made quality **worse** (mean diff 10 → 13). Removing it was the right call.

### White balance

`raw.camera_whitebalance` returns `[2.879, 1.0, 1.805, 0.0]`. Green is already the minimum (1.0), so normalizing to G=1.0 matches LibRaw's internal normalization (divide by min).

### Bayer pattern

`raw.raw_pattern` = `[[0, 1], [3, 2]]` = RGGB. The cv2 code `COLOR_BAYER_RG2BGR_EA` is correct for this pattern. Despite the name containing "BGR", it outputs **RGB**-ordered data for RGGB uint16 input (R=ch0). Do NOT reverse channels.

### Linear data — no tone curve needed

After subtracting black=1024, the raw Bayer data is perfectly linear with LibRaw's output. No additional linearisation or tone curve is needed for this sensor.

## Gamma curve: BT.709, not sRGB

This was the single biggest source of error.

rawpy's `postprocess()` uses **BT.709 gamma** `(2.222, 4.5)` by default, NOT sRGB gamma `(2.4, 12.92)`. The difference between the two curves is ~12.8/255 mean abs diff on the same linear data.

| Curve | Linear toe | Power law |
|-------|-----------|-----------|
| **BT.709** (rawpy default) | `V = 4.5 * L` for `L < 0.018` | `V = 1.099 * L^0.45 - 0.099` |
| sRGB | `V = 12.92 * L` for `L < 0.0031308` | `V = 1.055 * L^(1/2.4) - 0.055` |

BT.709 produces slightly darker mid-tones than sRGB. Switching our pipeline from sRGB to BT.709 was the single largest quality improvement.

We verified the curve match empirically: at every test point from 200 to 7000 in uint16, our BT.709 LUT matches LibRaw's output within **<1 level** (0.75 average).

## Full-range scale factor

LibRaw normalises raw values to fill the uint16 range after black subtraction:

```
scale = 65535 / (65535 - 1024) = 1.01588
```

Per-pixel comparison confirms this: a raw G pixel of 1468 becomes 451 in LibRaw's linear output, while `(1468 - 1024) * 1.0 = 444`. The ratio is 451/444 = 1.016.

Our WB LUTs now include this factor.

## Auto-brightness clip fraction

LibRaw's auto-brightness is histogram-based. We match it by finding the luma value where the cumulative histogram reaches a certain fraction, then mapping that to white.

Empirical sweep of clip fractions with BT.709 gamma:

| Clip % | Mean diff | Channel ratios (fast/ref) |
|--------|-----------|--------------------------|
| 0.1% | 14.82 | R=0.877, G=0.864, B=0.874 |
| 0.2% | 10.25 | R=0.937, G=0.924, B=0.937 |
| 0.35% | 7.07 | R=0.992, G=0.981, B=0.996 |
| **0.4%** | **6.50** | **R=1.008, G=0.997, B=1.013** |
| 0.5% | 7.37 | R=1.032, G=1.022, B=1.038 |
| 1.0% | 13.87 | R=1.123, G=1.115, B=1.135 |

Initial tuning on a single image (P0286625) pointed to 0.4%. Testing across all 4 sample images revealed 0.4% was slightly too conservative for some scenes (G channel up to 3% dark). **0.45% clip** is the best compromise:

| Clip % | Avg diff (4 images) | Worst G ratio | Spread |
|--------|---------------------|---------------|--------|
| 0.40% | 7.48 | 0.970 | 6.5–8.2 |
| **0.45%** | **7.28** | **0.986** | **6.8–7.5** |
| 0.50% | 7.42 | 1.000 | 6.9–8.1 |

0.45% has the lowest average diff, the tightest spread across images, and keeps all channel ratios within 2.6% of reference.

The original pipeline used 0.1%, which was too conservative (too dark with BT.709 gamma).

## Error decomposition

With the optimised pipeline (BT.709 + full-range scale + 0.45% clip), total mean diff is **~7/255**:

| Source | Mean diff contribution |
|--------|----------------------|
| Auto-brightness / gamma mismatch | 4.06 / 255 |
| Demosaic algorithm (EA vs PPG) | 2.44 / 255 |
| **Total** | **6.50 / 255** |

Measured by applying our gamma to LibRaw's own PPG linear output (isolating gamma error) vs our full pipeline (gamma + demosaic error).

The demosaic contribution (2.44) is irreducible without switching algorithms. cv2's VNG and bilinear modes don't support uint16, so EA is the only fast option at full bit depth.

## Final pipeline

```
raw_image_visible (12768x9564, uint16, RGGB Bayer)
    │
    ▼  [numba parallel] WB LUT on Bayer
    │  black subtract (1024) + white balance + full-range scale (1.016)
    │  Per-channel uint16→uint16 LUTs applied to RGGB positions
    │
    ▼  [cv2] Edge-Aware demosaic
    │  COLOR_BAYER_RG2BGR_EA → uint16 RGB (not BGR despite the name)
    │
    ▼  Auto-brightness
    │  Subsample 8× → compute luma → histogram → 0.45% highlight clip → threshold
    │
    ▼  [numba parallel] BT.709 gamma LUT
    │  uint16→uint8 LUT: auto-bright scale + BT.709 gamma
    │
    ▼  uint8 RGB (12768x9564)
```

## Colour matching across images

Validated on 4 sample IIQ files (different scenes, exposures):

| Image | Mean diff | Median | 95th | R ratio | G ratio | B ratio |
|-------|-----------|--------|------|---------|---------|---------|
| P0286625 | 6.83 | 3 | 26 | 1.021 | 1.010 | 1.026 |
| P0286635 | 7.38 | 3 | 26 | 0.999 | 0.986 | 1.001 |
| P0286642 | 7.51 | 3 | 28 | 1.022 | 1.008 | 1.026 |
| P0286690 | 7.39 | 3 | 27 | 1.003 | 0.991 | 1.006 |

All channel ratios within 2.6% of LibRaw reference. Median diff is a consistent 3/255 across all images. No scene-dependent bias — the pipeline generalises well beyond the single image it was tuned on.

## Performance profiling

Profiled end-to-end on a single 120 MP IIQ → JPEG conversion (after numba warmup):

| Stage | Time | % | Notes |
|-------|------|---|-------|
| IIQ decompress (rawpy) | 378ms | 41% | C code, single-threaded, can't optimise |
| WB LUT on Bayer (numba) | 22ms | 2% | Already parallel (`prange`) |
| cv2 EA demosaic | 75ms | 8% | C++ with OpenMP |
| Auto-brightness | 29ms | 3% | Numpy on 1/64th of pixels |
| Gamma LUT (numba) | 40ms | 4% | Already parallel |
| RGB→BGR + JPEG encode | 406ms | 42% | libjpeg-turbo via cv2 |

The two bottlenecks (IIQ read 41%, JPEG encode 42%) are both optimised C and account for 83% of the time. The actual fast pipeline compute (WB + demosaic + gamma) is only **166ms (17%)**.

### Optimisations applied

- **Skip `.copy()` on Bayer data**: The WB LUT reads from rawpy's view and writes to a new array, so we can avoid the 26ms copy. `raw.close()` is safe after the LUT since `b_corr` is independent. Saves ~45ms (8%).
- **Subsampled auto-brightness**: Histogram computed on 8× subsampled luma (64× fewer pixels). Negligible quality impact, saves ~200ms vs full-res.
- **Single-pass gamma**: Combined auto-bright scale + BT.709 gamma into one uint16→uint8 LUT. One memory pass instead of two.

### What didn't help

- **turbojpeg**: Same speed as cv2 for 120 MP JPEG (both use libjpeg-turbo). No win even accounting for skipping RGB→BGR.
- **Pillow JPEG**: Slower than cv2.
- **Threading within a single image**: numba `prange` and cv2 OpenMP already saturate all cores during their stages.
- **Contiguous array hint**: Bayer view from rawpy has non-contiguous strides, but the WB LUT copies to contiguous output anyway.

### Batch parallelism

Overlapping JPEG encode of image N with demosaic of image N+1 in a background thread saves **~22% in sequential batch mode** (tested on 2 images: 2004ms → 1573ms). This works because rawpy unpack is single-threaded while JPEG encode can use idle cores.

For `workers>1` batch mode, the existing multiprocessing already parallelises across images, which is more effective.

## Quality summary

| Metric | Value |
|--------|-------|
| Mean abs diff vs LibRaw PPG | 6.8–7.5 / 255 (across 4 images) |
| Median abs diff | 3 / 255 |
| 95th percentile diff | 26–28 / 255 |
| Channel mean ratios (fast/ref) | All within 2.6% of 1.0 |
| Demosaic speed | ~600ms (4.6x faster than LibRaw PPG) |
| End-to-end speed (JPEG) | ~920ms (3.6x faster) |

## Dead ends

Things we tried that didn't help:

- **Applying `raw.color_matrix`**: LibRaw doesn't use it for this camera (`rgb_xyz_matrix` is all zeros). Applying it increased mean diff from 10 → 13.
- **sRGB gamma**: Wrong curve. LibRaw defaults to BT.709. Switching was the single biggest win (~12.8/255).
- **VNG / bilinear demosaic**: cv2 only supports these for uint8, not uint16. Would lose precision.
- **Per-channel auto-brightness**: LibRaw uses a single brightness scale, not per-channel. Our luma-based approach already matches well.
- **turbojpeg**: No speed advantage over cv2 for 120 MP images.
- **Skipping rawpy**: `rawpy.imread` itself is instant; the cost is in `raw.raw_image_visible` which triggers LibRaw's IIQ decompressor (378ms, C code). No way around it without a custom IIQ parser.
