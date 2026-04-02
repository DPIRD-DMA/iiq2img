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

**0.4% clip** is optimal — all channel ratios within 1.3% of the reference.

The original pipeline used 0.1%, which was too conservative (too dark with BT.709 gamma).

## Error decomposition

With the optimised pipeline (BT.709 + full-range scale + 0.4% clip), total mean diff is **6.50/255**:

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
    │  Subsample 8× → compute luma → histogram → 0.4% highlight clip → threshold
    │
    ▼  [numba parallel] BT.709 gamma LUT
    │  uint16→uint8 LUT: auto-bright scale + BT.709 gamma
    │
    ▼  uint8 RGB (12768x9564)
```

## Quality summary

| Metric | Value |
|--------|-------|
| Mean abs diff vs LibRaw PPG | 6.50 / 255 |
| Median abs diff | 3 / 255 |
| 95th percentile diff | 25 / 255 |
| Per-channel mean diff | R=7.2, G=3.3, B=9.0 |
| Channel mean ratios (fast/ref) | R=1.008, G=0.997, B=1.013 |
| Demosaic speed | ~600ms (4.6x faster than LibRaw PPG) |
| End-to-end speed (JPEG) | ~1.0s (3.4x faster) |

## Dead ends

Things we tried that didn't help:

- **Applying `raw.color_matrix`**: LibRaw doesn't use it for this camera. Applying it increased mean diff from 10 → 13.
- **sRGB gamma**: Wrong curve. LibRaw defaults to BT.709.
- **VNG / bilinear demosaic**: cv2 only supports these for uint8, not uint16. Would lose precision.
- **Per-channel auto-brightness**: LibRaw uses a single brightness scale, not per-channel. Our luma-based approach already matches well.
