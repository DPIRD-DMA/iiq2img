"""
Fast IIQ demosaic research — development script.

Runs both pipelines on a real Phase One IIQ file and reports quality metrics.
See FINDINGS.md for the full write-up of what we learned.

Pipeline
--------
  raw_image_visible  →  numba WB+black LUT on Bayer (with full-range scale)
  →  cv2 EA demosaic (uint16)  →  auto-brightness (0.4% luma clip)
  →  BT.709 gamma LUT (numba parallel)  →  uint8 RGB

Timing (Phase One iXM-GS120, 120MP, 12768×9564):
  LibRaw PPG demosaic:  ~2900 ms
  Fast pipeline demosaic: ~600 ms  →  4.6× speedup
  End-to-end (JPEG):     ~1.0 s   →  3.4× speedup (JPEG encode is ~450ms shared cost)
  Quality: mean abs diff ~6.5/255 vs LibRaw PPG
"""

import time
from pathlib import Path

import cv2
import numba
import numpy as np
import rawpy


BLACK = 1024  # IIQ black level (rawpy incorrectly reports 0)


@numba.njit(parallel=True)
def _lut3(
    img: np.ndarray, lr: np.ndarray, lg: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """Apply 3 independent uint16→uint8 LUTs to an (H,W,3) uint16 image in parallel."""
    H, W = img.shape[0], img.shape[1]
    out = np.empty((H, W, 3), numba.uint8)
    for i in numba.prange(H):
        for j in range(W):
            out[i, j, 0] = lr[img[i, j, 0]]
            out[i, j, 1] = lg[img[i, j, 1]]
            out[i, j, 2] = lb[img[i, j, 2]]
    return out


@numba.njit(parallel=True)
def _wb_lut_bayer(
    bayer: np.ndarray, lr: np.ndarray, lg: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """Apply per-channel uint16→uint16 WB+black LUTs to RGGB Bayer in parallel."""
    H, W = bayer.shape
    out = np.empty_like(bayer)
    for i in numba.prange(H):
        if i % 2 == 0:
            for j in range(W):
                out[i, j] = lr[bayer[i, j]] if j % 2 == 0 else lg[bayer[i, j]]
        else:
            for j in range(W):
                out[i, j] = lg[bayer[i, j]] if j % 2 == 0 else lb[bayer[i, j]]
    return out


def _warmup_numba() -> None:
    dummy16 = np.zeros((4, 4, 3), dtype=np.uint16)
    dummy_bayer = np.zeros((4, 4), dtype=np.uint16)
    dummy_lut = np.zeros(65536, dtype=np.uint8)
    dummy_lut16 = np.zeros(65536, dtype=np.uint16)
    _lut3(dummy16, dummy_lut, dummy_lut, dummy_lut)
    _wb_lut_bayer(dummy_bayer, dummy_lut16, dummy_lut16, dummy_lut16)


def _build_wb_luts(wb: np.ndarray, black: int = BLACK) -> tuple:
    inp = np.arange(65536, dtype=np.float32)
    scale = 65535.0 / (65535 - black)
    lr = np.clip((inp - black) * wb[0] * scale, 0, 65535).astype(np.uint16)
    lg = np.clip((inp - black) * scale, 0, 65535).astype(np.uint16)
    lb = np.clip((inp - black) * wb[2] * scale, 0, 65535).astype(np.uint16)
    return lr, lg, lb


def _build_gamma_lut(threshold: float) -> np.ndarray:
    """Build a uint16→uint8 LUT combining auto-bright + BT.709 gamma."""
    x = np.arange(65536, dtype=np.float32) / (threshold * 65535.0)
    np.clip(x, 0.0, 1.0, out=x)
    gamma = np.where(
        x < 0.018,
        4.5 * x,
        1.099 * np.power(np.maximum(x, 1e-9), 0.45) - 0.099,
    )
    return np.clip(gamma * 255, 0, 255).astype(np.uint8)


def demosaic_fast(iiq_path: str | Path, clip_frac: float = 0.004) -> np.ndarray:
    """
    Fast IIQ demosaic bypassing LibRaw's single-threaded PPG algorithm.

    Parameters
    ----------
    iiq_path : path to .IIQ file
    clip_frac : auto-brightness highlight clip fraction (default 0.4%)

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8, RGB channel order
    """
    raw = rawpy.imread(str(iiq_path))
    b16 = raw.raw_image_visible.copy()
    wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
    wb /= wb[1]
    raw.close()

    # Black subtraction + WB on Bayer data (numba parallel)
    lr, lg, lb = _build_wb_luts(wb)
    b_corr = _wb_lut_bayer(b16, lr, lg, lb)

    # Edge-aware Bayer demosaic → uint16 RGB (cv2 RGGB→RGB output)
    rgb16 = cv2.cvtColor(b_corr, cv2.COLOR_BAYER_RG2BGR_EA)

    # Auto-brightness: subsample luma, clip 0.4% of highlights (empirically matched to LibRaw)
    sub = rgb16[::8, ::8].astype(np.float32) / 65535.0
    luma = 0.2126 * sub[:, :, 0] + 0.7152 * sub[:, :, 1] + 0.0722 * sub[:, :, 2]
    hist, edges = np.histogram(luma.ravel(), bins=4096, range=(0.0, 1.0))
    idx = np.searchsorted(np.cumsum(hist), luma.size * (1.0 - clip_frac))
    threshold = float(edges[min(idx + 1, len(edges) - 1)])

    gamma_lut = _build_gamma_lut(threshold)
    return _lut3(rgb16, gamma_lut, gamma_lut, gamma_lut)


def main() -> None:
    iiq = next(Path("PhaseOneSample").glob("*.IIQ"))
    print(f"File: {iiq}  ({iiq.stat().st_size / 1024**2:.0f} MB)\n")

    print("Warming up numba JIT (first-call compilation)...", end=" ", flush=True)
    _warmup_numba()
    print("done\n")

    # --- LibRaw PPG reference ---
    print("Running LibRaw PPG reference...")
    t0 = time.perf_counter()
    raw = rawpy.imread(str(iiq))
    rgb_ref = raw.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.PPG,
        half_size=False,
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=8,
    )
    raw.close()
    libraw_ms = (time.perf_counter() - t0) * 1000

    # --- fast pipeline ---
    print("Running fast pipeline...")
    t0 = time.perf_counter()
    rgb_fast = demosaic_fast(iiq)
    fast_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'':4}{'Pipeline':<30} {'Time':>8}  {'vs LibRaw':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'LibRaw PPG (current)':<30} {libraw_ms:>7.0f}ms  {'1.00×':>10}")
    print(
        f"  {'Fast (numba+cv2 EA)':<30} {fast_ms:>7.0f}ms  {libraw_ms / fast_ms:>9.2f}×"
    )

    # --- quality ---
    diff = rgb_ref.astype(np.int16) - rgb_fast.astype(np.int16)
    abs_diff = np.abs(diff)
    print("\nQuality vs LibRaw PPG:")
    print(f"  Mean abs diff:   {abs_diff.mean():.2f} / 255")
    print(f"  Median abs diff: {np.median(abs_diff):.1f} / 255")
    print(f"  95th pct diff:   {np.percentile(abs_diff, 95):.1f} / 255")
    print(f"  Max diff:        {abs_diff.max()} / 255")
    print(
        f"  Per-channel mean: R={abs_diff[:, :, 0].mean():.2f}  "
        f"G={abs_diff[:, :, 1].mean():.2f}  B={abs_diff[:, :, 2].mean():.2f}"
    )
    print(
        f"  Output means — fast: R={rgb_fast[:, :, 0].mean():.1f}  "
        f"G={rgb_fast[:, :, 1].mean():.1f}  B={rgb_fast[:, :, 2].mean():.1f}"
    )
    print(
        f"  Output means — ref:  R={rgb_ref[:, :, 0].mean():.1f}  "
        f"G={rgb_ref[:, :, 1].mean():.1f}  B={rgb_ref[:, :, 2].mean():.1f}"
    )
    mean_fast = [rgb_fast[:, :, c].mean() for c in range(3)]
    mean_ref = [rgb_ref[:, :, c].mean() for c in range(3)]
    print(
        f"  Mean ratio fast/ref: R={mean_fast[0] / mean_ref[0]:.3f}  "
        f"G={mean_fast[1] / mean_ref[1]:.3f}  B={mean_fast[2] / mean_ref[2]:.3f}"
    )

    # --- save crops + full images ---
    output_dir = Path("output/fast_demosaic_research")
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = rgb_ref.shape[:2]
    for name, (r0, c0, r1, c1) in {
        "centre": (H // 2 - 500, W // 2 - 500, H // 2 + 500, W // 2 + 500),
        "top_left": (200, 200, 1200, 1200),
        "bottom_right": (H - 1200, W - 1200, H - 200, W - 200),
    }.items():
        sbs = np.concatenate([rgb_ref[r0:r1, c0:c1], rgb_fast[r0:r1, c0:c1]], axis=1)
        cv2.putText(
            sbs, "LibRaw PPG", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2
        )
        cv2.putText(
            sbs,
            "Fast (EA+LUT)",
            (sbs.shape[1] // 2 + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            2,
        )
        cv2.imwrite(
            str(output_dir / f"compare_{name}.jpg"),
            cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    cv2.imwrite(
        str(output_dir / "libraw_ppg.jpg"),
        cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    cv2.imwrite(
        str(output_dir / "fast_demosaic.jpg"),
        cv2.cvtColor(rgb_fast, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    print(f"\nSaved crops + full images → {output_dir}/")


if __name__ == "__main__":
    main()
