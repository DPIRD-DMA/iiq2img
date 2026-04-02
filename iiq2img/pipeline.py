"""Fast demosaic pipeline: cv2 edge-aware Bayer + numba parallel LUTs.

~4.6x faster demosaic than LibRaw PPG (~600ms vs ~2.9s for 120 MP).
~3x faster end-to-end including JPEG encoding.
Mean pixel diff ~6.5/255 vs LibRaw. See FINDINGS.md for details.
"""

from pathlib import Path

import cv2
import numba
import numpy as np
import rawpy

_IIQ_BLACK_LEVEL = 1024  # rawpy incorrectly reports 0; true value confirmed empirically

_numba_warmed_up = False


def _ensure_numba_warmed_up() -> None:
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    dummy16 = np.zeros((4, 4, 3), dtype=np.uint16)
    dummy_bayer = np.zeros((4, 4), dtype=np.uint16)
    dummy_lut = np.zeros(65536, dtype=np.uint8)
    dummy_lut16 = np.zeros(65536, dtype=np.uint16)
    _fast_lut3(dummy16, dummy_lut, dummy_lut, dummy_lut)
    _fast_wb_lut_bayer(dummy_bayer, dummy_lut16, dummy_lut16, dummy_lut16)
    _numba_warmed_up = True


@numba.njit(parallel=True, cache=True)
def _fast_lut3(
    img: np.ndarray, lr: np.ndarray, lg: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """Apply 3 independent uint16->uint8 LUTs to an (H,W,3) image in parallel."""
    H, W = img.shape[0], img.shape[1]
    out = np.empty((H, W, 3), numba.uint8)  # type: ignore[call-overload]
    for i in numba.prange(H):
        for j in range(W):
            out[i, j, 0] = lr[img[i, j, 0]]
            out[i, j, 1] = lg[img[i, j, 1]]
            out[i, j, 2] = lb[img[i, j, 2]]
    return out


@numba.njit(parallel=True, cache=True)
def _fast_wb_lut_bayer(
    bayer: np.ndarray, lr: np.ndarray, lg: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """Apply per-channel uint16->uint16 black+WB LUTs to RGGB Bayer in parallel."""
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


def _build_wb_luts(wb: np.ndarray, black: int = _IIQ_BLACK_LEVEL) -> tuple:
    inp = np.arange(65536, dtype=np.float32)
    # Scale to fill uint16 range after black subtraction (matches LibRaw behavior)
    scale = 65535.0 / (65535 - black)
    lr = np.clip((inp - black) * wb[0] * scale, 0, 65535).astype(np.uint16)
    lg = np.clip((inp - black) * scale, 0, 65535).astype(np.uint16)
    lb = np.clip((inp - black) * wb[2] * scale, 0, 65535).astype(np.uint16)
    return lr, lg, lb


def _build_gamma_lut(threshold: float) -> np.ndarray:
    """uint16 -> uint8 LUT combining auto-brightness scale + BT.709 gamma.

    Uses BT.709 gamma (2.222, 4.5) to match LibRaw's default postprocess output.
    """
    x = np.arange(65536, dtype=np.float32) / (threshold * 65535.0)
    np.clip(x, 0.0, 1.0, out=x)
    gamma = np.where(
        x < 0.018,
        4.5 * x,
        1.099 * np.power(np.maximum(x, 1e-9), 0.45) - 0.099,
    )
    return np.clip(gamma * 255, 0, 255).astype(np.uint8)


def demosaic_fast(iiq_path: Path) -> np.ndarray:
    """
    Fast IIQ demosaic: cv2 edge-aware Bayer + numba LUT pipeline (~4.6x vs LibRaw PPG).

    Pipeline: raw Bayer -> black subtraction + WB + full-range scale (numba LUT)
    -> cv2 EA demosaic -> auto-brightness (0.4% luma clip) -> BT.709 gamma (numba LUT)
    -> uint8 RGB
    """
    _ensure_numba_warmed_up()

    try:
        raw = rawpy.imread(str(iiq_path))
    except rawpy.LibRawIOError:  # type: ignore[attr-defined]
        raise OSError(
            f"Failed to read IIQ file (corrupt or unreadable): {iiq_path}"
        ) from None

    b16 = raw.raw_image_visible.copy()
    wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
    wb /= wb[1]
    raw.close()

    # Black subtraction + WB on raw Bayer data
    lr, lg, lb = _build_wb_luts(wb)
    b_corr = _fast_wb_lut_bayer(b16, lr, lg, lb)

    # Edge-aware demosaic: RGGB uint16 -> uint16, output is RGB order (not BGR)
    rgb16 = cv2.cvtColor(b_corr, cv2.COLOR_BAYER_RG2BGR_EA)

    # Auto-brightness: subsample luma, clip 0.4% of highlights (empirically matched to LibRaw)
    sub = rgb16[::8, ::8].astype(np.float32) / 65535.0
    luma = 0.2126 * sub[:, :, 0] + 0.7152 * sub[:, :, 1] + 0.0722 * sub[:, :, 2]
    hist, edges = np.histogram(luma.ravel(), bins=4096, range=(0.0, 1.0))
    idx = np.searchsorted(np.cumsum(hist), luma.size * 0.996)
    threshold = float(edges[min(idx + 1, len(edges) - 1)])

    gamma_lut = _build_gamma_lut(threshold)
    return _fast_lut3(rgb16, gamma_lut, gamma_lut, gamma_lut)
