"""Defective row repair for Phase One iXM-GS120 raw Bayer data.

Detects severely corrupted sensor rows by comparing per-channel row means
to a local rolling median.  Only rows deviating by more than
``row_thresh_pct`` percent are flagged (default 10% — catches ~3 rows
per image on the iXM-GS120).

Flagged pixels are replaced with a weighted average of same-channel
neighbours.  The numba repair kernel parallelises over rows.
"""

from __future__ import annotations

import numba
import numpy as np


# ── Public API ───────────────────────────────────────────────────────────────


def repair_defective_rows(
    bayer: np.ndarray,
    row_thresh_pct: float = 10.0,
) -> np.ndarray:
    """Detect and repair severely corrupted rows in a raw RGGB Bayer array.

    Args:
        bayer: Raw Bayer image (H, W), uint16, RGGB pattern.
        row_thresh_pct: Minimum percent deviation of a row mean from its
                        local median to be flagged as defective.

    Returns:
        Corrected copy of the Bayer array (uint16).
    """
    out = bayer.copy()
    mask = detect_defective_rows(bayer, row_thresh_pct)

    if mask.any():
        _numba_repair_bayer(out, mask)

    return out


def detect_defective_rows(
    bayer: np.ndarray,
    row_thresh_pct: float = 10.0,
) -> np.ndarray:
    """Return a boolean mask of defective pixels in the Bayer array."""
    H, W = bayer.shape
    mask = np.zeros((H, W), dtype=np.bool_)
    _flag_bad_rows(bayer, mask, row_thresh_pct)
    return mask


# ── Row-defect detection ─────────────────────────────────────────────────────


def _flag_bad_rows(
    bayer: np.ndarray,
    mask: np.ndarray,
    row_thresh_pct: float,
) -> None:
    """Flag entire rows that deviate by more than row_thresh_pct from local median.

    Uses vectorised sliding-window median via stride tricks.
    """
    hw = 15
    channel_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for row_off, col_off in channel_offsets:
        plane = bayer[row_off::2, col_off::2]
        plane_mask = mask[row_off::2, col_off::2]
        nrows = plane.shape[0]

        # int32 sum is ~25% faster than float64 for uint16 data;
        # max possible sum = 65535 * 6384 = 418M, well within int32 range.
        row_means = plane.sum(axis=1, dtype=np.int32).astype(np.float32) / plane.shape[1]

        # Vectorised rolling median via stride tricks
        padded = np.pad(row_means, (hw, hw), mode="edge")
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(nrows, 2 * hw + 1), strides=strides,
        )
        smoothed = np.median(windows, axis=1)

        # Flag rows with large percent deviation
        safe_smooth = np.maximum(smoothed, 1.0)
        pct_dev = np.abs(row_means - smoothed) / safe_smooth * 100.0
        bad_rows = np.where(pct_dev > row_thresh_pct)[0]

        for r in bad_rows:
            plane_mask[r, :] = True


# ── Numba repair kernel ─────────────────────────────────────────────────────


@numba.njit(parallel=True, cache=True)
def _numba_repair_bayer(bayer: np.ndarray, mask: np.ndarray) -> None:
    """Replace masked pixels with weighted average of clean same-channel neighbours."""
    H, W = bayer.shape
    for r in numba.prange(H):
        for c in range(W):
            if not mask[r, c]:
                continue

            total_w = 0.0
            total_v = 0.0

            for dist in (2, 4):
                for dr, dc in ((-dist, 0), (dist, 0), (0, -dist), (0, dist)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < H and 0 <= nc < W and not mask[nr, nc]:
                        w = 1.0 / (abs(dr) + abs(dc))
                        total_w += w
                        total_v += w * bayer[nr, nc]

            if total_w > 0.0:
                val = total_v / total_w
                if val < 0.0:
                    val = 0.0
                if val > 65535.0:
                    val = 65535.0
                bayer[r, c] = numba.uint16(val)
