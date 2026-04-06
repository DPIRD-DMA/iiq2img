"""Tests for defective row repair."""

from pathlib import Path

import numpy as np
import pytest

from iiq2img.repair import (
    _flag_bad_rows,
    _numba_repair_bayer,
    detect_defective_rows,
    repair_defective_rows,
)


SAMPLE_DIR = Path("PhaseOneSample")
SAMPLE_IIQ = SAMPLE_DIR / "P0286625.IIQ"


# ── Synthetic Bayer helpers ──────────────────────────────────────────────────


def _make_clean_bayer(h: int = 40, w: int = 40, base: int = 2000) -> np.ndarray:
    """Create a synthetic RGGB Bayer array with plausible values."""
    bayer = np.full((h, w), base, dtype=np.uint16)
    bayer[0::2, 0::2] = base  # R
    bayer[0::2, 1::2] = base + 200  # G1
    bayer[1::2, 0::2] = base + 200  # G2
    bayer[1::2, 1::2] = base + 50  # B
    return bayer


def _inject_bad_row(bayer: np.ndarray, full_row: int, value: int = 60000) -> None:
    bayer[full_row, :] = value


# ── _numba_repair_bayer ──────────────────────────────────────────────────────


class TestNumbaRepairBayer:
    def test_repairs_single_masked_pixel(self):
        bayer = _make_clean_bayer(20, 20, base=1000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        bayer[10, 10] = 50000
        mask[10, 10] = True

        _numba_repair_bayer(bayer, mask)

        assert abs(int(bayer[10, 10]) - 1000) < 50

    def test_repairs_bad_row(self):
        bayer = _make_clean_bayer(40, 40, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        _inject_bad_row(bayer, 20, value=60000)
        mask[20, :] = True

        _numba_repair_bayer(bayer, mask)

        for c in range(2, 38):
            assert bayer[20, c] < 5000, f"pixel ({20},{c}) = {bayer[20, c]}"

    def test_leaves_unmasked_pixels_unchanged(self):
        bayer = _make_clean_bayer(20, 20, base=3000)
        original = bayer.copy()
        mask = np.zeros_like(bayer, dtype=np.bool_)

        mask[10, 10] = True
        _numba_repair_bayer(bayer, mask)

        mask[10, 10] = False
        assert np.array_equal(bayer[~mask], original[~mask])

    def test_handles_edge_pixel(self):
        bayer = _make_clean_bayer(20, 20, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        bayer[0, 0] = 50000
        mask[0, 0] = True

        _numba_repair_bayer(bayer, mask)

        assert bayer[0, 0] < 10000

    def test_handles_consecutive_bad_rows(self):
        bayer = _make_clean_bayer(40, 40, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        for r in [18, 19, 20]:
            _inject_bad_row(bayer, r, value=55000)
            mask[r, :] = True

        _numba_repair_bayer(bayer, mask)

        for r in [18, 19, 20]:
            for c in range(4, 36):
                assert bayer[r, c] < 5000, f"pixel ({r},{c}) = {bayer[r, c]}"


# ── _flag_bad_rows ───────────────────────────────────────────────────────────


class TestFlagBadRows:
    def test_flags_severe_row(self):
        bayer = _make_clean_bayer(100, 100, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        _inject_bad_row(bayer, 50, value=40000)

        _flag_bad_rows(bayer, mask, row_thresh_pct=10.0)

        assert mask[50, :].all()

    def test_does_not_flag_clean_rows(self):
        bayer = _make_clean_bayer(100, 100, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        _flag_bad_rows(bayer, mask, row_thresh_pct=10.0)

        assert not mask.any()

    def test_does_not_flag_small_deviation(self):
        bayer = _make_clean_bayer(100, 100, base=2000)
        mask = np.zeros_like(bayer, dtype=np.bool_)

        bayer[50, :] = (bayer[50, :].astype(np.float32) * 1.05).astype(np.uint16)

        _flag_bad_rows(bayer, mask, row_thresh_pct=10.0)

        assert not mask[50, :].any()

    def test_lower_threshold_catches_more(self):
        bayer = _make_clean_bayer(100, 100, base=2000)

        bayer[50, :] = (bayer[50, :].astype(np.float32) * 1.15).astype(np.uint16)

        mask_strict = np.zeros_like(bayer, dtype=np.bool_)
        _flag_bad_rows(bayer, mask_strict, row_thresh_pct=20.0)

        mask_loose = np.zeros_like(bayer, dtype=np.bool_)
        _flag_bad_rows(bayer, mask_loose, row_thresh_pct=10.0)

        assert mask_loose[50, :].any()
        assert not mask_strict[50, :].any()


# ── detect_defective_rows ───────────────────────────────────────────────────


class TestDetectDefectiveRows:
    def test_detects_bad_row(self):
        bayer = _make_clean_bayer(100, 100, base=2000)
        _inject_bad_row(bayer, 50, value=40000)

        mask = detect_defective_rows(bayer)

        assert mask[50, :].all()

    def test_clean_bayer_returns_empty_mask(self):
        bayer = _make_clean_bayer(100, 100, base=2000)

        mask = detect_defective_rows(bayer)

        assert not mask.any()


# ── repair_defective_rows ────────────────────────────────────────────────────


class TestRepairDefectiveRows:
    def test_returns_copy(self):
        bayer = _make_clean_bayer(40, 40)
        result = repair_defective_rows(bayer)
        assert result is not bayer

    def test_repairs_bad_row(self):
        bayer = _make_clean_bayer(100, 100, base=2000)
        _inject_bad_row(bayer, 50, value=40000)

        result = repair_defective_rows(bayer)

        assert bayer[50, 0] == 40000  # original unchanged
        for c in range(4, 96):
            assert result[50, c] < 5000

    def test_preserves_clean_data(self):
        bayer = _make_clean_bayer(40, 40, base=2000)
        result = repair_defective_rows(bayer)

        assert np.array_equal(bayer, result)

    def test_output_dtype_uint16(self):
        bayer = _make_clean_bayer(40, 40)
        result = repair_defective_rows(bayer)

        assert result.dtype == np.uint16


# ── Integration with real IIQ ────────────────────────────────────────────────


@pytest.mark.skipif(not SAMPLE_IIQ.exists(), reason="Sample IIQ file not available")
class TestRepairIntegration:
    def test_detect_defects_real_iiq(self):
        import rawpy

        raw = rawpy.imread(str(SAMPLE_IIQ))
        bayer = raw.raw_image_visible.copy()
        raw.close()

        mask = detect_defective_rows(bayer)

        assert mask[3882, :].any(), "Row 3882 should be flagged"

        bad_rows = np.where(mask.any(axis=1))[0]
        assert len(bad_rows) < 20, f"Too many flagged rows: {len(bad_rows)}"

    def test_repair_reduces_row_3882_deviation(self):
        import rawpy

        raw = rawpy.imread(str(SAMPLE_IIQ))
        bayer = raw.raw_image_visible.copy()
        raw.close()

        repaired = repair_defective_rows(bayer)

        orig_mean = float(bayer[3882, 0::2].mean())
        fixed_mean = float(repaired[3882, 0::2].mean())
        neighbor_mean = float(bayer[3880, 0::2].mean())

        assert orig_mean > 10000
        assert abs(fixed_mean - neighbor_mean) < 500

    def test_fast_pipeline_with_repair(self):
        from iiq2img.pipeline import demosaic_fast

        rgb = demosaic_fast(SAMPLE_IIQ, repair=True)

        assert rgb.shape == (9564, 12768, 3)
        assert rgb.dtype == np.uint8
