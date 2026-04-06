"""Tests for pipeline internal functions: LUTs, gamma curve, warmup."""

import numpy as np

from iiq2img.pipeline import (
    _IIQ_BLACK_LEVEL,
    _build_gamma_lut,
    _build_wb_luts,
    _ensure_numba_warmed_up,
    _fast_lut3,
    _fast_wb_lut_bayer,
)


class TestBuildGammaLut:
    def test_output_shape_and_dtype(self):
        lut = _build_gamma_lut(1.0)
        assert lut.shape == (65536,)
        assert lut.dtype == np.uint8

    def test_black_maps_to_zero(self):
        lut = _build_gamma_lut(1.0)
        assert lut[0] == 0

    def test_max_input_maps_to_255(self):
        lut = _build_gamma_lut(1.0)
        assert lut[65535] == 255

    def test_monotonically_increasing(self):
        lut = _build_gamma_lut(1.0)
        # Allow equal values (plateau) but never decreasing
        assert np.all(np.diff(lut.astype(np.int16)) >= 0)

    def test_lower_threshold_brighter(self):
        """Lower threshold means more of the range maps to bright values."""
        bright = _build_gamma_lut(0.5)
        normal = _build_gamma_lut(1.0)
        # At midpoint (32768), brighter LUT should map higher
        assert bright[32768] >= normal[32768]

    def test_bt709_linear_segment(self):
        """BT.709 has a linear segment for x < 0.018."""
        lut = _build_gamma_lut(1.0)
        # At very low input, output should be close to linear (4.5x)
        # Input 100/65535 ≈ 0.0015, well within linear range
        x = 100 / 65535
        expected = 4.5 * x * 255
        assert abs(int(lut[100]) - expected) < 2


class TestBuildWbLuts:
    def test_output_shapes(self):
        wb = np.array([2.0, 1.0, 1.5], dtype=np.float32)
        lr, lg, lb = _build_wb_luts(wb)
        assert lr.shape == (65536,)
        assert lg.shape == (65536,)
        assert lb.shape == (65536,)
        assert lr.dtype == np.uint16

    def test_black_level_subtracted(self):
        wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        lr, lg, lb = _build_wb_luts(wb)
        # Values at or below black level should map to 0
        assert lr[0] == 0
        assert lr[_IIQ_BLACK_LEVEL] == 0
        assert lg[_IIQ_BLACK_LEVEL] == 0

    def test_above_black_level_nonzero(self):
        wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        lr, lg, lb = _build_wb_luts(wb)
        # Well above black level should be non-zero
        assert lr[_IIQ_BLACK_LEVEL + 1000] > 0

    def test_wb_scales_channels(self):
        wb = np.array([2.0, 1.0, 1.5], dtype=np.float32)
        lr, lg, lb = _build_wb_luts(wb)
        mid = 40000
        # Red channel has 2x multiplier, should be brighter than green
        assert lr[mid] > lg[mid]
        # Blue channel has 1.5x multiplier
        assert lb[mid] > lg[mid]

    def test_clipping_at_65535(self):
        wb = np.array([3.0, 1.0, 3.0], dtype=np.float32)
        lr, _, lb = _build_wb_luts(wb)
        assert lr.max() <= 65535
        assert lb.max() <= 65535


class TestFastLut3:
    def test_basic_lut_application(self):
        _ensure_numba_warmed_up()
        img = np.array([[[100, 200, 300]]], dtype=np.uint16)
        lut_r = np.arange(65536, dtype=np.uint8)  # identity mod 256
        lut_g = np.zeros(65536, dtype=np.uint8)
        lut_b = np.full(65536, 42, dtype=np.uint8)

        result = _fast_lut3(img, lut_r, lut_g, lut_b)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 100  # identity
        assert result[0, 0, 1] == 0  # zero LUT
        assert result[0, 0, 2] == 42  # constant LUT

    def test_output_shape(self):
        _ensure_numba_warmed_up()
        img = np.zeros((10, 20, 3), dtype=np.uint16)
        lut = np.zeros(65536, dtype=np.uint8)
        result = _fast_lut3(img, lut, lut, lut)
        assert result.shape == (10, 20, 3)


class TestFastWbLutBayer:
    def test_rggb_pattern(self):
        """Verify RGGB Bayer pattern: R at even row/even col, G elsewhere, B at odd/odd."""
        _ensure_numba_warmed_up()
        bayer = np.full((4, 4), 1000, dtype=np.uint16)
        lr = np.full(65536, 10, dtype=np.uint16)  # R → 10
        lg = np.full(65536, 20, dtype=np.uint16)  # G → 20
        lb = np.full(65536, 30, dtype=np.uint16)  # B → 30

        result = _fast_wb_lut_bayer(bayer, lr, lg, lb)
        # Even row, even col = R
        assert result[0, 0] == 10
        # Even row, odd col = G
        assert result[0, 1] == 20
        # Odd row, even col = G
        assert result[1, 0] == 20
        # Odd row, odd col = B
        assert result[1, 1] == 30


class TestEnsureNumbaWarmedUp:
    def test_idempotent(self):
        """Calling warmup multiple times should not error."""
        _ensure_numba_warmed_up()
        _ensure_numba_warmed_up()
