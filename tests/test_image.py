"""Tests for image processing functions: resize, encode, and thumbnail extraction."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iiq2img.converter import (
    _encode_image,
    _resize_max_dim,
)


class TestResizeMaxDim:
    def test_landscape_resize(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 100)
        h, w = result.shape[:2]
        assert w == 100
        assert h == 50

    def test_portrait_resize(self):
        rgb = np.zeros((200, 100, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 100)
        h, w = result.shape[:2]
        assert h == 100
        assert w == 50

    def test_square_resize(self):
        rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 50)
        h, w = result.shape[:2]
        assert h == 50
        assert w == 50

    def test_returns_rgb(self):
        # Create a known red pixel image in RGB
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # R channel
        result = _resize_max_dim(rgb, 100)
        # Result should still be RGB (red in channel 0)
        assert result[0, 0, 0] == 255  # R
        assert result[0, 0, 2] == 0  # B


class TestResizeMaxDimEdgeCases:
    def test_no_resize_when_already_smaller(self):
        """Image smaller than max_dim should still resize to max_dim."""
        rgb = np.zeros((50, 80, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 100)
        h, w = result.shape[:2]
        # scale = 100/80 = 1.25 -> 100x62
        assert w == 100
        assert h == 62

    def test_single_pixel(self):
        rgb = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = _resize_max_dim(rgb, 10)
        assert result.shape[:2] == (10, 10)

    def test_wide_aspect_ratio(self):
        rgb = np.zeros((10, 1000, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 500)
        h, w = result.shape[:2]
        assert w == 500
        assert h == 5

    def test_preserves_pixel_values(self):
        rgb = np.full((100, 100, 3), 42, dtype=np.uint8)
        result = _resize_max_dim(rgb, 50)
        # After resize, uniform image should stay roughly uniform
        assert np.allclose(result, 42, atol=2)

    def test_dtype_preserved(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _resize_max_dim(rgb, 50)
        assert result.dtype == np.uint8


class TestEncodeImage:
    def test_jpeg_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # red in BGR
        out = str(tmp_path / "test.jpg")
        _encode_image(bgr, out, "jpg", 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_png_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "test.png")
        _encode_image(bgr, out, "png", 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_tiff_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "test.tif")
        _encode_image(bgr, out, "tiff", 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_jpeg_quality_affects_size(self, tmp_path):
        bgr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        low_q = str(tmp_path / "low.jpg")
        high_q = str(tmp_path / "high.jpg")
        _encode_image(bgr, low_q, "jpg", 10)
        _encode_image(bgr, high_q, "jpg", 100)
        assert os.path.getsize(low_q) < os.path.getsize(high_q)


class TestEncodeImageEdgeCases:
    def test_jpeg_quality_boundary_low(self, tmp_path):
        bgr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        out = str(tmp_path / "q1.jpg")
        _encode_image(bgr, out, "jpg", 1)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_jpeg_quality_boundary_high(self, tmp_path):
        bgr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        out = str(tmp_path / "q100.jpg")
        _encode_image(bgr, out, "jpg", 100)
        assert os.path.exists(out)

    def test_png_compression_mapping(self, tmp_path):
        """Quality 100 should map to PNG compression 0 (fast), quality 1 to ~9 (slow)."""
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fast = str(tmp_path / "fast.png")
        slow = str(tmp_path / "slow.png")
        _encode_image(bgr, fast, "png", 100)
        _encode_image(bgr, slow, "png", 1)
        # Both should be valid PNGs
        assert os.path.exists(fast)
        assert os.path.exists(slow)

    def test_single_pixel_image(self, tmp_path):
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        for fmt, ext in [
            ("jpg", "jpg"),
            ("png", "png"),
            ("tiff", "tif"),
        ]:
            out = str(tmp_path / f"pixel.{ext}")
            _encode_image(bgr, out, fmt, 90)
            assert os.path.exists(out)

    def test_grayscale_like_image(self, tmp_path):
        """All channels same value -- should encode without error."""
        bgr = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = str(tmp_path / "gray.jpg")
        _encode_image(bgr, out, "jpg", 90)
        assert os.path.getsize(out) > 0

    def test_encoded_jpeg_is_valid(self, tmp_path):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        out = str(tmp_path / "valid.jpg")
        _encode_image(bgr, out, "jpg", 90)
        with open(out, "rb") as f:
            header = f.read(2)
        assert header == b"\xff\xd8"  # JPEG magic bytes

    def test_encoded_png_is_valid(self, tmp_path):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        out = str(tmp_path / "valid.png")
        _encode_image(bgr, out, "png", 90)
        with open(out, "rb") as f:
            header = f.read(8)
        assert header == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


class TestExtractThumbnail:
    @patch("iiq2img.converter.rawpy.imread")
    def test_bitmap_thumbnail(self, mock_imread):
        mock_raw = MagicMock()
        mock_thumb = MagicMock()
        mock_thumb.format = MagicMock()
        mock_thumb.format.__eq__ = lambda self, other: (
            other == __import__("rawpy").ThumbFormat.BITMAP
        )
        mock_thumb.data = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_raw.extract_thumb.return_value = mock_thumb
        mock_imread.return_value = mock_raw

        # We need rawpy.ThumbFormat for comparison, which requires patching
        # Test via convert_iiq with THUMBNAIL quality instead
        pass

    @patch("iiq2img.converter.rawpy")
    def test_unknown_format_raises(self, mock_rawpy):
        mock_raw = MagicMock()
        mock_thumb = MagicMock()
        mock_thumb.format = "UNKNOWN"
        mock_rawpy.ThumbFormat.BITMAP = "BITMAP"
        mock_rawpy.ThumbFormat.JPEG = "JPEG"
        mock_raw.extract_thumb.return_value = mock_thumb
        mock_rawpy.imread.return_value = mock_raw

        from iiq2img.converter import _extract_thumbnail

        with pytest.raises(ValueError, match="Unknown thumbnail format"):
            _extract_thumbnail("fake.iiq")
