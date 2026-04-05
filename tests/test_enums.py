"""Tests for _resolve_pipeline, format constants, and encode helpers."""

import pytest

from iiq2img import (
    normalize_format,
    FORMAT_EXTENSIONS,
    format_from_path,
)
from iiq2img.converter import _resolve_pipeline, _resolve_rotate
from iiq2img.encode import resolve_format

from pathlib import Path


class TestFormatConstants:
    def test_normalize_format(self):
        assert normalize_format("jpg") == "jpg"
        assert normalize_format("jpeg") == "jpg"
        assert normalize_format("png") == "png"
        assert normalize_format("tif") == "tiff"
        assert normalize_format("tiff") == "tiff"

    def test_format_extensions(self):
        assert FORMAT_EXTENSIONS["jpg"] == ".jpg"
        assert FORMAT_EXTENSIONS["png"] == ".png"
        assert FORMAT_EXTENSIONS["tiff"] == ".tif"

    def test_format_extensions_complete(self):
        """Every canonical format has an extension mapping."""
        for fmt in ("jpg", "png", "tiff"):
            assert fmt in FORMAT_EXTENSIONS
            assert FORMAT_EXTENSIONS[fmt].startswith(".")

    def test_normalize_format_case_insensitive(self):
        assert normalize_format("JPG") == "jpg"
        assert normalize_format("PNG") == "png"
        assert normalize_format("TIFF") == "tiff"
        assert normalize_format(".jpg") == "jpg"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            normalize_format("bmp")


class TestResolvePipeline:
    def test_resolve_pipeline_from_string(self):
        assert _resolve_pipeline("libraw") == "libraw"
        assert _resolve_pipeline("fast") == "fast"

    def test_resolve_pipeline_case_insensitive(self):
        assert _resolve_pipeline("FAST") == "fast"
        assert _resolve_pipeline("LibRaw") == "libraw"

    def test_resolve_pipeline_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            _resolve_pipeline("invalid")


class TestFormatFromPath:
    def test_known_extensions(self):
        assert format_from_path(Path("photo.jpg")) == "jpg"
        assert format_from_path(Path("photo.png")) == "png"
        assert format_from_path(Path("photo.tif")) == "tiff"
        assert format_from_path(Path("photo.tiff")) == "tiff"

    def test_unknown_extension_returns_none(self):
        assert format_from_path(Path("photo.bmp")) is None
        assert format_from_path(Path("photo.gif")) is None

    def test_multiple_dots_in_path(self):
        assert format_from_path(Path("my.photo.v2.jpg")) == "jpg"


class TestResolveFormat:
    def test_explicit_format_wins(self):
        assert resolve_format("png", Path("photo.jpg")) == "png"

    def test_infer_from_path(self):
        assert resolve_format(None, Path("photo.tif")) == "tiff"

    def test_default_to_jpg(self):
        assert resolve_format(None, None) == "jpg"

    def test_explicit_format_normalized(self):
        assert resolve_format("JPEG", None) == "jpg"
        assert resolve_format(".tiff", None) == "tiff"

    def test_unknown_path_defaults_to_jpg(self):
        assert resolve_format(None, Path("photo.bmp")) == "jpg"

    def test_normalize_whitespace(self):
        assert normalize_format("  jpg  ") == "jpg"
        assert normalize_format(" .PNG ") == "png"


class TestResolveRotate:
    def test_valid_rotations(self):
        assert _resolve_rotate(0) == 0
        assert _resolve_rotate(90) == 90
        assert _resolve_rotate(180) == 180
        assert _resolve_rotate(270) == 270

    def test_360_maps_to_0(self):
        assert _resolve_rotate(360) == 0

    def test_invalid_angle_raises(self):
        with pytest.raises(ValueError, match="Invalid rotation: 45"):
            _resolve_rotate(45)

    def test_negative_angle_raises(self):
        with pytest.raises(ValueError, match="Invalid rotation: -90"):
            _resolve_rotate(-90)

    def test_arbitrary_angle_raises(self):
        with pytest.raises(ValueError, match="Invalid rotation: 123"):
            _resolve_rotate(123)
