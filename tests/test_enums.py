"""Tests for enums, ConvertResult dataclass, and constants."""

import pytest

from iiq2img import (
    ConvertResult,
    Pipeline,
    normalize_format,
    Quality,
    FORMAT_EXTENSIONS,
)
from iiq2img.converter import _resolve_pipeline


class TestEnums:
    def test_quality_values(self):
        assert Quality.THUMBNAIL.value == "thumbnail"
        assert Quality.FULL.value == "full"

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


class TestFormatEdgeCases:
    def test_quality_members(self):
        assert len(Quality) == 2
        assert set(q.value for q in Quality) == {"thumbnail", "full"}

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


class TestConvertResult:
    def test_default_metadata(self):
        r = ConvertResult(
            output_path="/tmp/out.jpg",
            width=100,
            height=200,
            elapsed_ms=50.0,
            file_size_bytes=1024,
        )
        assert r.metadata == {}

    def test_with_metadata(self):
        meta = {"Make": "Phase One"}
        r = ConvertResult(
            output_path="/tmp/out.jpg",
            width=100,
            height=200,
            elapsed_ms=50.0,
            file_size_bytes=1024,
            metadata=meta,
        )
        assert r.metadata == meta


class TestConvertResultEdgeCases:
    def test_all_fields_set(self):
        r = ConvertResult(
            output_path="/tmp/out.jpg",
            width=12768,
            height=9564,
            elapsed_ms=3400.5,
            file_size_bytes=15_000_000,
            metadata={"Make": "Phase One", "Model": "iXM-GS120"},
        )
        assert r.output_path == "/tmp/out.jpg"
        assert r.width == 12768
        assert r.height == 9564
        assert r.elapsed_ms == 3400.5
        assert r.file_size_bytes == 15_000_000
        assert len(r.metadata) == 2

    def test_equality(self):
        kwargs = dict(
            output_path="/tmp/out.jpg",
            width=100,
            height=200,
            elapsed_ms=50.0,
            file_size_bytes=1024,
        )
        r1 = ConvertResult(**kwargs)
        r2 = ConvertResult(**kwargs)
        assert r1 == r2

    def test_zero_dimensions(self):
        r = ConvertResult(
            output_path="/tmp/out.jpg",
            width=0,
            height=0,
            elapsed_ms=0.0,
            file_size_bytes=0,
        )
        assert r.width == 0
        assert r.height == 0


class TestPipeline:
    def test_pipeline_values(self):
        assert Pipeline.LIBRAW.value == "libraw"
        assert Pipeline.FAST.value == "fast"

    def test_pipeline_members(self):
        assert len(Pipeline) == 2
        assert set(p.value for p in Pipeline) == {"libraw", "fast"}

    def test_resolve_pipeline_from_string(self):
        assert _resolve_pipeline("libraw") == Pipeline.LIBRAW
        assert _resolve_pipeline("fast") == Pipeline.FAST

    def test_resolve_pipeline_case_insensitive(self):
        assert _resolve_pipeline("FAST") == Pipeline.FAST
        assert _resolve_pipeline("LibRaw") == Pipeline.LIBRAW

    def test_resolve_pipeline_from_enum(self):
        assert _resolve_pipeline(Pipeline.FAST) == Pipeline.FAST
        assert _resolve_pipeline(Pipeline.LIBRAW) == Pipeline.LIBRAW

    def test_resolve_pipeline_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            _resolve_pipeline("invalid")
