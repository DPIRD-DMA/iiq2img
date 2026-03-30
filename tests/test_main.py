"""Tests for the SKW_RAW IIQ converter."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from iiq2img import (
    ConvertResult,
    OutputFormat,
    Quality,
    FORMAT_EXTENSIONS,
    convert_iiq,
    extract_metadata,
)
from iiq2img.converter import (
    _encode_image,
    _inject_metadata_into_jpeg,
    _inject_metadata_into_png,
    _resize_max_dim,
    batch_convert,
)


# ---------------------------------------------------------------------------
# Enum / dataclass / constant tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_quality_values(self):
        assert Quality.THUMBNAIL.value == "thumbnail"
        assert Quality.FULL.value == "full"

    def test_output_format_values(self):
        assert OutputFormat.JPEG.value == "jpg"
        assert OutputFormat.PNG.value == "png"
        assert OutputFormat.TIFF.value == "tiff"

    def test_format_extensions(self):
        assert FORMAT_EXTENSIONS[OutputFormat.JPEG] == ".jpg"
        assert FORMAT_EXTENSIONS[OutputFormat.PNG] == ".png"
        assert FORMAT_EXTENSIONS[OutputFormat.TIFF] == ".tif"


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


# ---------------------------------------------------------------------------
# _resize_max_dim
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------


class TestEncodeImage:
    def test_jpeg_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # red in BGR
        out = str(tmp_path / "test.jpg")
        _encode_image(bgr, out, OutputFormat.JPEG, 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_png_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "test.png")
        _encode_image(bgr, out, OutputFormat.PNG, 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_tiff_encode(self, tmp_path):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "test.tif")
        _encode_image(bgr, out, OutputFormat.TIFF, 90)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_jpeg_quality_affects_size(self, tmp_path):
        bgr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        low_q = str(tmp_path / "low.jpg")
        high_q = str(tmp_path / "high.jpg")
        _encode_image(bgr, low_q, OutputFormat.JPEG, 10)
        _encode_image(bgr, high_q, OutputFormat.JPEG, 100)
        assert os.path.getsize(low_q) < os.path.getsize(high_q)


# ---------------------------------------------------------------------------
# _inject_metadata_into_jpeg
# ---------------------------------------------------------------------------


class TestInjectMetadataJpeg:
    def _make_jpeg(self, tmp_path) -> str:
        """Create a minimal valid JPEG file."""
        path = str(tmp_path / "test.jpg")
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(path, bgr)
        return path

    def test_injects_exif_segment(self, tmp_path):
        path = self._make_jpeg(tmp_path)
        exif_bytes = b"Exif\x00\x00" + b"\x00" * 10
        _inject_metadata_into_jpeg(path, exif_bytes, None)
        with open(path, "rb") as f:
            data = f.read()
        assert data[:2] == b"\xff\xd8"
        assert b"Exif\x00\x00" in data

    def test_injects_xmp_segment(self, tmp_path):
        path = self._make_jpeg(tmp_path)
        xmp_bytes = b"<x:xmpmeta>test</x:xmpmeta>"
        _inject_metadata_into_jpeg(path, None, xmp_bytes)
        with open(path, "rb") as f:
            data = f.read()
        assert b"http://ns.adobe.com/xap/1.0/\x00" in data

    def test_no_metadata_no_change(self, tmp_path):
        path = self._make_jpeg(tmp_path)
        with open(path, "rb") as f:
            original = f.read()
        _inject_metadata_into_jpeg(path, None, None)
        with open(path, "rb") as f:
            after = f.read()
        assert original == after

    def test_non_jpeg_file_ignored(self, tmp_path):
        path = str(tmp_path / "not_jpeg.jpg")
        with open(path, "wb") as f:
            f.write(b"not a jpeg")
        _inject_metadata_into_jpeg(path, b"\x00" * 10, None)
        with open(path, "rb") as f:
            data = f.read()
        assert data == b"not a jpeg"


# ---------------------------------------------------------------------------
# _inject_metadata_into_png
# ---------------------------------------------------------------------------


class TestInjectMetadataPng:
    def test_adds_text_chunks(self, tmp_path):
        path = str(tmp_path / "test.png")
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(path, bgr)
        metadata = {"Make": "Phase One", "Model": "iXM-GS120"}
        _inject_metadata_into_png(path, metadata)
        from PIL import Image as PILImage

        img = PILImage.open(path)
        assert img.info.get("Make") == "Phase One"
        assert img.info.get("Model") == "iXM-GS120"

    def test_skips_xmp_key(self, tmp_path):
        path = str(tmp_path / "test.png")
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(path, bgr)
        metadata = {"Make": "Phase One", "xmp": "<big xml blob>"}
        _inject_metadata_into_png(path, metadata)
        from PIL import Image as PILImage

        img = PILImage.open(path)
        assert "xmp" not in img.info


# ---------------------------------------------------------------------------
# extract_metadata (mocked)
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    @patch("iiq2img.converter.PILImage.open")
    def test_returns_exif_tags(self, mock_open):
        mock_img = MagicMock()
        mock_exif = MagicMock()
        mock_exif.items.return_value = [(271, "Phase One"), (272, "iXM-GS120")]
        mock_exif.get.return_value = None
        mock_img.getexif.return_value = mock_exif
        mock_open.return_value = mock_img

        result = extract_metadata("fake.iiq")
        assert result["Make"] == "Phase One"
        assert result["Model"] == "iXM-GS120"

    @patch("iiq2img.converter.PILImage.open")
    def test_extracts_xmp(self, mock_open):
        mock_img = MagicMock()
        mock_exif = MagicMock()
        mock_exif.items.return_value = []
        xmp_data = b"<x:xmpmeta>GPS data</x:xmpmeta>"
        mock_exif.get.return_value = xmp_data
        mock_img.getexif.return_value = mock_exif
        mock_open.return_value = mock_img

        result = extract_metadata("fake.iiq")
        assert "xmp" in result
        assert "GPS data" in result["xmp"]

    @patch("iiq2img.converter.PILImage.open", side_effect=Exception("bad file"))
    def test_returns_empty_on_error(self, mock_open):
        result = extract_metadata("nonexistent.iiq")
        assert result == {}


# ---------------------------------------------------------------------------
# convert_iiq (mocked end-to-end)
# ---------------------------------------------------------------------------


class TestConvertIiq:
    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"Make": "Phase One"})
    @patch("iiq2img.converter._demosaic")
    def test_full_quality_pipeline(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        rgb = np.zeros((200, 300, 3), dtype=np.uint8)
        mock_demosaic.return_value = rgb
        out = str(tmp_path / "out.jpg")

        result = convert_iiq("fake.iiq", out, quality=Quality.FULL)

        mock_demosaic.assert_called_once_with("fake.iiq")
        assert result.width == 300
        assert result.height == 200
        assert result.output_path == out
        assert result.metadata == {"Make": "Phase One"}
        assert os.path.exists(out)

    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_path(self, mock_thumb, mock_meta, mock_copy, tmp_path):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        out = str(tmp_path / "thumb.jpg")

        result = convert_iiq("fake.iiq", out, quality=Quality.THUMBNAIL)
        mock_thumb.assert_called_once_with("fake.iiq")
        assert result.width == 640
        assert result.height == 480

    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_default_output_path(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        iiq = str(tmp_path / "image.IIQ")
        Path(iiq).touch()

        result = convert_iiq(iiq)
        assert result.output_path == str(tmp_path / "image.jpg")

    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_png_output_format(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.png")

        result = convert_iiq("fake.iiq", out, output_format=OutputFormat.PNG)
        assert result.output_path == out
        assert os.path.exists(out)

    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_max_dimension_resize(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        mock_demosaic.return_value = np.zeros((1000, 2000, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        result = convert_iiq("fake.iiq", out, max_dimension=500)
        assert result.width == 500
        assert result.height == 250

    @patch("iiq2img.converter._copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata")
    @patch("iiq2img.converter._demosaic")
    def test_no_metadata_extraction(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        result = convert_iiq("fake.iiq", out, extract_meta=False)
        mock_meta.assert_not_called()
        mock_copy.assert_not_called()
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# batch_convert (mocked)
# ---------------------------------------------------------------------------


class TestBatchConvert:
    @patch("iiq2img.converter.convert_iiq")
    def test_converts_all_iiq_files(self, mock_convert, tmp_path):
        for name in ["a.IIQ", "b.IIQ", "c.IIQ"]:
            (tmp_path / name).touch()
        out_dir = str(tmp_path / "output")

        mock_convert.return_value = ConvertResult(
            output_path="out.jpg",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )

        results = batch_convert(str(tmp_path), out_dir, workers=1)
        assert len(results) == 3
        assert mock_convert.call_count == 3

    @patch("iiq2img.converter.convert_iiq")
    def test_empty_directory(self, mock_convert, tmp_path, capsys):
        results = batch_convert(str(tmp_path), str(tmp_path / "out"))
        assert results == []
        assert "No .IIQ files found" in capsys.readouterr().out

    @patch("iiq2img.converter.convert_iiq")
    def test_passes_options_through(self, mock_convert, tmp_path):
        (tmp_path / "test.IIQ").touch()
        out_dir = str(tmp_path / "output")

        mock_convert.return_value = ConvertResult(
            output_path="out.png",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )

        batch_convert(
            str(tmp_path),
            out_dir,
            output_format=OutputFormat.PNG,
            compress_quality=75,
            max_dimension=500,
            workers=1,
        )

        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["output_format"] == OutputFormat.PNG
        assert call_kwargs["compress_quality"] == 75
        assert call_kwargs["max_dimension"] == 500
