"""Tests for convert_iiq and _convert_one_for_batch."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from iiq2img import (
    ConvertResult,
    Pipeline,
    Quality,
    convert_iiq,
)
from iiq2img.converter import _convert_one_for_batch


class TestConvertIiq:
    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"Make": "Phase One"})
    @patch("iiq2img.converter._demosaic")
    def test_full_quality_pipeline(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        rgb = np.zeros((200, 300, 3), dtype=np.uint8)
        mock_demosaic.return_value = rgb
        out = str(tmp_path / "out.jpg")

        result = convert_iiq(str(fake_iiq), out, quality=Quality.FULL)

        mock_demosaic.assert_called_once_with(fake_iiq, Pipeline.LIBRAW)
        assert result.width == 300
        assert result.height == 200
        assert result.output_path == Path(out)
        assert result.metadata == {"Make": "Phase One"}
        assert os.path.exists(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_path(self, mock_thumb, mock_meta, mock_copy, tmp_path, fake_iiq):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        out = str(tmp_path / "thumb.jpg")

        result = convert_iiq(str(fake_iiq), out, quality=Quality.THUMBNAIL)
        mock_thumb.assert_called_once_with(fake_iiq)
        assert result.width == 640
        assert result.height == 480

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_default_output_path(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        iiq = str(tmp_path / "image.IIQ")
        Path(iiq).touch()

        result = convert_iiq(iiq)
        assert result.output_path == tmp_path / "image.jpg"

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_png_output_format(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.png")

        result = convert_iiq(str(fake_iiq), out, output_format="png")
        assert result.output_path == Path(out)
        assert os.path.exists(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_max_dimension_resize(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((1000, 2000, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        result = convert_iiq(str(fake_iiq), out, max_dimension=500)
        assert result.width == 500
        assert result.height == 250

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata")
    @patch("iiq2img.converter._demosaic")
    def test_no_metadata_extraction(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        result = convert_iiq(str(fake_iiq), out, extract_meta=False)
        mock_meta.assert_not_called()
        mock_copy.assert_not_called()
        assert result.metadata == {}


class TestConvertIiqEdgeCases:
    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_tiff_output_format(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.tif")
        result = convert_iiq(str(fake_iiq), out, output_format="tiff")
        assert result.output_path == Path(out)
        assert os.path.exists(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_max_dim_no_resize_when_smaller(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        """If image is already smaller than max_dimension, no resize."""
        mock_demosaic.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        result = convert_iiq(str(fake_iiq), out, max_dimension=500)
        # max(100, 200) = 200 < 500, so no resize
        assert result.width == 200
        assert result.height == 100

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"Make": "Phase One"})
    @patch("iiq2img.converter._demosaic")
    def test_metadata_copied_when_present(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        convert_iiq(str(fake_iiq), out)
        mock_copy.assert_called_once()

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_empty_metadata_skips_copy(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        """Even with extract_meta=True, empty metadata should skip copy."""
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        convert_iiq(str(fake_iiq), out, extract_meta=True)
        mock_copy.assert_not_called()

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_creates_output_directory(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        nested = tmp_path / "a" / "b" / "c"
        out = str(nested / "out.jpg")
        convert_iiq(str(fake_iiq), out)
        assert os.path.exists(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_result_has_elapsed_ms(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        result = convert_iiq(str(fake_iiq), out)
        assert result.elapsed_ms >= 0

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_result_has_file_size(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        result = convert_iiq(str(fake_iiq), out)
        assert result.file_size_bytes > 0

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_default_output_path_png(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        iiq = str(tmp_path / "image.IIQ")
        Path(iiq).touch()
        result = convert_iiq(iiq, output_format="png")
        assert result.output_path == tmp_path / "image.png"

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_default_output_path_tiff(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        iiq = str(tmp_path / "image.IIQ")
        Path(iiq).touch()
        result = convert_iiq(iiq, output_format="tiff")
        assert result.output_path == tmp_path / "image.tif"

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_compress_quality_passed_through(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        """Different compress quality should produce different file sizes."""
        rgb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mock_demosaic.return_value = rgb

        out_low = str(tmp_path / "low.jpg")
        out_high = str(tmp_path / "high.jpg")
        r_low = convert_iiq(str(fake_iiq), out_low, compress_quality=10)
        r_high = convert_iiq(str(fake_iiq), out_high, compress_quality=100)
        assert r_low.file_size_bytes < r_high.file_size_bytes


class TestConvertOneForBatch:
    @patch("iiq2img.converter.convert_iiq")
    def test_unpacks_args_correctly(self, mock_convert):
        mock_convert.return_value = ConvertResult(
            output_path="/out/test.jpg",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            Quality.FULL,
            "jpg",
            90,
            None,
            Pipeline.LIBRAW,
        )
        result = _convert_one_for_batch(args)
        mock_convert.assert_called_once_with(
            "/in/test.IIQ",
            "/out/test.jpg",
            quality=Quality.FULL,
            output_format="jpg",
            compress_quality=90,
            max_dimension=None,
            pipeline=Pipeline.LIBRAW,
        )
        assert result.output_path == "/out/test.jpg"

    @patch("iiq2img.converter.convert_iiq")
    def test_passes_max_dimension(self, mock_convert):
        mock_convert.return_value = ConvertResult(
            output_path="/out/test.jpg",
            width=500,
            height=250,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            Quality.FULL,
            "jpg",
            75,
            500,
            Pipeline.LIBRAW,
        )
        _convert_one_for_batch(args)
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["max_dimension"] == 500
        assert call_kwargs["compress_quality"] == 75

    @patch("iiq2img.converter.convert_iiq")
    def test_passes_fast_pipeline(self, mock_convert):
        mock_convert.return_value = ConvertResult(
            output_path="/out/test.jpg",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            Quality.FULL,
            "jpg",
            90,
            None,
            Pipeline.FAST,
        )
        _convert_one_for_batch(args)
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["pipeline"] == Pipeline.FAST
