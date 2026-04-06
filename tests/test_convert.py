"""Tests for convert_iiq and _convert_one_for_batch."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iiq2img import convert_iiq
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

        result = convert_iiq(str(fake_iiq), out)

        mock_demosaic.assert_called_once_with(fake_iiq, "fast", bgr=True)
        assert result == Path(out)
        assert os.path.exists(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_path(self, mock_thumb, mock_meta, mock_copy, tmp_path, fake_iiq):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        out = str(tmp_path / "thumb.jpg")

        result = convert_iiq(str(fake_iiq), out, thumbnail=True)
        mock_thumb.assert_called_once_with(fake_iiq)
        assert result == Path(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_default_output_path(self, mock_demosaic, mock_meta, mock_copy, tmp_path):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        iiq = str(tmp_path / "image.IIQ")
        Path(iiq).touch()

        result = convert_iiq(iiq)
        assert result == tmp_path / "image.jpg"

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_png_output_format(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.png")

        result = convert_iiq(str(fake_iiq), out, output_format="png")
        assert result == Path(out)
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
        assert result == Path(out)

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata")
    @patch("iiq2img.converter._demosaic")
    def test_no_metadata_extraction(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        convert_iiq(str(fake_iiq), out, extract_meta=False)
        mock_meta.assert_not_called()
        mock_copy.assert_not_called()


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
        assert result == Path(out)
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
        assert result == Path(out)

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
    def test_output_file_exists(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        result = convert_iiq(str(fake_iiq), out)
        assert result.stat().st_size > 0

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
        assert result == tmp_path / "image.png"

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
        assert result == tmp_path / "image.tif"

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
        assert r_low.stat().st_size < r_high.stat().st_size


class TestConvertOneForBatch:
    @patch("iiq2img.converter.convert_iiq")
    def test_unpacks_args_correctly(self, mock_convert):
        mock_convert.return_value = Path("/out/test.jpg")
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            False,
            "jpg",
            90,
            None,
            "libraw",
        )
        result = _convert_one_for_batch(args)
        mock_convert.assert_called_once_with(
            "/in/test.IIQ",
            "/out/test.jpg",
            thumbnail=False,
            output_format="jpg",
            compress_quality=90,
            max_dimension=None,
            pipeline="libraw",
        )
        assert result == Path("/out/test.jpg")

    @patch("iiq2img.converter.convert_iiq")
    def test_passes_max_dimension(self, mock_convert):
        mock_convert.return_value = Path("/out/test.jpg")
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            False,
            "jpg",
            75,
            500,
            "libraw",
        )
        _convert_one_for_batch(args)
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["max_dimension"] == 500
        assert call_kwargs["compress_quality"] == 75

    @patch("iiq2img.converter.convert_iiq")
    def test_passes_fast_pipeline(self, mock_convert):
        mock_convert.return_value = Path("/out/test.jpg")
        args = (
            "/in/test.IIQ",
            "/out/test.jpg",
            False,
            "jpg",
            90,
            None,
            "fast",
        )
        _convert_one_for_batch(args)
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["pipeline"] == "fast"


class TestConvertIiqGeoref:
    @patch("iiq2img.converter._apply_georef")
    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"xmp": "<gps/>"})
    @patch("iiq2img.converter._demosaic")
    def test_georef_jpeg_calls_apply_georef(
        self, mock_demosaic, mock_meta, mock_copy, mock_georef, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        convert_iiq(str(fake_iiq), out, georef=True)
        mock_georef.assert_called_once()

    @patch("iiq2img.converter._write_geotiff")
    @patch("iiq2img.converter.extract_metadata", return_value={"xmp": "<gps/>"})
    @patch("iiq2img.converter._demosaic")
    def test_georef_tiff_calls_write_geotiff(
        self, mock_demosaic, mock_meta, mock_geotiff, tmp_path, fake_iiq
    ):
        mock_demosaic.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        out = str(tmp_path / "out.tif")
        convert_iiq(str(fake_iiq), out, output_format="tiff", georef=True)
        mock_geotiff.assert_called_once()

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"xmp": "<gps/>"})
    @patch("iiq2img.converter._demosaic")
    def test_georef_forces_metadata_extraction(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        """georef=True should extract metadata even if extract_meta=False."""
        mock_demosaic.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        convert_iiq(str(fake_iiq), out, georef=True, extract_meta=False)
        mock_meta.assert_called_once()


class TestConvertIiqRotate:
    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_rotate_180_in_convert(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        mock_demosaic.return_value = rgb
        out = str(tmp_path / "out.png")
        convert_iiq(str(fake_iiq), out, output_format="png", rotate=180)
        assert Path(out).exists()

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_rotate_90_in_convert(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_demosaic.return_value = rgb
        out = str(tmp_path / "out.png")
        convert_iiq(str(fake_iiq), out, output_format="png", rotate=90)
        assert Path(out).exists()

    def test_invalid_rotation_in_convert(self, fake_iiq, tmp_path):
        with pytest.raises(ValueError, match="Invalid rotation: 45"):
            convert_iiq(str(fake_iiq), str(tmp_path / "out.jpg"), rotate=45)


class TestConvertIiqCombined:
    """Test combined parameter interactions."""

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_with_rotate_and_max_dim(
        self, mock_thumb, mock_meta, mock_copy, tmp_path, fake_iiq
    ):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")
        result = convert_iiq(
            str(fake_iiq), out, thumbnail=True, rotate=90, max_dimension=320
        )
        assert result == Path(out)
        assert Path(out).exists()


class TestRunBenchmark:
    @patch("iiq2img.converter.convert_iiq")
    def test_runs_all_approaches(self, mock_convert, tmp_path, capsys):
        from iiq2img.converter import run_benchmark

        iiq = tmp_path / "test.IIQ"
        iiq.write_bytes(b"\x00" * 1024)

        mock_convert.return_value = Path("/tmp/bench.jpg")

        # Mock the output file stat for each approach
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1_000_000
            run_benchmark(str(iiq))

        output = capsys.readouterr().out
        assert "Benchmarking" in output
        # Should run all 7 approaches
        assert mock_convert.call_count == 7

    @patch("iiq2img.converter.convert_iiq")
    def test_benchmark_output_format(self, mock_convert, tmp_path, capsys):
        from iiq2img.converter import run_benchmark

        iiq = tmp_path / "test.IIQ"
        iiq.write_bytes(b"\x00" * 1024)

        mock_convert.return_value = Path("/tmp/bench.jpg")
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 2_000_000
            run_benchmark(str(iiq))

        output = capsys.readouterr().out
        assert "Thumbnail" in output
        assert "LibRaw" in output
        assert "Fast" in output
        assert "ms" in output
        assert "MB" in output


class TestBatchConvertMultiprocessing:
    @patch("iiq2img.converter.ProcessPoolExecutor")
    def test_workers_greater_than_one_uses_pool(self, mock_pool_cls, tmp_path):
        """Verify that workers > 1 creates a ProcessPoolExecutor."""
        for name in ["a.IIQ", "b.IIQ"]:
            (tmp_path / name).touch()
        out_dir = str(tmp_path / "output")

        # Set up the mock executor context manager
        mock_executor = mock_pool_cls.return_value.__enter__.return_value
        future_mock = type("FakeFuture", (), {"result": lambda self: Path("out.jpg")})()
        mock_executor.submit.return_value = future_mock

        from iiq2img import batch_convert

        with patch(
            "iiq2img.converter.as_completed", return_value=[future_mock, future_mock]
        ):
            results = batch_convert(str(tmp_path), out_dir, workers=2)

        assert len(results) == 2
        mock_pool_cls.assert_called_once()
        assert mock_executor.submit.call_count == 2


class TestConvertIiqVerbose:
    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={"Make": "Phase One"})
    @patch("iiq2img.converter._demosaic")
    def test_verbose_prints_timing(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq, capsys
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        convert_iiq(str(fake_iiq), out, verbose=True)

        output = capsys.readouterr().out
        assert "Demosaic (fast):" in output
        assert "Metadata:" in output
        assert "Encode (jpg):" in output
        assert "Total:" in output
        assert "ms" in output

    @patch("iiq2img.converter.copy_metadata_to_output")
    @patch("iiq2img.converter.extract_metadata", return_value={})
    @patch("iiq2img.converter._demosaic")
    def test_verbose_false_prints_nothing(
        self, mock_demosaic, mock_meta, mock_copy, tmp_path, fake_iiq, capsys
    ):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        out = str(tmp_path / "out.jpg")

        convert_iiq(str(fake_iiq), out, verbose=False)

        output = capsys.readouterr().out
        assert output == ""


class TestConvertIiqValidation:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_iiq(str(tmp_path / "missing.iiq"))

    def test_not_iiq_extension(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.touch()
        with pytest.raises(ValueError, match="Expected a .IIQ file"):
            convert_iiq(str(f))

    def test_directory_not_file(self, tmp_path):
        d = tmp_path / "somedir.iiq"
        d.mkdir()
        with pytest.raises(ValueError, match="Expected a file"):
            convert_iiq(str(d))
