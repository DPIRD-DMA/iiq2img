"""Tests for batch_convert."""

import os
from pathlib import Path
from unittest.mock import patch

from iiq2img import (
    ConvertResult,
    batch_convert,
)


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
            output_format="png",
            compress_quality=75,
            max_dimension=500,
            workers=1,
        )

        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["output_format"] == "png"
        assert call_kwargs["compress_quality"] == 75
        assert call_kwargs["max_dimension"] == 500


class TestBatchConvertEdgeCases:
    @patch("iiq2img.converter.convert_iiq")
    def test_finds_lowercase_iiq(self, mock_convert, tmp_path):
        """Should find .iiq (lowercase) files when no .IIQ files exist."""
        for name in ["a.iiq", "b.iiq"]:
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
        assert len(results) == 2

    @patch("iiq2img.converter.convert_iiq")
    def test_creates_output_dir(self, mock_convert, tmp_path):
        (tmp_path / "test.IIQ").touch()
        out_dir = str(tmp_path / "new_output_dir")

        mock_convert.return_value = ConvertResult(
            output_path="out.jpg",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )

        batch_convert(str(tmp_path), out_dir, workers=1)
        assert os.path.isdir(out_dir)

    @patch("iiq2img.converter.convert_iiq")
    def test_output_paths_use_correct_extension(self, mock_convert, tmp_path):
        (tmp_path / "photo.IIQ").touch()
        out_dir = str(tmp_path / "output")

        mock_convert.return_value = ConvertResult(
            output_path="out.png",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )

        batch_convert(str(tmp_path), out_dir, output_format="png", workers=1)
        call_args = mock_convert.call_args
        output_path = (
            call_args[1]["output_path"]
            if "output_path" in call_args[1]
            else call_args[0][1]
        )
        assert output_path.endswith(".png")

    @patch("iiq2img.converter.convert_iiq")
    def test_sequential_preserves_order(self, mock_convert, tmp_path):
        for name in ["c.IIQ", "a.IIQ", "b.IIQ"]:
            (tmp_path / name).touch()
        out_dir = str(tmp_path / "output")

        call_order = []

        def track_call(*args, **kwargs):
            iiq_path = args[0] if args else kwargs.get("iiq_path", "")
            call_order.append(Path(iiq_path).name)
            return ConvertResult(
                output_path="out.jpg",
                width=100,
                height=100,
                elapsed_ms=10.0,
                file_size_bytes=1024,
            )

        mock_convert.side_effect = track_call
        batch_convert(str(tmp_path), out_dir, workers=1)
        # glob("*.IIQ") is sorted, so expect alphabetical order
        assert call_order == ["a.IIQ", "b.IIQ", "c.IIQ"]

    @patch("iiq2img.converter.convert_iiq")
    def test_non_iiq_files_ignored(self, mock_convert, tmp_path):
        (tmp_path / "photo.IIQ").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "other.jpg").touch()
        out_dir = str(tmp_path / "output")

        mock_convert.return_value = ConvertResult(
            output_path="out.jpg",
            width=100,
            height=100,
            elapsed_ms=10.0,
            file_size_bytes=1024,
        )

        results = batch_convert(str(tmp_path), out_dir, workers=1)
        assert len(results) == 1
