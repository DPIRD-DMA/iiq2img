"""Tests for the CLI entry point."""

import sys
from unittest.mock import patch

from iiq2img import ConvertResult, Pipeline
from iiq2img.converter import _cli_main


class TestCliMain:
    def test_usage_printed_with_no_args(self, capsys):
        with patch.object(sys, "argv", ["iiq2img"]):
            with patch("iiq2img.converter.run_benchmark"):
                _cli_main()
        output = capsys.readouterr().out
        assert "Usage:" in output

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_subcommand(self, mock_bench):
        with patch.object(sys, "argv", ["iiq2img", "benchmark", "test.IIQ"]):
            _cli_main()
        mock_bench.assert_called_once_with("test.IIQ")

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_default_path(self, mock_bench):
        with patch.object(sys, "argv", ["iiq2img", "benchmark"]):
            _cli_main()
        mock_bench.assert_called_once_with("PhaseOneSample/P0286625.IIQ")

    @patch("iiq2img.converter.batch_convert")
    def test_batch_subcommand(self, mock_batch):
        with patch.object(
            sys, "argv", ["iiq2img", "batch", "/in", "/out", "png", "75", "4"]
        ):
            _cli_main()
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="png",
            compress_quality=75,
            workers=4,
            pipeline=Pipeline.LIBRAW,
        )

    @patch("iiq2img.converter.batch_convert")
    def test_batch_defaults(self, mock_batch):
        with patch.object(sys, "argv", ["iiq2img", "batch"]):
            _cli_main()
        mock_batch.assert_called_once_with(
            "PhaseOneSample",
            "/tmp/iiq_output",
            output_format="jpg",
            compress_quality=90,
            workers=None,
            pipeline=Pipeline.LIBRAW,
        )

    @patch("iiq2img.converter.batch_convert")
    def test_batch_fast_flag(self, mock_batch):
        with patch.object(
            sys, "argv", ["iiq2img", "batch", "/in", "/out", "jpg", "90", "4", "--fast"]
        ):
            _cli_main()
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="jpg",
            compress_quality=90,
            workers=4,
            pipeline=Pipeline.FAST,
        )

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_conversion(self, mock_convert, capsys):
        mock_convert.return_value = ConvertResult(
            output_path="/tmp/out.jpg",
            width=1000,
            height=500,
            elapsed_ms=3400.0,
            file_size_bytes=2_000_000,
        )
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ"]):
            _cli_main()
        mock_convert.assert_called_once_with("photo.IIQ", pipeline=Pipeline.LIBRAW)
        output = capsys.readouterr().out
        assert "/tmp/out.jpg" in output
        assert "1000x500" in output

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_fast_flag(self, mock_convert, capsys):
        mock_convert.return_value = ConvertResult(
            output_path="/tmp/out.jpg",
            width=1000,
            height=500,
            elapsed_ms=600.0,
            file_size_bytes=2_000_000,
        )
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ", "--fast"]):
            _cli_main()
        mock_convert.assert_called_once_with("photo.IIQ", pipeline=Pipeline.FAST)
        output = capsys.readouterr().out
        assert "Pipeline: fast" in output
