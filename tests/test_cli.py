"""Tests for the CLI entry point."""

import sys
from pathlib import Path
from unittest.mock import patch


from iiq2img.converter import _cli_main


class TestCliMain:
    def test_help_printed_with_no_args(self, capsys):
        with patch.object(sys, "argv", ["iiq2img"]):
            _cli_main()
        output = capsys.readouterr().out
        assert "benchmark" in output
        assert "batch" in output

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
            sys,
            "argv",
            ["iiq2img", "batch", "/in", "/out", "--format", "png", "--quality", "75", "--workers", "4"],
        ):
            _cli_main()
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="png",
            compress_quality=75,
            workers=4,
            pipeline="fast",
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
            pipeline="fast",
        )

    @patch("iiq2img.converter.batch_convert")
    def test_batch_libraw_flag(self, mock_batch):
        with patch.object(
            sys,
            "argv",
            ["iiq2img", "batch", "/in", "/out", "--format", "jpg", "--quality", "90", "--workers", "4", "--libraw"],
        ):
            _cli_main()
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="jpg",
            compress_quality=90,
            workers=4,
            pipeline="libraw",
        )

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_conversion(self, mock_convert, capsys):
        out_path = Path("/tmp/out.jpg")
        mock_convert.return_value = out_path
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ"]):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 2_000_000
                _cli_main()
        mock_convert.assert_called_once_with("photo.IIQ", pipeline="fast")
        output = capsys.readouterr().out
        assert "/tmp/out.jpg" in output

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_libraw_flag(self, mock_convert, capsys):
        mock_convert.return_value = Path("/tmp/out.jpg")
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ", "--libraw"]):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 2_000_000
                _cli_main()
        mock_convert.assert_called_once_with("photo.IIQ", pipeline="libraw")
        output = capsys.readouterr().out
        assert "Pipeline: libraw" in output

    def test_libraw_flag_only_shows_help(self, capsys):
        """CLI with only --libraw and no file path should print help."""
        with patch.object(sys, "argv", ["iiq2img", "--libraw"]):
            _cli_main()
        output = capsys.readouterr().out
        assert "benchmark" in output
        assert "batch" in output

    def test_help_shows_all_subcommands(self, capsys):
        """Help output should document benchmark, batch, and --libraw."""
        with patch.object(sys, "argv", ["iiq2img"]):
            _cli_main()
        output = capsys.readouterr().out
        assert "benchmark" in output
        assert "batch" in output
        assert "--libraw" in output

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_shows_timing(self, mock_convert, capsys):
        """Single file conversion should print timing info."""
        mock_convert.return_value = Path("/tmp/out.jpg")
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ"]):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 5_000_000
                _cli_main()
        output = capsys.readouterr().out
        assert "Time:" in output
        assert "Size:" in output
        assert "Pipeline: fast" in output

    @patch("iiq2img.converter.batch_convert")
    def test_batch_partial_args(self, mock_batch):
        """Batch with only input/output dirs should use defaults for the rest."""
        with patch.object(sys, "argv", ["iiq2img", "batch", "/in", "/out"]):
            _cli_main()
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="jpg",
            compress_quality=90,
            workers=None,
            pipeline="fast",
        )
