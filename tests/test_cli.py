"""Tests for the CLI entry point and exit codes."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from iiq2img.converter import (
    EXIT_BAD_ARGS,
    EXIT_ERROR,
    EXIT_NO_FILES,
    EXIT_OK,
    _cli_main,
)


class TestCliMain:
    def test_help_printed_with_no_args(self, capsys):
        with patch.object(sys, "argv", ["iiq2img"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_BAD_ARGS
        output = capsys.readouterr().out
        assert "benchmark" in output
        assert "batch" in output

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_subcommand(self, mock_bench):
        with patch.object(sys, "argv", ["iiq2img", "benchmark", "test.IIQ"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
        mock_bench.assert_called_once_with("test.IIQ")

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_default_path(self, mock_bench):
        with patch.object(sys, "argv", ["iiq2img", "benchmark"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
        mock_bench.assert_called_once_with("PhaseOneSample/P0286625.IIQ")

    @patch("iiq2img.converter.batch_convert")
    def test_batch_subcommand(self, mock_batch):
        mock_batch.return_value = [Path("/out/img.jpg")]
        with patch.object(
            sys,
            "argv",
            [
                "iiq2img",
                "batch",
                "/in",
                "/out",
                "--format",
                "png",
                "--quality",
                "75",
                "--workers",
                "4",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
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
        mock_batch.return_value = [Path("/out/img.jpg")]
        with patch.object(sys, "argv", ["iiq2img", "batch"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
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
        mock_batch.return_value = [Path("/out/img.jpg")]
        with patch.object(
            sys,
            "argv",
            [
                "iiq2img",
                "batch",
                "/in",
                "/out",
                "--format",
                "jpg",
                "--quality",
                "90",
                "--workers",
                "4",
                "--libraw",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
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
                with pytest.raises(SystemExit) as exc_info:
                    _cli_main()
                assert exc_info.value.code == EXIT_OK
        mock_convert.assert_called_once_with("photo.IIQ", pipeline="fast")
        output = capsys.readouterr().out
        assert "/tmp/out.jpg" in output

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_libraw_flag(self, mock_convert, capsys):
        mock_convert.return_value = Path("/tmp/out.jpg")
        with patch.object(sys, "argv", ["iiq2img", "photo.IIQ", "--libraw"]):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 2_000_000
                with pytest.raises(SystemExit) as exc_info:
                    _cli_main()
                assert exc_info.value.code == EXIT_OK
        mock_convert.assert_called_once_with("photo.IIQ", pipeline="libraw")
        output = capsys.readouterr().out
        assert "Pipeline: libraw" in output

    def test_libraw_flag_only_shows_help(self, capsys):
        """CLI with only --libraw and no file path should print help."""
        with patch.object(sys, "argv", ["iiq2img", "--libraw"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_BAD_ARGS
        output = capsys.readouterr().out
        assert "benchmark" in output
        assert "batch" in output

    def test_help_shows_all_subcommands(self, capsys):
        """Help output should document benchmark, batch, and --libraw."""
        with patch.object(sys, "argv", ["iiq2img"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_BAD_ARGS
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
                with pytest.raises(SystemExit) as exc_info:
                    _cli_main()
                assert exc_info.value.code == EXIT_OK
        output = capsys.readouterr().out
        assert "Time:" in output
        assert "Size:" in output
        assert "Pipeline: fast" in output

    @patch("iiq2img.converter.batch_convert")
    def test_batch_partial_args(self, mock_batch):
        """Batch with only input/output dirs should use defaults for the rest."""
        mock_batch.return_value = [Path("/out/img.jpg")]
        with patch.object(sys, "argv", ["iiq2img", "batch", "/in", "/out"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK
        mock_batch.assert_called_once_with(
            "/in",
            "/out",
            output_format="jpg",
            compress_quality=90,
            workers=None,
            pipeline="fast",
        )


class TestCliExitCodes:
    """Tests for structured CLI exit codes."""

    def test_exit_code_constants(self):
        """Exit code constants have expected values."""
        assert EXIT_OK == 0
        assert EXIT_ERROR == 1
        assert EXIT_BAD_ARGS == 2
        assert EXIT_NO_FILES == 3

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_not_found_exits_error(self, mock_convert, capsys):
        """Missing IIQ file should exit with EXIT_ERROR."""
        mock_convert.side_effect = FileNotFoundError("IIQ file not found: missing.IIQ")
        with patch.object(sys, "argv", ["iiq2img", "missing.IIQ"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "missing.IIQ" in err

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_invalid_format_exits_error(self, mock_convert, capsys):
        """ValueError during conversion should exit with EXIT_ERROR."""
        mock_convert.side_effect = ValueError("Expected a .IIQ file")
        with patch.object(sys, "argv", ["iiq2img", "photo.png"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Expected a .IIQ file" in err

    @patch("iiq2img.converter.convert_iiq")
    def test_single_file_os_error_exits_error(self, mock_convert, capsys):
        """OSError during conversion should exit with EXIT_ERROR."""
        mock_convert.side_effect = OSError("corrupt or unreadable")
        with patch.object(sys, "argv", ["iiq2img", "bad.IIQ"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "corrupt or unreadable" in err

    @patch("iiq2img.converter.batch_convert")
    def test_batch_no_files_exits_no_files(self, mock_batch):
        """Batch with no IIQ files found should exit with EXIT_NO_FILES."""
        mock_batch.return_value = []
        with patch.object(sys, "argv", ["iiq2img", "batch", "/empty", "/out"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_NO_FILES

    @patch("iiq2img.converter.batch_convert")
    def test_batch_error_exits_error(self, mock_batch, capsys):
        """Runtime error in batch should exit with EXIT_ERROR."""
        mock_batch.side_effect = OSError("Permission denied")
        with patch.object(sys, "argv", ["iiq2img", "batch", "/in", "/out"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Permission denied" in err

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_error_exits_error(self, mock_bench, capsys):
        """Runtime error in benchmark should exit with EXIT_ERROR."""
        mock_bench.side_effect = FileNotFoundError("IIQ file not found")
        with patch.object(sys, "argv", ["iiq2img", "benchmark", "missing.IIQ"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "IIQ file not found" in err

    @patch("iiq2img.converter.run_benchmark")
    def test_benchmark_success_exits_ok(self, mock_bench):
        """Successful benchmark should exit with EXIT_OK."""
        with patch.object(sys, "argv", ["iiq2img", "benchmark", "test.IIQ"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK

    @patch("iiq2img.converter.batch_convert")
    def test_batch_success_exits_ok(self, mock_batch):
        """Successful batch should exit with EXIT_OK."""
        mock_batch.return_value = [Path("/out/img.jpg")]
        with patch.object(sys, "argv", ["iiq2img", "batch"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_OK

    def test_no_args_exits_bad_args(self):
        """No arguments should exit with EXIT_BAD_ARGS."""
        with patch.object(sys, "argv", ["iiq2img"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_BAD_ARGS

    def test_flags_only_exits_bad_args(self):
        """Only flags, no file should exit with EXIT_BAD_ARGS."""
        with patch.object(sys, "argv", ["iiq2img", "--libraw"]):
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == EXIT_BAD_ARGS
