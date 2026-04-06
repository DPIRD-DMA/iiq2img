"""Tests for read_iiq."""

from unittest.mock import patch

import numpy as np
import pytest

from iiq2img import read_iiq


class TestReadIiq:
    @patch("iiq2img.converter._demosaic")
    def test_returns_rgb_array(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((200, 300, 3), dtype=np.uint8)
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq))

        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 300, 3)
        assert result.dtype == np.uint8
        mock_demosaic.assert_called_once_with(fake_iiq, "fast")

    @patch("iiq2img.converter._demosaic")
    def test_fast_pipeline(self, mock_demosaic, fake_iiq):
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        read_iiq(str(fake_iiq), pipeline="fast")

        mock_demosaic.assert_called_once_with(fake_iiq, "fast")

    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail(self, mock_thumb, fake_iiq):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        result = read_iiq(str(fake_iiq), thumbnail=True)

        mock_thumb.assert_called_once_with(fake_iiq)
        assert result.shape == (480, 640, 3)

    @patch("iiq2img.converter._demosaic")
    def test_max_dimension_resize(self, mock_demosaic, fake_iiq):
        mock_demosaic.return_value = np.zeros((1000, 2000, 3), dtype=np.uint8)

        result = read_iiq(str(fake_iiq), max_dimension=500)

        assert result.shape[1] == 500  # width (longest edge)
        assert result.shape[0] == 250  # height

    @patch("iiq2img.converter._demosaic")
    def test_max_dimension_no_resize_when_smaller(self, mock_demosaic, fake_iiq):
        mock_demosaic.return_value = np.zeros((100, 200, 3), dtype=np.uint8)

        result = read_iiq(str(fake_iiq), max_dimension=500)

        assert result.shape == (100, 200, 3)

    @patch("iiq2img.converter._demosaic")
    def test_rotate_180(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]  # red pixel at top-left
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq), rotate=180)

        assert result.shape == (100, 200, 3)
        # red pixel should now be at bottom-right
        np.testing.assert_array_equal(result[99, 199], [255, 0, 0])

    @patch("iiq2img.converter._demosaic")
    def test_rotate_90(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]  # red pixel at top-left
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq), rotate=90)

        # 100x200 rotated 90° CCW becomes 200x100
        assert result.shape == (200, 100, 3)
        # top-left pixel moves to bottom-left after 90° CCW
        np.testing.assert_array_equal(result[199, 0], [255, 0, 0])

    @patch("iiq2img.converter._demosaic")
    def test_rotate_270(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq), rotate=270)

        assert result.shape == (200, 100, 3)
        # top-left pixel moves to top-right after 270° CCW
        np.testing.assert_array_equal(result[0, 99], [255, 0, 0])

    @patch("iiq2img.converter._demosaic")
    def test_rotate_360_same_as_0(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq), rotate=360)

        assert result.shape == (100, 200, 3)
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])

    @patch("iiq2img.converter._demosaic")
    def test_rotate_0_no_change(self, mock_demosaic, fake_iiq):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        mock_demosaic.return_value = rgb

        result = read_iiq(str(fake_iiq), rotate=0)

        assert result.shape == (100, 200, 3)
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])

    @patch("iiq2img.converter._demosaic")
    def test_no_file_written(self, mock_demosaic, fake_iiq, tmp_path):
        """read_iiq should not create any output files."""
        mock_demosaic.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        read_iiq(str(fake_iiq))

        # No new files should appear in the tmp directory
        files = list(tmp_path.iterdir())
        assert files == [fake_iiq]


class TestReadIiqErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_iiq(str(tmp_path / "missing.iiq"))

    def test_not_iiq_extension(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.touch()
        with pytest.raises(ValueError, match="Expected a .IIQ file"):
            read_iiq(str(f))

    def test_directory_not_file(self, tmp_path):
        d = tmp_path / "somedir.iiq"
        d.mkdir()
        with pytest.raises(ValueError, match="Expected a file"):
            read_iiq(str(d))

    def test_string_pipeline(self, fake_iiq):
        with patch("iiq2img.converter._demosaic") as mock:
            mock.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            read_iiq(str(fake_iiq), pipeline="fast")
            mock.assert_called_once_with(fake_iiq, "fast")

    def test_invalid_rotation(self, fake_iiq):
        with pytest.raises(ValueError, match="Invalid rotation: 45"):
            read_iiq(str(fake_iiq), rotate=45)

    def test_negative_rotation(self, fake_iiq):
        with pytest.raises(ValueError, match="Invalid rotation: -90"):
            read_iiq(str(fake_iiq), rotate=-90)

    def test_invalid_pipeline(self, fake_iiq):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            read_iiq(str(fake_iiq), pipeline="turbo")


class TestReadIiqCombined:
    """Test combined parameter interactions."""

    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_with_rotation(self, mock_thumb, fake_iiq):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        mock_thumb.return_value = rgb

        result = read_iiq(str(fake_iiq), thumbnail=True, rotate=90)

        assert result.shape == (640, 480, 3)
        np.testing.assert_array_equal(result[639, 0], [255, 0, 0])

    @patch("iiq2img.converter._extract_thumbnail")
    def test_thumbnail_with_max_dimension(self, mock_thumb, fake_iiq):
        mock_thumb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        result = read_iiq(str(fake_iiq), thumbnail=True, max_dimension=320)

        assert max(result.shape[:2]) == 320

    @patch("iiq2img.converter._demosaic")
    def test_rotate_with_max_dimension(self, mock_demosaic, fake_iiq):
        mock_demosaic.return_value = np.zeros((1000, 2000, 3), dtype=np.uint8)

        result = read_iiq(str(fake_iiq), max_dimension=500, rotate=90)

        # Resize happens before rotation: 1000x2000 → 250x500, then rotate → 500x250
        assert result.shape == (500, 250, 3)
