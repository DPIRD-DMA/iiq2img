"""Test that FAST and LIBRAW pipelines produce visually equivalent output."""

from pathlib import Path

import numpy as np
import pytest

from iiq2img import Pipeline, Quality, convert_iiq

SAMPLE_DIR = Path("PhaseOneSample")
SAMPLE_IIQ = next(SAMPLE_DIR.glob("*.IIQ"), None) if SAMPLE_DIR.exists() else None


@pytest.mark.skipif(SAMPLE_IIQ is None, reason="No sample IIQ file in PhaseOneSample/")
def test_fast_pipeline_matches_libraw(tmp_path: Path) -> None:
    """FAST pipeline output should be within ~7/255 mean abs diff of LIBRAW."""
    out_libraw = tmp_path / "libraw.jpg"
    out_fast = tmp_path / "fast.jpg"

    r_libraw = convert_iiq(
        SAMPLE_IIQ,
        out_libraw,
        quality=Quality.FULL,
        pipeline=Pipeline.LIBRAW,
        extract_meta=False,
    )
    r_fast = convert_iiq(
        SAMPLE_IIQ,
        out_fast,
        quality=Quality.FULL,
        pipeline=Pipeline.FAST,
        extract_meta=False,
    )

    assert r_libraw.width == r_fast.width
    assert r_libraw.height == r_fast.height

    # Compare the raw RGB arrays by re-reading the JPEGs
    import cv2

    img_libraw = cv2.imread(str(out_libraw))
    img_fast = cv2.imread(str(out_fast))
    assert img_libraw.shape == img_fast.shape

    diff = np.abs(img_libraw.astype(np.int16) - img_fast.astype(np.int16))
    mean_diff = diff.mean()

    # Mean abs diff should be under 10/255 (currently ~6.5 before JPEG re-encoding)
    assert mean_diff < 10, f"Mean pixel diff too large: {mean_diff:.2f}/255"
