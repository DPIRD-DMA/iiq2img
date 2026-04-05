"""Test that FAST and LIBRAW pipelines produce visually equivalent output."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from iiq2img import convert_iiq

SAMPLE_DIR = Path("PhaseOneSample")
SAMPLE_IIQ = next(SAMPLE_DIR.glob("*.IIQ"), None) if SAMPLE_DIR.exists() else None


@pytest.mark.skipif(SAMPLE_IIQ is None, reason="No sample IIQ file in PhaseOneSample/")
def test_fast_pipeline_matches_libraw(tmp_path: Path) -> None:
    """FAST pipeline output should be within ~7/255 mean abs diff of LIBRAW."""
    out_libraw = tmp_path / "libraw.jpg"
    out_fast = tmp_path / "fast.jpg"

    convert_iiq(
        SAMPLE_IIQ,
        out_libraw,
        pipeline="libraw",
        extract_meta=False,
    )
    convert_iiq(
        SAMPLE_IIQ,
        out_fast,
        pipeline="fast",
        extract_meta=False,
    )

    # Compare the raw RGB arrays by re-reading the JPEGs
    img_libraw = cv2.imread(str(out_libraw))
    img_fast = cv2.imread(str(out_fast))
    assert img_libraw.shape == img_fast.shape

    diff = np.abs(img_libraw.astype(np.int16) - img_fast.astype(np.int16))
    mean_diff = diff.mean()

    # Mean abs diff should be under 10/255 (currently ~6.5 before JPEG re-encoding)
    assert mean_diff < 10, f"Mean pixel diff too large: {mean_diff:.2f}/255"


@pytest.mark.skipif(SAMPLE_IIQ is None, reason="No sample IIQ file in PhaseOneSample/")
def test_fast_pipeline_faster_than_libraw(tmp_path: Path) -> None:
    """FAST pipeline should be at least 2x faster than LIBRAW."""
    out_libraw = tmp_path / "libraw.jpg"
    out_fast = tmp_path / "fast.jpg"

    t0 = time.perf_counter()
    convert_iiq(SAMPLE_IIQ, out_libraw, pipeline="libraw", extract_meta=False)
    libraw_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    convert_iiq(SAMPLE_IIQ, out_fast, pipeline="fast", extract_meta=False)
    fast_time = time.perf_counter() - t0

    speedup = libraw_time / fast_time
    assert speedup >= 2.0, (
        f"Fast pipeline not fast enough: {fast_time:.2f}s vs libraw {libraw_time:.2f}s "
        f"(speedup {speedup:.1f}x, expected >= 2x)"
    )
