"""
Fast IIQ converter for Phase One iXM-GS120 (120MP) raw images.

Two demosaic pipelines:
  - LIBRAW: LibRaw PPG demosaic — accurate reference, ~2.9s per image.
  - FAST:   cv2 edge-aware demosaic + BT.709 gamma + numba parallel LUTs.
            ~3x faster end-to-end (~550ms), mean pixel diff ~6.5/255 vs LibRaw.

Supports output formats: JPEG, PNG, TIFF.
Retains all EXIF/GPS/XMP metadata from the original IIQ file.
See FINDINGS.md for details on how the fast pipeline was matched to LibRaw.
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rawpy

from iiq2img.encode import (
    FORMAT_EXTENSIONS,
    encode_image,
    resolve_format,
    resize_max_dim,
)
from iiq2img.metadata import copy_metadata_to_output, extract_metadata
from iiq2img.pipeline import demosaic_fast


class Quality(Enum):
    THUMBNAIL = "thumbnail"  # 640x480 embedded bitmap, ~1ms
    FULL = "full"  # full-res (12768x9564), ~2.7s


class Pipeline(Enum):
    LIBRAW = "libraw"  # LibRaw PPG demosaic — accurate, ~2.8s, default
    FAST = "fast"  # cv2 EA + numba LUT — ~3x faster end-to-end, visually equivalent


def _resolve_pipeline(value: "str | Pipeline") -> Pipeline:
    if isinstance(value, Pipeline):
        return value
    try:
        return Pipeline(value.lower())
    except ValueError:
        raise ValueError(
            f"Unknown pipeline: {value!r}. Expected one of: 'libraw', 'fast'"
        ) from None


@dataclass
class ConvertResult:
    output_path: Path
    width: int
    height: int
    elapsed_ms: float
    file_size_bytes: int
    metadata: dict[str, str] = field(default_factory=dict)


def convert_iiq(
    iiq_path: str | Path,
    output_path: str | Path | None = None,
    quality: Quality = Quality.FULL,
    output_format: str | None = None,
    compress_quality: int = 90,
    max_dimension: int | None = None,
    extract_meta: bool = True,
    georef: bool = False,
    rotate_180: bool = False,
    pipeline: "str | Pipeline" = Pipeline.LIBRAW,
) -> ConvertResult:
    """
    Convert a Phase One IIQ raw file to JPEG, PNG, or TIFF.

    Args:
        iiq_path: Path to .IIQ file
        output_path: Output path. If None, uses same name with new extension.
        quality: Demosaic quality preset (THUMBNAIL or FULL)
        output_format: Output format string ('jpg', 'png', 'tif'),
                       or None to infer from output_path extension (default: 'jpg')
        compress_quality: Compression quality 1-100 (JPEG quality / PNG compression)
        max_dimension: If set, resize longest edge to this value
        extract_meta: Whether to extract and retain EXIF metadata
        georef: If True, georeference the output (world file for JPEG/PNG, GeoTIFF for TIFF)
        rotate_180: If True, rotate the image 180 degrees (for inverted-mount sensors)
        pipeline: Demosaic pipeline to use (Pipeline.LIBRAW or Pipeline.FAST).
                  LIBRAW uses LibRaw PPG — accurate, ~2.8s.
                  FAST uses cv2 edge-aware demosaic + numba LUTs — ~3x faster end-to-end,
                  visually equivalent (mean pixel diff ~6.5/255 vs LibRaw).

    Returns:
        ConvertResult with output details
    """
    t0 = time.perf_counter()
    iiq_path = Path(iiq_path)
    pipeline = _resolve_pipeline(pipeline)

    if not iiq_path.exists():
        raise FileNotFoundError(f"IIQ file not found: {iiq_path}")
    if not iiq_path.is_file():
        raise ValueError(f"Expected a file, got a directory: {iiq_path}")
    if iiq_path.suffix.lower() != ".iiq":
        raise ValueError(f"Expected a .IIQ file, got '{iiq_path.suffix}': {iiq_path}")

    out_path = Path(output_path) if output_path is not None else None
    output_fmt = resolve_format(output_format, out_path)

    if out_path is None:
        ext = FORMAT_EXTENSIONS[output_fmt]
        out_path = iiq_path.with_suffix(ext)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always extract metadata if georef is requested (need XMP for GPS)
    need_meta = extract_meta or georef
    metadata = extract_metadata(iiq_path) if need_meta else {}

    if quality == Quality.THUMBNAIL:
        rgb = _extract_thumbnail(iiq_path)
    else:
        rgb = _demosaic(iiq_path, pipeline)

    if max_dimension and max(rgb.shape[:2]) > max_dimension:
        rgb = resize_max_dim(rgb, max_dimension)

    if rotate_180:
        rgb = np.rot90(rgb, 2)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    is_geotiff = georef and output_fmt == "tiff"

    if is_geotiff:
        # Write GeoTIFF directly (skip EXIF injection — rasterio handles metadata)
        _write_geotiff(rgb, out_path, metadata, compress_quality)
    else:
        encode_image(bgr, out_path, output_fmt, compress_quality)
        if extract_meta and metadata:
            copy_metadata_to_output(iiq_path, out_path, metadata)

    if georef and not is_geotiff:
        _apply_georef(out_path, metadata, w, h)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    file_size = out_path.stat().st_size

    return ConvertResult(
        output_path=out_path,
        width=w,
        height=h,
        elapsed_ms=elapsed_ms,
        file_size_bytes=file_size,
        metadata=metadata,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _apply_georef(output_path: Path, metadata: dict[str, str], w: int, h: int) -> None:
    """Apply georeferencing to an output image using its metadata."""
    from iiq2img.georef import extract_geo_info, write_world_file

    xmp = metadata.get("xmp", "")
    if not xmp:
        return
    geo = extract_geo_info(xmp, 80.0, w, h)
    if geo is None:
        return

    if output_path.suffix.lower() not in (".tif", ".tiff"):
        write_world_file(output_path, geo)


def _write_geotiff(
    rgb: np.ndarray,
    output_path: Path,
    metadata: dict[str, str],
    compress_quality: int,
) -> None:
    """Write image as GeoTIFF with embedded georeferencing."""
    from iiq2img.georef import compute_transform, extract_geo_info

    h, w = rgb.shape[:2]
    xmp = metadata.get("xmp", "")

    geo = extract_geo_info(xmp, 80.0, w, h)
    if geo is None:
        # Fall back to plain TIFF if no GPS data
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)
        return

    import rasterio
    from rasterio.crs import CRS

    transform = compute_transform(geo)
    crs = CRS.from_epsg(4326)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": w,
        "height": h,
        "count": 3,
        "crs": crs,
        "transform": transform,
        "compress": "JPEG",
        "jpeg_quality": compress_quality,
        "photometric": "YCBCR",
    }

    with rasterio.open(str(output_path), "w", **profile) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)


def _extract_thumbnail(iiq_path: Path) -> np.ndarray:
    """Extract embedded thumbnail bitmap (~640x480, ~1ms)."""
    raw = rawpy.imread(str(iiq_path))
    thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.BITMAP:  # type: ignore[attr-defined]
        rgb = thumb.data
    elif thumb.format == rawpy.ThumbFormat.JPEG:  # type: ignore[attr-defined]
        arr = np.frombuffer(thumb.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode JPEG thumbnail")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown thumbnail format: {thumb.format}")
    raw.close()
    return rgb


def _demosaic(iiq_path: Path, pipeline: Pipeline = Pipeline.LIBRAW) -> np.ndarray:
    """Demosaic IIQ raw data to full-resolution RGB."""
    if pipeline == Pipeline.FAST:
        return demosaic_fast(iiq_path)
    try:
        raw = rawpy.imread(str(iiq_path))
    except rawpy.LibRawIOError:  # type: ignore[attr-defined]
        raise OSError(
            f"Failed to read IIQ file (corrupt or unreadable): {iiq_path}"
        ) from None
    rgb = raw.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.PPG,  # type: ignore[attr-defined]
        half_size=False,
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,  # type: ignore[attr-defined]
        output_bps=8,
    )
    raw.close()
    return rgb


# ── Batch conversion ─────────────────────────────────────────────────────────


def _convert_one_for_batch(args: tuple) -> ConvertResult:
    """Worker function for multiprocessing batch conversion."""
    (
        iiq_path,
        out_path,
        quality,
        output_format,
        compress_quality,
        max_dimension,
        pipeline,
    ) = args
    return convert_iiq(
        iiq_path,
        out_path,
        quality=quality,
        output_format=output_format,
        compress_quality=compress_quality,
        max_dimension=max_dimension,
        pipeline=pipeline,
    )


def batch_convert(
    input_dir: str | Path,
    output_dir: str | Path,
    quality: Quality = Quality.FULL,
    output_format: str = "jpg",
    compress_quality: int = 90,
    max_dimension: int | None = None,
    workers: int | None = None,
    pipeline: "str | Pipeline" = Pipeline.LIBRAW,
) -> list[ConvertResult]:
    """Convert all IIQ files in a directory using multiprocessing.

    Args:
        workers: Number of parallel workers. None = auto (min(cpu_count, 8)).
                 Set to 1 for sequential processing.
        pipeline: Demosaic pipeline — Pipeline.LIBRAW (default) or Pipeline.FAST (~3x faster end-to-end).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pipeline = _resolve_pipeline(pipeline)
    output_fmt = resolve_format(output_format, None)

    iiq_files = sorted(input_dir.glob("*.IIQ"))
    if not iiq_files:
        iiq_files = sorted(input_dir.glob("*.iiq"))
    if not iiq_files:
        print(f"No .IIQ files found in {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    ext = FORMAT_EXTENSIONS[output_fmt]

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)

    tasks = [
        (
            str(f),
            str(output_dir / (f.stem + ext)),
            quality,
            output_fmt,
            compress_quality,
            max_dimension,
            pipeline,
        )
        for f in iiq_files
    ]

    from tqdm.auto import tqdm

    total_t0 = time.perf_counter()
    results: list[ConvertResult] = []
    pbar = tqdm(total=len(tasks), unit="img")

    if workers <= 1:
        for task_args in tasks:
            result = _convert_one_for_batch(task_args)
            results.append(result)
            pbar.set_postfix_str(
                f"{Path(task_args[0]).stem} {result.width}x{result.height} "
                f"{result.elapsed_ms:.0f}ms {result.file_size_bytes / 1024 / 1024:.1f}MB"
            )
            pbar.update(1)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as exe:
            futures = {exe.submit(_convert_one_for_batch, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.set_postfix_str(
                    f"{result.output_path.stem} {result.width}x{result.height} "
                    f"{result.elapsed_ms:.0f}ms "
                    f"{result.file_size_bytes / 1024 / 1024:.1f}MB"
                )
                pbar.update(1)

    pbar.close()
    total_elapsed = (time.perf_counter() - total_t0) * 1000
    avg = total_elapsed / len(results)
    throughput = len(results) / (total_elapsed / 1000)
    print(
        f"Done: {len(results)} images in {total_elapsed / 1000:.1f}s "
        f"(avg {avg:.0f}ms/image, {throughput:.1f} images/sec, {workers} workers)"
    )
    return results


# ── Benchmark & CLI ──────────────────────────────────────────────────────────


def run_benchmark(iiq_path: str | Path) -> None:
    """Run benchmark comparing all conversion approaches on a single file."""
    iiq_path = Path(iiq_path)
    print(f"Benchmarking: {iiq_path}")
    print(f"File size: {iiq_path.stat().st_size / 1024 / 1024:.1f} MB\n")

    approaches = [
        (
            "Thumbnail (640x480) JPG",
            Quality.THUMBNAIL,
            "jpg",
            90,
            None,
            Pipeline.LIBRAW,
        ),
        ("LibRaw PPG  JPG q=90", Quality.FULL, "jpg", 90, None, Pipeline.LIBRAW),
        ("LibRaw PPG  JPG q=75", Quality.FULL, "jpg", 75, None, Pipeline.LIBRAW),
        ("Fast (cv2 EA)  JPG q=90", Quality.FULL, "jpg", 90, None, Pipeline.FAST),
        ("Fast (cv2 EA)  JPG q=75", Quality.FULL, "jpg", 75, None, Pipeline.FAST),
        ("Fast (cv2 EA)  PNG", Quality.FULL, "png", 90, None, Pipeline.FAST),
        ("Fast (cv2 EA)  TIFF", Quality.FULL, "tiff", 90, None, Pipeline.FAST),
    ]

    print(f"{'Approach':<35} {'Time':>8} {'Size':>10} {'Resolution':>14}")
    print("-" * 71)

    for name, qual, fmt, cq, max_dim, pl in approaches:
        out = Path(
            f"/tmp/bench_{qual.value}_{fmt}_{cq}_{pl.value}{FORMAT_EXTENSIONS[fmt]}"
        )
        r = convert_iiq(
            iiq_path,
            out,
            quality=qual,
            output_format=fmt,
            compress_quality=cq,
            max_dimension=max_dim,
            extract_meta=False,
            pipeline=pl,
        )
        print(
            f"{name:<35} {r.elapsed_ms:>7.0f}ms "
            f"{r.file_size_bytes / 1024 / 1024:>8.1f}MB "
            f"{r.width}x{r.height}"
        )


def _cli_main() -> None:
    """CLI entry point for the iiq2img command."""
    sample = "PhaseOneSample/P0286625.IIQ"

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        path = sys.argv[2] if len(sys.argv) > 2 else sample
        run_benchmark(path)

    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "PhaseOneSample"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/iiq_output"
        fmt_str = sys.argv[4] if len(sys.argv) > 4 else "jpg"
        cq = int(sys.argv[5]) if len(sys.argv) > 5 else 90
        workers = int(sys.argv[6]) if len(sys.argv) > 6 else None
        pl = Pipeline.FAST if "--fast" in sys.argv else Pipeline.LIBRAW
        batch_convert(
            input_dir,
            output_dir,
            output_format=fmt_str,
            compress_quality=cq,
            workers=workers,
            pipeline=pl,
        )

    elif len(sys.argv) > 1:
        pl = Pipeline.FAST if "--fast" in sys.argv else Pipeline.LIBRAW
        path_arg = next(a for a in sys.argv[1:] if not a.startswith("--"))
        result = convert_iiq(path_arg, pipeline=pl)
        print(f"Output:   {result.output_path}")
        print(f"Pipeline: {pl.value}")
        print(f"Res:      {result.width}x{result.height}")
        print(f"Time:     {result.elapsed_ms:.0f}ms")
        print(f"Size:     {result.file_size_bytes / 1024 / 1024:.1f}MB")

    else:
        print("Usage:")
        print(f"  {sys.argv[0]} benchmark [iiq_path]")
        print(
            f"  {sys.argv[0]} batch [in_dir] [out_dir] [jpg|png|tiff] [quality] [workers] [--fast]"
        )
        print(f"  {sys.argv[0]} <file.IIQ> [--fast]")
        print()
        print(
            "  --fast   Use fast pipeline (cv2 EA + numba, ~3x faster end-to-end, visually equivalent)"
        )
        print()
        print("Running benchmark on sample file...")
        print()
        run_benchmark(sample)


if __name__ == "__main__":
    _cli_main()
