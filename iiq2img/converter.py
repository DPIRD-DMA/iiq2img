"""
Fast IIQ converter for Phase One iXM-GS120 (120MP) raw images.

Two demosaic pipelines (default: fast):
  - FAST:   cv2 edge-aware demosaic + BT.709 gamma + numba parallel LUTs.
            ~3x faster end-to-end (~550ms), mean pixel diff ~6.5/255 vs LibRaw.
  - LIBRAW: LibRaw PPG demosaic — accurate reference, ~2.9s per image.

Supports output formats: JPEG, PNG, TIFF.
Retains all EXIF/GPS/XMP metadata from the original IIQ file.
See FINDINGS.md for details on how the fast pipeline was matched to LibRaw.
"""

import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

logger = logging.getLogger(__name__)


_VALID_PIPELINES = ("libraw", "fast")
_VALID_ROTATIONS = (0, 90, 180, 270, 360)


def _resolve_pipeline(value: str) -> str:
    """Validate and normalise a pipeline name to lowercase."""
    v = value.lower()
    if v not in _VALID_PIPELINES:
        raise ValueError(
            f"Unknown pipeline: {value!r}. Expected one of: {', '.join(_VALID_PIPELINES)!r}"
        )
    return v


def _validate_iiq_path(iiq_path: Path) -> None:
    """Raise if path doesn't exist, isn't a file, or isn't .IIQ."""
    if not iiq_path.exists():
        raise FileNotFoundError(f"IIQ file not found: {iiq_path}")
    if not iiq_path.is_file():
        raise ValueError(f"Expected a file, got a directory: {iiq_path}")
    if iiq_path.suffix.lower() != ".iiq":
        raise ValueError(f"Expected a .IIQ file, got '{iiq_path.suffix}': {iiq_path}")


def _resolve_rotate(value: int) -> int:
    """Validate and normalise a rotation angle. Returns 0, 90, 180, or 270 (360 maps to 0)."""
    if value not in _VALID_ROTATIONS:
        raise ValueError(
            f"Invalid rotation: {value}°. Must be one of: 0, 90, 180, 270 (360 is treated as 0)"
        )
    return value % 360


def read_iiq(
    iiq_path: str | Path,
    thumbnail: bool = False,
    max_dimension: int | None = None,
    rotate: int = 0,
    pipeline: str = "fast",
) -> np.ndarray:
    """
    Read a Phase One IIQ raw file and return it as an RGB numpy array.

    Args:
        iiq_path: Path to .IIQ file
        thumbnail: If True, extract the embedded 640x480 thumbnail instead of full demosaic.
        max_dimension: If set, resize longest edge to this value
        rotate: Rotation angle in degrees — 0, 90, 180, 270 (for different camera mounts).
                360 is accepted and treated as 0.
        pipeline: Demosaic pipeline — 'fast' (~3x faster, default) or 'libraw' (accurate, ~2.8s).

    Returns:
        np.ndarray with shape (H, W, 3), dtype uint8, in RGB order.
    """
    iiq_path = Path(iiq_path)
    pipeline = _resolve_pipeline(pipeline)
    rotate = _resolve_rotate(rotate)
    _validate_iiq_path(iiq_path)

    if thumbnail:
        rgb = _extract_thumbnail(iiq_path)
    else:
        rgb = _demosaic(iiq_path, pipeline)

    if max_dimension and max(rgb.shape[:2]) > max_dimension:
        rgb = resize_max_dim(rgb, max_dimension)

    if rotate:
        rgb = np.rot90(rgb, rotate // 90)

    return rgb


def convert_iiq(
    iiq_path: str | Path,
    output_path: str | Path | None = None,
    thumbnail: bool = False,
    output_format: str | None = None,
    compress_quality: int = 90,
    max_dimension: int | None = None,
    extract_meta: bool = True,
    georef: bool = False,
    rotate: int = 0,
    pipeline: str = "fast",
    verbose: bool = False,
) -> Path:
    """
    Convert a Phase One IIQ raw file to JPEG, PNG, or TIFF.

    Args:
        iiq_path: Path to .IIQ file
        output_path: Output path. If None, uses same name with new extension.
        thumbnail: If True, extract the embedded 640x480 thumbnail instead of full demosaic.
        output_format: Output format string ('jpg', 'png', 'tif'),
                       or None to infer from output_path extension (default: 'jpg')
        compress_quality: Compression quality 1-100 (JPEG quality / PNG compression)
        max_dimension: If set, resize longest edge to this value
        extract_meta: Whether to extract and retain EXIF metadata
        georef: If True, georeference the output (world file for JPEG/PNG, GeoTIFF for TIFF)
        rotate: Rotation angle in degrees — 0, 90, 180, 270 (for different camera mounts).
                360 is accepted and treated as 0.
        pipeline: Demosaic pipeline — 'fast' (~3x faster, default) or 'libraw' (accurate, ~2.8s).
        verbose: If True, print per-step timing information.

    Returns:
        Path to the output file.
    """
    iiq_path = Path(iiq_path)
    pipeline = _resolve_pipeline(pipeline)
    rotate = _resolve_rotate(rotate)
    _validate_iiq_path(iiq_path)
    logger.debug(
        "Converting %s -> %s (pipeline=%s)", iiq_path, output_path or "auto", pipeline
    )

    if verbose:
        t_total = time.perf_counter()

    out_path = Path(output_path) if output_path is not None else None
    output_fmt = resolve_format(output_format, out_path)

    if out_path is None:
        ext = FORMAT_EXTENSIONS[output_fmt]
        out_path = iiq_path.with_suffix(ext)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always extract metadata if georef is requested (need XMP for GPS)
    need_meta = extract_meta or georef

    if verbose:
        t0 = time.perf_counter()

    if thumbnail:
        rgb = _extract_thumbnail(iiq_path)
    else:
        rgb = _demosaic(iiq_path, pipeline)

    if verbose:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Demosaic ({pipeline}): {elapsed:.0f}ms")

    if verbose:
        t0 = time.perf_counter()

    metadata = extract_metadata(iiq_path) if need_meta else {}

    if verbose:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Metadata: {elapsed:.0f}ms")

    if max_dimension and max(rgb.shape[:2]) > max_dimension:
        rgb = resize_max_dim(rgb, max_dimension)

    if rotate:
        rgb = np.rot90(rgb, rotate // 90)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    is_geotiff = georef and output_fmt == "tiff"

    if verbose:
        t0 = time.perf_counter()

    if is_geotiff:
        # Write GeoTIFF directly (skip EXIF injection — rasterio handles metadata)
        _write_geotiff(rgb, out_path, metadata, compress_quality)
    else:
        encode_image(bgr, out_path, output_fmt, compress_quality)
        if extract_meta and metadata:
            copy_metadata_to_output(iiq_path, out_path, metadata)

    if verbose:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Encode ({output_fmt}): {elapsed:.0f}ms")

    if georef and not is_geotiff:
        _apply_georef(out_path, metadata, w, h)

    if verbose:
        total = (time.perf_counter() - t_total) * 1000
        print(f"  Total: {total:.0f}ms -> {out_path}")

    logger.debug("Conversion complete: %s", out_path)
    return out_path


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
    """Extract embedded thumbnail (~640x480, ~1ms). Handles BITMAP and JPEG formats."""
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


def _demosaic(iiq_path: Path, pipeline: str = "fast") -> np.ndarray:
    """Demosaic IIQ raw data to full-resolution RGB."""
    logger.debug("Demosaicing %s with %s pipeline", iiq_path, pipeline)
    if pipeline == "fast":
        return demosaic_fast(iiq_path)

    from iiq2img.repair import repair_defective_rows

    try:
        raw = rawpy.imread(str(iiq_path))
    except rawpy.LibRawIOError:  # type: ignore[attr-defined]
        raise OSError(
            f"Failed to read IIQ file (corrupt or unreadable): {iiq_path}"
        ) from None

    # Repair defective rows on the raw Bayer data before LibRaw postprocess
    repaired = repair_defective_rows(raw.raw_image_visible)
    raw.raw_image_visible[:] = repaired

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


def _convert_one_for_batch(
    args: tuple[str, str, bool, str, int, int | None, str],
) -> Path:
    """Worker function for multiprocessing batch conversion."""
    (
        iiq_path,
        out_path,
        thumbnail,
        output_format,
        compress_quality,
        max_dimension,
        pipeline,
    ) = args
    return convert_iiq(
        iiq_path,
        out_path,
        thumbnail=thumbnail,
        output_format=output_format,
        compress_quality=compress_quality,
        max_dimension=max_dimension,
        pipeline=pipeline,
    )


def batch_convert(
    input_dir: str | Path,
    output_dir: str | Path,
    thumbnail: bool = False,
    output_format: str = "jpg",
    compress_quality: int = 90,
    max_dimension: int | None = None,
    workers: int | None = None,
    pipeline: str = "fast",
) -> list[Path]:
    """Convert all IIQ files in a directory using multiprocessing.

    Args:
        workers: Number of parallel workers. None = auto (min(cpu_count, 8)).
                 Set to 1 for sequential processing.
        pipeline: Demosaic pipeline — 'fast' (default, ~3x faster) or 'libraw' (accurate).
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
            thumbnail,
            output_fmt,
            compress_quality,
            max_dimension,
            pipeline,
        )
        for f in iiq_files
    ]

    from tqdm.auto import tqdm

    total_t0 = time.perf_counter()
    results: list[Path] = []
    pbar = tqdm(total=len(tasks), unit="img")

    if workers <= 1:
        for task_args in tasks:
            result = _convert_one_for_batch(task_args)
            results.append(result)
            pbar.set_postfix_str(Path(task_args[0]).stem)
            pbar.update(1)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as exe:
            futures = {exe.submit(_convert_one_for_batch, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.set_postfix_str(result.stem)
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

    # (label, thumbnail, format, compress_quality, max_dim, pipeline)
    approaches = [
        ("Thumbnail (640x480) JPG", True, "jpg", 90, None, "libraw"),
        ("LibRaw PPG  JPG q=90", False, "jpg", 90, None, "libraw"),
        ("LibRaw PPG  JPG q=75", False, "jpg", 75, None, "libraw"),
        ("Fast (cv2 EA)  JPG q=90", False, "jpg", 90, None, "fast"),
        ("Fast (cv2 EA)  JPG q=75", False, "jpg", 75, None, "fast"),
        ("Fast (cv2 EA)  PNG", False, "png", 90, None, "fast"),
        ("Fast (cv2 EA)  TIFF", False, "tiff", 90, None, "fast"),
    ]

    print(f"{'Approach':<35} {'Time':>8} {'Size':>10}")
    print("-" * 57)

    for name, thumb, fmt, cq, max_dim, pl in approaches:
        tag = "thumb" if thumb else pl
        out = Path(f"/tmp/bench_{tag}_{fmt}_{cq}{FORMAT_EXTENSIONS[fmt]}")
        t0 = time.perf_counter()
        convert_iiq(
            iiq_path,
            out,
            thumbnail=thumb,
            output_format=fmt,
            compress_quality=cq,
            max_dimension=max_dim,
            extract_meta=False,
            pipeline=pl,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        file_size = out.stat().st_size
        print(f"{name:<35} {elapsed_ms:>7.0f}ms {file_size / 1024 / 1024:>8.1f}MB")


def _build_cli_parsers() -> tuple:
    """Build argparse parsers for benchmark, batch, and single-file modes."""
    import argparse

    sample = "PhaseOneSample/P0286625.IIQ"

    # Top-level parser (also used for help display)
    main_parser = argparse.ArgumentParser(
        prog="iiq2img",
        description="Fast IIQ converter for Phase One iXM-GS120 (120MP) raw images.",
        usage="%(prog)s <file.IIQ> [--libraw]\n"
        "       %(prog)s benchmark [iiq_path]\n"
        "       %(prog)s batch [input_dir] [output_dir] [options]",
    )
    main_parser.add_argument(
        "--libraw", action="store_true", help="Use LibRaw PPG pipeline"
    )

    # benchmark parser
    bench_parser = argparse.ArgumentParser(prog="iiq2img benchmark")
    bench_parser.add_argument(
        "iiq_path", nargs="?", default=sample, help="IIQ file to benchmark"
    )

    # batch parser
    batch_parser = argparse.ArgumentParser(prog="iiq2img batch")
    batch_parser.add_argument(
        "input_dir", nargs="?", default="PhaseOneSample", help="Input directory"
    )
    batch_parser.add_argument(
        "output_dir", nargs="?", default="/tmp/iiq_output", help="Output directory"
    )
    batch_parser.add_argument(
        "--format", default="jpg", dest="format", help="Output format (default: jpg)"
    )
    batch_parser.add_argument(
        "--quality", type=int, default=90, help="Compression quality (default: 90)"
    )
    batch_parser.add_argument(
        "--workers", type=int, default=None, help="Number of workers (default: auto)"
    )
    batch_parser.add_argument(
        "--libraw", action="store_true", help="Use LibRaw PPG pipeline"
    )

    # single-file parser
    file_parser = argparse.ArgumentParser(prog="iiq2img")
    file_parser.add_argument("file", help="IIQ file to convert")
    file_parser.add_argument(
        "--libraw", action="store_true", help="Use LibRaw PPG pipeline"
    )

    return main_parser, bench_parser, batch_parser, file_parser


def _cli_main() -> None:
    """CLI entry point for the iiq2img command."""
    main_parser, bench_parser, batch_parser, file_parser = _build_cli_parsers()

    argv = sys.argv[1:]
    if not argv:
        main_parser.print_help()
        return

    command = argv[0]

    if command == "benchmark":
        args = bench_parser.parse_args(argv[1:])
        run_benchmark(args.iiq_path)

    elif command == "batch":
        args = batch_parser.parse_args(argv[1:])
        pl = "libraw" if args.libraw else "fast"
        batch_convert(
            args.input_dir,
            args.output_dir,
            output_format=args.format,
            compress_quality=args.quality,
            workers=args.workers,
            pipeline=pl,
        )

    elif command.startswith("-"):
        # Only flags, no file — show help
        main_parser.print_help()

    else:
        args = file_parser.parse_args(argv)
        pl = "libraw" if args.libraw else "fast"
        t0 = time.perf_counter()
        out_path = convert_iiq(args.file, pipeline=pl)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"Output:   {out_path}")
        print(f"Pipeline: {pl}")
        print(f"Time:     {elapsed_ms:.0f}ms")
        print(f"Size:     {out_path.stat().st_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    _cli_main()
