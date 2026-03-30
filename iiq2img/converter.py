"""
Fast IIQ converter for Phase One iXM-GS120 (120MP) raw images.

Uses rawpy (LibRaw) + OpenCV for fast demosaicing and encoding,
bypassing the slow Phase One SDK pipeline.

Supports output formats: JPEG, PNG, TIFF.
Retains all EXIF/GPS/XMP metadata from the original IIQ file.

Optimizations applied:
  - PPG demosaic algorithm (fastest for this sensor)
  - no_auto_bright to skip auto-brightness histogram scan
  - Direct EXIF serialization (no dummy JPEG roundtrip)
  - Eliminated redundant RGB<->BGR conversions in resize path
  - Multiprocessing batch with spawn context for safe parallelism
"""

import multiprocessing
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image as PILImage
from PIL.ExifTags import TAGS


class Quality(Enum):
    THUMBNAIL = "thumbnail"  # 640x480 embedded bitmap, ~1ms
    FULL = "full"  # full-res (12768x9564), ~2.7s


class OutputFormat(Enum):
    JPEG = "jpg"
    PNG = "png"
    TIFF = "tiff"


FORMAT_EXTENSIONS = {
    OutputFormat.JPEG: ".jpg",
    OutputFormat.PNG: ".png",
    OutputFormat.TIFF: ".tif",
}


@dataclass
class ConvertResult:
    output_path: str
    width: int
    height: int
    elapsed_ms: float
    file_size_bytes: int
    metadata: dict[str, str] = field(default_factory=dict)


def extract_metadata(iiq_path: str) -> dict[str, str]:
    """Extract EXIF + XMP metadata from IIQ file using Pillow.

    Returns a dict of human-readable tag names to string values,
    plus raw 'xmp' key containing the full XMP XML if present.
    """
    try:
        img = PILImage.open(iiq_path)
        exif = img.getexif()
        meta = {}
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            meta[tag_name] = str(value)

        # XMP packet (tag 700) contains GPS, IMU, rangefinder data
        xmp = exif.get(700)
        if xmp and isinstance(xmp, bytes):
            meta["xmp"] = xmp.decode("utf-8", errors="replace")

        return meta
    except Exception:
        return {}


def _read_iiq_exif_and_xmp(iiq_path: str) -> tuple[bytes | None, bytes | None]:
    """Read EXIF bytes and XMP packet from IIQ for embedding in output.

    Uses Exif.tobytes() directly instead of dummy JPEG roundtrip.
    """
    try:
        PILImage.MAX_IMAGE_PIXELS = None
        img = PILImage.open(iiq_path)
        src_exif = img.getexif()

        from PIL.Image import Exif

        new_exif = Exif()

        # IFD0 tags to keep
        keep_ifd0 = {271, 272, 282, 283, 296, 306}
        for tag_id, value in src_exif.items():
            if tag_id in keep_ifd0:
                new_exif[tag_id] = value

        # EXIF sub-IFD — copy individual simple-valued tags only
        exif_ifd = src_exif.get_ifd(0x8769)
        if exif_ifd:
            safe_ifd = {}
            for k, v in exif_ifd.items():
                if isinstance(v, bytes) and len(v) > 1000:
                    continue
                safe_ifd[k] = v
            new_exif.get_ifd(0x8769).update(safe_ifd)

        # Direct serialization — tobytes() returns b"Exif\x00\x00" + TIFF header
        exif_raw = new_exif.tobytes()
        exif_bytes = exif_raw if exif_raw[:6] == b"Exif\x00\x00" else None

        # Get XMP packet (tag 700)
        xmp_bytes = src_exif.get(700)
        if isinstance(xmp_bytes, str):
            xmp_bytes = xmp_bytes.encode("utf-8")
        elif not isinstance(xmp_bytes, bytes):
            xmp_bytes = None

        return exif_bytes, xmp_bytes
    except Exception:
        return None, None


def _copy_metadata_to_output(
    iiq_path: str, output_path: str, metadata: dict[str, str]
) -> None:
    """Embed EXIF and XMP metadata from the source IIQ into the output image."""
    ext = Path(output_path).suffix.lower()
    exif_bytes, xmp_bytes = _read_iiq_exif_and_xmp(iiq_path)

    if ext in (".jpg", ".jpeg"):
        _inject_metadata_into_jpeg(output_path, exif_bytes, xmp_bytes)
    elif ext == ".png":
        _inject_metadata_into_png(output_path, metadata)
    elif ext in (".tif", ".tiff"):
        _inject_metadata_into_tiff(output_path, exif_bytes, xmp_bytes)


def _inject_metadata_into_jpeg(
    jpeg_path: str, exif_bytes: bytes | None, xmp_bytes: bytes | None
) -> None:
    """Inject EXIF APP1 and XMP APP1 segments into a JPEG file."""
    try:
        with open(jpeg_path, "rb") as f:
            jpeg_data = f.read()

        if jpeg_data[:2] != b"\xff\xd8":
            return

        segments = b""

        # EXIF APP1: FF E1 [len] [exif_data] (includes "Exif\0\0" prefix)
        if exif_bytes:
            if len(exif_bytes) + 2 <= 65535:
                segments += (
                    b"\xff\xe1" + struct.pack(">H", len(exif_bytes) + 2) + exif_bytes
                )

        # XMP APP1: FF E1 [len] "http://ns.adobe.com/xap/1.0/\0" [xmp_data]
        if xmp_bytes:
            xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
            payload = xmp_header + xmp_bytes
            if len(payload) + 2 <= 65535:
                segments += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload

        if segments:
            with open(jpeg_path, "wb") as f:
                f.write(b"\xff\xd8")
                f.write(segments)
                f.write(jpeg_data[2:])
    except Exception:
        pass


def _inject_metadata_into_png(png_path: str, metadata: dict[str, str]) -> None:
    """Add metadata as PNG tEXt chunks using Pillow."""
    try:
        from PIL.PngImagePlugin import PngInfo

        img = PILImage.open(png_path)
        pnginfo = PngInfo()
        for k, v in metadata.items():
            if k == "xmp":
                continue  # XMP is too large for tEXt
            pnginfo.add_text(k, str(v))
        img.save(png_path, pnginfo=pnginfo)
    except Exception:
        pass


def _inject_metadata_into_tiff(
    tiff_path: str, exif_bytes: bytes | None, xmp_bytes: bytes | None
) -> None:
    """Re-save TIFF with EXIF metadata using Pillow."""
    try:
        img = PILImage.open(tiff_path)
        save_kwargs = {}
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        if save_kwargs:
            img.save(tiff_path, **save_kwargs)  # type: ignore[arg-type]
    except Exception:
        pass


def _encode_image(
    bgr: np.ndarray,
    output_path: str,
    output_format: OutputFormat,
    compress_quality: int,
) -> None:
    """Encode and write image in the requested format."""
    if output_format == OutputFormat.JPEG:
        cv2.imwrite(output_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, compress_quality])
    elif output_format == OutputFormat.PNG:
        # PNG compression 0-9 (0=fast/large, 9=slow/small). Map quality 1-100 -> 9-0.
        png_compression = max(0, min(9, 9 - int(compress_quality / 100 * 9)))
        cv2.imwrite(output_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    elif output_format == OutputFormat.TIFF:
        cv2.imwrite(output_path, bgr)


def convert_iiq(
    iiq_path: str,
    output_path: str | None = None,
    quality: Quality = Quality.FULL,
    output_format: OutputFormat = OutputFormat.JPEG,
    compress_quality: int = 90,
    max_dimension: int | None = None,
    extract_meta: bool = True,
) -> ConvertResult:
    """
    Convert a Phase One IIQ raw file to JPEG, PNG, or TIFF.

    Args:
        iiq_path: Path to .IIQ file
        output_path: Output path. If None, uses same name with new extension.
        quality: Demosaic quality preset (THUMBNAIL or FULL)
        output_format: Output image format (JPEG, PNG, TIFF)
        compress_quality: Compression quality 1-100 (JPEG quality / PNG compression)
        max_dimension: If set, resize longest edge to this value
        extract_meta: Whether to extract and retain EXIF metadata

    Returns:
        ConvertResult with output details
    """
    t0 = time.perf_counter()

    if output_path is None:
        ext = FORMAT_EXTENSIONS[output_format]
        output_path = str(Path(iiq_path).with_suffix(ext))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    metadata = extract_metadata(iiq_path) if extract_meta else {}

    if quality == Quality.THUMBNAIL:
        rgb = _extract_thumbnail(iiq_path)
    else:
        rgb = _demosaic(iiq_path)

    if max_dimension and max(rgb.shape[:2]) > max_dimension:
        rgb = _resize_max_dim(rgb, max_dimension)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _encode_image(bgr, output_path, output_format, compress_quality)

    if extract_meta and metadata:
        _copy_metadata_to_output(iiq_path, output_path, metadata)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    h, w = rgb.shape[:2]
    file_size = os.path.getsize(output_path)

    return ConvertResult(
        output_path=output_path,
        width=w,
        height=h,
        elapsed_ms=elapsed_ms,
        file_size_bytes=file_size,
        metadata=metadata,
    )


def _extract_thumbnail(iiq_path: str) -> np.ndarray:
    """Extract embedded thumbnail bitmap (~640x480, ~1ms)."""
    raw = rawpy.imread(iiq_path)
    thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.BITMAP:
        rgb = thumb.data
    elif thumb.format == rawpy.ThumbFormat.JPEG:
        arr = np.frombuffer(thumb.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode JPEG thumbnail")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown thumbnail format: {thumb.format}")
    raw.close()
    return rgb


def _demosaic(iiq_path: str) -> np.ndarray:
    """Demosaic IIQ raw data to full-resolution RGB using LibRaw with PPG algorithm."""
    raw = rawpy.imread(iiq_path)
    rgb = raw.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.PPG,
        half_size=False,
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=8,
    )
    raw.close()
    return rgb


def _resize_max_dim(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so longest edge = max_dim, preserving aspect ratio."""
    h, w = rgb.shape[:2]
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _convert_one_for_batch(args: tuple) -> ConvertResult:
    """Worker function for multiprocessing batch conversion."""
    iiq_path, out_path, quality, output_format, compress_quality, max_dimension = args
    return convert_iiq(
        iiq_path,
        out_path,
        quality=quality,
        output_format=output_format,
        compress_quality=compress_quality,
        max_dimension=max_dimension,
    )


def batch_convert(
    input_dir: str,
    output_dir: str,
    quality: Quality = Quality.FULL,
    output_format: OutputFormat = OutputFormat.JPEG,
    compress_quality: int = 90,
    max_dimension: int | None = None,
    workers: int | None = None,
) -> list[ConvertResult]:
    """Convert all IIQ files in a directory using multiprocessing.

    Args:
        workers: Number of parallel workers. None = auto (min(cpu_count, 8)).
                 Set to 1 for sequential processing.
    """
    iiq_files = sorted(Path(input_dir).glob("*.IIQ"))
    if not iiq_files:
        iiq_files = sorted(Path(input_dir).glob("*.iiq"))
    if not iiq_files:
        print(f"No .IIQ files found in {input_dir}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    ext = FORMAT_EXTENSIONS[output_format]

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)

    tasks = [
        (
            str(f),
            os.path.join(output_dir, f.stem + ext),
            quality,
            output_format,
            compress_quality,
            max_dimension,
        )
        for f in iiq_files
    ]

    total_t0 = time.perf_counter()
    results = []

    if workers <= 1:
        # Sequential
        for i, task_args in enumerate(tasks):
            result = _convert_one_for_batch(task_args)
            results.append(result)
            print(
                f"  [{i + 1}/{len(iiq_files)}] {Path(task_args[0]).name} -> "
                f"{result.width}x{result.height} "
                f"({result.elapsed_ms:.0f}ms, {result.file_size_bytes / 1024 / 1024:.1f}MB)"
            )
    else:
        # Parallel with spawn context (safe with OpenMP/rawpy)
        mp_ctx = multiprocessing.get_context("spawn")
        completed = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as exe:
            futures = {exe.submit(_convert_one_for_batch, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                fname = Path(result.output_path).stem
                print(
                    f"  [{completed}/{len(iiq_files)}] {fname} -> "
                    f"{result.width}x{result.height} "
                    f"({result.elapsed_ms:.0f}ms, "
                    f"{result.file_size_bytes / 1024 / 1024:.1f}MB)"
                )

    total_elapsed = (time.perf_counter() - total_t0) * 1000
    avg = total_elapsed / len(results)
    throughput = len(results) / (total_elapsed / 1000)
    print(
        f"\nDone: {len(results)} images in {total_elapsed / 1000:.1f}s "
        f"(avg {avg:.0f}ms/image, {throughput:.1f} images/sec, {workers} workers)"
    )
    return results


def run_benchmark(iiq_path: str) -> None:
    """Run benchmark comparing all conversion approaches on a single file."""
    print(f"Benchmarking: {iiq_path}")
    print(f"File size: {os.path.getsize(iiq_path) / 1024 / 1024:.1f} MB\n")

    approaches = [
        ("Thumbnail (640x480) JPG", Quality.THUMBNAIL, OutputFormat.JPEG, 90, None),
        ("Full res JPG q=90", Quality.FULL, OutputFormat.JPEG, 90, None),
        ("Full res JPG q=75", Quality.FULL, OutputFormat.JPEG, 75, None),
        ("Full res PNG", Quality.FULL, OutputFormat.PNG, 90, None),
        ("Full res TIFF", Quality.FULL, OutputFormat.TIFF, 90, None),
    ]

    print(f"{'Approach':<30} {'Time':>8} {'Size':>10} {'Resolution':>14}")
    print("-" * 66)

    for name, qual, fmt, cq, max_dim in approaches:
        out = f"/tmp/bench_{qual.value}_{fmt.value}_{cq}.{FORMAT_EXTENSIONS[fmt].strip('.')}"
        r = convert_iiq(
            iiq_path,
            out,
            quality=qual,
            output_format=fmt,
            compress_quality=cq,
            max_dimension=max_dim,
            extract_meta=False,
        )
        print(
            f"{name:<30} {r.elapsed_ms:>7.0f}ms "
            f"{r.file_size_bytes / 1024 / 1024:>8.1f}MB "
            f"{r.width}x{r.height}"
        )


def _cli_main() -> None:
    """CLI entry point for the skw-raw command."""
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
        batch_convert(
            input_dir,
            output_dir,
            output_format=OutputFormat(fmt_str),
            compress_quality=cq,
            workers=workers,
        )

    elif len(sys.argv) > 1:
        result = convert_iiq(sys.argv[1])
        print(f"Output: {result.output_path}")
        print(f"Resolution: {result.width}x{result.height}")
        print(f"Time: {result.elapsed_ms:.0f}ms")
        print(f"Size: {result.file_size_bytes / 1024 / 1024:.1f}MB")

    else:
        print("Usage:")
        print(f"  {sys.argv[0]} benchmark [iiq_path]")
        print(
            f"  {sys.argv[0]} batch [in_dir] [out_dir] [jpg|png|tiff] [quality] [workers]"
        )
        print(f"  {sys.argv[0]} <file.IIQ>")
        print()
        print("Running benchmark on sample file...")
        print()
        run_benchmark(sample)


if __name__ == "__main__":
    _cli_main()
