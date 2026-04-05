"""EXIF and XMP metadata extraction and injection for IIQ output images."""

import logging
import struct
from pathlib import Path

from PIL import Image as PILImage
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


def extract_metadata(iiq_path: str | Path) -> dict[str, str]:
    """Extract EXIF + XMP metadata from IIQ file using Pillow.

    Returns a dict of human-readable tag names to string values,
    plus raw 'xmp' key containing the full XMP XML if present.
    Returns an empty dict on any error (logged at DEBUG level).
    """
    iiq_path = Path(iiq_path)
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
            meta["xmp"] = xmp.decode("utf-8", errors="replace").rstrip("\x00").strip()

        return meta
    except Exception:
        logger.debug("Failed to extract metadata from %s", iiq_path, exc_info=True)
        return {}


def _read_iiq_exif_and_xmp(iiq_path: str | Path) -> tuple[bytes | None, bytes | None]:
    """Read EXIF bytes and XMP packet from IIQ for embedding in output.

    Uses Exif.tobytes() directly instead of dummy JPEG roundtrip.
    Returns (None, None) on any error (logged at DEBUG level).
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
        logger.debug("Failed to read EXIF/XMP from %s", iiq_path, exc_info=True)
        return None, None


def copy_metadata_to_output(
    iiq_path: str | Path, output_path: str | Path, metadata: dict[str, str]
) -> None:
    """Embed EXIF and XMP metadata from the source IIQ into the output image."""
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    exif_bytes, xmp_bytes = _read_iiq_exif_and_xmp(iiq_path)

    if ext in (".jpg", ".jpeg"):
        _inject_metadata_into_jpeg(output_path, exif_bytes, xmp_bytes)
    elif ext == ".png":
        _inject_metadata_into_png(output_path, metadata)
    elif ext in (".tif", ".tiff"):
        _inject_metadata_into_tiff(output_path, exif_bytes, xmp_bytes)


def _inject_metadata_into_jpeg(
    jpeg_path: str | Path, exif_bytes: bytes | None, xmp_bytes: bytes | None
) -> None:
    """Inject EXIF APP1 and XMP APP1 segments into a JPEG file. Errors are logged at DEBUG level."""
    try:
        jpeg_path = Path(jpeg_path)
        jpeg_data = jpeg_path.read_bytes()

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
            jpeg_path.write_bytes(b"\xff\xd8" + segments + jpeg_data[2:])
    except Exception:
        logger.debug("Failed to inject metadata into JPEG %s", jpeg_path, exc_info=True)


def _inject_metadata_into_png(png_path: str | Path, metadata: dict[str, str]) -> None:
    """Add metadata as PNG tEXt chunks using Pillow. Errors are logged at DEBUG level."""
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
        logger.debug("Failed to inject metadata into PNG %s", png_path, exc_info=True)


def _inject_metadata_into_tiff(
    tiff_path: str | Path, exif_bytes: bytes | None, xmp_bytes: bytes | None
) -> None:
    """Re-save TIFF with EXIF metadata using Pillow. Errors are logged at DEBUG level."""
    try:
        img = PILImage.open(tiff_path)
        save_kwargs = {}
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        if save_kwargs:
            img.save(tiff_path, **save_kwargs)  # type: ignore[arg-type]
    except Exception:
        logger.debug("Failed to inject metadata into TIFF %s", tiff_path, exc_info=True)
