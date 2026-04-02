"""Image encoding, format helpers, and resize utilities."""

from pathlib import Path

import cv2
import numpy as np

_FORMAT_ALIASES: dict[str, str] = {
    "jpg": "jpg",
    "jpeg": "jpg",
    "png": "png",
    "tif": "tiff",
    "tiff": "tiff",
}

FORMAT_EXTENSIONS: dict[str, str] = {
    "jpg": ".jpg",
    "png": ".png",
    "tiff": ".tif",
}


def normalize_format(value: str) -> str:
    """Normalize a format string like 'jpg', 'JPEG', 'tif', 'tiff', 'PNG' etc.

    Returns canonical format name: 'jpg', 'png', or 'tiff'.
    """
    key = value.lower().strip().lstrip(".")
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unknown output format: {value!r}. Expected one of: jpg, jpeg, png, tif, tiff"
    )


def format_from_path(path: Path) -> str | None:
    """Infer output format from file extension. Returns None if unknown."""
    try:
        return normalize_format(path.suffix)
    except ValueError:
        return None


def resolve_format(output_format: str | None, output_path: Path | None) -> str:
    """Resolve output format from explicit value or file extension."""
    if isinstance(output_format, str):
        return normalize_format(output_format)
    if output_path is not None:
        fmt = format_from_path(output_path)
        if fmt is not None:
            return fmt
    return "jpg"


def encode_image(
    bgr: np.ndarray,
    output_path: str | Path,
    output_format: str,
    compress_quality: int,
) -> None:
    """Encode and write image in the requested format."""
    path_str = str(output_path)
    if output_format == "jpg":
        cv2.imwrite(path_str, bgr, [cv2.IMWRITE_JPEG_QUALITY, compress_quality])
    elif output_format == "png":
        # PNG compression 0-9 (0=fast/large, 9=slow/small). Map quality 1-100 -> 9-0.
        png_compression = max(0, min(9, 9 - int(compress_quality / 100 * 9)))
        cv2.imwrite(path_str, bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    elif output_format == "tiff":
        cv2.imwrite(path_str, bgr)


def resize_max_dim(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so longest edge = max_dim, preserving aspect ratio."""
    h, w = rgb.shape[:2]
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
