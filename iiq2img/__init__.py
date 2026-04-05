"""iiq2img — Fast IIQ converter for Phase One iXM-GS120 (120MP) raw images."""

from importlib.metadata import version

__version__ = version("iiq2img")

from iiq2img.converter import (
    batch_convert,
    convert_iiq,
    read_iiq,
)
from iiq2img.encode import (
    FORMAT_EXTENSIONS,
    format_from_path,
    normalize_format,
)
from iiq2img.georef import (
    GeoInfo,
    extract_geo_info,
    georeference_image,
    write_geotiff,
    write_world_file,
)
from iiq2img.metadata import extract_metadata

__all__ = [
    "GeoInfo",
    "FORMAT_EXTENSIONS",
    "batch_convert",
    "convert_iiq",
    "read_iiq",
    "extract_geo_info",
    "extract_metadata",
    "format_from_path",
    "georeference_image",
    "normalize_format",
    "write_geotiff",
    "write_world_file",
]
