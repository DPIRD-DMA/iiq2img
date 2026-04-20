"""Georeferencing support for iiq2img.

Computes ground footprint from camera position, altitude, focal length,
and sensor geometry, then writes:
  - World files (.jgw/.pgw) for JPEG/PNG
  - GeoTIFF for TIFF
  - Standalone function to georeference any existing image
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from rasterio.transform import Affine

# iXM-GS120 sensor: 3.45 μm pixel pitch
PIXEL_SIZE_MM = 0.00345  # mm per pixel


@dataclass
class GeoInfo:
    """Georeferencing parameters extracted from IIQ metadata."""

    latitude: float  # decimal degrees, negative = south
    longitude: float  # decimal degrees, negative = west
    altitude_agl: float  # meters above ground level
    focal_length_mm: float  # mm
    yaw_deg: float  # degrees, true north
    image_width: int
    image_height: int

    @property
    def gsd(self) -> float:
        """Ground sample distance in meters."""
        return (self.altitude_agl * PIXEL_SIZE_MM) / self.focal_length_mm

    @property
    def footprint_width(self) -> float:
        """Ground footprint width in meters."""
        return self.gsd * self.image_width

    @property
    def footprint_height(self) -> float:
        """Ground footprint height in meters."""
        return self.gsd * self.image_height


def _parse_dms_coordinate(dms_str: str) -> float:
    """Parse Phase One DMS string like '34,33.600456S' to decimal degrees."""
    match = re.match(r"(\d+),(\d+\.?\d*)(.*)", dms_str.strip())
    if not match:
        raise ValueError(f"Cannot parse coordinate: {dms_str}")
    degrees = float(match.group(1))
    minutes = float(match.group(2))
    suffix = match.group(3).strip()
    decimal = degrees + minutes / 60.0
    if suffix in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_rational(val: str) -> float:
    """Parse a rational string like '151615/1000' to float."""
    if "/" in val:
        num, den = val.split("/")
        return float(num) / float(den)
    return float(val)


def extract_geo_info(
    xmp: str,
    focal_length_mm: float,
    image_width: int,
    image_height: int,
) -> GeoInfo | None:
    """Extract georeferencing parameters from XMP metadata string.

    Args:
        xmp: XMP XML string from IIQ metadata
        focal_length_mm: Focal length in mm (from EXIF)
        image_width: Output image width in pixels
        image_height: Output image height in pixels

    Returns:
        GeoInfo if GPS data is available, None otherwise

    Note:
        Altitude defaults to 100m AGL if not found in metadata.
        Yaw defaults to 0 degrees (north) if not found.

    Example::

        metadata = extract_metadata("photo.IIQ")
        geo = extract_geo_info(metadata.get("xmp", ""), 80.0, 12768, 9564)
        if geo:
            print(f"GSD: {geo.gsd:.3f} m/px")
    """
    try:
        root = ET.fromstring(xmp)
    except ET.ParseError:
        return None

    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "exif": "http://ns.adobe.com/exif/1.0/",
        "aerialgps": "http://www.phaseone.com/aerialgps/",
        "Camera": "http://www.phaseone.com/camera/",
    }

    def _find_text(tag: str) -> str | None:
        for desc in root.iter(
            "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
        ):
            el = desc.find(tag, ns)
            if el is not None and el.text:
                return el.text
            # Also check attributes
            for attr_key, attr_val in desc.attrib.items():
                if attr_key.endswith(
                    tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                ):
                    return attr_val
        return None

    # Try flight GPS first (more accurate), fall back to trigger GPS
    lat_str = _find_text("aerialgps:GPSFlightLatitude") or _find_text(
        "exif:GPSLatitude"
    )
    lon_str = _find_text("aerialgps:GPSFlightLongitude") or _find_text(
        "exif:GPSLongitude"
    )
    alt_agl_str = _find_text("aerialgps:GPSAltitudeAboveTakeOff") or _find_text(
        "Camera:AboveGroundAltitude"
    )
    yaw_str = _find_text("Camera:Yaw") or _find_text("aerialgps:GPSIMUYaw")

    if not lat_str or not lon_str:
        return None

    lat = _parse_dms_coordinate(lat_str)
    lon = _parse_dms_coordinate(lon_str)
    alt_agl = _parse_rational(alt_agl_str) if alt_agl_str else 100.0
    yaw = _parse_rational(yaw_str) if yaw_str else 0.0

    return GeoInfo(
        latitude=lat,
        longitude=lon,
        altitude_agl=alt_agl,
        focal_length_mm=focal_length_mm,
        yaw_deg=yaw,
        image_width=image_width,
        image_height=image_height,
    )


def _meters_per_degree_lat(lat: float) -> float:
    """Approximate meters per degree of latitude at given latitude."""
    return 111320.0


def _meters_per_degree_lon(lat: float) -> float:
    """Approximate meters per degree of longitude at given latitude."""
    return 111320.0 * math.cos(math.radians(lat))


def compute_transform(geo: GeoInfo) -> Affine:
    """Compute an affine transform from GeoInfo.

    Returns an Affine transform mapping pixel coords to WGS84 (lon, lat),
    including yaw rotation so images align with their flight direction in
    GIS viewers. Assumes a nadir (straight-down) camera whose image "top"
    (row 0) points along the aircraft heading; for mounts where this isn't
    true, pre-rotate the image via ``convert_iiq(rotate=...)`` or adjust
    ``GeoInfo.yaw_deg`` before calling.

    Note:
        Uses a simplified spherical Earth model (111,320 m/degree latitude).
        Longitude scaling uses cos(lat) approximation, accurate to ~0.1%
        below 80 degrees latitude. For higher accuracy, consider a full UTM
        projection.
    """
    from rasterio.transform import Affine

    gsd = geo.gsd  # meters per pixel
    m_per_deg_lat = _meters_per_degree_lat(geo.latitude)
    m_per_deg_lon = _meters_per_degree_lon(geo.latitude)

    # Yaw is clockwise from true north; image "up" (−y) points along the heading.
    theta = math.radians(geo.yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Pixel basis vectors in degrees. +x (right) bears θ+90°, +y (down) bears θ+180°.
    a = cos_t * gsd / m_per_deg_lon
    b = -sin_t * gsd / m_per_deg_lon
    d = -sin_t * gsd / m_per_deg_lat
    e = -cos_t * gsd / m_per_deg_lat

    # Solve so the image center pixel (W/2, H/2) maps to (longitude, latitude).
    cx = geo.image_width / 2.0
    cy = geo.image_height / 2.0
    c = geo.longitude - a * cx - b * cy
    f = geo.latitude - d * cx - e * cy

    return Affine(a, b, c, d, e, f)


def write_world_file(image_path: str | Path, geo: GeoInfo) -> str:
    """Write a world file (.jgw/.pgw/.tfw) plus CRS sidecars for the given image.

    World files only carry the transform — GIS tools need a separate CRS hint.
    Two sidecars are written so both major toolchains recognise WGS84:

      - ``<basename>.prj``: WKT file read by ArcGIS-family tools.
      - ``<basename>.<ext>.aux.xml``: GDAL PAM file read by QGIS (GDAL's JPEG
        driver ignores .prj; without this QGIS reports "Unknown" CRS).

    Args:
        image_path: Path to the image file
        geo: Georeferencing parameters

    Returns:
        Path to the created world file

    Example::

        geo = extract_geo_info(xmp, 80.0, 12768, 9564)
        write_world_file("output.jpg", geo)
        # creates output.jgw, output.prj, output.jpg.aux.xml
    """
    image_path = Path(image_path)
    ext = image_path.suffix.lower()
    world_ext_map = {
        ".jpg": ".jgw",
        ".jpeg": ".jgw",
        ".png": ".pgw",
        ".tif": ".tfw",
        ".tiff": ".tfw",
    }
    world_ext = world_ext_map.get(ext, ".wld")
    world_path = str(image_path.with_suffix(world_ext))

    transform = compute_transform(geo)

    # World file format: 6 lines
    # Line 1: pixel size in x (a)
    # Line 2: rotation about y axis (d)
    # Line 3: rotation about x axis (b)
    # Line 4: pixel size in y (e, negative)
    # Line 5: x coordinate of center of upper-left pixel
    # Line 6: y coordinate of center of upper-left pixel
    with open(world_path, "w") as f:
        f.write(f"{transform.a:.15f}\n")
        f.write(f"{transform.d:.15f}\n")
        f.write(f"{transform.b:.15f}\n")
        f.write(f"{transform.e:.15f}\n")
        f.write(f"{transform.c:.15f}\n")
        f.write(f"{transform.f:.15f}\n")

    from rasterio.crs import CRS

    wkt = CRS.from_epsg(4326).to_wkt()
    image_path.with_suffix(".prj").write_text(wkt)
    aux_path = image_path.with_name(image_path.name + ".aux.xml")
    aux_path.write_text(f"<PAMDataset>\n  <SRS>{wkt}</SRS>\n</PAMDataset>\n")

    return world_path


def write_geotiff(
    image_path: str,
    output_path: str,
    geo: GeoInfo,
    compress: str = "JPEG",
    jpeg_quality: int = 90,
) -> str:
    """Convert an image to GeoTIFF with embedded georeferencing.

    Args:
        image_path: Path to source image (JPEG, PNG, TIFF)
        output_path: Path to output GeoTIFF
        geo: Georeferencing parameters
        compress: Compression method (JPEG, LZW, DEFLATE, NONE)
        jpeg_quality: JPEG quality if compress=JPEG

    Returns:
        Path to the created GeoTIFF
    """
    import rasterio
    from rasterio.crs import CRS

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # BGR to RGB for rasterio (bands-first format)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    transform = compute_transform(geo)
    crs = CRS.from_epsg(4326)  # WGS84

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": w,
        "height": h,
        "count": 3,
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }
    if compress.upper() == "JPEG":
        profile["jpeg_quality"] = jpeg_quality
        profile["photometric"] = "YCBCR"

    with rasterio.open(output_path, "w", **profile) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)

    return output_path


def georeference_image(
    image_path: str,
    xmp: str,
    focal_length_mm: float = 80.0,
    output_geotiff: str | None = None,
) -> str | None:
    """Add georeferencing to an existing image file.

    For JPEG/PNG: creates a sidecar world file (.jgw/.pgw).
    For TIFF: converts to GeoTIFF (in-place or to output_geotiff path).

    Args:
        image_path: Path to the image file
        xmp: XMP XML string containing GPS data
        focal_length_mm: Camera focal length in mm
        output_geotiff: Optional output path for GeoTIFF (TIFF only)

    Returns:
        Path to the created world file or GeoTIFF, or None if no GPS data
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    geo = extract_geo_info(xmp, focal_length_mm, w, h)
    if geo is None:
        return None

    ext = Path(image_path).suffix.lower()

    if ext in (".tif", ".tiff"):
        out = output_geotiff or image_path
        return write_geotiff(image_path, out, geo)
    else:
        return write_world_file(image_path, geo)
