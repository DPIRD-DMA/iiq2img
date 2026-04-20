"""Tests for georeferencing: coordinate parsing, GeoInfo, extract_geo_info, transforms, world files."""

from pathlib import Path

import numpy as np
import pytest

from iiq2img.georef import (
    GeoInfo,
    PIXEL_SIZE_MM,
    _meters_per_degree_lat,
    _meters_per_degree_lon,
    _parse_dms_coordinate,
    _parse_rational,
    compute_transform,
    extract_geo_info,
    write_world_file,
)


# ── Coordinate parsing ──────────────────────────────────────────────────────


class TestParseDmsCoordinate:
    def test_south_latitude(self):
        assert _parse_dms_coordinate("34,33.600456S") == pytest.approx(
            -(34 + 33.600456 / 60), abs=1e-8
        )

    def test_north_latitude(self):
        assert _parse_dms_coordinate("34,33.600456N") == pytest.approx(
            34 + 33.600456 / 60, abs=1e-8
        )

    def test_west_longitude(self):
        assert _parse_dms_coordinate("118,15.5W") == pytest.approx(
            -(118 + 15.5 / 60), abs=1e-8
        )

    def test_east_longitude(self):
        assert _parse_dms_coordinate("118,15.5E") == pytest.approx(
            118 + 15.5 / 60, abs=1e-8
        )

    def test_no_suffix(self):
        """No compass suffix — should return positive value."""
        result = _parse_dms_coordinate("34,33.6")
        assert result == pytest.approx(34 + 33.6 / 60, abs=1e-8)

    def test_zero_minutes(self):
        assert _parse_dms_coordinate("45,0.0N") == pytest.approx(45.0)

    def test_whitespace_stripped(self):
        assert _parse_dms_coordinate("  34,33.6S  ") == pytest.approx(
            -(34 + 33.6 / 60), abs=1e-8
        )

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse coordinate"):
            _parse_dms_coordinate("not a coordinate")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse coordinate"):
            _parse_dms_coordinate("")


class TestParseRational:
    def test_fraction(self):
        assert _parse_rational("151615/1000") == pytest.approx(151.615)

    def test_plain_float(self):
        assert _parse_rational("123.456") == pytest.approx(123.456)

    def test_integer(self):
        assert _parse_rational("42") == pytest.approx(42.0)

    def test_zero_numerator(self):
        assert _parse_rational("0/1000") == pytest.approx(0.0)


# ── GeoInfo dataclass ────────────────────────────────────────────────────────


class TestGeoInfo:
    @pytest.fixture
    def geo(self) -> GeoInfo:
        return GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=0.0,
            image_width=12768,
            image_height=9564,
        )

    def test_gsd(self, geo):
        expected = (120.0 * PIXEL_SIZE_MM) / 80.0
        assert geo.gsd == pytest.approx(expected)

    def test_footprint_width(self, geo):
        assert geo.footprint_width == pytest.approx(geo.gsd * 12768)

    def test_footprint_height(self, geo):
        assert geo.footprint_height == pytest.approx(geo.gsd * 9564)

    def test_gsd_scales_with_altitude(self):
        low = GeoInfo(0, 0, 50, 80, 0, 100, 100)
        high = GeoInfo(0, 0, 200, 80, 0, 100, 100)
        assert high.gsd > low.gsd

    def test_gsd_scales_with_focal_length(self):
        short = GeoInfo(0, 0, 100, 50, 0, 100, 100)
        long = GeoInfo(0, 0, 100, 150, 0, 100, 100)
        assert short.gsd > long.gsd


# ── Meters per degree helpers ────────────────────────────────────────────────


class TestMetersPerDegree:
    def test_lat_is_constant(self):
        assert _meters_per_degree_lat(0) == _meters_per_degree_lat(45)
        assert _meters_per_degree_lat(0) == 111320.0

    def test_lon_at_equator(self):
        assert _meters_per_degree_lon(0) == pytest.approx(111320.0, rel=1e-6)

    def test_lon_decreases_toward_poles(self):
        assert _meters_per_degree_lon(60) < _meters_per_degree_lon(0)

    def test_lon_at_pole(self):
        assert _meters_per_degree_lon(90) == pytest.approx(0.0, abs=1e-6)


# ── extract_geo_info from XMP ────────────────────────────────────────────────


# Minimal XMP matching Phase One iXM format
_SAMPLE_XMP = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:aerialgps="http://www.phaseone.com/aerialgps/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:Camera="http://www.phaseone.com/camera/"
      aerialgps:GPSFlightLatitude="34,33.600456S"
      aerialgps:GPSFlightLongitude="138,41.123456E"
      aerialgps:GPSAltitudeAboveTakeOff="151615/1000"
      Camera:Yaw="2366373/10000"
    />
  </rdf:RDF>
</x:xmpmeta>
"""

_XMP_TRIGGER_GPS = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:Camera="http://www.phaseone.com/camera/"
      exif:GPSLatitude="34,33.600456S"
      exif:GPSLongitude="138,41.123456E"
      Camera:AboveGroundAltitude="120000/1000"
    />
  </rdf:RDF>
</x:xmpmeta>
"""


class TestExtractGeoInfo:
    def test_flight_gps(self):
        geo = extract_geo_info(_SAMPLE_XMP, 80.0, 12768, 9564)
        assert geo is not None
        assert geo.latitude == pytest.approx(-(34 + 33.600456 / 60), abs=1e-6)
        assert geo.longitude == pytest.approx(138 + 41.123456 / 60, abs=1e-6)
        assert geo.altitude_agl == pytest.approx(151.615, abs=1e-3)
        assert geo.yaw_deg == pytest.approx(236.6373, abs=1e-3)
        assert geo.image_width == 12768
        assert geo.image_height == 9564

    def test_trigger_gps_fallback(self):
        geo = extract_geo_info(_XMP_TRIGGER_GPS, 80.0, 12768, 9564)
        assert geo is not None
        assert geo.latitude == pytest.approx(-(34 + 33.600456 / 60), abs=1e-6)
        assert geo.altitude_agl == pytest.approx(120.0, abs=1e-3)

    def test_missing_gps_returns_none(self):
        xmp = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description/></rdf:RDF></x:xmpmeta>'
        assert extract_geo_info(xmp, 80.0, 100, 100) is None

    def test_malformed_xml_returns_none(self):
        assert extract_geo_info("not xml at all!!!", 80.0, 100, 100) is None

    def test_empty_string_returns_none(self):
        assert extract_geo_info("", 80.0, 100, 100) is None

    def test_default_altitude_when_missing(self):
        """If no altitude tag, default to 100m."""
        xmp = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      exif:GPSLatitude="34,33.6S"
      exif:GPSLongitude="138,41.1E"
    />
  </rdf:RDF>
</x:xmpmeta>
"""
        geo = extract_geo_info(xmp, 80.0, 100, 100)
        assert geo is not None
        assert geo.altitude_agl == 100.0

    def test_default_yaw_when_missing(self):
        """If no yaw tag, default to 0.0."""
        xmp = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      exif:GPSLatitude="34,33.6S"
      exif:GPSLongitude="138,41.1E"
    />
  </rdf:RDF>
</x:xmpmeta>
"""
        geo = extract_geo_info(xmp, 80.0, 100, 100)
        assert geo is not None
        assert geo.yaw_deg == 0.0


# ── compute_transform ────────────────────────────────────────────────────────


class TestComputeTransform:
    @pytest.fixture
    def geo(self) -> GeoInfo:
        return GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=0.0,
            image_width=1000,
            image_height=800,
        )

    def test_transform_pixel_size(self, geo):
        t = compute_transform(geo)
        gsd = geo.gsd
        m_per_deg_lon = _meters_per_degree_lon(geo.latitude)
        m_per_deg_lat = _meters_per_degree_lat(geo.latitude)
        assert t.a == pytest.approx(gsd / m_per_deg_lon, rel=1e-6)
        assert t.e == pytest.approx(-gsd / m_per_deg_lat, rel=1e-6)

    def test_no_rotation(self, geo):
        t = compute_transform(geo)
        assert t.b == 0.0
        assert t.d == 0.0

    def test_upper_left_corner(self, geo):
        t = compute_transform(geo)
        # Upper left should be west and north of center
        assert t.c < geo.longitude
        assert t.f > geo.latitude

    def test_high_latitude(self):
        """Near-pole coordinates should not cause errors."""
        geo = GeoInfo(
            latitude=85.0,
            longitude=0.0,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=0.0,
            image_width=1000,
            image_height=800,
        )
        t = compute_transform(geo)
        # Pixel x size should be larger at high latitudes (fewer meters per degree lon)
        equator_geo = GeoInfo(0.0, 0.0, 120.0, 80.0, 0.0, 1000, 800)
        t_eq = compute_transform(equator_geo)
        assert t.a > t_eq.a  # degrees per pixel larger near poles

    def test_equator(self):
        """Equator should work with equal x/y scaling."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 0.0, 100, 100)
        t = compute_transform(geo)
        # At equator, meters_per_degree_lat == meters_per_degree_lon
        assert abs(t.a) == pytest.approx(abs(t.e), rel=1e-6)

    def test_center_pixel_maps_to_geo_center(self, geo):
        t = compute_transform(geo)
        # Apply full affine including rotation terms
        cx = t.c + (geo.image_width / 2.0) * t.a + (geo.image_height / 2.0) * t.b
        cy = t.f + (geo.image_width / 2.0) * t.d + (geo.image_height / 2.0) * t.e
        assert cx == pytest.approx(geo.longitude, abs=1e-6)
        assert cy == pytest.approx(geo.latitude, abs=1e-6)

    def test_yaw_rotates_transform(self):
        """Non-zero yaw should populate the rotation terms b and d."""
        geo = GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=90.0,  # heading east — image "up" points east
            image_width=1000,
            image_height=800,
        )
        t = compute_transform(geo)
        # 90° rotation: the pure-scale terms (a, e) vanish, rotation terms dominate
        assert abs(t.a) < 1e-12
        assert abs(t.e) < 1e-12
        assert abs(t.b) > 0
        assert abs(t.d) > 0

    def test_yaw_preserves_center(self):
        """Image center must still map to (longitude, latitude) under rotation."""
        geo = GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=236.64,
            image_width=1000,
            image_height=800,
        )
        t = compute_transform(geo)
        cx = t.c + (geo.image_width / 2.0) * t.a + (geo.image_height / 2.0) * t.b
        cy = t.f + (geo.image_width / 2.0) * t.d + (geo.image_height / 2.0) * t.e
        assert cx == pytest.approx(geo.longitude, abs=1e-9)
        assert cy == pytest.approx(geo.latitude, abs=1e-9)

    def test_yaw_zero_matches_axis_aligned(self):
        """Yaw=0 should produce the same transform as the previous axis-aligned form."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 0.0, 100, 100)
        t = compute_transform(geo)
        assert t.b == pytest.approx(0.0, abs=1e-15)
        assert t.d == pytest.approx(0.0, abs=1e-15)
        assert t.a > 0
        assert t.e < 0

    def test_yaw_zero_points_top_north(self):
        """Yaw=0 (north heading): top-center pixel should map north of center at same longitude."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 0.0, 100, 100)
        t = compute_transform(geo)
        # top-center pixel: (col=W/2, row=0)
        lon = t.a * (geo.image_width / 2) + t.c
        lat = t.d * (geo.image_width / 2) + t.f
        assert lon == pytest.approx(geo.longitude, abs=1e-12)
        assert lat > geo.latitude  # north of center

    def test_yaw_90_points_top_east(self):
        """Yaw=90 (east heading): top-center pixel should map east of center at same latitude."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 90.0, 100, 100)
        t = compute_transform(geo)
        lon = t.a * (geo.image_width / 2) + t.c
        lat = t.d * (geo.image_width / 2) + t.f
        assert lon > geo.longitude  # east of center
        assert lat == pytest.approx(geo.latitude, abs=1e-12)

    def test_yaw_180_points_top_south(self):
        """Yaw=180 (south heading): top-center pixel should map south of center."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 180.0, 100, 100)
        t = compute_transform(geo)
        lat = t.d * (geo.image_width / 2) + t.f
        assert lat < geo.latitude  # south of center

    def test_yaw_270_points_top_west(self):
        """Yaw=270 (west heading): top-center pixel should map west of center."""
        geo = GeoInfo(0.0, 0.0, 100.0, 80.0, 270.0, 100, 100)
        t = compute_transform(geo)
        lon = t.a * (geo.image_width / 2) + t.c
        assert lon < geo.longitude  # west of center


# ── write_world_file ─────────────────────────────────────────────────────────


class TestWriteWorldFile:
    @pytest.fixture
    def geo(self) -> GeoInfo:
        return GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=0.0,
            image_width=1000,
            image_height=800,
        )

    def test_creates_jgw_for_jpeg(self, tmp_path, geo):
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        assert result.endswith(".jgw")
        assert Path(result).exists()

    def test_creates_pgw_for_png(self, tmp_path, geo):
        img_path = tmp_path / "photo.png"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        assert result.endswith(".pgw")

    def test_creates_tfw_for_tiff(self, tmp_path, geo):
        img_path = tmp_path / "photo.tif"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        assert result.endswith(".tfw")

    def test_unknown_ext_uses_wld(self, tmp_path, geo):
        img_path = tmp_path / "photo.bmp"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        assert result.endswith(".wld")

    def test_world_file_has_six_lines(self, tmp_path, geo):
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        lines = Path(result).read_text().strip().splitlines()
        assert len(lines) == 6

    def test_world_file_values_are_floats(self, tmp_path, geo):
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        lines = Path(result).read_text().strip().splitlines()
        values = [float(line) for line in lines]
        # Line 1: pixel x size (positive)
        assert values[0] > 0
        # Line 2,3: rotation (zero for nadir)
        assert values[1] == 0.0
        assert values[2] == 0.0
        # Line 4: pixel y size (negative)
        assert values[3] < 0

    def test_world_file_precision(self, tmp_path, geo):
        """World file values should have 15 decimal places."""
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        lines = Path(result).read_text().strip().splitlines()
        for line in lines:
            # Each line should have a decimal point with digits after it
            parts = line.split(".")
            assert len(parts) == 2, f"Expected decimal point in: {line}"
            assert len(parts[1]) == 15, f"Expected 15 decimal places in: {line}"

    def test_world_file_upper_left_matches_transform(self, tmp_path, geo):
        """Lines 5-6 of world file should match transform upper-left corner."""
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        result = write_world_file(str(img_path), geo)
        lines = Path(result).read_text().strip().splitlines()
        values = [float(line) for line in lines]
        t = compute_transform(geo)
        assert values[4] == pytest.approx(t.c, rel=1e-10)
        assert values[5] == pytest.approx(t.f, rel=1e-10)

    def test_writes_prj_sidecar_with_wgs84_wkt(self, tmp_path, geo):
        """A .prj sidecar must accompany the world file so GIS tools can resolve the CRS."""
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        write_world_file(str(img_path), geo)
        prj_path = img_path.with_suffix(".prj")
        assert prj_path.exists()
        wkt = prj_path.read_text()
        # WGS84 WKT identifies the datum regardless of WKT1/WKT2 style
        assert "WGS" in wkt and "84" in wkt

    def test_writes_aux_xml_pam_sidecar(self, tmp_path, geo):
        """GDAL's JPEG driver reads CRS from <name>.<ext>.aux.xml, not .prj."""
        img_path = tmp_path / "photo.jpg"
        img_path.touch()
        write_world_file(str(img_path), geo)
        aux_path = img_path.with_name(img_path.name + ".aux.xml")
        assert aux_path.exists()
        content = aux_path.read_text()
        assert "<PAMDataset>" in content
        assert "<SRS>" in content
        assert "WGS" in content and "84" in content

    def test_rasterio_reads_crs_from_jpeg(self, tmp_path, geo):
        """End-to-end: GDAL/rasterio should resolve the CRS from the sidecars."""
        import cv2
        import rasterio

        img_path = tmp_path / "photo.jpg"
        cv2.imwrite(
            str(img_path),
            np.zeros((geo.image_height, geo.image_width, 3), dtype=np.uint8),
        )
        write_world_file(str(img_path), geo)

        with rasterio.open(str(img_path)) as ds:
            assert ds.crs is not None
            assert ds.crs.to_epsg() == 4326


# ── write_geotiff (requires rasterio) ────────────────────────────────────────


class TestWriteGeotiff:
    @pytest.fixture
    def geo(self) -> GeoInfo:
        return GeoInfo(
            latitude=-34.56,
            longitude=138.72,
            altitude_agl=120.0,
            focal_length_mm=80.0,
            yaw_deg=0.0,
            image_width=100,
            image_height=80,
        )

    def test_creates_geotiff(self, tmp_path, geo):
        import cv2

        img_path = str(tmp_path / "input.jpg")
        bgr = np.zeros((80, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, bgr)

        from iiq2img.georef import write_geotiff

        out = str(tmp_path / "out.tif")
        result = write_geotiff(img_path, out, geo)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_invalid_image_raises(self, tmp_path, geo):
        from iiq2img.georef import write_geotiff

        with pytest.raises(ValueError, match="Cannot read image"):
            write_geotiff(str(tmp_path / "nope.jpg"), str(tmp_path / "out.tif"), geo)


# ── georeference_image ───────────────────────────────────────────────────────


class TestGeoreferenceImage:
    def test_jpeg_creates_world_file(self, tmp_path):
        import cv2

        from iiq2img.georef import georeference_image

        img_path = str(tmp_path / "photo.jpg")
        cv2.imwrite(img_path, np.zeros((80, 100, 3), dtype=np.uint8))

        result = georeference_image(img_path, _SAMPLE_XMP, focal_length_mm=80.0)
        assert result is not None
        assert result.endswith(".jgw")
        assert Path(result).exists()

    def test_png_creates_world_file(self, tmp_path):
        import cv2

        from iiq2img.georef import georeference_image

        img_path = str(tmp_path / "photo.png")
        cv2.imwrite(img_path, np.zeros((80, 100, 3), dtype=np.uint8))

        result = georeference_image(img_path, _SAMPLE_XMP, focal_length_mm=80.0)
        assert result is not None
        assert result.endswith(".pgw")

    def test_no_gps_returns_none(self, tmp_path):
        import cv2

        from iiq2img.georef import georeference_image

        img_path = str(tmp_path / "photo.jpg")
        cv2.imwrite(img_path, np.zeros((80, 100, 3), dtype=np.uint8))

        result = georeference_image(img_path, "<x:xmpmeta/>")
        assert result is None

    def test_invalid_image_raises(self, tmp_path):
        from iiq2img.georef import georeference_image

        with pytest.raises(ValueError, match="Cannot read image"):
            georeference_image(str(tmp_path / "nope.jpg"), _SAMPLE_XMP)

    def test_tiff_creates_geotiff(self, tmp_path):
        import cv2

        from iiq2img.georef import georeference_image

        img_path = str(tmp_path / "photo.tif")
        cv2.imwrite(img_path, np.zeros((80, 100, 3), dtype=np.uint8))

        result = georeference_image(img_path, _SAMPLE_XMP, focal_length_mm=80.0)
        assert result is not None
        assert Path(result).exists()
