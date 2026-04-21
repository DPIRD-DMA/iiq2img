# Changelog

## [0.7.0] — 2026-04-21

### Added

- `geotiff_compress` parameter on `convert_iiq()` and `batch_convert()` — choose between `"jpeg"` (default, lossy/small), `"lzw"` or `"deflate"` (lossless), or `"none"`. JPEG keeps `photometric=YCBCR` + configurable `jpeg_quality`; the lossless modes skip those.
- CLI `--geotiff-compress {jpeg,lzw,deflate,none}` flag on `iiq2img batch`.
- `UserWarning` when `geotiff_compress` is set to a non-default value on an output that isn't actually a GeoTIFF (non-TIFF format, or TIFF without `georef=True`), so silent mis-targeting surfaces instead of being quietly ignored.

## [0.6.0] — 2026-04-20

### Added

- `.prj` and `<image>.<ext>.aux.xml` sidecars written alongside world files for JPEG/PNG outputs, so QGIS/ArcGIS resolve the CRS instead of reporting "Unknown". GDAL's JPEG driver ignores `.prj`, so the PAM `.aux.xml` is what actually makes QGIS recognise the projection.
- `batch_convert()` now accepts `extract_meta`, `georef`, and `rotate`, passing each through to `convert_iiq()`.
- CLI `batch` gains `--georef`, `--rotate {0,90,180,270}`, and `--no-meta` flags.

### Changed

- `compute_transform()` now applies the XMP yaw to the affine transform, so georeferenced images display aligned to the flight direction in QGIS (adjacent strip images line up naturally). Assumes image top aligns with aircraft heading — use `rotate` to correct for different camera mounts.
- Georef docstrings updated to reflect that world files do carry rotation.

## [0.5.0] — 2026-04-06

### Performance

- Fused BGR channel swap into numba gamma LUT (`_fast_lut3_bgr`), eliminating a full-image `cv2.cvtColor` pass (~55ms saved on 120MP images).
- Replaced numpy strided row-sum detection with numba-parallel `_row_means_all_channels`, ~4x faster defective-row detection.

### Changed

- `demosaic_fast()` accepts `bgr` flag to output BGR directly for encoding.
- `convert_iiq` skips the RGB→BGR conversion when the fast pipeline already produces BGR output (GeoTIFF path still gets RGB).

## [0.4.0] — 2026-04-06

### Added

- `verbose` parameter on `convert_iiq()` — prints per-step timing (demosaic, metadata, encode, total).
- Structured logging (`logging.debug`) in converter, pipeline, and repair modules.
- Docstring examples for `repair_defective_rows`, `detect_defective_rows`, `extract_geo_info`, and `write_world_file`.
- GitHub Actions CI workflow (ruff, mypy, pytest on push/PR).

### Changed

- CLI rewritten with `argparse` — batch args are now named flags (`--format`, `--quality`, `--workers`) instead of positional.
- Extracted `_validate_iiq_path()` helper to deduplicate path validation in `read_iiq` and `convert_iiq`.
- Moved `FINDINGS.md` to `docs/FINDINGS.md`.
- Documented georef limitations (spherical Earth model, nadir assumption) in `compute_transform` and `extract_geo_info` docstrings.

## [0.3.0] — 2026-04-05

### Added

- Defective row detection and repair for raw Bayer data (`repair_defective_rows`, `detect_defective_rows`).
  Fixes horizontal red/blue stripe artefacts caused by corrupted sensor rows (~3 per image on the iXM-GS120).
- Repair runs automatically in both fast and libraw pipelines before demosaicing.
  Can be disabled with `demosaic_fast(path, repair=False)`.
- New `iiq2img.repair` module with numba-parallel repair kernel.
- 18 new tests covering detection, repair, edge cases, and real-IIQ integration.

### Performance

- Repair overhead is ~90ms on top of a ~530ms pipeline (~17%).
- Detection uses vectorised stride-tricks rolling median and int32 row sums for speed.

## [0.2.0] — 2026-04-05

### Breaking changes

- Removed `ConvertResult` dataclass — `convert_iiq` now returns a `Path` directly.
- Removed `Quality` and `Pipeline` enums — use plain strings (`"fast"`, `"libraw"`) instead.
- Replaced `quality=Quality.THUMBNAIL` with `thumbnail=True`.
- Replaced `rotate_180=True` with `rotate=` accepting 0, 90, 180, 270.
- Default pipeline changed from `libraw` to `fast`.
- CLI flag changed from `--fast` to `--libraw` (fast is now the default).

### Added

- `read_iiq()` — read an IIQ file directly into a NumPy RGB array without writing to disk.
- Arbitrary rotation support (0, 90, 180, 270 degrees) for different camera mounts.
- CLI reference section in README.
- CHANGELOG.md.
- Performance regression test (fast pipeline must be ≥2x faster than libraw).
- Expanded test suite: 223 tests covering combined parameter interactions, benchmark, multiprocessing batch, world file precision, and high-latitude georeferencing.

### Changed

- Simplified public API surface — fewer exports, plain strings over enums.
- `batch_convert` returns `list[Path]` instead of `list[ConvertResult]`.
- Benchmark output simplified (no per-image resolution column).
- Updated README: new Quick Start examples, pipeline table, and Python API reflect the new defaults.

## [0.1.0] — 2025-04-04

Initial release.

### Added

- `convert_iiq()` for single-file IIQ → JPEG/PNG/TIFF conversion.
- `batch_convert()` for parallel directory conversion with progress bar.
- `run_benchmark()` for pipeline timing comparison.
- Two demosaic pipelines: LibRaw PPG (accurate) and fast (cv2 EA + numba LUTs, ~3x faster).
- Full EXIF/GPS/XMP metadata extraction and transfer.
- Georeferencing support: world files for JPEG/PNG, GeoTIFF for TIFF.
- Embedded thumbnail extraction (640x480).
- CLI entry point (`iiq2img`) with single-file, batch, and benchmark modes.
- Package structure with `pyproject.toml` and `uv` support.
