# Changelog

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
