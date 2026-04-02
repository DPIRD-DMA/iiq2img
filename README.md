# iiq2img

Fast IIQ-to-image converter for **Phase One iXM-GS120** (120 MP) raw files.

Converts 120 MP IIQ files (~115 MB) to JPEG, PNG, or TIFF with full EXIF/GPS/XMP metadata preservation and optional georeferencing (world files, GeoTIFF).

## Pipelines

Two demosaic pipelines are available:

| Pipeline | Algorithm | Demosaic | End-to-end (JPEG) | Quality vs LibRaw |
|----------|-----------|----------|-------------------|-------------------|
| **LIBRAW** (default) | LibRaw PPG | ~2.9 s | ~3.3 s | Reference |
| **FAST** | cv2 edge-aware + numba LUTs | ~600 ms (4.6x) | ~1.0 s (3.4x) | Mean diff 6.5/255, median 3/255 |

The fast pipeline matches LibRaw's output by using BT.709 gamma, full-range black level scaling, and empirically tuned auto-brightness. See [FINDINGS.md](FINDINGS.md) for the research behind it.

## Performance

| Approach | Per-file (JPEG) | Throughput (8 workers) |
|----------|-----------------|----------------------|
| Embedded thumbnail | ~1 ms | - |
| Full-res LIBRAW | ~3.3 s | ~1.1 img/s |
| Full-res FAST | ~1.0 s | ~3+ img/s |

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### CLI

```bash
# Convert a single file (LibRaw pipeline)
uv run iiq2img photo.IIQ

# Convert with the fast pipeline
uv run iiq2img photo.IIQ --fast

# Batch convert a directory (parallel)
uv run iiq2img batch ./input_dir ./output_dir jpg 90 8 --fast
#                                              fmt qual workers

# Run benchmark
uv run iiq2img benchmark photo.IIQ
```

### As a library

```python
from iiq2img import convert_iiq, batch_convert, Pipeline, Quality

# Full-resolution JPEG (LibRaw, default)
result = convert_iiq("photo.IIQ", "output.jpg")

# Fast pipeline (~3x faster end-to-end)
result = convert_iiq("photo.IIQ", "output.jpg", pipeline=Pipeline.FAST)

# Thumbnail (~1 ms)
result = convert_iiq("photo.IIQ", "thumb.jpg", quality=Quality.THUMBNAIL)

# PNG with max dimension constraint
result = convert_iiq("photo.IIQ", "output.png", output_format="png", max_dimension=4000)

# Batch convert (parallel, fast pipeline)
results = batch_convert(
    "./raw_images",
    "./converted",
    output_format="jpg",
    compress_quality=90,
    workers=8,
    pipeline=Pipeline.FAST,
)
```

### Georeferencing

```python
# JPEG/PNG: creates sidecar world file (.jgw/.pgw)
result = convert_iiq("photo.IIQ", "output.jpg", georef=True)

# TIFF: writes GeoTIFF with embedded CRS
result = convert_iiq("photo.IIQ", "output.tif", output_format="tiff", georef=True)
```

### Output formats

JPEG, PNG, and TIFF. Metadata (EXIF/XMP) is embedded in all formats.

## Development

```bash
# Run tests
uv run pytest

# Run fast-pipeline quality comparison
uv run python fast_demosaic_research.py
```
