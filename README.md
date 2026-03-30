# iiq2img

Fast IIQ-to-image converter for **Phase One iXM-GS120** (120 MP) raw files.

Uses [rawpy](https://github.com/letmaik/rawpy) (LibRaw) + OpenCV for demosaicing and encoding, bypassing the slow Phase One SDK pipeline. Retains all EXIF, GPS, and XMP metadata from the original IIQ file.

## Performance

| Approach | Per-file | Effective throughput |
|---|---|---|
| Embedded thumbnail | ~1 ms | - |
| Full-res JPEG (single) | ~3.4 s | 0.3 img/s |
| Full-res JPEG (8 workers) | ~880 ms | 1.1 img/s |

*Benchmarked on 120 MP IIQ files (~115 MB each), 32-core CPU.*

### Optimizations

- PPG demosaic algorithm (fastest for this sensor)
- Direct EXIF serialization (no dummy JPEG roundtrip)
- Multiprocessing batch with spawn context for safe parallelism

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### CLI

```bash
# Convert a single file
iiq2img photo.IIQ

# Batch convert a directory (parallel)
iiq2img batch ./input_dir ./output_dir jpg 90 8
#                                      fmt qual workers

# Run benchmark
iiq2img benchmark photo.IIQ
```

### As a library

```python
from iiq2img import convert_iiq, batch_convert, Quality, OutputFormat

# Full-resolution JPEG
result = convert_iiq("photo.IIQ", "output.jpg")

# Thumbnail (~1 ms)
result = convert_iiq("photo.IIQ", "thumb.jpg", quality=Quality.THUMBNAIL)

# PNG with max dimension constraint
result = convert_iiq(
    "photo.IIQ",
    "output.png",
    output_format=OutputFormat.PNG,
    max_dimension=4000,
)

# Batch convert (parallel, 8 workers)
results = batch_convert(
    "./raw_images",
    "./converted",
    output_format=OutputFormat.JPEG,
    compress_quality=90,
    workers=8,
)
```

### Output formats

JPEG, PNG, and TIFF. Metadata (EXIF/XMP) is embedded in all formats.

## Development

```bash
# Run tests
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy iiq2img/
```
