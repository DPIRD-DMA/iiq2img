# iiq2img

Fast converter for **Phase One iXM-GS120** (120 MP) IIQ raw files. Outputs JPEG, PNG, or TIFF with full EXIF/GPS/XMP metadata and optional georeferencing.

## Quick start

```bash
# Install from PyPI
pip install iiq2img

# Or install the latest dev version from git
uv pip install git+https://github.com/DPIRD-DMA/iiq2img.git

# Or clone and install locally
git clone https://github.com/DPIRD-DMA/iiq2img.git && cd iiq2img
uv sync

# Convert (uses fast pipeline by default)
iiq2img photo.IIQ                     # single file
iiq2img batch ./raw ./out             # batch directory
iiq2img photo.IIQ --libraw            # use LibRaw PPG instead
```

## Pipelines

| Pipeline | End-to-end | Speedup | Quality |
|----------|-----------|---------|---------|
| default (fast) | ~1.0 s | 3.6x | Mean diff ~7/255 vs reference |
| `--libraw` (LibRaw PPG) | ~3.3 s | baseline | Reference |

The fast pipeline uses cv2 edge-aware demosaic + numba parallel LUTs with BT.709 gamma, matched to LibRaw's output. See [FINDINGS.md](https://github.com/DPIRD-DMA/iiq2img/blob/main/docs/FINDINGS.md) for how.

## Python API

```python
from iiq2img import convert_iiq, read_iiq, batch_convert

# Single file (fast pipeline is the default)
out_path = convert_iiq("photo.IIQ", "output.jpg")

# Read as NumPy array (no file written)
rgb = read_iiq("photo.IIQ")

# Batch (parallel)
results = batch_convert("./raw", "./out", workers=8)

# Georeferenced output (requires `pip install iiq2img[geo]`)
# TIFF: CRS embedded in the GeoTIFF itself.
# JPEG/PNG: sidecars written next to the image (.jgw/.pgw + .prj + .aux.xml)
# so QGIS/ArcGIS recognise the projection.
convert_iiq("photo.IIQ", "output.tif", output_format="tiff", georef=True)
convert_iiq("photo.IIQ", "output.jpg", georef=True)
```

| Option | Default | Description |
|--------|---------|-------------|
| `thumbnail` | `False` | Extract embedded JPEG thumbnail instead of converting |
| `output_format` | `"jpg"` | Output format: `jpg`, `png`, or `tiff` |
| `compress_quality` | `90` | JPEG/PNG compression quality (1-100) |
| `max_dimension` | `None` | Downscale longest edge to this size |
| `rotate` | `0` | Rotate output: `0`, `90`, `180`, `270` |
| `georef` | `False` | World file + `.prj` + `.aux.xml` sidecars for JPEG/PNG, embedded CRS for TIFF (requires `[geo]` extra) |
| `extract_meta` | `True` | Copy EXIF/GPS/XMP metadata to output |
| `pipeline` | `"fast"` | Demosaic pipeline: `fast` or `libraw` |

## CLI

```bash
# Single file (outputs photo.jpg next to the original)
iiq2img photo.IIQ
iiq2img photo.IIQ --libraw            # use LibRaw PPG instead of fast

# Batch convert a directory (parallel)
iiq2img batch ./raw ./out                           # defaults: jpg, q=90, fast
iiq2img batch ./raw ./out --format tiff --workers 8 # tiff, q=90, 8 workers
iiq2img batch ./raw ./out --format jpg --quality 75 --workers 4 --libraw
iiq2img batch ./raw ./out --georef                  # write georef sidecars for every output
iiq2img batch ./raw ./out --rotate 180 --no-meta    # 180° flip, skip EXIF copy

# Benchmark all pipelines on a sample file
iiq2img benchmark photo.IIQ
```

## Changelog

See [CHANGELOG.md](https://github.com/DPIRD-DMA/iiq2img/blob/main/CHANGELOG.md) for version history.

## Contributing

```bash
git clone https://github.com/DPIRD-DMA/iiq2img.git && cd iiq2img
uv sync --dev
pre-commit install
```

Pre-commit hooks run **ruff** (lint + format) and **pytest** on every commit. To run them manually:

```bash
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run pytest                        # tests
uv run iiq2img benchmark photo.IIQ   # timing comparison
```
