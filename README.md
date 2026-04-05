# iiq2img

Fast converter for **Phase One iXM-GS120** (120 MP) IIQ raw files. Outputs JPEG, PNG, or TIFF with full EXIF/GPS/XMP metadata and optional georeferencing.

## Quick start

```bash
# Install from git
uv pip install git+https://github.com/nickponline/iiq2img.git

# Or clone and install locally
git clone https://github.com/nickponline/iiq2img.git && cd iiq2img
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

The fast pipeline uses cv2 edge-aware demosaic + numba parallel LUTs with BT.709 gamma, matched to LibRaw's output. See [FINDINGS.md](FINDINGS.md) for how.

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
out_path = convert_iiq("photo.IIQ", "output.tif", output_format="tiff", georef=True)
```

`convert_iiq` options: `output_format` (jpg/png/tiff), `compress_quality` (1-100), `max_dimension`, `rotate` (0/90/180/270), `georef`, `extract_meta`, `pipeline`.

## CLI

```bash
# Single file (outputs photo.jpg next to the original)
iiq2img photo.IIQ
iiq2img photo.IIQ --libraw            # use LibRaw PPG instead of fast

# Batch convert a directory (parallel)
iiq2img batch ./raw ./out                           # defaults: jpg, q=90, fast
iiq2img batch ./raw ./out tiff 90 8                 # tiff, q=90, 8 workers
iiq2img batch ./raw ./out jpg 75 4 --libraw         # jpg q=75, 4 workers, libraw

# Benchmark all pipelines on a sample file
iiq2img benchmark photo.IIQ
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Development

```bash
uv run pytest                        # run tests
uv run iiq2img benchmark photo.IIQ   # timing comparison
```
