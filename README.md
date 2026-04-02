# iiq2img

Fast converter for **Phase One iXM-GS120** (120 MP) IIQ raw files. Outputs JPEG, PNG, or TIFF with full EXIF/GPS/XMP metadata and optional georeferencing.

## Quick start

```bash
# Install from git
uv pip install git+https://github.com/nickponline/iiq2img.git

# Or clone and install locally
git clone https://github.com/nickponline/iiq2img.git && cd iiq2img
uv sync

# Convert
iiq2img photo.IIQ --fast              # single file
iiq2img batch ./raw ./out --fast      # batch directory
```

## Pipelines

| Pipeline | End-to-end | Speedup | Quality |
|----------|-----------|---------|---------|
| `--fast` | ~1.0 s | 3.6x | Mean diff ~7/255 vs reference |
| default (LibRaw PPG) | ~3.3 s | baseline | Reference |

The fast pipeline uses cv2 edge-aware demosaic + numba parallel LUTs with BT.709 gamma, matched to LibRaw's output. See [FINDINGS.md](FINDINGS.md) for how.

## Python API

```python
from iiq2img import convert_iiq, batch_convert, Pipeline

# Single file
result = convert_iiq("photo.IIQ", "output.jpg", pipeline=Pipeline.FAST)

# Batch (parallel)
results = batch_convert("./raw", "./out", pipeline=Pipeline.FAST, workers=8)

# Georeferenced output
result = convert_iiq("photo.IIQ", "output.tif", output_format="tiff", georef=True)
```

`convert_iiq` options: `output_format` (jpg/png/tiff), `compress_quality` (1-100), `max_dimension`, `rotate_180`, `georef`, `extract_meta`.

## Development

```bash
uv run pytest                        # 108 tests
uv run iiq2img benchmark photo.IIQ   # timing comparison
```
