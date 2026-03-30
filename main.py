"""CLI entry point for SKW_RAW IIQ converter."""

import sys

from iiq2img import OutputFormat, batch_convert, convert_iiq
from iiq2img.converter import run_benchmark

if __name__ == "__main__":
    sample = "PhaseOneSample/P0286625.IIQ"

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        path = sys.argv[2] if len(sys.argv) > 2 else sample
        run_benchmark(path)

    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "PhaseOneSample"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/iiq_output"
        fmt_str = sys.argv[4] if len(sys.argv) > 4 else "jpg"
        cq = int(sys.argv[5]) if len(sys.argv) > 5 else 90
        workers = int(sys.argv[6]) if len(sys.argv) > 6 else None
        batch_convert(
            input_dir,
            output_dir,
            output_format=OutputFormat(fmt_str),
            compress_quality=cq,
            workers=workers,
        )

    elif len(sys.argv) > 1:
        result = convert_iiq(sys.argv[1])
        print(f"Output: {result.output_path}")
        print(f"Resolution: {result.width}x{result.height}")
        print(f"Time: {result.elapsed_ms:.0f}ms")
        print(f"Size: {result.file_size_bytes / 1024 / 1024:.1f}MB")

    else:
        print("Usage:")
        print(f"  {sys.argv[0]} benchmark [iiq_path]")
        print(
            f"  {sys.argv[0]} batch [in_dir] [out_dir] [jpg|png|tiff] [quality] [workers]"
        )
        print(f"  {sys.argv[0]} <file.IIQ>")
        print()
        print("Running benchmark on sample file...")
        print()
        run_benchmark(sample)
