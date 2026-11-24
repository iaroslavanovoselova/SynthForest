"""Validate externally prepared benchmark data layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.paths import external_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Prepared dataset root; defaults to $SYNTHFOREST_DATA_ROOT/CHM_rasters_50cm/sao_paulo/benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = (
        args.root
        or external_root("SYNTHFOREST_DATA_ROOT") / "CHM_rasters_50cm" / "sao_paulo" / "benchmark"
    )
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"São Paulo benchmark directory does not exist: {root}")
    rasters = sorted((*root.glob("*.tif"), *root.glob("*.tiff")))
    if not rasters:
        raise SystemExit(f"No GeoTIFF rasters found under {root}")
    print(f"São Paulo benchmark layout: {root} ({len(rasters)} GeoTIFF files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
