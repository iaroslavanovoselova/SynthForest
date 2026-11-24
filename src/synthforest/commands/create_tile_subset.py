#!/usr/bin/env python3
"""Sample unlabeled real tiles as appearance-adaptation references.

Copies a random subset of RGB tiles from a ``dataset/location/tiles/split``
layout (CanopyRS crown benchmarks or tiled height benchmarks) while preserving
that structure. The defaults reproduce the crown references of the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

from synthforest.paths import data_path

try:
    import numpy as np
    import rasterio
except ImportError as exc:  # pragma: no cover - depends on local runtime.
    raise SystemExit(
        'This command needs rasterio. Install the geospatial extra: pip install -e ".[geospatial]"'
    ) from exc


LOGGER = logging.getLogger(__name__)

DEFAULT_SRC_ROOT = data_path("canopyrs")
DEFAULT_DST_ROOT = data_path("domain_adaptation/crowns")
DEFAULT_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect tile paths from dataset/location/tiles/split, keep tiles with "
            "a minimum non-black valid-pixel ratio, and copy a random subset while "
            "preserving the original relative paths."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help=f"Original dataset root. Default: {DEFAULT_SRC_ROOT}",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help=f"Subset output root. Default: {DEFAULT_DST_ROOT}",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Tile split to sample from. Default: train",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of valid tiles to copy. Default: 2000",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of non-black pixels required. Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling. Default: 42",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Only sample from these first-level dataset folders. Default: all.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions to include. Default: .tif .tiff .png .jpg .jpeg",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Copy all valid tiles if fewer than --sample-size are available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty destination and overwrite selected files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would be copied without writing files.",
    )
    parser.add_argument(
        "--manifest-name",
        default="subset_manifest.json",
        help="Manifest filename written in --dst-root. Default: subset_manifest.json",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=250,
        help="Log progress every N scanned tiles. Default: 250",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_extensions(extensions: list[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def _collect_tile_paths(
    src_root: Path,
    split: str,
    extensions: set[str],
    datasets: list[str] | None = None,
) -> list[Path]:
    split_dirs = sorted(
        path
        for path in src_root.glob(f"*/*/tiles/{split}")
        if path.is_dir() and (not datasets or path.parents[2].name in datasets)
    )
    paths: list[Path] = []

    for split_dir in split_dirs:
        for path in sorted(split_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                paths.append(path)

    return paths


def _array_valid_mask(array: np.ndarray) -> np.ndarray:
    """Return pixels that are not black, respecting alpha when it exists."""
    if array.ndim == 2:
        return array != 0

    if array.ndim != 3:
        raise ValueError(f"Unsupported raster array shape: {array.shape}")

    if array.shape[0] <= 4:
        color_bands = array[: min(3, array.shape[0]), :, :]
        valid = np.any(color_bands != 0, axis=0)
        if array.shape[0] == 4:
            valid &= array[3, :, :] != 0
        return valid

    color_bands = array[:, :, : min(3, array.shape[-1])]
    valid = np.any(color_bands != 0, axis=-1)
    if array.shape[-1] == 4:
        valid &= array[:, :, 3] != 0
    return valid


def _valid_pixel_ratio(path: Path) -> float:
    with rasterio.open(path) as dataset:
        array = dataset.read()
    if array.shape[0] == 1:
        array = array[0]
    valid = _array_valid_mask(np.asarray(array))
    return float(np.count_nonzero(valid) / valid.size)


def _filter_valid_tiles(
    paths: list[Path],
    min_valid_ratio: float,
    progress_interval: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for index, path in enumerate(paths, start=1):
        try:
            ratio = _valid_pixel_ratio(path)
        except Exception as exc:  # Keep scanning; report bad files in the manifest.
            LOGGER.warning("Skipping unreadable tile %s: %s", path, exc)
            rejected_records.append(
                {
                    "path": str(path),
                    "valid_ratio": None,
                    "reason": f"unreadable: {exc}",
                }
            )
            continue

        record = {"path": str(path), "valid_ratio": ratio}
        if ratio > min_valid_ratio:
            valid_records.append(record)
        else:
            rejected_records.append({**record, "reason": "below_min_valid_ratio"})

        if progress_interval > 0 and index % progress_interval == 0:
            LOGGER.info(
                "Scanned %s/%s tiles: %s valid, %s rejected",
                index,
                len(paths),
                len(valid_records),
                len(rejected_records),
            )

    return valid_records, rejected_records


def _select_tiles(
    valid_records: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    allow_fewer: bool,
) -> list[dict[str, Any]]:
    if len(valid_records) < sample_size and not allow_fewer:
        raise ValueError(
            f"Requested {sample_size} tiles, but only {len(valid_records)} pass the filter. "
            "Use --allow-fewer to copy all available valid tiles, lower --min-valid-ratio, "
            "or include more source data."
        )

    rng = random.Random(seed)
    if len(valid_records) <= sample_size:
        selected = list(valid_records)
        rng.shuffle(selected)
        return selected

    return rng.sample(valid_records, sample_size)


def _copy_selected_tiles(selected: list[dict[str, Any]], src_root: Path, dst_root: Path) -> None:
    for record in selected:
        src_path = Path(record["path"])
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _write_manifest(
    *,
    manifest_path: Path,
    src_root: Path,
    dst_root: Path,
    split: str,
    sample_size: int,
    min_valid_ratio: float,
    seed: int,
    candidates: list[Path],
    valid_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    selected_manifest = []
    for record in selected:
        src_path = Path(record["path"])
        rel_path = src_path.relative_to(src_root)
        selected_manifest.append(
            {
                "relative_path": rel_path.as_posix(),
                "source_path": str(src_path),
                "destination_path": str(dst_root / rel_path),
                "valid_ratio": record["valid_ratio"],
            }
        )

    manifest = {
        "source_root": str(src_root),
        "destination_root": str(dst_root),
        "split": split,
        "sample_size_requested": sample_size,
        "sample_size_copied": len(selected),
        "min_valid_ratio": min_valid_ratio,
        "seed": seed,
        "candidate_count": len(candidates),
        "valid_count": len(valid_records),
        "rejected_count": len(rejected_records),
        "selected_tiles": selected_manifest,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _configure_logging(args.verbose)

    src_root = _resolve_path(args.src_root)
    dst_root = _resolve_path(args.dst_root)
    extensions = _normalize_extensions(args.extensions)

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be between 0 and 1")
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {src_root}")
    if src_root == dst_root or _is_relative_to(dst_root, src_root):
        raise ValueError(
            "Destination root must be outside the source root so future scans do not include the subset."
        )
    if dst_root.exists() and any(dst_root.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Destination root is not empty: {dst_root}. Use --overwrite or choose a new --dst-root."
        )

    LOGGER.info("Collecting %s tile paths from %s", args.split, src_root)
    candidates = _collect_tile_paths(src_root, args.split, extensions, args.datasets)
    LOGGER.info("Collected %s candidate image tiles", len(candidates))

    if not candidates:
        raise ValueError(f"No image tiles found under */*/tiles/{args.split}")

    valid_records, rejected_records = _filter_valid_tiles(
        paths=candidates,
        min_valid_ratio=args.min_valid_ratio,
        progress_interval=args.progress_interval,
    )
    LOGGER.info(
        "Finished filtering: %s valid, %s rejected",
        len(valid_records),
        len(rejected_records),
    )

    selected = _select_tiles(
        valid_records=valid_records,
        sample_size=args.sample_size,
        seed=args.seed,
        allow_fewer=args.allow_fewer,
    )

    LOGGER.info(
        "%s copy %s tiles to %s",
        "Would" if args.dry_run else "Will",
        len(selected),
        dst_root,
    )

    if not args.dry_run:
        _copy_selected_tiles(selected, src_root, dst_root)
        _write_manifest(
            manifest_path=dst_root / args.manifest_name,
            src_root=src_root,
            dst_root=dst_root,
            split=args.split,
            sample_size=args.sample_size,
            min_valid_ratio=args.min_valid_ratio,
            seed=args.seed,
            candidates=candidates,
            valid_records=valid_records,
            rejected_records=rejected_records,
            selected=selected,
        )
        LOGGER.info("Wrote manifest to %s", dst_root / args.manifest_name)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
