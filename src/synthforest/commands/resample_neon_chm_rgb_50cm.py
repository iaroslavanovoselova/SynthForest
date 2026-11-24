#!/usr/bin/env python3
"""Resample downloaded NEON RGB/CHM rasters onto a shared 50 cm grid in place.

Native NEON RGB mosaics are 10 cm and are downsampled with area averaging.
Native CHMs are 1 m and are upsampled with nearest-neighbor replication. The
CHM therefore has 50 cm pixels after processing but retains an effective
resolution of 1 m.

For each RGB/CHM pair, invalid coverage is resolved on the output grid. An
existing meaningful RGB mask is preferred; otherwise, border-connected pixels
that are black in all three bands are considered invalid. This RGB mask is
combined with finite, non-nodata CHM coverage. The exact combined internal mask
is written to the RGB and CHM, and invalid CHM cells are set to nodata. This
matches the repository's ``valid_masks.py`` auto-mask logic.

The source file is replaced only after a temporary GeoTIFF has been written and
validated. Files that already have the requested 50 cm grid and matching masks
are skipped, so the script is safe to rerun. The downloader publishes complete
files atomically from ``*.part`` files; this script ignores parts and takes a
snapshot of the complete pairs available when it starts. Run it again after
downloading ends if it reports pending/unpaired files.

For compatible height metrics, use:

    metric_resolution:
      enabled: true
      rgb_gsd_m: 0.5
      height_gsd_m: 1.0
      downsample_factor: null
      downsample_interpolation: area
      upsample_interpolation: nearest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import threading
import time
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from synthforest.paths import data_path

DEFAULT_DATA_ROOT = data_path("NEON_CHM_1m_real_all")
DEFAULT_FOLDS = ("train", "test", "unmatched_rgb")
DEFAULT_TARGET_GSD_M = 0.5
DEFAULT_RGB_SOURCE_GSD_M = 0.1
DEFAULT_CHM_SOURCE_GSD_M = 1.0
DEFAULT_CHM_EFFECTIVE_GSD_M = 1.0
DEFAULT_CHM_NODATA = -9999.0
DEFAULT_BLACK_PIXEL_TOLERANCE = 0
DEFAULT_BLACK_MASK_MODE = "border-connected"
BLACK_MASK_MODES = ("border-connected", "all")
PRODUCT_SUFFIXES = {"RGB": "_RGB.tif", "CHM": "_CHM.tif"}
TAG_PIXEL_GSD = "SYNTHFOREST_PIXEL_GSD_M"
TAG_SOURCE_GSD = "SYNTHFOREST_SOURCE_GSD_M"
TAG_EFFECTIVE_GSD = "SYNTHFOREST_EFFECTIVE_GSD_M"
TAG_RESAMPLING = "SYNTHFOREST_RESAMPLING"
TAG_VALID_MASK_VERSION = "SYNTHFOREST_VALID_MASK_VERSION"
TAG_VALID_MASK_SOURCE = "SYNTHFOREST_VALID_MASK_SOURCE"
TAG_VALID_MASK_MODE = "SYNTHFOREST_VALID_MASK_MODE"
TAG_VALID_MASK_TOLERANCE = "SYNTHFOREST_VALID_MASK_BLACK_TOLERANCE"
TAG_VALID_FRACTION = "SYNTHFOREST_VALID_FRACTION"
TAG_CHM_RGB_MASKED = "SYNTHFOREST_CHM_RGB_MASKED"
VALID_MASK_VERSION = "2"


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    product: str
    width: int
    height: int
    count: int
    dtype: str
    crs: object
    transform: object
    bounds: object
    nodata: float | int | None
    gsd_x: float
    gsd_y: float
    tags: dict[str, str]


@dataclass(frozen=True)
class WorkItem:
    fold: str
    stem: str
    rgb_path: Path | None
    chm_path: Path


@dataclass(frozen=True)
class WorkResult:
    fold: str
    stem: str
    status: str
    rgb_status: str
    chm_status: str
    mask_source: str | None
    valid_fraction: float | None
    elapsed_seconds: float


@dataclass(frozen=True)
class Inventory:
    items: tuple[WorkItem, ...]
    pending: tuple[dict[str, str], ...]
    ignored_part_files: int


@dataclass(frozen=True)
class ResampleOptions:
    data_root: Path
    folds: tuple[str, ...]
    target_gsd_m: float
    rgb_source_gsd_m: float
    chm_source_gsd_m: float
    chm_effective_gsd_m: float
    gsd_tolerance_m: float
    black_pixel_tolerance: int
    black_mask_mode: str
    workers: int
    warp_memory_mb: int
    progress_every: int
    strict_pairs: bool
    dry_run: bool


def _stem_for(path: Path, product: str) -> str:
    suffix = PRODUCT_SUFFIXES[product]
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected filename ending in {suffix}: {path}")
    return path.name[: -len(suffix)]


def discover_inventory(data_root: Path, folds: Iterable[str]) -> Inventory:
    """Discover complete pair files while ignoring active downloader parts."""
    items: list[WorkItem] = []
    pending: list[dict[str, str]] = []
    part_files = 0
    for fold in folds:
        fold_root = data_root / fold
        if not fold_root.exists():
            continue
        if not fold_root.is_dir():
            raise NotADirectoryError(f"Fold path is not a directory: {fold_root}")
        part_files += sum(1 for _ in fold_root.glob("*.part"))
        rgb_by_stem = {
            _stem_for(path, "RGB"): path
            for path in sorted(fold_root.glob("*_RGB.tif"))
            if path.is_file()
        }
        chm_by_stem = {
            _stem_for(path, "CHM"): path
            for path in sorted(fold_root.glob("*_CHM.tif"))
            if path.is_file()
        }

        if fold == "unmatched_rgb":
            for stem, chm_path in sorted(chm_by_stem.items()):
                items.append(
                    WorkItem(
                        fold=fold,
                        stem=stem,
                        rgb_path=None,
                        chm_path=chm_path,
                    )
                )
            for stem, rgb_path in sorted(rgb_by_stem.items()):
                pending.append(
                    {
                        "fold": fold,
                        "stem": stem,
                        "reason": f"unexpected RGB without paired CHM: {rgb_path.name}",
                    }
                )
            continue

        for stem in sorted(set(rgb_by_stem) & set(chm_by_stem)):
            items.append(
                WorkItem(
                    fold=fold,
                    stem=stem,
                    rgb_path=rgb_by_stem[stem],
                    chm_path=chm_by_stem[stem],
                )
            )
        for stem in sorted(set(rgb_by_stem) - set(chm_by_stem)):
            pending.append(
                {
                    "fold": fold,
                    "stem": stem,
                    "reason": f"RGB present, CHM not yet present: {rgb_by_stem[stem].name}",
                }
            )
        for stem in sorted(set(chm_by_stem) - set(rgb_by_stem)):
            pending.append(
                {
                    "fold": fold,
                    "stem": stem,
                    "reason": f"CHM present, RGB not yet present: {chm_by_stem[stem].name}",
                }
            )

    return Inventory(
        items=tuple(items),
        pending=tuple(pending),
        ignored_part_files=part_files,
    )


def inspect_raster(path: Path, product: str) -> RasterInfo:
    import rasterio

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        if not _north_up(src.transform):
            raise ValueError(f"Raster is not north-up: {path}; transform={src.transform}")
        if product == "RGB" and src.count != 3:
            raise ValueError(f"Expected exactly 3 RGB bands in {path}, got {src.count}")
        if product == "CHM" and src.count != 1:
            raise ValueError(f"Expected one CHM band in {path}, got {src.count}")
        return RasterInfo(
            path=path,
            product=product,
            width=int(src.width),
            height=int(src.height),
            count=int(src.count),
            dtype=str(src.dtypes[0]),
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            nodata=src.nodata,
            gsd_x=abs(float(src.transform.a)),
            gsd_y=abs(float(src.transform.e)),
            tags=dict(src.tags()),
        )


def _north_up(transform: object, tolerance: float = 1e-12) -> bool:
    return (
        abs(float(transform.b)) <= tolerance
        and abs(float(transform.d)) <= tolerance
        and float(transform.a) > 0.0
        and float(transform.e) < 0.0
    )


def _is_gsd(info: RasterInfo, expected: float, tolerance: float) -> bool:
    return math.isclose(info.gsd_x, expected, rel_tol=0.0, abs_tol=tolerance) and math.isclose(
        info.gsd_y, expected, rel_tol=0.0, abs_tol=tolerance
    )


def _validate_source_or_target_gsd(
    info: RasterInfo,
    *,
    source_gsd_m: float,
    target_gsd_m: float,
    tolerance: float,
) -> str:
    if _is_gsd(info, target_gsd_m, tolerance):
        return "target"
    if _is_gsd(info, source_gsd_m, tolerance):
        return "source"
    raise ValueError(
        f"Unexpected {info.product} GSD for {info.path}: "
        f"got ({info.gsd_x}, {info.gsd_y}) m, expected source "
        f"{source_gsd_m} m or target {target_gsd_m} m"
    )


def _same_crs_and_bounds(
    first: RasterInfo,
    second: RasterInfo,
    *,
    tolerance: float,
) -> bool:
    if first.crs != second.crs:
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(first.bounds, second.bounds)
    )


def _target_grid(info: RasterInfo, target_gsd_m: float) -> tuple[int, int, object]:
    from rasterio.transform import from_origin

    span_x = float(info.bounds.right - info.bounds.left)
    span_y = float(info.bounds.top - info.bounds.bottom)
    width = int(round(span_x / target_gsd_m))
    height = int(round(span_y / target_gsd_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster extent for target grid: {info.path}")
    grid_tolerance = max(1e-7, target_gsd_m * 1e-6)
    if not math.isclose(width * target_gsd_m, span_x, rel_tol=0.0, abs_tol=grid_tolerance):
        raise ValueError(
            f"Raster width is not divisible by {target_gsd_m} m: {info.path}, span={span_x}"
        )
    if not math.isclose(height * target_gsd_m, span_y, rel_tol=0.0, abs_tol=grid_tolerance):
        raise ValueError(
            f"Raster height is not divisible by {target_gsd_m} m: {info.path}, span={span_y}"
        )
    transform = from_origin(
        float(info.bounds.left),
        float(info.bounds.top),
        target_gsd_m,
        target_gsd_m,
    )
    return width, height, transform


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.resample-{os.getpid()}-{threading.get_ident()}.tmp.tif")


def _output_profile(
    source: RasterInfo,
    *,
    width: int,
    height: int,
    transform: object,
) -> dict[str, object]:
    import numpy as np

    dtype_kind = np.dtype(source.dtype).kind
    predictor = 3 if dtype_kind == "f" else 2
    nodata = source.nodata
    if source.product == "CHM" and nodata is None:
        nodata = DEFAULT_CHM_NODATA
    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": source.count,
        "dtype": source.dtype,
        "crs": source.crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": predictor,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "IF_SAFER",
    }
    if source.product == "RGB":
        profile.update(interleave="pixel", photometric="RGB")
    return profile


def _copy_safe_tags(
    source_tags: dict[str, str],
    *,
    product: str,
    source_gsd_m: float,
    target_gsd_m: float,
    chm_effective_gsd_m: float,
) -> dict[str, str]:
    tags = {
        key: value
        for key, value in source_tags.items()
        if not key.startswith("TIFFTAG_")
        and key
        not in {
            TAG_PIXEL_GSD,
            TAG_SOURCE_GSD,
            TAG_EFFECTIVE_GSD,
            TAG_RESAMPLING,
            TAG_VALID_MASK_VERSION,
            TAG_VALID_MASK_SOURCE,
            TAG_VALID_MASK_MODE,
            TAG_VALID_MASK_TOLERANCE,
            TAG_VALID_FRACTION,
            TAG_CHM_RGB_MASKED,
        }
    }
    tags.update(
        {
            TAG_PIXEL_GSD: f"{target_gsd_m:g}",
            TAG_SOURCE_GSD: f"{source_gsd_m:g}",
            TAG_RESAMPLING: "average" if product == "RGB" else "nearest",
        }
    )
    if product == "CHM":
        tags[TAG_EFFECTIVE_GSD] = f"{chm_effective_gsd_m:g}"
    return tags


def resample_raster(
    source: RasterInfo,
    destination: Path,
    *,
    target_gsd_m: float,
    source_gsd_m: float,
    chm_effective_gsd_m: float,
    warp_memory_mb: int,
) -> None:
    """Write one validated resampled raster without modifying its source."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    width, height, transform = _target_grid(source, target_gsd_m)
    profile = _output_profile(
        source,
        width=width,
        height=height,
        transform=transform,
    )
    resampling = Resampling.average if source.product == "RGB" else Resampling.nearest
    tags = _copy_safe_tags(
        source.tags,
        product=source.product,
        source_gsd_m=source_gsd_m,
        target_gsd_m=target_gsd_m,
        chm_effective_gsd_m=chm_effective_gsd_m,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_NUM_THREADS="1"):
        with (
            rasterio.open(source.path) as src,
            rasterio.open(
                destination,
                "w",
                **profile,
            ) as dst,
        ):
            for band_index in range(1, source.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    dst_nodata=profile["nodata"],
                    resampling=resampling,
                    num_threads=1,
                    warp_mem_limit=warp_memory_mb,
                    init_dest_nodata=True,
                )
            if source.product == "RGB":
                dst.colorinterp = src.colorinterp
            dst.update_tags(**tags)


def _border_connected_component(candidate: object) -> object:
    """Return candidate pixels connected to any image edge using 4-connectivity."""
    import numpy as np

    candidate = np.asarray(candidate, dtype=bool)
    seed = np.zeros(candidate.shape, dtype=bool)
    seed[0, :] = candidate[0, :]
    seed[-1, :] = candidate[-1, :]
    seed[:, 0] = candidate[:, 0]
    seed[:, -1] = candidate[:, -1]
    if not bool(seed.any()):
        return seed

    try:
        from scipy.ndimage import binary_propagation

        return np.asarray(binary_propagation(seed, mask=candidate), dtype=bool)
    except ImportError:
        height, width = candidate.shape
        connected = seed.copy()
        pending = deque(zip(*np.nonzero(seed)))
        while pending:
            row, col = pending.popleft()
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and candidate[next_row, next_col]
                    and not connected[next_row, next_col]
                ):
                    connected[next_row, next_col] = True
                    pending.append((next_row, next_col))
        return connected


def resolve_rgb_valid_mask(
    path: Path,
    *,
    black_pixel_tolerance: int,
    black_mask_mode: str,
) -> tuple[object, str]:
    """Resolve an output-grid RGB valid mask using metadata or black pixels."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as dataset:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            metadata_valid = np.asarray(dataset.dataset_mask()) > 0
        if not bool(metadata_valid.all()):
            return np.ascontiguousarray(metadata_valid, dtype=bool), "metadata"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rgb = dataset.read((1, 2, 3))
    finite = np.all(np.isfinite(rgb), axis=0)
    black = np.all(rgb <= int(black_pixel_tolerance), axis=0)
    if black_mask_mode == "border-connected":
        invalid_black = _border_connected_component(black)
        source = "border_connected_black"
    elif black_mask_mode == "all":
        invalid_black = black
        source = "all_black"
    else:
        raise ValueError(f"Unsupported black mask mode: {black_mask_mode}")
    valid = finite & ~invalid_black
    if not bool(finite.all()):
        source += "_and_nonfinite"
    return np.ascontiguousarray(valid, dtype=bool), source


def resolve_chm_valid_mask(path: Path) -> object:
    """Resolve finite, non-nodata CHM coverage despite internal-mask precedence."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as dataset:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            metadata_valid = np.asarray(dataset.dataset_mask()) > 0
            values = dataset.read(1)
        nodata = dataset.nodata

    valid = metadata_valid & np.isfinite(values)
    if nodata is not None:
        if math.isnan(float(nodata)):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata
    return np.ascontiguousarray(valid, dtype=bool)


def _resolution_tags(
    existing_tags: dict[str, str],
    *,
    product: str,
    source_gsd_m: float,
    target_gsd_m: float,
    chm_effective_gsd_m: float,
) -> dict[str, str]:
    return _copy_safe_tags(
        existing_tags,
        product=product,
        source_gsd_m=source_gsd_m,
        target_gsd_m=target_gsd_m,
        chm_effective_gsd_m=chm_effective_gsd_m,
    )


def write_rgb_valid_mask(
    path: Path,
    valid: object,
    *,
    mask_source: str,
    options: ResampleOptions,
) -> None:
    """Write the resolved mask inside the RGB GeoTIFF and record its provenance."""
    import numpy as np
    import rasterio

    valid = np.asarray(valid, dtype=bool)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "r+") as dataset:
            if valid.shape != (dataset.height, dataset.width):
                raise ValueError(
                    f"RGB mask shape {valid.shape} does not match "
                    f"{path} shape {(dataset.height, dataset.width)}"
                )
            dataset.write_mask(valid.astype(np.uint8) * 255)
            tags = _resolution_tags(
                dict(dataset.tags()),
                product="RGB",
                source_gsd_m=options.rgb_source_gsd_m,
                target_gsd_m=options.target_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
            )
            tags.update(
                {
                    TAG_VALID_MASK_VERSION: VALID_MASK_VERSION,
                    TAG_VALID_MASK_SOURCE: mask_source,
                    TAG_VALID_MASK_MODE: options.black_mask_mode,
                    TAG_VALID_MASK_TOLERANCE: str(options.black_pixel_tolerance),
                    TAG_VALID_FRACTION: f"{float(valid.mean()):.12g}",
                }
            )
            dataset.update_tags(**tags)


def apply_rgb_mask_to_chm(
    path: Path,
    valid: object,
    *,
    mask_source: str,
    options: ResampleOptions,
) -> None:
    """Apply the shared RGB/CHM mask and write it as the CHM internal mask."""
    import numpy as np
    import rasterio

    valid = np.asarray(valid, dtype=bool)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "r+") as dataset:
            if valid.shape != (dataset.height, dataset.width):
                raise ValueError(
                    f"RGB mask shape {valid.shape} does not match "
                    f"{path} shape {(dataset.height, dataset.width)}"
                )
            nodata = dataset.nodata
            if nodata is None:
                nodata = DEFAULT_CHM_NODATA
                dataset.nodata = nodata
            for _, window in dataset.block_windows(1):
                row_start = int(window.row_off)
                row_stop = row_start + int(window.height)
                col_start = int(window.col_off)
                col_stop = col_start + int(window.width)
                window_valid = valid[row_start:row_stop, col_start:col_stop]
                if bool(window_valid.all()):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    values = dataset.read(1, window=window)
                values[~window_valid] = nodata
                dataset.write(values, 1, window=window)
            dataset.write_mask(valid.astype(np.uint8) * 255)
            tags = _resolution_tags(
                dict(dataset.tags()),
                product="CHM",
                source_gsd_m=options.chm_source_gsd_m,
                target_gsd_m=options.target_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
            )
            tags.update(
                {
                    TAG_VALID_MASK_VERSION: VALID_MASK_VERSION,
                    TAG_VALID_MASK_SOURCE: f"paired_rgb:{mask_source}",
                    TAG_VALID_MASK_MODE: options.black_mask_mode,
                    TAG_VALID_MASK_TOLERANCE: str(options.black_pixel_tolerance),
                    TAG_VALID_FRACTION: f"{float(valid.mean()):.12g}",
                    TAG_CHM_RGB_MASKED: "true",
                }
            )
            dataset.update_tags(**tags)


def _mask_tags_match(info: RasterInfo, options: ResampleOptions) -> bool:
    tags = info.tags
    if tags.get(TAG_VALID_MASK_VERSION) != VALID_MASK_VERSION:
        return False
    if tags.get(TAG_VALID_MASK_MODE) != options.black_mask_mode:
        return False
    if tags.get(TAG_VALID_MASK_TOLERANCE) != str(options.black_pixel_tolerance):
        return False
    if info.product == "CHM" and tags.get(TAG_CHM_RGB_MASKED) != "true":
        return False
    return True


def validate_shared_masks(rgb_path: Path, chm_path: Path) -> float:
    """Verify that paired rasters expose exactly the same dataset mask."""
    import numpy as np
    import rasterio

    with rasterio.open(rgb_path) as rgb, rasterio.open(chm_path) as chm:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rgb_valid = np.asarray(rgb.dataset_mask()) > 0
            chm_valid = np.asarray(chm.dataset_mask()) > 0
        if rgb_valid.shape != chm_valid.shape or not np.array_equal(
            rgb_valid,
            chm_valid,
        ):
            raise ValueError(f"RGB and CHM masks differ: {rgb_path.name} and {chm_path.name}")
        nodata = chm.nodata
        if nodata is None and not bool(chm_valid.all()):
            raise ValueError(f"Masked CHM does not define nodata: {chm_path}")
        for _, window in chm.block_windows(1):
            row_start = int(window.row_off)
            row_stop = row_start + int(window.height)
            col_start = int(window.col_off)
            col_stop = col_start + int(window.width)
            window_valid = chm_valid[row_start:row_stop, col_start:col_stop]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                values = chm.read(1, window=window)
            valid_values = values[window_valid]
            if not bool(np.all(np.isfinite(valid_values))):
                raise ValueError(f"CHM contains non-finite values inside its mask: {chm_path}")
            if nodata is not None:
                if math.isnan(float(nodata)):
                    valid_matches_nodata = np.isnan(valid_values)
                else:
                    valid_matches_nodata = valid_values == nodata
                if bool(np.any(valid_matches_nodata)):
                    raise ValueError(f"CHM contains nodata values inside its mask: {chm_path}")
            if bool(window_valid.all()):
                continue
            invalid_values = values[~window_valid]
            if nodata is not None and math.isnan(float(nodata)):
                matches_nodata = np.isnan(invalid_values)
            else:
                matches_nodata = invalid_values == nodata
            if not bool(np.all(matches_nodata)):
                raise ValueError(f"CHM contains non-nodata values outside the RGB mask: {chm_path}")
        return float(rgb_valid.mean())


def validate_target_raster(
    path: Path,
    *,
    product: str,
    target_gsd_m: float,
    chm_effective_gsd_m: float,
    tolerance: float,
) -> RasterInfo:
    info = inspect_raster(path, product)
    if not _is_gsd(info, target_gsd_m, tolerance):
        raise ValueError(
            f"Resampled {product} does not have {target_gsd_m} m pixels: "
            f"{path}, got ({info.gsd_x}, {info.gsd_y})"
        )
    expected_width, expected_height, expected_transform = _target_grid(info, target_gsd_m)
    if info.width != expected_width or info.height != expected_height:
        raise ValueError(
            f"Resampled dimensions do not match extent: {path}, "
            f"got {info.width}x{info.height}, expected "
            f"{expected_width}x{expected_height}"
        )
    if not info.transform.almost_equals(expected_transform):
        raise ValueError(f"Resampled transform is not on the target grid: {path}")
    pixel_tag = info.tags.get(TAG_PIXEL_GSD)
    if pixel_tag is not None and not math.isclose(
        float(pixel_tag),
        target_gsd_m,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"Incorrect {TAG_PIXEL_GSD} tag in {path}: {pixel_tag}")
    if product == "CHM":
        effective_tag = info.tags.get(TAG_EFFECTIVE_GSD)
        if effective_tag is not None and not math.isclose(
            float(effective_tag),
            chm_effective_gsd_m,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"Incorrect {TAG_EFFECTIVE_GSD} tag in {path}: {effective_tag}")
    return info


def _validate_pair(rgb: RasterInfo, chm: RasterInfo, tolerance: float) -> None:
    if not _same_crs_and_bounds(rgb, chm, tolerance=tolerance):
        raise ValueError(
            "RGB/CHM CRS or bounds differ: "
            f"RGB={rgb.path} {rgb.crs} {tuple(rgb.bounds)}, "
            f"CHM={chm.path} {chm.crs} {tuple(chm.bounds)}"
        )
    if (
        rgb.width != chm.width
        or rgb.height != chm.height
        or not rgb.transform.almost_equals(chm.transform)
    ):
        raise ValueError(f"RGB/CHM target grids differ: {rgb.path.name} and {chm.path.name}")


def process_item(item: WorkItem, options: ResampleOptions) -> WorkResult:
    started = time.monotonic()
    rgb_info = inspect_raster(item.rgb_path, "RGB") if item.rgb_path is not None else None
    chm_info = inspect_raster(item.chm_path, "CHM")
    if rgb_info is not None and not _same_crs_and_bounds(
        rgb_info,
        chm_info,
        tolerance=options.gsd_tolerance_m,
    ):
        raise ValueError(f"Native RGB/CHM CRS or bounds differ for {item.fold}/{item.stem}")

    rgb_grid_status = (
        _validate_source_or_target_gsd(
            rgb_info,
            source_gsd_m=options.rgb_source_gsd_m,
            target_gsd_m=options.target_gsd_m,
            tolerance=options.gsd_tolerance_m,
        )
        if rgb_info is not None
        else "absent"
    )
    chm_grid_status = _validate_source_or_target_gsd(
        chm_info,
        source_gsd_m=options.chm_source_gsd_m,
        target_gsd_m=options.target_gsd_m,
        tolerance=options.gsd_tolerance_m,
    )

    rgb_temp = _temporary_path(item.rgb_path) if item.rgb_path is not None else None
    chm_temp = _temporary_path(item.chm_path)
    created_temps: list[Path] = []
    try:
        if rgb_info is None:
            if chm_grid_status == "source":
                created_temps.append(chm_temp)
                resample_raster(
                    chm_info,
                    chm_temp,
                    target_gsd_m=options.target_gsd_m,
                    source_gsd_m=options.chm_source_gsd_m,
                    chm_effective_gsd_m=options.chm_effective_gsd_m,
                    warp_memory_mb=options.warp_memory_mb,
                )
                validate_target_raster(
                    chm_temp,
                    product="CHM",
                    target_gsd_m=options.target_gsd_m,
                    chm_effective_gsd_m=options.chm_effective_gsd_m,
                    tolerance=options.gsd_tolerance_m,
                )
                os.replace(chm_temp, item.chm_path)
                created_temps.remove(chm_temp)
                chm_status = "resampled"
            else:
                validate_target_raster(
                    item.chm_path,
                    product="CHM",
                    target_gsd_m=options.target_gsd_m,
                    chm_effective_gsd_m=options.chm_effective_gsd_m,
                    tolerance=options.gsd_tolerance_m,
                )
                chm_status = "already_target"
            return WorkResult(
                fold=item.fold,
                stem=item.stem,
                status=chm_status,
                rgb_status="absent",
                chm_status=chm_status,
                mask_source=None,
                valid_fraction=None,
                elapsed_seconds=time.monotonic() - started,
            )

        if (
            rgb_grid_status == "target"
            and chm_grid_status == "target"
            and _mask_tags_match(rgb_info, options)
            and _mask_tags_match(chm_info, options)
        ):
            final_rgb_info = validate_target_raster(
                item.rgb_path,
                product="RGB",
                target_gsd_m=options.target_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
                tolerance=options.gsd_tolerance_m,
            )
            final_chm_info = validate_target_raster(
                item.chm_path,
                product="CHM",
                target_gsd_m=options.target_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
                tolerance=options.gsd_tolerance_m,
            )
            _validate_pair(
                final_rgb_info,
                final_chm_info,
                tolerance=options.gsd_tolerance_m,
            )
            valid_fraction = validate_shared_masks(item.rgb_path, item.chm_path)
            return WorkResult(
                fold=item.fold,
                stem=item.stem,
                status="already_target",
                rgb_status="already_target",
                chm_status="already_target",
                mask_source=rgb_info.tags.get(TAG_VALID_MASK_SOURCE),
                valid_fraction=valid_fraction,
                elapsed_seconds=time.monotonic() - started,
            )

        if rgb_temp is None:
            raise AssertionError("RGB temporary path was not created")
        created_temps.append(rgb_temp)
        if rgb_grid_status == "source":
            if rgb_temp is None:
                raise AssertionError("RGB temporary path was not created")
            resample_raster(
                rgb_info,
                rgb_temp,
                target_gsd_m=options.target_gsd_m,
                source_gsd_m=options.rgb_source_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
                warp_memory_mb=options.warp_memory_mb,
            )
        else:
            shutil.copy2(item.rgb_path, rgb_temp)

        created_temps.append(chm_temp)
        if chm_grid_status == "source":
            resample_raster(
                chm_info,
                chm_temp,
                target_gsd_m=options.target_gsd_m,
                source_gsd_m=options.chm_source_gsd_m,
                chm_effective_gsd_m=options.chm_effective_gsd_m,
                warp_memory_mb=options.warp_memory_mb,
            )
        else:
            shutil.copy2(item.chm_path, chm_temp)

        final_rgb_info = validate_target_raster(
            rgb_temp,
            product="RGB",
            target_gsd_m=options.target_gsd_m,
            chm_effective_gsd_m=options.chm_effective_gsd_m,
            tolerance=options.gsd_tolerance_m,
        )
        final_chm_info = validate_target_raster(
            chm_temp,
            product="CHM",
            target_gsd_m=options.target_gsd_m,
            chm_effective_gsd_m=options.chm_effective_gsd_m,
            tolerance=options.gsd_tolerance_m,
        )
        _validate_pair(
            final_rgb_info,
            final_chm_info,
            tolerance=options.gsd_tolerance_m,
        )

        rgb_valid_mask, mask_source = resolve_rgb_valid_mask(
            rgb_temp,
            black_pixel_tolerance=options.black_pixel_tolerance,
            black_mask_mode=options.black_mask_mode,
        )
        chm_valid_mask = resolve_chm_valid_mask(chm_temp)
        valid_mask = rgb_valid_mask & chm_valid_mask
        if not bool(chm_valid_mask.all()):
            mask_source += "_and_chm_valid_data"
        write_rgb_valid_mask(
            rgb_temp,
            valid_mask,
            mask_source=mask_source,
            options=options,
        )
        apply_rgb_mask_to_chm(
            chm_temp,
            valid_mask,
            mask_source=mask_source,
            options=options,
        )

        final_rgb_info = validate_target_raster(
            rgb_temp,
            product="RGB",
            target_gsd_m=options.target_gsd_m,
            chm_effective_gsd_m=options.chm_effective_gsd_m,
            tolerance=options.gsd_tolerance_m,
        )
        final_chm_info = validate_target_raster(
            chm_temp,
            product="CHM",
            target_gsd_m=options.target_gsd_m,
            chm_effective_gsd_m=options.chm_effective_gsd_m,
            tolerance=options.gsd_tolerance_m,
        )
        _validate_pair(
            final_rgb_info,
            final_chm_info,
            tolerance=options.gsd_tolerance_m,
        )
        valid_fraction = validate_shared_masks(rgb_temp, chm_temp)

        os.replace(chm_temp, item.chm_path)
        created_temps.remove(chm_temp)
        os.replace(rgb_temp, item.rgb_path)
        created_temps.remove(rgb_temp)

        rgb_status = "resampled_masked" if rgb_grid_status == "source" else "masked"
        chm_status = "resampled_masked" if chm_grid_status == "source" else "masked"
        status = "resampled" if "source" in {rgb_grid_status, chm_grid_status} else "masked"
        return WorkResult(
            fold=item.fold,
            stem=item.stem,
            status=status,
            rgb_status=rgb_status,
            chm_status=chm_status,
            mask_source=mask_source,
            valid_fraction=valid_fraction,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        for path in created_temps:
            path.unlink(missing_ok=True)


def run_items(
    items: Iterable[WorkItem],
    options: ResampleOptions,
) -> tuple[list[WorkResult], list[dict[str, str]]]:
    item_list = list(items)
    results: list[WorkResult] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        future_to_item: dict[Future[WorkResult], WorkItem] = {
            executor.submit(process_item, item, options): item for item in item_list
        }
        for completed, future in enumerate(as_completed(future_to_item), start=1):
            item = future_to_item[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "fold": item.fold,
                        "stem": item.stem,
                        "error": str(exc),
                    }
                )
            if completed % options.progress_every == 0 or completed == len(item_list):
                statuses = Counter(result.status for result in results)
                print(
                    f"Resampling {completed}/{len(item_list)} "
                    f"(resampled={statuses['resampled']}, "
                    f"masked={statuses['masked']}, "
                    f"already_target={statuses['already_target']}, "
                    f"failed={len(failures)})",
                    flush=True,
                )
    return results, failures


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _report_payload(
    *,
    options: ResampleOptions,
    inventory: Inventory,
    results: Iterable[WorkResult],
    failures: Iterable[dict[str, str]],
) -> dict[str, object]:
    result_list = list(results)
    failure_list = list(failures)
    status_counts = Counter(result.status for result in result_list)
    mask_source_counts = Counter(
        result.mask_source for result in result_list if result.mask_source is not None
    )
    valid_fractions = [
        result.valid_fraction for result in result_list if result.valid_fraction is not None
    ]
    fold_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in result_list:
        fold_counts[result.fold][result.status] += 1
    return {
        "data_root": str(options.data_root),
        "folds": list(options.folds),
        "source_gsd_m": {
            "rgb": options.rgb_source_gsd_m,
            "chm": options.chm_source_gsd_m,
        },
        "output_pixel_gsd_m": options.target_gsd_m,
        "chm_effective_gsd_m": options.chm_effective_gsd_m,
        "resampling": {"rgb": "average", "chm": "nearest"},
        "valid_mask": {
            "rgb_preferred_source": "meaningful RGB dataset mask",
            "rgb_fallback": (
                "border-connected all-band black pixels"
                if options.black_mask_mode == "border-connected"
                else "all all-band black pixels"
            ),
            "combined_with": "finite, non-nodata CHM coverage",
            "black_pixel_tolerance": options.black_pixel_tolerance,
            "written_to": ["RGB internal mask", "CHM internal mask", "CHM nodata"],
            "mask_source_counts": dict(sorted(mask_source_counts.items())),
            "valid_fraction": {
                "minimum": min(valid_fractions) if valid_fractions else None,
                "mean": (sum(valid_fractions) / len(valid_fractions) if valid_fractions else None),
                "maximum": max(valid_fractions) if valid_fractions else None,
            },
        },
        "benchmark_metric_resolution": {
            "enabled": True,
            "rgb_gsd_m": options.target_gsd_m,
            "height_gsd_m": options.chm_effective_gsd_m,
            "downsample_factor": None,
            "min_valid_fraction": 0.5,
            "downsample_interpolation": "area",
            "upsample_interpolation": "nearest",
        },
        "snapshot": {
            "work_items": len(inventory.items),
            "pending_unpaired": len(inventory.pending),
            "ignored_part_files": inventory.ignored_part_files,
        },
        "results": {
            "successful": len(result_list),
            "failed": len(failure_list),
            "by_status": dict(sorted(status_counts.items())),
            "by_fold_and_status": {
                fold: dict(sorted(counts.items())) for fold, counts in sorted(fold_counts.items())
            },
        },
        "pending": list(inventory.pending),
        "failures": failure_list,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Atomically overwrite native NEON RGB/CHM GeoTIFFs with paired 50 cm rasters.")
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--fold",
        action="append",
        choices=DEFAULT_FOLDS,
        dest="folds",
        help="Fold to process; repeat as needed. Defaults to train, test, and unmatched_rgb.",
    )
    parser.add_argument("--target-gsd-m", type=float, default=DEFAULT_TARGET_GSD_M)
    parser.add_argument(
        "--rgb-source-gsd-m",
        type=float,
        default=DEFAULT_RGB_SOURCE_GSD_M,
    )
    parser.add_argument(
        "--chm-source-gsd-m",
        type=float,
        default=DEFAULT_CHM_SOURCE_GSD_M,
    )
    parser.add_argument(
        "--chm-effective-gsd-m",
        type=float,
        default=DEFAULT_CHM_EFFECTIVE_GSD_M,
    )
    parser.add_argument("--gsd-tolerance-m", type=float, default=1e-6)
    parser.add_argument(
        "--black-pixel-tolerance",
        type=int,
        default=DEFAULT_BLACK_PIXEL_TOLERANCE,
        help=(
            "Maximum value in every RGB band for a pixel to count as black "
            f"(default: {DEFAULT_BLACK_PIXEL_TOLERANCE})."
        ),
    )
    parser.add_argument(
        "--black-mask-mode",
        choices=BLACK_MASK_MODES,
        default=DEFAULT_BLACK_MASK_MODE,
        help=(
            "Mask only black regions connected to an image border (benchmark "
            "default), or every all-band black pixel."
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warp-memory-mb", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Fail before processing if train/test contain currently unpaired final files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the current inventory without changing rasters or writing a report.",
    )
    args = parser.parse_args()
    if args.target_gsd_m <= 0.0:
        parser.error("--target-gsd-m must be positive")
    if args.rgb_source_gsd_m <= 0.0 or args.chm_source_gsd_m <= 0.0:
        parser.error("source GSD values must be positive")
    if args.chm_effective_gsd_m < args.target_gsd_m:
        parser.error("--chm-effective-gsd-m must be at least --target-gsd-m")
    if args.gsd_tolerance_m <= 0.0:
        parser.error("--gsd-tolerance-m must be positive")
    if not 0 <= args.black_pixel_tolerance <= 255:
        parser.error("--black-pixel-tolerance must be between 0 and 255")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.warp_memory_mb <= 0:
        parser.error("--warp-memory-mb must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    folds = tuple(dict.fromkeys(args.folds or DEFAULT_FOLDS))
    options = ResampleOptions(
        data_root=data_root,
        folds=folds,
        target_gsd_m=float(args.target_gsd_m),
        rgb_source_gsd_m=float(args.rgb_source_gsd_m),
        chm_source_gsd_m=float(args.chm_source_gsd_m),
        chm_effective_gsd_m=float(args.chm_effective_gsd_m),
        gsd_tolerance_m=float(args.gsd_tolerance_m),
        black_pixel_tolerance=int(args.black_pixel_tolerance),
        black_mask_mode=str(args.black_mask_mode),
        workers=int(args.workers),
        warp_memory_mb=int(args.warp_memory_mb),
        progress_every=int(args.progress_every),
        strict_pairs=bool(args.strict_pairs),
        dry_run=bool(args.dry_run),
    )
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    inventory = discover_inventory(data_root, folds)
    print(
        f"Snapshot: {len(inventory.items)} complete work item(s), "
        f"{len(inventory.pending)} pending/unpaired final file(s), "
        f"{inventory.ignored_part_files} active .part file(s)."
    )
    if inventory.pending and options.strict_pairs:
        raise SystemExit(
            f"Found {len(inventory.pending)} pending/unpaired file(s); "
            "wait for downloading to finish or omit --strict-pairs."
        )
    if options.dry_run:
        print("Dry run: no rasters or metadata were changed.")
        return
    if not inventory.items:
        raise SystemExit("No complete RGB/CHM pairs or unmatched CHMs were found.")

    results, failures = run_items(inventory.items, options)
    report = _report_payload(
        options=options,
        inventory=inventory,
        results=results,
        failures=failures,
    )
    report_path = data_root / "metadata" / "resampling_50cm_report.json"
    _atomic_write_json(report_path, report)
    print(f"Report: {report_path}")
    if inventory.pending:
        print(
            "The download snapshot was incomplete; rerun this script after downloading "
            "finishes to process newly completed pairs."
        )
    if failures:
        print(
            f"Resampling finished with {len(failures)} failure(s); see {report_path}.",
            flush=True,
        )
        raise SystemExit(1)
    print("Resampling complete.")


if __name__ == "__main__":
    main()
