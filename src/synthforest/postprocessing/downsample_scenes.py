"""Downsample full synthetic forest scenes for canopy-height estimation."""

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window

logger = logging.getLogger(__name__)

SUPPORTED_HEIGHT_REDUCERS = {
    "p90_positive",
    "p95_positive",
    "p99_positive",
    "max_positive",
    "mean_positive",
}
SUPPORTED_EDGE_POLICIES = {"center_crop"}


@dataclass(frozen=True)
class CropWindow:
    row_start: int
    col_start: int
    height: int
    width: int
    out_height: int
    out_width: int


@dataclass(frozen=True)
class DownsampleSceneOptions:
    source_root: Path
    output_root: Path | None = None
    source_gsd: float = 0.05
    target_gsd: float = 0.50
    height_reducer: str = "p95_positive"
    min_canopy_fraction: float = 0.10
    edge_policy: str = "center_crop"
    images_subdir: str = "images"
    height_maps_subdir: str = "height_maps"
    canopy_coverage_subdir: str = "canopy_coverage_maps"
    manifest_name: str = "downsample_manifest.json"
    overwrite: bool = True


@dataclass(frozen=True)
class ScenePair:
    scene_id: str
    rgb_path: Path
    height_path: Path


def _extract_scene_id(stem: str) -> str | None:
    match = re.search(r"(\d{4})$", stem)
    return match.group(1) if match else None


def _find_raster_files(directory: Path) -> list[Path]:
    return [
        p.resolve() for p in sorted(list(directory.glob("*.tif")) + list(directory.glob("*.tiff")))
    ]


def _default_output_root(source_root: Path, target_gsd: float) -> Path:
    centimeters = int(round(float(target_gsd) * 100.0))
    return source_root.parent / f"{source_root.name}_{centimeters}cm"


def _resolve_output_root(options: DownsampleSceneOptions) -> Path:
    if options.output_root is not None:
        return Path(options.output_root).resolve()
    return _default_output_root(Path(options.source_root).resolve(), options.target_gsd).resolve()


def _scene_files_by_id(directory: Path, label: str) -> dict[str, Path]:
    files = _find_raster_files(directory)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff {label} files found in: {directory}")

    by_id: dict[str, Path] = {}
    for path in files:
        scene_id = _extract_scene_id(path.stem)
        if scene_id is None:
            logger.warning("Could not extract trailing scene id from %s. Skipping.", path.name)
            continue
        if scene_id in by_id:
            raise ValueError(
                f"Duplicate {label} scene id '{scene_id}': {by_id[scene_id]} and {path}"
            )
        by_id[scene_id] = path
    return by_id


def find_scene_pairs(
    source_root: Path, images_subdir: str, height_maps_subdir: str
) -> list[ScenePair]:
    images_dir = source_root / images_subdir
    height_maps_dir = source_root / height_maps_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not height_maps_dir.exists():
        raise FileNotFoundError(f"Height maps directory not found: {height_maps_dir}")

    rgb_by_id = _scene_files_by_id(images_dir, "RGB")
    height_by_id = _scene_files_by_id(height_maps_dir, "height map")

    missing_heights = sorted(set(rgb_by_id) - set(height_by_id))
    if missing_heights:
        raise ValueError(f"Missing height maps for scene id(s): {', '.join(missing_heights)}")

    missing_images = sorted(set(height_by_id) - set(rgb_by_id))
    if missing_images:
        logger.warning(
            "Ignoring height maps without RGB pairs for scene id(s): %s", ", ".join(missing_images)
        )

    return [
        ScenePair(
            scene_id=scene_id, rgb_path=rgb_by_id[scene_id], height_path=height_by_id[scene_id]
        )
        for scene_id in sorted(rgb_by_id)
    ]


def compute_downsample_factor(source_gsd: float, target_gsd: float) -> int:
    source = float(source_gsd)
    target = float(target_gsd)
    if source <= 0.0 or target <= 0.0:
        raise ValueError("source_gsd and target_gsd must both be positive.")
    ratio = target / source
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"target_gsd/source_gsd must be an integer factor. Got {target_gsd}/{source_gsd}={ratio}."
        )
    return factor


def center_crop_window(height: int, width: int, factor: int) -> CropWindow:
    if factor <= 0:
        raise ValueError("factor must be positive.")
    crop_height = int(height) - (int(height) % factor)
    crop_width = int(width) - (int(width) % factor)
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(
            f"Raster shape {(height, width)} is smaller than one {factor}x{factor} output cell."
        )
    row_start = (int(height) - crop_height) // 2
    col_start = (int(width) - crop_width) // 2
    return CropWindow(
        row_start=row_start,
        col_start=col_start,
        height=crop_height,
        width=crop_width,
        out_height=crop_height // factor,
        out_width=crop_width // factor,
    )


def downsample_rgb_block_average(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate an RGB raster in band-first shape with exact block averaging."""
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB data in band-first shape, got shape {rgb.shape}.")
    bands, height, width = rgb.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"RGB shape {(height, width)} is not divisible by factor={factor}.")

    blocks = rgb.reshape(bands, height // factor, factor, width // factor, factor)
    averaged = np.rint(blocks.astype(np.float32).mean(axis=(2, 4)))
    return np.clip(averaged, 0, 255).astype(np.uint8)


def _percentile_from_reducer(height_reducer: str) -> float | None:
    if not height_reducer.startswith("p") or not height_reducer.endswith("_positive"):
        return None
    percentile_text = height_reducer[1 : height_reducer.index("_positive")]
    return float(percentile_text)


def downsample_height_positive(
    height_map: np.ndarray,
    factor: int,
    *,
    height_reducer: str = "p95_positive",
    min_canopy_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a height map using positive canopy pixels only.

    Returns a tuple of `(height, canopy_fraction)`, both in output resolution.
    """
    if height_reducer not in SUPPORTED_HEIGHT_REDUCERS:
        supported = ", ".join(sorted(SUPPORTED_HEIGHT_REDUCERS))
        raise ValueError(
            f"Unsupported height_reducer='{height_reducer}'. Supported values: {supported}"
        )

    if height_map.ndim != 2:
        raise ValueError(f"Expected a single-band height map, got shape {height_map.shape}.")
    height, width = height_map.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"Height shape {(height, width)} is not divisible by factor={factor}.")

    min_fraction = float(min_canopy_fraction)
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_canopy_fraction must be between 0 and 1.")

    block_values = (
        np.asarray(height_map, dtype=np.float32)
        .reshape(height // factor, factor, width // factor, factor)
        .transpose(0, 2, 1, 3)
        .reshape(height // factor, width // factor, factor * factor)
    )
    positive = np.isfinite(block_values) & (block_values > 0.0)
    canopy_fraction = positive.mean(axis=2, dtype=np.float32).astype(np.float32)
    keep = canopy_fraction >= min_fraction

    if height_reducer == "mean_positive":
        summed = np.where(positive, block_values, 0.0).sum(axis=2, dtype=np.float32)
        counts = positive.sum(axis=2)
        reduced = np.divide(
            summed,
            np.maximum(counts, 1),
            out=np.zeros_like(summed, dtype=np.float32),
            where=counts > 0,
        )
    elif height_reducer == "max_positive":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced = np.nanmax(np.where(positive, block_values, np.nan), axis=2)
    else:
        percentile = _percentile_from_reducer(height_reducer)
        if percentile is None:
            raise ValueError(f"Unsupported percentile reducer: {height_reducer}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced = np.nanpercentile(
                np.where(positive, block_values, np.nan),
                percentile,
                axis=2,
            )

    reduced = np.asarray(reduced, dtype=np.float32)
    reduced[~np.isfinite(reduced)] = 0.0
    reduced[~keep] = 0.0
    return reduced.astype(np.float32), canopy_fraction


def output_transform(source_transform: Affine, crop: CropWindow, factor: int) -> Affine:
    return (
        source_transform
        * Affine.translation(crop.col_start, crop.row_start)
        * Affine.scale(factor, factor)
    )


def _pixel_size_from_transform(transform: Affine) -> tuple[float, float]:
    return float(math.hypot(transform.a, transform.d)), float(math.hypot(transform.b, transform.e))


def _assert_close(value: float, expected: float, name: str, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(float(value), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name}={value} does not match expected {expected}.")


def _validate_dataset_gsd(
    dataset: rasterio.io.DatasetReader, expected_gsd: float, label: str
) -> None:
    pixel_x, pixel_y = _pixel_size_from_transform(dataset.transform)
    _assert_close(pixel_x, expected_gsd, f"{label} pixel size x")
    _assert_close(pixel_y, expected_gsd, f"{label} pixel size y")


def _same_crs(left: object, right: object) -> bool:
    return left == right or str(left) == str(right)


def _validate_pair(
    rgb_ds: rasterio.io.DatasetReader,
    height_ds: rasterio.io.DatasetReader,
    *,
    source_gsd: float,
) -> None:
    if rgb_ds.width != height_ds.width or rgb_ds.height != height_ds.height:
        raise ValueError(
            "RGB and height map shapes differ: "
            f"rgb={(rgb_ds.height, rgb_ds.width)} height={(height_ds.height, height_ds.width)}"
        )
    if rgb_ds.count < 3:
        raise ValueError(f"RGB raster must have at least 3 bands, got {rgb_ds.count}.")
    if height_ds.count != 1:
        raise ValueError(f"Height map must have exactly one band, got {height_ds.count}.")
    if not _same_crs(rgb_ds.crs, height_ds.crs):
        raise ValueError(f"RGB and height map CRS differ: rgb={rgb_ds.crs} height={height_ds.crs}")
    if not np.allclose(tuple(rgb_ds.transform), tuple(height_ds.transform), rtol=1e-9, atol=1e-9):
        raise ValueError("RGB and height map transforms differ.")

    _validate_dataset_gsd(rgb_ds, source_gsd, "RGB")
    _validate_dataset_gsd(height_ds, source_gsd, "height")


def _target_profile(
    source_profile: dict,
    *,
    width: int,
    height: int,
    count: int,
    dtype: str,
    transform: Affine,
) -> dict:
    profile = source_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": int(count),
            "dtype": dtype,
            "transform": transform,
            "nodata": None,
            "BIGTIFF": "IF_NEEDED",
            "tiled": False,
        }
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    return profile


def _base_tags(
    *,
    source_gsd: float,
    target_gsd: float,
    factor: int,
    crop: CropWindow,
    height_reducer: str,
    min_canopy_fraction: float,
) -> dict[str, str]:
    return {
        "AREA_OR_POINT": "Area",
        "GSD_AT_NADIR": f"{target_gsd:.12g}",
        "SYNTHETIC_PIXEL_SIZE_X": f"{target_gsd:.12g}",
        "SYNTHETIC_PIXEL_SIZE_Y": f"{target_gsd:.12g}",
        "SOURCE_GSD_AT_NADIR": f"{source_gsd:.12g}",
        "DOWNSAMPLE_FACTOR": str(int(factor)),
        "DOWNSAMPLE_EDGE_POLICY": "center_crop",
        "DOWNSAMPLE_CROP_ROW_START": str(int(crop.row_start)),
        "DOWNSAMPLE_CROP_COL_START": str(int(crop.col_start)),
        "DOWNSAMPLE_CROP_HEIGHT": str(int(crop.height)),
        "DOWNSAMPLE_CROP_WIDTH": str(int(crop.width)),
        "HEIGHT_REDUCER": height_reducer,
        "MIN_CANOPY_FRACTION": f"{min_canopy_fraction:.12g}",
    }


def _write_raster(path: Path, data: np.ndarray, profile: dict, tags: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**tags)


def _relative_to_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def downsample_scene_pair(
    pair: ScenePair,
    *,
    output_root: Path,
    source_root: Path,
    source_gsd: float,
    target_gsd: float,
    height_reducer: str,
    min_canopy_fraction: float,
    edge_policy: str,
    images_subdir: str,
    height_maps_subdir: str,
    canopy_coverage_subdir: str,
    overwrite: bool,
) -> dict[str, object]:
    if edge_policy not in SUPPORTED_EDGE_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_EDGE_POLICIES))
        raise ValueError(f"Unsupported edge_policy='{edge_policy}'. Supported values: {supported}")

    factor = compute_downsample_factor(source_gsd=source_gsd, target_gsd=target_gsd)

    output_rgb_path = output_root / images_subdir / pair.rgb_path.name
    output_height_path = output_root / height_maps_subdir / pair.height_path.name
    output_coverage_path = (
        output_root / canopy_coverage_subdir / f"canopy_coverage_map{pair.scene_id}.tif"
    )
    if not overwrite and any(
        p.exists() for p in (output_rgb_path, output_height_path, output_coverage_path)
    ):
        raise FileExistsError(
            "Downsampled output already exists and overwrite=false: "
            f"{output_rgb_path}, {output_height_path}, {output_coverage_path}"
        )

    with rasterio.open(pair.rgb_path) as rgb_ds, rasterio.open(pair.height_path) as height_ds:
        _validate_pair(rgb_ds, height_ds, source_gsd=source_gsd)
        source_height = int(rgb_ds.height)
        source_width = int(rgb_ds.width)
        crop = center_crop_window(rgb_ds.height, rgb_ds.width, factor)
        window = Window(crop.col_start, crop.row_start, crop.width, crop.height)

        rgb = rgb_ds.read(indexes=[1, 2, 3], window=window, masked=False)
        height = height_ds.read(1, window=window, masked=False)

        rgb_downsampled = downsample_rgb_block_average(rgb, factor)
        height_downsampled, coverage = downsample_height_positive(
            height,
            factor,
            height_reducer=height_reducer,
            min_canopy_fraction=min_canopy_fraction,
        )

        transform = output_transform(rgb_ds.transform, crop, factor)
        tags = _base_tags(
            source_gsd=source_gsd,
            target_gsd=target_gsd,
            factor=factor,
            crop=crop,
            height_reducer=height_reducer,
            min_canopy_fraction=min_canopy_fraction,
        )

        rgb_profile = _target_profile(
            rgb_ds.profile,
            width=crop.out_width,
            height=crop.out_height,
            count=3,
            dtype="uint8",
            transform=transform,
        )
        height_profile = _target_profile(
            height_ds.profile,
            width=crop.out_width,
            height=crop.out_height,
            count=1,
            dtype="float32",
            transform=transform,
        )
        coverage_profile = _target_profile(
            height_ds.profile,
            width=crop.out_width,
            height=crop.out_height,
            count=1,
            dtype="float32",
            transform=transform,
        )

        _write_raster(output_rgb_path, rgb_downsampled, rgb_profile, tags)
        _write_raster(
            output_height_path, height_downsampled[np.newaxis, :, :], height_profile, tags
        )
        coverage_tags = {**tags, "RASTER_ROLE": "canopy_coverage_fraction"}
        _write_raster(
            output_coverage_path, coverage[np.newaxis, :, :], coverage_profile, coverage_tags
        )

    logger.info(
        "Downsampled scene %s: %s -> %s (%dx%d)",
        pair.scene_id,
        pair.rgb_path.name,
        output_rgb_path,
        crop.out_width,
        crop.out_height,
    )
    return {
        "scene_id": pair.scene_id,
        "source": {
            "rgb": _relative_to_or_absolute(pair.rgb_path, source_root),
            "height_map": _relative_to_or_absolute(pair.height_path, source_root),
        },
        "output": {
            "rgb": _relative_to_or_absolute(output_rgb_path, output_root),
            "height_map": _relative_to_or_absolute(output_height_path, output_root),
            "canopy_coverage_map": _relative_to_or_absolute(output_coverage_path, output_root),
        },
        "source_shape_px": {"height": source_height, "width": source_width},
        "crop": {
            "row_start": crop.row_start,
            "col_start": crop.col_start,
            "height": crop.height,
            "width": crop.width,
        },
        "output_shape_px": {"height": crop.out_height, "width": crop.out_width},
    }


def downsample_scenes(options: DownsampleSceneOptions) -> Path:
    source_root = Path(options.source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    output_root = _resolve_output_root(options)
    output_root.mkdir(parents=True, exist_ok=True)

    height_reducer = str(options.height_reducer)
    if height_reducer not in SUPPORTED_HEIGHT_REDUCERS:
        supported = ", ".join(sorted(SUPPORTED_HEIGHT_REDUCERS))
        raise ValueError(
            f"Unsupported height_reducer='{height_reducer}'. Supported values: {supported}"
        )

    edge_policy = str(options.edge_policy)
    if edge_policy not in SUPPORTED_EDGE_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_EDGE_POLICIES))
        raise ValueError(f"Unsupported edge_policy='{edge_policy}'. Supported values: {supported}")

    factor = compute_downsample_factor(options.source_gsd, options.target_gsd)
    scene_pairs = find_scene_pairs(source_root, options.images_subdir, options.height_maps_subdir)
    if not scene_pairs:
        raise RuntimeError(f"No RGB/height scene pairs found under: {source_root}")

    logger.info(
        "Downsampling %d scene(s) from %.6f m/px to %.6f m/px (factor=%d).",
        len(scene_pairs),
        float(options.source_gsd),
        float(options.target_gsd),
        factor,
    )

    scenes: list[dict[str, object]] = []
    for pair in scene_pairs:
        scenes.append(
            downsample_scene_pair(
                pair,
                output_root=output_root,
                source_root=source_root,
                source_gsd=float(options.source_gsd),
                target_gsd=float(options.target_gsd),
                height_reducer=height_reducer,
                min_canopy_fraction=float(options.min_canopy_fraction),
                edge_policy=edge_policy,
                images_subdir=options.images_subdir,
                height_maps_subdir=options.height_maps_subdir,
                canopy_coverage_subdir=options.canopy_coverage_subdir,
                overwrite=bool(options.overwrite),
            )
        )

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_gsd_m_per_px": float(options.source_gsd),
        "target_gsd_m_per_px": float(options.target_gsd),
        "downsample_factor": factor,
        "height_reducer": height_reducer,
        "min_canopy_fraction": float(options.min_canopy_fraction),
        "edge_policy": edge_policy,
        "scene_count": len(scenes),
        "scenes": scenes,
        "tiling_note": (
            "Run `synthforest prepare tiles` on this output with "
            "ground_resolution=null and scale_factor=null."
        ),
    }
    manifest_path = output_root / options.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote downsampling manifest: %s", manifest_path)
    return manifest_path
