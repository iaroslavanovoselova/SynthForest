#!/usr/bin/env python3
"""Preprocess Virtual Shizuoka LP zips into paired RGB and CHM rasters."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from synthforest.paths import data_path

LOGGER = logging.getLogger("preprocess_virtual_shizuoka_height")

DEFAULT_INPUT_ROOT = data_path("virtual-shizuoka_LIDAR")
DEFAULT_OUTPUT_ROOT = data_path("CHM_rasters_20cm/virtual-shizuoka_CHM")
EPSG_CODE = 6676
CRS = f"EPSG:{EPSG_CODE}"
NODATA = -9999.0
RGB_GSD_M = 0.2
DTM_GSD_M = 0.5
SUPPORTED_RESOLUTIONS = (0.2, 0.5)


@dataclass(frozen=True)
class SceneSet:
    grid_code: str
    original_zip: Path
    grid_zip: Path
    ground_zip: Path
    ortho_zip: Path


@dataclass(frozen=True)
class RasterSpec:
    width: int
    height: int
    transform: object
    crs: str = CRS


@dataclass(frozen=True)
class DtmRaster:
    values: np.ndarray
    valid_mask: np.ndarray
    spec: RasterSpec
    resolution: float
    source_has_valid_flag: bool


@dataclass(frozen=True)
class OutputPaths:
    benchmark_rgb: Path
    benchmark_chm: Path
    benchmark_chm_match_rgb: Path | None
    scene_rgb: Path
    scene_chm: Path
    diagnostic_dsm: Path
    diagnostic_dtm: Path
    diagnostic_point_count: Path
    qa_json: Path


@dataclass(frozen=True)
class ScenePlan:
    scene: SceneSet
    scene_index: int
    skip_existing: bool
    missing_outputs: tuple[Path, ...] = ()


@dataclass
class PreprocessOptions:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    year: str = "2025"
    resolution: float = DTM_GSD_M
    min_dsm_points: int = 1
    ignore_classes: tuple[int, ...] = (7, 18)
    chunk_size: int = 2_000_000
    max_scenes: int | None = None
    scene_codes: tuple[str, ...] = ()
    work_dir: Path | None = None
    overwrite: bool = False
    skip_existing: bool = False
    keep_work: bool = False


MANIFEST_FIELDNAMES = [
    "scene_id",
    "grid_code",
    "requested_resolution_m",
    "benchmark_rgb",
    "benchmark_chm",
    "benchmark_chm_match_rgb",
    "scene_rgb",
    "scene_chm",
    "chm_valid_fraction",
    "chm_max_m",
    "dsm_valid_fraction",
    "qa_json",
]


def discover_scene_sets(input_root: str | Path, year: str = "2025") -> list[SceneSet]:
    """Return grid-code matched scene sets with all required LP products present."""
    lp_root = Path(input_root).expanduser() / str(year) / "LP"
    product_maps = {
        product: _find_product_zips(lp_root, product)
        for product in ("Original", "Grid", "Ground", "Ortho")
    }
    complete_codes = set.intersection(*(set(paths) for paths in product_maps.values()))
    scenes = [
        SceneSet(
            grid_code=code,
            original_zip=product_maps["Original"][code],
            grid_zip=product_maps["Grid"][code],
            ground_zip=product_maps["Ground"][code],
            ortho_zip=product_maps["Ortho"][code],
        )
        for code in sorted(complete_codes)
    ]
    return scenes


def build_scene_stem(grid_code: str, scene_index: int) -> str:
    return f"Shizuoka_{grid_code.upper()}_{scene_index:04d}"


def build_output_paths(
    output_root: str | Path, grid_code: str, scene_index: int, resolution: float
) -> OutputPaths:
    output_root = Path(output_root).expanduser()
    stem = build_scene_stem(grid_code, scene_index)
    scene_id = f"{scene_index:04d}"
    chm_match_rgb = (
        output_root / "benchmark" / f"{stem}_CHM_matchRGB.tif"
        if _is_resolution(resolution, DTM_GSD_M)
        else None
    )
    return OutputPaths(
        benchmark_rgb=output_root / "benchmark" / f"{stem}_RGB.tif",
        benchmark_chm=output_root / "benchmark" / f"{stem}_CHM.tif",
        benchmark_chm_match_rgb=chm_match_rgb,
        scene_rgb=output_root / "scenes" / "images" / f"rgb_shizuoka{scene_id}.tif",
        scene_chm=output_root / "scenes" / "height_maps" / f"height_map_shizuoka{scene_id}.tif",
        diagnostic_dsm=output_root / "diagnostics" / "dsm" / f"{stem}_DSM.tif",
        diagnostic_dtm=output_root / "diagnostics" / "dtm" / f"{stem}_DTM.tif",
        diagnostic_point_count=output_root
        / "diagnostics"
        / "point_counts"
        / f"{stem}_DSM_point_count.tif",
        qa_json=output_root / "qa" / f"{stem}_qa.json",
    )


def read_grid_dtm_from_zip(grid_zip: str | Path, nodata: float = NODATA) -> DtmRaster:
    """Read a Virtual Shizuoka Grid TXT zip into a north-up DTM raster."""
    grid_zip = Path(grid_zip)
    txt_member = _single_zip_member(grid_zip, (".txt",))
    with zipfile.ZipFile(grid_zip) as archive:
        with archive.open(txt_member) as fp:
            data = np.loadtxt(fp, delimiter=",")

    data = np.atleast_2d(data)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 Grid columns in {grid_zip}, got {data.shape[1]}.")

    x = data[:, 1].astype(np.float64, copy=False)
    y = data[:, 2].astype(np.float64, copy=False)
    z = data[:, 3].astype(np.float32, copy=False)
    has_valid_flag = data.shape[1] >= 5
    row_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if has_valid_flag:
        row_valid &= data[:, 4].astype(np.int16, copy=False) == 1

    xs = np.unique(x[np.isfinite(x)])
    ys = np.unique(y[np.isfinite(y)])
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"No finite Grid coordinates found in {grid_zip}.")

    x_res = _coordinate_spacing(xs, "x", grid_zip)
    y_res = _coordinate_spacing(ys, "y", grid_zip)
    if not np.isclose(x_res, y_res, rtol=0.0, atol=1e-6):
        raise ValueError(f"Grid spacing is not square in {grid_zip}: x={x_res}, y={y_res}.")

    width = int(xs.size)
    height = int(ys.size)
    raster = np.full((height, width), nodata, dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    col = np.rint((x - xs[0]) / x_res).astype(np.int64)
    row = np.rint((ys[-1] - y) / y_res).astype(np.int64)
    in_bounds = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    assign = in_bounds & row_valid
    raster[row[assign], col[assign]] = z[assign]
    valid_mask[row[assign], col[assign]] = True

    from rasterio.transform import from_origin

    transform = from_origin(float(xs[0] - x_res / 2.0), float(ys[-1] + y_res / 2.0), x_res, y_res)
    return DtmRaster(
        values=raster,
        valid_mask=valid_mask,
        spec=RasterSpec(width=width, height=height, transform=transform),
        resolution=float(x_res),
        source_has_valid_flag=has_valid_flag,
    )


def rasterize_las_to_arrays(
    las_path: str | Path,
    spec: RasterSpec,
    *,
    ignore_classes: Iterable[int] = (7, 18),
    chunk_size: int = 2_000_000,
    nodata: float = NODATA,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Rasterize LAS points into max-Z DSM and point-count arrays."""
    import laspy

    ignore = np.asarray(tuple(int(v) for v in ignore_classes), dtype=np.uint8)
    transform = spec.transform
    west = float(transform.c)
    north = float(transform.f)
    x_res = float(transform.a)
    y_res = abs(float(transform.e))

    max_z = np.full((int(spec.height), int(spec.width)), -np.inf, dtype=np.float32)
    point_count = np.zeros((int(spec.height), int(spec.width)), dtype=np.uint32)
    max_z_flat = max_z.ravel()
    point_count_flat = point_count.ravel()
    stats = {
        "total_points": 0,
        "accepted_class_points": 0,
        "accepted_in_bounds_points": 0,
        "ignored_noise_class_points": 0,
    }

    with laspy.open(las_path) as reader:
        stats["total_points"] = int(reader.header.point_count)
        for points in reader.chunk_iterator(int(chunk_size)):
            classes = np.asarray(points.classification, dtype=np.uint8)
            keep_class = (
                ~np.isin(classes, ignore) if ignore.size else np.ones(classes.shape, dtype=bool)
            )
            stats["ignored_noise_class_points"] += int((~keep_class).sum())
            if not bool(keep_class.any()):
                continue

            x = np.asarray(points.x, dtype=np.float64)[keep_class]
            y = np.asarray(points.y, dtype=np.float64)[keep_class]
            z = np.asarray(points.z, dtype=np.float32)[keep_class]
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not bool(finite.any()):
                continue

            x = x[finite]
            y = y[finite]
            z = z[finite]
            stats["accepted_class_points"] += int(z.size)

            col = np.floor((x - west) / x_res).astype(np.int64)
            row = np.floor((north - y) / y_res).astype(np.int64)
            in_bounds = (row >= 0) & (row < spec.height) & (col >= 0) & (col < spec.width)
            if not bool(in_bounds.any()):
                continue

            flat_index = row[in_bounds] * int(spec.width) + col[in_bounds]
            z_in_bounds = z[in_bounds]
            np.maximum.at(max_z_flat, flat_index, z_in_bounds)
            np.add.at(point_count_flat, flat_index, 1)
            stats["accepted_in_bounds_points"] += int(z_in_bounds.size)

    dsm = np.where(point_count > 0, max_z, nodata).astype(np.float32, copy=False)
    return dsm, point_count, stats


def compute_chm(
    dsm: np.ndarray,
    point_count: np.ndarray,
    dtm: np.ndarray,
    dtm_valid_mask: np.ndarray,
    *,
    min_dsm_points: int,
    nodata: float = NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CHM with NoData propagation and non-negative clamping."""
    if (
        dsm.shape != dtm.shape
        or dsm.shape != point_count.shape
        or dsm.shape != dtm_valid_mask.shape
    ):
        raise ValueError(
            "DSM, point count, DTM, and DTM valid mask must have identical shapes; "
            f"got dsm={dsm.shape}, count={point_count.shape}, dtm={dtm.shape}, "
            f"dtm_valid={dtm_valid_mask.shape}."
        )
    dsm_valid = (
        (point_count >= int(min_dsm_points)) & np.isfinite(dsm) & (np.asarray(dsm) != float(nodata))
    )
    dtm_valid = (
        np.asarray(dtm_valid_mask, dtype=bool)
        & np.isfinite(dtm)
        & (np.asarray(dtm) != float(nodata))
    )
    valid_mask = np.ascontiguousarray(dsm_valid & dtm_valid, dtype=bool)
    chm = np.full(dsm.shape, float(nodata), dtype=np.float32)
    chm[valid_mask] = np.maximum(dsm[valid_mask] - dtm[valid_mask], 0.0).astype(
        np.float32, copy=False
    )
    return chm, valid_mask


def run_preprocessing(options: PreprocessOptions) -> Path:
    _require_runtime_dependencies()
    if not any(_is_resolution(options.resolution, value) for value in SUPPORTED_RESOLUTIONS):
        raise ValueError(
            f"resolution must be one of {SUPPORTED_RESOLUTIONS}, got {options.resolution}."
        )
    if options.overwrite and options.skip_existing:
        raise ValueError("--overwrite and --skip-existing are mutually exclusive.")

    scenes = discover_scene_sets(options.input_root, options.year)
    scene_code_filter = {code.upper() for code in options.scene_codes}
    if scene_code_filter:
        scenes = [scene for scene in scenes if scene.grid_code.upper() in scene_code_filter]
    if options.max_scenes is not None:
        scenes = scenes[: int(options.max_scenes)]
    if not scenes:
        raise ValueError("No complete Virtual Shizuoka scene sets matched the requested filters.")

    output_root = Path(options.output_root).expanduser()
    _prepare_output_dirs(output_root)
    manifest_path = output_root / "manifest.csv"
    scene_manifest_path = output_root / "scenes" / "manifest.csv"

    rows: list[dict[str, object]] = _read_manifest(manifest_path) if options.skip_existing else []
    resume_plans = (
        _build_resume_plan(
            scenes=scenes,
            output_root=output_root,
            resolution=options.resolution,
            existing_rows=rows,
        )
        if options.skip_existing
        else None
    )
    if options.work_dir is not None:
        work_root = Path(options.work_dir).expanduser()
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup_context = None
    else:
        cleanup_context = tempfile.TemporaryDirectory(prefix="virtual_shizuoka_", dir=None)
        work_root = Path(cleanup_context.name)

    try:
        plans = resume_plans or [
            ScenePlan(scene=scene, scene_index=index, skip_existing=False)
            for index, scene in enumerate(scenes, start=1)
        ]
        processed_count = 0
        skipped_count = 0
        for plan in plans:
            if plan.skip_existing:
                LOGGER.info(
                    "Skipping existing complete scene %s as scene %04d",
                    plan.scene.grid_code,
                    plan.scene_index,
                )
                skipped_count += 1
                continue
            if options.skip_existing and plan.missing_outputs:
                LOGGER.info(
                    "Reprocessing incomplete existing scene %s as scene %04d (%d missing output(s)).",
                    plan.scene.grid_code,
                    plan.scene_index,
                    len(plan.missing_outputs),
                )
                _remove_expected_outputs(
                    build_output_paths(
                        output_root, plan.scene.grid_code, plan.scene_index, options.resolution
                    )
                )

            LOGGER.info("Processing %s as scene %04d", plan.scene.grid_code, plan.scene_index)
            row = process_scene(
                scene=plan.scene,
                scene_index=plan.scene_index,
                options=options,
                work_root=work_root,
            )
            rows = _upsert_manifest_row(rows, row)
            _write_manifest(manifest_path, rows)
            _write_manifest(scene_manifest_path, rows)
            processed_count += 1
    finally:
        if cleanup_context is not None:
            cleanup_context.cleanup()

    if options.skip_existing and rows:
        _write_manifest(manifest_path, rows)
        _write_manifest(scene_manifest_path, rows)
    LOGGER.info(
        "Processed %d scene(s), skipped %d existing scene(s). Output: %s",
        processed_count,
        skipped_count,
        output_root,
    )
    return output_root


def process_scene(
    *,
    scene: SceneSet,
    scene_index: int,
    options: PreprocessOptions,
    work_root: Path,
) -> dict[str, object]:
    paths = build_output_paths(
        options.output_root, scene.grid_code, scene_index, options.resolution
    )
    _check_outputs(paths, overwrite=options.overwrite)

    scene_work = work_root / f"{scene.grid_code}_{scene_index:04d}"
    if scene_work.exists():
        if options.overwrite or options.skip_existing:
            shutil.rmtree(scene_work)
        else:
            raise FileExistsError(f"Work directory already exists: {scene_work}")
    scene_work.mkdir(parents=True, exist_ok=True)

    try:
        rgb_spec = write_native_rgb_from_zip(
            scene.ortho_zip,
            paths.benchmark_rgb,
            scene_work / "ortho",
            overwrite=options.overwrite,
        )
        dtm_source = read_grid_dtm_from_zip(scene.grid_zip)
        target_spec = rgb_spec if _is_resolution(options.resolution, RGB_GSD_M) else dtm_source.spec

        las_path = extract_las_from_zip(scene.original_zip, scene_work / "las")
        dsm, point_count, dsm_stats = rasterize_las_to_arrays(
            las_path,
            target_spec,
            ignore_classes=options.ignore_classes,
            chunk_size=options.chunk_size,
        )

        dtm_values, dtm_valid = align_dtm_to_spec(dtm_source, target_spec)
        chm, chm_valid = compute_chm(
            dsm,
            point_count,
            dtm_values,
            dtm_valid,
            min_dsm_points=options.min_dsm_points,
        )

        write_single_band_raster(paths.diagnostic_dsm, dsm, target_spec, nodata=NODATA)
        write_single_band_raster(paths.diagnostic_dtm, dtm_values, target_spec, nodata=NODATA)
        write_single_band_raster(
            paths.diagnostic_point_count, point_count, target_spec, nodata=0, dtype="uint32"
        )
        write_single_band_raster(paths.benchmark_chm, chm, target_spec, nodata=NODATA)

        benchmark_chm_for_scenes = paths.benchmark_chm
        if paths.benchmark_chm_match_rgb is not None:
            reproject_single_band_to_match(
                source_path=paths.benchmark_chm,
                match_path=paths.benchmark_rgb,
                output_path=paths.benchmark_chm_match_rgb,
                resampling="nearest",
                nodata=NODATA,
            )
            benchmark_chm_for_scenes = paths.benchmark_chm_match_rgb

        hardlink_replace(paths.benchmark_rgb, paths.scene_rgb, overwrite=options.overwrite)
        hardlink_replace(benchmark_chm_for_scenes, paths.scene_chm, overwrite=options.overwrite)

        qa = build_qa_payload(
            scene=scene,
            scene_index=scene_index,
            resolution=options.resolution,
            rgb_spec=rgb_spec,
            chm_spec=target_spec,
            dsm=dsm,
            point_count=point_count,
            dtm=dtm_values,
            dtm_valid=dtm_valid,
            chm=chm,
            chm_valid=chm_valid,
            dsm_stats=dsm_stats,
            paths=paths,
        )
        _write_json(paths.qa_json, qa)
        return _manifest_row(scene, scene_index, options.resolution, paths, qa)
    finally:
        if not options.keep_work and scene_work.exists():
            shutil.rmtree(scene_work)


def write_native_rgb_from_zip(
    ortho_zip: str | Path,
    output_path: str | Path,
    work_dir: str | Path,
    *,
    overwrite: bool,
) -> RasterSpec:
    import rasterio
    from rasterio.enums import ColorInterp

    output_path = Path(output_path)
    _prepare_path(output_path, overwrite=overwrite)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    tif_member = _single_zip_member(ortho_zip, (".tif", ".tiff"))
    sidecar_members = _zip_members(ortho_zip, (".tfw", ".wld", ".tifw"))
    tif_path = _extract_zip_member(ortho_zip, tif_member, work_dir)
    for member in sidecar_members:
        _extract_zip_member(ortho_zip, member, work_dir)

    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands in ortho image {tif_path}, got {src.count}."
            )
        data = src.read(indexes=[1, 2, 3])
        profile = src.profile.copy()
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(
            driver="GTiff",
            count=3,
            crs=CRS,
            dtype=data.dtype,
            nodata=None,
            compress="deflate",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

    with rasterio.open(output_path) as dataset:
        return RasterSpec(
            width=int(dataset.width), height=int(dataset.height), transform=dataset.transform
        )


def extract_las_from_zip(las_zip: str | Path, work_dir: str | Path) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    member = _single_zip_member(las_zip, (".las", ".laz"))
    return _extract_zip_member(las_zip, member, work_dir)


def align_dtm_to_spec(dtm: DtmRaster, target: RasterSpec) -> tuple[np.ndarray, np.ndarray]:
    if _same_grid(dtm.spec, target):
        return dtm.values.copy(), dtm.valid_mask.copy()

    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full((target.height, target.width), NODATA, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reproject(
            source=dtm.values,
            destination=destination,
            src_transform=dtm.spec.transform,
            src_crs=CRS,
            src_nodata=NODATA,
            dst_transform=target.transform,
            dst_crs=CRS,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )

    valid_destination = np.zeros((target.height, target.width), dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reproject(
            source=dtm.valid_mask.astype(np.uint8),
            destination=valid_destination,
            src_transform=dtm.spec.transform,
            src_crs=CRS,
            src_nodata=0,
            dst_transform=target.transform,
            dst_crs=CRS,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    valid = (valid_destination > 0) & np.isfinite(destination) & (destination != NODATA)
    return destination, np.ascontiguousarray(valid, dtype=bool)


def write_single_band_raster(
    path: str | Path,
    array: np.ndarray,
    spec: RasterSpec,
    *,
    nodata: float | int,
    dtype: str = "float32",
) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array, dtype=dtype)
    profile = {
        "driver": "GTiff",
        "height": int(spec.height),
        "width": int(spec.width),
        "count": 1,
        "dtype": dtype,
        "crs": CRS,
        "transform": spec.transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values[np.newaxis, :, :])


def reproject_single_band_to_match(
    *,
    source_path: str | Path,
    match_path: str | Path,
    output_path: str | Path,
    resampling: str,
    nodata: float,
) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampling_method = getattr(Resampling, resampling)

    with rasterio.open(source_path) as src, rasterio.open(match_path) as match:
        destination = np.full((match.height, match.width), float(nodata), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            reproject(
                source=src.read(1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=match.transform,
                dst_crs=match.crs,
                dst_nodata=nodata,
                resampling=resampling_method,
            )
        profile = {
            "driver": "GTiff",
            "height": int(match.height),
            "width": int(match.width),
            "count": 1,
            "dtype": "float32",
            "crs": match.crs,
            "transform": match.transform,
            "nodata": nodata,
            "compress": "deflate",
        }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination[np.newaxis, :, :])


def hardlink_replace(source: str | Path, destination: str | Path, *, overwrite: bool) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        destination.unlink()
    os.link(source, destination)


def build_qa_payload(
    *,
    scene: SceneSet,
    scene_index: int,
    resolution: float,
    rgb_spec: RasterSpec,
    chm_spec: RasterSpec,
    dsm: np.ndarray,
    point_count: np.ndarray,
    dtm: np.ndarray,
    dtm_valid: np.ndarray,
    chm: np.ndarray,
    chm_valid: np.ndarray,
    dsm_stats: dict[str, int],
    paths: OutputPaths,
) -> dict[str, object]:
    return {
        "scene_id": f"{scene_index:04d}",
        "grid_code": scene.grid_code,
        "crs": CRS,
        "requested_resolution_m": float(resolution),
        "rgb": {
            "path": str(paths.benchmark_rgb),
            "width": int(rgb_spec.width),
            "height": int(rgb_spec.height),
            "resolution_m": _spec_resolution(rgb_spec),
            "bounds": _spec_bounds(rgb_spec),
        },
        "chm": {
            "path": str(paths.benchmark_chm),
            "match_rgb_path": str(paths.benchmark_chm_match_rgb)
            if paths.benchmark_chm_match_rgb
            else None,
            "width": int(chm_spec.width),
            "height": int(chm_spec.height),
            "resolution_m": _spec_resolution(chm_spec),
            "bounds": _spec_bounds(chm_spec),
            "valid_fraction": _valid_fraction(chm_valid),
            "stats_m": _array_stats(chm, chm_valid),
        },
        "dtm": {
            "valid_fraction": _valid_fraction(dtm_valid),
            "stats_m": _array_stats(dtm, dtm_valid),
        },
        "dsm": {
            "stats": dsm_stats,
            "point_count": _point_count_stats(point_count),
            "stats_m": _array_stats(dsm, point_count > 0),
        },
        "outputs": {
            "scene_rgb": str(paths.scene_rgb),
            "scene_chm": str(paths.scene_chm),
            "diagnostic_dsm": str(paths.diagnostic_dsm),
            "diagnostic_dtm": str(paths.diagnostic_dtm),
            "diagnostic_point_count": str(paths.diagnostic_point_count),
        },
    }


def _find_product_zips(lp_root: Path, product: str) -> dict[str, Path]:
    product_root = lp_root / product
    if not product_root.exists():
        raise FileNotFoundError(f"Virtual Shizuoka product directory not found: {product_root}")
    zips: dict[str, Path] = {}
    for path in sorted(product_root.rglob("*.zip")):
        grid_code = path.stem.upper()
        if grid_code in zips:
            raise ValueError(
                f"Duplicate {product} zip for grid {grid_code}: {zips[grid_code]} and {path}"
            )
        zips[grid_code] = path.resolve()
    if not zips:
        raise FileNotFoundError(f"No {product} zips found under {product_root}")
    return zips


def _zip_members(zip_path: str | Path, suffixes: tuple[str, ...]) -> list[str]:
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    with zipfile.ZipFile(zip_path) as archive:
        return [
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(suffixes)
        ]


def _single_zip_member(zip_path: str | Path, suffixes: tuple[str, ...]) -> str:
    members = _zip_members(zip_path, suffixes)
    if len(members) != 1:
        raise ValueError(
            f"Expected exactly one {suffixes} member in {zip_path}, found {len(members)}."
        )
    return members[0]


def _extract_zip_member(zip_path: str | Path, member: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(member).name
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def _coordinate_spacing(values: np.ndarray, axis_name: str, source: Path) -> float:
    if values.size < 2:
        return DTM_GSD_M
    diffs = np.diff(values)
    spacing = float(np.median(diffs))
    if spacing <= 0:
        raise ValueError(f"Invalid {axis_name} spacing in {source}: {spacing}")
    if not np.allclose(diffs, spacing, rtol=0.0, atol=1e-4):
        raise ValueError(f"Irregular {axis_name} spacing in {source}.")
    return spacing


def _prepare_output_dirs(output_root: Path) -> None:
    for relative in (
        "benchmark",
        "scenes/images",
        "scenes/height_maps",
        "diagnostics/dsm",
        "diagnostics/dtm",
        "diagnostics/point_counts",
        "qa",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _prepare_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _check_outputs(paths: OutputPaths, *, overwrite: bool) -> None:
    candidates = [
        paths.benchmark_rgb,
        paths.benchmark_chm,
        paths.scene_rgb,
        paths.scene_chm,
        paths.diagnostic_dsm,
        paths.diagnostic_dtm,
        paths.diagnostic_point_count,
        paths.qa_json,
    ]
    if paths.benchmark_chm_match_rgb is not None:
        candidates.append(paths.benchmark_chm_match_rgb)
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        examples = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output already exists. Use --overwrite to replace: {examples}")
    if overwrite:
        for path in existing:
            path.unlink()


def _expected_output_paths(paths: OutputPaths) -> list[Path]:
    expected = [
        paths.benchmark_rgb,
        paths.benchmark_chm,
        paths.scene_rgb,
        paths.scene_chm,
        paths.diagnostic_dsm,
        paths.diagnostic_dtm,
        paths.diagnostic_point_count,
        paths.qa_json,
    ]
    if paths.benchmark_chm_match_rgb is not None:
        expected.append(paths.benchmark_chm_match_rgb)
    return expected


def _outputs_complete(paths: OutputPaths) -> tuple[bool, tuple[Path, ...]]:
    missing = tuple(path for path in _expected_output_paths(paths) if not path.exists())
    return len(missing) == 0, missing


def _remove_expected_outputs(paths: OutputPaths) -> None:
    for path in _expected_output_paths(paths):
        if path.exists() or path.is_symlink():
            path.unlink()


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _build_resume_plan(
    *,
    scenes: list[SceneSet],
    output_root: Path,
    resolution: float,
    existing_rows: list[dict[str, object]],
) -> list[ScenePlan]:
    existing_by_grid, used_scene_indices = _existing_manifest_maps(existing_rows, resolution)
    next_scene_index = max(used_scene_indices, default=0) + 1
    plans: list[ScenePlan] = []

    for scene in scenes:
        grid_code = scene.grid_code.upper()
        existing_row = existing_by_grid.get(grid_code)
        if existing_row is not None:
            scene_index = _parse_scene_id(existing_row["scene_id"])
            paths = build_output_paths(output_root, scene.grid_code, scene_index, resolution)
            complete, missing = _outputs_complete(paths)
            plans.append(
                ScenePlan(
                    scene=scene,
                    scene_index=scene_index,
                    skip_existing=complete,
                    missing_outputs=missing,
                )
            )
            continue

        scene_index = next_scene_index
        next_scene_index += 1
        plans.append(ScenePlan(scene=scene, scene_index=scene_index, skip_existing=False))

    return plans


def _existing_manifest_maps(
    rows: list[dict[str, object]],
    resolution: float,
) -> tuple[dict[str, dict[str, object]], set[int]]:
    rows_by_grid: dict[str, dict[str, object]] = {}
    scene_indices: set[int] = set()
    for row in rows:
        grid_code = str(row.get("grid_code", "")).upper()
        if not grid_code:
            raise ValueError("Existing manifest row is missing grid_code.")
        if grid_code in rows_by_grid:
            raise ValueError(f"Existing manifest contains duplicate grid_code: {grid_code}")

        scene_index = _parse_scene_id(row.get("scene_id"))
        if scene_index in scene_indices:
            raise ValueError(f"Existing manifest contains duplicate scene_id: {scene_index:04d}")

        row_resolution = row.get("requested_resolution_m")
        if row_resolution not in (None, "") and not _is_resolution(
            float(row_resolution), resolution
        ):
            raise ValueError(
                "Cannot resume into an output root created with a different resolution: "
                f"grid {grid_code} has requested_resolution_m={row_resolution}, current={resolution}."
            )

        rows_by_grid[grid_code] = row
        scene_indices.add(scene_index)
    return rows_by_grid, scene_indices


def _parse_scene_id(value: object) -> int:
    try:
        scene_index = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene_id in manifest: {value!r}") from exc
    if scene_index <= 0:
        raise ValueError(f"scene_id must be positive, got {value!r}.")
    return scene_index


def _upsert_manifest_row(
    rows: list[dict[str, object]],
    new_row: dict[str, object],
) -> list[dict[str, object]]:
    grid_code = str(new_row["grid_code"]).upper()
    updated: list[dict[str, object]] = []
    inserted = False
    for row in rows:
        if str(row.get("grid_code", "")).upper() == grid_code:
            updated.append(new_row)
            inserted = True
        else:
            updated.append(row)
    if not inserted:
        updated.append(new_row)
    return _sort_manifest_rows(updated)


def _sort_manifest_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: _parse_scene_id(row.get("scene_id")))


def _same_grid(left: RasterSpec, right: RasterSpec) -> bool:
    return (
        int(left.width) == int(right.width)
        and int(left.height) == int(right.height)
        and np.allclose(tuple(left.transform), tuple(right.transform), rtol=0.0, atol=1e-9)
    )


def _spec_resolution(spec: RasterSpec) -> dict[str, float]:
    return {"x": float(spec.transform.a), "y": abs(float(spec.transform.e))}


def _spec_bounds(spec: RasterSpec) -> dict[str, float]:
    from rasterio.transform import array_bounds

    west, south, east, north = array_bounds(spec.height, spec.width, spec.transform)
    return {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}


def _valid_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(mask.sum() / mask.size) if mask.size else 0.0


def _array_stats(values: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | None]:
    valid_values = np.asarray(values, dtype=np.float32)[np.asarray(valid_mask, dtype=bool)]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        return {"min": None, "max": None, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "p50": float(np.percentile(valid_values, 50)),
        "p95": float(np.percentile(valid_values, 95)),
        "p99": float(np.percentile(valid_values, 99)),
    }


def _point_count_stats(point_count: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(point_count)
    positive = values[values > 0]
    if positive.size == 0:
        return {
            "valid_cells": 0,
            "total_cells": int(values.size),
            "valid_fraction": 0.0,
            "min_positive": 0,
            "max": 0,
            "mean_positive": 0.0,
        }
    return {
        "valid_cells": int(positive.size),
        "total_cells": int(values.size),
        "valid_fraction": float(positive.size / values.size),
        "min_positive": int(positive.min()),
        "max": int(positive.max()),
        "mean_positive": float(positive.mean()),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in _sort_manifest_rows(rows):
            writer.writerow(
                {fieldname: row.get(fieldname, "") for fieldname in MANIFEST_FIELDNAMES}
            )


def _manifest_row(
    scene: SceneSet,
    scene_index: int,
    resolution: float,
    paths: OutputPaths,
    qa: dict[str, object],
) -> dict[str, object]:
    chm_qa = qa["chm"]
    dsm_qa = qa["dsm"]
    return {
        "scene_id": f"{scene_index:04d}",
        "grid_code": scene.grid_code,
        "requested_resolution_m": resolution,
        "benchmark_rgb": str(paths.benchmark_rgb),
        "benchmark_chm": str(paths.benchmark_chm),
        "benchmark_chm_match_rgb": str(paths.benchmark_chm_match_rgb)
        if paths.benchmark_chm_match_rgb
        else "",
        "scene_rgb": str(paths.scene_rgb),
        "scene_chm": str(paths.scene_chm),
        "chm_valid_fraction": chm_qa["valid_fraction"],
        "chm_max_m": chm_qa["stats_m"]["max"],
        "dsm_valid_fraction": dsm_qa["point_count"]["valid_fraction"],
        "qa_json": str(paths.qa_json),
    }


def _require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("rasterio", "laspy", "numpy", "scipy", "pyproj"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing required preprocessing package(s): "
            f"{', '.join(missing)}. Install the SynthForest geospatial dependencies "
            "in the active environment."
        )


def _is_resolution(value: float, expected: float) -> bool:
    return abs(float(value) - float(expected)) <= 1e-9


def _parse_ignore_classes(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_scene_codes(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    codes: list[str] = []
    for value in values:
        codes.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return tuple(codes)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> PreprocessOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--year", default="2025")
    parser.add_argument(
        "--resolution", type=float, choices=SUPPORTED_RESOLUTIONS, default=DTM_GSD_M
    )
    parser.add_argument("--min-dsm-points", type=int, default=1)
    parser.add_argument("--ignore-classes", default="7,18")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--scene-code",
        action="append",
        default=None,
        help="Grid code filter; can be repeated or comma-separated.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Resume from output_root/manifest.csv: preserve existing scene IDs, "
            "skip complete outputs, and rebuild incomplete outputs."
        ),
    )
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    return PreprocessOptions(
        input_root=args.input_root,
        output_root=args.output_root,
        year=args.year,
        resolution=args.resolution,
        min_dsm_points=args.min_dsm_points,
        ignore_classes=_parse_ignore_classes(args.ignore_classes),
        chunk_size=args.chunk_size,
        max_scenes=args.max_scenes,
        scene_codes=_parse_scene_codes(args.scene_code),
        work_dir=args.work_dir,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        keep_work=args.keep_work,
    )


def main() -> None:
    _configure_logging()
    options = parse_args()
    run_preprocessing(options)


if __name__ == "__main__":
    main()
