#!/usr/bin/env python3
"""Download France MNH LiDAR HD and Ortho Express into paired height scenes.

This script has two stages:

1. ``discover`` queries the GeoPF MNH WFS tile index for curated forest AOIs and
   writes a reproducible 1 km scene index CSV.
2. ``download`` reads that CSV, downloads the native 50 cm MNH/CHM tile and the
   native 20 cm Ortho Express RGB, creates a CHM matched to the RGB grid, and
   hardlinks scene-style pairs for ``synthforest prepare tiles``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from synthforest.paths import data_path

LOGGER = logging.getLogger("preprocess_france_height")

DEFAULT_OUTPUT_ROOT = data_path("CHM_rasters_20cm/France_CHM_OrthoExpress-20cm_MNH-50cm")
DEFAULT_SCENE_INDEX = DEFAULT_OUTPUT_ROOT / "metadata" / "france_mnh_orthoexpress_scene_index.csv"
DEFAULT_AOI_SUMMARY = DEFAULT_OUTPUT_ROOT / "metadata" / "france_mnh_orthoexpress_aoi_summary.json"
DEFAULT_TARGET_SCENES = 50

CRS = "EPSG:2154"
MNH_WFS_URL = "https://data.geopf.fr/wfs/ows"
WMS_URL = "https://data.geopf.fr/wms-r"
MNH_LAYER = "IGNF_MNH-LIDAR-HD:dalle"
ORTHO_EXPRESS_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2025"
NODATA = -9999.0
RGB_GSD_M = 0.2
CHM_GSD_M = 0.5
WMS_MAX_SIZE = 5010
USER_AGENT = "synthforest-france-height-preprocess/1.0"
DEFAULT_MIN_RGB_VALID_FRACTION = 0.5


@dataclass(frozen=True)
class AoiSpec:
    name: str
    minx: float
    miny: float
    maxx: float
    maxy: float
    target_scenes: int = 0
    rgb_layer: str | None = None


@dataclass(frozen=True)
class OutputPaths:
    benchmark_rgb: Path
    benchmark_chm: Path
    benchmark_chm_match_rgb: Path
    scene_rgb: Path
    scene_chm: Path
    qa_json: Path


DEFAULT_AOIS: tuple[AoiSpec, ...] = (
    AoiSpec(
        name="foret_orient_troyes",
        minx=803000.0,
        miny=6791000.0,
        maxx=824000.0,
        maxy=6810000.0,
        target_scenes=50,
        rgb_layer=ORTHO_EXPRESS_LAYER,
    ),
)


SCENE_INDEX_FIELDNAMES = [
    "scene_id",
    "aoi_name",
    "source_name",
    "mnh_name_download",
    "mnh_timestamp",
    "mnh_url",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "center_x",
    "center_y",
    "projection",
    "mnh_width",
    "mnh_height",
    "chm_gsd_m",
    "rgb_layer",
    "rgb_url",
    "rgb_width",
    "rgb_height",
    "rgb_gsd_m",
    "selected_rank",
]

MANIFEST_FIELDNAMES = [
    "scene_id",
    "aoi_name",
    "source_name",
    "mnh_timestamp",
    "benchmark_rgb",
    "benchmark_chm",
    "benchmark_chm_match_rgb",
    "scene_rgb",
    "scene_chm",
    "chm_valid_fraction",
    "chm_max_m",
    "qa_json",
]


def discover_scene_index(
    *,
    output_path: str | Path = DEFAULT_SCENE_INDEX,
    summary_path: str | Path = DEFAULT_AOI_SUMMARY,
    aois: Iterable[AoiSpec] = DEFAULT_AOIS,
    target_scenes: int = DEFAULT_TARGET_SCENES,
    rgb_layer: str | None = None,
    overwrite: bool = False,
    timeout: float = 120.0,
    retries: int = 3,
) -> Path:
    """Query GeoPF MNH tile metadata and write a 1 km scene index CSV."""
    output_path = Path(output_path).expanduser()
    summary_path = Path(summary_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Scene index already exists. Use --overwrite to replace: {output_path}"
        )

    aoi_list = list(aois)
    if target_scenes <= 0:
        raise ValueError("target_scenes must be positive.")
    if not aoi_list:
        raise ValueError("At least one AOI is required.")

    queried: dict[str, list[dict[str, object]]] = {}
    summary: dict[str, object] = {
        "crs": CRS,
        "mnh_layer": MNH_LAYER,
        "rgb_layer": rgb_layer or "per-aoi",
        "target_scenes": int(target_scenes),
        "aois": [],
    }

    for aoi in aoi_list:
        LOGGER.info("Querying MNH WFS tiles for AOI %s", aoi.name)
        features = query_mnh_features(aoi, timeout=timeout, retries=retries)
        candidate_rows = []
        aoi_rgb_layer = rgb_layer or aoi.rgb_layer or ORTHO_EXPRESS_LAYER
        for feature in features:
            row = scene_index_row_from_feature(feature, aoi.name, aoi_rgb_layer)
            if row is not None:
                candidate_rows.append(row)
        candidate_rows = _deduplicate_rows(candidate_rows)
        queried[aoi.name] = candidate_rows
        summary["aois"].append(
            {
                "name": aoi.name,
                "bbox": [aoi.minx, aoi.miny, aoi.maxx, aoi.maxy],
                "target_scenes": aoi.target_scenes,
                "rgb_layer": aoi_rgb_layer,
                "candidate_tiles": len(candidate_rows),
                "timestamps": sorted(
                    {row["mnh_timestamp"] for row in candidate_rows if row["mnh_timestamp"]}
                ),
            }
        )

    allocations = _allocate_counts(aoi_list, target_scenes)
    selected: list[dict[str, object]] = []
    used_sources: set[str] = set()

    for aoi in aoi_list:
        rows = [row for row in queried[aoi.name] if row["source_name"] not in used_sources]
        rows = _spatially_sort_rows(rows)
        picked = _evenly_sample(rows, allocations.get(aoi.name, 0))
        for row in picked:
            used_sources.add(str(row["source_name"]))
        selected.extend(picked)

    deficit = target_scenes - len(selected)
    if deficit > 0:
        overflow = []
        for aoi in aoi_list:
            overflow.extend(
                row for row in queried[aoi.name] if row["source_name"] not in used_sources
            )
        for row in _evenly_sample(_spatially_sort_rows(overflow), deficit):
            if row["source_name"] in used_sources:
                continue
            used_sources.add(str(row["source_name"]))
            selected.append(row)

    if len(selected) < target_scenes:
        LOGGER.warning(
            "Selected %d scenes, fewer than requested %d. Available AOI tiles may be limited.",
            len(selected),
            target_scenes,
        )

    selected = _spatially_sort_rows(selected)[:target_scenes]
    for index, row in enumerate(selected, start=1):
        row["scene_id"] = f"{index:04d}"
        row["selected_rank"] = index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCENE_INDEX_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row.get(field, "") for field in SCENE_INDEX_FIELDNAMES})

    summary["selected_scenes"] = len(selected)
    summary["selected_by_aoi"] = {
        _slug(aoi.name): sum(1 for row in selected if row["aoi_name"] == _slug(aoi.name))
        for aoi in aoi_list
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    LOGGER.info("Wrote scene index with %d scenes: %s", len(selected), output_path)
    LOGGER.info("Wrote AOI summary: %s", summary_path)
    return output_path


def download_scene_index(
    *,
    scene_index_path: str | Path = DEFAULT_SCENE_INDEX,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_scenes: int | None = None,
    scene_ids: Iterable[str] = (),
    overwrite: bool = False,
    skip_existing: bool = False,
    prune_stale: bool = False,
    min_rgb_valid_fraction: float = DEFAULT_MIN_RGB_VALID_FRACTION,
    timeout: float = 300.0,
    retries: int = 3,
) -> Path:
    """Download and organize all scenes listed in a scene index CSV."""
    scene_index_path = Path(scene_index_path).expanduser()
    output_root = Path(output_root).expanduser()
    rows = read_scene_index(scene_index_path)
    selected_ids = {_normalize_scene_id(value) for value in scene_ids}
    if selected_ids:
        rows = [row for row in rows if _normalize_scene_id(row["scene_id"]) in selected_ids]
    if max_scenes is not None:
        rows = rows[: int(max_scenes)]
    if not rows:
        raise ValueError(f"No scene rows selected from {scene_index_path}")

    _prepare_output_dirs(output_root)
    if prune_stale:
        prune_stale_outputs(output_root, rows)

    manifest_path = output_root / "manifest.csv"
    existing_manifest = _read_manifest(manifest_path)
    selected_manifest_keys = {_manifest_identity(row) for row in rows}
    manifest_rows = {
        str(row.get("scene_id")): row
        for row in existing_manifest
        if row.get("scene_id")
        and (not prune_stale or _manifest_identity(row) in selected_manifest_keys)
    }

    processed = 0
    skipped = 0
    for row in rows:
        paths = build_output_paths(output_root, row)
        if overwrite:
            _remove_outputs(paths)
        complete, _missing = outputs_complete(paths)
        if skip_existing and complete:
            try:
                validate_scene_outputs(paths, min_rgb_valid_fraction=min_rgb_valid_fraction)
            except Exception as exc:  # noqa: BLE001 - repair incomplete/bad resume outputs
                LOGGER.warning(
                    "Existing scene %s is invalid and will be rebuilt: %s", row["scene_id"], exc
                )
                _remove_outputs(paths)
                complete = False
            else:
                LOGGER.info("Skipping existing complete scene %s", row["scene_id"])
                skipped += 1
                manifest_rows[row["scene_id"]] = manifest_row(row, paths, _read_json(paths.qa_json))
                continue
        elif skip_existing and not complete:
            stale_expected = [
                path for path in _expected_output_paths(paths) if path.exists() or path.is_symlink()
            ]
            if stale_expected:
                LOGGER.info(
                    "Removing partial/stale outputs for scene %s before rebuild.",
                    row["scene_id"],
                )
                _remove_outputs(paths)

        LOGGER.info("Processing France scene %s (%s)", row["scene_id"], row["source_name"])
        qa = process_scene(
            row,
            paths,
            overwrite=overwrite,
            min_rgb_valid_fraction=min_rgb_valid_fraction,
            timeout=timeout,
            retries=retries,
        )
        manifest_rows[row["scene_id"]] = manifest_row(row, paths, qa)
        _write_manifest(manifest_path, list(manifest_rows.values()))
        _write_manifest(output_root / "scenes" / "manifest.csv", list(manifest_rows.values()))
        processed += 1

    _write_manifest(manifest_path, list(manifest_rows.values()))
    _write_manifest(output_root / "scenes" / "manifest.csv", list(manifest_rows.values()))
    LOGGER.info(
        "Processed %d scene(s), skipped %d existing scene(s). Output: %s",
        processed,
        skipped,
        output_root,
    )
    return output_root


def prune_output_root(
    *,
    scene_index_path: str | Path = DEFAULT_SCENE_INDEX,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_scenes: int | None = None,
    scene_ids: Iterable[str] = (),
) -> Path:
    """Remove France outputs and manifest rows not present in the selected scene index."""
    scene_index_path = Path(scene_index_path).expanduser()
    output_root = Path(output_root).expanduser()
    rows = read_scene_index(scene_index_path)
    selected_ids = {_normalize_scene_id(value) for value in scene_ids}
    if selected_ids:
        rows = [row for row in rows if _normalize_scene_id(row["scene_id"]) in selected_ids]
    if max_scenes is not None:
        rows = rows[: int(max_scenes)]
    if not rows:
        raise ValueError(f"No scene rows selected from {scene_index_path}")

    removed = prune_stale_outputs(output_root, rows)
    selected_manifest_keys = {_manifest_identity(row) for row in rows}
    for manifest_path in (output_root / "manifest.csv", output_root / "scenes" / "manifest.csv"):
        existing_rows = _read_manifest(manifest_path)
        kept_rows = [
            row for row in existing_rows if _manifest_identity(row) in selected_manifest_keys
        ]
        if existing_rows or manifest_path.exists():
            _write_manifest(manifest_path, kept_rows)
            LOGGER.info(
                "Pruned manifest %s: kept %d of %d row(s).",
                manifest_path,
                len(kept_rows),
                len(existing_rows),
            )
    LOGGER.info("Prune complete. Removed %d stale file(s).", len(removed))
    return output_root


def process_scene(
    row: dict[str, str],
    paths: OutputPaths,
    *,
    overwrite: bool,
    min_rgb_valid_fraction: float,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    """Download one scene and write QA metadata."""
    paths.benchmark_rgb.parent.mkdir(parents=True, exist_ok=True)
    paths.benchmark_chm.parent.mkdir(parents=True, exist_ok=True)
    paths.qa_json.parent.mkdir(parents=True, exist_ok=True)

    download_url(
        row["mnh_url"], paths.benchmark_chm, overwrite=overwrite, timeout=timeout, retries=retries
    )
    validate_geotiff(paths.benchmark_chm)
    download_url(
        row["rgb_url"], paths.benchmark_rgb, overwrite=overwrite, timeout=timeout, retries=retries
    )
    validate_geotiff(paths.benchmark_rgb)
    try:
        validate_rgb_content(paths.benchmark_rgb, min_valid_fraction=min_rgb_valid_fraction)
    except ValueError:
        if overwrite:
            raise
        paths.benchmark_rgb.unlink(missing_ok=True)
        download_url(
            row["rgb_url"], paths.benchmark_rgb, overwrite=True, timeout=timeout, retries=retries
        )
        validate_rgb_content(paths.benchmark_rgb, min_valid_fraction=min_rgb_valid_fraction)

    reproject_chm_to_match_rgb(
        chm_path=paths.benchmark_chm,
        rgb_path=paths.benchmark_rgb,
        output_path=paths.benchmark_chm_match_rgb,
        overwrite=overwrite,
    )

    hardlink_replace(paths.benchmark_rgb, paths.scene_rgb, overwrite=overwrite)
    hardlink_replace(paths.benchmark_chm_match_rgb, paths.scene_chm, overwrite=overwrite)

    qa = build_qa_payload(row, paths)
    _write_json(paths.qa_json, qa)
    return qa


def query_mnh_features(
    aoi: AoiSpec, *, timeout: float = 120.0, retries: int = 3
) -> list[dict[str, object]]:
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": MNH_LAYER,
        "SRSNAME": CRS,
        "BBOX": f"{aoi.minx:g},{aoi.miny:g},{aoi.maxx:g},{aoi.maxy:g},{CRS}",
        "OUTPUTFORMAT": "application/json",
        "COUNT": "10000",
    }
    payload = fetch_json(f"{MNH_WFS_URL}?{urlencode(params)}", timeout=timeout, retries=retries)
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Unexpected WFS response for {aoi.name}: features is not a list.")
    return features


def scene_index_row_from_feature(
    feature: dict[str, object],
    aoi_name: str,
    rgb_layer: str = ORTHO_EXPRESS_LAYER,
) -> dict[str, object] | None:
    props = feature.get("properties")
    if not isinstance(props, dict):
        return None
    mnh_url = str(props.get("url") or "")
    if not mnh_url:
        return None

    bbox_value = props.get("bbox")
    if bbox_value:
        minx, miny, maxx, maxy = parse_bbox(str(bbox_value))
    else:
        minx, miny, maxx, maxy = feature_geometry_bounds(feature)

    mnh_width = int(props.get("width") or _grid_size(minx, maxx, CHM_GSD_M))
    mnh_height = int(props.get("height") or _grid_size(miny, maxy, CHM_GSD_M))
    rgb_width = _grid_size(minx, maxx, RGB_GSD_M)
    rgb_height = _grid_size(miny, maxy, RGB_GSD_M)
    if rgb_width > WMS_MAX_SIZE or rgb_height > WMS_MAX_SIZE:
        LOGGER.warning(
            "Skipping %s because RGB WMS size would be %dx%d (> %d).",
            props.get("name"),
            rgb_width,
            rgb_height,
            WMS_MAX_SIZE,
        )
        return None

    source_name = str(props.get("name") or Path(str(props.get("name_download") or "")).stem)
    return {
        "scene_id": "",
        "aoi_name": _slug(aoi_name),
        "source_name": source_name,
        "mnh_name_download": str(props.get("name_download") or f"{source_name}.tif"),
        "mnh_timestamp": str(props.get("timestamp") or ""),
        "mnh_url": mnh_url,
        "bbox_minx": f"{minx:.3f}",
        "bbox_miny": f"{miny:.3f}",
        "bbox_maxx": f"{maxx:.3f}",
        "bbox_maxy": f"{maxy:.3f}",
        "center_x": f"{((minx + maxx) / 2.0):.3f}",
        "center_y": f"{((miny + maxy) / 2.0):.3f}",
        "projection": str(props.get("projection") or CRS),
        "mnh_width": mnh_width,
        "mnh_height": mnh_height,
        "chm_gsd_m": CHM_GSD_M,
        "rgb_layer": rgb_layer,
        "rgb_url": build_wms_url(
            layer=rgb_layer,
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            width=rgb_width,
            height=rgb_height,
        ),
        "rgb_width": rgb_width,
        "rgb_height": rgb_height,
        "rgb_gsd_m": RGB_GSD_M,
        "selected_rank": "",
    }


def build_wms_url(
    *,
    layer: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    width: int,
    height: int,
) -> str:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "FORMAT": "image/geotiff",
        "STYLES": "",
        "CRS": CRS,
        "BBOX": ",".join(_format_coord(value) for value in (minx, miny, maxx, maxy)),
        "WIDTH": str(int(width)),
        "HEIGHT": str(int(height)),
    }
    return f"{WMS_URL}?{urlencode(params)}"


def build_output_paths(output_root: str | Path, row: dict[str, str]) -> OutputPaths:
    output_root = Path(output_root).expanduser()
    scene_id = _normalize_scene_id(row["scene_id"])
    stem = build_scene_stem(row)
    return OutputPaths(
        benchmark_rgb=output_root / "benchmark" / f"{stem}_RGB.tif",
        benchmark_chm=output_root / "benchmark" / f"{stem}_CHM.tif",
        benchmark_chm_match_rgb=output_root / "benchmark" / f"{stem}_CHM_matchRGB.tif",
        scene_rgb=output_root / "scenes" / "images" / f"rgb_france{scene_id}.tif",
        scene_chm=output_root / "scenes" / "height_maps" / f"height_map_france{scene_id}.tif",
        qa_json=output_root / "qa" / f"{stem}_qa.json",
    )


def build_scene_stem(row: dict[str, str]) -> str:
    scene_id = _normalize_scene_id(row["scene_id"])
    aoi = _slug(str(row.get("aoi_name") or "france"))
    source = Path(str(row.get("mnh_name_download") or row.get("source_name") or "mnh")).stem
    return f"France_{aoi}_{scene_id}_{_slug(source)}"


def read_scene_index(path: str | Path) -> list[dict[str, str]]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Scene index not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"Scene index is empty: {path}")
    required = {
        "scene_id",
        "mnh_url",
        "rgb_url",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Scene index is missing required column(s): {', '.join(sorted(missing))}")
    return rows


def fetch_json(url: str, *, timeout: float, retries: int) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            text = payload.decode("utf-8")
            if text.lstrip().startswith("<"):
                raise RuntimeError(_compact_xml_error(text))
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object response.")
            return parsed
        except Exception as exc:  # noqa: BLE001 - retry and preserve final error
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}") from last_error


def download_url(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool,
    timeout: float,
    retries: int,
) -> Path:
    output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        validate_geotiff(output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
            _raise_if_wms_exception(temp_path)
            validate_geotiff(temp_path)
            temp_path.replace(output_path)
            return output_path
        except Exception as exc:  # noqa: BLE001 - retry and preserve final error
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url} -> {output_path}: {last_error}") from last_error


def validate_geotiff(path: str | Path) -> None:
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        if src.width <= 0 or src.height <= 0:
            raise ValueError(f"Invalid raster shape for {path}: {src.width}x{src.height}")
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")


def validate_rgb_content(path: str | Path, *, min_valid_fraction: float) -> float:
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 RGB bands in {path}, got {src.count}.")
        # Read a decimated view to avoid scanning 75 MB for every 1 km tile.
        out_height = min(src.height, 512)
        out_width = min(src.width, 512)
        data = src.read(
            indexes=[1, 2, 3],
            out_shape=(3, out_height, out_width),
            masked=False,
        )
        finite = np.isfinite(data)
        if src.nodata is not None:
            finite &= data != src.nodata
        valid_pixels = finite.all(axis=0)
        valid_fraction = float(valid_pixels.sum() / valid_pixels.size) if valid_pixels.size else 0.0
    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"RGB raster has too little valid imagery: {path} "
            f"valid_fraction={valid_fraction:.4f}, required={min_valid_fraction:.4f}."
        )
    return valid_fraction


def validate_scene_outputs(paths: OutputPaths, *, min_rgb_valid_fraction: float) -> None:
    validate_geotiff(paths.benchmark_chm)
    validate_geotiff(paths.benchmark_chm_match_rgb)
    validate_geotiff(paths.benchmark_rgb)
    validate_rgb_content(paths.benchmark_rgb, min_valid_fraction=min_rgb_valid_fraction)
    validate_geotiff(paths.scene_chm)
    validate_geotiff(paths.scene_rgb)


def reproject_chm_to_match_rgb(
    *,
    chm_path: str | Path,
    rgb_path: str | Path,
    output_path: str | Path,
    overwrite: bool,
) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        validate_geotiff(output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(chm_path) as chm, rasterio.open(rgb_path) as rgb:
        dtype = chm.dtypes[0]
        nodata = chm.nodata if chm.nodata is not None else NODATA
        destination = np.full((rgb.height, rgb.width), nodata, dtype=dtype)
        reproject(
            source=rasterio.band(chm, 1),
            destination=destination,
            src_transform=chm.transform,
            src_crs=chm.crs,
            src_nodata=chm.nodata,
            dst_transform=rgb.transform,
            dst_crs=rgb.crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )
        profile = chm.profile.copy()
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(
            driver="GTiff",
            width=rgb.width,
            height=rgb.height,
            transform=rgb.transform,
            crs=rgb.crs,
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            predictor=2 if np.dtype(dtype).kind == "f" else 1,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination, 1)


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


def build_qa_payload(row: dict[str, str], paths: OutputPaths) -> dict[str, object]:
    chm_summary = raster_summary(paths.benchmark_chm, include_stats=True)
    rgb_summary = raster_summary(paths.benchmark_rgb, include_stats=False)
    match_summary = raster_summary(paths.benchmark_chm_match_rgb, include_stats=True)
    return {
        "scene_id": _normalize_scene_id(row["scene_id"]),
        "aoi_name": row.get("aoi_name", ""),
        "source_name": row.get("source_name", ""),
        "mnh_timestamp": row.get("mnh_timestamp", ""),
        "crs": CRS,
        "bbox": {
            "minx": float(row["bbox_minx"]),
            "miny": float(row["bbox_miny"]),
            "maxx": float(row["bbox_maxx"]),
            "maxy": float(row["bbox_maxy"]),
        },
        "sources": {
            "mnh_url": row.get("mnh_url", ""),
            "rgb_layer": row.get("rgb_layer", ""),
            "rgb_url": row.get("rgb_url", ""),
        },
        "rgb": rgb_summary,
        "chm": chm_summary,
        "chm_match_rgb": match_summary,
        "outputs": {
            "benchmark_rgb": str(paths.benchmark_rgb),
            "benchmark_chm": str(paths.benchmark_chm),
            "benchmark_chm_match_rgb": str(paths.benchmark_chm_match_rgb),
            "scene_rgb": str(paths.scene_rgb),
            "scene_chm": str(paths.scene_chm),
        },
    }


def raster_summary(path: str | Path, *, include_stats: bool) -> dict[str, object]:
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        bounds = src.bounds
        payload: dict[str, object] = {
            "path": str(path),
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "crs": str(src.crs),
            "resolution_m": {"x": float(src.transform.a), "y": abs(float(src.transform.e))},
            "bounds": {
                "west": float(bounds.left),
                "south": float(bounds.bottom),
                "east": float(bounds.right),
                "north": float(bounds.top),
            },
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }
        if include_stats:
            values = src.read(1)
            valid = np.isfinite(values)
            if src.nodata is not None:
                valid &= values != src.nodata
            payload["valid_fraction"] = float(valid.sum() / valid.size) if valid.size else 0.0
            payload["stats_m"] = _array_stats(values, valid)
    return payload


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError(f"Expected bbox as minx,miny,maxx,maxy, got: {value!r}")
    minx, miny, maxx, maxy = parts
    if not (minx < maxx and miny < maxy):
        raise ValueError(f"Invalid bbox: {value!r}")
    return minx, miny, maxx, maxy


def feature_geometry_bounds(feature: dict[str, object]) -> tuple[float, float, float, float]:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Feature has neither properties.bbox nor geometry.")
    coords = geometry.get("coordinates")
    flat = list(_iter_coordinate_pairs(coords))
    if not flat:
        raise ValueError("Feature geometry has no coordinates.")
    xs = [point[0] for point in flat]
    ys = [point[1] for point in flat]
    return min(xs), min(ys), max(xs), max(ys)


def outputs_complete(paths: OutputPaths) -> tuple[bool, tuple[Path, ...]]:
    missing = tuple(path for path in _expected_output_paths(paths) if not path.exists())
    return len(missing) == 0, missing


def prune_stale_outputs(output_root: str | Path, rows: list[dict[str, str]]) -> list[Path]:
    """Remove France outputs not expected by the current scene index selection."""
    output_root = Path(output_root).expanduser()
    expected: set[Path] = set()
    for row in rows:
        expected.update(_expected_output_paths(build_output_paths(output_root, row)))

    candidates: list[Path] = []
    for directory, pattern in (
        (output_root / "benchmark", "France_*.tif"),
        (output_root / "qa", "France_*_qa.json"),
        (output_root / "scenes" / "images", "rgb_france*.tif"),
        (output_root / "scenes" / "height_maps", "height_map_france*.tif"),
    ):
        if directory.exists():
            candidates.extend(
                path for path in directory.glob(pattern) if path.is_file() or path.is_symlink()
            )

    removed: list[Path] = []
    for path in sorted(candidates):
        if path not in expected:
            path.unlink()
            removed.append(path)
    if removed:
        LOGGER.info("Pruned %d stale France output file(s).", len(removed))
    return removed


def _expected_output_paths(paths: OutputPaths) -> tuple[Path, ...]:
    return (
        paths.benchmark_rgb,
        paths.benchmark_chm,
        paths.benchmark_chm_match_rgb,
        paths.scene_rgb,
        paths.scene_chm,
        paths.qa_json,
    )


def manifest_row(
    row: dict[str, str], paths: OutputPaths, qa: dict[str, object]
) -> dict[str, object]:
    chm = qa.get("chm", {})
    stats = chm.get("stats_m", {}) if isinstance(chm, dict) else {}
    return {
        "scene_id": _normalize_scene_id(row["scene_id"]),
        "aoi_name": row.get("aoi_name", ""),
        "source_name": row.get("source_name", ""),
        "mnh_timestamp": row.get("mnh_timestamp", ""),
        "benchmark_rgb": str(paths.benchmark_rgb),
        "benchmark_chm": str(paths.benchmark_chm),
        "benchmark_chm_match_rgb": str(paths.benchmark_chm_match_rgb),
        "scene_rgb": str(paths.scene_rgb),
        "scene_chm": str(paths.scene_chm),
        "chm_valid_fraction": chm.get("valid_fraction", "") if isinstance(chm, dict) else "",
        "chm_max_m": stats.get("max", "") if isinstance(stats, dict) else "",
        "qa_json": str(paths.qa_json),
    }


def _manifest_identity(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        _normalize_scene_id(str(row.get("scene_id", "0"))),
        str(row.get("aoi_name", "")),
        str(row.get("source_name", "")),
    )


def load_aois(path: str | Path | None) -> tuple[AoiSpec, ...]:
    if path is None:
        return DEFAULT_AOIS
    path = Path(path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("aois", [])
    if not isinstance(payload, list):
        raise ValueError("AOI JSON must be a list or an object with an 'aois' list.")
    aois = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each AOI entry must be an object.")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"AOI {item.get('name')} must contain bbox [minx, miny, maxx, maxy].")
        aois.append(
            AoiSpec(
                name=str(item["name"]),
                minx=float(bbox[0]),
                miny=float(bbox[1]),
                maxx=float(bbox[2]),
                maxy=float(bbox[3]),
                target_scenes=int(item.get("target_scenes", 0) or 0),
                rgb_layer=str(item["rgb_layer"]) if item.get("rgb_layer") else None,
            )
        )
    return tuple(aois)


def _prepare_output_dirs(output_root: Path) -> None:
    for relative in ("benchmark", "scenes/images", "scenes/height_maps", "qa"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: _normalize_scene_id(str(row.get("scene_id", "0"))))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDNAMES})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_outputs(paths: OutputPaths) -> None:
    for path in (
        paths.benchmark_rgb,
        paths.benchmark_chm,
        paths.benchmark_chm_match_rgb,
        paths.scene_rgb,
        paths.scene_chm,
        paths.qa_json,
    ):
        if path.exists() or path.is_symlink():
            path.unlink()


def _array_stats(values: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | None]:
    valid_values = np.asarray(values)[np.asarray(valid_mask, dtype=bool)]
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


def _deduplicate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for row in rows:
        unique[str(row["source_name"])] = row
    return list(unique.values())


def _spatially_sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("aoi_name", "")),
            float(row.get("center_y", 0.0)),
            float(row.get("center_x", 0.0)),
            str(row.get("source_name", "")),
        ),
    )


def _evenly_sample(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return list(rows)
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]
    selected = []
    seen = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        selected.append(rows[index])
    cursor = 0
    while len(selected) < limit and cursor < len(rows):
        if cursor not in seen:
            selected.append(rows[cursor])
        cursor += 1
    return selected[:limit]


def _allocate_counts(aois: list[AoiSpec], target_scenes: int) -> dict[str, int]:
    weights = [aoi.target_scenes if aoi.target_scenes > 0 else 1 for aoi in aois]
    total_weight = sum(weights)
    raw = [target_scenes * weight / total_weight for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = target_scenes - sum(counts)
    order = sorted(range(len(aois)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return {aoi.name: count for aoi, count in zip(aois, counts, strict=True)}


def _grid_size(min_value: float, max_value: float, resolution: float) -> int:
    size = int(round((max_value - min_value) / resolution))
    if size <= 0:
        raise ValueError(f"Invalid grid size for {min_value}, {max_value}, resolution {resolution}")
    return size


def _format_coord(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _iter_coordinate_pairs(value: object) -> Iterable[tuple[float, float]]:
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            yield float(value[0]), float(value[1])
        else:
            for item in value:
                yield from _iter_coordinate_pairs(item)


def _raise_if_wms_exception(path: Path) -> None:
    with path.open("rb") as f:
        prefix = f.read(512)
    if prefix.lstrip().startswith(b"<"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        raise RuntimeError(_compact_xml_error(text))


def _compact_xml_error(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    match = re.search(r"<ServiceException[^>]*>(.*?)</ServiceException>", text)
    if match:
        return match.group(1).strip()
    return text[:500]


def _normalize_scene_id(value: str) -> str:
    return f"{int(str(value)):04d}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "france"


def _require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("rasterio", "numpy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing required package(s): "
            f"{', '.join(missing)}. Activate the synthforest environment or install them."
        )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover", help="Query MNH WFS metadata and write a scene index CSV."
    )
    discover.add_argument("--output", type=Path, default=DEFAULT_SCENE_INDEX)
    discover.add_argument("--summary-output", type=Path, default=DEFAULT_AOI_SUMMARY)
    discover.add_argument("--aoi-json", type=Path, default=None)
    discover.add_argument("--target-scenes", type=int, default=DEFAULT_TARGET_SCENES)
    discover.add_argument(
        "--rgb-layer",
        type=str,
        default=None,
        help="Override the RGB WMS layer for every AOI. By default each built-in AOI uses a verified Ortho Express year.",
    )
    discover.add_argument("--overwrite", action="store_true")
    discover.add_argument("--timeout", type=float, default=120.0)
    discover.add_argument("--retries", type=int, default=3)

    download = subparsers.add_parser(
        "download", help="Download and organize scenes from a scene index CSV."
    )
    download.add_argument("--scene-index", type=Path, default=DEFAULT_SCENE_INDEX)
    download.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    download.add_argument("--max-scenes", type=int, default=None)
    download.add_argument("--scene-id", action="append", default=[])
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--skip-existing", action="store_true")
    download.add_argument(
        "--prune-stale",
        action="store_true",
        help="Remove existing France outputs not present in the selected scene index before downloading.",
    )
    download.add_argument(
        "--min-rgb-valid-fraction", type=float, default=DEFAULT_MIN_RGB_VALID_FRACTION
    )
    download.add_argument("--timeout", type=float, default=300.0)
    download.add_argument("--retries", type=int, default=3)

    prune = subparsers.add_parser(
        "prune", help="Remove stale France outputs not present in a scene index."
    )
    prune.add_argument("--scene-index", type=Path, default=DEFAULT_SCENE_INDEX)
    prune.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prune.add_argument("--max-scenes", type=int, default=None)
    prune.add_argument("--scene-id", action="append", default=[])

    return parser.parse_args()


def main() -> None:
    _configure_logging()
    _require_runtime_dependencies()
    args = parse_args()

    if args.command == "discover":
        discover_scene_index(
            output_path=args.output,
            summary_path=args.summary_output,
            aois=load_aois(args.aoi_json),
            target_scenes=args.target_scenes,
            rgb_layer=args.rgb_layer,
            overwrite=args.overwrite,
            timeout=args.timeout,
            retries=args.retries,
        )
    elif args.command == "download":
        download_scene_index(
            scene_index_path=args.scene_index,
            output_root=args.output_root,
            max_scenes=args.max_scenes,
            scene_ids=args.scene_id,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            prune_stale=args.prune_stale,
            min_rgb_valid_fraction=args.min_rgb_valid_fraction,
            timeout=args.timeout,
            retries=args.retries,
        )
    elif args.command == "prune":
        prune_output_root(
            scene_index_path=args.scene_index,
            output_root=args.output_root,
            max_scenes=args.max_scenes,
            scene_ids=args.scene_id,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
