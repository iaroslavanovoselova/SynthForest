"""Utilities to convert full-scene labels and rasters into a tiled detection dataset."""

import json
import re
from pathlib import Path
from typing import Callable

import geopandas as gpd
from geodataset.utils import coco_to_geopackage


def _resolve_json_files(annotations_dir: Path) -> dict[str, Path]:
    json_files = sorted(annotations_dir.glob("*.json"))
    mapping: dict[str, Path] = {}
    for p in json_files:
        stem = p.stem
        if stem in mapping:
            raise ValueError(f"Duplicate annotation stem '{stem}' in {annotations_dir}")
        mapping[stem] = p.resolve()
    return mapping


def _resolve_gdf_files(labels_gdf_dir: Path) -> dict[str, Path]:
    patterns = ["*.gpkg", "*.geojson", "*.json", "*.parquet"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(labels_gdf_dir.glob(pattern)))

    mapping: dict[str, Path] = {}
    for p in files:
        stem = p.stem
        if stem in mapping:
            raise ValueError(
                f"Duplicate GeoDataFrame stem '{stem}' in {labels_gdf_dir}. "
                "Keep one file per raster stem."
            )
        mapping[stem] = p.resolve()
    return mapping


def _extract_scene_id(stem: str) -> str | None:
    match = re.search(r"(\d{4})$", stem)
    return match.group(1) if match else None


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[])


def _read_gdf(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        raise ValueError(f"File has no geometry column: {path}")
    return gdf


def _write_gdf(path: Path, gdf: gpd.GeoDataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(path)
    else:
        # Default and recommended format is GeoPackage.
        gdf.to_file(path, driver="GPKG")


def _load_categories(coco_json_path: Path) -> list[dict] | None:
    with coco_json_path.open("r") as f:
        data = json.load(f)
    categories = data.get("categories")
    return categories if categories else None


def _normalize_tile_path_column(labels_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "tile_path" not in labels_gdf.columns:
        raise ValueError(
            "The converted GeoDataFrame has no 'tile_path' column. "
            "Expected COCO images to map to source TIFF paths."
        )

    gdf = labels_gdf.copy()
    gdf["tile_path"] = gdf["tile_path"].apply(
        lambda v: str(Path(v).resolve()) if v is not None else v
    )
    return gdf


def _select_labels_for_raster(labels_gdf: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    raster_resolved = str(raster_path.resolve())

    exact_match = labels_gdf[labels_gdf["tile_path"] == raster_resolved]
    if len(exact_match) > 0:
        return exact_match

    # Fallback for COCO files that only contain relative image names.
    by_name = labels_gdf[
        labels_gdf["tile_path"].apply(lambda p: Path(p).name if p is not None else None)
        == raster_path.name
    ]
    return by_name


def load_categories_json(categories_json: str | None) -> list[dict] | None:
    if not categories_json:
        return None
    categories_path = Path(categories_json).resolve()
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories JSON not found: {categories_path}")
    with categories_path.open("r") as f:
        categories_from_file = json.load(f)
    if not isinstance(categories_from_file, list):
        raise ValueError("--categories_json must contain a COCO categories list.")
    return categories_from_file


def _cache_raster_labels(
    save_dir: str | None, filename: str, raster_labels: gpd.GeoDataFrame
) -> None:
    if not save_dir or len(raster_labels) == 0:
        return
    cache_path = Path(save_dir).resolve() / filename
    _write_gdf(cache_path, raster_labels)


def _build_labels_from_annotations_dir(
    annotations_dir: str,
    images_dir: Path,
    raster_files: list[Path],
    convert_to_crs_coordinates: bool,
    save_converted_gdf_dir: str | None,
    categories_from_file: list[dict] | None,
    log_warn: Callable[[str, object, object], None],
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, list[dict] | None]]:
    labels_gdf_by_raster: dict[str, gpd.GeoDataFrame] = {}
    categories_by_raster: dict[str, list[dict] | None] = {}

    annotations_path = Path(annotations_dir).resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_path}")

    json_by_stem = _resolve_json_files(annotations_path)
    if not json_by_stem:
        raise FileNotFoundError(f"No JSON files found in: {annotations_path}")
    json_by_id = {
        scene_id: path
        for stem, path in json_by_stem.items()
        for scene_id in [_extract_scene_id(stem)]
        if scene_id is not None
    }

    for raster_path in raster_files:
        raster_id = _extract_scene_id(raster_path.stem)
        if raster_id is None or raster_id not in json_by_id:
            log_warn(
                "No annotation JSON found for raster %s (ID %s). Skipping.",
                raster_path.name,
                raster_id,
            )
            labels_gdf_by_raster[raster_path.stem] = _empty_gdf()
            categories_by_raster[raster_path.stem] = categories_from_file
            continue
        ann_path = json_by_id[raster_id]

        categories = _load_categories(ann_path)
        labels_gdf = coco_to_geopackage(
            coco_json_path=str(ann_path),
            images_directory=str(images_dir),
            convert_to_crs_coordinates=convert_to_crs_coordinates,
            geopackage_output_path=None,
        )
        labels_gdf = _normalize_tile_path_column(labels_gdf)
        raster_labels = _select_labels_for_raster(labels_gdf, raster_path)

        labels_gdf_by_raster[raster_path.stem] = raster_labels
        categories_by_raster[raster_path.stem] = (
            categories if categories is not None else categories_from_file
        )
        _cache_raster_labels(save_converted_gdf_dir, f"labels{raster_id}.gpkg", raster_labels)

    return labels_gdf_by_raster, categories_by_raster


def _build_labels_from_gdf_dir(
    labels_gdf_dir: str,
    raster_files: list[Path],
    categories_from_file: list[dict] | None,
    log_warn: Callable[[str, object, object], None],
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, list[dict] | None]]:
    labels_gdf_by_raster: dict[str, gpd.GeoDataFrame] = {}
    categories_by_raster: dict[str, list[dict] | None] = {}

    labels_gdf_path = Path(labels_gdf_dir).resolve()
    if not labels_gdf_path.exists():
        raise FileNotFoundError(f"Labels GeoDataFrame directory not found: {labels_gdf_path}")

    gdf_by_stem = _resolve_gdf_files(labels_gdf_path)
    if not gdf_by_stem:
        raise FileNotFoundError(f"No GeoDataFrame files found in: {labels_gdf_path}")
    gdf_by_id = {
        scene_id: path
        for stem, path in gdf_by_stem.items()
        for scene_id in [_extract_scene_id(stem)]
        if scene_id is not None
    }

    for raster_path in raster_files:
        raster_id = _extract_scene_id(raster_path.stem)
        gdf_path = gdf_by_id.get(raster_id)
        if gdf_path is None:
            log_warn(
                "No labels GeoDataFrame found for raster %s (ID %s). Skipping.",
                raster_path.name,
                raster_id,
            )
            labels_gdf_by_raster[raster_path.stem] = _empty_gdf()
            categories_by_raster[raster_path.stem] = categories_from_file
            continue

        gdf = _read_gdf(gdf_path)
        if "tile_path" in gdf.columns:
            gdf = _normalize_tile_path_column(gdf)
            gdf = _select_labels_for_raster(gdf, raster_path)

        labels_gdf_by_raster[raster_path.stem] = gdf
        categories_by_raster[raster_path.stem] = categories_from_file

    return labels_gdf_by_raster, categories_by_raster


def build_labels_and_categories_by_raster(
    *,
    annotations_dir: str | None,
    labels_gdf_dir: str | None,
    images_dir: Path,
    raster_files: list[Path],
    convert_to_crs_coordinates: bool,
    save_converted_gdf_dir: str | None,
    categories_from_file: list[dict] | None,
    log_warn: Callable[[str, object, object], None],
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, list[dict] | None]]:
    if annotations_dir:
        return _build_labels_from_annotations_dir(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            raster_files=raster_files,
            convert_to_crs_coordinates=convert_to_crs_coordinates,
            save_converted_gdf_dir=save_converted_gdf_dir,
            categories_from_file=categories_from_file,
            log_warn=log_warn,
        )
    if labels_gdf_dir:
        return _build_labels_from_gdf_dir(
            labels_gdf_dir=labels_gdf_dir,
            raster_files=raster_files,
            categories_from_file=categories_from_file,
            log_warn=log_warn,
        )
    raise ValueError("No source configured to build labels.")
