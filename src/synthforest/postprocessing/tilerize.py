"""
Adapted from https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/data/detection/tilerize.py
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geodataset.aoi import AOIConfig
from geodataset.geodata import RasterTileMetadata
from geodataset.tilerize import LabeledRasterTilerizer, RasterTilerizer
from geodataset.utils import CocoNameConvention
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union


def combine_gdfs(gdf1, gdf2):
    gdf_combined = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))
    merged_polygons = unary_union(gdf_combined.geometry)
    if isinstance(merged_polygons, Polygon):
        gdf_combined = gpd.GeoDataFrame(geometry=[merged_polygons], crs=gdf_combined.crs)
    elif isinstance(merged_polygons, MultiPolygon):
        gdf_combined = gpd.GeoDataFrame(geometry=list(merged_polygons.geoms), crs=gdf_combined.crs)
    else:
        raise Exception

    return gdf_combined


def _tile_fits_inside_raster(
    tile: RasterTileMetadata, raster_width: int, raster_height: int
) -> bool:
    return (
        tile.col >= 0
        and tile.row >= 0
        and tile.col + tile.metadata["width"] <= raster_width
        and tile.row + tile.metadata["height"] <= raster_height
    )


class _DropEdgeTilesMixin:
    drop_edge_tiles: bool
    min_valid_fraction: float

    def _create_tiles(self):
        tiles = super()._create_tiles()

        raster_width = int(self.raster.metadata["width"])
        raster_height = int(self.raster.metadata["height"])
        if self.drop_edge_tiles:
            edge_filtered_tiles = [
                tile
                for tile in tiles
                if _tile_fits_inside_raster(
                    tile=tile,
                    raster_width=raster_width,
                    raster_height=raster_height,
                )
            ]
        else:
            edge_filtered_tiles = tiles
        edge_dropped_count = len(tiles) - len(edge_filtered_tiles)
        if edge_dropped_count:
            print(
                f"Dropped {edge_dropped_count} edge tile(s) whose windows extend outside "
                f"the raster bounds ({raster_width}x{raster_height})."
            )

        if self.min_valid_fraction <= 0.0:
            return edge_filtered_tiles
        with rasterio.open(self.raster.path) as source_dataset:
            valid_filtered_tiles = [
                tile
                for tile in edge_filtered_tiles
                if float(
                    _tile_source_valid_mask(
                        tile,
                        source_dataset=source_dataset,
                    ).mean()
                )
                >= self.min_valid_fraction
            ]
        invalid_dropped_count = len(edge_filtered_tiles) - len(valid_filtered_tiles)
        if invalid_dropped_count:
            print(
                f"Dropped {invalid_dropped_count} tile(s) with valid fraction below "
                f"{self.min_valid_fraction:g}."
            )
        return valid_filtered_tiles


class _EdgeAwareLabeledRasterTilerizer(_DropEdgeTilesMixin, LabeledRasterTilerizer):
    def __init__(
        self,
        *args,
        drop_edge_tiles: bool = False,
        min_valid_fraction: float = 0.0,
        **kwargs,
    ):
        self.drop_edge_tiles = drop_edge_tiles
        self.min_valid_fraction = float(min_valid_fraction)
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be between 0 and 1.")
        super().__init__(*args, **kwargs)


class _EdgeAwareRasterTilerizer(_DropEdgeTilesMixin, RasterTilerizer):
    def __init__(
        self,
        *args,
        drop_edge_tiles: bool = False,
        min_valid_fraction: float = 0.0,
        **kwargs,
    ):
        self.drop_edge_tiles = drop_edge_tiles
        self.min_valid_fraction = float(min_valid_fraction)
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be between 0 and 1.")
        super().__init__(*args, **kwargs)


def _as_bool_mask(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.shape != (height, width):
        raise ValueError(f"Expected mask with shape {(height, width)}, got {mask.shape}.")
    return np.ascontiguousarray(mask > 0, dtype=bool)


def _tile_bounds_valid_mask(
    tile: RasterTileMetadata, raster_width: int, raster_height: int
) -> np.ndarray:
    width = int(tile.metadata["width"])
    height = int(tile.metadata["height"])
    valid = np.zeros((height, width), dtype=bool)

    row_start = max(int(tile.row), 0)
    col_start = max(int(tile.col), 0)
    row_end = min(int(tile.row) + height, raster_height)
    col_end = min(int(tile.col) + width, raster_width)
    if row_end <= row_start or col_end <= col_start:
        return valid

    out_row_start = row_start - int(tile.row)
    out_col_start = col_start - int(tile.col)
    valid[
        out_row_start : out_row_start + (row_end - row_start),
        out_col_start : out_col_start + (col_end - col_start),
    ] = True
    return valid


def _tile_source_valid_mask(
    tile: RasterTileMetadata,
    *,
    source_dataset=None,
) -> np.ndarray:
    width = int(tile.metadata["width"])
    height = int(tile.metadata["height"])
    raster_width = int(tile.associated_raster.metadata["width"])
    raster_height = int(tile.associated_raster.metadata["height"])
    valid = _tile_bounds_valid_mask(
        tile=tile,
        raster_width=raster_width,
        raster_height=raster_height,
    )

    def apply_original_source_mask(dataset) -> None:
        tile_window = Window(int(tile.col), int(tile.row), width, height)
        tile_bounds = rasterio.windows.bounds(
            tile_window,
            tile.associated_raster.metadata["transform"],
        )
        source_window = rasterio.windows.from_bounds(
            *tile_bounds,
            transform=dataset.transform,
        )
        source_mask = dataset.dataset_mask(
            window=source_window,
            out_shape=(height, width),
            boundless=True,
        )
        valid[:] &= _as_bool_mask(source_mask, height=height, width=width)

        # A GDAL internal mask takes precedence over the nodata value. This can
        # make dataset_mask() report CHM nodata cells as valid when a shared RGB
        # mask was written to the CHM. Check finite/nodata values explicitly so
        # min_valid_fraction also measures usable target coverage.
        source_values = dataset.read(
            window=source_window,
            out_shape=(dataset.count, height, width),
            boundless=True,
        )
        value_valid = np.all(np.isfinite(source_values), axis=0)
        nodata = dataset.nodata
        if nodata is not None:
            if np.isnan(nodata):
                value_valid &= ~np.any(np.isnan(source_values), axis=0)
            else:
                value_valid &= np.all(source_values != nodata, axis=0)
        valid[:] &= np.ascontiguousarray(value_valid, dtype=bool)

    if source_dataset is not None:
        apply_original_source_mask(source_dataset)
    else:
        try:
            with rasterio.open(tile.associated_raster.path) as dataset:
                apply_original_source_mask(dataset)
        except Exception:
            try:
                source_mask = tile.associated_raster.data.dataset_mask(
                    window=Window(int(tile.col), int(tile.row), width, height),
                    out_shape=(height, width),
                    boundless=True,
                )
                valid &= _as_bool_mask(source_mask, height=height, width=width)
            except Exception:
                pass

    if tile.mask is not None:
        valid &= _as_bool_mask(tile.mask, height=height, width=width)
    return np.ascontiguousarray(valid, dtype=bool)


def _write_tile_with_internal_mask(
    *,
    tile_path: Path,
    valid_mask: np.ndarray,
) -> None:
    with rasterio.open(tile_path) as src:
        data = src.read()
        profile = src.profile.copy()
        colorinterp = src.colorinterp

    with rasterio.open(tile_path, "w", **profile) as dst:
        dst.write(data)
        if colorinterp:
            dst.colorinterp = colorinterp
        dst.write_mask(valid_mask.astype(np.uint8) * 255)


def _write_rgb_tile_with_alpha(
    *,
    tile_path: Path,
    valid_mask: np.ndarray,
) -> bool:
    with rasterio.open(tile_path) as src:
        data = src.read()
        profile = src.profile.copy()
        colorinterp = src.colorinterp

    if data.dtype != np.uint8 or data.shape[0] not in {3, 4}:
        return False
    if data.shape[0] == 4 and (len(colorinterp) < 4 or colorinterp[3] != ColorInterp.alpha):
        return False

    alpha = valid_mask.astype(np.uint8) * 255
    if data.shape[0] == 3:
        data = np.concatenate([data, alpha[None, :, :]], axis=0)
    else:
        data = data.copy()
        data[3] = np.minimum(data[3], alpha)

    profile.update(count=4, dtype="uint8", nodata=None)
    with rasterio.open(tile_path, "w", **profile) as dst:
        dst.write(data)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    return True


def _write_tile_valid_mask(
    *,
    tile_path: Path,
    tile: RasterTileMetadata,
    valid_mask_mode: str,
    source_dataset,
) -> dict:
    valid_mask = _tile_source_valid_mask(
        tile,
        source_dataset=source_dataset,
    )
    total_pixels = int(valid_mask.size)
    valid_pixels = int(valid_mask.sum())

    mode = valid_mask_mode.lower().replace("-", "_")
    if mode not in {"alpha", "internal_mask"}:
        raise ValueError("valid_mask_mode must be one of: alpha, internal_mask.")

    written_mode = "internal_mask"
    if mode == "alpha":
        if _write_rgb_tile_with_alpha(tile_path=tile_path, valid_mask=valid_mask):
            written_mode = "alpha"
        else:
            _write_tile_with_internal_mask(tile_path=tile_path, valid_mask=valid_mask)
    else:
        _write_tile_with_internal_mask(tile_path=tile_path, valid_mask=valid_mask)

    return {
        "file_name": tile_path.name,
        "aoi": tile_path.parent.name,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_pixels / total_pixels) if total_pixels else 0.0,
        "partial": valid_pixels != total_pixels,
        "mode": written_mode,
    }


def _write_valid_masks_for_tiles(
    *,
    tilerizer: _EdgeAwareRasterTilerizer | _EdgeAwareLabeledRasterTilerizer,
    valid_mask_mode: str,
) -> dict:
    if not tilerizer.aois_tiles:
        return {
            "enabled": True,
            "mode": valid_mask_mode,
            "total_tiles": 0,
            "partial_tiles": 0,
            "per_aoi": {},
        }

    per_tile = []
    with rasterio.open(tilerizer.raster.path) as source_dataset:
        for aoi, tiles in tilerizer.aois_tiles.items():
            tiles_folder_aoi = tilerizer.tiles_path / aoi
            for tile in tiles:
                tile_path = tiles_folder_aoi / tile.generate_name()
                if not tile_path.exists():
                    continue
                per_tile.append(
                    _write_tile_valid_mask(
                        tile_path=tile_path,
                        tile=tile,
                        valid_mask_mode=valid_mask_mode,
                        source_dataset=source_dataset,
                    )
                )

    per_aoi: dict[str, dict] = {}
    for item in per_tile:
        aoi_summary = per_aoi.setdefault(
            item["aoi"],
            {
                "total_tiles": 0,
                "partial_tiles": 0,
                "valid_pixels": 0,
                "total_pixels": 0,
                "write_modes": {},
            },
        )
        aoi_summary["total_tiles"] += 1
        aoi_summary["partial_tiles"] += int(item["partial"])
        aoi_summary["valid_pixels"] += item["valid_pixels"]
        aoi_summary["total_pixels"] += item["total_pixels"]
        aoi_summary["write_modes"][item["mode"]] = (
            aoi_summary["write_modes"].get(item["mode"], 0) + 1
        )

    for aoi_summary in per_aoi.values():
        total_pixels = aoi_summary["total_pixels"]
        aoi_summary["valid_fraction"] = (
            float(aoi_summary["valid_pixels"] / total_pixels) if total_pixels else 0.0
        )

    total_tiles = len(per_tile)
    partial_tiles = sum(int(item["partial"]) for item in per_tile)
    valid_pixels = sum(item["valid_pixels"] for item in per_tile)
    total_pixels = sum(item["total_pixels"] for item in per_tile)
    summary = {
        "enabled": True,
        "mode": valid_mask_mode,
        "total_tiles": total_tiles,
        "partial_tiles": partial_tiles,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_pixels / total_pixels) if total_pixels else 0.0,
        "per_aoi": per_aoi,
    }

    summary_path = tilerizer.output_path / "tile_validity_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote validity masks for {total_tiles} tile(s) "
        f"({partial_tiles} partial) to {tilerizer.tiles_path}. "
        f"Summary: {summary_path}"
    )
    return summary


def _labels_are_empty(labels: Union[str, Path, gpd.GeoDataFrame]) -> bool:
    if isinstance(labels, gpd.GeoDataFrame):
        return len(labels) == 0
    if labels is None:
        return True
    try:
        return len(gpd.read_file(labels)) == 0
    except Exception:
        return False


def _write_empty_coco_for_tiles(
    *,
    tilerizer: _EdgeAwareRasterTilerizer,
    coco_categories_list: Optional[list],
) -> dict[str, Path | None]:
    coco_paths: dict[str, Path | None] = {}
    for aoi, tiles in tilerizer.aois_tiles.items():
        if len(tiles) == 0:
            coco_paths[aoi] = None
            continue

        images = []
        tiles_folder_aoi = tilerizer.tiles_path / aoi
        for image_id, tile in enumerate(tiles):
            tile_path = tiles_folder_aoi / tile.generate_name()
            with rasterio.open(tile_path) as tile_src:
                width = int(tile_src.width)
                height = int(tile_src.height)
            images.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": tile_path.name,
                }
            )

        coco_output_path = tilerizer.output_path / CocoNameConvention.create_name(
            product_name=tilerizer.raster.output_name,
            ground_resolution=tilerizer.ground_resolution,
            scale_factor=tilerizer.scale_factor,
            fold=aoi,
        )
        with coco_output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "info": {
                        "description": (
                            f"Dataset for the product {tilerizer.raster.output_name}"
                            f" with fold {aoi}"
                            f" and scale_factor {tilerizer.scale_factor}"
                            f" and ground_resolution {tilerizer.ground_resolution}."
                        ),
                        "version": "1.0",
                        "year": str(date.today().year),
                        "date_created": str(date.today()),
                    },
                    "licenses": [],
                    "images": images,
                    "annotations": [],
                    "categories": coco_categories_list or [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        coco_paths[aoi] = coco_output_path
    return coco_paths


def _tilerize_empty_labels_with_overlap(
    raster_path: Union[str, Path],
    coco_categories_list: Optional[list],
    aois_config: AOIConfig,
    output_path: Union[str, Path],
    ground_resolution: float = None,
    scale_factor: float = None,
    tile_size: int = 1024,
    tile_overlap: float = 0.5,
    ignore_black_white_alpha_tiles_threshold: float = 0.8,
    drop_edge_tiles: bool = False,
    write_valid_mask: bool = True,
    valid_mask_mode: str = "alpha",
    min_valid_fraction: float = 0.0,
):
    tilerizer = _EdgeAwareRasterTilerizer(
        raster_path=raster_path,
        output_path=output_path,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        aois_config=aois_config,
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
        ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
        drop_edge_tiles=drop_edge_tiles,
        min_valid_fraction=min_valid_fraction,
    )

    tilerizer.generate_tiles()
    if write_valid_mask:
        _write_valid_masks_for_tiles(
            tilerizer=tilerizer,
            valid_mask_mode=valid_mask_mode,
        )
    coco_outputs = _write_empty_coco_for_tiles(
        tilerizer=tilerizer,
        coco_categories_list=coco_categories_list,
    )
    return coco_outputs, tilerizer.tiles_path


def tilerize_no_overlap(
    raster_path: Union[str, Path],
    labels: Union[str, Path, gpd.GeoDataFrame],
    main_label_category_column_name: Optional[str],
    coco_categories_list: Optional[list],
    aois_config: AOIConfig,
    output_path: Union[str, Path],
    ignore_tiles_without_labels: bool = True,
    drop_edge_tiles: bool = False,
    write_valid_mask: bool = True,
    valid_mask_mode: str = "alpha",
    min_valid_fraction: float = 0.0,
):
    labels_path = labels if not isinstance(labels, gpd.GeoDataFrame) else None
    labels_gdf = labels if isinstance(labels, gpd.GeoDataFrame) else None

    if not ignore_tiles_without_labels and _labels_are_empty(labels):
        coco_paths, _ = _tilerize_empty_labels_with_overlap(
            raster_path=raster_path,
            coco_categories_list=coco_categories_list,
            aois_config=aois_config,
            output_path=output_path,
            tile_size=2048,
            tile_overlap=0.0,
            ground_resolution=None,
            scale_factor=1.0,
            ignore_black_white_alpha_tiles_threshold=0.8,
            drop_edge_tiles=drop_edge_tiles,
            write_valid_mask=write_valid_mask,
            valid_mask_mode=valid_mask_mode,
            min_valid_fraction=min_valid_fraction,
        )
        return coco_paths

    tilerizer = _EdgeAwareLabeledRasterTilerizer(
        raster_path=raster_path,
        labels_path=labels_path,
        labels_gdf=labels_gdf,
        output_path=output_path,
        tile_size=2048,
        tile_overlap=0.0,
        aois_config=aois_config,
        ground_resolution=None,
        scale_factor=1.0,  # we keep the source resolution
        use_rle_for_labels=True,
        min_intersection_ratio=0.4,
        ignore_black_white_alpha_tiles_threshold=0.8,
        ignore_tiles_without_labels=ignore_tiles_without_labels,
        main_label_category_column_name=main_label_category_column_name,
        coco_categories_list=coco_categories_list,
        coco_n_workers=10,
        drop_edge_tiles=drop_edge_tiles,
        min_valid_fraction=min_valid_fraction,
    )

    coco_paths = tilerizer.generate_coco_dataset()
    if write_valid_mask:
        _write_valid_masks_for_tiles(
            tilerizer=tilerizer,
            valid_mask_mode=valid_mask_mode,
        )

    return coco_paths


def tilerize_with_overlap(
    raster_path: Union[str, Path],
    labels: Union[str, Path, gpd.GeoDataFrame],
    main_label_category_column_name: Optional[str],
    coco_categories_list: Optional[list],
    aois_config: AOIConfig,
    output_path: Union[str, Path],
    ground_resolution: float = None,
    scale_factor: float = None,
    tile_size: int = 1024,
    tile_overlap: float = 0.5,
    ignore_tiles_without_labels: bool = True,
    ignore_black_white_alpha_tiles_threshold: float = 0.8,
    drop_edge_tiles: bool = False,
    write_valid_mask: bool = True,
    valid_mask_mode: str = "alpha",
    min_valid_fraction: float = 0.0,
):
    labels_path = labels if not isinstance(labels, gpd.GeoDataFrame) else None
    labels_gdf = labels if isinstance(labels, gpd.GeoDataFrame) else None

    if not ignore_tiles_without_labels and _labels_are_empty(labels):
        return _tilerize_empty_labels_with_overlap(
            raster_path=raster_path,
            coco_categories_list=coco_categories_list,
            aois_config=aois_config,
            output_path=output_path,
            ground_resolution=ground_resolution,
            scale_factor=scale_factor,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
            drop_edge_tiles=drop_edge_tiles,
            write_valid_mask=write_valid_mask,
            valid_mask_mode=valid_mask_mode,
            min_valid_fraction=min_valid_fraction,
        )

    tilerizer = _EdgeAwareLabeledRasterTilerizer(
        raster_path=raster_path,
        labels_path=labels_path,
        labels_gdf=labels_gdf,
        output_path=output_path,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        aois_config=aois_config,
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
        use_rle_for_labels=True,
        min_intersection_ratio=0.4,
        ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
        ignore_tiles_without_labels=ignore_tiles_without_labels,
        main_label_category_column_name=main_label_category_column_name,
        coco_categories_list=coco_categories_list,
        coco_n_workers=10,
        drop_edge_tiles=drop_edge_tiles,
        min_valid_fraction=min_valid_fraction,
    )

    coco_outputs = tilerizer.generate_coco_dataset()
    if write_valid_mask:
        _write_valid_masks_for_tiles(
            tilerizer=tilerizer,
            valid_mask_mode=valid_mask_mode,
        )
    tiles_path = tilerizer.tiles_path

    return coco_outputs, tiles_path


def tilerize_raster_with_overlap(
    raster_path: Union[str, Path],
    aois_config: AOIConfig,
    output_path: Union[str, Path],
    ground_resolution: float = None,
    scale_factor: float = None,
    tile_size: int = 1024,
    tile_overlap: float = 0.5,
    output_name_suffix: str | None = None,
    ignore_black_white_alpha_tiles_threshold: float = 0.8,
    output_dtype: str | None = None,
    drop_edge_tiles: bool = False,
    write_valid_mask: bool = True,
    valid_mask_mode: str = "alpha",
    min_valid_fraction: float = 0.0,
):
    """Tile an unlabeled raster, e.g. a height map, using the same AOI logic."""

    tilerizer = _EdgeAwareRasterTilerizer(
        raster_path=raster_path,
        output_path=output_path,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        aois_config=aois_config,
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
        output_name_suffix=output_name_suffix,
        ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
        output_dtype=output_dtype,
        drop_edge_tiles=drop_edge_tiles,
        min_valid_fraction=min_valid_fraction,
    )

    try:
        tile_outputs = tilerizer.generate_tiles()
    except ValueError as exc:
        if "max() arg is an empty sequence" in str(exc):
            raise RuntimeError(
                "No tiles were generated before AOI assignment for raster "
                f"{raster_path}. Check tile_size={tile_size}, tile_overlap={tile_overlap}, "
                f"drop_edge_tiles={drop_edge_tiles}, and "
                "ignore_black_white_alpha_tiles_threshold="
                f"{ignore_black_white_alpha_tiles_threshold}. For height maps with valid "
                "zero-valued targets, use a threshold greater than 1.0."
            ) from exc
        raise
    if write_valid_mask:
        _write_valid_masks_for_tiles(
            tilerizer=tilerizer,
            valid_mask_mode=valid_mask_mode,
        )
    tiles_path = tilerizer.tiles_path

    return tile_outputs, tiles_path
