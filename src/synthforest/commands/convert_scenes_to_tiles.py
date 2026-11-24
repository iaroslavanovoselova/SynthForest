#!/usr/bin/env python3
"""Hydra launcher to convert full-scene labels/images into a tiled detection dataset."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import geopandas as gpd
import hydra
from geodataset.aoi import AOIGeneratorConfig
from omegaconf import DictConfig, OmegaConf

from synthforest.postprocessing.scene_label_builders import (
    _extract_scene_id,
    build_labels_and_categories_by_raster,
    load_categories_json,
)
from synthforest.postprocessing.tilerize import (
    tilerize_raster_with_overlap,
    tilerize_with_overlap,
)

logger = logging.getLogger(__name__)

HEIGHT_MAP_TILES_SUBDIR = "height_maps"
HEIGHT_TILE_EMPTY_FILTER_DISABLED_THRESHOLD = 1.01


def _configure_logging() -> None:
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    root_logger.addHandler(stream_handler)


def _find_raster_files(
    directory: Path,
    glob_pattern: str | None = None,
) -> list[Path]:
    if glob_pattern:
        candidates = sorted(directory.glob(str(glob_pattern)))
    else:
        candidates = sorted(list(directory.glob("*.tif")) + list(directory.glob("*.tiff")))
    return [
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]


def _scene_pair_key(stem: str, suffix: str | None) -> str | None:
    """Resolve a pair key from a configured suffix or the legacy numeric ID."""
    suffix = str(suffix or "").strip()
    if not suffix:
        return _extract_scene_id(stem)
    if not stem.lower().endswith(suffix.lower()):
        return None
    key = stem[: -len(suffix)]
    return key or None


def _replace_directory(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def _resolved_path_value(value: object) -> object:
    if value:
        return str(Path(str(value)).expanduser().resolve())
    return value


def _resolved_config_payload(
    cfg: DictConfig,
    *,
    images_dir: Path,
    output_root: Path,
    output_product_dir: Path,
    height_maps_dir: Path | None,
) -> dict[str, object]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        payload = {"value": payload}

    payload["images_dir"] = str(images_dir)
    payload["output_root"] = str(output_root)
    payload["output_product_dir"] = str(output_product_dir)
    if height_maps_dir is not None:
        payload["height_maps_dir"] = str(height_maps_dir)
    if "height_ignore_black_white_alpha_tiles_threshold" not in payload:
        payload["height_ignore_black_white_alpha_tiles_threshold"] = (
            HEIGHT_TILE_EMPTY_FILTER_DISABLED_THRESHOLD
        )

    for key in (
        "annotations_dir",
        "labels_gdf_dir",
        "save_converted_gdf_dir",
        "categories_json",
    ):
        payload[key] = _resolved_path_value(payload.get(key))
    return payload


def _save_resolved_config(output_product_dir: Path, payload: dict[str, object]) -> Path:
    config_path = output_product_dir / "config" / "resolved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=config_path)
    return config_path


def _export_tasks(cfg: DictConfig) -> tuple[bool, bool]:
    export_mode = str(cfg.get("export_mode", "instance_segmentation")).lower().replace("-", "_")
    aliases = {
        "instance": "instance_segmentation",
        "instances": "instance_segmentation",
        "segmentation": "instance_segmentation",
        "height_estimation": "height",
        "heights": "height",
    }
    export_mode = aliases.get(export_mode, export_mode)
    if export_mode not in {"instance_segmentation", "height", "both"}:
        raise ValueError("export_mode must be one of: instance_segmentation, height, both.")
    return export_mode in {"instance_segmentation", "both"}, export_mode in {"height", "both"}


def _height_tile_skip_threshold(cfg: DictConfig) -> float:
    return float(
        cfg.get(
            "height_ignore_black_white_alpha_tiles_threshold",
            HEIGHT_TILE_EMPTY_FILTER_DISABLED_THRESHOLD,
        )
    )


def _run_instance_tile_generation(
    *,
    raster_files: list[Path],
    labels_gdf_by_raster: dict[str, gpd.GeoDataFrame],
    categories_by_raster: dict[str, list[dict] | None],
    output_product_dir: Path,
    aois_config: AOIGeneratorConfig,
    cfg: DictConfig,
) -> list[tuple[str, object, Path]]:
    generated = []
    for raster_path in raster_files:
        raster_labels = labels_gdf_by_raster.get(raster_path.stem)
        if raster_labels is None:
            raster_labels = gpd.GeoDataFrame(geometry=[])
        if len(raster_labels) == 0 and bool(cfg.ignore_tiles_without_labels):
            logger.warning("No labels found for raster %s. Skipping.", raster_path.name)
            continue

        categories = categories_by_raster.get(raster_path.stem)
        coco_outputs, tiles_path = tilerize_with_overlap(
            raster_path=raster_path,
            labels=raster_labels,
            main_label_category_column_name=cfg.main_label_category_column_name,
            coco_categories_list=categories,
            aois_config=aois_config,
            output_path=output_product_dir,
            ground_resolution=cfg.ground_resolution,
            scale_factor=cfg.scale_factor,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            ignore_tiles_without_labels=cfg.ignore_tiles_without_labels,
            ignore_black_white_alpha_tiles_threshold=cfg.get(
                "ignore_black_white_alpha_tiles_threshold",
                1.0,
            ),
            drop_edge_tiles=bool(cfg.get("drop_edge_tiles", False)),
            write_valid_mask=bool(cfg.get("write_valid_mask", True)),
            valid_mask_mode=str(cfg.get("valid_mask_mode", "alpha")),
            min_valid_fraction=float(cfg.get("min_valid_fraction", 0.0)),
        )
        generated.append((raster_path.name, coco_outputs, tiles_path))
        logger.info("Processed %s", raster_path.name)
    return generated


def _run_image_tile_generation(
    *,
    raster_files: list[Path],
    output_product_dir: Path,
    aois_config: AOIGeneratorConfig,
    cfg: DictConfig,
) -> list[tuple[str, object, Path]]:
    generated = []

    for raster_path in raster_files:
        image_outputs, image_tiles_path = tilerize_raster_with_overlap(
            raster_path=raster_path,
            aois_config=aois_config,
            output_path=output_product_dir,
            ground_resolution=cfg.ground_resolution,
            scale_factor=cfg.scale_factor,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            ignore_black_white_alpha_tiles_threshold=cfg.get(
                "ignore_black_white_alpha_tiles_threshold",
                1.0,
            ),
            drop_edge_tiles=bool(cfg.get("drop_edge_tiles", False)),
            write_valid_mask=bool(cfg.get("write_valid_mask", True)),
            valid_mask_mode=str(cfg.get("valid_mask_mode", "alpha")),
            min_valid_fraction=float(cfg.get("min_valid_fraction", 0.0)),
        )
        generated.append((raster_path.name, image_outputs, image_tiles_path))
        logger.info("Processed image tiles for %s", raster_path.name)
    return generated


def _height_maps_by_scene_id(
    height_maps_dir: Path,
    *,
    glob_pattern: str | None,
    pair_suffix: str | None,
) -> dict[str, Path]:
    height_map_files = _find_raster_files(height_maps_dir, glob_pattern)
    if not height_map_files:
        raise FileNotFoundError(f"No .tif/.tiff height maps found in: {height_maps_dir}")

    height_maps: dict[str, Path] = {}
    for height_map_path in height_map_files:
        scene_id = _scene_pair_key(height_map_path.stem, pair_suffix)
        if scene_id is None:
            logger.warning(
                "Could not extract a pair key from height map %s. Skipping.",
                height_map_path.name,
            )
            continue
        if scene_id in height_maps:
            raise ValueError(
                f"Duplicate height map scene id '{scene_id}': "
                f"{height_maps[scene_id]} and {height_map_path}"
            )
        height_maps[scene_id] = height_map_path
    return height_maps


def _tile_spatial_key(path: Path, fold: str) -> tuple[str, str] | None:
    """Return (spatial key, full suffix) without depending on tiler tile IDs."""
    marker = f"_tile_{fold}_"
    if marker not in path.stem:
        return None
    suffix = path.stem.split(marker, 1)[1]
    spatial_key, separator, _tile_id = suffix.rpartition("_")
    if not separator or not spatial_key:
        return None
    return spatial_key, suffix


def _synchronize_paired_tiles(
    *,
    image_product_root: Path,
    height_tiles_root: Path,
) -> None:
    """Keep spatially paired tiles and align height filename suffixes to RGB."""
    image_tiles_root = image_product_root / "tiles"
    folds = sorted(
        {
            path.name
            for root in (image_tiles_root, height_tiles_root)
            if root.exists()
            for path in root.iterdir()
            if path.is_dir()
        }
    )
    for fold in folds:
        image_dir = image_tiles_root / fold
        height_dir = height_tiles_root / fold
        images: dict[str, tuple[Path, str]] = {}
        heights: dict[str, tuple[Path, str]] = {}
        for directory, mapping in ((image_dir, images), (height_dir, heights)):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.tif")):
                parsed = _tile_spatial_key(path, fold)
                if parsed is None:
                    continue
                spatial_key, suffix = parsed
                if spatial_key in mapping:
                    raise ValueError(
                        f"Duplicate {fold} tile spatial key {spatial_key!r}: "
                        f"{mapping[spatial_key][0]} and {path}"
                    )
                mapping[spatial_key] = (path, suffix)

        shared = set(images) & set(heights)
        for key in sorted(set(images) - shared):
            images[key][0].unlink()
        for key in sorted(set(heights) - shared):
            heights[key][0].unlink()

        renamed = 0
        for key in sorted(shared):
            height_path, _height_suffix = heights[key]
            image_path, image_suffix = images[key]
            marker = f"_tile_{fold}_"
            height_prefix = height_path.stem.split(marker, 1)[0]
            destination = height_path.with_name(
                f"{height_prefix}{marker}{image_suffix}{height_path.suffix}"
            )
            if destination != height_path:
                if destination.exists():
                    raise FileExistsError(
                        f"Cannot align height tile name; destination exists: {destination}"
                    )
                height_path.rename(destination)
                renamed += 1

        removed_images = len(images) - len(shared)
        removed_heights = len(heights) - len(shared)
        logger.info(
            "Synchronized %s fold for %s: paired=%d, removed_rgb=%d, "
            "removed_chm=%d, renamed_chm=%d.",
            fold,
            image_product_root.name,
            len(shared),
            removed_images,
            removed_heights,
            renamed,
        )


def _run_height_tile_generation(
    *,
    raster_files: list[Path],
    height_maps_by_id: dict[str, Path],
    image_product_roots: dict[str, Path],
    aois_config: AOIGeneratorConfig,
    cfg: DictConfig,
) -> list[tuple[str, object, Path]]:
    generated = []

    for raster_path in raster_files:
        scene_id = _scene_pair_key(
            raster_path.stem,
            cfg.get("image_pair_suffix"),
        )
        height_map_path = height_maps_by_id.get(scene_id)
        if height_map_path is None:
            logger.warning(
                "No height map found for raster %s (pair key %s). "
                "Skipping height export for this raster.",
                raster_path.name,
                scene_id,
            )
            continue
        image_product_root = image_product_roots.get(raster_path.name)
        if image_product_root is None:
            logger.warning(
                "No generated RGB product root found for %s. "
                "Skipping height export for this raster.",
                raster_path.name,
            )
            continue

        height_outputs, height_tiles_path = tilerize_raster_with_overlap(
            raster_path=height_map_path,
            aois_config=aois_config,
            output_path=image_product_root,
            ground_resolution=cfg.ground_resolution,
            scale_factor=cfg.scale_factor,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            ignore_black_white_alpha_tiles_threshold=_height_tile_skip_threshold(cfg),
            drop_edge_tiles=bool(cfg.get("drop_edge_tiles", False)),
            write_valid_mask=bool(cfg.get("write_valid_mask", True)),
            valid_mask_mode=str(cfg.get("valid_mask_mode", "alpha")),
            min_valid_fraction=float(cfg.get("min_valid_fraction", 0.0)),
        )
        desired_height_tiles_path = _replace_directory(
            source=height_tiles_path,
            destination=image_product_root / HEIGHT_MAP_TILES_SUBDIR,
        )
        height_product_root = height_tiles_path.parent
        if height_product_root != image_product_root:
            shutil.rmtree(height_product_root, ignore_errors=True)
        _synchronize_paired_tiles(
            image_product_root=image_product_root,
            height_tiles_root=desired_height_tiles_path,
        )

        generated.append((raster_path.name, height_outputs, desired_height_tiles_path))
        logger.info("Processed height pair %s -> %s", raster_path.name, height_map_path.name)
    return generated


def _force_single_tree_category(
    *,
    labels_gdf_by_raster: dict[str, gpd.GeoDataFrame],
    categories_by_raster: dict[str, list[dict] | None],
    raster_files: list[Path],
    cfg: DictConfig,
) -> None:
    if not bool(cfg.get("force_single_tree_category", False)):
        return

    category_column = str(cfg.get("main_label_category_column_name", "category_name"))
    category_name = str(cfg.get("single_tree_category_name", "tree"))
    category_id = int(cfg.get("single_tree_category_id", 1))

    single_category = [{"id": category_id, "name": category_name, "supercategory": category_name}]

    for raster_path in raster_files:
        raster_stem = raster_path.stem
        raster_labels = labels_gdf_by_raster.get(raster_stem)
        if raster_labels is not None and len(raster_labels) > 0:
            raster_labels = raster_labels.copy()
            raster_labels[category_column] = category_name
            labels_gdf_by_raster[raster_stem] = raster_labels
        categories_by_raster[raster_stem] = single_category

    logger.info(
        "Forced single category mapping for all annotations: %s (id=%d).",
        category_name,
        category_id,
    )


def run_conversion(cfg: DictConfig) -> None:
    export_instance, export_height = _export_tasks(cfg)

    if export_instance:
        # Validate source selection: exactly one of annotations_dir or labels_gdf_dir must be provided.
        source_values = [cfg.annotations_dir, cfg.labels_gdf_dir]
        selected_sources = [value for value in source_values if value]
        if len(selected_sources) != 1:
            raise ValueError(
                "Exactly one source must be configured for instance segmentation: "
                "annotations_dir or labels_gdf_dir."
            )

    train_ratio = float(cfg.train_ratio)
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("--train_ratio must be greater than 0 and at most 1.")

    images_dir = Path(cfg.images_dir).expanduser().resolve()
    output_root = Path(cfg.output_root).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    height_maps_dir = None
    if export_height:
        height_maps_dir = Path(cfg.height_maps_dir).expanduser().resolve()
        if not height_maps_dir.exists():
            raise FileNotFoundError(f"Height maps directory not found: {height_maps_dir}")

    raster_files = _find_raster_files(
        images_dir,
        cfg.get("images_glob"),
    )
    if not raster_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in: {images_dir}")

    output_product_dir = output_root / cfg.product_name
    output_product_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = _save_resolved_config(
        output_product_dir,
        _resolved_config_payload(
            cfg,
            images_dir=images_dir,
            output_root=output_root,
            output_product_dir=output_product_dir,
            height_maps_dir=height_maps_dir,
        ),
    )
    logger.info("Resolved tiling config saved to: %s", resolved_config_path)

    aois = {
        "train": {"percentage": train_ratio, "position": 1},
    }
    if train_ratio < 1.0:
        aois["valid"] = {
            "percentage": float(1.0 - train_ratio),
            "position": 2,
        }
    aois_config = AOIGeneratorConfig(aoi_type=cfg.aoi_type, aois=aois)

    instance_generated = []
    image_generated = []
    height_generated = []

    if export_instance:
        categories_from_file = load_categories_json(cfg.categories_json)
        labels_gdf_by_raster, categories_by_raster = build_labels_and_categories_by_raster(
            annotations_dir=cfg.annotations_dir,
            labels_gdf_dir=cfg.labels_gdf_dir,
            images_dir=images_dir,
            raster_files=raster_files,
            convert_to_crs_coordinates=cfg.convert_to_crs_coordinates,
            save_converted_gdf_dir=cfg.save_converted_gdf_dir,
            categories_from_file=categories_from_file,
            log_warn=logger.warning,
        )

        _force_single_tree_category(
            labels_gdf_by_raster=labels_gdf_by_raster,
            categories_by_raster=categories_by_raster,
            raster_files=raster_files,
            cfg=cfg,
        )

        logger.info(
            "Found %d raster(s). Starting instance tiling and COCO export...", len(raster_files)
        )
        instance_generated = _run_instance_tile_generation(
            raster_files=raster_files,
            labels_gdf_by_raster=labels_gdf_by_raster,
            categories_by_raster=categories_by_raster,
            output_product_dir=output_product_dir,
            aois_config=aois_config,
            cfg=cfg,
        )

    if export_height:
        if not export_instance:
            logger.info("Found %d raster(s). Starting shared image tiling...", len(raster_files))
            image_generated = _run_image_tile_generation(
                raster_files=raster_files,
                output_product_dir=output_product_dir,
                aois_config=aois_config,
                cfg=cfg,
            )

        height_maps_by_id = _height_maps_by_scene_id(
            height_maps_dir,
            glob_pattern=cfg.get("height_maps_glob"),
            pair_suffix=cfg.get("height_pair_suffix"),
        )
        image_product_roots = {
            raster_name: Path(tiles_path).parent
            for raster_name, _outputs, tiles_path in (instance_generated + image_generated)
        }
        logger.info("Found %d raster(s). Starting height-estimation tiling...", len(raster_files))
        height_generated = _run_height_tile_generation(
            raster_files=raster_files,
            height_maps_by_id=height_maps_by_id,
            image_product_roots=image_product_roots,
            aois_config=aois_config,
            cfg=cfg,
        )

    if export_instance and not instance_generated:
        raise RuntimeError(
            "No raster was processed for instance segmentation. "
            "Check image paths and COCO/GDF labels."
        )
    if export_height and not export_instance and not image_generated:
        raise RuntimeError(
            "No shared image tiles were processed for height estimation. Check image paths."
        )
    if export_height and not height_generated:
        raise RuntimeError(
            "No raster was processed for height estimation. Check image and height-map paths."
        )

    logger.info("Done. Generated dataset structure: %s", output_product_dir)
    if instance_generated:
        logger.info("Generated instance outputs summary:")
        for raster_name, coco_outputs, tiles_path in instance_generated:
            logger.info("- %s", raster_name)
            logger.info("  tiles: %s", tiles_path)
            logger.info("  coco: %s", coco_outputs)
    if image_generated:
        logger.info("Generated shared image outputs summary:")
        for raster_name, image_outputs, image_tiles_path in image_generated:
            logger.info("- %s", raster_name)
            logger.info("  images: %s (%s)", image_tiles_path, image_outputs)
    if height_generated:
        logger.info("Generated height outputs summary:")
        for raster_name, height_outputs, height_tiles_path in height_generated:
            logger.info("- %s", raster_name)
            logger.info("  height_maps: %s (%s)", height_tiles_path, height_outputs)


@hydra.main(
    version_base=None,
    config_path="../../../configs/generation",
    config_name="convert_scenes_to_tiles",
)
def main(cfg: DictConfig) -> None:
    try:
        _configure_logging()
        run_conversion(cfg)
    except Exception as exc:
        logger.exception("Dataset conversion failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
