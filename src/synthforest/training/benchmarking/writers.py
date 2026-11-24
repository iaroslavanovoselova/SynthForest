from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from synthforest.training.coco_masks import (
    CocoMaskCoverage,
    filter_coco_annotations_with_valid_masks,
    segmentation_area,
)

if TYPE_CHECKING:
    from synthforest.training.benchmarking.inference import ImagePrediction


DEFAULT_CATEGORY = {"id": 1, "name": "tree", "supercategory": "tree"}

# Raster aggregation in this module is adapted from CanopyRS's detector
# benchmarking pipeline. It is implemented locally so detector checkpoints and
# prediction artifacts keep this repository's existing formats and dependencies.


def load_coco(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_coco(path: str | Path, coco: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return path


def write_subset_truth_coco(
    truth_coco_path: str | Path,
    output_path: str | Path,
    file_names: Iterable[str],
) -> Path:
    truth = load_coco(truth_coco_path)
    keep_names = set(file_names)
    images = [image for image in truth.get("images", []) if image.get("file_name") in keep_names]
    image_ids = {image["id"] for image in images}
    annotations = [
        annotation
        for annotation in truth.get("annotations", [])
        if annotation.get("image_id") in image_ids
    ]
    subset = {
        "info": truth.get("info", {}),
        "licenses": truth.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": truth.get("categories", [DEFAULT_CATEGORY]),
    }
    return write_coco(output_path, subset)


def write_prediction_coco(
    truth_coco_path: str | Path,
    predictions: list[ImagePrediction],
    output_path: str | Path,
) -> Path:
    truth = load_coco(truth_coco_path)
    image_by_name = {image["file_name"]: image for image in truth.get("images", [])}
    images = list(truth.get("images", []))
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for prediction in predictions:
        file_name = prediction.image_info["file_name"]
        truth_image = image_by_name.get(file_name, prediction.image_info)
        width = int(truth_image.get("width", prediction.image_info.get("width", 0)))
        height = int(truth_image.get("height", prediction.image_info.get("height", 0)))
        image_id = int(truth_image["id"])
        for box, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
        ):
            x1, y1, x2, y2 = _clip_xyxy(box, width=width, height=height)
            w = x2 - x1
            h = y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": max(1, int(label)),
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
            ann_id += 1

    pred_coco = {
        "info": truth.get("info", {}),
        "licenses": truth.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": truth.get("categories", [DEFAULT_CATEGORY]),
    }
    return write_coco(output_path, pred_coco)


def write_segmentation_truth_coco(
    truth_coco_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, CocoMaskCoverage]:
    truth = load_coco(truth_coco_path)
    annotations, coverage = filter_coco_annotations_with_valid_masks(truth)
    segm_truth = {
        "info": truth.get("info", {}),
        "licenses": truth.get("licenses", []),
        "images": list(truth.get("images", [])),
        "annotations": annotations,
        "categories": truth.get("categories", [DEFAULT_CATEGORY]),
    }
    return write_coco(output_path, segm_truth), coverage


def write_segmentation_prediction_coco(
    truth_coco_path: str | Path,
    predictions: list[ImagePrediction],
    output_path: str | Path,
) -> Path:
    truth = load_coco(truth_coco_path)
    image_by_name = {image["file_name"]: image for image in truth.get("images", [])}
    images = list(truth.get("images", []))
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for prediction in predictions:
        if prediction.segmentations is None:
            continue
        if len(prediction.segmentations) != len(prediction.boxes_xyxy):
            raise ValueError("Prediction segmentations and boxes have different lengths.")
        file_name = prediction.image_info["file_name"]
        truth_image = image_by_name.get(file_name, prediction.image_info)
        width = int(truth_image.get("width", prediction.image_info.get("width", 0)))
        height = int(truth_image.get("height", prediction.image_info.get("height", 0)))
        image_id = int(truth_image["id"])
        areas = prediction.mask_areas or [
            segmentation_area(segmentation) for segmentation in prediction.segmentations
        ]
        for box, score, label, segmentation, area in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
            prediction.segmentations,
            areas,
        ):
            x1, y1, x2, y2 = _clip_xyxy(
                box,
                width=width,
                height=height,
            )
            w = x2 - x1
            h = y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": max(1, int(label)),
                    "bbox": [x1, y1, w, h],
                    "segmentation": segmentation,
                    "area": float(area),
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
            ann_id += 1

    pred_coco = {
        "info": truth.get("info", {}),
        "licenses": truth.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": truth.get("categories", [DEFAULT_CATEGORY]),
    }
    return write_coco(output_path, pred_coco)


def write_empty_prediction_coco(truth_coco_path: str | Path, output_path: str | Path) -> Path:
    truth = load_coco(truth_coco_path)
    empty = {
        "info": truth.get("info", {}),
        "licenses": truth.get("licenses", []),
        "images": truth.get("images", []),
        "annotations": [],
        "categories": truth.get("categories", [DEFAULT_CATEGORY]),
    }
    return write_coco(output_path, empty)


def merge_coco_jsons(coco_paths: list[str | Path], output_path: str | Path) -> Path:
    merged = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    next_image_id = 1
    next_annotation_id = 1
    categories_by_id: dict[int, dict[str, Any]] = {}

    for coco_path in coco_paths:
        coco = load_coco(coco_path)
        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            new_image = dict(image)
            old_image_id = int(new_image["id"])
            new_image["id"] = next_image_id
            image_id_map[old_image_id] = next_image_id
            merged["images"].append(new_image)
            next_image_id += 1

        for annotation in coco.get("annotations", []):
            old_image_id = int(annotation["image_id"])
            if old_image_id not in image_id_map:
                continue
            new_annotation = dict(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[old_image_id]
            merged["annotations"].append(new_annotation)
            next_annotation_id += 1

        for category in coco.get("categories", []):
            categories_by_id[int(category["id"])] = dict(category)

    merged["categories"] = [categories_by_id[key] for key in sorted(categories_by_id)] or [
        DEFAULT_CATEGORY
    ]
    return write_coco(output_path, merged)


def load_prediction_coco(
    prediction_coco_path: str | Path,
    *,
    tile_dir: str | Path,
    truth_coco_path: str | Path,
) -> tuple[list[ImagePrediction], list[str], dict[str, int]]:
    """Restore inference objects from a saved benchmark prediction COCO."""

    from synthforest.training.benchmarking.inference import ImagePrediction

    prediction_coco = load_coco(prediction_coco_path)
    truth_coco = load_coco(truth_coco_path)
    truth_names = {str(image["file_name"]) for image in truth_coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    all_annotations = list(prediction_coco.get("annotations", []))
    coco_has_segmentations = bool(all_annotations) and all(
        annotation.get("segmentation") is not None for annotation in all_annotations
    )
    for annotation in all_annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    predictions: list[ImagePrediction] = []
    evaluated_names: list[str] = []
    for image_info in prediction_coco.get("images", []):
        image_info = dict(image_info)
        file_name = str(image_info["file_name"])
        if file_name not in truth_names:
            raise ValueError(
                f"Reusable prediction image {file_name!r} is absent from {truth_coco_path}."
            )
        image_path = _resolve_saved_tile_path(tile_dir, file_name)
        image_annotations = annotations_by_image.get(int(image_info["id"]), [])
        boxes_xyxy: list[list[float]] = []
        scores: list[float] = []
        labels: list[int] = []
        segmentations: list[dict[str, Any]] = []
        mask_areas: list[float] = []
        has_segmentations = coco_has_segmentations and all(
            annotation.get("segmentation") is not None for annotation in image_annotations
        )
        for annotation in image_annotations:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            boxes_xyxy.append([x, y, x + width, y + height])
            scores.append(float(annotation.get("score", 1.0)))
            labels.append(int(annotation.get("category_id", 1)))
            if has_segmentations:
                segmentations.append(annotation["segmentation"])
                mask_areas.append(float(annotation.get("area", width * height)))
        predictions.append(
            ImagePrediction(
                image_info=image_info,
                image_path=image_path,
                boxes_xyxy=boxes_xyxy,
                scores=scores,
                labels=labels,
                segmentations=segmentations if has_segmentations else None,
                mask_areas=mask_areas if has_segmentations else None,
            )
        )
        evaluated_names.append(file_name)

    if not predictions:
        raise ValueError(f"Reusable prediction COCO contains no images: {prediction_coco_path}")
    prediction_count = sum(len(rows) for rows in annotations_by_image.values())
    return (
        predictions,
        evaluated_names,
        {
            "num_evaluated_tiles": len(predictions),
            "num_skipped_edge_tiles": max(0, len(truth_names) - len(predictions)),
            "num_predictions_before_mask_filter": prediction_count,
            "num_predictions_filtered_invalid": 0,
            "prediction_invalid_fraction": 0.0,
            "predictions_reused": 1,
        },
    )


def write_preaggregated_gpkg(
    predictions: list[ImagePrediction],
    output_path: str | Path,
) -> Path:
    try:
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box as shapely_box
    except ImportError as exc:
        raise ImportError(
            "Raster-level benchmarking requires geopandas, rasterio, and shapely. "
            "Use tile_only: true for COCO-only benchmarking."
        ) from exc

    rows: list[dict[str, Any]] = []
    crs = None
    object_id = 1
    transform_cache: dict[Path, tuple[Any, Any]] = {}
    for prediction in predictions:
        transform, tile_crs = _tile_transform_and_crs(
            prediction.image_path,
            rasterio,
            transform_cache,
        )
        if tile_crs is None:
            raise ValueError(
                "Tile has no CRS, so GPKG predictions cannot be georeferenced: "
                f"{prediction.image_path}"
            )
        if crs is None:
            crs = tile_crs
        for box_xyxy, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
        ):
            geom = _box_xyxy_to_geospatial_polygon(box_xyxy, transform, shapely_box)
            rows.append(
                {
                    "object_id": object_id,
                    "tile_path": str(prediction.image_path),
                    "detector_score": float(score),
                    "detector_class": int(label),
                    "geometry": geom,
                }
            )
            object_id += 1

    gdf = gpd.GeoDataFrame(
        rows,
        columns=[
            "object_id",
            "tile_path",
            "detector_score",
            "detector_class",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geodataframe(gdf, output_path)
    return output_path


def aggregate_gpkg_with_nms(
    preaggregated_gpkg_path: str | Path,
    output_path: str | Path,
    score_threshold: float,
    nms_threshold: float,
) -> Path:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Raster-level benchmarking requires geopandas. Use tile_only: true for "
            "COCO-only benchmarking."
        ) from exc

    gdf = gpd.read_file(preaggregated_gpkg_path)
    if "detector_score" not in gdf.columns:
        raise ValueError(f"Missing detector_score column in {preaggregated_gpkg_path}")
    gdf = gdf[gdf["detector_score"] >= float(score_threshold)].copy()
    gdf = gdf.sort_values("detector_score", ascending=False).reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.envelope
    keep_indices: list[int] = []
    active = [True] * len(gdf)
    try:
        spatial_index = gdf.sindex
    except Exception:
        spatial_index = None
    for idx, row in gdf.iterrows():
        if not active[idx]:
            continue
        keep_indices.append(idx)
        geom = row.geometry
        if spatial_index is None:
            candidates = range(idx + 1, len(gdf))
        else:
            candidates = spatial_index.intersection(geom.bounds)
        for candidate_idx in candidates:
            candidate_idx = int(candidate_idx)
            if candidate_idx <= idx or not active[candidate_idx]:
                continue
            if _geometry_iou(geom, gdf.at[candidate_idx, "geometry"]) > float(nms_threshold):
                active[candidate_idx] = False
    aggregated = gdf.loc[keep_indices].copy() if keep_indices else gdf.iloc[:0].copy()
    aggregated["aggregator_score"] = aggregated.get("detector_score", 0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geodataframe(aggregated, output_path)
    return output_path


def _clip_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    if width > 0:
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
    if height > 0:
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def _tile_transform_and_crs(image_path: Path, rasterio, cache: dict[Path, tuple[Any, Any]]):
    if image_path not in cache:
        with rasterio.open(image_path) as src:
            cache[image_path] = (src.transform, src.crs)
    return cache[image_path]


def _box_xyxy_to_geospatial_polygon(box_xyxy: list[float], transform, shapely_box):
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    corners = [
        transform * (x1, y1),
        transform * (x2, y1),
        transform * (x2, y2),
        transform * (x1, y2),
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return shapely_box(min(xs), min(ys), max(xs), max(ys))


def _geometry_iou(first, second) -> float:
    inter = first.intersection(second).area
    union = first.union(second).area
    return float(inter / union) if union > 0 else 0.0


def _resolve_saved_tile_path(tile_dir: str | Path, file_name: str) -> Path:
    tile_dir = Path(tile_dir)
    file_path = Path(file_name)
    for candidate in (tile_dir / file_path, tile_dir / file_path.name):
        if candidate.exists():
            return candidate
    matches = list(tile_dir.rglob(file_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find reusable prediction tile {file_name!r} under {tile_dir}"
    )


def _write_geodataframe(gdf, output_path: Path) -> None:
    if len(gdf) > 0:
        gdf.to_file(output_path, driver="GPKG")
        return
    try:
        gdf.to_file(output_path, driver="GPKG")
        return
    except ValueError as exc:
        if "empty" not in str(exc).lower():
            raise

    try:
        import fiona
    except ImportError as exc:
        raise ImportError("Writing an empty raster prediction GeoPackage requires fiona.") from exc
    properties: dict[str, str] = {}
    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue
        dtype_kind = getattr(gdf[column].dtype, "kind", "")
        if dtype_kind in {"i", "u"}:
            properties[column] = "int"
        elif dtype_kind == "f":
            properties[column] = "float"
        else:
            properties[column] = "str"
    with fiona.open(
        output_path,
        mode="w",
        driver="GPKG",
        crs_wkt=gdf.crs.to_wkt() if gdf.crs is not None else None,
        schema={"geometry": "Polygon", "properties": properties},
    ):
        pass
