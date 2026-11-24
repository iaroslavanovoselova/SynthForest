"""Tile metrics plus CanopyRS-adapted raster RF1 evaluation."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

SIZE_THRESHOLDS_SQ_METERS = {
    "tiny": (0, 9),
    "small": (9, 25),
    "medium": (25, 49),
    "large": (49, 100),
    "giant": (100, float("inf")),
}
SIZE_LABELS = ["tiny", "small", "medium", "large", "giant"]
RF1_50_95_IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]


def tile_level_bbox_metrics(
    preds_coco_path: str | Path,
    truth_coco_path: str | Path,
    max_dets: list[int],
    ground_resolution: Optional[float],
    iou_thresholds: Optional[list[float]] = None,
) -> dict[str, Any]:
    return tile_level_coco_metrics(
        preds_coco_path=preds_coco_path,
        truth_coco_path=truth_coco_path,
        max_dets=max_dets,
        ground_resolution=ground_resolution,
        iou_thresholds=iou_thresholds,
        iou_type="bbox",
    )


def tile_level_segmentation_metrics(
    preds_coco_path: str | Path,
    truth_coco_path: str | Path,
    max_dets: list[int],
    ground_resolution: Optional[float],
    iou_thresholds: Optional[list[float]] = None,
) -> dict[str, Any]:
    return tile_level_coco_metrics(
        preds_coco_path=preds_coco_path,
        truth_coco_path=truth_coco_path,
        max_dets=max_dets,
        ground_resolution=ground_resolution,
        iou_thresholds=iou_thresholds,
        iou_type="segm",
    )


def tile_level_coco_metrics(
    preds_coco_path: str | Path,
    truth_coco_path: str | Path,
    max_dets: list[int],
    ground_resolution: Optional[float],
    iou_thresholds: Optional[list[float]] = None,
    *,
    iou_type: str,
) -> dict[str, Any]:
    if iou_type not in {"bbox", "segm"}:
        raise ValueError("iou_type must be bbox or segm.")
    try:
        from faster_coco_eval.core.coco import COCO
        from faster_coco_eval.core.faster_eval_api import COCOeval_faster as COCOeval
    except ImportError as exc:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "Tile-level COCO benchmarking requires faster-coco-eval or pycocotools. "
                "Install the benchmark dependencies in the training environment."
            ) from exc

    truth_coco = COCO(str(truth_coco_path))
    preds_coco = COCO(str(preds_coco_path))
    _align_coco_datasets_by_name(truth_coco, preds_coco)

    evaluator = COCOeval(truth_coco, preds_coco, iou_type)
    evaluator.params.imgIds = sorted(truth_coco.getImgIds())
    evaluator.params.maxDets = _unique_max_dets(max_dets)
    primary_iou_thresholds = _primary_iou_thresholds(iou_thresholds)
    evaluator.params.iouThrs = np.array(
        sorted({*primary_iou_thresholds, 0.5, 0.75}),
        dtype=float,
    )
    if ground_resolution is not None:
        area_ranges = _area_ranges_pixels_from_gsd(float(ground_resolution))
        evaluator.params.areaRng = [
            [0, 1e10],
            [area_ranges["tiny"][0], area_ranges["tiny"][1]],
            [area_ranges["small"][0], area_ranges["small"][1]],
            [area_ranges["medium"][0], area_ranges["medium"][1]],
            [area_ranges["large"][0], area_ranges["large"][1]],
            [area_ranges["giant"][0], area_ranges["giant"][1]],
        ]
        evaluator.params.areaRngLbl = ["all", "tiny", "small", "medium", "large", "giant"]
    evaluator.evaluate()
    evaluator.accumulate()

    metrics = _summarize_coco_eval(evaluator, primary_iou_thresholds=primary_iou_thresholds)
    metrics["num_images"] = len(truth_coco.dataset.get("images", []))
    metrics["num_truths"] = len(truth_coco.dataset.get("annotations", []))
    metrics["num_preds"] = len(preds_coco.dataset.get("annotations", []))
    return metrics


def raster_level_bbox_metrics(
    preds_gpkg_path: str | Path,
    truth_gpkg_path: str | Path,
    aoi_gpkg_path: str | Path | None,
    ground_resolution: float,
    iou_thresholds: list[float],
) -> dict[str, Any]:
    try:
        import geopandas as gpd
        from shapely.affinity import affine_transform
        from shapely.validation import make_valid
    except ImportError as exc:
        raise ImportError(
            "Raster-level benchmarking requires geopandas and shapely. Use tile_only: true "
            "for COCO-only benchmarking."
        ) from exc

    pred_gdf = gpd.read_file(preds_gpkg_path)
    truth_gdf = gpd.read_file(truth_gpkg_path)
    pred_gdf = pred_gdf.copy()
    truth_gdf = truth_gdf.copy()
    pred_gdf["geometry"] = pred_gdf.geometry.envelope
    truth_gdf["geometry"] = truth_gdf.geometry.envelope

    if truth_gdf.crs is None:
        raise ValueError(f"Truth GPKG has no CRS: {truth_gpkg_path}")
    if pred_gdf.crs is None:
        raise ValueError(f"Prediction GPKG has no CRS: {preds_gpkg_path}")

    common_crs = truth_gdf.crs
    if not common_crs.is_projected:
        bounds = truth_gdf.total_bounds
        common_crs = _utm_epsg_crs(
            lon=(bounds[0] + bounds[2]) / 2.0,
            lat=(bounds[1] + bounds[3]) / 2.0,
        )
    pred_gdf = pred_gdf.to_crs(common_crs)
    truth_gdf = truth_gdf.to_crs(common_crs)

    if aoi_gpkg_path is not None:
        aoi_gdf = gpd.read_file(aoi_gpkg_path).to_crs(common_crs)
        aoi_union = aoi_gdf.geometry.unary_union
        truth_gdf = _filter_min_overlap(truth_gdf, aoi_union, make_valid=make_valid)
        pred_gdf = _filter_min_overlap(pred_gdf, aoi_union, make_valid=make_valid)
        truth_gdf = gpd.overlay(truth_gdf, aoi_gdf, how="intersection")
        pred_gdf = gpd.overlay(pred_gdf, aoi_gdf, how="intersection")
    else:
        warnings.warn(
            "AOI GPKG path is None. Raster-level metrics will not apply AOI filtering.",
            stacklevel=2,
        )

    truth_gdf, pred_gdf = _move_gdfs_to_ground_resolution(
        truth_gdf,
        pred_gdf,
        ground_resolution=float(ground_resolution),
        affine_transform=affine_transform,
        make_valid=make_valid,
    )

    score_column = _score_column(pred_gdf)
    pred_gdf = pred_gdf.sort_values(score_column, ascending=False).reset_index(drop=True)
    truth_gdf = truth_gdf.reset_index(drop=True)

    area_ranges = _area_ranges_pixels_from_gsd(float(ground_resolution))
    gt_sizes = [_size_label(geom.area, area_ranges) for geom in truth_gdf.geometry]
    pred_sizes = [_size_label(geom.area, area_ranges) for geom in pred_gdf.geometry]

    iou_thresholds = sorted({float(value) for value in iou_thresholds})
    candidates_by_pred = _candidate_ious(pred_gdf, truth_gdf)
    per_size_results: dict[str, list[dict[str, Any]]] = {
        label: [] for label in ["all", *SIZE_LABELS]
    }

    for iou_threshold in iou_thresholds:
        for size_label in ["all", *SIZE_LABELS]:
            gt_indices = [
                idx
                for idx, label in enumerate(gt_sizes)
                if size_label == "all" or label == size_label
            ]
            pred_indices = [
                idx
                for idx, label in enumerate(pred_sizes)
                if size_label == "all" or label == size_label
            ]
            per_size_results[size_label].append(
                _greedy_match(candidates_by_pred, gt_indices, pred_indices, iou_threshold)
                | {"iou_threshold": iou_threshold}
            )

    all_results = per_size_results["all"]
    metrics = {
        "precision": (
            float(np.mean([row["precision"] for row in all_results])) if all_results else 0.0
        ),
        "recall": float(np.mean([row["recall"] for row in all_results])) if all_results else 0.0,
        "f1": float(np.mean([row["f1"] for row in all_results])) if all_results else 0.0,
        "tp": int(all_results[0]["tp"]) if all_results else 0,
        "fp": int(all_results[0]["fp"]) if all_results else 0,
        "fn": int(all_results[0]["fn"]) if all_results else 0,
        "num_truths": int(len(truth_gdf)),
        "num_preds": int(len(pred_gdf)),
        "num_images": 1,
        "precision_per_iou": [float(row["precision"]) for row in all_results],
        "recall_per_iou": [float(row["recall"]) for row in all_results],
        "f1_per_iou": [float(row["f1"]) for row in all_results],
        "iou_thresholds": iou_thresholds,
        "size_labels": list(SIZE_LABELS),
    }
    for target, suffix in [(0.5, "50"), (0.75, "75")]:
        for idx, threshold in enumerate(iou_thresholds):
            if math.isclose(threshold, target):
                metrics[f"precision_{suffix}"] = float(all_results[idx]["precision"])
                metrics[f"recall_{suffix}"] = float(all_results[idx]["recall"])
                metrics[f"f1_{suffix}"] = float(all_results[idx]["f1"])
    metrics["rf1_75"] = _mean_per_iou_metric(
        metrics,
        requested_iou_thresholds=[0.75],
        metric_name="f1",
    )
    metrics["rf1_50_95"] = _mean_per_iou_metric(
        metrics,
        requested_iou_thresholds=RF1_50_95_IOU_THRESHOLDS,
        metric_name="f1",
    )

    for size_label in SIZE_LABELS:
        rows = per_size_results[size_label]
        metrics[f"precision_{size_label}"] = float(np.mean([row["precision"] for row in rows]))
        metrics[f"recall_{size_label}"] = float(np.mean([row["recall"] for row in rows]))
        metrics[f"f1_{size_label}"] = float(np.mean([row["f1"] for row in rows]))
        metrics[f"precision_per_iou_{size_label}"] = [float(row["precision"]) for row in rows]
        metrics[f"recall_per_iou_{size_label}"] = [float(row["recall"]) for row in rows]
        metrics[f"f1_per_iou_{size_label}"] = [float(row["f1"]) for row in rows]
    return metrics


def weighted_average_raster_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    weights = np.array([float(row.get("num_truths", 0.0)) for row in rows], dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(rows), dtype=float)
    out: dict[str, Any] = {
        "num_truths": int(sum(int(row.get("num_truths", 0)) for row in rows)),
        "num_preds": int(sum(int(row.get("num_preds", 0)) for row in rows)),
        "num_images": int(sum(int(row.get("num_images", 0)) for row in rows)),
        "tp": int(sum(int(row.get("tp", 0)) for row in rows)),
        "fp": int(sum(int(row.get("fp", 0)) for row in rows)),
        "fn": int(sum(int(row.get("fn", 0)) for row in rows)),
    }
    skip = {"location", "product_name", "size_labels", "iou_thresholds"}
    for key in sorted(set().union(*(row.keys() for row in rows))):
        if key in skip or key in out:
            continue
        values = [row.get(key) for row in rows]
        if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            out[key] = float(np.average(np.array(values, dtype=float), weights=weights))
        elif all(isinstance(value, list) for value in values):
            try:
                arrays = [np.array(value, dtype=float) for value in values]
            except (TypeError, ValueError):
                continue
            lengths = {array.shape for array in arrays}
            if len(lengths) == 1:
                out[key] = np.average(np.stack(arrays), axis=0, weights=weights).tolist()
    first = rows[0]
    if "iou_thresholds" in first:
        out["iou_thresholds"] = first["iou_thresholds"]
    if "size_labels" in first:
        out["size_labels"] = first["size_labels"]
    return out


def raster_f1_for_iou_thresholds(
    metrics: dict[str, Any],
    requested_iou_thresholds: list[float],
) -> float:
    """Return the RF1 mean for an explicit set of IoU thresholds."""

    return _mean_per_iou_metric(
        metrics,
        requested_iou_thresholds=requested_iou_thresholds,
        metric_name="f1",
    )


def _mean_per_iou_metric(
    metrics: dict[str, Any],
    *,
    requested_iou_thresholds: list[float],
    metric_name: str,
) -> float:
    available = [float(value) for value in metrics.get("iou_thresholds", [])]
    values = [float(value) for value in metrics.get(f"{metric_name}_per_iou", [])]
    selected: list[float] = []
    for requested in requested_iou_thresholds:
        match = next(
            (
                index
                for index, available_threshold in enumerate(available)
                if math.isclose(float(requested), available_threshold)
            ),
            None,
        )
        if match is None or match >= len(values):
            return float("nan")
        selected.append(values[match])
    return float(np.mean(selected)) if selected else float("nan")


def _area_ranges_pixels_from_gsd(ground_resolution: float) -> dict[str, tuple[float, float]]:
    pixel_area_per_sq_meter = 1.0 / (ground_resolution**2)
    ranges = {}
    for label, (min_m2, max_m2) in SIZE_THRESHOLDS_SQ_METERS.items():
        ranges[label] = (
            min_m2 * pixel_area_per_sq_meter,
            max_m2 * pixel_area_per_sq_meter if max_m2 != float("inf") else 1e10,
        )
    return ranges


def _size_label(area: float, area_ranges: dict[str, tuple[float, float]]) -> str:
    for label in SIZE_LABELS:
        low, high = area_ranges[label]
        if low <= area < high:
            return label
    return SIZE_LABELS[-1]


def _unique_max_dets(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) > 0})


def _align_coco_datasets_by_name(truth_coco, preds_coco) -> None:
    preds_by_name = {img["file_name"]: img for img in preds_coco.dataset.get("images", [])}
    id_mapping = {}
    new_images = []
    for truth_img in truth_coco.dataset.get("images", []):
        file_name = truth_img["file_name"]
        truth_id = truth_img["id"]
        pred_img = preds_by_name.get(file_name)
        if pred_img is None:
            new_images.append(dict(truth_img))
            continue
        id_mapping[pred_img["id"]] = truth_id
        new_img = dict(pred_img)
        new_img["id"] = truth_id
        new_images.append(new_img)
    preds_coco.dataset["images"] = new_images
    new_annotations = []
    for annotation in preds_coco.dataset.get("annotations", []):
        old_image_id = annotation["image_id"]
        if old_image_id not in id_mapping:
            continue
        new_annotation = dict(annotation)
        new_annotation["image_id"] = id_mapping[old_image_id]
        new_annotations.append(new_annotation)
    preds_coco.dataset["annotations"] = new_annotations
    preds_coco.createIndex()


def _primary_iou_thresholds(iou_thresholds: Optional[list[float]]) -> list[float]:
    if not iou_thresholds:
        return [round(0.50 + 0.05 * i, 2) for i in range(10)]
    return sorted({round(float(value), 4) for value in iou_thresholds})


def _summarize_coco_eval(
    evaluator,
    primary_iou_thresholds: Optional[list[float]] = None,
) -> dict[str, float]:
    max_det = evaluator.params.maxDets[-1]
    area_labels = set(evaluator.params.areaRngLbl)
    use_extended_sizes = all(label in area_labels for label in ["tiny", "giant"])
    metrics: dict[str, float] = {
        "AP": _coco_stat(
            evaluator,
            ap=True,
            iou_thrs=primary_iou_thresholds,
            max_det=max_det,
        ),
        "AP50": _coco_stat(evaluator, ap=True, iou_thr=0.5, max_det=max_det),
        "AP75": _coco_stat(evaluator, ap=True, iou_thr=0.75, max_det=max_det),
    }
    if use_extended_sizes:
        metrics["AP_tiny"] = _coco_stat(
            evaluator,
            ap=True,
            iou_thrs=primary_iou_thresholds,
            area="tiny",
            max_det=max_det,
        )
    metrics["AP_small"] = _coco_stat(
        evaluator,
        ap=True,
        iou_thrs=primary_iou_thresholds,
        area="small",
        max_det=max_det,
    )
    metrics["AP_medium"] = _coco_stat(
        evaluator,
        ap=True,
        iou_thrs=primary_iou_thresholds,
        area="medium",
        max_det=max_det,
    )
    metrics["AP_large"] = _coco_stat(
        evaluator,
        ap=True,
        iou_thrs=primary_iou_thresholds,
        area="large",
        max_det=max_det,
    )
    if use_extended_sizes:
        metrics["AP_giant"] = _coco_stat(
            evaluator,
            ap=True,
            iou_thrs=primary_iou_thresholds,
            area="giant",
            max_det=max_det,
        )
    for det in evaluator.params.maxDets[:3]:
        metrics[f"AR_{det}"] = _coco_stat(
            evaluator,
            ap=False,
            iou_thrs=primary_iou_thresholds,
            max_det=det,
        )
    metrics[f"AR_{max_det}"] = _coco_stat(
        evaluator,
        ap=False,
        iou_thrs=primary_iou_thresholds,
        max_det=max_det,
    )
    metrics["AR"] = metrics[f"AR_{max_det}"]
    metrics["AR50"] = _coco_stat(evaluator, ap=False, iou_thr=0.5, max_det=max_det)
    metrics["AR75"] = _coco_stat(evaluator, ap=False, iou_thr=0.75, max_det=max_det)
    if use_extended_sizes:
        metrics["AR_tiny"] = _coco_stat(
            evaluator,
            ap=False,
            iou_thrs=primary_iou_thresholds,
            area="tiny",
            max_det=max_det,
        )
    metrics["AR_small"] = _coco_stat(
        evaluator,
        ap=False,
        iou_thrs=primary_iou_thresholds,
        area="small",
        max_det=max_det,
    )
    metrics["AR_medium"] = _coco_stat(
        evaluator,
        ap=False,
        iou_thrs=primary_iou_thresholds,
        area="medium",
        max_det=max_det,
    )
    metrics["AR_large"] = _coco_stat(
        evaluator,
        ap=False,
        iou_thrs=primary_iou_thresholds,
        area="large",
        max_det=max_det,
    )
    if use_extended_sizes:
        metrics["AR_giant"] = _coco_stat(
            evaluator,
            ap=False,
            iou_thrs=primary_iou_thresholds,
            area="giant",
            max_det=max_det,
        )
    return metrics


def _coco_stat(
    evaluator,
    ap: bool,
    iou_thr: Optional[float] = None,
    iou_thrs: Optional[list[float]] = None,
    area: str = "all",
    max_det: int = 100,
) -> float:
    params = evaluator.params
    area_indices = [idx for idx, label in enumerate(params.areaRngLbl) if label == area]
    max_det_indices = [idx for idx, value in enumerate(params.maxDets) if value == max_det]
    if not area_indices or not max_det_indices:
        return -1.0
    if ap:
        values = evaluator.eval["precision"]
        if iou_thr is not None:
            iou_indices = np.where(np.isclose(params.iouThrs, iou_thr))[0]
            values = values[iou_indices]
        elif iou_thrs is not None:
            iou_indices = _iou_indices(params.iouThrs, iou_thrs)
            values = values[iou_indices]
        values = values[:, :, :, area_indices, max_det_indices]
    else:
        values = evaluator.eval["recall"]
        if iou_thr is not None:
            iou_indices = np.where(np.isclose(params.iouThrs, iou_thr))[0]
            values = values[iou_indices]
        elif iou_thrs is not None:
            iou_indices = _iou_indices(params.iouThrs, iou_thrs)
            values = values[iou_indices]
        values = values[:, :, area_indices, max_det_indices]
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def _iou_indices(available_iou_thresholds, requested_iou_thresholds: list[float]):
    indices = []
    for requested in requested_iou_thresholds:
        matches = np.where(np.isclose(available_iou_thresholds, requested))[0]
        if matches.size:
            indices.append(int(matches[0]))
    if not indices:
        return slice(None)
    return np.array(indices, dtype=int)


def _score_column(gdf) -> str:
    for column in ["score", "aggregator_score", "detector_score", "segmentation_score"]:
        if column in gdf.columns:
            return column
    raise ValueError(
        "No valid score column found in predictions GeoDataFrame. Expected one of: "
        "score, aggregator_score, detector_score, segmentation_score."
    )


def _filter_min_overlap(gdf, aoi_geom, make_valid, min_frac: float = 0.4):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )
    aoi_geom = make_valid(aoi_geom) if not aoi_geom.is_valid else aoi_geom
    original_areas = gdf.geometry.area
    intersection_areas = gdf.geometry.intersection(aoi_geom).area
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = intersection_areas / original_areas.replace({0: np.nan})
    return gdf[fractions.fillna(False) >= min_frac].copy()


def _move_gdfs_to_ground_resolution(
    truth_gdf,
    pred_gdf,
    ground_resolution,
    affine_transform,
    make_valid,
):
    if truth_gdf.crs != pred_gdf.crs:
        pred_gdf = pred_gdf.to_crs(truth_gdf.crs)
    if not truth_gdf.crs.is_projected:
        bounds = truth_gdf.total_bounds
        crs = _utm_epsg_crs(
            lon=(bounds[0] + bounds[2]) / 2.0,
            lat=(bounds[1] + bounds[3]) / 2.0,
        )
        truth_gdf = truth_gdf.to_crs(crs)
        pred_gdf = pred_gdf.to_crs(crs)
    if len(pred_gdf) == 0:
        combined_bounds = truth_gdf.total_bounds
    elif len(truth_gdf) == 0:
        combined_bounds = pred_gdf.total_bounds
    else:
        tb = truth_gdf.total_bounds
        pb = pred_gdf.total_bounds
        combined_bounds = [
            min(tb[0], pb[0]),
            min(tb[1], pb[1]),
            max(tb[2], pb[2]),
            max(tb[3], pb[3]),
        ]
    affine_params = [
        1.0 / ground_resolution,
        0,
        0,
        1.0 / ground_resolution,
        -combined_bounds[0] / ground_resolution,
        -combined_bounds[1] / ground_resolution,
    ]
    truth_gdf = truth_gdf.copy()
    pred_gdf = pred_gdf.copy()
    truth_gdf["geometry"] = truth_gdf.geometry.apply(
        lambda geom: affine_transform(geom, affine_params)
    )
    pred_gdf["geometry"] = pred_gdf.geometry.apply(
        lambda geom: affine_transform(geom, affine_params)
    )
    truth_gdf["geometry"] = truth_gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )
    pred_gdf["geometry"] = pred_gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )
    return truth_gdf[truth_gdf.geometry.notna()].copy(), pred_gdf[pred_gdf.geometry.notna()].copy()


def _candidate_ious(pred_gdf, truth_gdf) -> list[list[tuple[int, float]]]:
    if len(pred_gdf) == 0 or len(truth_gdf) == 0:
        return [[] for _ in range(len(pred_gdf))]
    try:
        truth_sindex = truth_gdf.sindex
    except Exception:
        truth_sindex = None

    candidate_ious: list[list[tuple[int, float]]] = []
    for _, pred in pred_gdf.iterrows():
        if truth_sindex is None:
            candidates = range(len(truth_gdf))
        else:
            candidates = list(truth_sindex.intersection(pred.geometry.bounds))
        rows: list[tuple[int, float]] = []
        for truth_idx in candidates:
            truth_geom = truth_gdf.at[truth_idx, "geometry"]
            inter_area = pred.geometry.intersection(truth_geom).area
            union_area = pred.geometry.union(truth_geom).area
            iou = float(inter_area / union_area) if union_area > 0 else 0.0
            rows.append((int(truth_idx), iou))
        rows.sort(key=lambda item: item[1], reverse=True)
        candidate_ious.append(rows)
    return candidate_ious


def _greedy_match(
    candidates_by_pred: list[list[tuple[int, float]]],
    gt_indices: list[int],
    pred_indices: list[int],
    iou_threshold: float,
) -> dict[str, Any]:
    if not gt_indices and not pred_indices:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    gt_idx_to_local = {idx: local_idx for local_idx, idx in enumerate(gt_indices)}
    matched = np.zeros(len(gt_indices), dtype=bool)
    tp = 0
    fp = 0
    for pred_idx in pred_indices:
        match_local_idx = None
        for gt_idx, iou in candidates_by_pred[pred_idx]:
            if iou < iou_threshold:
                break
            if gt_idx not in gt_idx_to_local:
                continue
            local_idx = gt_idx_to_local[gt_idx]
            if not matched[local_idx]:
                match_local_idx = local_idx
                break
        if match_local_idx is None:
            fp += 1
        else:
            tp += 1
            matched[match_local_idx] = True
    fn = int((~matched).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _utm_epsg_crs(lon: float, lat: float) -> str:
    zone = int((float(lon) + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"
