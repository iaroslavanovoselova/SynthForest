from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import (
    closing as skimage_closing,
)
from skimage.morphology import (
    convex_hull_image,
    disk,
)
from skimage.morphology import (
    opening as skimage_opening,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _VisibleCrownPostprocessParams:
    gsd: float
    min_visible_px: int
    min_final_px: int
    min_visible_brightness: float
    visible_brightness_percentile: float
    min_component_px: int
    gap_fill_radius_px: int
    sliver_px: int
    sliver_min_bbox_fill_ratio: float
    sliver_min_bbox_axis_ratio: float
    sliver_min_convex_fill_ratio: float
    sliver_min_largest_component_fraction: float
    boundary_regularization_min_px: int
    boundary_smooth_radius_px: int
    boundary_core_erosion_px: int
    boundary_component_merge_radius_px: int
    boundary_component_min_px: int
    boundary_component_keep_fraction: float
    boundary_component_merge_max_area_growth_fraction: float
    boundary_max_area_growth_fraction: float
    nested_island_parent_min_px: int
    nested_island_max_hole_area_fraction: float
    nested_island_absorb_max_area_fraction: float
    nested_island_absorb_max_px: int
    final_topology_min_px: int
    ordered_height_percentile: float
    ordered_area_priority_weight: float
    ordered_absorb_height_margin_m: float
    ordered_absorb_overlap_fraction: float
    ordered_absorb_max_px: int
    ordered_min_retained_visible_fraction: float
    ordered_component_keep_fraction: float
    ordered_outer_envelope_min_px: int
    ordered_outer_envelope_radius_px: int
    ordered_outer_envelope_max_growth_fraction: float
    ordered_reclaim_height_margin_m: float
    ordered_rescue_min_visible_fraction: float


@dataclass(frozen=True)
class _OrderedCrownStats:
    uid: int
    raw_area_px: int
    height_priority_m: float
    height_p50_m: float
    height_max_m: float
    priority: float


@dataclass(frozen=True)
class _OrderedCrownCandidate:
    uid: int
    y0: int
    y1: int
    x0: int
    x1: int
    seed: np.ndarray
    proposal: np.ndarray
    raw_area_px: int
    seed_area_px: int
    proposal_area_px: int
    height_priority_m: float
    height_p50_m: float
    height_max_m: float
    priority: float


@dataclass(frozen=True)
class _LabelRegionStats:
    ids: np.ndarray
    area_by_id: dict[int, int]
    bbox_by_id: dict[int, tuple[int, int, int, int]]


def _cfg_value(config: object | None, name: str, default: object) -> object:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _positive_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, number)


def _compute_visible_crown_postprocess_params(
    gsd: float,
    config: object | None,
) -> _VisibleCrownPostprocessParams:
    gsd = max(1e-3, float(gsd))
    pixel_area_m2 = gsd * gsd
    min_visible_area_m2 = _positive_float(_cfg_value(config, "min_visible_area_m2", 1.0), 1.0)
    min_final_area_m2 = _positive_float(_cfg_value(config, "min_final_area_m2", 1.5), 1.5)
    min_visible_brightness = min(
        1.0,
        _positive_float(_cfg_value(config, "min_visible_brightness", 0.08), 0.08),
    )
    visible_brightness_percentile = min(
        100.0,
        _positive_float(
            _cfg_value(config, "visible_brightness_percentile", 90.0),
            90.0,
        ),
    )
    min_component_area_m2 = _positive_float(_cfg_value(config, "min_component_area_m2", 0.25), 0.25)
    gap_fill_radius_m = _positive_float(_cfg_value(config, "gap_fill_radius_m", 1.75), 1.75)
    sliver_area_m2 = _positive_float(_cfg_value(config, "sliver_area_m2", 10.0), 10.0)
    sliver_min_bbox_fill_ratio = min(
        1.0, _positive_float(_cfg_value(config, "sliver_min_bbox_fill_ratio", 0.22), 0.22)
    )
    sliver_min_bbox_axis_ratio = min(
        1.0, _positive_float(_cfg_value(config, "sliver_min_bbox_axis_ratio", 0.30), 0.30)
    )
    sliver_min_convex_fill_ratio = min(
        1.0, _positive_float(_cfg_value(config, "sliver_min_convex_fill_ratio", 0.42), 0.42)
    )
    sliver_min_largest_component_fraction = min(
        1.0,
        _positive_float(_cfg_value(config, "sliver_min_largest_component_fraction", 0.70), 0.70),
    )
    boundary_regularization_min_area_m2 = _positive_float(
        _cfg_value(config, "boundary_regularization_min_area_m2", 8.0), 8.0
    )
    boundary_smooth_radius_m = _positive_float(
        _cfg_value(config, "boundary_smooth_radius_m", 1.0), 1.0
    )
    boundary_core_erosion_m = _positive_float(
        _cfg_value(config, "boundary_core_erosion_m", 0.50), 0.50
    )
    boundary_component_merge_radius_m = _positive_float(
        _cfg_value(config, "boundary_component_merge_radius_m", 3.0), 3.0
    )
    boundary_component_min_area_m2 = _positive_float(
        _cfg_value(config, "boundary_component_min_area_m2", 0.25), 0.25
    )
    boundary_component_keep_fraction = min(
        1.0, _positive_float(_cfg_value(config, "boundary_component_keep_fraction", 0.02), 0.02)
    )
    boundary_component_merge_max_area_growth_fraction = min(
        2.0,
        _positive_float(
            _cfg_value(config, "boundary_component_merge_max_area_growth_fraction", 1.20), 1.20
        ),
    )
    boundary_max_area_growth_fraction = min(
        2.0, _positive_float(_cfg_value(config, "boundary_max_area_growth_fraction", 0.90), 0.90)
    )
    nested_island_parent_min_area_m2 = _positive_float(
        _cfg_value(config, "nested_island_parent_min_area_m2", 8.0), 8.0
    )
    nested_island_max_hole_area_fraction = min(
        1.0, _positive_float(_cfg_value(config, "nested_island_max_hole_area_fraction", 0.35), 0.35)
    )
    nested_island_absorb_max_area_fraction = min(
        1.0,
        _positive_float(_cfg_value(config, "nested_island_absorb_max_area_fraction", 0.35), 0.35),
    )
    nested_island_absorb_max_area_m2 = _positive_float(
        _cfg_value(config, "nested_island_absorb_max_area_m2", 30.0), 30.0
    )
    final_topology_min_area_m2 = _positive_float(
        _cfg_value(config, "final_topology_min_area_m2", 5.0), 5.0
    )
    ordered_height_percentile = min(
        100.0,
        _positive_float(
            _cfg_value(config, "ordered_height_percentile", 90.0),
            90.0,
        ),
    )
    ordered_area_priority_weight = _positive_float(
        _cfg_value(config, "ordered_area_priority_weight", 0.12),
        0.12,
    )
    ordered_absorb_height_margin_m = _positive_float(
        _cfg_value(config, "ordered_absorb_height_margin_m", 2.0),
        2.0,
    )
    ordered_absorb_overlap_fraction = min(
        1.0,
        _positive_float(
            _cfg_value(config, "ordered_absorb_overlap_fraction", 0.60),
            0.60,
        ),
    )
    ordered_absorb_max_area_m2 = _positive_float(
        _cfg_value(config, "ordered_absorb_max_area_m2", 30.0),
        30.0,
    )
    ordered_min_retained_visible_fraction = min(
        1.0,
        _positive_float(
            _cfg_value(config, "ordered_min_retained_visible_fraction", 0.25),
            0.25,
        ),
    )
    ordered_component_keep_fraction = min(
        1.0,
        _positive_float(
            _cfg_value(config, "ordered_component_keep_fraction", 0.15),
            0.15,
        ),
    )
    ordered_outer_envelope_min_area_m2 = _positive_float(
        _cfg_value(config, "ordered_outer_envelope_min_area_m2", 3.0),
        3.0,
    )
    ordered_outer_envelope_radius_m = _positive_float(
        _cfg_value(config, "ordered_outer_envelope_radius_m", 2.0),
        2.0,
    )
    ordered_outer_envelope_max_growth_fraction = min(
        10.0,
        _positive_float(
            _cfg_value(config, "ordered_outer_envelope_max_growth_fraction", 0.75),
            0.75,
        ),
    )
    ordered_reclaim_height_margin_m = _positive_float(
        _cfg_value(config, "ordered_reclaim_height_margin_m", 0.5),
        0.5,
    )
    ordered_rescue_min_visible_fraction = min(
        1.0,
        _positive_float(
            _cfg_value(config, "ordered_rescue_min_visible_fraction", 0.02),
            0.02,
        ),
    )

    return _VisibleCrownPostprocessParams(
        gsd=gsd,
        min_visible_px=max(1, int(round(min_visible_area_m2 / pixel_area_m2))),
        min_final_px=max(1, int(round(min_final_area_m2 / pixel_area_m2))),
        min_visible_brightness=float(min_visible_brightness),
        visible_brightness_percentile=float(visible_brightness_percentile),
        min_component_px=max(1, int(round(min_component_area_m2 / pixel_area_m2))),
        gap_fill_radius_px=max(0, int(round(gap_fill_radius_m / gsd))),
        sliver_px=max(1, int(round(sliver_area_m2 / pixel_area_m2))),
        sliver_min_bbox_fill_ratio=float(sliver_min_bbox_fill_ratio),
        sliver_min_bbox_axis_ratio=float(sliver_min_bbox_axis_ratio),
        sliver_min_convex_fill_ratio=float(sliver_min_convex_fill_ratio),
        sliver_min_largest_component_fraction=float(sliver_min_largest_component_fraction),
        boundary_regularization_min_px=max(
            1,
            int(round(boundary_regularization_min_area_m2 / pixel_area_m2)),
        ),
        boundary_smooth_radius_px=max(0, int(round(boundary_smooth_radius_m / gsd))),
        boundary_core_erosion_px=max(0, int(round(boundary_core_erosion_m / gsd))),
        boundary_component_merge_radius_px=max(
            0,
            int(round(boundary_component_merge_radius_m / gsd)),
        ),
        boundary_component_min_px=max(
            1,
            int(round(boundary_component_min_area_m2 / pixel_area_m2)),
        ),
        boundary_component_keep_fraction=float(boundary_component_keep_fraction),
        boundary_component_merge_max_area_growth_fraction=float(
            boundary_component_merge_max_area_growth_fraction
        ),
        boundary_max_area_growth_fraction=float(boundary_max_area_growth_fraction),
        nested_island_parent_min_px=max(
            1,
            int(round(nested_island_parent_min_area_m2 / pixel_area_m2)),
        ),
        nested_island_max_hole_area_fraction=float(nested_island_max_hole_area_fraction),
        nested_island_absorb_max_area_fraction=float(nested_island_absorb_max_area_fraction),
        nested_island_absorb_max_px=max(
            1,
            int(round(nested_island_absorb_max_area_m2 / pixel_area_m2)),
        ),
        final_topology_min_px=max(
            1,
            int(round(final_topology_min_area_m2 / pixel_area_m2)),
        ),
        ordered_height_percentile=float(ordered_height_percentile),
        ordered_area_priority_weight=float(ordered_area_priority_weight),
        ordered_absorb_height_margin_m=float(ordered_absorb_height_margin_m),
        ordered_absorb_overlap_fraction=float(ordered_absorb_overlap_fraction),
        ordered_absorb_max_px=max(
            1,
            int(round(ordered_absorb_max_area_m2 / pixel_area_m2)),
        ),
        ordered_min_retained_visible_fraction=float(ordered_min_retained_visible_fraction),
        ordered_component_keep_fraction=float(ordered_component_keep_fraction),
        ordered_outer_envelope_min_px=max(
            1,
            int(round(ordered_outer_envelope_min_area_m2 / pixel_area_m2)),
        ),
        ordered_outer_envelope_radius_px=max(
            0,
            int(round(ordered_outer_envelope_radius_m / gsd)),
        ),
        ordered_outer_envelope_max_growth_fraction=float(
            ordered_outer_envelope_max_growth_fraction
        ),
        ordered_reclaim_height_margin_m=float(ordered_reclaim_height_margin_m),
        ordered_rescue_min_visible_fraction=float(ordered_rescue_min_visible_fraction),
    )


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = bbox
    height, width = shape
    return (
        max(0, y0 - pad),
        min(height, y1 + pad),
        max(0, x0 - pad),
        min(width, x1 + pad),
    )


@lru_cache(maxsize=None)
def _disk_footprint(radius: int) -> np.ndarray | None:
    radius = int(radius)
    if radius <= 0:
        return None
    return disk(radius)


def _close_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    footprint = _disk_footprint(radius)
    if footprint is None:
        return mask.astype(bool, copy=True)
    return skimage_closing(mask.astype(bool, copy=False), footprint=footprint)


def _open_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    footprint = _disk_footprint(radius)
    if footprint is None:
        return mask.astype(bool, copy=True)
    return skimage_opening(mask.astype(bool, copy=False), footprint=footprint)


def _apply_local_bool_op(
    mask: np.ndarray,
    radius: int,
    op: str,
) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool, copy=True)

    bbox = _mask_bbox(mask)
    if bbox is None:
        return np.zeros_like(mask, dtype=bool)

    # Two radii are enough for the dilation+erosion dependency radius used by
    # closing/opening while keeping large empty crop margins out of scipy/skimage.
    y0, y1, x0, x1 = _pad_bbox(bbox, mask.shape, int(radius) * 2 + 1)
    local = mask[y0:y1, x0:x1]
    if op == "closing":
        local_result = _close_bool(local, radius)
    elif op == "opening":
        local_result = _open_bool(local, radius)
    else:
        raise ValueError(f"Unknown boolean morphology operation: {op}")

    result = np.zeros_like(mask, dtype=bool)
    result[y0:y1, x0:x1] = local_result
    return result


def _compute_label_region_stats(label_map: np.ndarray) -> _LabelRegionStats:
    labels = np.asarray(label_map)
    height, width = labels.shape
    flat = labels.ravel()
    nonzero_positions = np.flatnonzero(flat > 0)
    if nonzero_positions.size == 0:
        return _LabelRegionStats(
            ids=np.asarray([], dtype=np.int32),
            area_by_id={},
            bbox_by_id={},
        )

    nonzero_labels = flat[nonzero_positions].astype(np.int64, copy=False)
    ids, inverse, counts = np.unique(
        nonzero_labels,
        return_inverse=True,
        return_counts=True,
    )
    ys = nonzero_positions // width
    xs = nonzero_positions - ys * width

    min_y = np.full(ids.shape, height, dtype=np.int64)
    min_x = np.full(ids.shape, width, dtype=np.int64)
    max_y = np.zeros(ids.shape, dtype=np.int64)
    max_x = np.zeros(ids.shape, dtype=np.int64)
    np.minimum.at(min_y, inverse, ys)
    np.minimum.at(min_x, inverse, xs)
    np.maximum.at(max_y, inverse, ys)
    np.maximum.at(max_x, inverse, xs)

    area_by_id = {int(uid): int(area) for uid, area in zip(ids, counts)}
    bbox_by_id = {
        int(uid): (
            int(y0),
            int(y1) + 1,
            int(x0),
            int(x1) + 1,
        )
        for uid, y0, y1, x0, x1 in zip(ids, min_y, max_y, min_x, max_x)
    }
    return _LabelRegionStats(
        ids=ids.astype(np.int32, copy=False),
        area_by_id=area_by_id,
        bbox_by_id=bbox_by_id,
    )


def _filter_small_components(
    mask: np.ndarray,
    min_component_px: int,
    *,
    keep_largest_if_all_small: bool = False,
) -> tuple[np.ndarray, int, int, float]:
    if not mask.any():
        return mask.astype(bool, copy=True), 0, 0, 0.0

    labeled, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count <= 0:
        return np.zeros_like(mask, dtype=bool), int(mask.sum()), 0, 0.0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep_labels = np.flatnonzero(sizes >= max(1, int(min_component_px)))
    if keep_labels.size == 0 and keep_largest_if_all_small:
        keep_labels = np.array([int(np.argmax(sizes))], dtype=np.int64)

    kept = np.isin(labeled, keep_labels) if keep_labels.size else np.zeros_like(mask, dtype=bool)
    removed_px = int(mask.sum() - kept.sum())
    largest_fraction = float(sizes.max() / max(1, int(mask.sum())))
    return kept, removed_px, int(component_count), largest_fraction


def _mask_shape_metrics(mask: np.ndarray) -> dict[str, float]:
    area = int(mask.sum())
    if area <= 0:
        return {
            "area_px": 0.0,
            "fill_ratio": 0.0,
            "bbox_axis_ratio": 0.0,
            "convex_fill_ratio": 0.0,
            "component_count": 0.0,
            "largest_component_fraction": 0.0,
        }

    bbox = _mask_bbox(mask)
    assert bbox is not None
    y0, y1, x0, x1 = bbox
    bbox_height = y1 - y0
    bbox_width = x1 - x0
    bbox_area = max(1, bbox_height * bbox_width)
    bbox_axis_ratio = min(bbox_height, bbox_width) / max(1, max(bbox_height, bbox_width))
    labeled, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if component_count > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_fraction = float(sizes.max() / max(1, area))
    else:
        largest_fraction = 0.0
    try:
        convex_area = int(convex_hull_image(mask).sum())
    except Exception:
        convex_area = area
    return {
        "area_px": float(area),
        "fill_ratio": float(area / bbox_area),
        "bbox_axis_ratio": float(bbox_axis_ratio),
        "convex_fill_ratio": float(area / max(1, convex_area)),
        "component_count": float(component_count),
        "largest_component_fraction": largest_fraction,
    }


def _is_low_compactness_sliver(
    mask: np.ndarray,
    params: _VisibleCrownPostprocessParams,
    *,
    metrics: dict[str, float] | None = None,
    largest_component_fraction: float | None = None,
) -> tuple[bool, dict[str, float]]:
    if metrics is None:
        metrics = _mask_shape_metrics(mask)
    area = int(metrics["area_px"])
    if area <= 0 or area >= params.sliver_px:
        return False, metrics

    largest_fraction = (
        float(metrics["largest_component_fraction"])
        if largest_component_fraction is None
        else float(largest_component_fraction)
    )
    is_sliver = (
        metrics["component_count"] > 1.0
        or metrics["fill_ratio"] < params.sliver_min_bbox_fill_ratio
        or metrics["bbox_axis_ratio"] < params.sliver_min_bbox_axis_ratio
        or metrics["convex_fill_ratio"] < params.sliver_min_convex_fill_ratio
        or largest_fraction < params.sliver_min_largest_component_fraction
    )
    return bool(is_sliver), metrics


def _percentiles(values: list[float], percentile_values: tuple[int, ...]) -> dict[str, float]:
    if not values:
        return {f"p{percentile}": 0.0 for percentile in percentile_values}
    return {
        f"p{percentile}": float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        for percentile in percentile_values
    }


def _normalize_height_map(
    height_map: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if height_map is None:
        return None
    height = np.asarray(height_map)
    if height.ndim == 3:
        height = height[..., 0]
    if height.shape != shape:
        logger.warning(
            "Ignoring height map for crown postprocess because shape %s does not match mask shape %s.",
            height.shape,
            shape,
        )
        return None
    return height.astype(np.float32, copy=False)


def _height_stats_for_mask(
    height_map: np.ndarray | None,
    mask: np.ndarray,
    percentile: float,
) -> tuple[float, float, float]:
    if height_map is None or not mask.any():
        return 0.0, 0.0, 0.0
    values = np.asarray(height_map[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(values, percentile)),
        float(np.percentile(values, 50.0)),
        float(np.max(values)),
    )


def _normalize_rgb_brightness_map(
    rgb_image: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image)
    if rgb.ndim == 2:
        if rgb.shape != shape:
            logger.warning(
                "Ignoring RGB image for crown postprocess because shape %s does not match mask shape %s.",
                rgb.shape,
                shape,
            )
            return None
        luminance = rgb.astype(np.float32, copy=False)
    elif rgb.ndim == 3:
        if rgb.shape[:2] != shape or rgb.shape[2] < 3:
            logger.warning(
                "Ignoring RGB image for crown postprocess because shape %s does not match mask shape %s.",
                rgb.shape,
                shape,
            )
            return None
        rgb_float = rgb[..., :3].astype(np.float32, copy=False)
        if np.issubdtype(rgb.dtype, np.integer):
            rgb_float = rgb_float / float(np.iinfo(rgb.dtype).max)
        luminance = (
            0.2126 * rgb_float[..., 0] + 0.7152 * rgb_float[..., 1] + 0.0722 * rgb_float[..., 2]
        )
    else:
        logger.warning(
            "Ignoring RGB image for crown postprocess because shape %s is unsupported.",
            rgb.shape,
        )
        return None

    if np.issubdtype(rgb.dtype, np.integer) and rgb.ndim == 2:
        luminance = luminance / float(np.iinfo(rgb.dtype).max)
    elif not np.issubdtype(rgb.dtype, np.integer):
        finite_values = luminance[np.isfinite(luminance)]
        max_value = float(np.max(finite_values)) if finite_values.size else 0.0
        if max_value > 10.0:
            if max_value <= 255.0:
                luminance = luminance / 255.0
            elif max_value <= 65535.0:
                luminance = luminance / 65535.0
            else:
                luminance = luminance / max_value

    luminance = np.nan_to_num(
        luminance.astype(np.float32, copy=False),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(luminance, 0.0, 1.0).astype(np.float32, copy=False)


def _brightness_percentile_for_mask(
    brightness_map: np.ndarray | None,
    mask: np.ndarray,
    percentile: float,
) -> float:
    if brightness_map is None or not mask.any():
        return 0.0
    values = np.asarray(brightness_map[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _ordered_priority(
    height_priority_m: float,
    raw_area_px: int,
    params: _VisibleCrownPostprocessParams,
) -> float:
    crown_radius_proxy_m = math.sqrt(max(0.0, float(raw_area_px))) * params.gsd
    if height_priority_m > 0.0:
        return float(height_priority_m + params.ordered_area_priority_weight * crown_radius_proxy_m)
    return float(crown_radius_proxy_m)


def _ordered_empty_diagnostics(params: _VisibleCrownPostprocessParams) -> dict[str, object]:
    return {
        "raw_instance_count": 0,
        "cleaned_instance_count": 0,
        "dropped_instance_count": 0,
        "dropped_small_visible_count": 0,
        "dropped_small_final_count": 0,
        "dropped_dark_crown_count": 0,
        "dropped_low_compactness_count": 0,
        "raw_visible_conflict_px": 0,
        "raw_visible_reassigned_px": 0,
        "raw_visible_removed_px": 0,
        "raw_visible_total_px": 0,
        "background_added_px": 0,
        "tiny_component_removed_px": 0,
        "boundary_regularized_instance_count": 0,
        "boundary_multicomponent_instance_count": 0,
        "boundary_component_merge_count": 0,
        "boundary_largest_component_only_count": 0,
        "boundary_component_removed_px": 0,
        "boundary_reassigned_px": 0,
        "boundary_added_px": 0,
        "boundary_removed_px": 0,
        "nested_island_parent_count": 0,
        "nested_island_hole_count": 0,
        "nested_island_absorbed_component_count": 0,
        "nested_island_absorbed_px": 0,
        "nested_island_filled_background_px": 0,
        "nested_island_skipped_large_hole_count": 0,
        "nested_island_skipped_large_component_count": 0,
        "final_topology_parent_count": 0,
        "final_topology_hole_count": 0,
        "final_topology_multicomponent_count": 0,
        "final_topology_component_merge_count": 0,
        "final_topology_absorbed_component_count": 0,
        "final_topology_absorbed_px": 0,
        "final_topology_filled_background_px": 0,
        "final_topology_removed_component_count": 0,
        "final_topology_removed_px": 0,
        "final_topology_skipped_large_hole_count": 0,
        "final_topology_skipped_large_component_count": 0,
        "ordered_height_map_used": 0,
        "ordered_rgb_brightness_used": 0,
        "ordered_candidate_count": 0,
        "ordered_placed_count": 0,
        "ordered_absorbed_instance_count": 0,
        "ordered_absorbed_px": 0,
        "ordered_reserved_px": 0,
        "ordered_dropped_occluded_count": 0,
        "ordered_dropped_component_count": 0,
        "ordered_background_hole_filled_px": 0,
        "ordered_reclaimed_visible_px": 0,
        "ordered_rescued_instance_count": 0,
        "ordered_rescued_px": 0,
        "ordered_rescue_reassigned_px": 0,
        "ordered_rescued_ids": [],
        "rescued_low_compactness_removed_count": 0,
        "rescued_low_compactness_removed_px": 0,
        "rescued_low_compactness_repaired_px": 0,
        "rescued_low_compactness_unrepaired_px": 0,
        "final_min_area_removed_count": 0,
        "final_min_area_removed_px": 0,
        "ordered_priority_height_p50": 0.0,
        "ordered_priority_height_p90": 0.0,
        "area_growth_percentiles": _percentiles([], (50, 90, 95, 99)),
        "params": _params_diagnostics(params),
        "component_count_by_id": {},
        "final_seed_area_by_id": {},
        "final_candidate_area_by_id": {},
        "visible_brightness_by_id": {},
    }


def _params_diagnostics(params: _VisibleCrownPostprocessParams) -> dict[str, object]:
    return {
        "gsd": float(params.gsd),
        "min_visible_px": int(params.min_visible_px),
        "min_final_px": int(params.min_final_px),
        "min_visible_brightness": float(params.min_visible_brightness),
        "visible_brightness_percentile": float(params.visible_brightness_percentile),
        "min_component_px": int(params.min_component_px),
        "gap_fill_radius_px": int(params.gap_fill_radius_px),
        "sliver_px": int(params.sliver_px),
        "sliver_min_bbox_fill_ratio": float(params.sliver_min_bbox_fill_ratio),
        "sliver_min_bbox_axis_ratio": float(params.sliver_min_bbox_axis_ratio),
        "sliver_min_convex_fill_ratio": float(params.sliver_min_convex_fill_ratio),
        "sliver_min_largest_component_fraction": float(
            params.sliver_min_largest_component_fraction
        ),
        "boundary_regularization_min_px": int(params.boundary_regularization_min_px),
        "boundary_smooth_radius_px": int(params.boundary_smooth_radius_px),
        "boundary_core_erosion_px": int(params.boundary_core_erosion_px),
        "boundary_component_merge_radius_px": int(params.boundary_component_merge_radius_px),
        "boundary_component_min_px": int(params.boundary_component_min_px),
        "boundary_component_keep_fraction": float(params.boundary_component_keep_fraction),
        "boundary_component_merge_max_area_growth_fraction": float(
            params.boundary_component_merge_max_area_growth_fraction
        ),
        "boundary_max_area_growth_fraction": float(params.boundary_max_area_growth_fraction),
        "nested_island_parent_min_px": int(params.nested_island_parent_min_px),
        "nested_island_max_hole_area_fraction": float(params.nested_island_max_hole_area_fraction),
        "nested_island_absorb_max_area_fraction": float(
            params.nested_island_absorb_max_area_fraction
        ),
        "nested_island_absorb_max_px": int(params.nested_island_absorb_max_px),
        "final_topology_min_px": int(params.final_topology_min_px),
        "ordered_height_percentile": float(params.ordered_height_percentile),
        "ordered_area_priority_weight": float(params.ordered_area_priority_weight),
        "ordered_absorb_height_margin_m": float(params.ordered_absorb_height_margin_m),
        "ordered_absorb_overlap_fraction": float(params.ordered_absorb_overlap_fraction),
        "ordered_absorb_max_px": int(params.ordered_absorb_max_px),
        "ordered_min_retained_visible_fraction": float(
            params.ordered_min_retained_visible_fraction
        ),
        "ordered_component_keep_fraction": float(params.ordered_component_keep_fraction),
        "ordered_outer_envelope_min_px": int(params.ordered_outer_envelope_min_px),
        "ordered_outer_envelope_radius_px": int(params.ordered_outer_envelope_radius_px),
        "ordered_outer_envelope_max_growth_fraction": float(
            params.ordered_outer_envelope_max_growth_fraction
        ),
        "ordered_reclaim_height_margin_m": float(params.ordered_reclaim_height_margin_m),
        "ordered_rescue_min_visible_fraction": float(params.ordered_rescue_min_visible_fraction),
    }


def _select_nearest_additions(
    addable: np.ndarray,
    distances: np.ndarray,
    max_add_px: int,
) -> np.ndarray:
    add_count = int(addable.sum())
    if add_count <= max(0, int(max_add_px)):
        return addable
    if max_add_px <= 0:
        return np.zeros_like(addable, dtype=bool)

    add_indices = np.flatnonzero(addable.ravel())
    add_distances = distances.ravel()[add_indices]
    selected_order = np.argpartition(add_distances, max_add_px - 1)[:max_add_px]
    selected = np.zeros_like(addable, dtype=bool)
    selected.ravel()[add_indices[selected_order]] = True
    return selected


def _compute_raw_visible_reassignments(
    raw_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    allowed_reassignments: np.ndarray | None = None,
) -> tuple[int, int]:
    reassigned = (raw_mask > 0) & (cleaned_mask > 0) & (raw_mask != cleaned_mask)
    reassigned_px = int(np.count_nonzero(reassigned))
    if allowed_reassignments is None:
        return reassigned_px, reassigned_px
    conflict_px = int(np.count_nonzero(reassigned & ~allowed_reassignments))
    return reassigned_px, conflict_px


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    labeled, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count <= 1:
        return mask.astype(bool, copy=True)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _prepare_boundary_proposal(
    mask: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    diagnostics = {
        "boundary_component_removed_px": 0,
        "boundary_multicomponent_instance_count": 0,
        "boundary_component_merge_count": 0,
        "boundary_largest_component_only_count": 0,
        "boundary_regularized_instance_count": 0,
    }
    mask = mask.astype(bool, copy=False)
    area = int(mask.sum())
    if area <= 0:
        return mask.copy(), mask.copy(), diagnostics

    labeled, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count <= 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool), diagnostics

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest_area = int(sizes[largest_label])

    keep_threshold = max(
        int(params.boundary_component_min_px),
        int(round(area * params.boundary_component_keep_fraction)),
    )
    keep_labels = np.flatnonzero(sizes >= keep_threshold)
    if largest_label not in keep_labels:
        keep_labels = np.append(keep_labels, largest_label)
    selected = np.isin(labeled, keep_labels)
    selected_area = int(selected.sum())
    diagnostics["boundary_component_removed_px"] += max(0, area - selected_area)

    if component_count > 1:
        diagnostics["boundary_multicomponent_instance_count"] = 1
        merge_source = selected
        if params.boundary_component_merge_radius_px > 0:
            merge_source = _apply_local_bool_op(
                selected,
                params.boundary_component_merge_radius_px,
                "closing",
            )
        merged = binary_fill_holes(merge_source)
        max_merged_area = int(
            round(selected_area * (1.0 + params.boundary_component_merge_max_area_growth_fraction))
        )
        if int(merged.sum()) <= max(selected_area, max_merged_area):
            selected = merged
            selected_area = int(selected.sum())
            diagnostics["boundary_component_merge_count"] = 1
        else:
            selected = labeled == largest_label
            diagnostics["boundary_component_removed_px"] += max(
                0,
                selected_area - largest_area,
            )
            selected_area = largest_area
            diagnostics["boundary_largest_component_only_count"] = 1

    holes = binary_fill_holes(selected) & ~selected
    should_regularize = (
        selected_area >= params.boundary_regularization_min_px
        or component_count > 1
        or bool(holes.any())
    )

    proposal = selected
    if should_regularize:
        diagnostics["boundary_regularized_instance_count"] = 1
        if params.boundary_smooth_radius_px > 0:
            proposal = _apply_local_bool_op(
                proposal,
                params.boundary_smooth_radius_px,
                "closing",
            )
            proposal = binary_fill_holes(proposal)
            opening_radius = max(0, params.boundary_smooth_radius_px // 2)
            if opening_radius > 0:
                opened = _apply_local_bool_op(proposal, opening_radius, "opening")
                if int(opened.sum()) >= int(round(selected_area * 0.80)):
                    proposal = opened
            proposal = binary_fill_holes(proposal)
        else:
            proposal = binary_fill_holes(proposal)

        if int(proposal.sum()) < int(round(selected_area * 0.80)):
            proposal = binary_fill_holes(selected)

        additions = proposal & ~selected
        max_add_px = int(round(selected_area * params.boundary_max_area_growth_fraction))
        if int(additions.sum()) > max_add_px:
            distances = ndimage.distance_transform_edt(~selected)
            additions = _select_nearest_additions(additions, distances, max_add_px)
            proposal = selected | additions
            proposal = binary_fill_holes(proposal)

    core = selected
    if should_regularize and params.boundary_core_erosion_px > 0:
        eroded = ndimage.binary_erosion(
            selected,
            structure=_disk_footprint(params.boundary_core_erosion_px),
            border_value=0,
        )
        if eroded.any():
            core = eroded
    if not core.any():
        core = _largest_component_mask(selected)

    return proposal.astype(bool), core.astype(bool), diagnostics


def _regularize_shared_boundaries(
    cleaned: np.ndarray,
    raw_instance: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    diagnostics = {
        "boundary_regularized_instance_count": 0,
        "boundary_multicomponent_instance_count": 0,
        "boundary_component_merge_count": 0,
        "boundary_largest_component_only_count": 0,
        "boundary_component_removed_px": 0,
        "boundary_reassigned_px": 0,
        "boundary_added_px": 0,
        "boundary_removed_px": 0,
    }
    allowed_reassignments = np.zeros(cleaned.shape, dtype=bool)

    label_stats = _compute_label_region_stats(cleaned)
    ids = label_stats.ids
    if ids.size == 0:
        return cleaned, diagnostics, allowed_reassignments

    nearest_labels = np.zeros(cleaned.shape, dtype=np.int32)
    nearest_distances = np.full(cleaned.shape, np.inf, dtype=np.float32)
    max_pad = (
        params.boundary_smooth_radius_px
        + params.boundary_component_merge_radius_px
        + params.boundary_core_erosion_px
        + 3
    )

    for uid_raw, _area in sorted(
        label_stats.area_by_id.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        uid = int(uid_raw)
        bbox = label_stats.bbox_by_id.get(uid)
        if bbox is None:
            continue

        y0, y1, x0, x1 = _pad_bbox(bbox, cleaned.shape, max_pad)
        crop_mask = cleaned[y0:y1, x0:x1] == uid
        proposal, core, local_diag = _prepare_boundary_proposal(crop_mask, params)
        if not proposal.any() or not core.any():
            continue

        for key, value in local_diag.items():
            diagnostics[key] += int(value)

        distances = ndimage.distance_transform_edt(~core).astype(np.float32, copy=False)
        current_distances = nearest_distances[y0:y1, x0:x1]
        current_labels = nearest_labels[y0:y1, x0:x1]
        take = proposal & (
            (distances < current_distances)
            | ((distances == current_distances) & ((current_labels == 0) | (uid < current_labels)))
        )
        if np.any(take):
            current_distances[take] = distances[take]
            current_labels[take] = uid
            nearest_distances[y0:y1, x0:x1] = current_distances
            nearest_labels[y0:y1, x0:x1] = current_labels

    result = nearest_labels
    diagnostics["boundary_reassigned_px"] = int(
        np.count_nonzero((cleaned > 0) & (result > 0) & (cleaned != result))
    )
    diagnostics["boundary_added_px"] = int(np.count_nonzero((cleaned == 0) & (result > 0)))
    diagnostics["boundary_removed_px"] = int(np.count_nonzero((cleaned > 0) & (result == 0)))
    allowed_reassignments = (raw_instance > 0) & (result > 0) & (raw_instance != result)
    return result, diagnostics, allowed_reassignments


def _absorb_nested_islands(
    cleaned: np.ndarray,
    raw_instance: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    diagnostics = {
        "nested_island_parent_count": 0,
        "nested_island_hole_count": 0,
        "nested_island_absorbed_component_count": 0,
        "nested_island_absorbed_px": 0,
        "nested_island_filled_background_px": 0,
        "nested_island_skipped_large_hole_count": 0,
        "nested_island_skipped_large_component_count": 0,
    }
    allowed_reassignments = np.zeros(cleaned.shape, dtype=bool)

    result = cleaned.copy()
    label_stats = _compute_label_region_stats(result)
    areas_by_id = dict(label_stats.area_by_id)
    bboxes_by_id = label_stats.bbox_by_id
    parent_ids = [
        int(uid)
        for uid, area in sorted(
            areas_by_id.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if area >= params.nested_island_parent_min_px
    ]

    for parent_id in parent_ids:
        parent_area = int(areas_by_id.get(parent_id, 0))
        if parent_area < params.nested_island_parent_min_px:
            continue

        bbox = bboxes_by_id.get(parent_id)
        if bbox is None:
            continue

        y0, y1, x0, x1 = _pad_bbox(bbox, result.shape, 1)
        crop = result[y0:y1, x0:x1]
        parent_crop = crop == parent_id
        holes = binary_fill_holes(parent_crop) & ~parent_crop
        if not holes.any():
            continue

        hole_labels, hole_count = ndimage.label(
            holes,
            structure=np.ones((3, 3), dtype=bool),
        )
        parent_used = False
        max_hole_px = int(round(parent_area * params.nested_island_max_hole_area_fraction))
        max_component_px = int(
            min(
                params.nested_island_absorb_max_px,
                round(parent_area * params.nested_island_absorb_max_area_fraction),
            )
        )

        for hole_label in range(1, hole_count + 1):
            hole = hole_labels == hole_label
            hole_area = int(hole.sum())
            if hole_area <= 0:
                continue
            if hole_area > max_hole_px:
                diagnostics["nested_island_skipped_large_hole_count"] += 1
                continue

            hole_values = crop[hole]
            child_ids = [
                int(value)
                for value in np.unique(hole_values)
                if int(value) > 0 and int(value) != parent_id
            ]

            blocked = False
            absorbed_px = 0
            child_component_sizes: dict[int, int] = {}
            for child_id in child_ids:
                child_component_px = int(np.count_nonzero(hole & (crop == child_id)))
                if child_component_px > max_component_px:
                    diagnostics["nested_island_skipped_large_component_count"] += 1
                    blocked = True
                    break
                child_component_sizes[child_id] = child_component_px
                absorbed_px += child_component_px

            if blocked:
                continue

            background_px = int(np.count_nonzero(hole & (crop == 0)))
            raw_crop = raw_instance[y0:y1, x0:x1]
            reassigned = hole & (raw_crop > 0) & (raw_crop != parent_id)
            allowed_reassignments[y0:y1, x0:x1] |= reassigned
            crop[hole] = parent_id

            parent_area += hole_area
            areas_by_id[parent_id] = parent_area
            for child_id, child_component_px in child_component_sizes.items():
                areas_by_id[child_id] = max(
                    0, int(areas_by_id.get(child_id, 0)) - child_component_px
                )

            diagnostics["nested_island_hole_count"] += 1
            diagnostics["nested_island_absorbed_component_count"] += len(child_ids)
            diagnostics["nested_island_absorbed_px"] += absorbed_px
            diagnostics["nested_island_filled_background_px"] += background_px
            parent_used = True

        if parent_used:
            result[y0:y1, x0:x1] = crop
            diagnostics["nested_island_parent_count"] += 1

    return result, diagnostics, allowed_reassignments


def _cleanup_final_crown_topology(
    cleaned: np.ndarray,
    raw_instance: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    diagnostics = {
        "final_topology_parent_count": 0,
        "final_topology_hole_count": 0,
        "final_topology_multicomponent_count": 0,
        "final_topology_component_merge_count": 0,
        "final_topology_absorbed_component_count": 0,
        "final_topology_absorbed_px": 0,
        "final_topology_filled_background_px": 0,
        "final_topology_removed_component_count": 0,
        "final_topology_removed_px": 0,
        "final_topology_skipped_large_hole_count": 0,
        "final_topology_skipped_large_component_count": 0,
    }
    allowed_reassignments = np.zeros(cleaned.shape, dtype=bool)

    result = cleaned.copy()
    label_stats = _compute_label_region_stats(result)
    areas_by_id = dict(label_stats.area_by_id)
    bboxes_by_id = label_stats.bbox_by_id
    parent_ids = [
        int(uid)
        for uid, area in sorted(
            areas_by_id.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if area >= params.final_topology_min_px
    ]
    if not parent_ids:
        return result, diagnostics, allowed_reassignments

    max_pad = (
        max(
            params.boundary_smooth_radius_px,
            params.boundary_component_merge_radius_px,
            params.boundary_core_erosion_px,
            1,
        )
        + 2
    )
    merge_footprint = _disk_footprint(params.boundary_component_merge_radius_px)

    for parent_id in parent_ids:
        parent_area = int(areas_by_id.get(parent_id, 0))
        if parent_area < params.final_topology_min_px:
            continue

        bbox = bboxes_by_id.get(parent_id)
        if bbox is None:
            continue

        y0, y1, x0, x1 = _pad_bbox(bbox, result.shape, max_pad)
        crop = result[y0:y1, x0:x1]
        raw_crop = raw_instance[y0:y1, x0:x1]
        changed = False
        parent_used = False

        def absorb_region(region: np.ndarray) -> bool:
            nonlocal parent_area, changed, parent_used

            region = region.astype(bool, copy=False)
            region_area = int(region.sum())
            if region_area <= 0:
                return False

            max_component_px = int(
                min(
                    params.nested_island_absorb_max_px,
                    round(parent_area * params.nested_island_absorb_max_area_fraction),
                )
            )
            region_values = crop[region]
            child_ids = [
                int(value)
                for value in np.unique(region_values)
                if int(value) > 0 and int(value) != parent_id
            ]

            child_component_sizes: dict[int, int] = {}
            absorbed_px = 0
            for child_id in child_ids:
                child_component_px = int(np.count_nonzero(region & (crop == child_id)))
                if child_component_px > max_component_px:
                    diagnostics["final_topology_skipped_large_component_count"] += 1
                    return False
                child_component_sizes[child_id] = child_component_px
                absorbed_px += child_component_px

            reassigned = region & (raw_crop > 0) & (raw_crop != parent_id)
            allowed_reassignments[y0:y1, x0:x1] |= reassigned
            background_px = int(np.count_nonzero(region & (crop == 0)))
            crop[region] = parent_id

            parent_area += region_area
            areas_by_id[parent_id] = parent_area
            for child_id, child_component_px in child_component_sizes.items():
                areas_by_id[child_id] = max(
                    0,
                    int(areas_by_id.get(child_id, 0)) - child_component_px,
                )

            diagnostics["final_topology_absorbed_component_count"] += len(child_ids)
            diagnostics["final_topology_absorbed_px"] += absorbed_px
            diagnostics["final_topology_filled_background_px"] += background_px
            changed = True
            parent_used = True
            return True

        parent_crop = crop == parent_id
        holes = binary_fill_holes(parent_crop) & ~parent_crop
        if holes.any():
            hole_labels, hole_count = ndimage.label(
                holes,
                structure=np.ones((3, 3), dtype=bool),
            )
            max_hole_px = int(round(parent_area * params.nested_island_max_hole_area_fraction))
            for hole_label in range(1, hole_count + 1):
                hole = hole_labels == hole_label
                hole_area = int(hole.sum())
                if hole_area <= 0:
                    continue
                if hole_area > max_hole_px:
                    diagnostics["final_topology_skipped_large_hole_count"] += 1
                    continue
                if absorb_region(hole):
                    diagnostics["final_topology_hole_count"] += 1

        parent_crop = crop == parent_id
        labeled, component_count = ndimage.label(
            parent_crop,
            structure=np.ones((3, 3), dtype=bool),
        )
        if component_count > 1:
            diagnostics["final_topology_multicomponent_count"] += 1
            selected_area = int(parent_crop.sum())
            if merge_footprint is not None:
                merged = _apply_local_bool_op(
                    parent_crop,
                    params.boundary_component_merge_radius_px,
                    "closing",
                )
            else:
                merged = parent_crop.copy()
            merged = binary_fill_holes(merged)
            additions = merged & ~parent_crop
            max_add_px = int(
                round(selected_area * params.boundary_component_merge_max_area_growth_fraction)
            )
            if int(additions.sum()) <= max_add_px and absorb_region(additions):
                diagnostics["final_topology_component_merge_count"] += 1

            parent_crop = crop == parent_id
            labeled, component_count = ndimage.label(
                parent_crop,
                structure=np.ones((3, 3), dtype=bool),
            )
            if component_count > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                largest_label = int(np.argmax(sizes))
                remove_threshold = max(
                    int(params.boundary_component_min_px),
                    int(round(int(parent_crop.sum()) * params.boundary_component_keep_fraction)),
                )
                remove_labels = [
                    label
                    for label in range(1, component_count + 1)
                    if label != largest_label and int(sizes[label]) < remove_threshold
                ]
                if remove_labels:
                    remove_mask = np.isin(labeled, remove_labels)
                    removed_px = int(remove_mask.sum())
                    crop[remove_mask] = 0
                    parent_area = max(0, parent_area - removed_px)
                    areas_by_id[parent_id] = parent_area
                    diagnostics["final_topology_removed_component_count"] += len(remove_labels)
                    diagnostics["final_topology_removed_px"] += removed_px
                    changed = True
                    parent_used = True

                parent_crop = crop == parent_id
                if parent_crop.any():
                    holes = binary_fill_holes(parent_crop) & ~parent_crop
                    if holes.any():
                        hole_labels, hole_count = ndimage.label(
                            holes,
                            structure=np.ones((3, 3), dtype=bool),
                        )
                        max_hole_px = int(
                            round(parent_area * params.nested_island_max_hole_area_fraction)
                        )
                        for hole_label in range(1, hole_count + 1):
                            hole = hole_labels == hole_label
                            hole_area = int(hole.sum())
                            if hole_area <= 0:
                                continue
                            if hole_area > max_hole_px:
                                diagnostics["final_topology_skipped_large_hole_count"] += 1
                                continue
                            if absorb_region(hole):
                                diagnostics["final_topology_hole_count"] += 1

        if changed:
            result[y0:y1, x0:x1] = crop
        if parent_used:
            diagnostics["final_topology_parent_count"] += 1

    return result, diagnostics, allowed_reassignments


def _build_ordered_crown_stats(
    instance: np.ndarray,
    height_map: np.ndarray | None,
    params: _VisibleCrownPostprocessParams,
    label_stats: _LabelRegionStats | None = None,
) -> dict[int, _OrderedCrownStats]:
    stats: dict[int, _OrderedCrownStats] = {}
    if label_stats is None:
        label_stats = _compute_label_region_stats(instance)
    raw_ids = label_stats.ids
    for raw_uid in raw_ids:
        uid = int(raw_uid)
        raw_area = int(label_stats.area_by_id[uid])
        bbox = label_stats.bbox_by_id.get(uid)
        if bbox is None:
            continue
        y0, y1, x0, x1 = bbox
        crop = instance[y0:y1, x0:x1]
        mask = crop == uid
        if height_map is None:
            height_priority, height_p50, height_max = 0.0, 0.0, 0.0
        else:
            height_priority, height_p50, height_max = _height_stats_for_mask(
                height_map[y0:y1, x0:x1],
                mask,
                params.ordered_height_percentile,
            )
        priority = _ordered_priority(height_priority, raw_area, params)
        stats[uid] = _OrderedCrownStats(
            uid=uid,
            raw_area_px=raw_area,
            height_priority_m=height_priority,
            height_p50_m=height_p50,
            height_max_m=height_max,
            priority=priority,
        )
    return stats


def _ordered_outer_envelope(
    seed: np.ndarray,
    proposal: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> np.ndarray:
    seed_area = int(seed.sum())
    if seed_area < params.ordered_outer_envelope_min_px:
        return proposal

    max_area = int(round(seed_area * (1.0 + params.ordered_outer_envelope_max_growth_fraction)))
    envelope = proposal.astype(bool, copy=True)

    if params.ordered_outer_envelope_radius_px > 0:
        closed = _apply_local_bool_op(
            seed,
            params.ordered_outer_envelope_radius_px,
            "closing",
        )
        closed = binary_fill_holes(closed)
        if int(closed.sum()) <= max(seed_area, max_area):
            envelope |= closed

    try:
        hull = convex_hull_image(seed)
        if int(hull.sum()) <= max(seed_area, max_area):
            envelope |= hull
    except Exception:
        logger.debug("Ordered crown convex envelope failed.", exc_info=True)

    if int(envelope.sum()) > max(seed_area, max_area):
        distances = ndimage.distance_transform_edt(~seed)
        additions = envelope & ~seed
        additions = _select_nearest_additions(
            additions,
            distances,
            max(0, max_area - seed_area),
        )
        envelope = seed | additions

    return binary_fill_holes(envelope)


def _build_ordered_crown_candidates(
    instance: np.ndarray,
    height_map: np.ndarray | None,
    brightness_map: np.ndarray | None,
    params: _VisibleCrownPostprocessParams,
    stats_by_id: dict[int, _OrderedCrownStats],
    label_stats: _LabelRegionStats | None = None,
) -> tuple[
    list[_OrderedCrownCandidate],
    dict[str, int],
    dict[int, int],
    dict[int, int],
    dict[int, int],
    dict[int, int],
    dict[int, float],
]:
    diagnostics = {
        "dropped_small_visible_count": 0,
        "dropped_small_final_count": 0,
        "dropped_dark_crown_count": 0,
        "dropped_low_compactness_count": 0,
        "tiny_component_removed_px": 0,
        "boundary_regularized_instance_count": 0,
        "boundary_multicomponent_instance_count": 0,
        "boundary_component_merge_count": 0,
        "boundary_largest_component_only_count": 0,
        "boundary_component_removed_px": 0,
    }
    candidates: list[_OrderedCrownCandidate] = []
    component_count_by_id: dict[int, int] = {}
    final_seed_area_by_id: dict[int, int] = {}
    final_candidate_area_by_id: dict[int, int] = {}
    visible_brightness_by_id: dict[int, float] = {}
    raw_area_by_id: dict[int, int] = {}
    if label_stats is None:
        label_stats = _compute_label_region_stats(instance)
    raw_ids = label_stats.ids
    crop_pad = (
        max(
            params.gap_fill_radius_px,
            params.boundary_smooth_radius_px,
            params.boundary_component_merge_radius_px,
            params.ordered_outer_envelope_radius_px,
        )
        + 3
    )

    for raw_uid in raw_ids:
        uid = int(raw_uid)
        raw_area = int(label_stats.area_by_id[uid])
        raw_area_by_id[uid] = raw_area
        if raw_area < params.min_visible_px:
            diagnostics["dropped_small_visible_count"] += 1
            continue

        bbox = label_stats.bbox_by_id.get(uid)
        if bbox is None:
            diagnostics["dropped_small_visible_count"] += 1
            continue
        y0, y1, x0, x1 = _pad_bbox(bbox, instance.shape, crop_pad)
        raw_crop = instance[y0:y1, x0:x1]
        seed = raw_crop == uid
        seed, removed_px, component_count, largest_fraction = _filter_small_components(
            seed,
            params.min_component_px,
            keep_largest_if_all_small=True,
        )
        diagnostics["tiny_component_removed_px"] += removed_px
        component_count_by_id[uid] = component_count

        seed_area = int(seed.sum())
        if seed_area < params.min_visible_px:
            diagnostics["dropped_small_visible_count"] += 1
            continue

        brightness_crop = brightness_map[y0:y1, x0:x1] if brightness_map is not None else None
        visible_brightness = _brightness_percentile_for_mask(
            brightness_crop,
            seed,
            params.visible_brightness_percentile,
        )
        if (
            params.min_visible_brightness > 0.0
            and visible_brightness < params.min_visible_brightness
        ):
            diagnostics["dropped_dark_crown_count"] += 1
            continue
        visible_brightness_by_id[uid] = visible_brightness

        proposal, _core, local_diag = _prepare_boundary_proposal(seed, params)
        proposal |= seed
        proposal = binary_fill_holes(proposal)
        for key, value in local_diag.items():
            diagnostics[key] = diagnostics.get(key, 0) + int(value)

        proposal = _ordered_outer_envelope(seed, proposal, params)

        proposal, removed_after_regularize, _proposal_component_count, _largest = (
            _filter_small_components(
                proposal,
                params.min_component_px,
                keep_largest_if_all_small=True,
            )
        )
        diagnostics["tiny_component_removed_px"] += removed_after_regularize
        proposal |= seed
        proposal = binary_fill_holes(proposal)

        metrics = _mask_shape_metrics(proposal)
        proposal_area = int(metrics["area_px"])
        if proposal_area < params.min_final_px:
            diagnostics["dropped_small_final_count"] += 1
            continue
        is_sliver, _ = _is_low_compactness_sliver(
            proposal,
            params,
            metrics=metrics,
            largest_component_fraction=largest_fraction,
        )
        if is_sliver:
            diagnostics["dropped_low_compactness_count"] += 1
            continue

        stats = stats_by_id[uid]
        candidates.append(
            _OrderedCrownCandidate(
                uid=uid,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                seed=seed.astype(bool, copy=True),
                proposal=proposal.astype(bool, copy=True),
                raw_area_px=raw_area,
                seed_area_px=seed_area,
                proposal_area_px=proposal_area,
                height_priority_m=stats.height_priority_m,
                height_p50_m=stats.height_p50_m,
                height_max_m=stats.height_max_m,
                priority=stats.priority,
            )
        )
        final_seed_area_by_id[uid] = seed_area
        final_candidate_area_by_id[uid] = proposal_area

    return (
        candidates,
        diagnostics,
        raw_area_by_id,
        component_count_by_id,
        final_seed_area_by_id,
        final_candidate_area_by_id,
        visible_brightness_by_id,
    )


def _candidate_overlap_in_crop(
    candidate: _OrderedCrownCandidate,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray | None:
    iy0 = max(y0, candidate.y0)
    iy1 = min(y1, candidate.y1)
    ix0 = max(x0, candidate.x0)
    ix1 = min(x1, candidate.x1)
    if iy0 >= iy1 or ix0 >= ix1:
        return None
    result = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    result[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = candidate.proposal[
        iy0 - candidate.y0 : iy1 - candidate.y0,
        ix0 - candidate.x0 : ix1 - candidate.x0,
    ]
    return result


def _ordered_should_absorb(
    parent: _OrderedCrownCandidate,
    child_stats: _OrderedCrownStats,
    overlap_px: int,
    params: _VisibleCrownPostprocessParams,
    *,
    height_map_used: bool,
) -> bool:
    if overlap_px <= 0 or child_stats.raw_area_px <= 0:
        return False
    if child_stats.raw_area_px > params.ordered_absorb_max_px:
        return False
    overlap_fraction = overlap_px / max(1, child_stats.raw_area_px)
    if overlap_fraction < params.ordered_absorb_overlap_fraction:
        return False
    return (
        parent.height_priority_m
        >= child_stats.height_priority_m + params.ordered_absorb_height_margin_m
    )


def _ordered_can_reclaim_from_owner(
    candidate: _OrderedCrownCandidate,
    owner_id: int,
    stats_by_id: dict[int, _OrderedCrownStats],
    params: _VisibleCrownPostprocessParams,
    *,
    height_map_used: bool,
) -> bool:
    if owner_id <= 0 or owner_id == candidate.uid:
        return True
    owner_stats = stats_by_id.get(int(owner_id))
    if owner_stats is None:
        return False
    return (
        candidate.height_priority_m
        >= owner_stats.height_priority_m + params.ordered_reclaim_height_margin_m
    )


def _ordered_reclaim_mask(
    candidate: _OrderedCrownCandidate,
    owner_labels: np.ndarray,
    stats_by_id: dict[int, _OrderedCrownStats],
    params: _VisibleCrownPostprocessParams,
    *,
    height_map_used: bool,
) -> np.ndarray:
    reclaimable = owner_labels == 0
    owner_ids = np.unique(owner_labels[(owner_labels > 0) & (owner_labels != candidate.uid)])
    for owner_raw in owner_ids:
        owner_id = int(owner_raw)
        if _ordered_can_reclaim_from_owner(
            candidate,
            owner_id,
            stats_by_id,
            params,
            height_map_used=height_map_used,
        ):
            reclaimable |= owner_labels == owner_id
    return reclaimable


def _postprocess_ordered_component_crop(
    mask: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, int, int]:
    if not mask.any():
        return mask.astype(bool, copy=True), 0, 0
    labeled, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count <= 1:
        return mask.astype(bool, copy=True), 0, 0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    keep_threshold = max(
        int(params.boundary_component_min_px),
        int(round(int(mask.sum()) * params.ordered_component_keep_fraction)),
    )
    keep_labels = np.flatnonzero(sizes >= keep_threshold)
    if largest_label not in keep_labels:
        keep_labels = np.append(keep_labels, largest_label)
    kept = np.isin(labeled, keep_labels)
    removed_px = int(mask.sum() - kept.sum())
    removed_count = max(0, int(component_count - len(keep_labels)))
    return kept.astype(bool), removed_px, removed_count


def _merge_remaining_same_label_components(
    cleaned: np.ndarray,
    raw_instance: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int]]:
    diagnostics = {
        "final_topology_component_merge_count": 0,
        "final_topology_removed_component_count": 0,
        "final_topology_removed_px": 0,
    }
    result = cleaned.copy()
    footprint = _disk_footprint(params.boundary_component_merge_radius_px)
    label_stats = _compute_label_region_stats(result)
    for uid_raw in label_stats.ids:
        uid = int(uid_raw)
        area = int(label_stats.area_by_id.get(uid, 0))
        if area <= 0:
            continue
        bbox = label_stats.bbox_by_id.get(uid)
        if bbox is None:
            continue
        y0, y1, x0, x1 = _pad_bbox(
            bbox,
            result.shape,
            params.boundary_component_merge_radius_px + 2,
        )
        crop = result[y0:y1, x0:x1]
        raw_crop = raw_instance[y0:y1, x0:x1]
        local = crop == uid
        labeled, component_count = ndimage.label(
            local,
            structure=np.ones((3, 3), dtype=bool),
        )
        if component_count <= 1:
            continue

        if area < params.sliver_px:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest_label = int(np.argmax(sizes))
            remove_mask = local & (labeled != largest_label)
            removed_px = int(remove_mask.sum())
            if removed_px > 0:
                crop[remove_mask] = 0
                result[y0:y1, x0:x1] = crop
                diagnostics["final_topology_removed_component_count"] += component_count - 1
                diagnostics["final_topology_removed_px"] += removed_px
            continue

        if footprint is not None:
            merged = _apply_local_bool_op(
                local,
                params.boundary_component_merge_radius_px,
                "closing",
            )
        else:
            merged = local.copy()
        merged = binary_fill_holes(merged)
        additions = merged & ~local & (crop == 0) & (raw_crop == 0)
        max_add_px = int(round(area * params.boundary_component_merge_max_area_growth_fraction))
        if int(additions.sum()) > max_add_px:
            distances = ndimage.distance_transform_edt(~local)
            additions = _select_nearest_additions(additions, distances, max_add_px)
        if additions.any():
            crop[additions] = uid
            result[y0:y1, x0:x1] = crop
            diagnostics["final_topology_component_merge_count"] += 1

    return result, diagnostics


def _rescue_ordered_visible_crowns(
    cleaned: np.ndarray,
    raw_instance: np.ndarray,
    candidates: list[_OrderedCrownCandidate],
    stats_by_id: dict[int, _OrderedCrownStats],
    absorbed_ids: set[int],
    params: _VisibleCrownPostprocessParams,
    *,
    height_map_used: bool,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    diagnostics = {
        "ordered_rescued_instance_count": 0,
        "ordered_rescued_px": 0,
        "ordered_rescue_reassigned_px": 0,
    }
    allowed_reassignments = np.zeros(cleaned.shape, dtype=bool)

    result = cleaned.copy()
    rescued_ids: set[int] = set()
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.priority,
            candidate.height_priority_m,
            candidate.raw_area_px,
            -candidate.uid,
        ),
        reverse=True,
    )

    for candidate in ordered_candidates:
        y0, y1, x0, x1 = candidate.y0, candidate.y1, candidate.x0, candidate.x1
        crop = result[y0:y1, x0:x1]
        raw_crop = raw_instance[y0:y1, x0:x1]
        existing_seed_px = int(np.count_nonzero(candidate.seed & (crop == candidate.uid)))
        if existing_seed_px >= params.min_visible_px:
            continue

        was_absorbed = candidate.uid in absorbed_ids
        if was_absorbed:
            # Absorption means a taller/larger crown already claimed some raw
            # pixels from this candidate. Still allow truly unassigned visible
            # pixels to survive, but do not let the rescue pass steal occupied
            # pixels back from the absorbing crown.
            reclaimable = crop == 0
            rescue_seed = candidate.seed & (crop == 0)
        else:
            reclaimable = _ordered_reclaim_mask(
                candidate,
                crop,
                stats_by_id,
                params,
                height_map_used=height_map_used,
            )
            rescue_seed = candidate.seed & (crop != candidate.uid)
        rescue_seed_px = int(rescue_seed.sum())
        rescue_fraction = rescue_seed_px / max(1, candidate.seed_area_px)
        if (
            rescue_seed_px < params.min_final_px
            and rescue_fraction < params.ordered_rescue_min_visible_fraction
        ):
            continue

        claim = candidate.proposal & reclaimable
        claim |= rescue_seed
        if not claim.any():
            continue

        holes = binary_fill_holes(claim) & ~claim
        fillable_holes = holes & ((crop == 0) | reclaimable) & (raw_crop == 0)
        claim |= fillable_holes
        claim, _removed_px, _removed_count = _postprocess_ordered_component_crop(
            claim,
            params,
        )
        claim &= reclaimable | rescue_seed
        if int(claim.sum()) < params.min_final_px:
            continue

        reassigned = claim & (raw_crop > 0) & (raw_crop != candidate.uid)
        allowed_reassignments[y0:y1, x0:x1] |= reassigned
        diagnostics["ordered_rescue_reassigned_px"] += int(np.count_nonzero(reassigned))
        crop[claim] = candidate.uid
        result[y0:y1, x0:x1] = crop
        rescued_ids.add(candidate.uid)
        diagnostics["ordered_rescued_px"] += int(np.count_nonzero(claim))

    diagnostics["ordered_rescued_instance_count"] = len(rescued_ids)
    diagnostics["ordered_rescued_ids"] = sorted(int(uid) for uid in rescued_ids)
    return result, diagnostics, allowed_reassignments


def _filter_rescued_low_compactness_slivers(
    cleaned: np.ndarray,
    rescued_ids: set[int],
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int]]:
    diagnostics = {
        "rescued_low_compactness_removed_count": 0,
        "rescued_low_compactness_removed_px": 0,
        "rescued_low_compactness_repaired_px": 0,
        "rescued_low_compactness_unrepaired_px": 0,
    }
    if not rescued_ids:
        return cleaned, diagnostics

    result = cleaned.copy()
    for uid in sorted(int(value) for value in rescued_ids):
        mask = result == uid
        is_sliver, metrics = _is_low_compactness_sliver(mask, params)
        if not is_sliver:
            continue
        removed_px = int(metrics["area_px"])
        if removed_px <= 0:
            continue
        result[mask] = 0
        repaired_px = _repair_removed_rescue_pixels(result, mask, params)
        diagnostics["rescued_low_compactness_removed_count"] += 1
        diagnostics["rescued_low_compactness_removed_px"] += removed_px
        diagnostics["rescued_low_compactness_repaired_px"] += repaired_px
        diagnostics["rescued_low_compactness_unrepaired_px"] += max(
            0,
            removed_px - repaired_px,
        )
    return result, diagnostics


def _repair_removed_rescue_pixels(
    cleaned: np.ndarray,
    removed_mask: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> int:
    if not removed_mask.any() or not np.any(cleaned > 0):
        return 0

    accepted = cleaned > 0
    distances, nearest_indices = ndimage.distance_transform_edt(
        ~accepted,
        return_indices=True,
    )
    nearest_labels = cleaned[nearest_indices[0], nearest_indices[1]]
    max_repair_distance_px = max(1, int(params.gap_fill_radius_px))
    repairable = removed_mask & (nearest_labels > 0) & (distances <= max_repair_distance_px)
    if not repairable.any():
        return 0

    cleaned[repairable] = nearest_labels[repairable]
    return int(np.count_nonzero(repairable))


def _filter_final_min_area(
    cleaned: np.ndarray,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, int]]:
    diagnostics = {
        "final_min_area_removed_count": 0,
        "final_min_area_removed_px": 0,
    }
    label_stats = _compute_label_region_stats(cleaned)
    remove_ids = [
        int(uid) for uid, area in label_stats.area_by_id.items() if int(area) < params.min_final_px
    ]
    if not remove_ids:
        return cleaned, diagnostics

    result = cleaned.copy()
    for uid in remove_ids:
        area_px = int(label_stats.area_by_id.get(uid, 0))
        if area_px <= 0:
            continue
        result[result == uid] = 0
        diagnostics["final_min_area_removed_count"] += 1
        diagnostics["final_min_area_removed_px"] += area_px
    return result, diagnostics


def _clean_instance_mask_ordered(
    instance: np.ndarray,
    *,
    height_map: np.ndarray | None,
    rgb_image: np.ndarray | None,
    params: _VisibleCrownPostprocessParams,
) -> tuple[np.ndarray, dict[str, object]]:
    raw_nonzero = instance > 0
    label_stats = _compute_label_region_stats(instance)
    raw_ids = label_stats.ids
    if raw_ids.size == 0:
        empty = np.zeros_like(instance, dtype=np.uint16)
        return empty, _ordered_empty_diagnostics(params)

    height_map = _normalize_height_map(height_map, instance.shape)
    if height_map is None:
        raise ValueError("Ordered crown mask postprocessing requires a valid height map.")
    height_map_used = True
    brightness_map = _normalize_rgb_brightness_map(rgb_image, instance.shape)
    if brightness_map is None:
        raise ValueError("Ordered crown mask postprocessing requires a valid rendered RGB image.")
    rgb_brightness_used = True
    stats_by_id = _build_ordered_crown_stats(instance, height_map, params, label_stats)
    (
        candidates,
        candidate_diagnostics,
        raw_area_by_id,
        component_count_by_id,
        final_seed_area_by_id,
        final_candidate_area_by_id,
        visible_brightness_by_id,
    ) = _build_ordered_crown_candidates(
        instance,
        height_map,
        brightness_map,
        params,
        stats_by_id,
        label_stats,
    )
    candidate_by_id = {candidate.uid: candidate for candidate in candidates}
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.priority,
            candidate.height_priority_m,
            candidate.raw_area_px,
            -candidate.uid,
        ),
        reverse=True,
    )

    result = np.zeros(instance.shape, dtype=np.int32)
    allowed_reassignments = np.zeros(instance.shape, dtype=bool)
    absorbed_ids: set[int] = set()
    absorbed_px = 0
    reserved_px = 0
    placed_count = 0
    dropped_occluded = 0
    ordered_component_removed_px = 0
    ordered_component_removed_count = 0
    background_hole_filled_px = 0
    reclaimed_visible_px = 0

    for candidate in ordered_candidates:
        y0, y1, x0, x1 = candidate.y0, candidate.y1, candidate.x0, candidate.x1
        crop_result = result[y0:y1, x0:x1]
        raw_crop = instance[y0:y1, x0:x1]
        proposal = candidate.proposal
        own_or_background = proposal & ((raw_crop == 0) | (raw_crop == candidate.uid))
        claim = own_or_background.copy()
        block = np.zeros_like(proposal, dtype=bool)

        other_raw = proposal & (raw_crop > 0) & (raw_crop != candidate.uid)
        other_ids = [
            int(value)
            for value in np.unique(raw_crop[other_raw])
            if int(value) > 0 and int(value) != candidate.uid
        ]
        for other_id in other_ids:
            other_region = other_raw & (raw_crop == other_id)
            other_stats = stats_by_id.get(other_id)
            if other_stats is None:
                continue
            overlap_px = int(other_region.sum())
            if _ordered_should_absorb(
                candidate,
                other_stats,
                overlap_px,
                params,
                height_map_used=height_map_used,
            ):
                claim |= other_region
                absorbed_ids.add(other_id)
                absorbed_px += overlap_px
                continue

            other_candidate = candidate_by_id.get(other_id)
            if other_candidate is None:
                block |= other_region
                continue
            other_proposal = _candidate_overlap_in_crop(
                other_candidate,
                y0,
                y1,
                x0,
                x1,
            )
            if other_proposal is not None:
                block |= other_proposal & proposal & (raw_crop != candidate.uid)
            else:
                block |= other_region

        reserved_px += int(np.count_nonzero(claim & block))
        claim &= ~block
        if candidate.uid in absorbed_ids:
            reclaimed_seed = np.zeros_like(candidate.seed, dtype=bool)
        else:
            reclaimed_seed = candidate.seed & (crop_result > 0) & (crop_result != candidate.uid)
        reclaimed_visible_px += int(np.count_nonzero(reclaimed_seed))
        claim = (claim & (crop_result == 0)) | reclaimed_seed

        retained_seed = candidate.seed & (crop_result == 0) & claim
        retained_seed |= reclaimed_seed
        retained_seed_px = int(retained_seed.sum())
        retained_fraction = retained_seed_px / max(1, candidate.seed_area_px)
        if (
            retained_seed_px < params.min_visible_px
            and retained_fraction < params.ordered_min_retained_visible_fraction
        ):
            dropped_occluded += 1
            continue

        holes = binary_fill_holes(claim) & ~claim
        fillable_holes = holes & (crop_result == 0) & (raw_crop == 0)
        if fillable_holes.any():
            claim |= fillable_holes
            background_hole_filled_px += int(fillable_holes.sum())

        claim, removed_px, removed_count = _postprocess_ordered_component_crop(
            claim,
            params,
        )
        claim |= reclaimed_seed
        ordered_component_removed_px += removed_px
        ordered_component_removed_count += removed_count
        claim = (claim & (crop_result == 0)) | reclaimed_seed

        final_area = int(claim.sum())
        if final_area < params.min_final_px:
            dropped_occluded += 1
            continue

        reassigned = claim & (raw_crop > 0) & (raw_crop != candidate.uid)
        allowed_reassignments[y0:y1, x0:x1] |= reassigned
        crop_result[claim] = candidate.uid
        result[y0:y1, x0:x1] = crop_result
        placed_count += 1

    coverage = result > 0
    enclosed_background = binary_fill_holes(coverage) & ~coverage & (instance == 0)
    if enclosed_background.any() and coverage.any():
        distances, nearest_indices = ndimage.distance_transform_edt(
            ~coverage,
            return_indices=True,
        )
        fillable = enclosed_background & (distances <= max(1, params.gap_fill_radius_px))
        if fillable.any():
            nearest_labels = result[nearest_indices[0], nearest_indices[1]]
            result[fillable] = nearest_labels[fillable]
            background_hole_filled_px += int(np.count_nonzero(fillable))

    result, boundary_diagnostics, boundary_reassignments = _regularize_shared_boundaries(
        result,
        instance,
        params,
    )
    allowed_reassignments |= boundary_reassignments
    result, nested_island_diagnostics, nested_island_reassignments = _absorb_nested_islands(
        result,
        instance,
        params,
    )
    allowed_reassignments |= nested_island_reassignments
    result, final_topology_diagnostics, final_topology_reassignments = (
        _cleanup_final_crown_topology(
            result,
            instance,
            params,
        )
    )
    allowed_reassignments |= final_topology_reassignments
    result, final_component_diagnostics = _merge_remaining_same_label_components(
        result,
        instance,
        params,
    )
    final_topology_diagnostics["final_topology_component_merge_count"] += (
        final_component_diagnostics["final_topology_component_merge_count"]
    )
    final_topology_diagnostics["final_topology_removed_component_count"] += (
        final_component_diagnostics["final_topology_removed_component_count"]
    )
    final_topology_diagnostics["final_topology_removed_px"] += final_component_diagnostics[
        "final_topology_removed_px"
    ]
    result, rescue_diagnostics, rescue_reassignments = _rescue_ordered_visible_crowns(
        result,
        instance,
        ordered_candidates,
        stats_by_id,
        absorbed_ids,
        params,
        height_map_used=height_map_used,
    )
    allowed_reassignments |= rescue_reassignments
    rescued_ids_raw = rescue_diagnostics.get("ordered_rescued_ids", [])
    rescued_ids = {int(value) for value in rescued_ids_raw if isinstance(value, (int, np.integer))}
    result, rescued_sliver_diagnostics = _filter_rescued_low_compactness_slivers(
        result,
        rescued_ids,
        params,
    )
    result, final_min_area_diagnostics = _filter_final_min_area(result, params)

    cleaned_ids = np.unique(result[result > 0])
    raw_visible_reassigned_px, raw_visible_conflict_px = _compute_raw_visible_reassignments(
        instance,
        result,
        allowed_reassignments,
    )
    raw_visible_removed_px = int(np.count_nonzero((instance > 0) & (result == 0)))
    background_added_px = int(np.count_nonzero((instance == 0) & (result > 0)))

    area_growth_values = []
    for uid in cleaned_ids:
        uid_int = int(uid)
        raw_area = raw_area_by_id.get(uid_int, 0)
        if raw_area <= 0:
            continue
        cleaned_area = int(np.count_nonzero(result == uid_int))
        area_growth_values.append((cleaned_area - raw_area) / raw_area)

    priority_heights = [candidate.height_priority_m for candidate in candidates]
    diagnostics: dict[str, object] = {
        **_ordered_empty_diagnostics(params),
        "raw_instance_count": int(raw_ids.size),
        "cleaned_instance_count": int(cleaned_ids.size),
        "dropped_instance_count": int(raw_ids.size - cleaned_ids.size),
        "dropped_small_visible_count": int(candidate_diagnostics["dropped_small_visible_count"]),
        "dropped_small_final_count": int(
            candidate_diagnostics["dropped_small_final_count"]
            + final_min_area_diagnostics["final_min_area_removed_count"]
        ),
        "dropped_dark_crown_count": int(candidate_diagnostics["dropped_dark_crown_count"]),
        "dropped_low_compactness_count": int(
            candidate_diagnostics["dropped_low_compactness_count"]
            + rescued_sliver_diagnostics["rescued_low_compactness_removed_count"]
        ),
        "raw_visible_conflict_px": int(raw_visible_conflict_px),
        "raw_visible_reassigned_px": int(raw_visible_reassigned_px),
        "raw_visible_removed_px": int(raw_visible_removed_px),
        "raw_visible_total_px": int(raw_nonzero.sum()),
        "background_added_px": int(background_added_px),
        "tiny_component_removed_px": int(candidate_diagnostics["tiny_component_removed_px"]),
        "boundary_regularized_instance_count": int(
            candidate_diagnostics["boundary_regularized_instance_count"]
            + boundary_diagnostics["boundary_regularized_instance_count"]
        ),
        "boundary_multicomponent_instance_count": int(
            candidate_diagnostics["boundary_multicomponent_instance_count"]
            + boundary_diagnostics["boundary_multicomponent_instance_count"]
        ),
        "boundary_component_merge_count": int(
            candidate_diagnostics["boundary_component_merge_count"]
            + boundary_diagnostics["boundary_component_merge_count"]
        ),
        "boundary_largest_component_only_count": int(
            candidate_diagnostics["boundary_largest_component_only_count"]
            + boundary_diagnostics["boundary_largest_component_only_count"]
        ),
        "boundary_component_removed_px": int(
            candidate_diagnostics["boundary_component_removed_px"]
            + boundary_diagnostics["boundary_component_removed_px"]
        ),
        "boundary_reassigned_px": int(boundary_diagnostics["boundary_reassigned_px"]),
        "boundary_added_px": int(boundary_diagnostics["boundary_added_px"]),
        "boundary_removed_px": int(boundary_diagnostics["boundary_removed_px"]),
        **nested_island_diagnostics,
        **final_topology_diagnostics,
        "ordered_height_map_used": int(height_map_used),
        "ordered_rgb_brightness_used": int(rgb_brightness_used),
        "ordered_candidate_count": int(len(candidates)),
        "ordered_placed_count": int(placed_count),
        "ordered_absorbed_instance_count": int(len(absorbed_ids)),
        "ordered_absorbed_px": int(absorbed_px),
        "ordered_reserved_px": int(reserved_px),
        "ordered_dropped_occluded_count": int(dropped_occluded),
        "ordered_dropped_component_count": int(ordered_component_removed_count),
        "ordered_component_removed_px": int(ordered_component_removed_px),
        "ordered_background_hole_filled_px": int(background_hole_filled_px),
        "ordered_reclaimed_visible_px": int(reclaimed_visible_px),
        **rescue_diagnostics,
        **rescued_sliver_diagnostics,
        **final_min_area_diagnostics,
        "ordered_priority_height_p50": float(
            np.percentile(priority_heights, 50) if priority_heights else 0.0
        ),
        "ordered_priority_height_p90": float(
            np.percentile(priority_heights, 90) if priority_heights else 0.0
        ),
        "area_growth_percentiles": _percentiles(
            area_growth_values,
            (50, 90, 95, 99),
        ),
        "params": _params_diagnostics(params),
        "component_count_by_id": {
            int(key): int(value) for key, value in component_count_by_id.items()
        },
        "final_seed_area_by_id": {
            int(key): int(value) for key, value in final_seed_area_by_id.items()
        },
        "final_candidate_area_by_id": {
            int(key): int(value) for key, value in final_candidate_area_by_id.items()
        },
        "visible_brightness_by_id": {
            int(key): float(value) for key, value in visible_brightness_by_id.items()
        },
    }
    logger.info(
        "Ordered visible-crown postprocess | raw_instances=%d candidates=%d placed=%d "
        "dropped=%d dark_dropped=%d height_map=%s rgb_brightness=%s "
        "conflicts=%d reassigned_px=%d bg_added_px=%d "
        "raw_removed_px=%d absorbed_instances=%d",
        diagnostics["raw_instance_count"],
        diagnostics["ordered_candidate_count"],
        diagnostics["ordered_placed_count"],
        diagnostics["dropped_instance_count"],
        diagnostics["dropped_dark_crown_count"],
        bool(height_map_used),
        bool(rgb_brightness_used),
        diagnostics["raw_visible_conflict_px"],
        diagnostics["raw_visible_reassigned_px"],
        diagnostics["background_added_px"],
        diagnostics["raw_visible_removed_px"],
        diagnostics["ordered_absorbed_instance_count"],
    )
    return np.clip(result, 0, 65535).astype(np.uint16), diagnostics


def clean_instance_mask(
    instance_mask: np.ndarray,
    *,
    gsd: float = 1.0,
    height_map: np.ndarray | None = None,
    rgb_image: np.ndarray | None = None,
    mask_postprocess_config: object | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    params = _compute_visible_crown_postprocess_params(gsd, mask_postprocess_config)
    logger.info(
        "Ordered visible-crown mask postprocess | gsd=%.3f m/px | "
        "min_visible_px=%d min_final_px=%d min_component_px=%d "
        "brightness_p%.0f>=%.3f gap_fill_r=%d sliver_px=%d",
        params.gsd,
        params.min_visible_px,
        params.min_final_px,
        params.min_component_px,
        params.visible_brightness_percentile,
        params.min_visible_brightness,
        params.gap_fill_radius_px,
        params.sliver_px,
    )

    if height_map is None:
        raise ValueError("Ordered crown mask postprocessing requires a height map.")
    if rgb_image is None:
        raise ValueError("Ordered crown mask postprocessing requires a rendered RGB image.")

    instance = instance_mask.astype(np.int32, copy=False)
    cleaned_ordered, ordered_diagnostics = _clean_instance_mask_ordered(
        instance,
        height_map=height_map,
        rgb_image=rgb_image,
        params=params,
    )
    return (cleaned_ordered, ordered_diagnostics) if return_diagnostics else cleaned_ordered


def _postprocess_skip_diagnostics(
    mask: np.ndarray,
    params: _VisibleCrownPostprocessParams,
    reason: str,
) -> dict[str, object]:
    return {
        "postprocess_skipped": 1,
        "skip_reason": reason,
        "raw_instance_count": int(np.unique(mask[mask > 0]).size),
        "cleaned_instance_count": 0,
        "dropped_instance_count": 0,
        "dropped_dark_crown_count": 0,
        "raw_visible_conflict_px": 0,
        "params": _params_diagnostics(params),
    }


def save_smoothed_instance_mask(
    mask_path: Path,
    output_path: Path,
    gsd: float = 1.0,
    height_map_path: Path | None = None,
    rgb_image_path: Path | None = None,
    mask_postprocess_config: object | None = None,
) -> dict[str, object]:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    params = _compute_visible_crown_postprocess_params(gsd, mask_postprocess_config)
    height_map = None
    if height_map_path is not None and height_map_path.exists():
        height_map = np.array(Image.open(height_map_path))
        if height_map.ndim == 3:
            height_map = height_map[..., 0]
    rgb_image = None
    if rgb_image_path is not None and rgb_image_path.exists():
        rgb_image = np.array(Image.open(rgb_image_path))
    if height_map is None:
        logger.error(
            "Ordered crown mask postprocessing requires a height map; skipping "
            "smoothed instance mask generation for %s.",
            mask_path,
        )
        return _postprocess_skip_diagnostics(mask, params, "missing_height_map")
    brightness_map = _normalize_rgb_brightness_map(rgb_image, mask.shape)
    if brightness_map is None:
        logger.error(
            "Ordered crown mask postprocessing requires a rendered RGB image; "
            "skipping smoothed instance mask generation for %s.",
            mask_path,
        )
        return _postprocess_skip_diagnostics(mask, params, "missing_rgb")
    repaired_mask, diagnostics = clean_instance_mask(
        mask,
        gsd=gsd,
        height_map=height_map,
        rgb_image=brightness_map,
        mask_postprocess_config=mask_postprocess_config,
        return_diagnostics=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(repaired_mask).save(output_path)
    return diagnostics


def _load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        return mask[..., 0]
    return mask


def _decode_id_map(mask: np.ndarray, id_max_value: int) -> np.ndarray:
    id_max = max(1, int(id_max_value))

    if np.issubdtype(mask.dtype, np.floating):
        max_val = float(np.nanmax(mask)) if mask.size > 0 else 0.0
        if max_val <= 1.0:
            return np.rint(mask * id_max).astype(np.int32)
        return np.rint(mask).astype(np.int32)

    mask_int = mask.astype(np.int64)
    if id_max == 65535:
        return mask_int.astype(np.int32)

    max_val = int(mask_int.max()) if mask_int.size > 0 else 0
    if max_val > id_max:
        return np.rint(mask_int.astype(np.float64) * id_max / 65535.0).astype(np.int32)
    return mask_int.astype(np.int32)


def compute_mask_statistics(
    instance_mask_path: Path,
    semantic_id_mask_path: Path,
    variant_id_mask_path: Path,
    semantic_id_max_value: int,
    variant_id_max_value: int,
) -> dict[str, object]:
    instance_mask = _load_mask(instance_mask_path)
    semantic_raw = _load_mask(semantic_id_mask_path)
    variant_raw = _load_mask(variant_id_mask_path)

    semantic_mask = _decode_id_map(semantic_raw, semantic_id_max_value)
    variant_mask = _decode_id_map(variant_raw, variant_id_max_value)

    if not (instance_mask.shape == semantic_mask.shape == variant_mask.shape):
        raise ValueError(
            "Mask shapes must match for summary statistics. "
            f"instance={instance_mask.shape}, semantic={semantic_mask.shape}, variant={variant_mask.shape}"
        )

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]

    per_semantic_tree_count: dict[int, int] = {}
    per_variant_tree_count: dict[int, int] = {}

    for instance_id in instance_ids:
        instance_binary = instance_mask == instance_id

        semantic_values = semantic_mask[instance_binary]
        semantic_values = semantic_values[semantic_values > 0]
        if semantic_values.size > 0:
            semantic_counts = np.bincount(semantic_values.astype(np.int64))
            semantic_id = int(semantic_counts.argmax())
            per_semantic_tree_count[semantic_id] = per_semantic_tree_count.get(semantic_id, 0) + 1

        variant_values = variant_mask[instance_binary]
        variant_values = variant_values[variant_values > 0]
        if variant_values.size > 0:
            variant_counts = np.bincount(variant_values.astype(np.int64))
            variant_id = int(variant_counts.argmax())
            per_variant_tree_count[variant_id] = per_variant_tree_count.get(variant_id, 0) + 1

    variant_ids_present = np.unique(variant_mask)
    variant_ids_present = variant_ids_present[variant_ids_present > 0]

    semantic_ids_present = np.unique(semantic_mask)
    semantic_ids_present = semantic_ids_present[semantic_ids_present > 0]

    return {
        "tree_count": int(len(instance_ids)),
        "per_semantic_tree_count": per_semantic_tree_count,
        "per_variant_tree_count": per_variant_tree_count,
        "semantic_ids_present": [int(value) for value in semantic_ids_present.tolist()],
        "variant_ids_present": [int(value) for value in variant_ids_present.tolist()],
    }


def save_class_id_map_from_variant_id_map(
    variant_id_map_path: Path,
    output_path: Path,
    variant_id_to_class_id: dict[int, int],
    variant_id_max_value: int,
    class_id_max_value: int,
) -> None:
    variant_map_raw = _load_mask(variant_id_map_path)
    variant_ids = _decode_id_map(variant_map_raw, variant_id_max_value)

    class_ids = np.zeros_like(variant_ids, dtype=np.int32)
    for variant_id, class_id in variant_id_to_class_id.items():
        class_ids[variant_ids == int(variant_id)] = int(class_id)

    class_max = max(1, int(class_id_max_value))
    class_ids = np.clip(class_ids, 0, class_max)
    class_map_uint16 = class_ids.astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_map_uint16).save(output_path)


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]


def _bbox_to_coco(bbox: tuple[int, int, int, int]) -> list[float]:
    y0, y1, x0, x1 = bbox
    return [
        float(x0),
        float(y0),
        float(x1 - x0),
        float(y1 - y0),
    ]


def _positive_values_for_label(
    label_map: np.ndarray,
    value_map: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if bbox is None:
        return np.asarray([], dtype=value_map.dtype)

    y0, y1, x0, x1 = bbox
    label_crop = label_map[y0:y1, x0:x1]
    values = value_map[y0:y1, x0:x1][label_crop == int(label_id)]
    return values[values > 0]


def _mask_to_polygons(
    mask: np.ndarray,
    contour_tolerance: float,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[list[float]]:
    polygons: list[list[float]] = []

    padded = np.pad(mask.astype(bool), pad_width=1, mode="constant", constant_values=False)
    contours = find_contours(padded.astype(float), level=0.5)
    for contour in contours:
        points = contour
        if x_offset or y_offset:
            # Preserve the full-frame Douglas-Peucker vertex choices while only
            # tracing the cropped mask.
            points = contour.copy()
            points[:, 0] += float(y_offset)
            points[:, 1] += float(x_offset)
        if contour_tolerance > 0.0:
            points = approximate_polygon(points, tolerance=contour_tolerance)

        if points.shape[0] < 3:
            continue
        if np.allclose(points[0], points[-1]):
            points = points[:-1]
        if points.shape[0] < 3:
            continue

        polygon: list[float] = []
        for point in points:
            y_coord = float(point[0] - 1.0)
            x_coord = float(point[1] - 1.0)
            polygon.extend([x_coord, y_coord])

        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def export_instance_segmentation_coco(
    instance_mask_raw_path: Path,
    instance_mask_polygon_path: Path,
    class_id_mask_path: Path,
    output_json_path: Path,
    image_file_name: str,
    category_id_to_name: dict[int, str],
    class_id_max_value: int,
    contour_tolerance: float = 2.5,
    min_area_px: int = 1,
) -> None:
    instance_mask_raw = _load_mask(instance_mask_raw_path)
    instance_mask_polygon = _load_mask(instance_mask_polygon_path)
    class_id_mask_raw = _load_mask(class_id_mask_path)
    class_id_mask = _decode_id_map(class_id_mask_raw, class_id_max_value)

    if not (instance_mask_raw.shape == instance_mask_polygon.shape == class_id_mask.shape):
        raise ValueError(
            "Raw instance, polygon instance, and semantic ID mask shapes must match. "
            f"Got raw={instance_mask_raw.shape}, polygon={instance_mask_polygon.shape}, "
            f"semantic={class_id_mask.shape}."
        )

    image_height, image_width = instance_mask_polygon.shape
    categories = [
        {"id": int(category_id), "name": name, "supercategory": "tree"}
        for category_id, name in sorted(category_id_to_name.items())
    ]

    annotations: list[dict[str, object]] = []
    annotation_id = 1
    raw_stats = _compute_label_region_stats(instance_mask_raw)
    polygon_stats = _compute_label_region_stats(instance_mask_polygon)
    unique_instance_ids = polygon_stats.ids
    min_area_px = max(1, int(min_area_px))

    for instance_id in unique_instance_ids:
        instance_id_int = int(instance_id)
        polygon_bbox = polygon_stats.bbox_by_id.get(instance_id_int)
        if polygon_bbox is None:
            continue

        y0, y1, x0, x1 = polygon_bbox
        instance_binary_polygon = instance_mask_polygon[y0:y1, x0:x1] == instance_id_int
        if not np.any(instance_binary_polygon):
            continue

        area_px = int(polygon_stats.area_by_id.get(instance_id_int, 0))
        if area_px < min_area_px:
            continue

        class_values = _positive_values_for_label(
            instance_mask_raw,
            class_id_mask,
            instance_id_int,
            raw_stats.bbox_by_id.get(instance_id_int),
        )
        if class_values.size == 0:
            class_values = _positive_values_for_label(
                instance_mask_polygon,
                class_id_mask,
                instance_id_int,
                polygon_bbox,
            )
        if class_values.size == 0:
            continue

        counts = np.bincount(class_values.astype(np.int64))
        category_id = int(counts.argmax())
        purity = counts[category_id] / counts.sum()

        if purity < 0.9:
            logger.warning(
                f"[warn] Mixed semantic assignment for instance {instance_id}: "
                f"semantic={category_id}, purity={purity:.3f}"
            )

        polygons = _mask_to_polygons(
            instance_binary_polygon,
            contour_tolerance=contour_tolerance,
            x_offset=x0,
            y_offset=y0,
        )
        if not polygons:
            continue

        annotation = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": int(category_id),
            "segmentation": polygons,
            "area": float(area_px),
            "bbox": _bbox_to_coco(polygon_bbox),
            "iscrowd": 0,
        }
        annotations.append(annotation)
        annotation_id += 1

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_file_name,
                "width": int(image_width),
                "height": int(image_height),
            }
        ],
        "annotations": annotations,
        "categories": categories,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
