from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any

import bpy
from mathutils import Matrix, Vector

from .botaniq_leaf_geometry import (
    capture_leaf_unit_geometry,
    restore_leaf_unit_geometry_to_centroids,
)

logger = logging.getLogger(__name__)


_DEFAULT_BRANCH_GROUP_PREFIX = "bq_branch_group_"
_TRUNK_GROUP = "bq_trunk"
_DISTANCE_TO_TRUNK_GROUP = "bq_distance_to_trunk"
_LEAVES_GRADIENT_GROUP = "bq_leaves_gradient"
BOTANIQ_BRANCH_RECIPE_SCHEMA = "synthforest.botaniq_branch_crown.v1"


@dataclass
class _BranchProfile:
    name: str
    weights: list[float]
    indices: list[int]
    pivot: Vector
    center: Vector
    direction_xy: Vector
    max_distance_from_pivot: float


@dataclass
class _BranchTransform:
    profile: _BranchProfile
    radial_scale: float
    tangent_scale: float
    z_scale: float
    rotation_rad: float
    tilt_rad: float
    translation_xy: Vector
    z_translation_m: float
    sector_swap_translation_xy: Vector
    sector_swap_target_group: str
    sector_swap_target_pivot: Vector
    sector_swap_target_center: Vector
    sector_swap_target_direction_xy: Vector
    sector_swap_strength: float
    influence_scale: float
    variation_role: str
    max_displacement_m: float


@dataclass
class _BranchMaskAttribute:
    transform: _BranchTransform
    name: str


def _vector_to_list(vector: Vector, dimensions: int = 3) -> list[float]:
    values = [float(vector[index]) for index in range(min(dimensions, len(vector)))]
    while len(values) < dimensions:
        values.append(0.0)
    return values


def _vector_from_recipe(value: object, dimensions: int = 3) -> Vector:
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value[:dimensions]]
    else:
        values = []
    while len(values) < dimensions:
        values.append(0.0)
    return Vector(tuple(values))


def _branch_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "branch"


def _cfg_bool(cfg: object, name: str, fallback: bool) -> bool:
    try:
        return bool(getattr(cfg, name, fallback))
    except Exception:
        return bool(fallback)


def _cfg_float(cfg: object, name: str, fallback: float) -> float:
    try:
        return float(getattr(cfg, name, fallback))
    except Exception:
        return float(fallback)


def _cfg_int(cfg: object, name: str, fallback: int) -> int:
    try:
        return int(getattr(cfg, name, fallback))
    except Exception:
        return int(fallback)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _limit_vector_length(vector: Vector, max_length: float) -> Vector:
    max_length = max(0.0, float(max_length))
    if max_length <= 0.0 or vector.length <= max_length:
        return vector.copy()
    if vector.length <= 1e-8:
        return vector.copy()
    return vector.normalized() * max_length


def _tag_variation_role(role: str, tag: str) -> str:
    tags = [part for part in str(role or "regular").split("+") if part]
    if tag not in tags:
        tags.append(tag)
    return "+".join(tags) or tag


def _sorted_branch_group_names(
    tree_obj: bpy.types.Object,
    prefix: str,
) -> list[str]:
    def sort_key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", name)
        if match:
            return int(match.group(1)), name
        return 10_000, name

    return sorted(
        [group.name for group in tree_obj.vertex_groups if group.name.startswith(prefix)],
        key=sort_key,
    )


def _read_vertex_group_weights(
    tree_obj: bpy.types.Object,
    group_name: str,
) -> tuple[list[float], bool]:
    vertex_count = len(tree_obj.data.vertices)
    weights = [0.0] * vertex_count
    group = tree_obj.vertex_groups.get(group_name)
    if group is None:
        return weights, False

    group_index = int(group.index)
    for vertex in tree_obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group_index:
                weights[vertex.index] = _clamp01(float(assignment.weight))
                break
    return weights, True


def _weighted_average_coords(
    coords: list[Vector],
    indices: list[int],
    weights: list[float],
) -> Vector:
    if not indices:
        return Vector((0.0, 0.0, 0.0))

    total = 0.0
    accum = Vector((0.0, 0.0, 0.0))
    for index in indices:
        weight = max(0.0, float(weights[index]))
        if weight <= 0.0:
            continue
        accum += coords[index] * weight
        total += weight

    if total > 1e-8:
        return accum / total

    accum = Vector((0.0, 0.0, 0.0))
    for index in indices:
        accum += coords[index]
    return accum / max(1, len(indices))


def _bounds_center_and_radius(coords: list[Vector]) -> tuple[Vector, float]:
    if not coords:
        return Vector((0.0, 0.0, 0.0)), 0.0

    min_x = min(coord.x for coord in coords)
    max_x = max(coord.x for coord in coords)
    min_y = min(coord.y for coord in coords)
    max_y = max(coord.y for coord in coords)
    min_z = min(coord.z for coord in coords)
    max_z = max(coord.z for coord in coords)
    center = Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5))
    radius = max(max_x - min_x, max_y - min_y) * 0.5
    return center, max(0.0, float(radius))


def _estimate_trunk_center(
    coords: list[Vector],
    trunk_weights: list[float],
    has_trunk_weights: bool,
) -> Vector:
    if has_trunk_weights:
        trunk_indices = [index for index, weight in enumerate(trunk_weights) if weight > 0.01]
        if trunk_indices:
            return _weighted_average_coords(coords, trunk_indices, trunk_weights)

    center, _radius = _bounds_center_and_radius(coords)
    return center


def _safe_direction_xy(vector: Vector, fallback_angle: float) -> Vector:
    direction = Vector((float(vector.x), float(vector.y)))
    if direction.length > 1e-8:
        return direction.normalized()
    return Vector((math.cos(fallback_angle), math.sin(fallback_angle)))


def _estimate_branch_pivot(
    coords: list[Vector],
    indices: list[int],
    weights: list[float],
    distance_weights: list[float],
    has_distance_weights: bool,
) -> Vector:
    if not indices:
        return Vector((0.0, 0.0, 0.0))

    sample_count = max(8, int(len(indices) * 0.12))
    if has_distance_weights:
        pivot_indices = sorted(indices, key=lambda index: distance_weights[index])[:sample_count]
        return _weighted_average_coords(coords, pivot_indices, weights)

    min_z = min(coords[index].z for index in indices)
    max_z = max(coords[index].z for index in indices)
    z_limit = min_z + (max_z - min_z) * 0.18
    pivot_indices = [index for index in indices if coords[index].z <= z_limit]
    if not pivot_indices:
        pivot_indices = sorted(indices, key=lambda index: coords[index].z)[:sample_count]
    return _weighted_average_coords(coords, pivot_indices, weights)


def _combined_branch_weights(
    raw_weights: list[float],
    trunk_weights: list[float],
    distance_weights: list[float],
    leaves_weights: list[float],
    has_trunk_weights: bool,
    has_distance_weights: bool,
    has_leaves_weights: bool,
    tip_power: float,
    base_floor: float,
) -> list[float]:
    combined: list[float] = []
    tip_power = max(0.05, float(tip_power))
    base_floor = _clamp01(base_floor)

    for index, raw_weight in enumerate(raw_weights):
        branch_weight = _clamp01(raw_weight)
        if branch_weight <= 0.0:
            combined.append(0.0)
            continue

        if has_trunk_weights:
            branch_weight *= (1.0 - _clamp01(trunk_weights[index])) ** 2

        tip_sources: list[float] = []
        if has_distance_weights:
            tip_sources.append(_clamp01(distance_weights[index]))
        if has_leaves_weights:
            tip_sources.append(_clamp01(leaves_weights[index]))
        if tip_sources:
            tip_weight = max(tip_sources) ** tip_power
            branch_weight *= base_floor + (1.0 - base_floor) * tip_weight

        combined.append(_clamp01(branch_weight))
    return combined


def _build_branch_profiles(
    tree_obj: bpy.types.Object,
    coords: list[Vector],
    branch_group_names: list[str],
    trunk_center: Vector,
    trunk_weights: list[float],
    distance_weights: list[float],
    leaves_weights: list[float],
    has_trunk_weights: bool,
    has_distance_weights: bool,
    has_leaves_weights: bool,
    rng: random.Random,
    cfg: object,
) -> list[_BranchProfile]:
    threshold = max(0.0, _cfg_float(cfg, "botaniq_branch_weight_threshold", 0.02))
    min_vertices = max(1, _cfg_int(cfg, "botaniq_branch_min_vertices", 128))
    tip_power = _cfg_float(cfg, "botaniq_branch_tip_influence_power", 0.85)
    base_floor = _cfg_float(cfg, "botaniq_branch_base_influence_floor", 0.12)

    profiles: list[_BranchProfile] = []
    for group_name in branch_group_names:
        raw_weights, _has_weights = _read_vertex_group_weights(tree_obj, group_name)
        weights = _combined_branch_weights(
            raw_weights=raw_weights,
            trunk_weights=trunk_weights,
            distance_weights=distance_weights,
            leaves_weights=leaves_weights,
            has_trunk_weights=has_trunk_weights,
            has_distance_weights=has_distance_weights,
            has_leaves_weights=has_leaves_weights,
            tip_power=tip_power,
            base_floor=base_floor,
        )
        indices = [index for index, weight in enumerate(weights) if weight >= threshold]
        if len(indices) < min_vertices:
            continue

        center = _weighted_average_coords(coords, indices, weights)
        pivot = _estimate_branch_pivot(
            coords,
            indices,
            weights,
            distance_weights,
            has_distance_weights,
        )
        direction_xy = _safe_direction_xy(
            center - trunk_center,
            fallback_angle=rng.uniform(0.0, math.tau),
        )
        max_distance = max((coords[index] - pivot).length for index in indices)
        profiles.append(
            _BranchProfile(
                name=group_name,
                weights=weights,
                indices=indices,
                pivot=pivot,
                center=center,
                direction_xy=direction_xy,
                max_distance_from_pivot=max(1e-6, float(max_distance)),
            )
        )

    return profiles


def _coerce_range(
    low: float,
    high: float,
    *,
    floor: float,
) -> tuple[float, float]:
    low = max(floor, float(low))
    high = max(floor, float(high))
    if high < low:
        low, high = high, low
    return low, high


def _sample_branch_transforms(
    profiles: list[_BranchProfile],
    crown_radius: float,
    rng: random.Random,
    cfg: object,
) -> list[_BranchTransform]:
    radial_min, radial_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_radial_scale_min", 0.82),
        _cfg_float(cfg, "botaniq_branch_radial_scale_max", 1.20),
        floor=0.05,
    )
    tangent_min, tangent_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_tangential_scale_min", 0.88),
        _cfg_float(cfg, "botaniq_branch_tangential_scale_max", 1.12),
        floor=0.05,
    )
    z_min, z_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_z_scale_min", 0.94),
        _cfg_float(cfg, "botaniq_branch_z_scale_max", 1.06),
        floor=0.05,
    )
    max_rotation = math.radians(max(0.0, _cfg_float(cfg, "botaniq_branch_rotation_max_deg", 18.0)))
    max_tilt = math.radians(max(0.0, _cfg_float(cfg, "botaniq_branch_tilt_max_deg", 5.0)))
    max_lateral = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_lateral_jitter_max_fraction", 0.035),
    ) * max(0.0, crown_radius)
    influence_min, influence_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_influence_scale_min", 0.85),
        _cfg_float(cfg, "botaniq_branch_influence_scale_max", 1.20),
        floor=0.0,
    )
    outlier_probability = _clamp01(_cfg_float(cfg, "botaniq_branch_outlier_probability", 0.30))
    retract_probability = _clamp01(
        _cfg_float(cfg, "botaniq_branch_outlier_retract_probability", 0.55)
    )
    retract_min, retract_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_retract_scale_min", 0.55),
        _cfg_float(cfg, "botaniq_branch_retract_scale_max", 0.82),
        floor=0.05,
    )
    extend_min, extend_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_extend_scale_min", 1.22),
        _cfg_float(cfg, "botaniq_branch_extend_scale_max", 1.48),
        floor=0.05,
    )
    outlier_rotation_multiplier = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_outlier_rotation_multiplier", 1.35),
    )
    outlier_lateral_multiplier = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_outlier_lateral_multiplier", 1.55),
    )
    outlier_vertical_multiplier = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_outlier_vertical_multiplier", 1.45),
    )
    max_vertical = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_vertical_jitter_max_fraction", 0.0),
    ) * max(0.0, crown_radius)
    max_vertical = min(
        max_vertical,
        max(0.0, _cfg_float(cfg, "botaniq_branch_vertical_jitter_max_m", 0.45)),
    )
    prune_probability = _clamp01(_cfg_float(cfg, "botaniq_branch_prune_probability", 0.0))
    prune_radial_min, prune_radial_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_prune_radial_scale_min", 0.60),
        _cfg_float(cfg, "botaniq_branch_prune_radial_scale_max", 0.78),
        floor=0.05,
    )
    prune_tangent_min, prune_tangent_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_prune_tangential_scale_min", 0.72),
        _cfg_float(cfg, "botaniq_branch_prune_tangential_scale_max", 0.92),
        floor=0.05,
    )
    prune_pullback_min, prune_pullback_max = _coerce_range(
        _cfg_float(cfg, "botaniq_branch_prune_pullback_min_fraction", 0.45),
        _cfg_float(cfg, "botaniq_branch_prune_pullback_max_fraction", 0.90),
        floor=0.0,
    )
    prune_lateral_multiplier = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_prune_lateral_multiplier", 0.65),
    )
    max_displacement_fraction = max(
        0.01,
        _cfg_float(cfg, "botaniq_branch_max_displacement_fraction", 0.22),
    )
    max_crown_displacement_fraction = max(
        0.01,
        _cfg_float(cfg, "botaniq_branch_max_crown_displacement_fraction", 0.35),
    )
    max_displacement_m = max(
        0.05,
        _cfg_float(cfg, "botaniq_branch_max_displacement_m", 1.25),
    )

    transforms: list[_BranchTransform] = []
    for profile in profiles:
        radial_axis = profile.direction_xy
        tangent_axis = Vector((-radial_axis.y, radial_axis.x))
        radial_scale = rng.uniform(radial_min, radial_max)
        tangent_scale = rng.uniform(tangent_min, tangent_max)
        rotation_rad = rng.uniform(-max_rotation, max_rotation)
        tilt_rad = rng.uniform(-max_tilt, max_tilt)
        influence_scale = rng.uniform(influence_min, influence_max)
        lateral_multiplier = 1.0
        z_translation_m = rng.uniform(-max_vertical, max_vertical)
        variation_role = "regular"

        if rng.random() < prune_probability:
            lateral_multiplier = prune_lateral_multiplier
            radial_scale = rng.uniform(prune_radial_min, prune_radial_max)
            tangent_scale = rng.uniform(prune_tangent_min, prune_tangent_max)
            influence_scale = max(influence_scale, rng.uniform(1.0, max(1.0, influence_max)))
            tilt_rad *= 0.65
            z_translation_m *= 0.6
            variation_role = "pruned"
        elif rng.random() < outlier_probability:
            lateral_multiplier = outlier_lateral_multiplier
            rotation_rad *= outlier_rotation_multiplier
            z_translation_m *= outlier_vertical_multiplier
            influence_scale = max(influence_scale, rng.uniform(1.0, max(1.0, influence_max)))
            if rng.random() < retract_probability:
                radial_scale = rng.uniform(retract_min, retract_max)
                tangent_scale = min(tangent_scale, rng.uniform(tangent_min, 1.0))
                variation_role = "retracted"
            else:
                radial_scale = rng.uniform(extend_min, extend_max)
                tangent_scale = max(tangent_scale, rng.uniform(1.0, tangent_max))
                variation_role = "extended"

        radial_offset = rng.uniform(-max_lateral, max_lateral) * lateral_multiplier
        if variation_role == "retracted":
            radial_offset -= max_lateral * rng.uniform(0.2, 0.9)
        elif variation_role == "extended":
            radial_offset += max_lateral * rng.uniform(0.2, 0.9)
        elif variation_role == "pruned":
            radial_offset -= max_lateral * rng.uniform(prune_pullback_min, prune_pullback_max)
        tangent_offset = rng.uniform(-max_lateral, max_lateral) * 0.65 * lateral_multiplier
        translation_xy = radial_axis * radial_offset + tangent_axis * tangent_offset
        displacement_limit = min(
            max_displacement_m,
            profile.max_distance_from_pivot * max_displacement_fraction,
            max(0.05, crown_radius * max_crown_displacement_fraction),
        )

        transforms.append(
            _BranchTransform(
                profile=profile,
                radial_scale=radial_scale,
                tangent_scale=tangent_scale,
                z_scale=rng.uniform(z_min, z_max),
                rotation_rad=rotation_rad,
                tilt_rad=tilt_rad,
                translation_xy=translation_xy,
                z_translation_m=z_translation_m,
                sector_swap_translation_xy=Vector((0.0, 0.0)),
                sector_swap_target_group="",
                sector_swap_target_pivot=profile.pivot.copy(),
                sector_swap_target_center=profile.center.copy(),
                sector_swap_target_direction_xy=profile.direction_xy.copy(),
                sector_swap_strength=0.0,
                influence_scale=influence_scale,
                variation_role=variation_role,
                max_displacement_m=max(0.05, float(displacement_limit)),
            )
        )

    return transforms


def _angle_delta(source: float, target: float) -> float:
    return (target - source + math.pi) % math.tau - math.pi


def _apply_branch_sector_swaps(
    transforms: list[_BranchTransform],
    crown_radius: float,
    rng: random.Random,
    cfg: object,
) -> list[tuple[str, str]]:
    if len(transforms) < 2:
        return []

    probability = _clamp01(_cfg_float(cfg, "botaniq_branch_swap_probability", 0.20))
    strength = _clamp01(_cfg_float(cfg, "botaniq_branch_swap_strength", 0.35))
    frame_strength = _clamp01(_cfg_float(cfg, "botaniq_branch_swap_frame_strength", 0.0))
    translation_strength = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_swap_translation_strength", 0.0),
    )
    translation_max = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_swap_translation_max_fraction", 0.0),
    ) * max(0.0, crown_radius)
    z_strength = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_swap_z_strength", 0.0),
    )
    z_max = max(
        0.0,
        _cfg_float(cfg, "botaniq_branch_swap_z_max_m", 0.0),
    )
    scale_exchange_strength = _clamp01(
        _cfg_float(cfg, "botaniq_branch_swap_scale_exchange_strength", 0.0)
    )
    max_pairs = max(0, _cfg_int(cfg, "botaniq_branch_swap_max_pairs", 2))
    if probability <= 0.0 or max_pairs <= 0:
        return []
    if (
        strength <= 0.0
        and frame_strength <= 0.0
        and translation_strength <= 0.0
        and z_strength <= 0.0
        and scale_exchange_strength <= 0.0
    ):
        return []

    shuffled = list(transforms)
    rng.shuffle(shuffled)
    pairs: list[tuple[str, str]] = []
    while len(shuffled) >= 2 and len(pairs) < max_pairs:
        first = shuffled.pop()
        second = shuffled.pop()
        if rng.random() > probability:
            continue

        first_angle = math.atan2(first.profile.direction_xy.y, first.profile.direction_xy.x)
        second_angle = math.atan2(second.profile.direction_xy.y, second.profile.direction_xy.x)
        residual_rotation_strength = strength * max(0.0, 1.0 - frame_strength)
        first.rotation_rad += _angle_delta(first_angle, second_angle) * residual_rotation_strength
        second.rotation_rad += _angle_delta(second_angle, first_angle) * residual_rotation_strength

        if frame_strength > 0.0:
            first.sector_swap_target_group = second.profile.name
            first.sector_swap_target_pivot = second.profile.pivot.copy()
            first.sector_swap_target_center = second.profile.center.copy()
            first.sector_swap_target_direction_xy = second.profile.direction_xy.copy()
            first.sector_swap_strength = max(first.sector_swap_strength, frame_strength)

            second.sector_swap_target_group = first.profile.name
            second.sector_swap_target_pivot = first.profile.pivot.copy()
            second.sector_swap_target_center = first.profile.center.copy()
            second.sector_swap_target_direction_xy = first.profile.direction_xy.copy()
            second.sector_swap_strength = max(second.sector_swap_strength, frame_strength)

        if translation_strength > 0.0 and translation_max > 0.0:
            first_to_second = Vector(
                (
                    second.profile.center.x - first.profile.center.x,
                    second.profile.center.y - first.profile.center.y,
                )
            )
            first_move = _limit_vector_length(
                first_to_second * translation_strength,
                translation_max,
            )
            second_move = -first_move
            first.translation_xy += first_move
            second.translation_xy += second_move
            first.sector_swap_translation_xy += first_move
            second.sector_swap_translation_xy += second_move
            if not first.sector_swap_target_group:
                first.sector_swap_target_group = second.profile.name
            if not second.sector_swap_target_group:
                second.sector_swap_target_group = first.profile.name

        if z_strength > 0.0 and z_max > 0.0:
            z_delta = max(
                -z_max,
                min(
                    z_max,
                    (second.profile.center.z - first.profile.center.z) * z_strength,
                ),
            )
            first.z_translation_m += z_delta
            second.z_translation_m -= z_delta

        if scale_exchange_strength > 0.0:
            first_radial = first.radial_scale
            first_tangent = first.tangent_scale
            first_z = first.z_scale
            first.radial_scale += (
                second.radial_scale - first.radial_scale
            ) * scale_exchange_strength
            first.tangent_scale += (
                second.tangent_scale - first.tangent_scale
            ) * scale_exchange_strength
            first.z_scale += (second.z_scale - first.z_scale) * scale_exchange_strength
            second.radial_scale += (first_radial - second.radial_scale) * scale_exchange_strength
            second.tangent_scale += (first_tangent - second.tangent_scale) * scale_exchange_strength
            second.z_scale += (first_z - second.z_scale) * scale_exchange_strength

        first.variation_role = _tag_variation_role(first.variation_role, "swapped")
        second.variation_role = _tag_variation_role(second.variation_role, "swapped")
        pairs.append((first.profile.name, second.profile.name))

    return pairs


def _branch_transform_to_recipe(transform: _BranchTransform) -> dict[str, Any]:
    profile = transform.profile
    return {
        "group_name": profile.name,
        "vertex_count": int(len(profile.indices)),
        "pivot": _vector_to_list(profile.pivot),
        "center": _vector_to_list(profile.center),
        "direction_xy": _vector_to_list(profile.direction_xy, 2),
        "max_distance_from_pivot_m": float(profile.max_distance_from_pivot),
        "radial_scale": float(transform.radial_scale),
        "tangential_scale": float(transform.tangent_scale),
        "z_scale": float(transform.z_scale),
        "rotation_rad": float(transform.rotation_rad),
        "tilt_rad": float(transform.tilt_rad),
        "translation_xy": _vector_to_list(transform.translation_xy, 2),
        "z_translation_m": float(transform.z_translation_m),
        "sector_swap_translation_xy": _vector_to_list(
            transform.sector_swap_translation_xy,
            2,
        ),
        "sector_swap_target_group": str(transform.sector_swap_target_group),
        "sector_swap_target_pivot": _vector_to_list(transform.sector_swap_target_pivot),
        "sector_swap_target_center": _vector_to_list(transform.sector_swap_target_center),
        "sector_swap_target_direction_xy": _vector_to_list(
            transform.sector_swap_target_direction_xy,
            2,
        ),
        "sector_swap_strength": float(transform.sector_swap_strength),
        "influence_scale": float(transform.influence_scale),
        "variation_role": str(transform.variation_role),
        "max_displacement_m": float(transform.max_displacement_m),
    }


def _branch_transform_from_recipe(
    tree_obj: bpy.types.Object,
    branch_recipe: dict[str, Any],
    cfg: object,
) -> _BranchTransform | None:
    group_name = str(branch_recipe.get("group_name", "")).strip()
    if not group_name:
        return None

    raw_weights, has_weights = _read_vertex_group_weights(tree_obj, group_name)
    if not has_weights:
        return None

    trunk_weights, has_trunk_weights = _read_vertex_group_weights(tree_obj, _TRUNK_GROUP)
    distance_weights, has_distance_weights = _read_vertex_group_weights(
        tree_obj,
        _DISTANCE_TO_TRUNK_GROUP,
    )
    leaves_weights, has_leaves_weights = _read_vertex_group_weights(
        tree_obj,
        _LEAVES_GRADIENT_GROUP,
    )
    weights = _combined_branch_weights(
        raw_weights=raw_weights,
        trunk_weights=trunk_weights,
        distance_weights=distance_weights,
        leaves_weights=leaves_weights,
        has_trunk_weights=has_trunk_weights,
        has_distance_weights=has_distance_weights,
        has_leaves_weights=has_leaves_weights,
        tip_power=_cfg_float(cfg, "botaniq_branch_tip_influence_power", 0.85),
        base_floor=_cfg_float(cfg, "botaniq_branch_base_influence_floor", 0.12),
    )
    threshold = max(0.0, _cfg_float(cfg, "botaniq_branch_weight_threshold", 0.02))
    indices = [index for index, weight in enumerate(weights) if weight >= threshold]
    if not indices:
        return None

    direction_xy = _vector_from_recipe(branch_recipe.get("direction_xy"), 2)
    if direction_xy.length <= 1e-8:
        direction_xy = Vector((1.0, 0.0))
    else:
        direction_xy.normalize()

    profile = _BranchProfile(
        name=group_name,
        weights=weights,
        indices=indices,
        pivot=_vector_from_recipe(branch_recipe.get("pivot")),
        center=_vector_from_recipe(branch_recipe.get("center")),
        direction_xy=direction_xy,
        max_distance_from_pivot=max(
            1e-6,
            float(branch_recipe.get("max_distance_from_pivot_m", 1.0)),
        ),
    )
    return _BranchTransform(
        profile=profile,
        radial_scale=float(branch_recipe.get("radial_scale", 1.0)),
        tangent_scale=float(branch_recipe.get("tangential_scale", 1.0)),
        z_scale=float(branch_recipe.get("z_scale", 1.0)),
        rotation_rad=float(branch_recipe.get("rotation_rad", 0.0)),
        tilt_rad=float(branch_recipe.get("tilt_rad", 0.0)),
        translation_xy=_vector_from_recipe(branch_recipe.get("translation_xy"), 2),
        z_translation_m=float(branch_recipe.get("z_translation_m", 0.0)),
        sector_swap_translation_xy=_vector_from_recipe(
            branch_recipe.get("sector_swap_translation_xy"),
            2,
        ),
        sector_swap_target_group=str(branch_recipe.get("sector_swap_target_group", "")),
        sector_swap_target_pivot=_vector_from_recipe(branch_recipe.get("sector_swap_target_pivot")),
        sector_swap_target_center=_vector_from_recipe(
            branch_recipe.get("sector_swap_target_center")
        ),
        sector_swap_target_direction_xy=_vector_from_recipe(
            branch_recipe.get("sector_swap_target_direction_xy"),
            2,
        ),
        sector_swap_strength=float(branch_recipe.get("sector_swap_strength", 0.0)),
        influence_scale=float(branch_recipe.get("influence_scale", 1.0)),
        variation_role=str(branch_recipe.get("variation_role", "regular")),
        max_displacement_m=max(
            0.05,
            float(branch_recipe.get("max_displacement_m", 1.25)),
        ),
    )


def build_botaniq_branch_group_crown_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: object,
) -> dict[str, Any]:
    """Plan deterministic branch deformation from editable botaniq vertex groups.

    The recipe is intentionally JSON-serializable and does not store per-vertex masks.
    Masks are rebuilt from the source botaniq vertex groups when the recipe is applied.
    """
    base_recipe: dict[str, Any] = {
        "schema": BOTANIQ_BRANCH_RECIPE_SCHEMA,
        "enabled": False,
        "reason": "",
        "branch_group_prefix": str(
            getattr(cfg, "botaniq_branch_group_prefix", _DEFAULT_BRANCH_GROUP_PREFIX)
            or _DEFAULT_BRANCH_GROUP_PREFIX
        ),
        "branch_groups_available": [],
        "branch_groups_used": [],
        "helper_groups": {
            "trunk": _TRUNK_GROUP,
            "distance_to_trunk": _DISTANCE_TO_TRUNK_GROUP,
            "leaves_gradient": _LEAVES_GRADIENT_GROUP,
        },
        "branches": [],
        "swap_pairs": [],
        "settings": {
            "weight_threshold": float(_cfg_float(cfg, "botaniq_branch_weight_threshold", 0.02)),
            "min_vertices": int(_cfg_int(cfg, "botaniq_branch_min_vertices", 128)),
            "influence_power": float(_cfg_float(cfg, "botaniq_branch_influence_power", 0.90)),
            "tip_influence_power": float(
                _cfg_float(cfg, "botaniq_branch_tip_influence_power", 0.85)
            ),
            "base_influence_floor": float(
                _cfg_float(cfg, "botaniq_branch_base_influence_floor", 0.12)
            ),
            "max_displacement_fraction": float(
                _cfg_float(cfg, "botaniq_branch_max_displacement_fraction", 0.22)
            ),
            "max_crown_displacement_fraction": float(
                _cfg_float(cfg, "botaniq_branch_max_crown_displacement_fraction", 0.35)
            ),
            "max_displacement_m": float(_cfg_float(cfg, "botaniq_branch_max_displacement_m", 1.25)),
            "prune_probability": float(_cfg_float(cfg, "botaniq_branch_prune_probability", 0.0)),
            "vertical_jitter_max_fraction": float(
                _cfg_float(cfg, "botaniq_branch_vertical_jitter_max_fraction", 0.0)
            ),
            "vertical_jitter_max_m": float(
                _cfg_float(cfg, "botaniq_branch_vertical_jitter_max_m", 0.45)
            ),
            "swap_frame_strength": float(
                _cfg_float(cfg, "botaniq_branch_swap_frame_strength", 0.0)
            ),
            "swap_translation_strength": float(
                _cfg_float(cfg, "botaniq_branch_swap_translation_strength", 0.0)
            ),
            "swap_translation_max_fraction": float(
                _cfg_float(cfg, "botaniq_branch_swap_translation_max_fraction", 0.0)
            ),
            "swap_scale_exchange_strength": float(
                _cfg_float(cfg, "botaniq_branch_swap_scale_exchange_strength", 0.0)
            ),
        },
    }

    if tree_obj.type != "MESH" or tree_obj.data is None:
        base_recipe["reason"] = "not_mesh"
        return base_recipe

    if not _cfg_bool(cfg, "botaniq_branch_variation_enabled", True):
        base_recipe["reason"] = "disabled"
        return base_recipe

    probability = _clamp01(_cfg_float(cfg, "botaniq_branch_variation_probability", 1.0))
    base_recipe["probability"] = float(probability)
    if rng.random() > probability:
        base_recipe["reason"] = "skipped_probability"
        return base_recipe

    prefix = str(base_recipe["branch_group_prefix"])
    branch_group_names = _sorted_branch_group_names(tree_obj, prefix)
    base_recipe["branch_groups_available"] = list(branch_group_names)
    if not branch_group_names:
        base_recipe["reason"] = "no_branch_groups"
        return base_recipe

    coords = [vertex.co.copy() for vertex in tree_obj.data.vertices]
    if not coords:
        base_recipe["reason"] = "empty_mesh"
        return base_recipe

    trunk_weights, has_trunk_weights = _read_vertex_group_weights(tree_obj, _TRUNK_GROUP)
    distance_weights, has_distance_weights = _read_vertex_group_weights(
        tree_obj,
        _DISTANCE_TO_TRUNK_GROUP,
    )
    leaves_weights, has_leaves_weights = _read_vertex_group_weights(
        tree_obj,
        _LEAVES_GRADIENT_GROUP,
    )

    trunk_center = _estimate_trunk_center(coords, trunk_weights, has_trunk_weights)
    bounds_center, crown_radius = _bounds_center_and_radius(coords)
    profiles = _build_branch_profiles(
        tree_obj=tree_obj,
        coords=coords,
        branch_group_names=branch_group_names,
        trunk_center=trunk_center,
        trunk_weights=trunk_weights,
        distance_weights=distance_weights,
        leaves_weights=leaves_weights,
        has_trunk_weights=has_trunk_weights,
        has_distance_weights=has_distance_weights,
        has_leaves_weights=has_leaves_weights,
        rng=rng,
        cfg=cfg,
    )
    base_recipe["helper_group_presence"] = {
        "has_trunk": bool(has_trunk_weights),
        "has_distance_to_trunk": bool(has_distance_weights),
        "has_leaves_gradient": bool(has_leaves_weights),
    }
    base_recipe["trunk_center"] = _vector_to_list(trunk_center)
    base_recipe["bounds_center"] = _vector_to_list(bounds_center)
    base_recipe["crown_radius_m"] = float(crown_radius)
    if not profiles:
        base_recipe["reason"] = "no_usable_branch_groups"
        return base_recipe

    transforms = _sample_branch_transforms(profiles, crown_radius, rng, cfg)
    swap_pairs = _apply_branch_sector_swaps(transforms, crown_radius, rng, cfg)
    base_recipe["enabled"] = True
    base_recipe["reason"] = "planned"
    base_recipe["branch_groups_used"] = [transform.profile.name for transform in transforms]
    base_recipe["branches"] = [_branch_transform_to_recipe(transform) for transform in transforms]
    base_recipe["swap_pairs"] = [[str(first), str(second)] for first, second in swap_pairs]
    return base_recipe


def _transforms_from_branch_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, Any],
    cfg: object,
) -> list[_BranchTransform]:
    transforms: list[_BranchTransform] = []
    for branch_recipe in recipe.get("branches", []):
        if not isinstance(branch_recipe, dict):
            continue
        transform = _branch_transform_from_recipe(tree_obj, branch_recipe, cfg)
        if transform is not None:
            transforms.append(transform)
    return transforms


def _safe_normalized_xy(vector: Vector, fallback: Vector) -> Vector:
    candidate = Vector((float(vector.x), float(vector.y)))
    if candidate.length > 1e-8:
        return candidate.normalized()
    fallback_xy = Vector((float(fallback.x), float(fallback.y)))
    if fallback_xy.length > 1e-8:
        return fallback_xy.normalized()
    return Vector((1.0, 0.0))


def _matrix_lerp(first: Matrix, second: Matrix, factor: float) -> Matrix:
    factor = _clamp01(factor)
    inverse = 1.0 - factor
    return Matrix(
        tuple(
            tuple(first[row][col] * inverse + second[row][col] * factor for col in range(3))
            for row in range(3)
        )
    )


def _sector_swap_frame_strength(transform: _BranchTransform) -> float:
    if not transform.sector_swap_target_group:
        return 0.0
    if transform.sector_swap_target_direction_xy.length <= 1e-8:
        return 0.0
    return _clamp01(transform.sector_swap_strength)


def _branch_linear_matrix_for_target(
    transform: _BranchTransform,
    target_direction_xy: Vector,
) -> Matrix:
    source_radial = _safe_normalized_xy(
        transform.profile.direction_xy,
        Vector((1.0, 0.0)),
    )
    source_tangent = Vector((-source_radial.y, source_radial.x))
    target_radial = _safe_normalized_xy(target_direction_xy, source_radial)
    target_tangent = Vector((-target_radial.y, target_radial.x))

    m00 = (
        target_radial.x * source_radial.x * transform.radial_scale
        + target_tangent.x * source_tangent.x * transform.tangent_scale
    )
    m01 = (
        target_radial.x * source_radial.y * transform.radial_scale
        + target_tangent.x * source_tangent.y * transform.tangent_scale
    )
    m10 = (
        target_radial.y * source_radial.x * transform.radial_scale
        + target_tangent.y * source_tangent.x * transform.tangent_scale
    )
    m11 = (
        target_radial.y * source_radial.y * transform.radial_scale
        + target_tangent.y * source_tangent.y * transform.tangent_scale
    )
    basis = Matrix(
        (
            (m00, m01, 0.0),
            (m10, m11, 0.0),
            (0.0, 0.0, transform.z_scale),
        )
    )

    rotation = Matrix.Rotation(transform.rotation_rad, 3, "Z")
    if abs(transform.tilt_rad) <= 1e-8:
        return rotation @ basis

    tilt_axis = Vector((-target_radial.y, target_radial.x, 0.0))
    if tilt_axis.length <= 1e-8:
        return rotation @ basis
    tilt = Matrix.Rotation(transform.tilt_rad, 3, tilt_axis.normalized())
    return tilt @ rotation @ basis


def _branch_affine_components(transform: _BranchTransform) -> tuple[Matrix, Vector]:
    profile = transform.profile
    source_matrix = _branch_linear_matrix_for_target(transform, profile.direction_xy)
    source_offset = profile.pivot.copy()
    swap_strength = _sector_swap_frame_strength(transform)
    if swap_strength <= 0.0:
        return source_matrix, source_offset

    target_direction = _safe_normalized_xy(
        transform.sector_swap_target_direction_xy,
        profile.direction_xy,
    )
    swap_matrix = _branch_linear_matrix_for_target(transform, target_direction)
    source_center_local = profile.center - profile.pivot
    target_center = transform.sector_swap_target_center.copy()
    if target_center.length <= 1e-8:
        target_center = transform.sector_swap_target_pivot.copy()
    swap_offset = target_center - (swap_matrix @ source_center_local)
    return (
        _matrix_lerp(source_matrix, swap_matrix, swap_strength),
        source_offset * (1.0 - swap_strength) + swap_offset * swap_strength,
    )


def _transform_branch_coord(coord: Vector, transform: _BranchTransform) -> Vector:
    matrix, offset = _branch_affine_components(transform)
    local = coord - transform.profile.pivot
    return (
        offset
        + (matrix @ local)
        + Vector(
            (
                transform.translation_xy.x,
                transform.translation_xy.y,
                transform.z_translation_m,
            )
        )
    )


def _clamped_branch_delta(coord: Vector, transform: _BranchTransform) -> Vector:
    delta = _transform_branch_coord(coord, transform) - coord
    max_displacement = max(0.0, float(transform.max_displacement_m))
    if max_displacement > 0.0 and delta.length > max_displacement:
        delta = delta.normalized() * max_displacement
    return delta


def _branch_linear_matrix(transform: _BranchTransform) -> Matrix:
    matrix, _offset = _branch_affine_components(transform)
    return matrix


def _remove_temp_mask_attributes(
    tree_obj: bpy.types.Object,
    attributes: list[_BranchMaskAttribute],
) -> None:
    mesh = tree_obj.data
    for mask_attribute in attributes:
        attribute = mesh.attributes.get(mask_attribute.name)
        if attribute is None:
            continue
        try:
            mesh.attributes.remove(attribute)
        except Exception:
            logger.debug("Failed to remove temporary attribute %s", mask_attribute.name)


def _remove_mesh_attribute(tree_obj: bpy.types.Object, attribute_name: str | None) -> None:
    if not attribute_name:
        return
    attribute = tree_obj.data.attributes.get(attribute_name)
    if attribute is None:
        return
    try:
        tree_obj.data.attributes.remove(attribute)
    except Exception:
        logger.debug("Failed to remove temporary attribute %s", attribute_name)


def _create_rest_position_attribute(
    tree_obj: bpy.types.Object,
    coords: list[Vector],
    attribute_name: str,
) -> str:
    mesh = tree_obj.data
    existing_attribute = mesh.attributes.get(attribute_name)
    if existing_attribute is not None:
        mesh.attributes.remove(existing_attribute)
    attribute = mesh.attributes.new(attribute_name, "FLOAT_VECTOR", "POINT")
    for vertex_index, coord in enumerate(coords):
        attribute.data[vertex_index].vector = coord
    return attribute_name


def _create_temp_branch_mask_attributes(
    tree_obj: bpy.types.Object,
    coords: list[Vector],
    transforms: list[_BranchTransform],
    cfg: object,
    attribute_prefix: str = "fg_bq_branch_mask",
) -> tuple[list[_BranchMaskAttribute], int, float]:
    mesh = tree_obj.data
    influence_power = max(0.05, _cfg_float(cfg, "botaniq_branch_influence_power", 0.90))
    raw_masks: list[dict[int, float]] = []
    influence_totals = [0.0] * len(coords)

    for transform in transforms:
        profile = transform.profile
        profile_masks: dict[int, float] = {}
        for index in profile.indices:
            distance_factor = min(
                1.0,
                (coords[index] - profile.pivot).length / profile.max_distance_from_pivot,
            )
            influence = _clamp01(
                profile.weights[index]
                * (distance_factor**influence_power)
                * transform.influence_scale
            )
            if influence <= 1e-5:
                continue
            profile_masks[index] = influence
            influence_totals[index] += influence
        raw_masks.append(profile_masks)

    branch_offsets = [Vector((0.0, 0.0, 0.0)) for _ in coords]
    mask_attributes: list[_BranchMaskAttribute] = []
    for branch_index, (transform, profile_masks) in enumerate(zip(transforms, raw_masks)):
        prefix = str(attribute_prefix).strip() or "fg_bq_branch_mask"
        if "{group}" in prefix or "{index}" in prefix:
            attribute_name = prefix.format(
                group=_branch_safe_name(transform.profile.name),
                index=f"{branch_index:02d}",
            )
        else:
            attribute_name = f"{prefix}_{branch_index:02d}"
        existing_attribute = mesh.attributes.get(attribute_name)
        if existing_attribute is not None:
            mesh.attributes.remove(existing_attribute)
        attribute = mesh.attributes.new(attribute_name, "FLOAT", "POINT")

        for vertex_index, influence in profile_masks.items():
            influence_total = influence_totals[vertex_index]
            mask = influence / influence_total if influence_total > 1.0 else influence
            if mask <= 1e-5:
                continue
            attribute.data[vertex_index].value = mask
            branch_offsets[vertex_index] += (
                _clamped_branch_delta(
                    coords[vertex_index],
                    transform,
                )
                * mask
            )

        mask_attributes.append(
            _BranchMaskAttribute(
                transform=transform,
                name=attribute_name,
            )
        )

    vertices_changed = 0
    max_displacement = 0.0
    for offset in branch_offsets:
        if offset.length <= 1e-7:
            continue
        vertices_changed += 1
        max_displacement = max(max_displacement, float(offset.length))

    return mask_attributes, vertices_changed, max_displacement


def _link_or_set(
    links: bpy.types.NodeLinks,
    value: object,
    input_socket: bpy.types.NodeSocket,
) -> None:
    if isinstance(value, bpy.types.NodeSocket):
        links.new(value, input_socket)
    else:
        input_socket.default_value = float(value)


def _new_math(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    operation: str,
    first: object,
    second: object,
    location: tuple[float, float],
    label: str,
) -> bpy.types.NodeSocket:
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    node.label = label
    node.name = label
    node.location = location
    _link_or_set(links, first, node.inputs[0])
    _link_or_set(links, second, node.inputs[1])
    return node.outputs["Value"]


def _new_vector_math(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    operation: str,
    first: object,
    second: object | None,
    location: tuple[float, float],
    label: str,
) -> bpy.types.NodeSocket:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    node.label = label
    node.name = label
    node.location = location

    if isinstance(first, bpy.types.NodeSocket):
        links.new(first, node.inputs["Vector"])
    else:
        node.inputs["Vector"].default_value = tuple(float(value) for value in first)

    if second is not None:
        input_socket = node.inputs[1]
        if isinstance(second, bpy.types.NodeSocket):
            links.new(second, input_socket)
        else:
            input_socket.default_value = tuple(float(value) for value in second)

    return node.outputs["Vector"]


def _clamp_delta_socket(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    delta_socket: bpy.types.NodeSocket,
    max_displacement: float,
    location: tuple[float, float],
    label: str,
) -> bpy.types.NodeSocket:
    max_displacement = max(0.0, float(max_displacement))
    if max_displacement <= 0.0:
        return delta_socket

    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    length.label = f"{label} length"
    length.name = length.label
    length.location = location
    links.new(delta_socket, length.inputs["Vector"])

    clamped_length = nodes.new("ShaderNodeMath")
    clamped_length.operation = "MINIMUM"
    clamped_length.label = f"{label} clamp length"
    clamped_length.name = clamped_length.label
    clamped_length.location = (location[0] + 180, location[1])
    links.new(length.outputs["Value"], clamped_length.inputs[0])
    clamped_length.inputs[1].default_value = max_displacement

    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    normalize.label = f"{label} direction"
    normalize.name = normalize.label
    normalize.location = (location[0] + 180, location[1] - 120)
    links.new(delta_socket, normalize.inputs["Vector"])

    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.label = f"{label} clamped"
    scale.name = scale.label
    scale.location = (location[0] + 420, location[1] - 80)
    links.new(normalize.outputs["Vector"], scale.inputs["Vector"])
    links.new(clamped_length.outputs["Value"], scale.inputs["Scale"])
    return scale.outputs["Vector"]


def _linear_component_socket(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    local_x: bpy.types.NodeSocket,
    local_y: bpy.types.NodeSocket,
    local_z: bpy.types.NodeSocket,
    row: tuple[float, float, float],
    offset: float,
    x_location: float,
    y_location: float,
    label: str,
) -> bpy.types.NodeSocket:
    x_term = _new_math(
        nodes,
        links,
        "MULTIPLY",
        local_x,
        row[0],
        (x_location, y_location),
        f"{label} X term",
    )
    y_term = _new_math(
        nodes,
        links,
        "MULTIPLY",
        local_y,
        row[1],
        (x_location, y_location - 40),
        f"{label} Y term",
    )
    z_term = _new_math(
        nodes,
        links,
        "MULTIPLY",
        local_z,
        row[2],
        (x_location, y_location - 80),
        f"{label} Z term",
    )
    xy_sum = _new_math(
        nodes,
        links,
        "ADD",
        x_term,
        y_term,
        (x_location + 180, y_location - 20),
        f"{label} XY",
    )
    xyz_sum = _new_math(
        nodes,
        links,
        "ADD",
        xy_sum,
        z_term,
        (x_location + 360, y_location - 40),
        f"{label} XYZ",
    )
    return _new_math(
        nodes,
        links,
        "ADD",
        xyz_sum,
        offset,
        (x_location + 540, y_location - 40),
        f"{label} offset",
    )


def _build_branch_delta_socket(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    position_socket: bpy.types.NodeSocket,
    mask_attribute: _BranchMaskAttribute,
    x_location: float,
    y_location: float,
) -> bpy.types.NodeSocket:
    transform = mask_attribute.transform
    pivot = transform.profile.pivot
    translation = transform.translation_xy
    matrix, output_offset = _branch_affine_components(transform)

    attr = nodes.new("GeometryNodeInputNamedAttribute")
    attr.label = f"{transform.profile.name} mask"
    attr.name = attr.label
    attr.data_type = "FLOAT"
    attr.location = (x_location, y_location - 300)
    attr.inputs["Name"].default_value = mask_attribute.name

    local = _new_vector_math(
        nodes,
        links,
        "SUBTRACT",
        position_socket,
        (pivot.x, pivot.y, pivot.z),
        (x_location, y_location),
        f"{transform.profile.name} local",
    )
    separate = nodes.new("ShaderNodeSeparateXYZ")
    separate.label = f"{transform.profile.name} local xyz"
    separate.name = separate.label
    separate.location = (x_location + 180, y_location)
    links.new(local, separate.inputs["Vector"])

    target_x = _linear_component_socket(
        nodes,
        links,
        separate.outputs["X"],
        separate.outputs["Y"],
        separate.outputs["Z"],
        (matrix[0][0], matrix[0][1], matrix[0][2]),
        output_offset.x + translation.x,
        x_location + 360,
        y_location + 120,
        f"{transform.profile.name} target x",
    )
    target_y = _linear_component_socket(
        nodes,
        links,
        separate.outputs["X"],
        separate.outputs["Y"],
        separate.outputs["Z"],
        (matrix[1][0], matrix[1][1], matrix[1][2]),
        output_offset.y + translation.y,
        x_location + 360,
        y_location - 120,
        f"{transform.profile.name} target y",
    )
    target_z = _linear_component_socket(
        nodes,
        links,
        separate.outputs["X"],
        separate.outputs["Y"],
        separate.outputs["Z"],
        (matrix[2][0], matrix[2][1], matrix[2][2]),
        output_offset.z + transform.z_translation_m,
        x_location + 360,
        y_location - 360,
        f"{transform.profile.name} target z",
    )

    combine = nodes.new("ShaderNodeCombineXYZ")
    combine.label = f"{transform.profile.name} target"
    combine.name = combine.label
    combine.location = (x_location + 1100, y_location - 120)
    links.new(target_x, combine.inputs["X"])
    links.new(target_y, combine.inputs["Y"])
    links.new(target_z, combine.inputs["Z"])

    delta = _new_vector_math(
        nodes,
        links,
        "SUBTRACT",
        combine.outputs["Vector"],
        position_socket,
        (x_location + 1300, y_location - 120),
        f"{transform.profile.name} delta",
    )
    clamped_delta = _clamp_delta_socket(
        nodes,
        links,
        delta,
        transform.max_displacement_m,
        (x_location + 1470, y_location - 20),
        f"{transform.profile.name} delta",
    )

    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.label = f"{transform.profile.name} weighted delta"
    scale.name = scale.label
    scale.location = (x_location + 1950, y_location - 120)
    links.new(clamped_delta, scale.inputs["Vector"])
    links.new(attr.outputs["Attribute"], scale.inputs["Scale"])
    return scale.outputs["Vector"]


def _build_botaniq_branch_variation_node_group(
    tree_obj: bpy.types.Object,
    mask_attributes: list[_BranchMaskAttribute],
) -> bpy.types.GeometryNodeTree:
    node_group = bpy.data.node_groups.new(
        f"{tree_obj.name}_BotaniqBranchBakeGN",
        "GeometryNodeTree",
    )
    node_group.use_fake_user = True
    node_group.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    node_group.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-600, 0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (2600, 0)

    position = nodes.new("GeometryNodeInputPosition")
    position.location = (-600, -240)

    total_offset: bpy.types.NodeSocket | None = None
    for branch_index, mask_attribute in enumerate(mask_attributes):
        branch_delta = _build_branch_delta_socket(
            nodes,
            links,
            position.outputs["Position"],
            mask_attribute,
            x_location=-300,
            y_location=-branch_index * 760,
        )
        if total_offset is None:
            total_offset = branch_delta
        else:
            total_offset = _new_vector_math(
                nodes,
                links,
                "ADD",
                total_offset,
                branch_delta,
                (1750, -branch_index * 760),
                f"Accumulated branch delta {branch_index}",
            )

    set_position = nodes.new("GeometryNodeSetPosition")
    set_position.label = "Bake botaniq branch crown variation"
    set_position.name = set_position.label
    set_position.location = (2200, 0)
    links.new(group_in.outputs["Geometry"], set_position.inputs["Geometry"])
    if total_offset is not None:
        links.new(total_offset, set_position.inputs["Offset"])
    links.new(set_position.outputs["Geometry"], group_out.inputs["Geometry"])
    return node_group


def _build_botaniq_single_branch_node_group(
    tree_obj: bpy.types.Object,
    mask_attribute: _BranchMaskAttribute,
    rest_position_attribute: str | None = None,
) -> bpy.types.GeometryNodeTree:
    group_name = _branch_safe_name(mask_attribute.transform.profile.name)
    node_group = bpy.data.node_groups.new(
        f"{tree_obj.name}_{group_name}_BranchDeformGN",
        "GeometryNodeTree",
    )
    node_group.use_fake_user = True
    node_group.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    node_group.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-600, 0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (2400, 0)
    position = nodes.new("GeometryNodeInputPosition")
    position.location = (-600, -240)
    base_position_socket = position.outputs["Position"]
    if rest_position_attribute:
        rest_position = nodes.new("GeometryNodeInputNamedAttribute")
        rest_position.label = "Authoring rest position"
        rest_position.name = rest_position.label
        rest_position.data_type = "FLOAT_VECTOR"
        rest_position.location = (-600, -420)
        rest_position.inputs["Name"].default_value = rest_position_attribute
        base_position_socket = rest_position.outputs["Attribute"]
    branch_delta = _build_branch_delta_socket(
        nodes,
        links,
        base_position_socket,
        mask_attribute,
        x_location=-300,
        y_location=0,
    )
    set_position = nodes.new("GeometryNodeSetPosition")
    set_position.label = f"Deform {mask_attribute.transform.profile.name}"
    set_position.name = set_position.label
    set_position.location = (2050, 0)
    links.new(group_in.outputs["Geometry"], set_position.inputs["Geometry"])
    links.new(branch_delta, set_position.inputs["Offset"])
    links.new(set_position.outputs["Geometry"], group_out.inputs["Geometry"])
    return node_group


def _apply_geometry_nodes_modifier(
    tree_obj: bpy.types.Object,
    modifier: bpy.types.Modifier,
) -> None:
    context = bpy.context
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(context.selected_objects)

    try:
        for obj in previous_selected:
            obj.select_set(False)
        tree_obj.select_set(True)
        view_layer.objects.active = tree_obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    finally:
        tree_obj.select_set(False)
        for obj in previous_selected:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if previous_active is not None and previous_active.name in bpy.data.objects:
            view_layer.objects.active = previous_active


def _apply_branch_variation_with_geometry_nodes(
    tree_obj: bpy.types.Object,
    coords: list[Vector],
    transforms: list[_BranchTransform],
    swap_pairs: list[tuple[str, str]],
    cfg: object,
    *,
    apply_modifier: bool | None = None,
    modifier_layout: str | None = None,
    mask_attribute_prefix: str = "fg_bq_branch_mask",
    keep_mask_attributes: bool | None = None,
) -> dict[str, Any]:
    leaf_geometry = capture_leaf_unit_geometry(tree_obj, cfg)
    mask_attributes, vertices_changed, max_displacement = _create_temp_branch_mask_attributes(
        tree_obj,
        coords,
        transforms,
        cfg,
        attribute_prefix=mask_attribute_prefix,
    )
    if not mask_attributes or vertices_changed <= 0:
        _remove_temp_mask_attributes(tree_obj, mask_attributes)
        return {
            "applied": False,
            "mode": "geometry_nodes_no_vertex_motion",
            "branch_group_count": len(transforms),
            "branch_groups_used": [transform.profile.name for transform in transforms],
            "variation_roles": [transform.variation_role for transform in transforms],
            "swap_pair_count": len(swap_pairs),
            "swap_pairs": swap_pairs,
            "vertices_changed": 0,
            "max_displacement_m": 0.0,
        }

    if apply_modifier is None:
        apply_modifier = _cfg_bool(cfg, "botaniq_branch_geometry_nodes_apply", True)
    if keep_mask_attributes is None:
        keep_mask_attributes = not bool(apply_modifier)

    layout = (
        str(
            modifier_layout
            or getattr(cfg, "botaniq_branch_geometry_nodes_layout", "combined")
            or "combined"
        )
        .strip()
        .lower()
    )
    modifiers: list[bpy.types.Modifier] = []
    rest_position_attribute: str | None = None
    if layout in {"per_branch", "branch", "modifiers"}:
        if not apply_modifier:
            rest_position_attribute = _create_rest_position_attribute(
                tree_obj,
                coords,
                "fg_bq_authoring_rest_position",
            )
        for branch_index, mask_attribute in enumerate(mask_attributes):
            node_group = _build_botaniq_single_branch_node_group(
                tree_obj,
                mask_attribute,
                rest_position_attribute=rest_position_attribute,
            )
            branch_name = mask_attribute.transform.profile.name
            modifier = tree_obj.modifiers.new(
                f"FG Branch {branch_index + 1:02d} {branch_name}",
                "NODES",
            )
            modifier.node_group = node_group
            modifiers.append(modifier)
    else:
        node_group = _build_botaniq_branch_variation_node_group(tree_obj, mask_attributes)
        modifier = tree_obj.modifiers.new("BotaniqBranchCrownBake", "NODES")
        modifier.node_group = node_group
        modifiers.append(modifier)

    if apply_modifier:
        try:
            for modifier in list(modifiers):
                if tree_obj.modifiers.get(modifier.name) is not None:
                    _apply_geometry_nodes_modifier(tree_obj, modifier)
            restored_leaf_vertices = restore_leaf_unit_geometry_to_centroids(
                tree_obj,
                leaf_geometry,
            )
        except Exception:
            _remove_temp_mask_attributes(tree_obj, mask_attributes)
            logger.exception(
                "Failed to apply botaniq branch Geometry Nodes bake for %s",
                tree_obj.name,
            )
            raise
        if not keep_mask_attributes:
            _remove_temp_mask_attributes(tree_obj, mask_attributes)
            _remove_mesh_attribute(tree_obj, rest_position_attribute)
    else:
        restored_leaf_vertices = 0

    tree_obj.data.update()
    return {
        "applied": True,
        "mode": (
            "geometry_nodes_bq_branch_vertex_groups"
            if apply_modifier
            else f"live_geometry_nodes_{layout}_bq_branch_vertex_groups"
        ),
        "branch_group_count": len(transforms),
        "branch_groups_used": [transform.profile.name for transform in transforms],
        "variation_roles": [transform.variation_role for transform in transforms],
        "swap_pair_count": len(swap_pairs),
        "swap_pairs": swap_pairs,
        "vertices_changed": vertices_changed,
        "max_displacement_m": float(max_displacement),
        "modifier_layout": layout,
        "mask_attributes": [mask_attribute.name for mask_attribute in mask_attributes],
        "rest_position_attribute": rest_position_attribute or "",
        "leaf_vertices_preserved": int(restored_leaf_vertices),
    }


def _apply_branch_variation_direct_mesh(
    tree_obj: bpy.types.Object,
    coords: list[Vector],
    transforms: list[_BranchTransform],
    swap_pairs: list[tuple[str, str]],
    cfg: object,
) -> dict[str, Any]:
    influence_power = max(0.05, _cfg_float(cfg, "botaniq_branch_influence_power", 0.90))
    leaf_geometry = capture_leaf_unit_geometry(tree_obj, cfg)

    deltas = [Vector((0.0, 0.0, 0.0)) for _ in coords]
    influence_totals = [0.0] * len(coords)
    for transform in transforms:
        profile = transform.profile
        for index in profile.indices:
            distance_factor = min(
                1.0,
                (coords[index] - profile.pivot).length / profile.max_distance_from_pivot,
            )
            influence = _clamp01(
                profile.weights[index]
                * (distance_factor**influence_power)
                * transform.influence_scale
            )
            if influence <= 1e-5:
                continue
            deltas[index] += _clamped_branch_delta(coords[index], transform) * influence
            influence_totals[index] += influence

    vertices_changed = 0
    max_displacement = 0.0
    for index, vertex in enumerate(tree_obj.data.vertices):
        influence_total = influence_totals[index]
        if influence_total <= 0.0:
            continue
        delta = deltas[index]
        if influence_total > 1.0:
            delta /= influence_total
        if delta.length <= 1e-7:
            continue
        vertex.co = coords[index] + delta
        vertices_changed += 1
        max_displacement = max(max_displacement, float(delta.length))

    tree_obj.data.update()
    restored_leaf_vertices = restore_leaf_unit_geometry_to_centroids(tree_obj, leaf_geometry)
    return {
        "applied": vertices_changed > 0,
        "mode": "mesh_bq_branch_vertex_groups" if vertices_changed > 0 else "no_vertex_motion",
        "branch_group_count": len(transforms),
        "branch_groups_used": [transform.profile.name for transform in transforms],
        "variation_roles": [transform.variation_role for transform in transforms],
        "swap_pair_count": len(swap_pairs),
        "swap_pairs": swap_pairs,
        "vertices_changed": vertices_changed,
        "max_displacement_m": float(max_displacement),
        "leaf_vertices_preserved": int(restored_leaf_vertices),
    }


def _store_branch_variation_result(
    tree_obj: bpy.types.Object,
    result: dict[str, Any],
) -> None:
    tree_obj["botaniq_branch_variation_applied"] = bool(result.get("applied", False))
    tree_obj["botaniq_branch_variation_mode"] = str(result.get("mode", ""))
    tree_obj["botaniq_branch_group_count"] = int(result.get("branch_group_count", 0))
    tree_obj["botaniq_branch_groups_used"] = ",".join(
        str(name) for name in result.get("branch_groups_used", [])
    )
    tree_obj["botaniq_branch_swap_pair_count"] = int(result.get("swap_pair_count", 0))
    tree_obj["botaniq_branch_vertices_changed"] = int(result.get("vertices_changed", 0))
    tree_obj["botaniq_branch_max_displacement_m"] = float(result.get("max_displacement_m", 0.0))
    tree_obj["botaniq_branch_leaf_vertices_preserved"] = int(
        result.get("leaf_vertices_preserved", 0)
    )
    tree_obj["botaniq_branch_variation_roles"] = ",".join(
        str(name) for name in result.get("variation_roles", [])
    )
    if result.get("recipe") is not None:
        try:
            tree_obj["botaniq_branch_recipe_json"] = json.dumps(
                result["recipe"],
                sort_keys=True,
            )
        except Exception:
            logger.debug("Failed to store botaniq branch recipe JSON on %s", tree_obj.name)


def apply_botaniq_branch_group_crown_variation_from_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, Any],
    cfg: object,
    *,
    apply_modifier: bool | None = None,
    modifier_layout: str | None = None,
    mask_attribute_prefix: str = "fg_bq_branch_mask",
    keep_mask_attributes: bool | None = None,
) -> dict[str, Any]:
    """Apply a branch-crown recipe as live GN modifiers or as baked mesh geometry."""
    if tree_obj.type != "MESH" or tree_obj.data is None:
        result = {"applied": False, "mode": "not_mesh", "branch_group_count": 0}
        result["recipe"] = recipe
        _store_branch_variation_result(tree_obj, result)
        return result

    if not recipe or not bool(recipe.get("enabled", False)):
        reason = (
            str(recipe.get("reason", "not_planned")) if isinstance(recipe, dict) else "not_planned"
        )
        result = {"applied": False, "mode": reason, "branch_group_count": 0}
        result["recipe"] = recipe
        _store_branch_variation_result(tree_obj, result)
        return result

    coords = [vertex.co.copy() for vertex in tree_obj.data.vertices]
    if not coords:
        result = {"applied": False, "mode": "empty_mesh", "branch_group_count": 0}
        result["recipe"] = recipe
        _store_branch_variation_result(tree_obj, result)
        return result

    transforms = _transforms_from_branch_recipe(tree_obj, recipe, cfg)
    if not transforms:
        result = {
            "applied": False,
            "mode": "recipe_no_usable_branch_groups",
            "branch_group_count": len(recipe.get("branches", [])),
        }
        result["recipe"] = recipe
        _store_branch_variation_result(tree_obj, result)
        return result

    swap_pairs = [
        (str(pair[0]), str(pair[1]))
        for pair in recipe.get("swap_pairs", [])
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    ]
    backend = (
        str(getattr(cfg, "botaniq_branch_variation_backend", "geometry_nodes") or "geometry_nodes")
        .strip()
        .lower()
    )
    if backend in {"geometry_nodes", "geometry_node", "gn", "nodes"}:
        result = _apply_branch_variation_with_geometry_nodes(
            tree_obj,
            coords,
            transforms,
            swap_pairs,
            cfg,
            apply_modifier=apply_modifier,
            modifier_layout=modifier_layout,
            mask_attribute_prefix=mask_attribute_prefix,
            keep_mask_attributes=keep_mask_attributes,
        )
    elif backend in {"mesh", "direct_mesh", "python"}:
        result = _apply_branch_variation_direct_mesh(
            tree_obj,
            coords,
            transforms,
            swap_pairs,
            cfg,
        )
    else:
        logger.warning(
            "Unknown botaniq_branch_variation_backend=%r; using geometry_nodes.",
            backend,
        )
        result = _apply_branch_variation_with_geometry_nodes(
            tree_obj,
            coords,
            transforms,
            swap_pairs,
            cfg,
            apply_modifier=apply_modifier,
            modifier_layout=modifier_layout,
            mask_attribute_prefix=mask_attribute_prefix,
            keep_mask_attributes=keep_mask_attributes,
        )

    result["recipe"] = recipe
    _store_branch_variation_result(tree_obj, result)
    if int(result.get("vertices_changed", 0)) > 0:
        logger.debug(
            "Applied botaniq branch variation to %s: groups=%s, changed_vertices=%s, "
            "max_displacement=%.3f",
            tree_obj.name,
            ",".join(str(name) for name in result.get("branch_groups_used", [])),
            int(result.get("vertices_changed", 0)),
            float(result.get("max_displacement_m", 0.0)),
        )
    return result


def apply_botaniq_branch_group_crown_variation(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: object,
) -> dict[str, Any]:
    """Bake crown morphology variation from botaniq editable vertex groups.

    Botaniq trees in the starter pack are static meshes, but editable assets expose branch,
    trunk, distance-to-trunk, and leaves-gradient vertex groups. This pass uses those groups
    as a deterministic deformation mask, then writes the changed coordinates back to the copied
    mesh so the scatter system samples pre-baked tree templates.
    """
    recipe = build_botaniq_branch_group_crown_recipe(tree_obj, rng, cfg)
    return apply_botaniq_branch_group_crown_variation_from_recipe(tree_obj, recipe, cfg)
