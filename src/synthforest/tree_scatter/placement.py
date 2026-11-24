from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DensityFieldResult:
    attribute_name: str
    values: list[float]
    total_area_m2: float
    eligible_area_m2: float
    eligible_mean: float
    eligible_min: float
    eligible_max: float


@dataclass(frozen=True)
class TreeVariantTemplate:
    variant_index: int
    variant_id: int
    crown_diameter_m: float
    preset_name: str


@dataclass(frozen=True)
class ScatterPlacement:
    instance_id: int
    x: float
    y: float
    z: float
    variant_index: int
    variant_id: int
    crown_radius_m: float
    stem_exclusion_radius_m: float
    scale_x: float
    scale_y: float
    scale_z: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    leaf_hue_shift: float
    leaf_brightness: float
    density_field_value: float


@dataclass(frozen=True)
class ScatterValidationResult:
    target_tree_count: float
    actual_tree_count: int
    target_trees_per_ha: float
    actual_trees_per_ha: float
    actual_to_target_count_ratio: float
    eligible_area_m2: float
    density_field_mean: float
    density_field_min: float
    density_field_max: float
    nearest_neighbor_min_m: float
    nearest_neighbor_p05_m: float
    nearest_neighbor_mean_m: float
    nearest_neighbor_median_m: float
    nearest_neighbor_p95_m: float
    crown_overlap_pair_count: int
    crown_overlap_area_sum_m2: float
    crown_overlap_area_fraction_of_crown_area: float
    canopy_cover_fraction_total: float
    canopy_cover_fraction_eligible_bbox: float
    density_grid_cells: int
    density_grid_counts: list[list[int]]
    density_grid_trees_per_ha: list[list[float]]
    density_grid_target_counts: list[list[float]]
    density_grid_target_trees_per_ha: list[list[float]]
    density_grid_actual_to_target_ratio: list[list[float]]


def _attribute_values(mesh: Any, name: str, fallback: float = 1.0) -> list[float]:
    attribute = getattr(mesh, "attributes", {}).get(name) if hasattr(mesh, "attributes") else None
    if attribute is None:
        return [float(fallback)] * len(mesh.polygons)

    values: list[float] = []
    for index in range(len(mesh.polygons)):
        value = fallback
        if index < len(attribute.data):
            try:
                value = float(attribute.data[index].value)
            except Exception:
                value = fallback
        values.append(float(value))
    return values


def _set_float_attribute(mesh: Any, name: str, domain: str, values: list[float]) -> None:
    attribute = mesh.attributes.get(name)
    if attribute is None:
        attribute = mesh.attributes.new(name=name, type="FLOAT", domain=domain)
    attribute.data.foreach_set("value", [float(value) for value in values])


def _hash_unit(ix: int, iy: int, seed: int) -> float:
    value = (
        int(ix) * 374761393 + int(iy) * 668265263 + int(seed) * 1442695040888963407
    ) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    value = value ^ (value >> 31)
    return float(value & 0xFFFFFFFF) / float(0xFFFFFFFF)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _value_noise_2d(x_coord: float, y_coord: float, seed: int) -> float:
    ix = math.floor(x_coord)
    iy = math.floor(y_coord)
    fx = _smoothstep(x_coord - ix)
    fy = _smoothstep(y_coord - iy)

    v00 = _hash_unit(ix, iy, seed)
    v10 = _hash_unit(ix + 1, iy, seed)
    v01 = _hash_unit(ix, iy + 1, seed)
    v11 = _hash_unit(ix + 1, iy + 1, seed)
    return _lerp(_lerp(v00, v10, fx), _lerp(v01, v11, fx), fy)


def _fbm_2d(x_coord: float, y_coord: float, scale: float, seed: int) -> float:
    scale = max(1e-6, float(scale))
    value = 0.0
    amplitude = 0.5
    total_amplitude = 0.0
    frequency = scale
    for octave in range(4):
        value += amplitude * _value_noise_2d(
            x_coord * frequency,
            y_coord * frequency,
            int(seed) + octave * 1013,
        )
        total_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    if total_amplitude <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, value / total_amplitude))


def _density_multiplier(
    x_coord: float,
    y_coord: float,
    *,
    noise_scale: float,
    threshold: float,
    variation_strength: float,
    cluster_exponent: float,
    seed: int,
) -> float:
    raw = _fbm_2d(x_coord, y_coord, noise_scale, seed)
    denom = max(1e-6, 1.0 - float(threshold))
    thresholded = max(0.0, min(1.0, (raw - float(threshold)) / denom))
    clustered = thresholded ** max(0.1, float(cluster_exponent))
    variation = max(0.0, min(1.0, float(variation_strength)))
    return max(0.0, min(1.0, (1.0 - variation) + variation * clustered))


def build_density_field_on_terrain(
    terrain_obj: Any,
    *,
    attribute_name: str,
    eligibility_attribute_name: str,
    noise_scale: float,
    threshold: float,
    variation_strength: float,
    cluster_exponent: float,
    seed: int,
) -> DensityFieldResult:
    mesh = terrain_obj.data
    eligibility = _attribute_values(mesh, eligibility_attribute_name, fallback=1.0)

    values: list[float] = []
    total_area = 0.0
    eligible_area = 0.0
    weighted_sum = 0.0
    eligible_values: list[float] = []

    for face in mesh.polygons:
        area = float(face.area)
        total_area += area
        eligible = eligibility[face.index] > 0.5
        center = face.center
        density = (
            _density_multiplier(
                float(center.x),
                float(center.y),
                noise_scale=noise_scale,
                threshold=threshold,
                variation_strength=variation_strength,
                cluster_exponent=cluster_exponent,
                seed=seed,
            )
            if eligible
            else 0.0
        )
        values.append(float(density))
        if eligible:
            eligible_area += area
            weighted_sum += density * area
            eligible_values.append(float(density))

    _set_float_attribute(mesh, attribute_name, "FACE", values)
    mean = weighted_sum / eligible_area if eligible_area > 1e-9 else 0.0
    return DensityFieldResult(
        attribute_name=attribute_name,
        values=values,
        total_area_m2=float(total_area),
        eligible_area_m2=float(eligible_area),
        eligible_mean=float(mean),
        eligible_min=float(min(eligible_values) if eligible_values else 0.0),
        eligible_max=float(max(eligible_values) if eligible_values else 0.0),
    )


def _sample_triangle(
    rng: random.Random,
    a: Any,
    b: Any,
    c: Any,
) -> tuple[float, float]:
    r1 = math.sqrt(rng.random())
    r2 = rng.random()
    wa = 1.0 - r1
    wb = r1 * (1.0 - r2)
    wc = r1 * r2
    return (
        float(a.x) * wa + float(b.x) * wb + float(c.x) * wc,
        float(a.y) * wa + float(b.y) * wb + float(c.y) * wc,
    )


def _sample_face_xy(rng: random.Random, mesh: Any, face: Any) -> tuple[float, float]:
    vertices = [mesh.vertices[int(index)].co for index in face.vertices]
    if len(vertices) < 3:
        center = face.center
        return float(center.x), float(center.y)
    if len(vertices) == 3:
        return _sample_triangle(rng, vertices[0], vertices[1], vertices[2])

    triangles = [
        (vertices[0], vertices[index], vertices[index + 1]) for index in range(1, len(vertices) - 1)
    ]
    areas = [
        abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) * 0.5 for a, b, c in triangles
    ]
    total = sum(float(area) for area in areas)
    if total <= 1e-12:
        center = face.center
        return float(center.x), float(center.y)

    pick = rng.random() * total
    running = 0.0
    for triangle, area in zip(triangles, areas):
        running += float(area)
        if pick <= running:
            return _sample_triangle(rng, *triangle)
    return _sample_triangle(rng, *triangles[-1])


def _sample_count(expected: float, rng: random.Random) -> int:
    expected = max(0.0, float(expected))
    base = int(math.floor(expected))
    return base + (1 if rng.random() < expected - base else 0)


def _placement_candidate(
    *,
    rng: random.Random,
    instance_id: int,
    x_coord: float,
    y_coord: float,
    variant: TreeVariantTemplate,
    scatter_cfg: Any,
    density_value: float,
) -> tuple[float, ScatterPlacement]:
    scale_min = max(0.01, float(getattr(scatter_cfg, "tree_scale_min", 1.0)))
    scale_max = max(scale_min, float(getattr(scatter_cfg, "tree_scale_max", 1.0)))
    base_scale = rng.uniform(scale_min, scale_max)

    height_jitter = max(
        0.0,
        min(0.25, float(getattr(scatter_cfg, "instance_height_scale_jitter", 0.035))),
    )
    height_scale = rng.uniform(1.0 - height_jitter, 1.0 + height_jitter)

    asymmetry_max = max(
        0.0,
        min(0.20, float(getattr(scatter_cfg, "instance_crown_asymmetry_max", 0.035))),
    )
    asymmetry = rng.uniform(-asymmetry_max, asymmetry_max)
    scale_x = base_scale * max(0.2, 1.0 + asymmetry)
    scale_y = base_scale * max(0.2, 1.0 - asymmetry)
    scale_z = base_scale * height_scale

    lean_probability = max(
        0.0,
        min(1.0, float(getattr(scatter_cfg, "instance_lean_probability", 0.75))),
    )
    lean_max = math.radians(
        max(0.0, min(12.0, float(getattr(scatter_cfg, "instance_lean_max_deg", 3.0))))
    )
    lean_angle = rng.uniform(0.0, lean_max) if rng.random() < lean_probability else 0.0
    lean_direction = rng.uniform(0.0, math.tau)
    rotation_x = math.sin(lean_direction) * lean_angle
    rotation_y = -math.cos(lean_direction) * lean_angle
    rotation_z = rng.uniform(0.0, math.tau)

    leaf_hue_min = float(getattr(scatter_cfg, "botaniq_instance_leaf_hue_shift_min", -0.006))
    leaf_hue_max = float(getattr(scatter_cfg, "botaniq_instance_leaf_hue_shift_max", 0.006))
    leaf_brightness_min = float(
        getattr(scatter_cfg, "botaniq_instance_leaf_brightness_min", -0.006)
    )
    leaf_brightness_max = float(getattr(scatter_cfg, "botaniq_instance_leaf_brightness_max", 0.008))
    leaf_hue_min, leaf_hue_max = sorted(
        (max(-0.05, min(0.05, leaf_hue_min)), max(-0.05, min(0.05, leaf_hue_max)))
    )
    leaf_brightness_min, leaf_brightness_max = sorted(
        (max(-0.05, min(0.05, leaf_brightness_min)), max(-0.05, min(0.05, leaf_brightness_max)))
    )

    crown_diameter = max(0.0, float(variant.crown_diameter_m)) * max(scale_x, scale_y)
    crown_radius = crown_diameter * 0.5
    spacing_fraction = max(
        0.0,
        float(getattr(scatter_cfg, "stem_spacing_crown_fraction_floor", 0.45)),
    )
    stem_radius = max(0.05, crown_radius * spacing_fraction)

    placement = ScatterPlacement(
        instance_id=int(instance_id),
        x=float(x_coord),
        y=float(y_coord),
        z=0.0,
        variant_index=int(variant.variant_index),
        variant_id=int(variant.variant_id),
        crown_radius_m=float(crown_radius),
        stem_exclusion_radius_m=float(stem_radius),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        scale_z=float(scale_z),
        rotation_x=float(rotation_x),
        rotation_y=float(rotation_y),
        rotation_z=float(rotation_z),
        leaf_hue_shift=float(rng.uniform(leaf_hue_min, leaf_hue_max)),
        leaf_brightness=float(rng.uniform(leaf_brightness_min, leaf_brightness_max)),
        density_field_value=float(density_value),
    )
    return rng.random(), placement


def _nearby_placements(
    grid: dict[tuple[int, int], list[ScatterPlacement]],
    x_coord: float,
    y_coord: float,
    radius: float,
    cell_size: float,
) -> list[ScatterPlacement]:
    cx = math.floor(float(x_coord) / cell_size)
    cy = math.floor(float(y_coord) / cell_size)
    reach = int(math.ceil(max(0.0, float(radius)) / cell_size)) + 1
    neighbors: list[ScatterPlacement] = []
    for gx in range(cx - reach, cx + reach + 1):
        for gy in range(cy - reach, cy + reach + 1):
            neighbors.extend(grid.get((gx, gy), []))
    return neighbors


def _build_poisson_candidate_pool(
    *,
    mesh: Any,
    density_field: DensityFieldResult,
    eligibility: list[float],
    target_stem_density_per_m2: float,
    variants: list[TreeVariantTemplate],
    scatter_cfg: Any,
    seed: int,
    candidate_multiplier: float,
) -> list[tuple[float, ScatterPlacement]]:
    rng = random.Random(int(seed) + 2029)
    density_mean = max(1e-9, float(density_field.eligible_mean))
    candidate_density_scale = (
        max(0.0, float(target_stem_density_per_m2))
        * max(1.0, float(candidate_multiplier))
        / density_mean
    )

    candidates: list[tuple[float, ScatterPlacement]] = []
    next_instance_id = 1
    for face in mesh.polygons:
        if face.index >= len(density_field.values) or face.index >= len(eligibility):
            continue
        if eligibility[face.index] <= 0.5:
            continue
        density_value = max(0.0, float(density_field.values[face.index]))
        if density_value <= 0.0:
            continue

        expected = candidate_density_scale * density_value * float(face.area)
        for _index in range(_sample_count(expected, rng)):
            x_coord, y_coord = _sample_face_xy(rng, mesh, face)
            variant = variants[rng.randrange(len(variants))]
            priority, placement = _placement_candidate(
                rng=rng,
                instance_id=next_instance_id,
                x_coord=x_coord,
                y_coord=y_coord,
                variant=variant,
                scatter_cfg=scatter_cfg,
                density_value=density_value,
            )
            candidates.append((priority, placement))
            next_instance_id += 1
    return candidates


def _accept_poisson_candidates(
    candidates: list[tuple[float, ScatterPlacement]],
) -> list[ScatterPlacement]:
    if not candidates:
        return []

    max_radius = max(candidate.stem_exclusion_radius_m for _priority, candidate in candidates)
    cell_size = max(0.25, max_radius)
    accepted: list[ScatterPlacement] = []
    grid: dict[tuple[int, int], list[ScatterPlacement]] = {}

    for _priority, candidate in sorted(candidates, key=lambda item: item[0]):
        blocked = False
        for other in _nearby_placements(
            grid,
            candidate.x,
            candidate.y,
            candidate.stem_exclusion_radius_m + max_radius,
            cell_size,
        ):
            min_distance = candidate.stem_exclusion_radius_m + other.stem_exclusion_radius_m
            dx = candidate.x - other.x
            dy = candidate.y - other.y
            if dx * dx + dy * dy < min_distance * min_distance:
                blocked = True
                break
        if blocked:
            continue

        accepted.append(candidate)
        key = (
            math.floor(candidate.x / cell_size),
            math.floor(candidate.y / cell_size),
        )
        grid.setdefault(key, []).append(candidate)
    return accepted


def generate_crown_aware_poisson_placements(
    terrain_obj: Any,
    *,
    density_field: DensityFieldResult,
    target_stem_density_per_m2: float,
    variants: list[TreeVariantTemplate],
    scatter_cfg: Any,
    seed: int,
    eligibility_attribute_name: str,
) -> list[ScatterPlacement]:
    if not variants:
        return []

    mesh = terrain_obj.data
    eligibility = _attribute_values(mesh, eligibility_attribute_name, fallback=1.0)
    multiplier = max(
        1.0,
        float(getattr(scatter_cfg, "precomputed_poisson_candidate_multiplier", 1.35)),
    )
    target_count = int(round(max(0.0, target_stem_density_per_m2) * density_field.eligible_area_m2))
    max_multiplier = max(
        multiplier,
        float(getattr(scatter_cfg, "precomputed_poisson_candidate_multiplier_max", 8.0)),
    )
    growth = max(
        1.05,
        float(getattr(scatter_cfg, "precomputed_poisson_candidate_multiplier_growth", 1.75)),
    )

    accepted: list[ScatterPlacement] = []
    best_accepted: list[ScatterPlacement] = []
    current_multiplier = multiplier
    while True:
        candidates = _build_poisson_candidate_pool(
            mesh=mesh,
            density_field=density_field,
            eligibility=eligibility,
            target_stem_density_per_m2=target_stem_density_per_m2,
            variants=variants,
            scatter_cfg=scatter_cfg,
            seed=seed,
            candidate_multiplier=current_multiplier,
        )
        accepted = _accept_poisson_candidates(candidates)
        if len(accepted) > len(best_accepted):
            best_accepted = accepted
        if target_count <= 0 or len(accepted) >= target_count:
            break
        if current_multiplier >= max_multiplier:
            accepted = best_accepted
            break
        next_multiplier = min(max_multiplier, current_multiplier * growth)
        if next_multiplier <= current_multiplier + 1e-9:
            accepted = best_accepted
            break
        current_multiplier = next_multiplier

    if not accepted:
        return []

    if target_count > 0 and len(accepted) > target_count:
        rng = random.Random(int(seed) + 2029)
        accepted = sorted(rng.sample(accepted, target_count), key=lambda item: item.instance_id)

    return [
        ScatterPlacement(
            instance_id=index,
            x=item.x,
            y=item.y,
            z=item.z,
            variant_index=item.variant_index,
            variant_id=item.variant_id,
            crown_radius_m=item.crown_radius_m,
            stem_exclusion_radius_m=item.stem_exclusion_radius_m,
            scale_x=item.scale_x,
            scale_y=item.scale_y,
            scale_z=item.scale_z,
            rotation_x=item.rotation_x,
            rotation_y=item.rotation_y,
            rotation_z=item.rotation_z,
            leaf_hue_shift=item.leaf_hue_shift,
            leaf_brightness=item.leaf_brightness,
            density_field_value=item.density_field_value,
        )
        for index, item in enumerate(accepted, start=1)
    ]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, float(fraction))) * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    t = pos - low
    return _lerp(sorted_values[low], sorted_values[high], t)


def _circle_overlap_area(radius_a: float, radius_b: float, distance: float) -> float:
    radius_a = max(0.0, float(radius_a))
    radius_b = max(0.0, float(radius_b))
    distance = max(0.0, float(distance))
    if radius_a <= 0.0 or radius_b <= 0.0:
        return 0.0
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return math.pi * min(radius_a, radius_b) ** 2

    alpha = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (distance * distance + radius_a * radius_a - radius_b * radius_b)
                / (2.0 * distance * radius_a),
            ),
        )
    )
    beta = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (distance * distance + radius_b * radius_b - radius_a * radius_a)
                / (2.0 * distance * radius_b),
            ),
        )
    )
    triangle = 0.5 * math.sqrt(
        max(
            0.0,
            (-distance + radius_a + radius_b)
            * (distance + radius_a - radius_b)
            * (distance - radius_a + radius_b)
            * (distance + radius_a + radius_b),
        )
    )
    return radius_a * radius_a * alpha + radius_b * radius_b * beta - triangle


def _terrain_bounds_xy(terrain_obj: Any) -> tuple[float, float, float, float]:
    coords = [vertex.co for vertex in terrain_obj.data.vertices]
    if not coords:
        return -0.5, 0.5, -0.5, 0.5
    return (
        min(float(coord.x) for coord in coords),
        max(float(coord.x) for coord in coords),
        min(float(coord.y) for coord in coords),
        max(float(coord.y) for coord in coords),
    )


def _canopy_cover_fraction(
    placements: list[ScatterPlacement],
    bounds: tuple[float, float, float, float],
    resolution: int,
) -> float:
    if not placements:
        return 0.0
    min_x, max_x, min_y, max_y = bounds
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    resolution = max(8, int(resolution))
    cell_size = max(width, height) / resolution
    spatial: dict[tuple[int, int], list[ScatterPlacement]] = {}
    max_radius = max(item.crown_radius_m for item in placements)
    for item in placements:
        key = (math.floor(item.x / cell_size), math.floor(item.y / cell_size))
        spatial.setdefault(key, []).append(item)

    covered = 0
    total = resolution * resolution
    reach = int(math.ceil(max_radius / cell_size)) + 1
    for ix in range(resolution):
        x_coord = min_x + (ix + 0.5) * width / resolution
        gx = math.floor(x_coord / cell_size)
        for iy in range(resolution):
            y_coord = min_y + (iy + 0.5) * height / resolution
            gy = math.floor(y_coord / cell_size)
            is_covered = False
            for nx in range(gx - reach, gx + reach + 1):
                for ny in range(gy - reach, gy + reach + 1):
                    for item in spatial.get((nx, ny), []):
                        dx = x_coord - item.x
                        dy = y_coord - item.y
                        if dx * dx + dy * dy <= item.crown_radius_m * item.crown_radius_m:
                            is_covered = True
                            break
                    if is_covered:
                        break
                if is_covered:
                    break
            if is_covered:
                covered += 1
    return float(covered) / float(total)


def validate_scatter_placements(
    terrain_obj: Any,
    placements: list[ScatterPlacement],
    *,
    target_tree_count: float,
    target_stem_density_per_m2: float,
    eligible_area_m2: float,
    density_field: DensityFieldResult,
    grid_cells: int,
    canopy_grid_resolution: int,
) -> ScatterValidationResult:
    count = len(placements)
    area_ha = max(1e-9, float(eligible_area_m2) / 10_000.0)
    target_trees_per_ha = max(0.0, float(target_stem_density_per_m2)) * 10_000.0
    actual_trees_per_ha = float(count) / area_ha

    nearest: list[float] = []
    overlap_pair_count = 0
    overlap_area = 0.0
    total_crown_area = sum(
        math.pi * item.crown_radius_m * item.crown_radius_m for item in placements
    )

    for index, item in enumerate(placements):
        best = None
        for other in placements[index + 1 :]:
            dx = item.x - other.x
            dy = item.y - other.y
            distance = math.sqrt(dx * dx + dy * dy)
            if best is None or distance < best:
                best = distance
            overlap = _circle_overlap_area(item.crown_radius_m, other.crown_radius_m, distance)
            if overlap > 0.0:
                overlap_pair_count += 1
                overlap_area += overlap
        if best is not None:
            nearest.append(best)

    if count > 1:
        reverse_nearest = []
        for index, item in enumerate(placements):
            best = float("inf")
            for other_index, other in enumerate(placements):
                if index == other_index:
                    continue
                dx = item.x - other.x
                dy = item.y - other.y
                best = min(best, math.sqrt(dx * dx + dy * dy))
            reverse_nearest.append(best)
        nearest = reverse_nearest

    min_x, max_x, min_y, max_y = _terrain_bounds_xy(terrain_obj)
    total_area = max(1e-9, (max_x - min_x) * (max_y - min_y))
    canopy_cover = _canopy_cover_fraction(
        placements,
        (min_x, max_x, min_y, max_y),
        canopy_grid_resolution,
    )

    grid_cells = max(1, int(grid_cells))
    grid_counts = [[0 for _x in range(grid_cells)] for _y in range(grid_cells)]
    grid_target_counts = [[0.0 for _x in range(grid_cells)] for _y in range(grid_cells)]
    width = max(1e-9, max_x - min_x)
    height = max(1e-9, max_y - min_y)
    for item in placements:
        gx = max(0, min(grid_cells - 1, int((item.x - min_x) / width * grid_cells)))
        gy = max(0, min(grid_cells - 1, int((item.y - min_y) / height * grid_cells)))
        grid_counts[gy][gx] += 1

    density_mean = max(1e-9, float(density_field.eligible_mean))
    target_density_scale = max(0.0, float(target_stem_density_per_m2)) / density_mean
    for face in terrain_obj.data.polygons:
        if face.index >= len(density_field.values):
            continue
        center = face.center
        gx = max(0, min(grid_cells - 1, int((float(center.x) - min_x) / width * grid_cells)))
        gy = max(0, min(grid_cells - 1, int((float(center.y) - min_y) / height * grid_cells)))
        expected_count = (
            target_density_scale
            * max(0.0, float(density_field.values[face.index]))
            * float(face.area)
        )
        grid_target_counts[gy][gx] += float(expected_count)

    cell_area_ha = (total_area / (grid_cells * grid_cells)) / 10_000.0
    grid_tpha = [
        [float(count_value) / max(1e-9, cell_area_ha) for count_value in row] for row in grid_counts
    ]
    grid_target_tpha = [
        [float(count_value) / max(1e-9, cell_area_ha) for count_value in row]
        for row in grid_target_counts
    ]
    grid_actual_to_target = [
        [
            float(actual) / float(target) if float(target) > 1e-9 else 0.0
            for actual, target in zip(actual_row, target_row)
        ]
        for actual_row, target_row in zip(grid_counts, grid_target_counts)
    ]

    return ScatterValidationResult(
        target_tree_count=float(target_tree_count),
        actual_tree_count=int(count),
        target_trees_per_ha=float(target_trees_per_ha),
        actual_trees_per_ha=float(actual_trees_per_ha),
        actual_to_target_count_ratio=(
            float(count) / float(target_tree_count) if float(target_tree_count) > 1e-9 else 0.0
        ),
        eligible_area_m2=float(eligible_area_m2),
        density_field_mean=float(density_field.eligible_mean),
        density_field_min=float(density_field.eligible_min),
        density_field_max=float(density_field.eligible_max),
        nearest_neighbor_min_m=float(min(nearest) if nearest else 0.0),
        nearest_neighbor_p05_m=float(_percentile(nearest, 0.05)),
        nearest_neighbor_mean_m=float(sum(nearest) / len(nearest) if nearest else 0.0),
        nearest_neighbor_median_m=float(_percentile(nearest, 0.50)),
        nearest_neighbor_p95_m=float(_percentile(nearest, 0.95)),
        crown_overlap_pair_count=int(overlap_pair_count),
        crown_overlap_area_sum_m2=float(overlap_area),
        crown_overlap_area_fraction_of_crown_area=(
            float(overlap_area) / float(total_crown_area) if total_crown_area > 1e-9 else 0.0
        ),
        canopy_cover_fraction_total=float(canopy_cover),
        canopy_cover_fraction_eligible_bbox=float(canopy_cover),
        density_grid_cells=int(grid_cells),
        density_grid_counts=grid_counts,
        density_grid_trees_per_ha=grid_tpha,
        density_grid_target_counts=grid_target_counts,
        density_grid_target_trees_per_ha=grid_target_tpha,
        density_grid_actual_to_target_ratio=grid_actual_to_target,
    )


def validation_to_dict(validation: ScatterValidationResult) -> dict[str, Any]:
    return asdict(validation)
