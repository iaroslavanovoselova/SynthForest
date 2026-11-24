from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import bpy

from ..config import TerrainConfig
from ..scene import create_terrain_object
from .buildings import add_sparse_context_buildings, collect_blocked_faces_by_buildings
from .geometry import (
    collect_blocked_faces_by_paths,
    collect_blocked_faces_for_rural_clearings,
    collect_blocked_faces_for_water,
    sample_polyline,
    split_polyline_by_clearance,
    strip_mesh_from_polyline,
)
from .resources import build_context_materials, context_value, load_profile_overrides
from .rules import pick_density_context_mode, pick_density_context_profile, scale_count_range

logger = logging.getLogger(__name__)
TREE_ELIGIBLE_ATTRIBUTE = "tree_eligible"


def _empty_feature_counts() -> dict[str, int]:
    return {
        "trails": 0,
        "trail_segments": 0,
        "rivers": 0,
        "buildings": 0,
        "rural_clearings": 0,
    }


@dataclass
class TerrainContextResult:
    terrain_obj: bpy.types.Object
    terrain_source_mode: str
    context_profile: str
    context_mode: str
    scatter_density_band: str = ""
    blocked_face_count: int = 0
    feature_counts: dict[str, int] = field(default_factory=_empty_feature_counts)


def _ensure_face_float_attribute(mesh: bpy.types.Mesh, name: str) -> bpy.types.Attribute:
    attribute = mesh.attributes.get(name)
    if attribute is None:
        attribute = mesh.attributes.new(name=name, type="FLOAT", domain="FACE")
    return attribute


def _set_terrain_tree_eligibility(
    terrain_obj: bpy.types.Object,
    blocked_face_indices: set[int],
) -> None:
    mesh = terrain_obj.data
    attribute = _ensure_face_float_attribute(mesh, TREE_ELIGIBLE_ATTRIBUTE)

    values = [1.0] * len(mesh.polygons)
    for face_index in blocked_face_indices:
        if 0 <= face_index < len(values):
            values[face_index] = 0.0

    attribute.data.foreach_set("value", values)


def _apply_context_geometry(
    scene: bpy.types.Scene,
    terrain_obj: bpy.types.Object,
    terrain_cfg: TerrainConfig,
    terrain_size: float,
    seed: int,
    mode: str,
    profile_overrides: dict[str, object],
) -> tuple[set[int], str, dict[str, int]]:
    rng = random.Random(seed + 7919)
    mode_name = str(mode).strip().lower()
    feature_counts = _empty_feature_counts()
    if mode_name == "none":
        logger.info("Context placement | mode='none' | trail=off | river=off | buildings=0")
        return set(), mode_name, feature_counts

    size_half = terrain_size * 0.5
    paths: list[list[tuple[float, float]]] = []
    widths: list[float] = []
    extra_margins: list[float] = []
    river_paths: list[list[tuple[float, float]]] = []
    river_widths: list[float] = []

    road_material, water_material = build_context_materials(
        terrain_cfg=terrain_cfg,
        rng=rng,
    )

    include_road = mode_name in {"trails_only", "river_and_trail", "sparse_rural_artifacts"}
    include_river = mode_name in {"river_only", "river_and_trail"}

    if mode_name == "sparse_rural_artifacts":
        include_river = rng.random() < 0.35

    road_widths: list[float] = []
    road_segment_count = 0
    if include_river:
        river_count_min, river_count_max = scale_count_range(
            int(context_value(terrain_cfg, profile_overrides, "river_count_min")),
            int(context_value(terrain_cfg, profile_overrides, "river_count_max")),
            terrain_size=terrain_size,
            reference_size=float(
                context_value(terrain_cfg, profile_overrides, "context_reference_size")
            ),
            scale_mode="linear",
            require_at_least_one=True,
        )
        river_count = rng.randint(river_count_min, river_count_max)
        for river_idx in range(river_count):
            river_points = sample_polyline(size_half, rng, point_count=6, kind="river")
            river_width = rng.uniform(
                float(context_value(terrain_cfg, profile_overrides, "river_width_min")),
                float(context_value(terrain_cfg, profile_overrides, "river_width_max")),
            )
            _, river_path = strip_mesh_from_polyline(
                scene=scene,
                terrain_obj=terrain_obj,
                points_xy=river_points,
                width=river_width,
                z_offset=0.03,
                surface_offset=float(
                    context_value(terrain_cfg, profile_overrides, "river_surface_offset")
                ),
                name="ContextRiver" if river_count == 1 else f"ContextRiver_{river_idx + 1:02d}",
                material=water_material,
                kind="river",
            )
            paths.append(river_path)
            widths.append(river_width)
            river_paths.append(river_path)
            river_widths.append(river_width)
            extra_margins.append(
                float(context_value(terrain_cfg, profile_overrides, "river_tree_buffer_extra"))
            )

    if include_road:
        trail_count_min, trail_count_max = scale_count_range(
            int(context_value(terrain_cfg, profile_overrides, "trail_count_min")),
            int(context_value(terrain_cfg, profile_overrides, "trail_count_max")),
            terrain_size=terrain_size,
            reference_size=float(
                context_value(terrain_cfg, profile_overrides, "context_reference_size")
            ),
            scale_mode="linear",
            require_at_least_one=True,
        )
        trail_count = rng.randint(trail_count_min, trail_count_max)
        for trail_idx in range(trail_count):
            road_width = rng.uniform(
                float(context_value(terrain_cfg, profile_overrides, "trail_width_min")),
                float(context_value(terrain_cfg, profile_overrides, "trail_width_max")),
            )
            road_points = sample_polyline(size_half, rng, point_count=5, kind="trail")
            road_segments: list[list[tuple[float, float]]] = [road_points]
            if river_paths and river_widths:
                # Break the trail only where it would geometrically overlap the river.
                # Keep the gap minimal so future bridge placement can connect the ends.
                break_clearance = max(
                    0.01,
                    0.5 * max(float(width) for width in river_widths) + 0.5 * float(road_width),
                )
                road_segments = split_polyline_by_clearance(
                    road_points,
                    river_paths,
                    clearance=break_clearance,
                    max_segment_length=max(0.25, road_width * 0.18),
                    min_points_per_segment=2,
                )
            if not road_segments:
                logger.warning(
                    "Skipping trail %d because river split removed the full path.", trail_idx + 1
                )
                continue

            created_any = False
            for segment_idx, road_segment in enumerate(road_segments):
                if len(road_segment) < 2:
                    continue
                _, road_path = strip_mesh_from_polyline(
                    scene=scene,
                    terrain_obj=terrain_obj,
                    points_xy=road_segment,
                    width=road_width,
                    z_offset=0.05,
                    surface_offset=float(
                        context_value(terrain_cfg, profile_overrides, "trail_surface_offset")
                    ),
                    name=f"ContextTrail_{trail_idx + 1:02d}_{segment_idx + 1:02d}",
                    material=road_material,
                    kind="trail",
                )
                paths.append(road_path)
                widths.append(road_width)
                extra_margins.append(
                    float(context_value(terrain_cfg, profile_overrides, "trail_tree_buffer_extra"))
                )
                created_any = True
                road_segment_count += 1
            if created_any:
                road_widths.append(road_width)

    building_footprints = add_sparse_context_buildings(
        scene=scene,
        terrain_obj=terrain_obj,
        terrain_cfg=terrain_cfg,
        profile_overrides=profile_overrides,
        terrain_size=terrain_size,
        paths=paths,
        widths=widths,
        rng=rng,
    )

    blocked = collect_blocked_faces_by_paths(
        terrain_obj=terrain_obj,
        paths=paths,
        widths=widths,
        margin=float(context_value(terrain_cfg, profile_overrides, "exclusion_margin")),
        extra_margins=extra_margins,
    )

    blocked.update(
        collect_blocked_faces_for_water(
            terrain_obj=terrain_obj,
            waterline_elevation=float(
                context_value(terrain_cfg, profile_overrides, "waterline_elevation")
            ),
        )
    )

    blocked.update(
        collect_blocked_faces_by_buildings(
            terrain_obj=terrain_obj,
            footprints=building_footprints,
            margin=float(
                context_value(terrain_cfg, profile_overrides, "building_tree_buffer_extra")
            ),
        )
    )

    clearing_count = 0
    if mode_name == "sparse_rural_artifacts":
        clearing_count_min, clearing_count_max = scale_count_range(
            int(context_value(terrain_cfg, profile_overrides, "rural_clearings_min")),
            int(context_value(terrain_cfg, profile_overrides, "rural_clearings_max")),
            terrain_size=terrain_size,
            reference_size=float(
                context_value(terrain_cfg, profile_overrides, "context_reference_size")
            ),
            scale_mode="area",
            require_at_least_one=False,
        )
        if clearing_count_max > 0:
            clearing_count = rng.randint(clearing_count_min, clearing_count_max)
            blocked.update(
                collect_blocked_faces_for_rural_clearings(
                    terrain_obj=terrain_obj,
                    rng=rng,
                    size_half=size_half,
                    clearing_count=clearing_count,
                    radius_min=float(
                        context_value(terrain_cfg, profile_overrides, "rural_clearing_radius_min")
                    ),
                    radius_max=float(
                        context_value(terrain_cfg, profile_overrides, "rural_clearing_radius_max")
                    ),
                )
            )

    road_text = "trail=off"
    if road_widths:
        road_text = (
            f"trail=on(count={len(road_widths)},"
            f" widths={min(road_widths):.2f}-{max(road_widths):.2f})"
        )
    if river_widths:
        river_text = (
            f"river=on(count={len(river_widths)},"
            f" widths={min(river_widths):.2f}-{max(river_widths):.2f})"
        )
    else:
        river_text = "river=off"
    logger.info(
        "Context placement | mode='%s' | %s | %s | buildings=%d",
        mode_name,
        road_text,
        river_text,
        len(building_footprints),
    )

    feature_counts.update(
        {
            "trails": len(road_widths),
            "trail_segments": road_segment_count,
            "rivers": len(river_widths),
            "buildings": len(building_footprints),
            "rural_clearings": clearing_count,
        }
    )
    return blocked, mode_name, feature_counts


def create_terrain_with_context(
    scene: bpy.types.Scene,
    terrain_cfg: TerrainConfig,
    terrain_size: float,
    seed: int,
    texture_image_path: Path | None,
    texture_gsd: float,
    density_band: str | None = None,
) -> TerrainContextResult:
    terrain_obj = create_terrain_object(
        scene=scene,
        size=terrain_size,
        subdivisions=terrain_cfg.subdivisions,
        color=tuple(terrain_cfg.color),
        texture_image_path=texture_image_path,
        texture_gsd=texture_gsd,
        texture_random_seed=seed,
    )

    source_mode = str(getattr(terrain_cfg, "terrain_source_mode", "native")).strip().lower()
    if source_mode not in {"native", "context_profile"}:
        logger.warning(
            "Unknown terrain.terrain_source_mode '%s'; falling back to native.",
            source_mode,
        )
        source_mode = "native"

    context_mode = "none"
    context_profile = ""
    blocked_faces: set[int] = set()
    feature_counts = _empty_feature_counts()
    if source_mode == "context_profile":
        context_profile = pick_density_context_profile(
            density_band,
            seed,
            getattr(terrain_cfg.context, "context_profile_weights_by_density_band", None),
        )
        logger.info(
            "Selected density context profile: '%s' (band=%s)",
            context_profile,
            str(density_band or "dense"),
        )
        profile_overrides = load_profile_overrides(terrain_cfg, context_profile)
        context_mode = pick_density_context_mode(
            density_band,
            seed,
            getattr(terrain_cfg.context, "context_mode_weights_by_density_band", None),
        )
        blocked_faces, context_mode, feature_counts = _apply_context_geometry(
            scene=scene,
            terrain_obj=terrain_obj,
            terrain_cfg=terrain_cfg,
            terrain_size=terrain_size,
            seed=seed,
            mode=context_mode,
            profile_overrides=profile_overrides,
        )
        _set_terrain_tree_eligibility(terrain_obj=terrain_obj, blocked_face_indices=blocked_faces)
        logger.info(
            "Applied context profile '%s' mode '%s' with %s blocked terrain faces.",
            context_profile,
            context_mode,
            len(blocked_faces),
        )
    else:
        _set_terrain_tree_eligibility(terrain_obj=terrain_obj, blocked_face_indices=set())

    return TerrainContextResult(
        terrain_obj=terrain_obj,
        terrain_source_mode=source_mode,
        context_profile=context_profile,
        context_mode=context_mode,
        scatter_density_band=str(density_band or ""),
        blocked_face_count=len(blocked_faces),
        feature_counts=feature_counts,
    )
