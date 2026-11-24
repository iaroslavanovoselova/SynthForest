from __future__ import annotations

import json
import logging
import math
import random
import re
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

import bpy
from mathutils import Vector
from omegaconf import DictConfig, OmegaConf

from synthforest.paths import project_root

from .config import (
    TreeScatterConfig,
    compute_density_patch_controls,
    get_available_forest_types,
    get_available_target_profiles,
    load_target_profile_forest_type_weights,
    load_tree_scatter_base_config,
    load_tree_scatter_config_for_forest_and_target_profile,
    sample_scatter_density,
    sample_scene_season,
)
from .constants import (
    DENSITY_FIELD_ATTRIBUTE_NAME,
    INSTANCE_ID_ATTRIBUTE_NAME,
    INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
    INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_X_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_Y_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_Z_ATTRIBUTE_NAME,
    INSTANCE_SCALE_X_ATTRIBUTE_NAME,
    INSTANCE_SCALE_Y_ATTRIBUTE_NAME,
    INSTANCE_SCALE_Z_ATTRIBUTE_NAME,
    INSTANCE_VARIANT_INDEX_ATTRIBUTE_NAME,
    VARIANT_ID_ATTRIBUTE_NAME,
)
from .geometry_nodes import build_precomputed_scatter_gn_tree
from .placement import (
    ScatterPlacement,
    TreeVariantTemplate,
    build_density_field_on_terrain,
    generate_crown_aware_poisson_placements,
    validate_scatter_placements,
    validation_to_dict,
)
from .postprocess import (
    compute_mask_statistics,
    export_instance_segmentation_coco,
    save_class_id_map_from_variant_id_map,
    save_smoothed_instance_mask,
)
from .rendering import (
    configure_for_density_map,
    configure_for_label_maps,
    configure_for_rgb,
    configure_for_rgb_height,
    configure_render_common,
    ensure_density_map_material,
    ensure_override_material,
    ensure_variant_id_material,
    setup_compositor_density_map,
    setup_compositor_label_maps,
    setup_compositor_rgb,
    setup_compositor_rgb_height,
    setup_density_map_view_layer,
    setup_instance_id_view_layer,
    setup_variant_id_view_layer,
)
from .run_summary import update_run_summary
from .scene import (
    create_camera,
    create_sun_light,
    find_layer_collection,
)
from .terrain_context import create_terrain_with_context, snap_context_buildings_to_terrain
from .tree_source import create_tree_collection_from_source

logger = logging.getLogger(__name__)
PROJECT_ROOT = project_root()
TERRAIN_TEXTURE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _resolve_project_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _normalize_texture_ref_list(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        return [item.lstrip("-").strip() for item in value.splitlines() if item.strip()]

    return []


def _load_biome_terrain_map(map_path: Path) -> dict[str, list[str]] | None:
    try:
        loaded = OmegaConf.to_container(OmegaConf.load(map_path), resolve=True)
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None
    mapping_root = loaded.get("biome_terrain_map", loaded)
    if not isinstance(mapping_root, dict):
        return None

    normalized: dict[str, list[str]] = {}
    for biome_key, terrain_value in mapping_root.items():
        biome_name = str(biome_key).strip().lower()
        texture_refs = _normalize_texture_ref_list(terrain_value)
        if biome_name and texture_refs:
            normalized[biome_name] = texture_refs
    return normalized


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")


def _is_texture_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TERRAIN_TEXTURE_EXTENSIONS


def _collect_texture_pool_candidates(
    *,
    texture_dir: Path,
    texture_refs: list[str],
) -> list[Path]:
    candidates: list[Path] = []
    for texture_ref in texture_refs:
        ref = texture_ref.strip()
        if not ref:
            continue

        if _looks_like_glob(ref):
            candidates.extend(path for path in texture_dir.glob(ref) if _is_texture_file(path))
            continue

        ref_path = Path(ref).expanduser()
        if not ref_path.is_absolute():
            ref_path = texture_dir / ref_path

        if _is_texture_file(ref_path):
            candidates.append(ref_path)
            continue

        if ref_path.is_dir():
            candidates.extend(path for path in ref_path.rglob("*") if _is_texture_file(path))

    return sorted(set(path.resolve() for path in candidates))


def _filter_texture_paths_by_season(paths: list[Path], season_name: str | None) -> list[Path]:
    """Return subset of `paths` whose filename tail encodes `season_name`.

    Filenames are expected to include a final token after '_' or '-' that may
    contain one or multiple hyphen-separated season tags, e.g.:
      'grass_strands-spring-summer-autumn.png'

    If no matches for `season_name` are found and `season_name` is not
    'summer', the function will attempt to return summer-matching paths. If
    still none, it returns the original list.
    """
    if not season_name:
        return paths
    season = season_name.lower()
    matched: list[Path] = []
    for p in paths:
        stem = p.stem.lower()
        m = re.search(r"[_\-]([a-z\-]+)$", stem)
        if not m:
            continue
        tail = m.group(1)
        parts = [part for part in tail.split("-") if part]
        if season in parts:
            matched.append(p)
    if matched:
        return matched
    if season != "summer":
        summer_matched = _filter_texture_paths_by_season(paths, "summer")
        if summer_matched:
            return summer_matched
    return paths


def _choose_terrain_texture_path(
    cfg: TreeScatterConfig, season: str | None = None
) -> tuple[Path | None, str | None]:
    if not bool(getattr(cfg.terrain, "use_texture", True)):
        return None, None

    texture_dir = _resolve_project_path(
        str(getattr(cfg.terrain, "texture_dir", "data/terrain_textures/botaniq_particles"))
    )
    texture_file = getattr(cfg.terrain, "texture_file", None)

    if texture_file:
        texture_path = Path(str(texture_file)).expanduser()
        if not texture_path.is_absolute():
            texture_path = texture_dir / texture_path
        texture_path = texture_path.resolve()
        if texture_path.is_file():
            return texture_path, texture_path.parent.name

        logger.warning(
            "Configured terrain.texture_file does not exist: %s. Falling back to terrain color.",
            texture_path,
        )
        return None, None

    if not texture_dir.is_dir():
        logger.warning(
            "Terrain texture directory does not exist: %s. Falling back to terrain color.",
            texture_dir,
        )
        return None, None

    biome_terrain_map_path = _resolve_project_path(
        str(
            getattr(
                cfg.terrain,
                "biome_terrain_map_path",
                str(Path(__file__).resolve().parent / "configs" / "biome_terrain_map.yaml"),
            )
        )
    )

    forest_type = str(getattr(cfg.forest, "type", "")).strip().lower()
    requested_season = str(season or "").strip().lower()
    if biome_terrain_map_path.is_file():
        biome_map = _load_biome_terrain_map(biome_terrain_map_path)
        if not biome_map:
            logger.warning("Failed to load biome terrain map: %s", biome_terrain_map_path)
            return None, None
        texture_refs = biome_map.get(forest_type, [])
        candidates = _collect_texture_pool_candidates(
            texture_dir=texture_dir,
            texture_refs=texture_refs,
        )
        if candidates:
            # Filter candidates by requested season if filenames encode seasons at the end.
            filtered = _filter_texture_paths_by_season(candidates, requested_season)
            rng = random.Random(int(cfg.seed))
            selected_path = rng.choice(filtered)
            selected_type = f"botaniq_particles:{forest_type}"
            logger.info(
                "Selected Botaniq particle terrain texture for forest '%s' from biome pool '%s': %s",
                forest_type,
                selected_type,
                selected_path,
            )
            return selected_path, selected_type

        if texture_refs:
            logger.warning(
                "No Botaniq particle texture files found in mapped pool for forest '%s': %s",
                forest_type,
                ", ".join(texture_refs),
            )
        else:
            logger.warning(
                "No Botaniq particle texture mapping found for forest '%s' in %s",
                forest_type,
                biome_terrain_map_path,
            )
        return None, None
    else:
        logger.warning("Biome terrain map file not found: %s", biome_terrain_map_path)

    fallback_candidates = sorted(path for path in texture_dir.rglob("*") if _is_texture_file(path))
    if fallback_candidates:
        # Filter fallback candidates by season if requested
        filtered_fallback = _filter_texture_paths_by_season(fallback_candidates, requested_season)
        rng = random.Random(int(cfg.seed))
        selected_path = rng.choice(filtered_fallback)
        selected_type = selected_path.parent.name
        logger.info(
            "Selected terrain texture from fallback texture directory scan: %s",
            selected_path,
        )
        return selected_path, selected_type

    logger.warning("Falling back to terrain color.")
    return None, None


def _object_crown_diameter_m(obj: bpy.types.Object) -> float:
    stored = obj.get("realized_crown_diameter_m")
    try:
        if stored is not None:
            return max(0.0, float(stored))
    except Exception:
        pass

    try:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    except Exception:
        corners = []
    if not corners:
        return 0.0

    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    return max(0.0, float(max(max(xs) - min(xs), max(ys) - min(ys))))


def _collection_crown_diameter_stats(tree_collection: bpy.types.Collection) -> dict[str, float]:
    diameters: list[float] = []
    for obj in tree_collection.objects:
        if obj.type != "MESH":
            continue
        diameter = _object_crown_diameter_m(obj)
        if diameter > 0.0:
            diameters.append(diameter)

    if not diameters:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
        }

    diameters_sorted = sorted(diameters)
    mid = len(diameters_sorted) // 2
    if len(diameters_sorted) % 2 == 0:
        median = 0.5 * (diameters_sorted[mid - 1] + diameters_sorted[mid])
    else:
        median = diameters_sorted[mid]

    return {
        "mean": float(sum(diameters_sorted) / len(diameters_sorted)),
        "median": float(median),
        "max": float(diameters_sorted[-1]),
        "mean_circle_area": float(
            sum(math.pi * (diameter * 0.5) ** 2 for diameter in diameters_sorted)
            / len(diameters_sorted)
        ),
    }


def _tree_variant_templates_from_collection(
    tree_collection: bpy.types.Collection,
) -> list[TreeVariantTemplate]:
    templates: list[TreeVariantTemplate] = []
    for index, obj in enumerate(tree_collection.objects):
        if obj.type != "MESH":
            continue
        diameter = _object_crown_diameter_m(obj)
        if diameter <= 0.0:
            continue
        preset_name = str(obj.get("preset_name", obj.name))
        templates.append(
            TreeVariantTemplate(
                variant_index=int(index),
                variant_id=int(index + 1),
                crown_diameter_m=float(diameter),
                preset_name=preset_name,
            )
        )
    return templates


def _set_point_attribute(
    mesh: bpy.types.Mesh,
    name: str,
    data_type: str,
    values: list[float] | list[int],
) -> None:
    attribute = mesh.attributes.get(name)
    if attribute is None:
        attribute = mesh.attributes.new(name=name, type=data_type, domain="POINT")
    attribute.data.foreach_set("value", values)


def _create_precomputed_scatter_points_object(
    scene: bpy.types.Scene,
    placements: list[ScatterPlacement],
    variant_attribute_name: str,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("TreeScatterPrecomputedPointsMesh")
    mesh.from_pydata(
        [(placement.x, placement.y, placement.z) for placement in placements],
        [],
        [],
    )
    mesh.update()

    _set_point_attribute(
        mesh, INSTANCE_ID_ATTRIBUTE_NAME, "INT", [p.instance_id for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_VARIANT_INDEX_ATTRIBUTE_NAME, "INT", [p.variant_index for p in placements]
    )
    _set_point_attribute(
        mesh, variant_attribute_name, "FLOAT", [float(p.variant_id) for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_SCALE_X_ATTRIBUTE_NAME, "FLOAT", [p.scale_x for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_SCALE_Y_ATTRIBUTE_NAME, "FLOAT", [p.scale_y for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_SCALE_Z_ATTRIBUTE_NAME, "FLOAT", [p.scale_z for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_ROTATION_X_ATTRIBUTE_NAME, "FLOAT", [p.rotation_x for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_ROTATION_Y_ATTRIBUTE_NAME, "FLOAT", [p.rotation_y for p in placements]
    )
    _set_point_attribute(
        mesh, INSTANCE_ROTATION_Z_ATTRIBUTE_NAME, "FLOAT", [p.rotation_z for p in placements]
    )
    _set_point_attribute(
        mesh,
        INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
        "FLOAT",
        [p.leaf_hue_shift for p in placements],
    )
    _set_point_attribute(
        mesh,
        INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
        "FLOAT",
        [p.leaf_brightness for p in placements],
    )

    obj = bpy.data.objects.new("TreeScatterPrecomputedPoints", mesh)
    obj.hide_render = True
    obj.display_type = "WIRE"
    scene.collection.objects.link(obj)
    return obj


def _write_scatter_validation_artifacts(
    output_dirs: dict[str, Path],
    suffix: str,
    validation: object,
) -> Path:
    validation_path = output_dirs["summaries"] / f"scatter_validation{suffix}.json"
    validation_path.write_text(
        json.dumps(validation_to_dict(validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Scatter validation: %s", validation_path)
    return validation_path


def _format_mask_postprocess_summary(diagnostics: dict[str, object]) -> list[str]:
    if not diagnostics:
        return []

    growth = diagnostics.get("area_growth_percentiles", {})
    if not isinstance(growth, dict):
        growth = {}
    params = diagnostics.get("params", {})
    if not isinstance(params, dict):
        params = {}

    def as_int(name: str) -> int:
        try:
            return int(diagnostics.get(name, 0))
        except Exception:
            return 0

    def as_float_from(source: dict[object, object], name: str) -> float:
        try:
            return float(source.get(name, 0.0))
        except Exception:
            return 0.0

    lines = [
        f"mask_postprocess_skipped: {as_int('postprocess_skipped')}",
        f"mask_postprocess_raw_instance_count: {as_int('raw_instance_count')}",
        f"mask_postprocess_cleaned_instance_count: {as_int('cleaned_instance_count')}",
        f"mask_postprocess_dropped_instance_count: {as_int('dropped_instance_count')}",
        f"mask_postprocess_dropped_small_visible_count: {as_int('dropped_small_visible_count')}",
        f"mask_postprocess_dropped_small_final_count: {as_int('dropped_small_final_count')}",
        f"mask_postprocess_dropped_dark_crown_count: {as_int('dropped_dark_crown_count')}",
        f"mask_postprocess_dropped_low_compactness_count: {as_int('dropped_low_compactness_count')}",
        f"mask_postprocess_raw_visible_conflict_px: {as_int('raw_visible_conflict_px')}",
        f"mask_postprocess_raw_visible_reassigned_px: {as_int('raw_visible_reassigned_px')}",
        f"mask_postprocess_raw_visible_removed_px: {as_int('raw_visible_removed_px')}",
        f"mask_postprocess_background_added_px: {as_int('background_added_px')}",
        f"mask_postprocess_tiny_component_removed_px: {as_int('tiny_component_removed_px')}",
        f"mask_postprocess_boundary_regularized_instance_count: {as_int('boundary_regularized_instance_count')}",
        f"mask_postprocess_boundary_multicomponent_instance_count: {as_int('boundary_multicomponent_instance_count')}",
        f"mask_postprocess_boundary_component_merge_count: {as_int('boundary_component_merge_count')}",
        f"mask_postprocess_boundary_largest_component_only_count: {as_int('boundary_largest_component_only_count')}",
        f"mask_postprocess_boundary_component_removed_px: {as_int('boundary_component_removed_px')}",
        f"mask_postprocess_boundary_reassigned_px: {as_int('boundary_reassigned_px')}",
        f"mask_postprocess_boundary_added_px: {as_int('boundary_added_px')}",
        f"mask_postprocess_boundary_removed_px: {as_int('boundary_removed_px')}",
        f"mask_postprocess_nested_island_parent_count: {as_int('nested_island_parent_count')}",
        f"mask_postprocess_nested_island_hole_count: {as_int('nested_island_hole_count')}",
        f"mask_postprocess_nested_island_absorbed_component_count: {as_int('nested_island_absorbed_component_count')}",
        f"mask_postprocess_nested_island_absorbed_px: {as_int('nested_island_absorbed_px')}",
        f"mask_postprocess_nested_island_filled_background_px: {as_int('nested_island_filled_background_px')}",
        f"mask_postprocess_nested_island_skipped_large_hole_count: {as_int('nested_island_skipped_large_hole_count')}",
        f"mask_postprocess_nested_island_skipped_large_component_count: {as_int('nested_island_skipped_large_component_count')}",
        f"mask_postprocess_final_topology_parent_count: {as_int('final_topology_parent_count')}",
        f"mask_postprocess_final_topology_hole_count: {as_int('final_topology_hole_count')}",
        f"mask_postprocess_final_topology_multicomponent_count: {as_int('final_topology_multicomponent_count')}",
        f"mask_postprocess_final_topology_component_merge_count: {as_int('final_topology_component_merge_count')}",
        f"mask_postprocess_final_topology_absorbed_component_count: {as_int('final_topology_absorbed_component_count')}",
        f"mask_postprocess_final_topology_absorbed_px: {as_int('final_topology_absorbed_px')}",
        f"mask_postprocess_final_topology_filled_background_px: {as_int('final_topology_filled_background_px')}",
        f"mask_postprocess_final_topology_removed_component_count: {as_int('final_topology_removed_component_count')}",
        f"mask_postprocess_final_topology_removed_px: {as_int('final_topology_removed_px')}",
        f"mask_postprocess_final_topology_skipped_large_hole_count: {as_int('final_topology_skipped_large_hole_count')}",
        f"mask_postprocess_final_topology_skipped_large_component_count: {as_int('final_topology_skipped_large_component_count')}",
        f"mask_postprocess_final_min_area_removed_count: {as_int('final_min_area_removed_count')}",
        f"mask_postprocess_final_min_area_removed_px: {as_int('final_min_area_removed_px')}",
        f"mask_postprocess_ordered_height_map_used: {as_int('ordered_height_map_used')}",
        f"mask_postprocess_ordered_rgb_brightness_used: {as_int('ordered_rgb_brightness_used')}",
        f"mask_postprocess_ordered_candidate_count: {as_int('ordered_candidate_count')}",
        f"mask_postprocess_ordered_placed_count: {as_int('ordered_placed_count')}",
        f"mask_postprocess_ordered_absorbed_instance_count: {as_int('ordered_absorbed_instance_count')}",
        f"mask_postprocess_ordered_absorbed_px: {as_int('ordered_absorbed_px')}",
        f"mask_postprocess_ordered_reserved_px: {as_int('ordered_reserved_px')}",
        f"mask_postprocess_ordered_dropped_occluded_count: {as_int('ordered_dropped_occluded_count')}",
        f"mask_postprocess_ordered_dropped_component_count: {as_int('ordered_dropped_component_count')}",
        f"mask_postprocess_ordered_component_removed_px: {as_int('ordered_component_removed_px')}",
        f"mask_postprocess_ordered_background_hole_filled_px: {as_int('ordered_background_hole_filled_px')}",
        f"mask_postprocess_ordered_reclaimed_visible_px: {as_int('ordered_reclaimed_visible_px')}",
        f"mask_postprocess_ordered_rescued_instance_count: {as_int('ordered_rescued_instance_count')}",
        f"mask_postprocess_ordered_rescued_px: {as_int('ordered_rescued_px')}",
        f"mask_postprocess_ordered_rescue_reassigned_px: {as_int('ordered_rescue_reassigned_px')}",
        f"mask_postprocess_rescued_low_compactness_removed_count: {as_int('rescued_low_compactness_removed_count')}",
        f"mask_postprocess_rescued_low_compactness_removed_px: {as_int('rescued_low_compactness_removed_px')}",
        f"mask_postprocess_rescued_low_compactness_repaired_px: {as_int('rescued_low_compactness_repaired_px')}",
        f"mask_postprocess_rescued_low_compactness_unrepaired_px: {as_int('rescued_low_compactness_unrepaired_px')}",
        f"mask_postprocess_ordered_priority_height_p50: {float(diagnostics.get('ordered_priority_height_p50', 0.0)):.6f}",
        f"mask_postprocess_ordered_priority_height_p90: {float(diagnostics.get('ordered_priority_height_p90', 0.0)):.6f}",
        f"mask_postprocess_area_growth_p50: {as_float_from(growth, 'p50'):.6f}",
        f"mask_postprocess_area_growth_p90: {as_float_from(growth, 'p90'):.6f}",
        f"mask_postprocess_area_growth_p95: {as_float_from(growth, 'p95'):.6f}",
        f"mask_postprocess_area_growth_p99: {as_float_from(growth, 'p99'):.6f}",
    ]
    skip_reason = diagnostics.get("skip_reason")
    if skip_reason:
        lines.append(f"mask_postprocess_skip_reason: {skip_reason}")
    if params:
        lines.extend(
            [
                f"mask_postprocess_min_visible_px: {as_float_from(params, 'min_visible_px'):.0f}",
                f"mask_postprocess_min_final_px: {as_float_from(params, 'min_final_px'):.0f}",
                f"mask_postprocess_min_visible_brightness: {as_float_from(params, 'min_visible_brightness'):.6f}",
                f"mask_postprocess_visible_brightness_percentile: {as_float_from(params, 'visible_brightness_percentile'):.6f}",
                f"mask_postprocess_min_component_px: {as_float_from(params, 'min_component_px'):.0f}",
                f"mask_postprocess_gap_fill_radius_px: {as_float_from(params, 'gap_fill_radius_px'):.0f}",
                f"mask_postprocess_sliver_px: {as_float_from(params, 'sliver_px'):.0f}",
                f"mask_postprocess_sliver_min_bbox_axis_ratio: {as_float_from(params, 'sliver_min_bbox_axis_ratio'):.6f}",
                f"mask_postprocess_boundary_regularization_min_px: {as_float_from(params, 'boundary_regularization_min_px'):.0f}",
                f"mask_postprocess_boundary_smooth_radius_px: {as_float_from(params, 'boundary_smooth_radius_px'):.0f}",
                f"mask_postprocess_boundary_core_erosion_px: {as_float_from(params, 'boundary_core_erosion_px'):.0f}",
                f"mask_postprocess_boundary_component_merge_radius_px: {as_float_from(params, 'boundary_component_merge_radius_px'):.0f}",
                f"mask_postprocess_boundary_component_min_px: {as_float_from(params, 'boundary_component_min_px'):.0f}",
                f"mask_postprocess_boundary_component_keep_fraction: {as_float_from(params, 'boundary_component_keep_fraction'):.6f}",
                f"mask_postprocess_boundary_component_merge_max_area_growth_fraction: {as_float_from(params, 'boundary_component_merge_max_area_growth_fraction'):.6f}",
                f"mask_postprocess_boundary_max_area_growth_fraction: {as_float_from(params, 'boundary_max_area_growth_fraction'):.6f}",
                f"mask_postprocess_nested_island_parent_min_px: {as_float_from(params, 'nested_island_parent_min_px'):.0f}",
                f"mask_postprocess_nested_island_max_hole_area_fraction: {as_float_from(params, 'nested_island_max_hole_area_fraction'):.6f}",
                f"mask_postprocess_nested_island_absorb_max_area_fraction: {as_float_from(params, 'nested_island_absorb_max_area_fraction'):.6f}",
                f"mask_postprocess_nested_island_absorb_max_px: {as_float_from(params, 'nested_island_absorb_max_px'):.0f}",
                f"mask_postprocess_final_topology_min_px: {as_float_from(params, 'final_topology_min_px'):.0f}",
                f"mask_postprocess_ordered_height_percentile: {as_float_from(params, 'ordered_height_percentile'):.6f}",
                f"mask_postprocess_ordered_area_priority_weight: {as_float_from(params, 'ordered_area_priority_weight'):.6f}",
                f"mask_postprocess_ordered_absorb_height_margin_m: {as_float_from(params, 'ordered_absorb_height_margin_m'):.6f}",
                f"mask_postprocess_ordered_absorb_overlap_fraction: {as_float_from(params, 'ordered_absorb_overlap_fraction'):.6f}",
                f"mask_postprocess_ordered_absorb_max_px: {as_float_from(params, 'ordered_absorb_max_px'):.0f}",
                f"mask_postprocess_ordered_min_retained_visible_fraction: {as_float_from(params, 'ordered_min_retained_visible_fraction'):.6f}",
                f"mask_postprocess_ordered_component_keep_fraction: {as_float_from(params, 'ordered_component_keep_fraction'):.6f}",
                f"mask_postprocess_ordered_outer_envelope_min_px: {as_float_from(params, 'ordered_outer_envelope_min_px'):.0f}",
                f"mask_postprocess_ordered_outer_envelope_radius_px: {as_float_from(params, 'ordered_outer_envelope_radius_px'):.0f}",
                f"mask_postprocess_ordered_outer_envelope_max_growth_fraction: {as_float_from(params, 'ordered_outer_envelope_max_growth_fraction'):.6f}",
                f"mask_postprocess_ordered_reclaim_height_margin_m: {as_float_from(params, 'ordered_reclaim_height_margin_m'):.6f}",
                f"mask_postprocess_ordered_rescue_min_visible_fraction: {as_float_from(params, 'ordered_rescue_min_visible_fraction'):.6f}",
            ]
        )
    return lines


def _terrain_face_eligibility_area_m2(
    terrain_obj: bpy.types.Object, attribute_name: str = "tree_eligible"
) -> tuple[float, float]:
    mesh = terrain_obj.data
    if mesh is None:
        return 0.0, 0.0

    total_area = 0.0
    eligible_area = 0.0
    attribute = (
        getattr(mesh, "attributes", {}).get(attribute_name) if hasattr(mesh, "attributes") else None
    )

    for poly in mesh.polygons:
        face_area = float(poly.area)
        total_area += face_area
        eligible = 1.0
        if attribute is not None and poly.index < len(attribute.data):
            try:
                eligible = float(attribute.data[poly.index].value)
            except Exception:
                eligible = 1.0
        if eligible > 0.5:
            eligible_area += face_area

    return total_area, eligible_area


def _compute_calibrated_scatter_parameters(
    cfg: TreeScatterConfig,
    terrain_obj: bpy.types.Object,
    tree_collection: bpy.types.Collection,
    scatter_density_relative: float,
) -> dict[str, float]:
    total_area_m2, eligible_area_m2 = _terrain_face_eligibility_area_m2(terrain_obj)
    crown_stats = _collection_crown_diameter_stats(tree_collection)

    crown_area_fill_factor = max(0.05, float(getattr(cfg.scatter, "crown_area_fill_factor", 0.72)))
    canopy_overlap_efficiency = max(
        0.05,
        float(getattr(cfg.scatter, "canopy_overlap_efficiency", 0.82)),
    )
    canopy_closure_at_full_density = max(
        0.01,
        min(0.995, float(getattr(cfg.scatter, "canopy_closure_at_full_density", 0.95))),
    )
    canopy_closure_gamma = max(
        0.1,
        float(getattr(cfg.scatter, "canopy_closure_gamma", 1.0)),
    )
    packing_efficiency = max(
        0.05,
        float(getattr(cfg.scatter, "poisson_packing_efficiency", 0.85)),
    )
    instance_scale_min = max(0.01, float(getattr(cfg.scatter, "tree_scale_min", 1.0)))
    instance_scale_max = max(instance_scale_min, float(getattr(cfg.scatter, "tree_scale_max", 1.0)))
    instance_scale_area_factor = (
        instance_scale_min**2 + instance_scale_min * instance_scale_max + instance_scale_max**2
    ) / 3.0
    effective_projected_crown_area_m2 = max(
        1e-6,
        crown_stats["mean_circle_area"]
        * instance_scale_area_factor
        * crown_area_fill_factor
        * canopy_overlap_efficiency,
    )
    density_slider = max(0.0, min(1.0, float(scatter_density_relative)))
    density_patch_controls = compute_density_patch_controls(cfg.scatter, density_slider)
    density_threshold = density_patch_controls["density_threshold"]
    density_variation_strength = density_patch_controls["density_variation_strength"]
    density_cluster_exponent = density_patch_controls["density_cluster_exponent"]
    # Approximate E[max(0, (noise - threshold) / (1 - threshold)) ** exponent]
    # with a uniform-noise assumption. This keeps the expected tree count stable
    # when the density field is made patchier for natural clumping.
    density_map_mean_factor = max(
        1e-6,
        (1.0 - density_threshold) / (density_cluster_exponent + 1.0),
    )
    density_mask_mean_factor = max(
        1e-6,
        (1.0 - density_variation_strength) + density_variation_strength * density_map_mean_factor,
    )
    target_canopy_coverage = canopy_closure_at_full_density * (density_slider**canopy_closure_gamma)
    target_crown_area_index = -math.log(max(1e-6, 1.0 - target_canopy_coverage))
    target_stem_density_for_coverage_per_m2 = (
        target_crown_area_index / effective_projected_crown_area_m2
    )

    required_stem_spacing_for_coverage_m = math.sqrt(
        packing_efficiency
        * 2.0
        / (math.sqrt(3.0) * max(1e-9, target_stem_density_for_coverage_per_m2))
    )
    stem_spacing_floor_m = max(
        0.1,
        crown_stats["mean"]
        * max(0.0, float(getattr(cfg.scatter, "stem_spacing_crown_fraction_floor", 0.45))),
    )

    # Keep Distance Min as an ecological collision limit, not as the spacing
    # implied by the average stand density. The density field handles patchy
    # openings and clusters; using the uniform spacing here makes sparse stands
    # look like plantations.
    applied_stem_spacing_m = stem_spacing_floor_m

    applied_spacing_cap_density_per_m2 = (
        packing_efficiency * 2.0 / (math.sqrt(3.0) * (applied_stem_spacing_m**2))
    )
    applied_stem_density_for_coverage_per_m2 = min(
        target_stem_density_for_coverage_per_m2,
        applied_spacing_cap_density_per_m2,
    )
    applied_crown_area_index = (
        applied_stem_density_for_coverage_per_m2 * effective_projected_crown_area_m2
    )
    applied_canopy_coverage = 1.0 - math.exp(-applied_crown_area_index)
    scatter_density_for_distribution = (
        applied_stem_density_for_coverage_per_m2 / density_mask_mean_factor
    )
    estimated_tree_count_target = applied_stem_density_for_coverage_per_m2 * eligible_area_m2

    return {
        "eligible_area_m2": float(eligible_area_m2),
        "total_area_m2": float(total_area_m2),
        "eligible_fraction": (
            float(eligible_area_m2 / total_area_m2) if total_area_m2 > 1e-9 else 0.0
        ),
        "crown_diameter_mean_m": float(crown_stats["mean"]),
        "crown_diameter_median_m": float(crown_stats["median"]),
        "crown_diameter_max_m": float(crown_stats["max"]),
        "crown_projected_area_mean_m2": float(crown_stats["mean_circle_area"]),
        "instance_scale_area_factor": float(instance_scale_area_factor),
        "effective_projected_crown_area_m2": float(effective_projected_crown_area_m2),
        "stem_spacing_floor_m": float(stem_spacing_floor_m),
        "estimated_tree_count_target": float(estimated_tree_count_target),
        "density_slider": float(density_slider),
        "canopy_closure_at_full_density": float(canopy_closure_at_full_density),
        "target_canopy_coverage": float(target_canopy_coverage),
        "target_crown_area_index": float(target_crown_area_index),
        "applied_crown_area_index": float(applied_crown_area_index),
        "density_mask_mean_factor": float(density_mask_mean_factor),
        "density_map_mean_factor": float(density_map_mean_factor),
        "density_cluster_exponent": float(density_cluster_exponent),
        "density_variation_strength_effective": float(density_variation_strength),
        "density_threshold_effective": float(density_threshold),
        "density_fade": float(density_patch_controls["density_fade"]),
        "target_stem_density_for_coverage_per_m2": float(target_stem_density_for_coverage_per_m2),
        "required_stem_spacing_for_coverage_m": float(required_stem_spacing_for_coverage_m),
        "applied_stem_spacing_m": float(applied_stem_spacing_m),
        "applied_spacing_cap_density_per_m2": float(applied_spacing_cap_density_per_m2),
        "applied_stem_density_for_coverage_per_m2": float(applied_stem_density_for_coverage_per_m2),
        "applied_canopy_coverage": float(applied_canopy_coverage),
        "scatter_density_for_distribution": float(scatter_density_for_distribution),
    }


def _compute_terrain_size_from_gsd(
    resolution_x: int,
    resolution_y: int,
    gsd_at_nadir: float,
    fallback_size: float,
) -> float:
    gsd = max(1e-6, float(gsd_at_nadir))
    width_world = max(1, int(resolution_x)) * gsd
    height_world = max(1, int(resolution_y)) * gsd
    return max(float(fallback_size), float(width_world), float(height_world))


def _georeference_rendered_tiff(
    tiff_path: Path,
    *,
    resolution_x: int,
    resolution_y: int,
    gsd_at_nadir: float,
) -> bool:
    """Stamp a rendered TIFF with the synthetic scene pixel size.

    Blender writes ordinary image TIFFs, so GDAL sees pixel coordinates
    0..width/height unless we add a transform after rendering.
    """
    if not tiff_path.exists():
        return False

    gdal_edit = shutil.which("gdal_edit.py")
    if gdal_edit is None:
        logger.warning("Cannot georeference %s: gdal_edit.py not found on PATH.", tiff_path)
        return False

    width_world = max(1, int(resolution_x)) * max(1e-6, float(gsd_at_nadir))
    height_world = max(1, int(resolution_y)) * max(1e-6, float(gsd_at_nadir))
    left = -0.5 * width_world
    right = 0.5 * width_world
    top = 0.5 * height_world
    bottom = -0.5 * height_world

    cmd = [
        gdal_edit,
        "-a_ullr",
        f"{left:.12g}",
        f"{top:.12g}",
        f"{right:.12g}",
        f"{bottom:.12g}",
        "-a_srs",
        'LOCAL_CS["Synthetic metric scene"]',
        "-mo",
        f"GSD_AT_NADIR={max(1e-6, float(gsd_at_nadir)):.12g}",
        "-mo",
        f"SYNTHETIC_PIXEL_SIZE_X={width_world / max(1, int(resolution_x)):.12g}",
        "-mo",
        f"SYNTHETIC_PIXEL_SIZE_Y={height_world / max(1, int(resolution_y)):.12g}",
        str(tiff_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Stamped GeoTIFF GSD metadata: %s (%.6f m/px)", tiff_path, float(gsd_at_nadir))
        return True
    except Exception as exc:
        logger.warning("Failed to stamp GeoTIFF GSD metadata for %s. Reason: %s", tiff_path, exc)
        return False


def _runtime_config_payload(cfg: object) -> dict[str, object]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True)
        return payload if isinstance(payload, dict) else {"value": payload}
    return {"value": cfg}


def _save_yaml_config(output_path: Path, cfg: object) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _runtime_config_payload(cfg)
    OmegaConf.save(config=OmegaConf.create(payload), f=output_path)
    return output_path


def _save_run_configs(
    root_output_dir: Path,
    requested_cfg: DictConfig,
    base_runtime_cfg: TreeScatterConfig,
) -> None:
    config_dir = root_output_dir / "config"
    requested_path = _save_yaml_config(config_dir / "requested.yaml", requested_cfg)
    base_path = _save_yaml_config(config_dir / "base_runtime.yaml", base_runtime_cfg)
    logger.info("Requested config saved to: %s", requested_path)
    logger.info("Base runtime config saved to: %s", base_path)


def _save_effective_scene_config(
    root_output_dir: Path,
    scene_cfg: TreeScatterConfig,
    scene_id: int,
    *,
    save_legacy_root_config: bool,
) -> None:
    config_path = _save_yaml_config(
        root_output_dir / "config" / "scenes" / f"scene_{int(scene_id):04d}.yaml",
        scene_cfg,
    )
    if save_legacy_root_config:
        legacy_path = _save_yaml_config(root_output_dir / "tree_scatter_scene.yaml", scene_cfg)
        logger.info("Effective first-scene config saved to: %s", legacy_path)
    logger.debug("Effective scene config saved to: %s", config_path)


def _purge_orphan_blender_data() -> None:
    try:
        purge = getattr(bpy.data, "orphans_purge", None)
        if purge is not None:
            try:
                purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
            except TypeError:
                purge()
            return
    except Exception:
        return


def _build_preset_category_mappings(
    variant_presets: list[str],
) -> tuple[dict[int, str], dict[int, int]]:
    preset_to_category_id: dict[str, int] = {}
    category_id_to_name: dict[int, str] = {}
    variant_id_to_category_id: dict[int, int] = {}

    for variant_id, preset_name in enumerate(variant_presets, start=1):
        category_id = preset_to_category_id.get(preset_name)
        if category_id is None:
            category_id = len(preset_to_category_id) + 1
            preset_to_category_id[preset_name] = category_id
            category_id_to_name[category_id] = preset_name
        variant_id_to_category_id[variant_id] = category_id

    return category_id_to_name, variant_id_to_category_id


def _compute_camera_angle_rad(camera_obj: bpy.types.Object) -> float:
    if camera_obj is None:
        return 0.0

    dx = -float(camera_obj.location.x)
    dy = -float(camera_obj.location.y)
    dz = -float(camera_obj.location.z)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-8:
        return 0.0

    dir_z = dz / norm
    dot = max(-1.0, min(1.0, -dir_z))
    return float(math.acos(dot))


def _create_flat_height_scene(
    main_scene: bpy.types.Scene,
    terrain_name: str,
) -> bpy.types.Scene | None:
    terrain_obj = main_scene.objects.get(terrain_name)
    if terrain_obj is None:
        for obj in main_scene.objects:
            if obj.type == "MESH" and (
                obj.name == terrain_name or obj.name.startswith(f"{terrain_name}.")
            ):
                terrain_obj = obj
                break

    if terrain_obj is None:
        logger.warning(
            "Flat height scene: terrain '%s' not found; skipping height map.",
            terrain_name,
        )
        return None

    flat_scene = None
    try:
        flat_scene = bpy.data.scenes.new("HeightFlatScene")
        flat_scene.frame_set(main_scene.frame_current)
        flat_scene.render.engine = main_scene.render.engine
        flat_scene.render.resolution_x = main_scene.render.resolution_x
        flat_scene.render.resolution_y = main_scene.render.resolution_y
        flat_scene.render.resolution_percentage = main_scene.render.resolution_percentage
        flat_scene.view_settings.view_transform = main_scene.view_settings.view_transform
        flat_scene.world = main_scene.world

        if hasattr(flat_scene.render, "use_persistent_data"):
            flat_scene.render.use_persistent_data = False
        if hasattr(flat_scene, "cycles") and hasattr(main_scene, "cycles"):
            flat_scene.cycles.device = main_scene.cycles.device
            flat_scene.cycles.samples = 1
            if hasattr(flat_scene.cycles, "use_denoising"):
                flat_scene.cycles.use_denoising = False
            if hasattr(flat_scene.cycles, "use_adaptive_sampling"):
                flat_scene.cycles.use_adaptive_sampling = False

        if main_scene.camera is not None:
            camera_obj = main_scene.camera.copy()
            camera_obj.data = main_scene.camera.data.copy()
            camera_obj.name = "HeightReferenceCamera"
            flat_scene.collection.objects.link(camera_obj)
            flat_scene.camera = camera_obj

        terrain_ref = terrain_obj.copy()
        terrain_ref.name = f"{terrain_obj.name}_HeightReference"
        flat_scene.collection.objects.link(terrain_ref)
        _zero_scatter_density_on_terrain(terrain_ref)

        _add_building_height_reference_pads(flat_scene, source_scene=main_scene)

        for view_layer in flat_scene.view_layers:
            if hasattr(view_layer, "use_pass_z"):
                view_layer.use_pass_z = True
            if hasattr(view_layer, "pass_alpha_threshold"):
                view_layer.pass_alpha_threshold = 0
    except Exception as exc:
        logger.warning("Flat height scene: failed to create lightweight terrain reference: %s", exc)
        if flat_scene is not None:
            try:
                bpy.data.scenes.remove(flat_scene)
            except Exception:
                pass
        return None

    if bpy.context.window is not None:
        bpy.context.window.scene = main_scene

    logger.debug("Height reference scene created as lightweight terrain-only scene.")
    return flat_scene


def _add_building_height_reference_pads(
    scene: bpy.types.Scene,
    *,
    source_scene: bpy.types.Scene | None = None,
) -> int:
    source = source_scene or scene
    created = 0
    for obj in list(source.objects):
        if obj.type != "MESH":
            continue
        if not str(obj.name).startswith("ContextBuilding_"):
            continue

        try:
            local_corners = list(obj.bound_box)
        except Exception:
            continue
        if not local_corners:
            continue

        half_w = max(0.1, max(abs(float(corner[0])) for corner in local_corners))
        half_d = max(0.1, max(abs(float(corner[1])) for corner in local_corners))
        pad_z = float(obj.get("height_reference_z", obj.location.z))
        margin = 0.05

        verts = [
            (-half_w - margin, -half_d - margin, 0.0),
            (half_w + margin, -half_d - margin, 0.0),
            (half_w + margin, half_d + margin, 0.0),
            (-half_w - margin, half_d + margin, 0.0),
        ]
        mesh = bpy.data.meshes.new(f"{obj.name}_HeightReferencePad_Mesh")
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()

        pad = bpy.data.objects.new(f"{obj.name}_HeightReferencePad", mesh)
        pad.location = (float(obj.location.x), float(obj.location.y), pad_z)
        pad.rotation_euler[2] = float(obj.rotation_euler[2])
        scene.collection.objects.link(pad)
        created += 1

    if created:
        logger.debug("Height reference scene: added %d building footprint pads.", created)
    return created


def _detach_compositor_scene_references(
    owner_scene: bpy.types.Scene | None,
    referenced_scene: bpy.types.Scene | None,
) -> None:
    if owner_scene is None or referenced_scene is None:
        return
    if not bool(getattr(owner_scene, "use_nodes", False)):
        return

    try:
        tree = getattr(owner_scene, "compositing_node_group", None) or owner_scene.node_tree
    except Exception:
        return
    if tree is None:
        return

    for node in list(tree.nodes):
        try:
            node_scene = getattr(node, "scene", None)
        except Exception:
            continue
        if node_scene == referenced_scene:
            try:
                tree.nodes.remove(node)
            except Exception:
                pass


def _clear_scene_compositor(scene: bpy.types.Scene | None) -> None:
    if scene is None or not bool(getattr(scene, "use_nodes", False)):
        return

    try:
        tree = getattr(scene, "compositing_node_group", None) or scene.node_tree
    except Exception:
        return
    if tree is None:
        return

    for node in list(tree.nodes):
        try:
            tree.nodes.remove(node)
        except Exception:
            pass
    try:
        scene.render.use_compositing = False
    except Exception:
        pass


def _remove_height_reference_scene(
    scene: bpy.types.Scene | None,
    *,
    owner_scene: bpy.types.Scene | None = None,
) -> None:
    if scene is None:
        return
    if owner_scene is not None:
        _detach_compositor_scene_references(owner_scene, scene)
    else:
        for candidate_scene in list(bpy.data.scenes):
            if candidate_scene != scene:
                _detach_compositor_scene_references(candidate_scene, scene)
    try:
        bpy.data.scenes.remove(scene)
    except Exception:
        pass
    _purge_orphan_blender_data()


def _set_scatter_density_on_terrain(
    terrain_obj: bpy.types.Object,
    value: float,
    *,
    warn: bool = True,
) -> bool:
    for modifier in terrain_obj.modifiers:
        if modifier.type != "NODES" or modifier.node_group is None:
            continue

        socket_identifiers = _geometry_node_input_identifiers(
            modifier.node_group,
            socket_name="Scatter Density",
        )
        socket_identifiers.extend(["Scatter Density", "Scatter Point Density"])

        for socket_identifier in socket_identifiers:
            try:
                modifier[socket_identifier] = float(value)
                logger.debug(
                    "Set %s.%s scatter density to %.3f.",
                    terrain_obj.name,
                    modifier.name,
                    float(value),
                )
                return True
            except Exception:
                continue
        if warn:
            logger.warning(
                "Could not find 'Scatter Density' input on terrain modifier '%s'.",
                modifier.name,
            )
    return False


def _zero_scatter_density_on_terrain(terrain_obj: bpy.types.Object) -> None:
    """Keep GN terrain relief in the height reference, but remove tree instances."""
    if not _set_scatter_density_on_terrain(terrain_obj, 0.0, warn=False):
        logger.warning(
            "Height reference scene: could not find 'Scatter Density' input on "
            "terrain modifiers; terrain relief is preserved but tree instances "
            "may remain in the reference depth pass."
        )


def _geometry_node_input_identifiers(
    node_group: bpy.types.GeometryNodeTree,
    socket_name: str,
) -> list[str]:
    identifiers: list[str] = []
    interface = getattr(node_group, "interface", None)
    if interface is None:
        return identifiers

    items = getattr(interface, "items_tree", None)
    if items is None:
        items = getattr(interface, "items", ())

    for item in items:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        if getattr(item, "name", None) != socket_name:
            continue
        identifier = getattr(item, "identifier", None)
        if identifier:
            identifiers.append(str(identifier))

    return identifiers


def _convert_height_exr_to_tiff(
    exr_path: Path,
    tiff_path: Path,
) -> bool:
    if not exr_path.exists():
        return False

    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return False

    image = None
    try:
        image = bpy.data.images.load(str(exr_path), check_existing=False)
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        height_data = pixels[:, :, 0][::-1]
        tiff_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(height_data, mode="F").save(str(tiff_path))
        return True
    except Exception:
        return False
    finally:
        if image is not None:
            try:
                bpy.data.images.remove(image)
            except Exception:
                pass


def _save_height_visualization_from_tiff(
    tiff_path: Path,
    vis_path: Path,
) -> bool:
    if not tiff_path.exists():
        return False

    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return False

    try:
        height_data = np.array(Image.open(tiff_path), dtype=np.float32)
        if height_data.ndim == 3:
            height_data = height_data[..., 0]

        finite = np.isfinite(height_data)
        if not finite.any():
            return False

        valid = finite & (height_data > 0.0)
        if valid.any():
            sample = height_data[valid]
        else:
            sample = height_data[finite]

        lo = float(np.percentile(sample, 1.0))
        hi = float(np.percentile(sample, 99.0))
        if hi <= lo:
            lo = float(np.min(sample))
            hi = float(np.max(sample))
        if hi <= lo:
            hi = lo + 1.0

        norm = (height_data - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        norm[~finite] = 0.0

        if valid.any():
            norm[~valid] = 0.0

        anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        anchors_rgb = np.array(
            [
                [68, 1, 84],
                [59, 82, 139],
                [33, 145, 140],
                [94, 201, 98],
                [253, 231, 37],
            ],
            dtype=np.float32,
        )

        r = np.interp(norm, anchors_x, anchors_rgb[:, 0])
        g = np.interp(norm, anchors_x, anchors_rgb[:, 1])
        b = np.interp(norm, anchors_x, anchors_rgb[:, 2])
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

        vis_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(str(vis_path))
        return True
    except Exception:
        return False


def _scene_id_suffix(scene_id: int) -> str:
    return f"{int(scene_id):04d}"


def _reset_main_scene(scene_name: str) -> bpy.types.Scene:
    try:
        bpy.ops.scene.new(type="NEW")
    except Exception as exc:
        raise RuntimeError(f"Failed to create a fresh scene: {exc}") from exc

    main_scene = bpy.context.scene
    main_scene.name = scene_name

    for existing_scene in list(bpy.data.scenes):
        if existing_scene != main_scene:
            try:
                bpy.data.scenes.remove(existing_scene)
            except Exception:
                continue

    _purge_orphan_blender_data()

    return main_scene


def _build_output_dirs(root_output_dir: Path) -> dict[str, Path]:
    output_dirs = {
        "images": root_output_dir / "images",
        "instance_id": root_output_dir / "instance_id_maps",
        "variant_id": root_output_dir / "variant_id_maps",
        "semantic_id": root_output_dir / "semantic_id_maps",
        "semantic_labels": root_output_dir / "semantic_id_labels",
        "smoothed_instance_id": root_output_dir / "instance_id_maps_smoothed",
        "density": root_output_dir / "density_maps",
        "height": root_output_dir / "height_maps",
        "height_vis": root_output_dir / "height_maps_vis",
        "blend": root_output_dir / "blend",
        "annotations": root_output_dir / "annotations",
        "summaries": root_output_dir / "summaries",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def _choose_forest_types_for_batch(
    cfg: TreeScatterConfig,
    scene_id_start: int,
) -> list[str]:
    num_scenes = max(1, int(cfg.generation.num_scenes))
    configured_weights = {
        str(forest_type).strip().lower(): float(weight)
        for forest_type, weight in cfg.generation.forest_type_weights.items()
    }

    if not configured_weights:
        return [cfg.forest.type] * num_scenes

    available = set(get_available_forest_types())
    unknown = sorted(name for name in configured_weights if name not in available)
    if unknown:
        raise ValueError(
            "Unknown biome profiles in generation.forest_type_weights: "
            f"{', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )

    positive_items = [(name, weight) for name, weight in configured_weights.items() if weight > 0.0]
    if not positive_items:
        raise ValueError(
            "generation.forest_type_weights must include at least one positive weight."
        )

    population = [item[0] for item in positive_items]
    weights = [item[1] for item in positive_items]
    rng = random.Random(cfg.seed)

    # Keep chunked runs consistent with a single long run by advancing the RNG
    # sequence up to the requested starting scene.
    sequence_offset = max(0, int(scene_id_start) - 1)
    total_draws = sequence_offset + num_scenes
    sampled = rng.choices(population=population, weights=weights, k=total_draws)
    return sampled[sequence_offset:]


def _choose_target_profiles_for_batch(
    runtime_cfg: TreeScatterConfig,
    raw_cfg: DictConfig,
    scene_id_start: int,
) -> list[str]:
    num_scenes = max(1, int(runtime_cfg.generation.num_scenes))
    configured_weights = {
        str(profile_name).strip(): float(weight)
        for profile_name, weight in runtime_cfg.generation.target_profile_weights.items()
    }

    if not configured_weights:
        return []

    available = set(get_available_target_profiles(raw_cfg))
    unknown = sorted(name for name in configured_weights if name not in available)
    if unknown:
        raise ValueError(
            "Unknown target profiles in generation.target_profile_weights: "
            f"{', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )

    positive_items = [(name, weight) for name, weight in configured_weights.items() if weight > 0.0]
    if not positive_items:
        raise ValueError(
            "generation.target_profile_weights must include at least one positive weight."
        )

    population = [item[0] for item in positive_items]
    weights = [item[1] for item in positive_items]
    rng = random.Random(int(runtime_cfg.seed) + 7919)

    sequence_offset = max(0, int(scene_id_start) - 1)
    total_draws = sequence_offset + num_scenes
    sampled = rng.choices(population=population, weights=weights, k=total_draws)
    return sampled[sequence_offset:]


def _choose_forest_type_for_target_profile(
    raw_cfg: DictConfig,
    runtime_cfg: TreeScatterConfig,
    target_profile: str,
    scene_id: int,
) -> str:
    configured_weights = load_target_profile_forest_type_weights(raw_cfg, target_profile)
    if not configured_weights:
        raise ValueError(
            f"Target profile '{target_profile}' must include at least one biome weight."
        )

    available = set(get_available_forest_types())
    unknown = sorted(name for name in configured_weights if name not in available)
    if unknown:
        raise ValueError(
            f"Unknown biome profiles in target profile '{target_profile}': "
            f"{', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )

    positive_items = [(name, weight) for name, weight in configured_weights.items() if weight > 0.0]
    if not positive_items:
        raise ValueError(
            f"Target profile '{target_profile}' forest_type_weights must include at least "
            "one positive weight."
        )

    population = [item[0] for item in positive_items]
    weights = [item[1] for item in positive_items]
    rng = random.Random(int(runtime_cfg.seed) + int(scene_id) * 1009 + 1543)
    return str(rng.choices(population=population, weights=weights, k=1)[0])


def _run_single_scene(
    cfg: TreeScatterConfig,
    root_output_dir: Path,
    scene_id: int,
) -> None:
    output_dirs = _build_output_dirs(root_output_dir)
    suffix = _scene_id_suffix(scene_id)
    scene_id_int = int(scene_id)

    scene = _reset_main_scene(f"TreeScatterScene_{suffix}")
    scene.frame_set(scene_id_int)

    passes = cfg.output_passes
    enable_smoothed = bool(passes.smoothed_instance_id or passes.coco_annotations)
    needs_rgb_for_postprocess = bool(enable_smoothed)
    enable_rgb = bool(passes.rgb or needs_rgb_for_postprocess)
    enable_instance = bool(
        passes.instance_id or passes.smoothed_instance_id or passes.coco_annotations
    )
    enable_variant = bool(
        passes.variant_id or passes.semantic_id or passes.semantic_labels or passes.coco_annotations
    )
    enable_density = bool(passes.density_map)
    needs_height_for_postprocess = bool(enable_smoothed)
    enable_height = bool(passes.height_map or passes.height_map_vis or needs_height_for_postprocess)
    enable_semantic = bool(passes.semantic_id or passes.semantic_labels or passes.coco_annotations)
    enable_coco = bool(passes.coco_annotations)
    enable_summary = bool(passes.scene_summary)
    mask_postprocess_diagnostics: dict[str, object] = {}

    requested_season = sample_scene_season(getattr(cfg.scatter, "season_weights", {}), cfg.seed)
    tree_collection, variant_presets, tree_collection_metadata = create_tree_collection_from_source(
        scene,
        cfg.scatter,
        cfg.seed,
        season=requested_season,
    )
    variant_source_seasons = list(tree_collection_metadata.get("variant_source_seasons", []))
    resolved_species_seasons = dict(tree_collection_metadata.get("resolved_species_seasons", {}))
    logger.info("Scene season requested: %s", requested_season)
    if resolved_species_seasons:
        logger.info(
            "Scene season resolved by species: %s",
            ", ".join(
                f"{species}={season}"
                for species, season in sorted(resolved_species_seasons.items())
            ),
        )
    terrain_size = _compute_terrain_size_from_gsd(
        resolution_x=cfg.render.resolution_x,
        resolution_y=cfg.render.resolution_y,
        gsd_at_nadir=cfg.render.gsd_at_nadir,
        fallback_size=cfg.terrain.size,
    )
    # Sample a relative canopy-density control in [0, 1] before context
    # generation so dense forest scenes can suppress rural artifacts.
    scatter_density_sample = sample_scatter_density(
        cfg.scatter.scatter_density_min,
        cfg.scatter.scatter_density_max,
        cfg.seed,
        band_weights=cfg.scatter.scatter_density_band_weights,
        step=cfg.scatter.scatter_density_step,
    )
    scatter_density = scatter_density_sample.value
    scatter_density_band = scatter_density_sample.band
    logger.info(
        "Sampled scatter density: %.2f (band=%s)",
        scatter_density,
        scatter_density_band,
    )
    terrain_texture_path, selected_terrain_type = _choose_terrain_texture_path(
        cfg, requested_season
    )
    terrain_context = create_terrain_with_context(
        scene=scene,
        terrain_cfg=cfg.terrain,
        terrain_size=terrain_size,
        seed=cfg.seed,
        texture_image_path=terrain_texture_path,
        texture_gsd=float(getattr(cfg.terrain, "texture_gsd", 0.05)),
        density_band=scatter_density_band,
    )
    terrain_obj = terrain_context.terrain_obj

    # Calibration converts the sampled density to target crown closure and
    # stems/m2, then caps the result by an ecological minimum stem spacing.
    scatter_calibration = _compute_calibrated_scatter_parameters(
        cfg=cfg,
        terrain_obj=terrain_obj,
        tree_collection=tree_collection,
        scatter_density_relative=scatter_density,
    )
    density_controls = compute_density_patch_controls(cfg.scatter, scatter_density)
    density_seed = int(cfg.seed) + int(getattr(cfg.scatter, "density_noise_seed_offset", 1009))
    density_field = build_density_field_on_terrain(
        terrain_obj,
        attribute_name=DENSITY_FIELD_ATTRIBUTE_NAME,
        eligibility_attribute_name="tree_eligible",
        noise_scale=float(cfg.scatter.density_noise_scale),
        threshold=float(density_controls["density_threshold"]),
        variation_strength=float(density_controls["density_variation_strength"]),
        cluster_exponent=float(density_controls["density_cluster_exponent"]),
        seed=density_seed,
    )
    logger.info(
        "Scatter calibration | eligible_area=%.1f/%.1f m2 (%.1f%%) | crown_mean=%.2f m | "
        "closure_target=%.2f | closure_applied=%.2f | cai_target=%.2f | cai_applied=%.2f | "
        "spacing_required=%.2f m | spacing_floor=%.2f m | spacing_applied=%.2f m | "
        "mask_mean=%.3f | patch_threshold=%.2f | patch_variation=%.2f | cluster_exp=%.2f | "
        "target_stems=%.4f/m2 | applied_stems=%.4f/m2 | density_input=%.4f/m2",
        scatter_calibration["eligible_area_m2"],
        scatter_calibration["total_area_m2"],
        scatter_calibration["eligible_fraction"] * 100.0,
        scatter_calibration["crown_diameter_mean_m"],
        scatter_calibration["target_canopy_coverage"],
        scatter_calibration["applied_canopy_coverage"],
        scatter_calibration["target_crown_area_index"],
        scatter_calibration["applied_crown_area_index"],
        scatter_calibration["required_stem_spacing_for_coverage_m"],
        scatter_calibration["stem_spacing_floor_m"],
        scatter_calibration["applied_stem_spacing_m"],
        scatter_calibration["density_mask_mean_factor"],
        scatter_calibration["density_threshold_effective"],
        scatter_calibration["density_variation_strength_effective"],
        scatter_calibration["density_cluster_exponent"],
        scatter_calibration["target_stem_density_for_coverage_per_m2"],
        scatter_calibration["applied_stem_density_for_coverage_per_m2"],
        scatter_calibration["scatter_density_for_distribution"],
    )
    scatter_density_value = float(
        scatter_calibration["applied_stem_density_for_coverage_per_m2"]
    ) / max(1e-9, density_field.eligible_mean)
    logger.info(
        "Scatter density input | value=%.6f per m2 | field_mean=%.4f min=%.4f max=%.4f seed=%s",
        scatter_density_value,
        density_field.eligible_mean,
        density_field.eligible_min,
        density_field.eligible_max,
        density_seed,
    )

    scatter_validation_path: Path | None = None
    variant_templates = _tree_variant_templates_from_collection(tree_collection)
    if not variant_templates:
        raise RuntimeError(
            "Botaniq tree templates are missing crown metadata required for crown-aware placement."
        )
    placements = generate_crown_aware_poisson_placements(
        terrain_obj,
        density_field=density_field,
        target_stem_density_per_m2=float(
            scatter_calibration["applied_stem_density_for_coverage_per_m2"]
        ),
        variants=variant_templates,
        scatter_cfg=cfg.scatter,
        seed=cfg.seed,
        eligibility_attribute_name="tree_eligible",
    )
    scatter_points_obj = _create_precomputed_scatter_points_object(
        scene,
        placements,
        VARIANT_ID_ATTRIBUTE_NAME,
    )
    build_precomputed_scatter_gn_tree(
        terrain_obj=terrain_obj,
        tree_collection=tree_collection,
        placement_points_obj=scatter_points_obj,
        terrain_cfg=cfg.terrain,
        variant_attribute_name=VARIANT_ID_ATTRIBUTE_NAME,
    )
    scatter_validation = validate_scatter_placements(
        terrain_obj,
        placements,
        target_tree_count=float(scatter_calibration["estimated_tree_count_target"]),
        target_stem_density_per_m2=float(
            scatter_calibration["applied_stem_density_for_coverage_per_m2"]
        ),
        eligible_area_m2=float(scatter_calibration["eligible_area_m2"]),
        density_field=density_field,
        grid_cells=int(getattr(cfg.scatter, "scatter_validation_grid_cells", 8)),
        canopy_grid_resolution=int(
            getattr(cfg.scatter, "scatter_validation_canopy_grid_resolution", 128)
        ),
    )
    scatter_validation_path = _write_scatter_validation_artifacts(
        output_dirs,
        suffix,
        scatter_validation,
    )
    logger.info(
        "Precomputed Botaniq scatter | target=%.1f actual=%d ratio=%.3f | "
        "trees_ha=%.1f | nn_median=%.2f m | crown_overlap_pairs=%d | canopy_cover=%.3f",
        scatter_validation.target_tree_count,
        scatter_validation.actual_tree_count,
        scatter_validation.actual_to_target_count_ratio,
        scatter_validation.actual_trees_per_ha,
        scatter_validation.nearest_neighbor_median_m,
        scatter_validation.crown_overlap_pair_count,
        scatter_validation.canopy_cover_fraction_total,
    )

    # Terrain elevation is applied by Geometry Nodes. Re-anchor sparse context
    # buildings to the evaluated terrain so roofs do not intersect hills.
    bpy.context.view_layer.update()
    snap_context_buildings_to_terrain(
        scene=scene,
        terrain_obj=terrain_obj,
        terrain_cfg=cfg.terrain,
    )

    create_camera(
        scene,
        terrain_size,
        cfg.render.resolution_x,
        cfg.render.resolution_y,
        cfg.render.gsd_at_nadir,
    )
    sun_obj = create_sun_light(scene, cfg.seed)

    configure_render_common(scene, cfg.render)

    for view_layer in scene.view_layers:
        if hasattr(view_layer, "use_pass_z"):
            view_layer.use_pass_z = True
        if hasattr(view_layer, "pass_alpha_threshold"):
            view_layer.pass_alpha_threshold = 0

    flat_scene = None
    camera_angle_rad = _compute_camera_angle_rad(scene.camera)

    if enable_instance:
        override_material = ensure_override_material(cfg.scatter.instance_id_max_value)
        setup_instance_id_view_layer(scene, override_material)

    variant_id_max_value = max(1, len(variant_presets))
    if enable_variant:
        variant_id_material = ensure_variant_id_material(
            variant_id_max_value=variant_id_max_value,
            attribute_name=VARIANT_ID_ATTRIBUTE_NAME,
        )
        setup_variant_id_view_layer(scene, variant_id_material)

    if enable_density:
        density_material = ensure_density_map_material(
            density_noise_scale=cfg.scatter.density_noise_scale,
            density_threshold=density_controls["density_threshold"],
            density_attribute_name=DENSITY_FIELD_ATTRIBUTE_NAME,
        )
        setup_density_map_view_layer(scene, density_material)

    template_collection_name = tree_collection.name
    for vl in scene.view_layers:
        layer_col = find_layer_collection(vl.layer_collection, template_collection_name)
        if layer_col is not None:
            layer_col.exclude = True

    flat_scene = _create_flat_height_scene(scene, terrain_obj.name) if enable_height else None
    bpy.context.view_layer.update()

    blend_path = output_dirs["blend"] / f"scene{suffix}.blend"
    if cfg.render.no_render:
        if flat_scene is not None:
            _remove_height_reference_scene(flat_scene, owner_scene=scene)
            flat_scene = None
        configure_for_rgb(scene, cfg.render)
        _clear_scene_compositor(scene)
        if cfg.output.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
            logger.info("Scene saved: %s", blend_path)
        else:
            logger.info("render.no_render=true and output.save_blend=false; skipping .blend save.")
        return

    instance_id_path = output_dirs["instance_id"] / f"instance_id_map{suffix}.tif"
    variant_id_path = output_dirs["variant_id"] / f"variant_id_map{suffix}.tif"
    rgb_path = output_dirs["images"] / f"rgb_satellite{suffix}.tif"
    height_map_exr_path = output_dirs["height"] / f"height_map{suffix}.exr"
    height_map_path = output_dirs["height"] / f"height_map{suffix}.tif"
    height_map_vis_path = output_dirs["height_vis"] / f"height_map_vis{suffix}.png"

    if enable_rgb or enable_height:
        configure_for_rgb_height(scene, cfg.render)
        setup_compositor_rgb_height(
            scene=scene,
            rgb_output_dir=str(output_dirs["images"]) if enable_rgb else None,
            flat_scene=flat_scene if enable_height else None,
            height_output_dir=str(output_dirs["height"]) if enable_height else None,
            angle_rad=camera_angle_rad,
            threshold=float(cfg.render.height_threshold),
            terrain_depth_max=(
                float(scene.camera.data.clip_end) * 0.99 if enable_height and scene.camera else None
            ),
        )
        bpy.context.view_layer.update()
        logger.info(
            "Render pass: RGB/Height | rgb=%s height=%s",
            enable_rgb,
            enable_height,
        )
        bpy.ops.render.render(write_still=False, scene=scene.name)

        if enable_rgb:
            _georeference_rendered_tiff(
                rgb_path,
                resolution_x=cfg.render.resolution_x,
                resolution_y=cfg.render.resolution_y,
                gsd_at_nadir=cfg.render.gsd_at_nadir,
            )
        else:
            logger.info("Render pass skipped: RGB")
    else:
        logger.info("Render pass skipped: RGB")
        logger.info("Render pass skipped: Height map")

    if enable_height and flat_scene is not None:
        try:
            if (passes.height_map or needs_height_for_postprocess) and _convert_height_exr_to_tiff(
                height_map_exr_path, height_map_path
            ):
                _georeference_rendered_tiff(
                    height_map_path,
                    resolution_x=cfg.render.resolution_x,
                    resolution_y=cfg.render.resolution_y,
                    gsd_at_nadir=cfg.render.gsd_at_nadir,
                )
                try:
                    height_map_exr_path.unlink()
                except Exception:
                    pass
                if passes.height_map:
                    logger.info("Height map output: %s", height_map_path)
                else:
                    logger.info("Height map generated for mask postprocess: %s", height_map_path)
            elif passes.height_map or needs_height_for_postprocess:
                logger.warning(
                    "[warn] Height EXR→TIFF conversion failed; keeping EXR output at "
                    f"{height_map_exr_path}"
                )
            else:
                try:
                    if height_map_exr_path.exists():
                        height_map_exr_path.unlink()
                except Exception:
                    pass

            if passes.height_map_vis:
                if not _save_height_visualization_from_tiff(height_map_path, height_map_vis_path):
                    logger.warning(
                        "Failed to generate TIFF-based height visualization: %s",
                        height_map_vis_path,
                    )
                else:
                    logger.info("Height map visualization output: %s", height_map_vis_path)
        except Exception as exc:
            logger.warning("Failed to render height map. Reason: %s", exc)
    elif enable_height:
        logger.warning("Height map skipped: failed to create flat reference scene.")

    if enable_instance or enable_variant:
        configure_for_label_maps(
            scene,
            enable_instance=enable_instance,
            enable_variant=enable_variant,
        )
        setup_compositor_label_maps(
            scene,
            instance_output_dir=str(output_dirs["instance_id"]) if enable_instance else None,
            variant_output_dir=str(output_dirs["variant_id"]) if enable_variant else None,
        )
        if flat_scene is not None:
            _remove_height_reference_scene(flat_scene, owner_scene=scene)
            flat_scene = None
        bpy.context.view_layer.update()
        logger.info(
            "Render pass: Label maps | instance=%s variant=%s",
            enable_instance,
            enable_variant,
        )
        bpy.ops.render.render(write_still=False, scene=scene.name)
        if enable_instance:
            _georeference_rendered_tiff(
                instance_id_path,
                resolution_x=cfg.render.resolution_x,
                resolution_y=cfg.render.resolution_y,
                gsd_at_nadir=cfg.render.gsd_at_nadir,
            )
            logger.info("Instance ID map saved: %s", instance_id_path)
        else:
            logger.info("Render pass skipped: Instance ID")

        if enable_variant:
            _georeference_rendered_tiff(
                variant_id_path,
                resolution_x=cfg.render.resolution_x,
                resolution_y=cfg.render.resolution_y,
                gsd_at_nadir=cfg.render.gsd_at_nadir,
            )
            logger.info("Variant ID map saved: %s", variant_id_path)
        else:
            logger.info("Render pass skipped: Variant ID")
    else:
        logger.info("Render pass skipped: Instance ID")
        logger.info("Render pass skipped: Variant ID")

    if flat_scene is not None:
        _remove_height_reference_scene(flat_scene, owner_scene=scene)
        flat_scene = None

    if enable_density:
        configure_for_density_map(scene)
        setup_compositor_density_map(scene, str(output_dirs["density"]))
        scatter_hidden_for_density = _set_scatter_density_on_terrain(
            terrain_obj,
            0.0,
            warn=True,
        )
        bpy.context.view_layer.update()
        logger.info("Render pass: Density map")
        try:
            bpy.ops.render.render(write_still=False, scene=scene.name)
        finally:
            if scatter_hidden_for_density:
                _set_scatter_density_on_terrain(
                    terrain_obj,
                    1.0 if variant_templates else scatter_density_value,
                    warn=False,
                )
                bpy.context.view_layer.update()
    else:
        logger.info("Render pass skipped: Density map")

    smoothed_mask_path = (
        output_dirs["smoothed_instance_id"] / f"instance_id_map_smoothed{suffix}.tif"
    )
    if enable_smoothed:
        if instance_id_path.exists():
            try:
                mask_postprocess_diagnostics = save_smoothed_instance_mask(
                    mask_path=instance_id_path,
                    output_path=smoothed_mask_path,
                    gsd=float(cfg.render.gsd_at_nadir),
                    height_map_path=height_map_path if height_map_path.exists() else None,
                    rgb_image_path=rgb_path if rgb_path.exists() else None,
                    mask_postprocess_config=cfg.mask_postprocess,
                )
                if smoothed_mask_path.exists():
                    _georeference_rendered_tiff(
                        smoothed_mask_path,
                        resolution_x=cfg.render.resolution_x,
                        resolution_y=cfg.render.resolution_y,
                        gsd_at_nadir=cfg.render.gsd_at_nadir,
                    )
                    if passes.smoothed_instance_id:
                        logger.info("Smoothed instance ID map: %s", smoothed_mask_path)
                elif mask_postprocess_diagnostics.get("postprocess_skipped"):
                    logger.warning(
                        "Smoothed instance ID skipped: %s",
                        mask_postprocess_diagnostics.get("skip_reason", "unknown"),
                    )
            except Exception as exc:
                logger.warning(
                    "[warn] Failed to generate smoothed instance ID map. "
                    "Ensure pillow, scipy, and scikit-image are available in Blender Python. "
                    f"Reason: {exc}"
                )
        else:
            logger.warning("Smoothed instance ID skipped: instance map is unavailable.")

    category_id_to_name, variant_id_to_category_id = _build_preset_category_mappings(
        variant_presets
    )
    class_id_max_value = max(1, len(category_id_to_name))
    semantic_id_path = output_dirs["semantic_id"] / f"semantic_id_map{suffix}.tif"
    semantic_id_legend_path = output_dirs["semantic_labels"] / f"semantic_id_map{suffix}_labels.txt"
    if enable_semantic:
        if variant_id_path.exists():
            try:
                save_class_id_map_from_variant_id_map(
                    variant_id_map_path=variant_id_path,
                    output_path=semantic_id_path,
                    variant_id_to_class_id=variant_id_to_category_id,
                    variant_id_max_value=variant_id_max_value,
                    class_id_max_value=class_id_max_value,
                )
                _georeference_rendered_tiff(
                    semantic_id_path,
                    resolution_x=cfg.render.resolution_x,
                    resolution_y=cfg.render.resolution_y,
                    gsd_at_nadir=cfg.render.gsd_at_nadir,
                )
                if passes.semantic_id:
                    logger.info("Semantic ID map saved: %s", semantic_id_path)

                if passes.semantic_labels:
                    legend_lines = ["0\tbackground"]
                    legend_lines.extend(
                        f"{category_id}\t{name}"
                        for category_id, name in sorted(category_id_to_name.items())
                    )
                    semantic_id_legend_path.write_text(
                        "\n".join(legend_lines) + "\n", encoding="utf-8"
                    )
                    logger.info("Semantic ID labels saved: %s", semantic_id_legend_path)
            except Exception as exc:
                logger.warning("Failed to generate semantic ID map. Reason: %s", exc)
        else:
            logger.warning("Semantic ID skipped: variant map is unavailable.")

    coco_annotations_path = output_dirs["annotations"] / f"annotations_coco{suffix}.json"
    if enable_coco:
        if instance_id_path.exists() and semantic_id_path.exists():
            # Export annotations from the cleaned ID map while using raw IDs for semantic voting.
            instance_mask_polygon = (
                smoothed_mask_path if smoothed_mask_path.exists() else instance_id_path
            )
            if mask_postprocess_diagnostics.get("postprocess_skipped"):
                logger.warning(
                    "COCO export is using raw instance IDs because mask postprocess was skipped."
                )
            try:
                min_coco_area_px = max(
                    1,
                    int(
                        round(
                            float(cfg.mask_postprocess.min_final_area_m2)
                            / max(1e-9, float(cfg.render.gsd_at_nadir) ** 2)
                        )
                    ),
                )
                export_instance_segmentation_coco(
                    instance_mask_raw_path=instance_id_path,
                    instance_mask_polygon_path=instance_mask_polygon,
                    class_id_mask_path=semantic_id_path,
                    output_json_path=coco_annotations_path,
                    image_file_name=f"rgb_satellite{suffix}.tif",
                    category_id_to_name=category_id_to_name,
                    class_id_max_value=class_id_max_value,
                    min_area_px=min_coco_area_px,
                )
                logger.info("COCO instance annotations: %s", coco_annotations_path)
            except Exception as exc:
                logger.warning("Failed to export COCO annotations. Reason: %s", exc)
        else:
            logger.warning("COCO export skipped: instance and/or semantic map is unavailable.")

    configure_for_rgb(scene, cfg.render)
    if enable_rgb:
        setup_compositor_rgb(scene, str(output_dirs["images"]))
    else:
        _clear_scene_compositor(scene)
    bpy.context.view_layer.update()
    if cfg.output.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    if cfg.output.save_blend:
        logger.info("Scene saved: %s", blend_path)
    else:
        logger.info("Scene save skipped (output.save_blend=false)")
    if enable_rgb and rgb_path.exists():
        logger.info("RGB output: %s", rgb_path)
    if passes.instance_id and instance_id_path.exists():
        logger.info("Instance ID output: %s", instance_id_path)
    if passes.variant_id and variant_id_path.exists():
        logger.info("Variant ID output: %s", variant_id_path)
    if passes.semantic_id and semantic_id_path.exists():
        logger.info("Semantic ID output: %s", semantic_id_path)
    if passes.semantic_labels and semantic_id_legend_path.exists():
        logger.info("Semantic ID labels output: %s", semantic_id_legend_path)
    if passes.smoothed_instance_id and smoothed_mask_path.exists():
        logger.info("Smoothed Instance ID output: %s", smoothed_mask_path)
    if passes.coco_annotations and coco_annotations_path.exists():
        logger.info("COCO output: %s", coco_annotations_path)
    if passes.instance_id or passes.variant_id or passes.semantic_id:
        logger.info(
            f"Decode IDs with: round(pixel_value * {max(1, cfg.scatter.instance_id_max_value)})"
        )
    density_map_path = output_dirs["density"] / f"density_map{suffix}.png"
    if passes.density_map and density_map_path.exists():
        logger.info("Density map output: %s", density_map_path)
    if passes.height_map and height_map_exr_path.exists():
        logger.info("Height EXR output: %s", height_map_exr_path)
    if passes.height_map and height_map_path.exists():
        logger.info("Height output: %s", height_map_path)
    if passes.height_map_vis and height_map_vis_path.exists():
        logger.info("Height vis output: %s", height_map_vis_path)

    summary_path = output_dirs["summaries"] / f"scene_summary{suffix}.txt"
    if enable_summary:
        try:
            stats: dict[str, object] | None = None
            species_present: list[str] = []
            per_species_tree_counts: dict[str, int] = {}
            per_species_tree_count_lines: list[str] = []
            has_mask_statistics = bool(
                enable_instance
                and enable_variant
                and enable_semantic
                and instance_id_path.exists()
                and semantic_id_path.exists()
                and variant_id_path.exists()
            )
            if has_mask_statistics:
                try:
                    mask_statistics_instance_path = (
                        smoothed_mask_path if smoothed_mask_path.exists() else instance_id_path
                    )
                    stats = compute_mask_statistics(
                        instance_mask_path=mask_statistics_instance_path,
                        semantic_id_mask_path=semantic_id_path,
                        variant_id_mask_path=variant_id_path,
                        semantic_id_max_value=class_id_max_value,
                        variant_id_max_value=variant_id_max_value,
                    )

                    species_present = [
                        category_id_to_name[semantic_id]
                        for semantic_id in stats["semantic_ids_present"]
                        if semantic_id in category_id_to_name
                    ]

                    for semantic_id, count in sorted(stats["per_semantic_tree_count"].items()):
                        species_name = category_id_to_name.get(
                            semantic_id, f"unknown_{semantic_id}"
                        )
                        per_species_tree_counts[species_name] = int(count)
                        per_species_tree_count_lines.append(f"  - {species_name}: {count}")
                except Exception as exc:
                    logger.warning("Mask-derived scene summary fields skipped. Reason: %s", exc)
                    stats = None
                    species_present = []
                    per_species_tree_counts = {}
                    per_species_tree_count_lines = []
            else:
                logger.info(
                    "Mask-derived scene summary fields skipped: "
                    "instance/semantic/variant maps are unavailable."
                )

            configured_species = list(tree_collection_metadata.get("botaniq_species_selected", []))

            botaniq_summary_lines = []
            botaniq_pack_dir = str(tree_collection_metadata.get("botaniq_pack_dir", "")).strip()
            if botaniq_pack_dir:
                botaniq_summary_lines.append(f"botaniq_pack_dir: {botaniq_pack_dir}")
            botaniq_asset_titles = sorted(
                {
                    str(title).strip()
                    for title in tree_collection_metadata.get("botaniq_asset_titles", [])
                    if str(title).strip()
                }
            )
            if botaniq_asset_titles:
                botaniq_summary_lines.append(
                    f"botaniq_assets_used: {', '.join(botaniq_asset_titles)}"
                )
            botaniq_species_roles = dict(tree_collection_metadata.get("botaniq_species_roles", {}))
            if botaniq_species_roles:
                botaniq_summary_lines.append(
                    "botaniq_species_roles: "
                    + ", ".join(
                        f"{species}:{role}"
                        for species, role in sorted(botaniq_species_roles.items())
                    )
                )
            botaniq_species_abundances = dict(
                tree_collection_metadata.get("botaniq_species_abundances", {})
            )
            if botaniq_species_abundances:
                botaniq_summary_lines.append(
                    "botaniq_species_abundances: "
                    + ", ".join(
                        f"{species}:{float(abundance):.4f}"
                        for species, abundance in sorted(botaniq_species_abundances.items())
                    )
                )
            botaniq_role_abundances = dict(
                tree_collection_metadata.get("botaniq_role_abundances", {})
            )
            if botaniq_role_abundances:
                botaniq_summary_lines.append(
                    "botaniq_role_abundances: "
                    + ", ".join(
                        f"{role}:{float(abundance):.4f}"
                        for role, abundance in sorted(botaniq_role_abundances.items())
                    )
                )
            botaniq_variant_allocation = dict(
                tree_collection_metadata.get("botaniq_variant_allocation", {})
            )
            if botaniq_variant_allocation:
                botaniq_summary_lines.append(
                    "botaniq_variant_allocation: "
                    + ", ".join(
                        f"{species}:{int(count)}"
                        for species, count in sorted(botaniq_variant_allocation.items())
                    )
                )
            botaniq_branch_groups_used = [
                str(group_name).strip()
                for group_name in tree_collection_metadata.get(
                    "botaniq_branch_groups_used",
                    [],
                )
                if str(group_name).strip()
            ]
            if botaniq_branch_groups_used:
                botaniq_summary_lines.append(
                    "botaniq_branch_groups_used: " + ", ".join(sorted(botaniq_branch_groups_used))
                )
            botaniq_branch_swap_pair_count = int(
                tree_collection_metadata.get("botaniq_branch_swap_pair_count", 0)
            )
            botaniq_branch_vertices_changed = int(
                tree_collection_metadata.get("botaniq_branch_vertices_changed", 0)
            )
            if botaniq_branch_swap_pair_count or botaniq_branch_vertices_changed:
                botaniq_branch_roles = [
                    str(role).strip()
                    for role in tree_collection_metadata.get(
                        "botaniq_branch_variation_roles",
                        [],
                    )
                    if str(role).strip()
                ]
                role_summary = ""
                if botaniq_branch_roles:
                    role_counts = {
                        role: botaniq_branch_roles.count(role)
                        for role in sorted(set(botaniq_branch_roles))
                    }
                    role_summary = ", roles=" + ",".join(
                        f"{role}:{count}" for role, count in role_counts.items()
                    )
                botaniq_summary_lines.append(
                    "botaniq_branch_variation: "
                    f"swap_pairs={botaniq_branch_swap_pair_count}, "
                    f"vertices_changed={botaniq_branch_vertices_changed}"
                    f"{role_summary}"
                )

            scatter_validation_lines = []
            if scatter_validation is not None:
                scatter_validation_lines = [
                    f"scatter_validation_json: {scatter_validation_path if scatter_validation_path is not None else 'none'}",
                    f"scatter_actual_tree_count: {scatter_validation.actual_tree_count}",
                    f"scatter_actual_to_target_count_ratio: {scatter_validation.actual_to_target_count_ratio:.6f}",
                    f"scatter_target_trees_per_ha: {scatter_validation.target_trees_per_ha:.4f}",
                    f"scatter_actual_trees_per_ha: {scatter_validation.actual_trees_per_ha:.4f}",
                    f"scatter_density_field_mean_actual: {scatter_validation.density_field_mean:.6f}",
                    f"scatter_density_field_min_actual: {scatter_validation.density_field_min:.6f}",
                    f"scatter_density_field_max_actual: {scatter_validation.density_field_max:.6f}",
                    f"scatter_nn_min_m: {scatter_validation.nearest_neighbor_min_m:.4f}",
                    f"scatter_nn_p05_m: {scatter_validation.nearest_neighbor_p05_m:.4f}",
                    f"scatter_nn_mean_m: {scatter_validation.nearest_neighbor_mean_m:.4f}",
                    f"scatter_nn_median_m: {scatter_validation.nearest_neighbor_median_m:.4f}",
                    f"scatter_nn_p95_m: {scatter_validation.nearest_neighbor_p95_m:.4f}",
                    f"scatter_crown_overlap_pair_count: {scatter_validation.crown_overlap_pair_count}",
                    f"scatter_crown_overlap_area_sum_m2: {scatter_validation.crown_overlap_area_sum_m2:.4f}",
                    f"scatter_crown_overlap_area_fraction_of_crown_area: {scatter_validation.crown_overlap_area_fraction_of_crown_area:.6f}",
                    f"scatter_canopy_cover_fraction_total: {scatter_validation.canopy_cover_fraction_total:.6f}",
                    "scatter_density_grid_counts:",
                    *[
                        "  - " + ", ".join(str(value) for value in row)
                        for row in scatter_validation.density_grid_counts
                    ],
                    "scatter_density_grid_target_counts:",
                    *[
                        "  - " + ", ".join(f"{value:.2f}" for value in row)
                        for row in scatter_validation.density_grid_target_counts
                    ],
                    "scatter_density_grid_actual_to_target_ratio:",
                    *[
                        "  - " + ", ".join(f"{value:.3f}" for value in row)
                        for row in scatter_validation.density_grid_actual_to_target_ratio
                    ],
                ]

            mask_postprocess_lines = _format_mask_postprocess_summary(mask_postprocess_diagnostics)
            terrain_texture_summary = (
                str(terrain_texture_path) if terrain_texture_path is not None else "none"
            )
            terrain_texture_type_summary = (
                selected_terrain_type if selected_terrain_type is not None else "none"
            )

            def relative_output_path(path: Path) -> str:
                try:
                    return str(path.relative_to(root_output_dir))
                except ValueError:
                    return str(path)

            output_candidates = [
                ("rgb", rgb_path, enable_rgb),
                ("instance_id", instance_id_path, enable_instance),
                ("variant_id", variant_id_path, enable_variant),
                ("semantic_id", semantic_id_path, enable_semantic),
                ("semantic_labels", semantic_id_legend_path, enable_semantic),
                ("density", density_map_path, enable_density),
                ("height", height_map_path, enable_height),
                ("height_exr", height_map_exr_path, enable_height),
                ("height_vis", height_map_vis_path, bool(passes.height_map_vis)),
                ("smoothed_instance_id", smoothed_mask_path, enable_smoothed),
                ("coco_annotations", coco_annotations_path, enable_coco),
                ("blend", blend_path, bool(cfg.output.save_blend)),
            ]
            existing_outputs = {
                name: relative_output_path(path)
                for name, path, include in output_candidates
                if include and path.exists()
            }
            output_lines = (
                [
                    "outputs:",
                    *[f"  - {name}: {path}" for name, path in existing_outputs.items()],
                ]
                if existing_outputs
                else []
            )

            summary_lines = [
                "Tree Scatter Scene Summary",
                "==========================",
                f"forest_type: {cfg.forest.type}",
                f"target_profile: {str(getattr(cfg.target_profile, 'name', '')).strip() or 'none'}",
                f"tree_source_mode: {cfg.scatter.tree_source_mode}",
                *botaniq_summary_lines,
                *scatter_validation_lines,
                *mask_postprocess_lines,
                f"seed: {cfg.seed}",
                f"season_requested: {requested_season}",
                f"season_variant_sources: {', '.join(variant_source_seasons) if variant_source_seasons else 'none'}",
                f"season_species_resolution: {', '.join(f'{species}:{season}' for species, season in sorted(resolved_species_seasons.items())) if resolved_species_seasons else 'none'}",
                f"gsd_at_nadir: {float(cfg.render.gsd_at_nadir):.6f}",
                f"scatter_density_sampled: {scatter_density:.2f}",
                f"scatter_density_band: {scatter_density_band}",
                f"scatter_density_input_per_m2: {scatter_density_value:.6f}",
                f"scatter_density_applied_per_m2: {scatter_calibration['applied_stem_density_for_coverage_per_m2']:.6f}",
                f"scatter_target_canopy_coverage: {scatter_calibration['target_canopy_coverage']:.6f}",
                f"scatter_applied_canopy_coverage: {scatter_calibration['applied_canopy_coverage']:.6f}",
                f"scatter_target_crown_area_index: {scatter_calibration['target_crown_area_index']:.6f}",
                f"scatter_applied_crown_area_index: {scatter_calibration['applied_crown_area_index']:.6f}",
                f"scatter_density_mask_mean_factor: {scatter_calibration['density_mask_mean_factor']:.6f}",
                f"scatter_density_map_mean_factor: {scatter_calibration['density_map_mean_factor']:.6f}",
                f"scatter_density_threshold_effective: {scatter_calibration['density_threshold_effective']:.6f}",
                f"scatter_density_variation_effective: {scatter_calibration['density_variation_strength_effective']:.6f}",
                f"scatter_density_cluster_exponent: {scatter_calibration['density_cluster_exponent']:.6f}",
                f"scatter_density_fade: {scatter_calibration['density_fade']:.6f}",
                f"scatter_target_stem_density_for_coverage_per_m2: {scatter_calibration['target_stem_density_for_coverage_per_m2']:.6f}",
                f"scatter_applied_stem_density_for_coverage_per_m2: {scatter_calibration['applied_stem_density_for_coverage_per_m2']:.6f}",
                f"scatter_estimated_tree_count_target: {scatter_calibration['estimated_tree_count_target']:.2f}",
                f"scatter_eligible_area_m2: {scatter_calibration['eligible_area_m2']:.2f}",
                f"scatter_total_area_m2: {scatter_calibration['total_area_m2']:.2f}",
                f"scatter_eligible_fraction: {scatter_calibration['eligible_fraction']:.6f}",
                f"scatter_crown_diameter_mean_m: {scatter_calibration['crown_diameter_mean_m']:.4f}",
                f"scatter_crown_diameter_median_m: {scatter_calibration['crown_diameter_median_m']:.4f}",
                f"scatter_crown_projected_area_mean_m2: {scatter_calibration['crown_projected_area_mean_m2']:.6f}",
                f"scatter_instance_scale_area_factor: {scatter_calibration['instance_scale_area_factor']:.6f}",
                f"scatter_effective_projected_crown_area_m2: {scatter_calibration['effective_projected_crown_area_m2']:.6f}",
                f"scatter_spacing_required_m: {scatter_calibration['required_stem_spacing_for_coverage_m']:.4f}",
                f"scatter_spacing_floor_m: {scatter_calibration['stem_spacing_floor_m']:.4f}",
                f"scatter_spacing_applied_m: {scatter_calibration['applied_stem_spacing_m']:.4f}",
                f"sun_elevation_deg: {sun_obj.get('sun_elevation_deg', 'n/a')}",
                f"sun_azimuth_deg: {sun_obj.get('sun_azimuth_deg', 'n/a')}",
                f"sun_roll_deg: {sun_obj.get('sun_roll_deg', 'n/a')}",
                f"terrain_size_effective: {terrain_size:.6f}",
                f"terrain_color_rgba: {tuple(cfg.terrain.color)}",
                f"terrain_texture: {terrain_texture_summary}",
                f"terrain_texture_type: {terrain_texture_type_summary}",
                f"terrain_texture_gsd: {float(getattr(cfg.terrain, 'texture_gsd', 0.05)):.6f}",
                f"terrain_texture_tiles_x: {float(terrain_obj.get('texture_tiles_x', 0.0)):.6f}",
                f"terrain_texture_tiles_y: {float(terrain_obj.get('texture_tiles_y', 0.0)):.6f}",
                f"terrain_source_mode: {terrain_context.terrain_source_mode}",
                f"terrain_context_profile: {terrain_context.context_profile}",
                f"terrain_context_mode: {terrain_context.context_mode}",
                f"terrain_context_blocked_face_count: {terrain_context.blocked_face_count}",
                f"terrain_context_trail_count: {terrain_context.feature_counts.get('trails', 0)}",
                "terrain_context_trail_segment_count: "
                f"{terrain_context.feature_counts.get('trail_segments', 0)}",
                f"terrain_context_river_count: {terrain_context.feature_counts.get('rivers', 0)}",
                "terrain_context_building_count: "
                f"{terrain_context.feature_counts.get('buildings', 0)}",
                "terrain_context_rural_clearing_count: "
                f"{terrain_context.feature_counts.get('rural_clearings', 0)}",
                f"species_configured: {', '.join(configured_species)}",
                f"variant_count_generated: {len(variant_presets)}",
                *output_lines,
            ]
            if scatter_validation is not None:
                summary_lines.append(
                    f"tree_count_scattered: {scatter_validation.actual_tree_count}"
                )
            if stats is not None:
                summary_lines.extend(
                    [
                        f"species_present: {', '.join(species_present) if species_present else 'none'}",
                        f"tree_count: {stats['tree_count']}",
                        f"variant_ids_present: {stats['variant_ids_present']}",
                        "per_species_tree_count:",
                    ]
                )
                summary_lines.extend(per_species_tree_count_lines or ["  - none"])

            summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
            logger.info("Scene summary: %s", summary_path)
            try:
                variant_source_season_counts = {
                    season_name: variant_source_seasons.count(season_name)
                    for season_name in sorted(set(variant_source_seasons))
                }
                tree_counts: dict[str, object] = {
                    "estimated_target": float(scatter_calibration["estimated_tree_count_target"]),
                    "variants_generated": len(variant_presets),
                }
                if scatter_validation is not None:
                    tree_counts["scattered"] = int(scatter_validation.actual_tree_count)
                if stats is not None:
                    tree_counts["visible"] = int(stats["tree_count"])

                species_overview: dict[str, object] = {"configured": configured_species}
                if botaniq_species_abundances:
                    species_overview["sampled_abundances"] = botaniq_species_abundances
                if botaniq_species_roles:
                    species_overview["roles"] = botaniq_species_roles
                if stats is not None:
                    species_overview["present"] = species_present
                    species_overview["visible_tree_counts"] = per_species_tree_counts

                scene_overview: dict[str, object] = {
                    "scene_id": scene_id_int,
                    "seed": int(cfg.seed),
                    "forest_type": str(cfg.forest.type),
                    "tree_source_mode": str(cfg.scatter.tree_source_mode),
                    "season": {
                        "requested": requested_season,
                        "variant_source_counts": variant_source_season_counts,
                        "species_resolution": resolved_species_seasons,
                    },
                    "sun": {
                        "elevation_deg": float(sun_obj.get("sun_elevation_deg", 0.0)),
                        "azimuth_deg": float(sun_obj.get("sun_azimuth_deg", 0.0)),
                        "zenith_deg": float(sun_obj.get("sun_zenith_deg", 0.0)),
                    },
                    "terrain": {
                        "source_mode": terrain_context.terrain_source_mode,
                        "size_m": float(terrain_size),
                        "image": (
                            str(terrain_texture_path) if terrain_texture_path is not None else None
                        ),
                        "image_type": selected_terrain_type,
                        "image_gsd_m_per_px": float(getattr(cfg.terrain, "texture_gsd", 0.05)),
                        "texture_tiles_x": float(terrain_obj.get("texture_tiles_x", 0.0)),
                        "texture_tiles_y": float(terrain_obj.get("texture_tiles_y", 0.0)),
                    },
                    "context": {
                        "profile": terrain_context.context_profile,
                        "mode": terrain_context.context_mode,
                        "blocked_terrain_face_count": int(terrain_context.blocked_face_count),
                        "counts": dict(terrain_context.feature_counts),
                    },
                    "density": {
                        "sampled_relative": float(scatter_density),
                        "band": scatter_density_band,
                        "input_per_m2": float(scatter_density_value),
                        "applied_stem_density_per_m2": float(
                            scatter_calibration["applied_stem_density_for_coverage_per_m2"]
                        ),
                        "target_canopy_coverage": float(
                            scatter_calibration["target_canopy_coverage"]
                        ),
                        "applied_canopy_coverage": float(
                            scatter_calibration["applied_canopy_coverage"]
                        ),
                        "eligible_area_m2": float(scatter_calibration["eligible_area_m2"]),
                        "eligible_fraction": float(scatter_calibration["eligible_fraction"]),
                    },
                    "tree_counts": tree_counts,
                    "species": species_overview,
                }
                target_profile_name = str(getattr(cfg.target_profile, "name", "")).strip()
                if target_profile_name:
                    scene_overview["target_profile"] = {
                        "name": target_profile_name,
                    }
                if existing_outputs:
                    scene_overview["outputs"] = existing_outputs
                if enable_rgb and rgb_path.exists():
                    scene_overview["image"] = {
                        "path": relative_output_path(rgb_path),
                        "resolution_x_px": int(cfg.render.resolution_x),
                        "resolution_y_px": int(cfg.render.resolution_y),
                        "gsd_at_nadir_m_per_px": float(cfg.render.gsd_at_nadir),
                    }
                run_summary_path = update_run_summary(root_output_dir, scene_overview)
                logger.info("Run summary: %s", run_summary_path)
            except Exception as exc:
                logger.warning("Failed to update run summary. Reason: %s", exc)
        except Exception as exc:
            logger.warning("Failed to write scene summary. Reason: %s", exc)


def run_tree_scatter_pipeline(cfg: DictConfig) -> None:
    base_runtime_cfg = load_tree_scatter_base_config(cfg)
    root_output_dir = Path(base_runtime_cfg.output.output_dir).resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_configs(root_output_dir, cfg, base_runtime_cfg)

    scene_id_start = max(1, int(base_runtime_cfg.generation.scene_id_start))
    selected_target_profiles = _choose_target_profiles_for_batch(
        base_runtime_cfg,
        cfg,
        scene_id_start=scene_id_start,
    )
    if selected_target_profiles:
        explicit_forest_type = OmegaConf.select(cfg, "forest.type", default=None)
        if explicit_forest_type is not None and str(explicit_forest_type).strip():
            logger.info(
                "generation.target_profile_weights is set; forest.type='%s' is ignored "
                "and each scene samples a biome from its selected target profile.",
                str(explicit_forest_type).strip().lower(),
            )
        selected_scene_profiles = []
        for index, target_profile in enumerate(selected_target_profiles):
            scene_id = scene_id_start + index
            forest_type = _choose_forest_type_for_target_profile(
                cfg,
                base_runtime_cfg,
                target_profile,
                scene_id,
            )
            selected_scene_profiles.append((forest_type, target_profile))
    else:
        explicit_forest_type = OmegaConf.select(cfg, "forest.type", default=None)
        if explicit_forest_type is not None and str(explicit_forest_type).strip():
            forest_type = str(explicit_forest_type).strip().lower()
            available = set(get_available_forest_types())
            if forest_type not in available:
                raise ValueError(
                    f"Unknown forest.type/biome profile '{forest_type}'. "
                    f"Available biome profiles: {', '.join(sorted(available))}"
                )
            selected_forest_types = [
                forest_type for _index in range(max(1, int(base_runtime_cfg.generation.num_scenes)))
            ]
            logger.info(
                "Using explicit forest.type='%s'; generation.forest_type_weights ignored.",
                forest_type,
            )
        else:
            selected_forest_types = _choose_forest_types_for_batch(
                base_runtime_cfg,
                scene_id_start=scene_id_start,
            )
        selected_scene_profiles = [(forest_type, "") for forest_type in selected_forest_types]

    for index, (forest_type, target_profile) in enumerate(selected_scene_profiles):
        scene_id = scene_id_start + index
        scene_cfg = load_tree_scatter_config_for_forest_and_target_profile(
            cfg,
            forest_type,
            target_profile,
        )
        scene_cfg.seed = int(base_runtime_cfg.seed) + scene_id - 1
        scene_cfg.output.output_dir = str(root_output_dir)
        _save_effective_scene_config(
            root_output_dir,
            scene_cfg,
            scene_id,
            save_legacy_root_config=(index == 0),
        )

        logger.info("-" * 50)
        if target_profile:
            logger.info(
                "Batch scene %s/%s | scene_id=%s | target_profile=%s | forest_type=%s | seed=%s",
                index + 1,
                len(selected_scene_profiles),
                scene_id,
                target_profile,
                forest_type,
                scene_cfg.seed,
            )
        else:
            logger.info(
                "Batch scene %s/%s | scene_id=%s | forest_type=%s | seed=%s",
                index + 1,
                len(selected_scene_profiles),
                scene_id,
                forest_type,
                scene_cfg.seed,
            )
        _run_single_scene(scene_cfg, root_output_dir, scene_id)
