from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from .botaniq_catalog import (
    _botaniq_asset_candidates_for_season,
    _BotaniqAsset,
    _choose_botaniq_asset_for_height_target,
    _load_botaniq_catalog,
    _load_botaniq_species_height_priors,
    _plan_botaniq_direct_pack_scales,
    _plan_botaniq_variant_height_scale,
    _resolve_botaniq_pack_dir,
    _sample_botaniq_target_height_m,
)
from .botaniq_composition import (
    _allocate_botaniq_variant_counts,
    _build_botaniq_stand_composition,
    _normalize_float_weights,
)
from .botaniq_leaf_geometry import (
    apply_anisotropic_scale_preserving_leaf_units,
    apply_position_transform_preserving_leaf_units,
)
from .botaniq_materials import (
    _apply_botaniq_color_adjustment_recipe,
    _apply_botaniq_leaf_density_recipe,
    _apply_botaniq_leaf_material_variation_recipe,
    _build_botaniq_color_adjustment_recipe,
    _build_botaniq_instance_leaf_material_recipe,
    _build_botaniq_leaf_density_recipe,
    _build_botaniq_leaf_material_variation_recipe,
    _set_botaniq_color_adjustment_properties,
)
from .botaniq_pool import (
    _botaniq_pool_catalog_from_manifest,
    _filtered_botaniq_pool_catalog,
    _load_pool_manifest,
    _positive_int_or_zero,
    load_botaniq_pool_manifest,
)
from .botaniq_quality import (
    botaniq_variant_quality_flags,
    botaniq_variant_quality_summary,
)
from .botaniq_utils import (
    coerce_float as _coerce_float,
)
from .botaniq_utils import (
    normalize_season_name as _normalize_season_name,
)
from .botaniq_utils import (
    resolve_project_path as _resolve_project_path,
)
from .botaniq_variants import (
    apply_botaniq_branch_group_crown_variation,
    apply_botaniq_branch_group_crown_variation_from_recipe,
    build_botaniq_branch_group_crown_recipe,
)
from .config import ScatterConfig
from .mesh_utils import infer_ground_anchor_local_z
from .mesh_utils import mesh_local_bounds as _mesh_local_bounds

logger = logging.getLogger(__name__)
BOTANIQ_VARIANT_RECIPE_SCHEMA = "synthforest.botaniq_variant.v1"


def _apply_botaniq_mesh_scale(
    tree_obj: bpy.types.Object,
    scale_xy: float,
    scale_z: float,
    cfg: ScatterConfig,
) -> None:
    apply_anisotropic_scale_preserving_leaf_units(tree_obj, scale_xy, scale_z, cfg)


def _refresh_variant_quality(
    tree_obj: bpy.types.Object,
    variant_recipe: dict[str, object],
    cfg: ScatterConfig,
    branch_variation_result: dict[str, object] | None = None,
) -> None:
    branch_recipe = variant_recipe.get("branch_variation", {})
    if isinstance(branch_recipe, dict) and isinstance(branch_variation_result, dict):
        try:
            branch_recipe["applied_max_displacement_m"] = float(
                branch_variation_result.get("max_displacement_m", 0.0)
            )
            branch_recipe["applied_vertices_changed"] = int(
                branch_variation_result.get("vertices_changed", 0)
            )
        except Exception:
            pass

    leaf_recipe = variant_recipe.get("leaf_density", {})
    if isinstance(leaf_recipe, dict):
        leaf_recipe["removed_face_fraction"] = float(
            tree_obj.get("botaniq_leaf_density_removed_face_fraction", 0.0)
        )
        leaf_recipe["removed_face_count"] = int(
            tree_obj.get("botaniq_leaf_density_removed_faces", 0)
        )
        leaf_recipe["removed_unit_count"] = int(
            tree_obj.get("botaniq_leaf_density_removed_units", 0)
        )

    flags = botaniq_variant_quality_flags(variant_recipe, cfg)
    variant_recipe["quality_flags"] = list(flags)
    tree_obj["botaniq_variant_quality_flags"] = ",".join(flags)
    tree_obj["botaniq_variant_quality_flag_count"] = int(len(flags))


@dataclass(frozen=True)
class _PoolSource:
    blend_path: Path
    collection_name: str
    resolved_season: str
    variant_count: int


@dataclass(frozen=True)
class _PlannedAssetVariant:
    variant_idx: int
    species_name: str
    object_name: str
    source_index: int
    species_role: str = "unknown"
    stand_abundance: float = 0.0
    role_abundance: float = 0.0
    variant_allocation_count: int = 0


@dataclass(frozen=True)
class _PlannedBotaniqVariant:
    variant_idx: int
    asset: _BotaniqAsset
    source_index: int
    resolved_season: str
    species_role: str
    stand_abundance: float
    role_abundance: float
    variant_allocation_count: int
    scale_xy: float
    scale_z: float
    target_height_m: float | None = None
    unclamped_target_height_m: float | None = None
    height_prior_min_m: float | None = None
    height_prior_mode_m: float | None = None
    height_prior_max_m: float | None = None
    height_prior_source: str = ""
    height_clamped: bool = False
    height_clamp_reason: str = ""
    scale_anisotropy: float = 1.0


def _resolve_species_pool_sources(
    species_list: list[str],
    pool_dir: Path,
    requested_season: str,
) -> tuple[dict[str, _PoolSource], dict[str, str]]:
    manifest_entries = _load_pool_manifest(pool_dir)
    species_sources: dict[str, _PoolSource] = {}
    resolved_seasons: dict[str, str] = {}
    requested_season = _normalize_season_name(requested_season)

    for species_name in species_list:
        manifest_entry = manifest_entries.get(species_name, {})
        pools = manifest_entry.get("pools", {}) if isinstance(manifest_entry, dict) else {}
        season_entry = pools.get(requested_season) if isinstance(pools, dict) else None
        resolved_season = requested_season
        if not isinstance(season_entry, dict):
            season_entry = pools.get("summer") if isinstance(pools, dict) else None
            resolved_season = "summer"
        if not isinstance(season_entry, dict) and isinstance(pools, dict) and pools:
            resolved_season, season_entry = next(
                (
                    (_normalize_season_name(str(season_name)), pool_entry)
                    for season_name, pool_entry in pools.items()
                    if isinstance(pool_entry, dict)
                ),
                (resolved_season, None),
            )

        if isinstance(season_entry, dict):
            asset_file = str(season_entry.get("asset_file", "")).strip()
            collection_name = str(season_entry.get("collection", "")).strip()
            variant_count = _positive_int_or_zero(season_entry.get("variant_count"))
            object_season = _asset_season_label(
                season_entry.get(
                    "object_season",
                    season_entry.get("season", resolved_season),
                )
            )
        else:
            asset_file = f"{species_name}.blend"
            collection_name = f"TreeAssets_{species_name}"
            variant_count = 0
            object_season = resolved_season

        blend_path = (pool_dir / asset_file).resolve()
        if not blend_path.is_file():
            raise FileNotFoundError(
                f"Asset pool file not found for species '{species_name}': {blend_path}"
            )
        if variant_count <= 0:
            variant_count = _infer_variant_count_from_library(
                blend_path,
                species_name,
                object_season,
            )
            if variant_count > 0:
                logger.info(
                    "Inferred asset variant_count=%s for species '%s' season '%s' from %s",
                    variant_count,
                    species_name,
                    object_season,
                    blend_path.name,
                )
        species_sources[species_name] = _PoolSource(
            blend_path=blend_path,
            collection_name=collection_name,
            resolved_season=object_season,
            variant_count=variant_count,
        )
        resolved_seasons[species_name] = object_season

    return species_sources, resolved_seasons


def _load_collection_from_library(
    blend_path: Path,
    collection_name: str,
) -> bpy.types.Collection:
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        if collection_name not in data_from.collections:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in asset file: {blend_path}"
            )
        data_to.collections = [collection_name]

    loaded_collection = data_to.collections[0]
    if loaded_collection is None:
        raise RuntimeError(
            f"Failed to load collection '{collection_name}' from asset file: {blend_path}"
        )
    return loaded_collection


def _asset_object_name(species_name: str, season: str, variant_idx: int) -> str:
    return f"{species_name}_{season}_asset_{int(variant_idx):03d}"


def _asset_season_label(season: object) -> str:
    label = "_".join(str(season or "").strip().lower().replace("-", "_").split())
    return label or "summer"


def _canonical_asset_object_name(obj: bpy.types.Object) -> str:
    species_name = str(obj.get("species_name", "")).strip().lower()
    season = _asset_season_label(obj.get("season", "summer"))
    variant_index = obj.get("variant_index")
    if species_name and variant_index is not None:
        try:
            return _asset_object_name(species_name, season, int(variant_index))
        except Exception:
            pass

    object_name = str(obj.name)
    if len(object_name) > 4 and object_name[-4] == "." and object_name[-3:].isdigit():
        return object_name[:-4]
    return object_name


def _infer_variant_count_from_library(
    blend_path: Path,
    species_name: str,
    season: str,
) -> int:
    prefix = f"{species_name}_{season}_asset_"
    try:
        with bpy.data.libraries.load(str(blend_path), link=True) as (data_from, _data_to):
            names = list(data_from.objects)
    except Exception as exc:
        logger.warning("Failed to inspect asset library %s for variant count: %s", blend_path, exc)
        return 0

    max_index = -1
    for name in names:
        if not str(name).startswith(prefix):
            continue
        suffix = str(name)[len(prefix) :]
        try:
            max_index = max(max_index, int(suffix))
        except ValueError:
            continue
    return max_index + 1


def _load_objects_from_library(
    blend_path: Path,
    object_names: list[str],
    *,
    link: bool = True,
) -> dict[str, bpy.types.Object]:
    requested_names = tuple(dict.fromkeys(object_names))
    with bpy.data.libraries.load(str(blend_path), link=link) as (data_from, data_to):
        available = set(data_from.objects)
        missing = [name for name in requested_names if name not in available]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            raise RuntimeError(f"Asset objects not found in {blend_path.name}: {preview}")
        data_to.objects = list(requested_names)

    returned_objects = list(data_to.objects)
    if len(returned_objects) != len(requested_names):
        raise RuntimeError(
            f"Requested {len(requested_names)} objects from {blend_path.name}, "
            f"but Blender returned {len(returned_objects)}."
        )

    loaded: dict[str, bpy.types.Object] = {}
    for requested_name, obj in zip(requested_names, returned_objects):
        if obj is None:
            raise RuntimeError(f"Failed to load asset object '{requested_name}' from {blend_path}")
        loaded[requested_name] = obj
    return loaded


def _mesh_objects_from_collection(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    return [obj for obj in collection.all_objects if obj.type == "MESH"]


def _cleanup_unlinked_asset_sources(
    objects: list[bpy.types.Object],
    collections: list[bpy.types.Collection],
) -> None:
    for collection in collections:
        try:
            bpy.data.collections.remove(collection)
        except Exception:
            pass

    for obj in objects:
        try:
            bpy.data.objects.remove(obj)
        except Exception:
            pass


def _build_botaniq_shape_jitter_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> dict[str, object]:
    probability = max(
        0.0,
        min(1.0, float(getattr(cfg, "botaniq_variant_shape_jitter_probability", 0.75))),
    )
    max_bend_fraction = max(
        0.0, float(getattr(cfg, "botaniq_variant_crown_bend_max_fraction", 0.035))
    )
    max_asymmetry = max(0.0, float(getattr(cfg, "botaniq_variant_crown_asymmetry_max", 0.08)))
    protected_fraction = max(
        0.0,
        min(0.95, float(getattr(cfg, "botaniq_variant_trunk_protected_fraction", 0.15))),
    )

    if rng.random() > probability or (max_bend_fraction <= 0.0 and max_asymmetry <= 0.0):
        return {
            "enabled": False,
            "reason": "skipped_probability_or_zero_strength",
            "bend_fraction": 0.0,
            "asymmetry": 0.0,
            "trunk_protected_fraction": float(protected_fraction),
        }

    min_v, max_v = _mesh_local_bounds(tree_obj)
    height = max(0.0, float(max_v.z - min_v.z))
    if height <= 1e-6:
        return {
            "enabled": False,
            "reason": "empty_height",
            "bend_fraction": 0.0,
            "asymmetry": 0.0,
            "trunk_protected_fraction": float(protected_fraction),
        }

    bend_fraction = rng.uniform(0.0, max_bend_fraction)
    bend_angle = rng.uniform(0.0, math.tau)
    asymmetry = rng.uniform(-max_asymmetry, max_asymmetry)
    asymmetry_angle = rng.uniform(0.0, math.tau)
    return {
        "enabled": True,
        "reason": "planned",
        "bend_fraction": float(bend_fraction),
        "bend_direction_rad": float(bend_angle),
        "asymmetry": float(asymmetry),
        "asymmetry_direction_rad": float(asymmetry_angle),
        "trunk_protected_fraction": float(protected_fraction),
    }


def _apply_botaniq_shape_jitter_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, object],
    cfg: ScatterConfig | None = None,
) -> None:
    if not bool(recipe.get("enabled", False)):
        tree_obj["botaniq_shape_bend_fraction"] = 0.0
        tree_obj["botaniq_shape_asymmetry"] = 0.0
        tree_obj["botaniq_shape_jitter_reason"] = str(recipe.get("reason", "disabled"))
        return

    min_v, max_v = _mesh_local_bounds(tree_obj)
    height = max(0.0, float(max_v.z - min_v.z))
    if height <= 1e-6:
        tree_obj["botaniq_shape_bend_fraction"] = 0.0
        tree_obj["botaniq_shape_asymmetry"] = 0.0
        tree_obj["botaniq_shape_jitter_reason"] = "empty_height"
        return

    protected_fraction = max(
        0.0,
        min(0.95, float(recipe.get("trunk_protected_fraction", 0.15))),
    )
    crown_start_z = float(min_v.z) + height * protected_fraction
    crown_height = max(1e-6, float(max_v.z) - crown_start_z)
    bend_fraction = float(recipe.get("bend_fraction", 0.0))
    bend_angle = float(recipe.get("bend_direction_rad", 0.0))
    bend_xy = Vector(
        (
            math.cos(bend_angle) * height * bend_fraction,
            math.sin(bend_angle) * height * bend_fraction,
        )
    )

    asymmetry = float(recipe.get("asymmetry", 0.0))
    asymmetry_angle = float(recipe.get("asymmetry_direction_rad", 0.0))
    axis_u = Vector((math.cos(asymmetry_angle), math.sin(asymmetry_angle)))
    axis_v = Vector((-math.sin(asymmetry_angle), math.cos(asymmetry_angle)))
    scale_u = max(0.2, 1.0 + asymmetry)
    scale_v = max(0.2, 1.0 - asymmetry)

    def transform_coord(co: Vector) -> Vector:
        t = max(0.0, min(1.0, (float(co.z) - crown_start_z) / crown_height))
        if t <= 0.0:
            return co.copy()
        weight = t * t

        xy = Vector((float(co.x), float(co.y)))
        target_xy = axis_u * (xy.dot(axis_u) * scale_u) + axis_v * (xy.dot(axis_v) * scale_v)
        varied_xy = xy + (target_xy - xy) * weight + bend_xy * weight
        return Vector((float(varied_xy.x), float(varied_xy.y), float(co.z)))

    if not apply_position_transform_preserving_leaf_units(tree_obj, cfg, transform_coord):
        for vertex in tree_obj.data.vertices:
            vertex.co = transform_coord(vertex.co.copy())

    tree_obj.data.update()
    tree_obj["botaniq_shape_bend_fraction"] = float(bend_fraction)
    tree_obj["botaniq_shape_bend_direction_rad"] = float(bend_angle)
    tree_obj["botaniq_shape_asymmetry"] = float(asymmetry)
    tree_obj["botaniq_shape_asymmetry_direction_rad"] = float(asymmetry_angle)
    tree_obj["botaniq_shape_jitter_reason"] = str(recipe.get("reason", "planned"))


def _apply_botaniq_shape_jitter(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> None:
    _apply_botaniq_shape_jitter_recipe(
        tree_obj,
        _build_botaniq_shape_jitter_recipe(tree_obj, rng, cfg),
        cfg,
    )


def _build_botaniq_variant_tilt_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> dict[str, object]:
    probability = max(0.0, min(1.0, float(getattr(cfg, "botaniq_variant_tilt_probability", 0.85))))
    max_tilt_deg = max(0.0, float(getattr(cfg, "botaniq_variant_tilt_max_deg", 4.0)))
    if rng.random() > probability or max_tilt_deg <= 0.0:
        return {
            "enabled": False,
            "reason": "skipped_probability_or_zero_strength",
            "tilt_deg": 0.0,
        }

    tilt_deg = rng.uniform(0.0, max_tilt_deg)
    direction = rng.uniform(0.0, math.tau)
    return {
        "enabled": True,
        "reason": "planned",
        "tilt_deg": float(tilt_deg),
        "direction_rad": float(direction),
    }


def _apply_botaniq_variant_tilt_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, object],
) -> None:
    if not bool(recipe.get("enabled", False)):
        tree_obj["botaniq_variant_tilt_deg"] = 0.0
        tree_obj["botaniq_variant_tilt_reason"] = str(recipe.get("reason", "disabled"))
        return

    tilt_deg = float(recipe.get("tilt_deg", 0.0))
    tilt_rad = math.radians(tilt_deg)
    direction = float(recipe.get("direction_rad", 0.0))
    axis = Vector((-math.sin(direction), math.cos(direction), 0.0))
    if axis.length <= 1e-8:
        tree_obj["botaniq_variant_tilt_deg"] = 0.0
        tree_obj["botaniq_variant_tilt_reason"] = "invalid_axis"
        return

    pivot_z = float(infer_ground_anchor_local_z(tree_obj))
    transform = (
        Matrix.Translation((0.0, 0.0, pivot_z))
        @ Matrix.Rotation(tilt_rad, 4, axis.normalized())
        @ Matrix.Translation((0.0, 0.0, -pivot_z))
    )
    tree_obj.data.transform(transform)
    tree_obj.data.update()

    tree_obj["botaniq_variant_tilt_deg"] = float(tilt_deg)
    tree_obj["botaniq_variant_tilt_direction_rad"] = float(direction)
    tree_obj["botaniq_variant_tilt_reason"] = str(recipe.get("reason", "planned"))


def _apply_botaniq_variant_tilt(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> None:
    _apply_botaniq_variant_tilt_recipe(
        tree_obj,
        _build_botaniq_variant_tilt_recipe(tree_obj, rng, cfg),
    )


def _stage_is_authoring(cfg: ScatterConfig) -> bool:
    stage = str(getattr(cfg, "botaniq_asset_pool_stage", "production") or "production")
    return stage.strip().lower() in {"authoring", "editable", "live"}


def _configured_variant_recipes(cfg: ScatterConfig) -> dict[int, dict[str, object]]:
    raw_recipes = getattr(cfg, "botaniq_variant_recipes", None)
    if isinstance(raw_recipes, dict):
        recipes = raw_recipes.get("variants", raw_recipes)
        if isinstance(recipes, list):
            return {
                int(recipe.get("variant_index", index)): recipe
                for index, recipe in enumerate(recipes)
                if isinstance(recipe, dict)
            }
        return {int(key): value for key, value in recipes.items() if isinstance(value, dict)}
    if isinstance(raw_recipes, list):
        return {
            int(recipe.get("variant_index", index)): recipe
            for index, recipe in enumerate(raw_recipes)
            if isinstance(recipe, dict)
        }
    return {}


def _build_botaniq_variant_recipe(
    tree_obj: bpy.types.Object,
    plan_item: _PlannedBotaniqVariant,
    rng: random.Random,
    cfg: ScatterConfig,
) -> dict[str, object]:
    height_recipe: dict[str, object] = {
        "enabled": plan_item.target_height_m is not None,
        "source_height_m": float(plan_item.asset.height_m),
        "scale_z": float(plan_item.scale_z),
        "scale_xy": float(plan_item.scale_xy),
        "anisotropy": float(plan_item.scale_anisotropy),
        "clamped": bool(plan_item.height_clamped),
        "clamp_reason": str(plan_item.height_clamp_reason),
        "crown_scale_power": float(getattr(cfg, "botaniq_height_to_crown_scale_power", 0.65)),
    }
    if plan_item.target_height_m is None:
        height_recipe["reason"] = "no_species_height_prior"
    else:
        height_recipe.update(
            {
                "target_height_m": float(plan_item.target_height_m),
                "unclamped_target_height_m": float(
                    plan_item.unclamped_target_height_m
                    if plan_item.unclamped_target_height_m is not None
                    else plan_item.target_height_m
                ),
                "height_min_m": float(plan_item.height_prior_min_m or 0.0),
                "height_mode_m": float(plan_item.height_prior_mode_m or 0.0),
                "height_max_m": float(plan_item.height_prior_max_m or 0.0),
                "prior_source": plan_item.height_prior_source,
            }
        )

    base_recipe: dict[str, object] = {
        "schema": BOTANIQ_VARIANT_RECIPE_SCHEMA,
        "variant_index": int(plan_item.variant_idx),
        "species_label": plan_item.asset.species_label,
        "species_key": plan_item.asset.species_key,
        "season": plan_item.resolved_season,
        "source_asset": {
            "asset_id": plan_item.asset.asset_id,
            "title": plan_item.asset.title,
            "blend_path": str(plan_item.asset.blend_path),
            "source_index": int(plan_item.source_index),
        },
        "stand": {
            "role": plan_item.species_role,
            "stand_abundance": float(plan_item.stand_abundance),
            "role_abundance": float(plan_item.role_abundance),
            "variant_allocation_count": int(plan_item.variant_allocation_count),
        },
        "scale": {
            "xy": float(plan_item.scale_xy),
            "z": float(plan_item.scale_z),
            "anisotropy": float(plan_item.scale_anisotropy),
        },
        "height": height_recipe,
        "leaf_density": _build_botaniq_leaf_density_recipe(tree_obj, rng, cfg),
        "color": _build_botaniq_color_adjustment_recipe(tree_obj, rng, cfg),
    }

    attempts = max(1, int(getattr(cfg, "botaniq_variant_quality_resample_attempts", 4)))
    resampleable_flags = {
        "branch_displacement_too_high",
        "branch_swap_pairs",
        "tilt_too_high",
    }
    best_recipe: dict[str, object] | None = None
    best_flags: list[str] = []
    for attempt_index in range(attempts):
        candidate = dict(base_recipe)
        candidate["branch_variation"] = build_botaniq_branch_group_crown_recipe(
            tree_obj,
            rng,
            cfg,
        )
        candidate["shape_jitter"] = _build_botaniq_shape_jitter_recipe(tree_obj, rng, cfg)
        candidate["tilt"] = _build_botaniq_variant_tilt_recipe(tree_obj, rng, cfg)
        flags = botaniq_variant_quality_flags(candidate, cfg)
        candidate["quality_flags"] = list(flags)
        candidate["quality_attempts"] = int(attempt_index + 1)
        if best_recipe is None or len(flags) < len(best_flags):
            best_recipe = candidate
            best_flags = list(flags)
        if not any(flag in resampleable_flags for flag in flags):
            return candidate

    assert best_recipe is not None
    return best_recipe


def _copy_botaniq_mesh_variant(
    source_obj: bpy.types.Object,
    tree_col: bpy.types.Collection,
    plan_item: _PlannedBotaniqVariant,
    rng: random.Random,
    cfg: ScatterConfig,
) -> bpy.types.Object:
    if source_obj.type != "MESH" or source_obj.data is None:
        raise RuntimeError(
            f"Botaniq source object '{source_obj.name}' is not a mesh and cannot be scattered."
        )

    tree_obj = source_obj.copy()
    tree_obj.data = source_obj.data.copy()
    tree_obj.name = f"BotaniqTree_{plan_item.variant_idx:03d}"
    tree_obj["preset_name"] = plan_item.asset.species_label
    tree_obj["species_name"] = plan_item.asset.species_label
    tree_obj["species_common_name"] = plan_item.asset.species_common_name
    tree_obj["botaniq_asset_id"] = plan_item.asset.asset_id
    tree_obj["botaniq_asset_title"] = plan_item.asset.title
    tree_obj["asset_source_name"] = source_obj.name
    tree_obj["season"] = plan_item.resolved_season
    tree_obj["botaniq_height_m"] = float(plan_item.asset.height_m)
    tree_obj["botaniq_width_m"] = float(plan_item.asset.width_m)
    tree_obj["botaniq_depth_m"] = float(plan_item.asset.depth_m)
    tree_obj["botaniq_model_detail"] = plan_item.asset.model_detail
    tree_obj["stand_species_role"] = plan_item.species_role
    tree_obj["stand_species_abundance"] = float(plan_item.stand_abundance)
    tree_obj["stand_role_abundance"] = float(plan_item.role_abundance)
    tree_obj["stand_variant_allocation_count"] = int(plan_item.variant_allocation_count)

    # Bake source transform and planned variant scale into the local mesh so
    # Geometry Nodes sees each copied child as a distinct tree template.
    tree_obj.data.transform(tree_obj.matrix_basis)
    tree_obj.location = (0.0, 0.0, 0.0)
    tree_obj.rotation_euler = (0.0, 0.0, 0.0)
    tree_obj.scale = (1.0, 1.0, 1.0)
    configured_recipes = _configured_variant_recipes(cfg)
    configured_recipe = configured_recipes.get(int(plan_item.variant_idx))
    recipe_scale = configured_recipe.get("scale", {}) if isinstance(configured_recipe, dict) else {}
    scale_xy = (
        float(recipe_scale.get("xy", plan_item.scale_xy))
        if isinstance(recipe_scale, dict)
        else float(plan_item.scale_xy)
    )
    scale_z = (
        float(recipe_scale.get("z", plan_item.scale_z))
        if isinstance(recipe_scale, dict)
        else float(plan_item.scale_z)
    )
    scale_anisotropy = float(plan_item.scale_anisotropy)
    if isinstance(recipe_scale, dict):
        scale_anisotropy = float(
            recipe_scale.get(
                "anisotropy",
                scale_z / max(scale_xy, 1e-6),
            )
        )
    recipe_height = (
        configured_recipe.get("height", {}) if isinstance(configured_recipe, dict) else {}
    )
    target_height_m = plan_item.target_height_m
    unclamped_target_height_m = plan_item.unclamped_target_height_m
    height_prior_min_m = plan_item.height_prior_min_m
    height_prior_mode_m = plan_item.height_prior_mode_m
    height_prior_max_m = plan_item.height_prior_max_m
    height_prior_source = plan_item.height_prior_source
    height_clamped = bool(plan_item.height_clamped)
    height_clamp_reason = str(plan_item.height_clamp_reason)
    if isinstance(recipe_height, dict):
        if recipe_height.get("target_height_m") is not None:
            recipe_target = _coerce_float(recipe_height.get("target_height_m"), 0.0)
            target_height_m = recipe_target if recipe_target > 0.0 else target_height_m
        if recipe_height.get("unclamped_target_height_m") is not None:
            recipe_unclamped = _coerce_float(
                recipe_height.get("unclamped_target_height_m"),
                0.0,
            )
            unclamped_target_height_m = (
                recipe_unclamped if recipe_unclamped > 0.0 else unclamped_target_height_m
            )
        if recipe_height.get("height_min_m") is not None:
            height_prior_min_m = _coerce_float(recipe_height.get("height_min_m"), 0.0)
        if recipe_height.get("height_mode_m") is not None:
            height_prior_mode_m = _coerce_float(recipe_height.get("height_mode_m"), 0.0)
        if recipe_height.get("height_max_m") is not None:
            height_prior_max_m = _coerce_float(recipe_height.get("height_max_m"), 0.0)
        height_prior_source = str(recipe_height.get("prior_source", height_prior_source))
        height_clamped = bool(recipe_height.get("clamped", height_clamped))
        height_clamp_reason = str(recipe_height.get("clamp_reason", height_clamp_reason))
        scale_anisotropy = _coerce_float(
            recipe_height.get("anisotropy"),
            scale_anisotropy,
        )
    _apply_botaniq_mesh_scale(tree_obj, scale_xy, scale_z, cfg)
    tree_col.objects.link(tree_obj)

    variant_recipe = (
        dict(configured_recipe)
        if isinstance(configured_recipe, dict)
        else _build_botaniq_variant_recipe(tree_obj, plan_item, rng, cfg)
    )
    authoring_stage = _stage_is_authoring(cfg)
    leaf_density_recipe = variant_recipe.get("leaf_density", {})
    if not isinstance(leaf_density_recipe, dict) or not leaf_density_recipe:
        leaf_density_recipe = _build_botaniq_leaf_density_recipe(tree_obj, rng, cfg)
        variant_recipe["leaf_density"] = leaf_density_recipe
    _apply_botaniq_leaf_density_recipe(tree_obj, leaf_density_recipe, cfg)
    leaf_density_recipe["removed_face_count"] = int(
        tree_obj.get("botaniq_leaf_density_removed_faces", 0)
    )
    leaf_density_recipe["removed_unit_count"] = int(
        tree_obj.get("botaniq_leaf_density_removed_units", 0)
    )
    leaf_density_recipe["removed_face_fraction"] = float(
        tree_obj.get("botaniq_leaf_density_removed_face_fraction", 0.0)
    )

    branch_recipe = variant_recipe.get("branch_variation", {})
    if isinstance(branch_recipe, dict):
        modifier_layout = (
            str(getattr(cfg, "botaniq_authoring_modifier_layout", "per_branch"))
            if authoring_stage
            else str(getattr(cfg, "botaniq_branch_geometry_nodes_layout", "combined"))
        )
        branch_variation_result = apply_botaniq_branch_group_crown_variation_from_recipe(
            tree_obj,
            branch_recipe,
            cfg,
            apply_modifier=(
                False
                if authoring_stage
                else bool(getattr(cfg, "botaniq_branch_geometry_nodes_apply", True))
            ),
            modifier_layout=modifier_layout,
            mask_attribute_prefix=(
                "fg_bq_authoring_mask_{group}" if authoring_stage else "fg_bq_branch_mask"
            ),
            keep_mask_attributes=authoring_stage,
        )
    else:
        branch_variation_result = apply_botaniq_branch_group_crown_variation(tree_obj, rng, cfg)
    tree_obj["botaniq_branch_variation_source"] = "editable_vertex_groups"
    tree_obj["botaniq_branch_variation_backend"] = str(branch_variation_result.get("mode", ""))
    tree_obj["botaniq_branch_variation_pre_shape_jitter_applied"] = bool(
        branch_variation_result.get("applied", False)
    )

    apply_global_deforms = not authoring_stage or bool(
        getattr(cfg, "botaniq_authoring_apply_global_deforms", False)
    )
    shape_recipe = variant_recipe.get("shape_jitter", {})
    if apply_global_deforms and isinstance(shape_recipe, dict):
        _apply_botaniq_shape_jitter_recipe(tree_obj, shape_recipe, cfg)
    elif apply_global_deforms:
        _apply_botaniq_shape_jitter(tree_obj, rng, cfg)
    else:
        tree_obj["botaniq_shape_jitter_reason"] = "stored_in_recipe_for_production"
        tree_obj["botaniq_shape_bend_fraction"] = float(
            shape_recipe.get("bend_fraction", 0.0) if isinstance(shape_recipe, dict) else 0.0
        )
        tree_obj["botaniq_shape_asymmetry"] = float(
            shape_recipe.get("asymmetry", 0.0) if isinstance(shape_recipe, dict) else 0.0
        )

    tilt_recipe = variant_recipe.get("tilt", {})
    if apply_global_deforms and isinstance(tilt_recipe, dict):
        _apply_botaniq_variant_tilt_recipe(tree_obj, tilt_recipe)
    elif apply_global_deforms:
        _apply_botaniq_variant_tilt(tree_obj, rng, cfg)
    else:
        tree_obj["botaniq_variant_tilt_reason"] = "stored_in_recipe_for_production"
        tree_obj["botaniq_variant_tilt_deg"] = float(
            tilt_recipe.get("tilt_deg", 0.0) if isinstance(tilt_recipe, dict) else 0.0
        )

    color_recipe = variant_recipe.get("color", {})
    if isinstance(color_recipe, dict):
        _apply_botaniq_color_adjustment_recipe(tree_obj, color_recipe, cfg)
    else:
        _set_botaniq_color_adjustment_properties(tree_obj, rng, cfg)

    anchor_z = float(infer_ground_anchor_local_z(tree_obj))
    tree_obj["ground_anchor_z_local"] = anchor_z
    tree_obj.location = (0.0, 0.0, -anchor_z)

    min_v, max_v = _mesh_local_bounds(tree_obj)
    realized_width = max(float(max_v.x - min_v.x), 0.0)
    realized_depth = max(float(max_v.y - min_v.y), 0.0)
    realized_height = max(float(max_v.z - min_v.z), 0.0)
    tree_obj["realized_crown_diameter_m"] = max(realized_width, realized_depth)
    tree_obj["realized_height_m"] = realized_height
    tree_obj["fit_scale_xy"] = float(scale_xy)
    tree_obj["fit_scale_z"] = float(scale_z)
    if target_height_m is not None:
        tree_obj["botaniq_target_height_m"] = float(target_height_m)
        if unclamped_target_height_m is not None:
            tree_obj["botaniq_unclamped_target_height_m"] = float(unclamped_target_height_m)
        tree_obj["botaniq_height_clamped"] = bool(height_clamped)
        tree_obj["botaniq_height_clamp_reason"] = str(height_clamp_reason)
        tree_obj["botaniq_height_prior_min_m"] = float(height_prior_min_m or 0.0)
        tree_obj["botaniq_height_prior_mode_m"] = float(height_prior_mode_m or 0.0)
        tree_obj["botaniq_height_prior_max_m"] = float(height_prior_max_m or 0.0)
        tree_obj["botaniq_height_prior_source"] = str(height_prior_source)
    tree_obj["fit_scale_anisotropy"] = float(scale_anisotropy)
    recipe_scale = variant_recipe.get("scale", {})
    if isinstance(recipe_scale, dict):
        recipe_scale.setdefault("xy", float(scale_xy))
        recipe_scale.setdefault("z", float(scale_z))
        recipe_scale["anisotropy"] = float(scale_anisotropy)
    recipe_height = variant_recipe.get("height", {})
    if isinstance(recipe_height, dict):
        recipe_height["scale_xy"] = float(scale_xy)
        recipe_height["scale_z"] = float(scale_z)
        recipe_height["anisotropy"] = float(scale_anisotropy)
        recipe_height["clamped"] = bool(height_clamped)
        recipe_height["clamp_reason"] = str(height_clamp_reason)
        if target_height_m is not None:
            recipe_height["target_height_m"] = float(target_height_m)
            recipe_height["unclamped_target_height_m"] = float(
                unclamped_target_height_m
                if unclamped_target_height_m is not None
                else target_height_m
            )
    tree_obj["botaniq_variant_recipe_schema"] = BOTANIQ_VARIANT_RECIPE_SCHEMA
    _refresh_variant_quality(tree_obj, variant_recipe, cfg, branch_variation_result)
    tree_obj["botaniq_variant_recipe_json"] = json.dumps(
        variant_recipe,
        sort_keys=True,
    )

    return tree_obj


def _map_plan_items_to_source_objects(
    species_name: str,
    plan_items: list[_PlannedAssetVariant],
    source_objects: list[bpy.types.Object],
    objects_by_name: dict[str, bpy.types.Object],
) -> dict[str, bpy.types.Object]:
    if not source_objects:
        raise RuntimeError(f"No mesh objects found in asset pool for species '{species_name}'")

    mapped: dict[str, bpy.types.Object] = {}
    for item in plan_items:
        obj = objects_by_name.get(item.object_name)
        if obj is None:
            obj = source_objects[item.source_index % len(source_objects)]
            logger.warning(
                "Selected asset '%s' for species '%s' was not found by name; "
                "using collection object at variant index %s.",
                item.object_name,
                species_name,
                item.source_index,
            )
        mapped[item.object_name] = obj
    return mapped


def _pop_asset_pool_source_index(
    species_name: str,
    species_sources: dict[str, _PoolSource],
    rng: random.Random,
    available_indices_by_species: dict[str, list[int]],
) -> int:
    source = species_sources[species_name]
    if source.variant_count <= 0:
        raise RuntimeError(
            "Asset pool manifest is missing variant_count for "
            f"species '{species_name}' season '{source.resolved_season}'. "
            "Regenerate the asset pool manifest before running selective loading."
        )
    available_indices = available_indices_by_species.get(species_name)
    if not available_indices:
        available_indices = list(range(source.variant_count))
        rng.shuffle(available_indices)
        available_indices_by_species[species_name] = available_indices
    return available_indices.pop()


def _plan_asset_variants_from_counts(
    species_counts: dict[str, int],
    species_sources: dict[str, _PoolSource],
    rng: random.Random,
    *,
    species_roles: dict[str, str] | None = None,
    stand_abundances: dict[str, float] | None = None,
    role_abundances: dict[str, float] | None = None,
) -> list[_PlannedAssetVariant]:
    planned_species: list[str] = []
    for species_name, count in species_counts.items():
        planned_species.extend([species_name] * max(0, int(count)))
    rng.shuffle(planned_species)

    available_indices_by_species: dict[str, list[int]] = {}
    plan: list[_PlannedAssetVariant] = []
    species_roles = species_roles or {}
    stand_abundances = stand_abundances or {}
    role_abundances = role_abundances or {}
    for variant_idx, species_name in enumerate(planned_species):
        source = species_sources[species_name]
        source_index = _pop_asset_pool_source_index(
            species_name,
            species_sources,
            rng,
            available_indices_by_species,
        )
        species_role = species_roles.get(species_name, "unknown")
        plan.append(
            _PlannedAssetVariant(
                variant_idx=variant_idx,
                species_name=species_name,
                object_name=_asset_object_name(
                    species_name,
                    source.resolved_season,
                    source_index,
                ),
                source_index=source_index,
                species_role=species_role,
                stand_abundance=float(stand_abundances.get(species_name, 0.0)),
                role_abundance=float(role_abundances.get(species_role, 0.0)),
                variant_allocation_count=int(species_counts.get(species_name, 0)),
            )
        )
    return plan


def create_tree_collection_from_botaniq_asset_pool(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    pool_dir = _resolve_project_path(str(getattr(cfg, "asset_pool_dir", "data/tree_assets")))
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Asset pool directory does not exist: {pool_dir}")
    manifest_entries = load_botaniq_pool_manifest(pool_dir)
    species_labels = {
        species_name: str(entry.get("species_label", species_name) or species_name)
        for species_name, entry in manifest_entries.items()
    }
    rng = random.Random(seed)
    variant_count = max(1, int(cfg.tree_variants))
    pool_catalog = _filtered_botaniq_pool_catalog(
        cfg,
        _botaniq_pool_catalog_from_manifest(pool_dir, manifest_entries),
    )
    if not pool_catalog:
        raise RuntimeError(
            "No Botaniq species are available from the prebaked asset pool. "
            "Check scatter.asset_pool_species and the pool manifest."
        )
    botaniq_composition = _build_botaniq_stand_composition(cfg, pool_catalog, rng)
    botaniq_variant_counts = _allocate_botaniq_variant_counts(
        botaniq_composition.species_keys,
        botaniq_composition.species_weights,
        variant_count,
    )
    requested = [
        species_key
        for species_key in botaniq_composition.species_keys
        if botaniq_variant_counts.get(species_key, 0) > 0
    ]
    botaniq_allocated_weights = _normalize_float_weights(
        {
            species_key: botaniq_composition.species_weights.get(species_key, 0.0)
            for species_key in requested
        }
    )
    dropped_species = [
        pool_catalog[species_key][0].species_label
        for species_key in botaniq_composition.species_keys
        if botaniq_variant_counts.get(species_key, 0) <= 0
    ]
    if dropped_species:
        logger.warning(
            "Dropped botaniq pool species because scatter.tree_variants=%s is too small: %s",
            variant_count,
            ", ".join(dropped_species),
        )

    species_sources, resolved_species_seasons = _resolve_species_pool_sources(
        requested,
        pool_dir,
        requested_season=season,
    )

    tree_col = bpy.data.collections.new("TreeTemplate")
    scene.collection.children.link(tree_col)

    plan = _plan_asset_variants_from_counts(
        botaniq_variant_counts,
        species_sources,
        rng,
        species_roles=botaniq_composition.species_roles,
        stand_abundances=botaniq_allocated_weights,
        role_abundances=botaniq_composition.role_abundances,
    )
    variant_species = [species_labels.get(item.species_name, item.species_name) for item in plan]
    variant_source_seasons = [
        resolved_species_seasons.get(item.species_name, "summer") for item in plan
    ]
    selected_species = sorted({item.species_name for item in plan})
    selected_object_count = len({(item.species_name, item.object_name) for item in plan})

    logger.info(
        "Asset pool mode enabled. Source=%s | species pool: %s | variants selected: %s",
        "botaniq_asset_pool",
        ", ".join(species_labels.get(species, species) for species in requested),
        variant_count,
    )
    logger.info(
        "Botaniq pool stand composition: %s",
        ", ".join(
            f"{species_labels.get(species_key, species_key)}"
            f"({botaniq_composition.species_roles.get(species_key, 'unknown')}:"
            f"{botaniq_allocated_weights.get(species_key, 0.0):.3f},"
            f"variants={botaniq_variant_counts.get(species_key, 0)})"
            for species_key in requested
        ),
    )
    logger.info("Asset pool requested season: %s", _normalize_season_name(season))
    logger.info(
        "Asset pool selective load plan: selected_species=%s | unique_assets=%s/%s",
        ", ".join(species_labels.get(species, species) for species in selected_species),
        selected_object_count,
        sum(max(0, species_sources[species].variant_count) for species in selected_species),
    )

    loaded_objects_by_species_name: dict[tuple[str, str], bpy.types.Object] = {}
    cleanup_objects: list[bpy.types.Object] = []
    cleanup_collections: list[bpy.types.Collection] = []
    scatter_leaf_material_variation_slots = 0
    scatter_leaf_material_variation_nodes = 0
    load_started = time.perf_counter()

    plan_by_species: dict[str, list[_PlannedAssetVariant]] = {
        species_name: [item for item in plan if item.species_name == species_name]
        for species_name in selected_species
    }

    for species_name in selected_species:
        source = species_sources[species_name]
        species_plan = plan_by_species[species_name]
        selected_names = [item.object_name for item in species_plan]
        unique_selected_names = list(dict.fromkeys(selected_names))
        species_load_started = time.perf_counter()
        load_mode = "linked_objects"

        try:
            loaded_objects = _load_objects_from_library(
                source.blend_path,
                unique_selected_names,
                link=True,
            )
            cleanup_objects.extend(loaded_objects.values())
        except Exception as exc:
            load_mode = "full_collection_fallback"
            logger.warning(
                "Selective asset object load failed for species '%s' from %s (%s); "
                "falling back to full collection load for this species.",
                species_name,
                source.blend_path.name,
                exc,
            )
            source_collection = _load_collection_from_library(
                blend_path=source.blend_path,
                collection_name=source.collection_name,
            )
            cleanup_collections.append(source_collection)
            source_objects = _mesh_objects_from_collection(source_collection)
            cleanup_objects.extend(source_objects)
            loaded_objects = _map_plan_items_to_source_objects(
                species_name,
                species_plan,
                source_objects,
                {_canonical_asset_object_name(obj): obj for obj in source_objects},
            )

        for object_name, obj in loaded_objects.items():
            loaded_objects_by_species_name[(species_name, object_name)] = obj
            if obj.type != "MESH":
                logger.warning(
                    "Loaded asset '%s' for species '%s' is type '%s' instead of MESH; "
                    "using it as-is.",
                    object_name,
                    species_name,
                    obj.type,
                )

        logger.info(
            "Asset pool loaded species '%s' | selected_assets=%s | mode=%s | file=%s | elapsed=%.1fs",
            species_labels.get(species_name, species_name),
            len(unique_selected_names),
            load_mode,
            source.blend_path.name,
            time.perf_counter() - species_load_started,
        )

    for item in plan:
        chosen_species = item.species_name
        source_obj = loaded_objects_by_species_name.get((chosen_species, item.object_name))
        if source_obj is None:
            raise RuntimeError(
                f"Selected asset '{item.object_name}' for species '{chosen_species}' "
                "was not available after asset loading."
            )

        tree_obj = source_obj.copy()

        species_label = species_labels.get(chosen_species, chosen_species)
        tree_obj.name = f"PoolTree_{item.variant_idx:03d}"
        tree_obj["preset_name"] = species_label
        tree_obj["species_name"] = chosen_species
        tree_obj["species_label"] = species_label
        tree_obj["asset_source_name"] = source_obj.name
        tree_obj["season"] = resolved_species_seasons.get(chosen_species, "summer")
        tree_obj["stand_species_role"] = item.species_role
        tree_obj["stand_species_abundance"] = float(item.stand_abundance)
        tree_obj["stand_role_abundance"] = float(item.role_abundance)
        tree_obj["stand_variant_allocation_count"] = int(item.variant_allocation_count)
        if bool(getattr(cfg, "botaniq_scatter_leaf_material_variation_enabled", True)):
            scatter_color_recipe = _build_botaniq_leaf_material_variation_recipe(
                tree_obj,
                rng,
                cfg,
                scope="scatter",
            )
            _apply_botaniq_leaf_material_variation_recipe(tree_obj, scatter_color_recipe)
            tree_obj["botaniq_scatter_leaf_material_recipe_json"] = json.dumps(
                scatter_color_recipe,
                sort_keys=True,
            )
            scatter_leaf_material_variation_slots += int(
                tree_obj.get("botaniq_leaf_material_variation_slots", 0)
            )
            scatter_leaf_material_variation_nodes += int(
                tree_obj.get("botaniq_leaf_material_variation_nodes", 0)
            )

        stored_anchor_z = tree_obj.get("ground_anchor_z_local")
        if stored_anchor_z is None:
            anchor_z = float(infer_ground_anchor_local_z(tree_obj))
        else:
            anchor_z = float(stored_anchor_z)
        tree_obj["ground_anchor_z_local"] = anchor_z
        tree_obj.location = (0.0, 0.0, -anchor_z)

        tree_col.objects.link(tree_obj)

    _cleanup_unlinked_asset_sources(cleanup_objects, cleanup_collections)

    logger.info(
        "Species distribution in generated variants: %s",
        ", ".join(
            f"{species_labels.get(species, species)}="
            f"{variant_species.count(species_labels.get(species, species))}"
            for species in requested
        ),
    )
    logger.info(
        "Resolved species pool seasons: %s",
        ", ".join(
            f"{species}={resolved_species_seasons.get(species, 'summer')}" for species in requested
        ),
    )
    logger.info(
        "Asset pool variants prepared in %.1fs",
        time.perf_counter() - load_started,
    )
    logger.info(
        "Botaniq asset-pool scatter: using prebaked mesh heights; "
        "species height priors are not reapplied at scatter time."
    )
    logger.info(
        "Botaniq scatter leaf-material variation: slots=%s | nodes=%s",
        scatter_leaf_material_variation_slots,
        scatter_leaf_material_variation_nodes,
    )

    collection_metadata = {
        "requested_season": _normalize_season_name(season),
        "resolved_species_seasons": dict(resolved_species_seasons),
        "variant_source_seasons": variant_source_seasons,
        "asset_pool_dir": str(pool_dir),
        "asset_pool_source": "botaniq_asset_pool",
        "asset_pool_species_selected": [
            species_labels.get(species, species) for species in requested
        ],
        "botaniq_height_priors_applied_at_scatter": False,
        "botaniq_height_priors_expected_stage": ("asset_pool_prebake"),
        "botaniq_scatter_leaf_material_variation_slots": int(scatter_leaf_material_variation_slots),
        "botaniq_scatter_leaf_material_variation_nodes": int(scatter_leaf_material_variation_nodes),
    }
    collection_metadata.update(
        {
            "botaniq_species_selected": [
                species_labels.get(species, species) for species in requested
            ],
            "botaniq_species_abundances": {
                species_labels.get(species, species): float(
                    botaniq_allocated_weights.get(species, 0.0)
                )
                for species in requested
            },
            "botaniq_species_roles": {
                species_labels.get(species, species): botaniq_composition.species_roles.get(
                    species,
                    "unknown",
                )
                for species in requested
            },
            "botaniq_role_abundances": dict(botaniq_composition.role_abundances),
            "botaniq_role_requested_counts": dict(botaniq_composition.role_requested_counts),
            "botaniq_role_selected_counts": dict(botaniq_composition.role_selected_counts),
            "botaniq_role_available_species": dict(botaniq_composition.role_available_species),
            "botaniq_variant_allocation": {
                species_labels.get(species, species): int(botaniq_variant_counts.get(species, 0))
                for species in requested
            },
        }
    )
    return tree_col, variant_species, collection_metadata


def _create_tree_collection_from_botaniq_asset_pack_simple(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    """Load raw botaniq pack assets with height-prior scaling only."""
    pack_dir = _resolve_botaniq_pack_dir(cfg)
    include_title_prefixes = [
        str(prefix) for prefix in getattr(cfg, "botaniq_include_title_prefixes", ["Tree "])
    ]
    catalog = _load_botaniq_catalog(pack_dir, include_title_prefixes)
    if not catalog:
        raise RuntimeError(
            "No botaniq model assets matched scatter.botaniq_include_title_prefixes "
            f"in {pack_dir / 'mapr_index.json'}"
        )

    rng = random.Random(seed)
    requested_season = _normalize_season_name(season)
    composition = _build_botaniq_stand_composition(cfg, catalog, rng)
    variant_count = max(1, int(cfg.tree_variants))
    variant_counts = _allocate_botaniq_variant_counts(
        composition.species_keys,
        composition.species_weights,
        variant_count,
    )
    allocated_species_keys = [
        species_key
        for species_key in composition.species_keys
        if variant_counts.get(species_key, 0) > 0
    ]
    if not allocated_species_keys:
        raise RuntimeError("No botaniq species were allocated to direct asset-pack templates.")

    allocated_weights = _normalize_float_weights(
        {
            species_key: composition.species_weights.get(species_key, 0.0)
            for species_key in allocated_species_keys
        }
    )
    dropped_species = [
        catalog[species_key][0].species_label
        for species_key in composition.species_keys
        if variant_counts.get(species_key, 0) <= 0
    ]
    if dropped_species:
        logger.warning(
            "Dropped botaniq asset-pack species because scatter.tree_variants=%s is too small: %s",
            variant_count,
            ", ".join(dropped_species),
        )

    available_indices_by_species: dict[str, list[int]] = {}

    def pop_asset_for_species(species_key: str) -> tuple[_BotaniqAsset, int, str]:
        assets = catalog[species_key]
        available_indices = available_indices_by_species.get(species_key)
        if not available_indices:
            seasonal_indices = [
                index for index, asset in enumerate(assets) if requested_season in asset.seasons
            ]
            if not seasonal_indices:
                seasonal_indices = [
                    index for index, asset in enumerate(assets) if "summer" in asset.seasons
                ]
            if not seasonal_indices:
                seasonal_indices = list(range(len(assets)))
            rng.shuffle(seasonal_indices)
            available_indices = seasonal_indices
            available_indices_by_species[species_key] = available_indices
        source_index = available_indices.pop()
        asset = assets[source_index]
        if requested_season in asset.seasons:
            resolved_season = requested_season
        elif "summer" in asset.seasons:
            resolved_season = "summer"
        else:
            resolved_season = asset.seasons[0] if asset.seasons else "summer"
        return asset, source_index, resolved_season

    planned_species_keys: list[str] = []
    for species_key in allocated_species_keys:
        planned_species_keys.extend([species_key] * variant_counts.get(species_key, 0))
    rng.shuffle(planned_species_keys)

    height_priors = _load_botaniq_species_height_priors(cfg)
    crown_scale_power = max(
        0.0,
        float(getattr(cfg, "botaniq_height_to_crown_scale_power", 0.65)),
    )

    plan: list[_PlannedBotaniqVariant] = []
    selected_assets: dict[str, _BotaniqAsset] = {}
    variant_species: list[str] = []
    variant_source_seasons: list[str] = []
    for variant_idx, species_key in enumerate(planned_species_keys):
        asset, source_index, resolved_season = pop_asset_for_species(species_key)
        species_role = composition.species_roles.get(species_key, "unknown")
        target_height_m, height_prior = _sample_botaniq_target_height_m(
            asset,
            height_priors,
            rng,
        )
        scale_xy, scale_z = _plan_botaniq_direct_pack_scales(
            asset,
            target_height_m,
            crown_scale_power=crown_scale_power,
        )
        plan.append(
            _PlannedBotaniqVariant(
                variant_idx=variant_idx,
                asset=asset,
                source_index=source_index,
                resolved_season=resolved_season,
                species_role=species_role,
                stand_abundance=float(allocated_weights.get(species_key, 0.0)),
                role_abundance=float(composition.role_abundances.get(species_role, 0.0)),
                variant_allocation_count=int(variant_counts.get(species_key, 0)),
                scale_xy=scale_xy,
                scale_z=scale_z,
                target_height_m=target_height_m,
                height_prior_min_m=(
                    height_prior.height_min_m if height_prior is not None else None
                ),
                height_prior_mode_m=(
                    height_prior.height_mode_m if height_prior is not None else None
                ),
                height_prior_max_m=(
                    height_prior.height_max_m if height_prior is not None else None
                ),
                height_prior_source=height_prior.source if height_prior is not None else "",
            )
        )
        selected_assets[asset.asset_id] = asset
        variant_species.append(asset.species_label)
        variant_source_seasons.append(resolved_season)

    logger.info("Botaniq direct asset-pack mode enabled. Pack: %s", pack_dir)
    logger.info(
        "Botaniq direct asset-pack composition: %s",
        ", ".join(
            f"{catalog[species_key][0].species_label}"
            f"({composition.species_roles.get(species_key, 'unknown')}:"
            f"{allocated_weights.get(species_key, 0.0):.3f},"
            f"templates={variant_counts.get(species_key, 0)})"
            for species_key in allocated_species_keys
        ),
    )
    logger.info(
        "Botaniq direct asset-pack uses raw meshes plus height-prior scaling; "
        "prebake branch, shape, tilt, and leaf-density variation are skipped."
    )

    tree_col = bpy.data.collections.new("TreeTemplate")
    scene.collection.children.link(tree_col)

    cleanup_objects: list[bpy.types.Object] = []
    cleanup_collections: list[bpy.types.Collection] = []
    source_mesh_by_asset_id: dict[str, bpy.types.Object] = {}
    load_started = time.perf_counter()

    for asset in selected_assets.values():
        source_collection = _load_collection_from_library(
            blend_path=asset.blend_path,
            collection_name=asset.blend_path.stem,
        )
        cleanup_collections.append(source_collection)
        cleanup_objects.extend(list(source_collection.all_objects))
        source_meshes = _mesh_objects_from_collection(source_collection)
        if not source_meshes:
            raise RuntimeError(f"No mesh objects found in botaniq asset: {asset.blend_path}")
        if len(source_meshes) > 1:
            logger.warning(
                "Botaniq asset '%s' contains %s mesh objects; using the first mesh "
                "for direct asset-pack scatter.",
                asset.title,
                len(source_meshes),
            )
        source_mesh_by_asset_id[asset.asset_id] = source_meshes[0]

    instance_material_slots = 0
    instance_material_nodes = 0
    target_height_values: list[float] = []
    for item in plan:
        source_obj = source_mesh_by_asset_id[item.asset.asset_id]
        if source_obj.type != "MESH" or source_obj.data is None:
            raise RuntimeError(
                f"Botaniq source object '{source_obj.name}' is not a mesh and cannot be scattered."
            )

        tree_obj = source_obj.copy()
        tree_obj.data = source_obj.data.copy()
        tree_obj.name = f"BotaniqPackTree_{item.variant_idx:03d}"
        tree_obj["preset_name"] = item.asset.species_label
        tree_obj["species_name"] = item.asset.species_key
        tree_obj["species_label"] = item.asset.species_label
        tree_obj["species_common_name"] = item.asset.species_common_name
        tree_obj["botaniq_asset_id"] = item.asset.asset_id
        tree_obj["botaniq_asset_title"] = item.asset.title
        tree_obj["asset_source_name"] = source_obj.name
        tree_obj["season"] = item.resolved_season
        tree_obj["botaniq_height_m"] = float(item.asset.height_m)
        tree_obj["botaniq_width_m"] = float(item.asset.width_m)
        tree_obj["botaniq_depth_m"] = float(item.asset.depth_m)
        tree_obj["botaniq_model_detail"] = item.asset.model_detail
        tree_obj["stand_species_role"] = item.species_role
        tree_obj["stand_species_abundance"] = float(item.stand_abundance)
        tree_obj["stand_role_abundance"] = float(item.role_abundance)
        tree_obj["stand_variant_allocation_count"] = int(item.variant_allocation_count)
        tree_obj["botaniq_direct_asset_pack"] = True

        tree_obj.data.transform(tree_obj.matrix_basis)
        tree_obj.location = (0.0, 0.0, 0.0)
        tree_obj.rotation_euler = (0.0, 0.0, 0.0)
        tree_obj.scale = (1.0, 1.0, 1.0)
        _apply_botaniq_mesh_scale(tree_obj, item.scale_xy, item.scale_z, cfg)

        material_recipe = _build_botaniq_instance_leaf_material_recipe(tree_obj, cfg)
        _apply_botaniq_leaf_material_variation_recipe(tree_obj, material_recipe)
        tree_obj["botaniq_instance_leaf_material_recipe_json"] = json.dumps(
            material_recipe,
            sort_keys=True,
        )
        instance_material_slots += int(tree_obj.get("botaniq_leaf_material_variation_slots", 0))
        instance_material_nodes += int(tree_obj.get("botaniq_leaf_material_variation_nodes", 0))

        anchor_z = float(infer_ground_anchor_local_z(tree_obj))
        tree_obj["ground_anchor_z_local"] = anchor_z
        tree_obj.location = (0.0, 0.0, -anchor_z)

        min_v, max_v = _mesh_local_bounds(tree_obj)
        realized_width = max(float(max_v.x - min_v.x), 0.0)
        realized_depth = max(float(max_v.y - min_v.y), 0.0)
        realized_height = max(float(max_v.z - min_v.z), 0.0)
        tree_obj["realized_crown_diameter_m"] = max(realized_width, realized_depth)
        tree_obj["realized_height_m"] = realized_height
        tree_obj["fit_scale_xy"] = float(item.scale_xy)
        tree_obj["fit_scale_z"] = float(item.scale_z)
        if item.target_height_m is not None:
            target_height_values.append(float(item.target_height_m))
            tree_obj["botaniq_target_height_m"] = float(item.target_height_m)
            tree_obj["botaniq_height_prior_min_m"] = float(item.height_prior_min_m or 0.0)
            tree_obj["botaniq_height_prior_mode_m"] = float(item.height_prior_mode_m or 0.0)
            tree_obj["botaniq_height_prior_max_m"] = float(item.height_prior_max_m or 0.0)
            tree_obj["botaniq_height_prior_source"] = str(item.height_prior_source)
        tree_obj["botaniq_prebake_variation_skipped"] = True
        tree_col.objects.link(tree_obj)

    _cleanup_unlinked_asset_sources(cleanup_objects, cleanup_collections)

    logger.info(
        "Botaniq direct asset-pack templates prepared in %.1fs | templates=%s | unique_assets=%s",
        time.perf_counter() - load_started,
        len(plan),
        len(selected_assets),
    )
    logger.info(
        "Botaniq direct asset-pack instance material nodes: slots=%s | nodes=%s",
        instance_material_slots,
        instance_material_nodes,
    )
    if target_height_values:
        logger.info(
            "Botaniq direct asset-pack height priors applied: target_height_range=%.2f-%.2fm | templates=%s/%s",
            min(target_height_values),
            max(target_height_values),
            len(target_height_values),
            len(plan),
        )

    collection_metadata = {
        "requested_season": requested_season,
        "variant_source_seasons": variant_source_seasons,
        "asset_pool_source": "botaniq_asset_pack",
        "botaniq_direct_asset_pack": True,
        "botaniq_pack_dir": str(pack_dir),
        "botaniq_asset_titles": [item.asset.title for item in plan],
        "botaniq_asset_ids": [item.asset.asset_id for item in plan],
        "botaniq_height_priors_applied_at_scatter": bool(target_height_values),
        "botaniq_height_priors_applied_at_prebake": False,
        "botaniq_height_priors_expected_stage": "runtime_asset_pack",
        "botaniq_target_heights_m": target_height_values,
        "botaniq_target_height_range_m": (
            [min(target_height_values), max(target_height_values)] if target_height_values else []
        ),
        "botaniq_species_height_priors": {
            catalog[species_key][0].species_label: {
                "height_min_m": float(height_priors[species_key].height_min_m),
                "height_mode_m": float(height_priors[species_key].height_mode_m),
                "height_max_m": float(height_priors[species_key].height_max_m),
                "source": height_priors[species_key].source,
            }
            for species_key in allocated_species_keys
            if species_key in height_priors
        },
        "botaniq_species_selected": [
            catalog[species_key][0].species_label for species_key in allocated_species_keys
        ],
        "botaniq_species_abundances": {
            catalog[species_key][0].species_label: float(allocated_weights.get(species_key, 0.0))
            for species_key in allocated_species_keys
        },
        "botaniq_species_roles": {
            catalog[species_key][0].species_label: composition.species_roles.get(
                species_key,
                "unknown",
            )
            for species_key in allocated_species_keys
        },
        "botaniq_role_abundances": dict(composition.role_abundances),
        "botaniq_role_requested_counts": dict(composition.role_requested_counts),
        "botaniq_role_selected_counts": dict(composition.role_selected_counts),
        "botaniq_role_available_species": dict(composition.role_available_species),
        "botaniq_variant_allocation": {
            catalog[species_key][0].species_label: int(variant_counts.get(species_key, 0))
            for species_key in allocated_species_keys
        },
        "botaniq_instance_leaf_material_variation_slots": int(instance_material_slots),
        "botaniq_instance_leaf_material_variation_nodes": int(instance_material_nodes),
        "botaniq_prebake_variation_skipped": True,
        "botaniq_branch_variation_enabled": False,
        "botaniq_branch_variation_modes": [],
        "botaniq_branch_groups_used": [],
        "botaniq_branch_swap_pair_count": 0,
        "botaniq_branch_vertices_changed": 0,
        "botaniq_leaf_density_removed_faces": 0,
        "botaniq_leaf_density_removed_units": 0,
        "botaniq_leaf_material_variation_slots": 0,
        "botaniq_leaf_material_variation_nodes": 0,
    }
    return tree_col, variant_species, collection_metadata


def _create_botaniq_height_scaled_variant_collection(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
    *,
    height_prior_stage: str = "asset_pool_prebake",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    pack_dir = _resolve_botaniq_pack_dir(cfg)
    include_title_prefixes = [
        str(prefix) for prefix in getattr(cfg, "botaniq_include_title_prefixes", ["Tree "])
    ]
    catalog = _load_botaniq_catalog(pack_dir, include_title_prefixes)
    if not catalog:
        raise RuntimeError(
            "No botaniq model assets matched scatter.botaniq_include_title_prefixes "
            f"in {pack_dir / 'mapr_index.json'}"
        )

    rng = random.Random(seed)
    composition = _build_botaniq_stand_composition(cfg, catalog, rng)
    variant_count = max(1, int(cfg.tree_variants))
    variant_counts = _allocate_botaniq_variant_counts(
        composition.species_keys,
        composition.species_weights,
        variant_count,
    )
    allocated_species_keys = [
        species_key
        for species_key in composition.species_keys
        if variant_counts.get(species_key, 0) > 0
    ]
    if not allocated_species_keys:
        raise RuntimeError("No botaniq species were allocated to generated variants.")

    dropped_species = [
        catalog[species_key][0].species_label
        for species_key in composition.species_keys
        if variant_counts.get(species_key, 0) <= 0
    ]
    if dropped_species:
        logger.warning(
            "Dropped botaniq species because scatter.tree_variants=%s is too small: %s",
            variant_count,
            ", ".join(dropped_species),
        )

    allocated_weights = _normalize_float_weights(
        {
            species_key: composition.species_weights.get(species_key, 0.0)
            for species_key in allocated_species_keys
        }
    )
    requested_season = _normalize_season_name(season)
    scale_min = max(0.01, float(getattr(cfg, "botaniq_variant_scale_min", 0.85)))
    scale_max = max(scale_min, float(getattr(cfg, "botaniq_variant_scale_max", 1.15)))
    height_scale_min = max(0.01, float(getattr(cfg, "botaniq_variant_height_scale_min", 0.90)))
    height_scale_max = max(
        height_scale_min,
        float(getattr(cfg, "botaniq_variant_height_scale_max", 1.10)),
    )
    height_priors = _load_botaniq_species_height_priors(cfg)
    crown_scale_power = max(
        0.0,
        float(getattr(cfg, "botaniq_height_to_crown_scale_power", 0.65)),
    )
    scale_z_max = max(0.01, float(getattr(cfg, "botaniq_height_scale_z_max", 3.0)))
    scale_xy_max = max(0.01, float(getattr(cfg, "botaniq_height_scale_xy_max", 3.0)))
    anisotropy_max = max(
        0.01,
        float(getattr(cfg, "botaniq_height_scale_anisotropy_max", 1.45)),
    )

    plan: list[_PlannedBotaniqVariant] = []
    variant_species: list[str] = []
    variant_source_seasons: list[str] = []
    selected_assets: dict[str, _BotaniqAsset] = {}
    source_candidates_by_species: dict[str, list[tuple[_BotaniqAsset, int, str]]] = {}
    planned_species_keys: list[str] = []
    for species_key in allocated_species_keys:
        planned_species_keys.extend([species_key] * variant_counts.get(species_key, 0))
        source_candidates = _botaniq_asset_candidates_for_season(
            catalog[species_key],
            requested_season,
        )
        if not source_candidates:
            raise RuntimeError(
                "No botaniq source assets were available for generated variants of "
                f"species '{catalog[species_key][0].species_label}'."
            )
        source_candidates_by_species[species_key] = source_candidates
    rng.shuffle(planned_species_keys)

    for variant_idx, species_key in enumerate(planned_species_keys):
        representative_asset = source_candidates_by_species[species_key][0][0]
        target_height_m, height_prior = _sample_botaniq_target_height_m(
            representative_asset,
            height_priors,
            rng,
        )
        asset, source_index, resolved_season, height_scale_plan = (
            _choose_botaniq_asset_for_height_target(
                source_candidates_by_species[species_key],
                target_height_m,
                rng,
                scale_min=scale_min,
                scale_max=scale_max,
                height_scale_min=height_scale_min,
                height_scale_max=height_scale_max,
                crown_scale_power=crown_scale_power,
                scale_z_max=scale_z_max,
                scale_xy_max=scale_xy_max,
                anisotropy_max=anisotropy_max,
            )
        )
        if height_scale_plan.clamped:
            logger.debug(
                "Clamped botaniq target height for %s using %s: %.2fm -> %.2fm (%s)",
                asset.species_label,
                asset.title,
                float(height_scale_plan.unclamped_target_height_m or 0.0),
                float(height_scale_plan.target_height_m or 0.0),
                height_scale_plan.clamp_reason,
            )
        if target_height_m is None:
            height_scale_plan = _plan_botaniq_variant_height_scale(
                asset,
                None,
                rng,
                scale_min=scale_min,
                scale_max=scale_max,
                height_scale_min=height_scale_min,
                height_scale_max=height_scale_max,
                crown_scale_power=crown_scale_power,
                scale_z_max=scale_z_max,
                scale_xy_max=scale_xy_max,
                anisotropy_max=anisotropy_max,
            )
        else:
            target_height_m = height_scale_plan.target_height_m
        scale_xy = height_scale_plan.scale_xy
        scale_z = height_scale_plan.scale_z
        scale_anisotropy = height_scale_plan.anisotropy
        plan.append(
            _PlannedBotaniqVariant(
                variant_idx=variant_idx,
                asset=asset,
                source_index=source_index,
                resolved_season=resolved_season,
                species_role=composition.species_roles.get(species_key, "unknown"),
                stand_abundance=allocated_weights.get(species_key, 0.0),
                role_abundance=composition.role_abundances.get(
                    composition.species_roles.get(species_key, ""),
                    0.0,
                ),
                variant_allocation_count=variant_counts.get(species_key, 0),
                scale_xy=scale_xy,
                scale_z=scale_z,
                target_height_m=target_height_m,
                unclamped_target_height_m=height_scale_plan.unclamped_target_height_m,
                height_prior_min_m=(
                    height_prior.height_min_m if height_prior is not None else None
                ),
                height_prior_mode_m=(
                    height_prior.height_mode_m if height_prior is not None else None
                ),
                height_prior_max_m=(
                    height_prior.height_max_m if height_prior is not None else None
                ),
                height_prior_source=height_prior.source if height_prior is not None else "",
                height_clamped=height_scale_plan.clamped,
                height_clamp_reason=height_scale_plan.clamp_reason,
                scale_anisotropy=scale_anisotropy,
            )
        )
        variant_species.append(asset.species_label)
        variant_source_seasons.append(resolved_season)
        selected_assets[asset.asset_id] = asset

    logger.info("Botaniq asset-pack mode enabled. Pack: %s", pack_dir)
    logger.info(
        "Botaniq selected stand composition: %s",
        ", ".join(
            f"{catalog[species_key][0].species_label}"
            f"({composition.species_roles.get(species_key, 'unknown')}:"
            f"{allocated_weights.get(species_key, 0.0):.3f},"
            f"variants={variant_counts.get(species_key, 0)})"
            for species_key in allocated_species_keys
        ),
    )
    logger.info("Botaniq requested season: %s", requested_season)
    logger.info(
        "Botaniq selected unique assets: %s",
        ", ".join(asset.title for asset in selected_assets.values()),
    )
    source_asset_allocation: dict[str, int] = {}
    for item in plan:
        source_asset_allocation[item.asset.title] = (
            source_asset_allocation.get(item.asset.title, 0) + 1
        )
    logger.info(
        "Botaniq source asset allocation: %s",
        ", ".join(f"{title}={count}" for title, count in sorted(source_asset_allocation.items())),
    )

    tree_col = bpy.data.collections.new("TreeTemplate")
    scene.collection.children.link(tree_col)

    cleanup_objects: list[bpy.types.Object] = []
    cleanup_collections: list[bpy.types.Collection] = []
    source_mesh_by_asset_id: dict[str, bpy.types.Object] = {}
    load_started = time.perf_counter()

    for asset in selected_assets.values():
        source_collection = _load_collection_from_library(
            blend_path=asset.blend_path,
            collection_name=asset.blend_path.stem,
        )
        cleanup_collections.append(source_collection)
        cleanup_objects.extend(list(source_collection.all_objects))
        source_meshes = _mesh_objects_from_collection(source_collection)
        if not source_meshes:
            raise RuntimeError(f"No mesh objects found in botaniq asset: {asset.blend_path}")
        if len(source_meshes) > 1:
            logger.warning(
                "Botaniq asset '%s' contains %s mesh objects; "
                "using the first mesh for this test loader.",
                asset.title,
                len(source_meshes),
            )
        source_mesh_by_asset_id[asset.asset_id] = source_meshes[0]

    branch_variation_modes: list[str] = []
    branch_variation_groups: set[str] = set()
    branch_variation_roles: list[str] = []
    branch_variation_swap_pair_count = 0
    branch_variation_changed_vertices = 0
    leaf_density_removed_faces = 0
    leaf_density_removed_units = 0
    leaf_material_variation_slots = 0
    leaf_material_variation_nodes = 0
    target_height_values: list[float] = []
    variant_recipes: list[dict[str, object]] = []

    for item in plan:
        source_obj = source_mesh_by_asset_id[item.asset.asset_id]
        tree_obj = _copy_botaniq_mesh_variant(
            source_obj=source_obj,
            tree_col=tree_col,
            plan_item=item,
            rng=rng,
            cfg=cfg,
        )
        branch_variation_modes.append(str(tree_obj.get("botaniq_branch_variation_mode", "")))
        branch_variation_groups.update(
            group_name
            for group_name in str(tree_obj.get("botaniq_branch_groups_used", "")).split(",")
            if group_name
        )
        branch_variation_roles.extend(
            role_name
            for role_name in str(tree_obj.get("botaniq_branch_variation_roles", "")).split(",")
            if role_name
        )
        try:
            branch_variation_swap_pair_count += int(
                tree_obj.get("botaniq_branch_swap_pair_count", 0)
            )
        except Exception:
            pass
        try:
            branch_variation_changed_vertices += int(
                tree_obj.get("botaniq_branch_vertices_changed", 0)
            )
        except Exception:
            pass
        try:
            leaf_density_removed_faces += int(tree_obj.get("botaniq_leaf_density_removed_faces", 0))
        except Exception:
            pass
        try:
            leaf_density_removed_units += int(tree_obj.get("botaniq_leaf_density_removed_units", 0))
        except Exception:
            pass
        try:
            leaf_material_variation_slots += int(
                tree_obj.get("botaniq_leaf_material_variation_slots", 0)
            )
            leaf_material_variation_nodes += int(
                tree_obj.get("botaniq_leaf_material_variation_nodes", 0)
            )
        except Exception:
            pass
        try:
            target_height = float(tree_obj.get("botaniq_target_height_m", 0.0))
            if target_height > 0.0:
                target_height_values.append(target_height)
        except Exception:
            pass
        recipe_json = str(tree_obj.get("botaniq_variant_recipe_json", "")).strip()
        if recipe_json:
            try:
                loaded_recipe = json.loads(recipe_json)
                if isinstance(loaded_recipe, dict):
                    variant_recipes.append(loaded_recipe)
            except Exception:
                logger.debug("Failed to parse botaniq variant recipe on %s", tree_obj.name)

    _cleanup_unlinked_asset_sources(cleanup_objects, cleanup_collections)
    quality_summary = botaniq_variant_quality_summary(variant_recipes)

    logger.info(
        "Botaniq variants prepared in %.1fs | variants=%s | unique_assets=%s",
        time.perf_counter() - load_started,
        variant_count,
        len(selected_assets),
    )
    logger.info(
        "Botaniq branch crown variation: groups=%s | changed_vertices=%s | swap_pairs=%s",
        ", ".join(sorted(branch_variation_groups)) if branch_variation_groups else "none",
        branch_variation_changed_vertices,
        branch_variation_swap_pair_count,
    )
    logger.info(
        "Botaniq leaf-density variation: removed_faces=%s | removed_units=%s",
        leaf_density_removed_faces,
        leaf_density_removed_units,
    )
    logger.info(
        "Botaniq prebake leaf-material variation: slots=%s | nodes=%s",
        leaf_material_variation_slots,
        leaf_material_variation_nodes,
    )
    logger.info(
        "Botaniq variant quality: clean=%s/%s | height_clamped=%s | flags=%s",
        quality_summary.get("clean_variant_count", 0),
        quality_summary.get("variant_count", 0),
        quality_summary.get("height_clamped_count", 0),
        quality_summary.get("flag_counts", {}),
    )
    if target_height_values:
        logger.info(
            "Botaniq species height priors applied: target_height_range=%.2f-%.2fm | variants=%s/%s",
            min(target_height_values),
            max(target_height_values),
            len(target_height_values),
            len(plan),
        )
    logger.info(
        "Botaniq species distribution in generated variants: %s",
        ", ".join(
            f"{catalog[species_key][0].species_label}="
            f"{variant_species.count(catalog[species_key][0].species_label)}"
            for species_key in allocated_species_keys
        ),
    )

    height_stage = str(height_prior_stage)
    collection_metadata = {
        "requested_season": requested_season,
        "variant_source_seasons": variant_source_seasons,
        "botaniq_pack_dir": str(pack_dir),
        "botaniq_asset_titles": [item.asset.title for item in plan],
        "botaniq_asset_ids": [item.asset.asset_id for item in plan],
        "botaniq_source_asset_allocation": dict(sorted(source_asset_allocation.items())),
        "botaniq_height_priors_applied_at_scatter": (
            height_stage == "runtime_asset_pack" and bool(target_height_values)
        ),
        "botaniq_height_priors_applied_at_prebake": (
            height_stage == "asset_pool_prebake" and bool(target_height_values)
        ),
        "botaniq_height_priors_expected_stage": height_stage,
        "botaniq_target_heights_m": target_height_values,
        "botaniq_target_height_range_m": (
            [min(target_height_values), max(target_height_values)] if target_height_values else []
        ),
        "botaniq_species_height_priors": {
            catalog[species_key][0].species_label: {
                "height_min_m": float(height_priors[species_key].height_min_m),
                "height_mode_m": float(height_priors[species_key].height_mode_m),
                "height_max_m": float(height_priors[species_key].height_max_m),
                "source": height_priors[species_key].source,
            }
            for species_key in allocated_species_keys
            if species_key in height_priors
        },
        "botaniq_species_selected": [
            catalog[species_key][0].species_label for species_key in allocated_species_keys
        ],
        "botaniq_species_abundances": {
            catalog[species_key][0].species_label: float(allocated_weights.get(species_key, 0.0))
            for species_key in allocated_species_keys
        },
        "botaniq_species_roles": {
            catalog[species_key][0].species_label: composition.species_roles.get(
                species_key,
                "unknown",
            )
            for species_key in allocated_species_keys
        },
        "botaniq_role_abundances": dict(composition.role_abundances),
        "botaniq_role_requested_counts": dict(composition.role_requested_counts),
        "botaniq_role_selected_counts": dict(composition.role_selected_counts),
        "botaniq_role_available_species": dict(composition.role_available_species),
        "botaniq_variant_allocation": {
            catalog[species_key][0].species_label: int(variant_counts.get(species_key, 0))
            for species_key in allocated_species_keys
        },
        "botaniq_branch_variation_enabled": bool(
            getattr(cfg, "botaniq_branch_variation_enabled", True)
        ),
        "botaniq_branch_variation_modes": branch_variation_modes,
        "botaniq_branch_groups_used": sorted(branch_variation_groups),
        "botaniq_branch_variation_roles": branch_variation_roles,
        "botaniq_branch_swap_pair_count": int(branch_variation_swap_pair_count),
        "botaniq_branch_vertices_changed": int(branch_variation_changed_vertices),
        "botaniq_leaf_density_removed_faces": int(leaf_density_removed_faces),
        "botaniq_leaf_density_removed_units": int(leaf_density_removed_units),
        "botaniq_leaf_material_variation_slots": int(leaf_material_variation_slots),
        "botaniq_leaf_material_variation_nodes": int(leaf_material_variation_nodes),
        "botaniq_variant_recipe_schema": BOTANIQ_VARIANT_RECIPE_SCHEMA,
        "botaniq_variant_recipes": variant_recipes,
        "botaniq_variant_quality_summary": quality_summary,
        "botaniq_asset_pool_stage": str(getattr(cfg, "botaniq_asset_pool_stage", "production")),
    }
    return tree_col, variant_species, collection_metadata


def create_tree_collection_from_botaniq_asset_pack(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    """Load botaniq pack assets directly and apply height priors on the fly."""
    return _create_tree_collection_from_botaniq_asset_pack_simple(
        scene=scene,
        cfg=cfg,
        seed=seed,
        season=season,
    )


def create_botaniq_prebake_tree_collection(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    """Build botaniq variants for asset-pool prebaking with height priors baked in."""
    return _create_botaniq_height_scaled_variant_collection(
        scene=scene,
        cfg=cfg,
        seed=seed,
        season=season,
        height_prior_stage="asset_pool_prebake",
    )
