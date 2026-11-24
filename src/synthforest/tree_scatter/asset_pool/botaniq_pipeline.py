from __future__ import annotations

import json
import logging
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import bpy
from omegaconf import DictConfig, OmegaConf

from ..botaniq_catalog import (
    _botaniq_asset_candidates_for_season,
    _load_botaniq_catalog,
    _resolve_botaniq_pack_dir,
)
from ..botaniq_templates import (
    _asset_object_name,
    create_botaniq_prebake_tree_collection,
)
from ..botaniq_utils import normalize_species_key as _normalize_species_key
from ..config import ScatterConfig
from .botaniq_config import load_botaniq_asset_pool_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PoolSeasonBakeGroup:
    seasons: tuple[str, ...]
    source_season: str
    season_label: str
    first_season_index: int
    source_asset_ids: tuple[str, ...]


def _purge_orphan_blender_data() -> None:
    try:
        purge = getattr(bpy.data, "orphans_purge", None)
        if purge is not None:
            try:
                purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
            except TypeError:
                purge()
    except Exception:
        return


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


def _safe_filename(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "botaniq_species"


def _configured_species_keys(
    configured_species: list[str],
    catalog: dict[str, list[object]],
) -> list[str]:
    if not configured_species or any(item.strip() in {"*", "all"} for item in configured_species):
        return sorted(catalog)

    available = set(catalog)
    output: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for species_name in configured_species:
        species_key = _normalize_species_key(species_name)
        if species_key not in available:
            unknown.append(species_name)
            continue
        if species_key in seen:
            continue
        seen.add(species_key)
        output.append(species_key)

    if unknown:
        available_names = ", ".join(
            assets[0].species_label for _key, assets in sorted(catalog.items())
        )
        raise ValueError(
            "Unknown botaniq species in generation.species: "
            f"{', '.join(unknown)}. Available: {available_names}"
        )
    return output


def _available_pool_seasons(
    assets: list[object],
    season_mode: str,
) -> list[str]:
    available = {season for asset in assets for season in getattr(asset, "seasons", ())}
    has_summer = "summer" in available
    has_autumn = "autumn" in available

    if season_mode == "summer":
        return ["summer"]
    if season_mode == "autumn":
        return ["autumn"] if has_autumn else ["summer"]

    seasons = ["summer"] if has_summer or not has_autumn else []
    if has_autumn:
        seasons.append("autumn")
    return seasons or ["summer"]


def _season_source_asset_ids(assets: list[object], season_name: str) -> tuple[str, ...]:
    return tuple(
        str(asset.asset_id)
        for asset, _index, _resolved_season in _botaniq_asset_candidates_for_season(
            assets,
            season_name,
        )
    )


def _season_group_label(seasons: tuple[str, ...]) -> str:
    return "_".join(seasons)


def _pool_season_bake_groups(
    assets: list[object],
    seasons: list[str],
) -> list[_PoolSeasonBakeGroup]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    first_indices: dict[tuple[str, ...], int] = {}
    order: list[tuple[str, ...]] = []

    for season_index, season_name in enumerate(seasons):
        source_asset_ids = _season_source_asset_ids(assets, season_name)
        if source_asset_ids not in grouped:
            grouped[source_asset_ids] = []
            first_indices[source_asset_ids] = season_index
            order.append(source_asset_ids)
        grouped[source_asset_ids].append(season_name)

    return [
        _PoolSeasonBakeGroup(
            seasons=tuple(grouped[source_asset_ids]),
            source_season=grouped[source_asset_ids][0],
            season_label=_season_group_label(tuple(grouped[source_asset_ids])),
            first_season_index=first_indices[source_asset_ids],
            source_asset_ids=source_asset_ids,
        )
        for source_asset_ids in order
    ]


def _build_species_scatter_cfg(
    base_scatter_cfg: ScatterConfig,
    species_label: str,
    variants_per_species: int,
) -> ScatterConfig:
    scatter_cfg = deepcopy(base_scatter_cfg)
    scatter_cfg.botaniq_species_role_pools = {
        "dominant": [species_label],
        "companion": [],
        "rare": [],
    }
    scatter_cfg.botaniq_role_count_ranges = {
        "dominant": [1, 1],
        "companion": [0, 0],
        "rare": [0, 0],
    }
    scatter_cfg.botaniq_role_abundance_ranges = {
        "dominant": [1.0, 1.0],
        "companion": [0.0, 0.0],
        "rare": [0.0, 0.0],
    }
    scatter_cfg.tree_variants = int(variants_per_species)
    return scatter_cfg


def _layout_collection_for_inspection(
    tree_collection: bpy.types.Collection,
    spacing_m: float,
) -> None:
    objects = [obj for obj in tree_collection.objects if obj.type == "MESH"]
    if not objects or spacing_m <= 0.0:
        return

    columns = max(1, math.ceil(math.sqrt(len(objects))))
    for index, obj in enumerate(objects):
        row = index // columns
        col = index % columns
        obj.location.x = (col - (columns - 1) * 0.5) * spacing_m
        obj.location.y = (row - (math.ceil(len(objects) / columns) - 1) * 0.5) * spacing_m
        obj["pool_inspection_grid_index"] = int(index)


def _save_pool_blend(
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists and output.overwrite=false: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=True)


def _save_resolved_config(pool_dir: Path, runtime_cfg: object) -> Path:
    payload = asdict(runtime_cfg) if is_dataclass(runtime_cfg) else {"value": runtime_cfg}
    config_path = pool_dir / "config" / "resolved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=config_path)
    return config_path


def _build_stage_scatter_cfg(
    base_scatter_cfg: ScatterConfig,
    species_label: str,
    variants_per_species: int,
    *,
    stage: str,
    variant_recipes: list[dict[str, object]] | None = None,
    production_apply_modifiers: bool = True,
) -> ScatterConfig:
    scatter_cfg = _build_species_scatter_cfg(
        base_scatter_cfg,
        species_label,
        variants_per_species,
    )
    scatter_cfg.botaniq_asset_pool_stage = stage
    if stage == "authoring":
        scatter_cfg.botaniq_branch_geometry_nodes_apply = False
        scatter_cfg.botaniq_branch_geometry_nodes_layout = str(
            getattr(scatter_cfg, "botaniq_authoring_modifier_layout", "per_branch")
        )
    else:
        scatter_cfg.botaniq_branch_geometry_nodes_apply = bool(production_apply_modifiers)
        scatter_cfg.botaniq_branch_geometry_nodes_layout = "combined"
    if variant_recipes is not None:
        scatter_cfg.botaniq_variant_recipes = variant_recipes
    return scatter_cfg


def _prepare_tree_collection_for_pool(
    tree_collection: bpy.types.Collection,
    species_key: str,
    species_label: str,
    season_name: str,
    collection_name: str,
    layout_for_inspection: bool,
    inspection_spacing_m: float,
) -> None:
    for variant_idx, tree_obj in enumerate(tree_collection.objects):
        tree_obj.name = _asset_object_name(species_key, season_name, variant_idx)
        tree_obj["species_name"] = species_key
        tree_obj["species_label"] = species_label
        tree_obj["variant_index"] = int(variant_idx)
        tree_obj["season"] = season_name

    tree_collection.name = collection_name
    if layout_for_inspection:
        _layout_collection_for_inspection(
            tree_collection,
            inspection_spacing_m,
        )


def _recipe_payload(
    *,
    species_key: str,
    species_label: str,
    season_name: str,
    requested_seasons: tuple[str, ...],
    seed: int,
    collection_metadata: dict[str, object],
    authoring_file: str | None,
    production_file: str | None,
) -> dict[str, object]:
    recipes = collection_metadata.get("botaniq_variant_recipes", [])
    if not isinstance(recipes, list):
        recipes = []
    return {
        "schema": "synthforest.botaniq_asset_pool_recipes.v1",
        "species": species_key,
        "species_label": species_label,
        "season": season_name,
        "requested_seasons": list(requested_seasons),
        "seed": int(seed),
        "authoring_file": authoring_file,
        "production_file": production_file,
        "variant_count": len(recipes),
        "target_height_range_m": list(collection_metadata.get("botaniq_target_height_range_m", [])),
        "height_priors_expected_stage": str(
            collection_metadata.get("botaniq_height_priors_expected_stage", "")
        ),
        "height_priors_applied_at_prebake": bool(
            collection_metadata.get("botaniq_height_priors_applied_at_prebake", False)
        ),
        "species_height_priors": dict(collection_metadata.get("botaniq_species_height_priors", {})),
        "variant_quality_summary": dict(
            collection_metadata.get("botaniq_variant_quality_summary", {})
        ),
        "variants": recipes,
    }


def _positive_range(values: list[float]) -> list[float]:
    positive = [float(value) for value in values if float(value) > 0.0]
    if not positive:
        return [0.0, 0.0]
    return [float(min(positive)), float(max(positive))]


def run_botaniq_asset_pool_pipeline(cfg: DictConfig) -> None:
    runtime_cfg = load_botaniq_asset_pool_config(cfg)
    pool_dir = Path(runtime_cfg.output.pool_dir).expanduser().resolve()
    pool_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = _save_resolved_config(pool_dir, runtime_cfg)
    logger.info("Resolved botaniq asset-pool config saved to: %s", resolved_config_path)

    pack_dir = _resolve_botaniq_pack_dir(runtime_cfg.scatter)
    include_title_prefixes = [
        str(prefix)
        for prefix in getattr(runtime_cfg.scatter, "botaniq_include_title_prefixes", ["Tree "])
    ]
    catalog = _load_botaniq_catalog(pack_dir, include_title_prefixes)
    if not catalog:
        raise RuntimeError(
            "No botaniq tree assets matched scatter.botaniq_include_title_prefixes "
            f"in {pack_dir / 'mapr_index.json'}"
        )

    species_keys = _configured_species_keys(runtime_cfg.generation.species, catalog)
    seed_base = int(runtime_cfg.seed)
    manifest: dict[str, object] = {
        "seed": seed_base,
        "source": "botaniq_asset_pool",
        "botaniq_pack_dir": str(pack_dir),
        "variants_per_species": int(runtime_cfg.generation.variants_per_species),
        "season_mode": str(runtime_cfg.generation.season_mode),
        "species": [],
    }

    logger.info("Generating botaniq asset pool in %s", pool_dir)
    logger.info(
        "Botaniq species to generate: %s",
        ", ".join(catalog[species_key][0].species_label for species_key in species_keys),
    )

    for species_index, species_key in enumerate(species_keys):
        assets = catalog[species_key]
        species_label = assets[0].species_label
        species_seed = seed_base + species_index * 101
        seasons_to_generate = _available_pool_seasons(
            assets,
            runtime_cfg.generation.season_mode,
        )
        season_bake_groups = _pool_season_bake_groups(assets, seasons_to_generate)
        pools_by_season: dict[str, dict[str, object]] = {}

        logger.info("-" * 50)
        logger.info("Generating botaniq assets for species: %s", species_label)

        for season_group in season_bake_groups:
            season_name = season_group.season_label
            source_season_name = season_group.source_season
            requested_seasons = season_group.seasons
            if len(requested_seasons) > 1:
                logger.info(
                    "Generating shared season '%s' for species '%s' "
                    "(requested seasons: %s; same source presets)",
                    season_name,
                    species_label,
                    ", ".join(requested_seasons),
                )
            else:
                logger.info("Generating season '%s' for species '%s'", season_name, species_label)
            season_seed = species_seed + season_group.first_season_index * 10007
            safe_species = _safe_filename(species_key)
            collection_name = f"TreeAssets_{species_key}_{season_name}"
            authoring_path: Path | None = None
            production_path: Path | None = None
            collection_metadata: dict[str, object] = {}
            variant_recipes: list[dict[str, object]] = []

            if bool(runtime_cfg.output.save_authoring):
                scene = _reset_main_scene(
                    f"BotaniqAssetPool_{safe_species}_{season_name}_authoring"
                )
                scatter_cfg = _build_stage_scatter_cfg(
                    runtime_cfg.scatter,
                    species_label,
                    runtime_cfg.generation.variants_per_species,
                    stage="authoring",
                    production_apply_modifiers=False,
                )
                tree_collection, _variant_species, authoring_metadata = (
                    create_botaniq_prebake_tree_collection(
                        scene=scene,
                        cfg=scatter_cfg,
                        seed=season_seed,
                        season=source_season_name,
                    )
                )
                _prepare_tree_collection_for_pool(
                    tree_collection,
                    species_key,
                    species_label,
                    season_name,
                    collection_name,
                    bool(runtime_cfg.output.layout_for_inspection),
                    float(runtime_cfg.output.inspection_spacing_m),
                )
                authoring_path = pool_dir / (
                    f"{safe_species}_{season_name}{runtime_cfg.output.authoring_suffix}.blend"
                )
                _save_pool_blend(authoring_path, bool(runtime_cfg.output.overwrite))
                collection_metadata = dict(authoring_metadata)
                raw_recipes = authoring_metadata.get("botaniq_variant_recipes", [])
                if isinstance(raw_recipes, list):
                    variant_recipes = [recipe for recipe in raw_recipes if isinstance(recipe, dict)]
                logger.info(
                    "Saved botaniq authoring blend: %s (variants=%s, season=%s)",
                    authoring_path,
                    len(tree_collection.objects),
                    season_name,
                )

            if bool(runtime_cfg.output.save_production):
                scene = _reset_main_scene(
                    f"BotaniqAssetPool_{safe_species}_{season_name}_production"
                )
                scatter_cfg = _build_stage_scatter_cfg(
                    runtime_cfg.scatter,
                    species_label,
                    runtime_cfg.generation.variants_per_species,
                    stage="production",
                    variant_recipes=variant_recipes or None,
                    production_apply_modifiers=bool(runtime_cfg.output.production_apply_modifiers),
                )
                tree_collection, _variant_species, production_metadata = (
                    create_botaniq_prebake_tree_collection(
                        scene=scene,
                        cfg=scatter_cfg,
                        seed=season_seed,
                        season=source_season_name,
                    )
                )
                _prepare_tree_collection_for_pool(
                    tree_collection,
                    species_key,
                    species_label,
                    season_name,
                    collection_name,
                    bool(runtime_cfg.output.layout_for_inspection),
                    float(runtime_cfg.output.inspection_spacing_m),
                )
                production_path = pool_dir / (
                    f"{safe_species}_{season_name}{runtime_cfg.output.production_suffix}.blend"
                )
                _save_pool_blend(production_path, bool(runtime_cfg.output.overwrite))
                collection_metadata = dict(production_metadata)
                if not variant_recipes:
                    raw_recipes = production_metadata.get("botaniq_variant_recipes", [])
                    if isinstance(raw_recipes, list):
                        variant_recipes = [
                            recipe for recipe in raw_recipes if isinstance(recipe, dict)
                        ]
                logger.info(
                    "Saved botaniq production blend: %s (variants=%s, season=%s)",
                    production_path,
                    len(tree_collection.objects),
                    season_name,
                )

            if not collection_metadata:
                raise RuntimeError(
                    "Botaniq asset pool generation produced no metadata. "
                    "Enable output.save_authoring or output.save_production."
                )

            recipe_path = pool_dir / (
                f"{safe_species}_{season_name}{runtime_cfg.output.recipe_suffix}.json"
            )
            recipe_payload = _recipe_payload(
                species_key=species_key,
                species_label=species_label,
                season_name=season_name,
                requested_seasons=requested_seasons,
                seed=season_seed,
                collection_metadata={
                    **collection_metadata,
                    "botaniq_variant_recipes": variant_recipes,
                },
                authoring_file=authoring_path.name if authoring_path else None,
                production_file=production_path.name if production_path else None,
            )
            recipe_path.write_text(
                json.dumps(recipe_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Saved botaniq variant recipes: %s", recipe_path)

            primary_asset_file = (
                production_path.name
                if production_path is not None
                else authoring_path.name
                if authoring_path is not None
                else ""
            )
            if not primary_asset_file:
                raise RuntimeError("No botaniq asset file was saved for this pool.")

            pool_entry = {
                "season": season_name,
                "object_season": season_name,
                "source_season": source_season_name,
                "requested_seasons": list(requested_seasons),
                "collection": collection_name,
                "asset_file": primary_asset_file,
                "production_asset_file": production_path.name if production_path else None,
                "authoring_asset_file": authoring_path.name if authoring_path else None,
                "recipe_file": recipe_path.name,
                "variant_count": int(recipe_payload.get("variant_count", 0)),
                "source_requested_season": collection_metadata.get(
                    "requested_season",
                    source_season_name,
                ),
                "botaniq_asset_titles": sorted(
                    {str(title) for title in collection_metadata.get("botaniq_asset_titles", [])}
                ),
                "botaniq_source_asset_ids": list(season_group.source_asset_ids),
                "botaniq_source_asset_allocation": dict(
                    collection_metadata.get("botaniq_source_asset_allocation", {})
                ),
                "botaniq_branch_groups_used": list(
                    collection_metadata.get("botaniq_branch_groups_used", [])
                ),
                "botaniq_branch_variation_roles": list(
                    collection_metadata.get("botaniq_branch_variation_roles", [])
                ),
                "botaniq_branch_vertices_changed": int(
                    collection_metadata.get("botaniq_branch_vertices_changed", 0)
                ),
                "botaniq_leaf_density_removed_faces": int(
                    collection_metadata.get("botaniq_leaf_density_removed_faces", 0)
                ),
                "botaniq_leaf_density_removed_units": int(
                    collection_metadata.get("botaniq_leaf_density_removed_units", 0)
                ),
                "botaniq_leaf_material_variation_slots": int(
                    collection_metadata.get("botaniq_leaf_material_variation_slots", 0)
                ),
                "botaniq_leaf_material_variation_nodes": int(
                    collection_metadata.get("botaniq_leaf_material_variation_nodes", 0)
                ),
                "botaniq_target_height_range_m": list(
                    collection_metadata.get("botaniq_target_height_range_m", [])
                ),
                "botaniq_height_priors_expected_stage": str(
                    collection_metadata.get("botaniq_height_priors_expected_stage", "")
                ),
                "botaniq_height_priors_applied_at_prebake": bool(
                    collection_metadata.get(
                        "botaniq_height_priors_applied_at_prebake",
                        False,
                    )
                ),
                "botaniq_species_height_priors": dict(
                    collection_metadata.get("botaniq_species_height_priors", {})
                ),
                "botaniq_variant_recipe_schema": recipe_payload.get("schema"),
                "botaniq_variant_quality_summary": dict(
                    collection_metadata.get("botaniq_variant_quality_summary", {})
                ),
            }

            for requested_season in requested_seasons:
                pools_by_season[requested_season] = {
                    **pool_entry,
                    "requested_season": requested_season,
                }

        manifest["species"].append(
            {
                "species": species_key,
                "species_label": species_label,
                "seed": species_seed,
                "available_seasons": sorted(pools_by_season.keys()),
                "pools": pools_by_season,
                "botaniq_asset_titles": sorted({asset.title for asset in assets}),
                "crown_diameter_range_m": _positive_range([asset.width_m for asset in assets]),
                "height_range_m": _positive_range([asset.height_m for asset in assets]),
            }
        )

    manifest_path = pool_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Botaniq asset pool manifest saved: %s", manifest_path)
