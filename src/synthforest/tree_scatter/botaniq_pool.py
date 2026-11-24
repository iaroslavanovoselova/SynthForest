from __future__ import annotations

import json
import logging
from pathlib import Path

from .botaniq_catalog import _BotaniqAsset
from .botaniq_utils import (
    as_list as _as_list,
)
from .botaniq_utils import (
    as_mapping as _as_mapping,
)
from .botaniq_utils import (
    coerce_float as _coerce_float,
)
from .botaniq_utils import (
    normalize_season_name as _normalize_season_name,
)
from .botaniq_utils import (
    normalize_species_key as _normalize_species_key,
)
from .config import ScatterConfig

logger = logging.getLogger(__name__)

_POOL_METADATA_KEYS = (
    "authoring_asset_file",
    "production_asset_file",
    "recipe_file",
    "season",
    "object_season",
    "source_season",
    "requested_seasons",
    "requested_season",
    "source_requested_season",
    "botaniq_asset_titles",
    "botaniq_source_asset_ids",
    "botaniq_source_asset_allocation",
    "botaniq_branch_groups_used",
    "botaniq_branch_variation_roles",
    "botaniq_branch_vertices_changed",
    "botaniq_leaf_density_removed_faces",
    "botaniq_leaf_density_removed_units",
    "botaniq_leaf_material_variation_slots",
    "botaniq_leaf_material_variation_nodes",
    "botaniq_target_height_range_m",
    "botaniq_height_priors_expected_stage",
    "botaniq_height_priors_applied_at_prebake",
    "botaniq_species_height_priors",
    "botaniq_variant_recipe_schema",
)

_SPECIES_METADATA_KEYS = (
    "botaniq_asset_titles",
    "crown_diameter_range_m",
    "height_range_m",
    "seed",
)


def _positive_int_or_zero(value: object) -> int:
    try:
        number = int(value)
    except Exception:
        return 0
    return max(0, number)


def _copy_metadata(source: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: source[key] for key in keys if key in source}


def _load_pool_manifest(pool_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = pool_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}

    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read asset pool manifest: %s", manifest_path)
        return {}

    species_entries = loaded.get("species", []) if isinstance(loaded, dict) else []
    if not isinstance(species_entries, list):
        return {}

    manifest_source = str(loaded.get("source", "") if isinstance(loaded, dict) else "").strip()
    mapping: dict[str, dict[str, object]] = {}
    for entry in species_entries:
        if not isinstance(entry, dict):
            continue
        species_name = _normalize_species_key(str(entry.get("species", "")))
        if not species_name:
            continue
        species_label = str(entry.get("species_label", "") or species_name).strip()

        pools = entry.get("pools")
        if isinstance(pools, dict) and pools:
            normalized_pools: dict[str, dict[str, object]] = {}
            for season_name, pool_entry in pools.items():
                if not isinstance(pool_entry, dict):
                    continue
                asset_file = str(pool_entry.get("asset_file", "")).strip()
                collection_name = str(pool_entry.get("collection", "")).strip()
                normalized_season = _normalize_season_name(str(season_name))
                if asset_file and collection_name:
                    normalized_pools[normalized_season] = {
                        "asset_file": asset_file,
                        "collection": collection_name,
                        "variant_count": _positive_int_or_zero(pool_entry.get("variant_count")),
                        **_copy_metadata(pool_entry, _POOL_METADATA_KEYS),
                    }
            if normalized_pools:
                mapping[species_name] = {
                    "pools": normalized_pools,
                    "available_seasons": sorted(normalized_pools.keys()),
                    "species_label": species_label,
                    "manifest_source": manifest_source,
                    **_copy_metadata(entry, _SPECIES_METADATA_KEYS),
                }
                continue

        asset_file = str(entry.get("asset_file", f"{species_name}.blend")).strip()
        collection_name = str(entry.get("collection", f"TreeAssets_{species_name}")).strip()
        if asset_file and collection_name:
            mapping[species_name] = {
                "pools": {
                    "summer": {
                        "asset_file": asset_file,
                        "collection": collection_name,
                        "variant_count": _positive_int_or_zero(entry.get("variant_count")),
                        **_copy_metadata(entry, _POOL_METADATA_KEYS),
                    }
                },
                "available_seasons": ["summer"],
                "species_label": species_label,
                "manifest_source": manifest_source,
                **_copy_metadata(entry, _SPECIES_METADATA_KEYS),
            }
    return mapping


def _pool_manifest_is_botaniq(manifest_entries: dict[str, dict[str, object]]) -> bool:
    return any(
        str(entry.get("manifest_source", "")).strip().lower()
        in {"botaniq_asset_pool", "botaniq_asset_pack"}
        for entry in manifest_entries.values()
    )


def load_botaniq_pool_manifest(pool_dir: Path) -> dict[str, dict[str, object]]:
    manifest_entries = _load_pool_manifest(pool_dir)
    if not manifest_entries or not _pool_manifest_is_botaniq(manifest_entries):
        raise ValueError(
            "scatter.tree_source_mode=botaniq_asset_pool requires a Botaniq pool manifest "
            "with source='botaniq_asset_pool' or a compatible legacy Botaniq source: "
            f"{pool_dir / 'manifest.json'}"
        )
    return manifest_entries


def _configured_pool_species_from_raw(
    raw_species: object,
    available_species: list[str],
) -> list[str]:
    configured = [
        _normalize_species_key(str(item)) for item in _as_list(raw_species) if str(item).strip()
    ]
    if not configured or "*" in configured or "all" in configured:
        return list(available_species)

    available = set(available_species)
    unknown = sorted(species for species in configured if species not in available)
    if unknown:
        raise ValueError(
            "Unknown species in asset pool selection: "
            f"{', '.join(unknown)}. Available: {', '.join(available_species)}"
        )
    return list(dict.fromkeys(configured))


def _range_midpoint(raw: object) -> float:
    values = [_coerce_float(item, 0.0) for item in _as_list(raw) if _coerce_float(item, 0.0) > 0.0]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _botaniq_pool_catalog_from_manifest(
    pool_dir: Path,
    manifest_entries: dict[str, dict[str, object]],
) -> dict[str, list[_BotaniqAsset]]:
    catalog: dict[str, list[_BotaniqAsset]] = {}
    for species_key, entry in sorted(manifest_entries.items()):
        pools = _as_mapping(entry.get("pools"))
        if not pools:
            continue

        seasons = tuple(
            dict.fromkeys(_normalize_season_name(str(season_name)) for season_name in pools)
        ) or ("summer",)
        first_pool = next(
            (pool for pool in pools.values() if isinstance(pool, dict)),
            {},
        )
        asset_file = str(first_pool.get("asset_file", "") if isinstance(first_pool, dict) else "")
        blend_path = (pool_dir / asset_file).resolve() if asset_file else pool_dir
        asset_titles = _as_list(
            first_pool.get("botaniq_asset_titles", []) if isinstance(first_pool, dict) else []
        )
        title = (
            str(asset_titles[0]).strip()
            if asset_titles
            else str(entry.get("species_label", species_key))
        )

        catalog[species_key] = [
            _BotaniqAsset(
                asset_id=f"asset_pool:{species_key}",
                title=title,
                species_key=species_key,
                species_label=str(entry.get("species_label", species_key) or species_key),
                species_common_name="",
                blend_path=blend_path,
                seasons=seasons,
                height_m=_range_midpoint(entry.get("height_range_m", [])),
                width_m=_range_midpoint(entry.get("crown_diameter_range_m", [])),
                depth_m=_range_midpoint(entry.get("crown_diameter_range_m", [])),
                model_detail="prebaked_asset_pool",
            )
        ]
    return catalog


def _filtered_botaniq_pool_catalog(
    cfg: ScatterConfig,
    catalog: dict[str, list[_BotaniqAsset]],
) -> dict[str, list[_BotaniqAsset]]:
    explicit_pool_species = _as_list(getattr(cfg, "asset_pool_species", []))
    if not explicit_pool_species:
        return dict(catalog)

    selected = _configured_pool_species_from_raw(explicit_pool_species, sorted(catalog))
    return {species_key: catalog[species_key] for species_key in selected}
