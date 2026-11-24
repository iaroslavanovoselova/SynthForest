from __future__ import annotations

import bpy

from .botaniq_templates import (
    create_tree_collection_from_botaniq_asset_pack,
    create_tree_collection_from_botaniq_asset_pool,
)
from .config import ScatterConfig


def _normalize_tree_source_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"botaniq_asset_pack", "botaniq_asset_pool"}:
        raise ValueError(
            f"Unknown scatter.tree_source_mode '{mode}'. "
            "Supported values: botaniq_asset_pack, botaniq_asset_pool"
        )
    return normalized


def create_tree_collection_from_source(
    scene: bpy.types.Scene,
    cfg: ScatterConfig,
    seed: int,
    season: str = "summer",
) -> tuple[bpy.types.Collection, list[str], dict[str, object]]:
    mode = _normalize_tree_source_mode(getattr(cfg, "tree_source_mode", "botaniq_asset_pack"))

    if mode == "botaniq_asset_pool":
        # Height priors are already baked into pool meshes. Scene loading keeps
        # only the small configured instance and material variations.
        return create_tree_collection_from_botaniq_asset_pool(
            scene,
            cfg,
            seed,
            season=season,
        )

    # Direct pack mode has no prebaked mesh cache, so species height priors
    # are applied while constructing the per-scene tree variants.
    return create_tree_collection_from_botaniq_asset_pack(
        scene,
        cfg,
        seed,
        season=season,
    )
