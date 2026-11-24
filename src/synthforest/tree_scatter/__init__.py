"""Tree scatter scene generation package."""

from typing import Any

__all__ = [
    "BotaniqAssetPoolConfig",
    "load_botaniq_asset_pool_config",
    "TreeScatterConfig",
    "load_tree_scatter_config",
    "run_botaniq_asset_pool_pipeline",
    "run_tree_scatter_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"BotaniqAssetPoolConfig", "load_botaniq_asset_pool_config"}:
        from .asset_pool.botaniq_config import (
            BotaniqAssetPoolConfig,
            load_botaniq_asset_pool_config,
        )

        return {
            "BotaniqAssetPoolConfig": BotaniqAssetPoolConfig,
            "load_botaniq_asset_pool_config": load_botaniq_asset_pool_config,
        }[name]
    if name == "TreeScatterConfig" or name == "load_tree_scatter_config":
        from .config import TreeScatterConfig, load_tree_scatter_config

        return {
            "TreeScatterConfig": TreeScatterConfig,
            "load_tree_scatter_config": load_tree_scatter_config,
        }[name]
    if name == "run_botaniq_asset_pool_pipeline":
        from .asset_pool.botaniq_pipeline import run_botaniq_asset_pool_pipeline

        return run_botaniq_asset_pool_pipeline
    if name == "run_tree_scatter_pipeline":
        from .pipeline import run_tree_scatter_pipeline

        return run_tree_scatter_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
