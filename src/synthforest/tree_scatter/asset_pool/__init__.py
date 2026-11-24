"""Tree asset pool generation helpers."""

from typing import Any

__all__ = [
    "BotaniqAssetPoolConfig",
    "load_botaniq_asset_pool_config",
    "run_botaniq_asset_pool_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"BotaniqAssetPoolConfig", "load_botaniq_asset_pool_config"}:
        from .botaniq_config import BotaniqAssetPoolConfig, load_botaniq_asset_pool_config

        return {
            "BotaniqAssetPoolConfig": BotaniqAssetPoolConfig,
            "load_botaniq_asset_pool_config": load_botaniq_asset_pool_config,
        }[name]
    if name == "run_botaniq_asset_pool_pipeline":
        from .botaniq_pipeline import run_botaniq_asset_pool_pipeline

        return run_botaniq_asset_pool_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
