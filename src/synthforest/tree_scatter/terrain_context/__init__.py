"""Terrain context generation helpers."""

from typing import Any

__all__ = [
    "TerrainContextResult",
    "create_terrain_with_context",
    "snap_context_buildings_to_terrain",
]


def __getattr__(name: str) -> Any:
    if name in {"TerrainContextResult", "create_terrain_with_context"}:
        from .pipeline import TerrainContextResult, create_terrain_with_context

        return {
            "TerrainContextResult": TerrainContextResult,
            "create_terrain_with_context": create_terrain_with_context,
        }[name]
    if name == "snap_context_buildings_to_terrain":
        from .buildings import snap_context_buildings_to_terrain

        return snap_context_buildings_to_terrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
