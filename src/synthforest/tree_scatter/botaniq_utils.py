from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def normalize_season_name(season: str | None) -> str:
    normalized = str(season or "summer").strip().lower()
    if normalized not in {"summer", "autumn"}:
        return "summer"
    return normalized


def normalize_species_key(species_name: str) -> str:
    return " ".join(str(species_name).strip().lower().split())


def as_mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def coerce_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback
