from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")

_PROFILE_TAGS = {
    "small_medium_crowns": "smc",
    "large_closed_canopy": "lcc",
    "hard_negative_sparse_context": "hneg",
}

_SEASON_TAGS = {
    "spring": "spr",
    "summer": "sum",
    "autumn": "aut",
    "fall": "aut",
    "winter": "win",
}


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("-_") or "run"


def resolve_output_dir(
    output_root: str | Path,
    run_name: str,
    append_run_name: bool = True,
) -> Path:
    root = Path(str(output_root)).expanduser()
    if not append_run_name:
        return root
    if _TIMESTAMP_RE.search(root.name) or root.name == run_name:
        return root
    return root / run_name


def resolve_log_file(log_root_or_file: str | Path, run_name: str) -> Path:
    path = Path(str(log_root_or_file)).expanduser()
    if path.suffix:
        if _TIMESTAMP_RE.search(path.stem) or path.stem == run_name:
            return path
        return path.parent / path.stem / f"{run_name}{path.suffix}"
    return path / f"{run_name}.log"


def tree_scatter_run_name(cfg: DictConfig, timestamp: str | None = None) -> str:
    explicit = _select_optional_str(cfg, "output.run_name")
    if explicit:
        return slugify(explicit)

    stamp = timestamp or timestamp_token()
    profile_tag = _target_profile_tag(
        _as_mapping(OmegaConf.select(cfg, "generation.target_profile_weights"))
    )
    gsd_tag = _gsd_tag(_select_float(cfg, "render.gsd_at_nadir", 0.1))
    size_tag = _scene_size_tag(cfg)
    season_tag = _season_tag(_as_mapping(OmegaConf.select(cfg, "scatter.season_weights")))
    num_scenes = max(1, int(_select_float(cfg, "generation.num_scenes", 1)))
    return "_".join(
        [
            "tree_scatter",
            profile_tag,
            gsd_tag,
            size_tag,
            season_tag,
            f"n{num_scenes}",
            stamp,
        ]
    )


def botaniq_asset_pool_run_name(cfg: DictConfig, timestamp: str | None = None) -> str:
    explicit = _select_optional_str(cfg, "output.run_name")
    if explicit:
        return slugify(explicit)

    stamp = timestamp or timestamp_token()
    species_tag = _species_tag(OmegaConf.select(cfg, "generation.species", default=[]))
    variants = max(1, int(_select_float(cfg, "generation.variants_per_species", 12)))
    season_mode = slugify(OmegaConf.select(cfg, "generation.season_mode", default="summer"))
    stage_tag = _pool_stage_tag(cfg)
    return "_".join(
        [
            "botaniq_pool",
            species_tag,
            f"v{variants}",
            season_mode,
            stage_tag,
            stamp,
        ]
    )


def _select_optional_str(cfg: DictConfig, key: str) -> str:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return ""
    return str(value).strip()


def _select_float(cfg: DictConfig, key: str, default: float) -> float:
    value = OmegaConf.select(cfg, key, default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        return {}
    output = {}
    for key, weight in value.items():
        try:
            output[str(key)] = float(weight)
        except (TypeError, ValueError):
            continue
    return output


def _positive_weight_keys(weights: dict[str, float]) -> list[str]:
    return [key for key, weight in weights.items() if weight > 0.0]


def _target_profile_tag(weights: dict[str, float]) -> str:
    positive = _positive_weight_keys(weights)
    if len(positive) == 1:
        profile = positive[0]
        return _PROFILE_TAGS.get(profile, slugify(profile))
    if len(positive) > 1:
        return f"mix{len(positive)}profiles"
    return "biome_mix"


def _gsd_tag(gsd_m: float) -> str:
    cm = float(gsd_m) * 100.0
    if abs(cm - round(cm)) < 1e-6:
        return f"gsd{int(round(cm))}cm"
    return f"gsd{_format_number(cm)}cm"


def _scene_size_tag(cfg: DictConfig) -> str:
    gsd = max(1e-6, _select_float(cfg, "render.gsd_at_nadir", 0.1))
    resolution_x = max(1, int(_select_float(cfg, "render.resolution_x", 1024)))
    resolution_y = max(1, int(_select_float(cfg, "render.resolution_y", 1024)))
    fallback_size = max(1.0, _select_float(cfg, "terrain.size", 100.0))
    width = max(fallback_size, resolution_x * gsd)
    height = max(fallback_size, resolution_y * gsd)
    if abs(width - height) < 1e-6:
        return f"size{_format_number(width)}m"
    return f"size{_format_number(width)}x{_format_number(height)}m"


def _season_tag(weights: dict[str, float]) -> str:
    positive = [(key, weight) for key, weight in weights.items() if weight > 0.0]
    if not positive:
        return "season_default"
    if len(positive) == 1:
        return _short_season_name(positive[0][0])

    total = sum(weight for _key, weight in positive)
    parts = []
    for key, weight in sorted(positive):
        pct = int(round((weight / total) * 100.0)) if total > 0 else int(round(weight))
        parts.append(f"{_short_season_name(key)}{pct}")
    return "-".join(parts)


def _short_season_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return _SEASON_TAGS.get(normalized, slugify(normalized)[:3] or "sea")


def _species_tag(species: object) -> str:
    if OmegaConf.is_config(species):
        species = OmegaConf.to_container(species, resolve=True)
    if not species or species == ["*"]:
        return "all"
    if isinstance(species, list):
        if len(species) == 1:
            return slugify(species[0])[:24]
        return f"{len(species)}species"
    return slugify(species)[:24]


def _pool_stage_tag(cfg: DictConfig) -> str:
    save_authoring = bool(OmegaConf.select(cfg, "output.save_authoring", default=True))
    save_production = bool(OmegaConf.select(cfg, "output.save_production", default=True))
    if save_authoring and save_production:
        return "auth-prod"
    if save_authoring:
        return "auth"
    return "prod"


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3g}".replace(".", "p")
