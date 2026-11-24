from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from synthforest.paths import external_root, project_root

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
from .botaniq_utils import (
    resolve_project_path as _resolve_project_path,
)
from .config import ScatterConfig

logger = logging.getLogger(__name__)
PROJECT_ROOT = project_root()


@dataclass(frozen=True)
class _BotaniqAsset:
    asset_id: str
    title: str
    species_key: str
    species_label: str
    species_common_name: str
    blend_path: Path
    seasons: tuple[str, ...]
    height_m: float
    width_m: float
    depth_m: float
    model_detail: str


@dataclass(frozen=True)
class _BotaniqSpeciesHeightPrior:
    species_key: str
    species_label: str
    common_name: str
    height_min_m: float
    height_mode_m: float
    height_max_m: float
    source: str


@dataclass(frozen=True)
class _BotaniqHeightScalePlan:
    scale_xy: float
    scale_z: float
    anisotropy: float
    target_height_m: float | None
    unclamped_target_height_m: float | None
    clamped: bool
    clamp_reason: str


def _resolve_botaniq_pack_dir(cfg: ScatterConfig) -> Path:
    configured = str(getattr(cfg, "botaniq_pack_dir", "") or "").strip()
    candidates: list[Path] = []

    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        candidates.append(candidate)

    asset_root = external_root("SYNTHFOREST_ASSET_ROOT", required=False)
    if asset_root is not None:
        candidates.extend(
            [
                asset_root / "botaniq_starter",
                asset_root / "botaniq_lite",
                asset_root / "botaniq_full",
                asset_root / "botaniq",
            ]
        )
        if asset_root.is_dir():
            candidates.extend(sorted(path for path in asset_root.glob("botaniq*") if path.is_dir()))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "mapr_index.json").is_file():
            return resolved

    searched = ", ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "Could not find a botaniq asset pack with mapr_index.json. "
        f"Configured scatter.botaniq_pack_dir='{configured}'. Searched: {searched}. "
        "Set SYNTHFOREST_ASSET_ROOT or scatter.botaniq_pack_dir."
    )


def _resolve_botaniq_pack_file(pack_dir: Path, file_id: str) -> Path:
    raw = str(file_id).strip()
    if not raw:
        raise ValueError("Empty botaniq file id")
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    raw = raw.lstrip("/")
    return (pack_dir / raw).resolve()


def _botaniq_species_height_config_path(cfg: ScatterConfig) -> Path | None:
    raw_path = str(getattr(cfg, "botaniq_species_height_config_path", "") or "").strip()
    if not raw_path:
        return None
    return _resolve_project_path(raw_path)


def _iter_botaniq_species_height_entries(
    config_data: object,
) -> list[tuple[str, dict[str, object], str]]:
    if not isinstance(config_data, dict):
        return []

    entries: list[tuple[str, dict[str, object], str]] = []
    species_map = _as_mapping(config_data.get("species"))
    for raw_key, raw_entry in species_map.items():
        if isinstance(raw_entry, dict):
            entries.append((str(raw_key), raw_entry, "species"))

    packs = _as_mapping(config_data.get("packs"))
    for pack_name, pack_data in packs.items():
        if not isinstance(pack_data, dict):
            continue
        pack_species = _as_mapping(pack_data.get("species"))
        for raw_key, raw_entry in pack_species.items():
            if isinstance(raw_entry, dict):
                entries.append((str(raw_key), raw_entry, f"packs.{pack_name}.species"))

    return entries


def _load_botaniq_species_height_priors(
    cfg: ScatterConfig,
) -> dict[str, _BotaniqSpeciesHeightPrior]:
    if not bool(getattr(cfg, "botaniq_use_species_height_priors", True)):
        return {}

    config_path = _botaniq_species_height_config_path(cfg)
    if config_path is None:
        return {}
    if not config_path.is_file():
        logger.warning("Botaniq species height-prior config not found: %s", config_path)
        return {}

    try:
        loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception as exc:
        logger.warning(
            "Failed to load botaniq species height-prior config %s: %s", config_path, exc
        )
        return {}

    priors: dict[str, _BotaniqSpeciesHeightPrior] = {}
    for raw_key, entry, source in _iter_botaniq_species_height_entries(loaded):
        species_label = str(entry.get("species_label") or entry.get("species") or raw_key).strip()
        species_key = _normalize_species_key(
            str(entry.get("species_key") or species_label or raw_key)
        )
        if not species_key:
            continue

        height_min = _coerce_float(
            entry.get("height_min_m", entry.get("min_height_m")),
            0.0,
        )
        height_max = _coerce_float(
            entry.get("height_max_m", entry.get("max_height_m")),
            0.0,
        )
        if height_min <= 0.0 or height_max <= 0.0:
            logger.warning(
                "Skipping invalid botaniq height prior for '%s': min=%s max=%s",
                species_label or raw_key,
                height_min,
                height_max,
            )
            continue
        if height_max < height_min:
            height_min, height_max = height_max, height_min
        height_mode = _coerce_float(
            entry.get("height_mode_m", entry.get("mode_height_m")),
            (height_min + height_max) * 0.5,
        )
        height_mode = max(height_min, min(height_max, height_mode))

        priors[species_key] = _BotaniqSpeciesHeightPrior(
            species_key=species_key,
            species_label=species_label or raw_key,
            common_name=str(entry.get("common_name", "") or "").strip(),
            height_min_m=float(height_min),
            height_mode_m=float(height_mode),
            height_max_m=float(height_max),
            source=f"{config_path}:{source}.{raw_key}",
        )

    if not priors:
        logger.warning(
            "Botaniq species height-prior config has no valid species entries: %s",
            config_path,
        )
    return priors


def _sample_botaniq_target_height_m(
    asset: _BotaniqAsset,
    height_priors: dict[str, _BotaniqSpeciesHeightPrior],
    rng: random.Random,
) -> tuple[float | None, _BotaniqSpeciesHeightPrior | None]:
    prior = height_priors.get(asset.species_key)
    if prior is None:
        return None, None
    return (
        rng.triangular(prior.height_min_m, prior.height_max_m, prior.height_mode_m),
        prior,
    )


def _plan_botaniq_variant_scales(
    asset: _BotaniqAsset,
    target_height_m: float | None,
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
    height_scale_min: float,
    height_scale_max: float,
    crown_scale_power: float,
) -> tuple[float, float]:
    plan = _plan_botaniq_variant_height_scale(
        asset,
        target_height_m,
        rng,
        scale_min=scale_min,
        scale_max=scale_max,
        height_scale_min=height_scale_min,
        height_scale_max=height_scale_max,
        crown_scale_power=crown_scale_power,
        scale_z_max=float("inf"),
        scale_xy_max=float("inf"),
        anisotropy_max=float("inf"),
    )
    return float(plan.scale_xy), float(plan.scale_z)


def _plan_botaniq_direct_pack_scales(
    asset: _BotaniqAsset,
    target_height_m: float | None,
    *,
    crown_scale_power: float,
) -> tuple[float, float]:
    plan = _plan_botaniq_variant_height_scale(
        asset,
        target_height_m,
        None,
        scale_min=1.0,
        scale_max=1.0,
        height_scale_min=1.0,
        height_scale_max=1.0,
        crown_scale_power=crown_scale_power,
        scale_z_max=float("inf"),
        scale_xy_max=float("inf"),
        anisotropy_max=float("inf"),
    )
    return float(plan.scale_xy), float(plan.scale_z)


def _scale_anisotropy(scale_xy: float, scale_z: float) -> float:
    return float(scale_z) / max(float(scale_xy), 1e-6)


def _target_height_max_for_scale_caps(
    asset: _BotaniqAsset,
    scale_noise: float,
    *,
    crown_scale_power: float,
    scale_z_max: float,
    scale_xy_max: float,
    anisotropy_max: float,
) -> tuple[float, list[str]]:
    source_height = max(float(asset.height_m), 1e-6)
    max_scale_z = max(0.01, float(scale_z_max))
    max_scale_xy = max(0.01, float(scale_xy_max))
    max_anisotropy = max(0.01, float(anisotropy_max))
    noise = max(0.01, float(scale_noise))
    power = max(0.0, float(crown_scale_power))

    candidates: list[tuple[str, float]] = [("scale_z_max", max_scale_z)]
    if power > 1e-8:
        candidates.append(("scale_xy_max", (max_scale_xy / noise) ** (1.0 / power)))
    if power < 1.0 - 1e-8:
        candidates.append(("anisotropy_max", (max_anisotropy * noise) ** (1.0 / (1.0 - power))))

    limit_name, scale_z_limit = min(candidates, key=lambda item: item[1])
    limiting_reasons = [name for name, value in candidates if value <= scale_z_limit + 1e-6]
    if limit_name not in limiting_reasons:
        limiting_reasons.append(limit_name)
    return source_height * max(0.01, scale_z_limit), limiting_reasons


def _plan_botaniq_height_scale_from_noise(
    asset: _BotaniqAsset,
    target_height_m: float | None,
    *,
    scale_noise: float,
    fallback_height_scale: float,
    crown_scale_power: float,
    scale_z_max: float,
    scale_xy_max: float,
    anisotropy_max: float,
) -> _BotaniqHeightScalePlan:
    if target_height_m is None or asset.height_m <= 0.0:
        scale_z = max(0.01, min(float(fallback_height_scale), float(scale_z_max)))
        scale_xy = max(0.01, min(float(scale_noise), float(scale_xy_max)))
        anisotropy = _scale_anisotropy(scale_xy, scale_z)
        if anisotropy > anisotropy_max:
            scale_z = max(0.01, float(anisotropy_max) * scale_xy)
            anisotropy = _scale_anisotropy(scale_xy, scale_z)
        return _BotaniqHeightScalePlan(
            scale_xy=float(scale_xy),
            scale_z=float(scale_z),
            anisotropy=float(anisotropy),
            target_height_m=None,
            unclamped_target_height_m=None,
            clamped=False,
            clamp_reason="",
        )

    source_height = max(float(asset.height_m), 1e-6)
    unclamped_target = max(0.01, float(target_height_m))
    max_target, limiting_reasons = _target_height_max_for_scale_caps(
        asset,
        scale_noise,
        crown_scale_power=crown_scale_power,
        scale_z_max=scale_z_max,
        scale_xy_max=scale_xy_max,
        anisotropy_max=anisotropy_max,
    )
    clamped_target = min(unclamped_target, max_target)
    scale_z = max(0.01, clamped_target / source_height)
    scale_xy = (scale_z ** max(0.0, float(crown_scale_power))) * max(0.01, float(scale_noise))
    scale_xy = min(scale_xy, max(0.01, float(scale_xy_max)))
    anisotropy = _scale_anisotropy(scale_xy, scale_z)
    if anisotropy > anisotropy_max:
        scale_z = max(0.01, float(anisotropy_max) * scale_xy)
        clamped_target = min(clamped_target, scale_z * source_height)
        anisotropy = _scale_anisotropy(scale_xy, scale_z)

    clamped = clamped_target < unclamped_target - 1e-6
    return _BotaniqHeightScalePlan(
        scale_xy=float(scale_xy),
        scale_z=float(scale_z),
        anisotropy=float(anisotropy),
        target_height_m=float(clamped_target),
        unclamped_target_height_m=float(unclamped_target),
        clamped=bool(clamped),
        clamp_reason="+".join(limiting_reasons) if clamped else "",
    )


def _plan_botaniq_variant_height_scale(
    asset: _BotaniqAsset,
    target_height_m: float | None,
    rng: random.Random | None,
    *,
    scale_min: float,
    scale_max: float,
    height_scale_min: float,
    height_scale_max: float,
    crown_scale_power: float,
    scale_z_max: float,
    scale_xy_max: float,
    anisotropy_max: float,
) -> _BotaniqHeightScalePlan:
    scale_min = max(0.01, float(scale_min))
    scale_max = max(scale_min, float(scale_max))
    height_scale_min = max(0.01, float(height_scale_min))
    height_scale_max = max(height_scale_min, float(height_scale_max))
    scale_noise = rng.uniform(scale_min, scale_max) if rng is not None else scale_min
    fallback_height_scale = (
        rng.uniform(height_scale_min, height_scale_max) if rng is not None else height_scale_min
    )
    return _plan_botaniq_height_scale_from_noise(
        asset,
        target_height_m,
        scale_noise=scale_noise,
        fallback_height_scale=fallback_height_scale,
        crown_scale_power=crown_scale_power,
        scale_z_max=scale_z_max,
        scale_xy_max=scale_xy_max,
        anisotropy_max=anisotropy_max,
    )


def _choose_botaniq_asset_for_height_target(
    candidates: list[tuple[_BotaniqAsset, int, str]],
    target_height_m: float | None,
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
    height_scale_min: float,
    height_scale_max: float,
    crown_scale_power: float,
    scale_z_max: float,
    scale_xy_max: float,
    anisotropy_max: float,
) -> tuple[_BotaniqAsset, int, str, _BotaniqHeightScalePlan]:
    if not candidates:
        raise ValueError("No botaniq source candidates are available.")

    scale_min = max(0.01, float(scale_min))
    scale_max = max(scale_min, float(scale_max))
    height_scale_min = max(0.01, float(height_scale_min))
    height_scale_max = max(height_scale_min, float(height_scale_max))
    scale_noise = rng.uniform(scale_min, scale_max)
    fallback_height_scale = rng.uniform(height_scale_min, height_scale_max)

    scored: list[
        tuple[
            bool,
            float,
            float,
            int,
            tuple[_BotaniqAsset, int, str],
            _BotaniqHeightScalePlan,
        ]
    ] = []
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for order_index, candidate in enumerate(shuffled):
        asset, _source_index, _resolved_season = candidate
        plan = _plan_botaniq_height_scale_from_noise(
            asset,
            target_height_m,
            scale_noise=scale_noise,
            fallback_height_scale=fallback_height_scale,
            crown_scale_power=crown_scale_power,
            scale_z_max=scale_z_max,
            scale_xy_max=scale_xy_max,
            anisotropy_max=anisotropy_max,
        )
        feasible = not plan.clamped
        target = float(plan.target_height_m or 0.0)
        scale_distance = abs(float(plan.scale_z) - 1.0)
        target_gap = (
            abs(float(target_height_m) - target) if target_height_m is not None else scale_distance
        )
        scored.append((feasible, target_gap, scale_distance, order_index, candidate, plan))

    scored.sort(key=lambda item: (not item[0], item[1], item[2], item[3]))
    _feasible, _gap, _scale_distance, _order, selected_candidate, selected_plan = scored[0]
    asset, source_index, resolved_season = selected_candidate
    return asset, source_index, resolved_season, selected_plan


def _botaniq_asset_seasons(tags: object) -> tuple[str, ...]:
    if not isinstance(tags, list):
        return ("summer",)
    seasons = []
    for tag in tags:
        normalized = str(tag).strip().lower()
        if normalized in {"spring", "summer", "autumn", "winter"}:
            seasons.append(normalized)
    return tuple(dict.fromkeys(seasons)) or ("summer",)


def _load_botaniq_catalog(
    pack_dir: Path,
    include_title_prefixes: list[str],
) -> dict[str, list[_BotaniqAsset]]:
    index_path = pack_dir / "mapr_index.json"
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load botaniq catalog: {index_path}") from exc

    asset_metadata = index_data.get("asset_metadata", {})
    asset_data = index_data.get("asset_data", {})
    child_asset_data = index_data.get("child_asset_data", {})
    if not isinstance(asset_metadata, dict) or not isinstance(asset_data, dict):
        raise RuntimeError(f"Invalid botaniq catalog structure: {index_path}")

    normalized_prefixes = [
        str(prefix).strip().lower() for prefix in include_title_prefixes if str(prefix).strip()
    ]
    if not normalized_prefixes:
        normalized_prefixes = ["tree "]

    catalog: dict[str, list[_BotaniqAsset]] = {}
    for asset_id, metadata in asset_metadata.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("type") != "blender_model":
            continue

        title = str(metadata.get("title", "")).strip()
        if not title or not any(title.lower().startswith(prefix) for prefix in normalized_prefixes):
            continue

        text_params = metadata.get("text_parameters", {})
        numeric_params = metadata.get("numeric_parameters", {})
        if not isinstance(text_params, dict) or not isinstance(numeric_params, dict):
            continue

        species_label = str(text_params.get("species", "")).strip()
        species_key = _normalize_species_key(species_label)
        if not species_key:
            continue

        data_ids = child_asset_data.get(asset_id, [])
        if not isinstance(data_ids, list) or not data_ids:
            continue
        data_entry = asset_data.get(str(data_ids[0]), {})
        if not isinstance(data_entry, dict):
            continue
        primary_blend_file = str(data_entry.get("primary_blend_file", "")).strip()
        if not primary_blend_file:
            continue
        blend_path = _resolve_botaniq_pack_file(pack_dir, primary_blend_file)
        if not blend_path.is_file():
            logger.warning("Skipping botaniq asset with missing blend file: %s", blend_path)
            continue

        asset = _BotaniqAsset(
            asset_id=str(asset_id),
            title=title,
            species_key=species_key,
            species_label=species_label,
            species_common_name=str(text_params.get("species_en", "")).strip(),
            blend_path=blend_path,
            seasons=_botaniq_asset_seasons(metadata.get("tags", [])),
            height_m=_coerce_float(numeric_params.get("height")),
            width_m=_coerce_float(numeric_params.get("width")),
            depth_m=_coerce_float(numeric_params.get("depth")),
            model_detail=str(text_params.get("model_detail", "")).strip(),
        )
        catalog.setdefault(species_key, []).append(asset)

    for assets in catalog.values():
        assets.sort(key=lambda item: (item.species_label, item.title, item.asset_id))
    return catalog


def _pick_botaniq_asset_for_species(
    assets: list[_BotaniqAsset],
    requested_season: str,
    rng: random.Random,
) -> tuple[_BotaniqAsset, str]:
    season = _normalize_season_name(requested_season)
    seasonal = [asset for asset in assets if season in asset.seasons]
    if seasonal:
        return rng.choice(seasonal), season

    summer = [asset for asset in assets if "summer" in asset.seasons]
    if summer:
        return rng.choice(summer), "summer"

    return rng.choice(assets), assets[0].seasons[0] if assets[0].seasons else "summer"


def _botaniq_asset_candidates_for_season(
    assets: list[_BotaniqAsset],
    requested_season: str,
) -> list[tuple[_BotaniqAsset, int, str]]:
    season = _normalize_season_name(requested_season)
    seasonal = [
        (asset, index, season) for index, asset in enumerate(assets) if season in asset.seasons
    ]
    if seasonal:
        return seasonal

    summer = [
        (asset, index, "summer") for index, asset in enumerate(assets) if "summer" in asset.seasons
    ]
    if summer:
        return summer

    return [
        (
            asset,
            index,
            asset.seasons[0] if asset.seasons else "summer",
        )
        for index, asset in enumerate(assets)
    ]


def _balanced_botaniq_assets_for_species(
    assets: list[_BotaniqAsset],
    requested_season: str,
    variant_count: int,
    rng: random.Random,
) -> list[tuple[_BotaniqAsset, int, str]]:
    """Return a shuffled, season-compatible source plan with near-equal counts."""
    count = max(0, int(variant_count))
    candidates = _botaniq_asset_candidates_for_season(assets, requested_season)
    if count <= 0 or not candidates:
        return []

    base_count = count // len(candidates)
    extra_count = count % len(candidates)
    candidate_indices = list(range(len(candidates)))
    rng.shuffle(candidate_indices)
    extra_indices = set(candidate_indices[:extra_count])

    plan: list[tuple[_BotaniqAsset, int, str]] = []
    for candidate_index, candidate in enumerate(candidates):
        repetitions = base_count + (1 if candidate_index in extra_indices else 0)
        plan.extend([candidate] * repetitions)
    rng.shuffle(plan)
    return plan
