from __future__ import annotations

import math
import random
from collections.abc import Mapping

from ..random_streams import make_random_stream

# Versioned stream names keep per-scene choices deterministic while avoiding
# accidental correlation with other first-draw uses of the scene seed.
DENSITY_CONTEXT_MODE_RANDOM_STREAM = "tree_scatter.density_context_mode.v1"

_DENSITY_MODE_CHOICES = {
    "dense": {
        "none": 1.0,
    },
    "medium": {
        "none": 0.25,
        "trails_only": 0.30,
        "river_only": 0.20,
        "river_and_trail": 0.25,
    },
    "sparse": {
        "none": 0.05,
        "trails_only": 0.15,
        "river_only": 0.15,
        "river_and_trail": 0.45,
        "sparse_rural_artifacts": 0.20,
    },
}

_DENSITY_PROFILE_CHOICES = {
    "dense": {
        "forest_sparse_context": 1.0,
    },
    "medium": {
        "forest_sparse_context": 0.55,
        "forest_river_bias": 0.45,
    },
    "sparse": {
        "forest_sparse_buildings": 0.60,
        "forest_rural_building_dominant": 0.40,
    },
}


def normalize_density_band(density_band: str | None) -> str:
    band = str(density_band or "").strip().lower()
    return band if band in _DENSITY_MODE_CHOICES else "dense"


def _normalize_choice_mapping(raw_choices: object) -> dict[str, float]:
    if not isinstance(raw_choices, Mapping):
        return {}
    choices: dict[str, float] = {}
    for name, weight in raw_choices.items():
        normalized_name = str(name).strip()
        if not normalized_name:
            continue
        choices[normalized_name] = max(0.0, float(weight))
    return {name: weight for name, weight in choices.items() if weight > 0.0}


def _choices_for_density_band(
    defaults: dict[str, dict[str, float]],
    density_band: str | None,
    weights_by_density_band: object | None,
) -> tuple[str, dict[str, float]]:
    band = normalize_density_band(density_band)
    if isinstance(weights_by_density_band, Mapping):
        override_choices = _normalize_choice_mapping(weights_by_density_band.get(band))
        if override_choices:
            return band, override_choices
    return band, defaults[band]


def pick_density_context_profile(
    density_band: str | None,
    seed: int,
    weights_by_density_band: object | None = None,
) -> str:
    _band, choices = _choices_for_density_band(
        _DENSITY_PROFILE_CHOICES,
        density_band,
        weights_by_density_band,
    )
    rng = random.Random(int(seed) + 1223)
    names = list(choices)
    weights = [choices[name] for name in names]
    return str(rng.choices(names, weights=weights, k=1)[0])


def pick_density_context_mode(
    density_band: str | None,
    seed: int,
    weights_by_density_band: object | None = None,
) -> str:
    _band, choices = _choices_for_density_band(
        _DENSITY_MODE_CHOICES,
        density_band,
        weights_by_density_band,
    )
    rng = make_random_stream(seed, DENSITY_CONTEXT_MODE_RANDOM_STREAM)
    modes = list(choices)
    weights = [choices[mode] for mode in modes]
    return str(rng.choices(modes, weights=weights, k=1)[0])


def scale_count_range(
    count_min: int,
    count_max: int,
    *,
    terrain_size: float,
    reference_size: float,
    scale_mode: str,
    require_at_least_one: bool = False,
) -> tuple[int, int]:
    low = max(0, int(count_min))
    high = max(0, int(count_max))
    if high < low:
        low, high = high, low

    reference = max(1e-6, float(reference_size))
    linear_scale = max(0.0, float(terrain_size)) / reference
    scale = linear_scale**2 if str(scale_mode).strip().lower() == "area" else linear_scale

    scaled_low = low
    scaled_high = max(0, int(math.floor(high * scale + 1e-9)))

    if require_at_least_one and high > 0:
        scaled_low = max(1, scaled_low)
        scaled_high = max(1, scaled_high)

    if scaled_high < scaled_low:
        scaled_high = scaled_low
    return scaled_low, scaled_high
