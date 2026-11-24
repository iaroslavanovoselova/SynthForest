from __future__ import annotations

import logging
import random
from dataclasses import dataclass

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
    normalize_species_key as _normalize_species_key,
)
from .config import ScatterConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BotaniqStandComposition:
    species_keys: list[str]
    species_weights: dict[str, float]
    species_roles: dict[str, str]
    role_abundances: dict[str, float]
    role_requested_counts: dict[str, int]
    role_selected_counts: dict[str, int]
    role_available_species: dict[str, list[str]]


_BOTANIQ_SPECIES_ROLES = ("dominant", "companion", "rare")
_BOTANIQ_DEFAULT_ROLE_COUNT_RANGES = {
    "dominant": (1, 3),
    "companion": (0, 5),
    "rare": (0, 6),
}
_BOTANIQ_DEFAULT_ROLE_ABUNDANCE_RANGES = {
    "dominant": (0.55, 0.80),
    "companion": (0.15, 0.38),
    "rare": (0.02, 0.12),
}
_BOTANIQ_DEFAULT_ROLE_DIRICHLET_ALPHA = {
    "dominant": 3.0,
    "companion": 1.5,
    "rare": 0.8,
}


def _coerce_numeric_range(
    raw: object,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if isinstance(raw, dict):
        first = raw.get("min", fallback[0])
        second = raw.get("max", fallback[1])
    else:
        values = _as_list(raw)
        if len(values) >= 2:
            first, second = values[0], values[1]
        elif len(values) == 1:
            first = second = values[0]
        else:
            first, second = fallback

    low = _coerce_float(first, fallback[0])
    high = _coerce_float(second, fallback[1])
    if high < low:
        low, high = high, low
    return low, high


def _coerce_int_range(
    raw: object,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    low, high = _coerce_numeric_range(raw, (float(fallback[0]), float(fallback[1])))
    low_i = max(0, int(round(low)))
    high_i = max(0, int(round(high)))
    if high_i < low_i:
        low_i, high_i = high_i, low_i
    return low_i, high_i


def _normalize_float_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        if not positive:
            return {}
        equal = 1.0 / len(positive)
        return {key: equal for key in positive}
    return {key: value / total for key, value in positive.items()}


def _sample_dirichlet(n: int, alpha: float, rng: random.Random) -> list[float]:
    if n <= 0:
        return []
    alpha = max(0.001, float(alpha))
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(draws)
    if total <= 0.0:
        return [1.0 / n] * n
    return [value / total for value in draws]


def _configured_botaniq_role_pool(cfg: ScatterConfig, role: str) -> list[str]:
    role = str(role).strip().lower()
    role_pools = _as_mapping(getattr(cfg, "botaniq_species_role_pools", {}))
    raw_species = _as_list(role_pools.get(role))

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_species:
        key = _normalize_species_key(str(item))
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _available_botaniq_role_pool(
    cfg: ScatterConfig,
    catalog: dict[str, list[_BotaniqAsset]],
    role: str,
) -> list[str]:
    configured = _configured_botaniq_role_pool(cfg, role)
    if not configured:
        return []

    available: list[str] = []
    unknown: list[str] = []
    for species_key in configured:
        if species_key in catalog:
            available.append(species_key)
        else:
            unknown.append(species_key)

    if unknown:
        logger.warning(
            "Skipping unavailable botaniq %s-pool species for this pack: %s",
            role,
            ", ".join(unknown),
        )
    return available


def _build_botaniq_stand_composition(
    cfg: ScatterConfig,
    catalog: dict[str, list[_BotaniqAsset]],
    rng: random.Random,
) -> _BotaniqStandComposition:
    role_count_ranges = _as_mapping(getattr(cfg, "botaniq_role_count_ranges", {}))
    role_abundance_ranges = _as_mapping(getattr(cfg, "botaniq_role_abundance_ranges", {}))
    role_alphas = _as_mapping(getattr(cfg, "botaniq_role_dirichlet_alpha", {}))

    selected_by_role: dict[str, list[str]] = {}
    role_requested_counts: dict[str, int] = {}
    role_available_species: dict[str, list[str]] = {}
    already_selected: set[str] = set()

    for role in _BOTANIQ_SPECIES_ROLES:
        available = [
            species_key
            for species_key in _available_botaniq_role_pool(cfg, catalog, role)
            if species_key not in already_selected
        ]
        role_available_species[role] = [
            catalog[species_key][0].species_label for species_key in available
        ]

        fallback_range = _BOTANIQ_DEFAULT_ROLE_COUNT_RANGES[role]
        min_count, max_count = _coerce_int_range(
            role_count_ranges.get(role),
            fallback_range,
        )
        max_count = min(max_count, len(available))
        min_count = min(min_count, max_count)
        count = rng.randint(min_count, max_count) if max_count > 0 else 0

        selected = rng.sample(available, count) if count > 0 else []
        selected_by_role[role] = selected
        role_requested_counts[role] = count
        already_selected.update(selected)

    selected_roles = [role for role in _BOTANIQ_SPECIES_ROLES if selected_by_role.get(role)]
    if not selected_roles:
        available_names = ", ".join(
            assets[0].species_label for _key, assets in sorted(catalog.items())
        )
        raise RuntimeError(
            "No available botaniq species were selected from role pools. "
            f"Available tree species in this pack: {available_names}"
        )

    raw_role_mass: dict[str, float] = {}
    for role in selected_roles:
        fallback_range = _BOTANIQ_DEFAULT_ROLE_ABUNDANCE_RANGES[role]
        low, high = _coerce_numeric_range(
            role_abundance_ranges.get(role),
            fallback_range,
        )
        low = max(0.0, low)
        high = max(low, high)
        raw_role_mass[role] = rng.uniform(low, high)

    role_abundances = _normalize_float_weights(raw_role_mass)
    species_weights: dict[str, float] = {}
    species_roles: dict[str, str] = {}
    species_keys: list[str] = []

    for role in _BOTANIQ_SPECIES_ROLES:
        selected = selected_by_role.get(role, [])
        if not selected:
            continue

        alpha = _coerce_float(
            role_alphas.get(role),
            _BOTANIQ_DEFAULT_ROLE_DIRICHLET_ALPHA[role],
        )
        within_role = _sample_dirichlet(len(selected), alpha, rng)
        for species_key, share in zip(selected, within_role):
            species_keys.append(species_key)
            species_roles[species_key] = role
            species_weights[species_key] = role_abundances.get(role, 0.0) * share

    species_weights = _normalize_float_weights(species_weights)
    return _BotaniqStandComposition(
        species_keys=species_keys,
        species_weights=species_weights,
        species_roles=species_roles,
        role_abundances=role_abundances,
        role_requested_counts=role_requested_counts,
        role_selected_counts={
            role: len(selected_by_role.get(role, [])) for role in _BOTANIQ_SPECIES_ROLES
        },
        role_available_species=role_available_species,
    )


def _allocate_botaniq_variant_counts(
    species_keys: list[str],
    species_weights: dict[str, float],
    variant_count: int,
) -> dict[str, int]:
    variant_count = max(1, int(variant_count))
    weighted = [
        (species_key, max(0.0, float(species_weights.get(species_key, 0.0))))
        for species_key in species_keys
    ]
    weighted = [(species_key, weight) for species_key, weight in weighted if weight > 0.0]
    if not weighted:
        return {}

    if len(weighted) > variant_count:
        weighted = sorted(weighted, key=lambda item: item[1], reverse=True)[:variant_count]

    active_keys = [species_key for species_key, _weight in weighted]
    normalized = _normalize_float_weights(dict(weighted))
    counts = {species_key: 1 for species_key in active_keys}
    remaining = max(0, variant_count - len(active_keys))
    if remaining <= 0:
        return counts

    raw_extras = {
        species_key: normalized.get(species_key, 0.0) * remaining for species_key in active_keys
    }
    extras = {species_key: int(raw_extras[species_key]) for species_key in active_keys}
    leftover = remaining - sum(extras.values())
    for species_key in sorted(
        active_keys,
        key=lambda key: (raw_extras[key] - extras[key], normalized.get(key, 0.0)),
        reverse=True,
    )[:leftover]:
        extras[species_key] += 1

    for species_key, extra_count in extras.items():
        counts[species_key] += extra_count
    return counts
