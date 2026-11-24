from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from synthforest.paths import project_root

from .random_streams import make_random_stream

DEFAULT_FOREST_TYPE = "eu_mixed"
SUPPORTED_TREE_SOURCE_MODES = (
    "botaniq_asset_pack",
    "botaniq_asset_pool",
)
BIOME_PROFILE_GLOBAL_SCATTER_KEYS = frozenset(
    {
        "tree_source_mode",
        "botaniq_pack_dir",
        "season_weights",
    }
)
SCATTER_DENSITY_BANDS = ("dense", "medium", "sparse")
# Versioned stream names keep per-scene choices deterministic while avoiding
# accidental correlation with other first-draw uses of the scene seed.
SCENE_SEASON_RANDOM_STREAM = "tree_scatter.scene_season.v1"


@dataclass(frozen=True)
class ScatterDensitySample:
    value: float
    band: str


def _default_scatter_density_band_weights() -> dict[str, float]:
    return {
        "dense": 0.80,
        "medium": 0.15,
        "sparse": 0.05,
    }


def _clamped_density_range(density_min: float, density_max: float) -> tuple[float, float]:
    low = max(0.0, min(1.0, float(density_min)))
    high = max(0.0, min(1.0, float(density_max)))
    if high < low:
        low, high = high, low
    return low, high


def _normalize_band_weights(weights: dict[str, float] | None) -> list[tuple[str, float]]:
    raw = weights if isinstance(weights, dict) else {}
    normalized: list[tuple[str, float]] = []
    for band in SCATTER_DENSITY_BANDS:
        weight = max(0.0, float(raw.get(band, 0.0)))
        if weight > 0.0:
            normalized.append((band, weight))
    if normalized:
        return normalized
    return [
        (band, weight)
        for band, weight in _default_scatter_density_band_weights().items()
        if weight > 0.0
    ]


def _density_band_range(density_min: float, density_max: float, band: str) -> tuple[float, float]:
    low, high = _clamped_density_range(density_min, density_max)
    span = high - low
    band_name = str(band).strip().lower()
    if band_name == "sparse":
        return low, low + span * 0.35
    if band_name == "medium":
        return low + span * 0.35, low + span * 0.65
    return low + span * 0.65, high


def _stepped_density_values(density_min: float, density_max: float, step: float) -> list[float]:
    low, high = _clamped_density_range(density_min, density_max)
    step = max(1e-6, float(step))
    min_steps = math.ceil((low - 1e-9) / step)
    max_steps = math.floor((high + 1e-9) / step)
    return [
        round(index * step, 10)
        for index in range(min_steps, max_steps + 1)
        if low - 1e-9 <= index * step <= high + 1e-9
    ]


def sample_scatter_density(
    density_min: float,
    density_max: float,
    seed: int,
    *,
    band_weights: dict[str, float] | None = None,
    step: float = 0.05,
) -> ScatterDensitySample:
    rng = random.Random(int(seed))
    bands = _normalize_band_weights(band_weights)
    band_names = [item[0] for item in bands]
    weights = [item[1] for item in bands]
    band = str(rng.choices(band_names, weights=weights, k=1)[0])

    band_min, band_max = _density_band_range(density_min, density_max, band)
    possible_values = _stepped_density_values(band_min, band_max, step)
    if not possible_values:
        possible_values = _stepped_density_values(density_min, density_max, step)
    if possible_values:
        return ScatterDensitySample(value=float(rng.choice(possible_values)), band=band)

    low, high = _clamped_density_range(density_min, density_max)
    return ScatterDensitySample(value=float((low + high) * 0.5), band=band)


def sample_scene_season(season_weights: dict[str, float], seed: int) -> str:
    normalized_weights: dict[str, float] = {}
    for season_name, weight in season_weights.items():
        normalized = str(season_name).strip().lower()
        if normalized not in {"summer", "autumn"}:
            continue
        normalized_weights[normalized] = max(0.0, float(weight))

    if not normalized_weights:
        return "summer"

    population = [name for name, weight in normalized_weights.items() if weight > 0.0]
    weights = [normalized_weights[name] for name in population]
    if not population:
        return "summer"

    rng = make_random_stream(seed, SCENE_SEASON_RANDOM_STREAM)
    return str(rng.choices(population=population, weights=weights, k=1)[0])


def compute_density_patch_controls(
    scatter_cfg: object, density_relative: float
) -> dict[str, float]:
    density_slider = max(0.0, min(1.0, float(density_relative)))
    fade_power = max(
        0.1,
        float(getattr(scatter_cfg, "density_variation_fade_power", 2.0)),
    )
    density_fade = density_slider**fade_power

    configured_variation = max(
        0.0,
        min(1.0, float(getattr(scatter_cfg, "density_variation_strength", 0.20))),
    )
    full_density_variation = max(
        0.0,
        min(
            configured_variation,
            float(getattr(scatter_cfg, "density_variation_min_at_full_density", 0.15)),
        ),
    )
    effective_variation = full_density_variation + (
        configured_variation - full_density_variation
    ) * (1.0 - density_fade)

    configured_threshold = max(
        0.0,
        min(0.999, float(getattr(scatter_cfg, "density_threshold", 0.4))),
    )
    threshold_fade_strength = max(
        0.0,
        min(1.0, float(getattr(scatter_cfg, "density_threshold_fade_strength", 0.75))),
    )
    effective_threshold = configured_threshold * (1.0 - threshold_fade_strength * density_fade)

    configured_exponent = max(
        0.1,
        float(getattr(scatter_cfg, "density_cluster_exponent", 1.0)),
    )
    full_density_exponent = max(
        0.1,
        float(getattr(scatter_cfg, "density_cluster_exponent_min_at_full_density", 1.0)),
    )
    effective_exponent = full_density_exponent + (configured_exponent - full_density_exponent) * (
        1.0 - density_fade
    )

    return {
        "density_slider": float(density_slider),
        "density_fade": float(density_fade),
        "density_threshold": float(effective_threshold),
        "density_variation_strength": float(effective_variation),
        "density_cluster_exponent": float(effective_exponent),
    }


def _biome_profiles_dir() -> Path:
    return Path(__file__).resolve().parent / "configs" / "biomes"


def _forest_profiles_dir() -> Path:
    # Keep the older helper name because the public config field is still
    # forest.type, but profiles now live in configs/biomes.
    return _biome_profiles_dir()


def _project_root() -> Path:
    return project_root()


def _internal_config_path(filename: str) -> str:
    return str((Path(__file__).resolve().parent / "configs" / filename).resolve())


def _resolve_project_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _available_forest_types() -> list[str]:
    profiles_dir = _forest_profiles_dir()
    if not profiles_dir.exists():
        return []
    return sorted(path.stem for path in profiles_dir.glob("*.yaml") if path.is_file())


def _load_forest_profile_cfg(forest_type: str) -> DictConfig:
    profile_path = _forest_profiles_dir() / f"{forest_type}.yaml"
    if not profile_path.is_file():
        available = ", ".join(_available_forest_types())
        raise ValueError(
            f"Unknown forest.type/biome profile '{forest_type}'. "
            f"Expected profile file at: {profile_path}. "
            f"Available biome profiles: {available}"
        )
    return OmegaConf.load(profile_path)


def _target_profile_catalog_path_from_cfg(cfg: DictConfig) -> Path:
    default = GenerationConfig().target_profile_catalog_path
    configured = OmegaConf.select(
        cfg,
        "generation.target_profile_catalog_path",
        default=default,
    )
    return _resolve_project_path(str(configured or default))


def _load_target_profile_catalog(cfg: DictConfig) -> DictConfig:
    catalog_path = _target_profile_catalog_path_from_cfg(cfg)
    if not catalog_path.is_file():
        raise ValueError(f"Cannot find generation.target_profile_catalog_path: {catalog_path}")
    loaded = OmegaConf.load(catalog_path)
    profiles = loaded.get("profiles", loaded)
    if not isinstance(profiles, DictConfig):
        raise ValueError(f"Target profile catalog must contain a mapping: {catalog_path}")
    return profiles


def get_available_target_profiles(cfg: DictConfig) -> list[str]:
    profiles = _load_target_profile_catalog(cfg)
    return sorted(str(name) for name in profiles.keys())


def load_target_profile_cfg(cfg: DictConfig, target_profile: str) -> DictConfig:
    profile_name = str(target_profile).strip()
    profiles = _load_target_profile_catalog(cfg)
    if profile_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles.keys()))
        raise ValueError(
            f"Unknown generation target profile '{profile_name}'. "
            f"Available target profiles: {available}"
        )
    return OmegaConf.create(OmegaConf.to_container(profiles[profile_name], resolve=False))


def load_target_profile_forest_type_weights(
    cfg: DictConfig,
    target_profile: str,
) -> dict[str, float]:
    profile_cfg = load_target_profile_cfg(cfg, target_profile)
    raw_weights = profile_cfg.get("forest_type_weights", {})
    if not isinstance(raw_weights, DictConfig):
        raise ValueError(
            f"Target profile '{target_profile}' must define forest_type_weights as a mapping."
        )
    return {
        str(forest_type).strip().lower(): float(weight)
        for forest_type, weight in raw_weights.items()
    }


def _to_plain_container(value: object) -> object:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _apply_runtime_overrides(target: object, overrides: object, path: str = "") -> None:
    plain_overrides = _to_plain_container(overrides)
    if not isinstance(plain_overrides, Mapping):
        return

    for key, value in plain_overrides.items():
        key_name = str(key)
        child_path = f"{path}.{key_name}" if path else key_name
        plain_value = _to_plain_container(value)

        if isinstance(target, dict):
            current_value = target.get(key_name)
            if isinstance(current_value, dict) and isinstance(plain_value, Mapping):
                _apply_runtime_overrides(current_value, plain_value, child_path)
            else:
                target[key_name] = plain_value
            continue

        if not hasattr(target, key_name):
            raise ValueError(f"Unknown target profile override key: {child_path}")

        current_value = getattr(target, key_name)
        if isinstance(plain_value, Mapping) and (
            isinstance(current_value, dict) or hasattr(current_value, "__dataclass_fields__")
        ):
            _apply_runtime_overrides(current_value, plain_value, child_path)
        else:
            setattr(target, key_name, plain_value)


def _strip_global_scatter_keys_from_profile(profile_cfg: DictConfig) -> DictConfig:
    cleaned = OmegaConf.create(OmegaConf.to_container(profile_cfg, resolve=False))
    scatter_cfg = cleaned.get("scatter")
    if isinstance(scatter_cfg, DictConfig):
        for key in BIOME_PROFILE_GLOBAL_SCATTER_KEYS:
            if key in scatter_cfg:
                del scatter_cfg[key]
    return cleaned


@dataclass
class OutputConfig:
    output_dir: str = "./tree_scatter_outputs"
    save_blend: bool = True
    run_name: str | None = None
    append_run_name: bool = True


@dataclass
class ForestConfig:
    type: str = DEFAULT_FOREST_TYPE


@dataclass
class TargetProfileConfig:
    name: str = ""
    quality_gates: Any = field(default_factory=dict)


@dataclass
class GenerationConfig:
    num_scenes: int = 1
    scene_id_start: int = 1
    forest_type_weights: dict[str, float] = field(default_factory=dict)
    target_profile_weights: dict[str, float] = field(default_factory=dict)
    target_profile_catalog_path: str = field(
        default_factory=lambda: _internal_config_path("target_profiles.yaml")
    )


@dataclass
class TerrainConfig:
    terrain_source_mode: str = "native"
    size: float = 100.0
    subdivisions: int = 200
    noise_scale: float = 0.05
    detail: float = 3.0
    roughness: float = 0.4
    elevation_min: float = -10.0
    elevation_max: float = 10.0
    color: list[float] = field(default_factory=lambda: [0.30, 0.25, 0.10, 1.0])
    use_texture: bool = True
    texture_dir: str = "data/terrain_textures/botaniq_particles"
    texture_file: str | None = None
    texture_gsd: float = 0.05
    biome_terrain_map_path: str = field(
        default_factory=lambda: _internal_config_path("biome_terrain_map.yaml")
    )
    context: TerrainContextConfig = field(default_factory=lambda: TerrainContextConfig())


@dataclass
class TerrainContextConfig:
    profile_catalog_path: str = field(
        default_factory=lambda: _internal_config_path("context_profiles.yaml")
    )
    context_mode_weights_by_density_band: Any = field(default_factory=dict)
    context_profile_weights_by_density_band: Any = field(default_factory=dict)
    # Optional path to the external city_rs asset root containing texture folders.
    # When omitted, the pipeline will try to resolve the default
    # "volumes/city_rs/assets" location relative to the repo.
    context_assets_dir: str | None = None
    context_reference_size: float = 100.0
    river_count_min: int = 1
    river_count_max: int = 1
    river_width_min: float = 4.0
    river_width_max: float = 8.5
    trail_width_min: float = 1.2
    trail_width_max: float = 2.8
    trail_count_min: int = 1
    trail_count_max: int = 2
    # How far above the terrain surface to place context geometry after shrinkwrap.
    # Keep small (millimeters to centimeters in scene units) to avoid a "floating" look.
    river_surface_offset: float = 0.005
    trail_surface_offset: float = 0.004
    exclusion_margin: float = 1.2
    # Extra no-tree buffer around context features in addition to exclusion_margin.
    trail_tree_buffer_extra: float = 0.8
    river_tree_buffer_extra: float = 1.3
    # Sparse context buildings (vegetation-dominant scenes with a few structures).
    buildings_enabled: bool = False
    building_count_min: int = 0
    building_count_max: int = 0
    building_width_min: float = 5.0
    building_width_max: float = 12.0
    building_depth_min: float = 5.0
    building_depth_max: float = 12.0
    building_height_min: float = 4
    building_height_max: float = 10
    building_base_offset: float = 0.02
    building_max_ground_relief: float = 1.0
    building_min_spacing: float = 4.0
    building_path_buffer: float = 8.0
    building_border_inset_ratio: float = 0.97
    building_tree_buffer_extra: float = 2.4
    waterline_elevation: float = -2.5
    rural_clearings_min: int = 1
    rural_clearings_max: int = 3
    rural_clearing_radius_min: float = 5.0
    rural_clearing_radius_max: float = 14.0


@dataclass
class ScatterConfig:
    tree_source_mode: str = "botaniq_asset_pack"
    asset_pool_dir: str = "data/tree_assets"
    asset_pool_species: list[str] = field(default_factory=list)
    botaniq_pack_dir: str = ""
    botaniq_include_title_prefixes: list[str] = field(default_factory=lambda: ["Tree "])
    botaniq_species_role_pools: dict[str, list[str]] = field(default_factory=dict)
    botaniq_role_count_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "dominant": [1, 3],
            "companion": [0, 5],
            "rare": [0, 6],
        }
    )
    botaniq_role_abundance_ranges: dict[str, list[float]] = field(
        default_factory=lambda: {
            "dominant": [0.55, 0.80],
            "companion": [0.15, 0.38],
            "rare": [0.02, 0.12],
        }
    )
    botaniq_role_dirichlet_alpha: dict[str, float] = field(
        default_factory=lambda: {
            "dominant": 3.0,
            "companion": 1.5,
            "rare": 0.8,
        }
    )
    botaniq_variant_scale_min: float = 0.92
    botaniq_variant_scale_max: float = 1.08
    botaniq_variant_height_scale_min: float = 0.82
    botaniq_variant_height_scale_max: float = 1.22
    botaniq_use_species_height_priors: bool = True
    botaniq_species_height_config_path: str = field(
        default_factory=lambda: _internal_config_path("botaniq_tree_heights.yaml")
    )
    botaniq_height_to_crown_scale_power: float = 0.85
    botaniq_height_scale_z_max: float = 3.0
    botaniq_height_scale_xy_max: float = 3.0
    botaniq_height_scale_anisotropy_max: float = 1.45
    botaniq_leaf_geometry_preservation_mode: str = "centroid_translate"
    botaniq_variant_quality_resample_attempts: int = 4
    botaniq_quality_max_scale_z: float = 3.0
    botaniq_quality_max_anisotropy: float = 1.45
    botaniq_quality_max_branch_displacement_m: float = 0.85
    botaniq_quality_max_leaf_removed_fraction: float = 0.12
    botaniq_quality_max_tilt_deg: float = 4.0
    botaniq_quality_allow_branch_swaps: bool = False
    botaniq_variant_tilt_probability: float = 0.35
    botaniq_variant_tilt_max_deg: float = 3.5
    botaniq_variant_shape_jitter_probability: float = 0.40
    botaniq_variant_crown_bend_max_fraction: float = 0.025
    botaniq_variant_crown_asymmetry_max: float = 0.07
    botaniq_variant_trunk_protected_fraction: float = 0.25
    botaniq_branch_variation_enabled: bool = True
    botaniq_branch_variation_backend: str = "geometry_nodes"
    botaniq_branch_variation_probability: float = 0.45
    botaniq_branch_geometry_nodes_apply: bool = True
    botaniq_branch_geometry_nodes_layout: str = "combined"
    botaniq_authoring_modifier_layout: str = "per_branch"
    botaniq_authoring_apply_global_deforms: bool = False
    botaniq_asset_pool_stage: str = "production"
    botaniq_branch_group_prefix: str = "bq_branch_group_"
    botaniq_branch_weight_threshold: float = 0.02
    botaniq_branch_min_vertices: int = 128
    botaniq_branch_radial_scale_min: float = 0.90
    botaniq_branch_radial_scale_max: float = 1.12
    botaniq_branch_tangential_scale_min: float = 0.92
    botaniq_branch_tangential_scale_max: float = 1.10
    botaniq_branch_z_scale_min: float = 0.97
    botaniq_branch_z_scale_max: float = 1.04
    botaniq_branch_rotation_max_deg: float = 10.0
    botaniq_branch_tilt_max_deg: float = 3.0
    botaniq_branch_lateral_jitter_max_fraction: float = 0.025
    botaniq_branch_vertical_jitter_max_fraction: float = 0.03
    botaniq_branch_vertical_jitter_max_m: float = 0.25
    botaniq_branch_influence_power: float = 1.05
    botaniq_branch_tip_influence_power: float = 0.95
    botaniq_branch_base_influence_floor: float = 0.08
    botaniq_branch_influence_scale_min: float = 0.85
    botaniq_branch_influence_scale_max: float = 1.45
    botaniq_branch_max_displacement_fraction: float = 0.16
    botaniq_branch_max_crown_displacement_fraction: float = 0.22
    botaniq_branch_max_displacement_m: float = 0.85
    botaniq_branch_prune_probability: float = 0.10
    botaniq_branch_prune_radial_scale_min: float = 0.72
    botaniq_branch_prune_radial_scale_max: float = 0.90
    botaniq_branch_prune_tangential_scale_min: float = 0.82
    botaniq_branch_prune_tangential_scale_max: float = 0.98
    botaniq_branch_prune_pullback_min_fraction: float = 0.25
    botaniq_branch_prune_pullback_max_fraction: float = 0.60
    botaniq_branch_prune_lateral_multiplier: float = 1.0
    botaniq_branch_outlier_probability: float = 0.05
    botaniq_branch_outlier_retract_probability: float = 0.58
    botaniq_branch_retract_scale_min: float = 0.42
    botaniq_branch_retract_scale_max: float = 0.78
    botaniq_branch_extend_scale_min: float = 1.02
    botaniq_branch_extend_scale_max: float = 1.18
    botaniq_branch_outlier_rotation_multiplier: float = 1.80
    botaniq_branch_outlier_lateral_multiplier: float = 1.70
    botaniq_branch_outlier_vertical_multiplier: float = 2.0
    botaniq_branch_swap_probability: float = 0.0
    botaniq_branch_swap_strength: float = 0.75
    botaniq_branch_swap_frame_strength: float = 1.0
    botaniq_branch_swap_translation_strength: float = 0.25
    botaniq_branch_swap_translation_max_fraction: float = 0.20
    botaniq_branch_swap_z_strength: float = 0.55
    botaniq_branch_swap_z_max_m: float = 0.75
    botaniq_branch_swap_scale_exchange_strength: float = 0.45
    botaniq_branch_swap_max_pairs: int = 3
    botaniq_leaf_density_variation_enabled: bool = True
    botaniq_leaf_density_variation_probability: float = 0.60
    botaniq_leaf_keep_ratio_min: float = 0.92
    botaniq_leaf_keep_ratio_max: float = 0.98
    botaniq_leaf_patch_count_min: int = 1
    botaniq_leaf_patch_count_max: int = 3
    botaniq_leaf_patch_radius_fraction_min: float = 0.12
    botaniq_leaf_patch_radius_fraction_max: float = 0.28
    botaniq_leaf_patch_drop_strength_min: float = 0.03
    botaniq_leaf_patch_drop_strength_max: float = 0.10
    botaniq_leaf_density_max_drop_probability: float = 0.25
    botaniq_leaf_density_material_keywords: list[str] = field(
        default_factory=lambda: ["leaf", "leaves", "foliage", "blossom", "flower"]
    )
    botaniq_leaf_density_vertex_group_threshold: float = 0.45
    botaniq_leaf_density_max_unit_faces: int = 8
    botaniq_color_variation_enabled: bool = True
    botaniq_brightness_jitter: float = 0.04
    botaniq_branch_hue_random_min: float = 0.0
    botaniq_branch_hue_random_max: float = 0.006
    botaniq_leaf_hue_random_min: float = 0.0
    botaniq_leaf_hue_random_max: float = 0.010
    botaniq_leaf_material_variation_enabled: bool = True
    botaniq_leaf_material_hue_shift_min: float = -0.035
    botaniq_leaf_material_hue_shift_max: float = 0.030
    botaniq_leaf_material_saturation_min: float = 0.88
    botaniq_leaf_material_saturation_max: float = 1.12
    botaniq_leaf_material_value_min: float = 0.86
    botaniq_leaf_material_value_max: float = 1.14
    botaniq_leaf_material_brightness_min: float = -0.025
    botaniq_leaf_material_brightness_max: float = 0.030
    botaniq_leaf_material_contrast_min: float = -0.040
    botaniq_leaf_material_contrast_max: float = 0.060
    botaniq_scatter_leaf_material_variation_enabled: bool = True
    botaniq_scatter_leaf_material_hue_shift_min: float = -0.008
    botaniq_scatter_leaf_material_hue_shift_max: float = 0.008
    botaniq_scatter_leaf_material_saturation_min: float = 0.97
    botaniq_scatter_leaf_material_saturation_max: float = 1.04
    botaniq_scatter_leaf_material_value_min: float = 0.96
    botaniq_scatter_leaf_material_value_max: float = 1.05
    botaniq_scatter_leaf_material_brightness_min: float = -0.008
    botaniq_scatter_leaf_material_brightness_max: float = 0.010
    botaniq_scatter_leaf_material_contrast_min: float = -0.015
    botaniq_scatter_leaf_material_contrast_max: float = 0.020
    season_weights: dict[str, float] = field(default_factory=lambda: {"summer": 1.0, "autumn": 0.0})
    density_noise_scale: float = 0.15
    density_noise_seed_offset: int = 1009
    density_threshold: float = 0.4
    scatter_density_min: float = 0.70
    scatter_density_max: float = 0.90
    scatter_density_band_weights: dict[str, float] = field(
        default_factory=_default_scatter_density_band_weights
    )
    scatter_density_step: float = 0.05
    crown_area_fill_factor: float = 0.72
    canopy_overlap_efficiency: float = 1.0
    canopy_closure_at_full_density: float = 0.95
    canopy_closure_gamma: float = 1.0
    poisson_packing_efficiency: float = 0.85
    precomputed_poisson_candidate_multiplier: float = 1.35
    precomputed_poisson_candidate_multiplier_max: float = 8.0
    precomputed_poisson_candidate_multiplier_growth: float = 1.75
    stem_spacing_crown_fraction_floor: float = 0.45
    density_variation_strength: float = 0.20
    density_cluster_exponent: float = 1.0
    density_variation_min_at_full_density: float = 0.15
    density_variation_fade_power: float = 2.0
    density_threshold_fade_strength: float = 0.75
    density_cluster_exponent_min_at_full_density: float = 1.0
    tree_scale_min: float = 1.0
    tree_scale_max: float = 1.0
    instance_height_scale_jitter: float = 0.035
    instance_crown_asymmetry_max: float = 0.035
    instance_lean_probability: float = 0.75
    instance_lean_max_deg: float = 3.0
    botaniq_instance_leaf_material_variation_enabled: bool = True
    botaniq_instance_leaf_hue_shift_min: float = -0.006
    botaniq_instance_leaf_hue_shift_max: float = 0.006
    botaniq_instance_leaf_brightness_min: float = -0.006
    botaniq_instance_leaf_brightness_max: float = 0.008
    scatter_validation_grid_cells: int = 8
    scatter_validation_canopy_grid_resolution: int = 128
    tree_variants: int = 6
    instance_id_max_value: int = 65535


@dataclass
class RenderConfig:
    resolution_x: int = 1024
    resolution_y: int = 1024
    gsd_at_nadir: float = 0.1
    samples: int = 64
    no_render: bool = False
    height_threshold: float = 1.5
    device: str = "AUTO"
    gpu_backend: str = "AUTO"
    use_persistent_data: bool = False


@dataclass
class LoggingConfig:
    file_path: str = "logs/tree_scatter.log"


@dataclass
class OutputPassesConfig:
    rgb: bool = True
    instance_id: bool = True
    variant_id: bool = True
    density_map: bool = True
    height_map: bool = True
    height_map_vis: bool = True
    smoothed_instance_id: bool = True
    semantic_id: bool = True
    semantic_labels: bool = True
    coco_annotations: bool = True
    scene_summary: bool = True


@dataclass
class MaskPostprocessConfig:
    min_visible_area_m2: float = 1.0
    min_final_area_m2: float = 1.5
    min_visible_brightness: float = 0.08
    visible_brightness_percentile: float = 90.0
    min_component_area_m2: float = 0.25
    gap_fill_radius_m: float = 1.0
    sliver_area_m2: float = 10.0
    sliver_min_bbox_fill_ratio: float = 0.22
    sliver_min_bbox_axis_ratio: float = 0.30
    sliver_min_convex_fill_ratio: float = 0.42
    sliver_min_largest_component_fraction: float = 0.70
    boundary_regularization_min_area_m2: float = 8.0
    boundary_smooth_radius_m: float = 0.3
    boundary_core_erosion_m: float = 0.50
    boundary_component_merge_radius_m: float = 2.0
    boundary_component_min_area_m2: float = 0.25
    boundary_component_keep_fraction: float = 0.02
    boundary_component_merge_max_area_growth_fraction: float = 0.75
    boundary_max_area_growth_fraction: float = 0.35
    nested_island_parent_min_area_m2: float = 8.0
    nested_island_max_hole_area_fraction: float = 0.35
    nested_island_absorb_max_area_fraction: float = 0.35
    nested_island_absorb_max_area_m2: float = 30.0
    final_topology_min_area_m2: float = 5.0
    ordered_height_percentile: float = 90.0
    ordered_area_priority_weight: float = 0.12
    ordered_absorb_height_margin_m: float = 2.0
    ordered_absorb_overlap_fraction: float = 0.60
    ordered_absorb_max_area_m2: float = 30.0
    ordered_min_retained_visible_fraction: float = 0.25
    ordered_component_keep_fraction: float = 0.15
    ordered_outer_envelope_min_area_m2: float = 3.0
    ordered_outer_envelope_radius_m: float = 0.75
    ordered_outer_envelope_max_growth_fraction: float = 0.5
    ordered_reclaim_height_margin_m: float = 0.5
    ordered_rescue_min_visible_fraction: float = 0.02


@dataclass
class TreeScatterConfig:
    seed: int = 42
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    target_profile: TargetProfileConfig = field(default_factory=TargetProfileConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output_passes: OutputPassesConfig = field(default_factory=OutputPassesConfig)
    mask_postprocess: MaskPostprocessConfig = field(default_factory=MaskPostprocessConfig)


def get_available_forest_types() -> list[str]:
    return _available_forest_types()


def get_available_biome_types() -> list[str]:
    return _available_forest_types()


def _validate_tree_source_mode(runtime_cfg: TreeScatterConfig) -> None:
    mode = str(runtime_cfg.scatter.tree_source_mode).strip().lower()
    if mode not in SUPPORTED_TREE_SOURCE_MODES:
        supported = ", ".join(SUPPORTED_TREE_SOURCE_MODES)
        raise ValueError(
            f"Unknown scatter.tree_source_mode '{runtime_cfg.scatter.tree_source_mode}'. "
            f"Supported values: {supported}"
        )
    runtime_cfg.scatter.tree_source_mode = mode


def load_tree_scatter_base_config(cfg: DictConfig) -> TreeScatterConfig:
    base_cfg = OmegaConf.structured(TreeScatterConfig)
    merged = OmegaConf.merge(base_cfg, cfg)
    runtime_cfg = OmegaConf.to_object(merged)

    runtime_cfg.generation.num_scenes = max(1, int(runtime_cfg.generation.num_scenes))
    runtime_cfg.generation.scene_id_start = max(1, int(runtime_cfg.generation.scene_id_start))
    _validate_tree_source_mode(runtime_cfg)
    return runtime_cfg


def load_tree_scatter_config_for_forest(
    cfg: DictConfig,
    forest_type: str,
) -> TreeScatterConfig:
    base_cfg = OmegaConf.structured(TreeScatterConfig)
    normalized_forest_type = str(forest_type).strip().lower()

    profile_cfg = _strip_global_scatter_keys_from_profile(
        _load_forest_profile_cfg(normalized_forest_type)
    )
    merged = OmegaConf.merge(
        base_cfg,
        profile_cfg,
        cfg,
        {"forest": {"type": normalized_forest_type}},
    )
    runtime_cfg = OmegaConf.to_object(merged)

    runtime_cfg.generation.num_scenes = max(1, int(runtime_cfg.generation.num_scenes))
    runtime_cfg.generation.scene_id_start = max(1, int(runtime_cfg.generation.scene_id_start))
    _validate_tree_source_mode(runtime_cfg)
    return runtime_cfg


def load_tree_scatter_config_for_forest_and_target_profile(
    cfg: DictConfig,
    forest_type: str,
    target_profile: str | None,
) -> TreeScatterConfig:
    profile_name = str(target_profile or "").strip()
    if not profile_name:
        return load_tree_scatter_config_for_forest(cfg, forest_type)

    base_cfg = OmegaConf.structured(TreeScatterConfig)
    normalized_forest_type = str(forest_type).strip().lower()

    biome_profile_cfg = _strip_global_scatter_keys_from_profile(
        _load_forest_profile_cfg(normalized_forest_type)
    )
    target_profile_cfg = load_target_profile_cfg(cfg, profile_name)
    target_overrides = target_profile_cfg.get("overrides", {})
    quality_gates = _to_plain_container(target_profile_cfg.get("quality_gates", {}))
    merged = OmegaConf.merge(
        base_cfg,
        biome_profile_cfg,
        cfg,
        {
            "forest": {"type": normalized_forest_type},
            "target_profile": {
                "name": profile_name,
            },
        },
    )
    runtime_cfg = OmegaConf.to_object(merged)
    runtime_cfg.target_profile.quality_gates = quality_gates
    _apply_runtime_overrides(runtime_cfg, target_overrides)

    runtime_cfg.generation.num_scenes = max(1, int(runtime_cfg.generation.num_scenes))
    runtime_cfg.generation.scene_id_start = max(1, int(runtime_cfg.generation.scene_id_start))
    _validate_tree_source_mode(runtime_cfg)
    return runtime_cfg


def load_tree_scatter_config(cfg: DictConfig) -> TreeScatterConfig:
    base_cfg = OmegaConf.structured(TreeScatterConfig)
    requested_cfg = OmegaConf.merge(base_cfg, cfg)

    raw_forest_type = OmegaConf.select(requested_cfg, "forest.type", default=DEFAULT_FOREST_TYPE)
    forest_type = str(raw_forest_type).strip().lower()
    return load_tree_scatter_config_for_forest(cfg, forest_type)
