from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf

from ..config import ScatterConfig


@dataclass
class BotaniqAssetPoolOutputConfig:
    pool_dir: str = "outputs/botaniq_tree_asset_pool"
    overwrite: bool = True
    run_name: str | None = None
    append_run_name: bool = True
    save_authoring: bool = True
    save_production: bool = True
    authoring_suffix: str = "_authoring"
    production_suffix: str = ""
    recipe_suffix: str = "_recipes"
    production_apply_modifiers: bool = True
    layout_for_inspection: bool = True
    inspection_spacing_m: float = 12.0


@dataclass
class BotaniqAssetPoolGenerationConfig:
    variants_per_species: int = 12
    species: list[str] = field(default_factory=lambda: ["Malus domestica"])
    season_mode: str = "summer"


@dataclass
class BotaniqAssetPoolLoggingConfig:
    file_path: str = "logs/botaniq_asset_pool.log"


@dataclass
class BotaniqAssetPoolConfig:
    seed: int = 42
    output: BotaniqAssetPoolOutputConfig = field(default_factory=BotaniqAssetPoolOutputConfig)
    generation: BotaniqAssetPoolGenerationConfig = field(
        default_factory=BotaniqAssetPoolGenerationConfig
    )
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    logging: BotaniqAssetPoolLoggingConfig = field(default_factory=BotaniqAssetPoolLoggingConfig)


def load_botaniq_asset_pool_config(cfg: DictConfig) -> BotaniqAssetPoolConfig:
    base_cfg = OmegaConf.structured(BotaniqAssetPoolConfig)
    merged = OmegaConf.merge(base_cfg, cfg)
    runtime_cfg = OmegaConf.to_object(merged)

    runtime_cfg.generation.variants_per_species = max(
        1,
        int(runtime_cfg.generation.variants_per_species),
    )
    normalized_species = []
    for species_name in runtime_cfg.generation.species:
        item = " ".join(str(species_name).strip().split())
        if item:
            normalized_species.append(item)
    runtime_cfg.generation.species = normalized_species

    season_mode = str(runtime_cfg.generation.season_mode).strip().lower()
    if season_mode not in {"summer", "autumn", "both"}:
        raise ValueError("generation.season_mode must be one of: summer, autumn, both.")
    runtime_cfg.generation.season_mode = season_mode

    runtime_cfg.output.inspection_spacing_m = max(
        0.0,
        float(runtime_cfg.output.inspection_spacing_m),
    )
    if not bool(runtime_cfg.output.save_authoring) and not bool(runtime_cfg.output.save_production):
        raise ValueError(
            "At least one of output.save_authoring or output.save_production must be true."
        )
    return runtime_cfg
