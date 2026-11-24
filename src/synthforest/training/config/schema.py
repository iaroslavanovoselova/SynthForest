from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import yaml

from synthforest.config_io import load_yaml
from synthforest.training.config.naming import (
    format_domain_adaptation_tag,
    format_float_tag,
    format_range,
    format_stage_tag,
    slugify,
)
from synthforest.training.layout import validate_checkpoint_name


def _data_path(relative: str) -> str:
    root = os.environ.get("SYNTHFOREST_DATA_ROOT")
    return str(Path(root).expanduser() / relative) if root else ""


def default_real_validation_root() -> str:
    """Resolve the default crown-benchmark root at config construction time."""

    return _data_path("canopyrs")


def default_real_height_validation_root() -> str:
    """Resolve the default height-benchmark root at config construction time."""

    return _data_path("NEON_CHM_1m_real_eval")


DEFAULT_REAL_VALIDATION_ROOT = default_real_validation_root()
DEFAULT_REAL_VALIDATION_DATASETS = [
    "QuebecTrees",
    "Detectree2",
    "NeonTreeEvaluation",
    "SelvaBox",
    "OAM-TCD",
]
DEFAULT_REAL_HEIGHT_VALIDATION_ROOT = default_real_height_validation_root()
DEFAULT_REAL_HEIGHT_VALIDATION_DATASETS = ["NEON_CHM_1m_real_eval"]


def _default_da_root() -> str:
    return _data_path("domain_adaptation")


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "synthforest"
    entity: Optional[str] = None
    tags: list[str] = field(default_factory=lambda: ["dinov3", "detector", "synthforest"])
    mode: Optional[str] = None
    log_artifacts: bool = True
    required: bool = True


@dataclass
class BranchInitializationConfig:
    checkpoint_path: str = ""
    completed_iterations: int = 0
    parent_run_id: Optional[str] = None
    parent_run_name: Optional[str] = None
    architecture_fingerprint: Optional[str] = None


@dataclass
class CheckpointInitializationConfig:
    checkpoint_path: str = ""


@dataclass
class TrainSamplingConfig:
    fraction: float = 1.0
    seed: Optional[int] = None
    min_samples: int = 1


@dataclass
class SweepRuntimeConfig:
    phase: str = ""
    objective_metric: Optional[str] = None
    objective_iterations: list[int] = field(default_factory=list)
    parent_control_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PostTrainingBenchmarkConfig:
    enabled: bool = False
    config_path: Optional[str] = None
    checkpoint_name: str = "last.pth"
    fail_on_error: bool = False


@dataclass
class DomainAdaptationConfig:
    enabled: bool = True
    reference_root_path: Optional[str] = field(default_factory=_default_da_root)
    reference_root_paths: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=lambda: ["FDA", "HM", "PDA"])
    prob: float = 0.8
    max_reference_images: int = 1200
    fda_beta_limit: list[float] = field(default_factory=lambda: [0.01, 0.05])
    hm_blend_ratio: list[float] = field(default_factory=lambda: [0.8, 1.0])
    pda_blend_ratio: list[float] = field(default_factory=lambda: [0.3, 0.5])
    pda_type: str = "standard"


@dataclass
class AugmentationConfig:
    image_mean: list[float] = field(default_factory=lambda: [0.430, 0.411, 0.296])
    image_std: list[float] = field(default_factory=lambda: [0.213, 0.156, 0.143])
    resize_min_sizes_train: list[int] = field(
        default_factory=lambda: [512, 608, 704, 800, 896, 992]
    )
    resize_min_size_eval: int = 992
    max_size: int = 992
    crop_prob: float = 0.5
    crop_size_range: list[int] = field(default_factory=lambda: [384, 992])
    crop_min_intersection_ratio: float = 0.5
    flip_horizontal: bool = True
    flip_vertical: bool = True
    rotation_degrees: float = 30.0
    rotation_prob: float = 0.5
    brightness: float = 0.2
    brightness_prob: float = 0.5
    contrast: float = 0.2
    contrast_prob: float = 0.5
    saturation: float = 0.2
    saturation_prob: float = 0.5
    hue: float = 0.05
    hue_prob: float = 0.3
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma: list[float] = field(default_factory=lambda: [0.1, 1.0])
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: list[float] = field(default_factory=lambda: [0.0, 0.02])
    jpeg_prob: float = 0.2
    jpeg_quality: list[int] = field(default_factory=lambda: [70, 100])
    downsample_upsample_prob: float = 0.2
    downsample_scale: list[float] = field(default_factory=lambda: [0.75, 1.0])


@dataclass
class ValidMaskConfig:
    enabled: bool = True
    source: str = "auto"
    black_pixel_tolerance: int = 0
    da_full_tile_min_valid_fraction: float = 0.95
    crop_min_valid_fraction: float = 0.25
    target_min_valid_fraction: float = 0.5
    prediction_min_valid_fraction: Optional[float] = 0.6


@dataclass
class HeightResolutionDegradationConfig:
    enabled: bool = False
    prob: float = 1.0
    base_gsd_m: float = 0.5
    target_gsd_m_values: list[float] = field(default_factory=lambda: [0.5, 1.0])
    height_reducer: str = "p90_positive"
    min_canopy_fraction: float = 0.10
    blur_sigma: list[float] = field(default_factory=lambda: [0.0, 1.0])
    downsample_interpolation: str = "box"
    upsample_interpolation: str = "bilinear"


@dataclass
class LoRAConfig:
    num_blocks: int = 4
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    gradient_checkpointing: bool = False


@dataclass
class StageOptimizerConfig:
    name: str = "AdamW"
    detector_lr: float = 1e-4
    detector_weight_decay: float = 1e-4
    lora_lr: Optional[float] = None
    lora_weight_decay: float = 0.0
    warmup_iterations: int = 500
    min_lr_factor: float = 0.01


@dataclass
class TrainingStageConfig:
    name: str = "detector_only"
    max_iterations: int = 12000
    eval_interval: int = 1000
    checkpoint_interval: int = 1000
    train_log_interval: int = 10
    train_lora: bool = False
    semi_supervised: bool = False
    optimizer: StageOptimizerConfig = field(default_factory=StageOptimizerConfig)


@dataclass
class HeightStageOptimizerConfig:
    name: str = "AdamW"
    decoder_lr: float = 1e-4
    decoder_weight_decay: float = 1e-4
    lora_lr: Optional[float] = None
    lora_weight_decay: float = 0.0
    warmup_iterations: int = 500
    min_lr_factor: float = 0.01


@dataclass
class HeightTrainingStageConfig:
    name: str = "decoder_only"
    max_iterations: int = 12000
    eval_interval: int = 1000
    checkpoint_interval: int = 1000
    train_log_interval: int = 10
    train_lora: bool = False
    optimizer: HeightStageOptimizerConfig = field(default_factory=HeightStageOptimizerConfig)


@dataclass
class SyntheticValidationConfig:
    data_root_path: Optional[str] = None
    dataset_names: Optional[list[str]] = None
    fold: str = "valid"


@dataclass
class RealValidationConfig:
    enabled: bool = True
    data_root_path: str = field(default_factory=default_real_validation_root)
    dataset_names: list[str] = field(default_factory=lambda: list(DEFAULT_REAL_VALIDATION_DATASETS))
    fold: str = "valid"
    drop_edge_tiles: bool = False
    edge_tile_black_pixel_threshold: float = 0.1
    edge_tile_black_pixel_tolerance: int = 0
    max_products_per_dataset: Optional[int] = None
    max_tiles_per_product: Optional[int] = None


@dataclass
class UnlabeledRealTrainingConfig:
    data_root_path: str = field(default_factory=default_real_validation_root)
    dataset_names: list[str] = field(default_factory=lambda: list(DEFAULT_REAL_VALIDATION_DATASETS))
    fold: str = "valid"
    drop_edge_tiles: bool = False
    edge_tile_black_pixel_threshold: float = 0.1
    edge_tile_black_pixel_tolerance: int = 0
    max_products_per_dataset: Optional[int] = None
    max_tiles_per_product: Optional[int] = None


@dataclass
class PseudoLabelConfig:
    score_threshold: float = 0.7
    min_box_size: float = 2.0
    min_box_area: float = 4.0
    max_labels_per_image: int = 200
    valid_mask_min_fraction: Optional[float] = 0.6
    nms_threshold: Optional[float] = None
    include_empty_images: bool = False


@dataclass
class SemiSupervisedStrongAugmentationConfig:
    brightness: float = 0.2
    brightness_prob: float = 0.5
    contrast: float = 0.2
    contrast_prob: float = 0.5
    saturation: float = 0.2
    saturation_prob: float = 0.5
    hue: float = 0.05
    hue_prob: float = 0.3
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma: list[float] = field(default_factory=lambda: [0.1, 1.0])
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: list[float] = field(default_factory=lambda: [0.0, 0.02])
    jpeg_prob: float = 0.2
    jpeg_quality: list[int] = field(default_factory=lambda: [70, 100])
    downsample_upsample_prob: float = 0.2
    downsample_scale: list[float] = field(default_factory=lambda: [0.75, 1.0])


@dataclass
class SemiSupervisedConfig:
    enabled: bool = False
    ema_decay: float = 0.999
    unsupervised_loss_start_weight: float = 0.0
    unsupervised_loss_weight: float = 1.0
    unsupervised_loss_ramp_fraction: float = 0.0
    unlabeled_real: UnlabeledRealTrainingConfig = field(default_factory=UnlabeledRealTrainingConfig)
    pseudo_labels: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    strong_augmentation: SemiSupervisedStrongAugmentationConfig = field(
        default_factory=SemiSupervisedStrongAugmentationConfig
    )


@dataclass
class ValidationConfig:
    selection_metric: str = "bbox_dense_map"
    map_max_detection_thresholds: list[int] = field(default_factory=lambda: [1, 10, 100])
    dense_eval_max_detections: Optional[int] = 500
    tasks: list[str] = field(default_factory=lambda: ["bbox"])
    batch_size: Optional[int] = None
    synthetic: SyntheticValidationConfig = field(default_factory=SyntheticValidationConfig)
    real: RealValidationConfig = field(default_factory=RealValidationConfig)


@dataclass
class HeightSyntheticValidationConfig:
    data_root_path: Optional[str] = None
    dataset_names: Optional[list[str]] = None
    fold: str = "valid"


@dataclass
class HeightRealValidationConfig:
    enabled: bool = True
    data_root_path: str = field(default_factory=default_real_height_validation_root)
    dataset_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_REAL_HEIGHT_VALIDATION_DATASETS)
    )
    fold: str = "valid"
    max_products_per_dataset: Optional[int] = None
    max_tiles_per_product: Optional[int] = None
    metric_resolution: "HeightMetricResolutionConfig" = field(
        default_factory=lambda: HeightMetricResolutionConfig()
    )
    sample_scenes_per_eval: Optional[int] = None
    sample_seed: Optional[int] = None


@dataclass
class HeightValidationConfig:
    selection_metric: str = "mae"
    batch_size: Optional[int] = None
    canopy_thresholds_m: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    synthetic: HeightSyntheticValidationConfig = field(
        default_factory=HeightSyntheticValidationConfig
    )
    real: HeightRealValidationConfig = field(default_factory=HeightRealValidationConfig)


@dataclass
class HeightMetricResolutionConfig:
    enabled: bool = False
    rgb_gsd_m: Optional[float] = None
    height_gsd_m: Optional[float] = None
    downsample_factor: Optional[float] = None
    min_valid_fraction: float = 0.5
    downsample_interpolation: str = "area"
    upsample_interpolation: str = "nearest"

    def resolved_downsample_factor(self) -> float:
        if not self.enabled:
            return 1.0
        if self.downsample_factor is not None:
            factor = float(self.downsample_factor)
            if factor < 1.0:
                raise ValueError("metric_resolution.downsample_factor must be >= 1.")
            return factor
        if self.rgb_gsd_m is None or self.height_gsd_m is None:
            raise ValueError(
                "metric_resolution requires downsample_factor or both "
                "rgb_gsd_m and height_gsd_m when enabled."
            )
        factor = float(self.height_gsd_m) / float(self.rgb_gsd_m)
        if factor < 1.0:
            raise ValueError(
                "metric_resolution height/RGB GSD factor must be >= 1; "
                f"got rgb_gsd_m={self.rgb_gsd_m} and "
                f"height_gsd_m={self.height_gsd_m}."
            )
        return factor


@dataclass
class HeightBenchmarkConfig:
    model_type: str = "dpt_height"
    train_config_path: Optional[str] = None
    run_folder: Optional[str] = None
    checkpoint_name: str = "best.pth"
    checkpoint_path: Optional[str] = None
    data_root_path: str = field(default_factory=default_real_height_validation_root)
    output_folder: str = "outputs/heights/benchmarks"
    benchmark_name: Optional[str] = None

    dataset_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_REAL_HEIGHT_VALIDATION_DATASETS)
    )
    fold_name: str = "valid"
    batch_size: int = 1
    num_workers: int = 4
    device: Optional[str] = None
    seed: int = 42
    max_products: Optional[int] = None
    max_tiles: Optional[int] = None
    save_predictions: bool = False
    metric_resolution: HeightMetricResolutionConfig = field(
        default_factory=HeightMetricResolutionConfig
    )
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            project="synthforest_height_benchmark",
            tags=[],
        )
    )

    @classmethod
    def from_yaml(cls, path: str | Path, validate: bool = True) -> "HeightBenchmarkConfig":
        _, raw = load_yaml(path)
        cfg = _coerce_dataclass(cls, raw)
        if validate:
            cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def resolved_checkpoint_path(self) -> Path:
        if self.run_folder:
            return (
                Path(self.run_folder)
                / "checkpoints"
                / validate_checkpoint_name(self.checkpoint_name)
            )
        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        raise ValueError("Either run_folder or checkpoint_path is required.")

    def resolved_benchmark_name(self) -> str:
        if self.benchmark_name:
            return self.benchmark_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_tag = "-".join(self.dataset_names)
        return slugify(f"height_benchmark_{self.fold_name}_{dataset_tag}_{stamp}")

    def validate(self) -> None:
        if self.model_type != "dpt_height":
            raise ValueError("height benchmark model_type must be dpt_height.")
        validate_checkpoint_name(self.checkpoint_name)
        if self.run_folder and self.checkpoint_path:
            raise ValueError("run_folder and checkpoint_path are mutually exclusive.")
        if not self.run_folder and not self.checkpoint_path:
            raise ValueError("Either run_folder or checkpoint_path is required.")
        if not self.dataset_names:
            raise ValueError("dataset_names must contain at least one dataset.")
        if self.fold_name not in {"train", "valid", "test"}:
            raise ValueError("fold_name must be train, valid, or test.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.max_products is not None and self.max_products <= 0:
            raise ValueError("max_products must be positive when set.")
        if self.max_tiles is not None and self.max_tiles <= 0:
            raise ValueError("max_tiles must be positive when set.")
        _validate_height_metric_resolution_config(self.metric_resolution)
        if self.wandb.enabled and not self.wandb.project.strip():
            raise ValueError("wandb.project must not be empty when wandb is enabled.")


@dataclass
class HeightRasterPairConfig:
    name: Optional[str] = None
    dataset_name: Optional[str] = None
    rgb_path: str = ""
    height_path: Optional[str] = None


@dataclass
class HeightRasterBenchmarkConfig:
    model_type: str = "dpt_height"
    train_config_path: Optional[str] = None
    run_folder: Optional[str] = None
    checkpoint_name: str = "last.pth"
    checkpoint_path: Optional[str] = None
    output_folder: str = "outputs/heights/raster_benchmarks"
    benchmark_name: Optional[str] = None

    rasters: list[HeightRasterPairConfig] = field(default_factory=list)
    rgb_root_path: Optional[str] = None
    height_root_path: Optional[str] = None
    rgb_glob: str = "**/*"
    height_glob: str = "**/*"
    rgb_name_suffix: str = "_RGB"
    height_name_suffix: str = "_CHM"
    image_suffixes: list[str] = field(
        default_factory=lambda: [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    )
    height_suffixes: list[str] = field(default_factory=lambda: [".tif", ".tiff"])

    tile_size: int = 512
    stride: int = 256
    batch_size: int = 1
    device: Optional[str] = None
    seed: int = 42
    max_rasters: Optional[int] = None
    save_predictions: bool = True
    save_weight_maps: bool = False
    metric_resolution: HeightMetricResolutionConfig = field(
        default_factory=HeightMetricResolutionConfig
    )
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            project="synthforest_height_benchmark",
            tags=[],
        )
    )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        validate: bool = True,
    ) -> "HeightRasterBenchmarkConfig":
        _, raw = load_yaml(path)
        cfg = _coerce_dataclass(cls, raw)
        if validate:
            cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def resolved_checkpoint_path(self) -> Path:
        if self.run_folder:
            return (
                Path(self.run_folder)
                / "checkpoints"
                / validate_checkpoint_name(self.checkpoint_name)
            )
        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        raise ValueError("Either run_folder or checkpoint_path is required.")

    def resolved_benchmark_name(self) -> str:
        if self.benchmark_name:
            return self.benchmark_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = "explicit"
        if self.rgb_root_path:
            source = Path(self.rgb_root_path).name
        return slugify(f"height_raster_benchmark_{source}_{stamp}")

    def validate(self) -> None:
        if self.model_type != "dpt_height":
            raise ValueError("height raster benchmark model_type must be dpt_height.")
        validate_checkpoint_name(self.checkpoint_name)
        if self.run_folder and self.checkpoint_path:
            raise ValueError("run_folder and checkpoint_path are mutually exclusive.")
        if not self.run_folder and not self.checkpoint_path:
            raise ValueError("Either run_folder or checkpoint_path is required.")
        if not self.rasters and not self.rgb_root_path:
            raise ValueError("Either rasters or rgb_root_path is required.")
        for raster in self.rasters:
            if not str(raster.rgb_path).strip():
                raise ValueError("Every raster entry must set rgb_path.")
        self.rgb_name_suffix = "" if self.rgb_name_suffix is None else str(self.rgb_name_suffix)
        self.height_name_suffix = (
            "" if self.height_name_suffix is None else str(self.height_name_suffix)
        )
        if not self.rgb_name_suffix:
            raise ValueError("rgb_name_suffix must not be empty.")
        if not self.height_name_suffix:
            raise ValueError("height_name_suffix must not be empty.")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_rasters is not None and self.max_rasters <= 0:
            raise ValueError("max_rasters must be positive when set.")
        if not self.image_suffixes:
            raise ValueError("image_suffixes must not be empty.")
        if any(not str(value).startswith(".") for value in self.image_suffixes):
            raise ValueError("image_suffixes values must include the leading dot.")
        if not self.height_suffixes:
            raise ValueError("height_suffixes must not be empty.")
        if any(not str(value).startswith(".") for value in self.height_suffixes):
            raise ValueError("height_suffixes values must include the leading dot.")
        _validate_height_metric_resolution_config(self.metric_resolution)
        if self.wandb.enabled and not self.wandb.project.strip():
            raise ValueError("wandb.project must not be empty when wandb is enabled.")


@dataclass
class DetectorHeightFusionConfig:
    """Shared-FPN height fusion for Faster R-CNN and Mask R-CNN detectors."""

    enabled: bool = False
    height_maps_subdir: str = "height_maps"
    generate_missing: bool = True
    height_checkpoint_path: Optional[str] = None
    height_train_config_path: Optional[str] = None
    source_gsd_m: float = 0.05
    target_gsd_m: float = 0.20
    levels: list[str] = field(default_factory=lambda: ["p8"])
    clip_min_m: float = 0.0
    clip_max_m: float = 60.0
    mean_m: float = 30.0
    std_m: float = 30.0
    encoder_hidden_channels: int = 64
    fusion_init_scale: float = 0.1
    use_uncertainty: bool = False
    uncertainty_log_variance_min: float = -4.0
    uncertainty_log_variance_max: float = 8.0
    uncertainty_reliability_tau: float = 0.5
    uncertainty_reliability_min: float = 0.1
    precompute_match_checkpoint: bool = True


@dataclass
class DetectorTrainingConfig:
    data_root_path: str = "outputs/tiles"
    train_dataset_names: list[str] = field(default_factory=list)
    test_dataset_names: list[str] = field(default_factory=list)
    train_output_path: str = "outputs/detectors/training"
    run_name: Optional[str] = None

    dinov3_model_dir: Optional[str] = None
    dinov3_model: str = "dinov3_vitl16"
    dinov3_pretraining: str = "sat493m"
    dinov3_embed_dim: int = 1024
    dinov3_patch_size: int = 16
    dinov3_use_fake_backbone: bool = False
    backbone_out_channels: int = 256
    model_type: str = ""
    mask_probability_threshold: float = 0.5

    lora: Optional[LoRAConfig] = None
    stages: list[TrainingStageConfig] = field(default_factory=lambda: [TrainingStageConfig()])
    initialization: Optional[BranchInitializationConfig] = None
    checkpoint_initialization: Optional[CheckpointInitializationConfig] = None
    train_sampling: TrainSamplingConfig = field(default_factory=TrainSamplingConfig)
    branch_checkpoint_iterations: list[int] = field(default_factory=list)
    sweep: Optional[SweepRuntimeConfig] = None
    post_training_benchmark: PostTrainingBenchmarkConfig = field(
        default_factory=PostTrainingBenchmarkConfig
    )

    num_classes: int = 1
    batch_size: int = 1
    grad_accumulation_steps: int = 4
    dataloader_num_workers: int = 4
    seed: int = 42
    use_amp: bool = True
    include_empty_images: bool = True
    debug_overfit_batches: int = 0
    debug_eval_on_train: bool = False

    pyramid_strides: list[int] = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list[Any] = field(default_factory=lambda: [[16, 32], [64, 128], [256, 512]])
    aspect_ratios: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_predictions_per_image: int = 500

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    valid_mask: ValidMaskConfig = field(default_factory=ValidMaskConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    domain_adaptation: DomainAdaptationConfig = field(default_factory=DomainAdaptationConfig)
    height_fusion: DetectorHeightFusionConfig = field(default_factory=DetectorHeightFusionConfig)
    semi_supervised: SemiSupervisedConfig = field(default_factory=SemiSupervisedConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectorTrainingConfig":
        _, raw = load_yaml(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
    ) -> "DetectorTrainingConfig":
        cfg = _coerce_dataclass(cls, raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def total_iterations(self) -> int:
        return sum(int(stage.max_iterations) for stage in self.stages)

    @property
    def final_stage(self) -> TrainingStageConfig:
        return self.stages[-1]

    def resolved_synthetic_validation_root(self) -> str:
        return self.validation.synthetic.data_root_path or self.data_root_path

    def resolved_synthetic_validation_names(self) -> list[str]:
        names = self.validation.synthetic.dataset_names
        return list(self.train_dataset_names if names is None else names)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        resize_tag = format_range(self.augmentation.resize_min_sizes_train)
        crop_tag = format_range(self.augmentation.crop_size_range)
        model_tag = self.dinov3_model.removeprefix("dinov3_")
        stage_tag = "_".join(format_stage_tag(stage) for stage in self.stages)
        adaptation_tag = "frozen"
        if self.lora is not None:
            adaptation_tag = f"lora{self.lora.num_blocks}b_r{self.lora.rank}"
        fine_tune_tags = []
        if self.checkpoint_initialization is not None:
            fraction_tag = format_float_tag(self.train_sampling.fraction)
            fine_tune_tags.append(f"ft_frac{fraction_tag}")

        parts = [
            "dinov3mrcnn" if self.model_type == "maskrcnn" else "dinov3frcnn",
            model_tag,
            *fine_tune_tags,
            # self.dinov3_pretraining,
            stage_tag,
            adaptation_tag,
            self.final_stage.optimizer.name.lower(),
            f"b{self.batch_size}",
            f"acc{self.grad_accumulation_steps}",
            # f"eff{effective_batch_size}",
            f"img{resize_tag}",
            f"crop{crop_tag}",
            format_domain_adaptation_tag(self.domain_adaptation),
            stamp,
        ]
        if slurm_job_id:
            parts.append(f"job{slurm_job_id}")
        return slugify("_".join(parts))

    def validate(self) -> None:
        if self.model_type not in {"fasterrcnn", "maskrcnn"}:
            raise ValueError("model_type must be one of: fasterrcnn, maskrcnn.")
        if not 0.0 <= float(self.mask_probability_threshold) <= 1.0:
            raise ValueError("mask_probability_threshold must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        if self.debug_overfit_batches < 0:
            raise ValueError("debug_overfit_batches must be non-negative.")
        if self.num_classes <= 0:
            raise ValueError(
                "num_classes is the number of foreground classes and must be positive."
            )
        if not self.train_dataset_names:
            raise ValueError("train_dataset_names must contain at least one dataset.")
        if not self.dinov3_use_fake_backbone and not self.dinov3_model_dir:
            raise ValueError(
                "dinov3_model_dir is required unless dinov3_use_fake_backbone is true."
            )
        if not self.stages:
            raise ValueError("stages must contain at least one stage.")

        stage_names = [slugify(stage.name) for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("stage names must be unique.")

        lora_stage_flags = [bool(stage.train_lora) for stage in self.stages]
        if any(lora_stage_flags) and self.lora is None:
            raise ValueError("A stage with train_lora: true requires a lora configuration.")
        if self.lora is not None and not any(lora_stage_flags):
            raise ValueError(
                "A lora configuration requires at least one stage with train_lora: true."
            )
        if any(
            earlier and not later for earlier, later in zip(lora_stage_flags, lora_stage_flags[1:])
        ):
            raise ValueError("train_lora may transition from false to true, but not back to false.")
        if self.lora is not None:
            _validate_lora_config(self.lora)
        for stage in self.stages:
            _validate_stage(stage)

        _validate_branch_config(self)
        _validate_checkpoint_initialization_config(self)
        _validate_train_sampling_config(self.train_sampling)
        _validate_post_training_benchmark_config(self.post_training_benchmark)
        _validate_valid_mask_config(self.valid_mask)
        _validate_validation_config(self)
        if self.pyramid_strides != [8, 16, 32]:
            raise ValueError(
                "The current simple DINOv3 pyramid supports pyramid_strides: [8, 16, 32]."
            )
        _validate_augmentation_config(self.augmentation)
        _validate_detector_height_fusion_config(self)
        _validate_semi_supervised_config(self)


@dataclass
class HeightTrainingConfig:
    data_root_path: str = "outputs/tiles_height"
    train_dataset_names: list[str] = field(default_factory=list)
    test_dataset_names: list[str] = field(default_factory=list)
    train_output_path: str = "outputs/heights/training"
    run_name: Optional[str] = None

    model_type: str = "dpt_height"
    dinov3_model_dir: Optional[str] = None
    dinov3_model: str = "dinov3_vitl16"
    dinov3_pretraining: str = "sat493m"
    dinov3_embed_dim: int = 1024
    dinov3_patch_size: int = 16
    dinov3_use_fake_backbone: bool = False
    dinov3_fake_num_blocks: int = 4

    dpt_feature_layers: list[int] = field(default_factory=lambda: [5, 11, 17, 23])
    dpt_feature_dim: int = 256
    dpt_out_channels: list[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    dpt_use_batch_norm: bool = False
    height_output_channels: int = 1

    min_valid_height_m: float = 0.0
    max_valid_height_m: float = 450.0
    log_variance_min: float = -4.0
    log_variance_max: float = 8.0
    loss_name: str = "smooth_l1"

    lora: Optional[LoRAConfig] = None
    stages: list[HeightTrainingStageConfig] = field(
        default_factory=lambda: [HeightTrainingStageConfig()]
    )
    post_training_benchmark: PostTrainingBenchmarkConfig = field(
        default_factory=PostTrainingBenchmarkConfig
    )

    batch_size: int = 2
    grad_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    seed: int = 42
    use_amp: bool = True
    debug_overfit_batches: int = 0
    debug_eval_on_train: bool = False
    preserve_input_resolution: bool = True

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    resolution_degradation: HeightResolutionDegradationConfig = field(
        default_factory=HeightResolutionDegradationConfig
    )
    valid_mask: ValidMaskConfig = field(default_factory=ValidMaskConfig)
    validation: HeightValidationConfig = field(default_factory=HeightValidationConfig)
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            project="synthforest_height",
            tags=["dinov3", "height", "dpt", "synthforest"],
        )
    )
    sweep: Optional[SweepRuntimeConfig] = None
    domain_adaptation: DomainAdaptationConfig = field(default_factory=DomainAdaptationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HeightTrainingConfig":
        _, raw = load_yaml(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
    ) -> "HeightTrainingConfig":
        cfg = _coerce_dataclass(cls, raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def total_iterations(self) -> int:
        return sum(int(stage.max_iterations) for stage in self.stages)

    @property
    def final_stage(self) -> HeightTrainingStageConfig:
        return self.stages[-1]

    def resolved_synthetic_validation_root(self) -> str:
        return self.validation.synthetic.data_root_path or self.data_root_path

    def resolved_synthetic_validation_names(self) -> list[str]:
        names = self.validation.synthetic.dataset_names
        return list(self.train_dataset_names if names is None else names)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        resize_tag = format_range(self.augmentation.resize_min_sizes_train)
        crop_tag = format_range(self.augmentation.crop_size_range)
        model_tag = self.dinov3_model.removeprefix("dinov3_")
        stage_tag = "_".join(format_stage_tag(stage) for stage in self.stages)
        adaptation_tag = "frozen"
        if self.lora is not None:
            adaptation_tag = f"lora{self.lora.num_blocks}b_r{self.lora.rank}"

        parts = [
            "dinov3dptheight",
            model_tag,
            stage_tag,
            adaptation_tag,
            self.final_stage.optimizer.name.lower(),
            f"b{self.batch_size}",
            f"acc{self.grad_accumulation_steps}",
            f"img{resize_tag}",
            f"crop{crop_tag}",
            format_domain_adaptation_tag(self.domain_adaptation),
            stamp,
        ]
        if slurm_job_id:
            parts.append(f"job{slurm_job_id}")
        return slugify("_".join(parts))

    def validate(self) -> None:
        if self.model_type != "dpt_height":
            raise ValueError("height model_type must be dpt_height.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative.")
        if self.debug_overfit_batches < 0:
            raise ValueError("debug_overfit_batches must be non-negative.")
        if not self.train_dataset_names:
            raise ValueError("train_dataset_names must contain at least one dataset.")
        if not self.dinov3_use_fake_backbone and not self.dinov3_model_dir:
            raise ValueError(
                "dinov3_model_dir is required unless dinov3_use_fake_backbone is true."
            )
        if self.dinov3_use_fake_backbone and self.dinov3_fake_num_blocks <= 0:
            raise ValueError("dinov3_fake_num_blocks must be positive.")
        if self.dinov3_patch_size <= 0:
            raise ValueError("dinov3_patch_size must be positive.")
        if not self.dpt_feature_layers:
            raise ValueError("dpt_feature_layers must contain at least one layer.")
        if self.dpt_feature_layers != sorted(set(int(v) for v in self.dpt_feature_layers)):
            raise ValueError("dpt_feature_layers must be sorted and contain no duplicates.")
        if len(self.dpt_feature_layers) != 4:
            raise ValueError("The DPT decoder expects exactly four feature layers.")
        if (
            self.dinov3_use_fake_backbone
            and max(self.dpt_feature_layers) >= self.dinov3_fake_num_blocks
        ):
            raise ValueError(
                "dpt_feature_layers must be less than dinov3_fake_num_blocks "
                "when using the fake backbone."
            )
        if len(self.dpt_out_channels) != 4:
            raise ValueError("dpt_out_channels must contain exactly four values.")
        if self.dpt_feature_dim <= 0 or any(int(value) <= 0 for value in self.dpt_out_channels):
            raise ValueError("DPT feature dimensions must be positive.")
        if int(self.height_output_channels) not in {1, 2}:
            raise ValueError("height_output_channels must be either 1 or 2.")
        self.height_output_channels = int(self.height_output_channels)
        if self.min_valid_height_m < 0:
            raise ValueError("min_valid_height_m must be non-negative.")
        if self.max_valid_height_m <= self.min_valid_height_m:
            raise ValueError("max_valid_height_m must exceed min_valid_height_m.")
        self.log_variance_min = float(self.log_variance_min)
        self.log_variance_max = float(self.log_variance_max)
        if not self.log_variance_max > self.log_variance_min:
            raise ValueError("log_variance_max must exceed log_variance_min.")
        self.loss_name = str(self.loss_name).strip().lower()
        if self.loss_name not in {"smooth_l1", "laplace_nll"}:
            raise ValueError("loss_name must be one of: smooth_l1, laplace_nll.")
        if self.loss_name == "smooth_l1" and self.height_output_channels != 1:
            raise ValueError("loss_name smooth_l1 requires height_output_channels: 1.")
        if self.loss_name == "laplace_nll" and self.height_output_channels != 2:
            raise ValueError("loss_name laplace_nll requires height_output_channels: 2.")
        if not self.stages:
            raise ValueError("stages must contain at least one stage.")

        stage_names = [slugify(stage.name) for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("stage names must be unique.")

        lora_stage_flags = [bool(stage.train_lora) for stage in self.stages]
        if any(lora_stage_flags) and self.lora is None:
            raise ValueError("A stage with train_lora: true requires a lora configuration.")
        if self.lora is not None and not any(lora_stage_flags):
            raise ValueError(
                "A lora configuration requires at least one stage with train_lora: true."
            )
        if any(
            earlier and not later for earlier, later in zip(lora_stage_flags, lora_stage_flags[1:])
        ):
            raise ValueError("train_lora may transition from false to true, but not back to false.")
        if self.lora is not None:
            _validate_lora_config(self.lora)
        for stage in self.stages:
            _validate_height_stage(stage)

        _validate_post_training_benchmark_config(self.post_training_benchmark)
        _validate_valid_mask_config(self.valid_mask)
        _validate_height_resolution_degradation_config(self.resolution_degradation)
        _validate_height_validation_config(self)
        _validate_height_sweep_config(self)
        _validate_augmentation_config(self.augmentation)


def _validate_stage(stage: TrainingStageConfig) -> None:
    if not str(stage.name).strip():
        raise ValueError("stage.name must not be empty.")
    if stage.max_iterations <= 0:
        raise ValueError(f"stage {stage.name}: max_iterations must be positive.")
    if stage.eval_interval <= 0:
        raise ValueError(f"stage {stage.name}: eval_interval must be positive.")
    if stage.checkpoint_interval <= 0:
        raise ValueError(f"stage {stage.name}: checkpoint_interval must be positive.")
    if stage.train_log_interval <= 0:
        raise ValueError(f"stage {stage.name}: train_log_interval must be positive.")
    optimizer = stage.optimizer
    if optimizer.detector_lr <= 0:
        raise ValueError(f"stage {stage.name}: optimizer.detector_lr must be positive.")
    if optimizer.detector_weight_decay < 0:
        raise ValueError(
            f"stage {stage.name}: optimizer.detector_weight_decay must be non-negative."
        )
    if stage.train_lora and optimizer.lora_lr is None:
        raise ValueError(
            f"stage {stage.name}: optimizer.lora_lr is required when train_lora is true."
        )
    if optimizer.lora_lr is not None and optimizer.lora_lr <= 0:
        raise ValueError(f"stage {stage.name}: optimizer.lora_lr must be positive when set.")
    if optimizer.lora_weight_decay < 0:
        raise ValueError(f"stage {stage.name}: optimizer.lora_weight_decay must be non-negative.")
    if optimizer.warmup_iterations < 0:
        raise ValueError(f"stage {stage.name}: optimizer.warmup_iterations must be non-negative.")
    if not 0.0 <= optimizer.min_lr_factor <= 1.0:
        raise ValueError(f"stage {stage.name}: optimizer.min_lr_factor must be in [0, 1].")


def _validate_height_stage(stage: HeightTrainingStageConfig) -> None:
    if not str(stage.name).strip():
        raise ValueError("stage.name must not be empty.")
    if stage.max_iterations <= 0:
        raise ValueError(f"stage {stage.name}: max_iterations must be positive.")
    if stage.eval_interval <= 0:
        raise ValueError(f"stage {stage.name}: eval_interval must be positive.")
    if stage.checkpoint_interval <= 0:
        raise ValueError(f"stage {stage.name}: checkpoint_interval must be positive.")
    if stage.train_log_interval <= 0:
        raise ValueError(f"stage {stage.name}: train_log_interval must be positive.")
    optimizer = stage.optimizer
    if optimizer.decoder_lr <= 0:
        raise ValueError(f"stage {stage.name}: optimizer.decoder_lr must be positive.")
    if optimizer.decoder_weight_decay < 0:
        raise ValueError(
            f"stage {stage.name}: optimizer.decoder_weight_decay must be non-negative."
        )
    if stage.train_lora and optimizer.lora_lr is None:
        raise ValueError(
            f"stage {stage.name}: optimizer.lora_lr is required when train_lora is true."
        )
    if optimizer.lora_lr is not None and optimizer.lora_lr <= 0:
        raise ValueError(f"stage {stage.name}: optimizer.lora_lr must be positive when set.")
    if optimizer.lora_weight_decay < 0:
        raise ValueError(f"stage {stage.name}: optimizer.lora_weight_decay must be non-negative.")
    if optimizer.warmup_iterations < 0:
        raise ValueError(f"stage {stage.name}: optimizer.warmup_iterations must be non-negative.")
    if not 0.0 <= optimizer.min_lr_factor <= 1.0:
        raise ValueError(f"stage {stage.name}: optimizer.min_lr_factor must be in [0, 1].")


def _validate_height_sweep_config(config: HeightTrainingConfig) -> None:
    if config.sweep is None:
        return
    if config.sweep.phase != "frozen":
        raise ValueError("height sweep.phase currently supports only 'frozen'.")
    if config.lora is not None or any(stage.train_lora for stage in config.stages):
        raise ValueError("Frozen height sweeps must keep DINOv3 frozen and set lora: null.")
    objective_metric = config.sweep.objective_metric or config.validation.selection_metric
    if objective_metric != config.validation.selection_metric:
        raise ValueError("height sweep.objective_metric must match validation.selection_metric.")
    objective_iterations = [int(value) for value in config.sweep.objective_iterations]
    if objective_iterations != sorted(set(objective_iterations)):
        raise ValueError(
            "height sweep.objective_iterations must be sorted and contain no duplicates."
        )
    if not objective_iterations:
        raise ValueError("height sweep.objective_iterations must not be empty.")
    if any(
        iteration <= 0 or iteration > config.total_iterations for iteration in objective_iterations
    ):
        raise ValueError(
            "height sweep.objective_iterations must be positive and no greater than "
            "the final global iteration."
        )
    if config.sweep.parent_control_metrics:
        raise ValueError("Frozen height sweeps do not support parent_control_metrics.")


def _validate_lora_config(config: LoRAConfig) -> None:
    if config.num_blocks <= 0:
        raise ValueError("lora.num_blocks must be positive.")
    if config.rank <= 0:
        raise ValueError("lora.rank must be positive.")
    if config.alpha <= 0:
        raise ValueError("lora.alpha must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("lora.dropout must be in [0, 1).")
    if not config.target_modules:
        raise ValueError("lora.target_modules must contain at least one module name.")
    if any(not str(target).strip() for target in config.target_modules):
        raise ValueError("lora.target_modules must not contain empty names.")


def _validate_branch_config(config: DetectorTrainingConfig) -> None:
    completed_iterations = 0
    if config.initialization is not None:
        initialization = config.initialization
        if not initialization.checkpoint_path.strip():
            raise ValueError("initialization.checkpoint_path must not be empty.")
        if initialization.completed_iterations < 0:
            raise ValueError("initialization.completed_iterations must be non-negative.")
        completed_iterations = int(initialization.completed_iterations)

    final_iteration = completed_iterations + config.total_iterations
    branch_iterations = [int(value) for value in config.branch_checkpoint_iterations]
    if branch_iterations != sorted(set(branch_iterations)):
        raise ValueError("branch_checkpoint_iterations must be sorted and contain no duplicates.")
    if any(
        iteration <= completed_iterations or iteration > final_iteration
        for iteration in branch_iterations
    ):
        raise ValueError(
            "branch_checkpoint_iterations must be greater than the initialization "
            "iteration and no greater than the final global iteration."
        )

    if config.sweep is None:
        return
    if config.sweep.phase not in {"frozen", "lora"}:
        raise ValueError("sweep.phase must be either 'frozen' or 'lora'.")
    objective_metric = config.sweep.objective_metric or config.validation.selection_metric
    allowed_objectives = {
        "bbox_dense_map",
        "segm_dense_map",
    }
    if objective_metric not in allowed_objectives:
        raise ValueError(
            "sweep.objective_metric must be one of: " + ", ".join(sorted(allowed_objectives))
        )
    if objective_metric == "segm_dense_map" and (
        config.model_type != "maskrcnn" or "segm" not in config.validation.tasks
    ):
        raise ValueError(
            "sweep.objective_metric segm_dense_map requires Mask R-CNN "
            "with segmentation validation enabled."
        )
    objective_iterations = [int(value) for value in config.sweep.objective_iterations]
    if objective_iterations != sorted(set(objective_iterations)):
        raise ValueError("sweep.objective_iterations must be sorted and contain no duplicates.")
    if not objective_iterations:
        raise ValueError("sweep.objective_iterations must not be empty.")
    if any(
        iteration <= completed_iterations or iteration > final_iteration
        for iteration in objective_iterations
    ):
        raise ValueError(
            "sweep.objective_iterations must be greater than the initialization "
            "iteration and no greater than the final global iteration."
        )
    for iteration, value in config.sweep.parent_control_metrics.items():
        if not isinstance(iteration, str):
            raise ValueError("sweep.parent_control_metrics keys must be iteration strings.")
        try:
            parsed_iteration = int(iteration)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sweep.parent_control_metrics keys must be global iterations."
            ) from exc
        if parsed_iteration <= 0:
            raise ValueError("sweep.parent_control_metrics keys must be positive iterations.")
        if not isinstance(value, (int, float)):
            raise ValueError("sweep.parent_control_metrics values must be numeric.")


def _validate_checkpoint_initialization_config(config: DetectorTrainingConfig) -> None:
    if config.initialization is not None and config.checkpoint_initialization is not None:
        raise ValueError("initialization and checkpoint_initialization are mutually exclusive.")
    if config.checkpoint_initialization is None:
        return
    if not config.checkpoint_initialization.checkpoint_path.strip():
        raise ValueError("checkpoint_initialization.checkpoint_path must not be empty.")


def _validate_train_sampling_config(config: TrainSamplingConfig) -> None:
    if not 0.0 < float(config.fraction) <= 1.0:
        raise ValueError("train_sampling.fraction must be in (0, 1].")
    if int(config.min_samples) <= 0:
        raise ValueError("train_sampling.min_samples must be positive.")


def _validate_post_training_benchmark_config(
    config: PostTrainingBenchmarkConfig,
) -> None:
    validate_checkpoint_name(config.checkpoint_name)
    if config.config_path is not None and not str(config.config_path).strip():
        raise ValueError("post_training_benchmark.config_path must not be empty when set.")
    if config.enabled and config.config_path is None:
        raise ValueError(
            "post_training_benchmark.config_path is required when "
            "post_training_benchmark.enabled is true."
        )


def _validate_augmentation_config(config: AugmentationConfig) -> None:
    if not config.resize_min_sizes_train:
        raise ValueError("augmentation.resize_min_sizes_train must not be empty.")
    if any(int(size) <= 0 for size in config.resize_min_sizes_train):
        raise ValueError("augmentation.resize_min_sizes_train values must be positive.")
    if int(config.resize_min_size_eval) <= 0:
        raise ValueError("augmentation.resize_min_size_eval must be positive.")


def _validate_valid_mask_config(config: ValidMaskConfig) -> None:
    if config.source not in {"auto", "metadata", "border_black"}:
        raise ValueError("valid_mask.source must be one of: auto, metadata, border_black.")
    if not 0 <= config.black_pixel_tolerance <= 255:
        raise ValueError("valid_mask.black_pixel_tolerance must be in [0, 255].")
    fraction_fields = {
        "da_full_tile_min_valid_fraction": config.da_full_tile_min_valid_fraction,
        "crop_min_valid_fraction": config.crop_min_valid_fraction,
        "target_min_valid_fraction": config.target_min_valid_fraction,
    }
    for name, value in fraction_fields.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"valid_mask.{name} must be in [0, 1].")
    prediction_threshold = config.prediction_min_valid_fraction
    if prediction_threshold is not None and not 0.0 <= float(prediction_threshold) <= 1.0:
        raise ValueError("valid_mask.prediction_min_valid_fraction must be null or in [0, 1].")


def _validate_detector_height_fusion_config(config: DetectorTrainingConfig) -> None:
    fusion = config.height_fusion
    if bool(fusion.use_uncertainty) and not bool(fusion.enabled):
        raise ValueError("height_fusion.use_uncertainty requires height_fusion.enabled: true.")
    if bool(fusion.use_uncertainty) and config.model_type != "fasterrcnn":
        raise ValueError("height_fusion.use_uncertainty v1 supports only model_type: fasterrcnn.")
    if float(fusion.uncertainty_log_variance_max) <= float(fusion.uncertainty_log_variance_min):
        raise ValueError(
            "height_fusion.uncertainty_log_variance_max must exceed uncertainty_log_variance_min."
        )
    if float(fusion.uncertainty_reliability_tau) <= 0.0:
        raise ValueError("height_fusion.uncertainty_reliability_tau must be positive.")
    if not 0.0 <= float(fusion.uncertainty_reliability_min) <= 1.0:
        raise ValueError("height_fusion.uncertainty_reliability_min must be in [0, 1].")
    if not fusion.enabled:
        return
    fusion.height_maps_subdir = str(fusion.height_maps_subdir).strip()
    if not fusion.height_maps_subdir:
        raise ValueError("height_fusion.height_maps_subdir must not be empty.")
    if Path(fusion.height_maps_subdir).is_absolute():
        raise ValueError("height_fusion.height_maps_subdir must be a relative subdirectory.")
    levels = [str(level).strip() for level in fusion.levels]
    if not levels:
        raise ValueError("height_fusion.levels must contain at least one FPN level.")
    allowed_levels = {"p8", "p16", "p32"}
    unsupported = sorted(set(levels) - allowed_levels)
    if unsupported:
        raise ValueError(
            "height_fusion.levels supports only p8, p16, and p32; got: " + ", ".join(unsupported)
        )
    if len(set(levels)) != len(levels):
        raise ValueError("height_fusion.levels must not contain duplicates.")
    fusion.levels = levels
    if fusion.generate_missing and not str(fusion.height_checkpoint_path or "").strip():
        raise ValueError(
            "height_fusion.height_checkpoint_path is required when "
            "height_fusion.generate_missing is true."
        )
    if fusion.height_checkpoint_path is not None:
        fusion.height_checkpoint_path = str(fusion.height_checkpoint_path).strip()
        if not fusion.height_checkpoint_path:
            fusion.height_checkpoint_path = None
    if fusion.height_train_config_path is not None:
        fusion.height_train_config_path = str(fusion.height_train_config_path).strip()
        if not fusion.height_train_config_path:
            fusion.height_train_config_path = None
    if float(fusion.source_gsd_m) <= 0.0:
        raise ValueError("height_fusion.source_gsd_m must be positive.")
    if float(fusion.target_gsd_m) <= 0.0:
        raise ValueError("height_fusion.target_gsd_m must be positive.")
    if float(fusion.target_gsd_m) < float(fusion.source_gsd_m):
        raise ValueError("height_fusion.target_gsd_m must be >= source_gsd_m.")
    if float(fusion.clip_max_m) <= float(fusion.clip_min_m):
        raise ValueError("height_fusion.clip_max_m must exceed clip_min_m.")
    if float(fusion.std_m) <= 0.0:
        raise ValueError("height_fusion.std_m must be positive.")
    if int(fusion.encoder_hidden_channels) <= 0:
        raise ValueError("height_fusion.encoder_hidden_channels must be positive.")
    if float(fusion.fusion_init_scale) < 0.0:
        raise ValueError("height_fusion.fusion_init_scale must be non-negative.")


def _validate_semi_supervised_config(config: DetectorTrainingConfig) -> None:
    stage_flags = [bool(stage.semi_supervised) for stage in config.stages]
    semi = config.semi_supervised
    if any(stage_flags) and not semi.enabled:
        raise ValueError(
            "A stage with semi_supervised: true requires semi_supervised.enabled: true."
        )
    if not semi.enabled:
        return
    if not any(stage_flags):
        raise ValueError(
            "semi_supervised.enabled requires at least one stage with semi_supervised: true."
        )
    if not 0.0 <= float(semi.ema_decay) < 1.0:
        raise ValueError("semi_supervised.ema_decay must be in [0, 1).")
    if float(semi.unsupervised_loss_start_weight) < 0.0:
        raise ValueError("semi_supervised.unsupervised_loss_start_weight must be non-negative.")
    if float(semi.unsupervised_loss_weight) < 0.0:
        raise ValueError("semi_supervised.unsupervised_loss_weight must be non-negative.")
    if float(semi.unsupervised_loss_start_weight) > float(semi.unsupervised_loss_weight):
        raise ValueError(
            "semi_supervised.unsupervised_loss_start_weight must be <= "
            "semi_supervised.unsupervised_loss_weight."
        )
    if not 0.0 <= float(semi.unsupervised_loss_ramp_fraction) <= 1.0:
        raise ValueError("semi_supervised.unsupervised_loss_ramp_fraction must be in [0, 1].")

    real = semi.unlabeled_real
    if not str(real.data_root_path).strip():
        raise ValueError("semi_supervised.unlabeled_real.data_root_path is required.")
    if not real.dataset_names:
        raise ValueError("semi_supervised.unlabeled_real.dataset_names must not be empty.")
    if real.fold not in {"train", "valid", "test"}:
        raise ValueError("semi_supervised.unlabeled_real.fold must be train, valid, or test.")
    if not 0.0 <= real.edge_tile_black_pixel_threshold <= 1.0:
        raise ValueError(
            "semi_supervised.unlabeled_real.edge_tile_black_pixel_threshold must be in [0, 1]."
        )
    if not 0 <= real.edge_tile_black_pixel_tolerance <= 255:
        raise ValueError(
            "semi_supervised.unlabeled_real.edge_tile_black_pixel_tolerance must be in [0, 255]."
        )
    if real.max_products_per_dataset is not None and real.max_products_per_dataset <= 0:
        raise ValueError(
            "semi_supervised.unlabeled_real.max_products_per_dataset must be positive when set."
        )
    if real.max_tiles_per_product is not None and real.max_tiles_per_product <= 0:
        raise ValueError(
            "semi_supervised.unlabeled_real.max_tiles_per_product must be positive when set."
        )

    pseudo = semi.pseudo_labels
    if not 0.0 <= float(pseudo.score_threshold) <= 1.0:
        raise ValueError("semi_supervised.pseudo_labels.score_threshold must be in [0, 1].")
    if float(pseudo.min_box_size) < 0.0:
        raise ValueError("semi_supervised.pseudo_labels.min_box_size must be non-negative.")
    if float(pseudo.min_box_area) < 0.0:
        raise ValueError("semi_supervised.pseudo_labels.min_box_area must be non-negative.")
    if int(pseudo.max_labels_per_image) <= 0:
        raise ValueError("semi_supervised.pseudo_labels.max_labels_per_image must be positive.")
    if pseudo.valid_mask_min_fraction is not None and not (
        0.0 <= float(pseudo.valid_mask_min_fraction) <= 1.0
    ):
        raise ValueError(
            "semi_supervised.pseudo_labels.valid_mask_min_fraction must be null or in [0, 1]."
        )
    if pseudo.nms_threshold is not None and not 0.0 <= float(pseudo.nms_threshold) <= 1.0:
        raise ValueError("semi_supervised.pseudo_labels.nms_threshold must be null or in [0, 1].")
    _validate_semi_supervised_strong_augmentation_config(semi.strong_augmentation)


def _validate_semi_supervised_strong_augmentation_config(
    config: SemiSupervisedStrongAugmentationConfig,
) -> None:
    prefix = "semi_supervised.strong_augmentation"
    for name in (
        "brightness_prob",
        "contrast_prob",
        "saturation_prob",
        "hue_prob",
        "gaussian_blur_prob",
        "gaussian_noise_prob",
        "jpeg_prob",
        "downsample_upsample_prob",
    ):
        if not 0.0 <= float(getattr(config, name)) <= 1.0:
            raise ValueError(f"{prefix}.{name} must be in [0, 1].")
    for name in ("brightness", "contrast", "saturation", "hue"):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{prefix}.{name} must be non-negative.")
    _validate_float_range(
        config.gaussian_blur_sigma,
        f"{prefix}.gaussian_blur_sigma",
        min_value=0.0,
    )
    _validate_float_range(
        config.gaussian_noise_std,
        f"{prefix}.gaussian_noise_std",
        min_value=0.0,
    )
    _validate_float_range(
        config.downsample_scale,
        f"{prefix}.downsample_scale",
        min_value=0.0,
        max_value=1.0,
    )
    _validate_int_range(
        config.jpeg_quality,
        f"{prefix}.jpeg_quality",
        min_value=1,
        max_value=100,
    )


def _validate_float_range(
    values: list[float],
    name: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = [float(value) for value in values]
    if low > high:
        raise ValueError(f"{name} must be ordered as [min, max].")
    if min_value is not None and low < float(min_value):
        raise ValueError(f"{name} values must be >= {float(min_value):g}.")
    if max_value is not None and high > float(max_value):
        raise ValueError(f"{name} values must be <= {float(max_value):g}.")


def _validate_int_range(
    values: list[int],
    name: str,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = [int(value) for value in values]
    if low > high:
        raise ValueError(f"{name} must be ordered as [min, max].")
    if min_value is not None and low < int(min_value):
        raise ValueError(f"{name} values must be >= {int(min_value)}.")
    if max_value is not None and high > int(max_value):
        raise ValueError(f"{name} values must be <= {int(max_value)}.")


def _validate_height_resolution_degradation_config(
    config: HeightResolutionDegradationConfig,
) -> None:
    if not 0.0 <= float(config.prob) <= 1.0:
        raise ValueError("resolution_degradation.prob must be in [0, 1].")
    if float(config.base_gsd_m) <= 0.0:
        raise ValueError("resolution_degradation.base_gsd_m must be positive.")
    if not config.target_gsd_m_values:
        raise ValueError("resolution_degradation.target_gsd_m_values must not be empty.")
    if any(float(value) <= 0.0 for value in config.target_gsd_m_values):
        raise ValueError("resolution_degradation.target_gsd_m_values must be positive.")
    if any(float(value) < float(config.base_gsd_m) for value in config.target_gsd_m_values):
        raise ValueError("resolution_degradation.target_gsd_m_values must be >= base_gsd_m.")
    supported_reducers = {
        "p90_positive",
        "p95_positive",
        "p99_positive",
        "max_positive",
        "mean_positive",
    }
    if config.height_reducer not in supported_reducers:
        raise ValueError(
            "resolution_degradation.height_reducer must be one of: "
            + ", ".join(sorted(supported_reducers))
        )
    if not 0.0 <= float(config.min_canopy_fraction) <= 1.0:
        raise ValueError("resolution_degradation.min_canopy_fraction must be in [0, 1].")
    if len(config.blur_sigma) != 2:
        raise ValueError("resolution_degradation.blur_sigma must contain [min, max].")
    blur_min, blur_max = (float(config.blur_sigma[0]), float(config.blur_sigma[1]))
    if blur_min < 0.0 or blur_max < blur_min:
        raise ValueError("resolution_degradation.blur_sigma must be non-negative and sorted.")
    supported_interpolations = {"nearest", "box", "bilinear", "bicubic", "lanczos"}
    if str(config.downsample_interpolation).lower() not in supported_interpolations:
        raise ValueError(
            "resolution_degradation.downsample_interpolation must be one of: "
            + ", ".join(sorted(supported_interpolations))
        )
    if str(config.upsample_interpolation).lower() not in supported_interpolations:
        raise ValueError(
            "resolution_degradation.upsample_interpolation must be one of: "
            + ", ".join(sorted(supported_interpolations))
        )


def _validate_height_metric_resolution_config(
    config: HeightMetricResolutionConfig,
) -> None:
    if not 0.0 <= float(config.min_valid_fraction) <= 1.0:
        raise ValueError("metric_resolution.min_valid_fraction must be in [0, 1].")
    if config.rgb_gsd_m is not None and float(config.rgb_gsd_m) <= 0.0:
        raise ValueError("metric_resolution.rgb_gsd_m must be positive when set.")
    if config.height_gsd_m is not None and float(config.height_gsd_m) <= 0.0:
        raise ValueError("metric_resolution.height_gsd_m must be positive when set.")
    if config.downsample_factor is not None:
        factor = float(config.downsample_factor)
        if factor < 1.0:
            raise ValueError("metric_resolution.downsample_factor must be >= 1 when set.")
    if str(config.downsample_interpolation).lower() != "area":
        raise ValueError("metric_resolution.downsample_interpolation currently supports only area.")
    supported_upsample_interpolations = {"nearest", "bilinear"}
    if str(config.upsample_interpolation).lower() not in supported_upsample_interpolations:
        raise ValueError(
            "metric_resolution.upsample_interpolation must be one of: "
            + ", ".join(sorted(supported_upsample_interpolations))
        )
    if config.enabled:
        config.resolved_downsample_factor()


def _validate_validation_config(config: DetectorTrainingConfig) -> None:
    validation = config.validation
    allowed_metrics = {
        "bbox_map",
        "bbox_map_50",
        "bbox_map_75",
        "bbox_dense_map",
        "bbox_dense_map_50",
        "bbox_dense_map_75",
        "segm_map",
        "segm_map_50",
        "segm_map_75",
        "segm_dense_map",
        "segm_dense_map_50",
        "segm_dense_map_75",
    }
    if validation.selection_metric not in allowed_metrics:
        raise ValueError(
            "validation.selection_metric must be one of: " + ", ".join(sorted(allowed_metrics))
        )
    thresholds = validation.map_max_detection_thresholds
    if len(thresholds) != 3:
        raise ValueError(
            "validation.map_max_detection_thresholds must contain exactly three values."
        )
    if thresholds != sorted(thresholds) or any(int(value) <= 0 for value in thresholds):
        raise ValueError("validation.map_max_detection_thresholds must be positive and sorted.")
    dense_max = validation.dense_eval_max_detections
    if dense_max is not None and int(dense_max) <= thresholds[-1]:
        raise ValueError(
            "validation.dense_eval_max_detections must exceed the largest standard "
            "max-detection threshold."
        )
    if "dense_" in validation.selection_metric and dense_max is None:
        raise ValueError("A dense selection metric requires validation.dense_eval_max_detections.")
    if not validation.tasks:
        raise ValueError("validation.tasks must contain at least one task.")
    normalized_tasks = [str(task).lower() for task in validation.tasks]
    if len(set(normalized_tasks)) != len(normalized_tasks):
        raise ValueError("validation.tasks must not contain duplicates.")
    unsupported_tasks = sorted(set(normalized_tasks) - {"bbox", "segm"})
    if unsupported_tasks:
        raise ValueError(
            "validation.tasks supports only bbox and segm; got: " + ", ".join(unsupported_tasks)
        )
    validation.tasks = normalized_tasks
    if "bbox" not in normalized_tasks:
        raise ValueError("validation.tasks must always include bbox.")
    if "segm" in normalized_tasks and config.model_type != "maskrcnn":
        raise ValueError("validation.tasks may include segm only when model_type is maskrcnn.")
    if validation.selection_metric.startswith("segm_") and "segm" not in normalized_tasks:
        raise ValueError("A segm selection metric requires segm in validation.tasks.")
    if validation.batch_size is not None and int(validation.batch_size) <= 0:
        raise ValueError("validation.batch_size must be positive when set.")
    if validation.synthetic.fold not in {"train", "valid", "test"}:
        raise ValueError("validation.synthetic.fold must be train, valid, or test.")
    if not config.resolved_synthetic_validation_names():
        raise ValueError("Synthetic validation must contain at least one dataset.")

    real = validation.real
    if not real.enabled:
        return
    if real.fold != "valid":
        raise ValueError(
            "Training-time real validation must use fold: valid to avoid test leakage."
        )
    if not real.data_root_path:
        raise ValueError("validation.real.data_root_path is required.")
    if not real.dataset_names:
        raise ValueError("validation.real.dataset_names must not be empty.")
    if not 0.0 <= real.edge_tile_black_pixel_threshold <= 1.0:
        raise ValueError("validation.real.edge_tile_black_pixel_threshold must be in [0, 1].")
    if not 0 <= real.edge_tile_black_pixel_tolerance <= 255:
        raise ValueError("validation.real.edge_tile_black_pixel_tolerance must be in [0, 255].")
    if real.max_products_per_dataset is not None and real.max_products_per_dataset <= 0:
        raise ValueError("validation.real.max_products_per_dataset must be positive when set.")
    if real.max_tiles_per_product is not None and real.max_tiles_per_product <= 0:
        raise ValueError("validation.real.max_tiles_per_product must be positive when set.")


def _validate_height_validation_config(config: HeightTrainingConfig) -> None:
    validation = config.validation
    if not validation.canopy_thresholds_m:
        raise ValueError("validation.canopy_thresholds_m must not be empty.")
    if any(float(value) < 0.0 for value in validation.canopy_thresholds_m):
        raise ValueError("validation.canopy_thresholds_m values must be non-negative.")
    validation.canopy_thresholds_m = sorted(
        dict.fromkeys(float(value) for value in validation.canopy_thresholds_m)
    )
    allowed_metrics = {
        "mae",
        "rmse",
        "mse",
        "bias",
    }
    for threshold in validation.canopy_thresholds_m:
        tag = _height_threshold_tag(threshold)
        allowed_metrics.add(f"mae_gt_{tag}")
        allowed_metrics.add(f"rmse_gt_{tag}")
    if validation.selection_metric not in allowed_metrics:
        raise ValueError(
            "validation.selection_metric must be one of: " + ", ".join(sorted(allowed_metrics))
        )
    if validation.batch_size is not None and int(validation.batch_size) <= 0:
        raise ValueError("validation.batch_size must be positive when set.")
    if validation.synthetic.fold not in {"train", "valid", "test"}:
        raise ValueError("validation.synthetic.fold must be train, valid, or test.")
    if not config.resolved_synthetic_validation_names():
        raise ValueError("Synthetic validation must contain at least one dataset.")

    real = validation.real
    if not real.enabled:
        return
    if real.fold != "valid":
        raise ValueError("Training-time real height validation must use fold: valid.")
    if not real.data_root_path:
        raise ValueError("validation.real.data_root_path is required.")
    if not real.dataset_names:
        raise ValueError("validation.real.dataset_names must not be empty.")
    if real.max_products_per_dataset is not None and real.max_products_per_dataset <= 0:
        raise ValueError("validation.real.max_products_per_dataset must be positive when set.")
    if real.max_tiles_per_product is not None and real.max_tiles_per_product <= 0:
        raise ValueError("validation.real.max_tiles_per_product must be positive when set.")
    _validate_height_metric_resolution_config(real.metric_resolution)
    if real.sample_scenes_per_eval is not None and real.sample_scenes_per_eval <= 0:
        raise ValueError("validation.real.sample_scenes_per_eval must be positive when set.")


def _height_threshold_tag(value: float) -> str:
    text = f"{float(value):.3g}".replace(".", "p")
    return f"{text}m"


def _coerce_dataclass(dataclass_type: type, raw: dict[str, Any]) -> Any:
    known = {item.name: item for item in fields(dataclass_type)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise TypeError(f"Unknown {dataclass_type.__name__} field(s): {', '.join(unknown)}")
    type_hints = get_type_hints(dataclass_type)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        kwargs[key] = _coerce_value(type_hints.get(key, known[key].type), value)
    return dataclass_type(**kwargs)


def _coerce_value(target_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is list:
        element_type = args[0] if args else Any
        return [_coerce_value(element_type, item) for item in value]
    if origin is not None and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        return _coerce_value(non_none_args[0], value) if non_none_args else value
    if isinstance(target_type, type) and is_dataclass(target_type):
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected a mapping for {target_type.__name__}, got {type(value).__name__}."
            )
        return _coerce_dataclass(target_type, value)
    return value
