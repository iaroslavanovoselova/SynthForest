from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import yaml

from synthforest.config_io import load_yaml
from synthforest.training.config import default_real_validation_root
from synthforest.training.layout import (
    source_run_name,
    training_run_folder_from_checkpoint,
    validate_checkpoint_name,
)

DEFAULT_BENCHMARK_DATASETS = [
    "QuebecTrees",
    "Detectree2",
    "NeonTreeEvaluation",
    "OAM-TCD",
]


@dataclass
class BenchmarkWandbConfig:
    enabled: bool = True
    project: str = "synthforest_real_eval"
    entity: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: Optional[str] = None
    required: bool = True
    step: int = 0
    resume_existing: bool = True


@dataclass
class BenchmarkConfig:
    model_type: str = ""
    train_config_path: Optional[str] = None
    run_folder: Optional[str] = None
    checkpoint_name: str = "last.pth"
    checkpoint_path: Optional[str] = None
    real_data_root: str = field(default_factory=default_real_validation_root)
    output_folder: str = "outputs/detectors/benchmarks"
    benchmark_name: Optional[str] = None
    prediction_source: Optional[str] = None
    reuse_predictions: bool = True
    resume_existing_benchmark: bool = True

    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARK_DATASETS))
    fold_name: str = "test"
    tile_only: bool = True
    score_threshold: float = 0.05
    nms_threshold: float = 0.5
    nms_algorithm: str = "iou"
    eval_iou_threshold: str | list[float] | float = "50:95"
    raster_eval_iou_threshold: str | list[float] | float = "50:95"
    require_raster_inputs: bool = False
    evaluate_segmentation_when_available: bool = True
    mask_probability_threshold: Optional[float] = None

    batch_size: int = 2
    num_workers: int = 4
    device: Optional[str] = None
    seed: int = 42
    max_products: Optional[int] = None
    max_tiles: Optional[int] = None
    drop_edge_tiles: bool = False
    edge_tile_black_pixel_threshold: float = 0.5
    edge_tile_black_pixel_tolerance: int = 0
    allow_hf_gpkg_download: bool = False
    wandb: BenchmarkWandbConfig = field(default_factory=BenchmarkWandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, validate: bool = True) -> "BenchmarkConfig":
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

    def resolved_run_folder(self) -> Path | None:
        if self.run_folder:
            return Path(self.run_folder)
        if self.checkpoint_path:
            return training_run_folder_from_checkpoint(self.checkpoint_path)
        return None

    def resolved_benchmark_name(self) -> str:
        if self.benchmark_name:
            return self.benchmark_name
        checkpoint_tag = Path(self.checkpoint_name).with_suffix("").as_posix()
        if self.run_folder is None:
            checkpoint_tag = (
                f"{source_run_name(self.resolved_checkpoint_path(), None)}-{checkpoint_tag}"
            )
        dataset_tag = "-".join(self.datasets)
        parts = [
            "benchmark",
            checkpoint_tag,
            self.fold_name,
            dataset_tag,
        ]
        if not self.resume_existing_benchmark:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        return _slugify("_".join(parts))

    def validate(self) -> None:
        if self.model_type not in {"fasterrcnn", "maskrcnn"}:
            raise ValueError("model_type must be one of: fasterrcnn, maskrcnn.")
        self.run_folder = _clean_optional_path(self.run_folder)
        self.checkpoint_path = _clean_optional_path(self.checkpoint_path)
        self.prediction_source = _clean_optional_path(self.prediction_source)
        validate_checkpoint_name(self.checkpoint_name)
        if self.run_folder and self.checkpoint_path:
            raise ValueError("run_folder and checkpoint_path are mutually exclusive.")
        if not self.run_folder and not self.checkpoint_path:
            raise ValueError("Either run_folder or checkpoint_path is required.")
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset name.")
        if self.fold_name not in {"train", "valid", "test"}:
            raise ValueError("fold_name must be one of: train, valid, test.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.max_products is not None and self.max_products <= 0:
            raise ValueError("max_products must be positive when set.")
        if self.max_tiles is not None and self.max_tiles <= 0:
            raise ValueError("max_tiles must be positive when set.")
        if not 0.0 <= self.edge_tile_black_pixel_threshold <= 1.0:
            raise ValueError("edge_tile_black_pixel_threshold must be between 0 and 1.")
        if not 0 <= self.edge_tile_black_pixel_tolerance <= 255:
            raise ValueError("edge_tile_black_pixel_tolerance must be between 0 and 255.")
        if self.wandb.step < 0:
            raise ValueError("wandb.step must be non-negative.")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1.")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1.")
        if self.nms_algorithm != "iou":
            raise ValueError("nms_algorithm must currently be 'iou'.")
        if (
            self.mask_probability_threshold is not None
            and not 0.0 <= float(self.mask_probability_threshold) <= 1.0
        ):
            raise ValueError("mask_probability_threshold must be null or in [0, 1].")
        if self.allow_hf_gpkg_download:
            raise ValueError(
                "allow_hf_gpkg_download is not implemented in this repo. Download or copy "
                "CanopyRS GPKG files into real_data_root, or use tile_only: true."
            )
        parse_iou_thresholds(self.eval_iou_threshold)
        parse_iou_thresholds(self.raster_eval_iou_threshold)


@dataclass
class NmsSearchConfig(BenchmarkConfig):
    """Validation-set NMS grid search adapted from CanopyRS evaluation logic."""

    fold_name: str = "valid"
    tile_only: bool = True
    objective_iou_threshold: str | list[float] | float = 0.75
    nms_iou_thresholds: list[float] = field(
        default_factory=lambda: [round(index / 20.0, 2) for index in range(1, 21)]
    )
    nms_score_thresholds: list[float] = field(
        default_factory=lambda: [round(index / 20.0, 2) for index in range(1, 21)]
    )
    n_workers: int = 6
    search_name: Optional[str] = None
    wandb: BenchmarkWandbConfig = field(
        default_factory=lambda: BenchmarkWandbConfig(enabled=False, required=False)
    )

    def validate(self) -> None:
        super().validate()
        if self.fold_name != "valid":
            raise ValueError("NMS parameter search must use fold_name: valid.")
        if not self.nms_iou_thresholds:
            raise ValueError("nms_iou_thresholds must not be empty.")
        if not self.nms_score_thresholds:
            raise ValueError("nms_score_thresholds must not be empty.")
        for threshold in [*self.nms_iou_thresholds, *self.nms_score_thresholds]:
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError("All NMS search thresholds must be in [0, 1].")
        self.nms_iou_thresholds = sorted(
            {round(float(threshold), 6) for threshold in self.nms_iou_thresholds}
        )
        self.nms_score_thresholds = sorted(
            {round(float(threshold), 6) for threshold in self.nms_score_thresholds}
        )
        if self.n_workers <= 0:
            raise ValueError("n_workers must be positive.")
        parse_iou_thresholds(self.objective_iou_threshold)

    def resolved_search_name(self) -> str:
        if self.search_name:
            return _slugify(self.search_name)
        return _slugify(
            "_".join(
                [
                    "nms_search",
                    Path(self.checkpoint_name).with_suffix("").as_posix(),
                    "-".join(self.datasets),
                    f"objective_iou{_format_iou_tag(self.objective_iou_threshold)}",
                ]
            )
        )

    def resolved_search_output(self) -> Path:
        if self.resolved_run_folder() is not None:
            parent = self.resolved_run_folder() / "nms_searches"
        else:
            parent = Path(self.output_folder) / self.model_type / "nms_searches"
        return parent / self.resolved_search_name()

    def as_prediction_benchmark_config(self) -> BenchmarkConfig:
        benchmark_field_names = {item.name for item in fields(BenchmarkConfig)}
        raw = {key: value for key, value in self.to_dict().items() if key in benchmark_field_names}
        raw["tile_only"] = True
        raw["benchmark_name"] = None
        raw["wandb"] = {
            **raw["wandb"],
            "enabled": False,
            "required": False,
        }
        return _coerce_dataclass(BenchmarkConfig, raw)


def parse_iou_thresholds(value: str | float | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        thresholds = [float(item) for item in value]
    elif isinstance(value, (int, float)):
        thresholds = [float(value)]
    else:
        text = str(value).strip().lower()
        if text == "50:95":
            thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
        elif "," in text:
            thresholds = [float(item.strip()) for item in text.split(",") if item.strip()]
        else:
            thresholds = [float(text)]

    if not thresholds:
        raise ValueError("eval_iou_threshold must define at least one threshold.")
    for threshold in thresholds:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("IoU thresholds must be in the interval (0, 1].")
    return sorted(dict.fromkeys(round(float(t), 4) for t in thresholds))


def _coerce_dataclass(dataclass_type: type, raw: dict[str, Any]) -> Any:
    known = {field.name: field for field in fields(dataclass_type)}
    type_hints = get_type_hints(dataclass_type)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise TypeError(f"Unknown {dataclass_type.__name__} field: {key}")
        kwargs[key] = _coerce_value(type_hints.get(key, known[key].type), value)
    return dataclass_type(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (list, tuple):
        return list(value)
    if origin is Optional:
        return value
    if origin is not None:
        args = get_args(annotation)
        if type(None) in args and value is None:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce_value(arg, value)
            except Exception:
                continue
        return value
    if is_dataclass(annotation) and isinstance(value, dict):
        return _coerce_dataclass(annotation, value)
    return value


def _clean_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_iou_tag(value: str | float | list[float]) -> str:
    if isinstance(value, str):
        return value.lower().replace(":", "-").replace(",", "-").replace(".", "p")
    thresholds = parse_iou_thresholds(value)
    if len(thresholds) == 1:
        return _format_float(thresholds[0])
    return f"{_format_float(thresholds[0])}-{_format_float(thresholds[-1])}"


def _format_float(value: float) -> str:
    return f"{float(value):.3g}".replace(".", "p")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value
