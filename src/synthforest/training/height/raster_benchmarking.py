from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from synthforest.training.config import (
    HeightRasterBenchmarkConfig,
    HeightRasterPairConfig,
)
from synthforest.training.config.naming import slugify
from synthforest.training.height.benchmarking import _merged_wandb_tags
from synthforest.training.height.checkpoint import load_height_checkpoint_model
from synthforest.training.height.dataset import image_to_normalized_tensor, load_height_map
from synthforest.training.height.metric_resolution import apply_height_metric_resolution
from synthforest.training.height.metrics import (
    HeightMetricsAccumulator,
    macro_average_height_metrics,
)
from synthforest.training.layout import BenchmarkRunLayout, source_run_name
from synthforest.training.utils import set_all_seeds
from synthforest.training.valid_masks import load_image_with_valid_mask


@dataclass(frozen=True)
class HeightRasterRecord:
    dataset_name: str
    name: str
    rgb_path: Path
    height_path: Path | None


@dataclass(frozen=True)
class SlidingWindowPrediction:
    prediction: np.ndarray
    weight_sum: np.ndarray
    num_windows: int


def run_height_raster_benchmark(config: HeightRasterBenchmarkConfig) -> Path:
    config.validate()
    set_all_seeds(config.seed)
    device = torch.device(
        config.device
        if config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, training_config, checkpoint_metadata = load_height_checkpoint_model(
        config.resolved_checkpoint_path(),
        device=device,
        config_path=config.train_config_path,
    )
    config.wandb.tags = _merged_wandb_tags(
        training_tags=training_config.wandb.tags,
        benchmark_tags=config.wandb.tags,
    )
    layout = BenchmarkRunLayout.resolve(
        benchmark_name=config.resolved_benchmark_name(),
        output_root=config.output_folder,
        model_type=config.model_type,
        run_folder=config.run_folder,
    ).create()
    config.save_yaml(layout.benchmark_config)
    training_config.save_yaml(layout.training_config)
    _write_json(layout.checkpoint_metadata, checkpoint_metadata)

    records = resolve_height_raster_records(config)
    _write_json(
        layout.metadata_dir / "rasters.json",
        {
            "num_rasters": len(records),
            "rasters": [
                {
                    "dataset_name": record.dataset_name,
                    "name": record.name,
                    "rgb_path": str(record.rgb_path),
                    "height_path": str(record.height_path) if record.height_path else None,
                }
                for record in records
            ],
        },
    )

    training_run_name = _training_run_name(config, checkpoint_metadata)
    wandb_run = _init_wandb(
        config,
        layout.root,
        training_run_name=training_run_name,
        checkpoint_metadata=checkpoint_metadata,
    )
    rows: list[dict] = []
    metrics_by_raster: dict[str, dict[str, float]] = {}
    metrics_by_dataset_raster: dict[str, dict[str, dict[str, float]]] = {}
    metric_downsample_factor = config.metric_resolution.resolved_downsample_factor()

    model.eval()
    for record in records:
        image, rgb_valid_mask = _load_rgb_raster(record.rgb_path, training_config)
        result = _sliding_window_predict_height(
            model=model,
            image=image,
            valid_mask=rgb_valid_mask,
            training_config=training_config,
            benchmark_config=config,
            device=device,
        )

        prediction_path: Path | None = None
        if config.save_predictions:
            prediction_path = layout.predictions_dir / "rasters" / f"{record.name}.tif"
            save_prediction_raster(
                source_path=record.rgb_path,
                output_path=prediction_path,
                array=result.prediction,
            )

        weight_map_path: Path | None = None
        if config.save_weight_maps:
            weight_map_path = layout.predictions_dir / "weights" / f"{record.name}.tif"
            save_prediction_raster(
                source_path=record.rgb_path,
                output_path=weight_map_path,
                array=result.weight_sum.astype(np.float32, copy=False),
            )

        row: dict[str, object] = {
            "dataset_name": record.dataset_name,
            "raster_name": record.name,
            "rgb_path": str(record.rgb_path),
            "height_path": str(record.height_path) if record.height_path else "",
            "prediction_path": str(prediction_path) if prediction_path else "",
            "weight_map_path": str(weight_map_path) if weight_map_path else "",
            "width": float(image.width),
            "height": float(image.height),
            "tile_size": float(config.tile_size),
            "stride": float(config.stride),
            "num_windows": float(result.num_windows),
            "has_ground_truth": record.height_path is not None,
            "metric_resolution_enabled": config.metric_resolution.enabled,
            "metric_resolution_downsample_factor": metric_downsample_factor,
        }

        if record.height_path is not None:
            raster_metrics = _compute_raster_metrics(
                prediction=result.prediction,
                prediction_valid_mask=result.weight_sum > 0.0,
                height_path=record.height_path,
                target_size=image.size,
                training_config=training_config,
                benchmark_config=config,
            )
            metrics_by_raster[record.name] = raster_metrics
            metrics_by_dataset_raster.setdefault(record.dataset_name, {})[record.name] = (
                raster_metrics
            )
            row.update(raster_metrics)
        rows.append(row)
        _log_wandb_raster(wandb_run, row)

    metrics_by_dataset = _summarize_raster_metrics_by_dataset(metrics_by_dataset_raster)
    for dataset_name, dataset_metrics in metrics_by_dataset.items():
        average_row: dict[str, object] = {
            "dataset_name": dataset_name,
            "raster_name": "average_over_rasters",
            "rgb_path": "",
            "height_path": "",
            "prediction_path": "",
            "weight_map_path": "",
            "has_ground_truth": True,
            "metric_resolution_enabled": config.metric_resolution.enabled,
            "metric_resolution_downsample_factor": metric_downsample_factor,
            "num_rasters_with_ground_truth": float(
                dataset_metrics["num_rasters_with_ground_truth"]
            ),
        }
        average_row.update(
            {
                key: value
                for key, value in dataset_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        rows.append(average_row)

    aggregate: dict[str, object] = {
        "by_dataset": metrics_by_dataset,
        "by_raster": metrics_by_raster,
    }
    if len(metrics_by_dataset) == 1:
        only_metrics = next(iter(metrics_by_dataset.values()))
        aggregate.update(
            {key: value for key, value in only_metrics.items() if isinstance(value, (int, float))}
        )
    summary = {
        "benchmark_name": layout.root.name,
        "checkpoint": checkpoint_metadata,
        "metric_resolution": {
            "enabled": config.metric_resolution.enabled,
            "downsample_factor": metric_downsample_factor,
        },
        "metrics": aggregate,
        "num_rasters": len(records),
        "num_rasters_with_ground_truth": len(metrics_by_raster),
    }
    _write_json(layout.metrics_dir / "summary.json", summary)
    _write_rows(layout.metrics_dir / "rasters.csv", rows)
    _write_jsonl(layout.metrics_dir / "rasters.jsonl", rows)
    _log_wandb_summary(wandb_run, aggregate)
    _write_wandb_metadata(layout.wandb_metadata, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
    return layout.root


def resolve_height_raster_records(
    config: HeightRasterBenchmarkConfig,
) -> list[HeightRasterRecord]:
    records: list[HeightRasterRecord] = [_record_from_pair(pair) for pair in config.rasters]
    if config.rgb_root_path:
        records.extend(_records_from_roots(config))
    if config.max_rasters is not None:
        records = records[: config.max_rasters]
    if not records:
        raise ValueError("No RGB rasters were resolved for raster height benchmark.")
    return records


def save_prediction_raster(
    *,
    source_path: str | Path,
    output_path: str | Path,
    array: np.ndarray,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array, dtype=np.float32)
    try:
        import rasterio

        with rasterio.open(source_path) as src:
            transform = src.transform
            if values.shape != (src.height, src.width):
                transform = transform * transform.scale(
                    src.width / values.shape[1],
                    src.height / values.shape[0],
                )
            profile = {
                "driver": "GTiff",
                "height": int(values.shape[0]),
                "width": int(values.shape[1]),
                "count": 1,
                "dtype": "float32",
                "crs": src.crs,
                "transform": transform,
                "nodata": None,
                "compress": "deflate",
            }
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(values[np.newaxis, :, :])
    except Exception:
        Image.fromarray(values, mode="F").save(output_path)


def _record_from_pair(pair: HeightRasterPairConfig) -> HeightRasterRecord:
    rgb_path = Path(pair.rgb_path)
    height_path = Path(pair.height_path) if pair.height_path else None
    name = pair.name or slugify(rgb_path.stem)
    dataset_name = pair.dataset_name or _infer_dataset_name_from_path(rgb_path)
    return HeightRasterRecord(
        dataset_name=dataset_name,
        name=name,
        rgb_path=rgb_path,
        height_path=height_path,
    )


def _infer_dataset_name_from_relative(*, relative: Path, rgb_root: Path) -> str:
    if len(relative.parts) > 1:
        return relative.parts[0]
    return _infer_dataset_name_from_root(rgb_root)


def _infer_dataset_name_from_root(root: Path) -> str:
    if root.name in {"benchmark", "validation"} and root.parent.name:
        return root.parent.name
    return root.name or "rasters"


def _infer_dataset_name_from_path(path: Path) -> str:
    parent = path.parent
    if parent.name in {"benchmark", "validation"} and parent.parent.name:
        return parent.parent.name
    if parent.parent.name in {"benchmark", "validation"} and parent.parent.parent.name:
        return parent.parent.parent.name
    return parent.name or "rasters"


def _records_from_roots(config: HeightRasterBenchmarkConfig) -> list[HeightRasterRecord]:
    rgb_root = Path(config.rgb_root_path or "")
    if not rgb_root.exists():
        raise FileNotFoundError(f"RGB raster root does not exist: {rgb_root}")
    image_suffixes = {suffix.lower() for suffix in config.image_suffixes}
    rgb_paths = [
        path
        for path in sorted(rgb_root.glob(config.rgb_glob))
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]
    height_root = Path(config.height_root_path) if config.height_root_path else None
    height_by_stem: dict[str, Path] = {}
    height_suffixes = {suffix.lower() for suffix in config.height_suffixes}
    if height_root is not None:
        if not height_root.exists():
            raise FileNotFoundError(f"Height raster root does not exist: {height_root}")
        for path in sorted(height_root.glob(config.height_glob)):
            if path.is_file() and path.suffix.lower() in height_suffixes:
                height_by_stem.setdefault(path.stem, path)

    records = []
    for rgb_path in rgb_paths:
        height_path = (
            _matching_height_path(
                rgb_path=rgb_path,
                height_by_stem=height_by_stem,
                rgb_name_suffix=config.rgb_name_suffix,
                height_name_suffix=config.height_name_suffix,
            )
            if height_root is not None
            else None
        )
        relative = rgb_path.relative_to(rgb_root)
        dataset_name = _infer_dataset_name_from_relative(
            relative=relative,
            rgb_root=rgb_root,
        )
        record_stem = _stem_without_suffix(relative.stem, config.rgb_name_suffix)
        record_stem = record_stem if record_stem is not None else relative.stem
        record_relative = relative.with_name(record_stem).with_suffix("")
        name = slugify(str(record_relative).replace("/", "_"))
        records.append(
            HeightRasterRecord(
                dataset_name=dataset_name,
                name=name,
                rgb_path=rgb_path,
                height_path=height_path,
            )
        )
    if height_root is not None and records:
        missing = [record for record in records if record.height_path is None]
        if missing:
            examples = ", ".join(record.rgb_path.name for record in missing[:5])
            raise ValueError(
                "height_root_path was provided, but some RGB rasters did not match "
                f"a full-raster height target by unique prefix: {examples}"
            )
    return records


def _matching_height_path(
    *,
    rgb_path: Path,
    height_by_stem: dict[str, Path],
    rgb_name_suffix: str,
    height_name_suffix: str,
) -> Path | None:
    rgb_name_suffix = rgb_name_suffix or ""
    height_name_suffix = height_name_suffix or ""
    shared = _stem_without_suffix(rgb_path.stem, rgb_name_suffix)
    if shared is None:
        return None
    mapped_match = height_by_stem.get(f"{shared}{height_name_suffix}")
    if mapped_match is not None and mapped_match != rgb_path:
        return mapped_match
    return None


def _stem_without_suffix(stem: str, suffix: str) -> str | None:
    if not suffix or not stem.endswith(suffix):
        return None
    shared = stem[: -len(suffix)]
    return shared or None


def _summarize_raster_metrics_by_dataset(
    metrics_by_dataset_raster: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for dataset_name, dataset_raster_metrics in sorted(metrics_by_dataset_raster.items()):
        dataset_metrics = macro_average_height_metrics(dataset_raster_metrics)
        summaries[dataset_name] = {
            **dataset_metrics,
            "num_rasters_with_ground_truth": len(dataset_raster_metrics),
            "by_raster": dataset_raster_metrics,
        }
    return summaries


def _load_rgb_raster(rgb_path: Path, training_config) -> tuple[Image.Image, np.ndarray]:
    Image.MAX_IMAGE_PIXELS = None
    image, valid_mask = load_image_with_valid_mask(rgb_path, training_config.valid_mask)
    return image.convert("RGB"), valid_mask


def _sliding_window_predict_height(
    *,
    model: torch.nn.Module,
    image: Image.Image,
    valid_mask: np.ndarray,
    training_config,
    benchmark_config: HeightRasterBenchmarkConfig,
    device: torch.device,
) -> SlidingWindowPrediction:
    width, height = image.size
    tile_size = int(benchmark_config.tile_size)
    stride = int(benchmark_config.stride)
    prediction_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    base_weight = _distance_weight_map(tile_size, tile_size)
    x_starts = _window_starts(width, tile_size, stride)
    y_starts = _window_starts(height, tile_size, stride)
    batch_tensors: list[torch.Tensor] = []
    batch_metadata: list[dict] = []
    num_windows = 0

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_metadata
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            output = model(batch)["height"].clamp_min(0.0)
            if output.shape[-2:] != (tile_size, tile_size):
                output = F.interpolate(
                    output,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
        output_np = output.detach().cpu().numpy()
        for sample_index, metadata in enumerate(batch_metadata):
            x = metadata["x"]
            y = metadata["y"]
            crop_height = metadata["height"]
            crop_width = metadata["width"]
            crop_mask = metadata["valid_mask"]
            prediction = output_np[sample_index, 0, :crop_height, :crop_width]
            weights = base_weight[:crop_height, :crop_width] * crop_mask.astype(np.float64)
            prediction_sum[y : y + crop_height, x : x + crop_width] += prediction * weights
            weight_sum[y : y + crop_height, x : x + crop_width] += weights
        batch_tensors = []
        batch_metadata = []

    for y in y_starts:
        for x in x_starts:
            right = min(width, x + tile_size)
            bottom = min(height, y + tile_size)
            crop_width = right - x
            crop_height = bottom - y
            crop_mask = np.ascontiguousarray(valid_mask[y:bottom, x:right], dtype=bool)
            if not bool(crop_mask.any()):
                continue
            crop = image.crop((x, y, right, bottom))
            if crop_width != tile_size or crop_height != tile_size:
                crop, tensor_mask = _pad_window(
                    crop,
                    crop_mask,
                    tile_size=tile_size,
                    image_mean=training_config.augmentation.image_mean,
                )
            else:
                tensor_mask = crop_mask
            batch_tensors.append(
                image_to_normalized_tensor(
                    crop,
                    tensor_mask,
                    training_config.augmentation.image_mean,
                    training_config.augmentation.image_std,
                )
            )
            batch_metadata.append(
                {
                    "x": x,
                    "y": y,
                    "height": crop_height,
                    "width": crop_width,
                    "valid_mask": crop_mask,
                }
            )
            num_windows += 1
            if len(batch_tensors) >= int(benchmark_config.batch_size):
                flush_batch()
    flush_batch()

    prediction = np.divide(
        prediction_sum,
        weight_sum,
        out=np.zeros_like(prediction_sum, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    return SlidingWindowPrediction(
        prediction=prediction.astype(np.float32, copy=False),
        weight_sum=weight_sum.astype(np.float32, copy=False),
        num_windows=num_windows,
    )


def _pad_window(
    image: Image.Image,
    valid_mask: np.ndarray,
    *,
    tile_size: int,
    image_mean: list[float],
) -> tuple[Image.Image, np.ndarray]:
    fill = tuple(int(round(float(value) * 255.0)) for value in image_mean)
    canvas = Image.new("RGB", (tile_size, tile_size), fill)
    canvas.paste(image.convert("RGB"), (0, 0))
    padded_mask = np.zeros((tile_size, tile_size), dtype=bool)
    padded_mask[: valid_mask.shape[0], : valid_mask.shape[1]] = valid_mask
    return canvas, padded_mask


def _distance_weight_map(height: int, width: int) -> np.ndarray:
    min_edge_weight = 0.05
    if height <= 1:
        y_weight = np.ones((height,), dtype=np.float64)
    else:
        y_weight = 1.0 - np.abs(np.linspace(-1.0, 1.0, height, dtype=np.float64))
    if width <= 1:
        x_weight = np.ones((width,), dtype=np.float64)
    else:
        x_weight = 1.0 - np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float64))
    y_weight = np.clip(y_weight, min_edge_weight, 1.0)
    x_weight = np.clip(x_weight, min_edge_weight, 1.0)
    return np.clip(y_weight[:, None] * x_weight[None, :], min_edge_weight, 1.0)


def _window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _compute_raster_metrics(
    *,
    prediction: np.ndarray,
    prediction_valid_mask: np.ndarray,
    height_path: Path,
    target_size: tuple[int, int],
    training_config,
    benchmark_config: HeightRasterBenchmarkConfig,
) -> dict[str, float]:
    target, target_valid_mask = load_height_map(
        height_path,
        min_valid_height_m=training_config.min_valid_height_m,
        max_valid_height_m=training_config.max_valid_height_m,
    )
    if target.shape != (target_size[1], target_size[0]):
        target, target_valid_mask = _resize_height_and_mask(
            target,
            target_valid_mask,
            size=target_size,
        )
    valid_mask = np.ascontiguousarray(target_valid_mask & prediction_valid_mask, dtype=bool)
    prediction_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
    metric_prediction, metric_mask = apply_height_metric_resolution(
        prediction_tensor,
        torch.from_numpy(valid_mask).unsqueeze(0),
        benchmark_config.metric_resolution,
    )
    accumulator = HeightMetricsAccumulator(training_config.validation.canopy_thresholds_m)
    accumulator.update(
        metric_prediction,
        torch.from_numpy(target).unsqueeze(0).unsqueeze(0),
        metric_mask,
    )
    return accumulator.compute()


def _resize_height_and_mask(
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            Image.fromarray(np.asarray(height, dtype=np.float32), mode="F").resize(
                size,
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ),
        np.asarray(
            Image.fromarray(valid_mask.astype(np.uint8) * 255).resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
        > 0,
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _training_run_name(
    config: HeightRasterBenchmarkConfig,
    checkpoint_metadata: dict,
) -> str:
    return source_run_name(
        config.resolved_checkpoint_path(),
        checkpoint_metadata.get("run_name"),
    )


def _init_wandb(
    config: HeightRasterBenchmarkConfig,
    output_dir: Path,
    *,
    training_run_name: str,
    checkpoint_metadata: dict,
):
    if not config.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        message = (
            "raster benchmark wandb.enabled is true, but wandb is not installed. "
            "Install wandb or set benchmark wandb.enabled: false."
        )
        if config.wandb.required:
            raise RuntimeError(message)
        print(f"{message} Continuing without wandb logging.")
        return None
    try:
        wandb_config = config.to_dict()
        wandb_config["training_run_name"] = training_run_name
        wandb_config["checkpoint_metadata"] = checkpoint_metadata
        return wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=training_run_name,
            tags=config.wandb.tags,
            mode=config.wandb.mode,
            config=wandb_config,
            dir=str(output_dir),
            job_type="height_raster_eval",
        )
    except Exception as exc:
        message = f"raster benchmark wandb initialization failed: {exc}"
        if config.wandb.required:
            raise RuntimeError(message) from exc
        print(f"{message} Continuing without wandb logging.")
        return None


def _log_wandb_raster(wandb_run, row: dict) -> None:
    if wandb_run is None or row.get("has_ground_truth") is not True:
        return
    payload = {
        f"raster/{key}": float(value)
        for key, value in row.items()
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    }
    if payload:
        wandb_run.log(payload)


def _log_wandb_summary(wandb_run, metrics: dict) -> None:
    if wandb_run is None:
        return
    payload = {}
    by_dataset = metrics.get("by_dataset")
    if isinstance(by_dataset, dict):
        for dataset_name, dataset_metrics in by_dataset.items():
            if not isinstance(dataset_metrics, dict):
                continue
            for key, value in dataset_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    payload[f"raster_benchmark/{dataset_name}/{key}"] = float(value)
    for key, value in metrics.items():
        if key in {"by_dataset", "by_raster"}:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[f"raster_benchmark/{key}"] = float(value)
    if payload:
        wandb_run.log(payload)
        wandb_run.summary.update(payload)


def _write_wandb_metadata(path: Path, wandb_run) -> None:
    payload = {
        "enabled": wandb_run is not None,
        "run_id": getattr(wandb_run, "id", None),
        "run_name": getattr(wandb_run, "name", None),
    }
    _write_json(path, payload)
