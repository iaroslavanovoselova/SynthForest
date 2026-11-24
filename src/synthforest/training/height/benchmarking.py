from __future__ import annotations

import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from synthforest.training.config import HeightBenchmarkConfig
from synthforest.training.height.checkpoint import load_height_checkpoint_model
from synthforest.training.height.dataset import HeightTileDataset, collate_height_batch
from synthforest.training.height.metric_resolution import apply_height_metric_resolution
from synthforest.training.height.metrics import (
    HeightMetricsAccumulator,
    macro_average_height_metrics,
)
from synthforest.training.layout import BenchmarkRunLayout, source_run_name
from synthforest.training.utils import set_all_seeds


def run_height_benchmark(config: HeightBenchmarkConfig) -> Path:
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

    training_run_name = _training_run_name(config, checkpoint_metadata)
    wandb_run = _init_wandb(
        config,
        layout.root,
        training_run_name=training_run_name,
        checkpoint_metadata=checkpoint_metadata,
    )
    rows: list[dict] = []
    metrics_by_dataset: dict[str, dict[str, float]] = {}
    metric_downsample_factor = config.metric_resolution.resolved_downsample_factor()
    for dataset_name in config.dataset_names:
        dataset = HeightTileDataset(
            data_root_path=config.data_root_path,
            dataset_names=[dataset_name],
            fold=config.fold_name,
            config=training_config,
            train=False,
            max_products_per_dataset=config.max_products,
            max_tiles_per_product=config.max_tiles,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_height_batch,
                patch_size_multiple=training_config.dinov3_patch_size,
            ),
            num_workers=config.num_workers,
        )
        dataset_accumulator = HeightMetricsAccumulator(
            training_config.validation.canopy_thresholds_m
        )
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device)
                heights = batch["heights"].to(device)
                valid_masks = batch["valid_masks"].to(device)
                predictions = model(images)["height"].clamp_min(0.0)
                metric_predictions, metric_masks = apply_height_metric_resolution(
                    predictions,
                    valid_masks,
                    config.metric_resolution,
                )
                dataset_accumulator.update(metric_predictions, heights, metric_masks)
                for sample_index, metadata in enumerate(batch["metadata"]):
                    sample_native_prediction = predictions[sample_index : sample_index + 1]
                    sample_metric_prediction = metric_predictions[sample_index : sample_index + 1]
                    sample_height = heights[sample_index : sample_index + 1]
                    sample_metric_mask = metric_masks[sample_index : sample_index + 1]
                    sample_accumulator = HeightMetricsAccumulator(
                        training_config.validation.canopy_thresholds_m
                    )
                    sample_accumulator.update(
                        sample_metric_prediction,
                        sample_height,
                        sample_metric_mask,
                    )
                    sample_metrics = sample_accumulator.compute()
                    row = {
                        "row_type": "tile",
                        "dataset_name": metadata["dataset_name"],
                        "product_name": metadata["product_name"],
                        "tile_key": metadata["tile_key"],
                        "image_path": metadata["image_path"],
                        "height_path": metadata["height_path"],
                        "metric_resolution_enabled": config.metric_resolution.enabled,
                        "metric_resolution_downsample_factor": metric_downsample_factor,
                        **sample_metrics,
                    }
                    rows.append(row)
                    if config.save_predictions:
                        output_path = (
                            layout.fold_predictions(config.fold_name)
                            / metadata["dataset_name"]
                            / metadata["product_name"]
                            / f"{metadata['tile_key']}.tif"
                        )
                        save_prediction_tile(
                            source_path=metadata["height_path"],
                            output_path=output_path,
                            prediction=sample_native_prediction.squeeze(0)
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy(),
                        )
        dataset_metrics = dataset_accumulator.compute()
        metrics_by_dataset[dataset_name] = dataset_metrics
        rows.append(
            {
                "row_type": "dataset_summary",
                "dataset_name": dataset_name,
                "product_name": "average_over_tiles",
                "tile_key": "",
                "image_path": "",
                "height_path": "",
                "metric_resolution_enabled": config.metric_resolution.enabled,
                "metric_resolution_downsample_factor": metric_downsample_factor,
                **dataset_metrics,
            }
        )

    aggregate: dict[str, object] = {"by_dataset": metrics_by_dataset}
    if len(metrics_by_dataset) == 1:
        aggregate.update(next(iter(metrics_by_dataset.values())))
    else:
        aggregate["macro_average"] = macro_average_height_metrics(metrics_by_dataset)
    summary = {
        "benchmark_name": layout.root.name,
        "checkpoint": checkpoint_metadata,
        "metric_resolution": {
            "enabled": config.metric_resolution.enabled,
            "downsample_factor": metric_downsample_factor,
        },
        "metrics": aggregate,
        "num_tiles": sum(row.get("row_type") == "tile" for row in rows),
        "num_summary_rows": sum(row.get("row_type") == "dataset_summary" for row in rows),
    }
    _write_json(layout.metrics_dir / "summary.json", summary)
    _write_rows(layout.metrics_dir / "tiles.csv", rows)
    _write_jsonl(layout.metrics_dir / "tiles.jsonl", rows)
    _log_wandb_summary(wandb_run, aggregate)
    _write_wandb_metadata(layout.wandb_metadata, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
    return layout.root


def save_prediction_tile(
    *,
    source_path: str | Path,
    output_path: str | Path,
    prediction: np.ndarray,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(prediction, dtype=np.float32)
    try:
        import rasterio

        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            if array.shape != (src.height, src.width):
                transform = transform * transform.scale(
                    src.width / array.shape[1],
                    src.height / array.shape[0],
                )
            profile.update(
                count=1,
                dtype="float32",
                height=int(array.shape[0]),
                width=int(array.shape[1]),
                transform=transform,
                nodata=None,
            )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array[np.newaxis, :, :])
    except Exception:
        from PIL import Image

        Image.fromarray(array, mode="F").save(output_path)


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
    config: HeightBenchmarkConfig,
    checkpoint_metadata: dict,
) -> str:
    return source_run_name(
        config.resolved_checkpoint_path(),
        checkpoint_metadata.get("run_name"),
    )


def _init_wandb(
    config: HeightBenchmarkConfig,
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
            "benchmark wandb.enabled is true, but wandb is not installed. "
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
            job_type="height_eval",
        )
    except Exception as exc:
        message = f"benchmark wandb initialization failed: {exc}"
        if config.wandb.required:
            raise RuntimeError(message) from exc
        print(f"{message} Continuing without wandb logging.")
        return None


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
                if isinstance(value, (int, float)):
                    payload[f"benchmark/{dataset_name}/{key}"] = float(value)
    for key, value in metrics.items():
        if key in {"by_dataset", "macro_average"}:
            continue
        if isinstance(by_dataset, dict) and len(by_dataset) > 1:
            continue
        if isinstance(value, (int, float)):
            payload[f"benchmark/{key}"] = float(value)
    wandb_run.log(payload)
    wandb_run.summary.update(payload)


def _write_wandb_metadata(path: Path, wandb_run) -> None:
    payload = {
        "enabled": wandb_run is not None,
        "run_id": getattr(wandb_run, "id", None),
        "run_name": getattr(wandb_run, "name", None),
    }
    _write_json(path, payload)


def _merged_wandb_tags(
    *,
    training_tags: list[str],
    benchmark_tags: list[str],
) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in [*training_tags, *benchmark_tags]:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags
