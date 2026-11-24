from __future__ import annotations

import hashlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from synthforest.training.config import HeightMetricResolutionConfig, HeightTrainingConfig
from synthforest.training.dataloader import dataloader_worker_kwargs
from synthforest.training.height.checkpoint import load_height_checkpoint_model
from synthforest.training.height.dataset import HeightTileDataset, collate_height_batch
from synthforest.training.height.metric_resolution import apply_height_metric_resolution
from synthforest.training.height.metrics import (
    HeightMetricsAccumulator,
    macro_average_height_metrics,
)


def build_height_train_loader(config: HeightTrainingConfig) -> DataLoader:
    dataset: Dataset = HeightTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )
    dataset = limit_height_to_debug_batches(dataset, config=config, label="train")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.debug_overfit_batches <= 0,
        collate_fn=partial(
            collate_height_batch,
            patch_size_multiple=config.dinov3_patch_size,
        ),
        **dataloader_worker_kwargs(
            config.dataloader_num_workers,
            avoid_fork=config.wandb.enabled,
        ),
    )


def build_synthetic_height_validation_loader(config: HeightTrainingConfig) -> DataLoader:
    if config.debug_eval_on_train:
        root = config.data_root_path
        names = config.train_dataset_names
        fold = "train"
    else:
        root = config.resolved_synthetic_validation_root()
        names = config.resolved_synthetic_validation_names()
        fold = config.validation.synthetic.fold
    dataset: Dataset = HeightTileDataset(
        data_root_path=root,
        dataset_names=names,
        fold=fold,
        config=config,
        train=False,
    )
    dataset = limit_height_to_debug_batches(dataset, config=config, label=f"{fold}-synthetic-eval")
    return build_height_eval_loader(dataset, config)


def build_synthetic_height_fold_loader(
    config: HeightTrainingConfig,
    fold: str,
) -> DataLoader:
    if fold == "train":
        root = config.data_root_path
        names = config.train_dataset_names
    elif fold == "test":
        root = config.data_root_path
        names = config.test_dataset_names
    else:
        root = config.resolved_synthetic_validation_root()
        names = config.resolved_synthetic_validation_names()
    if not names:
        raise ValueError(f"No synthetic height datasets are configured for fold {fold!r}.")
    dataset = HeightTileDataset(
        data_root_path=root,
        dataset_names=names,
        fold=fold,
        config=config,
        train=False,
    )
    return build_height_eval_loader(dataset, config, avoid_fork=False)


def build_real_height_validation_loaders(
    config: HeightTrainingConfig,
) -> dict[str, DataLoader]:
    real = config.validation.real
    if not real.enabled:
        return {}
    loaders = {}
    for dataset_name in real.dataset_names:
        dataset = HeightTileDataset(
            data_root_path=real.data_root_path,
            dataset_names=[dataset_name],
            fold=real.fold,
            config=config,
            train=False,
            max_products_per_dataset=real.max_products_per_dataset,
            max_tiles_per_product=real.max_tiles_per_product,
        )
        loaders[dataset_name] = build_height_eval_loader(dataset, config)
    return loaders


def build_height_eval_loader(
    dataset: Dataset,
    config: HeightTrainingConfig,
    *,
    avoid_fork: bool | None = None,
) -> DataLoader:
    batch_size = (
        int(config.validation.batch_size)
        if config.validation.batch_size is not None
        else config.batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_height_batch,
            patch_size_multiple=config.dinov3_patch_size,
        ),
        **dataloader_worker_kwargs(
            config.dataloader_num_workers,
            avoid_fork=(config.wandb.enabled if avoid_fork is None else avoid_fork),
        ),
    )


def evaluate_height_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: HeightTrainingConfig,
    max_batches: Optional[int] = None,
    metric_resolution: Optional[HeightMetricResolutionConfig] = None,
) -> dict[str, float]:
    model.eval()
    accumulator = HeightMetricsAccumulator(config.validation.canopy_thresholds_m)
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["images"].to(device)
            heights = batch["heights"].to(device)
            valid_masks = batch["valid_masks"].to(device)
            outputs = model(images)
            predictions = outputs["height"]
            if metric_resolution is not None:
                predictions, valid_masks = apply_height_metric_resolution(
                    predictions,
                    valid_masks,
                    metric_resolution,
                )
            accumulator.update(predictions, heights, valid_masks)
    return accumulator.compute()


def evaluate_height_validation_domains(
    *,
    model: torch.nn.Module,
    synthetic_loader: DataLoader,
    real_loaders: dict[str, DataLoader],
    device: torch.device,
    config: HeightTrainingConfig,
    evaluation_iteration: Optional[int] = None,
) -> dict:
    synthetic_metrics = evaluate_height_model(model, synthetic_loader, device, config)
    real_metrics_for_average = {}
    real_by_dataset = {}
    for dataset_name, loader in real_loaders.items():
        eval_loader, sampling_metadata = _sample_real_validation_loader(
            loader,
            dataset_name=dataset_name,
            config=config,
            evaluation_iteration=evaluation_iteration,
        )
        dataset_metrics = evaluate_height_model(
            model,
            eval_loader,
            device,
            config,
            metric_resolution=config.validation.real.metric_resolution,
        )
        real_metrics_for_average[dataset_name] = dataset_metrics
        real_by_dataset[dataset_name] = {
            **dataset_metrics,
            **sampling_metadata,
        }
    real_metrics = (
        macro_average_height_metrics(real_metrics_for_average) if real_metrics_for_average else {}
    )
    return {
        "valid_synthetic": synthetic_metrics,
        "valid_real": {
            **real_metrics,
            "by_dataset": real_by_dataset,
        },
    }


def evaluate_height_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: Optional[HeightTrainingConfig] = None,
    config_path: Optional[str | Path] = None,
    validation_set: str = "both",
    fold: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
) -> dict:
    if validation_set not in {"synthetic", "real", "both"}:
        raise ValueError("validation_set must be synthetic, real, or both.")
    if fold is not None and fold not in {"train", "valid", "test"}:
        raise ValueError("fold must be train, valid, or test.")
    if validation_set in {"real", "both"} and fold not in {None, "valid"}:
        raise ValueError("Real height evaluation through this CLI uses the registered valid fold.")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, resolved_config, _ = load_height_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        config=config,
        config_path=config_path,
    )
    metrics: dict = {}
    if validation_set in {"synthetic", "both"}:
        if fold in {None, "valid"}:
            loader = build_synthetic_height_validation_loader(resolved_config)
        else:
            loader = build_synthetic_height_fold_loader(resolved_config, fold)
        metrics["valid_synthetic"] = evaluate_height_model(
            model,
            loader,
            resolved_device,
            resolved_config,
        )
    if validation_set in {"real", "both"}:
        real_loaders = build_real_height_validation_loaders(resolved_config)
        real_metrics_for_average = {}
        real_by_dataset = {}
        for dataset_name, loader in real_loaders.items():
            eval_loader, sampling_metadata = _sample_real_validation_loader(
                loader,
                dataset_name=dataset_name,
                config=resolved_config,
                evaluation_iteration=0,
            )
            dataset_metrics = evaluate_height_model(
                model,
                eval_loader,
                resolved_device,
                resolved_config,
                metric_resolution=resolved_config.validation.real.metric_resolution,
            )
            real_metrics_for_average[dataset_name] = dataset_metrics
            real_by_dataset[dataset_name] = {
                **dataset_metrics,
                **sampling_metadata,
            }
        metrics["valid_real"] = {
            **macro_average_height_metrics(real_metrics_for_average),
            "by_dataset": real_by_dataset,
        }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def limit_height_to_debug_batches(
    dataset: Dataset,
    *,
    config: HeightTrainingConfig,
    label: str,
) -> Dataset:
    if config.debug_overfit_batches <= 0:
        return dataset
    max_samples = config.debug_overfit_batches * config.batch_size
    sample_count = min(len(dataset), max_samples)
    if sample_count <= 0:
        raise ValueError(f"Cannot build debug {label} subset from an empty dataset.")
    print(f"Debug subset for {label}: keeping {sample_count}/{len(dataset)} sample(s).")
    return Subset(dataset, list(range(sample_count)))


def _sample_real_validation_loader(
    loader: DataLoader,
    *,
    dataset_name: str,
    config: HeightTrainingConfig,
    evaluation_iteration: Optional[int],
) -> tuple[DataLoader, dict]:
    sample_count = config.validation.real.sample_scenes_per_eval
    indexed_records = _indexed_height_records(loader.dataset)
    scene_names = sorted({record.product_name for _, record in indexed_records})
    metadata = {
        "num_available_scenes": float(len(scene_names)),
        "num_sampled_scenes": float(len(scene_names)),
        "sampled_scenes": scene_names,
    }
    if sample_count is None or int(sample_count) >= len(scene_names):
        return loader, metadata

    seed = _stable_scene_sample_seed(
        config.validation.real.sample_seed
        if config.validation.real.sample_seed is not None
        else config.seed,
        dataset_name,
        0 if evaluation_iteration is None else int(evaluation_iteration),
    )
    rng = random.Random(seed)
    selected_scenes = sorted(rng.sample(scene_names, int(sample_count)))
    selected_scene_set = set(selected_scenes)
    selected_indices = [
        index for index, record in indexed_records if record.product_name in selected_scene_set
    ]
    if not selected_indices:
        raise ValueError(
            f"Scene sampling selected no validation tiles for real dataset {dataset_name!r}."
        )
    metadata = {
        "num_available_scenes": float(len(scene_names)),
        "num_sampled_scenes": float(len(selected_scenes)),
        "sampled_scenes": selected_scenes,
    }
    return build_height_eval_loader(Subset(loader.dataset, selected_indices), config), metadata


def _indexed_height_records(dataset: Dataset) -> list[tuple[int, object]]:
    return [(index, _height_record_for_index(dataset, index)) for index in range(len(dataset))]


def _height_record_for_index(dataset: Dataset, index: int):
    if hasattr(dataset, "records"):
        return dataset.records[int(index)]
    if isinstance(dataset, Subset):
        return _height_record_for_index(dataset.dataset, int(dataset.indices[int(index)]))
    raise TypeError("Real height scene sampling requires a HeightTileDataset or Subset thereof.")


def _stable_scene_sample_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
