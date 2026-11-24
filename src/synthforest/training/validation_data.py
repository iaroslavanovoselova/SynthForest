from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, Subset

from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.dataloader import dataloader_worker_kwargs
from synthforest.training.dataset import (
    CocoTileDataset,
    CocoTileSource,
    collate_detection_batch,
)
from synthforest.training.real_datasets import get_dataset_spec


def build_validation_loaders(
    config: DetectorTrainingConfig,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    return (
        build_synthetic_validation_loader(config),
        build_real_validation_loaders(config),
    )


def build_synthetic_validation_loader(
    config: DetectorTrainingConfig,
) -> DataLoader:
    if config.debug_eval_on_train:
        synthetic_root = config.data_root_path
        synthetic_names = config.train_dataset_names
        synthetic_fold = "train"
    else:
        synthetic_root = config.resolved_synthetic_validation_root()
        synthetic_names = config.resolved_synthetic_validation_names()
        synthetic_fold = config.validation.synthetic.fold

    _ensure_synthetic_fold_height_maps(config, synthetic_fold)
    dataset: Dataset = CocoTileDataset(
        data_root_path=synthetic_root,
        dataset_names=synthetic_names,
        fold=synthetic_fold,
        config=config,
        train=False,
    )
    dataset = limit_to_debug_batches(
        dataset,
        config=config,
        label=f"{synthetic_fold}-synthetic-eval",
    )
    if config.debug_eval_on_train:
        print("Debug evaluation mode: synthetic validation metrics are computed on the train fold.")
    return build_eval_loader(dataset, config)


def build_synthetic_fold_loader(
    config: DetectorTrainingConfig,
    fold: str,
) -> DataLoader:
    if fold == "train":
        data_root_path = config.data_root_path
        dataset_names = config.train_dataset_names
    elif fold == "test":
        data_root_path = config.data_root_path
        dataset_names = config.test_dataset_names
    else:
        data_root_path = config.resolved_synthetic_validation_root()
        dataset_names = config.resolved_synthetic_validation_names()
    if not dataset_names:
        raise ValueError(f"No synthetic datasets are configured for fold {fold!r}.")
    _ensure_synthetic_fold_height_maps(config, fold)
    dataset = CocoTileDataset(
        data_root_path=data_root_path,
        dataset_names=dataset_names,
        fold=fold,
        config=config,
        train=False,
    )
    return build_eval_loader(dataset, config, avoid_fork=False)


def build_real_validation_loaders(
    config: DetectorTrainingConfig,
) -> dict[str, DataLoader]:
    real_config = config.validation.real
    if not real_config.enabled:
        return {}
    loaders: dict[str, DataLoader] = {}
    for dataset_name in real_config.dataset_names:
        dataset_spec = get_dataset_spec(dataset_name)
        dataset_spec.verify_fold(
            root_output_path=real_config.data_root_path,
            fold=real_config.fold,
            include_raster_inputs=False,
        )
        products = list(
            dataset_spec.iter_fold(
                root_output_path=real_config.data_root_path,
                fold=real_config.fold,
                include_raster_inputs=False,
            )
        )
        if real_config.max_products_per_dataset is not None:
            products = products[: real_config.max_products_per_dataset]
        sources = [
            CocoTileSource(
                name=f"{dataset_name}/{product.product_name}",
                tile_dir=product.tile_dir,
                coco_path=product.truth_coco_path,
            )
            for product in products
        ]
        _ensure_coco_source_height_maps(
            config,
            sources,
            fold=real_config.fold,
            source_gsd=(
                dataset_spec.ground_resolution
                if dataset_spec.ground_resolution is not None
                else config.height_fusion.source_gsd_m
            ),
        )
        dataset = CocoTileDataset(
            data_root_path=real_config.data_root_path,
            dataset_names=[],
            fold=real_config.fold,
            config=config,
            train=False,
            coco_sources=sources,
            max_tiles_per_source=real_config.max_tiles_per_product,
            drop_edge_tiles=real_config.drop_edge_tiles,
            edge_tile_black_pixel_threshold=(real_config.edge_tile_black_pixel_threshold),
            edge_tile_black_pixel_tolerance=(real_config.edge_tile_black_pixel_tolerance),
        )
        loaders[dataset_name] = build_eval_loader(dataset, config)
    return loaders


def _ensure_synthetic_fold_height_maps(
    config: DetectorTrainingConfig,
    fold: str,
) -> None:
    fusion = config.height_fusion
    if not fusion.enabled or not fusion.generate_missing:
        return
    from synthforest.training.height.detector_precompute import (
        resolve_detector_height_tile_records,
        run_detector_height_precompute,
    )

    records = resolve_detector_height_tile_records(config, splits=[fold])
    if not records:
        return
    print(
        "Height fusion precompute: checking/generating "
        f"{len(records)} synthetic {fold} height map(s)."
    )
    run_detector_height_precompute(
        detector_config=config,
        height_checkpoint_path=fusion.height_checkpoint_path,
        height_train_config_path=fusion.height_train_config_path,
        splits=[fold],
        batch_size=max(1, int(config.validation.batch_size or config.batch_size)),
        source_gsd=float(fusion.source_gsd_m),
        target_gsd=float(fusion.target_gsd_m),
    )


def _ensure_coco_source_height_maps(
    config: DetectorTrainingConfig,
    sources: list[CocoTileSource],
    *,
    fold: str,
    source_gsd: float,
) -> None:
    fusion = config.height_fusion
    if not fusion.enabled or not fusion.generate_missing:
        return
    from synthforest.training.height.detector_precompute import (
        resolve_detector_height_tile_records_from_coco,
        run_detector_height_precompute_for_coco,
    )

    for source in sources:
        records = resolve_detector_height_tile_records_from_coco(
            tile_dir=source.tile_dir,
            truth_coco_path=source.coco_path,
            split=fold,
            height_maps_subdir=fusion.height_maps_subdir,
        )
        if not records:
            continue
        print(
            "Height fusion precompute: checking/generating "
            f"{len(records)} height map(s) for {source.name}."
        )
        run_detector_height_precompute_for_coco(
            tile_dir=source.tile_dir,
            truth_coco_path=source.coco_path,
            split=fold,
            height_maps_subdir=fusion.height_maps_subdir,
            height_checkpoint_path=fusion.height_checkpoint_path,
            height_train_config_path=fusion.height_train_config_path,
            batch_size=max(1, int(config.validation.batch_size or config.batch_size)),
            source_gsd=float(source_gsd),
            target_gsd=float(fusion.target_gsd_m),
            precompute_match_checkpoint=bool(fusion.precompute_match_checkpoint),
        )


def build_eval_loader(
    dataset: Dataset,
    config: DetectorTrainingConfig,
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
        collate_fn=collate_detection_batch,
        **dataloader_worker_kwargs(
            config.dataloader_num_workers,
            avoid_fork=(config.wandb.enabled if avoid_fork is None else avoid_fork),
        ),
    )


def limit_to_debug_batches(
    dataset: Dataset,
    *,
    config: DetectorTrainingConfig,
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
