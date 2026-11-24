from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from synthforest.training.branch_checkpoint import (
    load_branch_checkpoint,
    save_branch_checkpoint,
)
from synthforest.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint_weights_into_model,
)
from synthforest.training.config import (
    DetectorTrainingConfig,
    TrainingStageConfig,
    _slugify,
)
from synthforest.training.dataloader import dataloader_worker_kwargs
from synthforest.training.dataset import (
    CocoTileDataset,
    UnlabeledRealTileDataset,
    UnlabeledRealTileSource,
    collate_detection_batch,
    collate_unlabeled_real_batch,
)
from synthforest.training.engine.optimization import (
    build_stage_optimizer,
    build_stage_scheduler,
    lr_metrics_by_group,
)
from synthforest.training.evaluation import (
    evaluate_model,
    macro_average_metrics,
)
from synthforest.training.layout import TrainingRunLayout, validate_checkpoint_name
from synthforest.training.mean_teacher import (
    build_pseudo_label_targets,
    copy_ema_teacher_weights,
    freeze_teacher,
    update_ema_teacher,
)
from synthforest.training.models import (
    assert_stage_trainability,
    build_dinov3_detector,
    set_stage_trainability,
    summarize_trainable_parameters,
)
from synthforest.training.real_datasets import get_dataset_spec
from synthforest.training.utils import move_targets_to_device, set_all_seeds
from synthforest.training.validation_data import (
    build_validation_loaders,
    limit_to_debug_batches,
)


def train_from_config(
    config: DetectorTrainingConfig,
    *,
    wandb_run=None,
) -> Path:
    config.validate()
    set_all_seeds(config.seed)
    config.run_name = config.resolved_run_name()
    layout = TrainingRunLayout.resolve(
        output_root=config.train_output_path,
        model_type=config.model_type,
        run_name=config.run_name,
        sweep_phase=(config.sweep.phase if config.sweep is not None else None),
    ).create()
    output_dir = layout.root
    _clear_run_metrics(layout)
    config.save_yaml(layout.resolved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_synthetic_height_maps(config)
    train_loader = _build_train_loader(config, layout=layout)
    unlabeled_real_loader = _build_unlabeled_real_loader(config, layout=layout)
    synthetic_valid_loader, real_valid_loaders = build_validation_loaders(config)

    model = build_dinov3_detector(config).to(device)
    if config.checkpoint_initialization is not None:
        initialization_metadata = load_checkpoint_weights_into_model(
            config.checkpoint_initialization.checkpoint_path,
            model=model,
            config=config,
        )
        _write_json(layout.initialization_metadata, initialization_metadata)
    elif config.initialization is not None:
        initialization_metadata = load_branch_checkpoint(
            config.initialization.checkpoint_path,
            model=model,
            config=config,
            expected_iteration=config.initialization.completed_iterations,
            expected_fingerprint=config.initialization.architecture_fingerprint,
        )
        _write_json(layout.initialization_metadata, initialization_metadata)

    owns_wandb_run = wandb_run is None
    if wandb_run is None:
        wandb_run = _init_wandb(config, output_dir)
    run_summary = {
        "run/output_dir": str(output_dir.resolve()),
        "run/phase": (config.sweep.phase if config.sweep is not None else "training"),
    }
    if config.sweep is not None:
        run_summary["sweep/objective_metric"] = _sweep_objective_metric(config)
    _update_wandb_summary(
        wandb_run,
        {
            **run_summary,
            "run/objective_metric": (
                f"{_validation_metric_domain(config)}/{config.validation.selection_metric}"
            ),
        },
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    global_iteration = (
        int(config.initialization.completed_iterations) if config.initialization is not None else 0
    )
    data_iter = iter(train_loader)
    unlabeled_real_iter = iter(unlabeled_real_loader) if unlabeled_real_loader is not None else None
    ema_teacher: Optional[torch.nn.Module] = None
    final_stage_index = len(config.stages) - 1
    best_final_metric = -math.inf
    best_bbox_metric = -math.inf
    best_segm_metric = -math.inf
    trainability_by_stage: dict[str, dict[str, int]] = {}
    saved_branch_iterations: set[int] = set()

    for stage_index, stage in enumerate(config.stages):
        set_stage_trainability(model, train_lora=stage.train_lora)
        if stage.semi_supervised:
            if unlabeled_real_loader is None or unlabeled_real_iter is None:
                raise RuntimeError("Semi-supervised stage requires an unlabeled real loader.")
            if ema_teacher is None:
                ema_teacher = build_dinov3_detector(config).to(device)
                copy_ema_teacher_weights(student=model, teacher=ema_teacher)
                print(f"Stage {stage_index}:{stage.name} initialized EMA teacher.")
            else:
                freeze_teacher(ema_teacher)
        trainability_summary = summarize_trainable_parameters(model)
        trainability_by_stage[stage.name] = trainability_summary
        _save_trainability_summary(layout, trainability_by_stage)
        print(f"Stage {stage_index}:{stage.name} trainability: {trainability_summary}")
        _log_wandb(
            wandb_run,
            {
                f"trainability/{stage.name}/{key}": value
                for key, value in trainability_summary.items()
            },
            step=global_iteration,
        )

        optimizer = build_stage_optimizer(model, stage)
        scheduler = build_stage_scheduler(optimizer, stage)
        optimizer.zero_grad(set_to_none=True)
        stage_best_metric = -math.inf

        for stage_iteration in range(1, stage.max_iterations + 1):
            model.train()
            if stage.semi_supervised and ema_teacher is not None:
                ema_teacher.eval()
            pseudo_loss_weight = (
                _semi_supervised_loss_weight(
                    config,
                    stage=stage,
                    stage_iteration=stage_iteration,
                )
                if stage.semi_supervised
                else 0.0
            )
            losses_for_log: dict[str, float] = {}
            pseudo_losses_for_log: dict[str, float] = {}
            total_loss_for_log = 0.0
            sup_total_loss_for_log = 0.0
            pseudo_total_loss_for_log = 0.0
            pseudo_stats_for_log = {
                "pseudo_num_labels_before_filter": 0,
                "pseudo_num_labels": 0,
                "pseudo_images_used": 0,
                "pseudo_images_skipped": 0,
            }

            for _ in range(config.grad_accumulation_steps):
                try:
                    images, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, targets = next(data_iter)

                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)
                pseudo_images: list[torch.Tensor] = []
                pseudo_targets: list[dict[str, torch.Tensor]] = []
                pseudo_stats: dict[str, int] = {}
                if stage.semi_supervised:
                    if ema_teacher is None or unlabeled_real_iter is None:
                        raise RuntimeError("Semi-supervised stage has no EMA teacher.")
                    try:
                        weak_images, strong_images, real_metadata = next(unlabeled_real_iter)
                    except StopIteration:
                        unlabeled_real_iter = iter(unlabeled_real_loader)
                        weak_images, strong_images, real_metadata = next(unlabeled_real_iter)
                    weak_images = [image.to(device) for image in weak_images]
                    strong_images = [image.to(device) for image in strong_images]
                    valid_masks = [item["valid_mask"].to(device) for item in real_metadata]
                    with torch.no_grad():
                        with torch.amp.autocast(
                            "cuda",
                            enabled=config.use_amp and device.type == "cuda",
                        ):
                            teacher_outputs = ema_teacher(weak_images)
                    selected_indices, pseudo_targets, pseudo_stats = build_pseudo_label_targets(
                        teacher_outputs,
                        valid_masks,
                        config.semi_supervised.pseudo_labels,
                        require_masks=config.model_type == "maskrcnn",
                        mask_probability_threshold=config.mask_probability_threshold,
                    )
                    pseudo_images = [strong_images[index] for index in selected_indices]
                    for key, value in pseudo_stats.items():
                        pseudo_stats_for_log[key] = pseudo_stats_for_log.get(key, 0) + int(value)

                with torch.amp.autocast(
                    "cuda",
                    enabled=config.use_amp and device.type == "cuda",
                ):
                    loss_dict = model(images, targets)
                    supervised_loss = sum(loss_value for loss_value in loss_dict.values())
                    pseudo_loss = supervised_loss.new_zeros(())
                    pseudo_loss_dict: dict[str, torch.Tensor] = {}
                    if stage.semi_supervised and pseudo_targets:
                        pseudo_loss_dict = model(pseudo_images, pseudo_targets)
                        pseudo_loss = sum(loss_value for loss_value in pseudo_loss_dict.values())
                    loss = supervised_loss + pseudo_loss_weight * pseudo_loss
                    scaled_loss = loss / config.grad_accumulation_steps

                scaler.scale(scaled_loss).backward()
                total_loss_for_log += float(loss.detach().cpu())
                sup_total_loss_for_log += float(supervised_loss.detach().cpu())
                pseudo_total_loss_for_log += float(pseudo_loss.detach().cpu())
                for key, value in loss_dict.items():
                    losses_for_log[key] = losses_for_log.get(key, 0.0) + float(value.detach().cpu())
                for key, value in pseudo_loss_dict.items():
                    pseudo_losses_for_log[key] = pseudo_losses_for_log.get(key, 0.0) + float(
                        value.detach().cpu()
                    )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_iteration += 1
            if stage.semi_supervised and ema_teacher is not None:
                update_ema_teacher(
                    student=model,
                    teacher=ema_teacher,
                    decay=float(config.semi_supervised.ema_decay),
                )
            assert_stage_trainability(model, train_lora=stage.train_lora)

            if (
                global_iteration in config.branch_checkpoint_iterations
                and global_iteration not in saved_branch_iterations
            ):
                save_branch_checkpoint(
                    layout.branch_checkpoint(global_iteration),
                    model=(
                        ema_teacher if stage.semi_supervised and ema_teacher is not None else model
                    ),
                    config=config,
                    iteration=global_iteration,
                    stage=stage.name,
                    output_dir=output_dir,
                    source_run_id=getattr(wandb_run, "id", None),
                    source_run_name=getattr(wandb_run, "name", None),
                )
                saved_branch_iterations.add(global_iteration)

            if stage_iteration % stage.train_log_interval == 0 or stage_iteration == 1:
                log_data = {
                    "iteration": global_iteration,
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "stage_iteration": stage_iteration,
                    "train/total_loss": (total_loss_for_log / config.grad_accumulation_steps),
                    "train/sup_total_loss": (
                        sup_total_loss_for_log / config.grad_accumulation_steps
                    ),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if stage.semi_supervised:
                    log_data.update(
                        {
                            "train/pseudo_total_loss": (
                                pseudo_total_loss_for_log / config.grad_accumulation_steps
                            ),
                            "train/pseudo_loss_weight": float(pseudo_loss_weight),
                            "train/pseudo_loss_target_weight": float(
                                config.semi_supervised.unsupervised_loss_weight
                            ),
                            "train/pseudo_loss_start_weight": float(
                                config.semi_supervised.unsupervised_loss_start_weight
                            ),
                            "train/pseudo_loss_ramp_fraction": float(
                                config.semi_supervised.unsupervised_loss_ramp_fraction
                            ),
                            "train/pseudo_num_labels": float(
                                pseudo_stats_for_log["pseudo_num_labels"]
                            ),
                            "train/pseudo_num_labels_before_filter": float(
                                pseudo_stats_for_log["pseudo_num_labels_before_filter"]
                            ),
                            "train/pseudo_images_used": float(
                                pseudo_stats_for_log["pseudo_images_used"]
                            ),
                            "train/pseudo_images_skipped": float(
                                pseudo_stats_for_log["pseudo_images_skipped"]
                            ),
                            "ema/decay": float(config.semi_supervised.ema_decay),
                        }
                    )
                    for key, value in pseudo_stats_for_log.items():
                        if key.startswith("pseudo_height_"):
                            log_data[f"train/{key}"] = float(value)
                log_data.update(lr_metrics_by_group(optimizer))
                for key, value in losses_for_log.items():
                    log_data[f"train/{key}"] = value / config.grad_accumulation_steps
                for key, value in pseudo_losses_for_log.items():
                    log_data[f"train/pseudo_{key}"] = value / config.grad_accumulation_steps
                _append_jsonl(layout.train_metrics, log_data)
                _log_wandb(
                    wandb_run,
                    _numeric_metrics(log_data),
                    global_iteration,
                )
                print(
                    f"stage={stage.name} "
                    f"iter={global_iteration} "
                    f"stage_iter={stage_iteration} "
                    f"loss={log_data['train/total_loss']:.4f} "
                    f"lr={log_data['train/lr']:.6g}"
                )

            if stage_iteration % stage.eval_interval == 0:
                eval_model = (
                    ema_teacher if stage.semi_supervised and ema_teacher is not None else model
                )
                validation_payload = evaluate_validation_domains(
                    model=eval_model,
                    synthetic_loader=synthetic_valid_loader,
                    real_loaders=real_valid_loaders,
                    device=device,
                    config=config,
                )
                payload = {
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "stage_iteration": stage_iteration,
                    **validation_payload,
                }
                metric_value = _selection_metric_value(
                    validation_payload,
                    config.validation.selection_metric,
                    domain=_validation_metric_domain(config),
                )
                bbox_metric_value = _optional_selection_metric_value(
                    validation_payload,
                    "bbox_dense_map",
                    domain=_validation_metric_domain(config),
                )
                segm_metric_value = _optional_selection_metric_value(
                    validation_payload,
                    "segm_dense_map",
                    domain=_validation_metric_domain(config),
                )
                sweep_metric_value = (
                    _selection_metric_value(
                        validation_payload,
                        _sweep_objective_metric(config),
                        domain=_validation_metric_domain(config),
                    )
                    if config.sweep is not None
                    else metric_value
                )
                sweep_metrics = _sweep_metrics(
                    config,
                    global_iteration=global_iteration,
                    metric_value=sweep_metric_value,
                )
                payload.update(sweep_metrics)
                _append_validation_metrics(
                    layout,
                    global_iteration,
                    payload,
                )
                _log_wandb(
                    wandb_run,
                    _validation_wandb_payload(validation_payload)
                    | {
                        "stage_index": stage_index,
                        "stage_iteration": stage_iteration,
                    }
                    | sweep_metrics,
                    global_iteration,
                )

                eligible_for_best = (
                    config.sweep is None or global_iteration in config.sweep.objective_iterations
                )
                if eligible_for_best and metric_value > stage_best_metric:
                    stage_best_metric = metric_value
                    _save_checkpoint(
                        layout.stage_checkpoint(
                            _slugify(stage.name),
                            "best",
                        ),
                        model=eval_model,
                        student_model=model if stage.semi_supervised else None,
                        ema_teacher_model=ema_teacher if stage.semi_supervised else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=stage_best_metric,
                    )
                if (
                    eligible_for_best
                    and stage_index == final_stage_index
                    and metric_value > best_final_metric
                ):
                    best_final_metric = metric_value
                    _save_checkpoint(
                        layout.best_checkpoint,
                        model=eval_model,
                        student_model=model if stage.semi_supervised else None,
                        ema_teacher_model=ema_teacher if stage.semi_supervised else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=best_final_metric,
                    )
                if (
                    eligible_for_best
                    and stage_index == final_stage_index
                    and bbox_metric_value is not None
                    and bbox_metric_value > best_bbox_metric
                ):
                    best_bbox_metric = bbox_metric_value
                    _save_checkpoint(
                        layout.best_bbox_checkpoint,
                        model=eval_model,
                        student_model=model if stage.semi_supervised else None,
                        ema_teacher_model=ema_teacher if stage.semi_supervised else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=best_bbox_metric,
                        best_metric_name=(f"{_validation_metric_domain(config)}/bbox_dense_map"),
                    )
                if (
                    eligible_for_best
                    and stage_index == final_stage_index
                    and segm_metric_value is not None
                    and segm_metric_value > best_segm_metric
                ):
                    best_segm_metric = segm_metric_value
                    _save_checkpoint(
                        layout.best_segm_checkpoint,
                        model=eval_model,
                        student_model=model if stage.semi_supervised else None,
                        ema_teacher_model=ema_teacher if stage.semi_supervised else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=best_segm_metric,
                        best_metric_name=(f"{_validation_metric_domain(config)}/segm_dense_map"),
                    )
                print(
                    f"valid stage={stage.name} iter={global_iteration} "
                    f"{_validation_metric_domain(config)}_"
                    f"{config.validation.selection_metric}={metric_value:.6f}"
                )

            if stage_iteration % stage.checkpoint_interval == 0:
                _save_stage_last(
                    layout,
                    model=(
                        ema_teacher if stage.semi_supervised and ema_teacher is not None else model
                    ),
                    student_model=model if stage.semi_supervised else None,
                    ema_teacher_model=ema_teacher if stage.semi_supervised else None,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    stage=stage,
                    stage_index=stage_index,
                    stage_iteration=stage_iteration,
                    iteration=global_iteration,
                    best_metric=stage_best_metric,
                )

        _save_stage_last(
            layout,
            model=(ema_teacher if stage.semi_supervised and ema_teacher is not None else model),
            student_model=model if stage.semi_supervised else None,
            ema_teacher_model=ema_teacher if stage.semi_supervised else None,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            stage=stage,
            stage_index=stage_index,
            stage_iteration=stage.max_iterations,
            iteration=global_iteration,
            best_metric=stage_best_metric,
        )

    _update_wandb_summary(
        wandb_run,
        {
            "run/completed_global_iteration": global_iteration,
            "run/completed": True,
        },
    )
    _finish_wandb(
        wandb_run,
        layout.best_checkpoint,
        config,
        finish=owns_wandb_run,
    )
    _run_post_training_benchmark(config, layout)
    return output_dir


def _semi_supervised_loss_weight(
    config: DetectorTrainingConfig,
    *,
    stage: TrainingStageConfig,
    stage_iteration: int,
) -> float:
    semi = config.semi_supervised
    start = float(semi.unsupervised_loss_start_weight)
    target = float(semi.unsupervised_loss_weight)
    ramp_fraction = float(semi.unsupervised_loss_ramp_fraction)
    if ramp_fraction <= 0.0 or start == target:
        return target

    stage_iterations = max(1, int(stage.max_iterations))
    ramp_iterations = max(2, int(math.ceil(stage_iterations * ramp_fraction)))
    progress = (int(stage_iteration) - 1) / float(ramp_iterations - 1)
    progress = min(1.0, max(0.0, progress))
    return start + (target - start) * progress


def _run_post_training_benchmark(
    config: DetectorTrainingConfig,
    layout: TrainingRunLayout,
) -> None:
    benchmark = config.post_training_benchmark
    if not benchmark.enabled:
        return

    checkpoint_name = validate_checkpoint_name(benchmark.checkpoint_name)
    checkpoint_path = layout.checkpoints_dir / checkpoint_name
    metadata = {
        "enabled": True,
        "config_path": benchmark.config_path,
        "checkpoint_name": checkpoint_name,
        "resolved_checkpoint_path": str(checkpoint_path.resolve()),
        "run_folder": str(layout.root.resolve()),
        "status": "started",
    }
    _write_json(layout.post_training_benchmark_metadata, metadata)

    try:
        from synthforest.training.benchmarking import BenchmarkConfig, run_benchmark

        if benchmark.config_path is None:
            raise ValueError(
                "post_training_benchmark.config_path is required when "
                "post_training_benchmark.enabled is true."
            )
        benchmark_config = BenchmarkConfig.from_yaml(
            benchmark.config_path,
            validate=False,
        )
        if benchmark_config.model_type != config.model_type:
            raise ValueError(
                "Benchmark model_type does not match the training config: "
                f"benchmark={benchmark_config.model_type!r}, "
                f"training={config.model_type!r}."
            )
        benchmark_config.run_folder = str(layout.root)
        benchmark_config.checkpoint_name = checkpoint_name
        benchmark_config.checkpoint_path = None
        output_dir = run_benchmark(benchmark_config)
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(layout.post_training_benchmark_metadata, metadata)
        message = f"Post-training benchmark failed: {exc}"
        if benchmark.fail_on_error:
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}")
        return

    metadata.update(
        {
            "status": "success",
            "benchmark_output_path": str(output_dir.resolve()),
            "benchmark_name": output_dir.name,
        }
    )
    _write_json(layout.post_training_benchmark_metadata, metadata)


def _build_train_loader(
    config: DetectorTrainingConfig,
    *,
    layout: Optional[TrainingRunLayout] = None,
) -> DataLoader:
    dataset: Dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )
    dataset, sampling_metadata = _apply_train_sampling(dataset, config)
    if layout is not None:
        _write_json(layout.train_sampling_metadata, sampling_metadata)
    dataset = limit_to_debug_batches(
        dataset,
        config=config,
        label="train",
    )
    if config.debug_overfit_batches > 0:
        print(
            "Debug overfit mode: using "
            f"{len(dataset)} train sample(s) "
            f"({config.debug_overfit_batches} batch(es) at "
            f"batch_size={config.batch_size})."
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.debug_overfit_batches <= 0,
        collate_fn=collate_detection_batch,
        **dataloader_worker_kwargs(
            config.dataloader_num_workers,
            avoid_fork=config.wandb.enabled,
        ),
    )


def _build_unlabeled_real_loader(
    config: DetectorTrainingConfig,
    *,
    layout: Optional[TrainingRunLayout] = None,
) -> Optional[DataLoader]:
    if not config.semi_supervised.enabled:
        return None
    real_config = config.semi_supervised.unlabeled_real
    height_config = _active_unlabeled_height_config(config)
    sources: list[UnlabeledRealTileSource] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in real_config.dataset_names:
        dataset_spec = get_dataset_spec(dataset_name)
        products = list(
            dataset_spec.iter_fold(
                root_output_path=real_config.data_root_path,
                fold=real_config.fold,
                include_raster_inputs=False,
            )
        )
        if real_config.max_products_per_dataset is not None:
            products = products[: real_config.max_products_per_dataset]
        source_gsd = (
            float(dataset_spec.ground_resolution)
            if dataset_spec.ground_resolution is not None
            else float(
                height_config.source_gsd_m
                if height_config is not None
                else config.height_fusion.source_gsd_m
            )
        )
        dataset_sources = [
            UnlabeledRealTileSource(
                name=f"{dataset_name}/{product.product_name}",
                tile_dir=product.tile_dir,
                source_gsd_m=source_gsd,
            )
            for product in products
        ]
        sources.extend(dataset_sources)
        per_dataset[dataset_name] = {
            "products": [source.name for source in dataset_sources],
            "num_products": len(dataset_sources),
            "source_gsd_m": source_gsd,
        }
    _ensure_unlabeled_real_height_maps(
        config,
        sources,
        max_tiles_per_source=real_config.max_tiles_per_product,
        data_root_path=real_config.data_root_path,
    )
    dataset: Dataset = UnlabeledRealTileDataset(
        data_root_path=real_config.data_root_path,
        sources=sources,
        config=config,
        max_tiles_per_source=real_config.max_tiles_per_product,
        drop_edge_tiles=real_config.drop_edge_tiles,
        edge_tile_black_pixel_threshold=real_config.edge_tile_black_pixel_threshold,
        edge_tile_black_pixel_tolerance=real_config.edge_tile_black_pixel_tolerance,
    )
    dataset = limit_to_debug_batches(
        dataset,
        config=config,
        label="unlabeled-real-train",
    )
    metadata = {
        "enabled": True,
        "data_root_path": real_config.data_root_path,
        "dataset_names": list(real_config.dataset_names),
        "fold": real_config.fold,
        "total_samples": len(dataset),
        "max_tiles_per_product": real_config.max_tiles_per_product,
        "max_products_per_dataset": real_config.max_products_per_dataset,
        "drop_edge_tiles": real_config.drop_edge_tiles,
        "per_dataset": per_dataset,
        "ema_decay": config.semi_supervised.ema_decay,
        "unsupervised_loss_start_weight": (config.semi_supervised.unsupervised_loss_start_weight),
        "unsupervised_loss_weight": config.semi_supervised.unsupervised_loss_weight,
        "unsupervised_loss_ramp_fraction": (config.semi_supervised.unsupervised_loss_ramp_fraction),
        "pseudo_labels": config.semi_supervised.pseudo_labels.__dict__,
        "strong_augmentation": config.semi_supervised.strong_augmentation.__dict__,
    }
    if layout is not None:
        _write_json(layout.semi_supervised_metadata, metadata)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.debug_overfit_batches <= 0,
        collate_fn=collate_unlabeled_real_batch,
        **dataloader_worker_kwargs(
            config.dataloader_num_workers,
            avoid_fork=config.wandb.enabled,
        ),
    )


def _ensure_unlabeled_real_height_maps(
    config: DetectorTrainingConfig,
    sources: list[UnlabeledRealTileSource],
    *,
    max_tiles_per_source: Optional[int],
    data_root_path: str,
) -> None:
    height_config = _active_unlabeled_height_config(config)
    if height_config is None or not height_config.generate_missing:
        return
    from synthforest.training.height.detector_precompute import (
        DetectorHeightTileRecord,
        resolve_detector_height_tile_records_from_tile_dir,
        run_detector_height_precompute_records,
    )

    grouped: dict[float, list[DetectorHeightTileRecord]] = {}
    for source in sources:
        split = source.tile_dir.name
        records = resolve_detector_height_tile_records_from_tile_dir(
            tile_dir=source.tile_dir,
            split=split,
            height_maps_subdir=height_config.height_maps_subdir,
            dataset_name=source.name.split("/", 1)[0],
            product_name=source.name,
        )
        if max_tiles_per_source is not None:
            records = records[: int(max_tiles_per_source)]
        if not records:
            continue
        source_gsd = float(
            source.source_gsd_m if source.source_gsd_m is not None else height_config.source_gsd_m
        )
        grouped.setdefault(source_gsd, []).extend(records)

    for source_gsd, records in grouped.items():
        if not records:
            continue
        print(
            "Height precompute: checking/generating "
            f"{len(records)} unlabeled-real height map(s) at source_gsd={source_gsd:g}."
        )
        gsd_tag = str(source_gsd).replace(".", "p")
        run_detector_height_precompute_records(
            records=records,
            height_checkpoint_path=height_config.height_checkpoint_path,
            height_train_config_path=height_config.height_train_config_path,
            batch_size=max(1, int(config.batch_size)),
            source_gsd=float(source_gsd),
            target_gsd=float(height_config.target_gsd_m),
            precompute_match_checkpoint=bool(height_config.precompute_match_checkpoint),
            summary_path=(
                Path(data_root_path) / f"height_maps_precompute_unlabeled_gsd{gsd_tag}_summary.json"
            ),
            context={
                "split": config.semi_supervised.unlabeled_real.fold,
                "mode": "unlabeled_real",
            },
        )


def _active_unlabeled_height_config(config: DetectorTrainingConfig) -> Any | None:
    if config.height_fusion.enabled:
        return config.height_fusion
    return None


def _ensure_synthetic_height_maps(config: DetectorTrainingConfig) -> None:
    fusion = config.height_fusion
    if not fusion.enabled or not fusion.generate_missing:
        return
    splits = ["train", config.validation.synthetic.fold]
    splits = list(dict.fromkeys(split for split in splits if split in {"train", "valid", "test"}))
    from synthforest.training.height.detector_precompute import (
        resolve_detector_height_tile_records,
        run_detector_height_precompute,
    )

    records = resolve_detector_height_tile_records(config, splits=splits)
    if not records:
        return
    print(f"Height fusion precompute: checking/generating {len(records)} synthetic height map(s).")
    run_detector_height_precompute(
        detector_config=config,
        height_checkpoint_path=fusion.height_checkpoint_path,
        height_train_config_path=fusion.height_train_config_path,
        splits=splits,
        batch_size=max(1, int(config.batch_size)),
        source_gsd=float(fusion.source_gsd_m),
        target_gsd=float(fusion.target_gsd_m),
    )


def _apply_train_sampling(
    dataset: Dataset,
    config: DetectorTrainingConfig,
) -> tuple[Dataset, dict]:
    total_samples = len(dataset)
    if total_samples <= 0:
        raise ValueError("Cannot sample from an empty training dataset.")
    sampling = config.train_sampling
    fraction = float(sampling.fraction)
    min_samples = int(sampling.min_samples)
    seed = int(sampling.seed if sampling.seed is not None else config.seed)
    groups = _train_sampling_groups(dataset)
    selected_indices: list[int] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    for group_index, (source_name, source_indices) in enumerate(groups.items()):
        source_total = len(source_indices)
        selected_count = min(
            source_total,
            max(min_samples, int(math.ceil(source_total * fraction))),
        )
        if selected_count >= source_total:
            group_selected = list(source_indices)
        else:
            group_seed = f"{seed}:{group_index}:{source_name}"
            rng = random.Random(group_seed)
            group_selected = sorted(rng.sample(source_indices, selected_count))
        selected_indices.extend(group_selected)
        per_dataset[source_name] = {
            "total_samples": source_total,
            "selected_samples": selected_count,
            "indices": group_selected,
        }

    indices = sorted(selected_indices)
    selected_count = len(indices)
    if selected_count == total_samples and indices == list(range(total_samples)):
        indices = list(range(total_samples))
        sampled_dataset = dataset
    else:
        sampled_dataset = Subset(dataset, indices)
    metadata = {
        "total_samples": total_samples,
        "selected_samples": selected_count,
        "fraction": fraction,
        "seed": seed,
        "min_samples": min_samples,
        "sampling_unit": "records_per_dataset",
        "indices": indices,
        "per_dataset": per_dataset,
    }
    if selected_count < total_samples:
        print(
            "Train sampling: keeping "
            f"{selected_count}/{total_samples} sample(s) "
            f"(fraction={fraction:g}, seed={seed})."
        )
    return sampled_dataset, metadata


def _train_sampling_groups(dataset: Dataset) -> dict[str, list[int]]:
    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or len(records) != len(dataset):
        return {"__all__": list(range(len(dataset)))}

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        source_name = None
        if isinstance(record, dict):
            source_name = record.get("source_name")
        if not source_name:
            source_name = "__unknown__"
        groups.setdefault(str(source_name), []).append(index)
    return groups


def evaluate_validation_domains(
    *,
    model: torch.nn.Module,
    synthetic_loader: DataLoader,
    real_loaders: dict[str, DataLoader],
    device: torch.device,
    config: DetectorTrainingConfig,
) -> dict:
    synthetic_metrics = evaluate_model(
        model,
        synthetic_loader,
        device,
        config=config,
    )
    real_by_dataset = {
        dataset_name: evaluate_model(
            model,
            loader,
            device,
            config=config,
        )
        for dataset_name, loader in real_loaders.items()
    }
    real_metrics = macro_average_metrics(real_by_dataset) if real_by_dataset else {}
    return {
        "valid_synthetic": synthetic_metrics,
        "valid_real": {
            **real_metrics,
            "by_dataset": real_by_dataset,
        },
    }


def _validation_metric_domain(config: DetectorTrainingConfig) -> str:
    return "valid_real" if config.validation.real.enabled else "valid_synthetic"


def _selection_metric_value(
    payload: dict,
    metric_name: str,
    *,
    domain: str = "valid_real",
) -> float:
    value = payload.get(domain, {}).get(metric_name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{domain} selection metric {metric_name!r} is missing or invalid.")
    value = float(value)
    if value < 0:
        raise RuntimeError(f"{domain} selection metric {metric_name!r} is negative ({value}).")
    return value


def _optional_selection_metric_value(
    payload: dict,
    metric_name: str,
    *,
    domain: str = "valid_real",
) -> Optional[float]:
    value = payload.get(domain, {}).get(metric_name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    value = float(value)
    return value if value >= 0.0 else None


def _validation_wandb_payload(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload["valid_synthetic"].items():
        if _is_number(value):
            metrics[f"valid_synthetic/{key}"] = float(value)
    for key, value in payload["valid_real"].items():
        if key == "by_dataset":
            continue
        if _is_number(value):
            metrics[f"valid_real/{key}"] = float(value)
    for dataset_name, dataset_metrics in payload["valid_real"]["by_dataset"].items():
        for key, value in dataset_metrics.items():
            if _is_number(value):
                metrics[f"valid_real_{dataset_name}/{key}"] = float(value)
    return metrics


def _sweep_metrics(
    config: DetectorTrainingConfig,
    *,
    global_iteration: int,
    metric_value: float,
) -> dict[str, float]:
    if config.sweep is None:
        return {}
    if global_iteration not in config.sweep.objective_iterations:
        return {}
    metrics = {"sweep/objective": float(metric_value)}
    control = config.sweep.parent_control_metrics.get(str(global_iteration))
    if control is not None:
        control_value = float(control)
        objective_metric = _sweep_objective_metric(config)
        metrics["sweep/control"] = control_value
        metrics[f"sweep/control_{objective_metric}"] = control_value
        metrics["sweep/staged_lora_uplift"] = float(metric_value) - control_value
    return metrics


def _sweep_objective_metric(
    config: DetectorTrainingConfig,
) -> str:
    if config.sweep is None:
        return config.validation.selection_metric
    return config.sweep.objective_metric or config.validation.selection_metric


def _save_stage_last(
    layout: TrainingRunLayout,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: DetectorTrainingConfig,
    stage: TrainingStageConfig,
    stage_index: int,
    stage_iteration: int,
    iteration: int,
    best_metric: float,
    student_model: Optional[torch.nn.Module] = None,
    ema_teacher_model: Optional[torch.nn.Module] = None,
) -> None:
    stage_name = _slugify(stage.name)
    for path in (
        layout.stage_checkpoint(stage_name, "last"),
        layout.last_checkpoint,
    ):
        _save_checkpoint(
            path,
            model=model,
            student_model=student_model,
            ema_teacher_model=ema_teacher_model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            stage=stage,
            stage_index=stage_index,
            stage_iteration=stage_iteration,
            iteration=iteration,
            best_metric=best_metric,
        )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: DetectorTrainingConfig,
    stage: TrainingStageConfig,
    stage_index: int,
    stage_iteration: int,
    iteration: int,
    best_metric: float,
    best_metric_name: Optional[str] = None,
    student_model: Optional[torch.nn.Module] = None,
    ema_teacher_model: Optional[torch.nn.Module] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = config.to_dict()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_config": config_payload,
        "run_name": config.run_name,
        "stage": stage.name,
        "stage_index": stage_index,
        "stage_iteration": stage_iteration,
        "iteration": iteration,
        "best_metric": best_metric,
        "best_metric_name": (
            best_metric_name
            or f"{_validation_metric_domain(config)}/{config.validation.selection_metric}"
        ),
    }
    if student_model is not None:
        payload["student_state_dict"] = student_model.state_dict()
    if ema_teacher_model is not None:
        payload["ema_teacher_state_dict"] = ema_teacher_model.state_dict()
    torch.save(payload, path)


def _append_validation_metrics(
    layout: TrainingRunLayout,
    iteration: int,
    metrics: dict,
) -> None:
    payload = {"iteration": iteration, **metrics}
    _append_jsonl(layout.validation_metrics, payload)
    with open(
        layout.latest_validation_metrics,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_trainability_summary(
    layout: TrainingRunLayout,
    summary: dict[str, dict[str, int]],
) -> None:
    with open(layout.trainability_metadata, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _clear_run_metrics(layout: TrainingRunLayout) -> None:
    for path in (
        layout.train_metrics,
        layout.validation_metrics,
        layout.latest_validation_metrics,
    ):
        if path.exists():
            path.unlink()


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if _is_number(value)}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _init_wandb(config: DetectorTrainingConfig, output_dir: Path):
    if not config.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        message = (
            "wandb.enabled is true, but wandb is not installed in this Python "
            "environment. Install wandb or set wandb.enabled: false."
        )
        if config.wandb.required:
            raise RuntimeError(message)
        print(f"{message} Continuing without wandb logging.")
        return None
    try:
        return wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=output_dir.name,
            tags=config.wandb.tags,
            mode=config.wandb.mode,
            config=config.to_dict(),
            dir=str(output_dir),
        )
    except Exception as exc:
        message = f"wandb initialization failed: {exc}"
        if config.wandb.required:
            raise RuntimeError(message) from exc
        print(f"{message} Continuing without wandb logging.")
        return None


def _log_wandb(wandb_run, metrics: dict[str, float], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def _finish_wandb(
    wandb_run,
    best_checkpoint: Path,
    config: DetectorTrainingConfig,
    *,
    finish: bool = True,
) -> None:
    if wandb_run is None:
        return
    if config.wandb.log_artifacts and best_checkpoint.exists():
        import wandb

        artifact = wandb.Artifact(f"{wandb_run.name}_best", type="model")
        artifact.add_file(str(best_checkpoint))
        wandb_run.log_artifact(artifact)
    if finish:
        wandb_run.finish()


def _update_wandb_summary(wandb_run, values: dict) -> None:
    if wandb_run is None:
        return
    for key, value in values.items():
        wandb_run.summary[key] = value
