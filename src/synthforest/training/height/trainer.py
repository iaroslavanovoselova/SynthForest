from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from synthforest.training.config import HeightTrainingConfig, HeightTrainingStageConfig, _slugify
from synthforest.training.engine.optimization import build_stage_scheduler, lr_metrics_by_group
from synthforest.training.height.checkpoint import HEIGHT_CHECKPOINT_SCHEMA_VERSION
from synthforest.training.height.evaluation import (
    build_height_train_loader,
    build_real_height_validation_loaders,
    build_synthetic_height_validation_loader,
    evaluate_height_validation_domains,
)
from synthforest.training.height.metrics import (
    masked_laplace_nll_loss,
    masked_resolution_aware_smooth_l1_loss,
)
from synthforest.training.layout import TrainingRunLayout, validate_checkpoint_name
from synthforest.training.models import (
    assert_stage_trainability,
    build_dinov3_height_model,
    set_stage_trainability,
    summarize_trainable_parameters,
)
from synthforest.training.utils import set_all_seeds


def train_height_from_config(
    config: HeightTrainingConfig,
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
    ).create()
    output_dir = layout.root
    _clear_run_metrics(layout)
    config.save_yaml(layout.resolved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_height_train_loader(config)
    synthetic_valid_loader = build_synthetic_height_validation_loader(config)
    real_valid_loaders = build_real_height_validation_loaders(config)

    model = build_dinov3_height_model(config).to(device)
    owns_wandb_run = wandb_run is None
    if wandb_run is None:
        wandb_run = _init_wandb(config, output_dir)
    _update_wandb_summary(
        wandb_run,
        {
            "run/output_dir": str(output_dir.resolve()),
            "run/phase": "training",
            "run/objective_metric": (
                f"{_validation_metric_domain(config)}/{config.validation.selection_metric}"
            ),
            "run/objective_direction": "minimize",
        },
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    global_iteration = 0
    data_iter = iter(train_loader)
    final_stage_index = len(config.stages) - 1
    best_final_metric = math.inf
    trainability_by_stage: dict[str, dict[str, int]] = {}

    for stage_index, stage in enumerate(config.stages):
        set_stage_trainability(model, train_lora=stage.train_lora)
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

        optimizer = build_height_stage_optimizer(model, stage)
        scheduler = build_stage_scheduler(optimizer, stage)
        optimizer.zero_grad(set_to_none=True)
        stage_best_metric = math.inf

        for stage_iteration in range(1, stage.max_iterations + 1):
            model.train()
            losses_for_log: dict[str, float] = {}
            total_loss_for_log = 0.0

            for _ in range(config.grad_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                images = batch["images"].to(device)
                heights = batch["heights"].to(device)
                valid_masks = batch["valid_masks"].to(device)
                resolution_factors = batch["resolution_factors"].to(device)
                with torch.amp.autocast(
                    "cuda",
                    enabled=config.use_amp and device.type == "cuda",
                ):
                    outputs = model(images)
                    predictions = outputs["height"]
                    if config.loss_name == "smooth_l1":
                        loss = masked_resolution_aware_smooth_l1_loss(
                            predictions,
                            heights,
                            valid_masks,
                            resolution_factors=resolution_factors,
                            height_reducer=config.resolution_degradation.height_reducer,
                            min_canopy_fraction=(config.resolution_degradation.min_canopy_fraction),
                        )
                        loss_log_key = "loss_height"
                    elif config.loss_name == "laplace_nll":
                        log_variance = outputs.get("log_variance")
                        if log_variance is None:
                            raise RuntimeError(
                                "loss_name laplace_nll requires model output 'log_variance'."
                            )
                        loss = masked_laplace_nll_loss(
                            predictions,
                            log_variance,
                            heights,
                            valid_masks,
                            log_variance_min=config.log_variance_min,
                            log_variance_max=config.log_variance_max,
                        )
                        loss_log_key = "loss_laplace_nll"
                    else:
                        raise RuntimeError(f"Unsupported height loss_name {config.loss_name!r}.")
                    scaled_loss = loss / config.grad_accumulation_steps

                scaler.scale(scaled_loss).backward()
                loss_value = float(loss.detach().cpu())
                losses_for_log[loss_log_key] = losses_for_log.get(loss_log_key, 0.0) + loss_value
                if config.loss_name == "laplace_nll":
                    mean_log_variance = _masked_tensor_mean(
                        outputs["log_variance"].detach(),
                        valid_masks,
                    )
                    if mean_log_variance is not None:
                        losses_for_log["log_variance_mean"] = (
                            losses_for_log.get("log_variance_mean", 0.0) + mean_log_variance
                        )
                    mean_variance = _masked_tensor_mean(
                        outputs["log_variance"].detach().exp(),
                        valid_masks,
                    )
                    if mean_variance is not None:
                        losses_for_log["variance_mean"] = (
                            losses_for_log.get("variance_mean", 0.0) + mean_variance
                        )
                total_loss_for_log += loss_value

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_iteration += 1
            assert_stage_trainability(model, train_lora=stage.train_lora)

            if stage_iteration % stage.train_log_interval == 0 or stage_iteration == 1:
                log_data = {
                    "iteration": global_iteration,
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "stage_iteration": stage_iteration,
                    "train/total_loss": total_loss_for_log / config.grad_accumulation_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                log_data.update(lr_metrics_by_group(optimizer))
                for key, value in losses_for_log.items():
                    log_data[f"train/{key}"] = value / config.grad_accumulation_steps
                _append_jsonl(layout.train_metrics, log_data)
                _log_wandb(wandb_run, _numeric_metrics(log_data), global_iteration)
                print(
                    f"stage={stage.name} "
                    f"iter={global_iteration} "
                    f"stage_iter={stage_iteration} "
                    f"loss={log_data['train/total_loss']:.4f} "
                    f"lr={log_data['train/lr']:.6g}"
                )

            if stage_iteration % stage.eval_interval == 0:
                validation_payload = evaluate_height_validation_domains(
                    model=model,
                    synthetic_loader=synthetic_valid_loader,
                    real_loaders=real_valid_loaders,
                    device=device,
                    config=config,
                    evaluation_iteration=global_iteration,
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
                sweep_metrics = _height_sweep_metrics(
                    config,
                    global_iteration=global_iteration,
                    metric_value=metric_value,
                )
                payload.update(sweep_metrics)
                _append_validation_metrics(layout, global_iteration, payload)
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

                if metric_value < stage_best_metric:
                    stage_best_metric = metric_value
                    _save_height_checkpoint(
                        layout.stage_checkpoint(_slugify(stage.name), "best"),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=stage_best_metric,
                    )
                if stage_index == final_stage_index and metric_value < best_final_metric:
                    best_final_metric = metric_value
                    _save_height_checkpoint(
                        layout.best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        stage=stage,
                        stage_index=stage_index,
                        stage_iteration=stage_iteration,
                        iteration=global_iteration,
                        best_metric=best_final_metric,
                    )
                print(
                    f"valid stage={stage.name} iter={global_iteration} "
                    f"{_validation_metric_domain(config)}_"
                    f"{config.validation.selection_metric}={metric_value:.6f}"
                )

            if stage_iteration % stage.checkpoint_interval == 0:
                _save_stage_last(
                    layout,
                    model=model,
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
            model=model,
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


def build_height_stage_optimizer(
    model: torch.nn.Module,
    stage: HeightTrainingStageConfig,
) -> torch.optim.Optimizer:
    lora_params = []
    decoder_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_parameter_name(name):
            lora_params.append(parameter)
        else:
            decoder_params.append(parameter)
    param_groups = [
        {
            "params": decoder_params,
            "lr": stage.optimizer.decoder_lr,
            "weight_decay": stage.optimizer.decoder_weight_decay,
            "name": "decoder",
        }
    ]
    if stage.train_lora:
        if not lora_params:
            raise ValueError(f"Stage {stage.name} enables LoRA but found no trainable adapters.")
        param_groups.append(
            {
                "params": lora_params,
                "lr": stage.optimizer.lora_lr,
                "weight_decay": stage.optimizer.lora_weight_decay,
                "name": "lora",
            }
        )
    elif lora_params:
        raise ValueError(f"Stage {stage.name} has trainable LoRA params while train_lora is false.")
    name = stage.optimizer.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    raise ValueError(f"Unsupported optimizer for stage {stage.name}: {stage.optimizer.name}")


def _run_post_training_benchmark(
    config: HeightTrainingConfig,
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
        from synthforest.training.config import (
            HeightBenchmarkConfig,
            HeightRasterBenchmarkConfig,
        )
        from synthforest.training.height.benchmarking import run_height_benchmark
        from synthforest.training.height.raster_benchmarking import (
            run_height_raster_benchmark,
        )

        if benchmark.config_path is None:
            raise ValueError(
                "post_training_benchmark.config_path is required when "
                "post_training_benchmark.enabled is true."
            )
        if _is_raster_height_benchmark_config(benchmark.config_path):
            benchmark_config = HeightRasterBenchmarkConfig.from_yaml(
                benchmark.config_path,
                validate=False,
            )
            benchmark_kind = "raster"
            benchmark_runner = run_height_raster_benchmark
        else:
            benchmark_config = HeightBenchmarkConfig.from_yaml(
                benchmark.config_path,
                validate=False,
            )
            benchmark_kind = "tile"
            benchmark_runner = run_height_benchmark
        if benchmark_config.model_type != config.model_type:
            raise ValueError(
                "Benchmark model_type does not match the training config: "
                f"benchmark={benchmark_config.model_type!r}, training={config.model_type!r}."
            )
        metadata["benchmark_kind"] = benchmark_kind
        _write_json(layout.post_training_benchmark_metadata, metadata)
        benchmark_config.run_folder = str(layout.root)
        benchmark_config.checkpoint_name = checkpoint_name
        benchmark_config.checkpoint_path = None
        benchmark_config.validate()
        output_dir = benchmark_runner(benchmark_config)
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(layout.post_training_benchmark_metadata, metadata)
        message = f"Post-training height benchmark failed: {exc}"
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


def _is_raster_height_benchmark_config(config_path: str | Path) -> bool:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Benchmark config must be a mapping: {config_path}")
    raster_keys = {
        "rasters",
        "rgb_root_path",
        "height_root_path",
        "rgb_glob",
        "height_glob",
        "rgb_name_suffix",
        "height_name_suffix",
    }
    return bool(raster_keys & set(raw))


def _save_stage_last(
    layout: TrainingRunLayout,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: HeightTrainingConfig,
    stage: HeightTrainingStageConfig,
    stage_index: int,
    stage_iteration: int,
    iteration: int,
    best_metric: float,
) -> None:
    stage_name = _slugify(stage.name)
    for path in (
        layout.stage_checkpoint(stage_name, "last"),
        layout.last_checkpoint,
    ):
        _save_height_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            stage=stage,
            stage_index=stage_index,
            stage_iteration=stage_iteration,
            iteration=iteration,
            best_metric=best_metric,
        )


def _save_height_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: HeightTrainingConfig,
    stage: HeightTrainingStageConfig,
    stage_index: int,
    stage_iteration: int,
    iteration: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": HEIGHT_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "height",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_config": config.to_dict(),
            "run_name": config.run_name,
            "stage": stage.name,
            "stage_index": stage_index,
            "stage_iteration": stage_iteration,
            "iteration": iteration,
            "best_metric": best_metric,
            "best_metric_name": (
                f"{_validation_metric_domain(config)}/{config.validation.selection_metric}"
            ),
            "best_metric_direction": "minimize",
        },
        path,
    )


def _validation_metric_domain(config: HeightTrainingConfig) -> str:
    return "valid_real" if config.validation.real.enabled else "valid_synthetic"


def _selection_metric_value(
    payload: dict,
    metric_name: str,
    *,
    domain: str = "valid_real",
) -> float:
    value = payload.get(domain, {}).get(metric_name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{domain} selection metric {metric_name!r} is missing.")
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"{domain} selection metric {metric_name!r} is invalid ({value}).")
    return value


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


def _height_sweep_metrics(
    config: HeightTrainingConfig,
    *,
    global_iteration: int,
    metric_value: float,
) -> dict[str, float]:
    if config.sweep is None or global_iteration not in config.sweep.objective_iterations:
        return {}
    return {"sweep/objective": float(metric_value)}


def _append_validation_metrics(layout: TrainingRunLayout, iteration: int, metrics: dict) -> None:
    payload = {"iteration": iteration, **metrics}
    _append_jsonl(layout.validation_metrics, payload)
    with layout.latest_validation_metrics.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_trainability_summary(
    layout: TrainingRunLayout,
    summary: dict[str, dict[str, int]],
) -> None:
    with layout.trainability_metadata.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _clear_run_metrics(layout: TrainingRunLayout) -> None:
    for path in (
        layout.train_metrics,
        layout.validation_metrics,
        layout.latest_validation_metrics,
    ):
        if path.exists():
            path.unlink()


def _masked_tensor_mean(tensor: torch.Tensor, valid_mask: torch.Tensor) -> float | None:
    mask = valid_mask.to(device=tensor.device, dtype=torch.bool)
    if mask.ndim == tensor.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != tensor.shape:
        mask = mask.expand_as(tensor)
    if not bool(mask.any()):
        return None
    value = tensor[mask].mean()
    if not torch.isfinite(value):
        return None
    return float(value.detach().cpu())


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if _is_number(value)}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _init_wandb(config: HeightTrainingConfig, output_dir: Path):
    if not config.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        message = (
            "wandb.enabled is true, but wandb is not installed in this Python environment. "
            "Install wandb or set wandb.enabled: false."
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
    config: HeightTrainingConfig,
    *,
    finish: bool = True,
) -> None:
    if wandb_run is None:
        return
    if config.wandb.log_artifacts and best_checkpoint.exists():
        import wandb

        artifact = wandb.Artifact(
            name=f"{wandb_run.name}-height-checkpoint",
            type="model",
        )
        artifact.add_file(str(best_checkpoint))
        wandb_run.log_artifact(artifact)
    if finish:
        wandb_run.finish()


def _update_wandb_summary(wandb_run, values: dict) -> None:
    if wandb_run is not None:
        wandb_run.summary.update(values)


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or ".lora_" in name
