from __future__ import annotations

import math

import torch

from synthforest.training.config import TrainingStageConfig


def build_stage_optimizer(
    model: torch.nn.Module,
    stage: TrainingStageConfig,
) -> torch.optim.Optimizer:
    lora_params = []
    detector_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_parameter_name(name):
            lora_params.append(parameter)
        else:
            detector_params.append(parameter)

    param_groups = [
        {
            "params": detector_params,
            "lr": stage.optimizer.detector_lr,
            "weight_decay": stage.optimizer.detector_weight_decay,
            "name": "detector",
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


def build_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    stage: TrainingStageConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, stage.optimizer.warmup_iterations)
    total = max(1, stage.max_iterations)
    min_factor = stage.optimizer.min_lr_factor

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def lr_metrics_by_group(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name") or f"group_{index}"
        metrics[f"train/lr_{group_name}"] = group["lr"]
    return metrics


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or ".lora_" in name
