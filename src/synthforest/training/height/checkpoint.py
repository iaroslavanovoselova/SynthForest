from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from synthforest.training.config import HeightTrainingConfig
from synthforest.training.models import build_dinov3_height_model
from synthforest.training.utils import load_trained_state_dict, set_all_seeds

HEIGHT_CHECKPOINT_SCHEMA_VERSION = 1


def load_height_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    config: Optional[HeightTrainingConfig] = None,
    config_path: Optional[str | Path] = None,
) -> tuple[torch.nn.Module, HeightTrainingConfig, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_height_checkpoint_payload(checkpoint, checkpoint_path)
    embedded_config = _embedded_height_config(checkpoint)
    external_config = config
    if config_path is not None:
        if external_config is not None:
            raise ValueError("Pass config or config_path, not both.")
        external_config = HeightTrainingConfig.from_yaml(config_path)

    if external_config is not None:
        _assert_height_architecture_compatible(
            checkpoint_config=embedded_config,
            external_config=external_config,
        )
        resolved_config = external_config
        config_source = "external"
    else:
        resolved_config = embedded_config
        config_source = "checkpoint"

    set_all_seeds(resolved_config.seed)
    model = build_dinov3_height_model(resolved_config).to(device)
    try:
        load_trained_state_dict(model, checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint state does not match the resolved DINOv3-DPT height architecture."
        ) from exc
    model.requires_grad_(False)
    if hasattr(model, "backbone_grad_enabled"):
        model.backbone_grad_enabled = False
    model.eval()
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "schema_version": checkpoint["schema_version"],
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "iteration": checkpoint.get("iteration"),
        "stage": checkpoint.get("stage"),
        "stage_index": checkpoint.get("stage_index"),
        "stage_iteration": checkpoint.get("stage_iteration"),
        "best_metric": checkpoint.get("best_metric"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "config_source": config_source,
        "has_lora": resolved_config.lora is not None,
        "model_type": resolved_config.model_type,
        "run_name": checkpoint.get("run_name") or resolved_config.run_name,
    }
    return model, resolved_config, metadata


def load_height_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> HeightTrainingConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_height_checkpoint_payload(checkpoint, Path(checkpoint_path))
    embedded = _embedded_height_config(checkpoint)
    if config_path is None:
        return embedded
    external = HeightTrainingConfig.from_yaml(config_path)
    _assert_height_architecture_compatible(embedded, external)
    return external


def _embedded_height_config(checkpoint: Any) -> HeightTrainingConfig:
    raw = checkpoint["training_config"]
    try:
        return HeightTrainingConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint training_config is not a valid height configuration.") from exc


def _validate_height_checkpoint_payload(checkpoint: Any, checkpoint_path: Path) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Height checkpoint must be a mapping: {checkpoint_path}")
    if checkpoint.get("checkpoint_type") != "height":
        raise ValueError(f"Checkpoint is not a height checkpoint: {checkpoint_path}")
    schema_version = checkpoint.get("schema_version")
    if schema_version != HEIGHT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported height checkpoint schema {schema_version!r}; "
            f"expected {HEIGHT_CHECKPOINT_SCHEMA_VERSION}."
        )
    if not isinstance(checkpoint.get("training_config"), dict):
        raise ValueError("Checkpoint is missing training_config.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint is missing model_state_dict.")


def _assert_height_architecture_compatible(
    checkpoint_config: HeightTrainingConfig,
    external_config: HeightTrainingConfig,
) -> None:
    checkpoint_signature = _height_architecture_signature(checkpoint_config)
    external_signature = _height_architecture_signature(external_config)
    mismatches = [
        key for key in checkpoint_signature if checkpoint_signature[key] != external_signature[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={checkpoint_signature[key]!r}, external={external_signature[key]!r}"
            for key in mismatches
        )
        raise ValueError("External height config is incompatible with the checkpoint: " + details)


def _height_architecture_signature(config: HeightTrainingConfig) -> dict[str, Any]:
    lora = None
    if config.lora is not None:
        lora = {
            "num_blocks": config.lora.num_blocks,
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": tuple(config.lora.target_modules),
        }
    return {
        "model_type": config.model_type,
        "dinov3_model": config.dinov3_model,
        "dinov3_embed_dim": config.dinov3_embed_dim,
        "dinov3_patch_size": config.dinov3_patch_size,
        "dinov3_use_fake_backbone": config.dinov3_use_fake_backbone,
        "dinov3_fake_num_blocks": config.dinov3_fake_num_blocks,
        "dpt_feature_layers": tuple(config.dpt_feature_layers),
        "dpt_feature_dim": config.dpt_feature_dim,
        "dpt_out_channels": tuple(config.dpt_out_channels),
        "dpt_use_batch_norm": config.dpt_use_batch_norm,
        "height_output_channels": config.height_output_channels,
        "log_variance_min": config.log_variance_min,
        "log_variance_max": config.log_variance_max,
        "lora": lora,
    }
