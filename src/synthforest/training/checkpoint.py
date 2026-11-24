from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.models import build_dinov3_detector
from synthforest.training.utils import load_trained_state_dict, set_all_seeds

CHECKPOINT_SCHEMA_VERSION = 4


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    config: Optional[DetectorTrainingConfig] = None,
    config_path: Optional[str | Path] = None,
) -> tuple[torch.nn.Module, DetectorTrainingConfig, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_payload(checkpoint, checkpoint_path)
    embedded_config = _embedded_training_config(checkpoint)
    external_config = config
    if config_path is not None:
        if external_config is not None:
            raise ValueError("Pass config or config_path, not both.")
        external_config = DetectorTrainingConfig.from_yaml(config_path)

    if external_config is not None:
        _assert_architecture_compatible(
            checkpoint_config=embedded_config,
            external_config=external_config,
        )
        resolved_config = external_config
        config_source = "external"
    else:
        resolved_config = embedded_config
        config_source = "checkpoint"

    set_all_seeds(resolved_config.seed)
    model = build_dinov3_detector(resolved_config).to(device)
    state_dict = _checkpoint_eval_state_dict(checkpoint)
    try:
        load_trained_state_dict(model, state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint state does not match the resolved DINOv3 detector "
            "architecture. Check the DINO variant, FPN/anchor settings, and LoRA "
            "configuration and model_type."
        ) from exc
    model.requires_grad_(False)
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "backbone_grad_enabled"):
        backbone.backbone_grad_enabled = False
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "schema_version": (checkpoint["schema_version"]),
        "iteration": (checkpoint.get("iteration")),
        "stage": (checkpoint.get("stage")),
        "stage_index": (checkpoint.get("stage_index")),
        "stage_iteration": (checkpoint.get("stage_iteration")),
        "best_metric": (checkpoint.get("best_metric")),
        "best_metric_name": (checkpoint.get("best_metric_name")),
        "config_source": config_source,
        "has_lora": resolved_config.lora is not None,
        "model_type": resolved_config.model_type,
        "run_name": checkpoint.get("run_name") or resolved_config.run_name,
        "valid_mask": resolved_config.to_dict()["valid_mask"],
        "valid_mask_policy_source": config_source,
    }
    return model, resolved_config, metadata


def load_checkpoint_weights_into_model(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    config: DetectorTrainingConfig,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_payload(checkpoint, checkpoint_path)
    checkpoint_config = _embedded_training_config(checkpoint)
    _assert_architecture_compatible(
        checkpoint_config=checkpoint_config,
        external_config=config,
    )
    try:
        load_trained_state_dict(model, _checkpoint_eval_state_dict(checkpoint))
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint state does not match the requested DINOv3 detector "
            "architecture. Check the DINO variant, FPN/anchor settings, LoRA "
            "configuration and model_type."
        ) from exc

    return {
        "type": "full_checkpoint",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "schema_version": checkpoint["schema_version"],
        "iteration": checkpoint.get("iteration"),
        "stage": checkpoint.get("stage"),
        "stage_index": checkpoint.get("stage_index"),
        "stage_iteration": checkpoint.get("stage_iteration"),
        "best_metric": checkpoint.get("best_metric"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "source_run_name": checkpoint.get("run_name") or checkpoint_config.run_name,
        "model_type": checkpoint_config.model_type,
        "has_lora": checkpoint_config.lora is not None,
    }


def load_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> DetectorTrainingConfig:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_payload(checkpoint, Path(checkpoint_path))
    embedded = _embedded_training_config(checkpoint)
    if config_path is None:
        return embedded
    external = DetectorTrainingConfig.from_yaml(config_path)
    _assert_architecture_compatible(embedded, external)
    return external


def _embedded_training_config(
    checkpoint: Any,
) -> DetectorTrainingConfig:
    raw = dict(checkpoint["training_config"])
    # Checkpoints from earlier code versions carry a disabled pseudo-label
    # height-verifier section that is no longer part of the configuration.
    legacy_verifier = raw.pop("height_verifier", None)
    if isinstance(legacy_verifier, dict) and legacy_verifier.get("enabled"):
        raise ValueError("Checkpoints trained with the height verifier are not supported.")
    try:
        return DetectorTrainingConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint training_config is not a valid detector configuration."
        ) from exc


def _validate_checkpoint_payload(
    checkpoint: Any,
    checkpoint_path: Path,
) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint must be a schema-{CHECKPOINT_SCHEMA_VERSION} mapping: {checkpoint_path}"
        )
    schema_version = checkpoint.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported full checkpoint schema {schema_version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    if not isinstance(checkpoint.get("training_config"), dict):
        raise ValueError("Checkpoint is missing training_config.")
    if _checkpoint_eval_state_dict(checkpoint, validate_only=True) is None:
        raise ValueError("Checkpoint is missing model_state_dict.")


def _checkpoint_eval_state_dict(
    checkpoint: dict[str, Any],
    *,
    validate_only: bool = False,
) -> Optional[dict[str, Any]]:
    for key in ("model_state_dict", "ema_teacher_state_dict", "student_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    if validate_only:
        return None
    raise ValueError("Checkpoint is missing a model state dict.")


def _assert_architecture_compatible(
    checkpoint_config: DetectorTrainingConfig,
    external_config: DetectorTrainingConfig,
) -> None:
    checkpoint_signature = _architecture_signature(checkpoint_config)
    external_signature = _architecture_signature(external_config)
    mismatches = [
        key for key in checkpoint_signature if checkpoint_signature[key] != external_signature[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={checkpoint_signature[key]!r}, external={external_signature[key]!r}"
            for key in mismatches
        )
        raise ValueError("External training config is incompatible with the checkpoint: " + details)


def _architecture_signature(config: DetectorTrainingConfig) -> dict[str, Any]:
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
        "backbone_out_channels": config.backbone_out_channels,
        "num_classes": config.num_classes,
        "pyramid_strides": tuple(config.pyramid_strides),
        "anchor_sizes": tuple(
            tuple(value) if isinstance(value, (list, tuple)) else value
            for value in config.anchor_sizes
        ),
        "aspect_ratios": tuple(config.aspect_ratios),
        "height_fusion": _height_fusion_signature(config),
        "lora": lora,
    }


def _height_fusion_signature(config: DetectorTrainingConfig) -> dict[str, Any]:
    fusion = config.height_fusion
    if not fusion.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "levels": tuple(fusion.levels),
        "encoder_hidden_channels": int(fusion.encoder_hidden_channels),
        "fusion_init_scale": float(fusion.fusion_init_scale),
        "clip_min_m": float(fusion.clip_min_m),
        "clip_max_m": float(fusion.clip_max_m),
        "mean_m": float(fusion.mean_m),
        "std_m": float(fusion.std_m),
        "use_uncertainty": bool(fusion.use_uncertainty),
        "uncertainty_log_variance_min": float(fusion.uncertainty_log_variance_min),
        "uncertainty_log_variance_max": float(fusion.uncertainty_log_variance_max),
        "uncertainty_reliability_tau": float(fusion.uncertainty_reliability_tau),
        "uncertainty_reliability_min": float(fusion.uncertainty_reliability_min),
    }
