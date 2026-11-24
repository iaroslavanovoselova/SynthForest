from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import torch

from synthforest.training.config import DetectorTrainingConfig

BRANCH_CHECKPOINT_SCHEMA_VERSION = 2
_DINOV3_STATE_PREFIXES = ("backbone.dinov3.", "dinov3.")


def branch_architecture_signature(
    config: DetectorTrainingConfig,
) -> dict[str, Any]:
    """Describe weights shared by frozen and LoRA detector models."""
    signature = {
        "dinov3_model_dir": config.dinov3_model_dir,
        "dinov3_model": config.dinov3_model,
        "dinov3_pretraining": config.dinov3_pretraining,
        "dinov3_embed_dim": int(config.dinov3_embed_dim),
        "dinov3_patch_size": int(config.dinov3_patch_size),
        "dinov3_use_fake_backbone": bool(config.dinov3_use_fake_backbone),
        "backbone_out_channels": int(config.backbone_out_channels),
        "num_classes": int(config.num_classes),
        "pyramid_strides": [int(value) for value in config.pyramid_strides],
        "anchor_sizes": config.anchor_sizes,
        "aspect_ratios": [float(value) for value in config.aspect_ratios],
        "box_score_thresh": float(config.box_score_thresh),
        "box_nms_thresh": float(config.box_nms_thresh),
        "box_predictions_per_image": int(config.box_predictions_per_image),
        "image_mean": [float(value) for value in config.augmentation.image_mean],
        "image_std": [float(value) for value in config.augmentation.image_std],
        "resize_min_size_eval": int(config.augmentation.resize_min_size_eval),
        "max_size": int(config.augmentation.max_size),
        "height_fusion": _height_fusion_signature(config),
    }
    signature["model_type"] = config.model_type
    return signature


def _height_fusion_signature(config: DetectorTrainingConfig) -> dict[str, Any]:
    fusion = config.height_fusion
    if not fusion.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "levels": list(fusion.levels),
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


def branch_architecture_fingerprint(
    config: DetectorTrainingConfig,
) -> str:
    payload = json.dumps(
        branch_architecture_signature(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_branch_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: DetectorTrainingConfig,
    iteration: int,
    stage: str,
    output_dir: str | Path,
    source_run_id: Optional[str] = None,
    source_run_name: Optional[str] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    detector_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(_DINOV3_STATE_PREFIXES)
    }
    fingerprint = branch_architecture_fingerprint(config)
    torch.save(
        {
            "schema_version": BRANCH_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "detector_branch",
            "detector_state_dict": detector_state,
            "iteration": int(iteration),
            "stage": str(stage),
            "source_run_id": source_run_id,
            "source_run_name": source_run_name,
            "source_output_dir": str(Path(output_dir).resolve()),
            "architecture_signature": branch_architecture_signature(config),
            "architecture_fingerprint": fingerprint,
            "training_config": config.to_dict(),
        },
        path,
    )
    return path


def load_branch_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: DetectorTrainingConfig,
    expected_iteration: Optional[int] = None,
    expected_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Branch checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid branch checkpoint payload: {path}")
    if checkpoint.get("checkpoint_type") != "detector_branch":
        raise ValueError(f"Checkpoint is not a detector branch checkpoint: {path}")
    schema_version = checkpoint.get("schema_version")
    if schema_version != BRANCH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported detector branch checkpoint schema: {schema_version!r}")

    actual_fingerprint = checkpoint.get("architecture_fingerprint")
    config_fingerprint = branch_architecture_fingerprint(config)
    if actual_fingerprint != config_fingerprint:
        raise ValueError(
            "Branch checkpoint architecture does not match the requested model. "
            f"checkpoint={actual_fingerprint!r}, config={config_fingerprint!r}"
        )
    if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Branch checkpoint fingerprint does not match the parent manifest. "
            f"checkpoint={actual_fingerprint!r}, manifest={expected_fingerprint!r}"
        )

    checkpoint_iteration = int(checkpoint.get("iteration", -1))
    if expected_iteration is not None and checkpoint_iteration != int(expected_iteration):
        raise ValueError(
            "Branch checkpoint iteration does not match initialization metadata. "
            f"checkpoint={checkpoint_iteration}, expected={expected_iteration}"
        )

    detector_state = checkpoint.get("detector_state_dict")
    if not isinstance(detector_state, dict) or not detector_state:
        raise ValueError(f"Branch checkpoint has no detector state: {path}")
    incompatible = model.load_state_dict(detector_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        name for name in incompatible.missing_keys if not name.startswith(_DINOV3_STATE_PREFIXES)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Detector branch checkpoint did not load cleanly. "
            f"unexpected={unexpected[:5]}, missing={disallowed_missing[:5]}"
        )

    return {
        "path": str(path.resolve()),
        "iteration": checkpoint_iteration,
        "stage": checkpoint.get("stage"),
        "source_run_id": checkpoint.get("source_run_id"),
        "source_run_name": checkpoint.get("source_run_name"),
        "source_output_dir": checkpoint.get("source_output_dir"),
        "architecture_fingerprint": actual_fingerprint,
    }
