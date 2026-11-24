"""Public API for DINOv3 crown detector and height workflows."""

from synthforest.training.config import DetectorTrainingConfig, HeightTrainingConfig
from synthforest.training.engine import train_from_config
from synthforest.training.evaluation import evaluate_checkpoint
from synthforest.training.height import evaluate_height_checkpoint, train_height_from_config
from synthforest.training.models import build_dinov3_detector, build_dinov3_height_model

__all__ = [
    "DetectorTrainingConfig",
    "HeightTrainingConfig",
    "build_dinov3_detector",
    "build_dinov3_height_model",
    "evaluate_checkpoint",
    "evaluate_height_checkpoint",
    "train_height_from_config",
    "train_from_config",
]
