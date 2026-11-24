"""DINOv3 detector construction and trainability."""

from synthforest.training.models.detector import (
    DinoV3FeatureBackbone,
    FakeDinoV3Backbone,
    TrainEvalResizeTransform,
    build_dinov3_detector,
)
from synthforest.training.models.height import (
    DinoV3DPTHeightModel,
    build_dinov3_height_model,
)
from synthforest.training.models.trainability import (
    assert_dinov3_frozen,
    assert_dinov3_trainability,
    assert_stage_trainability,
    set_stage_trainability,
    summarize_trainable_parameters,
)

__all__ = [
    "DinoV3FeatureBackbone",
    "DinoV3DPTHeightModel",
    "FakeDinoV3Backbone",
    "TrainEvalResizeTransform",
    "assert_dinov3_frozen",
    "assert_dinov3_trainability",
    "assert_stage_trainability",
    "build_dinov3_detector",
    "build_dinov3_height_model",
    "set_stage_trainability",
    "summarize_trainable_parameters",
]
