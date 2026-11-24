"""Detector validation and standalone checkpoint evaluation."""

from synthforest.training.evaluation.aggregation import macro_average_metrics
from synthforest.training.evaluation.runner import (
    evaluate_checkpoint,
    evaluate_model,
)

__all__ = ["evaluate_checkpoint", "evaluate_model", "macro_average_metrics"]
