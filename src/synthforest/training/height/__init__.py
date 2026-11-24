"""DINOv3 height-estimation training, evaluation, and benchmarking."""

from synthforest.training.height.benchmarking import run_height_benchmark
from synthforest.training.height.detector_precompute import (
    run_detector_height_precompute,
    run_detector_height_precompute_for_coco,
)
from synthforest.training.height.evaluation import evaluate_height_checkpoint
from synthforest.training.height.raster_benchmarking import run_height_raster_benchmark
from synthforest.training.height.trainer import train_height_from_config

__all__ = [
    "evaluate_height_checkpoint",
    "run_detector_height_precompute",
    "run_detector_height_precompute_for_coco",
    "run_height_benchmark",
    "run_height_raster_benchmark",
    "train_height_from_config",
]
