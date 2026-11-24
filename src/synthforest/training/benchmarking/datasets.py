"""Compatibility exports for the shared real-dataset registry."""

from synthforest.training.real_datasets import (
    DATASET_REGISTRY,
    PreprocessedDatasetSpec,
    ProductFold,
    get_dataset_spec,
)

__all__ = [
    "DATASET_REGISTRY",
    "PreprocessedDatasetSpec",
    "ProductFold",
    "get_dataset_spec",
]
