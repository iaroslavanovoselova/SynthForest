"""Sequential detector NMS search and raster benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from synthforest.training.benchmarking.config import (
    BenchmarkConfig,
    NmsSearchConfig,
)
from synthforest.training.benchmarking.nms_search import run_nms_search
from synthforest.training.benchmarking.runner import run_benchmark


@dataclass(frozen=True)
class NmsSearchBenchmarkResult:
    """Outputs from an NMS search followed by a raster benchmark."""

    search_output: Path
    optimal_parameters_path: Path
    benchmark_output: Path
    nms_threshold: float
    score_threshold: float


def run_nms_search_then_benchmark(
    search_config: NmsSearchConfig,
    benchmark_config: BenchmarkConfig,
) -> NmsSearchBenchmarkResult:
    """Optimize NMS on validation data, then raster-benchmark with the optimum."""
    # This pipeline is intended to extend existing prediction benchmarks. Compatible
    # saved predictions are still validated by the benchmark runner before reuse.
    search_config.reuse_predictions = True
    benchmark_config.reuse_predictions = True
    search_config.validate()
    benchmark_config.validate()
    _validate_shared_checkpoint(search_config, benchmark_config)

    print("Phase 1/2: searching validation raster NMS parameters.")
    search_output = run_nms_search(search_config)
    optimal_parameters_path = search_output / "optimal_nms.yaml"
    optimal_parameters = _load_optimal_parameters(optimal_parameters_path)

    benchmark_config.nms_threshold = optimal_parameters["nms_threshold"]
    benchmark_config.score_threshold = optimal_parameters["score_threshold"]
    benchmark_config.tile_only = False
    benchmark_config.validate()

    print(
        "Phase 2/2: running raster benchmark with "
        f"NMS IoU={benchmark_config.nms_threshold} and "
        f"score={benchmark_config.score_threshold}."
    )
    benchmark_output = run_benchmark(benchmark_config)
    return NmsSearchBenchmarkResult(
        search_output=search_output,
        optimal_parameters_path=optimal_parameters_path,
        benchmark_output=benchmark_output,
        nms_threshold=benchmark_config.nms_threshold,
        score_threshold=benchmark_config.score_threshold,
    )


def _validate_shared_checkpoint(
    search_config: NmsSearchConfig,
    benchmark_config: BenchmarkConfig,
) -> None:
    if search_config.model_type != benchmark_config.model_type:
        raise ValueError(
            "NMS search and benchmark model_type values must match: "
            f"{search_config.model_type!r} != {benchmark_config.model_type!r}."
        )
    search_checkpoint = search_config.resolved_checkpoint_path().resolve()
    benchmark_checkpoint = benchmark_config.resolved_checkpoint_path().resolve()
    if search_checkpoint != benchmark_checkpoint:
        raise ValueError(
            "NMS search and benchmark must use the same checkpoint: "
            f"{search_checkpoint} != {benchmark_checkpoint}."
        )


def _load_optimal_parameters(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"NMS search did not write optimal parameters: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload: Any = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid optimal NMS parameter file: {path}")

    parameters: dict[str, float] = {}
    for name in ("nms_threshold", "score_threshold"):
        if name not in payload:
            raise ValueError(f"Optimal NMS parameter file is missing {name!r}: {path}")
        try:
            value = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Optimal NMS parameter {name!r} is not numeric: {path}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Optimal NMS parameter {name!r} must be in [0, 1], got {value}.")
        parameters[name] = value
    return parameters
