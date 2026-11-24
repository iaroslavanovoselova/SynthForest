"""CanopyRS-style real-data benchmarking for DINOv3 detector checkpoints."""

from synthforest.training.benchmarking.config import (
    BenchmarkConfig,
    NmsSearchConfig,
    parse_iou_thresholds,
)

__all__ = [
    "BenchmarkConfig",
    "NmsSearchConfig",
    "parse_iou_thresholds",
    "run_benchmark",
    "run_nms_search",
    "run_nms_search_then_benchmark",
]


def run_benchmark(config: BenchmarkConfig):
    from synthforest.training.benchmarking.runner import run_benchmark as _run_benchmark

    return _run_benchmark(config)


def run_nms_search(config: NmsSearchConfig):
    from synthforest.training.benchmarking.nms_search import (
        run_nms_search as _run_nms_search,
    )

    return _run_nms_search(config)


def run_nms_search_then_benchmark(
    search_config: NmsSearchConfig,
    benchmark_config: BenchmarkConfig,
):
    from synthforest.training.benchmarking.pipeline import (
        run_nms_search_then_benchmark as _run_nms_search_then_benchmark,
    )

    return _run_nms_search_then_benchmark(search_config, benchmark_config)
