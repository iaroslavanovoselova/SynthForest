#!/usr/bin/env python3
"""Find detector NMS parameters, then run the raster benchmark with them."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training.benchmarking import (
    BenchmarkConfig,
    NmsSearchConfig,
    run_nms_search_then_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search validation raster NMS parameters and then run a raster-level "
            "benchmark with the optimal parameters. Compatible predictions are "
            "reused automatically."
        )
    )
    parser.add_argument(
        "--nms-search-config",
        type=Path,
        required=True,
        help="Path to a detector NMS-search YAML config.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        required=True,
        help="Path to the final detector benchmark YAML config.",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--run-folder", type=Path, default=None)
    checkpoint.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--nms-prediction-source", type=Path, default=None)
    parser.add_argument("--benchmark-prediction-source", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    return parser.parse_args()


def _apply_checkpoint_overrides(
    config: BenchmarkConfig,
    args: argparse.Namespace,
) -> None:
    if args.run_folder is not None:
        config.run_folder = str(args.run_folder)
        config.checkpoint_path = None
    if args.checkpoint is not None:
        config.checkpoint_path = str(args.checkpoint)
        config.run_folder = None
    if args.checkpoint_name is not None:
        config.checkpoint_name = args.checkpoint_name


def main() -> None:
    args = parse_args()
    search_config = NmsSearchConfig.from_yaml(
        args.nms_search_config,
        validate=False,
    )
    benchmark_config = BenchmarkConfig.from_yaml(
        args.benchmark_config,
        validate=False,
    )
    _apply_checkpoint_overrides(search_config, args)
    _apply_checkpoint_overrides(benchmark_config, args)
    if args.nms_prediction_source is not None:
        search_config.prediction_source = str(args.nms_prediction_source)
    if args.benchmark_prediction_source is not None:
        benchmark_config.prediction_source = str(args.benchmark_prediction_source)
    if args.n_workers is not None:
        search_config.n_workers = args.n_workers

    result = run_nms_search_then_benchmark(search_config, benchmark_config)
    print("Sequential NMS search and raster benchmark completed.")
    print(f"NMS search output: {result.search_output}")
    print(f"Optimal parameters: {result.optimal_parameters_path}")
    print(f"Selected thresholds: NMS IoU={result.nms_threshold}, score={result.score_threshold}")
    print(f"Raster benchmark output: {result.benchmark_output}")


if __name__ == "__main__":
    main()
