#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training.config import HeightTrainingConfig
from synthforest.training.height import train_height_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stage-configured DINOv3 DPT canopy-height regression model."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to training YAML config.")
    parser.add_argument(
        "--train-dataset",
        dest="train_datasets",
        action="append",
        default=None,
        metavar="NAME",
        help="Override train_dataset_names with this dataset. Repeat for multiple datasets.",
    )
    parser.add_argument("--run-name", default=None, help="Override run_name from the YAML config.")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=None,
        help="Run a post-training height benchmark with this YAML after training completes.",
    )
    parser.add_argument(
        "--benchmark-checkpoint-name",
        default=None,
        help="Checkpoint filename under checkpoints/ for post-training benchmark.",
    )
    parser.add_argument(
        "--no-post-training-benchmark",
        action="store_true",
        help="Disable post-training benchmarking even if the YAML config enables it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HeightTrainingConfig.from_yaml(args.config)
    if args.train_datasets is not None:
        config.train_dataset_names = list(args.train_datasets)
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.benchmark_config is not None:
        config.post_training_benchmark.enabled = True
        config.post_training_benchmark.config_path = str(args.benchmark_config)
    if args.benchmark_checkpoint_name is not None:
        config.post_training_benchmark.checkpoint_name = args.benchmark_checkpoint_name
    if args.no_post_training_benchmark:
        config.post_training_benchmark.enabled = False
    output_dir = train_height_from_config(config)
    print(f"Height training completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
