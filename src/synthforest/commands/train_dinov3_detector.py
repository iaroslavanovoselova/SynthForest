#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training import DetectorTrainingConfig, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a stage-configured DINOv3 Faster R-CNN or Mask R-CNN "
            "crown detector with a frozen backbone or LoRA adapters."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_datasets",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Override train_dataset_names with this dataset. Repeat the option "
            "to train on multiple datasets."
        ),
    )
    parser.add_argument(
        "--train-data-fraction",
        type=float,
        default=None,
        help="Override train_sampling.fraction.",
    )
    parser.add_argument(
        "--train-data-seed",
        type=int,
        default=None,
        help="Override train_sampling.seed.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override run_name from the YAML config.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=None,
        help=(
            "Run a post-training benchmark with this detector benchmark YAML "
            "after training completes."
        ),
    )
    parser.add_argument(
        "--benchmark-checkpoint-name",
        default=None,
        help=(
            "Checkpoint filename or relative path under the run's checkpoints/ "
            "folder for the post-training benchmark. Defaults to best.pth."
        ),
    )
    parser.add_argument(
        "--no-post-training-benchmark",
        action="store_true",
        help="Disable post-training benchmarking even if the YAML config enables it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectorTrainingConfig.from_yaml(args.config)
    if args.train_datasets is not None:
        config.train_dataset_names = list(args.train_datasets)
    if args.train_data_fraction is not None:
        config.train_sampling.fraction = args.train_data_fraction
    if args.train_data_seed is not None:
        config.train_sampling.seed = args.train_data_seed
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.benchmark_config is not None:
        config.post_training_benchmark.enabled = True
        config.post_training_benchmark.config_path = str(args.benchmark_config)
    if args.benchmark_checkpoint_name is not None:
        config.post_training_benchmark.checkpoint_name = args.benchmark_checkpoint_name
    if args.no_post_training_benchmark:
        config.post_training_benchmark.enabled = False
    config.validate()
    output_dir = train_from_config(config)
    print(f"Training completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
