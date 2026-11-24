#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training import DetectorTrainingConfig, train_from_config
from synthforest.training.config import CheckpointInitializationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a DINOv3 Faster R-CNN or Mask R-CNN detector from an "
            "existing full checkpoint on a sampled fraction of real data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to fine-tuning YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint_initialization.checkpoint_path.",
    )
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=None,
        help=(
            "Override the real training data root. Also overrides validation.real.data_root_path."
        ),
    )
    parser.add_argument(
        "--real-dataset",
        dest="real_datasets",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Override train_dataset_names with this real dataset folder name. "
            "Repeat the option to fine-tune on multiple datasets."
        ),
    )
    parser.add_argument(
        "--real-data-fraction",
        type=float,
        default=None,
        help="Override train_sampling.fraction.",
    )
    parser.add_argument(
        "--real-data-seed",
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
            "after fine-tuning completes."
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
    if args.checkpoint is not None:
        config.checkpoint_initialization = CheckpointInitializationConfig(
            checkpoint_path=str(args.checkpoint)
        )
    if config.checkpoint_initialization is None:
        raise ValueError(
            "Fine-tuning requires checkpoint_initialization.checkpoint_path "
            "in the config or --checkpoint."
        )
    if args.real_data_root is not None:
        config.data_root_path = str(args.real_data_root)
        config.validation.real.data_root_path = str(args.real_data_root)
    if args.real_datasets is not None:
        config.train_dataset_names = list(args.real_datasets)
    if args.real_data_fraction is not None:
        config.train_sampling.fraction = args.real_data_fraction
    if args.real_data_seed is not None:
        config.train_sampling.seed = args.real_data_seed
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
    print(f"Fine-tuning completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
