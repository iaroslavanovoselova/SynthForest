#!/usr/bin/env python3
"""CanopyRS-adapted validation search for detector raster NMS parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training.benchmarking import NmsSearchConfig, run_nms_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find optimal raster NMS IoU and score thresholds on validation datasets. "
            "Saved compatible benchmark predictions are reused automatically."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--run-folder", type=Path, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prediction-source", type=Path, default=None)
    parser.add_argument("--real-data-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--objective-iou-threshold", type=str, default=None)
    parser.add_argument("--search-name", type=str, default=None)
    return parser.parse_args()


def _apply_overrides(
    config: NmsSearchConfig,
    args: argparse.Namespace,
) -> None:
    if args.train_config is not None:
        config.train_config_path = str(args.train_config)
    if args.run_folder is not None:
        config.run_folder = str(args.run_folder)
        config.checkpoint_path = None
    if args.checkpoint_name is not None:
        config.checkpoint_name = args.checkpoint_name
    if args.checkpoint is not None:
        config.checkpoint_path = str(args.checkpoint)
        config.run_folder = None
    if args.prediction_source is not None:
        config.prediction_source = str(args.prediction_source)
    if args.real_data_root is not None:
        config.real_data_root = str(args.real_data_root)
    if args.datasets is not None:
        config.datasets = list(args.datasets)
    if args.n_workers is not None:
        config.n_workers = args.n_workers
    if args.objective_iou_threshold is not None:
        config.objective_iou_threshold = args.objective_iou_threshold
    if args.search_name is not None:
        config.search_name = args.search_name
    config.validate()


def main() -> None:
    args = parse_args()
    config = NmsSearchConfig.from_yaml(args.config, validate=False)
    _apply_overrides(config, args)
    output_dir = run_nms_search(config)
    print(f"NMS search completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
