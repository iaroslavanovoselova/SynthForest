#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training.config import HeightRasterBenchmarkConfig
from synthforest.training.height import run_height_raster_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window full-raster inference for a trained DINOv3 DPT height checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--run-folder", type=Path, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rgb-root", type=Path, default=None)
    parser.add_argument("--height-root", type=Path, default=None)
    parser.add_argument("--rgb-glob", type=str, default=None)
    parser.add_argument("--height-glob", type=str, default=None)
    parser.add_argument("--rgb-name-suffix", type=str, default=None)
    parser.add_argument("--height-name-suffix", type=str, default=None)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--benchmark-name", type=str, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-rasters", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--save-weight-maps", action="store_true")
    return parser.parse_args()


def _apply_overrides(config: HeightRasterBenchmarkConfig, args: argparse.Namespace) -> None:
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
    if args.rgb_root is not None:
        config.rgb_root_path = str(args.rgb_root)
    if args.height_root is not None:
        config.height_root_path = str(args.height_root)
    if args.rgb_glob is not None:
        config.rgb_glob = args.rgb_glob
    if args.height_glob is not None:
        config.height_glob = args.height_glob
    if args.rgb_name_suffix is not None:
        config.rgb_name_suffix = args.rgb_name_suffix
    if args.height_name_suffix is not None:
        config.height_name_suffix = args.height_name_suffix
    if args.output_folder is not None:
        config.output_folder = str(args.output_folder)
    if args.benchmark_name is not None:
        config.benchmark_name = args.benchmark_name
    if args.tile_size is not None:
        config.tile_size = args.tile_size
    if args.stride is not None:
        config.stride = args.stride
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.max_rasters is not None:
        config.max_rasters = args.max_rasters
    if args.save_predictions:
        config.save_predictions = True
    if args.no_save_predictions:
        config.save_predictions = False
    if args.save_weight_maps:
        config.save_weight_maps = True
    config.validate()


def main() -> None:
    args = parse_args()
    config = HeightRasterBenchmarkConfig.from_yaml(args.config, validate=False)
    _apply_overrides(config, args)
    output_dir = run_height_raster_benchmark(config)
    print(f"Height raster benchmark completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
