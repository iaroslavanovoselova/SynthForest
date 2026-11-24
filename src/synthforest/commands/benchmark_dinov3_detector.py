#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.training.benchmarking import BenchmarkConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a trained DINOv3 Faster R-CNN or Mask R-CNN checkpoint "
            "on CanopyRS-style real data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a detector benchmark YAML config.",
    )
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--run-folder", type=Path, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--real-data-root", type=Path, default=None)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--benchmark-name", type=str, default=None)
    parser.add_argument("--prediction-source", type=Path, default=None)
    parser.add_argument("--no-reuse-predictions", action="store_true")
    parser.add_argument("--no-resume-existing-benchmark", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--fold-name",
        choices=["train", "valid", "test"],
        default=None,
    )
    parser.add_argument("--tile-only", action="store_true")
    parser.add_argument("--raster-level", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-threshold", type=float, default=None)
    parser.add_argument("--eval-iou-threshold", type=str, default=None)
    parser.add_argument("--raster-eval-iou-threshold", type=str, default=None)
    parser.add_argument("--require-raster-inputs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    return parser.parse_args()


def _apply_overrides(
    config: BenchmarkConfig,
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
    if args.real_data_root is not None:
        config.real_data_root = str(args.real_data_root)
    if args.output_folder is not None:
        config.output_folder = str(args.output_folder)
    if args.benchmark_name is not None:
        config.benchmark_name = args.benchmark_name
    if args.prediction_source is not None:
        config.prediction_source = str(args.prediction_source)
    if args.no_reuse_predictions:
        config.reuse_predictions = False
    if args.no_resume_existing_benchmark:
        config.resume_existing_benchmark = False
        config.wandb.resume_existing = False
    if args.datasets is not None:
        config.datasets = list(args.datasets)
    if args.fold_name is not None:
        config.fold_name = args.fold_name
    if args.tile_only:
        config.tile_only = True
    if args.raster_level:
        config.tile_only = False
    if args.score_threshold is not None:
        config.score_threshold = args.score_threshold
    if args.nms_threshold is not None:
        config.nms_threshold = args.nms_threshold
    if args.eval_iou_threshold is not None:
        config.eval_iou_threshold = args.eval_iou_threshold
    if args.raster_eval_iou_threshold is not None:
        config.raster_eval_iou_threshold = args.raster_eval_iou_threshold
    if args.require_raster_inputs:
        config.require_raster_inputs = True
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device
    if args.max_products is not None:
        config.max_products = args.max_products
    if args.max_tiles is not None:
        config.max_tiles = args.max_tiles
    config.validate()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig.from_yaml(args.config, validate=False)
    _apply_overrides(config, args)
    output_dir = run_benchmark(config)
    print(f"Benchmark completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
