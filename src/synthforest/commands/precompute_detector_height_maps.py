#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synthforest.paths import project_root

PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute detector-aligned height maps from a trained DINOv3 DPT height checkpoint."
        )
    )
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--height-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--height-train-config",
        type=Path,
        default=None,
        help="Optional height training config override for relocated model/data paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source-gsd", type=float, default=0.05)
    parser.add_argument("--target-gsd", type=float, default=0.20)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from synthforest.training.height import run_detector_height_precompute

    summary = run_detector_height_precompute(
        detector_config_path=args.detector_config,
        height_checkpoint_path=args.height_checkpoint,
        height_train_config_path=args.height_train_config,
        splits=args.splits,
        batch_size=args.batch_size,
        device=args.device,
        source_gsd=args.source_gsd,
        target_gsd=args.target_gsd,
        max_tiles=args.max_tiles,
        overwrite=args.overwrite,
        summary_path=args.summary_path,
    )
    print(
        "Detector height-map precompute completed. "
        f"processed={summary['num_processed']}, "
        f"skipped_existing={summary['num_skipped_existing']}, "
        f"failed={summary['num_failed']}. "
        f"Summary written to: {summary['summary_path']}"
    )


if __name__ == "__main__":
    main()
