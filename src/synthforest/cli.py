"""Unified command-line interface for SynthForest workflows."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, Sequence

from synthforest import __version__
from synthforest.paths import environment_status, project_root

CommandTarget = tuple[str, str]

COMMANDS: dict[tuple[str, ...], CommandTarget] = {
    # Synthetic data generation (requires Blender and the Botaniq library).
    ("generate", "asset-pool"): ("generate_asset_pool", "Create the Botaniq tree asset pool."),
    ("generate", "textures"): (
        "generate_textures",
        "Render Botaniq forest-floor terrain textures.",
    ),
    ("generate", "scenes"): ("generate_scenes", "Generate synthetic forest scenes."),
    # Synthetic data preparation.
    ("prepare", "downsample"): ("downsample_scenes", "Downsample rendered scenes (Syn-20)."),
    ("prepare", "tiles"): ("convert_scenes_to_tiles", "Convert scenes to training tiles."),
    ("prepare", "split"): ("split_tiles_by_scene", "Split tiles into train/valid by scene."),
    ("prepare", "height-maps"): (
        "precompute_detector_height_maps",
        "Precompute Syn-20 height maps for height fusion.",
    ),
    # Real benchmark data.
    ("data", "download", "neon"): ("download_neon_chm_rgb", "Download NEON height data."),
    ("data", "download", "canopyrs"): (
        "download_canopyrs_raster_gpkgs",
        "Download CanopyRS crown benchmark rasters.",
    ),
    ("data", "prepare", "france"): ("preprocess_france_height", "Prepare France height data."),
    ("data", "prepare", "neon"): ("resample_neon_chm_rgb_50cm", "Prepare NEON height data."),
    ("data", "prepare", "virtual-shizuoka"): (
        "preprocess_virtual_shizuoka_height",
        "Prepare Virtual Shizuoka height data.",
    ),
    ("data", "prepare", "sao-paulo"): (
        "validate_external_dataset",
        "Validate the prepared Sao Paulo raster layout.",
    ),
    ("data", "prepare", "references"): (
        "create_tile_subset",
        "Sample unlabeled real tiles for appearance adaptation.",
    ),
    ("data", "split", "height"): (
        "split_real_height_data_valid_test",
        "Create scene-level valid/test splits for real height data.",
    ),
    # Training and evaluation.
    ("train", "height"): ("train_dinov3_height", "Train canopy-height estimation."),
    ("train", "crowns"): ("train_dinov3_detector", "Train crown delineation."),
    ("finetune", "crowns"): ("finetune_dinov3_detector", "Fine-tune crown delineation."),
    ("evaluate", "height"): (
        "benchmark_dinov3_height_raster",
        "Evaluate canopy height on real rasters.",
    ),
    ("evaluate", "crowns"): ("benchmark_dinov3_detector", "Evaluate crown delineation."),
    ("evaluate", "crowns-nms"): (
        "find_optimal_and_benchmark_dinov3_detector",
        "Tune raster NMS on validation data, then evaluate.",
    ),
    ("evaluate", "crowns-nms-search"): (
        "find_optimal_dinov3_detector_nms",
        "Tune raster NMS without the final benchmark.",
    ),
    # Dataset release.
    ("publish", "dataset"): (
        "upload_synthetic_data_to_huggingface",
        "Upload the generated dataset to Hugging Face (dry run by default).",
    ),
}


def _usage() -> str:
    lines = ["usage: synthforest <command> [options]", "", "commands:", "  doctor"]
    for command, (_, description) in COMMANDS.items():
        lines.append(f"  {' '.join(command):<36} {description}")
    lines.extend(("", "Use 'synthforest <command> --help' for command-specific options."))
    return "\n".join(lines)


def doctor() -> int:
    """Print a side-effect-free environment and installation diagnostic."""

    print(f"SynthForest {__version__}")
    print(f"Project root: {project_root()}")
    print(f"Python: {sys.version.split()[0]}")
    missing = []
    for name, value in environment_status().items():
        display = value or "<not set>"
        print(f"{name}: {display}")
        if name != "SYNTHFOREST_BLENDER_BIN" and not value:
            missing.append(name)
    if missing:
        print(
            "Status: ready for metadata/config inspection; external data operations need the roots above."
        )
    else:
        print("Status: external roots configured.")
    return 0


def _match_command(argv: list[str]) -> tuple[tuple[str, ...], CommandTarget] | None:
    matches = [(key, target) for key, target in COMMANDS.items() if argv[: len(key)] == list(key)]
    return max(matches, key=lambda item: len(item[0])) if matches else None


def _dispatch(module_name: str, command: tuple[str, ...], args: list[str]) -> int:
    module = import_module(f"synthforest.commands.{module_name}")
    entry: Callable[[], object] = getattr(module, "main")
    old_argv = sys.argv
    old_cwd = Path.cwd()
    sys.argv = [f"synthforest {' '.join(command)}", *args]
    try:
        os.chdir(project_root())
        result = entry()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return int(result) if isinstance(result, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a SynthForest subcommand."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    if args[0] == "doctor":
        if len(args) > 1 and args[1] not in {"-h", "--help"}:
            print("doctor does not accept arguments", file=sys.stderr)
            return 2
        return doctor()

    matched = _match_command(args)
    if matched is None:
        print(f"Unknown command: {' '.join(args[:3])}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    command, (module_name, _) = matched
    return _dispatch(module_name, command, args[len(command) :])


if __name__ == "__main__":
    raise SystemExit(main())
