"""Launch SynthForest generation commands inside Blender.

Scene generation requires Blender and the commercial Botaniq tree library
(https://polygoniq.com/3d/botaniq/), which cannot be redistributed. All other
parts of the repository (data preparation, training, evaluation) run without
either of them. The checks below fail early with an explanatory message instead
of starting a Blender job that cannot succeed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path

from synthforest.paths import asset_path

BOTANIQ_MESSAGE = """\
SynthForest scene generation requires the Botaniq tree and grass library
(https://polygoniq.com/3d/botaniq/), which is a commercial asset pack and is
not distributed with this repository.

Expected location: {path}

To generate scenes, obtain Botaniq and place it at $SYNTHFOREST_ASSET_ROOT/botaniq_full
(or pass {override}). See docs/generation.md for details.

The released SynthForest dataset can be used for training and evaluation
without Botaniq or Blender."""

# Command module -> which Botaniq-derived input it needs.
_BOTANIQ_REQUIREMENTS = {
    "create_botaniq_asset_pool": "pack",
    "render_botaniq_forest_floor_textures": "pack",
    "create_tree_scatter_scene": "pool",
}


def _override(args: Sequence[str], key: str) -> str | None:
    """Return the value of a Hydra ``key=value`` or argparse ``key value`` override."""

    for index, arg in enumerate(args):
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
        if arg == key and index + 1 < len(args):
            return args[index + 1]
    return None


def botaniq_problem(module_name: str, args: Sequence[str]) -> str | None:
    """Return a user-facing message if the Botaniq input for a command is missing."""

    requirement = _BOTANIQ_REQUIREMENTS.get(module_name)
    if requirement is None:
        return None

    tree_source = _override(args, "scatter.tree_source_mode") or "botaniq_asset_pool"
    if requirement == "pool" and tree_source == "botaniq_asset_pool":
        pool_dir = Path(
            _override(args, "scatter.asset_pool_dir") or asset_path("botaniq_tree_asset_pool")
        )
        if (pool_dir / "manifest.json").is_file():
            return None
        return (
            BOTANIQ_MESSAGE.format(path=pool_dir, override="scatter.asset_pool_dir=...")
            + "\n\nThe asset pool is built from Botaniq with `synthforest generate asset-pool`."
        )

    pack_key = (
        "--pack-dir"
        if module_name == "render_botaniq_forest_floor_textures"
        else ("scatter.botaniq_pack_dir")
    )
    pack_dir = Path(_override(args, pack_key) or asset_path("botaniq_full"))
    if pack_dir.is_dir() and any(pack_dir.iterdir()):
        return None
    override = f"{pack_key} ..." if pack_key.startswith("--") else f"{pack_key}=..."
    return BOTANIQ_MESSAGE.format(path=pack_dir, override=override)


def launch_blender(module_name: str, args: Sequence[str]) -> int:
    """Execute a SynthForest command as a Blender background Python script."""

    if any(arg in {"-h", "--help"} for arg in args):
        print(
            "Generation commands run inside Blender (set $SYNTHFOREST_BLENDER_BIN).\n"
            "Pass Hydra overrides as KEY=VALUE arguments, for example:\n\n"
            "  synthforest generate scenes generation.num_scenes=1 "
            "generation.target_profile_weights.large_closed_canopy=1\n\n"
            "See docs/generation.md for the full workflow."
        )
        return 0

    problem = botaniq_problem(module_name, args)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1

    blender_value = os.environ.get("SYNTHFOREST_BLENDER_BIN")
    if not blender_value or not Path(blender_value).expanduser().is_file():
        print(
            "Scene generation runs inside Blender 4.1. Set SYNTHFOREST_BLENDER_BIN to the "
            f"Blender executable (currently: {blender_value or '<not set>'}).",
            file=sys.stderr,
        )
        return 1
    blender = Path(blender_value).expanduser()

    qualified_name = f"synthforest.commands.{module_name}"
    spec = find_spec(qualified_name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"Cannot locate Blender command: {qualified_name}")
    script = Path(spec.origin).resolve()

    child_environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(source_root)
    )
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        *[arg for arg in args if arg != "--"],
    ]
    return subprocess.run(command, env=child_environment, check=False).returncode
