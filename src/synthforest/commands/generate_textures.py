"""Launch Botaniq forest-floor texture rendering through Blender."""

from __future__ import annotations

import sys

from synthforest.commands.blender_launcher import launch_blender


def main() -> int:
    return launch_blender("render_botaniq_forest_floor_textures", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
