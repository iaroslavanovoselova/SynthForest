"""Launch synthetic scene generation through Blender."""

from __future__ import annotations

import sys

from synthforest.commands.blender_launcher import launch_blender


def main() -> int:
    return launch_blender("create_tree_scatter_scene", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
