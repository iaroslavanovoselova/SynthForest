#!/usr/bin/env python3
"""
Hydra-based Blender launcher for the tree scatter pipeline.

Example Hydra overrides passed by ``synthforest generate scenes``::

  output.output_dir=/tmp/tree_scatter_test \
  terrain.size=80 \
  scatter.scatter_density=0.5 \
  render.samples=32
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Keep Hydra overrides while removing Blender CLI args.
# In Blender, user script args are passed after "--".
_raw_argv = sys.argv[1:]
if "--" in _raw_argv:
    _raw_argv = _raw_argv[_raw_argv.index("--") + 1 :]
elif any(arg in {"--background", "--factory-startup", "--addons", "--python"} for arg in _raw_argv):
    # Running under Blender without explicit script overrides.
    _raw_argv = []
sys.argv = [sys.argv[0], *_raw_argv]

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

logger = logging.getLogger(__name__)


from synthforest.paths import project_root  # noqa: E402

PROJECT_ROOT = project_root()

from synthforest.tree_scatter import (  # noqa: E402
    run_tree_scatter_pipeline,
)
from synthforest.tree_scatter.run_naming import (  # noqa: E402
    resolve_log_file,
    resolve_output_dir,
    timestamp_token,
    tree_scatter_run_name,
)


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _configure_logging(log_file_path: Path) -> None:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def _prepare_run_paths(cfg: DictConfig) -> tuple[str, Path, Path]:
    stamp = timestamp_token()
    run_name = tree_scatter_run_name(cfg, stamp)
    append_run_name = bool(cfg.get("output", {}).get("append_run_name", True))

    configured_output = cfg.get("output", {}).get("output_dir", "outputs/tree_scatter")
    output_dir = _resolve_project_path(
        resolve_output_dir(configured_output, run_name, append_run_name)
    )
    cfg.output.output_dir = str(output_dir)
    cfg.output.run_name = run_name

    configured_log = cfg.get("logging", {}).get("file_path", "logs/tree_scatter")
    log_file_path = _resolve_project_path(resolve_log_file(configured_log, run_name))
    cfg.logging.file_path = str(log_file_path)
    return run_name, output_dir, log_file_path


@hydra.main(
    version_base=None,
    config_path="../../../configs/generation",
    config_name="tree_scatter_scene",
)
def main(cfg: DictConfig) -> None:
    try:
        run_name, output_dir, log_path = _prepare_run_paths(cfg)
        _configure_logging(log_path)
        logger.info("Run name: %s", run_name)
        logger.info("Log file: %s", log_path)
        logger.info("Output directory: %s", output_dir)
        run_tree_scatter_pipeline(cfg)
    except Exception as exc:
        logger.exception("Tree scatter pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
