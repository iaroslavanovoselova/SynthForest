#!/usr/bin/env python3
"""Hydra launcher to downsample generated scenes for canopy-height estimation."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from synthforest.postprocessing.downsample_scenes import (
    DownsampleSceneOptions,
    downsample_scenes,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    root_logger.addHandler(stream_handler)


def _options_from_cfg(cfg: DictConfig) -> DownsampleSceneOptions:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Downsampling config must resolve to a mapping.")

    source_root = data.get("source_root")
    if not source_root:
        raise ValueError("source_root must be set to a generated scene output directory.")

    output_root = data.get("output_root")
    return DownsampleSceneOptions(
        source_root=Path(str(source_root)),
        output_root=Path(str(output_root)) if output_root else None,
        source_gsd=float(data.get("source_gsd", 0.05)),
        target_gsd=float(data.get("target_gsd", 0.50)),
        height_reducer=str(data.get("height_reducer", "p95_positive")),
        min_canopy_fraction=float(data.get("min_canopy_fraction", 0.10)),
        edge_policy=str(data.get("edge_policy", "center_crop")),
        images_subdir=str(data.get("images_subdir", "images")),
        height_maps_subdir=str(data.get("height_maps_subdir", "height_maps")),
        canopy_coverage_subdir=str(data.get("canopy_coverage_subdir", "canopy_coverage_maps")),
        manifest_name=str(data.get("manifest_name", "downsample_manifest.json")),
        overwrite=bool(data.get("overwrite", True)),
    )


def _resolved_output_root(options: DownsampleSceneOptions) -> Path:
    if options.output_root is not None:
        return Path(options.output_root).expanduser().resolve()
    source_root = Path(options.source_root).expanduser().resolve()
    centimeters = int(round(float(options.target_gsd) * 100.0))
    return source_root.parent / f"{source_root.name}_{centimeters}cm"


def _resolved_config_payload(
    options: DownsampleSceneOptions,
    output_root: Path,
) -> dict[str, object]:
    return {
        "source_root": str(Path(options.source_root).expanduser().resolve()),
        "output_root": str(output_root),
        "source_gsd": float(options.source_gsd),
        "target_gsd": float(options.target_gsd),
        "height_reducer": str(options.height_reducer),
        "min_canopy_fraction": float(options.min_canopy_fraction),
        "edge_policy": str(options.edge_policy),
        "images_subdir": str(options.images_subdir),
        "height_maps_subdir": str(options.height_maps_subdir),
        "canopy_coverage_subdir": str(options.canopy_coverage_subdir),
        "manifest_name": str(options.manifest_name),
        "overwrite": bool(options.overwrite),
    }


def _save_resolved_config(output_root: Path, options: DownsampleSceneOptions) -> Path:
    config_path = output_root / "config" / "resolved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=OmegaConf.create(_resolved_config_payload(options, output_root)),
        f=config_path,
    )
    return config_path


@hydra.main(
    version_base=None,
    config_path="../../../configs/generation",
    config_name="downsample_scenes",
)
def main(cfg: DictConfig) -> None:
    try:
        _configure_logging()
        options = _options_from_cfg(cfg)
        output_root = _resolved_output_root(options)
        resolved_config_path = _save_resolved_config(output_root, options)
        logger.info("Resolved downsampling config saved to: %s", resolved_config_path)
        manifest_path = downsample_scenes(options)
        logger.info("Done. Manifest: %s", manifest_path)
    except Exception as exc:
        logger.exception("Scene downsampling failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
