from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.dataset import (
    SUPPORTED_IMAGE_SUFFIXES,
    resolve_height_path_for_tile,
    resolve_tile_path,
)
from synthforest.training.height.checkpoint import load_height_checkpoint_model
from synthforest.training.height.dataset import image_to_normalized_tensor
from synthforest.training.valid_masks import fill_invalid_pixels, load_image_with_valid_mask

DETECTOR_HEIGHT_PRECOMPUTE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DetectorHeightTileRecord:
    dataset_name: str
    product_name: str
    split: str
    image_path: Path
    output_path: Path


@dataclass(frozen=True)
class DetectorHeightPreparedTile:
    record: DetectorHeightTileRecord
    tensor: torch.Tensor
    valid_mask: np.ndarray
    original_size: tuple[int, int]
    low_size: tuple[int, int]
    padded_size: tuple[int, int]
    crop_left: int
    crop_top: int


def run_detector_height_precompute(
    *,
    height_checkpoint_path: str | Path,
    detector_config_path: str | Path | None = None,
    detector_config: DetectorTrainingConfig | None = None,
    height_train_config_path: str | Path | None = None,
    splits: Iterable[str] = ("train", "valid"),
    batch_size: int = 1,
    device: str | torch.device | None = None,
    source_gsd: float = 0.05,
    target_gsd: float = 0.20,
    max_tiles: Optional[int] = None,
    overwrite: bool = False,
    summary_path: str | Path | None = None,
) -> dict:
    if detector_config is None:
        if detector_config_path is None:
            raise ValueError("Pass detector_config or detector_config_path.")
        detector_config = DetectorTrainingConfig.from_yaml(detector_config_path)
    elif detector_config_path is not None:
        raise ValueError("Pass detector_config or detector_config_path, not both.")

    splits = normalize_detector_height_splits(splits)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_tiles is not None and int(max_tiles) <= 0:
        raise ValueError("max_tiles must be positive when set.")
    _validate_gsds(source_gsd=source_gsd, target_gsd=target_gsd)

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, height_config, checkpoint_metadata = load_height_checkpoint_model(
        height_checkpoint_path,
        device=resolved_device,
        config_path=height_train_config_path,
    )
    model.eval()
    required_output_bands = _required_detector_height_output_bands(height_config)
    precompute_match_checkpoint = bool(detector_config.height_fusion.precompute_match_checkpoint)
    producer_tags = _detector_height_precompute_tags(
        height_checkpoint_path=height_checkpoint_path,
        height_train_config_path=height_train_config_path,
        required_output_bands=required_output_bands,
        source_gsd=source_gsd,
        target_gsd=target_gsd,
    )

    records = resolve_detector_height_tile_records(detector_config, splits=splits)
    if max_tiles is not None:
        records = records[: int(max_tiles)]
    if not records:
        raise ValueError(
            "No detector tiles were found for height-map precompute. "
            f"splits={splits}, data_root_path={detector_config.data_root_path!r}."
        )

    stats = {
        "num_records": len(records),
        "num_processed": 0,
        "num_skipped_existing": 0,
        "num_failed": 0,
        "failures": [],
    }
    pending: list[DetectorHeightPreparedTile] = []
    pending_shape: tuple[int, int] | None = None

    def flush_batch() -> None:
        nonlocal pending, pending_shape
        if not pending:
            return
        batch = torch.stack([item.tensor for item in pending], dim=0).to(resolved_device)
        try:
            with torch.no_grad():
                predictions, log_variances = _predict_detector_height_outputs(
                    model,
                    batch,
                    required_output_bands=required_output_bands,
                )
        except Exception as exc:
            for item in pending:
                stats["num_failed"] += 1
                stats["failures"].append(
                    {
                        "image_path": str(item.record.image_path),
                        "output_path": str(item.record.output_path),
                        "error": str(exc),
                    }
                )
            pending = []
            pending_shape = None
            return
        for sample_index, item in enumerate(pending):
            try:
                array = _prediction_to_detector_grid(predictions[sample_index], item)
                log_variance_array = (
                    _prediction_to_detector_grid(log_variances[sample_index], item)
                    if log_variances is not None
                    else None
                )
                save_detector_height_tile(
                    source_path=item.record.image_path,
                    output_path=item.record.output_path,
                    prediction=array,
                    log_variance=log_variance_array,
                    metadata_tags=producer_tags,
                )
                stats["num_processed"] += 1
            except Exception as exc:
                stats["num_failed"] += 1
                stats["failures"].append(
                    {
                        "image_path": str(item.record.image_path),
                        "output_path": str(item.record.output_path),
                        "error": str(exc),
                    }
                )
        pending = []
        pending_shape = None

    for record in records:
        if (
            record.output_path.exists()
            and not overwrite
            and _detector_height_tile_is_complete(
                record.output_path,
                required_output_bands,
                expected_tags=producer_tags if precompute_match_checkpoint else None,
            )
        ):
            stats["num_skipped_existing"] += 1
            continue
        try:
            prepared = prepare_detector_height_tile(
                record,
                height_config=height_config,
                source_gsd=source_gsd,
                target_gsd=target_gsd,
            )
        except Exception as exc:
            stats["num_failed"] += 1
            stats["failures"].append(
                {
                    "image_path": str(record.image_path),
                    "output_path": str(record.output_path),
                    "error": str(exc),
                }
            )
            continue

        shape = tuple(int(value) for value in prepared.tensor.shape[-2:])
        if pending and shape != pending_shape:
            flush_batch()
        pending.append(prepared)
        pending_shape = shape
        if len(pending) >= batch_size:
            flush_batch()
    flush_batch()

    if summary_path is None:
        summary_path = Path(detector_config.data_root_path) / "height_maps_precompute_summary.json"
    summary_path = Path(summary_path)
    summary = {
        "detector_config_path": str(detector_config_path) if detector_config_path else None,
        "height_checkpoint_path": str(Path(height_checkpoint_path).expanduser().resolve()),
        "height_train_config_path": (
            str(Path(height_train_config_path).expanduser().resolve())
            if height_train_config_path is not None
            else None
        ),
        "checkpoint": checkpoint_metadata,
        "splits": splits,
        "source_gsd_m": float(source_gsd),
        "target_gsd_m": float(target_gsd),
        "downsample_scale": float(source_gsd) / float(target_gsd),
        "output_mode": "detector_aligned",
        "output_bands": required_output_bands,
        "precompute_match_checkpoint": precompute_match_checkpoint,
        "producer_tags": producer_tags,
        "overwrite": bool(overwrite),
        "batch_size": batch_size,
        "device": str(resolved_device),
        **stats,
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def run_detector_height_precompute_for_coco(
    *,
    tile_dir: str | Path,
    truth_coco_path: str | Path,
    split: str,
    height_maps_subdir: str,
    height_checkpoint_path: str | Path,
    height_train_config_path: str | Path | None = None,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    source_gsd: float = 0.05,
    target_gsd: float = 0.20,
    overwrite: bool = False,
    precompute_match_checkpoint: bool = True,
    summary_path: str | Path | None = None,
) -> dict:
    records = resolve_detector_height_tile_records_from_coco(
        tile_dir=tile_dir,
        truth_coco_path=truth_coco_path,
        split=split,
        height_maps_subdir=height_maps_subdir,
    )
    if summary_path is None:
        summary_path = (
            Path(tile_dir).parent.parent
            / f"height_maps_precompute_{str(split).lower()}_summary.json"
        )
    return run_detector_height_precompute_records(
        records=records,
        height_checkpoint_path=height_checkpoint_path,
        height_train_config_path=height_train_config_path,
        batch_size=batch_size,
        device=device,
        source_gsd=source_gsd,
        target_gsd=target_gsd,
        overwrite=overwrite,
        precompute_match_checkpoint=precompute_match_checkpoint,
        summary_path=summary_path,
        context={
            "tile_dir": str(Path(tile_dir)),
            "truth_coco_path": str(Path(truth_coco_path)),
            "split": str(split),
            "height_maps_subdir": str(height_maps_subdir),
        },
    )


def run_detector_height_precompute_records(
    *,
    records: Iterable[DetectorHeightTileRecord],
    height_checkpoint_path: str | Path,
    height_train_config_path: str | Path | None = None,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    source_gsd: float = 0.05,
    target_gsd: float = 0.20,
    overwrite: bool = False,
    precompute_match_checkpoint: bool = True,
    summary_path: str | Path | None = None,
    context: Optional[dict] = None,
) -> dict:
    records = list(records)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    _validate_gsds(source_gsd=source_gsd, target_gsd=target_gsd)
    if summary_path is None:
        summary_path = Path("height_maps_precompute_summary.json")
    summary_path = Path(summary_path)

    stats = {
        "num_records": len(records),
        "num_processed": 0,
        "num_skipped_existing": 0,
        "num_failed": 0,
        "failures": [],
    }
    if not records:
        summary = {
            "height_checkpoint_path": str(Path(height_checkpoint_path).expanduser().resolve()),
            "height_train_config_path": (
                str(Path(height_train_config_path).expanduser().resolve())
                if height_train_config_path is not None
                else None
            ),
            "source_gsd_m": float(source_gsd),
            "target_gsd_m": float(target_gsd),
            "downsample_scale": float(source_gsd) / float(target_gsd),
            "output_mode": "detector_aligned",
            "overwrite": bool(overwrite),
            "precompute_match_checkpoint": bool(precompute_match_checkpoint),
            "batch_size": batch_size,
            "device": str(device) if device is not None else None,
            **(context or {}),
            **stats,
            "summary_path": str(summary_path),
        }
        _write_json(summary_path, summary)
        return summary

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, height_config, checkpoint_metadata = load_height_checkpoint_model(
        height_checkpoint_path,
        device=resolved_device,
        config_path=height_train_config_path,
    )
    model.eval()
    required_output_bands = _required_detector_height_output_bands(height_config)
    producer_tags = _detector_height_precompute_tags(
        height_checkpoint_path=height_checkpoint_path,
        height_train_config_path=height_train_config_path,
        required_output_bands=required_output_bands,
        source_gsd=source_gsd,
        target_gsd=target_gsd,
    )

    pending: list[DetectorHeightPreparedTile] = []
    pending_shape: tuple[int, int] | None = None

    def flush_batch() -> None:
        nonlocal pending, pending_shape
        if not pending:
            return
        batch = torch.stack([item.tensor for item in pending], dim=0).to(resolved_device)
        try:
            with torch.no_grad():
                predictions, log_variances = _predict_detector_height_outputs(
                    model,
                    batch,
                    required_output_bands=required_output_bands,
                )
        except Exception as exc:
            for item in pending:
                stats["num_failed"] += 1
                stats["failures"].append(
                    {
                        "image_path": str(item.record.image_path),
                        "output_path": str(item.record.output_path),
                        "error": str(exc),
                    }
                )
            pending = []
            pending_shape = None
            return
        for sample_index, item in enumerate(pending):
            try:
                array = _prediction_to_detector_grid(predictions[sample_index], item)
                log_variance_array = (
                    _prediction_to_detector_grid(log_variances[sample_index], item)
                    if log_variances is not None
                    else None
                )
                save_detector_height_tile(
                    source_path=item.record.image_path,
                    output_path=item.record.output_path,
                    prediction=array,
                    log_variance=log_variance_array,
                    metadata_tags=producer_tags,
                )
                stats["num_processed"] += 1
            except Exception as exc:
                stats["num_failed"] += 1
                stats["failures"].append(
                    {
                        "image_path": str(item.record.image_path),
                        "output_path": str(item.record.output_path),
                        "error": str(exc),
                    }
                )
        pending = []
        pending_shape = None

    for record in records:
        if (
            record.output_path.exists()
            and not overwrite
            and _detector_height_tile_is_complete(
                record.output_path,
                required_output_bands,
                expected_tags=producer_tags if precompute_match_checkpoint else None,
            )
        ):
            stats["num_skipped_existing"] += 1
            continue
        try:
            prepared = prepare_detector_height_tile(
                record,
                height_config=height_config,
                source_gsd=source_gsd,
                target_gsd=target_gsd,
            )
        except Exception as exc:
            stats["num_failed"] += 1
            stats["failures"].append(
                {
                    "image_path": str(record.image_path),
                    "output_path": str(record.output_path),
                    "error": str(exc),
                }
            )
            continue

        shape = tuple(int(value) for value in prepared.tensor.shape[-2:])
        if pending and shape != pending_shape:
            flush_batch()
        pending.append(prepared)
        pending_shape = shape
        if len(pending) >= batch_size:
            flush_batch()
    flush_batch()

    summary = {
        "height_checkpoint_path": str(Path(height_checkpoint_path).expanduser().resolve()),
        "height_train_config_path": (
            str(Path(height_train_config_path).expanduser().resolve())
            if height_train_config_path is not None
            else None
        ),
        "checkpoint": checkpoint_metadata,
        "source_gsd_m": float(source_gsd),
        "target_gsd_m": float(target_gsd),
        "downsample_scale": float(source_gsd) / float(target_gsd),
        "output_mode": "detector_aligned",
        "output_bands": required_output_bands,
        "precompute_match_checkpoint": bool(precompute_match_checkpoint),
        "producer_tags": producer_tags,
        "overwrite": bool(overwrite),
        "batch_size": batch_size,
        "device": str(resolved_device),
        **(context or {}),
        **stats,
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def resolve_detector_height_tile_records_from_coco(
    *,
    tile_dir: str | Path,
    truth_coco_path: str | Path,
    split: str,
    height_maps_subdir: str,
) -> list[DetectorHeightTileRecord]:
    tile_dir = Path(tile_dir)
    truth_coco_path = Path(truth_coco_path)
    with truth_coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    records = []
    for image_info in coco.get("images", []):
        image_path = resolve_tile_path(tile_dir, image_info["file_name"])
        output_path = resolve_height_path_for_tile(
            image_path=image_path,
            fold=split,
            height_maps_subdir=height_maps_subdir,
        )
        records.append(
            DetectorHeightTileRecord(
                dataset_name=str(tile_dir.parent.parent.name),
                product_name=str(tile_dir.parent.parent.name),
                split=str(split),
                image_path=image_path,
                output_path=output_path,
            )
        )
    return records


def resolve_detector_height_tile_records_from_tile_dir(
    *,
    tile_dir: str | Path,
    split: str,
    height_maps_subdir: str,
    dataset_name: str = "",
    product_name: str = "",
) -> list[DetectorHeightTileRecord]:
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Detector tile directory does not exist: {tile_dir}")
    resolved_product_name = str(product_name or tile_dir.parent.parent.name)
    resolved_dataset_name = str(dataset_name or tile_dir.parent.parent.name)
    records = []
    seen_outputs: set[Path] = set()
    for image_path in sorted(tile_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        output_path = resolve_height_path_for_tile(
            image_path=image_path,
            fold=split,
            height_maps_subdir=height_maps_subdir,
        )
        resolved_output = output_path.resolve()
        if resolved_output in seen_outputs:
            continue
        seen_outputs.add(resolved_output)
        records.append(
            DetectorHeightTileRecord(
                dataset_name=resolved_dataset_name,
                product_name=resolved_product_name,
                split=str(split),
                image_path=image_path,
                output_path=output_path,
            )
        )
    return records


def resolve_detector_height_tile_records(
    config: DetectorTrainingConfig,
    *,
    splits: Iterable[str] = ("train", "valid"),
    dataset_names: Iterable[str] | None = None,
) -> list[DetectorHeightTileRecord]:
    records: list[DetectorHeightTileRecord] = []
    seen_outputs: set[Path] = set()
    normalized_splits = normalize_detector_height_splits(splits)
    for split in normalized_splits:
        for source_root, names in _dataset_sources_for_split(
            config,
            split=split,
            dataset_names=dataset_names,
        ):
            for dataset_name in names:
                dataset_root = _resolve_detector_dataset_root(source_root, dataset_name)
                for product_root in _find_detector_product_roots(dataset_root, split):
                    tile_dir = product_root / "tiles" / split
                    height_dir = product_root / config.height_fusion.height_maps_subdir / split
                    for image_path in sorted(tile_dir.iterdir()):
                        if (
                            not image_path.is_file()
                            or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
                        ):
                            continue
                        output_path = height_dir / f"{image_path.stem}.tif"
                        resolved_output = output_path.resolve()
                        if resolved_output in seen_outputs:
                            continue
                        seen_outputs.add(resolved_output)
                        records.append(
                            DetectorHeightTileRecord(
                                dataset_name=str(dataset_name),
                                product_name=product_root.name,
                                split=split,
                                image_path=image_path,
                                output_path=output_path,
                            )
                        )
    return records


def prepare_detector_height_tile(
    record: DetectorHeightTileRecord,
    *,
    height_config,
    source_gsd: float,
    target_gsd: float,
) -> DetectorHeightPreparedTile:
    _validate_gsds(source_gsd=source_gsd, target_gsd=target_gsd)
    image, valid_mask = load_image_with_valid_mask(record.image_path, height_config.valid_mask)
    image = fill_invalid_pixels(
        image.convert("RGB"),
        valid_mask,
        height_config.augmentation.image_mean,
    )
    original_size = image.size
    low_size = detector_height_low_size(
        original_size,
        source_gsd=source_gsd,
        target_gsd=target_gsd,
    )
    low_image = image.resize(low_size, Image.Resampling.BOX)
    low_mask = _resize_valid_mask(valid_mask, low_size)
    padded_image, padded_mask, crop_left, crop_top = _center_pad_image_and_mask(
        low_image,
        low_mask,
        multiple=int(height_config.dinov3_patch_size),
        image_mean=height_config.augmentation.image_mean,
    )
    tensor = image_to_normalized_tensor(
        padded_image,
        padded_mask,
        height_config.augmentation.image_mean,
        height_config.augmentation.image_std,
    )
    return DetectorHeightPreparedTile(
        record=record,
        tensor=tensor,
        valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
        original_size=original_size,
        low_size=low_size,
        padded_size=padded_image.size,
        crop_left=crop_left,
        crop_top=crop_top,
    )


def detector_height_low_size(
    original_size: tuple[int, int],
    *,
    source_gsd: float,
    target_gsd: float,
) -> tuple[int, int]:
    _validate_gsds(source_gsd=source_gsd, target_gsd=target_gsd)
    width, height = (int(original_size[0]), int(original_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid detector tile size: {original_size}.")
    scale = float(source_gsd) / float(target_gsd)
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def save_detector_height_tile(
    *,
    source_path: str | Path,
    output_path: str | Path,
    prediction: np.ndarray,
    log_variance: np.ndarray | None = None,
    metadata_tags: dict[str, str] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(prediction, dtype=np.float32)
    log_values = None
    if log_variance is not None:
        log_values = np.asarray(log_variance, dtype=np.float32)
        if log_values.shape != values.shape:
            raise ValueError(
                "log_variance must match prediction shape, "
                f"got {log_values.shape} vs {values.shape}."
            )
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "Saving detector height uncertainty as band 2 requires rasterio."
            ) from exc

        with rasterio.Env():
            with rasterio.open(source_path) as src:
                profile = _detector_height_raster_profile(src, values)
                profile["count"] = 2
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.stack([values, log_values], axis=0))
            dst.set_band_description(1, "height_m")
            dst.set_band_description(2, "log_variance_m2")
            dst.update_tags(1, name="height_m", units="m")
            dst.update_tags(2, name="log_variance_m2", units="m2")
            if metadata_tags:
                dst.update_tags(**metadata_tags)
        return

    try:
        import rasterio

        with rasterio.Env():
            with rasterio.open(source_path) as src:
                profile = _detector_height_raster_profile(src, values)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(values[np.newaxis, :, :])
            dst.set_band_description(1, "height_m")
            dst.update_tags(1, name="height_m", units="m")
            if metadata_tags:
                dst.update_tags(**metadata_tags)
    except Exception:
        Image.fromarray(values, mode="F").save(output_path)


def normalize_detector_height_splits(splits: Iterable[str]) -> list[str]:
    normalized = []
    for split in splits:
        value = str(split).strip().lower()
        if value not in {"train", "valid", "test"}:
            raise ValueError("splits must contain only train, valid, or test.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one split is required.")
    return normalized


def _required_detector_height_output_bands(height_config) -> int:
    return 2 if int(getattr(height_config, "height_output_channels", 1)) >= 2 else 1


def _detector_height_tile_has_required_bands(path: Path, required_bands: int) -> bool:
    return _detector_height_tile_is_complete(path, required_bands, expected_tags=None)


def _detector_height_tile_is_complete(
    path: Path,
    required_bands: int,
    *,
    expected_tags: dict[str, str] | None = None,
) -> bool:
    if not path.exists():
        return False
    if required_bands <= 1 and expected_tags is None:
        return True
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            if int(dataset.count) < int(required_bands):
                return False
            if expected_tags is None:
                return True
            tags = dataset.tags()
            return all(str(tags.get(key, "")) == str(value) for key, value in expected_tags.items())
    except Exception:
        return False


def _detector_height_precompute_tags(
    *,
    height_checkpoint_path: str | Path,
    height_train_config_path: str | Path | None,
    required_output_bands: int,
    source_gsd: float,
    target_gsd: float,
) -> dict[str, str]:
    checkpoint_path = Path(height_checkpoint_path).expanduser().resolve()
    stat = checkpoint_path.stat()
    train_config = (
        str(Path(height_train_config_path).expanduser().resolve())
        if height_train_config_path is not None
        else ""
    )
    return {
        "detector_height_precompute_schema": str(DETECTOR_HEIGHT_PRECOMPUTE_SCHEMA_VERSION),
        "height_checkpoint_path": str(checkpoint_path),
        "height_checkpoint_size_bytes": str(int(stat.st_size)),
        "height_checkpoint_mtime_ns": str(int(stat.st_mtime_ns)),
        "height_train_config_path": train_config,
        "height_output_bands": str(int(required_output_bands)),
        "source_gsd_m": f"{float(source_gsd):.12g}",
        "target_gsd_m": f"{float(target_gsd):.12g}",
    }


def _predict_detector_height_outputs(
    model: torch.nn.Module,
    batch: torch.Tensor,
    *,
    required_output_bands: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    outputs = model(batch)
    if "height" not in outputs:
        raise RuntimeError("Height precompute model output is missing 'height'.")
    predictions = outputs["height"].clamp_min(0.0)
    log_variances = None
    if required_output_bands >= 2:
        log_variances = outputs.get("log_variance")
        if log_variances is None:
            raise RuntimeError("Two-band height precompute requires model output 'log_variance'.")
    if predictions.shape[-2:] != batch.shape[-2:]:
        predictions = F.interpolate(
            predictions,
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if log_variances is not None and log_variances.shape[-2:] != batch.shape[-2:]:
        log_variances = F.interpolate(
            log_variances,
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return predictions, log_variances


def _detector_height_raster_profile(src, values: np.ndarray) -> dict:
    transform = src.transform
    if values.shape != (src.height, src.width):
        transform = transform * transform.scale(
            src.width / values.shape[1],
            src.height / values.shape[0],
        )
    return {
        "driver": "GTiff",
        "height": int(values.shape[0]),
        "width": int(values.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": src.crs,
        "transform": transform,
        "nodata": None,
        "compress": "deflate",
    }


def _prediction_to_detector_grid(
    prediction: torch.Tensor,
    item: DetectorHeightPreparedTile,
) -> np.ndarray:
    crop = prediction[
        :,
        item.crop_top : item.crop_top + item.low_size[1],
        item.crop_left : item.crop_left + item.low_size[0],
    ].unsqueeze(0)
    upsampled = F.interpolate(
        crop,
        size=(item.original_size[1], item.original_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    array = upsampled.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if array.shape != (item.valid_mask.shape[0], item.valid_mask.shape[1]):
        raise ValueError(
            "Upsampled prediction does not match the source valid mask: "
            f"prediction={array.shape}, mask={item.valid_mask.shape}."
        )
    return np.where(item.valid_mask, array, 0.0).astype(np.float32, copy=False)


def _dataset_sources_for_split(
    config: DetectorTrainingConfig,
    *,
    split: str,
    dataset_names: Iterable[str] | None,
) -> list[tuple[Path, list[str]]]:
    if dataset_names is not None:
        return [(Path(config.data_root_path), [str(name) for name in dataset_names])]
    if split == "valid":
        return [
            (
                Path(config.resolved_synthetic_validation_root()),
                list(config.resolved_synthetic_validation_names()),
            )
        ]
    if split == "test" and config.test_dataset_names:
        return [(Path(config.data_root_path), list(config.test_dataset_names))]
    return [(Path(config.data_root_path), list(config.train_dataset_names))]


def _resolve_detector_dataset_root(source_root: Path, dataset_name: str) -> Path:
    candidate = source_root / dataset_name
    if candidate.exists():
        return candidate
    if source_root.exists() and source_root.name == dataset_name:
        return source_root
    raise FileNotFoundError(f"Detector dataset {dataset_name!r} was not found under {source_root}.")


def _find_detector_product_roots(dataset_root: Path, split: str) -> list[Path]:
    candidates = []
    if (dataset_root / "tiles" / split).exists():
        candidates.append(dataset_root)
    for tile_dir in sorted(dataset_root.rglob(f"tiles/{split}")):
        candidates.append(tile_dir.parent.parent)

    seen = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _resize_valid_mask(valid_mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray(np.asarray(valid_mask, dtype=np.uint8) * 255)
    resized = mask_image.resize(size, Image.Resampling.BOX)
    return np.ascontiguousarray(np.asarray(resized) > 0, dtype=bool)


def _center_pad_image_and_mask(
    image: Image.Image,
    valid_mask: np.ndarray,
    *,
    multiple: int,
    image_mean: list[float],
) -> tuple[Image.Image, np.ndarray, int, int]:
    multiple = max(1, int(multiple))
    width, height = image.size
    padded_width = int(math.ceil(width / multiple) * multiple)
    padded_height = int(math.ceil(height / multiple) * multiple)
    crop_left = (padded_width - width) // 2
    crop_top = (padded_height - height) // 2
    if padded_width == width and padded_height == height:
        return image.convert("RGB"), np.ascontiguousarray(valid_mask, dtype=bool), 0, 0

    fill = tuple(int(round(float(value) * 255.0)) for value in image_mean)
    canvas = Image.new("RGB", (padded_width, padded_height), fill)
    canvas.paste(image.convert("RGB"), (crop_left, crop_top))
    padded_mask = np.zeros((padded_height, padded_width), dtype=bool)
    padded_mask[crop_top : crop_top + height, crop_left : crop_left + width] = valid_mask
    return canvas, np.ascontiguousarray(padded_mask, dtype=bool), crop_left, crop_top


def _validate_gsds(*, source_gsd: float, target_gsd: float) -> None:
    source = float(source_gsd)
    target = float(target_gsd)
    if source <= 0.0 or target <= 0.0:
        raise ValueError("source_gsd and target_gsd must be positive.")
    if target < source:
        raise ValueError(
            "Detector height precompute expects target_gsd >= source_gsd "
            f"for downsampling, got source_gsd={source_gsd}, target_gsd={target_gsd}."
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
