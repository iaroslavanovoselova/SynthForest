from __future__ import annotations

import json
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from synthforest.training.coco_masks import (
    CocoMaskCoverage,
    decode_coco_segmentation,
    decode_coco_segmentation_crop,
    filter_coco_annotations_with_valid_masks,
    inspect_coco_mask_coverage,
)
from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.valid_masks import (
    box_valid_fractions,
    fill_invalid_pixels,
    image_and_mask_to_tensors,
    load_image_with_valid_mask,
    valid_fraction,
)

SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPPORTED_HEIGHT_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class CocoTileSource:
    name: str
    tile_dir: Path
    coco_path: Path


@dataclass(frozen=True)
class UnlabeledRealTileSource:
    name: str
    tile_dir: Path
    source_gsd_m: Optional[float] = None


@dataclass(frozen=True)
class DetectionGeometry:
    crop: Optional[tuple[int, int, int]] = None
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotation_angle: Optional[float] = None


def _unlabeled_height_config(config: DetectorTrainingConfig) -> Any | None:
    if config.height_fusion.enabled:
        return config.height_fusion
    return None


class CocoTileDataset(Dataset):
    """Read CanopyRS/geodataset-style per-raster COCO tile folders."""

    def __init__(
        self,
        data_root_path: str | Path,
        dataset_names: Iterable[str],
        fold: str,
        config: DetectorTrainingConfig,
        train: bool,
        coco_sources: Optional[Iterable[CocoTileSource]] = None,
        max_tiles_per_source: Optional[int] = None,
        drop_edge_tiles: bool = False,
        edge_tile_black_pixel_threshold: float = 0.1,
        edge_tile_black_pixel_tolerance: int = 0,
    ) -> None:
        self.data_root_path = Path(data_root_path)
        self.dataset_names = list(dataset_names)
        self.fold = fold
        self.config = config
        self.train = train
        self.coco_sources = list(coco_sources or [])
        self.max_tiles_per_source = max_tiles_per_source
        self.drop_edge_tiles = drop_edge_tiles
        self.edge_tile_black_pixel_threshold = edge_tile_black_pixel_threshold
        self.edge_tile_black_pixel_tolerance = edge_tile_black_pixel_tolerance
        self.require_instance_masks = config.model_type == "maskrcnn"
        self.mask_coverage_by_source: dict[Path, CocoMaskCoverage] = {}
        self.augmentor = DetectionAugmentor(config, train=train)
        self.records = self._load_records()
        if not self.records:
            raise ValueError(
                f"No {fold!r} COCO tile records found under {self.data_root_path} "
                f"for datasets: {self.dataset_names}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image, valid_mask = load_image_with_valid_mask(
            record["image_path"],
            self.config.valid_mask,
        )
        height = None
        if self.config.height_fusion.enabled:
            height, height_valid_mask = load_detector_height_features(
                record["height_path"],
                use_uncertainty=bool(self.config.height_fusion.use_uncertainty),
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
            if _height_spatial_shape(height) != (image.height, image.width):
                height, height_valid_mask = _resize_height_and_mask(
                    height,
                    height_valid_mask,
                    size=image.size,
                )
            valid_mask = np.ascontiguousarray(valid_mask & height_valid_mask, dtype=bool)
        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(record["labels"], dtype=torch.int64)
        area = torch.as_tensor(record["area"], dtype=torch.float32)
        iscrowd = torch.as_tensor(record["iscrowd"], dtype=torch.int64)
        masks = None
        if self.require_instance_masks:
            if self.train:
                (
                    image,
                    valid_mask,
                    boxes,
                    labels,
                    area,
                    iscrowd,
                    masks,
                ) = self.augmentor(
                    image,
                    valid_mask,
                    boxes,
                    labels,
                    area,
                    iscrowd,
                    segmentations=record["segmentations"],
                    source_image_size=(
                        int(record["height"]),
                        int(record["width"]),
                    ),
                    height=height,
                )
            else:
                masks = _decode_record_masks(record)
                (
                    image,
                    valid_mask,
                    boxes,
                    labels,
                    area,
                    iscrowd,
                    masks,
                ) = self.augmentor(
                    image,
                    valid_mask,
                    boxes,
                    labels,
                    area,
                    iscrowd,
                    masks=masks,
                    height=height,
                )
        else:
            image, valid_mask, boxes, labels, area, iscrowd = self.augmentor(
                image,
                valid_mask,
                boxes,
                labels,
                area,
                iscrowd,
                height=height,
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        if masks is not None:
            target["masks"] = masks
        if not self.train:
            target["valid_mask"] = valid_mask
        return image, target

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self.coco_sources:
            for source in self.coco_sources:
                records.extend(
                    self._records_from_json(
                        source.coco_path,
                        tile_dir=source.tile_dir,
                        source_name=source.name,
                    )
                )
            return records

        for dataset_name in self.dataset_names:
            dataset_root = self.data_root_path / dataset_name
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
            for json_path in sorted(dataset_root.rglob("*.json")):
                if not _json_matches_fold(json_path, self.fold):
                    continue
                records.extend(self._records_from_json(json_path, source_name=dataset_name))
        return records

    def _records_from_json(
        self,
        json_path: Path,
        *,
        tile_dir: Optional[Path] = None,
        source_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        if self.require_instance_masks:
            annotations, coverage = filter_coco_annotations_with_valid_masks(coco)
        else:
            annotations = list(coco.get("annotations", []))
            coverage = inspect_coco_mask_coverage(coco)
        self.mask_coverage_by_source[json_path] = coverage
        skipped_mask_count = coverage.total_annotations - coverage.valid_segmentations
        if self.require_instance_masks and skipped_mask_count:
            warnings.warn(
                f"Mask R-CNN is skipping {skipped_mask_count} annotation(s) with "
                f"invalid instance masks from {json_path} "
                f"(missing={coverage.missing_segmentations}, "
                f"structurally_invalid={coverage.invalid_segmentations}, "
                f"decode_errors={coverage.decode_errors}).",
                stacklevel=2,
            )

        raster_dir = json_path.parent
        images = {int(image["id"]): image for image in coco.get("images", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            anns_by_image[int(annotation["image_id"])].append(annotation)

        records = []
        for image_id, image_info in images.items():
            if self.max_tiles_per_source is not None and len(records) >= self.max_tiles_per_source:
                break
            annotations = anns_by_image.get(image_id, [])
            if not annotations and not self.config.include_empty_images:
                continue

            if tile_dir is None:
                image_path = _resolve_image_path(
                    raster_dir=raster_dir,
                    fold=self.fold,
                    file_name=image_info["file_name"],
                )
            else:
                image_path = resolve_tile_path(tile_dir, image_info["file_name"])
            height_path = None
            if self.config.height_fusion.enabled:
                height_path = resolve_height_path_for_tile(
                    image_path=image_path,
                    fold=self.fold,
                    height_maps_subdir=self.config.height_fusion.height_maps_subdir,
                )
                if not height_path.exists() and not self.config.height_fusion.generate_missing:
                    raise FileNotFoundError(
                        "Height fusion is enabled but the paired height map is missing: "
                        f"{height_path}"
                    )
            if self.drop_edge_tiles and is_edge_tile(
                image_path=image_path,
                black_pixel_threshold=self.edge_tile_black_pixel_threshold,
                black_pixel_tolerance=self.edge_tile_black_pixel_tolerance,
            ):
                continue
            boxes, labels, area, iscrowd, segmentations = _annotations_to_tensors(
                annotations=annotations,
                force_binary=self.config.num_classes == 1,
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
                include_segmentations=True,
            )
            records.append(
                {
                    "image_path": image_path,
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                    "boxes": boxes,
                    "labels": labels,
                    "area": area,
                    "iscrowd": iscrowd,
                    "segmentations": segmentations,
                    "source_json": json_path,
                    "source_image_id": image_id,
                    "source_name": source_name,
                    "height_path": height_path,
                }
            )
        return records


class UnlabeledRealTileDataset(Dataset):
    """Read real RGB tiles without requiring labels."""

    def __init__(
        self,
        data_root_path: str | Path,
        sources: Iterable[UnlabeledRealTileSource],
        config: DetectorTrainingConfig,
        max_tiles_per_source: Optional[int] = None,
        drop_edge_tiles: bool = False,
        edge_tile_black_pixel_threshold: float = 0.1,
        edge_tile_black_pixel_tolerance: int = 0,
    ) -> None:
        self.data_root_path = Path(data_root_path)
        self.sources = list(sources)
        self.config = config
        self.max_tiles_per_source = max_tiles_per_source
        self.drop_edge_tiles = drop_edge_tiles
        self.edge_tile_black_pixel_threshold = edge_tile_black_pixel_threshold
        self.edge_tile_black_pixel_tolerance = edge_tile_black_pixel_tolerance
        self.augmentor = DetectionAugmentor(
            config,
            train=True,
            enable_domain_adaptation=False,
        )
        self.records = self._load_records()
        if not self.records:
            source_names = [source.name for source in self.sources]
            raise ValueError(
                f"No unlabeled real tile records found under {self.data_root_path} "
                f"for sources: {source_names}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image, valid_mask = load_image_with_valid_mask(
            record["image_path"],
            self.config.valid_mask,
        )
        height = None
        height_valid_mask = None
        height_config = _unlabeled_height_config(self.config)
        if height_config is not None:
            height, height_valid_mask = load_detector_height_features(
                record["height_path"],
                use_uncertainty=bool(
                    self.config.height_fusion.enabled and self.config.height_fusion.use_uncertainty
                ),
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
            if _height_spatial_shape(height) != (image.height, image.width):
                height, height_valid_mask = _resize_height_and_mask(
                    height,
                    height_valid_mask,
                    size=image.size,
                )
            if self.config.height_fusion.enabled:
                valid_mask = np.ascontiguousarray(valid_mask & height_valid_mask, dtype=bool)
        (
            weak,
            strong,
            mask,
            _aligned_height,
            _aligned_height_valid,
        ) = self.augmentor.weak_strong_pair(
            image,
            valid_mask,
            height=height,
            height_valid_mask=height_valid_mask,
            return_height=False,
        )
        metadata: dict[str, Any] = {
            "valid_mask": mask,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "image_path": str(record["image_path"]),
            "source_name": record["source_name"],
        }
        return weak, strong, metadata

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source in self.sources:
            height_config = _unlabeled_height_config(self.config)
            source_gsd_m = (
                float(source.source_gsd_m)
                if source.source_gsd_m is not None
                else float(height_config.source_gsd_m)
                if height_config is not None
                else None
            )
            if not source.tile_dir.exists():
                raise FileNotFoundError(
                    f"Unlabeled real tile directory is missing: {source.tile_dir}"
                )
            source_records = []
            for path in sorted(source.tile_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                height_path = None
                if height_config is not None:
                    height_path = resolve_height_path_for_tile(
                        image_path=path,
                        fold=path.parent.name,
                        height_maps_subdir=height_config.height_maps_subdir,
                    )
                    if not height_path.exists() and not height_config.generate_missing:
                        raise FileNotFoundError(
                            "Height use is enabled but the paired unlabeled-real "
                            f"height map is missing: {height_path}"
                        )
                if self.drop_edge_tiles and is_edge_tile(
                    image_path=path,
                    black_pixel_threshold=self.edge_tile_black_pixel_threshold,
                    black_pixel_tolerance=self.edge_tile_black_pixel_tolerance,
                ):
                    continue
                source_records.append(
                    {
                        "image_path": path,
                        "source_name": source.name,
                        "height_path": height_path,
                        "source_gsd_m": source_gsd_m,
                    }
                )
                if (
                    self.max_tiles_per_source is not None
                    and len(source_records) >= self.max_tiles_per_source
                ):
                    break
            records.extend(source_records)
        return records


class DetectionAugmentor:
    def __init__(
        self,
        config: DetectorTrainingConfig,
        train: bool,
        *,
        enable_domain_adaptation: bool = True,
    ) -> None:
        self.config = config
        self.train = train
        self.reference_paths = (
            _collect_reference_images(config) if train and enable_domain_adaptation else []
        )
        self._reference_is_full: dict[Path, bool] = {}
        self._warned_domain_adaptation = False

    def __call__(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        area: torch.Tensor,
        iscrowd: torch.Tensor,
        *,
        masks: Optional[torch.Tensor] = None,
        segmentations: Optional[list[Any]] = None,
        source_image_size: Optional[tuple[int, int]] = None,
        height: Optional[np.ndarray] = None,
    ) -> tuple:
        if masks is not None and segmentations is not None:
            raise ValueError("Pass either dense masks or segmentations, not both.")
        has_instance_masks = masks is not None or segmentations is not None
        if self.train:
            image = self._apply_domain_adaptation(image, valid_mask)
            if self.config.valid_mask.enabled:
                image = fill_invalid_pixels(
                    image,
                    valid_mask,
                    self.config.augmentation.image_mean,
                )
            image = self._apply_color(image)
            if has_instance_masks:
                if segmentations is not None:
                    if source_image_size is None:
                        raise ValueError(
                            "source_image_size is required when segmentations are passed."
                        )
                    (
                        image,
                        valid_mask,
                        boxes,
                        labels,
                        iscrowd,
                        masks,
                        height,
                    ) = self._apply_crop_with_lazy_masks(
                        image,
                        valid_mask,
                        boxes,
                        labels,
                        area,
                        iscrowd,
                        segmentations=segmentations,
                        source_image_size=source_image_size,
                        height=height,
                    )
                else:
                    (
                        image,
                        valid_mask,
                        boxes,
                        labels,
                        iscrowd,
                        masks,
                        height,
                    ) = self._apply_crop(
                        image,
                        valid_mask,
                        boxes,
                        labels,
                        iscrowd,
                        masks=masks,
                        height=height,
                    )
                image, valid_mask, boxes, masks, height = self._apply_flips(
                    image,
                    valid_mask,
                    boxes,
                    masks=masks,
                    height=height,
                )
                image, valid_mask, boxes, masks, height = self._apply_rotation(
                    image,
                    valid_mask,
                    boxes,
                    masks=masks,
                    height=height,
                    return_height=True,
                )
            else:
                image, valid_mask, boxes, labels, iscrowd, height = self._apply_crop(
                    image,
                    valid_mask,
                    boxes,
                    labels,
                    iscrowd,
                    height=height,
                )
                image, valid_mask, boxes, height = self._apply_flips(
                    image,
                    valid_mask,
                    boxes,
                    height=height,
                )
                image, valid_mask, boxes, height = self._apply_rotation(
                    image,
                    valid_mask,
                    boxes,
                    height=height,
                    return_height=True,
                )
            if self.config.valid_mask.enabled:
                image = fill_invalid_pixels(
                    image,
                    valid_mask,
                    self.config.augmentation.image_mean,
                )
            image = self._apply_image_degradations(image)
        if self.config.valid_mask.enabled:
            image = fill_invalid_pixels(
                image,
                valid_mask,
                self.config.augmentation.image_mean,
            )

        if has_instance_masks:
            assert masks is not None
            boxes, labels, iscrowd, masks = _finalize_mask_targets(
                boxes=boxes,
                labels=labels,
                iscrowd=iscrowd,
                masks=masks,
                valid_mask=valid_mask,
                apply_valid_filter=self.train and self.config.valid_mask.enabled,
                min_valid_fraction=self.config.valid_mask.target_min_valid_fraction,
                derive_boxes=self.train,
                image_size=image.size,
            )
        else:
            boxes, labels, iscrowd = _clip_and_filter_boxes(
                boxes,
                labels,
                iscrowd,
                image.size,
            )
        if self.train and self.config.valid_mask.enabled:
            if not has_instance_masks:
                boxes, labels, iscrowd = _filter_targets_by_valid_mask(
                    boxes,
                    labels,
                    iscrowd,
                    valid_mask,
                    min_valid_fraction=self.config.valid_mask.target_min_valid_fraction,
                )
            if not bool(valid_mask.any()):
                raise ValueError(
                    "Training augmentation produced an entirely invalid image. "
                    "Increase valid_mask.crop_min_valid_fraction or remove the tile."
                )
        area = _box_area(boxes)
        tensor, mask_tensor = image_and_mask_to_tensors(
            image,
            valid_mask,
            self.config.augmentation.image_mean,
        )
        if height is not None:
            if _height_spatial_shape(height) != (image.height, image.width):
                raise ValueError(
                    "Augmented height map shape does not match RGB image size: "
                    f"height={height.shape}, image={(image.height, image.width)}."
                )
            height_array = _mask_detector_height_features(
                height,
                valid_mask,
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
            height_tensor = torch.from_numpy(np.ascontiguousarray(height_array))
            if height_tensor.ndim == 2:
                height_tensor = height_tensor.unsqueeze(0)
            tensor = torch.cat([tensor, height_tensor], dim=0)
        base = (tensor, mask_tensor, boxes, labels, area, iscrowd)
        if has_instance_masks:
            assert masks is not None
            return (*base, masks.to(torch.uint8).contiguous())
        return base

    def weak_strong_pair(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        height: Optional[np.ndarray] = None,
        height_valid_mask: Optional[np.ndarray] = None,
        return_height: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        geometry = self._sample_paired_geometry(image, valid_mask)
        weak_image, weak_mask, weak_height, weak_height_valid = self._apply_paired_geometry(
            image.copy(),
            valid_mask.copy(),
            geometry,
            height=height.copy() if height is not None else None,
            height_valid_mask=(height_valid_mask.copy() if height_valid_mask is not None else None),
        )
        strong_image, strong_mask, strong_height, _ = self._apply_paired_geometry(
            image.copy(),
            valid_mask.copy(),
            geometry,
            height=height.copy() if height is not None else None,
            height_valid_mask=(height_valid_mask.copy() if height_valid_mask is not None else None),
        )

        if self.config.valid_mask.enabled:
            weak_image = fill_invalid_pixels(
                weak_image,
                weak_mask,
                self.config.augmentation.image_mean,
            )
            strong_image = fill_invalid_pixels(
                strong_image,
                strong_mask,
                self.config.augmentation.image_mean,
            )
        strong_aug = self.config.semi_supervised.strong_augmentation
        strong_image = self._apply_color(strong_image, aug=strong_aug)
        if self.config.valid_mask.enabled:
            strong_image = fill_invalid_pixels(
                strong_image,
                strong_mask,
                self.config.augmentation.image_mean,
            )
        strong_image = self._apply_image_degradations(strong_image, aug=strong_aug)
        if self.config.valid_mask.enabled:
            strong_image = fill_invalid_pixels(
                strong_image,
                strong_mask,
                self.config.augmentation.image_mean,
            )

        weak_tensor, weak_mask_tensor = image_and_mask_to_tensors(
            weak_image,
            weak_mask,
            self.config.augmentation.image_mean,
        )
        strong_tensor, strong_mask_tensor = image_and_mask_to_tensors(
            strong_image,
            strong_mask,
            self.config.augmentation.image_mean,
        )
        if weak_mask_tensor.shape != strong_mask_tensor.shape:
            raise ValueError(
                "Weak and strong real views must share geometry; got masks "
                f"{tuple(weak_mask_tensor.shape)} and {tuple(strong_mask_tensor.shape)}."
            )
        if weak_height is not None and self.config.height_fusion.enabled:
            weak_tensor = _append_detector_height_tensor(
                weak_tensor,
                weak_height,
                weak_mask,
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
        if strong_height is not None and self.config.height_fusion.enabled:
            strong_tensor = _append_detector_height_tensor(
                strong_tensor,
                strong_height,
                strong_mask,
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
        aligned_height = None
        aligned_height_valid = None
        if return_height:
            if weak_height is None or weak_height_valid is None:
                raise ValueError(
                    "Height verification requested an aligned height map, but none was loaded."
                )
            height_array = np.asarray(weak_height, dtype=np.float32)
            if height_array.ndim == 3:
                height_array = height_array[0]
            combined_valid = np.ascontiguousarray(
                weak_height_valid & weak_mask & np.isfinite(height_array),
                dtype=bool,
            )
            aligned_height = torch.from_numpy(np.ascontiguousarray(height_array, dtype=np.float32))
            aligned_height_valid = torch.from_numpy(combined_valid)
        return (
            weak_tensor,
            strong_tensor,
            weak_mask_tensor,
            aligned_height,
            aligned_height_valid,
        )

    def _sample_paired_geometry(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
    ) -> DetectionGeometry:
        crop = self._sample_crop_window(image, valid_mask)
        crop_box = None if crop is None else (int(crop[0]), int(crop[1]), int(crop[2]))
        aug = self.config.augmentation
        rotation_angle = None
        if aug.rotation_degrees > 0 and random.random() < aug.rotation_prob:
            rotation_angle = random.uniform(-aug.rotation_degrees, aug.rotation_degrees)
        return DetectionGeometry(
            crop=crop_box,
            flip_horizontal=bool(aug.flip_horizontal and random.random() < 0.5),
            flip_vertical=bool(aug.flip_vertical and random.random() < 0.5),
            rotation_angle=rotation_angle,
        )

    def _apply_paired_geometry(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        geometry: DetectionGeometry,
        *,
        height: Optional[np.ndarray] = None,
        height_valid_mask: Optional[np.ndarray] = None,
    ) -> tuple[
        Image.Image,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if geometry.crop is not None:
            side, left, top = geometry.crop
            image = image.crop((left, top, left + side, top + side))
            valid_mask = valid_mask[top : top + side, left : left + side].copy()
            if height is not None:
                height = _crop_detector_height_features(height, top=top, left=left, side=side)
            if height_valid_mask is not None:
                height_valid_mask = height_valid_mask[top : top + side, left : left + side].copy()
        if geometry.flip_horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            valid_mask = np.fliplr(valid_mask).copy()
            if height is not None:
                height = _flip_detector_height_features_left_right(height)
            if height_valid_mask is not None:
                height_valid_mask = np.fliplr(height_valid_mask).copy()
        if geometry.flip_vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            valid_mask = np.flipud(valid_mask).copy()
            if height is not None:
                height = _flip_detector_height_features_top_bottom(height)
            if height_valid_mask is not None:
                height_valid_mask = np.flipud(height_valid_mask).copy()
        if geometry.rotation_angle is not None:
            fill = tuple(
                round(float(value) * 255.0) for value in self.config.augmentation.image_mean
            )
            image = image.rotate(
                float(geometry.rotation_angle),
                resample=Image.Resampling.BILINEAR,
                fillcolor=fill if self.config.valid_mask.enabled else None,
            )
            if self.config.valid_mask.enabled or height is not None:
                valid_mask = (
                    np.asarray(
                        Image.fromarray(valid_mask.astype(np.uint8) * 255).rotate(
                            float(geometry.rotation_angle),
                            resample=Image.Resampling.NEAREST,
                            fillcolor=0,
                        )
                    )
                    > 0
                )
            if height is not None:
                height = _rotate_detector_height_features(
                    height,
                    angle_degrees=float(geometry.rotation_angle),
                    uncertainty_fill_value=float(
                        self.config.height_fusion.uncertainty_log_variance_max
                    ),
                )
            if height_valid_mask is not None:
                height_valid_mask = (
                    np.asarray(
                        Image.fromarray(height_valid_mask.astype(np.uint8) * 255).rotate(
                            float(geometry.rotation_angle),
                            resample=Image.Resampling.NEAREST,
                            fillcolor=0,
                        )
                    )
                    > 0
                )
        return (
            image,
            np.ascontiguousarray(valid_mask, dtype=bool),
            height,
            (
                np.ascontiguousarray(height_valid_mask, dtype=bool)
                if height_valid_mask is not None
                else None
            ),
        )

    def _apply_color(self, image: Image.Image, *, aug: Optional[Any] = None) -> Image.Image:
        aug = self.config.augmentation if aug is None else aug
        if random.random() < aug.brightness_prob:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(1.0 - aug.brightness, 1.0 + aug.brightness)
            )
        if random.random() < aug.contrast_prob:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(1.0 - aug.contrast, 1.0 + aug.contrast)
            )
        if random.random() < aug.saturation_prob:
            image = ImageEnhance.Color(image).enhance(
                random.uniform(1.0 - aug.saturation, 1.0 + aug.saturation)
            )
        if random.random() < aug.hue_prob:
            image = _shift_hue(image, random.uniform(-aug.hue, aug.hue))
        return image

    def _apply_crop(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        iscrowd: torch.Tensor,
        *,
        masks: Optional[torch.Tensor] = None,
        height: Optional[np.ndarray] = None,
    ) -> tuple:
        has_instance_masks = masks is not None

        def result(
            out_image,
            out_valid_mask,
            out_boxes,
            out_labels,
            out_iscrowd,
            out_masks=masks,
            out_height=height,
        ):
            base = (
                out_image,
                out_valid_mask,
                out_boxes,
                out_labels,
                out_iscrowd,
            )
            return (*base, out_masks, out_height) if has_instance_masks else (*base, out_height)

        aug = self.config.augmentation
        crop = self._sample_crop_window(image, valid_mask)
        if crop is None:
            return result(image, valid_mask, boxes, labels, iscrowd)

        side, left, top, valid_mask = crop
        image = image.crop((left, top, left + side, top + side))
        cropped_height = (
            _crop_detector_height_features(height, top=top, left=left, side=side)
            if height is not None
            else None
        )

        if boxes.numel() == 0:
            cropped_masks = (
                masks[:, top : top + side, left : left + side] if masks is not None else masks
            )
            return result(
                image,
                valid_mask.copy(),
                boxes,
                labels,
                iscrowd,
                cropped_masks,
                cropped_height,
            )

        if masks is not None:
            original_area = masks.flatten(1).sum(1).to(torch.float32).clamp_min(1.0)
            cropped_masks = masks[:, top : top + side, left : left + side]
            kept_area = cropped_masks.flatten(1).sum(1).to(torch.float32)
            intersection_ratio = kept_area / original_area
            keep = (kept_area > 1.0) & (
                intersection_ratio >= float(aug.crop_min_intersection_ratio)
            )
            kept_masks = cropped_masks[keep].contiguous()
            kept_boxes = _boxes_from_masks(kept_masks)
            return result(
                image,
                valid_mask.copy(),
                kept_boxes,
                labels[keep],
                iscrowd[keep],
                kept_masks,
                cropped_height,
            )

        original_area = _box_area(boxes).clamp_min(1.0)
        cropped = boxes.clone()
        cropped[:, [0, 2]] -= left
        cropped[:, [1, 3]] -= top
        cropped[:, 0::2] = cropped[:, 0::2].clamp(0, side)
        cropped[:, 1::2] = cropped[:, 1::2].clamp(0, side)
        kept_area = _box_area(cropped)
        intersection_ratio = kept_area / original_area
        keep = (kept_area > 1.0) & (intersection_ratio >= aug.crop_min_intersection_ratio)
        return result(
            image,
            valid_mask.copy(),
            cropped[keep],
            labels[keep],
            iscrowd[keep],
            out_height=cropped_height,
        )

    def _apply_crop_with_lazy_masks(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        area: torch.Tensor,
        iscrowd: torch.Tensor,
        *,
        segmentations: list[Any],
        source_image_size: tuple[int, int],
        height: Optional[np.ndarray] = None,
    ) -> tuple:
        crop = self._sample_crop_window(
            image,
            valid_mask,
            fallback_to_valid_pixel=True,
        )
        source_height, width = source_image_size
        if crop is None:
            masks = _decode_segmentations(
                segmentations,
                image_height=source_height,
                image_width=width,
            )
            return image, valid_mask, boxes, labels, iscrowd, masks, height

        side, left, top, valid_mask = crop
        image = image.crop((left, top, left + side, top + side))
        cropped_height = (
            _crop_detector_height_features(height, top=top, left=left, side=side)
            if height is not None
            else None
        )
        crop_box = (left, top, left + side, top + side)

        if boxes.numel() == 0:
            masks = torch.zeros((0, side, side), dtype=torch.uint8)
            return image, valid_mask.copy(), boxes, labels, iscrowd, masks, cropped_height

        cropped_boxes = boxes.clone()
        cropped_boxes[:, [0, 2]] -= left
        cropped_boxes[:, [1, 3]] -= top
        cropped_boxes[:, 0::2] = cropped_boxes[:, 0::2].clamp(0, side)
        cropped_boxes[:, 1::2] = cropped_boxes[:, 1::2].clamp(0, side)
        bbox_keep = _box_area(cropped_boxes) > 1.0
        if not bool(bbox_keep.any()):
            masks = torch.zeros((0, side, side), dtype=torch.uint8)
            return (
                image,
                valid_mask.copy(),
                cropped_boxes[:0],
                labels[:0],
                iscrowd[:0],
                masks,
                cropped_height,
            )

        candidate_indexes = torch.nonzero(bbox_keep, as_tuple=False).flatten().tolist()
        decoded = [
            torch.from_numpy(
                decode_coco_segmentation_crop(
                    segmentations[index],
                    image_height=source_height,
                    image_width=width,
                    crop_box=crop_box,
                )
            )
            for index in candidate_indexes
        ]
        candidate_masks = (
            torch.stack(decoded, dim=0).to(torch.uint8).contiguous()
            if decoded
            else torch.zeros((0, side, side), dtype=torch.uint8)
        )
        if candidate_masks.shape[0] == 0:
            masks = torch.zeros((0, side, side), dtype=torch.uint8)
            return (
                image,
                valid_mask.copy(),
                cropped_boxes[:0],
                labels[:0],
                iscrowd[:0],
                masks,
                cropped_height,
            )

        candidate_tensor = torch.as_tensor(candidate_indexes, dtype=torch.long)
        original_area = area[candidate_tensor].to(torch.float32).clamp_min(1.0)
        kept_area = candidate_masks.flatten(1).sum(1).to(torch.float32)
        keep = (kept_area > 1.0) & (
            kept_area / original_area >= float(self.config.augmentation.crop_min_intersection_ratio)
        )
        kept_masks = candidate_masks[keep].contiguous()
        kept_boxes = _boxes_from_masks(kept_masks)
        kept_indexes = candidate_tensor[keep]
        return (
            image,
            valid_mask.copy(),
            kept_boxes,
            labels[kept_indexes],
            iscrowd[kept_indexes],
            kept_masks,
            cropped_height,
        )

    def _sample_crop_window(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        *,
        fallback_to_valid_pixel: bool = False,
    ) -> Optional[tuple[int, int, int, np.ndarray]]:
        aug = self.config.augmentation
        if random.random() >= aug.crop_prob:
            return None

        width, height = image.size
        max_possible = min(width, height, max(aug.crop_size_range))
        min_possible = min(max_possible, min(aug.crop_size_range))
        if min_possible <= 1 or max_possible <= 1:
            return None

        for _ in range(10):
            side = random.randint(int(min_possible), int(max_possible))
            left = random.randint(0, max(0, width - side))
            top = random.randint(0, max(0, height - side))
            candidate_mask = valid_mask[top : top + side, left : left + side]
            candidate = (side, left, top, candidate_mask)
            if (
                not self.config.valid_mask.enabled
                or valid_fraction(candidate_mask) >= self.config.valid_mask.crop_min_valid_fraction
            ):
                return candidate
        if fallback_to_valid_pixel:
            return self._sample_valid_pixel_crop_window(
                valid_mask,
                min_side=int(min_possible),
                max_side=int(max_possible),
            )
        return None

    def _sample_valid_pixel_crop_window(
        self,
        valid_mask: np.ndarray,
        *,
        min_side: int,
        max_side: int,
    ) -> Optional[tuple[int, int, int, np.ndarray]]:
        height, width = valid_mask.shape
        if not self.config.valid_mask.enabled:
            side = random.randint(min_side, max_side)
            left = random.randint(0, max(0, width - side))
            top = random.randint(0, max(0, height - side))
            return side, left, top, valid_mask[top : top + side, left : left + side]

        valid_y, valid_x = np.nonzero(valid_mask)
        if valid_y.size == 0:
            return None

        best: Optional[tuple[int, int, int, np.ndarray]] = None
        best_fraction = -1.0
        for _ in range(25):
            index = random.randrange(valid_y.size)
            side = random.randint(min_side, max_side)
            center_x = int(valid_x[index])
            center_y = int(valid_y[index])
            left = max(0, min(width - side, center_x - side // 2))
            top = max(0, min(height - side, center_y - side // 2))
            candidate_mask = valid_mask[top : top + side, left : left + side]
            fraction = valid_fraction(candidate_mask)
            candidate = (side, left, top, candidate_mask)
            if fraction > best_fraction:
                best_fraction = fraction
                best = candidate
            if fraction >= self.config.valid_mask.crop_min_valid_fraction:
                return candidate
        return best

    def _apply_flips(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        boxes: torch.Tensor,
        *,
        masks: Optional[torch.Tensor] = None,
        height: Optional[np.ndarray] = None,
    ) -> tuple:
        has_instance_masks = masks is not None
        aug = self.config.augmentation
        width, image_height = image.size
        if aug.flip_horizontal and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            valid_mask = np.fliplr(valid_mask).copy()
            if height is not None:
                height = _flip_detector_height_features_left_right(height)
            if masks is not None:
                masks = torch.flip(masks, dims=(-1,)).contiguous()
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
        if aug.flip_vertical and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            valid_mask = np.flipud(valid_mask).copy()
            if height is not None:
                height = _flip_detector_height_features_top_bottom(height)
            if masks is not None:
                masks = torch.flip(masks, dims=(-2,)).contiguous()
            if boxes.numel() > 0:
                y1 = boxes[:, 1].clone()
                y2 = boxes[:, 3].clone()
                boxes[:, 1] = image_height - y2
                boxes[:, 3] = image_height - y1
        base = (image, valid_mask, boxes)
        return (*base, masks, height) if has_instance_masks else (*base, height)

    def _apply_rotation(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
        boxes: torch.Tensor,
        *,
        masks: Optional[torch.Tensor] = None,
        height: Optional[np.ndarray] = None,
        return_height: bool = False,
    ) -> tuple:
        has_instance_masks = masks is not None

        def result(out_image, out_valid_mask, out_boxes, out_masks=masks, out_height=height):
            base = (out_image, out_valid_mask, out_boxes)
            if has_instance_masks:
                return (
                    (*base, out_masks, out_height)
                    if return_height or out_height is not None
                    else (*base, out_masks)
                )
            return (*base, out_height) if return_height or out_height is not None else base

        aug = self.config.augmentation
        if aug.rotation_degrees <= 0 or random.random() >= aug.rotation_prob:
            return result(image, valid_mask, boxes)
        angle = random.uniform(-aug.rotation_degrees, aug.rotation_degrees)
        width, image_height = image.size
        fill = tuple(round(float(value) * 255.0) for value in self.config.augmentation.image_mean)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill if self.config.valid_mask.enabled else None,
        )
        if height is not None:
            height = _rotate_detector_height_features(
                height,
                angle_degrees=angle,
                uncertainty_fill_value=float(
                    self.config.height_fusion.uncertainty_log_variance_max
                ),
            )
        if self.config.valid_mask.enabled or height is not None:
            valid_mask = (
                np.asarray(
                    Image.fromarray(valid_mask.astype(np.uint8) * 255).rotate(
                        angle,
                        resample=Image.Resampling.NEAREST,
                        fillcolor=0,
                    )
                )
                > 0
            )
        if masks is not None:
            masks = _rotate_instance_masks(
                masks,
                angle_degrees=angle,
            )
            return result(
                image,
                valid_mask,
                _boxes_from_masks(masks),
                masks,
                height,
            )
        if boxes.numel() == 0:
            return result(image, valid_mask, boxes)
        return result(
            image,
            valid_mask,
            _rotate_boxes(
                boxes,
                width=width,
                height=image_height,
                angle_degrees=angle,
            ),
        )

    def _apply_image_degradations(
        self,
        image: Image.Image,
        *,
        aug: Optional[Any] = None,
    ) -> Image.Image:
        aug = self.config.augmentation if aug is None else aug
        if random.random() < aug.downsample_upsample_prob:
            image = _downsample_then_upsample(image, _sample_float_range(aug.downsample_scale))
        if random.random() < aug.gaussian_blur_prob:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=_sample_float_range(aug.gaussian_blur_sigma))
            )
        if random.random() < aug.gaussian_noise_prob:
            image = _add_gaussian_noise(image, _sample_float_range(aug.gaussian_noise_std))
        if random.random() < aug.jpeg_prob:
            image = _apply_jpeg_compression(image, _sample_int_range(aug.jpeg_quality))
        return image

    def _apply_domain_adaptation(
        self,
        image: Image.Image,
        valid_mask: np.ndarray,
    ) -> Image.Image:
        da_cfg = self.config.domain_adaptation
        if not da_cfg.enabled or not da_cfg.methods or not self.reference_paths:
            return image
        if (
            self.config.valid_mask.enabled
            and valid_fraction(valid_mask) < self.config.valid_mask.da_full_tile_min_valid_fraction
        ):
            return image
        if random.random() >= da_cfg.prob:
            return image

        try:
            import albumentations as A
        except ImportError:
            if not self._warned_domain_adaptation:
                warnings.warn("albumentations is not installed; domain adaptation is skipped.")
                self._warned_domain_adaptation = True
            return image

        source = np.asarray(image)
        reference_path = self._choose_full_reference_path()
        if reference_path is None:
            if not self._warned_domain_adaptation:
                warnings.warn(
                    "No fully valid domain-adaptation reference image was found; "
                    "domain adaptation is skipped."
                )
                self._warned_domain_adaptation = True
            return image
        reference = _load_reference_image(reference_path, image.size)
        transforms = []
        transform_kwargs = {"image": source}
        methods = {method.upper() for method in da_cfg.methods}
        if "FDA" in methods and hasattr(A, "FDA"):
            transform, metadata = _build_domain_adaptation_transform(
                A.FDA,
                metadata_key="fda_metadata",
                reference=reference,
                beta_limit=tuple(da_cfg.fda_beta_limit),
                p=1.0,
            )
            transforms.append(transform)
            transform_kwargs.update(metadata)
        if "HM" in methods and hasattr(A, "HistogramMatching"):
            transform, metadata = _build_domain_adaptation_transform(
                A.HistogramMatching,
                metadata_key="hm_metadata",
                reference=reference,
                blend_ratio=tuple(da_cfg.hm_blend_ratio),
                p=1.0,
            )
            transforms.append(transform)
            transform_kwargs.update(metadata)
        try:
            adapted = A.Compose(transforms)(**transform_kwargs)["image"] if transforms else source
            if "PDA" in methods:
                adapted = _adapt_pixel_distribution(
                    adapted,
                    reference,
                    transform_type=da_cfg.pda_type,
                    weight=random.uniform(*da_cfg.pda_blend_ratio),
                )
        except Exception as exc:
            if not self._warned_domain_adaptation:
                warnings.warn(f"domain adaptation failed and will be skipped: {exc}")
                self._warned_domain_adaptation = True
            return image
        return Image.fromarray(adapted)

    def _choose_full_reference_path(self) -> Optional[Path]:
        if not self.config.valid_mask.enabled:
            return random.choice(self.reference_paths)
        candidates = random.sample(
            self.reference_paths,
            k=min(10, len(self.reference_paths)),
        )
        for path in candidates:
            is_full = self._reference_is_full.get(path)
            if is_full is None:
                _, reference_mask = load_image_with_valid_mask(
                    path,
                    self.config.valid_mask,
                )
                is_full = (
                    valid_fraction(reference_mask)
                    >= self.config.valid_mask.da_full_tile_min_valid_fraction
                )
                self._reference_is_full[path] = is_full
            if is_full:
                return path
        return None


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def collate_unlabeled_real_batch(batch):
    weak_images, strong_images, metadata = zip(*batch)
    return list(weak_images), list(strong_images), list(metadata)


def _build_domain_adaptation_transform(
    transform_cls, metadata_key: str, reference: np.ndarray, **kwargs
):
    try:
        return transform_cls(metadata_key=metadata_key, **kwargs), {metadata_key: [reference]}
    except TypeError:
        return (
            transform_cls(reference_images=[reference], read_fn=lambda x: x, **kwargs),
            {},
        )


def _adapt_pixel_distribution(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    transform_type: str,
    weight: float,
) -> np.ndarray:
    """Match RGB pixel statistics without Albumentations 2.0.8's uint8 scaling bug."""
    if image.ndim not in (2, 3) or reference.ndim != image.ndim:
        raise ValueError("PDA image and reference must both be grayscale or channel-last images.")
    if image.ndim == 3 and image.shape[2] != reference.shape[2]:
        raise ValueError("PDA image and reference must have the same number of channels.")

    squeeze_channel = image.ndim == 2
    source = np.asarray(image, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    if squeeze_channel:
        source = source[..., None]
        target = target[..., None]

    source_pixels = source.reshape(-1, source.shape[-1])
    target_pixels = target.reshape(-1, target.shape[-1])
    transform_type = str(transform_type).lower()

    if transform_type == "standard":
        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0)
        target_mean = target_pixels.mean(axis=0)
        target_std = target_pixels.std(axis=0)
        standardized = (source_pixels - source_mean) / np.where(
            source_std > 1e-6,
            source_std,
            1.0,
        )
        adapted_pixels = standardized * target_std + target_mean
    elif transform_type == "minmax":
        source_min = source_pixels.min(axis=0)
        source_range = source_pixels.max(axis=0) - source_min
        target_min = target_pixels.min(axis=0)
        target_range = target_pixels.max(axis=0) - target_min
        normalized = (source_pixels - source_min) / np.where(
            source_range > 1e-6,
            source_range,
            1.0,
        )
        adapted_pixels = normalized * target_range + target_min
    elif transform_type == "pca":
        source_mean, source_components = _fit_pca_components(source_pixels)
        target_mean, target_components = _fit_pca_components(target_pixels)
        representation = (source_pixels - source_mean) @ source_components
        adapted_pixels = representation @ target_components.T + target_mean
    else:
        raise ValueError(
            "domain_adaptation.pda_type must be one of: standard, minmax, pca; "
            f"got {transform_type!r}."
        )

    adapted = adapted_pixels.reshape(source.shape)
    blend_weight = float(np.clip(weight, 0.0, 1.0))
    blended = source * (1.0 - blend_weight) + adapted * blend_weight
    result = np.clip(blended, 0.0, 255.0).round().astype(np.uint8)
    return result[..., 0] if squeeze_channel else result


def _fit_pca_components(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pixels.mean(axis=0)
    centered = pixels - mean
    denominator = max(1, centered.shape[0] - 1)
    covariance = (centered.T @ centered) / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    for component_index in range(components.shape[1]):
        component = components[:, component_index]
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            components[:, component_index] *= -1
    return mean, components


def _json_matches_fold(json_path: Path, fold: str) -> bool:
    stem = json_path.stem.lower()
    fold = fold.lower()
    return stem.endswith(f"_{fold}") or f"_{fold}_" in stem


def _resolve_image_path(raster_dir: Path, fold: str, file_name: str) -> Path:
    file_path = Path(file_name)
    candidates = [
        raster_dir / "tiles" / fold / file_path,
        raster_dir / file_path,
        raster_dir / "tiles" / file_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list((raster_dir / "tiles" / fold).glob(file_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not resolve tile {file_name!r} under {raster_dir / 'tiles' / fold}"
    )


def resolve_tile_path(tile_dir: str | Path, file_name: str) -> Path:
    tile_dir = Path(tile_dir)
    file_path = Path(file_name)
    candidates = [
        tile_dir / file_path,
        tile_dir / file_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(tile_dir.rglob(file_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find tile {file_name!r} under {tile_dir}")


def resolve_height_path_for_tile(
    *,
    image_path: str | Path,
    fold: str,
    height_maps_subdir: str = "height_maps",
) -> Path:
    image_path = Path(image_path)
    fold = str(fold)
    tile_dir = image_path.parent
    if tile_dir.name == fold and tile_dir.parent.name == "tiles":
        product_root = tile_dir.parent.parent
        return product_root / height_maps_subdir / fold / f"{image_path.stem}.tif"
    if tile_dir.name == "tiles":
        product_root = tile_dir.parent
        return product_root / height_maps_subdir / f"{image_path.stem}.tif"
    return tile_dir.parent / height_maps_subdir / fold / f"{image_path.stem}.tif"


def load_detector_height_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Detector height-fusion map is missing. Run the configured "
            f"height precompute first or disable height_fusion.generate_missing: {path}"
        )
    height: np.ndarray
    valid_mask: np.ndarray
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            with rasterio.open(path) as dataset:
                masked = dataset.read(1, masked=True)
        height = np.asarray(masked.filled(np.nan), dtype=np.float32)
        valid_mask = ~np.ma.getmaskarray(masked)
    except ImportError:
        with Image.open(path) as image:
            height = np.asarray(image, dtype=np.float32)
        if height.ndim == 3:
            height = height[..., 0]
        valid_mask = np.ones(height.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(height)
    height = np.where(valid_mask, height, 0.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(height), np.ascontiguousarray(valid_mask, dtype=bool)


def load_detector_height_features(
    path: str | Path,
    *,
    use_uncertainty: bool = False,
    uncertainty_fill_value: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_uncertainty:
        return load_detector_height_map(path)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Detector height-fusion map is missing. Run the configured "
            f"height precompute first or disable height_fusion.generate_missing: {path}"
        )
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError as exc:
        raise RuntimeError(
            "Uncertainty-aware detector height fusion requires rasterio to read "
            "two-band height maps."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        with rasterio.open(path) as dataset:
            if int(dataset.count) < 2:
                raise ValueError(
                    "Uncertainty-aware detector height fusion requires a two-band "
                    f"height map [height,log_variance], got {dataset.count} band(s): {path}"
                )
            masked = dataset.read([1, 2], masked=True)
    features = np.asarray(masked.filled(np.nan), dtype=np.float32)
    invalid = np.ma.getmaskarray(masked)
    valid_mask = (~np.any(invalid, axis=0)) & np.all(np.isfinite(features), axis=0)
    features = _mask_detector_height_features(
        features,
        valid_mask,
        uncertainty_fill_value=float(uncertainty_fill_value),
    )
    return np.ascontiguousarray(features), np.ascontiguousarray(valid_mask, dtype=bool)


def _resize_height_and_mask(
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _resize_detector_height_features(height, size=size),
        np.asarray(
            Image.fromarray(valid_mask.astype(np.uint8) * 255).resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
        > 0,
    )


def _height_spatial_shape(height: np.ndarray) -> tuple[int, int]:
    array = np.asarray(height)
    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3:
        return int(array.shape[1]), int(array.shape[2])
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _resize_detector_height_features(
    height: np.ndarray,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(height, dtype=np.float32)
    if array.ndim == 2:
        return np.asarray(
            _float_image_from_array(array).resize(size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    if array.ndim == 3:
        channels = [
            np.asarray(
                _float_image_from_array(channel).resize(size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            for channel in array
        ]
        return np.ascontiguousarray(np.stack(channels, axis=0), dtype=np.float32)
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _mask_detector_height_features(
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    uncertainty_fill_value: float,
) -> np.ndarray:
    array = np.asarray(height, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if array.ndim == 2:
        return np.where(mask, array, 0.0).astype(np.float32, copy=False)
    if array.ndim == 3:
        if array.shape[-2:] != mask.shape:
            raise ValueError(
                "height feature shape does not match valid mask: "
                f"height={array.shape}, valid_mask={mask.shape}."
            )
        fill = np.zeros((array.shape[0], 1, 1), dtype=np.float32)
        if array.shape[0] > 1:
            fill[1:, :, :] = float(uncertainty_fill_value)
        return np.where(mask[None, :, :], array, fill).astype(np.float32, copy=False)
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _append_detector_height_tensor(
    image_tensor: torch.Tensor,
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    uncertainty_fill_value: float,
) -> torch.Tensor:
    if _height_spatial_shape(height) != tuple(int(value) for value in image_tensor.shape[-2:]):
        raise ValueError(
            "Height feature shape does not match tensor image size: "
            f"height={height.shape}, image={tuple(image_tensor.shape[-2:])}."
        )
    height_array = _mask_detector_height_features(
        height,
        valid_mask,
        uncertainty_fill_value=float(uncertainty_fill_value),
    )
    height_tensor = torch.from_numpy(np.ascontiguousarray(height_array))
    if height_tensor.ndim == 2:
        height_tensor = height_tensor.unsqueeze(0)
    return torch.cat([image_tensor, height_tensor], dim=0)


def _crop_detector_height_features(
    height: np.ndarray,
    *,
    top: int,
    left: int,
    side: int,
) -> np.ndarray:
    array = np.asarray(height)
    if array.ndim == 2:
        return array[top : top + side, left : left + side].copy()
    if array.ndim == 3:
        return array[:, top : top + side, left : left + side].copy()
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _flip_detector_height_features_left_right(height: np.ndarray) -> np.ndarray:
    array = np.asarray(height)
    if array.ndim == 2:
        return np.fliplr(array).copy()
    if array.ndim == 3:
        return np.flip(array, axis=-1).copy()
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _flip_detector_height_features_top_bottom(height: np.ndarray) -> np.ndarray:
    array = np.asarray(height)
    if array.ndim == 2:
        return np.flipud(array).copy()
    if array.ndim == 3:
        return np.flip(array, axis=-2).copy()
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _rotate_detector_height_features(
    height: np.ndarray,
    *,
    angle_degrees: float,
    uncertainty_fill_value: float,
) -> np.ndarray:
    array = np.asarray(height, dtype=np.float32)
    if array.ndim == 2:
        return np.asarray(
            _float_image_from_array(array).rotate(
                angle_degrees,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0.0,
            ),
            dtype=np.float32,
        )
    if array.ndim == 3:
        rotated = []
        for channel_index, channel in enumerate(array):
            fill = float(uncertainty_fill_value) if channel_index > 0 else 0.0
            rotated.append(
                np.asarray(
                    _float_image_from_array(channel).rotate(
                        angle_degrees,
                        resample=Image.Resampling.BILINEAR,
                        fillcolor=fill,
                    ),
                    dtype=np.float32,
                )
            )
        return np.ascontiguousarray(np.stack(rotated, axis=0), dtype=np.float32)
    raise ValueError(f"height features must have shape [H,W] or [C,H,W], got {array.shape}.")


def _float_image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")


def is_edge_tile(
    image_path: str | Path,
    black_pixel_threshold: float,
    black_pixel_tolerance: int,
) -> bool:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
    black_pixels = np.all(array <= int(black_pixel_tolerance), axis=-1)
    black_fraction = float(black_pixels.mean()) if black_pixels.size else 0.0
    return black_fraction >= float(black_pixel_threshold)


def _annotations_to_tensors(
    annotations: list[dict[str, Any]],
    force_binary: bool,
    image_width: int,
    image_height: int,
    include_segmentations: bool = False,
) -> tuple:
    boxes = []
    labels = []
    area = []
    iscrowd = []
    segmentations = []
    for ann in annotations:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        x1 = max(0.0, min(float(image_width), x))
        y1 = max(0.0, min(float(image_height), y))
        x2 = max(0.0, min(float(image_width), x + w))
        y2 = max(0.0, min(float(image_height), y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(1 if force_binary else max(1, int(ann.get("category_id", 1))))
        area.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
        iscrowd.append(int(ann.get("iscrowd", 0)))
        segmentations.append(ann.get("segmentation"))
    base = (boxes, labels, area, iscrowd)
    return (*base, segmentations) if include_segmentations else base


def _decode_record_masks(record: dict[str, Any]) -> torch.Tensor:
    height = int(record["height"])
    width = int(record["width"])
    segmentations = list(record.get("segmentations", []))
    return _decode_segmentations(
        segmentations,
        image_height=height,
        image_width=width,
    )


def _decode_segmentations(
    segmentations: list[Any],
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    height = int(image_height)
    width = int(image_width)
    if not segmentations:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    decoded = [
        torch.from_numpy(
            decode_coco_segmentation(
                segmentation,
                image_height=height,
                image_width=width,
            )
        )
        for segmentation in segmentations
    ]
    return torch.stack(decoded, dim=0).to(torch.uint8).contiguous()


def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _sample_float_range(values: list[float]) -> float:
    if not values:
        raise ValueError("Expected a non-empty float range.")
    low = float(min(values))
    high = float(max(values))
    return random.uniform(low, high)


def _sample_int_range(values: list[int]) -> int:
    if not values:
        raise ValueError("Expected a non-empty integer range.")
    low = int(min(values))
    high = int(max(values))
    return random.randint(low, high)


def _downsample_then_upsample(image: Image.Image, scale: float) -> Image.Image:
    scale = float(scale)
    if scale >= 1.0:
        return image
    if scale <= 0.0:
        raise ValueError(f"downsample_scale values must be positive, got {scale}.")
    width, height = image.size
    low_width = max(1, int(round(width * scale)))
    low_height = max(1, int(round(height * scale)))
    low_res = image.resize((low_width, low_height), Image.Resampling.BILINEAR)
    return low_res.resize((width, height), Image.Resampling.BILINEAR)


def _add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    std = max(0.0, float(std))
    if std == 0.0:
        return image
    array = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=array.shape).astype(np.float32)
    noisy = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def _apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    quality = int(np.clip(quality, 1, 100))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    return (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)


def _clip_and_filter_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    iscrowd: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), labels.reshape(0), iscrowd.reshape(0)
    width, height = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    keep = _box_area(boxes) > 1.0
    return boxes[keep], labels[keep], iscrowd[keep]


def _finalize_mask_targets(
    *,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    iscrowd: torch.Tensor,
    masks: torch.Tensor,
    valid_mask: np.ndarray,
    apply_valid_filter: bool,
    min_valid_fraction: float,
    derive_boxes: bool,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if masks.ndim != 3:
        raise ValueError(f"Instance masks must have shape [N, H, W], got {tuple(masks.shape)}.")
    if not (masks.shape[0] == boxes.shape[0] == labels.shape[0] == iscrowd.shape[0]):
        raise ValueError(
            "Mask targets are misaligned: "
            f"masks={masks.shape[0]}, boxes={boxes.shape[0]}, "
            f"labels={labels.shape[0]}, iscrowd={iscrowd.shape[0]}."
        )
    masks = masks.to(torch.bool)
    keep = masks.flatten(1).sum(1) > 1
    if apply_valid_filter and masks.shape[0] > 0:
        valid = torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=bool))
        mask_area = masks.flatten(1).sum(1).clamp_min(1)
        valid_area = (masks & valid.unsqueeze(0)).flatten(1).sum(1)
        keep &= valid_area > 1
        keep &= valid_area.to(torch.float32) / mask_area.to(torch.float32) >= float(
            min_valid_fraction
        )
        masks = masks & valid.unsqueeze(0)

    masks = masks[keep].contiguous()
    labels = labels[keep]
    iscrowd = iscrowd[keep]
    boxes = boxes[keep]
    if derive_boxes:
        boxes = _boxes_from_masks(masks)
    else:
        boxes, labels, iscrowd, masks = _clip_and_filter_mask_targets_by_box(
            boxes,
            labels,
            iscrowd,
            masks,
            image_size,
        )
    return boxes, labels, iscrowd, masks


def _clip_and_filter_mask_targets_by_box(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    iscrowd: torch.Tensor,
    masks: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return (
            boxes.reshape(0, 4),
            labels.reshape(0),
            iscrowd.reshape(0),
            masks,
        )
    width, height = image_size
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    keep = _box_area(boxes) > 1.0
    return boxes[keep], labels[keep], iscrowd[keep], masks[keep]


def _boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N, H, W], got {tuple(masks.shape)}.")
    if masks.shape[0] == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    boxes = []
    for mask in masks:
        ys, xs = torch.where(mask > 0)
        if xs.numel() == 0:
            boxes.append(mask.new_zeros((4,), dtype=torch.float32))
            continue
        boxes.append(
            torch.stack(
                [
                    xs.min().to(torch.float32),
                    ys.min().to(torch.float32),
                    (xs.max() + 1).to(torch.float32),
                    (ys.max() + 1).to(torch.float32),
                ]
            )
        )
    return torch.stack(boxes, dim=0)


def _rotate_instance_masks(
    masks: torch.Tensor,
    *,
    angle_degrees: float,
) -> torch.Tensor:
    if masks.shape[0] == 0:
        return masks.contiguous()
    rotated = []
    for mask in masks.detach().cpu():
        array = np.asarray(
            Image.fromarray(mask.numpy().astype(np.uint8) * 255).rotate(
                angle_degrees,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
        )
        rotated.append(torch.from_numpy(np.ascontiguousarray(array > 0)))
    return torch.stack(rotated, dim=0).to(
        device=masks.device,
        dtype=torch.uint8,
    )


def _filter_targets_by_valid_mask(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    iscrowd: torch.Tensor,
    valid_mask: np.ndarray,
    *,
    min_valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, labels, iscrowd

    mask_tensor = torch.from_numpy(np.ascontiguousarray(valid_mask))
    keep_tensor = box_valid_fractions(mask_tensor, boxes) >= float(min_valid_fraction)
    return boxes[keep_tensor], labels[keep_tensor], iscrowd[keep_tensor]


def _rotate_boxes(
    boxes: torch.Tensor, width: int, height: int, angle_degrees: float
) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # PIL rotates positive angles counter-clockwise around (width / 2, height / 2).
    # Pixel coordinates have a downward-pointing y axis, so their forward
    # rotation matrix has the opposite sine signs from Cartesian coordinates.
    cx = width / 2.0
    cy = height / 2.0
    corners = torch.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        dim=1,
    )
    x = corners[..., 0] - cx
    y = corners[..., 1] - cy
    rotated_x = cos_a * x + sin_a * y + cx
    rotated_y = -sin_a * x + cos_a * y + cy
    rotated = torch.stack([rotated_x, rotated_y], dim=-1)
    return torch.stack(
        [
            rotated[..., 0].min(dim=1).values,
            rotated[..., 1].min(dim=1).values,
            rotated[..., 0].max(dim=1).values,
            rotated[..., 1].max(dim=1).values,
        ],
        dim=1,
    )


def _shift_hue(image: Image.Image, delta: float) -> Image.Image:
    hsv = image.convert("HSV")
    arr = np.asarray(hsv).copy()
    shift = int(delta * 255)
    arr[..., 0] = (arr[..., 0].astype(np.int16) + shift) % 255
    return Image.fromarray(arr, mode="HSV").convert("RGB")


def _collect_reference_images(config: DetectorTrainingConfig) -> list[Path]:
    da_cfg = config.domain_adaptation
    roots = _domain_adaptation_reference_roots(da_cfg)
    if not roots:
        return []

    rng = random.Random(config.seed)
    paths_by_root: list[list[Path]] = []
    for root in roots:
        paths = _collect_reference_images_from_root(root)
        if paths:
            rng.shuffle(paths)
            paths_by_root.append(paths)
    if not paths_by_root:
        return []
    return _balanced_reference_sample(
        paths_by_root,
        max_count=int(da_cfg.max_reference_images),
        rng=rng,
    )


def _domain_adaptation_reference_roots(da_cfg: Any) -> list[Path]:
    roots: list[Path] = []
    raw_roots = []
    if getattr(da_cfg, "reference_root_path", None):
        raw_roots.append(da_cfg.reference_root_path)
    raw_roots.extend(getattr(da_cfg, "reference_root_paths", []) or [])
    seen: set[Path] = set()
    for raw_root in raw_roots:
        if raw_root is None or not str(raw_root).strip():
            continue
        root = Path(raw_root)
        if root in seen:
            continue
        roots.append(root)
        seen.add(root)
    return roots


def _collect_reference_images_from_root(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            return [root]
        return _collect_reference_images_from_manifest(root.parent, root)

    manifest_path = root / "subset_manifest.json"
    if manifest_path.exists():
        return _collect_reference_images_from_manifest(root, manifest_path)

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return paths


def _collect_reference_images_from_manifest(root: Path, manifest_path: Path) -> list[Path]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return []
    paths: list[Path] = []
    for tile in manifest.get("selected_tiles", []):
        relative_path = tile.get("relative_path")
        if not relative_path:
            continue
        relative = Path(relative_path)
        candidate = relative if relative.is_absolute() else root / relative
        if candidate.exists() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            paths.append(candidate)
    return paths


def _balanced_reference_sample(
    paths_by_root: list[list[Path]],
    *,
    max_count: int,
    rng: random.Random,
) -> list[Path]:
    if max_count <= 0:
        return []
    total_count = sum(len(paths) for paths in paths_by_root)
    if total_count <= max_count:
        combined = [path for paths in paths_by_root for path in paths]
        rng.shuffle(combined)
        return combined

    queues = [list(paths) for paths in paths_by_root if paths]
    selected: list[Path] = []
    while queues and len(selected) < max_count:
        next_queues: list[list[Path]] = []
        for queue in queues:
            if len(selected) >= max_count:
                next_queues.append(queue)
                continue
            selected.append(queue.pop())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    rng.shuffle(selected)
    return selected


def _load_reference_image(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    )
