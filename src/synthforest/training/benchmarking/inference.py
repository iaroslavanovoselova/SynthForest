from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from synthforest.training.coco_masks import (
    encode_binary_mask,
    segmentation_area,
)
from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.dataloader import dataloader_worker_kwargs
from synthforest.training.dataset import (
    _height_spatial_shape,
    _mask_detector_height_features,
    _pil_to_float_tensor,
    _resize_height_and_mask,
    is_edge_tile,
    load_detector_height_features,
    resolve_height_path_for_tile,
    resolve_tile_path,
)
from synthforest.training.valid_masks import (
    filter_detections_by_valid_masks,
    image_and_mask_to_tensors,
    load_image_with_valid_mask,
)


@dataclass(frozen=True)
class TileRecord:
    image_info: dict[str, Any]
    image_path: Path


@dataclass(frozen=True)
class ImagePrediction:
    image_info: dict[str, Any]
    image_path: Path
    boxes_xyxy: list[list[float]]
    scores: list[float]
    labels: list[int]
    segmentations: Optional[list[dict[str, Any]]] = None
    mask_areas: Optional[list[float]] = None


class BenchmarkTileDataset:
    def __init__(
        self,
        tile_dir: str | Path,
        truth_coco_path: str | Path,
        max_tiles: Optional[int] = None,
        drop_edge_tiles: bool = False,
        edge_tile_black_pixel_threshold: float = 0.5,
        edge_tile_black_pixel_tolerance: int = 0,
        training_config: Optional[DetectorTrainingConfig] = None,
    ) -> None:
        self.tile_dir = Path(tile_dir)
        self.truth_coco_path = Path(truth_coco_path)
        self.training_config = training_config
        with open(self.truth_coco_path, "r", encoding="utf-8") as f:
            self.truth_coco = json.load(f)
        images = list(self.truth_coco.get("images", []))
        records: list[TileRecord] = []
        self.skipped_edge_file_names: list[str] = []
        for image_info in images:
            image_path = resolve_tile_path(self.tile_dir, image_info["file_name"])
            if drop_edge_tiles and is_edge_tile(
                image_path=image_path,
                black_pixel_threshold=edge_tile_black_pixel_threshold,
                black_pixel_tolerance=edge_tile_black_pixel_tolerance,
            ):
                self.skipped_edge_file_names.append(str(image_info["file_name"]))
                continue
            records.append(TileRecord(image_info=dict(image_info), image_path=image_path))
        if max_tiles is not None:
            records = records[: int(max_tiles)]
        self.records = records
        if not self.records:
            if drop_edge_tiles and self.skipped_edge_file_names:
                raise ValueError(
                    f"No full tile images remain after edge filtering for {self.truth_coco_path}. "
                    f"Skipped {len(self.skipped_edge_file_names)} edge tile(s)."
                )
            raise ValueError(f"No tile images found in truth COCO: {self.truth_coco_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if self.training_config is None:
            image = Image.open(record.image_path).convert("RGB")
            image_tensor = _pil_to_float_tensor(image)
            valid_mask = image_tensor.new_ones(
                image_tensor.shape[-2:],
                dtype=torch.bool,
            )
        else:
            image, valid_mask_array = load_image_with_valid_mask(
                record.image_path,
                self.training_config.valid_mask,
            )
            height = None
            if self.training_config.height_fusion.enabled:
                height, height_valid_mask = load_detector_height_features(
                    resolve_height_path_for_tile(
                        image_path=record.image_path,
                        fold=record.image_path.parent.name,
                        height_maps_subdir=(self.training_config.height_fusion.height_maps_subdir),
                    ),
                    use_uncertainty=bool(self.training_config.height_fusion.use_uncertainty),
                    uncertainty_fill_value=float(
                        self.training_config.height_fusion.uncertainty_log_variance_max
                    ),
                )
                if _height_spatial_shape(height) != (image.height, image.width):
                    height, height_valid_mask = _resize_height_and_mask(
                        height,
                        height_valid_mask,
                        size=image.size,
                    )
                valid_mask_array = np.ascontiguousarray(
                    valid_mask_array & height_valid_mask,
                    dtype=bool,
                )
            image_tensor, valid_mask = image_and_mask_to_tensors(
                image,
                valid_mask_array,
                self.training_config.augmentation.image_mean,
            )
            if height is not None:
                height = _mask_detector_height_features(
                    height,
                    valid_mask_array,
                    uncertainty_fill_value=float(
                        self.training_config.height_fusion.uncertainty_log_variance_max
                    ),
                )
                height_tensor = torch.from_numpy(np.ascontiguousarray(height))
                if height_tensor.ndim == 2:
                    height_tensor = height_tensor.unsqueeze(0)
                image_tensor = torch.cat(
                    [image_tensor, height_tensor],
                    dim=0,
                )
        return image_tensor, valid_mask, record

    @property
    def evaluated_file_names(self) -> list[str]:
        return [str(record.image_info["file_name"]) for record in self.records]

    @property
    def edge_filter_stats(self) -> dict[str, int]:
        return {
            "num_evaluated_tiles": len(self.records),
            "num_skipped_edge_tiles": len(self.skipped_edge_file_names),
        }


def collate_benchmark_batch(batch):
    images, valid_masks, records = zip(*batch)
    return list(images), list(valid_masks), list(records)


def run_product_inference(
    model,
    tile_dir: str | Path,
    truth_coco_path: str | Path,
    device,
    batch_size: int,
    num_workers: int,
    max_tiles: Optional[int] = None,
    drop_edge_tiles: bool = False,
    edge_tile_black_pixel_threshold: float = 0.5,
    edge_tile_black_pixel_tolerance: int = 0,
    avoid_fork: bool = False,
    training_config: Optional[DetectorTrainingConfig] = None,
    mask_probability_threshold: Optional[float] = None,
    source_gsd_m: Optional[float] = None,
) -> tuple[list[ImagePrediction], list[str], dict[str, int]]:
    import torch
    from torch.utils.data import DataLoader

    if (
        training_config is not None
        and training_config.height_fusion.enabled
        and training_config.height_fusion.generate_missing
    ):
        _ensure_benchmark_height_maps(
            tile_dir=tile_dir,
            truth_coco_path=truth_coco_path,
            training_config=training_config,
            batch_size=batch_size,
            device=device,
            source_gsd_m=source_gsd_m,
        )

    dataset = BenchmarkTileDataset(
        tile_dir=tile_dir,
        truth_coco_path=truth_coco_path,
        max_tiles=max_tiles,
        drop_edge_tiles=drop_edge_tiles,
        edge_tile_black_pixel_threshold=edge_tile_black_pixel_threshold,
        edge_tile_black_pixel_tolerance=edge_tile_black_pixel_tolerance,
        training_config=training_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_benchmark_batch,
        **dataloader_worker_kwargs(num_workers, avoid_fork=avoid_fork),
    )
    predictions: list[ImagePrediction] = []
    prediction_filter_stats = {
        "num_predictions_before_mask_filter": 0,
        "num_predictions_filtered_invalid": 0,
    }
    model.eval()
    with torch.no_grad():
        for images, valid_masks, records in loader:
            images = [image.to(device) for image in images]
            valid_masks = [valid_mask.to(device) for valid_mask in valid_masks]
            outputs = model(images)
            outputs, batch_filter_stats = filter_detections_by_valid_masks(
                outputs,
                valid_masks,
                min_valid_fraction=(
                    training_config.valid_mask.prediction_min_valid_fraction
                    if training_config is not None and training_config.valid_mask.enabled
                    else None
                ),
            )
            for key, value in batch_filter_stats.items():
                prediction_filter_stats[key] += int(value)
            threshold = (
                float(mask_probability_threshold)
                if mask_probability_threshold is not None
                else float(training_config.mask_probability_threshold)
                if training_config is not None
                else 0.5
            )
            for output, record, valid_mask in zip(
                outputs,
                records,
                valid_masks,
            ):
                predictions.append(
                    _prediction_from_output(
                        output,
                        record,
                        valid_mask=valid_mask,
                        mask_probability_threshold=threshold,
                    )
                )
    checked = prediction_filter_stats["num_predictions_before_mask_filter"]
    removed = prediction_filter_stats["num_predictions_filtered_invalid"]
    inference_stats = {
        **dataset.edge_filter_stats,
        **prediction_filter_stats,
        "prediction_invalid_fraction": removed / checked if checked > 0 else 0.0,
    }
    return predictions, dataset.evaluated_file_names, inference_stats


def _ensure_benchmark_height_maps(
    *,
    tile_dir: str | Path,
    truth_coco_path: str | Path,
    training_config: DetectorTrainingConfig,
    batch_size: int,
    device,
    source_gsd_m: Optional[float],
) -> None:
    fusion = training_config.height_fusion
    from synthforest.training.height.detector_precompute import (
        resolve_detector_height_tile_records_from_coco,
        run_detector_height_precompute_for_coco,
    )

    split = Path(tile_dir).name
    records = resolve_detector_height_tile_records_from_coco(
        tile_dir=tile_dir,
        truth_coco_path=truth_coco_path,
        split=split,
        height_maps_subdir=fusion.height_maps_subdir,
    )
    if not records:
        return
    print(
        "Height fusion precompute: checking/generating "
        f"{len(records)} benchmark height map(s) for {tile_dir}."
    )
    run_detector_height_precompute_for_coco(
        tile_dir=tile_dir,
        truth_coco_path=truth_coco_path,
        split=split,
        height_maps_subdir=fusion.height_maps_subdir,
        height_checkpoint_path=fusion.height_checkpoint_path,
        height_train_config_path=fusion.height_train_config_path,
        batch_size=max(1, int(batch_size)),
        device=device,
        source_gsd=float(source_gsd_m or fusion.source_gsd_m),
        target_gsd=float(fusion.target_gsd_m),
        precompute_match_checkpoint=bool(fusion.precompute_match_checkpoint),
    )


def _prediction_from_output(
    output: dict[str, Any],
    record: TileRecord,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    mask_probability_threshold: float = 0.5,
) -> ImagePrediction:
    boxes = output.get("boxes")
    scores = output.get("scores")
    labels = output.get("labels")
    if boxes is None:
        boxes_list: list[list[float]] = []
    else:
        boxes_list = boxes.detach().cpu().float().tolist()
    if scores is None:
        scores_list: list[float] = [1.0 for _ in boxes_list]
    else:
        scores_list = [float(value) for value in scores.detach().cpu().float().tolist()]
    if labels is None:
        labels_list: list[int] = [1 for _ in boxes_list]
    else:
        labels_list = [int(value) for value in labels.detach().cpu().tolist()]
    segmentations = None
    mask_areas = None
    masks = output.pop("masks", None)
    if torch.is_tensor(masks):
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[0] != len(boxes_list):
            raise ValueError(
                "Predicted masks must align with predicted boxes and have shape "
                f"[N,1,H,W] or [N,H,W], got {tuple(masks.shape)}."
            )
        valid_cpu = None
        if valid_mask is not None:
            valid_cpu = valid_mask.detach().cpu().to(torch.bool)
            if valid_cpu.shape != masks.shape[-2:]:
                raise ValueError(
                    "Prediction valid mask does not match predicted mask size: "
                    f"{tuple(valid_cpu.shape)} vs {tuple(masks.shape[-2:])}."
                )
        segmentations = []
        for mask in masks:
            binary_mask = mask.detach().cpu() >= float(mask_probability_threshold)
            if valid_cpu is not None:
                binary_mask &= valid_cpu
            segmentations.append(encode_binary_mask(binary_mask.numpy()))
        mask_areas = [segmentation_area(segmentation) for segmentation in segmentations]

    return ImagePrediction(
        image_info=dict(record.image_info),
        image_path=record.image_path,
        boxes_xyxy=boxes_list,
        scores=scores_list,
        labels=labels_list,
        segmentations=segmentations,
        mask_areas=mask_areas,
    )
