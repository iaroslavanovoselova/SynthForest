from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from synthforest.training.checkpoint import load_checkpoint_model
from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.evaluation.aggregation import macro_average_metrics
from synthforest.training.evaluation.segmentation import (
    SegmentationCocoAccumulator,
)
from synthforest.training.valid_masks import (
    filter_detections_by_valid_masks,
)
from synthforest.training.validation_data import (
    build_real_validation_loaders,
    build_synthetic_fold_loader,
    build_synthetic_validation_loader,
)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: Optional[DetectorTrainingConfig] = None,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    validation_tasks = set(config.validation.tasks) if config is not None else {"bbox"}
    metric = _build_map_metric(
        device,
        max_detection_thresholds=(
            config.validation.map_max_detection_thresholds if config is not None else None
        ),
    )
    dense_metric = _build_dense_map_metric(device, config)
    segmentation_metric = (
        SegmentationCocoAccumulator(
            mask_probability_threshold=config.mask_probability_threshold,
        )
        if config is not None and "segm" in validation_tasks
        else None
    )
    num_images = 0
    num_predictions = 0
    num_targets = 0
    num_predictions_before_mask_filter = 0
    num_predictions_filtered_invalid = 0
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = [image.to(device) for image in images]
            targets = [
                {
                    key: (value.to(device) if torch.is_tensor(value) and key != "masks" else value)
                    for key, value in target.items()
                }
                for target in targets
            ]
            outputs = model(images)
            batch_valid_masks = [
                target.get(
                    "valid_mask",
                    torch.ones(
                        image.shape[-2:],
                        dtype=torch.bool,
                        device=image.device,
                    ),
                )
                for image, target in zip(images, targets)
            ]
            outputs, filter_stats = filter_detections_by_valid_masks(
                outputs,
                batch_valid_masks,
                min_valid_fraction=(
                    config.valid_mask.prediction_min_valid_fraction
                    if config is not None and config.valid_mask.enabled
                    else None
                ),
            )
            num_predictions_before_mask_filter += filter_stats["num_predictions_before_mask_filter"]
            num_predictions_filtered_invalid += filter_stats["num_predictions_filtered_invalid"]
            num_images += len(images)
            num_predictions += sum(int(output["boxes"].shape[0]) for output in outputs)
            num_targets += sum(int(target["boxes"].shape[0]) for target in targets)
            if metric is not None:
                metric.update(outputs, targets)
            if dense_metric is not None:
                dense_metric.update(outputs, targets)
            if segmentation_metric is not None:
                segmentation_metric.update(
                    outputs,
                    targets,
                    batch_valid_masks,
                )
                for output in outputs:
                    output.pop("masks", None)

    metrics = {
        "num_images": float(num_images),
        "num_predictions": float(num_predictions),
        "num_predictions_before_mask_filter": float(num_predictions_before_mask_filter),
        "num_predictions_filtered_invalid": float(num_predictions_filtered_invalid),
        "prediction_invalid_fraction": (
            num_predictions_filtered_invalid / num_predictions_before_mask_filter
            if num_predictions_before_mask_filter > 0
            else 0.0
        ),
        "num_targets": float(num_targets),
    }
    if metric is None:
        metrics.update(
            {
                "bbox_map": float("nan"),
                "bbox_map_50": float("nan"),
                "bbox_map_75": float("nan"),
            }
        )
        if segmentation_metric is not None:
            _add_segmentation_metric_values(
                metrics,
                segmentation_metric,
                config=config,
            )
        else:
            metrics["segm_available"] = False
        return metrics

    _add_metric_values(metrics, metric.compute(), prefix="bbox_")
    if dense_metric is not None:
        dense_raw = dense_metric.compute()
        _add_dense_metric_values(
            metrics,
            dense_raw,
            dense_max=int(config.validation.dense_eval_max_detections),
        )
    if segmentation_metric is not None:
        _add_segmentation_metric_values(
            metrics,
            segmentation_metric,
            config=config,
        )
    else:
        metrics["segm_available"] = False
    return metrics


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: Optional[DetectorTrainingConfig] = None,
    config_path: Optional[str | Path] = None,
    validation_set: str = "both",
    fold: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
) -> dict:
    if validation_set not in {"synthetic", "real", "both"}:
        raise ValueError("validation_set must be synthetic, real, or both.")
    if fold is not None and fold not in {"train", "valid", "test"}:
        raise ValueError("fold must be train, valid, or test.")
    if validation_set in {"real", "both"} and fold not in {None, "valid"}:
        raise ValueError("Real evaluation through this CLI uses the registered valid fold.")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, resolved_config, _ = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        config=config,
        config_path=config_path,
    )

    metrics: dict = {}
    if validation_set in {"synthetic", "both"}:
        if fold in {None, "valid"}:
            loader = build_synthetic_validation_loader(resolved_config)
        else:
            loader = build_synthetic_fold_loader(resolved_config, fold)
        metrics["valid_synthetic"] = evaluate_model(
            model,
            loader,
            resolved_device,
            config=resolved_config,
        )
    if validation_set in {"real", "both"}:
        real_loaders = build_real_validation_loaders(resolved_config)
        real_by_dataset = {
            dataset_name: evaluate_model(
                model,
                loader,
                resolved_device,
                config=resolved_config,
            )
            for dataset_name, loader in real_loaders.items()
        }
        metrics["valid_real"] = {
            **macro_average_metrics(real_by_dataset),
            "by_dataset": real_by_dataset,
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def _build_dense_map_metric(
    device: torch.device,
    config: Optional[DetectorTrainingConfig],
):
    if config is None or config.validation.dense_eval_max_detections is None:
        return None
    dense_max = int(config.validation.dense_eval_max_detections)
    official_thresholds = [int(value) for value in config.validation.map_max_detection_thresholds]
    if dense_max <= official_thresholds[-1]:
        return None
    return _build_map_metric(
        device,
        max_detection_thresholds=[1, 10, dense_max],
        extended_summary=True,
    )


def _build_map_metric(
    device: torch.device,
    max_detection_thresholds: Optional[list[int]] = None,
    extended_summary: bool = False,
):
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        return None
    kwargs = {
        "iou_type": "bbox",
        "extended_summary": extended_summary,
    }
    if max_detection_thresholds:
        kwargs["max_detection_thresholds"] = [int(value) for value in max_detection_thresholds]
    try:
        return MeanAveragePrecision(**kwargs).to(device)
    except TypeError as exc:
        if "max_detection_thresholds" not in kwargs:
            raise
        print(
            "torchmetrics mAP backend does not accept max_detection_thresholds; "
            f"falling back to its default thresholds: {exc}"
        )
        kwargs.pop("max_detection_thresholds")
        return MeanAveragePrecision(**kwargs).to(device)
    except ModuleNotFoundError as exc:
        print(f"torchmetrics mAP backend is unavailable; falling back to count metrics: {exc}")
        return None


def _add_dense_metric_values(
    metrics: dict[str, float],
    raw: dict[str, torch.Tensor],
    dense_max: int,
) -> None:
    precision = raw.get("precision")
    recall = raw.get("recall")
    if not torch.is_tensor(precision) or precision.ndim != 5:
        raise RuntimeError(
            "torchmetrics did not return the precision tensor required for dense mAP evaluation."
        )
    if not torch.is_tensor(recall) or recall.ndim != 4:
        raise RuntimeError(
            "torchmetrics did not return the recall tensor required for dense mAR evaluation."
        )

    max_det_index = precision.shape[-1] - 1
    all_areas_index = 0
    metrics["bbox_dense_map"] = _valid_tensor_mean(
        precision[:, :, :, all_areas_index, max_det_index]
    )
    metrics["bbox_dense_map_50"] = _valid_tensor_mean(
        precision[0, :, :, all_areas_index, max_det_index]
    )
    iou_75_index = 5
    if precision.shape[0] > iou_75_index:
        metrics["bbox_dense_map_75"] = _valid_tensor_mean(
            precision[
                iou_75_index,
                :,
                :,
                all_areas_index,
                max_det_index,
            ]
        )
    dense_mar = _valid_tensor_mean(recall[:, :, all_areas_index, max_det_index])
    metrics["bbox_dense_mar"] = dense_mar
    metrics[f"bbox_dense_mar_{dense_max}"] = dense_mar
    metrics["bbox_dense_mar_50"] = _valid_tensor_mean(recall[0, :, all_areas_index, max_det_index])
    if recall.shape[0] > iou_75_index:
        metrics["bbox_dense_mar_75"] = _valid_tensor_mean(
            recall[iou_75_index, :, all_areas_index, max_det_index]
        )


def _valid_tensor_mean(values: torch.Tensor) -> float:
    valid = values[values >= 0]
    if valid.numel() == 0:
        return -1.0
    return float(valid.float().mean().detach().cpu())


def _add_metric_values(
    metrics: dict[str, float],
    raw: dict[str, torch.Tensor],
    prefix: str = "",
    skip_negative: bool = False,
) -> None:
    for key, value in raw.items():
        if not (torch.is_tensor(value) and value.numel() == 1):
            continue
        scalar = float(value.detach().cpu())
        if skip_negative and scalar < 0:
            continue
        metrics[f"{prefix}{key}"] = scalar


def _add_segmentation_metric_values(
    metrics: dict[str, float],
    accumulator: SegmentationCocoAccumulator,
    *,
    config: DetectorTrainingConfig,
) -> None:
    standard = accumulator.compute(
        max_detections=[int(value) for value in config.validation.map_max_detection_thresholds],
    )
    _add_coco_summary(
        metrics,
        standard,
        prefix="segm_",
    )
    dense_max = config.validation.dense_eval_max_detections
    if dense_max is not None and int(dense_max) > int(
        config.validation.map_max_detection_thresholds[-1]
    ):
        dense = accumulator.compute(
            max_detections=[1, 10, int(dense_max)],
        )
        _add_coco_summary(
            metrics,
            dense,
            prefix="segm_dense_",
            include_counts=False,
        )
    metrics["segm_available"] = True


def _add_coco_summary(
    metrics: dict[str, float],
    summary: dict[str, float],
    *,
    prefix: str,
    include_counts: bool = True,
) -> None:
    key_mapping = {
        "AP": "map",
        "AP50": "map_50",
        "AP75": "map_75",
        "AR": "mar",
        "AR50": "mar_50",
        "AR75": "mar_75",
    }
    for source, target in key_mapping.items():
        value = summary.get(source)
        if isinstance(value, (int, float)):
            metrics[f"{prefix}{target}"] = float(value)
    for source, value in summary.items():
        if source.startswith("AR_") and source[3:].isdigit():
            metrics[f"{prefix}mar_{source[3:]}"] = float(value)
    if include_counts:
        for key in ("num_images", "num_truths", "num_preds"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                metrics[f"{prefix}{key}"] = float(value)
