from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from synthforest.training.coco_masks import (
    encode_binary_mask,
    segmentation_area,
)


class SegmentationCocoAccumulator:
    """Accumulate compressed COCO RLEs instead of full-resolution mask tensors."""

    def __init__(self, *, mask_probability_threshold: float) -> None:
        self.mask_probability_threshold = float(mask_probability_threshold)
        self.images: list[dict[str, Any]] = []
        self.ground_truth_annotations: list[dict[str, Any]] = []
        self.prediction_annotations: list[dict[str, Any]] = []
        self.category_ids: set[int] = set()
        self._next_image_id = 1
        self._next_ground_truth_id = 1
        self._next_prediction_id = 1

    def update(
        self,
        outputs: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        valid_masks: list[torch.Tensor],
    ) -> None:
        if not (len(outputs) == len(targets) == len(valid_masks)):
            raise ValueError(
                "Segmentation evaluation expects one output, target, and valid mask per image."
            )
        for output, target, valid_mask in zip(outputs, targets, valid_masks):
            target_masks = target.get("masks")
            if not torch.is_tensor(target_masks):
                raise ValueError("Segmentation evaluation requires masks for every target.")
            height, width = _mask_image_size(
                target_masks,
                valid_mask=valid_mask,
            )
            image_id = self._next_image_id
            self._next_image_id += 1
            self.images.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": f"validation_{image_id:08d}.png",
                }
            )
            self._add_ground_truth(
                image_id=image_id,
                target=target,
                valid_mask=valid_mask,
                height=height,
                width=width,
            )
            self._add_predictions(
                image_id=image_id,
                output=output,
                valid_mask=valid_mask,
                height=height,
                width=width,
            )

    def compute(
        self,
        *,
        max_detections: list[int],
        iou_thresholds: Optional[list[float]] = None,
    ) -> dict[str, float]:
        if len(max_detections) != 3:
            raise ValueError("COCO segmentation max_detections must contain exactly three values.")
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as exc:
            raise ImportError("Segmentation COCO evaluation requires pycocotools.") from exc

        categories = [
            {
                "id": category_id,
                "name": f"class_{category_id}",
                "supercategory": "object",
            }
            for category_id in sorted(self.category_ids or {1})
        ]
        truth = COCO()
        truth.dataset = {
            "images": list(self.images),
            "annotations": list(self.ground_truth_annotations),
            "categories": categories,
        }
        truth.createIndex()
        predictions = COCO()
        predictions.dataset = {
            "images": list(self.images),
            "annotations": list(self.prediction_annotations),
            "categories": categories,
        }
        predictions.createIndex()

        evaluator = COCOeval(truth, predictions, "segm")
        evaluator.params.imgIds = sorted(truth.getImgIds())
        evaluator.params.maxDets = [int(value) for value in max_detections]
        if iou_thresholds is not None:
            evaluator.params.iouThrs = np.asarray(
                sorted({float(value) for value in iou_thresholds}),
                dtype=float,
            )
        evaluator.evaluate()
        evaluator.accumulate()

        from synthforest.training.benchmarking.metrics import (
            _summarize_coco_eval,
        )

        metrics = _summarize_coco_eval(
            evaluator,
            primary_iou_thresholds=([float(value) for value in evaluator.params.iouThrs]),
        )
        metrics.update(
            {
                "num_images": float(len(self.images)),
                "num_truths": float(len(self.ground_truth_annotations)),
                "num_preds": float(len(self.prediction_annotations)),
            }
        )
        return metrics

    def _add_ground_truth(
        self,
        *,
        image_id: int,
        target: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        masks = target["masks"].detach().cpu().to(torch.bool)
        valid = valid_mask.detach().cpu().to(torch.bool)
        if valid.shape != (height, width):
            raise ValueError(
                f"Valid mask shape {tuple(valid.shape)} does not match {(height, width)}."
            )
        masks = masks & valid.unsqueeze(0)
        labels = target["labels"].detach().cpu()
        iscrowd = target.get("iscrowd")
        crowds = (
            iscrowd.detach().cpu().to(torch.int64)
            if torch.is_tensor(iscrowd)
            else torch.zeros_like(labels, dtype=torch.int64)
        )
        if not (masks.shape[0] == labels.shape[0] == crowds.shape[0]):
            raise ValueError("Ground-truth mask fields have inconsistent lengths.")

        for index in range(masks.shape[0]):
            mask = masks[index].numpy()
            if not mask.any():
                continue
            category_id = max(1, int(labels[index]))
            self.category_ids.add(category_id)
            rle = encode_binary_mask(mask)
            self.ground_truth_annotations.append(
                {
                    "id": self._next_ground_truth_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": _bbox_from_binary_mask(mask),
                    "area": segmentation_area(rle),
                    "iscrowd": int(crowds[index]),
                }
            )
            self._next_ground_truth_id += 1

    def _add_predictions(
        self,
        *,
        image_id: int,
        output: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        masks = output.get("masks")
        scores = output.get("scores")
        labels = output.get("labels")
        boxes = output.get("boxes")
        if not torch.is_tensor(masks):
            raise ValueError("Mask R-CNN output is missing the masks tensor.")
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(
                f"Predicted masks must have shape [N,1,H,W] or [N,H,W], got {masks.shape}."
            )
        count = int(masks.shape[0])
        if not (
            torch.is_tensor(scores)
            and torch.is_tensor(labels)
            and scores.shape[0] == labels.shape[0] == count
        ):
            raise ValueError("Predicted mask fields have inconsistent lengths.")

        valid = valid_mask.detach().cpu().to(torch.bool)
        if valid.shape != (height, width):
            raise ValueError(
                f"Valid mask shape {tuple(valid.shape)} does not match {(height, width)}."
            )
        labels_cpu = labels.detach().cpu()
        scores_cpu = scores.detach().cpu()
        boxes_cpu = boxes.detach().cpu().to(torch.float32) if torch.is_tensor(boxes) else None

        for index in range(count):
            binary_mask = (masks[index].detach().cpu() >= self.mask_probability_threshold) & valid
            category_id = max(1, int(labels_cpu[index]))
            self.category_ids.add(category_id)
            rle = encode_binary_mask(binary_mask.numpy())
            bbox = (
                _xyxy_to_xywh(boxes_cpu[index].tolist())
                if boxes_cpu is not None
                else _bbox_from_binary_mask(binary_mask.numpy())
            )
            self.prediction_annotations.append(
                {
                    "id": self._next_prediction_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": segmentation_area(rle),
                    "iscrowd": 0,
                    "score": float(scores_cpu[index]),
                }
            )
            self._next_prediction_id += 1


def _mask_image_size(
    target_masks: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> tuple[int, int]:
    if target_masks.ndim != 3:
        raise ValueError(f"Target masks must have shape [N,H,W], got {target_masks.shape}.")
    if target_masks.shape[-2] > 0 and target_masks.shape[-1] > 0:
        return int(target_masks.shape[-2]), int(target_masks.shape[-1])
    return int(valid_mask.shape[-2]), int(valid_mask.shape[-1])


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _bbox_from_binary_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = float(xs.min())
    y1 = float(ys.min())
    return [
        x1,
        y1,
        float(xs.max() + 1) - x1,
        float(ys.max() + 1) - y1,
    ]
