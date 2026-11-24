from __future__ import annotations

from typing import Any

import torch

from synthforest.training.valid_masks import box_valid_fractions


def copy_ema_teacher_weights(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
) -> None:
    teacher.load_state_dict(student.state_dict())
    freeze_teacher(teacher)


def freeze_teacher(teacher: torch.nn.Module) -> None:
    teacher.requires_grad_(False)
    backbone = getattr(teacher, "backbone", None)
    if backbone is not None and hasattr(backbone, "backbone_grad_enabled"):
        backbone.backbone_grad_enabled = False
        dinov3 = getattr(backbone, "dinov3", None)
        if dinov3 is not None:
            dinov3.eval()
    teacher.eval()


def update_ema_teacher(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    decay: float,
) -> None:
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.mul_(decay).add_(student_param.detach(), alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)
    freeze_teacher(teacher)


def build_pseudo_label_targets(
    outputs: list[dict[str, torch.Tensor]],
    valid_masks: list[torch.Tensor],
    pseudo_config: Any,
    *,
    require_masks: bool = False,
    mask_probability_threshold: float = 0.5,
) -> tuple[list[int], list[dict[str, torch.Tensor]], dict[str, int]]:
    if len(outputs) != len(valid_masks):
        raise ValueError(
            "Pseudo-label filtering expects one valid mask per teacher output; "
            f"got {len(valid_masks)} masks for {len(outputs)} outputs."
        )
    if require_masks and not 0.0 <= float(mask_probability_threshold) <= 1.0:
        raise ValueError("mask_probability_threshold must be in [0, 1].")
    score_threshold = float(pseudo_config.score_threshold)

    stats = {
        "pseudo_num_labels_before_filter": 0,
        "pseudo_num_labels": 0,
        "pseudo_images_used": 0,
        "pseudo_images_skipped": 0,
    }
    selected_indices: list[int] = []
    targets: list[dict[str, torch.Tensor]] = []
    for image_index, (output, valid_mask) in enumerate(zip(outputs, valid_masks)):
        boxes = output.get("boxes")
        scores = output.get("scores")
        labels = output.get("labels")
        if not (torch.is_tensor(boxes) and torch.is_tensor(scores) and torch.is_tensor(labels)):
            stats["pseudo_images_skipped"] += 1
            continue
        boxes = boxes.detach()
        scores = scores.detach()
        labels = labels.detach().to(torch.int64)
        masks = (
            _prepare_pseudo_masks(
                output.get("masks"),
                expected_instances=int(boxes.shape[0]),
                image_size=tuple(valid_mask.shape),
                valid_mask=valid_mask,
                probability_threshold=float(mask_probability_threshold),
                device=boxes.device,
            )
            if require_masks
            else None
        )
        stats["pseudo_num_labels_before_filter"] += int(boxes.shape[0])
        if boxes.numel() == 0:
            boxes, labels = _empty_like(boxes, labels)
            scores = scores[:0]
            if masks is not None:
                masks = masks[:0]
        else:
            keep = scores >= score_threshold
            widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
            heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
            areas = widths * heights
            keep &= widths >= float(pseudo_config.min_box_size)
            keep &= heights >= float(pseudo_config.min_box_size)
            keep &= areas >= float(pseudo_config.min_box_area)
            if pseudo_config.valid_mask_min_fraction is not None:
                fractions = box_valid_fractions(valid_mask, boxes)
                keep &= fractions >= float(pseudo_config.valid_mask_min_fraction)
            if masks is not None:
                keep &= masks.flatten(start_dim=1).any(dim=1)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            if masks is not None:
                masks = masks[keep]

        if boxes.numel() > 0 and pseudo_config.nms_threshold is not None:
            from torchvision.ops import nms

            keep_indices = nms(boxes, scores, float(pseudo_config.nms_threshold))
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            if masks is not None:
                masks = masks[keep_indices]
        if boxes.numel() > 0:
            max_labels = int(pseudo_config.max_labels_per_image)
            order = torch.argsort(scores, descending=True)[:max_labels]
            boxes = boxes[order]
            labels = labels[order]
            if masks is not None:
                masks = masks[order]
        elif not bool(pseudo_config.include_empty_images):
            stats["pseudo_images_skipped"] += 1
            continue

        target = _target_from_boxes_and_labels(boxes, labels, masks=masks)
        selected_indices.append(image_index)
        targets.append(target)
        stats["pseudo_images_used"] += 1
        stats["pseudo_num_labels"] += int(target["boxes"].shape[0])
    return selected_indices, targets, stats


def _target_from_boxes_and_labels(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    *,
    masks: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    boxes = boxes.detach().to(torch.float32).reshape(-1, 4)
    labels = labels.detach().to(torch.int64).reshape(-1)
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError("Pseudo-label boxes and labels have inconsistent lengths.")
    if boxes.numel() == 0:
        area = boxes.new_zeros((0,))
    else:
        area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(
            0.0
        )
    target = {
        "boxes": boxes,
        "labels": labels,
        "area": area,
        "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64, device=boxes.device),
    }
    if masks is not None:
        masks = masks.detach().to(device=boxes.device, dtype=torch.uint8)
        if masks.ndim != 3 or masks.shape[0] != boxes.shape[0]:
            raise ValueError("Pseudo-label masks must have shape [N, H, W] and match the boxes.")
        target["masks"] = masks.contiguous()
    return target


def _prepare_pseudo_masks(
    masks: Any,
    *,
    expected_instances: int,
    image_size: tuple[int, ...],
    valid_mask: torch.Tensor,
    probability_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(masks):
        raise ValueError("Mask R-CNN Mean Teacher outputs must contain a masks tensor.")
    masks = masks.detach()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError("Mask R-CNN Mean Teacher masks must have shape [N, 1, H, W] or [N, H, W].")
    if int(masks.shape[0]) != expected_instances:
        raise ValueError("Mask R-CNN Mean Teacher outputs have inconsistent box and mask counts.")
    if len(image_size) != 2 or tuple(masks.shape[-2:]) != image_size:
        raise ValueError(
            "Mask R-CNN Mean Teacher masks must match the paired-view geometry; "
            f"got masks {tuple(masks.shape[-2:])} and valid mask {image_size}."
        )
    valid_mask = valid_mask.detach().to(device=masks.device, dtype=torch.bool)
    binary_masks = masks >= probability_threshold
    binary_masks &= valid_mask.unsqueeze(0)
    return binary_masks.to(device=device)


def _empty_like(
    boxes: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return boxes.reshape(0, 4), labels.reshape(0)
