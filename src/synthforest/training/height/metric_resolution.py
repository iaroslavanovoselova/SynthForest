from __future__ import annotations

import torch
from torch.nn import functional as F

from synthforest.training.config import HeightMetricResolutionConfig


def apply_height_metric_resolution(
    prediction: torch.Tensor,
    valid_mask: torch.Tensor,
    config: HeightMetricResolutionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Degrade predictions to the target CHM effective resolution for metrics."""
    factor = float(config.resolved_downsample_factor())
    if factor <= 1.0:
        return prediction, valid_mask.to(torch.bool)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            f"height prediction must have shape [B, 1, H, W], got {tuple(prediction.shape)}."
        )
    mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != prediction.shape:
        mask = mask.expand_as(prediction)

    height, width = prediction.shape[-2:]
    low_size = _scaled_size(height, width, factor)

    mask_float = mask.to(dtype=prediction.dtype)
    numerator = F.interpolate(
        prediction * mask_float,
        size=low_size,
        mode="area",
    )
    denominator = F.interpolate(mask_float, size=low_size, mode="area")
    low_valid = denominator >= float(config.min_valid_fraction)
    low_prediction = numerator / denominator.clamp_min(1e-6)
    low_prediction = torch.where(low_valid, low_prediction, torch.zeros_like(low_prediction))

    mode = str(config.upsample_interpolation).lower()
    if mode == "nearest":
        metric_prediction = F.interpolate(low_prediction, size=(height, width), mode=mode)
    else:
        metric_prediction = F.interpolate(
            low_prediction,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )
    metric_valid = F.interpolate(
        low_valid.to(dtype=prediction.dtype),
        size=(height, width),
        mode="nearest",
    ).to(torch.bool)

    metric_prediction = metric_prediction.contiguous()
    metric_mask = (metric_valid & mask).squeeze(1).contiguous()
    return metric_prediction, metric_mask


def _scaled_size(height: int, width: int, factor: float) -> tuple[int, int]:
    return (
        max(1, int(round(float(height) / float(factor)))),
        max(1, int(round(float(width) / float(factor)))),
    )
