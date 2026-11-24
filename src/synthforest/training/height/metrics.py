from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return losses[mask].mean()


def masked_laplace_nll_loss(
    prediction: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    log_variance_min: float | None = None,
    log_variance_max: float | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if log_variance.shape != prediction.shape:
        raise ValueError(
            f"log_variance and prediction shapes differ: {log_variance.shape} vs {prediction.shape}"
        )
    mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
    if not bool(mask.any()):
        return (prediction.sum() + log_variance.sum()) * 0.0
    prediction_f = prediction.float()
    target_f = target.to(device=prediction.device, dtype=torch.float32)
    log_variance_f = log_variance.float()
    if log_variance_min is not None or log_variance_max is not None:
        log_variance_f = log_variance_f.clamp(
            min=log_variance_min,
            max=log_variance_max,
        )
    residual = (target_f - prediction_f).abs()
    losses = math.sqrt(2.0) * residual * torch.exp(-0.5 * log_variance_f) + 0.5 * log_variance_f
    return losses[mask].mean()


def masked_resolution_aware_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    resolution_factors: torch.Tensor | None = None,
    height_reducer: str = "p90_positive",
    min_canopy_fraction: float = 0.10,
) -> torch.Tensor:
    if resolution_factors is None:
        return masked_smooth_l1_loss(prediction, target, valid_mask)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            f"height prediction must have shape [B, 1, H, W], got {tuple(prediction.shape)}."
        )
    factors = (
        resolution_factors.detach().to(device=prediction.device, dtype=torch.float32).flatten()
    )
    if factors.numel() != prediction.shape[0]:
        raise ValueError(
            f"Expected {prediction.shape[0]} resolution factor(s), got {factors.numel()}."
        )
    losses = []
    for sample_index, factor_tensor in enumerate(factors):
        factor = float(factor_tensor.item())
        sample_prediction = prediction[sample_index : sample_index + 1]
        sample_target = target[sample_index : sample_index + 1]
        sample_mask = valid_mask[sample_index : sample_index + 1]
        if factor <= 1.0:
            sample_loss = _masked_smooth_l1_values(
                sample_prediction,
                sample_target,
                sample_mask,
            )
        elif _is_integer_factor(factor):
            reduced_prediction, reduced_target, reduced_mask = _aggregate_height_loss_inputs(
                sample_prediction,
                sample_target,
                sample_mask,
                factor=int(round(factor)),
                height_reducer=height_reducer,
                min_canopy_fraction=min_canopy_fraction,
            )
            sample_loss = _masked_smooth_l1_values(
                reduced_prediction,
                reduced_target,
                reduced_mask,
            )
        else:
            reduced_prediction, reduced_target, reduced_mask = (
                _aggregate_fractional_height_loss_inputs(
                    sample_prediction,
                    sample_target,
                    sample_mask,
                    factor=factor,
                    min_canopy_fraction=min_canopy_fraction,
                )
            )
            sample_loss = _masked_smooth_l1_values(
                reduced_prediction,
                reduced_target,
                reduced_mask,
            )
        if sample_loss.numel() > 0:
            losses.append(sample_loss)
    if not losses:
        return prediction.sum() * 0.0
    return torch.cat(losses).mean()


def _aggregate_fractional_height_loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    factor: float,
    min_canopy_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if factor <= 1.0:
        return prediction, target, valid_mask
    mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
    low_size = _scaled_size(prediction.shape[-2], prediction.shape[-1], factor)
    mask_float = mask.to(dtype=prediction.dtype)
    low_valid_fraction = F.interpolate(mask_float, size=low_size, mode="area")
    reduced_prediction = F.interpolate(
        prediction * mask_float,
        size=low_size,
        mode="area",
    ) / low_valid_fraction.clamp_min(1e-6)

    # Fractional cells do not align with exact source-pixel blocks, so use
    # masked positive-area target resampling here. Integer factors keep the
    # configured pXX/max/mean reducer path above.
    positive = mask & torch.isfinite(target) & (target > 0.0)
    positive_float = positive.to(dtype=prediction.dtype)
    low_positive_fraction = F.interpolate(positive_float, size=low_size, mode="area")
    reduced_target = F.interpolate(
        torch.where(positive, target, torch.zeros_like(target)),
        size=low_size,
        mode="area",
    ) / low_positive_fraction.clamp_min(1e-6)
    reduced_mask = (low_valid_fraction > 0.0) & (
        low_positive_fraction >= float(min_canopy_fraction)
    )
    reduced_prediction = torch.where(
        reduced_mask,
        reduced_prediction,
        torch.zeros_like(reduced_prediction),
    )
    reduced_target = torch.where(
        reduced_mask,
        reduced_target,
        torch.zeros_like(reduced_target),
    )
    return reduced_prediction, reduced_target.to(dtype=prediction.dtype), reduced_mask.squeeze(1)


def _is_integer_factor(value: float, *, tolerance: float = 1e-6) -> bool:
    rounded = round(float(value))
    return rounded >= 1 and abs(float(value) - rounded) <= tolerance


def _scaled_size(height: int, width: int, factor: float) -> tuple[int, int]:
    return (
        max(1, int(round(float(height) / float(factor)))),
        max(1, int(round(float(width) / float(factor)))),
    )


def _masked_smooth_l1_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
    if not bool(mask.any()):
        return prediction.new_zeros((0,))
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return losses[mask]


def _aggregate_height_loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    factor: int,
    height_reducer: str,
    min_canopy_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if factor <= 1:
        return prediction, target, valid_mask
    mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
    prediction, target, mask = _pad_to_factor(prediction, target, mask, factor=factor)
    pred_blocks = _block_view(prediction, factor)
    target_blocks = _block_view(target, factor).to(torch.float32)
    mask_blocks = _block_view(mask.to(prediction.dtype), factor).to(torch.bool)
    block_area = factor * factor
    valid_count = mask_blocks.sum(dim=-1)
    prediction_sum = torch.where(mask_blocks, pred_blocks, torch.zeros_like(pred_blocks)).sum(
        dim=-1
    )
    reduced_prediction = prediction_sum / valid_count.clamp_min(1).to(prediction_sum.dtype)

    positive = mask_blocks & torch.isfinite(target_blocks) & (target_blocks > 0.0)
    positive_count = positive.sum(dim=-1)
    canopy_fraction = positive_count.to(torch.float32) / float(block_area)
    reduced_mask = (valid_count > 0) & (canopy_fraction >= float(min_canopy_fraction))
    reduced_target = _reduce_positive_height_blocks(
        target_blocks,
        positive,
        reducer=height_reducer,
    ).to(dtype=prediction.dtype)
    reduced_target = torch.where(
        reduced_mask,
        reduced_target,
        torch.zeros_like(reduced_target),
    )
    return reduced_prediction, reduced_target, reduced_mask.squeeze(1)


def _pad_to_factor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = prediction.shape[-2:]
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    if pad_height == 0 and pad_width == 0:
        return prediction, target, mask
    pad = (0, pad_width, 0, pad_height)
    return (
        F.pad(prediction, pad, value=0.0),
        F.pad(target, pad, value=0.0),
        F.pad(mask, pad, value=False),
    )


def _block_view(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    batch, channels, height, width = tensor.shape
    return (
        tensor.reshape(batch, channels, height // factor, factor, width // factor, factor)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, channels, height // factor, width // factor, factor * factor)
    )


def _reduce_positive_height_blocks(
    values: torch.Tensor,
    positive: torch.Tensor,
    *,
    reducer: str,
) -> torch.Tensor:
    masked_values = torch.where(
        positive,
        values,
        torch.full_like(values, float("nan")),
    )
    if reducer == "mean_positive":
        summed = torch.where(positive, values, torch.zeros_like(values)).sum(dim=-1)
        counts = positive.sum(dim=-1).clamp_min(1).to(summed.dtype)
        return summed / counts
    if reducer == "max_positive":
        reduced = torch.nan_to_num(masked_values, nan=float("-inf")).amax(dim=-1)
        return torch.where(torch.isfinite(reduced), reduced, torch.zeros_like(reduced))
    percentile = _percentile_from_positive_reducer(reducer)
    if percentile is None:
        raise ValueError(f"Unsupported height reducer for resolution degradation: {reducer}")
    reduced = torch.nanquantile(masked_values, percentile / 100.0, dim=-1)
    return torch.nan_to_num(reduced, nan=0.0)


def _percentile_from_positive_reducer(reducer: str) -> float | None:
    if not reducer.startswith("p") or not reducer.endswith("_positive"):
        return None
    try:
        return float(reducer[1 : reducer.index("_positive")])
    except ValueError:
        return None


class HeightMetricsAccumulator:
    def __init__(self, thresholds_m: list[float]) -> None:
        self.thresholds_m = [float(value) for value in thresholds_m]
        self.reset()

    def reset(self) -> None:
        self.total_pixels = 0
        self.valid_pixels = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.threshold_stats = {
            _threshold_tag(threshold): {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0}
            for threshold in self.thresholds_m
        }

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
            )
        mask = _expand_mask(valid_mask, prediction).to(device=prediction.device)
        self.total_pixels += int(mask.numel())
        self.valid_pixels += int(mask.sum().detach().cpu())
        if not bool(mask.any()):
            return
        pred = prediction.detach().clamp_min(0.0)[mask].to(torch.float64)
        truth = target.detach()[mask].to(torch.float64)
        error = pred - truth
        self.sum_abs += float(error.abs().sum().cpu())
        self.sum_sq += float((error * error).sum().cpu())
        self.sum_err += float(error.sum().cpu())
        for threshold in self.thresholds_m:
            threshold_mask = truth > float(threshold)
            if not bool(threshold_mask.any()):
                continue
            tag = _threshold_tag(threshold)
            threshold_error = error[threshold_mask]
            stats = self.threshold_stats[tag]
            stats["count"] += int(threshold_mask.sum().cpu())
            stats["sum_abs"] += float(threshold_error.abs().sum().cpu())
            stats["sum_sq"] += float((threshold_error * threshold_error).sum().cpu())

    def compute(self) -> dict[str, float]:
        count = max(0, int(self.valid_pixels))
        metrics = {
            "num_pixels": float(self.total_pixels),
            "valid_pixels": float(self.valid_pixels),
            "valid_fraction": (
                float(self.valid_pixels / self.total_pixels) if self.total_pixels else 0.0
            ),
        }
        if count == 0:
            metrics.update({"mae": -1.0, "mse": -1.0, "rmse": -1.0, "bias": 0.0})
        else:
            mse = self.sum_sq / count
            metrics.update(
                {
                    "mae": self.sum_abs / count,
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "bias": self.sum_err / count,
                }
            )
        for tag, stats in self.threshold_stats.items():
            threshold_count = int(stats["count"])
            metrics[f"valid_pixels_gt_{tag}"] = float(threshold_count)
            if threshold_count <= 0:
                metrics[f"mae_gt_{tag}"] = -1.0
                metrics[f"rmse_gt_{tag}"] = -1.0
                continue
            threshold_mse = stats["sum_sq"] / threshold_count
            metrics[f"mae_gt_{tag}"] = stats["sum_abs"] / threshold_count
            metrics[f"rmse_gt_{tag}"] = math.sqrt(threshold_mse)
        return metrics


def macro_average_height_metrics(
    metrics_by_dataset: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not metrics_by_dataset:
        raise ValueError("Real height validation requires at least one evaluated dataset.")
    count_keys = {"num_pixels", "valid_pixels"} | {
        key
        for metrics in metrics_by_dataset.values()
        for key in metrics
        if key.startswith("valid_pixels_gt_")
    }
    metric_keys = sorted(set().union(*(metrics.keys() for metrics in metrics_by_dataset.values())))
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        values = [
            float(metrics[key])
            for metrics in metrics_by_dataset.values()
            if key in metrics
            and isinstance(metrics[key], (int, float))
            and math.isfinite(float(metrics[key]))
            and _is_aggregate_height_metric_value(key, float(metrics[key]), count_keys)
        ]
        if not values:
            continue
        aggregate[key] = (
            float(sum(values)) if key in count_keys else float(sum(values) / len(values))
        )
    if aggregate.get("num_pixels", 0.0) > 0:
        aggregate["valid_fraction"] = aggregate.get("valid_pixels", 0.0) / aggregate["num_pixels"]
    return aggregate


def _is_aggregate_height_metric_value(
    key: str,
    value: float,
    count_keys: set[str],
) -> bool:
    if key in count_keys:
        return True
    if key == "bias" or key.endswith("_bias"):
        return True
    return value >= 0.0


def _expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.bool)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = mask.expand_as(reference)
    return mask


def _threshold_tag(value: float) -> str:
    text = f"{float(value):.3g}".replace(".", "p")
    return f"{text}m"
