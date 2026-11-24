from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from synthforest.training.config import ValidMaskConfig


def load_image_with_valid_mask(
    image_path: str | Path,
    config: ValidMaskConfig,
) -> tuple[Image.Image, np.ndarray]:
    """Load RGB pixels and resolve a boolean valid-data mask."""
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        alpha_mask = _alpha_valid_mask(source)
        image = source.convert("RGB").copy()

    if not config.enabled:
        return image, np.ones((image.height, image.width), dtype=bool)

    metadata_mask = _raster_valid_mask(image_path)
    if alpha_mask is not None:
        metadata_mask = alpha_mask if metadata_mask is None else metadata_mask & alpha_mask
    if metadata_mask is not None and metadata_mask.shape != (image.height, image.width):
        metadata_mask = (
            np.asarray(
                Image.fromarray(metadata_mask.astype(np.uint8) * 255).resize(
                    image.size,
                    Image.Resampling.NEAREST,
                )
            )
            > 0
        )

    if config.source == "metadata":
        valid_mask = metadata_mask
    elif config.source == "border_black":
        valid_mask = _border_black_valid_mask(
            image,
            tolerance=config.black_pixel_tolerance,
        )
    else:
        # An all-valid metadata mask carries no nodata information. In that case,
        # synthetic edge padding is recovered from border-connected black pixels.
        valid_mask = (
            metadata_mask
            if metadata_mask is not None and not bool(metadata_mask.all())
            else _border_black_valid_mask(
                image,
                tolerance=config.black_pixel_tolerance,
            )
        )

    if valid_mask is None:
        valid_mask = np.ones((image.height, image.width), dtype=bool)
    return image, np.ascontiguousarray(valid_mask, dtype=bool)


def image_and_mask_to_tensors(
    image: Image.Image,
    valid_mask: np.ndarray,
    image_mean: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(valid_mask)).to(torch.bool)
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if not bool(mask_tensor.all()):
        mean = torch.as_tensor(
            image_mean,
            dtype=image_tensor.dtype,
        ).view(-1, 1)
        image_tensor[:, ~mask_tensor] = mean
    return image_tensor, mask_tensor


def fill_invalid_pixels(
    image: Image.Image,
    valid_mask: np.ndarray,
    image_mean: list[float],
) -> Image.Image:
    if bool(valid_mask.all()):
        return image
    array = np.asarray(image.convert("RGB")).copy()
    fill = np.asarray(
        [round(float(value) * 255.0) for value in image_mean],
        dtype=np.uint8,
    )
    array[~valid_mask] = fill
    return Image.fromarray(array)


def valid_fraction(valid_mask: np.ndarray) -> float:
    if valid_mask.size == 0:
        return 0.0
    return float(valid_mask.mean())


def box_valid_fractions(
    valid_mask: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """Return the fraction of valid pixels covered by each XYXY box."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}.")
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        valid_mask = valid_mask[0]
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [H, W], got {tuple(valid_mask.shape)}.")

    valid_mask = valid_mask.to(device=boxes.device, dtype=torch.bool)
    if bool(valid_mask.all()):
        return boxes.new_ones((boxes.shape[0],))

    height, width = valid_mask.shape
    x1 = boxes[:, 0].floor().clamp(0, width).to(torch.int64)
    y1 = boxes[:, 1].floor().clamp(0, height).to(torch.int64)
    x2 = boxes[:, 2].ceil().clamp(0, width).to(torch.int64)
    y2 = boxes[:, 3].ceil().clamp(0, height).to(torch.int64)

    integral = torch.nn.functional.pad(
        valid_mask.to(torch.int64).cumsum(0).cumsum(1),
        (1, 0, 1, 0),
    )
    valid_area = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    box_area = (x2 - x1) * (y2 - y1)
    fractions = valid_area.to(boxes.dtype) / box_area.clamp_min(1).to(boxes.dtype)
    return torch.where(box_area > 0, fractions, torch.zeros_like(fractions))


def filter_detections_by_valid_masks(
    detections: list[dict[str, Any]],
    valid_masks: list[torch.Tensor],
    *,
    min_valid_fraction: Optional[float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove final detections that mostly cover invalid image support."""
    if len(detections) != len(valid_masks):
        raise ValueError(
            "Expected one valid mask per detection result, got "
            f"{len(valid_masks)} masks for {len(detections)} results."
        )
    if min_valid_fraction is None:
        checked = sum(
            int(detection["boxes"].shape[0])
            for detection in detections
            if torch.is_tensor(detection.get("boxes"))
        )
        return detections, {
            "num_predictions_before_mask_filter": checked,
            "num_predictions_filtered_invalid": 0,
        }

    checked = 0
    removed = 0
    filtered: list[dict[str, Any]] = []
    for detection, valid_mask in zip(detections, valid_masks):
        boxes = detection.get("boxes")
        if not torch.is_tensor(boxes):
            filtered.append(detection)
            continue
        fractions = box_valid_fractions(valid_mask, boxes)
        keep = fractions >= float(min_valid_fraction)
        checked += int(keep.numel())
        removed += int((~keep).sum())
        filtered.append(
            {
                key: (
                    value[keep]
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == keep.shape[0]
                    else value
                )
                for key, value in detection.items()
            }
        )
    return filtered, {
        "num_predictions_before_mask_filter": checked,
        "num_predictions_filtered_invalid": removed,
    }


def _alpha_valid_mask(image: Image.Image) -> Optional[np.ndarray]:
    if "A" not in image.getbands():
        return None
    alpha = np.asarray(image.getchannel("A"))
    return alpha > 0


def _raster_valid_mask(image_path: Path) -> Optional[np.ndarray]:
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            with rasterio.open(image_path) as dataset:
                return np.asarray(dataset.dataset_mask()) > 0
    except Exception:
        return None


def _border_black_valid_mask(
    image: Image.Image,
    *,
    tolerance: int,
) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    black = np.all(array <= int(tolerance), axis=-1)
    if not bool(black.any()):
        return np.ones(black.shape, dtype=bool)
    invalid = _border_connected_component(black)
    return ~invalid


def _border_connected_component(candidate: np.ndarray) -> np.ndarray:
    seed = np.zeros(candidate.shape, dtype=bool)
    seed[0, :] = candidate[0, :]
    seed[-1, :] = candidate[-1, :]
    seed[:, 0] = candidate[:, 0]
    seed[:, -1] = candidate[:, -1]
    if not bool(seed.any()):
        return seed

    try:
        from scipy.ndimage import binary_propagation

        return np.asarray(binary_propagation(seed, mask=candidate), dtype=bool)
    except ImportError:
        return _flood_fill_from_seed(candidate, seed)


def _flood_fill_from_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    height, width = candidate.shape
    connected = seed.copy()
    pending = deque(zip(*np.nonzero(seed)))
    while pending:
        row, col = pending.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and candidate[next_row, next_col]
                and not connected[next_row, next_col]
            ):
                connected[next_row, next_col] = True
                pending.append((next_row, next_col))
    return connected
