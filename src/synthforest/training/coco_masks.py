from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class CocoMaskCoverage:
    status: str
    total_annotations: int
    valid_segmentations: int
    missing_segmentations: int
    invalid_segmentations: int
    decode_errors: int

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            **asdict(self),
            "is_complete": self.is_complete,
        }


def load_coco_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO payload must be an object: {path}")
    return payload


def inspect_coco_mask_coverage(
    coco: Mapping[str, Any] | str | Path,
    *,
    image_ids: Optional[Iterable[int]] = None,
    validate_decoding: bool = False,
) -> CocoMaskCoverage:
    payload = load_coco_payload(coco) if isinstance(coco, (str, Path)) else dict(coco)
    coverage, _ = _inspect_coco_masks(
        payload,
        image_ids=image_ids,
        validate_decoding=validate_decoding,
    )
    return coverage


def filter_coco_annotations_with_valid_masks(
    coco: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], CocoMaskCoverage]:
    """Return annotations whose instance masks decode to non-empty masks."""
    payload = dict(coco)
    coverage, valid_annotation_indexes = _inspect_coco_masks(
        payload,
        image_ids=None,
        validate_decoding=True,
    )
    annotations = list(payload.get("annotations", []))
    return (
        [annotations[index] for index in valid_annotation_indexes],
        coverage,
    )


def _inspect_coco_masks(
    payload: Mapping[str, Any],
    *,
    image_ids: Optional[Iterable[int]],
    validate_decoding: bool,
) -> tuple[CocoMaskCoverage, tuple[int, ...]]:
    selected_ids = {int(image_id) for image_id in image_ids} if image_ids is not None else None
    images = {int(image["id"]): image for image in payload.get("images", [])}

    total = 0
    valid = 0
    missing = 0
    invalid = 0
    decode_errors = 0
    valid_annotation_indexes: list[int] = []
    for annotation_index, annotation in enumerate(payload.get("annotations", [])):
        image_id = int(annotation["image_id"])
        if selected_ids is not None and image_id not in selected_ids:
            continue
        total += 1
        segmentation = annotation.get("segmentation")
        if segmentation is None or segmentation == [] or segmentation == {}:
            missing += 1
            continue
        image_info = images.get(image_id)
        if image_info is None or not _is_structurally_valid_segmentation(
            segmentation,
            image_height=int(image_info.get("height", 0)),
            image_width=int(image_info.get("width", 0)),
        ):
            invalid += 1
            continue
        if validate_decoding:
            try:
                _validate_coco_segmentation_encoding(
                    segmentation,
                    image_height=int(image_info["height"]),
                    image_width=int(image_info["width"]),
                )
            except (TypeError, ValueError, RuntimeError):
                decode_errors += 1
                continue
        valid += 1
        valid_annotation_indexes.append(annotation_index)

    if invalid or decode_errors or (0 < valid < total):
        status = "partial"
    elif missing == total and total > 0:
        status = "none"
    elif valid == total:
        # Empty COCO files are mask-compatible: there are no positive targets
        # that require segmentation supervision.
        status = "complete"
    else:
        status = "partial"
    return (
        CocoMaskCoverage(
            status=status,
            total_annotations=total,
            valid_segmentations=valid,
            missing_segmentations=missing,
            invalid_segmentations=invalid,
            decode_errors=decode_errors,
        ),
        tuple(valid_annotation_indexes),
    )


def decode_coco_segmentation(
    segmentation: Any,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "Mask R-CNN training and segmentation evaluation require pycocotools."
        ) from exc

    height = int(image_height)
    width = int(image_width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask dimensions: height={height}, width={width}.")
    rle = _coco_segmentation_to_rle(
        segmentation,
        image_height=height,
        image_width=width,
        mask_utils=mask_utils,
    )

    decoded = np.asarray(mask_utils.decode(rle))
    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)
    if decoded.shape != (height, width):
        raise ValueError(f"Decoded mask has shape {decoded.shape}, expected {(height, width)}.")
    return np.ascontiguousarray(decoded > 0, dtype=np.uint8)


def decode_coco_segmentation_crop(
    segmentation: Any,
    *,
    image_height: int,
    image_width: int,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Decode one COCO segmentation only inside an XYXY crop box.

    Polygon segmentations are shifted and rasterized directly at crop size.
    RLE segmentations are decoded one instance at a time and sliced, which keeps
    peak memory bounded by one full-image mask rather than an N-instance stack.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "Mask R-CNN training and segmentation evaluation require pycocotools."
        ) from exc

    height = int(image_height)
    width = int(image_width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask dimensions: height={height}, width={width}.")
    left, top, right, bottom = [int(value) for value in crop_box]
    left = max(0, min(width, left))
    right = max(left, min(width, right))
    top = max(0, min(height, top))
    bottom = max(top, min(height, bottom))
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Invalid crop box {crop_box!r} for image {(height, width)}.")

    if isinstance(segmentation, list):
        shifted = _shift_polygon_segmentation(
            segmentation,
            x_offset=float(left),
            y_offset=float(top),
        )
        rle = mask_utils.merge(mask_utils.frPyObjects(shifted, crop_height, crop_width))
        decoded = np.asarray(mask_utils.decode(rle))
    else:
        decoded = decode_coco_segmentation(
            segmentation,
            image_height=height,
            image_width=width,
        )[top:bottom, left:right]

    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)
    if decoded.shape != (crop_height, crop_width):
        raise ValueError(
            f"Decoded crop has shape {decoded.shape}, expected {(crop_height, crop_width)}."
        )
    return np.ascontiguousarray(decoded > 0, dtype=np.uint8)


def _shift_polygon_segmentation(
    segmentation: list[Any],
    *,
    x_offset: float,
    y_offset: float,
) -> list[list[float]]:
    shifted: list[list[float]] = []
    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            raise ValueError("COCO polygon segmentation is structurally invalid.")
        coords = [float(value) for value in polygon]
        shifted.append(
            [
                coord - (x_offset if index % 2 == 0 else y_offset)
                for index, coord in enumerate(coords)
            ]
        )
    if not shifted:
        raise ValueError("COCO polygon segmentation must not be empty.")
    return shifted


def _validate_coco_segmentation_encoding(
    segmentation: Any,
    *,
    image_height: int,
    image_width: int,
) -> None:
    """Validate through pycocotools without allocating a dense image mask."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "Mask R-CNN training and segmentation evaluation require pycocotools."
        ) from exc

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            runs = [_validated_rle_run(value) for value in counts]
        elif isinstance(counts, (str, bytes)):
            runs = _decode_compressed_rle_counts(counts)
        else:
            runs = []
        if runs and sum(runs) != int(image_height) * int(image_width):
            raise ValueError("COCO RLE runs do not cover exactly the declared image size.")

    rle = _coco_segmentation_to_rle(
        segmentation,
        image_height=int(image_height),
        image_width=int(image_width),
        mask_utils=mask_utils,
    )
    area = float(np.asarray(mask_utils.area(rle)).sum())
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("COCO segmentation decodes to an empty or invalid mask.")


def _validated_rle_run(value: Any) -> int:
    try:
        run = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("COCO RLE runs must be integers.") from exc
    if run < 0 or float(value) != float(run):
        raise ValueError("COCO RLE runs must be non-negative integers.")
    return run


def _decode_compressed_rle_counts(counts: str | bytes) -> list[int]:
    """Decode COCO's compact run lengths without expanding the image mask."""
    try:
        encoded = counts.encode("ascii") if isinstance(counts, str) else counts
    except UnicodeEncodeError as exc:
        raise ValueError("Compressed COCO RLE must contain ASCII bytes.") from exc
    runs: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift_groups = 0
        while True:
            if position >= len(encoded):
                raise ValueError("Compressed COCO RLE ends inside a run.")
            code = int(encoded[position]) - 48
            position += 1
            if not 0 <= code <= 0x3F:
                raise ValueError("Compressed COCO RLE contains an invalid byte.")
            value |= (code & 0x1F) << (5 * shift_groups)
            has_more = bool(code & 0x20)
            shift_groups += 1
            if not has_more:
                if code & 0x10:
                    value |= -1 << (5 * shift_groups)
                break
            if shift_groups > 12:
                raise ValueError("Compressed COCO RLE run is unreasonably long.")
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("Compressed COCO RLE produced a negative run.")
        runs.append(value)
    if not runs:
        raise ValueError("Compressed COCO RLE contains no runs.")
    return runs


def _coco_segmentation_to_rle(
    segmentation: Any,
    *,
    image_height: int,
    image_width: int,
    mask_utils,
):
    height = int(image_height)
    width = int(image_width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask dimensions: height={height}, width={width}.")
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError("COCO polygon segmentation must not be empty.")
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if not isinstance(segmentation, dict):
        raise TypeError("COCO segmentation must be a polygon list or RLE dictionary.")

    rle = dict(segmentation)
    size = rle.get("size")
    if size is not None and [int(value) for value in size] != [height, width]:
        raise ValueError(
            f"COCO RLE size does not match its image: rle={size}, image={[height, width]}."
        )
    rle["size"] = [height, width]
    counts = rle.get("counts")
    if isinstance(counts, list):
        return mask_utils.frPyObjects(rle, height, width)
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    elif not isinstance(counts, bytes):
        raise ValueError("COCO RLE counts must be a list, string, or bytes.")
    return rle


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("Encoding instance-mask predictions requires pycocotools.") from exc

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Binary mask must have shape [H, W], got {array.shape}.")
    rle = mask_utils.encode(np.asfortranarray(array.astype(np.uint8, copy=False)))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    rle["size"] = [int(value) for value in rle["size"]]
    return rle


def segmentation_area(segmentation: Mapping[str, Any]) -> float:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("Computing instance-mask areas requires pycocotools.") from exc

    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return float(mask_utils.area(rle))


def _is_structurally_valid_segmentation(
    segmentation: Any,
    *,
    image_height: int,
    image_width: int,
) -> bool:
    if image_height <= 0 or image_width <= 0:
        return False
    if isinstance(segmentation, list):
        return bool(segmentation) and all(
            isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
            for polygon in segmentation
        )
    if not isinstance(segmentation, dict):
        return False
    counts = segmentation.get("counts")
    if not isinstance(counts, (list, str, bytes)) or len(counts) == 0:
        return False
    size = segmentation.get("size")
    if size is None:
        return True
    try:
        parsed_size = [int(value) for value in size]
    except (TypeError, ValueError):
        return False
    return parsed_size == [int(image_height), int(image_width)]
