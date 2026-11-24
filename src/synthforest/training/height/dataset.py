from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn import functional as F
from torch.utils.data import Dataset

from synthforest.training.config import HeightTrainingConfig
from synthforest.training.dataset import (
    SUPPORTED_IMAGE_SUFFIXES,
    _adapt_pixel_distribution,
    _add_gaussian_noise,
    _apply_jpeg_compression,
    _build_domain_adaptation_transform,
    _collect_reference_images,
    _downsample_then_upsample,
    _load_reference_image,
    _sample_float_range,
    _sample_int_range,
    _shift_hue,
)
from synthforest.training.valid_masks import (
    fill_invalid_pixels,
    load_image_with_valid_mask,
    valid_fraction,
)

HEIGHT_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class HeightTileRecord:
    dataset_name: str
    product_name: str
    image_path: Path
    height_path: Path
    tile_key: str


class HeightTileDataset(Dataset):
    """Read paired RGB and canopy-height tiles exported by convert_scenes_to_tiles."""

    def __init__(
        self,
        data_root_path: str | Path,
        dataset_names: Iterable[str],
        fold: str,
        config: HeightTrainingConfig,
        train: bool,
        max_products_per_dataset: Optional[int] = None,
        max_tiles_per_product: Optional[int] = None,
    ) -> None:
        self.data_root_path = Path(data_root_path)
        self.dataset_names = list(dataset_names)
        self.fold = str(fold)
        self.config = config
        self.train = train
        self.max_products_per_dataset = max_products_per_dataset
        self.max_tiles_per_product = max_tiles_per_product
        self.augmentor = HeightAugmentor(config, train=train)
        self._warned_invalid_record = False
        self.records = self._load_records()
        if not self.records:
            raise ValueError(
                f"No paired height tile records found under {self.data_root_path} "
                f"for fold={self.fold!r}, datasets={self.dataset_names}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        if not self.records:
            raise IndexError("Cannot load an item from an empty height dataset.")
        index = int(index) % len(self.records)
        max_attempts = min(len(self.records), 128)
        minimum_valid_fraction = (
            float(self.config.valid_mask.target_min_valid_fraction)
            if self.config.valid_mask.enabled
            else 0.0
        )
        rejected: list[tuple[HeightTileRecord, float]] = []
        for offset in range(max_attempts):
            record = self.records[(index + offset) % len(self.records)]
            image, height, valid_mask = self._load_record(record)
            fraction = valid_fraction(valid_mask)
            if fraction > 0.0 and fraction >= minimum_valid_fraction:
                break
            rejected.append((record, fraction))
        else:
            examples = ", ".join(
                f"{item.height_path} ({fraction:.3f})" for item, fraction in rejected[:3]
            )
            raise ValueError(
                "Could not find a usable height tile within "
                f"{max_attempts} consecutive records. Required valid fraction is "
                f"{minimum_valid_fraction:g}; examples: "
                f"{examples}"
            )

        if rejected and not self._warned_invalid_record:
            first_record, first_fraction = rejected[0]
            warnings.warn(
                "Skipping height tile(s) below the required valid fraction "
                f"of {minimum_valid_fraction:g}. "
                f"First skipped CHM: {first_record.height_path} "
                f"(usable fraction={first_fraction:.3f}).",
                stacklevel=2,
            )
            self._warned_invalid_record = True

        image, height, valid_mask, resolution_degradation = self.augmentor(
            image,
            height,
            valid_mask,
        )
        image_tensor = image_to_normalized_tensor(
            image,
            valid_mask,
            self.config.augmentation.image_mean,
            self.config.augmentation.image_std,
        )
        height_tensor = torch.from_numpy(
            np.array(height, dtype=np.float32, copy=True, order="C")
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(valid_mask, dtype=bool, copy=True, order="C"))
        return {
            "image": image_tensor,
            "height": height_tensor,
            "valid_mask": mask_tensor,
            "metadata": {
                "dataset_name": record.dataset_name,
                "product_name": record.product_name,
                "image_path": str(record.image_path),
                "height_path": str(record.height_path),
                "tile_key": record.tile_key,
                "resolution_degradation": resolution_degradation,
            },
            "resolution_degradation": resolution_degradation,
        }

    def _load_record(
        self,
        record: HeightTileRecord,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        image, rgb_valid_mask = load_image_with_valid_mask(
            record.image_path,
            self.config.valid_mask,
        )
        height, height_valid_mask = load_height_map(
            record.height_path,
            min_valid_height_m=self.config.min_valid_height_m,
            max_valid_height_m=self.config.max_valid_height_m,
        )
        if height.shape != (image.height, image.width):
            height, height_valid_mask = _resize_height_and_mask(
                height,
                height_valid_mask,
                size=image.size,
            )
        valid_mask = np.ascontiguousarray(
            rgb_valid_mask & height_valid_mask,
            dtype=bool,
        )
        return image, height, valid_mask

    def _load_records(self) -> list[HeightTileRecord]:
        records: list[HeightTileRecord] = []
        for dataset_name, dataset_root in resolve_height_dataset_roots(
            self.data_root_path,
            self.dataset_names,
        ):
            product_roots = _find_product_roots(dataset_root, self.fold)
            if self.max_products_per_dataset is not None:
                product_roots = product_roots[: self.max_products_per_dataset]
            for product_root in product_roots:
                product_records = _pair_product_tiles(
                    dataset_name=dataset_name,
                    product_root=product_root,
                    fold=self.fold,
                )
                if self.max_tiles_per_product is not None:
                    product_records = product_records[: self.max_tiles_per_product]
                records.extend(product_records)
        return records


class HeightAugmentor:
    def __init__(self, config: HeightTrainingConfig, train: bool) -> None:
        self.config = config
        self.train = train
        self.reference_paths = _collect_reference_images(config) if train else []
        self._reference_is_full: dict[Path, bool] = {}
        self._warned_domain_adaptation = False

    def __call__(
        self,
        image: Image.Image,
        height: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray, dict[str, float | bool]]:
        resolution_degradation = _default_resolution_degradation(
            self.config.resolution_degradation.base_gsd_m
        )
        if self.train:
            image = self._apply_domain_adaptation(image, valid_mask)
            if self.config.valid_mask.enabled:
                image = fill_invalid_pixels(
                    image,
                    valid_mask,
                    self.config.augmentation.image_mean,
                )
            image = self._apply_color(image)
            image, height, valid_mask = self._apply_crop(image, height, valid_mask)
            image, height, valid_mask = self._apply_flips(image, height, valid_mask)
            image, height, valid_mask = self._apply_rotation(image, height, valid_mask)
            if self.config.valid_mask.enabled:
                image = fill_invalid_pixels(
                    image,
                    valid_mask,
                    self.config.augmentation.image_mean,
                )
            image = self._apply_image_degradations(image)
            image, resolution_degradation = self._apply_resolution_degradation(image)

        if not self.config.preserve_input_resolution:
            image, height, valid_mask = self._apply_resize(image, height, valid_mask)
        if self.config.valid_mask.enabled:
            image = fill_invalid_pixels(
                image,
                valid_mask,
                self.config.augmentation.image_mean,
            )
        if self.train and self.config.valid_mask.enabled and not bool(valid_mask.any()):
            raise ValueError(
                "Training augmentation produced an entirely invalid height target. "
                "Increase valid_mask.crop_min_valid_fraction or remove the tile."
            )
        return (
            image,
            np.ascontiguousarray(height, dtype=np.float32),
            np.ascontiguousarray(valid_mask, dtype=bool),
            resolution_degradation,
        )

    def _apply_color(self, image: Image.Image) -> Image.Image:
        aug = self.config.augmentation
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
        height: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        aug = self.config.augmentation
        if random.random() >= aug.crop_prob:
            return image, height, valid_mask
        width, image_height = image.size
        max_possible = min(width, image_height, max(aug.crop_size_range))
        min_possible = min(max_possible, min(aug.crop_size_range))
        if min_possible <= 1 or max_possible <= 1:
            return image, height, valid_mask
        crop: tuple[int, int, int] | None = None
        for _ in range(10):
            side = random.randint(int(min_possible), int(max_possible))
            left = random.randint(0, max(0, width - side))
            top = random.randint(0, max(0, image_height - side))
            candidate_mask = valid_mask[top : top + side, left : left + side]
            if (
                not self.config.valid_mask.enabled
                or valid_fraction(candidate_mask) >= self.config.valid_mask.crop_min_valid_fraction
            ):
                crop = (side, left, top)
                break
        if crop is None:
            return image, height, valid_mask
        side, left, top = crop
        return (
            image.crop((left, top, left + side, top + side)),
            height[top : top + side, left : left + side].copy(),
            valid_mask[top : top + side, left : left + side].copy(),
        )

    def _apply_flips(
        self,
        image: Image.Image,
        height: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        aug = self.config.augmentation
        if aug.flip_horizontal and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            height = np.fliplr(height).copy()
            valid_mask = np.fliplr(valid_mask).copy()
        if aug.flip_vertical and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            height = np.flipud(height).copy()
            valid_mask = np.flipud(valid_mask).copy()
        return image, height, valid_mask

    def _apply_rotation(
        self,
        image: Image.Image,
        height: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        aug = self.config.augmentation
        if aug.rotation_degrees <= 0 or random.random() >= aug.rotation_prob:
            return image, height, valid_mask
        angle = random.uniform(-aug.rotation_degrees, aug.rotation_degrees)
        fill = tuple(round(float(value) * 255.0) for value in self.config.augmentation.image_mean)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill if self.config.valid_mask.enabled else None,
        )
        height = np.asarray(
            _float_image_from_array(height).rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0.0,
            ),
            dtype=np.float32,
        )
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
        return image, height, valid_mask

    def _apply_resize(
        self,
        image: Image.Image,
        height: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        aug = self.config.augmentation
        min_size = (
            random.choice(list(aug.resize_min_sizes_train))
            if self.train
            else int(aug.resize_min_size_eval)
        )
        width, image_height = image.size
        min_original = min(width, image_height)
        max_original = max(width, image_height)
        if min_original <= 0:
            return image, height, valid_mask
        scale = float(min_size) / float(min_original)
        if max_original * scale > float(aug.max_size):
            scale = float(aug.max_size) / float(max_original)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(image_height * scale)))
        if (new_width, new_height) == image.size:
            return image, height, valid_mask
        return (
            image.resize((new_width, new_height), Image.Resampling.BILINEAR),
            np.asarray(
                _float_image_from_array(height).resize(
                    (new_width, new_height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            ),
            np.asarray(
                Image.fromarray(valid_mask.astype(np.uint8) * 255).resize(
                    (new_width, new_height),
                    Image.Resampling.NEAREST,
                )
            )
            > 0,
        )

    def _apply_image_degradations(self, image: Image.Image) -> Image.Image:
        aug = self.config.augmentation
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

    def _apply_resolution_degradation(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, dict[str, float | bool]]:
        cfg = self.config.resolution_degradation
        metadata = _default_resolution_degradation(cfg.base_gsd_m)
        if not cfg.enabled or random.random() >= float(cfg.prob):
            return image, metadata
        target_gsd = float(random.choice(list(cfg.target_gsd_m_values)))
        factor = target_gsd / float(cfg.base_gsd_m)
        metadata = {
            "enabled": True,
            "applied": factor > 1.0,
            "factor": float(factor),
            "base_gsd_m": float(cfg.base_gsd_m),
            "target_gsd_m": target_gsd,
        }
        if factor <= 1.0:
            return image, metadata

        blur_min, blur_max = float(cfg.blur_sigma[0]), float(cfg.blur_sigma[1])
        blur_sigma = random.uniform(blur_min, blur_max)
        if blur_sigma > 0.0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        width, height = image.size
        low_size = (
            max(1, int(round(width / factor))),
            max(1, int(round(height / factor))),
        )
        downsampled = image.resize(
            low_size,
            _pil_resampling(cfg.downsample_interpolation),
        )
        return (
            downsampled.resize(
                image.size,
                _pil_resampling(cfg.upsample_interpolation),
            ),
            metadata,
        )

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
                _, reference_mask = load_image_with_valid_mask(path, self.config.valid_mask)
                is_full = (
                    valid_fraction(reference_mask)
                    >= self.config.valid_mask.da_full_tile_min_valid_fraction
                )
                self._reference_is_full[path] = is_full
            if is_full:
                return path
        return None


def resolve_height_dataset_roots(
    data_root_path: str | Path,
    dataset_names: Iterable[str],
) -> list[tuple[str, Path]]:
    root = Path(data_root_path)
    resolved = []
    for dataset_name in dataset_names:
        dataset_name = str(dataset_name)
        candidate = root / dataset_name
        if candidate.exists():
            resolved.append((dataset_name, candidate))
        elif root.name == dataset_name and root.exists():
            resolved.append((dataset_name, root))
        else:
            raise FileNotFoundError(
                f"Height dataset {dataset_name!r} was not found at {candidate} "
                f"and data_root_path itself is {root.name!r}."
            )
    return resolved


def load_height_map(
    path: str | Path,
    *,
    min_valid_height_m: float,
    max_valid_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
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
    valid_mask = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(height)
        & (height >= float(min_valid_height_m))
        & (height <= float(max_valid_height_m))
    )
    height = np.where(valid_mask, height, 0.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(height), np.ascontiguousarray(valid_mask, dtype=bool)


def image_to_normalized_tensor(
    image: Image.Image,
    valid_mask: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(valid_mask, dtype=bool)
    if not bool(mask.all()):
        fill = np.asarray(image_mean, dtype=np.float32)
        array[~mask] = fill
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.as_tensor(image_mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.as_tensor(image_std, dtype=tensor.dtype).view(-1, 1, 1).clamp_min(1e-6)
    return (tensor - mean) / std


def collate_height_batch(
    batch: list[dict],
    *,
    patch_size_multiple: int,
) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty height batch.")
    max_height = max(item["image"].shape[-2] for item in batch)
    max_width = max(item["image"].shape[-1] for item in batch)
    multiple = max(1, int(patch_size_multiple))
    max_height = int(np.ceil(max_height / multiple) * multiple)
    max_width = int(np.ceil(max_width / multiple) * multiple)
    images = []
    heights = []
    valid_masks = []
    metadata = []
    resolution_factors = []
    target_gsd_m = []
    resolution_degradation_applied = []
    for item in batch:
        image = item["image"]
        height = item["height"]
        valid_mask = item["valid_mask"]
        degradation = item.get("resolution_degradation", {})
        pad_height = max_height - image.shape[-2]
        pad_width = max_width - image.shape[-1]
        images.append(F.pad(image, (0, pad_width, 0, pad_height), value=0.0))
        heights.append(F.pad(height, (0, pad_width, 0, pad_height), value=0.0))
        valid_masks.append(F.pad(valid_mask, (0, pad_width, 0, pad_height), value=False))
        metadata.append(item["metadata"])
        resolution_factors.append(float(degradation.get("factor", 1.0)))
        target_gsd_m.append(float(degradation.get("target_gsd_m", 0.0)))
        resolution_degradation_applied.append(bool(degradation.get("applied", False)))
    return {
        "images": torch.stack(images, dim=0),
        "heights": torch.stack(heights, dim=0),
        "valid_masks": torch.stack(valid_masks, dim=0).to(torch.bool),
        "metadata": metadata,
        "resolution_factors": torch.as_tensor(resolution_factors, dtype=torch.float32),
        "target_gsd_m": torch.as_tensor(target_gsd_m, dtype=torch.float32),
        "resolution_degradation_applied": torch.as_tensor(
            resolution_degradation_applied,
            dtype=torch.bool,
        ),
    }


def _find_product_roots(dataset_root: Path, fold: str) -> list[Path]:
    candidates = []
    if (dataset_root / "tiles" / fold).exists() and (dataset_root / "height_maps" / fold).exists():
        candidates.append(dataset_root)
    for tile_dir in sorted(dataset_root.rglob(f"tiles/{fold}")):
        product_root = tile_dir.parent.parent
        if (product_root / "height_maps" / fold).exists():
            candidates.append(product_root)
    seen = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _pair_product_tiles(
    *,
    dataset_name: str,
    product_root: Path,
    fold: str,
) -> list[HeightTileRecord]:
    image_dir = product_root / "tiles" / fold
    height_dir = product_root / "height_maps" / fold
    images = {
        key: path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and (key := _tile_pair_key(path.name, fold)) is not None
    }
    heights = {
        key: path
        for path in sorted(height_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in HEIGHT_SUFFIXES
        and (key := _tile_pair_key(path.name, fold)) is not None
    }
    shared = sorted(set(images) & set(heights))
    if not shared:
        return []
    product_name = product_root.name
    return [
        HeightTileRecord(
            dataset_name=dataset_name,
            product_name=product_name,
            image_path=images[key],
            height_path=heights[key],
            tile_key=key,
        )
        for key in shared
    ]


def _tile_pair_key(filename: str, fold: str) -> Optional[str]:
    marker = f"_tile_{fold}_"
    if marker not in filename:
        return None
    suffix = filename.split(marker, 1)[1]
    return Path(suffix).with_suffix("").name


def _resize_height_and_mask(
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            _float_image_from_array(height).resize(size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ),
        np.asarray(
            Image.fromarray(valid_mask.astype(np.uint8) * 255).resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
        > 0,
    )


def _float_image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")


def _default_resolution_degradation(base_gsd_m: float) -> dict[str, float | bool]:
    return {
        "enabled": False,
        "applied": False,
        "factor": 1.0,
        "base_gsd_m": float(base_gsd_m),
        "target_gsd_m": float(base_gsd_m),
    }


def _pil_resampling(name: str) -> Image.Resampling:
    normalized = str(name).lower()
    if normalized == "nearest":
        return Image.Resampling.NEAREST
    if normalized == "box":
        return Image.Resampling.BOX
    if normalized == "bilinear":
        return Image.Resampling.BILINEAR
    if normalized == "bicubic":
        return Image.Resampling.BICUBIC
    if normalized == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unsupported PIL resampling mode: {name}")
