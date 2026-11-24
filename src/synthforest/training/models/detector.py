from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.models.trainability import (
    assert_dinov3_trainability,
    configure_dinov3_trainability,
)

PYRAMID_FEATURE_NAMES = ("p8", "p16", "p32")


class FakeDinoV3Backbone(nn.Module):
    """Small deterministic DINO-like module for tests and CLI smoke runs."""

    def __init__(self, embed_dim: int = 16, patch_size: int = 16, num_blocks: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.model = FakeDinoEncoder(embed_dim=embed_dim, num_blocks=num_blocks)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.proj(x)
        patch_tokens = features.flatten(2).transpose(1, 2).contiguous()
        patch_tokens = self.model(patch_tokens)
        return {"x_norm_patchtokens": patch_tokens}

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor]:
        if pixel_values is None:
            if not args:
                raise ValueError(
                    "FakeDinoV3Backbone.forward requires pixel_values or a positional tensor."
                )
            pixel_values = args[0]
        features = self.forward_features(pixel_values)
        if return_dict:
            return {"last_hidden_state": features["x_norm_patchtokens"]}
        return (features["x_norm_patchtokens"],)


class FakeDinoEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(FakeDinoBlock(embed_dim) for _ in range(num_blocks))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.layer:
            tokens = block(tokens)
        return tokens


class FakeDinoBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.up_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.down_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_like = self.o_proj(
            self.q_proj(tokens) + self.k_proj(tokens) + self.v_proj(tokens)
        )
        mlp_like = self.down_proj(F.gelu(self.up_proj(tokens)))
        return tokens + 0.1 * (attention_like + mlp_like)


class DinoV3FeatureBackbone(nn.Module):
    def __init__(
        self,
        dinov3: nn.Module,
        embed_dim: int,
        patch_size: int,
        out_channels: int,
        config: DetectorTrainingConfig,
    ) -> None:
        super().__init__()
        self.backbone_grad_enabled = False
        self.dinov3 = configure_dinov3_trainability(dinov3, config)
        self.patch_size = patch_size
        self.input_projection = _conv_norm_relu(embed_dim, out_channels, kernel_size=1)
        self.p8_smooth = _conv_norm_relu(out_channels, out_channels, kernel_size=3)
        self.p16_smooth = _conv_norm_relu(out_channels, out_channels, kernel_size=3)
        self.p32_downsample = _conv_norm_relu(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
        )
        self.height_fusion = (
            HeightPyramidFusion(
                out_channels=out_channels,
                hidden_channels=int(config.height_fusion.encoder_hidden_channels),
                levels=tuple(config.height_fusion.levels),
                init_scale=float(config.height_fusion.fusion_init_scale),
                use_uncertainty=bool(config.height_fusion.use_uncertainty),
                uncertainty_reliability_tau=float(config.height_fusion.uncertainty_reliability_tau),
                uncertainty_reliability_min=float(config.height_fusion.uncertainty_reliability_min),
            )
            if config.height_fusion.enabled
            else None
        )
        self.height_fusion_uses_uncertainty = bool(
            config.height_fusion.enabled and config.height_fusion.use_uncertainty
        )
        self.out_channels = out_channels
        self.dinov3.eval()
        assert_dinov3_trainability(self, config)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_grad_enabled:
            self.dinov3.eval()
        return self

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        height = None
        uncertainty = None
        if self.height_fusion is not None:
            expected_channels = 5 if self.height_fusion_uses_uncertainty else 4
            expected_layout = (
                "[RGB,height,log_variance]" if expected_channels == 5 else "[RGB,height]"
            )
            if images.ndim != 4 or images.shape[1] != expected_channels:
                raise ValueError(
                    f"Height-fusion detector expects normalized {expected_channels}-channel "
                    f"inputs {expected_layout}, got shape {tuple(images.shape)}."
                )
            height = images[:, 3:4]
            if self.height_fusion_uses_uncertainty:
                uncertainty = images[:, 4:5]
            images = images[:, :3]
        elif images.ndim == 4 and images.shape[1] != 3:
            raise ValueError(
                f"RGB-only detector expects 3-channel inputs, got shape {tuple(images.shape)}."
            )
        if not self.backbone_grad_enabled:
            with torch.no_grad():
                raw_features = _forward_dinov3(self.dinov3, images)
        else:
            raw_features = _forward_dinov3(self.dinov3, images)
        patch_tokens = _extract_patch_tokens(raw_features)
        feature_map = _tokens_to_feature_map(
            patch_tokens=patch_tokens,
            images=images,
            patch_size=self.patch_size,
        )
        base = self.input_projection(feature_map)
        p8 = self.p8_smooth(
            F.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False)
        )
        p16 = self.p16_smooth(base)
        p32 = self.p32_downsample(base)
        features = OrderedDict([("p8", p8), ("p16", p16), ("p32", p32)])
        if self.height_fusion is not None:
            if height is None:
                raise AssertionError("height_fusion is enabled but no height tensor was split.")
            features = self.height_fusion(features, height, uncertainty=uncertainty)
        return features


class HeightPyramidFusion(nn.Module):
    """Fuse aligned height into FPN features shared by every RCNN head."""

    def __init__(
        self,
        *,
        out_channels: int,
        hidden_channels: int,
        levels: tuple[str, ...],
        init_scale: float,
        use_uncertainty: bool = False,
        uncertainty_reliability_tau: float = 0.5,
        uncertainty_reliability_min: float = 0.1,
    ) -> None:
        super().__init__()
        self.levels = tuple(levels)
        self.use_uncertainty = bool(use_uncertainty)
        self.uncertainty_reliability_tau = float(uncertainty_reliability_tau)
        self.uncertainty_reliability_min = float(uncertainty_reliability_min)
        self.encoder = nn.Sequential(
            _conv_norm_relu(1, int(hidden_channels), kernel_size=3),
            _conv_norm_relu(int(hidden_channels), int(out_channels), kernel_size=3),
        )
        self.level_scales = nn.ParameterDict(
            {
                level: nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
                for level in self.levels
            }
        )

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        height: torch.Tensor,
        *,
        uncertainty: torch.Tensor | None = None,
    ) -> OrderedDict[str, torch.Tensor]:
        if self.use_uncertainty and uncertainty is None:
            raise ValueError("Uncertainty-aware height fusion requires an uncertainty tensor.")
        fused = OrderedDict(features)
        for level in self.levels:
            if level not in fused:
                raise ValueError(f"Height fusion requested unknown FPN level {level!r}.")
            target = fused[level]
            resized = _resize_height_for_feature(height, target.shape[-2:])
            encoded = self.encoder(resized)
            if self.use_uncertainty:
                assert uncertainty is not None
                resized_uncertainty = _resize_height_for_feature(uncertainty, target.shape[-2:])
                reliability = torch.exp(
                    -resized_uncertainty / max(self.uncertainty_reliability_tau, 1e-6)
                ).clamp(min=self.uncertainty_reliability_min, max=1.0)
                encoded = encoded * reliability.to(dtype=encoded.dtype)
            fused[level] = target + self.level_scales[level].to(dtype=target.dtype) * encoded
        return fused


def build_dinov3_detector(
    config: DetectorTrainingConfig,
) -> FasterRCNN | MaskRCNN:
    if config.model_type == "fasterrcnn":
        return _build_dinov3_faster_rcnn(config)
    if config.model_type == "maskrcnn":
        return _build_dinov3_mask_rcnn(config)
    raise ValueError(f"Unsupported detector model_type {config.model_type!r}.")


def _build_dinov3_faster_rcnn(config: DetectorTrainingConfig) -> FasterRCNN:
    if config.model_type != "fasterrcnn":
        raise ValueError("The Faster R-CNN builder requires model_type: fasterrcnn.")
    return _build_dinov3_rcnn(config, model_type="fasterrcnn")


def _build_dinov3_mask_rcnn(config: DetectorTrainingConfig) -> MaskRCNN:
    if config.model_type != "maskrcnn":
        raise ValueError("The Mask R-CNN builder requires model_type: maskrcnn.")
    model = _build_dinov3_rcnn(config, model_type="maskrcnn")
    if not isinstance(model, MaskRCNN):
        raise AssertionError("Mask R-CNN builder returned the wrong model type.")
    return model


def _build_dinov3_rcnn(
    config: DetectorTrainingConfig,
    *,
    model_type: str,
) -> FasterRCNN | MaskRCNN:
    dinov3 = load_dinov3_module(config)
    backbone = DinoV3FeatureBackbone(
        dinov3=dinov3,
        embed_dim=config.dinov3_embed_dim,
        patch_size=config.dinov3_patch_size,
        out_channels=config.backbone_out_channels,
        config=config,
    )
    anchor_sizes = _anchor_sizes_for_feature_maps(config.anchor_sizes, len(PYRAMID_FEATURE_NAMES))
    aspect_ratios = (tuple(config.aspect_ratios),) * len(PYRAMID_FEATURE_NAMES)
    _validate_rpn_anchor_counts(anchor_sizes, aspect_ratios)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )
    box_roi_pooler = MultiScaleRoIAlign(
        featmap_names=list(PYRAMID_FEATURE_NAMES),
        output_size=7,
        sampling_ratio=2,
    )
    train_min_sizes = tuple(config.augmentation.resize_min_sizes_train)
    common_kwargs = {
        "num_classes": config.num_classes + 1,
        "rpn_anchor_generator": anchor_generator,
        "box_roi_pool": box_roi_pooler,
        "min_size": train_min_sizes,
        "max_size": config.augmentation.max_size,
        "image_mean": config.augmentation.image_mean,
        "image_std": config.augmentation.image_std,
        "box_score_thresh": config.box_score_thresh,
        "box_nms_thresh": config.box_nms_thresh,
        "box_detections_per_img": config.box_predictions_per_image,
    }
    if model_type == "fasterrcnn":
        model = FasterRCNN(backbone, **common_kwargs)
    elif model_type == "maskrcnn":
        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=list(PYRAMID_FEATURE_NAMES),
            output_size=14,
            sampling_ratio=2,
        )
        model = MaskRCNN(
            backbone,
            mask_roi_pool=mask_roi_pooler,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported RCNN model_type {model_type!r}.")
    model.transform = TrainEvalResizeTransform(
        train_min_sizes=train_min_sizes,
        eval_min_size=config.augmentation.resize_min_size_eval,
        max_size=config.augmentation.max_size,
        image_mean=config.augmentation.image_mean,
        image_std=config.augmentation.image_std,
        height_fusion_config=config.height_fusion,
    )
    assert_dinov3_trainability(model, config)
    return model


class TrainEvalResizeTransform(GeneralizedRCNNTransform):
    """Use random train resize sizes and a separate fixed eval resize size."""

    def __init__(
        self,
        *,
        train_min_sizes: tuple[int, ...],
        eval_min_size: int,
        max_size: int,
        image_mean: list[float],
        image_std: list[float],
        height_fusion_config: Any | None = None,
    ) -> None:
        self.height_fusion_config = height_fusion_config
        self.height_fusion_enabled = bool(
            height_fusion_config is not None and height_fusion_config.enabled
        )
        self.height_fusion_uses_uncertainty = bool(
            self.height_fusion_enabled and height_fusion_config.use_uncertainty
        )
        if self.height_fusion_enabled:
            image_mean = list(image_mean) + [float(height_fusion_config.mean_m)]
            image_std = list(image_std) + [float(height_fusion_config.std_m)]
            if self.height_fusion_uses_uncertainty:
                image_mean = list(image_mean) + [0.0]
                image_std = list(image_std) + [1.0]
        super().__init__(
            min_size=tuple(int(size) for size in train_min_sizes),
            max_size=int(max_size),
            image_mean=image_mean,
            image_std=image_std,
        )
        self.train_min_sizes = tuple(int(size) for size in train_min_sizes)
        self.eval_min_size = int(eval_min_size)

    def normalize(self, image):
        if not self.height_fusion_enabled:
            return super().normalize(image)
        expected_channels = 5 if self.height_fusion_uses_uncertainty else 4
        if image.shape[0] != expected_channels:
            raise ValueError(
                f"Height-fusion transform expects {expected_channels}-channel tensors "
                "before normalization, "
                f"got shape {tuple(image.shape)}."
            )
        dtype = image.dtype
        device = image.device
        rgb_mean = torch.as_tensor(self.image_mean[:3], dtype=dtype, device=device)
        rgb_std = torch.as_tensor(self.image_std[:3], dtype=dtype, device=device)
        rgb = (image[:3] - rgb_mean[:, None, None]) / rgb_std[:, None, None]

        cfg = self.height_fusion_config
        height = image[3:4].clamp(float(cfg.clip_min_m), float(cfg.clip_max_m))
        height = (height - float(cfg.mean_m)) / max(float(cfg.std_m), 1e-6)
        if not self.height_fusion_uses_uncertainty:
            return torch.cat([rgb, height], dim=0)

        lower = float(cfg.uncertainty_log_variance_min)
        upper = float(cfg.uncertainty_log_variance_max)
        uncertainty = image[4:5].clamp(lower, upper)
        uncertainty = (uncertainty - lower) / max(upper - lower, 1e-6)
        return torch.cat([rgb, height, uncertainty], dim=0)

    def resize(self, image, target):
        original_min_size = self.min_size
        self.min_size = self.train_min_sizes if self.training else (self.eval_min_size,)
        try:
            return super().resize(image, target)
        finally:
            self.min_size = original_min_size


def _conv_norm_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Sequential:
    padding = kernel_size // 2
    groups = 32 if out_channels % 32 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(inplace=True),
    )


def _resize_height_for_feature(
    height: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if tuple(height.shape[-2:]) == tuple(size):
        return height
    in_height, in_width = (int(height.shape[-2]), int(height.shape[-1]))
    out_height, out_width = (int(size[0]), int(size[1]))
    if out_height <= in_height and out_width <= in_width:
        return F.interpolate(height, size=size, mode="area")
    return F.interpolate(height, size=size, mode="bilinear", align_corners=False)


def _anchor_sizes_for_feature_maps(
    anchor_sizes: list[Any],
    num_feature_maps: int,
) -> tuple[tuple[int, ...], ...]:
    if not anchor_sizes:
        raise ValueError("anchor_sizes must not be empty.")
    if isinstance(anchor_sizes[0], (list, tuple)):
        sizes = tuple(tuple(int(size) for size in level_sizes) for level_sizes in anchor_sizes)
        if len(sizes) != num_feature_maps:
            raise ValueError(
                f"Expected anchor sizes for {num_feature_maps} feature maps, got {len(sizes)}."
            )
        return sizes

    flat_sizes = [int(size) for size in anchor_sizes]
    if len(flat_sizes) == num_feature_maps:
        return tuple((size,) for size in flat_sizes)
    if num_feature_maps == 1:
        return (tuple(flat_sizes),)
    raise ValueError(
        "Flat anchor_sizes must either contain one size per feature map or be nested, "
        f"for example [[16, 32], [64, 128], [256, 512]]. Got: {anchor_sizes}"
    )


def _validate_rpn_anchor_counts(
    anchor_sizes: tuple[tuple[int, ...], ...],
    aspect_ratios: tuple[tuple[float, ...], ...],
) -> None:
    counts = [
        len(level_sizes) * len(level_ratios)
        for level_sizes, level_ratios in zip(anchor_sizes, aspect_ratios)
    ]
    if len(set(counts)) <= 1:
        return
    details = ", ".join(
        f"{name}: {count} anchors/location ({len(level_sizes)} sizes x {len(level_ratios)} ratios)"
        for name, count, level_sizes, level_ratios in zip(
            PYRAMID_FEATURE_NAMES,
            counts,
            anchor_sizes,
            aspect_ratios,
        )
    )
    raise ValueError(
        "torchvision Faster R-CNN's default RPN head requires the same number of "
        f"anchors per spatial location on every feature level. Got {details}. "
        "Use the same number of anchor sizes per pyramid level, or provide a custom RPN head."
    )


def load_dinov3_module(config: Any) -> nn.Module:
    if config.dinov3_use_fake_backbone:
        return FakeDinoV3Backbone(
            embed_dim=config.dinov3_embed_dim,
            patch_size=config.dinov3_patch_size,
            num_blocks=int(getattr(config, "dinov3_fake_num_blocks", 4)),
        )
    model_dir_value = config.dinov3_model_dir
    if not model_dir_value:
        raise ValueError("dinov3_model_dir is required.")
    model_dir = Path(model_dir_value)
    if not model_dir.exists():
        raise FileNotFoundError(f"DINOv3 Hugging Face model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"DINOv3 Hugging Face model directory must contain config.json: {model_dir}"
        )
    if not _has_huggingface_weight_file(model_dir):
        raise FileNotFoundError(
            "DINOv3 Hugging Face model directory must contain model.safetensors, "
            f"pytorch_model.bin, or sharded weight index files: {model_dir}"
        )

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Loading DINOv3 from a Hugging Face snapshot requires the optional training "
            "dependencies `transformers` and `safetensors`. Install them in the training "
            "environment before starting the run."
        ) from exc
    if (model_dir / "model.safetensors").exists():
        try:
            import safetensors.torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "This DINOv3 snapshot uses model.safetensors, so the training environment "
                "must have `safetensors` installed."
            ) from exc

    try:
        return AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the local DINOv3 Hugging Face model. Make sure the installed "
            "`transformers` version supports model_type='dinov3_vit' and that the snapshot "
            f"is complete: {model_dir}"
        ) from exc


def _has_huggingface_weight_file(model_dir: Path) -> bool:
    weight_filenames = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((model_dir / filename).exists() for filename in weight_filenames)


def _forward_dinov3(dinov3: nn.Module, images: torch.Tensor) -> Any:
    if hasattr(dinov3, "forward_features"):
        return dinov3.forward_features(images)
    try:
        return dinov3(pixel_values=images, return_dict=True)
    except TypeError:
        return dinov3(images)


def _extract_patch_tokens(raw_features: Any) -> torch.Tensor:
    if isinstance(raw_features, dict):
        for key in (
            "x_norm_patchtokens",
            "patch_tokens",
            "last_hidden_state",
            "features",
        ):
            value = raw_features.get(key)
            if value is not None:
                return value
        tensor_values = [value for value in raw_features.values() if torch.is_tensor(value)]
        if tensor_values:
            return tensor_values[-1]
    for attr in (
        "x_norm_patchtokens",
        "patch_tokens",
        "last_hidden_state",
        "features",
    ):
        value = getattr(raw_features, attr, None)
        if torch.is_tensor(value):
            return value
    if torch.is_tensor(raw_features):
        return raw_features
    if isinstance(raw_features, (list, tuple)):
        for value in reversed(raw_features):
            if torch.is_tensor(value):
                return value
    to_tuple = getattr(raw_features, "to_tuple", None)
    if callable(to_tuple):
        for value in reversed(to_tuple()):
            if torch.is_tensor(value):
                return value
    raise TypeError(
        f"Could not extract patch tokens from DINOv3 output type {type(raw_features)!r}."
    )


def _tokens_to_feature_map(
    patch_tokens: torch.Tensor,
    images: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    if patch_tokens.ndim == 4:
        return patch_tokens
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected DINOv3 patch tokens with 3 or 4 dims, got {patch_tokens.shape}."
        )

    batch_size, num_tokens, channels = patch_tokens.shape
    height = images.shape[-2] // patch_size
    width = images.shape[-1] // patch_size
    expected_tokens = height * width
    if expected_tokens <= 0:
        raise ValueError(
            f"Input shape {tuple(images.shape[-2:])} is smaller than patch_size={patch_size}."
        )
    if num_tokens > expected_tokens:
        patch_tokens = patch_tokens[:, num_tokens - expected_tokens :, :]
        num_tokens = expected_tokens
    if expected_tokens != num_tokens:
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"Cannot reshape {num_tokens} DINOv3 tokens into a feature map "
                f"for input shape {tuple(images.shape[-2:])} and patch_size={patch_size}."
            )
        height = width = side
    return patch_tokens.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()
