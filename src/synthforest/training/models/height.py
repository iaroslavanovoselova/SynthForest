from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from synthforest.training.config import HeightTrainingConfig
from synthforest.training.models.detector import _tokens_to_feature_map, load_dinov3_module
from synthforest.training.models.trainability import (
    assert_dinov3_trainability,
    configure_dinov3_trainability,
)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, use_batch_norm: bool) -> None:
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.norm1 = _norm_layer(features, use_batch_norm)
        self.norm2 = _norm_layer(features, use_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.norm1(self.conv1(out))
        out = self.activation(out)
        out = self.norm2(self.conv2(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, use_batch_norm: bool) -> None:
        super().__init__()
        self.residual1 = ResidualConvUnit(features, use_batch_norm)
        self.residual2 = ResidualConvUnit(features, use_batch_norm)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        *,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.residual1(skip)
        x = self.residual2(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class DPTHeightDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        features: int,
        out_channels: list[int],
        use_batch_norm: bool,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError("DPTHeightDecoder requires exactly four out_channels.")
        if int(output_channels) not in {1, 2}:
            raise ValueError("DPTHeightDecoder output_channels must be either 1 or 2.")
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channel, kernel_size=1) for out_channel in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features, use_batch_norm)
        self.refinenet2 = FeatureFusionBlock(features, use_batch_norm)
        self.refinenet3 = FeatureFusionBlock(features, use_batch_norm)
        self.refinenet4 = FeatureFusionBlock(features, use_batch_norm)
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, int(output_channels), kernel_size=1),
        )

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(feature_maps) != 4:
            raise ValueError(f"DPT decoder expects four feature maps, got {len(feature_maps)}.")
        resized = []
        for feature, project, resize in zip(feature_maps, self.projects, self.resize_layers):
            resized.append(resize(project(feature)))
        layer1, layer2, layer3, layer4 = resized
        layer1_rn = self.layer1_rn(layer1)
        layer2_rn = self.layer2_rn(layer2)
        layer3_rn = self.layer3_rn(layer3)
        layer4_rn = self.layer4_rn(layer4)

        path4 = self.refinenet4(layer4_rn, size=layer3_rn.shape[-2:])
        path3 = self.refinenet3(path4, layer3_rn, size=layer2_rn.shape[-2:])
        path2 = self.refinenet2(path3, layer2_rn, size=layer1_rn.shape[-2:])
        path1 = self.refinenet1(path2, layer1_rn)
        out = self.output_conv(path1)
        return F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)


class DinoV3DPTHeightModel(nn.Module):
    def __init__(self, config: HeightTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_grad_enabled = False
        self.patch_size = int(config.dinov3_patch_size)
        self.feature_layers = [int(value) for value in config.dpt_feature_layers]
        self.height_output_channels = int(config.height_output_channels)
        self.log_variance_min = float(config.log_variance_min)
        self.log_variance_max = float(config.log_variance_max)
        self.dinov3 = configure_dinov3_trainability(load_dinov3_module(config), config)
        self.decoder = DPTHeightDecoder(
            in_channels=int(config.dinov3_embed_dim),
            features=int(config.dpt_feature_dim),
            out_channels=[int(value) for value in config.dpt_out_channels],
            use_batch_norm=bool(config.dpt_use_batch_norm),
            output_channels=self.height_output_channels,
        )
        self.dinov3.eval()
        assert_dinov3_trainability(self, config)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_grad_enabled:
            self.dinov3.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.backbone_grad_enabled:
            with torch.no_grad():
                feature_maps = self._extract_feature_maps(images)
        else:
            feature_maps = self._extract_feature_maps(images)
        output = self.decoder(feature_maps, output_size=tuple(images.shape[-2:]))
        if output.shape[1] == 1:
            return {"height": output}
        if output.shape[1] == 2:
            height = output[:, 0:1]
            log_variance = _bounded_log_variance(
                output[:, 1:2],
                log_variance_min=self.log_variance_min,
                log_variance_max=self.log_variance_max,
            )
            return {"height": height, "log_variance": log_variance}
        raise RuntimeError(f"Unexpected height decoder channel count: {output.shape[1]}.")

    def _extract_feature_maps(self, images: torch.Tensor) -> list[torch.Tensor]:
        if _is_fake_dino(self.dinov3):
            hidden_states = _fake_intermediate_states(self.dinov3, images, self.feature_layers)
        else:
            hidden_states = _hf_intermediate_states(self.dinov3, images, self.feature_layers)
        return [
            _tokens_to_feature_map(
                patch_tokens=tokens,
                images=images,
                patch_size=self.patch_size,
            )
            for tokens in hidden_states
        ]


def build_dinov3_height_model(config: HeightTrainingConfig) -> DinoV3DPTHeightModel:
    if config.model_type != "dpt_height":
        raise ValueError(f"Unsupported height model_type {config.model_type!r}.")
    return DinoV3DPTHeightModel(config)


def _norm_layer(features: int, use_batch_norm: bool) -> nn.Module:
    if use_batch_norm:
        return nn.BatchNorm2d(features)
    groups = 32 if features % 32 == 0 else 1
    return nn.GroupNorm(groups, features)


def _bounded_log_variance(
    logits: torch.Tensor,
    *,
    log_variance_min: float,
    log_variance_max: float,
) -> torch.Tensor:
    lower = float(log_variance_min)
    upper = float(log_variance_max)
    return lower + (upper - lower) * torch.sigmoid(logits)


def _is_fake_dino(dinov3: nn.Module) -> bool:
    return hasattr(dinov3, "proj") and hasattr(dinov3, "model")


def _fake_intermediate_states(
    dinov3: nn.Module,
    images: torch.Tensor,
    layers: list[int],
) -> list[torch.Tensor]:
    tokens = dinov3.proj(images).flatten(2).transpose(1, 2).contiguous()
    wanted = set(layers)
    outputs: dict[int, torch.Tensor] = {}
    blocks = getattr(dinov3.model, "layer")
    for index, block in enumerate(blocks):
        tokens = block(tokens)
        if index in wanted:
            outputs[index] = tokens
    missing = [index for index in layers if index not in outputs]
    if missing:
        raise ValueError(f"Fake DINOv3 did not produce requested layer(s): {missing}")
    return [outputs[index] for index in layers]


def _hf_intermediate_states(
    dinov3: nn.Module,
    images: torch.Tensor,
    layers: list[int],
) -> list[torch.Tensor]:
    outputs = _forward_hf_dinov3(dinov3, images)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        raise RuntimeError(
            "DINOv3 height model requires the loaded backbone to return hidden_states "
            "when called with output_hidden_states=True."
        )
    max_layer = max(layers)
    if len(hidden_states) <= max_layer + 1:
        raise ValueError(
            f"DINOv3 returned {len(hidden_states)} hidden-state tensors, which cannot "
            f"satisfy requested zero-based layer {max_layer}."
        )
    return [hidden_states[layer + 1] for layer in layers]


def _forward_hf_dinov3(dinov3: nn.Module, images: torch.Tensor) -> Any:
    try:
        return dinov3(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
    except TypeError:
        return dinov3(
            images,
            output_hidden_states=True,
            return_dict=True,
        )
