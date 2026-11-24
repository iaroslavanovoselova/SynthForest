from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from synthforest.training.config import DetectorTrainingConfig


def configure_dinov3_trainability(
    dinov3: nn.Module,
    config: Any,
) -> nn.Module:
    for parameter in dinov3.parameters():
        parameter.requires_grad = False

    if config.lora is None:
        return dinov3
    dinov3 = _apply_peft_lora_to_dinov3(dinov3, config)
    for parameter in dinov3.parameters():
        parameter.requires_grad = False
    return dinov3


def _apply_peft_lora_to_dinov3(
    dinov3: nn.Module,
    config: DetectorTrainingConfig,
) -> nn.Module:
    if config.lora is None:
        raise ValueError("Cannot attach LoRA without a lora configuration.")
    blocks = _resolve_transformer_blocks(dinov3)
    if config.lora.num_blocks > len(blocks):
        raise ValueError(
            f"lora.num_blocks={config.lora.num_blocks} exceeds the available "
            f"DINOv3 transformer blocks ({len(blocks)})."
        )
    first_layer = len(blocks) - config.lora.num_blocks
    layers_to_transform = list(range(first_layer, len(blocks)))

    if config.lora.gradient_checkpointing:
        enable_checkpointing = getattr(
            dinov3,
            "gradient_checkpointing_enable",
            None,
        )
        if not callable(enable_checkpointing):
            raise ValueError(
                "lora.gradient_checkpointing is true, but the loaded DINOv3 "
                "model does not expose gradient_checkpointing_enable()."
            )
        enable_checkpointing()

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError(
            "LoRA fine-tuning requires an importable `peft` installation. "
            "Use an environment with peft installed and make sure the conda "
            "environment's lib directory is first in LD_LIBRARY_PATH, or set "
            f"lora: null. Original import error: {exc}"
        ) from exc

    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=config.lora.dropout,
        bias="none",
        layers_to_transform=layers_to_transform,
        layers_pattern="layer",
    )
    try:
        dinov3 = get_peft_model(dinov3, lora_config)
    except Exception as exc:
        raise RuntimeError(
            "Failed to attach PEFT LoRA adapters to DINOv3. The local "
            "Hugging Face model is expected to expose transformer blocks at "
            "`model.layer` and targets such as q_proj/v_proj."
        ) from exc

    trainable = [name for name, parameter in dinov3.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "PEFT did not create any trainable LoRA parameters. Check "
            f"lora.target_modules and lora.num_blocks: {config.lora.target_modules}"
        )
    return dinov3


def _resolve_transformer_blocks(dinov3: nn.Module) -> nn.ModuleList:
    for path in ("model.layer", "encoder.layer", "blocks", "layers"):
        module = _get_nested_module(dinov3, path)
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return module
    raise ValueError(
        "Could not find DINOv3 transformer blocks. Expected one of: "
        "model.layer, encoder.layer, blocks, layers."
    )


def _get_nested_module(module: nn.Module, path: str) -> nn.Module | None:
    current: nn.Module | None = module
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def assert_dinov3_frozen(module: nn.Module) -> None:
    dinov3_modules = [
        child
        for name, child in module.named_modules()
        if name.endswith("dinov3") or name == "backbone.dinov3"
    ]
    for dinov3 in dinov3_modules:
        trainable = [
            name for name, parameter in dinov3.named_parameters() if parameter.requires_grad
        ]
        if trainable:
            sample = ", ".join(trainable[:5])
            raise AssertionError(f"DINOv3 backbone is not frozen. Trainable params: {sample}")


def assert_dinov3_trainability(
    module: nn.Module,
    config: Any,
) -> None:
    if config.lora is None:
        assert_dinov3_frozen(module)
        return
    _assert_dinov3_base_frozen(module)


def set_stage_trainability(
    module: nn.Module,
    *,
    train_lora: bool,
) -> None:
    """Train detector parameters and optionally enable attached LoRA adapters."""
    for name, parameter in module.named_parameters():
        if _is_dinov3_parameter_name(name):
            parameter.requires_grad = train_lora and _is_lora_parameter_name(name)
        else:
            parameter.requires_grad = True

    backbone = getattr(module, "backbone", None)
    if backbone is not None and hasattr(backbone, "backbone_grad_enabled"):
        backbone.backbone_grad_enabled = train_lora
        if not train_lora:
            backbone.dinov3.eval()
    if hasattr(module, "backbone_grad_enabled") and hasattr(module, "dinov3"):
        module.backbone_grad_enabled = train_lora
        if not train_lora:
            module.dinov3.eval()

    assert_stage_trainability(module, train_lora=train_lora)


def assert_stage_trainability(
    module: nn.Module,
    *,
    train_lora: bool,
) -> None:
    detector_trainable: list[str] = []
    lora_trainable: list[str] = []
    base_dino_trainable: list[str] = []

    for name, parameter in module.named_parameters():
        if _is_dinov3_parameter_name(name):
            if _is_lora_parameter_name(name):
                if parameter.requires_grad:
                    lora_trainable.append(name)
            elif parameter.requires_grad:
                base_dino_trainable.append(name)
        elif parameter.requires_grad:
            detector_trainable.append(name)

    if base_dino_trainable:
        sample = ", ".join(base_dino_trainable[:5])
        raise AssertionError(f"Base DINOv3 params must stay frozen. Trainable params: {sample}")
    if not detector_trainable:
        raise AssertionError("Stage expects trainable detector params, but none were found.")
    if train_lora and not lora_trainable:
        raise AssertionError("Stage expects trainable LoRA params, but none were found.")
    if not train_lora and lora_trainable:
        sample = ", ".join(lora_trainable[:5])
        raise AssertionError(f"Stage expects frozen LoRA params. Trainable params: {sample}")


def _assert_dinov3_base_frozen(module: nn.Module) -> None:
    non_lora = [
        name
        for name, parameter in module.named_parameters()
        if _is_dinov3_parameter_name(name)
        and not _is_lora_parameter_name(name)
        and parameter.requires_grad
    ]
    if non_lora:
        sample = ", ".join(non_lora[:5])
        raise AssertionError(f"DINOv3 has trainable base parameters: {sample}")


def summarize_trainable_parameters(module: nn.Module) -> dict[str, int]:
    summary = {
        "total_params": 0,
        "trainable_params": 0,
        "dinov3_total_params": 0,
        "dinov3_trainable_params": 0,
        "dinov3_lora_trainable_params": 0,
        "detector_trainable_params": 0,
    }
    for name, parameter in module.named_parameters():
        count = parameter.numel()
        is_trainable = parameter.requires_grad
        is_dinov3 = name.startswith("backbone.dinov3.") or name.startswith("dinov3.")
        is_lora = _is_lora_parameter_name(name)
        summary["total_params"] += count
        if is_trainable:
            summary["trainable_params"] += count
        if is_dinov3:
            summary["dinov3_total_params"] += count
            if is_trainable:
                summary["dinov3_trainable_params"] += count
            if is_trainable and is_lora:
                summary["dinov3_lora_trainable_params"] += count
        elif is_trainable:
            summary["detector_trainable_params"] += count
    return summary


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or ".lora_" in name


def _is_dinov3_parameter_name(name: str) -> bool:
    return name.startswith("backbone.dinov3.") or name.startswith("dinov3.")
