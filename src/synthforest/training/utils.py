from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


FROZEN_BACKBONE_PREFIXES = ("backbone.dinov3.", "dinov3.")


def load_trained_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load a checkpoint whose frozen DINOv3 backbone weights may be omitted.

    Released SynthForest checkpoints store only the trained parameters; the
    frozen backbone is initialised from the pretrained DINOv3 weights when the
    model is built. Any other missing or unexpected key is an error.
    """

    result = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if not key.startswith(FROZEN_BACKBONE_PREFIXES) or "lora" in key.lower()
    ]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state does not match the model: "
            f"missing={missing[:5]}, unexpected={list(result.unexpected_keys)[:5]}"
        )
