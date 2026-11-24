from __future__ import annotations

import re
from typing import Any


def format_count(value: int) -> str:
    value = int(value)
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def format_lr(value: float) -> str:
    mantissa, exponent = f"{float(value):.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def format_float_tag(value: float) -> str:
    return f"{float(value):.3g}".replace(".", "p")


def format_range(values: list[int]) -> str:
    if not values:
        return "none"
    numbers = [int(value) for value in values]
    if len(numbers) == 1:
        return str(numbers[0])
    return f"{min(numbers)}-{max(numbers)}"


def format_stage_tag(stage: Any) -> str:
    name = slugify(stage.name)
    prefix = {
        "detector_only": "det",
        "decoder_only": "dec",
        "joint_lora": "joint",
    }.get(name, name)
    tag = f"{prefix}{format_count(stage.max_iterations)}"
    stage_lr = getattr(stage.optimizer, "detector_lr", None)
    if stage_lr is None:
        stage_lr = getattr(stage.optimizer, "decoder_lr", None)
    if stage_lr is not None:
        tag += f"-dlr{format_lr(stage_lr)}"
    if stage.train_lora and stage.optimizer.lora_lr is not None:
        tag += f"-llr{format_lr(stage.optimizer.lora_lr)}"
    return tag


def format_domain_adaptation_tag(config: Any) -> str:
    if not config.enabled or not config.methods:
        return "noda"
    methods = "-".join(slugify(method) for method in config.methods)
    return f"da{methods}_p{format_float_tag(config.prob)}"


def slugify(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("-_") or "run"
