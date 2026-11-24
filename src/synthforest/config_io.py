"""Shared YAML loading with portable environment expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from synthforest.paths import config_path, expand_environment


def load_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a YAML mapping after resolving identifiers and environment values."""

    resolved = config_path(path)
    text = expand_environment(resolved.read_text(encoding="utf-8"), source=str(resolved))
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}, got {type(loaded).__name__}.")
    return resolved, loaded
