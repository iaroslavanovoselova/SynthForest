"""Portable path and configuration resolution for SynthForest."""

from __future__ import annotations

import os
import re
from pathlib import Path

EXTERNAL_ROOT_VARIABLES = (
    "SYNTHFOREST_DATA_ROOT",
    "SYNTHFOREST_MODEL_ROOT",
    "SYNTHFOREST_ASSET_ROOT",
    "SYNTHFOREST_OUTPUT_ROOT",
    "SYNTHFOREST_BLENDER_BIN",
)

_UNRESOLVED_ENV = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


class PathConfigurationError(ValueError):
    """Raised when a configured path cannot be resolved portably."""


def project_root(start: str | Path | None = None) -> Path:
    """Return the source-checkout root, independent of the current directory.

    Falls back to the current directory when no checkout is found.
    """

    candidates = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    candidates.extend((Path.cwd().resolve(), Path(__file__).resolve()))
    for candidate in candidates:
        if candidate.is_file():
            candidate = candidate.parent
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "src" / "synthforest").is_dir():
                return parent
    return Path.cwd().resolve()


def expand_environment(value: str, *, source: str = "value") -> str:
    """Expand environment variables and reject unresolved placeholders."""

    expanded = os.path.expandvars(os.path.expanduser(value))
    unresolved = sorted({left or right for left, right in _UNRESOLVED_ENV.findall(expanded)})
    if unresolved:
        names = ", ".join(unresolved)
        raise PathConfigurationError(
            f"Unresolved environment variable(s) in {source}: {names}. "
            "Set them explicitly or pass an overriding CLI path."
        )
    return expanded


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    """Resolve a configured path relative to the checkout root by default."""

    expanded = Path(expand_environment(str(value), source="path"))
    if expanded.is_absolute():
        return expanded.resolve()
    root = Path(base).expanduser().resolve() if base is not None else project_root()
    return (root / expanded).resolve()


def external_root(variable: str, *, required: bool = True) -> Path | None:
    """Return one of the documented external roots."""

    if variable not in EXTERNAL_ROOT_VARIABLES:
        raise KeyError(f"Unknown SynthForest external root: {variable}")
    value = os.environ.get(variable)
    if value:
        return resolve_path(value)
    if required:
        raise PathConfigurationError(
            f"{variable} is required for this operation. See docs/data.md "
            "or pass the corresponding CLI override."
        )
    return None


def output_root() -> Path:
    """Return the configured output root, defaulting to ``<repo>/outputs``."""

    return external_root("SYNTHFOREST_OUTPUT_ROOT", required=False) or project_root() / "outputs"


def data_path(relative: str | Path = "") -> Path:
    """Return a path below the configured data root or checkout ``data/``."""

    root = external_root("SYNTHFOREST_DATA_ROOT", required=False) or project_root() / "data"
    return (root / relative).resolve()


def model_path(relative: str | Path = "") -> Path:
    """Return a path below the configured model root or checkout ``models/``."""

    root = external_root("SYNTHFOREST_MODEL_ROOT", required=False) or project_root() / "models"
    return (root / relative).resolve()


def asset_path(relative: str | Path = "") -> Path:
    """Return a path below the configured asset root or checkout ``data/assets``."""

    root = (
        external_root("SYNTHFOREST_ASSET_ROOT", required=False)
        or project_root() / "data" / "assets"
    )
    return (root / relative).resolve()


def config_root() -> Path:
    """Return the repository ``configs/`` directory."""

    return project_root() / "configs"


def config_path(value: str | Path) -> Path:
    """Resolve an explicit YAML path or a config identifier.

    Identifiers are relative to ``configs/`` and may omit the YAML suffix,
    for example ``height/train_syn20`` or ``crowns/training/synthetic_hf``.
    """

    raw = expand_environment(str(value), source="config path")
    candidate = Path(raw).expanduser()
    direct_candidates = [candidate]
    if not candidate.is_absolute():
        direct_candidates.append(project_root() / candidate)
    for direct in direct_candidates:
        if direct.is_file():
            return direct.resolve()

    resource = config_root() / candidate
    resource_candidates = [resource]
    if resource.suffix not in {".yaml", ".yml"}:
        resource_candidates.extend((resource.with_suffix(".yaml"), resource.with_suffix(".yml")))
    for packaged in resource_candidates:
        if packaged.is_file():
            return packaged.resolve()
    raise FileNotFoundError(
        f"Configuration {value!s} was not found as a file or config identifier under configs/."
    )


def environment_status() -> dict[str, str | None]:
    """Return documented external-root values for diagnostics."""

    return {name: os.environ.get(name) for name in EXTERNAL_ROOT_VARIABLES}
