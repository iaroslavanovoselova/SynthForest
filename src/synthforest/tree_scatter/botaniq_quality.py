from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _cfg_float(cfg: object, name: str, fallback: float) -> float:
    try:
        return float(getattr(cfg, name, fallback))
    except Exception:
        return float(fallback)


def _cfg_bool(cfg: object, name: str, fallback: bool) -> bool:
    try:
        return bool(getattr(cfg, name, fallback))
    except Exception:
        return bool(fallback)


def _coerce_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _max_branch_motion(branch_recipe: dict[str, Any]) -> float:
    max_motion = _coerce_float(branch_recipe.get("applied_max_displacement_m"))
    max_motion = max(max_motion, _coerce_float(branch_recipe.get("max_displacement_m")))

    translation_xy = branch_recipe.get("translation_xy", [])
    if isinstance(translation_xy, (list, tuple)) and len(translation_xy) >= 2:
        max_motion = max(
            max_motion,
            math.hypot(
                _coerce_float(translation_xy[0]),
                _coerce_float(translation_xy[1]),
            ),
        )
    max_motion = max(max_motion, abs(_coerce_float(branch_recipe.get("z_translation_m"))))
    return float(max_motion)


def botaniq_variant_quality_flags(
    recipe: dict[str, Any],
    cfg: object,
) -> list[str]:
    flags: list[str] = []
    height = recipe.get("height", {})
    scale = recipe.get("scale", {})
    if not isinstance(height, dict):
        height = {}
    if not isinstance(scale, dict):
        scale = {}

    scale_z = _coerce_float(height.get("scale_z", scale.get("z", 1.0)), 1.0)
    scale_xy = _coerce_float(height.get("scale_xy", scale.get("xy", 1.0)), 1.0)
    anisotropy = _coerce_float(
        height.get("anisotropy"),
        scale_z / max(scale_xy, 1e-6),
    )

    max_scale_z = _cfg_float(
        cfg,
        "botaniq_quality_max_scale_z",
        _cfg_float(cfg, "botaniq_height_scale_z_max", 3.0),
    )
    max_anisotropy = _cfg_float(
        cfg,
        "botaniq_quality_max_anisotropy",
        _cfg_float(cfg, "botaniq_height_scale_anisotropy_max", 1.45),
    )
    if scale_z > max_scale_z + 1e-6:
        flags.append("scale_z_too_high")
    if anisotropy > max_anisotropy + 1e-6:
        flags.append("scale_anisotropy_too_high")

    branch_variation = recipe.get("branch_variation", {})
    if isinstance(branch_variation, dict):
        swap_pairs = branch_variation.get("swap_pairs", [])
        if (
            isinstance(swap_pairs, list)
            and swap_pairs
            and not _cfg_bool(
                cfg,
                "botaniq_quality_allow_branch_swaps",
                False,
            )
        ):
            flags.append("branch_swap_pairs")

        max_branch_displacement = _cfg_float(
            cfg,
            "botaniq_quality_max_branch_displacement_m",
            _cfg_float(cfg, "botaniq_branch_max_displacement_m", 0.85),
        )
        branch_motion = max(
            (
                _max_branch_motion(branch)
                for branch in branch_variation.get("branches", [])
                if isinstance(branch, dict)
            ),
            default=0.0,
        )
        if branch_motion > max_branch_displacement + 1e-6:
            flags.append("branch_displacement_too_high")

    leaf_density = recipe.get("leaf_density", {})
    if isinstance(leaf_density, dict):
        removed_fraction = _coerce_float(leaf_density.get("removed_face_fraction"), 0.0)
        max_leaf_removed = _cfg_float(
            cfg,
            "botaniq_quality_max_leaf_removed_fraction",
            0.12,
        )
        if removed_fraction > max_leaf_removed + 1e-6:
            flags.append("leaf_density_drop_too_high")

    tilt = recipe.get("tilt", {})
    if isinstance(tilt, dict):
        tilt_deg = abs(_coerce_float(tilt.get("tilt_deg"), 0.0))
        max_tilt_deg = _cfg_float(cfg, "botaniq_quality_max_tilt_deg", 4.0)
        if tilt_deg > max_tilt_deg + 1e-6:
            flags.append("tilt_too_high")

    return flags


def botaniq_variant_quality_summary(
    recipes: list[dict[str, Any]],
) -> dict[str, object]:
    flag_counts: Counter[str] = Counter()
    clean_count = 0
    height_clamped_count = 0
    for recipe in recipes:
        flags = [str(flag) for flag in recipe.get("quality_flags", []) if str(flag).strip()]
        if flags:
            flag_counts.update(flags)
        else:
            clean_count += 1

        height = recipe.get("height", {})
        if isinstance(height, dict) and bool(height.get("clamped", False)):
            height_clamped_count += 1

    return {
        "variant_count": int(len(recipes)),
        "clean_variant_count": int(clean_count),
        "flagged_variant_count": int(len(recipes) - clean_count),
        "height_clamped_count": int(height_clamped_count),
        "flag_counts": dict(sorted(flag_counts.items())),
    }
