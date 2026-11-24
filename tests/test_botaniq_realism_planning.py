import random
import sys
import types
from pathlib import Path

if "omegaconf" not in sys.modules:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.DictConfig = object
    omegaconf_stub.OmegaConf = object
    sys.modules["omegaconf"] = omegaconf_stub

from synthforest.tree_scatter.botaniq_catalog import (  # noqa: E402
    _BotaniqAsset,
    _choose_botaniq_asset_for_height_target,
    _plan_botaniq_variant_height_scale,
)
from synthforest.tree_scatter.botaniq_quality import (  # noqa: E402
    botaniq_variant_quality_flags,
)


class _Cfg:
    botaniq_height_scale_z_max = 3.0
    botaniq_height_scale_anisotropy_max = 1.45
    botaniq_quality_max_scale_z = 3.0
    botaniq_quality_max_anisotropy = 1.45
    botaniq_quality_max_branch_displacement_m = 0.85
    botaniq_quality_max_leaf_removed_fraction = 0.12
    botaniq_quality_max_tilt_deg = 4.0
    botaniq_quality_allow_branch_swaps = False


def _asset(asset_id, title, height_m):
    return _BotaniqAsset(
        asset_id=asset_id,
        title=title,
        species_key="test species",
        species_label="Test species",
        species_common_name="test",
        blend_path=Path(f"/tmp/{asset_id}.blend"),
        seasons=("summer",),
        height_m=float(height_m),
        width_m=2.0,
        depth_m=2.0,
        model_detail="test",
    )


def test_height_scale_plan_clamps_extreme_target_height():
    plan = _plan_botaniq_variant_height_scale(
        _asset("small", "Small source", 1.0),
        20.0,
        None,
        scale_min=1.0,
        scale_max=1.0,
        height_scale_min=1.0,
        height_scale_max=1.0,
        crown_scale_power=0.85,
        scale_z_max=3.0,
        scale_xy_max=3.0,
        anisotropy_max=1.45,
    )

    assert plan.clamped is True
    assert abs(plan.target_height_m - 3.0) < 1e-9
    assert abs(plan.scale_z - 3.0) < 1e-9
    assert plan.scale_xy <= 3.0
    assert plan.anisotropy <= 1.45
    assert "scale_z_max" in plan.clamp_reason


def test_source_selection_prefers_compatible_taller_source():
    small = _asset("small", "Small source", 1.0)
    tall = _asset("tall", "Tall source", 8.0)

    selected, _source_index, _season, plan = _choose_botaniq_asset_for_height_target(
        [(small, 0, "summer"), (tall, 1, "summer")],
        18.0,
        random.Random(7),
        scale_min=1.0,
        scale_max=1.0,
        height_scale_min=1.0,
        height_scale_max=1.0,
        crown_scale_power=0.85,
        scale_z_max=3.0,
        scale_xy_max=3.0,
        anisotropy_max=1.45,
    )

    assert selected.asset_id == "tall"
    assert plan.clamped is False
    assert abs(plan.target_height_m - 18.0) < 1e-9
    assert abs(plan.scale_z - 2.25) < 1e-9


def test_quality_flags_aggressive_recipe_values():
    recipe = {
        "scale": {"xy": 1.0, "z": 4.0},
        "height": {"scale_xy": 1.0, "scale_z": 4.0, "anisotropy": 4.0},
        "branch_variation": {
            "swap_pairs": [["a", "b"]],
            "branches": [
                {
                    "max_displacement_m": 2.0,
                    "translation_xy": [1.5, 0.0],
                    "z_translation_m": 0.5,
                }
            ],
        },
        "leaf_density": {"removed_face_fraction": 0.20},
        "tilt": {"tilt_deg": 8.0},
    }

    assert botaniq_variant_quality_flags(recipe, _Cfg()) == [
        "scale_z_too_high",
        "scale_anisotropy_too_high",
        "branch_swap_pairs",
        "branch_displacement_too_high",
        "leaf_density_drop_too_high",
        "tilt_too_high",
    ]
