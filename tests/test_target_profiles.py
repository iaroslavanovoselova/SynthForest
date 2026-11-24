import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from synthforest.tree_scatter.config import (  # noqa: E402
    get_available_target_profiles,
    load_target_profile_forest_type_weights,
    load_tree_scatter_config_for_forest_and_target_profile,
)
from synthforest.tree_scatter.terrain_context.rules import (  # noqa: E402
    pick_density_context_mode,
    pick_density_context_profile,
)

TARGET_PROFILE_CATALOG = "src/synthforest/tree_scatter/configs/target_profiles.yaml"


def _base_cfg():
    return OmegaConf.create(
        {
            "generation": {
                "target_profile_catalog_path": TARGET_PROFILE_CATALOG,
            }
        }
    )


def test_target_profile_catalog_lists_initial_profiles():
    profiles = get_available_target_profiles(_base_cfg())

    assert profiles == [
        "hard_negative_sparse_context",
        "large_closed_canopy",
        "small_medium_crowns",
    ]


def test_target_profile_overrides_win_over_scene_defaults():
    cfg = OmegaConf.merge(
        _base_cfg(),
        {
            "scatter": {
                "tree_scale_min": 1.0,
                "tree_scale_max": 1.0,
            },
            "mask_postprocess": {
                "min_final_area_m2": 9.0,
            },
        },
    )

    runtime_cfg = load_tree_scatter_config_for_forest_and_target_profile(
        cfg,
        "sa_tropical",
        "small_medium_crowns",
    )

    assert runtime_cfg.forest.type == "sa_tropical"
    assert runtime_cfg.target_profile.name == "small_medium_crowns"
    assert runtime_cfg.scatter.tree_scale_min == 0.55
    assert runtime_cfg.scatter.tree_scale_max == 0.90
    assert runtime_cfg.mask_postprocess.min_final_area_m2 == 0.40
    assert runtime_cfg.target_profile.quality_gates["bbox_eq_diameter_m_p50_range"] == [
        3.0,
        5.5,
    ]


def test_target_profile_does_not_override_render_gsd():
    cfg = OmegaConf.merge(
        _base_cfg(),
        {
            "render": {
                "gsd_at_nadir": 0.1,
            },
        },
    )

    runtime_cfg = load_tree_scatter_config_for_forest_and_target_profile(
        cfg,
        "sa_tropical",
        "large_closed_canopy",
    )

    assert runtime_cfg.render.gsd_at_nadir == 0.1


def test_unknown_target_profile_fails_clearly():
    with pytest.raises(ValueError, match="Unknown generation target profile"):
        load_tree_scatter_config_for_forest_and_target_profile(
            _base_cfg(),
            "sa_tropical",
            "not_a_profile",
        )


def test_profile_forest_type_weights_are_normalized():
    weights = load_target_profile_forest_type_weights(_base_cfg(), "large_closed_canopy")

    assert weights["af_tropical"] == 1.2
    assert weights["savanna_woodland"] == 0.0


def test_context_mode_overrides_density_band_choices():
    mode = pick_density_context_mode(
        "sparse",
        42,
        {"sparse": {"sparse_rural_artifacts": 1.0}},
    )

    assert mode == "sparse_rural_artifacts"


def test_context_profile_overrides_density_band_choices():
    profile = pick_density_context_profile(
        "medium",
        42,
        {"medium": {"forest_sparse_buildings": 1.0}},
    )

    assert profile == "forest_sparse_buildings"
