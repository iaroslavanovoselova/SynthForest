import sys
import types
from collections import Counter

from synthforest.tree_scatter.random_streams import make_random_stream
from synthforest.tree_scatter.terrain_context.rules import pick_density_context_mode

if "omegaconf" not in sys.modules:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.DictConfig = object
    omegaconf_stub.OmegaConf = object
    sys.modules["omegaconf"] = omegaconf_stub

from synthforest.tree_scatter.config import sample_scatter_density, sample_scene_season

DENSITY_BAND_WEIGHTS = {
    "dense": 0.65,
    "medium": 0.20,
    "sparse": 0.15,
}
SEASON_WEIGHTS = {
    "summer": 0.70,
    "autumn": 0.30,
}


def test_named_random_streams_are_repeatable_and_independent():
    first = make_random_stream(42, "tree_scatter.test.first").random()
    repeat = make_random_stream(42, "tree_scatter.test.first").random()
    second = make_random_stream(42, "tree_scatter.test.second").random()

    assert first == repeat
    assert first != second


def test_density_band_sequence_for_existing_seed_range_is_preserved():
    bands = [
        sample_scatter_density(
            0.70,
            0.90,
            seed,
            band_weights=DENSITY_BAND_WEIGHTS,
        ).band
        for seed in range(42, 51)
    ]

    assert bands == [
        "dense",
        "dense",
        "dense",
        "dense",
        "sparse",
        "dense",
        "dense",
        "dense",
        "dense",
    ]


def test_season_stream_is_not_locked_to_density_band():
    samples = [
        (
            sample_scatter_density(
                0.70,
                0.90,
                seed,
                band_weights=DENSITY_BAND_WEIGHTS,
            ).band,
            sample_scene_season(SEASON_WEIGHTS, seed),
        )
        for seed in range(42, 92)
    ]

    dense_seasons = {season for band, season in samples if band == "dense"}
    sparse_seasons = {season for band, season in samples if band == "sparse"}

    assert "autumn" in dense_seasons
    assert sparse_seasons != {"autumn"}


def test_sparse_context_mode_stream_varies_independently_from_sparse_band():
    sparse_seeds = [
        seed
        for seed in range(42, 92)
        if sample_scatter_density(
            0.70,
            0.90,
            seed,
            band_weights=DENSITY_BAND_WEIGHTS,
        ).band
        == "sparse"
    ]
    modes = [pick_density_context_mode("sparse", seed) for seed in sparse_seeds]
    mode_counts = Counter(modes)

    assert len(mode_counts) > 1
    assert mode_counts["sparse_rural_artifacts"] < len(modes)
