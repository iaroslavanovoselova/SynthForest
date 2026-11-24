import json
from types import SimpleNamespace

import pytest

pytest.importorskip("geodataset")
pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from geodataset.aoi import AOIGeneratorConfig  # noqa: E402

from synthforest.postprocessing.tilerize import (  # noqa: E402
    _DropEdgeTilesMixin,
    _tile_fits_inside_raster,
    _tile_source_valid_mask,
    _write_tile_valid_mask,
    tilerize_with_overlap,
)


def _tile(col, row, width=32, height=32):
    return SimpleNamespace(col=col, row=row, metadata={"width": width, "height": height})


class _DummyBase:
    def __init__(self, tiles, drop_edge_tiles=True):
        self._tiles = tiles
        self.drop_edge_tiles = drop_edge_tiles
        self.raster = SimpleNamespace(metadata={"width": 100, "height": 80})

    def _create_tiles(self):
        return self._tiles


class _DummyTilerizer(_DropEdgeTilesMixin, _DummyBase):
    pass


def test_tile_fits_inside_raster_excludes_crossing_windows():
    assert _tile_fits_inside_raster(_tile(0, 0), raster_width=100, raster_height=80)
    assert _tile_fits_inside_raster(_tile(68, 48), raster_width=100, raster_height=80)
    assert not _tile_fits_inside_raster(_tile(69, 48), raster_width=100, raster_height=80)
    assert not _tile_fits_inside_raster(_tile(68, 49), raster_width=100, raster_height=80)


def test_drop_edge_tiles_mixin_filters_only_when_enabled():
    tiles = [_tile(0, 0), _tile(68, 48), _tile(80, 0), _tile(0, 60)]

    filtered = _DummyTilerizer(tiles, drop_edge_tiles=True)._create_tiles()
    unfiltered = _DummyTilerizer(tiles, drop_edge_tiles=False)._create_tiles()

    assert filtered == tiles[:2]
    assert unfiltered == tiles


def test_tilerize_with_overlap_writes_empty_coco_for_empty_labels(tmp_path):
    raster_path = tmp_path / "rgb_satellite0001.tif"
    data = np.full((3, 16, 16), 80, dtype=np.uint8)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(data)

    categories = [{"id": 1, "name": "tree", "supercategory": "tree"}]
    coco_outputs, tiles_path = tilerize_with_overlap(
        raster_path=raster_path,
        labels=gpd.GeoDataFrame({"category_name": []}, geometry=[]),
        main_label_category_column_name="category_name",
        coco_categories_list=categories,
        aois_config=AOIGeneratorConfig(
            aois={"train": {"percentage": 1.0, "position": 1}},
            aoi_type="band",
        ),
        output_path=tmp_path / "tiles",
        tile_size=8,
        tile_overlap=0.0,
        ignore_tiles_without_labels=False,
        ignore_black_white_alpha_tiles_threshold=1.0,
    )

    coco_path = coco_outputs["train"]
    assert coco_path is not None
    assert len(list((tiles_path / "train").glob("*.tif"))) == 4

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 4
    assert coco["annotations"] == []
    assert coco["categories"] == categories


def test_write_tile_valid_mask_adds_alpha_for_edge_window(tmp_path):
    source_path = tmp_path / "source.tif"
    tile_path = tmp_path / "tile.tif"
    source_data = np.full((3, 3, 3), 80, dtype=np.uint8)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(source_data)

    with rasterio.open(source_path) as source:
        tile_data = source.read(
            window=rasterio.windows.Window(2, 1, 3, 3),
            boundless=True,
            fill_value=0,
        )

    with rasterio.open(
        tile_path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(tile_data)

    source = rasterio.open(source_path)
    try:
        tile = SimpleNamespace(
            col=2,
            row=1,
            metadata={"width": 3, "height": 3},
            mask=None,
            associated_raster=SimpleNamespace(
                data=source,
                metadata={"width": 3, "height": 3},
            ),
        )

        expected = np.array(
            [
                [True, False, False],
                [True, False, False],
                [False, False, False],
            ]
        )
        np.testing.assert_array_equal(_tile_source_valid_mask(tile), expected)

        result = _write_tile_valid_mask(
            tile_path=tile_path,
            tile=tile,
            valid_mask_mode="alpha",
        )
    finally:
        source.close()

    assert result["mode"] == "alpha"
    assert result["valid_pixels"] == 2
    with rasterio.open(tile_path) as tile_src:
        assert tile_src.count == 4
        np.testing.assert_array_equal(tile_src.read(4), expected.astype(np.uint8) * 255)
