from __future__ import annotations

import csv
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from synthforest.commands import preprocess_france_height as france


def _feature(
    name: str, minx: float, miny: float, timestamp: str = "2025-05-01"
) -> dict[str, object]:
    maxx = minx + 1000.0
    maxy = miny + 1000.0
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny],
                ]
            ],
        },
        "properties": {
            "name": name,
            "url": f"https://example.test/{name}.tif",
            "name_download": f"{name}.tif",
            "timestamp": timestamp,
            "projection": france.CRS,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "width": 2000,
            "height": 2000,
        },
    }


def test_scene_index_row_uses_mnh_bbox_and_builds_20cm_rgb_url() -> None:
    row = france.scene_index_row_from_feature(
        _feature("LHD_FXX_0806_6797_MNH_O_0M50_LAMB93_IGN69", 805999.75, 6796000.25),
        "Forêt d'Orient / Troyes",
    )

    assert row is not None
    assert row["aoi_name"] == "For_t_d_Orient_Troyes"
    assert row["mnh_width"] == 2000
    assert row["mnh_height"] == 2000
    assert row["rgb_width"] == 5000
    assert row["rgb_height"] == 5000

    query = parse_qs(urlparse(str(row["rgb_url"])).query)
    assert query["LAYERS"] == [france.ORTHO_EXPRESS_LAYER]
    assert query["CRS"] == [france.CRS]
    assert query["BBOX"] == ["805999.75,6796000.25,806999.75,6797000.25"]
    assert query["WIDTH"] == ["5000"]
    assert query["HEIGHT"] == ["5000"]


def test_output_names_and_hardlinks(tmp_path: Path) -> None:
    row = {
        "scene_id": "7",
        "aoi_name": "foret_orient_troyes",
        "source_name": "LHD_FXX_0806_6797_MNH_O_0M50_LAMB93_IGN69",
        "mnh_name_download": "LHD_FXX_0806_6797_MNH_O_0M50_LAMB93_IGN69.tif",
    }

    paths = france.build_output_paths(tmp_path, row)

    assert paths.benchmark_rgb.name == (
        "France_foret_orient_troyes_0007_LHD_FXX_0806_6797_MNH_O_0M50_LAMB93_IGN69_RGB.tif"
    )
    assert paths.benchmark_chm.name.endswith("_CHM.tif")
    assert paths.benchmark_chm_match_rgb.name.endswith("_CHM_matchRGB.tif")
    assert paths.scene_rgb.name == "rgb_france0007.tif"
    assert paths.scene_chm.name == "height_map_france0007.tif"

    source = tmp_path / "benchmark" / "source.tif"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raster")
    destination = tmp_path / "scenes" / "images" / "linked.tif"
    france.hardlink_replace(source, destination, overwrite=False)

    assert destination.read_bytes() == b"raster"
    assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_default_discovery_writes_50_foret_orient_scene_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_query(aoi: france.AoiSpec, *, timeout: float, retries: int) -> list[dict[str, object]]:
        assert aoi.name == "foret_orient_troyes"
        base_x = 803000
        base_y = 6791000
        count = 60
        return [
            _feature(
                f"{aoi.name}_{idx:03d}", base_x + (idx % 10) * 1000, base_y + (idx // 10) * 1000
            )
            for idx in range(count)
        ]

    monkeypatch.setattr(france, "query_mnh_features", fake_query)

    scene_index = tmp_path / "scene_index.csv"
    summary = tmp_path / "summary.json"
    france.discover_scene_index(output_path=scene_index, summary_path=summary)

    with scene_index.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 50
    assert rows[0]["scene_id"] == "0001"
    assert rows[-1]["scene_id"] == "0050"
    counts = {
        aoi_name: sum(1 for row in rows if row["aoi_name"] == aoi_name)
        for aoi_name in {row["aoi_name"] for row in rows}
    }
    assert counts == {"foret_orient_troyes": 50}
    assert {row["rgb_layer"] for row in rows} == {france.ORTHO_EXPRESS_LAYER}
    assert summary.exists()


def test_load_custom_aois_and_count_allocation(tmp_path: Path) -> None:
    payload = {
        "aois": [
            {"name": "one", "bbox": [0, 0, 1000, 1000], "target_scenes": 10},
            {"name": "two", "bbox": [0, 0, 1000, 1000], "target_scenes": 20},
        ]
    }
    path = tmp_path / "aois.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    aois = france.load_aois(path)
    assert [aoi.name for aoi in aois] == ["one", "two"]
    assert france._allocate_counts(list(aois), 60) == {"one": 20, "two": 40}


def test_wms_exception_message_is_compacted(tmp_path: Path) -> None:
    error_path = tmp_path / "bad.tif"
    error_path.write_text(
        '<ServiceExceptionReport><ServiceException code="InvalidParameterValue">'
        "WIDTH query parameter exceed the limit (5010)"
        "</ServiceException></ServiceExceptionReport>",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="WIDTH query parameter"):
        france._raise_if_wms_exception(error_path)
