from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from synthforest.commands import preprocess_virtual_shizuoka_height as shizuoka


def _write_zip(path: Path, members: dict[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_discover_scene_sets_requires_all_products(tmp_path: Path) -> None:
    for product in ("Original", "Grid", "Ground", "Ortho"):
        _write_zip(
            tmp_path / "2025" / "LP" / product / "08" / "NC" / "38" / "08NC3800.zip",
            {"placeholder.txt": ""},
        )
    _write_zip(
        tmp_path / "2025" / "LP" / "Original" / "08" / "NC" / "38" / "08NC3801.zip",
        {"placeholder.txt": ""},
    )

    scenes = shizuoka.discover_scene_sets(tmp_path, "2025")

    assert [scene.grid_code for scene in scenes] == ["08NC3800"]
    assert scenes[0].original_zip.name == "08NC3800.zip"
    assert scenes[0].grid_zip.name == "08NC3800.zip"


def test_read_grid_dtm_from_zip_handles_valid_flag(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    grid_zip = tmp_path / "08NC3800.zip"
    _write_zip(
        grid_zip,
        {
            "08nc3800_0.5g.txt": (
                "1,0.25,0.75,10.0,1\n2,0.75,0.75,12.0,0\n3,0.25,0.25,8.0,1\n4,0.75,0.25,9.0,1\n"
            )
        },
    )

    dtm = shizuoka.read_grid_dtm_from_zip(grid_zip)

    assert dtm.source_has_valid_flag is True
    np.testing.assert_allclose(dtm.values, [[10.0, shizuoka.NODATA], [8.0, 9.0]])
    np.testing.assert_array_equal(dtm.valid_mask, [[True, False], [True, True]])
    assert dtm.spec.width == 2
    assert dtm.spec.height == 2
    assert dtm.resolution == pytest.approx(0.5)


def test_read_grid_dtm_from_zip_four_column_grid_is_all_valid(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    grid_zip = tmp_path / "08NC3800.zip"
    _write_zip(
        grid_zip,
        {
            "08nc3800_0.5g.txt": (
                "1,0.25,0.75,10.0\n2,0.75,0.75,12.0\n3,0.25,0.25,8.0\n4,0.75,0.25,9.0\n"
            )
        },
    )

    dtm = shizuoka.read_grid_dtm_from_zip(grid_zip)

    assert dtm.source_has_valid_flag is False
    np.testing.assert_allclose(dtm.values, [[10.0, 12.0], [8.0, 9.0]])
    assert bool(dtm.valid_mask.all())


def test_rasterize_las_filters_noise_classes_and_uses_max_z(tmp_path: Path) -> None:
    laspy = pytest.importorskip("laspy")
    from rasterio.transform import from_origin

    las_path = tmp_path / "points.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = np.array([0.10, 0.20, 0.20, 0.70, 0.70])
    las.y = np.array([0.90, 0.80, 0.80, 0.90, 0.40])
    las.z = np.array([5.0, 7.0, 6.0, 4.0, 3.0])
    las.classification = np.array([1, 7, 1, 18, 2], dtype=np.uint8)
    las.write(las_path)

    spec = shizuoka.RasterSpec(width=2, height=2, transform=from_origin(0.0, 1.0, 0.5, 0.5))
    dsm, counts, stats = shizuoka.rasterize_las_to_arrays(
        las_path,
        spec,
        ignore_classes=(7, 18),
        chunk_size=2,
    )

    np.testing.assert_allclose(dsm, [[6.0, shizuoka.NODATA], [shizuoka.NODATA, 3.0]])
    np.testing.assert_array_equal(counts, [[2, 0], [0, 1]])
    assert stats["total_points"] == 5
    assert stats["ignored_noise_class_points"] == 2
    assert stats["accepted_in_bounds_points"] == 3


def test_compute_chm_propagates_nodata_and_min_point_count() -> None:
    dsm = np.array([[5.0, shizuoka.NODATA], [8.0, 9.0]], dtype=np.float32)
    counts = np.array([[1, 1], [0, 2]], dtype=np.uint32)
    dtm = np.array([[6.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    dtm_valid = np.array([[True, True], [True, True]])

    chm, valid = shizuoka.compute_chm(
        dsm,
        counts,
        dtm,
        dtm_valid,
        min_dsm_points=1,
    )

    np.testing.assert_allclose(chm, [[0.0, shizuoka.NODATA], [shizuoka.NODATA, 6.0]])
    np.testing.assert_array_equal(valid, [[True, False], [False, True]])


def test_output_names_and_hardlinks(tmp_path: Path) -> None:
    paths = shizuoka.build_output_paths(tmp_path, "08NC3800", 1, 0.5)

    assert paths.benchmark_rgb.name == "Shizuoka_08NC3800_0001_RGB.tif"
    assert paths.benchmark_chm.name == "Shizuoka_08NC3800_0001_CHM.tif"
    assert paths.benchmark_chm_match_rgb.name == "Shizuoka_08NC3800_0001_CHM_matchRGB.tif"
    assert paths.scene_rgb.name == "rgb_shizuoka0001.tif"
    assert paths.scene_chm.name == "height_map_shizuoka0001.tif"
    assert shizuoka.build_output_paths(tmp_path, "08NC3800", 1, 0.2).benchmark_chm_match_rgb is None

    source = tmp_path / "benchmark" / "source.tif"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raster")
    destination = tmp_path / "scenes" / "images" / "linked.tif"
    shizuoka.hardlink_replace(source, destination, overwrite=False)

    assert destination.read_bytes() == b"raster"
    assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_resume_plan_preserves_existing_ids_and_assigns_new_after_max(tmp_path: Path) -> None:
    existing_paths = shizuoka.build_output_paths(tmp_path, "08NC3801", 7, 0.5)
    for path in shizuoka._expected_output_paths(existing_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"done")
    existing_rows = [
        {
            "scene_id": "0007",
            "grid_code": "08NC3801",
            "requested_resolution_m": "0.5",
        }
    ]
    scenes = [
        shizuoka.SceneSet(
            grid_code="08NC3800",
            original_zip=tmp_path / "new_original.zip",
            grid_zip=tmp_path / "new_grid.zip",
            ground_zip=tmp_path / "new_ground.zip",
            ortho_zip=tmp_path / "new_ortho.zip",
        ),
        shizuoka.SceneSet(
            grid_code="08NC3801",
            original_zip=tmp_path / "old_original.zip",
            grid_zip=tmp_path / "old_grid.zip",
            ground_zip=tmp_path / "old_ground.zip",
            ortho_zip=tmp_path / "old_ortho.zip",
        ),
    ]

    plans = shizuoka._build_resume_plan(
        scenes=scenes,
        output_root=tmp_path,
        resolution=0.5,
        existing_rows=existing_rows,
    )

    assert [(plan.scene.grid_code, plan.scene_index, plan.skip_existing) for plan in plans] == [
        ("08NC3800", 8, False),
        ("08NC3801", 7, True),
    ]


def test_resume_plan_rebuilds_incomplete_existing_outputs(tmp_path: Path) -> None:
    paths = shizuoka.build_output_paths(tmp_path, "08NC3801", 7, 0.5)
    paths.benchmark_rgb.parent.mkdir(parents=True, exist_ok=True)
    paths.benchmark_rgb.write_bytes(b"partial")
    scene = shizuoka.SceneSet(
        grid_code="08NC3801",
        original_zip=tmp_path / "old_original.zip",
        grid_zip=tmp_path / "old_grid.zip",
        ground_zip=tmp_path / "old_ground.zip",
        ortho_zip=tmp_path / "old_ortho.zip",
    )

    plans = shizuoka._build_resume_plan(
        scenes=[scene],
        output_root=tmp_path,
        resolution=0.5,
        existing_rows=[
            {
                "scene_id": "0007",
                "grid_code": "08NC3801",
                "requested_resolution_m": "0.5",
            }
        ],
    )

    assert plans[0].scene_index == 7
    assert plans[0].skip_existing is False
    assert paths.benchmark_chm in plans[0].missing_outputs


def test_resume_plan_rejects_different_resolution_manifest(tmp_path: Path) -> None:
    scene = shizuoka.SceneSet(
        grid_code="08NC3801",
        original_zip=tmp_path / "old_original.zip",
        grid_zip=tmp_path / "old_grid.zip",
        ground_zip=tmp_path / "old_ground.zip",
        ortho_zip=tmp_path / "old_ortho.zip",
    )

    with pytest.raises(ValueError, match="different resolution"):
        shizuoka._build_resume_plan(
            scenes=[scene],
            output_root=tmp_path,
            resolution=0.2,
            existing_rows=[
                {
                    "scene_id": "0007",
                    "grid_code": "08NC3801",
                    "requested_resolution_m": "0.5",
                }
            ],
        )
