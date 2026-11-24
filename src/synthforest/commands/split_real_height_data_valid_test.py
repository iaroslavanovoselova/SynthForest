#!/usr/bin/env python3
"""Create a stable scene-level valid/test split for height benchmark datasets.

Dry-run is the default. Use --apply only after checking the printed split.
The source rasters under scenes/ are left untouched. For a validation scene this
moves its full benchmark rasters to <data-root>/validation; test rasters stay
in <data-root>/benchmark. Tiled products are retargeted as whole scenes by
renaming tiles/<fold>, height_maps/<fold>, and split tokens in tile filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from synthforest.paths import data_path

DEFAULT_FRANCE_ROOT = data_path("CHM_rasters_20cm/France_CHM_OrthoExpress-20cm_MNH-50cm")
DEFAULT_HEIGHT_TILES_ROOT = data_path("CHM_tiles_20cm")
DEFAULT_FRANCE_DATASET = "france_mnh_orthoexpress_height_50cm"
DEFAULT_SHIZUOKA_ROOT = data_path("CHM_rasters_20cm/virtual-shizuoka_CHM")
DEFAULT_SHIZUOKA_DATASET = "virtual_shizuoka_height_50cm"
DEFAULT_NEON_ROOT = data_path("CHM_rasters_50cm/NEON_CHM_1m_real_eval_rasters")
DEFAULT_NEON_TILES_ROOT = data_path()
DEFAULT_NEON_DATASET = "NEON_CHM_1m_real_eval"
DEFAULT_FORCED_TEST_SCENES = ()
SPLITS = ("valid", "test")
HEIGHT_BINS_M = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0)


@dataclass(frozen=True)
class DatasetPreset:
    label: str
    root: Path
    tiles_root: Path
    dataset_name: str
    product_prefix: str
    scene_chm_prefix: str
    split_manifest_stem: str
    reference_description: str
    manifest_kind: str = "standard"


DATASET_PRESETS = {
    "france": DatasetPreset(
        label="France",
        root=DEFAULT_FRANCE_ROOT,
        tiles_root=DEFAULT_HEIGHT_TILES_ROOT,
        dataset_name=DEFAULT_FRANCE_DATASET,
        product_prefix="rgb_france",
        scene_chm_prefix="height_map_france",
        split_manifest_stem="france_height_valid_test_split",
        reference_description=(
            "Validation-only France RGB tile references for height domain adaptation."
        ),
    ),
    "shizuoka": DatasetPreset(
        label="Shizuoka",
        root=DEFAULT_SHIZUOKA_ROOT,
        tiles_root=DEFAULT_HEIGHT_TILES_ROOT,
        dataset_name=DEFAULT_SHIZUOKA_DATASET,
        product_prefix="rgb_shizuoka",
        scene_chm_prefix="height_map_shizuoka",
        split_manifest_stem="shizuoka_height_valid_test_split",
        reference_description=(
            "Validation-only Shizuoka RGB tile references for height evaluation."
        ),
    ),
    "neon": DatasetPreset(
        label="NEON",
        root=DEFAULT_NEON_ROOT,
        tiles_root=DEFAULT_NEON_TILES_ROOT,
        dataset_name=DEFAULT_NEON_DATASET,
        product_prefix="",
        scene_chm_prefix="",
        split_manifest_stem="neon_height_valid_test_split",
        reference_description=("Validation-only NEON RGB tile references for height evaluation."),
        manifest_kind="neon_rasters",
    ),
}


@dataclass(frozen=True)
class HeightStats:
    source: str
    valid_fraction: float
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    frac_le_1: float | None = None
    frac_gt_2: float | None = None
    frac_gt_10: float | None = None
    frac_gt_20: float | None = None
    frac_gt_30: float | None = None
    histogram: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    product_name: str
    product_root: Path
    qa_path: Path
    scene_chm_path: Path
    benchmark_filenames: tuple[str, ...]
    stats: HeightStats
    stratum: str


def _scene_id(value: str) -> str:
    match = re.search(r"(\d{1,4})$", str(value).strip())
    if not match:
        raise ValueError(f"Could not parse scene id from {value!r}")
    return f"{int(match.group(1)):04d}"


def _neon_scene_id(value: str) -> str:
    stem = Path(str(value).strip()).stem.lower()
    for suffix in ("_rgb", "_chm", "_maxar1"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError(f"Could not parse NEON scene id from {value!r}")
    return stem


def _normalise_scene_id(value: str, args: argparse.Namespace) -> str:
    if args.dataset_preset == "neon":
        return _neon_scene_id(value)
    return _scene_id(value)


def _load_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = args.data_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    dataset_root = args.tiles_root / args.dataset_name
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scene_id = _scene_id(row["scene_id"])
            product_name = f"{args.product_prefix}{scene_id}"
            scene_chm_name = (
                Path(row["scene_chm"]).name
                if row.get("scene_chm")
                else f"{args.scene_chm_prefix}{scene_id}.tif"
            )
            benchmark_names = tuple(
                Path(row[key]).name
                for key in ("benchmark_rgb", "benchmark_chm", "benchmark_chm_match_rgb")
                if row.get(key)
            )
            rows.append(
                {
                    "scene_id": scene_id,
                    "product_name": product_name,
                    "product_root": dataset_root / product_name,
                    "qa_path": args.data_root / "qa" / Path(row["qa_json"]).name,
                    "scene_chm_path": args.data_root / "scenes" / "height_maps" / scene_chm_name,
                    "benchmark_filenames": benchmark_names,
                }
            )
    if not rows:
        raise ValueError(f"No scenes found in {manifest_path}")
    return rows


def _load_neon_raster_inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_root = args.tiles_root / args.dataset_name
    raster_dirs = [
        args.data_root / "benchmark",
        args.data_root / "validation",
    ]
    chm_by_scene_id: dict[str, Path] = {}
    rgb_by_scene_id: dict[str, Path] = {}
    for raster_dir in raster_dirs:
        if not raster_dir.exists():
            continue
        for chm_path in sorted(raster_dir.glob("*_CHM.tif")):
            chm_by_scene_id.setdefault(_neon_scene_id(chm_path.name), chm_path)
        for rgb_path in sorted(raster_dir.glob("*_RGB.tif")):
            scene_id = _neon_scene_id(rgb_path.name)
            if scene_id in rgb_by_scene_id:
                raise ValueError(
                    f"Duplicate NEON RGB raster for scene {scene_id!r}: "
                    f"{rgb_by_scene_id[scene_id]} and {rgb_path}"
                )
            rgb_by_scene_id[scene_id] = rgb_path

    rows: list[dict[str, Any]] = []
    for scene_id, rgb_path in sorted(rgb_by_scene_id.items()):
        chm_path = chm_by_scene_id.get(scene_id)
        if chm_path is None:
            raise FileNotFoundError(f"Missing NEON CHM raster for RGB scene {scene_id!r}")
        product_name = Path(rgb_path.name).stem.lower()
        rows.append(
            {
                "scene_id": scene_id,
                "product_name": product_name,
                "product_root": dataset_root / product_name,
                "qa_path": args.data_root,
                "scene_chm_path": chm_path,
                "benchmark_filenames": (rgb_path.name, chm_path.name),
            }
        )
    if not rows:
        raise ValueError(
            f"No NEON *_RGB.tif rasters found under {args.data_root / 'benchmark'} "
            f"or {args.data_root / 'validation'}"
        )
    return rows


def _stats_from_qa(path: Path) -> HeightStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    chm = data.get("chm_match_rgb") or data.get("chm") or {}
    stats = chm.get("stats_m") or {}
    return HeightStats(
        source=f"qa:{path.name}",
        valid_fraction=float(chm.get("valid_fraction", 1.0)),
        mean=float(stats["mean"]),
        p50=float(stats["p50"]),
        p95=float(stats["p95"]),
        p99=float(stats["p99"]),
        max=float(stats["max"]),
    )


def _stats_from_raster(
    path: Path,
    *,
    sample_max_dim: int,
    min_height_m: float,
    max_height_m: float,
) -> HeightStats:
    try:
        import warnings

        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError as exc:
        raise RuntimeError(
            "Raster stats require numpy and rasterio. Run in the preprocessing/training "
            "environment, or pass --stats-source qa to use existing QA summaries."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as dataset:
            scale = max(dataset.width, dataset.height) / float(sample_max_dim)
            if scale > 1.0:
                out_height = max(1, int(round(dataset.height / scale)))
                out_width = max(1, int(round(dataset.width / scale)))
                masked = dataset.read(
                    1,
                    out_shape=(out_height, out_width),
                    masked=True,
                    resampling=Resampling.nearest,
                )
            else:
                masked = dataset.read(1, masked=True)

    values = np.asarray(masked.filled(np.nan), dtype=np.float32)
    valid = (
        ~np.ma.getmaskarray(masked)
        & np.isfinite(values)
        & (values >= float(min_height_m))
        & (values <= float(max_height_m))
    )
    if not bool(valid.any()):
        raise ValueError(f"No valid height pixels in {path}")
    valid_values = values[valid]
    hist_counts, _ = np.histogram(valid_values, bins=np.asarray(HEIGHT_BINS_M, dtype=np.float32))
    hist_sum = float(hist_counts.sum())
    histogram = tuple(float(value) / hist_sum for value in hist_counts)
    return HeightStats(
        source=f"raster:{path.name}",
        valid_fraction=float(valid.mean()),
        mean=float(valid_values.mean()),
        p50=float(np.percentile(valid_values, 50)),
        p95=float(np.percentile(valid_values, 95)),
        p99=float(np.percentile(valid_values, 99)),
        max=float(valid_values.max()),
        frac_le_1=float((valid_values <= 1.0).mean()),
        frac_gt_2=float((valid_values > 2.0).mean()),
        frac_gt_10=float((valid_values > 10.0).mean()),
        frac_gt_20=float((valid_values > 20.0).mean()),
        frac_gt_30=float((valid_values > 30.0).mean()),
        histogram=histogram,
    )


def _load_stats(row: dict[str, Any], args: argparse.Namespace) -> HeightStats:
    if args.stats_source in {"auto", "raster"}:
        try:
            return _stats_from_raster(
                row["scene_chm_path"],
                sample_max_dim=args.sample_max_dim,
                min_height_m=args.min_height_m,
                max_height_m=args.max_height_m,
            )
        except Exception:
            if args.stats_source == "raster" or args.manifest_kind == "neon_rasters":
                raise
    return _stats_from_qa(row["qa_path"])


def _stratum(stats: HeightStats, args: argparse.Namespace) -> str:
    if stats.frac_gt_2 is not None and stats.frac_gt_2 <= args.field_canopy_fraction:
        return "field"
    if stats.frac_gt_2 is None and (
        stats.p95 <= args.field_p95_m or stats.mean <= args.field_mean_m
    ):
        return "field"
    if stats.p95 < args.low_p95_m:
        return "low"
    if stats.max >= args.tall_max_m or stats.p99 >= args.tall_p99_m:
        return "tall"
    return "mid"


def _load_records(args: argparse.Namespace) -> list[SceneRecord]:
    if args.manifest_kind == "neon_rasters":
        rows = _load_neon_raster_inventory(args)
    else:
        rows = _load_manifest(args)
    records: list[SceneRecord] = []
    for row in rows:
        if not row["product_root"].exists():
            raise FileNotFoundError(f"Tile product not found: {row['product_root']}")
        stats = _load_stats(row, args)
        records.append(
            SceneRecord(
                scene_id=row["scene_id"],
                product_name=row["product_name"],
                product_root=row["product_root"],
                qa_path=row["qa_path"],
                scene_chm_path=row["scene_chm_path"],
                benchmark_filenames=row["benchmark_filenames"],
                stats=stats,
                stratum=_stratum(stats, args),
            )
        )
    return records


def _target_valid_count(total: int, args: argparse.Namespace) -> int:
    if args.valid_count is not None:
        count = int(args.valid_count)
    else:
        count = int(round(total * float(args.valid_ratio)))
    return max(1, min(total - 1, count))


def _feature_names(records: list[SceneRecord]) -> list[str]:
    names = [
        "valid_fraction",
        "mean",
        "p50",
        "p95",
        "p99",
        "max",
        "frac_le_1",
        "frac_gt_2",
        "frac_gt_10",
        "frac_gt_20",
        "frac_gt_30",
    ]
    return [
        name for name in names if all(getattr(record.stats, name) is not None for record in records)
    ]


def _mean_feature(records: list[SceneRecord], name: str) -> float:
    return fmean(float(getattr(record.stats, name)) for record in records)


def _histogram_mean(records: list[SceneRecord]) -> tuple[float, ...] | None:
    if not records or any(record.stats.histogram is None for record in records):
        return None
    width = len(records[0].stats.histogram or ())
    return tuple(
        fmean(float(record.stats.histogram[index]) for record in records) for index in range(width)
    )


def _split_score(
    records: list[SceneRecord],
    valid_ids: set[str],
    feature_names: list[str],
) -> float:
    valid = [record for record in records if record.scene_id in valid_ids]
    test = [record for record in records if record.scene_id not in valid_ids]
    score = 0.0
    for name in feature_names:
        values = [float(getattr(record.stats, name)) for record in records]
        scale = max(values) - min(values)
        if scale <= 1e-9:
            scale = max(1.0, abs(fmean(values)))
        score += abs(_mean_feature(valid, name) - _mean_feature(test, name)) / scale
    valid_hist = _histogram_mean(valid)
    test_hist = _histogram_mean(test)
    if valid_hist is not None and test_hist is not None:
        score += 0.5 * sum(abs(a - b) for a, b in zip(valid_hist, test_hist))
    return score


def _strata_ok(records: list[SceneRecord], valid_ids: set[str]) -> bool:
    counts = Counter(record.stratum for record in records)
    valid_counts = Counter(record.stratum for record in records if record.scene_id in valid_ids)
    for stratum, total in counts.items():
        if total < 2:
            continue
        if valid_counts[stratum] == 0 or valid_counts[stratum] == total:
            return False
    return True


def _choose_valid_ids(records: list[SceneRecord], args: argparse.Namespace) -> set[str]:
    all_ids = {record.scene_id for record in records}
    forced_test = {_normalise_scene_id(scene_id, args) for scene_id in args.force_test_scene_id}
    forced_valid = {_normalise_scene_id(scene_id, args) for scene_id in args.force_valid_scene_id}
    unknown = (forced_test | forced_valid) - all_ids
    if unknown:
        raise ValueError(f"Forced scene ids are not present in the dataset: {sorted(unknown)}")
    overlap = forced_test & forced_valid
    if overlap:
        raise ValueError(f"Scenes cannot be forced to both valid and test: {sorted(overlap)}")

    valid_count = _target_valid_count(len(records), args)
    if len(forced_valid) > valid_count:
        raise ValueError("--valid-count/--valid-ratio leaves no room for all forced valid scenes")
    eligible = sorted(all_ids - forced_test - forced_valid)
    need = valid_count - len(forced_valid)
    if need > len(eligible):
        raise ValueError("Forced test scenes leave too few scenes for the validation split")

    rng = random.Random(args.seed)
    feature_names = _feature_names(records)
    best_ids: set[str] | None = None
    best_score = math.inf
    for _ in range(args.search_iterations):
        candidate = set(forced_valid)
        candidate.update(rng.sample(eligible, need))
        if args.balance_strata and not _strata_ok(records, candidate):
            continue
        score = _split_score(records, candidate, feature_names)
        if score < best_score:
            best_ids = candidate
            best_score = score
    if best_ids is None:
        for _ in range(args.search_iterations):
            candidate = set(forced_valid)
            candidate.update(rng.sample(eligible, need))
            score = _split_score(records, candidate, feature_names)
            if score < best_score:
                best_ids = candidate
                best_score = score
    if best_ids is None:
        raise RuntimeError("Could not find a valid split")
    return best_ids


def _retarget_name(name: str, target: str) -> str:
    result = name
    for old in ("train", "valid", "test"):
        if old == target:
            continue
        result = result.replace(f"_tile_{old}_", f"_tile_{target}_")
        result = result.replace(f"_{old}.", f"_{target}.")
        result = result.replace(f"_{old}_", f"_{target}_")
    return result


def _rename_file(path: Path, destination: Path, *, apply: bool, actions: list[str]) -> None:
    if path == destination:
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    actions.append(f"rename {path} -> {destination}")
    if apply:
        path.rename(destination)


def _move_file(path: Path, destination: Path, *, apply: bool, actions: list[str]) -> None:
    if path == destination:
        return
    if destination.exists():
        return
    if not path.exists():
        return
    actions.append(f"move {path} -> {destination}")
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))


def _merge_dir(source: Path, destination: Path, *, apply: bool, actions: list[str]) -> None:
    if not source.exists() or source == destination:
        return
    actions.append(f"merge {source} -> {destination}")
    if not apply:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        child_destination = destination / child.name
        if child_destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {child_destination}")
        shutil.move(str(child), str(child_destination))
    source.rmdir()


def _retarget_tile_product(
    record: SceneRecord,
    target: str,
    *,
    apply: bool,
    actions: list[str],
) -> None:
    for subdir_name in ("tiles", "height_maps"):
        subdir = record.product_root / subdir_name
        target_dir = subdir / target
        for old in SPLITS:
            old_dir = subdir / old
            if old == target or not old_dir.exists():
                continue
            for file_path in sorted(path for path in old_dir.iterdir() if path.is_file()):
                _rename_file(
                    file_path,
                    file_path.with_name(_retarget_name(file_path.name, target)),
                    apply=apply,
                    actions=actions,
                )
            _merge_dir(old_dir, target_dir, apply=apply, actions=actions)
    for file_path in sorted(path for path in record.product_root.iterdir() if path.is_file()):
        new_name = _retarget_name(file_path.name, target)
        if new_name != file_path.name:
            _rename_file(
                file_path,
                file_path.with_name(new_name),
                apply=apply,
                actions=actions,
            )


def _retarget_benchmark_rasters(
    record: SceneRecord,
    target: str,
    data_root: Path,
    *,
    apply: bool,
    actions: list[str],
) -> None:
    target_dir = data_root / ("validation" if target == "valid" else "benchmark")
    other_dir = data_root / ("benchmark" if target == "valid" else "validation")
    for filename in record.benchmark_filenames:
        _move_file(other_dir / filename, target_dir / filename, apply=apply, actions=actions)


def _write_reference_manifest(
    valid_records: list[SceneRecord],
    reference_root: Path,
    description: str,
    *,
    apply: bool,
    actions: list[str],
) -> None:
    manifest_path = reference_root / "subset_manifest.json"
    selected_tiles: list[dict[str, str]] = []
    for record in sorted(valid_records, key=lambda item: item.scene_id):
        tile_dir = record.product_root / "tiles" / "valid"
        for tile_path in sorted(tile_dir.glob("*.tif")):
            selected_tiles.append(
                {"relative_path": os.path.relpath(tile_path, start=reference_root)}
            )
    payload = {
        "description": description,
        "num_scenes": len(valid_records),
        "scene_ids": [
            record.scene_id for record in sorted(valid_records, key=lambda item: item.scene_id)
        ],
        "selected_tiles": selected_tiles,
    }
    actions.append(f"write {manifest_path} ({len(selected_tiles)} tile references)")
    if apply:
        reference_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_split_manifest(
    records: list[SceneRecord],
    valid_ids: set[str],
    data_root: Path,
    split_manifest_stem: str,
    *,
    apply: bool,
    actions: list[str],
) -> None:
    metadata_dir = data_root / "metadata"
    json_path = metadata_dir / f"{split_manifest_stem}.json"
    csv_path = metadata_dir / f"{split_manifest_stem}.csv"
    rows = [
        {
            "scene_id": record.scene_id,
            "product_name": record.product_name,
            "split": "valid" if record.scene_id in valid_ids else "test",
            "stratum": record.stratum,
            "stats_source": record.stats.source,
            "mean_m": record.stats.mean,
            "p50_m": record.stats.p50,
            "p95_m": record.stats.p95,
            "p99_m": record.stats.p99,
            "max_m": record.stats.max,
            "frac_gt_2m": record.stats.frac_gt_2,
            "frac_gt_20m": record.stats.frac_gt_20,
            "frac_gt_30m": record.stats.frac_gt_30,
        }
        for record in sorted(records, key=lambda item: item.scene_id)
    ]
    actions.append(f"write {json_path}")
    actions.append(f"write {csv_path}")
    if not apply:
        return
    metadata_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"scenes": rows}, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _split_records(records: list[SceneRecord], valid_ids: set[str]) -> dict[str, list[SceneRecord]]:
    return {
        "valid": [record for record in records if record.scene_id in valid_ids],
        "test": [record for record in records if record.scene_id not in valid_ids],
    }


def _print_summary(
    records: list[SceneRecord], valid_ids: set[str], args: argparse.Namespace
) -> None:
    split_records = _split_records(records, valid_ids)
    print(f"dataset_preset: {args.dataset_preset}")
    print(f"data_root: {args.data_root}")
    print(f"tiles_root: {args.tiles_root / args.dataset_name}")
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"seed: {args.seed}")
    print(f"stats_source: {records[0].stats.source.split(':', 1)[0]}")
    print(f"scene_counts: valid={len(split_records['valid'])} test={len(split_records['test'])}")
    if args.force_test_scene_id:
        print(f"forced_test_scene_ids: {', '.join(args.force_test_scene_id)}")
    print("")
    print("split   count  mean   p50   p95   p99   max   >2m    >20m   >30m")
    for split in SPLITS:
        group = split_records[split]
        stats = [record.stats for record in group]
        frac_gt_2 = [stat.frac_gt_2 for stat in stats if stat.frac_gt_2 is not None]
        frac_gt_20 = [stat.frac_gt_20 for stat in stats if stat.frac_gt_20 is not None]
        frac_gt_30 = [stat.frac_gt_30 for stat in stats if stat.frac_gt_30 is not None]
        print(
            f"{split:<7} {len(group):>5} "
            f"{fmean(stat.mean for stat in stats):>5.1f} "
            f"{fmean(stat.p50 for stat in stats):>5.1f} "
            f"{fmean(stat.p95 for stat in stats):>5.1f} "
            f"{fmean(stat.p99 for stat in stats):>5.1f} "
            f"{max(stat.max for stat in stats):>5.1f} "
            f"{(fmean(frac_gt_2) if frac_gt_2 else float('nan')):>5.2f} "
            f"{(fmean(frac_gt_20) if frac_gt_20 else float('nan')):>6.2f} "
            f"{(fmean(frac_gt_30) if frac_gt_30 else float('nan')):>6.2f}"
        )
    print("")
    print("stratum  total  valid  test")
    total_counts = Counter(record.stratum for record in records)
    valid_counts = Counter(record.stratum for record in split_records["valid"])
    for stratum in sorted(total_counts):
        print(
            f"{stratum:<8} {total_counts[stratum]:>5} "
            f"{valid_counts[stratum]:>6} {total_counts[stratum] - valid_counts[stratum]:>5}"
        )
    print("")
    print("valid_scene_ids: " + ", ".join(record.scene_id for record in split_records["valid"]))
    print("test_scene_ids : " + ", ".join(record.scene_id for record in split_records["test"]))


def _resolve_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = DATASET_PRESETS[args.dataset_preset]
    if args.data_root is None:
        args.data_root = preset.root
    if args.tiles_root is None:
        args.tiles_root = preset.tiles_root
    if args.dataset_name is None:
        args.dataset_name = preset.dataset_name
    args.product_prefix = preset.product_prefix
    args.scene_chm_prefix = preset.scene_chm_prefix
    args.split_manifest_stem = preset.split_manifest_stem
    args.reference_description = preset.reference_description
    args.manifest_kind = preset.manifest_kind
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split height scenes into stable validation and benchmark/test subsets."
    )
    parser.add_argument(
        "--dataset-preset",
        choices=tuple(DATASET_PRESETS),
        default="france",
        help="Dataset defaults to use for roots, product prefixes, and metadata names.",
    )
    parser.add_argument(
        "--data-root",
        "--france-root",
        dest="data_root",
        type=Path,
        default=None,
        help=(
            "Raster root containing benchmark/ and validation/. Standard presets also "
            "expect manifest.csv, scenes/, and qa/; NEON discovers paired rasters "
            "directly from benchmark/ and validation/."
        ),
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=None,
        help="Root containing the tiled dataset folder. Defaults come from --dataset-preset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name of the tiled dataset directory. Defaults come from --dataset-preset.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.40)
    parser.add_argument("--valid-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-iterations", type=int, default=50000)
    parser.add_argument("--stats-source", choices=("auto", "raster", "qa"), default="auto")
    parser.add_argument("--sample-max-dim", type=int, default=1200)
    parser.add_argument("--min-height-m", type=float, default=0.0)
    parser.add_argument("--max-height-m", type=float, default=60.0)
    parser.add_argument("--field-canopy-fraction", type=float, default=0.08)
    parser.add_argument("--field-p95-m", type=float, default=3.0)
    parser.add_argument("--field-mean-m", type=float, default=1.5)
    parser.add_argument("--low-p95-m", type=float, default=18.0)
    parser.add_argument("--tall-p99-m", type=float, default=35.0)
    parser.add_argument("--tall-max-m", type=float, default=35.0)
    parser.add_argument("--balance-strata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-test-scene-id",
        action="append",
        default=None,
        help="Scene id forced into test. May be passed multiple times.",
    )
    parser.add_argument("--force-valid-scene-id", action="append", default=[])
    parser.add_argument(
        "--no-default-forced-test-scenes",
        action="store_true",
        help="Do not use any built-in forced-test scene ids, if configured.",
    )
    parser.add_argument("--reference-manifest-dir", default="valid_reference_images")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-actions", action="store_true")
    args = parser.parse_args()
    args = _resolve_dataset_args(args)
    if args.force_test_scene_id is None:
        args.force_test_scene_id = (
            [] if args.no_default_forced_test_scenes else list(DEFAULT_FORCED_TEST_SCENES)
        )
    args.force_test_scene_id = [
        _normalise_scene_id(value, args) for value in args.force_test_scene_id
    ]
    args.force_valid_scene_id = [
        _normalise_scene_id(value, args) for value in args.force_valid_scene_id
    ]
    if not 0.0 < args.valid_ratio < 1.0:
        raise SystemExit("--valid-ratio must be between 0 and 1")
    if args.search_iterations <= 0:
        raise SystemExit("--search-iterations must be positive")
    if args.sample_max_dim <= 0:
        raise SystemExit("--sample-max-dim must be positive")
    if args.manifest_kind == "neon_rasters" and args.stats_source == "qa":
        raise SystemExit(
            "NEON splitting is raster-discovered and requires --stats-source auto or raster"
        )
    return args


def main() -> None:
    args = _parse_args()
    records = _load_records(args)
    valid_ids = _choose_valid_ids(records, args)
    _print_summary(records, valid_ids, args)

    split_records = _split_records(records, valid_ids)
    actions: list[str] = []
    for split, group in split_records.items():
        for record in group:
            _retarget_tile_product(record, split, apply=args.apply, actions=actions)
            _retarget_benchmark_rasters(
                record,
                split,
                args.data_root,
                apply=args.apply,
                actions=actions,
            )

    reference_root = args.tiles_root / args.dataset_name / args.reference_manifest_dir
    _write_reference_manifest(
        split_records["valid"],
        reference_root,
        args.reference_description,
        apply=args.apply,
        actions=actions,
    )
    _write_split_manifest(
        records,
        valid_ids,
        args.data_root,
        args.split_manifest_stem,
        apply=args.apply,
        actions=actions,
    )

    print("")
    if args.show_actions:
        for action in actions:
            print(action)
    verb = "applied" if args.apply else "dry_run"
    print(f"{verb}: {len(actions)} planned update(s).")
    if not args.apply:
        print("Re-run with --apply to move files and write split/reference manifests.")


if __name__ == "__main__":
    main()
