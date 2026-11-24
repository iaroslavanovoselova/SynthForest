#!/usr/bin/env python3
"""Move tiled rasters between train/valid splits at whole-scene granularity.

The converter can create per-tile AOI splits. This helper keeps each raster scene
intact by assigning the full scene directory to either train or valid, then
renaming split-specific tile folders, tile files, COCO JSON files, and split
tokens inside COCO JSON strings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthforest.paths import config_path, project_root

PROJECT_ROOT = project_root()
DEFAULT_CONFIG = config_path("generation/convert_scenes_to_tiles")
SPLITS = ("train", "valid")
UNKNOWN_DENSITY = "unknown"


@dataclass(frozen=True)
class SceneRecord:
    stem: str
    directory: Path
    density_band: str
    scene_id: int | None


def _strip_inline_comment(value: str) -> str:
    if "#" in value:
        value = value.split("#", 1)[0]
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    return value


def _load_simple_top_level_config(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[0].isspace() or ":" not in raw_line:
            continue

        key, value = raw_line.split(":", 1)
        value = _strip_inline_comment(value)
        if value and value.lower() not in {"null", "none"}:
            values[key.strip()] = value
    return values


def _resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _default_tiles_root(args: argparse.Namespace) -> Path:
    if args.tiles_root is not None:
        return _resolve_repo_path(args.tiles_root)

    config = _load_simple_top_level_config(_resolve_repo_path(args.config))
    output_root = config.get("output_root")
    product_name = config.get("product_name")
    if not output_root or not product_name:
        raise ValueError(
            "Could not infer --tiles-root. Pass it explicitly or set output_root "
            "and product_name in the tiling config."
        )
    return (_resolve_repo_path(output_root) / product_name).resolve()


def _default_run_summary(args: argparse.Namespace, tiles_root: Path) -> Path:
    if args.run_summary is not None:
        return _resolve_repo_path(args.run_summary)

    config = _load_simple_top_level_config(_resolve_repo_path(args.config))
    images_dir = config.get("images_dir")
    if images_dir:
        return (_resolve_repo_path(images_dir).parent / "run_summary.json").resolve()

    return (tiles_root.parent.parent / tiles_root.name / "run_summary.json").resolve()


def _scene_id_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    return int(match.group(1))


def _summary_scene_stem(scene: dict[str, Any]) -> str | None:
    image = scene.get("image")
    if isinstance(image, dict) and image.get("path"):
        return Path(str(image["path"])).stem

    scene_id = scene.get("scene_id")
    if isinstance(scene_id, int):
        return f"rgb_satellite{scene_id:04d}"
    return None


def _summary_density(scene: dict[str, Any]) -> str:
    density = scene.get("density")
    if isinstance(density, dict):
        band = str(density.get("band", "")).strip()
        if band:
            return band
    return UNKNOWN_DENSITY


def _load_summary_metadata(summary_path: Path) -> dict[str, tuple[str, int | None]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"Run summary has no scenes list: {summary_path}")

    metadata: dict[str, tuple[str, int | None]] = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        stem = _summary_scene_stem(scene)
        if stem is None:
            continue
        scene_id = scene.get("scene_id")
        metadata[stem] = (
            _summary_density(scene),
            scene_id if isinstance(scene_id, int) else _scene_id_from_stem(stem),
        )
    return metadata


def _load_scene_records(
    tiles_root: Path, summary_path: Path | None
) -> tuple[list[SceneRecord], list[str]]:
    if not tiles_root.exists():
        raise FileNotFoundError(f"Tiles root not found: {tiles_root}")

    metadata = _load_summary_metadata(summary_path) if summary_path is not None else {}
    warnings: list[str] = []
    records: list[SceneRecord] = []

    for scene_dir in sorted(path for path in tiles_root.iterdir() if path.is_dir()):
        density_band, scene_id = metadata.get(
            scene_dir.name,
            (UNKNOWN_DENSITY, _scene_id_from_stem(scene_dir.name)),
        )
        if summary_path is not None and scene_dir.name not in metadata:
            warnings.append(f"No run-summary metadata for {scene_dir.name}; using density=unknown.")
        records.append(
            SceneRecord(
                stem=scene_dir.name,
                directory=scene_dir,
                density_band=density_band,
                scene_id=scene_id,
            )
        )

    if not records:
        raise ValueError(f"No scene directories found in: {tiles_root}")
    return records, warnings


def _target_valid_count(total: int, val_ratio: float) -> int:
    target = int(round(total * val_ratio))
    if 0.0 < val_ratio < 1.0 and total > 1:
        target = max(1, min(total - 1, target))
    return target


def _valid_counts_by_density(
    groups: dict[str, list[SceneRecord]],
    val_ratio: float,
    ensure_each_density: bool,
) -> dict[str, int]:
    bands = sorted(groups)
    counts = {band: len(groups[band]) for band in bands}
    exact = {band: counts[band] * val_ratio for band in bands}

    mins = {band: 0 for band in bands}
    maxs = {band: counts[band] for band in bands}
    if ensure_each_density and 0.0 < val_ratio < 1.0:
        for band in bands:
            if counts[band] > 1:
                mins[band] = 1
                maxs[band] = counts[band] - 1

    target = _target_valid_count(sum(counts.values()), val_ratio)
    target = max(target, sum(mins.values()))
    target = min(target, sum(maxs.values()))

    allocated = {
        band: max(mins[band], min(maxs[band], int(math.floor(exact[band])))) for band in bands
    }

    def allocated_total() -> int:
        return sum(allocated.values())

    while allocated_total() < target:
        candidates = [band for band in bands if allocated[band] < maxs[band]]
        if not candidates:
            break
        band = max(
            candidates,
            key=lambda item: (exact[item] - allocated[item], counts[item], item),
        )
        allocated[band] += 1

    while allocated_total() > target:
        candidates = [band for band in bands if allocated[band] > mins[band]]
        if not candidates:
            break
        band = min(
            candidates,
            key=lambda item: (exact[item] - (allocated[item] - 1), counts[item], item),
        )
        allocated[band] -= 1

    return allocated


def _choose_balanced_valid_scenes(
    records: list[SceneRecord],
    val_ratio: float,
    seed: int,
    ensure_each_density: bool,
) -> set[str]:
    groups: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in records:
        groups[record.density_band].append(record)

    target_counts = _valid_counts_by_density(groups, val_ratio, ensure_each_density)
    rng = random.Random(seed)
    valid_stems: set[str] = set()

    for band in sorted(groups):
        band_records = sorted(
            groups[band],
            key=lambda item: (item.scene_id is None, item.scene_id, item.stem),
        )
        rng.shuffle(band_records)
        valid_stems.update(record.stem for record in band_records[: target_counts[band]])
    return valid_stems


def _choose_random_valid_scenes(
    records: list[SceneRecord],
    val_ratio: float,
    seed: int,
) -> set[str]:
    target_count = _target_valid_count(len(records), val_ratio)
    shuffled = sorted(records, key=lambda item: (item.scene_id is None, item.scene_id, item.stem))
    random.Random(seed).shuffle(shuffled)
    return {record.stem for record in shuffled[:target_count]}


def _retarget_split_text(value: str, target_split: str) -> str:
    result = value
    for old_split in SPLITS:
        if old_split == target_split:
            continue

        result = result.replace(f"_{old_split}_", f"_{target_split}_")
        result = result.replace(f"_{old_split}.", f"_{target_split}.")
        result = re.sub(rf"(?<=_){re.escape(old_split)}$", target_split, result)
        result = result.replace(f"/{old_split}/", f"/{target_split}/")
        result = result.replace(f"\\{old_split}\\", f"\\{target_split}\\")
        result = re.sub(rf"\bfold {re.escape(old_split)}\b", f"fold {target_split}", result)
        if result == old_split:
            result = target_split
    return result


def _retarget_json_strings(value: Any, target_split: str) -> Any:
    if isinstance(value, str):
        return _retarget_split_text(value, target_split)
    if isinstance(value, list):
        return [_retarget_json_strings(item, target_split) for item in value]
    if isinstance(value, dict):
        return {key: _retarget_json_strings(item, target_split) for key, item in value.items()}
    return value


def _merge_directory(source: Path, destination: Path, dry_run: bool, actions: list[str]) -> None:
    if destination == source:
        return
    if not source.exists():
        return
    if not destination.exists():
        actions.append(f"rename {source} -> {destination}")
        if not dry_run:
            source.rename(destination)
        return
    if not destination.is_dir():
        raise FileExistsError(f"Refusing to merge directory into non-directory: {destination}")

    for child in sorted(source.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
        child_destination = destination / child.name
        if child_destination.exists():
            if child.is_dir() and child_destination.is_dir():
                _merge_directory(child, child_destination, dry_run, actions)
            else:
                raise FileExistsError(f"Refusing to overwrite existing path: {child_destination}")
        else:
            actions.append(f"rename {child} -> {child_destination}")
            if not dry_run:
                child.rename(child_destination)

    actions.append(f"remove empty dir {source}")
    if not dry_run:
        source.rmdir()


def _rename_path(path: Path, destination: Path, dry_run: bool, actions: list[str]) -> None:
    if destination == path:
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    actions.append(f"rename {path} -> {destination}")
    if not dry_run:
        path.rename(destination)


def _apply_scene_split(record: SceneRecord, target_split: str, dry_run: bool) -> list[str]:
    scene_dir = record.directory
    actions: list[str] = []

    file_paths = sorted(
        (path for path in scene_dir.rglob("*") if path.is_file()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in file_paths:
        new_name = _retarget_split_text(path.name, target_split)
        if new_name != path.name:
            _rename_path(path, path.with_name(new_name), dry_run, actions)

    dir_paths = sorted(
        (path for path in scene_dir.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in dir_paths:
        if path.name in SPLITS and path.name != target_split:
            _merge_directory(path, path.with_name(target_split), dry_run, actions)

    for json_path in sorted(scene_dir.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        updated = _retarget_json_strings(data, target_split)
        if updated != data:
            actions.append(f"rewrite {json_path}")
            if not dry_run:
                json_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")

    return actions


def _split_summary(records: list[SceneRecord], valid_stems: set[str]) -> dict[str, Counter[str]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = "valid" if record.stem in valid_stems else "train"
        summary[record.density_band][split] += 1
    return dict(summary)


def _print_summary(
    records: list[SceneRecord],
    valid_stems: set[str],
    split_mode: str,
    val_ratio: float,
    seed: int,
    tiles_root: Path,
    run_summary: Path | None,
) -> None:
    split_counts = Counter("valid" if record.stem in valid_stems else "train" for record in records)
    density_summary = _split_summary(records, valid_stems)

    print(f"tiles_root: {tiles_root}")
    print(f"split_mode: {split_mode}")
    print(f"run_summary: {run_summary if run_summary is not None else 'none'}")
    print(f"seed: {seed}")
    print(f"requested_valid_ratio: {val_ratio:.3f}")
    print(f"scene_counts: train={split_counts['train']} valid={split_counts['valid']}")
    print("")
    print("density_band        total  train  valid")
    for band in sorted(density_summary):
        counts = density_summary[band]
        total = counts["train"] + counts["valid"]
        print(f"{band:<18} {total:>5} {counts['train']:>6} {counts['valid']:>6}")

    valid_records = sorted(
        (record for record in records if record.stem in valid_stems),
        key=lambda item: (item.density_band, item.scene_id is None, item.scene_id, item.stem),
    )
    valid_labels = ", ".join(f"{record.stem}({record.density_band})" for record in valid_records)
    print("")
    print(f"valid_scenes: {valid_labels}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a scene-level train/valid split from an already tiled dataset. "
            "Without --apply this only prints the planned split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Tiling config used to infer default paths.",
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=None,
        help="Product tile root, e.g. outputs/tiles/<product_name>.",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help="Scene generator run_summary.json used for balanced density-band stratification.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("balanced", "random"),
        default="balanced",
        help=(
            "Scene selection mode. balanced uses run-summary density bands; random "
            "does not require a run summary and applies only --val-ratio."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        "--valid-ratio",
        dest="val_ratio",
        type=float,
        default=0.20,
        help="Target fraction of scenes assigned to valid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic selection seed.")
    parser.add_argument(
        "--strict-ratio",
        action="store_true",
        help=(
            "Balanced mode only: do not force at least one validation and one "
            "training scene per density band."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rename files/directories and rewrite COCO JSON strings in place.",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="Print every filesystem/JSON action.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.val_ratio <= 1.0:
        raise SystemExit("--val-ratio must be strictly between 0 and 1.")

    tiles_root = _default_tiles_root(args)
    run_summary = _default_run_summary(args, tiles_root) if args.split_mode == "balanced" else None
    records, warnings = _load_scene_records(tiles_root, run_summary)
    if args.split_mode == "balanced":
        valid_stems = _choose_balanced_valid_scenes(
            records=records,
            val_ratio=args.val_ratio,
            seed=args.seed,
            ensure_each_density=not args.strict_ratio,
        )
    else:
        valid_stems = _choose_random_valid_scenes(
            records=records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    _print_summary(
        records,
        valid_stems,
        args.split_mode,
        args.val_ratio,
        args.seed,
        tiles_root,
        run_summary,
    )
    if warnings:
        print("")
        for warning in warnings:
            print(f"warning: {warning}")

    dry_run = not args.apply
    all_actions: list[str] = []
    for record in records:
        target_split = "valid" if record.stem in valid_stems else "train"
        all_actions.extend(_apply_scene_split(record, target_split, dry_run=dry_run))

    print("")
    if args.show_actions:
        for action in all_actions:
            print(action)
    if dry_run:
        print(
            f"dry_run: {len(all_actions)} updates would be made. "
            "Re-run with --apply to modify files."
        )
    else:
        print(f"applied: {len(all_actions)} updates.")


if __name__ == "__main__":
    main()
