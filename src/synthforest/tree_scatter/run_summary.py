from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_SUMMARY_FILENAME = "run_summary.json"
RUN_SUMMARY_SCHEMA_VERSION = 1


def _increment(counts: dict[str, int], key: object, amount: int = 1) -> None:
    normalized = str(key).strip()
    if not normalized:
        return
    counts[normalized] = counts.get(normalized, 0) + int(amount)


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_aggregates(
    scenes: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    forest_types: dict[str, int] = {}
    target_profiles: dict[str, int] = {}
    seasons: dict[str, int] = {}
    density_bands: dict[str, int] = {}
    context_profiles: dict[str, int] = {}
    context_modes: dict[str, int] = {}
    species_scene_counts: dict[str, int] = {}
    species_visible_tree_counts: dict[str, int] = {}
    total_context_counts: dict[str, int] = {}
    visible_tree_count = 0
    scattered_tree_count = 0
    scenes_with_visible_tree_count = 0
    scenes_with_scattered_tree_count = 0

    for scene in scenes:
        _increment(forest_types, scene.get("forest_type"))

        target_profile = _as_mapping(scene.get("target_profile"))
        _increment(target_profiles, target_profile.get("name"))

        season = _as_mapping(scene.get("season"))
        _increment(seasons, season.get("requested"))

        density = _as_mapping(scene.get("density"))
        _increment(density_bands, density.get("band"))

        context = _as_mapping(scene.get("context"))
        _increment(context_profiles, context.get("profile"))
        _increment(context_modes, context.get("mode"))
        for name, count in _as_mapping(context.get("counts")).items():
            _increment(total_context_counts, name, _as_int(count))

        tree_counts = _as_mapping(scene.get("tree_counts"))
        if tree_counts.get("visible") is not None:
            visible_tree_count += _as_int(tree_counts.get("visible"))
            scenes_with_visible_tree_count += 1
        if tree_counts.get("scattered") is not None:
            scattered_tree_count += _as_int(tree_counts.get("scattered"))
            scenes_with_scattered_tree_count += 1

        species = _as_mapping(scene.get("species"))
        if species.get("visible_tree_counts") is not None:
            visible_counts = _as_mapping(species.get("visible_tree_counts"))
            for species_name, count in visible_counts.items():
                _increment(species_visible_tree_counts, species_name, _as_int(count))
                _increment(species_scene_counts, species_name)

    totals: dict[str, object] = {"context_counts": total_context_counts}
    if scenes_with_visible_tree_count:
        totals["visible_tree_count"] = visible_tree_count
        totals["scenes_with_visible_tree_count"] = scenes_with_visible_tree_count
    if scenes_with_scattered_tree_count:
        totals["scattered_tree_count"] = scattered_tree_count
        totals["scenes_with_scattered_tree_count"] = scenes_with_scattered_tree_count

    distributions = {
        "forest_types": forest_types,
        "target_profiles": target_profiles,
        "seasons_requested": seasons,
        "density_bands": density_bands,
        "terrain_context_profiles": context_profiles,
        "terrain_context_modes": context_modes,
    }
    if species_scene_counts:
        distributions["species_scene_counts"] = species_scene_counts
    if species_visible_tree_counts:
        distributions["species_visible_tree_counts"] = species_visible_tree_counts
    return totals, distributions


def _load_existing_scenes(summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read existing run summary: {summary_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Existing run summary must contain a JSON object: {summary_path}")
    if payload.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run summary schema version in {summary_path}: "
            f"{payload.get('schema_version')!r}"
        )

    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not all(isinstance(scene, dict) for scene in scenes):
        raise ValueError(f"Existing run summary has an invalid scenes list: {summary_path}")
    return scenes


def _write_json_atomic(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def update_run_summary(
    root_output_dir: Path,
    scene_overview: dict[str, object],
) -> Path:
    scene_id = _as_int(scene_overview.get("scene_id"))
    if scene_id < 1:
        raise ValueError("scene_overview.scene_id must be a positive integer")

    summary_path = Path(root_output_dir) / RUN_SUMMARY_FILENAME
    scenes = [
        scene
        for scene in _load_existing_scenes(summary_path)
        if _as_int(scene.get("scene_id")) != scene_id
    ]
    scenes.append(scene_overview)
    scenes.sort(key=lambda scene: _as_int(scene.get("scene_id")))

    totals, distributions = _build_aggregates(scenes)
    payload = {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "scene_count": len(scenes),
        "totals": totals,
        "distributions": distributions,
        "scenes": scenes,
    }
    _write_json_atomic(summary_path, payload)
    return summary_path
