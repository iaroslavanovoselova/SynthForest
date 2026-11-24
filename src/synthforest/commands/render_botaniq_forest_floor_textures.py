#!/usr/bin/env python3
"""Bake Botaniq particle presets into ordinary terrain textures.

Run through the documented Blender launcher with ``SYNTHFOREST_BLENDER_BIN``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import bpy
from mathutils import Vector

from synthforest.paths import asset_path, project_root

LOGGER = logging.getLogger(__name__)


DEFAULT_PARTICLE_CATEGORIES: dict[str, str] = {
    "forest-floor": "blends/particles/forest-floor",
    "grass": "blends/particles/grass",
    "misc": "blends/particles/misc",
    "moss": "blends/particles/mosses-and-lichens",
    "rocks": "blends/particles/rocks",
    "shrubs": "blends/particles/shrubs",
    "weeds": "blends/particles/weed",
}


def _script_argv() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _repo_root() -> Path:
    return project_root()


def _default_log_dir() -> Path:
    return _repo_root() / "logs"


def _configure_logging(log_dir: Path) -> Path:
    resolved_log_dir = log_dir.expanduser()
    if not resolved_log_dir.is_absolute():
        resolved_log_dir = (_repo_root() / resolved_log_dir).resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    log_path = resolved_log_dir / (
        f"render_botaniq_forest_floor_textures_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return log_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Botaniq particle .blend presets to RGB texture PNGs."
    )
    parser.add_argument(
        "--pack-dir",
        default=None,
        help="Botaniq full pack root (defaults to $SYNTHFOREST_ASSET_ROOT/botaniq_full).",
    )
    parser.add_argument(
        "--category",
        choices=sorted(DEFAULT_PARTICLE_CATEGORIES),
        action="append",
        default=[],
        help="Particle category to render. Can be repeated. Defaults to all supported categories.",
    )
    parser.add_argument(
        "--category-dir",
        action="append",
        default=[],
        help=(
            "Override category scanning with a preset directory inside the Botaniq pack. "
            "Can be repeated. When omitted, all supported particle categories are rendered."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder for texture PNGs (defaults to "
            "$SYNTHFOREST_ASSET_ROOT/terrain_textures/botaniq_particles_5cm)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Folder where this script writes its run log file.",
    )
    parser.add_argument(
        "--image-size",
        "--resolution",
        dest="image_size",
        type=int,
        default=1024,
        help=(
            "Final square image size in pixels. The scene width/height is derived from "
            "this value and --gsd."
        ),
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=0.05,
        help="Ground sampling distance in meters per pixel.",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--normal-factor",
        type=float,
        default=0.6,
        help="Minimum particle normal offset while baking onto the visible floor plane.",
    )
    parser.add_argument(
        "--sun-energy",
        type=float,
        default=3.0,
        help="Strength of the directional sun used for texture renders.",
    )
    parser.add_argument(
        "--sun-elevation-deg",
        type=float,
        default=55.0,
        help="Sun elevation angle above the horizon in degrees.",
    )
    parser.add_argument(
        "--sun-azimuth-deg",
        type=float,
        default=135.0,
        help="Sun azimuth in degrees, measured around the XY plane.",
    )
    parser.add_argument(
        "--fill-energy",
        type=float,
        default=1600.0,
        help="Strength of the large overhead area fill light.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.4,
        help="Color-management exposure applied to the baked RGB texture.",
    )
    parser.add_argument(
        "--show-emitter-surfaces",
        action="store_true",
        help=(
            "Render each Botaniq particle emitter surface. By default the script hides "
            "them and renders one continuous floor plane to avoid baked tile seams."
        ),
    )
    parser.add_argument(
        "--floor-plane-padding",
        type=float,
        default=1.04,
        help="Scale multiplier for the continuous bake floor plane under the particles.",
    )
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a preset fails instead of recording the error in the manifest.",
    )
    return parser.parse_args(_script_argv())


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return re.sub(r"_+", "_", slug).strip("_")


def _matches(path: Path, include: list[str], exclude: list[str]) -> bool:
    name = path.stem.lower()
    include_tokens = [item.lower() for item in include if item.strip()]
    exclude_tokens = [item.lower() for item in exclude if item.strip()]
    if include_tokens and not any(token in name for token in include_tokens):
        return False
    return not any(token in name for token in exclude_tokens)


def _find_emitter() -> bpy.types.Object:
    emitters = [
        obj for obj in bpy.data.objects if obj.type == "MESH" and len(obj.particle_systems) > 0
    ]
    if not emitters:
        raise RuntimeError("No particle emitter mesh found in Botaniq particle preset.")
    emitters.sort(key=lambda obj: (not obj.name.startswith("bq_pps_"), obj.name))
    return emitters[0]


def _world_bbox_center(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / len(corners)


def _configure_emitter(
    emitter: bpy.types.Object,
    *,
    normal_factor: float,
    seed_offset: int,
    show_instancer: bool,
) -> None:
    emitter.hide_render = False
    emitter.hide_viewport = False
    emitter.show_instancer_for_render = bool(show_instancer)
    emitter.show_instancer_for_viewport = bool(show_instancer)
    for ps_index, ps in enumerate(emitter.particle_systems):
        if hasattr(ps.settings, "normal_factor"):
            ps.settings.normal_factor = max(float(ps.settings.normal_factor), normal_factor)
        if hasattr(ps, "seed"):
            ps.seed = max(1, (int(ps.seed) + int(seed_offset) + ps_index * 1009) % 2_000_000_000)


def _odd_at_least(value: float, minimum: int = 1) -> int:
    repeat = max(minimum, int(math.ceil(value)))
    if repeat % 2 == 0:
        repeat += 1
    return repeat


def _tile_emitter_grid(
    emitter: bpy.types.Object,
    *,
    scene_size: float,
    tile_size: float,
    normal_factor: float,
    show_instancer: bool,
) -> list[bpy.types.Object]:
    repeat = _odd_at_least(scene_size / tile_size)
    if repeat == 1:
        _configure_emitter(
            emitter,
            normal_factor=normal_factor,
            seed_offset=0,
            show_instancer=show_instancer,
        )
        return [emitter]

    emitters = [emitter]
    center_index = repeat // 2
    original_location = emitter.location.copy()
    for ix in range(repeat):
        for iy in range(repeat):
            if ix == center_index and iy == center_index:
                tile = emitter
            else:
                tile = emitter.copy()
                tile.data = emitter.data.copy()
                bpy.context.collection.objects.link(tile)
                emitters.append(tile)
            tile.location = (
                original_location.x + (ix - center_index) * tile_size,
                original_location.y + (iy - center_index) * tile_size,
                original_location.z,
            )
            _configure_emitter(
                tile,
                normal_factor=normal_factor,
                seed_offset=(ix * repeat + iy) * 104_729,
                show_instancer=show_instancer,
            )
    return emitters


def _create_continuous_floor_plane(
    *,
    emitter: bpy.types.Object,
    center: Vector,
    view_size: float,
    padding: float,
) -> bpy.types.Object:
    size = max(1.0, float(view_size) * max(1.0, float(padding)))
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(float(center.x), float(center.y), float(center.z) - 0.03),
    )
    floor = bpy.context.active_object
    floor.name = "BotaniqBakeContinuousFloor"
    floor.hide_render = False
    floor.hide_viewport = False

    material = emitter.active_material
    if material is None and emitter.data.materials:
        material = emitter.data.materials[0]
    if material is not None:
        floor.data.materials.append(material)
    else:
        mat = bpy.data.materials.new("BotaniqBakeContinuousFloor_Material")
        mat.diffuse_color = (0.34, 0.30, 0.22, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.34, 0.30, 0.22, 1.0)
            bsdf.inputs["Roughness"].default_value = 1.0
        floor.data.materials.append(mat)
    return floor


def _sun_rotation(elevation_deg: float, azimuth_deg: float):
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    incoming_direction = Vector(
        (
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        )
    )
    return incoming_direction.to_track_quat("-Z", "Y").to_euler()


def _prepare_scene(
    *,
    emitter: bpy.types.Object,
    image_size: int,
    gsd: float,
    samples: int,
    normal_factor: float,
    sun_energy: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    fill_energy: float,
    exposure: float,
    show_emitter_surfaces: bool,
    floor_plane_padding: float,
    output_path: Path,
) -> tuple[int, float]:
    scene = bpy.context.scene

    for obj in bpy.data.objects:
        obj.hide_render = False
        obj.hide_viewport = False

    center = _world_bbox_center(emitter)
    tile_size = max(float(emitter.dimensions.x), float(emitter.dimensions.y), 1.0)
    scene_size = max(1.0, float(image_size) * float(gsd))
    effective_tile_repeat = _odd_at_least(scene_size / tile_size)
    LOGGER.info(
        f"Scene size: {scene_size:.2f} m x {scene_size:.2f} m from "
        f"{image_size} px @ {gsd:.4f} m/px."
    )
    LOGGER.info(
        "Emitter tile repeat selected: %s x %s", effective_tile_repeat, effective_tile_repeat
    )
    emitters = _tile_emitter_grid(
        emitter,
        scene_size=scene_size,
        tile_size=tile_size,
        normal_factor=normal_factor,
        show_instancer=show_emitter_surfaces,
    )
    repeat_used = int(round(len(emitters) ** 0.5))
    camera_height = max(8.0, scene_size * 1.2)

    if not show_emitter_surfaces:
        _create_continuous_floor_plane(
            emitter=emitter,
            center=center,
            view_size=scene_size,
            padding=floor_plane_padding,
        )

    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"} and obj.name.startswith("BotaniqBake"):
            bpy.data.objects.remove(obj, do_unlink=True)

    camera_data = bpy.data.cameras.new("BotaniqBakeCamera")
    camera = bpy.data.objects.new("BotaniqBakeCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (center.x, center.y, center.z + camera_height)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = scene_size
    scene.camera = camera

    light_data = bpy.data.lights.new("BotaniqBakeAreaLight", "AREA")
    light = bpy.data.objects.new("BotaniqBakeAreaLight", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (center.x, center.y, center.z + camera_height * 0.75)
    light_data.energy = max(0.0, float(fill_energy))
    light_data.size = max(scene_size * 0.85, tile_size * 2.0)
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = False

    sun_data = bpy.data.lights.new("BotaniqBakeSun", "SUN")
    sun = bpy.data.objects.new("BotaniqBakeSun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.location = (center.x, center.y, center.z + camera_height)
    sun.rotation_euler = _sun_rotation(sun_elevation_deg, sun_azimuth_deg)
    sun_data.energy = max(0.0, float(sun_energy))
    sun_data.angle = math.radians(4.0)

    if scene.world is not None:
        scene.world.color = (0.9, 0.9, 0.88)
    else:
        scene.world = bpy.data.worlds.new("BotaniqBakeWorld")
        scene.world.color = (0.9, 0.9, 0.88)

    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.render.resolution_x = max(64, int(image_size))
    scene.render.resolution_y = max(64, int(image_size))
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_path)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(exposure)
    scene.view_settings.gamma = 1.0
    bpy.context.view_layer.update()
    return repeat_used, scene_size


def _render_preset(
    *,
    category: str,
    blend_path: Path,
    output_path: Path,
    image_size: int,
    gsd: float,
    samples: int,
    normal_factor: float,
    sun_energy: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    fill_energy: float,
    exposure: float,
    show_emitter_surfaces: bool,
    floor_plane_padding: float,
) -> dict[str, object]:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    emitter = _find_emitter()
    tile_repeat_used, scene_size_m = _prepare_scene(
        emitter=emitter,
        image_size=image_size,
        gsd=gsd,
        samples=samples,
        normal_factor=normal_factor,
        sun_energy=sun_energy,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        fill_energy=fill_energy,
        exposure=exposure,
        show_emitter_surfaces=show_emitter_surfaces,
        floor_plane_padding=floor_plane_padding,
        output_path=output_path,
    )
    bpy.ops.render.render(write_still=True)
    return {
        "preset": blend_path.stem,
        "category": category,
        "source_blend": str(blend_path),
        "texture": str(output_path),
        "image_size_px": int(image_size),
        "gsd_m_per_px": float(gsd),
        "emitter_size_m": max(float(emitter.dimensions.x), float(emitter.dimensions.y)),
        "tile_repeat": int(tile_repeat_used),
        "scene_size_m": float(scene_size_m),
        "show_emitter_surfaces": bool(show_emitter_surfaces),
        "particle_systems": [
            {
                "name": ps.name,
                "count": int(ps.settings.count),
                "render_type": str(ps.settings.render_type),
            }
            for ps in emitter.particle_systems
        ],
    }


def _resolve_category_dirs(pack_dir: Path, args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.category_dir:
        resolved: list[tuple[str, Path]] = []
        for raw_dir in args.category_dir:
            category_dir = Path(raw_dir).expanduser()
            if not category_dir.is_absolute():
                category_dir = pack_dir / category_dir
            resolved.append((_slug(category_dir.name), category_dir.resolve()))
        return resolved

    categories = args.category or list(DEFAULT_PARTICLE_CATEGORIES)
    return [
        (category, (pack_dir / DEFAULT_PARTICLE_CATEGORIES[category]).resolve())
        for category in categories
    ]


def _collect_presets(
    *,
    category_dirs: list[tuple[str, Path]],
    include: list[str],
    exclude: list[str],
) -> list[tuple[str, Path]]:
    presets: list[tuple[str, Path]] = []
    for category, preset_dir in category_dirs:
        if not preset_dir.exists():
            LOGGER.warning("Category directory does not exist, skipping: %s", preset_dir)
            continue
        presets.extend(
            (category, path)
            for path in sorted(preset_dir.glob("*.blend"))
            if _matches(path, include, exclude)
        )
    return presets


def main() -> None:
    args = _parse_args()
    log_path = _configure_logging(Path(args.log_dir))
    LOGGER.info("Writing run log to %s", log_path)

    configured_pack_dir = args.pack_dir
    if configured_pack_dir is None:
        configured_pack_dir = asset_path("botaniq_full")
    pack_dir = Path(configured_pack_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else asset_path("terrain_textures/botaniq_particles_5cm")
    )
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    category_dirs = _resolve_category_dirs(pack_dir, args)
    presets = _collect_presets(
        category_dirs=category_dirs,
        include=args.include,
        exclude=args.exclude,
    )
    if args.limit > 0:
        presets = presets[: args.limit]
    if not presets:
        searched = ", ".join(str(path) for _, path in category_dirs)
        raise SystemExit(f"No Botaniq particle presets found in: {searched}")

    manifest: list[dict[str, object]] = []
    for index, (category, blend_path) in enumerate(presets, start=1):
        output_path = output_dir / f"{_slug(category)}__{_slug(blend_path.stem)}.png"
        if output_path.exists() and not args.force:
            LOGGER.info("[%s/%s] exists, skipping: %s", index, len(presets), output_path)
            manifest.append(
                {
                    "preset": blend_path.stem,
                    "category": category,
                    "source_blend": str(blend_path),
                    "texture": str(output_path),
                    "skipped_existing": True,
                }
            )
            continue
        LOGGER.info(
            "[%s/%s] rendering %s/%s -> %s",
            index,
            len(presets),
            category,
            blend_path.name,
            output_path,
        )
        try:
            manifest.append(
                _render_preset(
                    category=category,
                    blend_path=blend_path,
                    output_path=output_path,
                    image_size=args.image_size,
                    gsd=args.gsd,
                    samples=args.samples,
                    normal_factor=args.normal_factor,
                    sun_energy=args.sun_energy,
                    sun_elevation_deg=args.sun_elevation_deg,
                    sun_azimuth_deg=args.sun_azimuth_deg,
                    fill_energy=args.fill_energy,
                    exposure=args.exposure,
                    show_emitter_surfaces=args.show_emitter_surfaces,
                    floor_plane_padding=args.floor_plane_padding,
                )
            )
        except Exception as exc:
            if args.stop_on_error:
                raise
            LOGGER.exception("Failed to render %s", blend_path)
            manifest.append(
                {
                    "preset": blend_path.stem,
                    "category": category,
                    "source_blend": str(blend_path),
                    "texture": str(output_path),
                    "error": str(exc),
                }
            )

    manifest_path = output_dir / "botaniq_particle_textures.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
