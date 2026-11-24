from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

import bpy

from ..config import TerrainConfig
from .geometry import distance_to_paths
from .resources import build_building_material_pair, collect_building_textures, context_value
from .rules import scale_count_range

logger = logging.getLogger(__name__)


@dataclass
class BuildingFootprint:
    cx: float
    cy: float
    half_w: float
    half_d: float
    yaw: float = 0.0


def point_in_oriented_rect(
    px: float,
    py: float,
    center_x: float,
    center_y: float,
    half_w: float,
    half_d: float,
    rotation_z: float,
) -> bool:
    dx = float(px) - float(center_x)
    dy = float(py) - float(center_y)
    c = math.cos(float(rotation_z))
    s = math.sin(float(rotation_z))

    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return abs(lx) <= float(half_w) and abs(ly) <= float(half_d)


def collect_blocked_faces_by_buildings(
    terrain_obj: bpy.types.Object,
    footprints: list[BuildingFootprint],
    margin: float,
) -> set[int]:
    blocked: set[int] = set()
    if not footprints:
        return blocked

    extra = max(0.0, float(margin))
    for face in terrain_obj.data.polygons:
        cx = float(face.center.x)
        cy = float(face.center.y)
        for fp in footprints:
            if point_in_oriented_rect(
                px=cx,
                py=cy,
                center_x=fp.cx,
                center_y=fp.cy,
                half_w=fp.half_w + extra,
                half_d=fp.half_d + extra,
                rotation_z=fp.yaw,
            ):
                blocked.add(face.index)
                break
    return blocked


def _terrain_height_at_xy(
    terrain_obj: bpy.types.Object,
    x_coord: float,
    y_coord: float,
) -> float | None:
    from mathutils import Vector  # type: ignore[import-untyped]

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        terr_eval = terrain_obj.evaluated_get(depsgraph)
        inv = terr_eval.matrix_world.inverted()
        ray_origin_w = Vector((float(x_coord), float(y_coord), 10000.0))
        ray_target_w = Vector((float(x_coord), float(y_coord), -10000.0))
        ray_origin_l = inv @ ray_origin_w
        ray_target_l = inv @ ray_target_w
        ray_dir_l = (ray_target_l - ray_origin_l).normalized()
        hit, loc_l, _, _ = terr_eval.ray_cast(ray_origin_l, ray_dir_l)
        if not hit:
            return None
        loc_w = terr_eval.matrix_world @ loc_l
        return float(loc_w.z)
    except Exception:
        return None


def _terrain_heights_for_footprint(
    terrain_obj: bpy.types.Object,
    center_x: float,
    center_y: float,
    half_w: float,
    half_d: float,
    rotation_z: float,
) -> list[float]:
    heights: list[float] = []
    local_samples = [
        (-half_w, -half_d),
        (half_w, -half_d),
        (half_w, half_d),
        (-half_w, half_d),
        (0.0, 0.0),
        (0.0, -half_d),
        (0.0, half_d),
        (-half_w, 0.0),
        (half_w, 0.0),
    ]
    c = math.cos(float(rotation_z))
    s = math.sin(float(rotation_z))
    for lx, ly in local_samples:
        dx = c * float(lx) - s * float(ly)
        dy = s * float(lx) + c * float(ly)
        h = _terrain_height_at_xy(
            terrain_obj=terrain_obj,
            x_coord=float(center_x) + float(dx),
            y_coord=float(center_y) + float(dy),
        )
        if h is not None:
            heights.append(float(h))
    return heights


def terrain_vertex_heights_in_footprint(
    terrain_obj: bpy.types.Object,
    center_x: float,
    center_y: float,
    half_w: float,
    half_d: float,
    rotation_z: float,
) -> list[float]:
    c = math.cos(float(rotation_z))
    s = math.sin(float(rotation_z))
    extent_x = abs(c) * float(half_w) + abs(s) * float(half_d)
    extent_y = abs(s) * float(half_w) + abs(c) * float(half_d)

    x_min = float(center_x) - extent_x
    x_max = float(center_x) + extent_x
    y_min = float(center_y) - extent_y
    y_max = float(center_y) + extent_y

    heights: list[float] = []
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        terr_eval = terrain_obj.evaluated_get(depsgraph)
        mw = terr_eval.matrix_world
        mesh = terr_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            for v in mesh.vertices:
                vw = mw @ v.co
                xw = float(vw.x)
                yw = float(vw.y)
                if not (x_min <= xw <= x_max and y_min <= yw <= y_max):
                    continue
                if point_in_oriented_rect(
                    px=xw,
                    py=yw,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    half_w=float(half_w),
                    half_d=float(half_d),
                    rotation_z=float(rotation_z),
                ):
                    heights.append(float(vw.z))
        finally:
            terr_eval.to_mesh_clear()
    except Exception:
        return []

    return heights


def _create_building_object(
    scene: bpy.types.Scene,
    terrain_obj: bpy.types.Object,
    name: str,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    wall_material: bpy.types.Material,
    roof_material: bpy.types.Material,
    base_offset: float,
    rotation_z: float,
    roof_style: str,
    roof_rise_ratio: float,
    ridge_along_x: bool,
) -> bpy.types.Object:
    half_w = max(0.5, float(width) * 0.5)
    half_d = max(0.5, float(depth) * 0.5)
    h = max(1.5, float(height))

    cx = float(center_x)
    cy = float(center_y)
    sampled = terrain_vertex_heights_in_footprint(
        terrain_obj=terrain_obj,
        center_x=cx,
        center_y=cy,
        half_w=half_w,
        half_d=half_d,
        rotation_z=float(rotation_z),
    )
    if not sampled:
        sampled = _terrain_heights_for_footprint(
            terrain_obj=terrain_obj,
            center_x=cx,
            center_y=cy,
            half_w=half_w,
            half_d=half_d,
            rotation_z=float(rotation_z),
        )
    if sampled:
        top_ground_z = max(sampled)
        low_ground_z = min(sampled)
    else:
        top_ground_z = 0.0
        low_ground_z = 0.0
    base_z = float(top_ground_z)
    base_z += max(0.0, float(base_offset))

    relief = max(0.0, float(top_ground_z) - float(low_ground_z))
    # Keep a small skirt for terrain contact, but cap its influence so
    # roof/world height remains visually consistent across local slopes.
    foundation_skirt = min(0.45, max(0.20, relief * 0.35 + 0.15))
    z0 = -foundation_skirt

    roof_mode = str(roof_style).strip().lower()
    roof_faces_start = 0

    if roof_mode == "gable":
        rise_ratio = max(0.12, min(0.45, float(roof_rise_ratio)))
        roof_rise = max(0.6, h * rise_ratio)
        wall_top = max(1.0, h - roof_rise)
        ridge_z = wall_top + roof_rise
        verts = [
            (-half_w, -half_d, z0),
            (half_w, -half_d, z0),
            (half_w, half_d, z0),
            (-half_w, half_d, z0),
            (-half_w, -half_d, wall_top),
            (half_w, -half_d, wall_top),
            (half_w, half_d, wall_top),
            (-half_w, half_d, wall_top),
        ]

        faces: list[tuple[int, int, int]] = [
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
            (0, 1, 2),
            (0, 2, 3),
        ]

        if ridge_along_x:
            verts.extend(
                [
                    (-half_w, 0.0, ridge_z),
                    (half_w, 0.0, ridge_z),
                ]
            )
            faces.extend(
                [
                    (4, 8, 7),
                    (5, 6, 9),
                ]
            )
            roof_faces_start = len(faces)
            faces.extend(
                [
                    (4, 5, 9),
                    (4, 9, 8),
                    (7, 8, 9),
                    (7, 9, 6),
                ]
            )
        else:
            verts.extend(
                [
                    (0.0, -half_d, ridge_z),
                    (0.0, half_d, ridge_z),
                ]
            )
            faces.extend(
                [
                    (4, 5, 8),
                    (7, 9, 6),
                ]
            )
            roof_faces_start = len(faces)
            faces.extend(
                [
                    (4, 8, 9),
                    (4, 9, 7),
                    (5, 6, 9),
                    (5, 9, 8),
                ]
            )
    else:
        z_top = h
        verts = [
            (-half_w, -half_d, z0),
            (half_w, -half_d, z0),
            (half_w, half_d, z0),
            (-half_w, half_d, z0),
            (-half_w, -half_d, z_top),
            (half_w, -half_d, z_top),
            (half_w, half_d, z_top),
            (-half_w, half_d, z_top),
        ]

        faces = [
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
            (0, 1, 2),
            (0, 2, 3),
        ]
        roof_faces_start = len(faces)
        faces.extend(
            [
                (4, 6, 5),
                (4, 7, 6),
            ]
        )

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (cx, cy, base_z)
    obj.rotation_euler[2] = float(rotation_z)
    obj["height_reference_z"] = float(base_z)
    obj["height_reference_ground_z"] = float(top_ground_z)
    scene.collection.objects.link(obj)

    mesh.materials.append(wall_material)
    mesh.materials.append(roof_material)
    for poly in mesh.polygons:
        poly.material_index = 1 if poly.index >= roof_faces_start else 0

    return obj


def add_sparse_context_buildings(
    scene: bpy.types.Scene,
    terrain_obj: bpy.types.Object,
    terrain_cfg: TerrainConfig,
    profile_overrides: dict[str, object],
    terrain_size: float,
    paths: list[list[tuple[float, float]]],
    widths: list[float],
    rng: random.Random,
) -> list[BuildingFootprint]:
    enabled = bool(context_value(terrain_cfg, profile_overrides, "buildings_enabled"))
    if not enabled:
        return []

    min_count, max_count = scale_count_range(
        int(context_value(terrain_cfg, profile_overrides, "building_count_min")),
        int(context_value(terrain_cfg, profile_overrides, "building_count_max")),
        terrain_size=terrain_size,
        reference_size=float(
            context_value(terrain_cfg, profile_overrides, "context_reference_size")
        ),
        scale_mode="area",
        require_at_least_one=False,
    )
    target_count = rng.randint(max(0, min_count), max(0, max_count))
    if target_count <= 0:
        return []

    size_half = terrain_size * 0.5
    border = float(context_value(terrain_cfg, profile_overrides, "building_border_inset_ratio"))
    border = max(0.70, min(0.995, border)) * size_half

    path_buffer = float(context_value(terrain_cfg, profile_overrides, "building_path_buffer"))
    min_spacing = float(context_value(terrain_cfg, profile_overrides, "building_min_spacing"))
    w_min = float(context_value(terrain_cfg, profile_overrides, "building_width_min"))
    w_max = float(context_value(terrain_cfg, profile_overrides, "building_width_max"))
    d_min = float(context_value(terrain_cfg, profile_overrides, "building_depth_min"))
    d_max = float(context_value(terrain_cfg, profile_overrides, "building_depth_max"))
    h_min = float(context_value(terrain_cfg, profile_overrides, "building_height_min"))
    h_max = float(context_value(terrain_cfg, profile_overrides, "building_height_max"))
    base_offset = float(context_value(terrain_cfg, profile_overrides, "building_base_offset"))

    max_ground_relief = float(
        context_value(terrain_cfg, profile_overrides, "building_max_ground_relief")
    )
    wall_candidates, roof_candidates = collect_building_textures(terrain_cfg=terrain_cfg)

    placed: list[BuildingFootprint] = []
    placed_widths: list[float] = []
    placed_depths: list[float] = []
    placed_heights: list[float] = []
    gable_count = 0
    attempts = max(50, target_count * 25)
    for _ in range(attempts):
        if len(placed) >= target_count:
            break

        bw = rng.uniform(w_min, w_max)
        bd = rng.uniform(d_min, d_max)
        bh = rng.uniform(h_min, h_max)
        half_w = max(0.5, bw * 0.5)
        half_d = max(0.5, bd * 0.5)

        x = rng.uniform(-border, border)
        y = rng.uniform(-border, border)
        yaw = rng.uniform(0.0, math.tau)
        roof_style = "gable" if rng.random() < 0.5 else "flat"
        roof_rise_ratio = rng.uniform(0.12, 0.42)
        ridge_along_x = bool(rng.getrandbits(1))

        sampled = terrain_vertex_heights_in_footprint(
            terrain_obj=terrain_obj,
            center_x=x,
            center_y=y,
            half_w=half_w,
            half_d=half_d,
            rotation_z=yaw,
        )
        if not sampled:
            sampled = _terrain_heights_for_footprint(
                terrain_obj=terrain_obj,
                center_x=x,
                center_y=y,
                half_w=half_w,
                half_d=half_d,
                rotation_z=yaw,
            )
        if len(sampled) < 5:
            continue
        relief = max(sampled) - min(sampled)
        if relief > max(0.0, max_ground_relief):
            continue

        dist_paths = distance_to_paths(x, y, paths)
        eff_buffer = path_buffer
        if widths:
            eff_buffer += 0.5 * max(float(v) for v in widths)
        if dist_paths < eff_buffer:
            continue

        overlap = False
        for fp in placed:
            dx = abs(x - fp.cx)
            dy = abs(y - fp.cy)
            if dx <= (half_w + fp.half_w + min_spacing) and dy <= (
                half_d + fp.half_d + min_spacing
            ):
                overlap = True
                break
        if overlap:
            continue

        wall_mat, roof_mat = build_building_material_pair(
            rng=rng,
            index=len(placed) + 1,
            wall_candidates=wall_candidates,
            roof_candidates=roof_candidates,
        )

        _create_building_object(
            scene=scene,
            terrain_obj=terrain_obj,
            name=f"ContextBuilding_{len(placed) + 1:02d}",
            center_x=x,
            center_y=y,
            width=bw,
            depth=bd,
            height=bh,
            wall_material=wall_mat,
            roof_material=roof_mat,
            base_offset=base_offset,
            rotation_z=yaw,
            roof_style=roof_style,
            roof_rise_ratio=roof_rise_ratio,
            ridge_along_x=ridge_along_x,
        )
        placed.append(BuildingFootprint(cx=x, cy=y, half_w=half_w, half_d=half_d, yaw=yaw))
        placed_widths.append(float(bw))
        placed_depths.append(float(bd))
        placed_heights.append(float(bh))
        if roof_style == "gable":
            gable_count += 1

    if placed:
        logger.info(
            "Buildings placed: %d/%d | size_w=[%.2f,%.2f] size_d=[%.2f,%.2f] h=[%.2f,%.2f] roofs(gable/flat)=%d/%d",
            len(placed),
            target_count,
            min(placed_widths),
            max(placed_widths),
            min(placed_depths),
            max(placed_depths),
            min(placed_heights),
            max(placed_heights),
            gable_count,
            len(placed) - gable_count,
        )
    else:
        logger.info("Buildings placed: 0/%d", target_count)
    return placed


def snap_context_buildings_to_terrain(
    scene: bpy.types.Scene,
    terrain_obj: bpy.types.Object,
    terrain_cfg: TerrainConfig,
) -> int:
    _ = scene

    base_offset = max(0.0, float(getattr(terrain_cfg.context, "building_base_offset", 0.02)))
    snapped = 0
    for obj in list(bpy.data.objects):
        if not str(getattr(obj, "name", "")).startswith("ContextBuilding_"):
            continue

        try:
            local_corners = list(obj.bound_box)
        except Exception:
            continue
        if not local_corners:
            continue

        cx = float(obj.location.x)
        cy = float(obj.location.y)
        yaw = float(obj.rotation_euler[2])
        half_w = max(0.1, max(abs(float(corner[0])) for corner in local_corners))
        half_d = max(0.1, max(abs(float(corner[1])) for corner in local_corners))

        sampled = terrain_vertex_heights_in_footprint(
            terrain_obj=terrain_obj,
            center_x=cx,
            center_y=cy,
            half_w=half_w,
            half_d=half_d,
            rotation_z=yaw,
        )
        if not sampled:
            sampled = _terrain_heights_for_footprint(
                terrain_obj=terrain_obj,
                center_x=cx,
                center_y=cy,
                half_w=half_w,
                half_d=half_d,
                rotation_z=yaw,
            )
        if not sampled:
            continue

        new_base_z = max(sampled) + base_offset
        obj.location.z = float(new_base_z)
        obj["height_reference_z"] = float(new_base_z)
        obj["height_reference_ground_z"] = float(max(sampled))
        snapped += 1

    if snapped > 0:
        logger.info(
            "Re-anchored %d context buildings to highest terrain-under-footprint Z.", snapped
        )
    return snapped
