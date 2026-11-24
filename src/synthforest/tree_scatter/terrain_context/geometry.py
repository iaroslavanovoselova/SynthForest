from __future__ import annotations

import math
import random

import bpy


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _polyline_lengths(points_xy: list[tuple[float, float]]) -> tuple[list[float], float]:
    if not points_xy:
        return [], 0.0
    lengths = [0.0]
    total = 0.0
    for idx in range(1, len(points_xy)):
        ax, ay = points_xy[idx - 1]
        bx, by = points_xy[idx]
        seg_len = math.hypot(float(bx) - float(ax), float(by) - float(ay))
        total += seg_len
        lengths.append(total)
    return lengths, total


def distance_point_to_segment_2d(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-8:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)

    t = (wx * vx + wy * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * vx
    cy = ay + t * vy
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx * dx + dy * dy)


def _edge_point(size_half: float, edge: str, rng: random.Random) -> tuple[float, float]:
    border = float(size_half) * 0.988
    jitter = rng.uniform(-0.72, 0.72) * border
    if edge == "left":
        return (-border, jitter)
    if edge == "right":
        return (border, jitter)
    if edge == "top":
        return (jitter, border)
    return (jitter, -border)


def sample_polyline(
    size_half: float,
    rng: random.Random,
    point_count: int,
    kind: str = "trail",
) -> list[tuple[float, float]]:
    edges = ["left", "right", "top", "bottom"]
    start_edge = rng.choice(edges)
    opposite = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }
    end_edge = opposite[start_edge]
    start_point = _edge_point(size_half, start_edge, rng)
    end_point = _edge_point(size_half, end_edge, rng)

    dx = float(end_point[0]) - float(start_point[0])
    dy = float(end_point[1]) - float(start_point[1])
    axis_len = max(1e-6, math.hypot(dx, dy))
    axis_x = dx / axis_len
    axis_y = dy / axis_len
    normal_x = -axis_y
    normal_y = axis_x

    normalized_kind = str(kind).strip().lower()
    meander_strength = 0.09 if normalized_kind == "trail" else 0.18
    axial_jitter_strength = 0.015 if normalized_kind == "trail" else 0.025
    points: list[tuple[float, float]] = [start_point]

    interior_count = max(0, int(point_count) - 2)
    if interior_count > 0:
        interior_ts = []
        for idx in range(interior_count):
            base_t = (idx + 1) / float(interior_count + 1)
            jitter = rng.uniform(-0.08, 0.08)
            interior_ts.append(_clamp(base_t + jitter, 0.04, 0.96))
        interior_ts.sort()

        drift = rng.uniform(-0.35, 0.35)
        for t in interior_ts:
            drift = 0.62 * drift + rng.uniform(-0.65, 0.65)
            base_x = _lerp(start_point[0], end_point[0], t)
            base_y = _lerp(start_point[1], end_point[1], t)
            envelope = math.sin(math.pi * t)
            lateral = drift * envelope * float(size_half) * meander_strength
            axial = rng.uniform(-1.0, 1.0) * envelope * float(size_half) * axial_jitter_strength
            x_coord = base_x + normal_x * lateral + axis_x * axial
            y_coord = base_y + normal_y * lateral + axis_y * axial
            points.append(
                (
                    _clamp(x_coord, -0.93 * size_half, 0.93 * size_half),
                    _clamp(y_coord, -0.93 * size_half, 0.93 * size_half),
                )
            )

    points.append(end_point)
    return _dedupe_polyline(points, min_spacing=0.4)


def _densify_polyline(
    points_xy: list[tuple[float, float]],
    max_segment_length: float,
) -> list[tuple[float, float]]:
    if len(points_xy) < 2:
        return list(points_xy)

    max_len = max(0.001, float(max_segment_length))
    densified: list[tuple[float, float]] = []
    for idx in range(len(points_xy) - 1):
        ax, ay = points_xy[idx]
        bx, by = points_xy[idx + 1]
        dx = float(bx) - float(ax)
        dy = float(by) - float(ay)
        seg_len = math.sqrt(dx * dx + dy * dy)
        steps = max(1, int(math.ceil(seg_len / max_len)))
        for step in range(steps):
            t = step / float(steps)
            densified.append((float(ax) + dx * t, float(ay) + dy * t))
    densified.append((float(points_xy[-1][0]), float(points_xy[-1][1])))
    return densified


def _dedupe_polyline(
    points_xy: list[tuple[float, float]],
    min_spacing: float,
) -> list[tuple[float, float]]:
    if not points_xy:
        return []
    spacing = max(1e-6, float(min_spacing))
    out: list[tuple[float, float]] = [(float(points_xy[0][0]), float(points_xy[0][1]))]
    for x_coord, y_coord in points_xy[1:]:
        px, py = out[-1]
        dx = float(x_coord) - px
        dy = float(y_coord) - py
        if (dx * dx + dy * dy) >= (spacing * spacing):
            out.append((float(x_coord), float(y_coord)))
    if len(out) == 1 and len(points_xy) > 1:
        out.append((float(points_xy[-1][0]), float(points_xy[-1][1])))
    return out


def _chaikin_smooth(
    points_xy: list[tuple[float, float]], iterations: int
) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points_xy]
    iters = max(0, int(iterations))
    if len(pts) < 3 or iters == 0:
        return pts

    for _ in range(iters):
        if len(pts) < 3:
            break
        new_pts: list[tuple[float, float]] = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            ax, ay = a
            bx, by = b
            qx = 0.75 * ax + 0.25 * bx
            qy = 0.75 * ay + 0.25 * by
            rx = 0.25 * ax + 0.75 * bx
            ry = 0.25 * ay + 0.75 * by
            new_pts.append((qx, qy))
            new_pts.append((rx, ry))
        new_pts.append(pts[-1])
        pts = new_pts

    return pts


def _remove_near_backtracks(
    points_xy: list[tuple[float, float]],
    dot_threshold: float = -0.25,
    min_seg_len: float = 1e-3,
) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points_xy]
    if len(pts) < 3:
        return pts

    keep: list[tuple[float, float]] = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = keep[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]

        v1x = bx - ax
        v1y = by - ay
        v2x = cx - bx
        v2y = cy - by

        l1 = math.sqrt(v1x * v1x + v1y * v1y)
        l2 = math.sqrt(v2x * v2x + v2y * v2y)
        if l1 < min_seg_len or l2 < min_seg_len:
            continue

        dot = (v1x / l1) * (v2x / l2) + (v1y / l1) * (v2y / l2)
        if dot < float(dot_threshold):
            continue
        keep.append((bx, by))
    keep.append(pts[-1])
    return keep


def add_project_plus_nearest_shrinkwrap(
    obj: bpy.types.Object,
    terrain_obj: bpy.types.Object,
    surface_offset: float,
) -> None:
    # Mirror river/trail placement: project downward first, then nearest fallback.
    clearance = max(0.0, float(surface_offset)) + 0.16

    shrinkwrap = obj.modifiers.new("Shrinkwrap", "SHRINKWRAP")
    shrinkwrap.wrap_method = "PROJECT"
    shrinkwrap.wrap_mode = "ON_SURFACE"
    shrinkwrap.target = terrain_obj
    if hasattr(shrinkwrap, "use_project_z"):
        shrinkwrap.use_project_z = True
    if hasattr(shrinkwrap, "use_project_x"):
        shrinkwrap.use_project_x = False
    if hasattr(shrinkwrap, "use_project_y"):
        shrinkwrap.use_project_y = False
    if hasattr(shrinkwrap, "use_negative_direction"):
        shrinkwrap.use_negative_direction = True
    if hasattr(shrinkwrap, "use_positive_direction"):
        shrinkwrap.use_positive_direction = False
    shrinkwrap.offset = float(clearance)

    shrinkwrap_fallback = obj.modifiers.new("ShrinkwrapFallback", "SHRINKWRAP")
    shrinkwrap_fallback.wrap_method = "NEAREST_SURFACEPOINT"
    shrinkwrap_fallback.wrap_mode = "ON_SURFACE"
    shrinkwrap_fallback.target = terrain_obj
    shrinkwrap_fallback.offset = float(clearance)


def strip_mesh_from_polyline(
    scene: bpy.types.Scene,
    terrain_obj: bpy.types.Object,
    points_xy: list[tuple[float, float]],
    width: float,
    z_offset: float,
    surface_offset: float,
    name: str,
    material: bpy.types.Material,
    kind: str = "trail",
) -> tuple[bpy.types.Object, list[tuple[float, float]]]:
    base_points = _dedupe_polyline(points_xy, min_spacing=0.25)
    smooth_iterations = 2 if str(kind).strip().lower() == "trail" else 3
    base_points = _chaikin_smooth(base_points, iterations=smooth_iterations)
    base_points = _remove_near_backtracks(base_points, dot_threshold=-0.25)
    base_points = _dedupe_polyline(base_points, min_spacing=0.15)

    max_seg = max(0.25, float(width) * (0.14 if str(kind).strip().lower() == "trail" else 0.12))
    densified_points = _densify_polyline(base_points, max_segment_length=max_seg)
    if len(densified_points) < 2:
        raise ValueError("Polyline must have at least 2 points")

    start_z = float(z_offset) + 50.0
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    vert_uvs: list[tuple[float, float]] = []
    path_masks: list[float] = []

    normalized_kind = str(kind).strip().lower()
    path_lengths, total_length = _polyline_lengths(densified_points)
    total_length = max(1e-6, total_length)

    width_factors: list[float] = []
    phase_a = 0.173 if normalized_kind == "trail" else 0.337
    phase_b = 0.611 if normalized_kind == "trail" else 0.487
    for distance in path_lengths:
        t = distance / total_length
        wave_a = math.sin((t + phase_a) * math.tau)
        wave_b = math.sin((t * 2.0 + phase_b) * math.tau)
        envelope = 0.82 + 0.18 * math.sin(math.pi * t)
        if normalized_kind == "trail":
            factor = 0.88 + 0.08 * wave_a + 0.04 * wave_b
            factor *= envelope
            factor = _clamp(factor, 0.74, 1.05)
        else:
            factor = 0.95 + 0.16 * wave_a + 0.08 * wave_b
            factor *= 0.90 + 0.12 * math.sin(math.pi * t)
            factor = _clamp(factor, 0.78, 1.20)
        width_factors.append(factor)

    if float(width) < 4.0:
        width_samples = 9
    elif float(width) < 8.0:
        width_samples = 11
    else:
        width_samples = 13
    prev_nx: float | None = None
    prev_ny: float | None = None

    n = len(densified_points)
    for i, (x_coord, y_coord) in enumerate(densified_points):
        if i == 0:
            x1, y1 = densified_points[1]
            dx = float(x1) - float(x_coord)
            dy = float(y1) - float(y_coord)
        elif i == n - 1:
            x0, y0 = densified_points[n - 2]
            dx = float(x_coord) - float(x0)
            dy = float(y_coord) - float(y0)
        else:
            x0, y0 = densified_points[i - 1]
            x1, y1 = densified_points[i + 1]
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)

        norm = math.sqrt(dx * dx + dy * dy)
        if norm <= 1e-8:
            nx, ny = 1.0, 0.0
        else:
            tx = dx / norm
            ty = dy / norm
            nx = -ty
            ny = tx

        if prev_nx is not None and prev_ny is not None:
            if (nx * prev_nx + ny * prev_ny) < 0.0:
                nx = -nx
                ny = -ny
        prev_nx, prev_ny = nx, ny

        cx = float(x_coord)
        cy = float(y_coord)
        row_width = max(0.4, float(width) * width_factors[i])
        row_half_width = max(0.05, row_width * 0.5)
        width_offsets = [
            row_half_width - (2.0 * row_half_width * (j / float(width_samples - 1)))
            for j in range(width_samples)
        ]
        u_coord = path_lengths[i] / max(
            0.75, row_width * (1.6 if normalized_kind == "trail" else 2.5)
        )
        for offset_idx, off in enumerate(width_offsets):
            vx = cx + nx * float(off)
            vy = cy + ny * float(off)
            verts.append((vx, vy, start_z))
            lateral = abs(float(off)) / max(1e-6, row_half_width)
            mask = _clamp(1.0 - lateral ** (1.55 if normalized_kind == "trail" else 1.20), 0.0, 1.0)
            v_coord = offset_idx / max(1.0, float(width_samples - 1))
            vert_uvs.append((u_coord, v_coord))
            path_masks.append(mask)

    cols = len(width_offsets)
    for i in range(n - 1):
        row0 = i * cols
        row1 = (i + 1) * cols
        for j in range(cols - 1):
            a = row0 + j
            b = row0 + j + 1
            c = row1 + j + 1
            d = row1 + j
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="PathUV")
    for polygon in mesh.polygons:
        for vert_idx, loop_idx in zip(polygon.vertices, polygon.loop_indices):
            uv_layer.data[loop_idx].uv = vert_uvs[vert_idx]

    if hasattr(mesh, "color_attributes"):
        color_attr = mesh.color_attributes.new(
            name="path_mask",
            type="FLOAT_COLOR",
            domain="POINT",
        )
        for idx, mask in enumerate(path_masks):
            color_attr.data[idx].color = (mask, mask, mask, 1.0)

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    scene.collection.objects.link(obj)

    add_project_plus_nearest_shrinkwrap(
        obj=obj,
        terrain_obj=terrain_obj,
        surface_offset=surface_offset,
    )

    return obj, densified_points


def collect_blocked_faces_by_paths(
    terrain_obj: bpy.types.Object,
    paths: list[list[tuple[float, float]]],
    widths: list[float],
    margin: float,
    extra_margins: list[float] | None = None,
) -> set[int]:
    blocked: set[int] = set()
    if not paths:
        return blocked

    if extra_margins is None:
        extras = [0.0] * len(paths)
    else:
        extras = [max(0.0, float(v)) for v in extra_margins]
        if len(extras) < len(paths):
            extras.extend([0.0] * (len(paths) - len(extras)))

    for face in terrain_obj.data.polygons:
        center = face.center
        cx = float(center.x)
        cy = float(center.y)

        for points, width, extra in zip(paths, widths, extras):
            radius = max(0.0, float(width) * 0.5 + margin + extra)
            for idx in range(len(points) - 1):
                ax, ay = points[idx]
                bx, by = points[idx + 1]
                if distance_point_to_segment_2d(cx, cy, ax, ay, bx, by) <= radius:
                    blocked.add(face.index)
                    break
            if face.index in blocked:
                break

    return blocked


def collect_blocked_faces_for_water(
    terrain_obj: bpy.types.Object,
    waterline_elevation: float,
) -> set[int]:
    blocked: set[int] = set()
    for face in terrain_obj.data.polygons:
        if float(face.center.z) <= float(waterline_elevation):
            blocked.add(face.index)
    return blocked


def distance_to_paths(
    x_coord: float,
    y_coord: float,
    paths: list[list[tuple[float, float]]],
) -> float:
    if not paths:
        return 1e9
    best = 1e9
    for points in paths:
        if len(points) < 2:
            continue
        for idx in range(len(points) - 1):
            ax, ay = points[idx]
            bx, by = points[idx + 1]
            d = distance_point_to_segment_2d(x_coord, y_coord, ax, ay, bx, by)
            if d < best:
                best = d
    return best


def split_polyline_by_clearance(
    points_xy: list[tuple[float, float]],
    blocker_paths: list[list[tuple[float, float]]],
    clearance: float,
    *,
    max_segment_length: float = 0.6,
    min_points_per_segment: int = 2,
) -> list[list[tuple[float, float]]]:
    if len(points_xy) < 2 or not blocker_paths:
        return [list(points_xy)] if len(points_xy) >= 2 else []

    densified = _densify_polyline(points_xy, max_segment_length=max_segment_length)
    kept_segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    radius = max(0.0, float(clearance))

    for x_coord, y_coord in densified:
        blocked = distance_to_paths(float(x_coord), float(y_coord), blocker_paths) <= radius
        if blocked:
            if len(current) >= int(min_points_per_segment):
                kept_segments.append(_dedupe_polyline(current, min_spacing=0.15))
            current = []
            continue
        current.append((float(x_coord), float(y_coord)))

    if len(current) >= int(min_points_per_segment):
        kept_segments.append(_dedupe_polyline(current, min_spacing=0.15))

    return [segment for segment in kept_segments if len(segment) >= int(min_points_per_segment)]


def collect_blocked_faces_for_rural_clearings(
    terrain_obj: bpy.types.Object,
    rng: random.Random,
    size_half: float,
    clearing_count: int,
    radius_min: float,
    radius_max: float,
) -> set[int]:
    blocked: set[int] = set()
    for _ in range(max(0, int(clearing_count))):
        cx = rng.uniform(-0.75 * size_half, 0.75 * size_half)
        cy = rng.uniform(-0.75 * size_half, 0.75 * size_half)
        radius = rng.uniform(float(radius_min), float(radius_max))
        radius_sq = radius * radius
        for face in terrain_obj.data.polygons:
            dx = float(face.center.x) - cx
            dy = float(face.center.y) - cy
            if dx * dx + dy * dy <= radius_sq:
                blocked.add(face.index)
    return blocked
