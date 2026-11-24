from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import bpy
from mathutils import Matrix, Vector

from .botaniq_materials import (
    _botaniq_leaf_polygon_indices,
)


@dataclass(frozen=True)
class _LeafUnitGeometry:
    vertex_indices: tuple[int, ...]
    centroid: Vector
    offsets: tuple[Vector, ...]


def _leaf_geometry_preservation_enabled(cfg: object | None) -> bool:
    if cfg is None:
        return False
    mode = str(getattr(cfg, "botaniq_leaf_geometry_preservation_mode", "off") or "off")
    return mode.strip().lower() in {"centroid_translate", "centroid", "translate"}


def _leaf_unit_vertex_indices(
    tree_obj: bpy.types.Object,
    cfg: object,
) -> list[tuple[int, ...]]:
    leaf_polygons = _botaniq_leaf_polygon_indices(tree_obj, cfg)
    if not leaf_polygons:
        return []
    mesh = tree_obj.data
    leaf_set = {int(index) for index in leaf_polygons if 0 <= int(index) < len(mesh.polygons)}
    vertex_to_faces: dict[int, list[int]] = {}
    for polygon_index in leaf_set:
        for vertex_index in mesh.polygons[polygon_index].vertices:
            vertex_to_faces.setdefault(int(vertex_index), []).append(polygon_index)

    units: list[tuple[int, ...]] = []
    remaining = set(leaf_set)
    while remaining:
        start = remaining.pop()
        stack = [start]
        vertices: set[int] = set()
        while stack:
            polygon_index = stack.pop()
            polygon = mesh.polygons[polygon_index]
            for vertex_index in polygon.vertices:
                vertex_index = int(vertex_index)
                vertices.add(vertex_index)
                for neighbor in vertex_to_faces.get(vertex_index, []):
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        if vertices:
            units.append(tuple(sorted(vertices)))
    return units


def capture_leaf_unit_geometry(
    tree_obj: bpy.types.Object,
    cfg: object | None,
) -> list[_LeafUnitGeometry]:
    if not _leaf_geometry_preservation_enabled(cfg):
        return []
    if tree_obj.type != "MESH" or tree_obj.data is None:
        return []

    captures: list[_LeafUnitGeometry] = []
    for vertex_indices in _leaf_unit_vertex_indices(tree_obj, cfg):
        coords = [tree_obj.data.vertices[index].co.copy() for index in vertex_indices]
        if not coords:
            continue
        centroid = Vector((0.0, 0.0, 0.0))
        for coord in coords:
            centroid += coord
        centroid /= len(coords)
        captures.append(
            _LeafUnitGeometry(
                vertex_indices=vertex_indices,
                centroid=centroid,
                offsets=tuple(coord - centroid for coord in coords),
            )
        )
    return captures


def restore_leaf_unit_geometry_to_centroids(
    tree_obj: bpy.types.Object,
    captures: list[_LeafUnitGeometry],
) -> int:
    if not captures or tree_obj.type != "MESH" or tree_obj.data is None:
        return 0

    restored_vertices: set[int] = set()
    for capture in captures:
        current_centroid = Vector((0.0, 0.0, 0.0))
        valid_indices = [
            index for index in capture.vertex_indices if 0 <= index < len(tree_obj.data.vertices)
        ]
        if not valid_indices:
            continue
        for vertex_index in valid_indices:
            current_centroid += tree_obj.data.vertices[vertex_index].co
        current_centroid /= len(valid_indices)
        for vertex_index, offset in zip(capture.vertex_indices, capture.offsets):
            if not 0 <= vertex_index < len(tree_obj.data.vertices):
                continue
            tree_obj.data.vertices[vertex_index].co = current_centroid + offset
            restored_vertices.add(int(vertex_index))
    if restored_vertices:
        tree_obj.data.update()
    return len(restored_vertices)


def apply_anisotropic_scale_preserving_leaf_units(
    tree_obj: bpy.types.Object,
    scale_xy: float,
    scale_z: float,
    cfg: object | None,
) -> bool:
    if not _leaf_geometry_preservation_enabled(cfg):
        tree_obj.data.transform(Matrix.Diagonal((scale_xy, scale_xy, scale_z, 1.0)))
        tree_obj.data.update()
        return False

    captures = capture_leaf_unit_geometry(tree_obj, cfg)
    if not captures:
        tree_obj.data.transform(Matrix.Diagonal((scale_xy, scale_xy, scale_z, 1.0)))
        tree_obj.data.update()
        return False

    leaf_vertices = {
        int(vertex_index) for capture in captures for vertex_index in capture.vertex_indices
    }
    for vertex in tree_obj.data.vertices:
        if int(vertex.index) in leaf_vertices:
            continue
        vertex.co.x *= scale_xy
        vertex.co.y *= scale_xy
        vertex.co.z *= scale_z

    for capture in captures:
        target_centroid = Vector(
            (
                capture.centroid.x * scale_xy,
                capture.centroid.y * scale_xy,
                capture.centroid.z * scale_z,
            )
        )
        for vertex_index, offset in zip(capture.vertex_indices, capture.offsets):
            if not 0 <= vertex_index < len(tree_obj.data.vertices):
                continue
            tree_obj.data.vertices[vertex_index].co = target_centroid + offset * scale_xy
    tree_obj.data.update()
    tree_obj["botaniq_leaf_geometry_preservation_mode"] = "centroid_translate"
    tree_obj["botaniq_leaf_geometry_units_preserved"] = int(len(captures))
    return True


def apply_position_transform_preserving_leaf_units(
    tree_obj: bpy.types.Object,
    cfg: object | None,
    transform_coord: Callable[[Vector], Vector],
) -> bool:
    if not _leaf_geometry_preservation_enabled(cfg):
        return False
    captures = capture_leaf_unit_geometry(tree_obj, cfg)
    if not captures:
        return False

    leaf_vertices = {
        int(vertex_index) for capture in captures for vertex_index in capture.vertex_indices
    }
    for vertex in tree_obj.data.vertices:
        if int(vertex.index) in leaf_vertices:
            continue
        vertex.co = transform_coord(vertex.co.copy())

    for capture in captures:
        target_centroid = transform_coord(capture.centroid.copy())
        for vertex_index, offset in zip(capture.vertex_indices, capture.offsets):
            if not 0 <= vertex_index < len(tree_obj.data.vertices):
                continue
            tree_obj.data.vertices[vertex_index].co = target_centroid + offset
    tree_obj.data.update()
    tree_obj["botaniq_leaf_geometry_preservation_mode"] = "centroid_translate"
    tree_obj["botaniq_leaf_geometry_units_preserved"] = int(len(captures))
    return True
