from __future__ import annotations

import bpy
from mathutils import Vector


def mesh_local_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [Vector(corner) for corner in obj.bound_box]
    min_v = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    max_v = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    return min_v, max_v


def infer_ground_anchor_local_z(obj: bpy.types.Object) -> float:
    """Return the lowest trunk-like vertex so foliage does not anchor the tree."""
    if obj.type != "MESH" or obj.data is None:
        return float(min(corner[2] for corner in obj.bound_box))

    mesh = obj.data
    leaf_tokens = ("leaf", "leaves", "foliage", "blossom", "flower")
    bark_poly_indices: list[int] = []

    for poly in mesh.polygons:
        slot_idx = int(poly.material_index)
        slot_name = ""
        if 0 <= slot_idx < len(obj.material_slots):
            slot = obj.material_slots[slot_idx]
            if slot.material is not None:
                slot_name = slot.material.name.lower()
        if any(token in slot_name for token in leaf_tokens):
            continue
        bark_poly_indices.append(int(poly.index))

    if bark_poly_indices:
        bark_vertex_indices: set[int] = set()
        for poly_idx in bark_poly_indices:
            bark_vertex_indices.update(
                int(vertex_idx) for vertex_idx in mesh.polygons[poly_idx].vertices
            )
        if bark_vertex_indices:
            return float(min(mesh.vertices[index].co.z for index in bark_vertex_indices))

    return float(min(corner[2] for corner in obj.bound_box))
