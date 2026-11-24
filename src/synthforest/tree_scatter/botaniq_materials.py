from __future__ import annotations

import colorsys
import logging
import math
import random

import bpy
from mathutils import Vector

from .botaniq_utils import as_list as _as_list
from .config import ScatterConfig
from .constants import (
    INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
    INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
)
from .mesh_utils import mesh_local_bounds as _mesh_local_bounds

logger = logging.getLogger(__name__)
BOTANIQ_LEAF_DENSITY_RECIPE_SCHEMA = "synthforest.botaniq_leaf_density.v1"
_BOTANIQ_LEAVES_GRADIENT_GROUP = "bq_leaves_gradient"


def _sample_clamped_range(
    rng: random.Random,
    low: float,
    high: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    low = max(min_value, min(max_value, float(low)))
    high = max(min_value, min(max_value, float(high)))
    if high < low:
        low, high = high, low
    return rng.uniform(low, high)


def _sample_int_bounds(
    rng: random.Random,
    low: object,
    high: object,
    *,
    min_value: int = 0,
) -> int:
    try:
        low_i = int(round(float(low)))
    except Exception:
        low_i = min_value
    try:
        high_i = int(round(float(high)))
    except Exception:
        high_i = low_i
    low_i = max(min_value, low_i)
    high_i = max(min_value, high_i)
    if high_i < low_i:
        low_i, high_i = high_i, low_i
    return rng.randint(low_i, high_i)


def _botaniq_leaf_density_keywords(cfg: ScatterConfig) -> tuple[str, ...]:
    raw_keywords = getattr(cfg, "botaniq_leaf_density_material_keywords", None)
    if isinstance(raw_keywords, str):
        values = [item.strip() for item in raw_keywords.split(",")]
    elif isinstance(raw_keywords, (list, tuple)):
        values = [str(item).strip() for item in raw_keywords]
    else:
        values = []
    keywords = tuple(item.lower() for item in values if item)
    return keywords or ("leaf", "leaves", "foliage", "blossom", "flower")


def _botaniq_material_indices_matching_keywords(
    tree_obj: bpy.types.Object,
    keywords: tuple[str, ...],
) -> set[int]:
    material_indices: set[int] = set()
    for index, slot in enumerate(tree_obj.material_slots):
        material = slot.material
        material_name = material.name.lower() if material is not None else ""
        if any(keyword in material_name for keyword in keywords):
            material_indices.add(index)
    return material_indices


def _read_object_vertex_group_weights(
    tree_obj: bpy.types.Object,
    group_name: str,
) -> tuple[list[float], bool]:
    mesh = tree_obj.data
    vertex_count = len(mesh.vertices)
    weights = [0.0] * vertex_count
    group = tree_obj.vertex_groups.get(group_name)
    if group is None:
        return weights, False

    group_index = int(group.index)
    for vertex in mesh.vertices:
        for assignment in vertex.groups:
            if assignment.group == group_index:
                weights[vertex.index] = max(0.0, min(1.0, float(assignment.weight)))
                break
    return weights, True


def _botaniq_leaf_polygon_indices(
    tree_obj: bpy.types.Object,
    cfg: ScatterConfig,
) -> list[int]:
    if tree_obj.type != "MESH" or tree_obj.data is None:
        return []

    mesh = tree_obj.data
    material_indices = _botaniq_material_indices_matching_keywords(
        tree_obj,
        _botaniq_leaf_density_keywords(cfg),
    )
    if material_indices:
        return [
            int(poly.index)
            for poly in mesh.polygons
            if int(poly.material_index) in material_indices
        ]

    weights, has_weights = _read_object_vertex_group_weights(
        tree_obj,
        _BOTANIQ_LEAVES_GRADIENT_GROUP,
    )
    if not has_weights:
        return []

    threshold = max(
        0.0,
        min(
            1.0,
            float(getattr(cfg, "botaniq_leaf_density_vertex_group_threshold", 0.45)),
        ),
    )
    leaf_polygons: list[int] = []
    for poly in mesh.polygons:
        poly_vertices = list(poly.vertices)
        if not poly_vertices:
            continue
        average_weight = sum(weights[int(index)] for index in poly_vertices) / len(poly_vertices)
        if average_weight >= threshold:
            leaf_polygons.append(int(poly.index))
    return leaf_polygons


def _polygon_centroid(tree_obj: bpy.types.Object, polygon_index: int) -> Vector:
    mesh = tree_obj.data
    poly = mesh.polygons[polygon_index]
    vertices = [mesh.vertices[int(vertex_index)].co for vertex_index in poly.vertices]
    if not vertices:
        return Vector((0.0, 0.0, 0.0))
    accum = Vector((0.0, 0.0, 0.0))
    for coord in vertices:
        accum += coord
    return accum / len(vertices)


def _build_botaniq_leaf_face_units(
    tree_obj: bpy.types.Object,
    leaf_polygon_indices: list[int],
    cfg: ScatterConfig,
) -> list[dict[str, object]]:
    mesh = tree_obj.data
    leaf_set = {
        int(index) for index in leaf_polygon_indices if 0 <= int(index) < len(mesh.polygons)
    }
    if not leaf_set:
        return []

    vertex_to_faces: dict[int, list[int]] = {}
    for polygon_index in leaf_set:
        for vertex_index in mesh.polygons[polygon_index].vertices:
            vertex_to_faces.setdefault(int(vertex_index), []).append(polygon_index)

    components: list[list[int]] = []
    remaining = set(leaf_set)
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            polygon_index = stack.pop()
            for vertex_index in mesh.polygons[polygon_index].vertices:
                for neighbor in vertex_to_faces.get(int(vertex_index), []):
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))

    max_unit_faces = max(1, int(getattr(cfg, "botaniq_leaf_density_max_unit_faces", 8)))
    split_units: list[list[int]] = []
    for component in sorted(components, key=lambda item: item[0]):
        if len(component) <= max_unit_faces:
            split_units.append(component)
            continue
        for index in range(0, len(component), max_unit_faces):
            split_units.append(component[index : index + max_unit_faces])

    units: list[dict[str, object]] = []
    for face_indices in split_units:
        centroid = Vector((0.0, 0.0, 0.0))
        for polygon_index in face_indices:
            centroid += _polygon_centroid(tree_obj, polygon_index)
        centroid /= max(1, len(face_indices))
        units.append(
            {
                "face_indices": list(face_indices),
                "centroid": [float(centroid.x), float(centroid.y), float(centroid.z)],
            }
        )
    return units


def _build_botaniq_leaf_density_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> dict[str, object]:
    recipe: dict[str, object] = {
        "schema": BOTANIQ_LEAF_DENSITY_RECIPE_SCHEMA,
        "enabled": False,
        "reason": "",
        "leaf_face_count": 0,
        "leaf_unit_count": 0,
        "removed_face_count": 0,
    }
    if not bool(getattr(cfg, "botaniq_leaf_density_variation_enabled", True)):
        recipe["reason"] = "disabled"
        return recipe

    probability = max(
        0.0,
        min(1.0, float(getattr(cfg, "botaniq_leaf_density_variation_probability", 1.0))),
    )
    recipe["probability"] = float(probability)
    if rng.random() > probability:
        recipe["reason"] = "skipped_probability"
        return recipe

    leaf_polygons = _botaniq_leaf_polygon_indices(tree_obj, cfg)
    recipe["leaf_face_count"] = len(leaf_polygons)
    if not leaf_polygons:
        recipe["reason"] = "no_leaf_faces"
        return recipe

    leaf_units = _build_botaniq_leaf_face_units(tree_obj, leaf_polygons, cfg)
    recipe["leaf_unit_count"] = len(leaf_units)
    if not leaf_units:
        recipe["reason"] = "no_leaf_units"
        return recipe

    keep_ratio = _sample_clamped_range(
        rng,
        float(getattr(cfg, "botaniq_leaf_keep_ratio_min", 0.86)),
        float(getattr(cfg, "botaniq_leaf_keep_ratio_max", 0.96)),
        min_value=0.50,
        max_value=1.0,
    )
    min_v, max_v = _mesh_local_bounds(tree_obj)
    crown_radius = max(float(max_v.x - min_v.x), float(max_v.y - min_v.y)) * 0.5
    radius_min = float(getattr(cfg, "botaniq_leaf_patch_radius_fraction_min", 0.12))
    radius_max = float(getattr(cfg, "botaniq_leaf_patch_radius_fraction_max", 0.28))
    patch_count = _sample_int_bounds(
        rng,
        getattr(cfg, "botaniq_leaf_patch_count_min", 2),
        getattr(cfg, "botaniq_leaf_patch_count_max", 5),
        min_value=0,
    )
    patch_count = min(patch_count, max(0, len(leaf_units)))

    patches: list[dict[str, object]] = []
    for _index in range(patch_count):
        unit = rng.choice(leaf_units)
        center = list(unit["centroid"])
        patches.append(
            {
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "radius_m": float(
                    max(
                        0.05,
                        crown_radius
                        * _sample_clamped_range(
                            rng,
                            radius_min,
                            radius_max,
                            min_value=0.01,
                            max_value=1.0,
                        ),
                    )
                ),
                "drop_strength": float(
                    _sample_clamped_range(
                        rng,
                        float(getattr(cfg, "botaniq_leaf_patch_drop_strength_min", 0.06)),
                        float(getattr(cfg, "botaniq_leaf_patch_drop_strength_max", 0.18)),
                        min_value=0.0,
                        max_value=0.45,
                    )
                ),
            }
        )

    recipe.update(
        {
            "enabled": True,
            "reason": "planned",
            "keep_ratio": float(keep_ratio),
            "base_drop_probability": float(max(0.0, 1.0 - keep_ratio)),
            "max_drop_probability": float(
                max(
                    0.0,
                    min(
                        0.90,
                        float(getattr(cfg, "botaniq_leaf_density_max_drop_probability", 0.42)),
                    ),
                )
            ),
            "random_seed": int(rng.randrange(1, 2_147_483_647)),
            "patches": patches,
        }
    )
    return recipe


def _delete_botaniq_leaf_faces(
    tree_obj: bpy.types.Object,
    face_indices: list[int],
) -> int:
    if not face_indices:
        return 0

    import bmesh

    mesh = tree_obj.data
    requested = sorted({int(index) for index in face_indices})
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        faces = [bm.faces[index] for index in requested if 0 <= index < len(bm.faces)]
        if not faces:
            return 0
        bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    return len(faces)


def _apply_botaniq_leaf_density_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, object],
    cfg: ScatterConfig,
) -> None:
    if not bool(recipe.get("enabled", False)):
        tree_obj["botaniq_leaf_density_variation_applied"] = False
        tree_obj["botaniq_leaf_density_reason"] = str(recipe.get("reason", "disabled"))
        tree_obj["botaniq_leaf_density_removed_faces"] = 0
        return

    leaf_polygons = _botaniq_leaf_polygon_indices(tree_obj, cfg)
    leaf_units = _build_botaniq_leaf_face_units(tree_obj, leaf_polygons, cfg)
    if not leaf_units:
        tree_obj["botaniq_leaf_density_variation_applied"] = False
        tree_obj["botaniq_leaf_density_reason"] = "no_leaf_units_at_apply"
        tree_obj["botaniq_leaf_density_removed_faces"] = 0
        return

    keep_ratio = max(0.0, min(1.0, float(recipe.get("keep_ratio", 0.92))))
    base_drop = max(0.0, min(1.0, float(recipe.get("base_drop_probability", 1.0 - keep_ratio))))
    max_drop = max(0.0, min(0.90, float(recipe.get("max_drop_probability", 0.42))))
    seed = int(recipe.get("random_seed", 1))
    patches = recipe.get("patches", [])
    if not isinstance(patches, list):
        patches = []

    faces_to_remove: list[int] = []
    removed_units = 0
    for unit_index, unit in enumerate(leaf_units):
        centroid_values = unit.get("centroid", [0.0, 0.0, 0.0])
        if not isinstance(centroid_values, list) or len(centroid_values) < 2:
            centroid_values = [0.0, 0.0, 0.0]
        centroid_x = float(centroid_values[0])
        centroid_y = float(centroid_values[1])

        patch_drop = 0.0
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            center = patch.get("center", [0.0, 0.0, 0.0])
            if not isinstance(center, list) or len(center) < 2:
                continue
            radius = max(1e-6, float(patch.get("radius_m", 0.0)))
            distance = math.hypot(centroid_x - float(center[0]), centroid_y - float(center[1]))
            if distance >= radius:
                continue
            influence = (1.0 - distance / radius) ** 2
            patch_drop = max(
                patch_drop,
                influence * max(0.0, float(patch.get("drop_strength", 0.0))),
            )

        drop_probability = min(max_drop, base_drop + patch_drop)
        unit_faces = [int(index) for index in unit.get("face_indices", [])]
        unit_seed = seed + unit_index * 1_000_003 + (min(unit_faces) if unit_faces else 0)
        if random.Random(unit_seed).random() >= drop_probability:
            continue
        faces_to_remove.extend(unit_faces)
        removed_units += 1

    try:
        removed_faces = _delete_botaniq_leaf_faces(tree_obj, faces_to_remove)
    except Exception as exc:
        logger.warning(
            "Failed to apply botaniq leaf-density variation to %s: %s", tree_obj.name, exc
        )
        tree_obj["botaniq_leaf_density_variation_applied"] = False
        tree_obj["botaniq_leaf_density_reason"] = "delete_failed"
        tree_obj["botaniq_leaf_density_removed_faces"] = 0
        return

    tree_obj["botaniq_leaf_density_variation_applied"] = bool(removed_faces > 0)
    tree_obj["botaniq_leaf_density_reason"] = str(recipe.get("reason", "planned"))
    tree_obj["botaniq_leaf_density_keep_ratio_target"] = float(keep_ratio)
    tree_obj["botaniq_leaf_density_leaf_faces_before"] = int(len(leaf_polygons))
    tree_obj["botaniq_leaf_density_leaf_units_before"] = int(len(leaf_units))
    tree_obj["botaniq_leaf_density_removed_units"] = int(removed_units)
    tree_obj["botaniq_leaf_density_removed_faces"] = int(removed_faces)
    tree_obj["botaniq_leaf_density_removed_face_fraction"] = (
        float(removed_faces) / len(leaf_polygons) if leaf_polygons else 0.0
    )


def _object_float_property(
    obj: bpy.types.Object,
    name: str,
    fallback: float,
) -> float:
    try:
        return float(obj.get(name, fallback))
    except Exception:
        return float(fallback)


def _jitter_rgba_hsv(
    color: object,
    *,
    hue_shift: float,
    saturation: float,
    value: float,
    brightness: float,
    contrast: float,
) -> tuple[float, float, float, float]:
    try:
        red = max(0.0, min(1.0, float(color[0])))
        green = max(0.0, min(1.0, float(color[1])))
        blue = max(0.0, min(1.0, float(color[2])))
        alpha = max(0.0, min(1.0, float(color[3] if len(color) > 3 else 1.0)))
    except Exception:
        red, green, blue, alpha = 0.35, 0.50, 0.18, 1.0

    hue, sat, val = colorsys.rgb_to_hsv(red, green, blue)
    hue = (hue + float(hue_shift)) % 1.0
    sat = max(0.0, min(1.0, sat * float(saturation)))
    val = max(0.0, min(1.0, val * float(value)))
    out_r, out_g, out_b = colorsys.hsv_to_rgb(hue, sat, val)

    adjusted = []
    for channel in (out_r, out_g, out_b):
        channel = (channel - 0.5) * (1.0 + float(contrast)) + 0.5 + float(brightness)
        adjusted.append(max(0.0, min(1.0, channel)))
    return float(adjusted[0]), float(adjusted[1]), float(adjusted[2]), float(alpha)


def _principled_bsdf_nodes(material: bpy.types.Material) -> list[bpy.types.Node]:
    if material.node_tree is None:
        return []
    return [
        node
        for node in material.node_tree.nodes
        if getattr(node, "bl_idname", "") == "ShaderNodeBsdfPrincipled"
    ]


def _material_has_instance_leaf_variation_nodes(material: bpy.types.Material) -> bool:
    if material.node_tree is None:
        return False
    return any(
        str(getattr(node, "name", "")).startswith("FG Instance Leaf ")
        for node in material.node_tree.nodes
    )


def _new_instancer_attribute_node(
    material: bpy.types.Material,
    attribute_name: str,
    label: str,
    location: tuple[float, float],
) -> bpy.types.Node:
    node = material.node_tree.nodes.new("ShaderNodeAttribute")
    node.name = label
    node.label = label
    node.location = location
    node.attribute_name = attribute_name
    try:
        node.attribute_type = "INSTANCER"
    except Exception:
        pass
    return node


def _append_instance_leaf_variation_nodes(
    material: bpy.types.Material,
    source_socket: bpy.types.NodeSocket,
    target_socket: bpy.types.NodeSocket,
    *,
    hue_attribute_name: str,
    brightness_attribute_name: str,
    location: tuple[float, float],
) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    base_x, base_y = location

    hue_attr = _new_instancer_attribute_node(
        material,
        hue_attribute_name,
        "FG Instance Leaf Hue Attribute",
        (base_x - 460.0, base_y + 30.0),
    )
    hue_add = nodes.new("ShaderNodeMath")
    hue_add.name = "FG Instance Leaf Hue Offset"
    hue_add.label = "FG Instance Leaf Hue Offset"
    hue_add.operation = "ADD"
    hue_add.location = (base_x - 260.0, base_y + 30.0)
    hue_add.inputs[0].default_value = 0.5

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    hue_sat.name = "FG Instance Leaf Hue Variation"
    hue_sat.label = "FG Instance Leaf Hue Variation"
    hue_sat.location = (base_x, base_y + 30.0)
    hue_sat.inputs["Saturation"].default_value = 1.0
    hue_sat.inputs["Value"].default_value = 1.0

    brightness_attr = _new_instancer_attribute_node(
        material,
        brightness_attribute_name,
        "FG Instance Leaf Brightness Attribute",
        (base_x - 460.0, base_y - 170.0),
    )
    bright_contrast = nodes.new("ShaderNodeBrightContrast")
    bright_contrast.name = "FG Instance Leaf Brightness Variation"
    bright_contrast.label = "FG Instance Leaf Brightness Variation"
    bright_contrast.location = (base_x + 240.0, base_y + 30.0)
    bright_contrast.inputs["Contrast"].default_value = 0.0

    links.new(source_socket, hue_sat.inputs["Color"])
    links.new(hue_attr.outputs["Fac"], hue_add.inputs[1])
    links.new(hue_add.outputs["Value"], hue_sat.inputs["Hue"])
    links.new(hue_sat.outputs["Color"], bright_contrast.inputs["Color"])
    links.new(brightness_attr.outputs["Fac"], bright_contrast.inputs["Bright"])
    links.new(bright_contrast.outputs["Color"], target_socket)


def _apply_leaf_color_nodes_to_material(
    material: bpy.types.Material,
    recipe: dict[str, object],
) -> int:
    material.use_nodes = True
    if material.node_tree is None:
        return 0

    hue_shift = float(recipe.get("hue_shift", 0.0))
    saturation = float(recipe.get("saturation", 1.0))
    value = float(recipe.get("value", 1.0))
    brightness = float(recipe.get("brightness", 0.0))
    contrast = float(recipe.get("contrast", 0.0))
    instance_variation_enabled = bool(recipe.get("instance_variation_enabled", False))
    hue_attribute_name = str(
        recipe.get("instance_leaf_hue_attribute", INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME)
    )
    brightness_attribute_name = str(
        recipe.get(
            "instance_leaf_brightness_attribute",
            INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
        )
    )
    has_instance_variation_nodes = _material_has_instance_leaf_variation_nodes(material)
    changed = 0

    for node in material.node_tree.nodes:
        if getattr(node, "bl_idname", "") != "ShaderNodeGroup":
            continue
        hue_input = node.inputs.get("Hue")
        saturation_input = node.inputs.get("Saturation")
        value_input = node.inputs.get("Value")
        if hue_input is None or saturation_input is None or value_input is None:
            continue
        if not hue_input.is_linked:
            hue_input.default_value = float(hue_input.default_value) + hue_shift
        if not saturation_input.is_linked:
            saturation_input.default_value = max(
                0.70,
                min(1.35, float(saturation_input.default_value) * saturation),
            )
        if not value_input.is_linked:
            value_input.default_value = max(
                0.70,
                min(1.35, float(value_input.default_value) * value + brightness),
            )
        translucency_hue = node.inputs.get("Translucency Hue")
        translucency_saturation = node.inputs.get("Translucency Saturation")
        translucency_value = node.inputs.get("Translucency Value")
        if translucency_hue is not None and not translucency_hue.is_linked:
            translucency_hue.default_value = float(translucency_hue.default_value) + hue_shift * 0.5
        if translucency_saturation is not None and not translucency_saturation.is_linked:
            translucency_saturation.default_value = max(
                0.70,
                min(
                    1.50,
                    float(translucency_saturation.default_value) * (1.0 + (saturation - 1.0) * 0.5),
                ),
            )
        if translucency_value is not None and not translucency_value.is_linked:
            translucency_value.default_value = max(
                0.70,
                min(1.80, float(translucency_value.default_value) * (1.0 + (value - 1.0) * 0.5)),
            )
        changed += 1

    for node in _principled_bsdf_nodes(material):
        base_input = node.inputs.get("Base Color")
        if base_input is None:
            continue

        if base_input.is_linked and base_input.links:
            old_link = base_input.links[0]
            source_socket = old_link.from_socket
            material.node_tree.links.remove(old_link)

            hue_sat = material.node_tree.nodes.new("ShaderNodeHueSaturation")
            hue_sat.name = "FG Leaf Hue Variation"
            hue_sat.label = "FG Leaf Hue Variation"
            hue_sat.location = (node.location.x - 520.0, node.location.y + 120.0)
            hue_sat.inputs["Hue"].default_value = 0.5 + hue_shift
            hue_sat.inputs["Saturation"].default_value = saturation
            hue_sat.inputs["Value"].default_value = value

            bright_contrast = material.node_tree.nodes.new("ShaderNodeBrightContrast")
            bright_contrast.name = "FG Leaf Brightness Variation"
            bright_contrast.label = "FG Leaf Brightness Variation"
            bright_contrast.location = (node.location.x - 280.0, node.location.y + 120.0)
            bright_contrast.inputs["Bright"].default_value = brightness
            bright_contrast.inputs["Contrast"].default_value = contrast

            material.node_tree.links.new(source_socket, hue_sat.inputs["Color"])
            material.node_tree.links.new(hue_sat.outputs["Color"], bright_contrast.inputs["Color"])
            if instance_variation_enabled and not has_instance_variation_nodes:
                _append_instance_leaf_variation_nodes(
                    material,
                    bright_contrast.outputs["Color"],
                    base_input,
                    hue_attribute_name=hue_attribute_name,
                    brightness_attribute_name=brightness_attribute_name,
                    location=(node.location.x - 40.0, node.location.y - 140.0),
                )
                has_instance_variation_nodes = True
            else:
                material.node_tree.links.new(bright_contrast.outputs["Color"], base_input)
            changed += 1
            continue

        varied_color = _jitter_rgba_hsv(
            base_input.default_value,
            hue_shift=hue_shift,
            saturation=saturation,
            value=value,
            brightness=brightness,
            contrast=contrast,
        )
        if instance_variation_enabled and not has_instance_variation_nodes:
            color_node = material.node_tree.nodes.new("ShaderNodeRGB")
            color_node.name = "FG Leaf Base Color Variation"
            color_node.label = "FG Leaf Base Color Variation"
            color_node.location = (node.location.x - 560.0, node.location.y + 120.0)
            color_node.outputs["Color"].default_value = varied_color
            _append_instance_leaf_variation_nodes(
                material,
                color_node.outputs["Color"],
                base_input,
                hue_attribute_name=hue_attribute_name,
                brightness_attribute_name=brightness_attribute_name,
                location=(node.location.x - 320.0, node.location.y + 100.0),
            )
            has_instance_variation_nodes = True
        else:
            base_input.default_value = varied_color
        changed += 1

    if changed:
        try:
            material.diffuse_color = _jitter_rgba_hsv(
                material.diffuse_color,
                hue_shift=hue_shift,
                saturation=saturation,
                value=value,
                brightness=brightness,
                contrast=contrast,
            )
        except Exception:
            pass
    return changed


def _botaniq_leaf_material_slot_indices(
    tree_obj: bpy.types.Object,
    cfg: ScatterConfig,
) -> list[int]:
    keyword_indices = _botaniq_material_indices_matching_keywords(
        tree_obj,
        _botaniq_leaf_density_keywords(cfg),
    )
    if keyword_indices:
        return sorted(keyword_indices)

    leaf_polygons = _botaniq_leaf_polygon_indices(tree_obj, cfg)
    return sorted(
        {
            int(tree_obj.data.polygons[index].material_index)
            for index in leaf_polygons
            if 0 <= index < len(tree_obj.data.polygons)
        }
    )


def _build_botaniq_leaf_material_variation_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
    *,
    scope: str,
) -> dict[str, object]:
    prefix = "botaniq_scatter_leaf_material" if scope == "scatter" else "botaniq_leaf_material"
    template_variation_enabled = bool(getattr(cfg, f"{prefix}_variation_enabled", True))
    instance_variation_enabled = bool(
        getattr(cfg, "botaniq_instance_leaf_material_variation_enabled", True)
    )
    if not template_variation_enabled and not instance_variation_enabled:
        return {"enabled": False, "reason": "disabled", "scope": scope}

    slot_indices = _botaniq_leaf_material_slot_indices(tree_obj, cfg)
    if not slot_indices:
        return {"enabled": False, "reason": "no_leaf_material_slots", "scope": scope}

    recipe = {
        "enabled": True,
        "reason": "planned",
        "scope": scope,
        "slot_indices": [int(index) for index in slot_indices],
        "hue_shift": (
            _sample_clamped_range(
                rng,
                float(getattr(cfg, f"{prefix}_hue_shift_min", -0.02)),
                float(getattr(cfg, f"{prefix}_hue_shift_max", 0.02)),
                min_value=-0.12,
                max_value=0.12,
            )
            if template_variation_enabled
            else 0.0
        ),
        "saturation": (
            _sample_clamped_range(
                rng,
                float(getattr(cfg, f"{prefix}_saturation_min", 0.92)),
                float(getattr(cfg, f"{prefix}_saturation_max", 1.08)),
                min_value=0.70,
                max_value=1.35,
            )
            if template_variation_enabled
            else 1.0
        ),
        "value": (
            _sample_clamped_range(
                rng,
                float(getattr(cfg, f"{prefix}_value_min", 0.92)),
                float(getattr(cfg, f"{prefix}_value_max", 1.08)),
                min_value=0.70,
                max_value=1.35,
            )
            if template_variation_enabled
            else 1.0
        ),
        "brightness": (
            _sample_clamped_range(
                rng,
                float(getattr(cfg, f"{prefix}_brightness_min", -0.02)),
                float(getattr(cfg, f"{prefix}_brightness_max", 0.02)),
                min_value=-0.12,
                max_value=0.12,
            )
            if template_variation_enabled
            else 0.0
        ),
        "contrast": (
            _sample_clamped_range(
                rng,
                float(getattr(cfg, f"{prefix}_contrast_min", -0.03)),
                float(getattr(cfg, f"{prefix}_contrast_max", 0.04)),
                min_value=-0.20,
                max_value=0.25,
            )
            if template_variation_enabled
            else 0.0
        ),
        "instance_variation_enabled": instance_variation_enabled,
        "instance_leaf_hue_attribute": INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
        "instance_leaf_brightness_attribute": INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
        "material_names": [
            str(tree_obj.material_slots[index].material.name)
            for index in slot_indices
            if 0 <= index < len(tree_obj.material_slots)
            and tree_obj.material_slots[index].material is not None
        ],
    }
    return recipe


def _build_botaniq_instance_leaf_material_recipe(
    tree_obj: bpy.types.Object,
    cfg: ScatterConfig,
) -> dict[str, object]:
    if not bool(getattr(cfg, "botaniq_instance_leaf_material_variation_enabled", True)):
        return {"enabled": False, "reason": "disabled", "scope": "instance"}

    slot_indices = _botaniq_leaf_material_slot_indices(tree_obj, cfg)
    if not slot_indices:
        return {"enabled": False, "reason": "no_leaf_material_slots", "scope": "instance"}

    return {
        "enabled": True,
        "reason": "instance_variation_nodes",
        "scope": "instance",
        "slot_indices": [int(index) for index in slot_indices],
        "hue_shift": 0.0,
        "saturation": 1.0,
        "value": 1.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "instance_variation_enabled": True,
        "instance_leaf_hue_attribute": INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
        "instance_leaf_brightness_attribute": INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
        "material_names": [
            str(tree_obj.material_slots[index].material.name)
            for index in slot_indices
            if 0 <= index < len(tree_obj.material_slots)
            and tree_obj.material_slots[index].material is not None
        ],
    }


def _apply_botaniq_leaf_material_variation_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, object],
) -> None:
    if not bool(recipe.get("enabled", False)):
        tree_obj["botaniq_leaf_material_variation_reason"] = str(recipe.get("reason", "disabled"))
        return

    slot_indices = [
        int(index) for index in _as_list(recipe.get("slot_indices", [])) if str(index).strip()
    ]
    changed_slots = 0
    changed_nodes = 0
    for slot_index in slot_indices:
        if slot_index < 0 or slot_index >= len(tree_obj.material_slots):
            continue
        slot = tree_obj.material_slots[slot_index]
        material = slot.material
        if material is None:
            continue

        local_material = material.copy()
        local_material.name = (
            f"{material.name}_{tree_obj.name}_{str(recipe.get('scope', 'variant'))}_leaf_color"
        )
        try:
            slot.link = "OBJECT"
        except Exception:
            pass
        slot.material = local_material
        changed = _apply_leaf_color_nodes_to_material(local_material, recipe)
        if changed <= 0:
            continue
        changed_slots += 1
        changed_nodes += changed

    tree_obj["botaniq_leaf_material_variation_applied"] = bool(changed_slots > 0)
    tree_obj["botaniq_leaf_material_variation_scope"] = str(recipe.get("scope", "variant"))
    tree_obj["botaniq_leaf_material_variation_slots"] = int(changed_slots)
    tree_obj["botaniq_leaf_material_variation_nodes"] = int(changed_nodes)
    tree_obj["botaniq_leaf_material_hue_shift"] = float(recipe.get("hue_shift", 0.0))
    tree_obj["botaniq_leaf_material_saturation"] = float(recipe.get("saturation", 1.0))
    tree_obj["botaniq_leaf_material_value"] = float(recipe.get("value", 1.0))
    tree_obj["botaniq_leaf_material_instance_variation_enabled"] = bool(
        recipe.get("instance_variation_enabled", False)
    )


def _sample_realistic_bq_random_property(
    rng: random.Random,
    base_value: float,
    jitter_min: float,
    jitter_max: float,
) -> tuple[float, float]:
    base_value = max(0.0, min(1.0, float(base_value)))
    jitter = _sample_clamped_range(
        rng,
        jitter_min,
        jitter_max,
        min_value=0.0,
        max_value=0.05,
    )
    if jitter <= 0.0:
        return base_value, 0.0

    low = max(0.0, base_value - jitter)
    high = min(1.0, base_value + jitter)
    if base_value >= 1.0 - jitter:
        value = rng.uniform(low, base_value)
    elif base_value <= jitter:
        value = rng.uniform(base_value, high)
    else:
        value = rng.uniform(low, high)
    return float(value), float(value - base_value)


def _build_botaniq_color_adjustment_recipe(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> dict[str, object]:
    if not bool(getattr(cfg, "botaniq_color_variation_enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    brightness_base = max(0.01, _object_float_property(tree_obj, "bq_brightness", 1.0))
    brightness_jitter = max(0.0, float(getattr(cfg, "botaniq_brightness_jitter", 0.04)))
    brightness = brightness_base * rng.uniform(
        max(0.01, 1.0 - brightness_jitter),
        1.0 + brightness_jitter,
    )

    branch_base = _object_float_property(tree_obj, "bq_random_per_branch", 0.0)
    leaf_base = _object_float_property(tree_obj, "bq_random_per_leaf", 0.0)
    branch_random, branch_delta = _sample_realistic_bq_random_property(
        rng,
        branch_base,
        float(getattr(cfg, "botaniq_branch_hue_random_min", 0.0)),
        float(getattr(cfg, "botaniq_branch_hue_random_max", 0.006)),
    )
    leaf_random, leaf_delta = _sample_realistic_bq_random_property(
        rng,
        leaf_base,
        float(getattr(cfg, "botaniq_leaf_hue_random_min", 0.0)),
        float(getattr(cfg, "botaniq_leaf_hue_random_max", 0.010)),
    )

    return {
        "enabled": True,
        "reason": "planned",
        "bq_brightness": float(max(0.75, min(1.25, brightness))),
        "bq_random_per_branch": float(branch_random),
        "bq_random_per_leaf": float(leaf_random),
        "branch_base": float(branch_base),
        "leaf_base": float(leaf_base),
        "branch_delta": float(branch_delta),
        "leaf_delta": float(leaf_delta),
        "leaf_material": _build_botaniq_leaf_material_variation_recipe(
            tree_obj,
            rng,
            cfg,
            scope="prebake",
        ),
    }


def _apply_botaniq_color_adjustment_recipe(
    tree_obj: bpy.types.Object,
    recipe: dict[str, object],
    cfg: ScatterConfig | None = None,
) -> None:
    if not bool(recipe.get("enabled", False)):
        tree_obj["botaniq_color_variation_mode"] = str(recipe.get("reason", "disabled"))
        return

    branch_random = float(recipe.get("bq_random_per_branch", 0.0))
    leaf_random = float(recipe.get("bq_random_per_leaf", 0.0))
    branch_delta = float(recipe.get("branch_delta", 0.0))
    leaf_delta = float(recipe.get("leaf_delta", 0.0))
    if branch_random >= 1.0 and branch_delta > 0.0:
        branch_random = max(0.0, 1.0 - branch_delta)
    if leaf_random >= 1.0 and leaf_delta > 0.0:
        leaf_random = max(0.0, 1.0 - leaf_delta)

    tree_obj["bq_brightness"] = float(recipe.get("bq_brightness", 1.0))
    tree_obj["bq_random_per_branch"] = max(0.0, min(1.0, branch_random))
    tree_obj["bq_random_per_leaf"] = max(0.0, min(1.0, leaf_random))
    tree_obj["botaniq_color_variation_mode"] = "native_bq_adjustments"
    leaf_material_recipe = recipe.get("leaf_material", {})
    if isinstance(leaf_material_recipe, dict):
        if cfg is not None and not leaf_material_recipe.get("slot_indices"):
            rebuilt = _build_botaniq_leaf_material_variation_recipe(
                tree_obj,
                random.Random(int(recipe.get("leaf_material_seed", 1))),
                cfg,
                scope=str(leaf_material_recipe.get("scope", "prebake")),
            )
            rebuilt.update(leaf_material_recipe)
            leaf_material_recipe = rebuilt
        _apply_botaniq_leaf_material_variation_recipe(tree_obj, leaf_material_recipe)


def _set_botaniq_color_adjustment_properties(
    tree_obj: bpy.types.Object,
    rng: random.Random,
    cfg: ScatterConfig,
) -> None:
    _apply_botaniq_color_adjustment_recipe(
        tree_obj,
        _build_botaniq_color_adjustment_recipe(tree_obj, rng, cfg),
        cfg,
    )
