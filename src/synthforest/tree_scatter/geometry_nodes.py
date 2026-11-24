from __future__ import annotations

import bpy

from .config import TerrainConfig
from .constants import (
    INSTANCE_ID_ATTRIBUTE_NAME,
    INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
    INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_X_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_Y_ATTRIBUTE_NAME,
    INSTANCE_ROTATION_Z_ATTRIBUTE_NAME,
    INSTANCE_SCALE_X_ATTRIBUTE_NAME,
    INSTANCE_SCALE_Y_ATTRIBUTE_NAME,
    INSTANCE_SCALE_Z_ATTRIBUTE_NAME,
    INSTANCE_VARIANT_INDEX_ATTRIBUTE_NAME,
)


def _add_group_input_socket(
    node_group: bpy.types.GeometryNodeTree,
    name: str,
    socket_type: str,
    default_value=None,
    min_value=None,
    max_value=None,
) -> str:
    socket = node_group.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
    if default_value is not None:
        socket.default_value = default_value
    if min_value is not None:
        socket.min_value = min_value
    if max_value is not None:
        socket.max_value = max_value
    return socket.identifier


def _named_attribute_node(
    nodes: bpy.types.Nodes,
    name: str,
    data_type: str,
    location: tuple[float, float],
) -> bpy.types.Node:
    node = nodes.new("GeometryNodeInputNamedAttribute")
    node.location = location
    node.data_type = data_type
    if "Name" in node.inputs:
        node.inputs["Name"].default_value = name
    return node


def build_precomputed_scatter_gn_tree(
    terrain_obj: bpy.types.Object,
    tree_collection: bpy.types.Collection,
    placement_points_obj: bpy.types.Object,
    terrain_cfg: TerrainConfig,
    variant_attribute_name: str = "tree_variant_id",
) -> bpy.types.GeometryNodeTree:
    node_group = bpy.data.node_groups.new("TreeScatterPrecomputedNodes", "GeometryNodeTree")

    node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    sid_terrain_scale = _add_group_input_socket(
        node_group,
        "Terrain Noise Scale",
        "NodeSocketFloat",
        default_value=terrain_cfg.noise_scale,
        min_value=0.001,
        max_value=5.0,
    )
    sid_terrain_detail = _add_group_input_socket(
        node_group,
        "Terrain Detail",
        "NodeSocketFloat",
        default_value=terrain_cfg.detail,
        min_value=0.0,
        max_value=15.0,
    )
    sid_terrain_rough = _add_group_input_socket(
        node_group,
        "Terrain Roughness",
        "NodeSocketFloat",
        default_value=terrain_cfg.roughness,
        min_value=0.0,
        max_value=1.0,
    )
    sid_elev_min = _add_group_input_socket(
        node_group,
        "Elevation Min",
        "NodeSocketFloat",
        default_value=terrain_cfg.elevation_min,
        min_value=-500.0,
        max_value=500.0,
    )
    sid_elev_max = _add_group_input_socket(
        node_group,
        "Elevation Max",
        "NodeSocketFloat",
        default_value=terrain_cfg.elevation_max,
        min_value=-500.0,
        max_value=500.0,
    )
    sid_scatter_density = _add_group_input_socket(
        node_group,
        "Scatter Density",
        "NodeSocketFloat",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
    )

    nodes = node_group.nodes
    links = node_group.links
    x_step = 220

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (0, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (x_step * 11, 0)

    terrain_position = nodes.new("GeometryNodeInputPosition")
    terrain_position.location = (x_step * 1, 260)

    terrain_noise = nodes.new("ShaderNodeTexNoise")
    terrain_noise.name = "Terrain Noise"
    terrain_noise.label = "Terrain Noise"
    terrain_noise.location = (x_step * 2, 260)
    terrain_noise.noise_dimensions = "3D"

    terrain_map = nodes.new("ShaderNodeMapRange")
    terrain_map.name = "Elevation Map Range"
    terrain_map.label = "Noise → Elevation"
    terrain_map.location = (x_step * 3, 260)
    terrain_map.inputs["From Min"].default_value = 0.18
    terrain_map.inputs["From Max"].default_value = 0.82
    terrain_map.clamp = True

    terrain_offset = nodes.new("ShaderNodeCombineXYZ")
    terrain_offset.name = "Terrain Offset XYZ"
    terrain_offset.label = "Terrain Offset XYZ"
    terrain_offset.location = (x_step * 4, 260)

    set_terrain_position = nodes.new("GeometryNodeSetPosition")
    set_terrain_position.location = (x_step * 5, 80)

    points_info = nodes.new("GeometryNodeObjectInfo")
    points_info.location = (x_step * 1, -260)
    points_info.inputs["Object"].default_value = placement_points_obj
    if "As Instance" in points_info.inputs:
        points_info.inputs["As Instance"].default_value = False

    point_position = nodes.new("GeometryNodeInputPosition")
    point_position.location = (x_step * 2, -20)

    point_noise = nodes.new("ShaderNodeTexNoise")
    point_noise.name = "Point Terrain Noise"
    point_noise.label = "Point Terrain Noise"
    point_noise.location = (x_step * 3, -20)
    point_noise.noise_dimensions = "3D"

    point_map = nodes.new("ShaderNodeMapRange")
    point_map.name = "Point Elevation Map Range"
    point_map.label = "Point Noise → Elevation"
    point_map.location = (x_step * 4, -20)
    point_map.inputs["From Min"].default_value = 0.18
    point_map.inputs["From Max"].default_value = 0.82
    point_map.clamp = True

    point_offset = nodes.new("ShaderNodeCombineXYZ")
    point_offset.name = "Point Offset XYZ"
    point_offset.label = "Point Offset XYZ"
    point_offset.location = (x_step * 5, -20)

    set_point_position = nodes.new("GeometryNodeSetPosition")
    set_point_position.location = (x_step * 6, -240)

    attr_instance_id = _named_attribute_node(
        nodes,
        INSTANCE_ID_ATTRIBUTE_NAME,
        "INT",
        (x_step * 5, -520),
    )
    attr_variant_index = _named_attribute_node(
        nodes,
        INSTANCE_VARIANT_INDEX_ATTRIBUTE_NAME,
        "INT",
        (x_step * 6, -560),
    )
    attr_variant_id = _named_attribute_node(
        nodes,
        variant_attribute_name,
        "FLOAT",
        (x_step * 6, -720),
    )
    attr_scale_x = _named_attribute_node(
        nodes,
        INSTANCE_SCALE_X_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 6, 300),
    )
    attr_scale_y = _named_attribute_node(
        nodes,
        INSTANCE_SCALE_Y_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 6, 180),
    )
    attr_scale_z = _named_attribute_node(
        nodes,
        INSTANCE_SCALE_Z_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 6, 60),
    )
    attr_rot_x = _named_attribute_node(
        nodes,
        INSTANCE_ROTATION_X_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 7, -360),
    )
    attr_rot_y = _named_attribute_node(
        nodes,
        INSTANCE_ROTATION_Y_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 7, -480),
    )
    attr_rot_z = _named_attribute_node(
        nodes,
        INSTANCE_ROTATION_Z_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 7, -600),
    )
    attr_leaf_hue = _named_attribute_node(
        nodes,
        INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 7, -840),
    )
    attr_leaf_brightness = _named_attribute_node(
        nodes,
        INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME,
        "FLOAT",
        (x_step * 8, -840),
    )

    set_id = nodes.new("GeometryNodeSetID")
    set_id.location = (x_step * 7, -240)

    store_variant_id = nodes.new("GeometryNodeStoreNamedAttribute")
    store_variant_id.name = "Store Variant ID"
    store_variant_id.label = "Store Variant ID"
    store_variant_id.location = (x_step * 8, -240)
    store_variant_id.data_type = "FLOAT"
    store_variant_id.domain = "POINT"
    if "Name" in store_variant_id.inputs:
        store_variant_id.inputs["Name"].default_value = variant_attribute_name

    store_leaf_hue = nodes.new("GeometryNodeStoreNamedAttribute")
    store_leaf_hue.name = "Store Instance Leaf Hue"
    store_leaf_hue.label = "Store Instance Leaf Hue"
    store_leaf_hue.location = (x_step * 9, -240)
    store_leaf_hue.data_type = "FLOAT"
    store_leaf_hue.domain = "POINT"
    if "Name" in store_leaf_hue.inputs:
        store_leaf_hue.inputs["Name"].default_value = INSTANCE_LEAF_HUE_SHIFT_ATTRIBUTE_NAME

    store_leaf_brightness = nodes.new("GeometryNodeStoreNamedAttribute")
    store_leaf_brightness.name = "Store Instance Leaf Brightness"
    store_leaf_brightness.label = "Store Instance Leaf Brightness"
    store_leaf_brightness.location = (x_step * 10, -240)
    store_leaf_brightness.data_type = "FLOAT"
    store_leaf_brightness.domain = "POINT"
    if "Name" in store_leaf_brightness.inputs:
        store_leaf_brightness.inputs["Name"].default_value = INSTANCE_LEAF_BRIGHTNESS_ATTRIBUTE_NAME

    combine_scale = nodes.new("ShaderNodeCombineXYZ")
    combine_scale.name = "Instance Scale XYZ"
    combine_scale.label = "Instance Scale XYZ"
    combine_scale.location = (x_step * 8, 160)

    combine_rot = nodes.new("ShaderNodeCombineXYZ")
    combine_rot.name = "Rotation XYZ"
    combine_rot.label = "Rotation XYZ"
    combine_rot.location = (x_step * 9, -520)

    euler_to_rot = nodes.new("FunctionNodeEulerToRotation")
    euler_to_rot.location = (x_step * 10, -520)

    col_info = nodes.new("GeometryNodeCollectionInfo")
    col_info.location = (x_step * 9, -700)
    col_info.inputs["Separate Children"].default_value = True
    col_info.inputs["Reset Children"].default_value = True
    col_info.inputs["Collection"].default_value = tree_collection

    instance_on = nodes.new("GeometryNodeInstanceOnPoints")
    instance_on.location = (x_step * 11, -220)
    if "Pick Instance" in instance_on.inputs:
        instance_on.inputs["Pick Instance"].default_value = True

    rotate_instances = nodes.new("GeometryNodeRotateInstances")
    rotate_instances.location = (x_step * 12, -220)

    join_geo = nodes.new("GeometryNodeJoinGeometry")
    join_geo.location = (x_step * 13, 0)
    group_out.location = (x_step * 14, 0)

    links.new(terrain_position.outputs["Position"], terrain_noise.inputs["Vector"])
    links.new(group_in.outputs[sid_terrain_scale], terrain_noise.inputs["Scale"])
    links.new(group_in.outputs[sid_terrain_detail], terrain_noise.inputs["Detail"])
    links.new(group_in.outputs[sid_terrain_rough], terrain_noise.inputs["Roughness"])
    links.new(terrain_noise.outputs["Fac"], terrain_map.inputs["Value"])
    links.new(group_in.outputs[sid_elev_min], terrain_map.inputs["To Min"])
    links.new(group_in.outputs[sid_elev_max], terrain_map.inputs["To Max"])
    links.new(terrain_map.outputs["Result"], terrain_offset.inputs["Z"])
    links.new(group_in.outputs["Geometry"], set_terrain_position.inputs["Geometry"])
    links.new(terrain_offset.outputs["Vector"], set_terrain_position.inputs["Offset"])

    links.new(points_info.outputs["Geometry"], set_point_position.inputs["Geometry"])
    links.new(point_position.outputs["Position"], point_noise.inputs["Vector"])
    links.new(group_in.outputs[sid_terrain_scale], point_noise.inputs["Scale"])
    links.new(group_in.outputs[sid_terrain_detail], point_noise.inputs["Detail"])
    links.new(group_in.outputs[sid_terrain_rough], point_noise.inputs["Roughness"])
    links.new(point_noise.outputs["Fac"], point_map.inputs["Value"])
    links.new(group_in.outputs[sid_elev_min], point_map.inputs["To Min"])
    links.new(group_in.outputs[sid_elev_max], point_map.inputs["To Max"])
    links.new(point_map.outputs["Result"], point_offset.inputs["Z"])
    links.new(point_offset.outputs["Vector"], set_point_position.inputs["Offset"])

    links.new(set_point_position.outputs["Geometry"], set_id.inputs["Geometry"])
    links.new(attr_instance_id.outputs["Attribute"], set_id.inputs["ID"])
    links.new(set_id.outputs["Geometry"], store_variant_id.inputs["Geometry"])
    links.new(attr_variant_id.outputs["Attribute"], store_variant_id.inputs["Value"])
    links.new(store_variant_id.outputs["Geometry"], store_leaf_hue.inputs["Geometry"])
    links.new(attr_leaf_hue.outputs["Attribute"], store_leaf_hue.inputs["Value"])
    links.new(store_leaf_hue.outputs["Geometry"], store_leaf_brightness.inputs["Geometry"])
    links.new(attr_leaf_brightness.outputs["Attribute"], store_leaf_brightness.inputs["Value"])

    links.new(attr_scale_x.outputs["Attribute"], combine_scale.inputs["X"])
    links.new(attr_scale_y.outputs["Attribute"], combine_scale.inputs["Y"])
    links.new(attr_scale_z.outputs["Attribute"], combine_scale.inputs["Z"])
    links.new(attr_rot_x.outputs["Attribute"], combine_rot.inputs["X"])
    links.new(attr_rot_y.outputs["Attribute"], combine_rot.inputs["Y"])
    links.new(attr_rot_z.outputs["Attribute"], combine_rot.inputs["Z"])
    links.new(combine_rot.outputs["Vector"], euler_to_rot.inputs["Euler"])

    links.new(store_leaf_brightness.outputs["Geometry"], instance_on.inputs["Points"])
    links.new(col_info.outputs["Instances"], instance_on.inputs["Instance"])
    links.new(combine_scale.outputs["Vector"], instance_on.inputs["Scale"])
    if "Selection" in instance_on.inputs:
        links.new(group_in.outputs[sid_scatter_density], instance_on.inputs["Selection"])
    if "Instance Index" in instance_on.inputs:
        links.new(attr_variant_index.outputs["Attribute"], instance_on.inputs["Instance Index"])
    else:
        raise RuntimeError(
            "GeometryNodeInstanceOnPoints is missing required 'Instance Index' input. "
            "Cannot map precomputed Botaniq placements to template variants."
        )

    links.new(instance_on.outputs["Instances"], rotate_instances.inputs["Instances"])
    links.new(euler_to_rot.outputs["Rotation"], rotate_instances.inputs["Rotation"])

    links.new(set_terrain_position.outputs["Geometry"], join_geo.inputs["Geometry"])
    links.new(rotate_instances.outputs["Instances"], join_geo.inputs["Geometry"])
    links.new(join_geo.outputs["Geometry"], group_out.inputs["Geometry"])

    modifier = terrain_obj.modifiers.new("TreeScatter", "NODES")
    modifier.node_group = node_group

    return node_group
