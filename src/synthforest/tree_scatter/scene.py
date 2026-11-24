from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import bpy

logger = logging.getLogger(__name__)


def _texture_tiling_xy(texture_tiling: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(texture_tiling, (tuple, list)):
        if len(texture_tiling) != 2:
            raise ValueError("Texture tiling must contain exactly two values.")
        tiling_x, tiling_y = texture_tiling
    else:
        tiling_x = texture_tiling
        tiling_y = texture_tiling
    return max(1e-9, float(tiling_x)), max(1e-9, float(tiling_y))


def _terrain_texture_tiling(
    image: bpy.types.Image,
    *,
    terrain_size: float,
    texture_gsd: float,
) -> tuple[float, float]:
    image_width_px, image_height_px = (int(value) for value in image.size)
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError(f"Terrain texture has invalid dimensions: {tuple(image.size)}")

    gsd = float(texture_gsd)
    if gsd <= 0.0:
        raise ValueError(f"Terrain texture GSD must be positive, got {gsd}.")

    terrain_extent = float(terrain_size)
    if terrain_extent <= 0.0:
        raise ValueError(f"Terrain size must be positive, got {terrain_extent}.")

    return (
        terrain_extent / (image_width_px * gsd),
        terrain_extent / (image_height_px * gsd),
    )


def _randomized_tiled_texture_vector(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    source_socket: bpy.types.NodeSocket,
    *,
    tiling: tuple[float, float],
    seed: int,
    allow_quarter_turns: bool,
) -> bpy.types.NodeSocket:
    """Return UVs with a deterministic random orientation for each repeated tile."""
    rng = random.Random(int(seed) + 1709)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-1500.0, 120.0)
    mapping.inputs["Location"].default_value[0] = rng.random()
    mapping.inputs["Location"].default_value[1] = rng.random()
    mapping.inputs["Scale"].default_value[0] = float(tiling[0])
    mapping.inputs["Scale"].default_value[1] = float(tiling[1])
    mapping.inputs["Scale"].default_value[2] = 1.0

    separate = nodes.new("ShaderNodeSeparateXYZ")
    separate.location = (-1260.0, 120.0)

    cell_x = nodes.new("ShaderNodeMath")
    cell_x.operation = "FLOOR"
    cell_x.location = (-1040.0, 300.0)

    cell_y = nodes.new("ShaderNodeMath")
    cell_y.operation = "FLOOR"
    cell_y.location = (-1040.0, 220.0)

    local_x = nodes.new("ShaderNodeMath")
    local_x.operation = "FRACT"
    local_x.location = (-1040.0, 100.0)

    local_y = nodes.new("ShaderNodeMath")
    local_y.operation = "FRACT"
    local_y.location = (-1040.0, 20.0)

    cell_vector = nodes.new("ShaderNodeCombineXYZ")
    cell_vector.location = (-820.0, 280.0)
    cell_vector.inputs["Z"].default_value = rng.uniform(-10000.0, 10000.0)

    tile_random = nodes.new("ShaderNodeTexWhiteNoise")
    tile_random.noise_dimensions = "3D"
    tile_random.location = (-600.0, 280.0)

    variant_scale = nodes.new("ShaderNodeMath")
    variant_scale.operation = "MULTIPLY"
    variant_scale.location = (-380.0, 320.0)
    variant_scale.inputs[1].default_value = 8.0 if allow_quarter_turns else 4.0

    variant_index = nodes.new("ShaderNodeMath")
    variant_index.operation = "FLOOR"
    variant_index.location = (-180.0, 320.0)

    rotation_divide = nodes.new("ShaderNodeMath")
    rotation_divide.operation = "DIVIDE"
    rotation_divide.location = (20.0, 380.0)
    rotation_divide.inputs[1].default_value = 2.0

    rotation_index = nodes.new("ShaderNodeMath")
    rotation_index.operation = "FLOOR"
    rotation_index.location = (220.0, 380.0)

    rotation_angle = nodes.new("ShaderNodeMath")
    rotation_angle.operation = "MULTIPLY"
    rotation_angle.location = (420.0, 380.0)
    rotation_angle.inputs[1].default_value = math.pi * (0.5 if allow_quarter_turns else 1.0)

    mirror_flag = nodes.new("ShaderNodeMath")
    mirror_flag.operation = "MODULO"
    mirror_flag.location = (20.0, 240.0)
    mirror_flag.inputs[1].default_value = 2.0

    local_vector = nodes.new("ShaderNodeCombineXYZ")
    local_vector.location = (-380.0, 60.0)

    rotate = nodes.new("ShaderNodeVectorRotate")
    rotate.rotation_type = "AXIS_ANGLE"
    rotate.location = (640.0, 120.0)
    rotate.inputs["Center"].default_value = (0.5, 0.5, 0.0)
    rotate.inputs["Axis"].default_value = (0.0, 0.0, 1.0)

    separate_rotated = nodes.new("ShaderNodeSeparateXYZ")
    separate_rotated.location = (860.0, 120.0)

    doubled_x = nodes.new("ShaderNodeMath")
    doubled_x.operation = "MULTIPLY"
    doubled_x.location = (1060.0, 220.0)
    doubled_x.inputs[1].default_value = 2.0

    one_minus_doubled_x = nodes.new("ShaderNodeMath")
    one_minus_doubled_x.operation = "SUBTRACT"
    one_minus_doubled_x.location = (1240.0, 220.0)
    one_minus_doubled_x.inputs[0].default_value = 1.0

    mirrored_x_delta = nodes.new("ShaderNodeMath")
    mirrored_x_delta.operation = "MULTIPLY"
    mirrored_x_delta.location = (1420.0, 220.0)

    randomized_x = nodes.new("ShaderNodeMath")
    randomized_x.operation = "ADD"
    randomized_x.location = (1600.0, 220.0)

    randomized_vector = nodes.new("ShaderNodeCombineXYZ")
    randomized_vector.location = (1800.0, 120.0)

    links.new(source_socket, mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], separate.inputs["Vector"])
    links.new(separate.outputs["X"], cell_x.inputs[0])
    links.new(separate.outputs["Y"], cell_y.inputs[0])
    links.new(separate.outputs["X"], local_x.inputs[0])
    links.new(separate.outputs["Y"], local_y.inputs[0])
    links.new(cell_x.outputs["Value"], cell_vector.inputs["X"])
    links.new(cell_y.outputs["Value"], cell_vector.inputs["Y"])
    links.new(cell_vector.outputs["Vector"], tile_random.inputs["Vector"])
    links.new(tile_random.outputs["Value"], variant_scale.inputs[0])
    links.new(variant_scale.outputs["Value"], variant_index.inputs[0])
    links.new(variant_index.outputs["Value"], rotation_divide.inputs[0])
    links.new(rotation_divide.outputs["Value"], rotation_index.inputs[0])
    links.new(rotation_index.outputs["Value"], rotation_angle.inputs[0])
    links.new(variant_index.outputs["Value"], mirror_flag.inputs[0])
    links.new(local_x.outputs["Value"], local_vector.inputs["X"])
    links.new(local_y.outputs["Value"], local_vector.inputs["Y"])
    links.new(local_vector.outputs["Vector"], rotate.inputs["Vector"])
    links.new(rotation_angle.outputs["Value"], rotate.inputs["Angle"])
    links.new(rotate.outputs["Vector"], separate_rotated.inputs["Vector"])
    links.new(separate_rotated.outputs["X"], doubled_x.inputs[0])
    links.new(doubled_x.outputs["Value"], one_minus_doubled_x.inputs[1])
    links.new(one_minus_doubled_x.outputs["Value"], mirrored_x_delta.inputs[0])
    links.new(mirror_flag.outputs["Value"], mirrored_x_delta.inputs[1])
    links.new(separate_rotated.outputs["X"], randomized_x.inputs[0])
    links.new(mirrored_x_delta.outputs["Value"], randomized_x.inputs[1])
    links.new(randomized_x.outputs["Value"], randomized_vector.inputs["X"])
    links.new(separate_rotated.outputs["Y"], randomized_vector.inputs["Y"])
    return randomized_vector.outputs["Vector"]


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def make_material(
    name: str, base_color: tuple[float, ...], roughness: float = 0.8
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBsdfPrincipled":
                bsdf = node
                break
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _mirrored_cropped_texture_vector(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    source_socket: bpy.types.NodeSocket,
    *,
    tiling: tuple[float, float],
    crop_margin_fraction: float,
    location: tuple[float, float],
    rotation_z: float,
    x_offset: float,
) -> bpy.types.NodeSocket:
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (x_offset, 120.0)
    mapping.inputs["Location"].default_value[0] = float(location[0])
    mapping.inputs["Location"].default_value[1] = float(location[1])
    mapping.inputs["Rotation"].default_value[2] = float(rotation_z)
    mapping.inputs["Scale"].default_value[0] = float(tiling[0])
    mapping.inputs["Scale"].default_value[1] = float(tiling[1])
    mapping.inputs["Scale"].default_value[2] = 1.0

    separate = nodes.new("ShaderNodeSeparateXYZ")
    separate.location = (x_offset + 220.0, 120.0)

    combine = nodes.new("ShaderNodeCombineXYZ")
    combine.location = (x_offset + 1260.0, 120.0)

    margin = max(0.0, min(0.25, float(crop_margin_fraction)))
    interior_scale = max(0.001, 1.0 - margin * 2.0)

    def mirror_crop_axis(
        axis_socket: bpy.types.NodeSocket, y_offset: float
    ) -> bpy.types.NodeSocket:
        half = nodes.new("ShaderNodeMath")
        half.operation = "MULTIPLY"
        half.inputs[1].default_value = 0.5
        half.location = (x_offset + 430.0, y_offset)

        fract = nodes.new("ShaderNodeMath")
        fract.operation = "FRACT"
        fract.location = (x_offset + 580.0, y_offset)

        double = nodes.new("ShaderNodeMath")
        double.operation = "MULTIPLY"
        double.inputs[1].default_value = 2.0
        double.location = (x_offset + 730.0, y_offset)

        centered = nodes.new("ShaderNodeMath")
        centered.operation = "SUBTRACT"
        centered.inputs[1].default_value = 1.0
        centered.location = (x_offset + 880.0, y_offset)

        absolute = nodes.new("ShaderNodeMath")
        absolute.operation = "ABSOLUTE"
        absolute.location = (x_offset + 1030.0, y_offset)

        mirrored = nodes.new("ShaderNodeMath")
        mirrored.operation = "SUBTRACT"
        mirrored.inputs[0].default_value = 1.0
        mirrored.location = (x_offset + 1180.0, y_offset)

        crop_scale = nodes.new("ShaderNodeMath")
        crop_scale.operation = "MULTIPLY"
        crop_scale.inputs[1].default_value = interior_scale
        crop_scale.location = (x_offset + 1330.0, y_offset)

        crop_bias = nodes.new("ShaderNodeMath")
        crop_bias.operation = "ADD"
        crop_bias.inputs[1].default_value = margin
        crop_bias.location = (x_offset + 1480.0, y_offset)

        links.new(axis_socket, half.inputs[0])
        links.new(half.outputs["Value"], fract.inputs[0])
        links.new(fract.outputs["Value"], double.inputs[0])
        links.new(double.outputs["Value"], centered.inputs[0])
        links.new(centered.outputs["Value"], absolute.inputs[0])
        links.new(absolute.outputs["Value"], mirrored.inputs[1])
        links.new(mirrored.outputs["Value"], crop_scale.inputs[0])
        links.new(crop_scale.outputs["Value"], crop_bias.inputs[0])
        return crop_bias.outputs["Value"]

    links.new(source_socket, mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], separate.inputs["Vector"])
    links.new(mirror_crop_axis(separate.outputs["X"], 250.0), combine.inputs["X"])
    links.new(mirror_crop_axis(separate.outputs["Y"], -20.0), combine.inputs["Y"])
    return combine.outputs["Vector"]


def make_material_from_image(
    name: str,
    image_path: Path,
    roughness: float = 1.0,
    texture_tiling: float | tuple[float, float] = 1.0,
    seam_fix: bool = False,
    crop_margin_fraction: float = 0.035,
    anti_tile_blend: float = 0.18,
    tile_random_seed: int | None = None,
    tile_allow_quarter_turns: bool = True,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    links = node_tree.links

    image = bpy.data.images.load(str(image_path), check_existing=True)
    tiling = _texture_tiling_xy(texture_tiling)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1700.0, 120.0)

    if seam_fix:
        # Primary sampling keeps the full image extent. Terrain materials add
        # per-tile orientation changes; other callers retain plain mapping.
        if tile_random_seed is None:
            mapping = nodes.new("ShaderNodeMapping")
            mapping.location = (-1500.0, 120.0)
            mapping.inputs["Scale"].default_value[0] = tiling[0]
            mapping.inputs["Scale"].default_value[1] = tiling[1]
            mapping.inputs["Scale"].default_value[2] = 1.0

            links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
            primary_vector = mapping.outputs["Vector"]
        else:
            primary_vector = _randomized_tiled_texture_vector(
                nodes,
                links,
                tex_coord.outputs["Generated"],
                tiling=tiling,
                seed=tile_random_seed,
                allow_quarter_turns=tile_allow_quarter_turns,
            )

        # Secondary sampling keeps the mirrored/cropped variant to help hide seams
        secondary_vector = _mirrored_cropped_texture_vector(
            nodes,
            links,
            tex_coord.outputs["Generated"],
            tiling=(tiling[0] * 0.73, tiling[1] * 0.73),
            crop_margin_fraction=crop_margin_fraction,
            location=(0.37, 0.53),
            rotation_z=math.radians(23.0),
            x_offset=-1500.0,
        )

        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.location = (160.0, 210.0)
        tex_image.image = image
        tex_image.extension = "REPEAT"

        tex_image_alt = nodes.new("ShaderNodeTexImage")
        tex_image_alt.location = (160.0, -80.0)
        tex_image_alt.image = image
        tex_image_alt.extension = "REPEAT"

        mix = nodes.new("ShaderNodeMixRGB")
        mix.location = (420.0, 120.0)
        mix.inputs["Fac"].default_value = max(0.0, min(0.45, float(anti_tile_blend)))

        links.new(primary_vector, tex_image.inputs["Vector"])
        links.new(secondary_vector, tex_image_alt.inputs["Vector"])
        links.new(tex_image.outputs["Color"], mix.inputs["Color1"])
        links.new(tex_image_alt.outputs["Color"], mix.inputs["Color2"])
        color_socket = mix.outputs["Color"]
    else:
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-780.0, 120.0)
        mapping.inputs["Scale"].default_value[0] = tiling[0]
        mapping.inputs["Scale"].default_value[1] = tiling[1]
        mapping.inputs["Scale"].default_value[2] = 1.0

        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.location = (-560.0, 120.0)
        tex_image.image = image
        tex_image.extension = "REPEAT"

        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
        color_socket = tex_image.outputs["Color"]

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (660.0, 120.0)
    bsdf.inputs["Roughness"].default_value = roughness

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (900.0, 120.0)

    links.new(color_socket, bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_material_from_image_with_variation(
    name: str,
    image_path: Path,
    *,
    roughness: float = 1.0,
    texture_tiling: float = 1.0,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    value: float = 1.0,
    contrast: float = 0.0,
    brightness: float = 0.0,
    noise_scale: float = 2.0,
    noise_strength: float = 0.0,
    shadow_color: tuple[float, float, float, float] = (0.18, 0.16, 0.14, 1.0),
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    links = node_tree.links

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1180.0, 120.0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-960.0, 120.0)
    tiling = max(0.001, float(texture_tiling))
    mapping.inputs["Scale"].default_value[0] = tiling
    mapping.inputs["Scale"].default_value[1] = tiling
    mapping.inputs["Scale"].default_value[2] = 1.0

    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.location = (-740.0, 120.0)
    tex_image.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex_image.extension = "REPEAT"

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    hue_sat.location = (-500.0, 120.0)
    hue_sat.inputs["Hue"].default_value = 0.5 + float(hue_shift)
    hue_sat.inputs["Saturation"].default_value = float(saturation)
    hue_sat.inputs["Value"].default_value = float(value)

    bright_contrast = nodes.new("ShaderNodeBrightContrast")
    bright_contrast.location = (-260.0, 120.0)
    bright_contrast.inputs["Bright"].default_value = float(brightness)
    bright_contrast.inputs["Contrast"].default_value = float(contrast)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-500.0, -180.0)
    noise.inputs["Scale"].default_value = max(0.1, float(noise_scale))
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.5

    noise_strength_mult = nodes.new("ShaderNodeMath")
    noise_strength_mult.location = (-260.0, -180.0)
    noise_strength_mult.operation = "MULTIPLY"
    noise_strength_mult.inputs[1].default_value = max(0.0, float(noise_strength)) * 0.35

    variation_mix_fac = nodes.new("ShaderNodeValue")
    variation_mix_fac.location = (-20.0, 260.0)
    variation_mix_fac.outputs[0].default_value = 0.12

    shadow_rgb = nodes.new("ShaderNodeRGB")
    shadow_rgb.location = (-20.0, -120.0)
    shadow_rgb.outputs[0].default_value = shadow_color

    variation_mix = nodes.new("ShaderNodeMixRGB")
    variation_mix.location = (40.0, 120.0)

    noise_mix = nodes.new("ShaderNodeMixRGB")
    noise_mix.location = (280.0, 90.0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (520.0, 90.0)
    bsdf.inputs["Roughness"].default_value = roughness

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (750.0, 90.0)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], hue_sat.inputs["Color"])
    links.new(hue_sat.outputs["Color"], bright_contrast.inputs["Color"])
    links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], noise_strength_mult.inputs[0])
    links.new(tex_image.outputs["Color"], variation_mix.inputs["Color1"])
    links.new(bright_contrast.outputs["Color"], variation_mix.inputs["Color2"])
    links.new(variation_mix_fac.outputs["Value"], variation_mix.inputs["Fac"])
    links.new(noise_strength_mult.outputs["Value"], noise_mix.inputs["Fac"])
    links.new(shadow_rgb.outputs["Color"], noise_mix.inputs["Color1"])
    links.new(variation_mix.outputs["Color"], noise_mix.inputs["Color2"])
    links.new(noise_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_path_blend_material_from_image(
    name: str,
    image_path: Path,
    *,
    roughness: float = 0.9,
    uv_map_name: str = "PathUV",
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    links = node_tree.links

    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.location = (-1080.0, 150.0)
    uv_map.uv_map = uv_map_name

    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.location = (-860.0, 150.0)
    tex_image.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex_image.extension = "REPEAT"

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    hue_sat.location = (-620.0, 220.0)
    hue_sat.inputs["Hue"].default_value = 0.5
    hue_sat.inputs["Saturation"].default_value = 1.0
    hue_sat.inputs["Value"].default_value = 1.0

    bright_contrast = nodes.new("ShaderNodeBrightContrast")
    bright_contrast.location = (-320.0, 180.0)
    bright_contrast.inputs["Bright"].default_value = 0.0
    bright_contrast.inputs["Contrast"].default_value = 0.02

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (320.0, 110.0)
    bsdf.inputs["Roughness"].default_value = roughness

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (560.0, 110.0)

    links.new(uv_map.outputs["UV"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], hue_sat.inputs["Color"])
    links.new(hue_sat.outputs["Color"], bright_contrast.inputs["Color"])
    links.new(bright_contrast.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_water_material_from_image(
    name: str,
    image_path: Path,
    *,
    shallow_tint: tuple[float, float, float, float],
    deep_tint: tuple[float, float, float, float],
    bank_tint: tuple[float, float, float, float],
    roughness: float = 0.22,
    uv_map_name: str = "PathUV",
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    links = node_tree.links

    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.location = (-1120.0, 150.0)
    uv_map.uv_map = uv_map_name

    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.location = (-900.0, 150.0)
    tex_image.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex_image.extension = "REPEAT"

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-900.0, -420.0)
    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.45

    separate_uv = nodes.new("ShaderNodeSeparateXYZ")
    separate_uv.location = (-1020.0, -120.0)

    center_sub = nodes.new("ShaderNodeMath")
    center_sub.location = (-860.0, -140.0)
    center_sub.operation = "SUBTRACT"
    center_sub.inputs[1].default_value = 0.5

    center_abs = nodes.new("ShaderNodeMath")
    center_abs.location = (-700.0, -140.0)
    center_abs.operation = "ABSOLUTE"

    center_scale = nodes.new("ShaderNodeMath")
    center_scale.location = (-540.0, -140.0)
    center_scale.operation = "MULTIPLY"
    center_scale.inputs[1].default_value = 2.0

    bank_ramp = nodes.new("ShaderNodeValToRGB")
    bank_ramp.location = (-360.0, -120.0)
    bank_ramp.color_ramp.elements[0].position = 0.46
    bank_ramp.color_ramp.elements[1].position = 0.88

    shallow_rgb = nodes.new("ShaderNodeRGB")
    shallow_rgb.location = (-640.0, 320.0)
    shallow_rgb.outputs[0].default_value = shallow_tint

    deep_rgb = nodes.new("ShaderNodeRGB")
    deep_rgb.location = (-640.0, 120.0)
    deep_rgb.outputs[0].default_value = deep_tint

    bank_rgb = nodes.new("ShaderNodeRGB")
    bank_rgb.location = (-640.0, -80.0)
    bank_rgb.outputs[0].default_value = bank_tint

    depth_mix = nodes.new("ShaderNodeMixRGB")
    depth_mix.location = (-390.0, 210.0)
    depth_mix.inputs["Fac"].default_value = 0.60

    texture_mix = nodes.new("ShaderNodeMixRGB")
    texture_mix.location = (-140.0, 130.0)
    texture_mix.inputs["Fac"].default_value = 0.72

    center_dark_mix = nodes.new("ShaderNodeMixRGB")
    center_dark_mix.location = (100.0, 150.0)
    center_dark_mix.inputs["Fac"].default_value = 0.08

    center_dark_rgb = nodes.new("ShaderNodeRGB")
    center_dark_rgb.location = (-140.0, 300.0)
    center_dark_rgb.outputs[0].default_value = (0.07, 0.12, 0.14, 1.0)

    soften_mix = nodes.new("ShaderNodeMixRGB")
    soften_mix.location = (340.0, 150.0)
    soften_mix.inputs["Fac"].default_value = 0.18

    soften_rgb = nodes.new("ShaderNodeRGB")
    soften_rgb.location = (100.0, 300.0)
    soften_rgb.outputs[0].default_value = (0.16, 0.20, 0.20, 1.0)

    bank_mix = nodes.new("ShaderNodeMixRGB")
    bank_mix.location = (580.0, 60.0)

    bump = nodes.new("ShaderNodeBump")
    bump.location = (120.0, -220.0)
    bump.inputs["Strength"].default_value = 0.03
    bump.inputs["Distance"].default_value = 0.2

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (820.0, 70.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1050.0, 70.0)

    links.new(uv_map.outputs["UV"], tex_image.inputs["Vector"])
    links.new(uv_map.outputs["UV"], noise.inputs["Vector"])
    links.new(uv_map.outputs["UV"], separate_uv.inputs["Vector"])
    links.new(separate_uv.outputs["Y"], center_sub.inputs[0])
    links.new(center_sub.outputs["Value"], center_abs.inputs[0])
    links.new(center_abs.outputs["Value"], center_scale.inputs[0])
    links.new(center_scale.outputs["Value"], bank_ramp.inputs["Fac"])

    links.new(shallow_rgb.outputs["Color"], depth_mix.inputs["Color1"])
    links.new(deep_rgb.outputs["Color"], depth_mix.inputs["Color2"])

    links.new(depth_mix.outputs["Color"], texture_mix.inputs["Color1"])
    links.new(tex_image.outputs["Color"], texture_mix.inputs["Color2"])

    links.new(center_dark_rgb.outputs["Color"], center_dark_mix.inputs["Color1"])
    links.new(texture_mix.outputs["Color"], center_dark_mix.inputs["Color2"])
    links.new(bank_ramp.outputs["Color"], center_dark_mix.inputs["Fac"])

    links.new(soften_rgb.outputs["Color"], soften_mix.inputs["Color1"])
    links.new(center_dark_mix.outputs["Color"], soften_mix.inputs["Color2"])
    links.new(bank_ramp.outputs["Color"], soften_mix.inputs["Fac"])

    links.new(soften_mix.outputs["Color"], bank_mix.inputs["Color1"])
    links.new(bank_rgb.outputs["Color"], bank_mix.inputs["Color2"])
    links.new(bank_ramp.outputs["Color"], bank_mix.inputs["Fac"])

    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bank_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def create_terrain_object(
    scene: bpy.types.Scene,
    size: float,
    subdivisions: int,
    color: tuple[float, float, float, float],
    texture_image_path: Path | None = None,
    texture_gsd: float = 0.05,
    texture_random_seed: int = 0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=subdivisions,
        y_subdivisions=subdivisions,
        size=size,
        location=(0, 0, 0),
    )
    terrain = bpy.context.active_object
    terrain.name = "Terrain"

    if texture_image_path is not None:
        try:
            image = bpy.data.images.load(str(texture_image_path), check_existing=True)
            texture_tiling = _terrain_texture_tiling(
                image,
                terrain_size=size,
                texture_gsd=texture_gsd,
            )
            ground_mat = make_material_from_image(
                "Ground_Material",
                image_path=texture_image_path,
                roughness=1.0,
                texture_tiling=texture_tiling,
                seam_fix=True,
                tile_random_seed=texture_random_seed,
                tile_allow_quarter_turns=int(image.size[0]) == int(image.size[1]),
            )
            terrain["texture_image_gsd"] = float(texture_gsd)
            terrain["texture_tiles_x"] = float(texture_tiling[0])
            terrain["texture_tiles_y"] = float(texture_tiling[1])
            terrain["texture_image_width_px"] = int(image.size[0])
            terrain["texture_image_height_px"] = int(image.size[1])
            logger.info(
                "Applied terrain texture: %s | image=%sx%s px | gsd=%.6f m/px | tiles=(%.4f, %.4f)",
                texture_image_path,
                int(image.size[0]),
                int(image.size[1]),
                float(texture_gsd),
                texture_tiling[0],
                texture_tiling[1],
            )
        except Exception as exc:
            logger.warning(
                "Failed to apply terrain texture '%s'. Falling back to flat color. Reason: %s",
                texture_image_path,
                exc,
            )
            ground_mat = make_material("Ground_Material", color, roughness=1.0)
    else:
        ground_mat = make_material("Ground_Material", color, roughness=1.0)

    terrain.data.materials.append(ground_mat)
    return terrain


def create_camera(
    scene: bpy.types.Scene,
    terrain_size: float,
    resolution_x: int,
    resolution_y: int,
    gsd_at_nadir: float,
) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("SatelliteCamera")
    cam_data.type = "ORTHO"

    gsd = max(1e-6, float(gsd_at_nadir))
    ortho_width = max(1.0, float(resolution_x)) * gsd
    ortho_height = max(1.0, float(resolution_y)) * gsd
    world_extent = max(float(terrain_size), ortho_width, ortho_height)

    # Blender's orthographic scale is the vertical view size.  For the
    # square satellite chips used here this matches loom's max(width,height)
    # convention, while keeping rectangular outputs metrically exact.
    cam_data.ortho_scale = ortho_height
    cam_data.clip_end = world_extent * 2

    cam_obj = bpy.data.objects.new("SatelliteCamera", cam_data)
    cam_obj.location = (0, 0, world_extent * 0.5)
    cam_obj.rotation_euler = (0, 0, 0)
    cam_obj["gsd_at_nadir"] = gsd
    cam_obj["orthographic_width"] = ortho_width
    cam_obj["orthographic_height"] = ortho_height
    cam_obj["effective_gsd_x"] = ortho_width / max(1.0, float(resolution_x))
    cam_obj["effective_gsd_y"] = ortho_height / max(1.0, float(resolution_y))
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def create_sun_light(scene: bpy.types.Scene, seed: int) -> bpy.types.Object:
    rng = random.Random(seed)

    elevation_deg = rng.uniform(55, 80)
    azimuth_deg = rng.uniform(0, 360)  # full sun rotation

    zenith_deg = 90.0 - elevation_deg

    # --- Create light ---
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light_data.angle = math.radians(2.0)  # softer shadows

    light_obj = bpy.data.objects.new("Sun", light_data)

    # --- Apply SAME mapping as set_sun_angle ---
    light_obj.rotation_euler[0] = math.radians(zenith_deg)
    light_obj.rotation_euler[1] = 0.0
    light_obj.rotation_euler[2] = math.radians(180.0 - azimuth_deg)

    # --- Store metadata ---
    light_obj["sun_elevation_deg"] = float(elevation_deg)
    light_obj["sun_azimuth_deg"] = float(azimuth_deg)
    light_obj["sun_zenith_deg"] = float(zenith_deg)

    scene.collection.objects.link(light_obj)

    logger.info("Sun light: elevation=%.2f°, azimuth=%.2f°", elevation_deg, azimuth_deg)

    return light_obj


def find_layer_collection(
    parent: bpy.types.LayerCollection, name: str
) -> bpy.types.LayerCollection | None:
    if parent.name == name:
        return parent
    for child in parent.children:
        found = find_layer_collection(child, name)
        if found is not None:
            return found
    return None
