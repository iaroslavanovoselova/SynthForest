from __future__ import annotations

import logging

import bpy

from .config import RenderConfig
from .constants import (
    DENSITY_MAP_MATERIAL_NAME,
    DENSITY_MAP_VIEW_LAYER_NAME,
    INSTANCE_ID_MATERIAL_NAME,
    INSTANCE_ID_VIEW_LAYER_NAME,
    VARIANT_ID_MATERIAL_NAME,
    VARIANT_ID_VIEW_LAYER_NAME,
)

logger = logging.getLogger(__name__)


def _set_cycles_compute_backend(backend: str) -> None:
    requested_backend = str(backend).strip().upper()
    if requested_backend == "AUTO":
        return

    try:
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon is None:
            logger.warning(
                "Cycles addon preferences not found; cannot set gpu_backend=%s", requested_backend
            )
            return

        cycles_prefs = cycles_addon.preferences
        cycles_prefs.compute_device_type = requested_backend
        logger.info("Cycles compute backend set to: %s", requested_backend)
    except Exception as exc:
        logger.warning("Failed to set gpu_backend=%s (%s)", requested_backend, exc)


def _enable_cycles_gpu() -> bool:
    try:
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon is None:
            logger.warning("Cycles addon preferences not found; using CPU rendering")
            return False

        cycles_prefs = cycles_addon.preferences
        if hasattr(cycles_prefs, "get_devices"):
            cycles_prefs.get_devices()

        devices = list(getattr(cycles_prefs, "devices", []))
        if not devices:
            logger.warning("No Cycles devices detected; using CPU rendering")
            return False

        gpu_devices = [device for device in devices if getattr(device, "type", "CPU") != "CPU"]
        if not gpu_devices:
            logger.warning("No GPU devices detected by Cycles; using CPU rendering")
            return False

        for device in devices:
            device.use = device in gpu_devices

        for scene in bpy.data.scenes:
            if hasattr(scene, "cycles") and hasattr(scene.cycles, "device"):
                scene.cycles.device = "GPU"

        logger.info(
            "Using GPU rendering on devices: %s",
            ", ".join(getattr(device, "name", "<unknown>") for device in gpu_devices),
        )
        return True
    except Exception as exc:
        logger.warning("Failed to enable Cycles GPU rendering (%s); using CPU", exc)
        return False


def _enable_cycles_cpu() -> None:
    try:
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon is not None:
            cycles_prefs = cycles_addon.preferences
            if hasattr(cycles_prefs, "get_devices"):
                cycles_prefs.get_devices()

            devices = list(getattr(cycles_prefs, "devices", []))
            for device in devices:
                device.use = getattr(device, "type", "CPU") == "CPU"

        for scene in bpy.data.scenes:
            if hasattr(scene, "cycles") and hasattr(scene.cycles, "device"):
                scene.cycles.device = "CPU"
    except Exception as exc:
        logger.warning("Failed to force CPU rendering (%s)", exc)


def _configure_cycles_device(scene: bpy.types.Scene, render_cfg: RenderConfig) -> None:
    requested_device = str(getattr(render_cfg, "device", "AUTO")).strip().upper()
    requested_backend = str(getattr(render_cfg, "gpu_backend", "AUTO")).strip().upper()

    if requested_device not in {"AUTO", "GPU", "CPU"}:
        logger.warning("Unknown render.device='%s'; falling back to AUTO", requested_device)
        requested_device = "AUTO"

    _set_cycles_compute_backend(requested_backend)

    if requested_device == "CPU":
        _enable_cycles_cpu()
        logger.info("Render device: CPU")
        return

    gpu_enabled = _enable_cycles_gpu()
    if gpu_enabled:
        logger.info("Render device: GPU")
        return

    _enable_cycles_cpu()
    if requested_device == "GPU":
        logger.warning("render.device=GPU requested but unavailable; fell back to CPU")
    else:
        logger.info("Render device: CPU (AUTO fallback)")


def ensure_override_material(instance_id_max_value: int) -> bpy.types.Material:
    mat = bpy.data.materials.new(INSTANCE_ID_MATERIAL_NAME)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    link = node_tree.links

    attribute = nodes.new("ShaderNodeAttribute")
    attribute.location = (-420.0, 40.0)
    attribute.attribute_name = "id"
    attribute.attribute_type = "INSTANCER"

    divide = nodes.new("ShaderNodeMath")
    divide.operation = "DIVIDE"
    divide.location = (-260.0, 40.0)
    divide.inputs[1].default_value = float(max(1, instance_id_max_value))

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (-80.0, 40.0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (160.0, 40.0)

    link.new(attribute.outputs["Fac"], divide.inputs[0])
    link.new(divide.outputs["Value"], emission.inputs["Color"])
    link.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def setup_instance_id_view_layer(
    scene: bpy.types.Scene, override_material: bpy.types.Material
) -> bpy.types.ViewLayer:
    view_layer = scene.view_layers.new(INSTANCE_ID_VIEW_LAYER_NAME)
    view_layer.material_override = override_material
    view_layer.samples = 1

    cycles_layer = getattr(view_layer, "cycles", None)
    if cycles_layer is not None and hasattr(cycles_layer, "use_denoising"):
        cycles_layer.use_denoising = False

    view_layer.use = False
    return view_layer


def ensure_variant_id_material(
    variant_id_max_value: int,
    attribute_name: str,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(VARIANT_ID_MATERIAL_NAME)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    link = node_tree.links

    attribute = nodes.new("ShaderNodeAttribute")
    attribute.location = (-420.0, 40.0)
    attribute.attribute_name = attribute_name
    attribute.attribute_type = "INSTANCER"

    divide = nodes.new("ShaderNodeMath")
    divide.operation = "DIVIDE"
    divide.location = (-260.0, 40.0)
    divide.inputs[1].default_value = float(max(1, variant_id_max_value))

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (-80.0, 40.0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (160.0, 40.0)

    link.new(attribute.outputs["Fac"], divide.inputs[0])
    link.new(divide.outputs["Value"], emission.inputs["Color"])
    link.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def setup_variant_id_view_layer(
    scene: bpy.types.Scene, override_material: bpy.types.Material
) -> bpy.types.ViewLayer:
    view_layer = scene.view_layers.new(VARIANT_ID_VIEW_LAYER_NAME)
    view_layer.material_override = override_material
    view_layer.samples = 1

    cycles_layer = getattr(view_layer, "cycles", None)
    if cycles_layer is not None and hasattr(cycles_layer, "use_denoising"):
        cycles_layer.use_denoising = False

    view_layer.use = False
    return view_layer


def ensure_density_map_material(
    density_noise_scale: float,
    density_threshold: float,
    eligibility_attribute_name: str = "tree_eligible",
    density_attribute_name: str | None = None,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(DENSITY_MAP_MATERIAL_NAME)
    mat.use_nodes = True

    node_tree = mat.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    nodes = node_tree.nodes
    lnk = node_tree.links

    if density_attribute_name:
        density_attr = nodes.new("ShaderNodeAttribute")
        density_attr.location = (-420.0, 40.0)
        density_attr.attribute_name = density_attribute_name

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-140.0, 40.0)
        cr = ramp.color_ramp
        cr.interpolation = "LINEAR"
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.0, 0.0, 0.05, 1.0)
        cr.elements[1].position = 1.0
        cr.elements[1].color = (1.0, 0.0, 0.0, 1.0)

        e_blue = cr.elements.new(0.3)
        e_blue.color = (0.0, 0.1, 0.5, 1.0)
        e_green = cr.elements.new(0.5)
        e_green.color = (0.0, 0.6, 0.1, 1.0)
        e_yellow = cr.elements.new(0.7)
        e_yellow.color = (1.0, 0.8, 0.0, 1.0)

        emission = nodes.new("ShaderNodeEmission")
        emission.location = (160.0, 40.0)
        emission.inputs["Strength"].default_value = 1.0

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (380.0, 40.0)

        lnk.new(density_attr.outputs["Fac"], ramp.inputs["Fac"])
        lnk.new(ramp.outputs["Color"], emission.inputs["Color"])
        lnk.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    geo_node = nodes.new("ShaderNodeNewGeometry")
    geo_node.location = (-800.0, 40.0)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-600.0, 40.0)
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = density_noise_scale
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.5

    eligibility = nodes.new("ShaderNodeAttribute")
    eligibility.location = (-600.0, -120.0)
    eligibility.attribute_name = eligibility_attribute_name

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = (-400.0, 40.0)
    map_range.inputs["From Min"].default_value = max(0.0, min(0.99, float(density_threshold)))
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    if hasattr(map_range, "use_clamp"):
        map_range.use_clamp = True
    elif hasattr(map_range, "clamp"):
        map_range.clamp = True

    eligible_mul = nodes.new("ShaderNodeMath")
    eligible_mul.location = (-200.0, 40.0)
    eligible_mul.operation = "MULTIPLY"

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (0.0, 40.0)
    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"

    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.0, 0.0, 0.05, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (1.0, 0.0, 0.0, 1.0)

    e_blue = cr.elements.new(0.3)
    e_blue.color = (0.0, 0.1, 0.5, 1.0)
    e_green = cr.elements.new(0.5)
    e_green.color = (0.0, 0.6, 0.1, 1.0)
    e_yellow = cr.elements.new(0.7)
    e_yellow.color = (1.0, 0.8, 0.0, 1.0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (260.0, 40.0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (460.0, 40.0)

    lnk.new(geo_node.outputs["Position"], noise.inputs["Vector"])
    lnk.new(noise.outputs["Fac"], map_range.inputs["Value"])
    lnk.new(map_range.outputs["Result"], eligible_mul.inputs[0])
    lnk.new(eligibility.outputs["Fac"], eligible_mul.inputs[1])
    lnk.new(eligible_mul.outputs["Value"], ramp.inputs["Fac"])
    lnk.new(ramp.outputs["Color"], emission.inputs["Color"])
    lnk.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def setup_density_map_view_layer(
    scene: bpy.types.Scene, density_material: bpy.types.Material
) -> bpy.types.ViewLayer:
    view_layer = scene.view_layers.new(DENSITY_MAP_VIEW_LAYER_NAME)
    view_layer.material_override = density_material
    view_layer.samples = 1

    cycles_layer = getattr(view_layer, "cycles", None)
    if cycles_layer is not None and hasattr(cycles_layer, "use_denoising"):
        cycles_layer.use_denoising = False

    view_layer.use = False
    return view_layer


def _get_compositor_tree(scene: bpy.types.Scene) -> bpy.types.CompositorNodeTree:
    tree = getattr(scene, "compositing_node_group", None) or scene.node_tree
    if tree is None:
        raise RuntimeError("Failed to access compositor node tree")
    return tree


def _clear_compositor(scene: bpy.types.Scene) -> bpy.types.CompositorNodeTree:
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = _get_compositor_tree(scene)

    if hasattr(tree, "interface"):
        has_image_output = False
        for item in tree.interface.items_tree:
            if item.item_type == "SOCKET" and item.in_out == "OUTPUT" and item.name == "Image":
                has_image_output = True
                break
        if not has_image_output:
            tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    for node in list(tree.nodes):
        tree.nodes.remove(node)
    return tree


def _configure_file_output(
    tree: bpy.types.NodeTree,
    name: str,
    directory: str,
    slot_name: str,
    file_format: str,
    location: tuple[float, float],
    color_mode: str = "RGBA",
    color_depth: str = "8",
) -> tuple[bpy.types.CompositorNodeOutputFile, bpy.types.NodeSocket]:
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = name
    node.label = name
    node.location = location
    node.base_path = directory

    if hasattr(node, "file_output_items"):
        node.file_output_items.clear()
        socket_type = "FLOAT" if color_mode == "BW" else "RGBA"
        item = node.file_output_items.new(socket_type, slot_name)
        item.override_node_format = True
        item.format.file_format = file_format
        try:
            item.format.color_mode = color_mode
        except Exception:
            if color_mode == "BW":
                item.format.color_mode = "RGB"
                logger.warning(
                    "Compositor output '%s' does not support BW for %s; falling back to RGB.",
                    slot_name,
                    file_format,
                )
            else:
                raise
        item.format.color_depth = color_depth
    else:
        node.file_slots.clear()
        node.file_slots.new(slot_name)
        slot = node.file_slots[0]
        slot.use_node_format = False
        slot.format.file_format = file_format
        try:
            slot.format.color_mode = color_mode
        except Exception:
            if color_mode == "BW":
                slot.format.color_mode = "RGB"
                logger.warning(
                    "Compositor output '%s' does not support BW for %s; falling back to RGB.",
                    slot_name,
                    file_format,
                )
            else:
                raise
        slot.format.color_depth = color_depth

    return node, node.inputs[slot_name]


def setup_compositor_rgb(scene: bpy.types.Scene, output_dir: str) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    rl.layer = "ViewLayer"

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (400, 0)

    _, file_socket = _configure_file_output(
        tree,
        "RGB Output",
        output_dir,
        "rgb_satellite",
        file_format="TIFF",
        location=(200, -200),
        color_mode="RGB",
        color_depth="8",
    )

    links.new(rl.outputs["Image"], composite.inputs["Image"])
    links.new(rl.outputs["Image"], file_socket)


def setup_compositor_rgb_height(
    scene: bpy.types.Scene,
    rgb_output_dir: str | None,
    flat_scene: bpy.types.Scene | None,
    height_output_dir: str | None,
    angle_rad: float,
    threshold: float = 1.5,
    max_height: float = 300.0,
    terrain_depth_max: float | None = None,
) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    main_rl = nodes.new("CompositorNodeRLayers")
    main_rl.location = (-900, 140)
    main_rl.layer = "ViewLayer"

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (900, 220)
    links.new(main_rl.outputs["Image"], composite.inputs["Image"])

    if rgb_output_dir is not None:
        _, rgb_socket = _configure_file_output(
            tree,
            "RGB Output",
            rgb_output_dir,
            "rgb_satellite",
            file_format="TIFF",
            location=(720, 20),
            color_mode="RGB",
            color_depth="8",
        )
        links.new(main_rl.outputs["Image"], rgb_socket)

    if flat_scene is None or height_output_dir is None:
        return

    flat_rl = nodes.new("CompositorNodeRLayers")
    flat_rl.location = (-900, -120)
    flat_rl.scene = flat_scene
    flat_rl.layer = "ViewLayer"

    cos_node = nodes.new("CompositorNodeMath")
    cos_node.operation = "COSINE"
    cos_node.location = (-680, 20)
    cos_node.inputs[0].default_value = float(angle_rad)

    depth_main_mul = nodes.new("CompositorNodeMath")
    depth_main_mul.operation = "MULTIPLY"
    depth_main_mul.location = (-480, 140)

    depth_flat_mul = nodes.new("CompositorNodeMath")
    depth_flat_mul.operation = "MULTIPLY"
    depth_flat_mul.location = (-480, -120)

    depth_sub = nodes.new("CompositorNodeMath")
    depth_sub.operation = "SUBTRACT"
    depth_sub.location = (-280, 20)

    depth_gt = nodes.new("CompositorNodeMath")
    depth_gt.operation = "GREATER_THAN"
    depth_gt.location = (-80, 140)
    depth_gt.inputs[1].default_value = float(threshold)

    depth_lt = nodes.new("CompositorNodeMath")
    depth_lt.operation = "LESS_THAN"
    depth_lt.location = (-80, -120)
    depth_lt.inputs[1].default_value = float(max_height)

    valid_range = nodes.new("CompositorNodeMath")
    valid_range.operation = "MULTIPLY"
    valid_range.location = (120, 140)

    valid_mask_socket = valid_range.outputs[0]
    if terrain_depth_max is not None:
        terrain_sane = nodes.new("CompositorNodeMath")
        terrain_sane.operation = "LESS_THAN"
        terrain_sane.location = (120, -120)
        terrain_sane.inputs[1].default_value = float(terrain_depth_max)

        valid_with_terrain = nodes.new("CompositorNodeMath")
        valid_with_terrain.operation = "MULTIPLY"
        valid_with_terrain.location = (320, 140)
        valid_mask_socket = valid_with_terrain.outputs[0]
    else:
        terrain_sane = None
        valid_with_terrain = None

    depth_masked = nodes.new("CompositorNodeMath")
    depth_masked.operation = "MULTIPLY"
    depth_masked.location = (520, 20)

    _, height_socket = _configure_file_output(
        tree,
        "Height Map Output",
        height_output_dir,
        "height_map",
        file_format="OPEN_EXR",
        location=(720, -180),
        color_mode="RGB",
        color_depth="32",
    )

    links.new(main_rl.outputs["Depth"], depth_main_mul.inputs[0])
    links.new(flat_rl.outputs["Depth"], depth_flat_mul.inputs[0])
    links.new(cos_node.outputs[0], depth_main_mul.inputs[1])
    links.new(cos_node.outputs[0], depth_flat_mul.inputs[1])

    links.new(depth_flat_mul.outputs[0], depth_sub.inputs[0])
    links.new(depth_main_mul.outputs[0], depth_sub.inputs[1])

    links.new(depth_sub.outputs[0], depth_gt.inputs[0])
    links.new(depth_sub.outputs[0], depth_lt.inputs[0])
    links.new(depth_gt.outputs[0], valid_range.inputs[0])
    links.new(depth_lt.outputs[0], valid_range.inputs[1])
    if terrain_sane is not None and valid_with_terrain is not None:
        links.new(depth_flat_mul.outputs[0], terrain_sane.inputs[0])
        links.new(valid_range.outputs[0], valid_with_terrain.inputs[0])
        links.new(terrain_sane.outputs[0], valid_with_terrain.inputs[1])
    links.new(depth_sub.outputs[0], depth_masked.inputs[0])
    links.new(valid_mask_socket, depth_masked.inputs[1])

    links.new(depth_masked.outputs[0], height_socket)


def setup_compositor_instance_id(scene: bpy.types.Scene, output_dir: str) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-540, 0)
    rl.layer = INSTANCE_ID_VIEW_LAYER_NAME

    separate = nodes.new("CompositorNodeSeparateColor")
    separate.location = (-240, 0)

    normalize = nodes.new("CompositorNodeNormalize")
    normalize.location = (-20, 140)

    combine = nodes.new("CompositorNodeCombineColor")
    combine.location = (180, 140)

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (420, 140)

    _, file_socket = _configure_file_output(
        tree,
        "Instance ID Output",
        output_dir,
        "instance_id_map",
        file_format="TIFF",
        location=(200, -200),
        color_mode="BW",
        color_depth="16",
    )

    links.new(rl.outputs["Image"], separate.inputs["Image"])
    links.new(separate.outputs["Red"], file_socket)
    links.new(separate.outputs["Red"], normalize.inputs["Value"])
    links.new(normalize.outputs["Value"], combine.inputs["Red"])
    links.new(normalize.outputs["Value"], combine.inputs["Green"])
    links.new(normalize.outputs["Value"], combine.inputs["Blue"])
    links.new(combine.outputs["Image"], composite.inputs["Image"])


def setup_compositor_variant_id(scene: bpy.types.Scene, output_dir: str) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-540, 0)
    rl.layer = VARIANT_ID_VIEW_LAYER_NAME

    separate = nodes.new("CompositorNodeSeparateColor")
    separate.location = (-240, 0)

    normalize = nodes.new("CompositorNodeNormalize")
    normalize.location = (-20, 140)

    combine = nodes.new("CompositorNodeCombineColor")
    combine.location = (180, 140)

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (420, 140)

    _, file_socket = _configure_file_output(
        tree,
        "Variant ID Output",
        output_dir,
        "variant_id_map",
        file_format="TIFF",
        location=(200, -200),
        color_mode="BW",
        color_depth="16",
    )

    links.new(rl.outputs["Image"], separate.inputs["Image"])
    links.new(separate.outputs["Red"], file_socket)
    links.new(separate.outputs["Red"], normalize.inputs["Value"])
    links.new(normalize.outputs["Value"], combine.inputs["Red"])
    links.new(normalize.outputs["Value"], combine.inputs["Green"])
    links.new(normalize.outputs["Value"], combine.inputs["Blue"])
    links.new(combine.outputs["Image"], composite.inputs["Image"])


def setup_compositor_label_maps(
    scene: bpy.types.Scene,
    instance_output_dir: str | None = None,
    variant_output_dir: str | None = None,
) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (520, 120)

    composite_image_socket = None

    if instance_output_dir is not None:
        inst_rl = nodes.new("CompositorNodeRLayers")
        inst_rl.location = (-540, 160)
        inst_rl.layer = INSTANCE_ID_VIEW_LAYER_NAME

        inst_separate = nodes.new("CompositorNodeSeparateColor")
        inst_separate.location = (-240, 160)

        inst_normalize = nodes.new("CompositorNodeNormalize")
        inst_normalize.location = (-20, 300)

        inst_combine = nodes.new("CompositorNodeCombineColor")
        inst_combine.location = (180, 300)

        _, inst_socket = _configure_file_output(
            tree,
            "Instance ID Output",
            instance_output_dir,
            "instance_id_map",
            file_format="TIFF",
            location=(220, -120),
            color_mode="BW",
            color_depth="16",
        )

        links.new(inst_rl.outputs["Image"], inst_separate.inputs["Image"])
        links.new(inst_separate.outputs["Red"], inst_socket)
        links.new(inst_separate.outputs["Red"], inst_normalize.inputs["Value"])
        links.new(inst_normalize.outputs["Value"], inst_combine.inputs["Red"])
        links.new(inst_normalize.outputs["Value"], inst_combine.inputs["Green"])
        links.new(inst_normalize.outputs["Value"], inst_combine.inputs["Blue"])
        composite_image_socket = inst_combine.outputs["Image"]

    if variant_output_dir is not None:
        var_rl = nodes.new("CompositorNodeRLayers")
        var_rl.location = (-540, -160)
        var_rl.layer = VARIANT_ID_VIEW_LAYER_NAME

        var_separate = nodes.new("CompositorNodeSeparateColor")
        var_separate.location = (-240, -160)

        var_normalize = nodes.new("CompositorNodeNormalize")
        var_normalize.location = (-20, -20)

        var_combine = nodes.new("CompositorNodeCombineColor")
        var_combine.location = (180, -20)

        _, var_socket = _configure_file_output(
            tree,
            "Variant ID Output",
            variant_output_dir,
            "variant_id_map",
            file_format="TIFF",
            location=(220, -320),
            color_mode="BW",
            color_depth="16",
        )

        links.new(var_rl.outputs["Image"], var_separate.inputs["Image"])
        links.new(var_separate.outputs["Red"], var_socket)
        links.new(var_separate.outputs["Red"], var_normalize.inputs["Value"])
        links.new(var_normalize.outputs["Value"], var_combine.inputs["Red"])
        links.new(var_normalize.outputs["Value"], var_combine.inputs["Green"])
        links.new(var_normalize.outputs["Value"], var_combine.inputs["Blue"])
        if composite_image_socket is None:
            composite_image_socket = var_combine.outputs["Image"]

    if composite_image_socket is not None:
        links.new(composite_image_socket, composite.inputs["Image"])


def setup_compositor_density_map(scene: bpy.types.Scene, output_dir: str) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    rl.layer = DENSITY_MAP_VIEW_LAYER_NAME

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (400, 0)

    _, file_socket = _configure_file_output(
        tree,
        "Density Map Output",
        output_dir,
        "density_map",
        file_format="PNG",
        location=(200, -200),
        color_mode="RGBA",
        color_depth="8",
    )

    links.new(rl.outputs["Image"], composite.inputs["Image"])
    links.new(rl.outputs["Image"], file_socket)


def setup_compositor_height_map(
    scene: bpy.types.Scene,
    flat_scene: bpy.types.Scene,
    output_dir: str,
    angle_rad: float,
    threshold: float = 1.5,
    max_height: float = 300.0,
    terrain_depth_max: float | None = None,
) -> None:
    tree = _clear_compositor(scene)
    nodes = tree.nodes
    links = tree.links

    main_rl = nodes.new("CompositorNodeRLayers")
    main_rl.location = (-900, 140)
    main_rl.layer = "ViewLayer"

    flat_rl = nodes.new("CompositorNodeRLayers")
    flat_rl.location = (-900, -80)
    flat_rl.scene = flat_scene
    flat_rl.layer = "ViewLayer"

    cos_node = nodes.new("CompositorNodeMath")
    cos_node.operation = "COSINE"
    cos_node.location = (-680, 20)
    cos_node.inputs[0].default_value = float(angle_rad)

    depth_main_mul = nodes.new("CompositorNodeMath")
    depth_main_mul.operation = "MULTIPLY"
    depth_main_mul.location = (-480, 140)

    depth_flat_mul = nodes.new("CompositorNodeMath")
    depth_flat_mul.operation = "MULTIPLY"
    depth_flat_mul.location = (-480, -80)

    depth_sub = nodes.new("CompositorNodeMath")
    depth_sub.operation = "SUBTRACT"
    depth_sub.location = (-280, 20)

    depth_gt = nodes.new("CompositorNodeMath")
    depth_gt.operation = "GREATER_THAN"
    depth_gt.location = (-80, 140)
    depth_gt.inputs[1].default_value = float(threshold)

    depth_lt = nodes.new("CompositorNodeMath")
    depth_lt.operation = "LESS_THAN"
    depth_lt.location = (-80, -80)
    depth_lt.inputs[1].default_value = float(max_height)

    valid_range = nodes.new("CompositorNodeMath")
    valid_range.operation = "MULTIPLY"
    valid_range.location = (120, 140)

    valid_mask_socket = valid_range.outputs[0]
    if terrain_depth_max is not None:
        terrain_sane = nodes.new("CompositorNodeMath")
        terrain_sane.operation = "LESS_THAN"
        terrain_sane.location = (120, -80)
        terrain_sane.inputs[1].default_value = float(terrain_depth_max)

        valid_with_terrain = nodes.new("CompositorNodeMath")
        valid_with_terrain.operation = "MULTIPLY"
        valid_with_terrain.location = (320, 140)
        valid_mask_socket = valid_with_terrain.outputs[0]
    else:
        terrain_sane = None
        valid_with_terrain = None

    depth_masked = nodes.new("CompositorNodeMath")
    depth_masked.operation = "MULTIPLY"
    depth_masked.location = (520, 20)

    _, height_socket = _configure_file_output(
        tree,
        "Height Map Output",
        output_dir,
        "height_map",
        file_format="OPEN_EXR",
        location=(380, -120),
        color_mode="RGB",
        color_depth="32",
    )

    links.new(main_rl.outputs["Depth"], depth_main_mul.inputs[0])
    links.new(flat_rl.outputs["Depth"], depth_flat_mul.inputs[0])
    links.new(cos_node.outputs[0], depth_main_mul.inputs[1])
    links.new(cos_node.outputs[0], depth_flat_mul.inputs[1])

    links.new(depth_flat_mul.outputs[0], depth_sub.inputs[0])
    links.new(depth_main_mul.outputs[0], depth_sub.inputs[1])

    links.new(depth_sub.outputs[0], depth_gt.inputs[0])
    links.new(depth_sub.outputs[0], depth_lt.inputs[0])
    links.new(depth_gt.outputs[0], valid_range.inputs[0])
    links.new(depth_lt.outputs[0], valid_range.inputs[1])
    if terrain_sane is not None and valid_with_terrain is not None:
        links.new(depth_flat_mul.outputs[0], terrain_sane.inputs[0])
        links.new(valid_range.outputs[0], valid_with_terrain.inputs[0])
        links.new(terrain_sane.outputs[0], valid_with_terrain.inputs[1])
    links.new(depth_sub.outputs[0], depth_masked.inputs[0])
    links.new(valid_mask_socket, depth_masked.inputs[1])

    links.new(depth_masked.outputs[0], height_socket)


def configure_render_common(scene: bpy.types.Scene, render_cfg: RenderConfig) -> None:
    scene.render.engine = "CYCLES"
    _configure_cycles_device(scene, render_cfg)
    scene.render.resolution_x = render_cfg.resolution_x
    scene.render.resolution_y = render_cfg.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = bool(getattr(render_cfg, "use_persistent_data", True))


def configure_for_rgb(scene: bpy.types.Scene, render_cfg: RenderConfig) -> None:
    scene.cycles.samples = render_cfg.samples
    scene.render.filter_size = 1.5
    scene.view_settings.view_transform = "Standard"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True

    for vl in scene.view_layers:
        vl.use = vl.name == "ViewLayer"


def configure_for_instance_id(scene: bpy.types.Scene) -> None:
    scene.cycles.samples = 1
    scene.render.filter_size = 0.01
    scene.view_settings.view_transform = "Raw"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"

    for vl in scene.view_layers:
        vl.use = vl.name == INSTANCE_ID_VIEW_LAYER_NAME


def configure_for_variant_id(scene: bpy.types.Scene) -> None:
    scene.cycles.samples = 1
    scene.render.filter_size = 0.01
    scene.view_settings.view_transform = "Raw"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"

    for vl in scene.view_layers:
        vl.use = vl.name == VARIANT_ID_VIEW_LAYER_NAME


def configure_for_density_map(scene: bpy.types.Scene) -> None:
    scene.cycles.samples = 1
    scene.render.filter_size = 1.5
    scene.view_settings.view_transform = "Raw"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = False

    for vl in scene.view_layers:
        vl.use = vl.name == DENSITY_MAP_VIEW_LAYER_NAME


def configure_for_height_map(scene: bpy.types.Scene) -> None:
    scene.cycles.samples = 1
    scene.render.filter_size = 0.01
    scene.view_settings.view_transform = "Raw"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"

    for vl in scene.view_layers:
        vl.use = vl.name == "ViewLayer"


def configure_for_rgb_height(scene: bpy.types.Scene, render_cfg: RenderConfig) -> None:
    scene.cycles.samples = render_cfg.samples
    scene.render.filter_size = 1.5
    scene.view_settings.view_transform = "Standard"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "GAUSSIAN"

    for vl in scene.view_layers:
        vl.use = vl.name == "ViewLayer"


def configure_for_label_maps(
    scene: bpy.types.Scene,
    *,
    enable_instance: bool,
    enable_variant: bool,
) -> None:
    scene.cycles.samples = 1
    scene.render.filter_size = 0.01
    scene.view_settings.view_transform = "Raw"

    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"

    enabled_layers = set()
    if enable_instance:
        enabled_layers.add(INSTANCE_ID_VIEW_LAYER_NAME)
    if enable_variant:
        enabled_layers.add(VARIANT_ID_VIEW_LAYER_NAME)

    for vl in scene.view_layers:
        vl.use = vl.name in enabled_layers
