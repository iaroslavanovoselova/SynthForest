import json

import numpy as np
from PIL import Image

from synthforest.tree_scatter.postprocess import (
    _bbox_from_mask,
    _mask_to_polygons,
    export_instance_segmentation_coco,
)


def _write_mask(path, array):
    Image.fromarray(array).save(path)


def _reference_coco(
    instance_mask_raw,
    instance_mask_polygon,
    class_id_mask,
    image_file_name,
    category_id_to_name,
    contour_tolerance,
    min_area_px,
):
    image_height, image_width = instance_mask_polygon.shape
    categories = [
        {"id": int(category_id), "name": name, "supercategory": "tree"}
        for category_id, name in sorted(category_id_to_name.items())
    ]

    annotations = []
    annotation_id = 1
    unique_instance_ids = np.unique(instance_mask_polygon)
    unique_instance_ids = unique_instance_ids[unique_instance_ids != 0]
    for instance_id in unique_instance_ids:
        instance_binary_raw = instance_mask_raw == instance_id
        instance_binary_polygon = instance_mask_polygon == instance_id
        if not np.any(instance_binary_polygon):
            continue

        area_px = int(np.count_nonzero(instance_binary_polygon))
        if area_px < min_area_px:
            continue

        class_values = class_id_mask[instance_binary_raw]
        class_values = class_values[class_values > 0]
        if class_values.size == 0:
            class_values = class_id_mask[instance_binary_polygon]
            class_values = class_values[class_values > 0]
        if class_values.size == 0:
            continue

        counts = np.bincount(class_values.astype(np.int64))
        category_id = int(counts.argmax())
        polygons = _mask_to_polygons(
            instance_binary_polygon,
            contour_tolerance=contour_tolerance,
        )
        if not polygons:
            continue

        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": int(category_id),
                "segmentation": polygons,
                "area": float(area_px),
                "bbox": _bbox_from_mask(instance_binary_polygon),
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    return {
        "images": [
            {
                "id": 1,
                "file_name": image_file_name,
                "width": int(image_width),
                "height": int(image_height),
            }
        ],
        "annotations": annotations,
        "categories": categories,
    }


def test_cropped_coco_export_matches_full_frame_reference(tmp_path):
    raw = np.zeros((12, 14), dtype=np.uint16)
    polygon = np.zeros_like(raw)
    semantic = np.zeros_like(raw)

    polygon[0:4, 0:4] = 7
    polygon[2:4, 2:4] = 0
    raw[0:4, 0:4] = 7
    raw[2:4, 2:4] = 0
    semantic[raw == 7] = 2

    polygon[6:9, 8:11] = 12
    polygon[10:12, 11:14] = 12
    raw[6:9, 8:11] = 12
    raw[10:12, 11:14] = 12
    semantic[polygon == 12] = 3
    semantic[6:9, 8:10] = 1

    polygon[5:7, 1:3] = 21
    semantic[polygon == 21] = 4

    raw_path = tmp_path / "raw.tif"
    polygon_path = tmp_path / "polygon.tif"
    semantic_path = tmp_path / "semantic.tif"
    output_path = tmp_path / "annotations.json"
    _write_mask(raw_path, raw)
    _write_mask(polygon_path, polygon)
    _write_mask(semantic_path, semantic)

    categories = {1: "cedar", 2: "palm", 3: "pine", 4: "oak"}
    export_instance_segmentation_coco(
        instance_mask_raw_path=raw_path,
        instance_mask_polygon_path=polygon_path,
        class_id_mask_path=semantic_path,
        output_json_path=output_path,
        image_file_name="rgb_satellite0001.tif",
        category_id_to_name=categories,
        class_id_max_value=10,
        contour_tolerance=0.0,
        min_area_px=1,
    )

    expected = _reference_coco(
        raw,
        polygon,
        semantic,
        "rgb_satellite0001.tif",
        categories,
        contour_tolerance=0.0,
        min_area_px=1,
    )
    actual = json.loads(output_path.read_text(encoding="utf-8"))

    assert actual == expected
