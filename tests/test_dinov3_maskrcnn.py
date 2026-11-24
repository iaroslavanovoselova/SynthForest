import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from synthforest.training.coco_masks import (
    decode_coco_segmentation,
    decode_coco_segmentation_crop,
    encode_binary_mask,
    inspect_coco_mask_coverage,
    segmentation_area,
)
from synthforest.training.config import (
    CheckpointInitializationConfig,
    DetectorTrainingConfig,
    SemiSupervisedConfig,
    TrainingStageConfig,
    UnlabeledRealTrainingConfig,
)


def _binary_mask(height=32, width=32):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[6:20, 8:24] = 1
    return mask


def _mask_config(tmp_path):
    config = DetectorTrainingConfig(
        data_root_path=str(tmp_path / "data"),
        train_dataset_names=["tiny"],
        train_output_path=str(tmp_path / "runs"),
        run_name="mask-smoke",
        model_type="maskrcnn",
        dinov3_use_fake_backbone=True,
        dinov3_embed_dim=8,
        backbone_out_channels=8,
        batch_size=1,
        grad_accumulation_steps=1,
        dataloader_num_workers=0,
        use_amp=False,
        anchor_sizes=[8, 16, 32],
        box_score_thresh=0.0,
        box_predictions_per_image=20,
    )
    config.augmentation.resize_min_sizes_train = [32]
    config.augmentation.resize_min_size_eval = 32
    config.augmentation.max_size = 32
    config.augmentation.crop_prob = 0.0
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.rotation_prob = 0.0
    config.domain_adaptation.enabled = False
    config.wandb.enabled = False
    config.validation.tasks = ["bbox", "segm"]
    config.validation.selection_metric = "bbox_dense_map"
    return config


def _write_mask_dataset(root, *, segmentation, with_height=False):
    raster_root = root / "tiny" / "raster"
    for fold in ("train", "valid"):
        tile_dir = raster_root / "tiles" / fold
        tile_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(80, 120, 70)).save(tile_dir / f"tile_{fold}.tif")
        if with_height:
            height_dir = raster_root / "height_maps" / fold
            height_dir.mkdir(parents=True, exist_ok=True)
            height = _binary_mask().astype(np.float32) * 20.0
            Image.fromarray(height, mode="F").save(height_dir / f"tile_{fold}.tif")
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": f"tile_{fold}.tif",
                    "width": 32,
                    "height": 32,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 6, 16, 14],
                    "area": 224,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            ],
            "categories": [{"id": 1, "name": "tree"}],
        }
        (raster_root / f"raster_coco_{fold}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def _write_quebec_mask_product(root):
    product_name = "20210902_sblz1_p4rtk_rgb"
    product_root = root / "quebec_trees" / product_name
    tile_dir = product_root / "tiles" / "valid"
    tile_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(80, 120, 70)).save(tile_dir / "tile_valid.tif")
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "tile_valid.tif",
                "width": 32,
                "height": 32,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 6, 16, 14],
                "area": 224,
                "iscrowd": 0,
                "segmentation": [[8, 6, 24, 6, 24, 20, 8, 20]],
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }
    (product_root / f"{product_name}_coco_gr0p03_valid.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_maskrcnn_pseudo_labels_keep_masks_aligned_through_filtering():
    torch = pytest.importorskip("torch")
    from synthforest.training.mean_teacher import build_pseudo_label_targets

    masks = torch.zeros((3, 1, 24, 24), dtype=torch.float32)
    masks[0, 0, 2:8, 2:8] = 0.4
    masks[1, 0, 4:10, 4:10] = 0.8
    masks[2, 0, 14:22, 14:22] = 0.9
    output = {
        "boxes": torch.tensor(
            [
                [2.0, 2.0, 8.0, 8.0],
                [4.0, 4.0, 10.0, 10.0],
                [14.0, 14.0, 22.0, 22.0],
            ]
        ),
        "scores": torch.tensor([0.95, 0.9, 0.8]),
        "labels": torch.ones((3,), dtype=torch.int64),
        "masks": masks,
    }
    valid = torch.ones((24, 24), dtype=torch.bool)
    valid[:, 20:] = False
    pseudo = SimpleNamespace(
        score_threshold=0.7,
        min_box_size=2.0,
        min_box_area=4.0,
        max_labels_per_image=2,
        valid_mask_min_fraction=None,
        nms_threshold=None,
        include_empty_images=False,
    )

    indices, targets, stats = build_pseudo_label_targets(
        [output],
        [valid],
        pseudo,
        require_masks=True,
        mask_probability_threshold=0.5,
    )

    assert indices == [0]
    assert stats["pseudo_num_labels_before_filter"] == 3
    assert stats["pseudo_num_labels"] == 2
    target = targets[0]
    assert target["boxes"].tolist() == [
        [4.0, 4.0, 10.0, 10.0],
        [14.0, 14.0, 22.0, 22.0],
    ]
    assert target["masks"].dtype == torch.uint8
    assert target["masks"].shape == (2, 24, 24)
    assert target["masks"][0, 4:10, 4:10].all()
    assert target["masks"][1, 14:22, 14:20].all()
    assert not target["masks"][:, :, 20:].any()

    with pytest.raises(ValueError, match="must contain a masks tensor"):
        build_pseudo_label_targets(
            [{key: value for key, value in output.items() if key != "masks"}],
            [valid],
            pseudo,
            require_masks=True,
        )


def test_coco_mask_coverage_and_decoding_support_all_formats():
    mask = _binary_mask()
    compressed = encode_binary_mask(mask)
    polygon = [[8, 6, 24, 6, 24, 20, 8, 20]]
    uncompressed = {
        "size": [32, 32],
        "counts": _uncompressed_rle_counts(mask),
    }
    images = [{"id": 1, "width": 32, "height": 32}]

    for segmentation in (compressed, polygon, uncompressed):
        payload = {
            "images": images,
            "annotations": [{"id": 1, "image_id": 1, "segmentation": segmentation}],
        }
        coverage = inspect_coco_mask_coverage(
            payload,
            validate_decoding=True,
        )
        assert coverage.status == "complete"
        decoded = decode_coco_segmentation(
            segmentation,
            image_height=32,
            image_width=32,
        )
        assert decoded.shape == (32, 32)
        assert decoded.dtype == np.uint8

    no_masks = inspect_coco_mask_coverage(
        {
            "images": images,
            "annotations": [{"id": 1, "image_id": 1, "segmentation": None}],
        }
    )
    assert no_masks.status == "none"

    malformed = inspect_coco_mask_coverage(
        {
            "images": images,
            "annotations": [{"id": 1, "image_id": 1, "segmentation": [[1, 2, 3]]}],
        }
    )
    assert malformed.status == "partial"

    malformed_compressed = inspect_coco_mask_coverage(
        {
            "images": images,
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": {
                        "size": [32, 32],
                        "counts": "definitely_invalid",
                    },
                }
            ],
        },
        validate_decoding=True,
    )
    assert malformed_compressed.status == "partial"
    assert malformed_compressed.decode_errors == 1

    empty = inspect_coco_mask_coverage({"images": images, "annotations": []})
    assert empty.status == "complete"


def test_coco_mask_crop_decoding_rasterizes_only_crop_extent():
    pytest.importorskip("pycocotools")
    segmentation = [[8, 6, 24, 6, 24, 20, 8, 20]]

    cropped = decode_coco_segmentation_crop(
        segmentation,
        image_height=32,
        image_width=32,
        crop_box=(8, 6, 24, 20),
    )

    assert cropped.shape == (14, 16)
    assert cropped.sum() == 224


def test_mask_dataset_returns_aligned_masks_and_skips_invalid_masks(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset

    root = tmp_path / "mask_data"
    _write_mask_dataset(
        root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
    )
    train_json = root / "tiny" / "raster" / "raster_coco_train.json"
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    payload["annotations"].append(
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 2, 1, 6],
            "area": 6,
            "iscrowd": 0,
            "segmentation": [[0, 2, 0, 8, 0, 2]],
        }
    )
    train_json.write_text(json.dumps(payload), encoding="utf-8")
    config = _mask_config(tmp_path)
    config.data_root_path = str(root)
    with pytest.warns(UserWarning, match=r"skipping 1 annotation.*decode_errors=1"):
        dataset = CocoTileDataset(
            data_root_path=root,
            dataset_names=["tiny"],
            fold="train",
            config=config,
            train=False,
        )
    _, target = dataset[0]
    assert target["masks"].dtype == torch.uint8
    assert target["masks"].shape == (1, 32, 32)
    assert target["masks"].shape[0] == target["boxes"].shape[0]
    assert target["labels"].shape[0] == target["boxes"].shape[0]

    bbox_root = tmp_path / "bbox_data"
    _write_mask_dataset(bbox_root, segmentation=None)
    with pytest.warns(UserWarning, match=r"skipping 1 annotation.*missing=1"):
        bbox_dataset = CocoTileDataset(
            data_root_path=bbox_root,
            dataset_names=["tiny"],
            fold="train",
            config=config,
            train=True,
        )
    _, bbox_target = bbox_dataset[0]
    assert bbox_target["boxes"].shape == (0, 4)
    assert bbox_target["masks"].shape == (0, 32, 32)


def test_mask_training_dataset_decodes_only_cropped_masks(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset

    root = tmp_path / "mask_data"
    raster_root = root / "tiny" / "raster"
    tile_dir = raster_root / "tiles" / "train"
    tile_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(80, 120, 70)).save(tile_dir / "tile_train.tif")
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "tile_train.tif",
                "width": 64,
                "height": 64,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 6, 16, 14],
                "area": 224,
                "iscrowd": 0,
                "segmentation": [[8, 6, 24, 6, 24, 20, 8, 20]],
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }
    (raster_root / "raster_coco_train.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    config = _mask_config(tmp_path)
    config.data_root_path = str(root)
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [32, 32]
    config.augmentation.crop_min_intersection_ratio = 0.1
    config.augmentation.brightness_prob = 0.0
    config.augmentation.contrast_prob = 0.0
    config.augmentation.saturation_prob = 0.0
    config.augmentation.hue_prob = 0.0
    config.augmentation.gaussian_blur_prob = 0.0
    config.augmentation.gaussian_noise_prob = 0.0
    config.augmentation.jpeg_prob = 0.0
    config.augmentation.downsample_upsample_prob = 0.0
    randint_values = iter([32, 0, 0])
    monkeypatch.setattr(
        "synthforest.training.dataset.random.random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "synthforest.training.dataset.random.randint",
        lambda *_: next(randint_values),
    )

    def fail_full_decode(*_args, **_kwargs):
        raise AssertionError("training should not decode full-image masks before crop")

    monkeypatch.setattr(
        "synthforest.training.dataset.decode_coco_segmentation",
        fail_full_decode,
    )

    dataset = CocoTileDataset(
        data_root_path=root,
        dataset_names=["tiny"],
        fold="train",
        config=config,
        train=True,
    )
    _, target = dataset[0]

    assert target["masks"].shape == (1, 32, 32)
    assert target["boxes"].shape == (1, 4)
    assert target["masks"].sum() > 0


def test_mask_height_dataset_keeps_height_and_masks_aligned_through_geometry(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset

    root = tmp_path / "mask_height_data"
    _write_mask_dataset(
        root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
        with_height=True,
    )
    config = _mask_config(tmp_path)
    config.data_root_path = str(root)
    config.height_fusion.enabled = True
    config.height_fusion.generate_missing = False
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [24, 24]
    config.augmentation.crop_min_intersection_ratio = 0.1
    config.augmentation.flip_horizontal = True
    config.augmentation.flip_vertical = True
    config.augmentation.rotation_degrees = 90.0
    config.augmentation.rotation_prob = 1.0
    config.augmentation.brightness_prob = 0.0
    config.augmentation.contrast_prob = 0.0
    config.augmentation.saturation_prob = 0.0
    config.augmentation.hue_prob = 0.0
    config.augmentation.gaussian_blur_prob = 0.0
    config.augmentation.gaussian_noise_prob = 0.0
    config.augmentation.jpeg_prob = 0.0
    config.augmentation.downsample_upsample_prob = 0.0

    randint_values = iter([24, 4, 4])
    monkeypatch.setattr(
        "synthforest.training.dataset.random.random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "synthforest.training.dataset.random.randint",
        lambda *_: next(randint_values),
    )
    monkeypatch.setattr(
        "synthforest.training.dataset.random.uniform",
        lambda *_: 90.0,
    )

    dataset = CocoTileDataset(
        data_root_path=root,
        dataset_names=["tiny"],
        fold="train",
        config=config,
        train=True,
    )
    image, target = dataset[0]

    assert image.shape == (4, 24, 24)
    assert target["masks"].shape == (1, 24, 24)
    assert torch.equal(image[3] >= 10.0, target["masks"][0].bool())
    mask_y, mask_x = torch.where(target["masks"][0] > 0)
    expected_box = torch.tensor(
        [
            [
                mask_x.min(),
                mask_y.min(),
                mask_x.max() + 1,
                mask_y.max() + 1,
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.equal(target["boxes"], expected_box)


def test_lazy_mask_crop_fallback_keeps_valid_pixels(tmp_path, monkeypatch):
    from synthforest.training.dataset import DetectionAugmentor

    config = _mask_config(tmp_path)
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [16, 16]
    config.valid_mask.crop_min_valid_fraction = 0.25
    augmentor = DetectionAugmentor(config, train=True)
    image = Image.new("RGB", (64, 64), color=(80, 120, 70))
    valid_mask = np.zeros((64, 64), dtype=bool)
    valid_mask[60, 60] = True

    randint_values = iter([16, 0, 0] * 10 + [16] * 25)
    monkeypatch.setattr(
        "synthforest.training.dataset.random.random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "synthforest.training.dataset.random.randint",
        lambda *_: next(randint_values),
    )
    monkeypatch.setattr(
        "synthforest.training.dataset.random.randrange",
        lambda *_: 0,
    )

    crop = augmentor._sample_crop_window(
        image,
        valid_mask,
        fallback_to_valid_pixel=True,
    )

    assert crop is not None
    _, _, _, cropped_valid_mask = crop
    assert cropped_valid_mask.any()


def test_mask_augmentations_derive_tight_boxes_and_filter_invalid_pixels(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import DetectionAugmentor

    config = _mask_config(tmp_path)
    config.augmentation.crop_prob = 0.0
    config.augmentation.flip_horizontal = True
    config.augmentation.flip_vertical = True
    config.augmentation.rotation_prob = 0.0
    config.valid_mask.target_min_valid_fraction = 0.4
    augmentor = DetectionAugmentor(config, train=True)
    monkeypatch.setattr(
        "synthforest.training.dataset.random.random",
        lambda: 0.0,
    )

    mask = torch.from_numpy(_binary_mask()).unsqueeze(0)
    valid_mask = np.ones((32, 32), dtype=bool)
    valid_mask[:, :12] = False
    _, _, boxes, labels, _, crowds, masks = augmentor(
        Image.new("RGB", (32, 32), color=(80, 120, 70)),
        valid_mask,
        torch.tensor([[8.0, 6.0, 24.0, 20.0]]),
        torch.tensor([1]),
        torch.tensor([224.0]),
        torch.tensor([0]),
        masks=mask,
    )

    assert labels.tolist() == [1]
    assert crowds.tolist() == [0]
    assert boxes.tolist() == [[8.0, 12.0, 20.0, 26.0]]
    assert masks.shape == (1, 32, 32)
    assert not masks[:, :, 20:].any()


def test_maskrcnn_fake_backbone_trains_populated_and_empty_targets(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import build_dinov3_detector

    config = _mask_config(tmp_path)
    model = build_dinov3_detector(config)
    assert model.backbone.height_fusion is None
    model.train()
    mask = torch.zeros((1, 32, 32), dtype=torch.uint8)
    mask[:, 6:20, 8:24] = 1
    losses = model(
        [torch.rand(3, 32, 32), torch.rand(3, 32, 32)],
        [
            {
                "boxes": torch.tensor([[8.0, 6.0, 24.0, 20.0]]),
                "labels": torch.tensor([1]),
                "masks": mask,
            },
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, 32, 32), dtype=torch.uint8),
            },
        ],
    )
    assert "loss_mask" in losses
    assert torch.isfinite(losses["loss_mask"])
    sum(losses.values()).backward()
    assert any(parameter.grad is not None for parameter in model.roi_heads.mask_head.parameters())


def test_maskrcnn_mean_teacher_pseudo_target_backpropagates_mask_loss(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.mean_teacher import build_pseudo_label_targets
    from synthforest.training.models import build_dinov3_detector

    config = _mask_config(tmp_path)
    model = build_dinov3_detector(config)
    model.train()
    teacher_mask = torch.zeros((1, 1, 32, 32))
    teacher_mask[:, :, 6:20, 8:24] = 0.9
    pseudo = SimpleNamespace(
        score_threshold=0.5,
        min_box_size=2.0,
        min_box_area=4.0,
        max_labels_per_image=20,
        valid_mask_min_fraction=0.6,
        nms_threshold=0.5,
        include_empty_images=False,
    )
    selected, targets, _ = build_pseudo_label_targets(
        [
            {
                "boxes": torch.tensor([[8.0, 6.0, 24.0, 20.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
                "masks": teacher_mask,
            }
        ],
        [torch.ones((32, 32), dtype=torch.bool)],
        pseudo,
        require_masks=True,
        mask_probability_threshold=config.mask_probability_threshold,
    )

    losses = model([torch.rand(3, 32, 32)], targets)

    assert selected == [0]
    assert {
        "loss_box_reg",
        "loss_classifier",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    } <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert any(parameter.grad is not None for parameter in model.roi_heads.mask_head.parameters())


def test_maskrcnn_height_fusion_trains_populated_and_empty_targets(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import build_dinov3_detector

    config = _mask_config(tmp_path)
    config.height_fusion.enabled = True
    config.height_fusion.generate_missing = False
    config.height_fusion.levels = ["p8", "p16"]
    config.height_fusion.encoder_hidden_channels = 8
    config.validate()
    model = build_dinov3_detector(config)
    model.train()

    populated_image = torch.rand(4, 32, 32)
    populated_image[3].mul_(60.0)
    empty_image = torch.rand(4, 32, 32)
    empty_image[3].mul_(60.0)
    mask = torch.zeros((1, 32, 32), dtype=torch.uint8)
    mask[:, 6:20, 8:24] = 1
    losses = model(
        [populated_image, empty_image],
        [
            {
                "boxes": torch.tensor([[8.0, 6.0, 24.0, 20.0]]),
                "labels": torch.tensor([1]),
                "masks": mask,
            },
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, 32, 32), dtype=torch.uint8),
            },
        ],
    )

    expected_losses = {
        "loss_box_reg",
        "loss_classifier",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert expected_losses <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.backbone.height_fusion is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.backbone.height_fusion.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.roi_heads.mask_head.parameters())


def test_perfect_dual_evaluation_and_bbox_only_path():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    from torch.utils.data import DataLoader, Dataset

    from synthforest.training.dataset import collate_detection_batch
    from synthforest.training.evaluation import evaluate_model

    class TinyDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            mask = torch.zeros((1, 16, 16), dtype=torch.uint8)
            mask[:, 3:10, 4:12] = 1
            return torch.zeros((3, 16, 16)), {
                "boxes": torch.tensor([[4.0, 3.0, 12.0, 10.0]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([56.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor([index]),
                "masks": mask,
                "valid_mask": torch.ones((16, 16), dtype=torch.bool),
            }

    class PerfectModel(torch.nn.Module):
        def forward(self, images):
            outputs = []
            for image in images:
                mask = torch.zeros(
                    (1, 1, 16, 16),
                    device=image.device,
                )
                mask[:, :, 3:10, 4:12] = 1.0
                outputs.append(
                    {
                        "boxes": torch.tensor(
                            [[4.0, 3.0, 12.0, 10.0]],
                            device=image.device,
                        ),
                        "labels": torch.tensor([1], device=image.device),
                        "scores": torch.tensor([0.9], device=image.device),
                        "masks": mask,
                    }
                )
            return outputs

    config = DetectorTrainingConfig(
        train_dataset_names=["unused"],
        dinov3_use_fake_backbone=True,
        model_type="maskrcnn",
    )
    config.valid_mask.enabled = False
    config.validation.tasks = ["bbox", "segm"]
    config.validation.selection_metric = "bbox_dense_map"
    config.validate()
    loader = DataLoader(
        TinyDataset(),
        batch_size=1,
        collate_fn=collate_detection_batch,
    )
    metrics = evaluate_model(
        PerfectModel(),
        loader,
        torch.device("cpu"),
        config=config,
    )
    assert metrics["bbox_map"] == pytest.approx(1.0)
    assert metrics["segm_map"] == pytest.approx(1.0)
    assert metrics["segm_dense_map"] == pytest.approx(1.0)
    assert metrics["segm_available"] is True

    config.model_type = "fasterrcnn"
    config.validation.tasks = ["bbox"]
    config.validation.selection_metric = "bbox_dense_map"
    config.validate()

    class BoxModel(PerfectModel):
        def forward(self, images):
            outputs = super().forward(images)
            for output in outputs:
                output.pop("masks")
            return outputs

    bbox_metrics = evaluate_model(
        BoxModel(),
        loader,
        torch.device("cpu"),
        config=config,
    )
    assert bbox_metrics["bbox_map"] == pytest.approx(1.0)
    assert "map" not in bbox_metrics
    assert "dense_map" not in bbox_metrics
    assert bbox_metrics["segm_available"] is False
    assert "segm_map" not in bbox_metrics


def test_segmentation_evaluation_clips_truth_to_valid_pixels():
    torch = pytest.importorskip("torch")
    from synthforest.training.evaluation.segmentation import (
        SegmentationCocoAccumulator,
    )

    valid = torch.ones((8, 8), dtype=torch.bool)
    valid[:, 4:] = False
    accumulator = SegmentationCocoAccumulator(mask_probability_threshold=0.5)
    accumulator.update(
        outputs=[
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "masks": torch.ones((1, 1, 8, 8)),
            }
        ],
        targets=[
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
                "labels": torch.tensor([1]),
                "iscrowd": torch.tensor([0]),
                "masks": torch.ones((1, 8, 8), dtype=torch.uint8),
            }
        ],
        valid_masks=[valid],
    )

    truth = accumulator.ground_truth_annotations[0]
    decoded = decode_coco_segmentation(
        truth["segmentation"],
        image_height=8,
        image_width=8,
    )
    assert decoded[:, :4].all()
    assert not decoded[:, 4:].any()
    assert truth["area"] == 32.0
    assert truth["bbox"] == [0.0, 0.0, 4.0, 8.0]


def test_benchmark_segmentation_writer_metrics_and_eligibility(tmp_path):
    pytest.importorskip("faster_coco_eval")
    from synthforest.training.benchmarking.metrics import (
        tile_level_segmentation_metrics,
    )
    from synthforest.training.benchmarking.runner import (
        _segmentation_benchmark_eligibility,
    )
    from synthforest.training.benchmarking.writers import (
        write_segmentation_prediction_coco,
    )

    mask = _binary_mask(16, 16)
    rle = encode_binary_mask(mask)
    truth = {
        "images": [{"id": 1, "file_name": "tile.tif", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 6, 8, 10],
                "area": segmentation_area(rle),
                "iscrowd": 0,
                "segmentation": rle,
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    prediction = SimpleNamespace(
        image_info=truth["images"][0],
        image_path=tmp_path / "tile.tif",
        boxes_xyxy=[[8, 6, 16, 16]],
        scores=[0.9],
        labels=[1],
        segmentations=[rle],
        mask_areas=[segmentation_area(rle)],
    )
    pred_path = write_segmentation_prediction_coco(
        truth_path,
        [prediction],
        tmp_path / "pred.json",
    )
    metrics = tile_level_segmentation_metrics(
        pred_path,
        truth_path,
        [1, 10, 100],
        ground_resolution=None,
    )
    assert metrics["AP"] == pytest.approx(1.0)

    coverage = inspect_coco_mask_coverage(truth)
    assert _segmentation_benchmark_eligibility(
        enabled=True,
        mask_coverage=coverage,
        predictions=[prediction],
    ) == (True, None)
    bbox_only = SimpleNamespace(segmentations=None)
    eligible, reason = _segmentation_benchmark_eligibility(
        enabled=True,
        mask_coverage=coverage,
        predictions=[bbox_only],
    )
    assert eligible is False
    assert reason == "model_has_no_mask_predictions"


def test_benchmark_inference_encodes_and_clips_predicted_masks(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.benchmarking.inference import (
        TileRecord,
        _prediction_from_output,
    )

    mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    valid = torch.ones((8, 8), dtype=torch.bool)
    valid[:, 4:] = False
    record = TileRecord(
        image_info={
            "id": 1,
            "file_name": "tile.tif",
            "width": 8,
            "height": 8,
        },
        image_path=tmp_path / "tile.tif",
    )
    output = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
        "masks": mask,
    }
    prediction = _prediction_from_output(
        output,
        record,
        valid_mask=valid,
        mask_probability_threshold=0.5,
    )
    assert "masks" not in output
    decoded = decode_coco_segmentation(
        prediction.segmentations[0],
        image_height=8,
        image_width=8,
    )
    assert decoded[:, :4].all()
    assert not decoded[:, 4:].any()
    assert prediction.mask_areas == [32.0]


def test_mixed_benchmark_emits_mask_availability_by_product(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("faster_coco_eval")
    from synthforest.training.benchmarking import inference, runner
    from synthforest.training.benchmarking.config import BenchmarkConfig
    from synthforest.training.benchmarking.inference import ImagePrediction

    products = {}
    for dataset_name, has_mask in (
        ("MaskData", True),
        ("BoxData", False),
    ):
        product_name = f"{dataset_name.lower()}_product"
        product_root = tmp_path / dataset_name / product_name
        tile_dir = product_root / "tiles"
        tile_dir.mkdir(parents=True)
        tile_path = tile_dir / "tile.tif"
        Image.new("RGB", (16, 16), color=(80, 120, 70)).save(tile_path)
        mask = _binary_mask(16, 16)
        rle = encode_binary_mask(mask)
        truth = {
            "images": [
                {
                    "id": 1,
                    "file_name": "tile.tif",
                    "width": 16,
                    "height": 16,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 6, 8, 10],
                    "area": 80,
                    "iscrowd": 0,
                    "segmentation": rle if has_mask else None,
                }
            ],
            "categories": [{"id": 1, "name": "tree"}],
        }
        truth_path = product_root / "truth.json"
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        products[dataset_name] = SimpleNamespace(
            location=dataset_name,
            product_name=product_name,
            tile_dir=tile_dir,
            truth_coco_path=truth_path,
            truth_gpkg_path=None,
            aoi_gpkg_path=None,
        )

    class FakeDatasetSpec:
        tile_level_eval_max_dets = 500
        ground_resolution = None

        def __init__(self, product):
            self.product = product

        def verify_fold(self, **kwargs):
            return None

        def iter_fold(self, **kwargs):
            yield self.product

    specs = {name: FakeDatasetSpec(product) for name, product in products.items()}
    train_config = _mask_config(tmp_path)
    train_config.valid_mask.enabled = False
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()

    monkeypatch.setattr(
        runner,
        "load_checkpoint_model",
        lambda **kwargs: (
            object(),
            train_config,
            {
                "model_type": "maskrcnn",
                "valid_mask_policy_source": "checkpoint",
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "get_dataset_spec",
        lambda name: specs[name],
    )

    def fake_inference(*, truth_coco_path, tile_dir, **kwargs):
        truth = json.loads(truth_coco_path.read_text(encoding="utf-8"))
        mask = _binary_mask(16, 16)
        prediction = ImagePrediction(
            image_info=truth["images"][0],
            image_path=tile_dir / "tile.tif",
            boxes_xyxy=[[8.0, 6.0, 16.0, 16.0]],
            scores=[0.9],
            labels=[1],
            segmentations=[encode_binary_mask(mask)],
            mask_areas=[float(mask.sum())],
        )
        return (
            [prediction],
            ["tile.tif"],
            {
                "num_evaluated_tiles": 1,
                "num_skipped_edge_tiles": 0,
                "num_predictions_before_mask_filter": 1,
                "num_predictions_filtered_invalid": 0,
            },
        )

    monkeypatch.setattr(inference, "run_product_inference", fake_inference)
    config = BenchmarkConfig(
        model_type="maskrcnn",
        checkpoint_path=str(checkpoint_path),
        real_data_root=str(tmp_path),
        output_folder=str(tmp_path / "benchmark_outputs"),
        benchmark_name="mixed",
        datasets=["MaskData", "BoxData"],
        fold_name="valid",
        batch_size=1,
        num_workers=0,
        device="cpu",
    )
    config.wandb.enabled = False
    output_dir = runner.run_benchmark(config)

    with open(
        output_dir / "metrics" / "segmentation_coverage.csv",
        newline="",
        encoding="utf-8",
    ) as file:
        coverage = {row["dataset_name"]: row for row in csv.DictReader(file)}
    assert coverage["MaskData"]["segmentation_eligible"] == "True"
    assert coverage["MaskData"]["status"] == "complete"
    assert coverage["BoxData"]["segmentation_eligible"] == "False"
    assert coverage["BoxData"]["status"] == "none"

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_segmentation_eligible_products"] == 1
    assert summary["num_segmentation_skipped_products"] == 1
    assert summary["num_segmentation_eligible_images"] == 1
    assert summary["num_segmentation_skipped_images"] == 1
    assert (
        output_dir
        / "predictions"
        / "valid"
        / "MaskData"
        / "maskdata_product"
        / "maskdata_product_segm_preds_coco.json"
    ).exists()
    assert not (
        output_dir
        / "predictions"
        / "valid"
        / "BoxData"
        / "boxdata_product"
        / "boxdata_product_segm_preds_coco.json"
    ).exists()


def test_model_type_is_part_of_checkpoint_compatibility():
    from synthforest.training.checkpoint import (
        _assert_architecture_compatible,
    )

    faster = DetectorTrainingConfig(
        train_dataset_names=["unused"],
        dinov3_use_fake_backbone=True,
        model_type="fasterrcnn",
    )
    mask = DetectorTrainingConfig.from_dict(faster.to_dict())
    mask.model_type = "maskrcnn"
    mask.validation.tasks = ["bbox", "segm"]
    mask.validation.selection_metric = "bbox_dense_map"
    mask.validate()
    with pytest.raises(ValueError, match="model_type"):
        _assert_architecture_compatible(faster, mask)


def test_maskrcnn_branch_checkpoint_round_trip_and_cross_type_rejection(
    tmp_path,
):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.branch_checkpoint import (
        BRANCH_CHECKPOINT_SCHEMA_VERSION,
        load_branch_checkpoint,
        save_branch_checkpoint,
    )
    from synthforest.training.models import build_dinov3_detector

    mask_config = _mask_config(tmp_path)
    source_model = build_dinov3_detector(mask_config)
    path = save_branch_checkpoint(
        tmp_path / "mask_branch.pth",
        model=source_model,
        config=mask_config,
        iteration=10,
        stage="detector_only",
        output_dir=tmp_path,
    )
    target_model = build_dinov3_detector(mask_config)
    metadata = load_branch_checkpoint(
        path,
        model=target_model,
        config=mask_config,
        expected_iteration=10,
    )
    assert metadata["iteration"] == 10

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == BRANCH_CHECKPOINT_SCHEMA_VERSION

    faster_config = DetectorTrainingConfig.from_dict(mask_config.to_dict())
    faster_config.model_type = "fasterrcnn"
    faster_config.validation.tasks = ["bbox"]
    faster_config.validation.selection_metric = "bbox_dense_map"
    faster_config.validate()
    faster_model = build_dinov3_detector(faster_config)
    with pytest.raises(ValueError, match="architecture"):
        load_branch_checkpoint(
            path,
            model=faster_model,
            config=faster_config,
        )


def test_maskrcnn_training_loop_writes_bbox_and_segmentation_bests(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.checkpoint import load_checkpoint_model
    from synthforest.training.engine.trainer import train_from_config

    data_root = tmp_path / "data"
    _write_mask_dataset(
        data_root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
    )
    real_root = tmp_path / "real"
    _write_quebec_mask_product(real_root)
    config = _mask_config(tmp_path)
    config.data_root_path = str(data_root)
    config.validation.synthetic.dataset_names = ["tiny"]
    config.validation.real.data_root_path = str(real_root)
    config.validation.real.dataset_names = ["QuebecTrees"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    config.stages = [
        TrainingStageConfig(
            name="detector_only",
            max_iterations=1,
            eval_interval=1,
            checkpoint_interval=1,
            train_log_interval=1,
        )
    ]
    config.validate()

    output_dir = train_from_config(config)

    assert (output_dir / "checkpoints" / "best.pth").exists()
    assert (output_dir / "checkpoints" / "best_bbox.pth").exists()
    assert (output_dir / "checkpoints" / "best_segm.pth").exists()
    metrics = json.loads(
        (output_dir / "metrics" / "validation_latest.json").read_text(encoding="utf-8")
    )
    assert "bbox_dense_map" in metrics["valid_real"]
    assert "segm_dense_map" in metrics["valid_real"]
    checkpoint_payload = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint_payload["schema_version"] == 4
    loaded_model, loaded_config, metadata = load_checkpoint_model(
        output_dir / "checkpoints" / "best.pth",
        device=torch.device("cpu"),
    )
    assert loaded_config.model_type == "maskrcnn"
    assert metadata["model_type"] == "maskrcnn"
    assert loaded_model.roi_heads.mask_predictor is not None


def test_maskrcnn_mean_teacher_training_loop_uses_mask_pseudo_targets(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    data_root = tmp_path / "data"
    _write_mask_dataset(
        data_root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
    )
    real_root = tmp_path / "real"
    _write_quebec_mask_product(real_root)
    config = _mask_config(tmp_path)
    config.run_name = "mask-mean-teacher-smoke"
    config.data_root_path = str(data_root)
    config.validation.synthetic.dataset_names = ["tiny"]
    config.validation.real.data_root_path = str(real_root)
    config.validation.real.dataset_names = ["QuebecTrees"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    config.stages = [
        TrainingStageConfig(
            name="rgb_mean_teacher",
            max_iterations=1,
            eval_interval=1,
            checkpoint_interval=1,
            train_log_interval=1,
            semi_supervised=True,
        )
    ]
    config.semi_supervised = SemiSupervisedConfig(
        enabled=True,
        ema_decay=0.5,
        unsupervised_loss_start_weight=0.25,
        unsupervised_loss_weight=1.0,
        unsupervised_loss_ramp_fraction=0.5,
        unlabeled_real=UnlabeledRealTrainingConfig(
            data_root_path=str(real_root),
            dataset_names=["QuebecTrees"],
            fold="valid",
            max_products_per_dataset=1,
            max_tiles_per_product=1,
        ),
    )
    config.semi_supervised.pseudo_labels.score_threshold = 1.0
    config.semi_supervised.pseudo_labels.include_empty_images = True
    config.validate()

    output_dir = train_from_config(config)

    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    train_metrics = json.loads(
        (output_dir / "metrics" / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert "student_state_dict" in checkpoint
    assert "ema_teacher_state_dict" in checkpoint
    assert train_metrics["train/pseudo_images_used"] == 1.0
    assert "train/pseudo_loss_mask" in train_metrics
    assert np.isfinite(train_metrics["train/pseudo_loss_mask"])


def test_maskrcnn_finetune_training_loads_checkpoint_and_writes_segmentation_bests(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config
    from synthforest.training.models import build_dinov3_detector

    data_root = tmp_path / "data"
    _write_mask_dataset(
        data_root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
    )
    real_root = tmp_path / "real"
    _write_quebec_mask_product(real_root)
    config = _mask_config(tmp_path)
    config.data_root_path = str(data_root)
    config.validation.synthetic.dataset_names = ["tiny"]
    config.validation.real.data_root_path = str(real_root)
    config.validation.real.dataset_names = ["QuebecTrees"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    config.stages = [
        TrainingStageConfig(
            name="detector_only",
            max_iterations=1,
            eval_interval=1,
            checkpoint_interval=1,
            train_log_interval=1,
        )
    ]
    checkpoint_path = tmp_path / "synthetic_mask_best.pth"
    source_model = build_dinov3_detector(config)
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": source_model.state_dict(),
            "training_config": config.to_dict(),
            "run_name": "synthetic-mask-source",
            "iteration": 10,
            "stage": "detector_only",
        },
        checkpoint_path,
    )
    config.checkpoint_initialization = CheckpointInitializationConfig(
        checkpoint_path=str(checkpoint_path)
    )
    config.validate()

    output_dir = train_from_config(config)

    initialization = json.loads(
        (output_dir / "metadata" / "initialization.json").read_text(encoding="utf-8")
    )
    assert initialization["type"] == "full_checkpoint"
    assert initialization["model_type"] == "maskrcnn"
    assert (output_dir / "checkpoints" / "best.pth").exists()
    assert (output_dir / "checkpoints" / "best_bbox.pth").exists()
    assert (output_dir / "checkpoints" / "best_segm.pth").exists()


def _uncompressed_rle_counts(mask):
    flat = np.asarray(mask, order="F").reshape(-1, order="F")
    counts = []
    previous = 0
    run = 0
    for value in flat:
        value = int(value > 0)
        if value == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = value
    counts.append(run)
    return counts


def test_maskrcnn_height_mean_teacher_training_loop(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    data_root = tmp_path / "data"
    _write_mask_dataset(
        data_root,
        segmentation=[[8, 6, 24, 6, 24, 20, 8, 20]],
        with_height=True,
    )
    real_root = tmp_path / "real"
    _write_quebec_mask_product(real_root)
    real_height_dir = (
        real_root / "quebec_trees" / "20210902_sblz1_p4rtk_rgb" / "height_maps" / "valid"
    )
    real_height_dir.mkdir(parents=True)
    height = np.zeros((32, 32), dtype=np.float32)
    height[6:20, 8:24] = 20.0
    Image.fromarray(height, mode="F").save(real_height_dir / "tile_valid.tif")

    config = _mask_config(tmp_path)
    config.run_name = "mask-height-mean-teacher-smoke"
    config.data_root_path = str(data_root)
    config.height_fusion.enabled = True
    config.height_fusion.generate_missing = False
    config.height_fusion.levels = ["p8", "p16"]
    config.height_fusion.encoder_hidden_channels = 8
    config.validation.synthetic.dataset_names = ["tiny"]
    config.validation.real.data_root_path = str(real_root)
    config.validation.real.dataset_names = ["QuebecTrees"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    config.stages = [
        TrainingStageConfig(
            name="height_mean_teacher",
            max_iterations=1,
            eval_interval=1,
            checkpoint_interval=1,
            train_log_interval=1,
            semi_supervised=True,
        )
    ]
    config.semi_supervised = SemiSupervisedConfig(
        enabled=True,
        ema_decay=0.5,
        unsupervised_loss_start_weight=0.25,
        unsupervised_loss_weight=1.0,
        unsupervised_loss_ramp_fraction=0.5,
        unlabeled_real=UnlabeledRealTrainingConfig(
            data_root_path=str(real_root),
            dataset_names=["QuebecTrees"],
            fold="valid",
            max_products_per_dataset=1,
            max_tiles_per_product=1,
        ),
    )
    config.semi_supervised.pseudo_labels.score_threshold = 1.0
    config.semi_supervised.pseudo_labels.include_empty_images = True
    config.validate()

    output_dir = train_from_config(config)

    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    train_metrics = json.loads(
        (output_dir / "metrics" / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert "student_state_dict" in checkpoint
    assert "ema_teacher_state_dict" in checkpoint
    assert checkpoint["training_config"]["height_fusion"]["enabled"] is True
    assert train_metrics["train/pseudo_images_used"] == 1.0
    assert np.isfinite(train_metrics["train/pseudo_loss_mask"])
