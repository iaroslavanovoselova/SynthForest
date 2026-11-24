from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from synthforest.training.checkpoint import _architecture_signature
from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.dataset import CocoTileDataset
from synthforest.training.models.detector import (
    DinoV3FeatureBackbone,
    FakeDinoV3Backbone,
    TrainEvalResizeTransform,
    build_dinov3_detector,
)


def _base_config_dict() -> dict:
    return {
        "data_root_path": "unused",
        "train_dataset_names": ["toy"],
        "model_type": "fasterrcnn",
        "dinov3_use_fake_backbone": True,
        "dinov3_embed_dim": 8,
        "dinov3_patch_size": 8,
        "backbone_out_channels": 16,
        "batch_size": 1,
        "grad_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "anchor_sizes": [[16], [32], [64]],
        "aspect_ratios": [1.0],
        "augmentation": {
            "resize_min_sizes_train": [64],
            "resize_min_size_eval": 64,
            "max_size": 64,
            "crop_prob": 0.0,
            "flip_horizontal": False,
            "flip_vertical": False,
            "rotation_degrees": 0.0,
            "rotation_prob": 0.0,
            "brightness_prob": 0.0,
            "contrast_prob": 0.0,
            "saturation_prob": 0.0,
            "hue_prob": 0.0,
            "gaussian_blur_prob": 0.0,
            "gaussian_noise_prob": 0.0,
            "jpeg_prob": 0.0,
            "downsample_upsample_prob": 0.0,
        },
        "valid_mask": {
            "enabled": False,
        },
        "validation": {
            "selection_metric": "bbox_dense_map",
            "map_max_detection_thresholds": [1, 10, 100],
            "dense_eval_max_detections": 500,
            "tasks": ["bbox"],
            "synthetic": {"dataset_names": ["toy"], "fold": "valid"},
            "real": {"dataset_names": ["SelvaMask"], "fold": "valid"},
        },
        "wandb": {"enabled": False, "required": False},
        "domain_adaptation": {"enabled": False},
    }


def _height_config_dict() -> dict:
    raw = _base_config_dict()
    raw["height_fusion"] = {
        "enabled": True,
        "generate_missing": False,
        "height_checkpoint_path": None,
        "levels": ["p8"],
        "clip_min_m": 0.0,
        "clip_max_m": 60.0,
        "mean_m": 30.0,
        "std_m": 30.0,
        "encoder_hidden_channels": 8,
        "fusion_init_scale": 0.1,
    }
    return raw


def _uncertainty_height_config_dict() -> dict:
    raw = _height_config_dict()
    raw["height_fusion"].update(
        {
            "use_uncertainty": True,
            "uncertainty_log_variance_min": -4.0,
            "uncertainty_log_variance_max": 8.0,
            "uncertainty_reliability_tau": 0.5,
            "uncertainty_reliability_min": 0.1,
            "precompute_match_checkpoint": True,
        }
    )
    return raw


def test_height_fusion_defaults_disabled() -> None:
    config = DetectorTrainingConfig.from_dict(_base_config_dict())

    assert config.height_fusion.enabled is False
    assert _architecture_signature(config)["height_fusion"] == {"enabled": False}


def test_height_fusion_strict_config_and_signature() -> None:
    config = DetectorTrainingConfig.from_dict(_height_config_dict())
    signature = _architecture_signature(config)["height_fusion"]

    assert signature["enabled"] is True
    assert signature["levels"] == ("p8",)

    raw = _height_config_dict()
    raw["height_fusion"]["unknown"] = 1
    with pytest.raises(TypeError):
        DetectorTrainingConfig.from_dict(raw)


def test_uncertainty_height_fusion_config_and_signature() -> None:
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())
    signature = _architecture_signature(config)["height_fusion"]

    assert config.height_fusion.use_uncertainty is True
    assert signature["use_uncertainty"] is True
    assert signature["uncertainty_log_variance_min"] == pytest.approx(-4.0)
    assert signature["uncertainty_log_variance_max"] == pytest.approx(8.0)

    height_only = DetectorTrainingConfig.from_dict(_height_config_dict())
    assert signature != _architecture_signature(height_only)["height_fusion"]


def test_maskrcnn_height_fusion_example_preserves_rgb_default() -> None:
    rgb_config = DetectorTrainingConfig.from_yaml("configs/crowns/training/synthetic.yaml")
    height_config = DetectorTrainingConfig.from_yaml("configs/crowns/training/synthetic_hf.yaml")

    assert rgb_config.model_type == "maskrcnn"
    assert rgb_config.height_fusion.enabled is False
    assert _architecture_signature(rgb_config)["height_fusion"] == {"enabled": False}

    assert height_config.model_type == "maskrcnn"
    assert height_config.height_fusion.enabled is True
    assert height_config.height_fusion.use_uncertainty is False
    assert height_config.height_fusion.levels == ["p8", "p16"]
    assert height_config.validation.tasks == ["bbox", "segm"]
    assert height_config.validation.selection_metric == "segm_dense_map"
    assert height_config.stages[0].max_iterations == 8_000
    assert height_config.post_training_benchmark.config_path == ("crowns/benchmark/maskrcnn")


def test_uncertainty_height_fusion_validation() -> None:
    raw = _base_config_dict()
    raw["height_fusion"] = {
        "enabled": False,
        "use_uncertainty": True,
        "generate_missing": False,
    }
    with pytest.raises(ValueError, match="use_uncertainty"):
        DetectorTrainingConfig.from_dict(raw)

    raw = _uncertainty_height_config_dict()
    raw["model_type"] = "maskrcnn"
    with pytest.raises(ValueError, match="fasterrcnn"):
        DetectorTrainingConfig.from_dict(raw)

    raw = _uncertainty_height_config_dict()
    raw["height_fusion"]["uncertainty_reliability_tau"] = 0.0
    with pytest.raises(ValueError, match="reliability_tau"):
        DetectorTrainingConfig.from_dict(raw)


def _write_toy_height_file(
    path: Path,
    height: np.ndarray,
    *,
    log_variance: np.ndarray | None = None,
    nodata: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if log_variance is None:
        Image.fromarray(height.astype(np.float32), mode="F").save(path)
        return

    rasterio = pytest.importorskip("rasterio")
    profile = {
        "driver": "GTiff",
        "height": int(height.shape[0]),
        "width": int(height.shape[1]),
        "count": 2,
        "dtype": "float32",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.stack([height, log_variance], axis=0).astype(np.float32))
        dst.set_band_description(1, "height_m")
        dst.set_band_description(2, "log_variance_m2")


def _write_toy_coco_product(
    root: Path,
    *,
    with_height: bool = True,
    with_uncertainty: bool = False,
    invalid_pixel: bool = False,
) -> None:
    product = root / "toy" / "product1"
    tile_dir = product / "tiles" / "train"
    height_dir = product / "height_maps" / "train"
    tile_dir.mkdir(parents=True)
    if with_height:
        height_dir.mkdir(parents=True)

    image_name = "toy_tile_train_0_0_4_4_1.tif"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 64
    rgb[..., 1] = 128
    rgb[..., 2] = 192
    Image.fromarray(rgb).save(tile_dir / image_name)
    if with_height:
        height = np.arange(16, dtype=np.float32).reshape(4, 4)
        if with_uncertainty:
            log_variance = np.full((4, 4), 2.0, dtype=np.float32)
            nodata = None
            if invalid_pixel:
                nodata = -9999.0
                height[0, 0] = nodata
                log_variance[0, 0] = nodata
            _write_toy_height_file(
                height_dir / image_name,
                height,
                log_variance=log_variance,
                nodata=nodata,
            )
        else:
            _write_toy_height_file(height_dir / image_name, height)

    coco = {
        "images": [
            {"id": 1, "file_name": image_name, "width": 4, "height": 4},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }
    with (product / "product1_coco_train.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f)


def test_dataset_appends_height_channel(tmp_path: Path) -> None:
    _write_toy_coco_product(tmp_path, with_height=True)
    config = DetectorTrainingConfig.from_dict(_height_config_dict())

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=False,
    )
    image, target = dataset[0]

    assert image.shape == (4, 4, 4)
    assert torch.allclose(image[3], torch.arange(16, dtype=torch.float32).reshape(4, 4))
    assert target["boxes"].shape == (1, 4)


def test_dataset_appends_height_and_uncertainty_channels(tmp_path: Path) -> None:
    _write_toy_coco_product(tmp_path, with_height=True, with_uncertainty=True)
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=False,
    )
    image, target = dataset[0]

    assert image.shape == (5, 4, 4)
    assert torch.allclose(image[3], torch.arange(16, dtype=torch.float32).reshape(4, 4))
    assert torch.allclose(image[4], torch.full((4, 4), 2.0, dtype=torch.float32))
    assert target["boxes"].shape == (1, 4)


def test_dataset_height_only_uses_band_one_from_two_band_map(tmp_path: Path) -> None:
    _write_toy_coco_product(tmp_path, with_height=True, with_uncertainty=True)
    config = DetectorTrainingConfig.from_dict(_height_config_dict())

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=False,
    )
    image, _ = dataset[0]

    assert image.shape == (4, 4, 4)
    assert torch.allclose(image[3], torch.arange(16, dtype=torch.float32).reshape(4, 4))


def test_dataset_fills_invalid_uncertainty_with_max_log_variance(tmp_path: Path) -> None:
    _write_toy_coco_product(
        tmp_path,
        with_height=True,
        with_uncertainty=True,
        invalid_pixel=True,
    )
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=False,
    )
    image, _ = dataset[0]

    assert image[3, 0, 0].item() == pytest.approx(0.0)
    assert image[4, 0, 0].item() == pytest.approx(8.0)


def test_dataset_missing_height_fails_when_generation_disabled(tmp_path: Path) -> None:
    _write_toy_coco_product(tmp_path, with_height=False)
    config = DetectorTrainingConfig.from_dict(_height_config_dict())

    with pytest.raises(FileNotFoundError):
        CocoTileDataset(
            data_root_path=tmp_path,
            dataset_names=["toy"],
            fold="train",
            config=config,
            train=False,
        )


def test_height_rotation_keeps_height_map_aligned(tmp_path: Path, monkeypatch) -> None:
    _write_toy_coco_product(tmp_path, with_height=True)
    raw = _height_config_dict()
    raw["augmentation"]["rotation_degrees"] = 15.0
    raw["augmentation"]["rotation_prob"] = 1.0
    config = DetectorTrainingConfig.from_dict(raw)
    monkeypatch.setattr("random.uniform", lambda low, high: 0.0)

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=True,
    )
    image, _ = dataset[0]

    assert image.shape == (4, 4, 4)
    assert torch.allclose(image[3], torch.arange(16, dtype=torch.float32).reshape(4, 4))


def test_uncertainty_height_rotation_keeps_aux_maps_aligned(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_toy_coco_product(tmp_path, with_height=True, with_uncertainty=True)
    raw = _uncertainty_height_config_dict()
    raw["augmentation"]["rotation_degrees"] = 15.0
    raw["augmentation"]["rotation_prob"] = 1.0
    config = DetectorTrainingConfig.from_dict(raw)
    monkeypatch.setattr("random.uniform", lambda low, high: 0.0)

    dataset = CocoTileDataset(
        data_root_path=tmp_path,
        dataset_names=["toy"],
        fold="train",
        config=config,
        train=True,
    )
    image, _ = dataset[0]

    assert image.shape == (5, 4, 4)
    assert torch.allclose(image[3], torch.arange(16, dtype=torch.float32).reshape(4, 4))
    assert torch.allclose(image[4], torch.full((4, 4), 2.0, dtype=torch.float32))


def test_transform_normalizes_height_fixed_range() -> None:
    config = DetectorTrainingConfig.from_dict(_height_config_dict())
    transform = TrainEvalResizeTransform(
        train_min_sizes=(64,),
        eval_min_size=64,
        max_size=64,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        height_fusion_config=config.height_fusion,
    )
    image = torch.zeros((4, 4, 4), dtype=torch.float32)
    image[3] = 60.0

    normalized = transform.normalize(image)

    assert normalized.shape == image.shape
    assert torch.allclose(normalized[3], torch.ones((4, 4)))


def test_transform_normalizes_uncertainty_to_unit_range() -> None:
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())
    transform = TrainEvalResizeTransform(
        train_min_sizes=(64,),
        eval_min_size=64,
        max_size=64,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        height_fusion_config=config.height_fusion,
    )
    image = torch.zeros((5, 4, 4), dtype=torch.float32)
    image[3] = 60.0
    image[4] = 2.0

    normalized = transform.normalize(image)

    assert normalized.shape == image.shape
    assert torch.allclose(normalized[3], torch.ones((4, 4)))
    assert torch.allclose(normalized[4], torch.full((4, 4), 0.5))


def test_backbone_fuses_height_only_into_p8() -> None:
    config = DetectorTrainingConfig.from_dict(_height_config_dict())
    backbone = DinoV3FeatureBackbone(
        dinov3=FakeDinoV3Backbone(embed_dim=8, patch_size=8),
        embed_dim=8,
        patch_size=8,
        out_channels=16,
        config=config,
    )
    backbone.eval()
    rgb = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    low_height = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    high_height = torch.ones((1, 1, 64, 64), dtype=torch.float32)

    with torch.no_grad():
        low = backbone(torch.cat([rgb, low_height], dim=1))
        high = backbone(torch.cat([rgb, high_height], dim=1))

    assert set(low) == {"p8", "p16", "p32"}
    assert low["p8"].shape[-2:] == (16, 16)
    assert low["p16"].shape[-2:] == (8, 8)
    assert low["p32"].shape[-2:] == (4, 4)
    assert not torch.allclose(low["p8"], high["p8"])
    assert torch.allclose(low["p16"], high["p16"])
    assert torch.allclose(low["p32"], high["p32"])


def test_backbone_reliability_gates_height_features() -> None:
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())
    backbone = DinoV3FeatureBackbone(
        dinov3=FakeDinoV3Backbone(embed_dim=8, patch_size=8),
        embed_dim=8,
        patch_size=8,
        out_channels=16,
        config=config,
    )
    backbone.eval()
    rgb = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    height = torch.ones((1, 1, 64, 64), dtype=torch.float32)
    low_uncertainty = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    high_uncertainty = torch.ones((1, 1, 64, 64), dtype=torch.float32)

    with torch.no_grad():
        low = backbone(torch.cat([rgb, height, low_uncertainty], dim=1))
        high = backbone(torch.cat([rgb, height, high_uncertainty], dim=1))

    assert set(low) == {"p8", "p16", "p32"}
    assert not torch.allclose(low["p8"], high["p8"])
    assert torch.allclose(low["p16"], high["p16"])
    assert torch.allclose(low["p32"], high["p32"])


def test_uncertainty_height_fusion_expects_five_channels() -> None:
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())
    backbone = DinoV3FeatureBackbone(
        dinov3=FakeDinoV3Backbone(embed_dim=8, patch_size=8),
        embed_dim=8,
        patch_size=8,
        out_channels=16,
        config=config,
    )

    with pytest.raises(ValueError, match="5-channel"):
        backbone(torch.zeros((1, 4, 64, 64), dtype=torch.float32))


def test_height_fusion_detector_training_smoke() -> None:
    config = DetectorTrainingConfig.from_dict(_height_config_dict())
    model = build_dinov3_detector(config)
    model.train()
    image = torch.rand((4, 64, 64), dtype=torch.float32)
    image[3] = image[3] * 60.0
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    losses = model([image], [target])

    assert losses
    assert all(torch.isfinite(value) for value in losses.values())


def test_uncertainty_height_fusion_detector_training_smoke() -> None:
    config = DetectorTrainingConfig.from_dict(_uncertainty_height_config_dict())
    model = build_dinov3_detector(config)
    model.train()
    image = torch.rand((5, 64, 64), dtype=torch.float32)
    image[3] = image[3] * 60.0
    image[4] = image[4] * 12.0 - 4.0
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    losses = model([image], [target])

    assert losses
    assert all(torch.isfinite(value) for value in losses.values())
