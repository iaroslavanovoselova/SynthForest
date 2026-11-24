import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from synthforest.training.config import (
    BranchInitializationConfig,
    CheckpointInitializationConfig,
    DetectorTrainingConfig,
    LoRAConfig,
    SemiSupervisedConfig,
    SweepRuntimeConfig,
    TrainingStageConfig,
    TrainSamplingConfig,
    UnlabeledRealTrainingConfig,
    ValidMaskConfig,
)


def _coco_payload(fold, *, two_images=True):
    images = [
        {
            "id": 0,
            "width": 64,
            "height": 64,
            "file_name": f"tile_{fold}_box.tif",
        }
    ]
    if two_images:
        images.append(
            {
                "id": 1,
                "width": 64,
                "height": 64,
                "file_name": f"tile_{fold}_empty.tif",
            }
        )
    return {
        "images": images,
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [12, 10, 28, 30],
                "area": 840,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
    }


def _write_tiny_coco_dataset(root):
    dataset_root = root / "tiny_dataset" / "raster001"
    for fold in ("train", "valid"):
        tiles_dir = dataset_root / "tiles" / fold
        tiles_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(80, 120, 70)).save(tiles_dir / f"tile_{fold}_box.tif")
        Image.new("RGB", (64, 64), color=(20, 30, 40)).save(tiles_dir / f"tile_{fold}_empty.tif")
        (dataset_root / f"raster001_coco_sf1p0_{fold}.json").write_text(
            json.dumps(_coco_payload(fold)),
            encoding="utf-8",
        )
    return root


def _write_tiny_real_product(root):
    product_name = "20210902_sblz1_p4rtk_rgb"
    product_root = root / "quebec_trees" / product_name
    tiles_dir = product_root / "tiles" / "valid"
    tiles_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(80, 120, 70)).save(tiles_dir / "tile_valid_box.tif")
    (product_root / f"{product_name}_coco_gr0p03_valid.json").write_text(
        json.dumps(_coco_payload("valid", two_images=False)),
        encoding="utf-8",
    )
    return root


def _write_tiny_unlabeled_real_product(root):
    product_name = "20210902_sblz1_p4rtk_rgb"
    product_root = root / "quebec_trees" / product_name
    tiles_dir = product_root / "tiles" / "valid"
    tiles_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(90, 120, 80)).save(tiles_dir / "tile_unlabeled_a.tif")
    Image.new("RGB", (64, 64), color=(60, 100, 70)).save(tiles_dir / "tile_unlabeled_b.tif")
    return root


def _write_detector_height_maps_for_tiles(root, *, channels=1):
    if channels >= 2:
        rasterio = pytest.importorskip("rasterio")
    else:
        rasterio = None
    for tile_dir in Path(root).rglob("tiles/*"):
        if not tile_dir.is_dir():
            continue
        fold = tile_dir.name
        height_dir = tile_dir.parent.parent / "height_maps" / fold
        height_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(tile_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {
                ".tif",
                ".tiff",
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            height_map = np.arange(width * height, dtype=np.float32).reshape(height, width)
            output_path = height_dir / f"{image_path.stem}.tif"
            if channels >= 2:
                log_variance = np.full((height, width), 2.0, dtype=np.float32)
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=2,
                    dtype="float32",
                ) as dataset:
                    dataset.write(np.stack([height_map, log_variance], axis=0))
            else:
                Image.fromarray(height_map, mode="F").save(output_path)


def _tiny_config(tmp_path):
    data_root = _write_tiny_coco_dataset(tmp_path / "data")
    real_root = _write_tiny_real_product(tmp_path / "real")
    config = DetectorTrainingConfig(
        data_root_path=str(data_root),
        train_dataset_names=["tiny_dataset"],
        train_output_path=str(tmp_path / "runs"),
        run_name="smoke",
        model_type="fasterrcnn",
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
        stages=[
            TrainingStageConfig(
                name="detector_only",
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
            )
        ],
    )
    config.augmentation.resize_min_sizes_train = [64]
    config.augmentation.resize_min_size_eval = 64
    config.augmentation.max_size = 64
    config.domain_adaptation.enabled = False
    config.wandb.enabled = False
    config.validation.synthetic.dataset_names = ["tiny_dataset"]
    config.validation.real.data_root_path = str(real_root)
    config.validation.real.dataset_names = ["QuebecTrees"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    return config


def _skip_if_peft_unavailable():
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed in this environment")
    try:
        import peft  # noqa: F401
    except Exception as exc:
        pytest.skip(f"peft cannot be imported in this environment: {exc}")


def _enable_lora(config, *, staged=True):
    config.lora = LoRAConfig(
        num_blocks=2,
        rank=2,
        alpha=4,
        target_modules=["q_proj", "v_proj"],
    )
    if staged:
        config.stages = [
            TrainingStageConfig(
                name="detector_only",
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
            ),
            TrainingStageConfig(
                name="joint_lora",
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
                train_lora=True,
            ),
        ]
        config.stages[1].optimizer.detector_lr = 5e-5
        config.stages[1].optimizer.lora_lr = 1e-5
    else:
        config.stages = [
            TrainingStageConfig(
                name="joint_lora",
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
                train_lora=True,
            )
        ]
        config.stages[0].optimizer.lora_lr = 1e-5
    return config


def test_canonical_config_parses_nested_yaml(tmp_path):
    config = _tiny_config(tmp_path)
    config.run_name = None
    config.checkpoint_initialization = CheckpointInitializationConfig(
        checkpoint_path=str(tmp_path / "synthetic_best.pth")
    )
    config.train_sampling = TrainSamplingConfig(fraction=0.25, seed=7, min_samples=1)
    config_path = tmp_path / "config.yaml"
    config.save_yaml(config_path)

    parsed = DetectorTrainingConfig.from_yaml(config_path)

    assert parsed.lora is None
    assert parsed.total_iterations == 1
    assert parsed.stages[0].optimizer.detector_lr == pytest.approx(1e-4)
    assert parsed.validation.selection_metric == "bbox_dense_map"
    assert parsed.validation.real.dataset_names == ["QuebecTrees"]
    assert parsed.resolved_synthetic_validation_names() == ["tiny_dataset"]
    assert parsed.valid_mask.enabled is True
    assert parsed.valid_mask.prediction_min_valid_fraction == pytest.approx(0.6)
    assert parsed.post_training_benchmark.enabled is False
    assert parsed.post_training_benchmark.checkpoint_name == "last.pth"
    assert parsed.checkpoint_initialization is not None
    assert parsed.checkpoint_initialization.checkpoint_path.endswith("synthetic_best.pth")
    assert parsed.train_sampling.fraction == pytest.approx(0.25)
    assert parsed.train_sampling.seed == 7


def test_semi_supervised_loss_weight_ramps_by_stage_fraction(tmp_path):
    from synthforest.training.engine.trainer import _semi_supervised_loss_weight

    config = _tiny_config(tmp_path)
    config.semi_supervised = SemiSupervisedConfig(
        enabled=True,
        unsupervised_loss_start_weight=0.1,
        unsupervised_loss_weight=0.5,
        unsupervised_loss_ramp_fraction=0.6,
    )
    stage = TrainingStageConfig(
        name="mean_teacher",
        max_iterations=5,
        semi_supervised=True,
    )

    assert _semi_supervised_loss_weight(
        config,
        stage=stage,
        stage_iteration=1,
    ) == pytest.approx(0.1)
    assert _semi_supervised_loss_weight(
        config,
        stage=stage,
        stage_iteration=2,
    ) == pytest.approx(0.3)
    assert _semi_supervised_loss_weight(
        config,
        stage=stage,
        stage_iteration=3,
    ) == pytest.approx(0.5)
    assert _semi_supervised_loss_weight(
        config,
        stage=stage,
        stage_iteration=4,
    ) == pytest.approx(0.5)


def test_finetune_config_validation_rejects_invalid_initialization_and_sampling(tmp_path):
    config = _tiny_config(tmp_path / "empty_checkpoint")
    config.checkpoint_initialization = CheckpointInitializationConfig(checkpoint_path="")
    with pytest.raises(ValueError, match="checkpoint_initialization.checkpoint_path"):
        config.validate()

    config = _tiny_config(tmp_path / "fraction")
    config.train_sampling = TrainSamplingConfig(fraction=0.0)
    with pytest.raises(ValueError, match="train_sampling.fraction"):
        config.validate()

    config = _tiny_config(tmp_path / "min_samples")
    config.train_sampling = TrainSamplingConfig(fraction=0.5, min_samples=0)
    with pytest.raises(ValueError, match="train_sampling.min_samples"):
        config.validate()

    config = _tiny_config(tmp_path / "mutual")
    config.initialization = BranchInitializationConfig(checkpoint_path="branch.pth")
    config.checkpoint_initialization = CheckpointInitializationConfig(checkpoint_path="full.pth")
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_valid_mask_config_validation(tmp_path):
    config = _tiny_config(tmp_path)
    config.valid_mask = ValidMaskConfig(source="unsupported")
    with pytest.raises(ValueError, match="valid_mask.source"):
        config.validate()

    config.valid_mask = ValidMaskConfig(prediction_min_valid_fraction=1.1)
    with pytest.raises(ValueError, match="prediction_min_valid_fraction"):
        config.validate()


def test_post_training_benchmark_config_validation(tmp_path):
    config = _tiny_config(tmp_path)
    config.post_training_benchmark.enabled = True
    with pytest.raises(ValueError, match="config_path"):
        config.validate()

    config.post_training_benchmark.enabled = False
    config.post_training_benchmark.checkpoint_name = "../best.pth"
    with pytest.raises(ValueError, match="checkpoint_name"):
        config.validate()


def test_model_type_is_required_in_canonical_config(tmp_path):
    payload = _tiny_config(tmp_path).to_dict()
    payload.pop("model_type")

    with pytest.raises(ValueError, match="model_type"):
        DetectorTrainingConfig.from_dict(payload)


def test_legacy_mask_aware_config_is_rejected(tmp_path):
    config = _tiny_config(tmp_path)
    payload = config.to_dict()
    payload["valid_mask"].pop("prediction_min_valid_fraction")
    payload["valid_mask"].update(
        {
            "require_anchor_center_valid": True,
            "feature_location_min_valid_fraction": 0.5,
            "anchor_min_valid_fraction": 0.5,
            "proposal_min_valid_fraction": 0.6,
            "detection_min_valid_fraction": 0.7,
        }
    )

    with pytest.raises(TypeError, match="Unknown ValidMaskConfig field"):
        DetectorTrainingConfig.from_dict(payload)


def test_legacy_single_stage_config_is_rejected(tmp_path):
    base = {
        "data_root_path": "data",
        "train_dataset_names": ["tiny"],
        "valid_dataset_names": ["tiny"],
        "dinov3_use_fake_backbone": True,
        "freeze_backbone": True,
        "max_iterations": 12,
        "eval_interval": 4,
        "checkpoint_interval": 4,
        "train_log_interval": 2,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_iterations": 3,
            "min_lr_factor": 0.01,
        },
        "lora": {"enabled": False},
    }
    with pytest.raises(TypeError, match="Unknown DetectorTrainingConfig"):
        DetectorTrainingConfig.from_dict(base)


def test_post_training_benchmark_disabled_does_nothing(tmp_path, monkeypatch):
    from synthforest.training.engine import trainer as trainer_module
    from synthforest.training.layout import TrainingRunLayout

    config = _tiny_config(tmp_path)
    layout = TrainingRunLayout(tmp_path / "run").create()

    def fail_if_called(_config):
        raise AssertionError("Disabled benchmark hook must not call run_benchmark.")

    monkeypatch.setattr(
        "synthforest.training.benchmarking.run_benchmark",
        fail_if_called,
    )

    trainer_module._run_post_training_benchmark(config, layout)

    assert not layout.post_training_benchmark_metadata.exists()


def test_post_training_benchmark_passes_run_folder_checkpoint_and_records_success(
    tmp_path,
    monkeypatch,
):
    from synthforest.training.engine import trainer as trainer_module
    from synthforest.training.layout import TrainingRunLayout

    benchmark_config_path = tmp_path / "benchmark.yaml"
    benchmark_config_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "fasterrcnn",
                "datasets": ["QuebecTrees"],
                "checkpoint_path": None,
            }
        ),
        encoding="utf-8",
    )
    config = _tiny_config(tmp_path)
    config.post_training_benchmark.enabled = True
    config.post_training_benchmark.config_path = str(benchmark_config_path)
    config.post_training_benchmark.checkpoint_name = "best_bbox.pth"
    layout = TrainingRunLayout(tmp_path / "runs" / "fasterrcnn" / "smoke").create()
    captured = {}

    def fake_run_benchmark(benchmark_config):
        captured["model_type"] = benchmark_config.model_type
        captured["run_folder"] = benchmark_config.run_folder
        captured["checkpoint_name"] = benchmark_config.checkpoint_name
        captured["checkpoint_path"] = benchmark_config.checkpoint_path
        return layout.benchmark_root("smoke_benchmark")

    monkeypatch.setattr(
        "synthforest.training.benchmarking.run_benchmark",
        fake_run_benchmark,
    )

    trainer_module._run_post_training_benchmark(config, layout)

    assert captured == {
        "model_type": "fasterrcnn",
        "run_folder": str(layout.root),
        "checkpoint_name": "best_bbox.pth",
        "checkpoint_path": None,
    }
    metadata = json.loads(layout.post_training_benchmark_metadata.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["checkpoint_name"] == "best_bbox.pth"
    assert metadata["benchmark_output_path"].endswith("benchmarks/smoke_benchmark")


def test_post_training_benchmark_warn_only_failure_records_metadata(
    tmp_path,
    monkeypatch,
):
    from synthforest.training.engine import trainer as trainer_module
    from synthforest.training.layout import TrainingRunLayout

    benchmark_config_path = tmp_path / "benchmark.yaml"
    benchmark_config_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "fasterrcnn",
                "datasets": ["QuebecTrees"],
                "checkpoint_path": None,
            }
        ),
        encoding="utf-8",
    )
    config = _tiny_config(tmp_path)
    config.post_training_benchmark.enabled = True
    config.post_training_benchmark.config_path = str(benchmark_config_path)
    layout = TrainingRunLayout(tmp_path / "run").create()

    def fail_benchmark(_config):
        raise RuntimeError("benchmark exploded")

    monkeypatch.setattr(
        "synthforest.training.benchmarking.run_benchmark",
        fail_benchmark,
    )

    trainer_module._run_post_training_benchmark(config, layout)

    metadata = json.loads(layout.post_training_benchmark_metadata.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"
    assert "benchmark exploded" in metadata["error"]


def test_config_rejects_inconsistent_lora_stages(tmp_path):
    config = _tiny_config(tmp_path / "missing")
    config.stages[0].train_lora = True
    with pytest.raises(ValueError, match="requires a lora configuration"):
        config.validate()

    config = _tiny_config(tmp_path / "unused")
    config.lora = LoRAConfig(num_blocks=2)
    with pytest.raises(ValueError, match="requires at least one stage"):
        config.validate()

    config = _enable_lora(_tiny_config(tmp_path / "backwards"))
    config.stages.append(TrainingStageConfig(name="frozen_again", train_lora=False))
    with pytest.raises(ValueError, match="not back to false"):
        config.validate()


def test_run_name_describes_stage_learning_rates(tmp_path, monkeypatch):
    config = _enable_lora(_tiny_config(tmp_path))
    config.run_name = None
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    run_name = config.resolved_run_name()

    assert "det1-dlr1e-4" in run_name
    assert "joint1-dlr5e-5-llr1e-5" in run_name
    assert "lora2b_r2" in run_name
    assert run_name.endswith("_job12345")


def test_finetune_run_name_includes_fraction_without_dataset_names(tmp_path, monkeypatch):
    config = _tiny_config(tmp_path)
    config.run_name = None
    config.train_dataset_names = ["selvamask", "bamforests"]
    config.checkpoint_initialization = CheckpointInitializationConfig(
        checkpoint_path=str(tmp_path / "synthetic_best.pth")
    )
    config.train_sampling = TrainSamplingConfig(fraction=0.25, seed=7, min_samples=1)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    run_name = config.resolved_run_name()

    assert "ft_frac0p25" in run_name
    assert "selvamask" not in run_name
    assert "bamforests" not in run_name


def test_finetune_run_name_omits_long_dataset_names(tmp_path, monkeypatch):
    config = _tiny_config(tmp_path)
    config.run_name = None
    config.train_dataset_names = [
        "botaniq_tree_scatter_hard_negative_sparse_context_20260701_115916",
        "botaniq_tree_scatter_large_closed_canopy_20260701_114608",
        "botaniq_tree_scatter_small_medium_crowns_20260701_120917",
        "botaniq_tree_scatter_v2_small_medium_crowns_20260616_181018_overlap0p1",
    ]
    config.checkpoint_initialization = CheckpointInitializationConfig(
        checkpoint_path=str(tmp_path / "synthetic_best.pth")
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    run_name = config.resolved_run_name()

    assert len(run_name) < 255
    assert "frac1" in run_name
    assert "botaniq_tree_scatter" not in run_name


def test_coco_tile_dataset_loads_box_and_empty_tile(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset

    config = _tiny_config(tmp_path)
    dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=False,
    )

    assert len(dataset) == 2
    assert dataset[0][1]["boxes"].shape == (1, 4)
    assert dataset[1][1]["boxes"].shape == (0, 4)
    assert dataset[0][1]["valid_mask"].dtype == torch.bool

    train_dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )
    assert "valid_mask" not in train_dataset[0][1]


def test_unlabeled_real_tile_dataset_loads_without_coco_and_returns_paired_views(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import (
        UnlabeledRealTileDataset,
        UnlabeledRealTileSource,
    )

    root = _write_tiny_unlabeled_real_product(tmp_path / "real")
    product_name = "20210902_sblz1_p4rtk_rgb"
    config = _tiny_config(tmp_path / "config")
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [32, 32]
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.rotation_prob = 0.0
    config.augmentation.brightness_prob = 0.0
    config.augmentation.gaussian_blur_prob = 0.0
    config.augmentation.gaussian_noise_prob = 0.0
    config.augmentation.jpeg_prob = 0.0
    config.augmentation.downsample_upsample_prob = 0.0
    config.semi_supervised.strong_augmentation.brightness = 1.0
    config.semi_supervised.strong_augmentation.brightness_prob = 1.0
    config.semi_supervised.strong_augmentation.contrast_prob = 0.0
    config.semi_supervised.strong_augmentation.saturation_prob = 0.0
    config.semi_supervised.strong_augmentation.hue_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_blur_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_noise_prob = 0.0
    config.semi_supervised.strong_augmentation.jpeg_prob = 0.0
    config.semi_supervised.strong_augmentation.downsample_upsample_prob = 0.0
    monkeypatch.setattr(
        "synthforest.training.dataset.random.uniform",
        lambda *_: 0.5,
    )
    dataset = UnlabeledRealTileDataset(
        data_root_path=root,
        sources=[
            UnlabeledRealTileSource(
                name="QuebecTrees/20210902_sblz1_p4rtk_rgb",
                tile_dir=root / "quebec_trees" / product_name / "tiles" / "valid",
            )
        ],
        config=config,
        max_tiles_per_source=1,
    )

    weak, strong, metadata = dataset[0]

    assert len(dataset) == 1
    assert weak.shape == strong.shape == (3, 32, 32)
    assert torch.mean(torch.abs(strong - weak)) > 0.05
    assert metadata["valid_mask"].shape == torch.Size([32, 32])
    assert metadata["source_name"] == "QuebecTrees/20210902_sblz1_p4rtk_rgb"


def test_unlabeled_real_tile_dataset_appends_height_to_paired_views(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import (
        UnlabeledRealTileDataset,
        UnlabeledRealTileSource,
    )

    root = _write_tiny_unlabeled_real_product(tmp_path / "real")
    _write_detector_height_maps_for_tiles(root, channels=1)
    product_name = "20210902_sblz1_p4rtk_rgb"
    config = _tiny_config(tmp_path / "config")
    config.height_fusion.enabled = True
    config.height_fusion.generate_missing = False
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [32, 32]
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.rotation_prob = 0.0
    config.augmentation.brightness_prob = 0.0
    config.semi_supervised.strong_augmentation.brightness = 1.0
    config.semi_supervised.strong_augmentation.brightness_prob = 1.0
    config.semi_supervised.strong_augmentation.contrast_prob = 0.0
    config.semi_supervised.strong_augmentation.saturation_prob = 0.0
    config.semi_supervised.strong_augmentation.hue_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_blur_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_noise_prob = 0.0
    config.semi_supervised.strong_augmentation.jpeg_prob = 0.0
    config.semi_supervised.strong_augmentation.downsample_upsample_prob = 0.0
    monkeypatch.setattr(
        "synthforest.training.dataset.random.uniform",
        lambda *_: 0.5,
    )
    dataset = UnlabeledRealTileDataset(
        data_root_path=root,
        sources=[
            UnlabeledRealTileSource(
                name="QuebecTrees/20210902_sblz1_p4rtk_rgb",
                tile_dir=root / "quebec_trees" / product_name / "tiles" / "valid",
            )
        ],
        config=config,
        max_tiles_per_source=1,
    )

    weak, strong, metadata = dataset[0]

    assert weak.shape == strong.shape == (4, 32, 32)
    assert torch.mean(torch.abs(strong[:3] - weak[:3])) > 0.05
    assert torch.allclose(weak[3], strong[3])
    assert metadata["valid_mask"].shape == torch.Size([32, 32])


def test_unlabeled_real_tile_dataset_appends_uncertainty_to_paired_views(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import (
        UnlabeledRealTileDataset,
        UnlabeledRealTileSource,
    )

    root = _write_tiny_unlabeled_real_product(tmp_path / "real")
    _write_detector_height_maps_for_tiles(root, channels=2)
    product_name = "20210902_sblz1_p4rtk_rgb"
    config = _tiny_config(tmp_path / "config")
    config.height_fusion.enabled = True
    config.height_fusion.use_uncertainty = True
    config.height_fusion.generate_missing = False
    config.augmentation.crop_prob = 1.0
    config.augmentation.crop_size_range = [32, 32]
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.rotation_prob = 0.0
    config.augmentation.brightness_prob = 0.0
    config.semi_supervised.strong_augmentation.brightness = 1.0
    config.semi_supervised.strong_augmentation.brightness_prob = 1.0
    config.semi_supervised.strong_augmentation.contrast_prob = 0.0
    config.semi_supervised.strong_augmentation.saturation_prob = 0.0
    config.semi_supervised.strong_augmentation.hue_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_blur_prob = 0.0
    config.semi_supervised.strong_augmentation.gaussian_noise_prob = 0.0
    config.semi_supervised.strong_augmentation.jpeg_prob = 0.0
    config.semi_supervised.strong_augmentation.downsample_upsample_prob = 0.0
    monkeypatch.setattr(
        "synthforest.training.dataset.random.uniform",
        lambda *_: 0.5,
    )
    dataset = UnlabeledRealTileDataset(
        data_root_path=root,
        sources=[
            UnlabeledRealTileSource(
                name="QuebecTrees/20210902_sblz1_p4rtk_rgb",
                tile_dir=root / "quebec_trees" / product_name / "tiles" / "valid",
            )
        ],
        config=config,
        max_tiles_per_source=1,
    )

    weak, strong, metadata = dataset[0]

    assert weak.shape == strong.shape == (5, 32, 32)
    assert torch.mean(torch.abs(strong[:3] - weak[:3])) > 0.05
    assert torch.allclose(weak[3:], strong[3:])
    assert torch.allclose(weak[4], torch.full_like(weak[4], 2.0))
    assert metadata["valid_mask"].shape == torch.Size([32, 32])


def test_train_sampling_selects_deterministic_record_fraction(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset
    from synthforest.training.engine.trainer import _apply_train_sampling

    config = _tiny_config(tmp_path)
    config.train_sampling = TrainSamplingConfig(fraction=0.5, seed=7, min_samples=1)
    dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )

    sampled, metadata = _apply_train_sampling(dataset, config)
    sampled_again, metadata_again = _apply_train_sampling(dataset, config)

    assert len(dataset) == 2
    assert len(sampled) == 1
    assert len(sampled_again) == 1
    assert metadata == metadata_again
    assert metadata["total_samples"] == 2
    assert metadata["selected_samples"] == 1
    assert metadata["fraction"] == pytest.approx(0.5)
    assert metadata["seed"] == 7
    assert metadata["indices"] == metadata_again["indices"]


def test_train_sampling_samples_fraction_per_dataset(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset
    from synthforest.training.engine.trainer import _apply_train_sampling

    data_root = tmp_path / "data"
    for dataset_name in ("selvamask", "bamforests"):
        dataset_root = data_root / dataset_name / "raster001"
        tiles_dir = dataset_root / "tiles" / "train"
        tiles_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(80, 120, 70)).save(tiles_dir / "tile_train_box.tif")
        Image.new("RGB", (64, 64), color=(20, 30, 40)).save(tiles_dir / "tile_train_empty.tif")
        (dataset_root / "raster001_coco_train.json").write_text(
            json.dumps(_coco_payload("train")),
            encoding="utf-8",
        )

    config = _tiny_config(tmp_path / "config")
    config.data_root_path = str(data_root)
    config.train_dataset_names = ["selvamask", "bamforests"]
    config.train_sampling = TrainSamplingConfig(fraction=0.5, seed=7, min_samples=1)
    dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )

    sampled, metadata = _apply_train_sampling(dataset, config)

    assert len(dataset) == 4
    assert len(sampled) == 2
    assert metadata["selected_samples"] == 2
    assert metadata["sampling_unit"] == "records_per_dataset"
    assert set(metadata["per_dataset"]) == {"selvamask", "bamforests"}
    assert metadata["per_dataset"]["selvamask"]["total_samples"] == 2
    assert metadata["per_dataset"]["selvamask"]["selected_samples"] == 1
    assert metadata["per_dataset"]["bamforests"]["total_samples"] == 2
    assert metadata["per_dataset"]["bamforests"]["selected_samples"] == 1


def test_train_sampling_fraction_one_preserves_full_dataset(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.dataset import CocoTileDataset
    from synthforest.training.engine.trainer import _apply_train_sampling

    config = _tiny_config(tmp_path)
    config.train_sampling = TrainSamplingConfig(fraction=1.0, seed=7, min_samples=1)
    dataset = CocoTileDataset(
        data_root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        config=config,
        train=True,
    )

    sampled, metadata = _apply_train_sampling(dataset, config)

    assert sampled is dataset
    assert len(sampled) == len(dataset) == 2
    assert metadata["indices"] == [0, 1]
    assert metadata["selected_samples"] == 2


def test_pseudo_label_filtering_applies_score_geometry_valid_mask_topk_and_empty_policy():
    torch = pytest.importorskip("torch")
    from synthforest.training.mean_teacher import build_pseudo_label_targets

    output = {
        "boxes": torch.tensor(
            [
                [1.0, 1.0, 8.0, 8.0],
                [10.0, 10.0, 11.0, 20.0],
                [15.0, 15.0, 28.0, 28.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        ),
        "scores": torch.tensor([0.95, 0.99, 0.2, 0.9]),
        "labels": torch.tensor([1, 1, 1, 1]),
    }
    valid_mask = torch.ones((48, 48), dtype=torch.bool)
    valid_mask[30:40, 30:40] = False
    pseudo_config = SimpleNamespace(
        score_threshold=0.7,
        min_box_size=2.0,
        min_box_area=20.0,
        max_labels_per_image=1,
        valid_mask_min_fraction=0.6,
        nms_threshold=None,
        include_empty_images=False,
    )

    indices, targets, stats = build_pseudo_label_targets(
        [output],
        [valid_mask],
        pseudo_config,
    )

    assert indices == [0]
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[0]["boxes"][0].tolist() == pytest.approx([1.0, 1.0, 8.0, 8.0])
    assert stats["pseudo_num_labels_before_filter"] == 4
    assert stats["pseudo_num_labels"] == 1
    assert stats["pseudo_images_used"] == 1
    assert stats["pseudo_images_skipped"] == 0

    pseudo_config.score_threshold = 1.0
    indices, targets, stats = build_pseudo_label_targets(
        [output],
        [valid_mask],
        pseudo_config,
    )
    assert indices == []
    assert targets == []
    assert stats["pseudo_images_skipped"] == 1


def test_ema_teacher_copy_update_and_freeze():
    torch = pytest.importorskip("torch")
    from synthforest.training.mean_teacher import (
        copy_ema_teacher_weights,
        update_ema_teacher,
    )

    student = torch.nn.Linear(2, 1)
    teacher = torch.nn.Linear(2, 1)
    with torch.no_grad():
        student.weight.fill_(2.0)
        student.bias.fill_(1.0)
        teacher.weight.zero_()
        teacher.bias.zero_()

    copy_ema_teacher_weights(student=student, teacher=teacher)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.training is False
    assert torch.equal(teacher.weight, student.weight)

    with torch.no_grad():
        student.weight.fill_(4.0)
        student.bias.fill_(3.0)
    update_ema_teacher(student=student, teacher=teacher, decay=0.5)

    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 3.0))
    assert torch.allclose(teacher.bias, torch.full_like(teacher.bias, 2.0))
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_border_black_mask_keeps_interior_black_regions(tmp_path):
    from synthforest.training.valid_masks import load_image_with_valid_mask

    array = np.full((16, 16, 3), 100, dtype=np.uint8)
    array[:, :3] = 0
    array[7:9, 8:10] = 0
    image_path = tmp_path / "edge.png"
    Image.fromarray(array).save(image_path)

    _, valid_mask = load_image_with_valid_mask(
        image_path,
        ValidMaskConfig(source="border_black"),
    )

    assert not valid_mask[:, :3].any()
    assert valid_mask[7:9, 8:10].all()


def test_alpha_channel_takes_precedence_for_valid_mask(tmp_path):
    from synthforest.training.valid_masks import load_image_with_valid_mask

    array = np.full((12, 12, 4), 100, dtype=np.uint8)
    array[:, :, 3] = 255
    array[:4, :, 3] = 0
    image_path = tmp_path / "alpha.png"
    Image.fromarray(array, mode="RGBA").save(image_path)

    _, valid_mask = load_image_with_valid_mask(
        image_path,
        ValidMaskConfig(source="auto"),
    )

    assert not valid_mask[:4].any()
    assert valid_mask[4:].all()


def test_invalid_pixels_are_mean_filled_after_augmentation(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import DetectionAugmentor

    config = _tiny_config(tmp_path)
    config.augmentation.crop_prob = 0.0
    config.augmentation.rotation_prob = 0.0
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.brightness_prob = 1.0
    config.augmentation.contrast_prob = 1.0
    config.augmentation.gaussian_noise_prob = 1.0
    config.augmentation.gaussian_noise_std = [0.02, 0.02]
    augmentor = DetectionAugmentor(config, train=True)
    image = Image.new("RGB", (16, 16), color=(70, 100, 60))
    valid_mask = np.ones((16, 16), dtype=bool)
    valid_mask[:, 12:] = False
    empty_boxes = torch.empty((0, 4), dtype=torch.float32)

    tensor, tensor_mask, *_ = augmentor(
        image,
        valid_mask,
        empty_boxes,
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0,), dtype=torch.float32),
        torch.empty((0,), dtype=torch.int64),
    )

    expected_mean = torch.tensor(config.augmentation.image_mean)[:, None]
    assert torch.equal(tensor_mask, torch.from_numpy(valid_mask))
    assert torch.allclose(tensor[:, ~tensor_mask], expected_mean)


def test_disabling_valid_masks_preserves_legacy_rotation_fill(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import DetectionAugmentor

    config = _tiny_config(tmp_path)
    config.valid_mask.enabled = False
    config.augmentation.rotation_degrees = 30.0
    config.augmentation.rotation_prob = 1.0
    augmentor = DetectionAugmentor(config, train=True)
    monkeypatch.setattr(
        "synthforest.training.dataset.random.uniform",
        lambda *_: 30.0,
    )

    image, valid_mask, _ = augmentor._apply_rotation(
        Image.new("RGB", (16, 16), color=(70, 100, 60)),
        np.ones((16, 16), dtype=bool),
        torch.empty((0, 4), dtype=torch.float32),
    )

    assert valid_mask.all()
    assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("angle", [30.0, -30.0])
def test_rotate_boxes_matches_pillow_rotation(angle):
    torch = pytest.importorskip("torch")
    from synthforest.training.dataset import _rotate_boxes

    object_mask = np.zeros((100, 100), dtype=np.uint8)
    object_mask[10:20, 10:20] = 255
    rotated_mask = np.asarray(
        Image.fromarray(object_mask).rotate(
            angle,
            resample=Image.Resampling.NEAREST,
        )
    )
    ys, xs = np.nonzero(rotated_mask)
    observed = torch.tensor(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=torch.float32,
    )

    rotated_box = _rotate_boxes(
        torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        width=100,
        height=100,
        angle_degrees=angle,
    )[0]
    rotated_box[0::2].clamp_(0, 100)
    rotated_box[1::2].clamp_(0, 100)

    assert torch.allclose(rotated_box, observed, atol=1.1)


def test_standard_pda_matches_reference_statistics_without_darkening():
    from synthforest.training.dataset import _adapt_pixel_distribution

    gradient = np.linspace(40, 100, num=64, dtype=np.float32).reshape(8, 8)
    source = np.stack([gradient, gradient + 10, gradient + 20], axis=-1).astype(np.uint8)
    reference = np.stack(
        [gradient * 0.75 + 80, gradient * 0.5 + 100, gradient * 0.25 + 120],
        axis=-1,
    ).astype(np.uint8)

    adapted = _adapt_pixel_distribution(
        source,
        reference,
        transform_type="standard",
        weight=1.0,
    )

    adapted_pixels = adapted.reshape(-1, 3).astype(np.float32)
    reference_pixels = reference.reshape(-1, 3).astype(np.float32)
    assert adapted.dtype == np.uint8
    assert np.allclose(adapted_pixels.mean(axis=0), reference_pixels.mean(axis=0), atol=0.6)
    assert np.allclose(adapted_pixels.std(axis=0), reference_pixels.std(axis=0), atol=0.6)
    assert float(adapted.mean()) > float(source.mean())


def test_partial_tiles_skip_domain_adaptation(tmp_path, monkeypatch):
    from synthforest.training.dataset import DetectionAugmentor

    config = _tiny_config(tmp_path)
    config.domain_adaptation.enabled = True
    config.domain_adaptation.prob = 1.0
    augmentor = DetectionAugmentor(config, train=True)
    augmentor.reference_paths = [tmp_path / "reference.png"]
    partial_mask = np.ones((16, 16), dtype=bool)
    partial_mask[:, -1] = False

    def fail_if_reference_is_selected():
        raise AssertionError("DA reference selection must not run for a partial tile")

    monkeypatch.setattr(
        augmentor,
        "_choose_full_reference_path",
        fail_if_reference_is_selected,
    )

    image = Image.new("RGB", (16, 16), color=(70, 100, 60))
    assert augmentor._apply_domain_adaptation(image, partial_mask) is image


def test_real_validation_loader_uses_registered_products(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.validation_data import (
        build_real_validation_loaders,
    )

    config = _tiny_config(tmp_path)
    loaders = build_real_validation_loaders(config)

    assert list(loaders) == ["QuebecTrees"]
    assert len(loaders["QuebecTrees"].dataset) == 1


def test_model_smoke_and_frozen_backbone(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import (
        assert_dinov3_frozen,
        build_dinov3_detector,
        set_stage_trainability,
    )

    config = _tiny_config(tmp_path)
    model = build_dinov3_detector(config)
    set_stage_trainability(model, train_lora=False)
    model.train()

    pyramid = model.backbone(torch.rand(1, 3, 64, 64))
    assert [pyramid[key].shape[-2:] for key in ("p8", "p16", "p32")] == [
        (8, 8),
        (4, 4),
        (2, 2),
    ]
    losses = model(
        [torch.rand(3, 64, 64)],
        [
            {
                "boxes": torch.tensor([[8.0, 8.0, 32.0, 32.0]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([576.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor([0]),
            }
        ],
    )
    sum(losses.values()).backward()
    assert model.backbone.backbone_grad_enabled is False
    assert_dinov3_frozen(model)


def test_model_uses_explicit_eval_resize_min_size(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import build_dinov3_detector

    config = _tiny_config(tmp_path)
    config.augmentation.resize_min_sizes_train = [32, 64]
    config.augmentation.resize_min_size_eval = 48
    config.augmentation.max_size = 128
    model = build_dinov3_detector(config).eval()

    assert model.transform.train_min_sizes == (32, 64)
    assert model.transform.eval_min_size == 48
    with torch.no_grad():
        images, _ = model.transform([torch.rand(3, 60, 80)], None)
    assert images.image_sizes == [(48, 64)]


def test_model_uses_stock_torchvision_rpn(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import RegionProposalNetwork

    from synthforest.training.models import build_dinov3_detector

    model = build_dinov3_detector(_tiny_config(tmp_path))

    assert type(model) is FasterRCNN
    assert type(model.rpn) is RegionProposalNetwork


def test_final_prediction_filter_uses_valid_area_only():
    torch = pytest.importorskip("torch")
    from synthforest.training.valid_masks import (
        filter_detections_by_valid_masks,
    )

    valid_mask = torch.ones((20, 20), dtype=torch.bool)
    valid_mask[:, 10:] = False
    detection = {
        "boxes": torch.tensor(
            [
                [1.0, 1.0, 9.0, 9.0],
                [8.0, 1.0, 18.0, 9.0],
                [11.0, 1.0, 18.0, 9.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "labels": torch.tensor([1, 2, 3]),
    }

    filtered, stats = filter_detections_by_valid_masks(
        [detection],
        [valid_mask],
        min_valid_fraction=0.6,
    )

    assert filtered[0]["boxes"].shape == (1, 4)
    assert filtered[0]["scores"].tolist() == pytest.approx([0.9])
    assert filtered[0]["labels"].tolist() == [1]
    assert stats == {
        "num_predictions_before_mask_filter": 3,
        "num_predictions_filtered_invalid": 2,
    }


def test_valid_mask_config_does_not_change_detector_state_dict(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import build_dinov3_detector

    enabled = _tiny_config(tmp_path / "enabled")
    disabled = _tiny_config(tmp_path / "disabled")
    disabled.valid_mask.enabled = False

    enabled_keys = set(build_dinov3_detector(enabled).state_dict())
    disabled_keys = set(build_dinov3_detector(disabled).state_dict())

    assert enabled_keys == disabled_keys


def test_all_valid_and_all_invalid_prediction_filtering():
    torch = pytest.importorskip("torch")
    from synthforest.training.valid_masks import (
        filter_detections_by_valid_masks,
    )

    detection = {
        "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }
    all_valid, valid_stats = filter_detections_by_valid_masks(
        [detection],
        [torch.ones((10, 10), dtype=torch.bool)],
        min_valid_fraction=0.6,
    )
    all_invalid, invalid_stats = filter_detections_by_valid_masks(
        [detection],
        [torch.zeros((10, 10), dtype=torch.bool)],
        min_valid_fraction=0.6,
    )

    assert torch.equal(all_valid[0]["boxes"], detection["boxes"])
    assert valid_stats["num_predictions_filtered_invalid"] == 0
    assert all_invalid[0]["boxes"].shape == (0, 4)
    assert invalid_stats["num_predictions_filtered_invalid"] == 1


def test_lora_stage_toggle_only_trains_last_block_adapters(tmp_path):
    _skip_if_peft_unavailable()
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.models import (
        build_dinov3_detector,
        set_stage_trainability,
    )

    config = _enable_lora(_tiny_config(tmp_path))
    model = build_dinov3_detector(config)

    set_stage_trainability(model, train_lora=False)
    assert not any(parameter.requires_grad for parameter in model.backbone.dinov3.parameters())

    set_stage_trainability(model, train_lora=True)
    trainable = [
        name
        for name, parameter in model.backbone.dinov3.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert any(".layer.2." in name for name in trainable)
    assert any(".layer.3." in name for name in trainable)
    assert not any(".layer.0." in name for name in trainable)


def test_stage_optimizers_have_expected_parameter_groups(tmp_path):
    _skip_if_peft_unavailable()
    pytest.importorskip("torch")
    from synthforest.training.engine.optimization import (
        build_stage_optimizer,
    )
    from synthforest.training.models import (
        build_dinov3_detector,
        set_stage_trainability,
    )

    config = _enable_lora(_tiny_config(tmp_path))
    model = build_dinov3_detector(config)

    set_stage_trainability(model, train_lora=False)
    detector_optimizer = build_stage_optimizer(model, config.stages[0])
    assert [group["name"] for group in detector_optimizer.param_groups] == ["detector"]

    set_stage_trainability(model, train_lora=True)
    joint_optimizer = build_stage_optimizer(model, config.stages[1])
    assert [group["name"] for group in joint_optimizer.param_groups] == [
        "detector",
        "lora",
    ]
    assert joint_optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)


def test_real_metrics_are_macro_averaged_and_selected(tmp_path):
    from synthforest.training.engine.trainer import _selection_metric_value
    from synthforest.training.evaluation import macro_average_metrics

    aggregate = macro_average_metrics(
        {
            "Small": {
                "bbox_dense_map": 0.8,
                "bbox_map": 0.6,
                "num_images": 10.0,
                "num_targets": 20.0,
                "num_predictions": 30.0,
            },
            "Huge": {
                "bbox_dense_map": 0.2,
                "bbox_map": 0.4,
                "num_images": 4000.0,
                "num_targets": 90000.0,
                "num_predictions": 80000.0,
            },
        }
    )
    payload = {
        "valid_synthetic": {"bbox_dense_map": 0.99},
        "valid_real": aggregate,
    }

    assert aggregate["bbox_dense_map"] == pytest.approx(0.5)
    assert aggregate["bbox_map"] == pytest.approx(0.5)
    assert aggregate["num_images"] == 4010.0
    assert _selection_metric_value(payload, "bbox_dense_map") == pytest.approx(0.5)


def test_unified_frozen_training_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config
    from synthforest.training.evaluation import evaluate_checkpoint

    config = _tiny_config(tmp_path)
    config.validate()
    output_dir = train_from_config(config)

    assert (output_dir / "config" / "resolved.yaml").exists()
    assert (output_dir / "checkpoints" / "stages" / "detector_only" / "last.pth").exists()
    assert (output_dir / "checkpoints" / "stages" / "detector_only" / "best.pth").exists()
    assert (output_dir / "checkpoints" / "last.pth").exists()
    assert (output_dir / "checkpoints" / "best.pth").exists()
    metrics = json.loads(
        (output_dir / "metrics" / "validation_latest.json").read_text(encoding="utf-8")
    )
    assert "valid_synthetic" in metrics
    assert "valid_real" in metrics
    assert "QuebecTrees" in metrics["valid_real"]["by_dataset"]
    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["schema_version"] == 4
    assert "model_config" not in checkpoint
    assert "config" not in checkpoint
    assert checkpoint["stage"] == "detector_only"
    assert checkpoint["best_metric_name"] == "valid_real/bbox_dense_map"
    evaluated = evaluate_checkpoint(
        output_dir / "checkpoints" / "best.pth",
        validation_set="both",
        device="cpu",
    )
    assert set(evaluated) == {"valid_synthetic", "valid_real"}


def test_mean_teacher_training_smoke_saves_student_and_teacher_states(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    config = _tiny_config(tmp_path)
    config.run_name = "mean_teacher_smoke"
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
            data_root_path=config.validation.real.data_root_path,
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
    train_metrics = [
        json.loads(line)
        for line in (output_dir / "metrics" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert "student_state_dict" in checkpoint
    assert "ema_teacher_state_dict" in checkpoint
    first_key = next(iter(checkpoint["model_state_dict"]))
    assert torch.equal(
        checkpoint["model_state_dict"][first_key],
        checkpoint["ema_teacher_state_dict"][first_key],
    )
    assert train_metrics[-1]["ema/decay"] == pytest.approx(0.5)
    assert train_metrics[-1]["train/pseudo_loss_weight"] == pytest.approx(0.25)
    assert train_metrics[-1]["train/pseudo_loss_target_weight"] == pytest.approx(1.0)
    assert train_metrics[-1]["train/pseudo_loss_ramp_fraction"] == pytest.approx(0.5)
    assert train_metrics[-1]["train/pseudo_images_used"] == 1.0
    semi_metadata = json.loads(
        (output_dir / "metadata" / "semi_supervised.json").read_text(encoding="utf-8")
    )
    assert semi_metadata["unsupervised_loss_start_weight"] == pytest.approx(0.25)
    assert semi_metadata["unsupervised_loss_weight"] == pytest.approx(1.0)
    assert semi_metadata["unsupervised_loss_ramp_fraction"] == pytest.approx(0.5)
    assert "strong_augmentation" in semi_metadata


@pytest.mark.parametrize("height_channels", [1, 2])
def test_height_fusion_mean_teacher_training_smoke_saves_student_and_teacher_states(
    tmp_path,
    height_channels,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    config = _tiny_config(tmp_path)
    _write_detector_height_maps_for_tiles(config.data_root_path, channels=height_channels)
    _write_detector_height_maps_for_tiles(
        config.validation.real.data_root_path,
        channels=height_channels,
    )
    config.run_name = f"height_mean_teacher_smoke_c{height_channels}"
    config.height_fusion.enabled = True
    config.height_fusion.use_uncertainty = height_channels >= 2
    config.height_fusion.generate_missing = False
    config.height_fusion.levels = ["p8"]
    config.height_fusion.encoder_hidden_channels = 8
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
            data_root_path=config.validation.real.data_root_path,
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
    assert "student_state_dict" in checkpoint
    assert "ema_teacher_state_dict" in checkpoint
    assert checkpoint["training_config"]["height_fusion"]["enabled"] is True
    assert checkpoint["training_config"]["height_fusion"]["use_uncertainty"] is (
        height_channels >= 2
    )


def test_external_wandb_run_and_branch_checkpoint_smoke(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine import trainer as train_module

    class FakeRun:
        id = "sweep123"
        name = "frozen-sweep"

        def __init__(self):
            self.summary = {}
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((step, metrics))

        def finish(self):
            raise AssertionError("The trainer must not finish an external W&B run.")

    monkeypatch.setattr(
        train_module,
        "_init_wandb",
        lambda *args, **kwargs: pytest.fail("Trainer initialized a second W&B run."),
    )
    config = _tiny_config(tmp_path)
    config.branch_checkpoint_iterations = [1]
    config.sweep = SweepRuntimeConfig(
        phase="frozen",
        objective_iterations=[1],
    )
    config.wandb.log_artifacts = False
    fake_run = FakeRun()

    output_dir = train_module.train_from_config(config, wandb_run=fake_run)

    branch_path = output_dir / "checkpoints" / "branches" / "iter_00000001.pth"
    assert branch_path.exists()
    branch = torch.load(branch_path, map_location="cpu", weights_only=False)
    assert branch["source_run_id"] == "sweep123"
    assert fake_run.summary["run/completed"] is True
    assert any(metrics.get("sweep/objective") is not None for _, metrics in fake_run.logged)


def test_unified_staged_training_smoke_and_final_best(tmp_path):
    _skip_if_peft_unavailable()
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    config = _enable_lora(_tiny_config(tmp_path))
    config.run_name = "staged_smoke"
    config.validate()
    output_dir = train_from_config(config)

    assert (output_dir / "checkpoints" / "stages" / "detector_only" / "last.pth").exists()
    assert (output_dir / "checkpoints" / "stages" / "joint_lora" / "last.pth").exists()
    assert (output_dir / "checkpoints" / "stages" / "joint_lora" / "best.pth").exists()
    best = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert best["stage"] == "joint_lora"
    assert best["stage_index"] == 1


@pytest.mark.parametrize("with_lora", [False, True])
def test_checkpoint_loader_reconstructs_schema_four_models(
    tmp_path,
    with_lora,
):
    if with_lora:
        _skip_if_peft_unavailable()
    torch = pytest.importorskip("torch")
    from synthforest.training.checkpoint import load_checkpoint_model
    from synthforest.training.models import build_dinov3_detector

    config = _tiny_config(tmp_path)
    if with_lora:
        _enable_lora(config, staged=False)
    model = build_dinov3_detector(config)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "iteration": 4,
        },
        checkpoint_path,
    )

    loaded, loaded_config, metadata = load_checkpoint_model(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert loaded.training is False
    assert (loaded_config.lora is not None) is with_lora
    assert metadata["iteration"] == 4
    assert metadata["config_source"] == "checkpoint"
    assert metadata["valid_mask_policy_source"] == "checkpoint"
    assert loaded_config.valid_mask.enabled is True


def test_full_checkpoint_weights_initialize_finetune_config_with_new_data(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.checkpoint import load_checkpoint_weights_into_model
    from synthforest.training.models import build_dinov3_detector

    checkpoint_config = _tiny_config(tmp_path / "source")
    source_model = build_dinov3_detector(checkpoint_config)
    checkpoint_path = tmp_path / "synthetic_best.pth"
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": source_model.state_dict(),
            "training_config": checkpoint_config.to_dict(),
            "run_name": "synthetic-source",
            "iteration": 12,
            "stage": "detector_only",
        },
        checkpoint_path,
    )
    finetune_config = _tiny_config(tmp_path / "finetune")
    finetune_config.data_root_path = str(tmp_path / "new_real_root")
    finetune_config.train_dataset_names = ["new_real_dataset"]
    target_model = build_dinov3_detector(finetune_config)

    metadata = load_checkpoint_weights_into_model(
        checkpoint_path,
        model=target_model,
        config=finetune_config,
    )

    assert metadata["type"] == "full_checkpoint"
    assert metadata["iteration"] == 12
    assert metadata["source_run_name"] == "synthetic-source"
    for key, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[key], value)


def test_full_checkpoint_weights_reject_cross_model_type(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from synthforest.training.checkpoint import load_checkpoint_weights_into_model
    from synthforest.training.models import build_dinov3_detector

    faster_config = _tiny_config(tmp_path / "source")
    faster_model = build_dinov3_detector(faster_config)
    checkpoint_path = tmp_path / "faster_best.pth"
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": faster_model.state_dict(),
            "training_config": faster_config.to_dict(),
        },
        checkpoint_path,
    )
    mask_config = DetectorTrainingConfig.from_dict(faster_config.to_dict())
    mask_config.model_type = "maskrcnn"
    mask_config.validation.tasks = ["bbox", "segm"]
    mask_config.validate()
    mask_model = build_dinov3_detector(mask_config)

    with pytest.raises(ValueError, match="model_type"):
        load_checkpoint_weights_into_model(
            checkpoint_path,
            model=mask_model,
            config=mask_config,
        )


def test_finetune_training_smoke_loads_checkpoint_and_records_sampling(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from synthforest.training.engine.trainer import train_from_config

    source_config = _tiny_config(tmp_path / "source")
    source_config.run_name = "source"
    source_config.train_output_path = str(tmp_path / "source_runs")
    source_dir = train_from_config(source_config)

    finetune_config = _tiny_config(tmp_path / "finetune")
    finetune_config.run_name = "finetune"
    finetune_config.train_output_path = str(tmp_path / "finetune_runs")
    finetune_config.checkpoint_initialization = CheckpointInitializationConfig(
        checkpoint_path=str(source_dir / "checkpoints" / "best.pth")
    )
    finetune_config.train_sampling = TrainSamplingConfig(
        fraction=0.5,
        seed=11,
        min_samples=1,
    )
    output_dir = train_from_config(finetune_config)

    initialization = json.loads(
        (output_dir / "metadata" / "initialization.json").read_text(encoding="utf-8")
    )
    sampling = json.loads(
        (output_dir / "metadata" / "train_sampling.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert initialization["type"] == "full_checkpoint"
    assert sampling["total_samples"] == 2
    assert sampling["selected_samples"] == 1
    assert sampling["fraction"] == pytest.approx(0.5)
    assert sampling["seed"] == 11
    assert checkpoint["iteration"] == 1


def test_checkpoint_loader_rejects_legacy_schema(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.checkpoint import load_checkpoint_model

    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(
        {
            "schema_version": 3,
            "model_state_dict": {},
            "training_config": _tiny_config(tmp_path).to_dict(),
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="Unsupported full checkpoint schema"):
        load_checkpoint_model(checkpoint_path, torch.device("cpu"))
