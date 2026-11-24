import json

import numpy as np
import pytest
import yaml
from PIL import Image

from synthforest.training.config import (
    DetectorTrainingConfig,
    HeightBenchmarkConfig,
    HeightMetricResolutionConfig,
    HeightRasterBenchmarkConfig,
    HeightRasterPairConfig,
    HeightResolutionDegradationConfig,
    HeightStageOptimizerConfig,
    HeightTrainingConfig,
    HeightTrainingStageConfig,
    LoRAConfig,
)


def _height_config(tmp_path, *, real_root=None):
    config = HeightTrainingConfig(
        data_root_path=str(tmp_path / "data"),
        train_dataset_names=["tiny_height"],
        train_output_path=str(tmp_path / "runs"),
        run_name="height-smoke",
        dinov3_use_fake_backbone=True,
        dinov3_embed_dim=8,
        dinov3_patch_size=8,
        dinov3_fake_num_blocks=4,
        dpt_feature_layers=[0, 1, 2, 3],
        dpt_feature_dim=8,
        dpt_out_channels=[8, 8, 8, 8],
        batch_size=1,
        grad_accumulation_steps=1,
        dataloader_num_workers=0,
        use_amp=False,
        stages=[
            HeightTrainingStageConfig(
                name="decoder_only",
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
                optimizer=HeightStageOptimizerConfig(
                    decoder_lr=1e-4,
                    decoder_weight_decay=0.0,
                    warmup_iterations=0,
                ),
            )
        ],
    )
    config.augmentation.resize_min_sizes_train = [32]
    config.augmentation.resize_min_size_eval = 32
    config.augmentation.max_size = 32
    config.augmentation.crop_prob = 0.0
    config.augmentation.flip_horizontal = False
    config.augmentation.flip_vertical = False
    config.augmentation.rotation_prob = 0.0
    config.augmentation.brightness_prob = 0.0
    config.augmentation.contrast_prob = 0.0
    config.augmentation.saturation_prob = 0.0
    config.augmentation.hue_prob = 0.0
    config.augmentation.gaussian_blur_prob = 0.0
    config.augmentation.gaussian_noise_prob = 0.0
    config.augmentation.jpeg_prob = 0.0
    config.augmentation.downsample_upsample_prob = 0.0
    config.valid_mask.prediction_min_valid_fraction = None
    config.domain_adaptation.enabled = False
    config.wandb.enabled = False
    config.post_training_benchmark.enabled = False
    config.validation.synthetic.dataset_names = ["tiny_height"]
    config.validation.synthetic.fold = "valid"
    config.validation.real.data_root_path = str(
        real_root if real_root is not None else tmp_path / "real" / "NEON_CHM_1m_real_eval"
    )
    config.validation.real.dataset_names = ["NEON_CHM_1m_real_eval"]
    config.validation.real.max_products_per_dataset = 1
    config.validation.real.max_tiles_per_product = 1
    return config


def _write_height_tile(product_root, fold, suffix, *, image_size=(32, 32), height=None):
    tiles_dir = product_root / "tiles" / fold
    heights_dir = product_root / "height_maps" / fold
    tiles_dir.mkdir(parents=True, exist_ok=True)
    heights_dir.mkdir(parents=True, exist_ok=True)
    width, height_px = image_size
    rgba = np.zeros((height_px, width, 4), dtype=np.uint8)
    rgba[..., 0] = 80
    rgba[..., 1] = 120
    rgba[..., 2] = 70
    rgba[..., 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(tiles_dir / f"scene_tile_{fold}_{suffix}.tif")
    if height is None:
        height = np.linspace(0.0, 12.0, num=width * height_px, dtype=np.float32).reshape(
            height_px,
            width,
        )
    Image.fromarray(np.asarray(height, dtype=np.float32), mode="F").save(
        heights_dir / f"chm_tile_{fold}_{suffix}.tif"
    )


def _write_tiny_height_dataset(root):
    dataset_root = root / "tiny_height" / "product_a"
    for fold in ("train", "valid"):
        _write_height_tile(dataset_root, fold, "0001")
    return root


def _write_tiny_real_height_root(root):
    product_root = root / "product_real"
    _write_height_tile(product_root, "valid", "0001")
    _write_height_tile(product_root, "test", "0001")
    return root


def _write_tiny_detector_dataset(root, *, image_size=(32, 32)):
    dataset_root = root / "tiny_detector" / "product_a"
    tiles_dir = dataset_root / "tiles" / "train"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[..., 1] = 120
    rgb[..., 2] = 70
    Image.fromarray(rgb, mode="RGB").save(tiles_dir / "tile_train_box.tif")
    return root


def _detector_config(tmp_path, data_root):
    config = DetectorTrainingConfig(
        data_root_path=str(data_root),
        train_dataset_names=["tiny_detector"],
        train_output_path=str(tmp_path / "detector_runs"),
        run_name="detector-smoke",
        model_type="fasterrcnn",
        dinov3_use_fake_backbone=True,
        dinov3_embed_dim=8,
        backbone_out_channels=8,
        batch_size=1,
        grad_accumulation_steps=1,
        dataloader_num_workers=0,
        use_amp=False,
        anchor_sizes=[8, 16, 32],
    )
    config.augmentation.resize_min_sizes_train = [32]
    config.augmentation.resize_min_size_eval = 32
    config.augmentation.max_size = 32
    config.domain_adaptation.enabled = False
    config.wandb.enabled = False
    config.validation.synthetic.dataset_names = ["tiny_detector"]
    return config


def _write_fake_height_checkpoint(path, config):
    import torch

    from synthforest.training.height.checkpoint import HEIGHT_CHECKPOINT_SCHEMA_VERSION
    from synthforest.training.models import build_dinov3_height_model

    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_dinov3_height_model(config)
    torch.save(
        {
            "schema_version": HEIGHT_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "height",
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "run_name": config.run_name,
            "iteration": 0,
            "stage": "fake",
            "stage_index": 0,
            "stage_iteration": 0,
            "best_metric": None,
            "best_metric_name": None,
        },
        path,
    )


def test_height_config_rejects_invalid_fake_dpt_layers(tmp_path):
    config = _height_config(tmp_path)
    config.dinov3_fake_num_blocks = 3

    with pytest.raises(ValueError, match="dpt_feature_layers"):
        config.validate()


def test_height_config_validates_uncertainty_loss_channel_pairing(tmp_path):
    config = _height_config(tmp_path)
    config.height_output_channels = 2
    config.loss_name = "smooth_l1"
    with pytest.raises(ValueError, match="smooth_l1"):
        config.validate()

    config = _height_config(tmp_path)
    config.height_output_channels = 1
    config.loss_name = "laplace_nll"
    with pytest.raises(ValueError, match="laplace_nll"):
        config.validate()

    config = _height_config(tmp_path)
    config.height_output_channels = 2
    config.loss_name = "laplace_nll"
    config.validate()


def test_height_checkpoint_rejects_output_channel_mismatch(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.height.checkpoint import load_height_checkpoint_config

    checkpoint_config = _height_config(tmp_path)
    checkpoint_path = tmp_path / "height" / "best.pth"
    _write_fake_height_checkpoint(checkpoint_path, checkpoint_config)

    external_config = _height_config(tmp_path)
    external_config.height_output_channels = 2
    external_config.loss_name = "laplace_nll"
    external_config.validate()
    external_config_path = tmp_path / "external_uncertainty.yaml"
    external_config.save_yaml(external_config_path)

    with pytest.raises(ValueError, match="height_output_channels"):
        load_height_checkpoint_config(checkpoint_path, config_path=external_config_path)


def test_height_config_rejects_invalid_resolution_degradation(tmp_path):
    config = _height_config(tmp_path)
    config.resolution_degradation = HeightResolutionDegradationConfig(
        enabled=True,
        base_gsd_m=0.5,
        target_gsd_m_values=[0.5, 0.6, 0.75, 1.0],
    )
    config.validate()

    config = _height_config(tmp_path)
    config.resolution_degradation = HeightResolutionDegradationConfig(
        enabled=True,
        base_gsd_m=0.5,
        target_gsd_m_values=[0.4],
    )
    with pytest.raises(ValueError, match="target_gsd_m_values must be >= base_gsd_m"):
        config.validate()

    config = _height_config(tmp_path)
    config.resolution_degradation = HeightResolutionDegradationConfig(
        enabled=True,
        height_reducer="median",
    )
    with pytest.raises(ValueError, match="height_reducer"):
        config.validate()


def test_height_benchmark_config_rejects_invalid_metric_resolution():
    benchmark = HeightBenchmarkConfig(
        checkpoint_path="best.pth",
        metric_resolution=HeightMetricResolutionConfig(
            enabled=True,
            rgb_gsd_m=1.0,
            height_gsd_m=0.5,
        ),
    )
    with pytest.raises(ValueError, match="factor must be >= 1"):
        benchmark.validate()

    fractional = HeightBenchmarkConfig(
        checkpoint_path="best.pth",
        metric_resolution=HeightMetricResolutionConfig(
            enabled=True,
            rgb_gsd_m=0.6,
            height_gsd_m=1.0,
        ),
    )
    fractional.validate()
    assert fractional.metric_resolution.resolved_downsample_factor() == pytest.approx(1.6666667)

    benchmark = HeightBenchmarkConfig(
        checkpoint_path="best.pth",
        metric_resolution=HeightMetricResolutionConfig(
            enabled=True,
            downsample_factor=2,
            upsample_interpolation="bicubic",
        ),
    )
    with pytest.raises(ValueError, match="upsample_interpolation"):
        benchmark.validate()


def test_height_dataset_pairs_suffixes_masks_zero_and_resolves_direct_root(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.height.dataset import (
        HeightTileDataset,
        collate_height_batch,
        resolve_height_dataset_roots,
    )

    dataset_root = tmp_path / "direct_real_root"
    product_root = dataset_root / "product_a"
    height = np.arange(16, dtype=np.float32).reshape(4, 4)
    height[0, 0] = 0.0
    height[0, 1] = 999.0
    _write_height_tile(product_root, "valid", "shared", image_size=(4, 4), height=height)
    _write_height_tile(product_root, "valid", "orphan_image", image_size=(4, 4), height=height)
    (product_root / "height_maps" / "valid" / "chm_tile_valid_orphan_image.tif").unlink()

    config = _height_config(tmp_path, real_root=dataset_root)
    config.data_root_path = str(dataset_root)
    config.train_dataset_names = [dataset_root.name]
    config.validation.synthetic.dataset_names = [dataset_root.name]
    config.augmentation.resize_min_sizes_train = [4]
    config.augmentation.resize_min_size_eval = 4
    config.augmentation.max_size = 4
    config.max_valid_height_m = 20.0
    config.validate()

    assert resolve_height_dataset_roots(dataset_root, [dataset_root.name]) == [
        (dataset_root.name, dataset_root)
    ]

    dataset = HeightTileDataset(
        data_root_path=dataset_root,
        dataset_names=[dataset_root.name],
        fold="valid",
        config=config,
        train=False,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["metadata"]["tile_key"] == "shared"
    assert item["height"].shape == (1, 4, 4)
    assert item["height"][0, 0, 0].item() == pytest.approx(0.0)
    assert item["valid_mask"][0, 0].item() is True
    assert item["valid_mask"][0, 1].item() is False

    smaller = {
        "image": torch.zeros((3, 3, 5), dtype=torch.float32),
        "height": torch.zeros((1, 3, 5), dtype=torch.float32),
        "valid_mask": torch.ones((3, 5), dtype=torch.bool),
        "metadata": {"tile_key": "small"},
    }
    collated = collate_height_batch([item, smaller], patch_size_multiple=4)
    assert collated["images"].shape == (2, 3, 4, 8)
    assert collated["heights"].shape == (2, 1, 4, 8)
    assert collated["valid_masks"][1, 3, 7].item() is False
    assert collated["resolution_factors"].tolist() == [1, 1]


def test_height_training_dataset_skips_tiles_below_target_valid_fraction(tmp_path):
    from synthforest.training.height.dataset import HeightTileDataset

    root = tmp_path / "data"
    product_root = root / "tiny_height" / "product_a"
    _write_height_tile(
        product_root,
        "train",
        "0001_invalid",
        height=np.full((32, 32), 999.0, dtype=np.float32),
    )
    _write_height_tile(
        product_root,
        "train",
        "0002_valid",
        height=np.full((32, 32), 12.0, dtype=np.float32),
    )

    config = _height_config(tmp_path)
    config.max_valid_height_m = 60.0
    config.valid_mask.target_min_valid_fraction = 0.5
    config.validate()

    dataset = HeightTileDataset(
        data_root_path=root,
        dataset_names=["tiny_height"],
        fold="train",
        config=config,
        train=True,
    )
    with pytest.warns(UserWarning, match="First skipped CHM"):
        item = dataset[0]

    assert item["metadata"]["tile_key"] == "0002_valid"
    assert bool(item["valid_mask"].all())
    assert item["height"].numpy().flags.writeable


def test_resolution_aware_height_loss_aggregates_low_resolution_targets():
    torch = pytest.importorskip("torch")
    from synthforest.training.height.metrics import (
        masked_resolution_aware_smooth_l1_loss,
        masked_smooth_l1_loss,
    )

    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]])
    valid_mask = torch.ones((1, 2, 2), dtype=torch.bool)

    factor_one = masked_resolution_aware_smooth_l1_loss(
        prediction,
        target,
        valid_mask,
        resolution_factors=torch.tensor([1]),
    )
    assert factor_one == pytest.approx(float(masked_smooth_l1_loss(prediction, target, valid_mask)))

    factor_two = masked_resolution_aware_smooth_l1_loss(
        prediction,
        target,
        valid_mask,
        resolution_factors=torch.tensor([2]),
        height_reducer="p90_positive",
        min_canopy_fraction=0.1,
    )
    reduced_target = torch.quantile(torch.tensor([10.0, 20.0, 30.0]), 0.9)
    reduced_prediction = prediction.mean()
    expected = torch.nn.functional.smooth_l1_loss(
        reduced_prediction.view(1),
        reduced_target.view(1),
    )
    assert factor_two == pytest.approx(float(expected))

    masked_out = masked_resolution_aware_smooth_l1_loss(
        prediction,
        target,
        valid_mask,
        resolution_factors=torch.tensor([2]),
        height_reducer="p90_positive",
        min_canopy_fraction=1.0,
    )
    assert masked_out == pytest.approx(0.0)

    fractional = masked_resolution_aware_smooth_l1_loss(
        torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        torch.ones((1, 1, 4, 4), dtype=torch.float32),
        torch.ones((1, 4, 4), dtype=torch.bool),
        resolution_factors=torch.tensor([1.5]),
        height_reducer="p90_positive",
        min_canopy_fraction=0.1,
    )
    assert fractional == pytest.approx(0.5)


def test_masked_laplace_nll_loss_uses_log_variance_and_mask():
    torch = pytest.importorskip("torch")
    from synthforest.training.height.metrics import masked_laplace_nll_loss

    prediction = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]], requires_grad=True)
    log_variance = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[[[2.0, 3.0], [1.0, 7.0]]]])
    valid_mask = torch.tensor([[[True, False], [True, True]]])

    loss = masked_laplace_nll_loss(prediction, log_variance, target, valid_mask)
    expected_values = torch.tensor([1.0, 4.0, 0.0]) * np.sqrt(2.0)
    assert float(loss.detach()) == pytest.approx(float(expected_values.mean()))

    loss.backward()
    assert prediction.grad is not None
    assert log_variance.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.isfinite(log_variance.grad).all()

    invalid = masked_laplace_nll_loss(
        prediction,
        log_variance,
        target,
        torch.zeros((1, 2, 2), dtype=torch.bool),
    )
    assert float(invalid.detach()) == pytest.approx(0.0)


def test_height_metric_resolution_degrades_prediction_to_chm_effective_grid():
    torch = pytest.importorskip("torch")
    from synthforest.training.height.metric_resolution import apply_height_metric_resolution

    prediction = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    valid_mask = torch.ones((1, 4, 4), dtype=torch.bool)
    config = HeightMetricResolutionConfig(
        enabled=True,
        downsample_factor=1.5,
        min_valid_fraction=0.5,
        upsample_interpolation="nearest",
    )

    metric_prediction, metric_mask = apply_height_metric_resolution(
        prediction,
        valid_mask,
        config,
    )

    assert metric_prediction.shape == prediction.shape
    assert not torch.allclose(metric_prediction, prediction)
    assert metric_mask.all()

    sparse_mask = valid_mask.clone()
    sparse_mask[:, :2, :2] = False
    sparse_prediction, sparse_metric_mask = apply_height_metric_resolution(
        prediction,
        sparse_mask,
        config,
    )
    assert sparse_metric_mask[:, :2, :2].any().item() is False
    assert sparse_prediction[:, :, :2, :2].sum().item() == pytest.approx(0.0)


def test_height_raster_records_and_sliding_window_prediction(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.height.raster_benchmarking import (
        _sliding_window_predict_height,
        resolve_height_raster_records,
    )

    raster_root = tmp_path / "rasters"
    raster_root.mkdir()
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[..., 1] = 120
    rgb[..., 2] = 70
    Image.fromarray(rgb, mode="RGB").save(raster_root / "scene_a_RGB.tif")
    Image.fromarray(np.ones((6, 6), dtype=np.float32), mode="F").save(
        raster_root / "scene_a_CHM.tif"
    )
    Image.fromarray(rgb, mode="RGB").save(raster_root / "scene_a_maxar1.tif")

    benchmark_config = HeightRasterBenchmarkConfig(
        checkpoint_path="best.pth",
        rgb_root_path=str(raster_root),
        height_root_path=str(raster_root),
        rgb_glob="*_RGB.tif",
        height_glob="*_CHM.tif",
        tile_size=4,
        stride=2,
        batch_size=2,
        wandb=_height_config(tmp_path).wandb,
    )
    records = resolve_height_raster_records(benchmark_config)
    assert len(records) == 1
    assert records[0].name == "scene_a"
    assert records[0].rgb_path == raster_root / "scene_a_RGB.tif"
    assert records[0].height_path == raster_root / "scene_a_CHM.tif"

    class ConstantHeightModel(torch.nn.Module):
        def forward(self, images):
            return {
                "height": torch.full(
                    (images.shape[0], 1, images.shape[-2], images.shape[-1]),
                    7.0,
                    dtype=images.dtype,
                    device=images.device,
                )
            }

    config = _height_config(tmp_path)
    image = Image.fromarray(rgb, mode="RGB")
    valid_mask = np.ones((6, 6), dtype=bool)
    valid_mask[0, 0] = False
    result = _sliding_window_predict_height(
        model=ConstantHeightModel(),
        image=image,
        valid_mask=valid_mask,
        training_config=config,
        benchmark_config=benchmark_config,
        device=torch.device("cpu"),
    )

    assert result.prediction.shape == (6, 6)
    assert result.weight_sum.shape == (6, 6)
    assert result.num_windows == 4
    assert result.prediction[1:, 1:].mean() == pytest.approx(7.0)
    assert result.weight_sum[0, 0] == pytest.approx(0.0)


def test_detector_height_precompute_geometry_uses_physical_20cm_grid(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.height.detector_precompute import (
        DetectorHeightTileRecord,
        detector_height_low_size,
        prepare_detector_height_tile,
    )

    image_path = tmp_path / "tile_train_box.tif"
    rgb = np.zeros((992, 992, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[..., 1] = 120
    rgb[..., 2] = 70
    Image.fromarray(rgb, mode="RGB").save(image_path)
    record = DetectorHeightTileRecord(
        dataset_name="tiny_detector",
        product_name="product_a",
        split="train",
        image_path=image_path,
        output_path=tmp_path / "height_maps" / "train" / image_path.name,
    )
    config = _height_config(tmp_path)
    config.dinov3_patch_size = 16

    assert detector_height_low_size((992, 992), source_gsd=0.05, target_gsd=0.20) == (
        248,
        248,
    )

    prepared = prepare_detector_height_tile(
        record,
        height_config=config,
        source_gsd=0.05,
        target_gsd=0.20,
    )

    assert prepared.original_size == (992, 992)
    assert prepared.low_size == (248, 248)
    assert prepared.padded_size == (256, 256)
    assert prepared.crop_left == 4
    assert prepared.crop_top == 4
    assert tuple(prepared.tensor.shape) == (3, 256, 256)


def test_detector_height_record_resolution_matches_height_tile_layout(tmp_path):
    from synthforest.training.height.detector_precompute import (
        resolve_detector_height_tile_records,
        resolve_detector_height_tile_records_from_tile_dir,
    )

    data_root = _write_tiny_detector_dataset(tmp_path / "detector_data")
    config = _detector_config(tmp_path, data_root)

    records = resolve_detector_height_tile_records(config, splits=["train"])

    assert len(records) == 1
    assert records[0].image_path == (
        data_root / "tiny_detector" / "product_a" / "tiles" / "train" / "tile_train_box.tif"
    )
    assert records[0].output_path == (
        data_root / "tiny_detector" / "product_a" / "height_maps" / "train" / "tile_train_box.tif"
    )

    tile_dir_records = resolve_detector_height_tile_records_from_tile_dir(
        tile_dir=data_root / "tiny_detector" / "product_a" / "tiles" / "train",
        split="train",
        height_maps_subdir="height_maps",
        dataset_name="tiny_detector",
        product_name="product_a",
    )
    assert len(tile_dir_records) == 1
    assert tile_dir_records[0].image_path == records[0].image_path
    assert tile_dir_records[0].output_path == records[0].output_path


def test_detector_height_precompute_smoke_and_skip_existing(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.height.detector_precompute import (
        run_detector_height_precompute,
    )

    data_root = _write_tiny_detector_dataset(tmp_path / "detector_data")
    detector_config = _detector_config(tmp_path, data_root)
    height_config = _height_config(tmp_path)
    checkpoint_path = tmp_path / "height" / "best.pth"
    _write_fake_height_checkpoint(checkpoint_path, height_config)

    summary = run_detector_height_precompute(
        detector_config=detector_config,
        height_checkpoint_path=checkpoint_path,
        splits=["train"],
        batch_size=1,
        device="cpu",
        source_gsd=1.0,
        target_gsd=2.0,
        summary_path=tmp_path / "summary.json",
    )

    output_path = (
        data_root / "tiny_detector" / "product_a" / "height_maps" / "train" / "tile_train_box.tif"
    )
    assert summary["num_processed"] == 1
    assert summary["num_skipped_existing"] == 0
    assert summary["num_failed"] == 0
    assert output_path.exists()
    with Image.open(output_path) as image:
        prediction = np.asarray(image)
    assert prediction.shape == (32, 32)
    assert prediction.dtype == np.float32
    assert np.isfinite(prediction).all()

    skipped = run_detector_height_precompute(
        detector_config=detector_config,
        height_checkpoint_path=checkpoint_path,
        splits=["train"],
        batch_size=1,
        device="cpu",
        source_gsd=1.0,
        target_gsd=2.0,
        summary_path=tmp_path / "summary.json",
    )

    assert skipped["num_processed"] == 0
    assert skipped["num_skipped_existing"] == 1
    assert skipped["num_failed"] == 0


def test_detector_height_precompute_writes_uncertainty_band_and_replaces_one_band(tmp_path):
    pytest.importorskip("torch")
    rasterio = pytest.importorskip("rasterio")
    from synthforest.training.dataset import load_detector_height_map
    from synthforest.training.height.detector_precompute import (
        run_detector_height_precompute,
    )

    data_root = _write_tiny_detector_dataset(tmp_path / "detector_data")
    detector_config = _detector_config(tmp_path, data_root)
    height_config = _height_config(tmp_path)
    height_config.height_output_channels = 2
    height_config.loss_name = "laplace_nll"
    height_config.validate()
    checkpoint_path = tmp_path / "height" / "best_uncertainty.pth"
    _write_fake_height_checkpoint(checkpoint_path, height_config)

    output_path = (
        data_root / "tiny_detector" / "product_a" / "height_maps" / "train" / "tile_train_box.tif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ones((32, 32), dtype=np.float32), mode="F").save(output_path)

    summary = run_detector_height_precompute(
        detector_config=detector_config,
        height_checkpoint_path=checkpoint_path,
        splits=["train"],
        batch_size=1,
        device="cpu",
        source_gsd=1.0,
        target_gsd=2.0,
        summary_path=tmp_path / "summary_uncertainty.json",
    )

    assert summary["output_bands"] == 2
    assert summary["num_processed"] == 1
    assert summary["num_skipped_existing"] == 0
    assert summary["num_failed"] == 0
    with rasterio.open(output_path) as dataset:
        assert dataset.count == 2
        assert dataset.descriptions == ("height_m", "log_variance_m2")
        tags = dataset.tags()
        height = dataset.read(1)
        log_variance = dataset.read(2)
    assert tags["detector_height_precompute_schema"] == "2"
    assert tags["height_checkpoint_path"] == str(checkpoint_path.resolve())
    assert tags["height_output_bands"] == "2"
    assert height.shape == (32, 32)
    assert log_variance.shape == (32, 32)
    assert np.isfinite(height).all()
    assert np.isfinite(log_variance).all()

    loaded_height, loaded_mask = load_detector_height_map(output_path)
    assert loaded_height.shape == (32, 32)
    assert loaded_mask.shape == (32, 32)
    assert np.allclose(loaded_height, height)

    skipped = run_detector_height_precompute(
        detector_config=detector_config,
        height_checkpoint_path=checkpoint_path,
        splits=["train"],
        batch_size=1,
        device="cpu",
        source_gsd=1.0,
        target_gsd=2.0,
        summary_path=tmp_path / "summary_uncertainty.json",
    )

    assert skipped["num_processed"] == 0
    assert skipped["num_skipped_existing"] == 1
    assert skipped["num_failed"] == 0

    second_checkpoint_path = tmp_path / "height" / "best_uncertainty_second.pth"
    _write_fake_height_checkpoint(second_checkpoint_path, height_config)
    regenerated = run_detector_height_precompute(
        detector_config=detector_config,
        height_checkpoint_path=second_checkpoint_path,
        splits=["train"],
        batch_size=1,
        device="cpu",
        source_gsd=1.0,
        target_gsd=2.0,
        summary_path=tmp_path / "summary_uncertainty_second.json",
    )

    assert regenerated["num_processed"] == 1
    assert regenerated["num_skipped_existing"] == 0
    assert regenerated["num_failed"] == 0
    with rasterio.open(output_path) as dataset:
        assert dataset.tags()["height_checkpoint_path"] == str(second_checkpoint_path.resolve())


def test_detector_height_tile_save_preserves_raster_metadata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from synthforest.training.height.detector_precompute import save_detector_height_tile

    source_path = tmp_path / "source_rgb.tif"
    output_path = tmp_path / "height.tif"
    transform = from_origin(140.0, 35.0, 0.05, 0.05)
    rgb = np.zeros((3, 4, 4), dtype=np.uint8)
    rgb[0] = 80
    rgb[1] = 120
    rgb[2] = 70
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(rgb)

    save_detector_height_tile(
        source_path=source_path,
        output_path=output_path,
        prediction=np.ones((4, 4), dtype=np.float32),
    )

    with rasterio.open(output_path) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes == ("float32",)
        assert dataset.width == 4
        assert dataset.height == 4
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.transform == transform


def test_dinov3_dpt_height_model_fake_backbone_shape_and_trainability(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.models import (
        assert_stage_trainability,
        build_dinov3_height_model,
        set_stage_trainability,
    )

    config = _height_config(tmp_path)
    model = build_dinov3_height_model(config)
    outputs = model(torch.zeros((2, 3, 32, 32), dtype=torch.float32))

    assert outputs["height"].shape == (2, 1, 32, 32)
    assert "log_variance" not in outputs
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("dinov3.")
    )

    set_stage_trainability(model, train_lora=False)
    assert_stage_trainability(model, train_lora=False)

    uncertainty_config = _height_config(tmp_path)
    uncertainty_config.height_output_channels = 2
    uncertainty_config.loss_name = "laplace_nll"
    uncertainty_model = build_dinov3_height_model(uncertainty_config)
    uncertainty_outputs = uncertainty_model(torch.zeros((2, 3, 32, 32), dtype=torch.float32))

    assert uncertainty_outputs["height"].shape == (2, 1, 32, 32)
    assert uncertainty_outputs["log_variance"].shape == (2, 1, 32, 32)
    assert uncertainty_outputs["log_variance"].min().item() >= (
        uncertainty_config.log_variance_min - 1e-6
    )
    assert uncertainty_outputs["log_variance"].max().item() <= (
        uncertainty_config.log_variance_max + 1e-6
    )


def test_dinov3_dpt_height_model_lora_trainability_when_peft_is_available(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from synthforest.training.models import (
        assert_stage_trainability,
        build_dinov3_height_model,
        set_stage_trainability,
    )

    config = _height_config(tmp_path)
    config.lora = LoRAConfig(
        num_blocks=2,
        rank=2,
        alpha=4.0,
        target_modules=["q_proj", "v_proj"],
    )
    config.stages.append(
        HeightTrainingStageConfig(
            name="joint_lora",
            max_iterations=1,
            eval_interval=1,
            checkpoint_interval=1,
            train_lora=True,
            optimizer=HeightStageOptimizerConfig(lora_lr=1e-5),
        )
    )
    model = build_dinov3_height_model(config)
    set_stage_trainability(model, train_lora=True)

    assert_stage_trainability(model, train_lora=True)
    assert any(
        "lora" in name and parameter.requires_grad for name, parameter in model.named_parameters()
    )


def test_height_training_evaluation_and_benchmark_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.height.benchmarking import run_height_benchmark
    from synthforest.training.height.evaluation import evaluate_height_checkpoint
    from synthforest.training.height.raster_benchmarking import run_height_raster_benchmark
    from synthforest.training.height.trainer import train_height_from_config

    data_root = _write_tiny_height_dataset(tmp_path / "data")
    real_root = _write_tiny_real_height_root(tmp_path / "real" / "NEON_CHM_1m_real_eval")
    config = _height_config(tmp_path, real_root=real_root)
    config.data_root_path = str(data_root)
    config.validate()

    output_dir = train_height_from_config(config)
    checkpoint_path = output_dir / "checkpoints" / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["checkpoint_type"] == "height"
    assert checkpoint["best_metric_name"] == "valid_real/mae"
    assert checkpoint["best_metric_direction"] == "minimize"

    eval_path = tmp_path / "eval" / "metrics.json"
    metrics = evaluate_height_checkpoint(
        checkpoint_path,
        validation_set="real",
        output_path=eval_path,
        device="cpu",
    )
    assert eval_path.exists()
    assert metrics["valid_real"]["mae"] >= 0.0
    assert metrics["valid_real"]["by_dataset"]["NEON_CHM_1m_real_eval"]["valid_pixels"] > 0

    benchmark_config = HeightBenchmarkConfig(
        checkpoint_path=str(checkpoint_path),
        data_root_path=str(real_root),
        output_folder=str(tmp_path / "benchmarks"),
        dataset_names=["NEON_CHM_1m_real_eval"],
        fold_name="test",
        batch_size=1,
        num_workers=0,
        device="cpu",
        metric_resolution=HeightMetricResolutionConfig(enabled=True, downsample_factor=2),
        wandb=config.wandb,
    )
    benchmark_output = run_height_benchmark(benchmark_config)
    summary = json.loads((benchmark_output / "metrics" / "summary.json").read_text())
    tiles = (benchmark_output / "metrics" / "tiles.jsonl").read_text().strip().splitlines()

    assert summary["metrics"]["mae"] >= 0.0
    assert summary["metric_resolution"]["downsample_factor"] == 2
    saved_benchmark_config = yaml.safe_load(
        (benchmark_output / "config" / "benchmark.yaml").read_text()
    )
    assert saved_benchmark_config["wandb"]["tags"] == config.wandb.tags
    assert summary["num_tiles"] == 1
    assert summary["num_summary_rows"] == 1
    assert len(tiles) == 2
    assert {json.loads(row)["row_type"] for row in tiles} == {"tile", "dataset_summary"}

    raster_rgb = tmp_path / "raster_rgb.tif"
    raster_height = tmp_path / "raster_height.tif"
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[..., 1] = 120
    rgb[..., 2] = 70
    Image.fromarray(rgb, mode="RGB").save(raster_rgb)
    Image.fromarray(np.ones((32, 32), dtype=np.float32), mode="F").save(raster_height)
    raster_config = HeightRasterBenchmarkConfig(
        checkpoint_path=str(checkpoint_path),
        output_folder=str(tmp_path / "raster_benchmarks"),
        rasters=[
            HeightRasterPairConfig(
                name="tiny_scene",
                rgb_path=str(raster_rgb),
                height_path=str(raster_height),
            )
        ],
        tile_size=16,
        stride=8,
        batch_size=2,
        device="cpu",
        metric_resolution=HeightMetricResolutionConfig(enabled=True, downsample_factor=2),
        wandb=config.wandb,
    )
    raster_output = run_height_raster_benchmark(raster_config)
    raster_summary = json.loads((raster_output / "metrics" / "summary.json").read_text())

    assert raster_summary["metrics"]["mae"] >= 0.0
    assert raster_summary["metric_resolution"]["downsample_factor"] == 2
    saved_raster_config = yaml.safe_load((raster_output / "config" / "benchmark.yaml").read_text())
    assert saved_raster_config["wandb"]["tags"] == config.wandb.tags
    assert raster_summary["num_rasters"] == 1
    assert (raster_output / "predictions" / "rasters" / "tiny_scene.tif").exists()


def test_height_laplace_uncertainty_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.height.trainer import train_height_from_config

    data_root = _write_tiny_height_dataset(tmp_path / "data")
    real_root = _write_tiny_real_height_root(tmp_path / "real" / "NEON_CHM_1m_real_eval")
    config = _height_config(tmp_path, real_root=real_root)
    config.data_root_path = str(data_root)
    config.run_name = "height-laplace-smoke"
    config.height_output_channels = 2
    config.loss_name = "laplace_nll"
    config.validate()

    output_dir = train_height_from_config(config)
    train_metrics = (output_dir / "metrics" / "train.jsonl").read_text()

    assert (output_dir / "checkpoints" / "best.pth").exists()
    assert "loss_laplace_nll" in train_metrics
    assert "log_variance_mean" in train_metrics


def test_height_resolution_robust_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.height.trainer import train_height_from_config

    data_root = _write_tiny_height_dataset(tmp_path / "data")
    real_root = _write_tiny_real_height_root(tmp_path / "real" / "NEON_CHM_1m_real_eval")
    config = _height_config(tmp_path, real_root=real_root)
    config.data_root_path = str(data_root)
    config.run_name = "height-resolution-smoke"
    config.resolution_degradation = HeightResolutionDegradationConfig(
        enabled=True,
        prob=1.0,
        base_gsd_m=0.5,
        target_gsd_m_values=[1.0],
        blur_sigma=[0.0, 0.0],
    )
    config.validate()

    output_dir = train_height_from_config(config)

    assert (output_dir / "checkpoints" / "best.pth").exists()
