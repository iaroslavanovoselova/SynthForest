import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from synthforest.training.benchmarking.config import BenchmarkConfig, parse_iou_thresholds
from synthforest.training.benchmarking.datasets import PreprocessedDatasetSpec
from synthforest.training.benchmarking.writers import (
    merge_coco_jsons,
    write_prediction_coco,
    write_segmentation_truth_coco,
    write_subset_truth_coco,
)
from synthforest.training.config import (
    CheckpointInitializationConfig,
    DetectorTrainingConfig,
    TrainSamplingConfig,
)


def _truth_coco():
    return {
        "images": [
            {"id": 10, "file_name": "tile_a.tif", "width": 64, "height": 64},
            {"id": 11, "file_name": "tile_b.tif", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 1,
                "bbox": [5, 6, 10, 12],
                "area": 120,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 11,
                "category_id": 1,
                "bbox": [1, 2, 5, 6],
                "area": 30,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
    }


def test_parse_iou_thresholds_formats():
    assert parse_iou_thresholds("0.75") == [0.75]
    assert parse_iou_thresholds("0.5,0.75") == [0.5, 0.75]
    assert parse_iou_thresholds("50:95") == [
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    with pytest.raises(ValueError):
        parse_iou_thresholds("1.5")


def test_benchmark_config_parses_yaml(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "fasterrcnn",
                "train_config_path": "train.yaml",
                "checkpoint_path": "last.pth",
                "real_data_root": "real",
                "datasets": ["QuebecTrees"],
                "fold_name": "valid",
                "tile_only": True,
                "eval_iou_threshold": "50:95",
                "max_products": 1,
                "max_tiles": 2,
                "drop_edge_tiles": True,
                "edge_tile_black_pixel_threshold": 0.5,
                "edge_tile_black_pixel_tolerance": 0,
            }
        ),
        encoding="utf-8",
    )

    config = BenchmarkConfig.from_yaml(config_path)

    assert config.checkpoint_path == "last.pth"
    assert config.datasets == ["QuebecTrees"]
    assert config.fold_name == "valid"
    assert config.max_products == 1
    assert config.drop_edge_tiles is True
    assert config.edge_tile_black_pixel_threshold == 0.5
    assert config.resolved_checkpoint_path().as_posix() == "last.pth"
    assert parse_iou_thresholds(config.eval_iou_threshold)[-1] == 0.95


def test_benchmark_config_resolves_run_folder_checkpoint(tmp_path):
    run_folder = tmp_path / "runs" / "fasterrcnn" / "train_run"
    config = BenchmarkConfig(
        model_type="fasterrcnn",
        run_folder=str(run_folder),
        checkpoint_name="best_bbox.pth",
        datasets=["QuebecTrees"],
    )

    config.validate()

    assert config.resolved_checkpoint_path() == (run_folder / "checkpoints" / "best_bbox.pth")
    assert config.resolved_run_folder() == run_folder


def test_benchmark_config_infers_run_folder_from_checkpoint_path(tmp_path):
    run_folder = tmp_path / "runs" / "fasterrcnn" / "train_run"
    checkpoint_path = run_folder / "checkpoints" / "best.pth"
    config = BenchmarkConfig(
        model_type="fasterrcnn",
        checkpoint_path=str(checkpoint_path),
        datasets=["QuebecTrees"],
    )

    config.validate()

    assert config.resolved_checkpoint_path() == checkpoint_path
    assert config.resolved_run_folder() == run_folder


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "",
        ".",
        "/tmp/best.pth",
        "../best.pth",
        "stages/../../best.pth",
    ],
)
def test_benchmark_config_rejects_unsafe_checkpoint_names(
    tmp_path,
    checkpoint_name,
):
    config = BenchmarkConfig(
        model_type="fasterrcnn",
        run_folder=str(tmp_path / "run"),
        checkpoint_name=checkpoint_name,
        datasets=["QuebecTrees"],
    )

    with pytest.raises(ValueError, match="checkpoint_name"):
        config.validate()


def test_benchmark_config_can_defer_validation_for_cli_overrides(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "fasterrcnn",
                "checkpoint_path": None,
                "datasets": ["QuebecTrees"],
            }
        ),
        encoding="utf-8",
    )

    config = BenchmarkConfig.from_yaml(config_path, validate=False)
    config.checkpoint_path = "last.pth"
    config.validate()

    assert config.checkpoint_path == "last.pth"


def test_preprocessed_dataset_iter_fold_resolves_scale_factor_coco(tmp_path):
    root = tmp_path / "real"
    product_root = root / "location_a" / "product_a"
    tiles = product_root / "tiles" / "test"
    tiles.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(tiles / "tile_a.tif")
    (product_root / "product_a_coco_sf1p0_test.json").write_text(
        json.dumps(_truth_coco()),
        encoding="utf-8",
    )
    spec = PreprocessedDatasetSpec(
        dataset_name="TinyReal",
        ground_resolution=None,
        scale_factor=1.0,
        tile_level_eval_max_dets=100,
        locations=["location_a"],
        products_per_location={"location_a": ["product_a"]},
        raster_level_eval_inputs={"test": {"product_a": {}}},
    )

    spec.verify_fold(root, fold="test", include_raster_inputs=False)
    product = next(spec.iter_fold(root, fold="test", include_raster_inputs=False))

    assert product.tile_dir == tiles
    assert product.truth_coco_path.name == "product_a_coco_sf1p0_test.json"
    assert product.truth_gpkg_path is None


def test_prediction_coco_subset_and_merge(tmp_path):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(_truth_coco()), encoding="utf-8")
    subset_path = write_subset_truth_coco(truth_path, tmp_path / "subset.json", ["tile_a.tif"])
    prediction = SimpleNamespace(
        image_info={"id": 999, "file_name": "tile_a.tif", "width": 64, "height": 64},
        image_path=tmp_path / "tile_a.tif",
        boxes_xyxy=[[10.0, 12.0, 20.0, 30.0]],
        scores=[0.9],
        labels=[1],
    )

    pred_path = write_prediction_coco(subset_path, [prediction], tmp_path / "preds.json")
    pred_coco = json.loads(pred_path.read_text(encoding="utf-8"))

    assert pred_coco["images"][0]["id"] == 10
    assert pred_coco["annotations"][0]["image_id"] == 10
    assert pred_coco["annotations"][0]["bbox"] == [10.0, 12.0, 10.0, 18.0]
    assert pred_coco["annotations"][0]["score"] == 0.9

    merged_path = merge_coco_jsons([pred_path, pred_path], tmp_path / "merged.json")
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert len(merged["images"]) == 2
    assert len(merged["annotations"]) == 2
    assert {ann["image_id"] for ann in merged["annotations"]} == {1, 2}


def test_segmentation_truth_coco_filters_invalid_masks(tmp_path):
    pytest.importorskip("pycocotools")
    truth = _truth_coco()
    truth["annotations"][0]["segmentation"] = [[5, 6, 15, 6, 15, 18, 5, 18]]
    truth["annotations"][1]["segmentation"] = None
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")

    segm_truth_path, coverage = write_segmentation_truth_coco(
        truth_path,
        tmp_path / "segm_truth.json",
    )

    segm_truth = json.loads(segm_truth_path.read_text(encoding="utf-8"))
    assert len(segm_truth["images"]) == 2
    assert [ann["id"] for ann in segm_truth["annotations"]] == [1]
    assert coverage.status == "partial"
    assert coverage.total_annotations == 2
    assert coverage.valid_segmentations == 1
    assert coverage.missing_segmentations == 1


def test_segmentation_eligibility_allows_partial_valid_ground_truth():
    from synthforest.training.benchmarking.runner import (
        _segmentation_benchmark_eligibility,
    )

    partial_with_valid_masks = SimpleNamespace(
        is_complete=False,
        status="partial",
        valid_segmentations=1,
    )
    no_valid_masks = SimpleNamespace(
        is_complete=False,
        status="partial",
        valid_segmentations=0,
    )
    predictions = [SimpleNamespace(segmentations=[])]

    assert _segmentation_benchmark_eligibility(
        enabled=True,
        mask_coverage=partial_with_valid_masks,
        predictions=predictions,
    ) == (True, None)
    assert _segmentation_benchmark_eligibility(
        enabled=True,
        mask_coverage=no_valid_masks,
        predictions=predictions,
    ) == (False, "ground_truth_mask_coverage_partial")


def test_wandb_payload_uses_fold_dataset_sections():
    from synthforest.training.benchmarking.runner import _build_wandb_metric_payload

    config = BenchmarkConfig(
        model_type="fasterrcnn",
        checkpoint_path="outputs/training/train_run/best.pth",
        datasets=["QuebecTrees", "Detectree2"],
        fold_name="valid",
    )
    tile_rows = [
        {
            "dataset_name": "QuebecTrees",
            "product_name": "average_over_rasters",
            "AP": 0.1,
            "AR": 0.2,
            "AP50": 0.3,
            "AP75": 0.4,
            "AR50": 0.5,
            "AR75": 0.6,
        },
        {
            "dataset_name": "QuebecTrees",
            "product_name": "some_product",
            "AP": 0.99,
        },
    ]
    raster_rows = [
        {
            "dataset_name": "QuebecTrees",
            "product_name": "average_over_rasters",
            "f1": 0.7,
        }
    ]

    payload = _build_wandb_metric_payload(config, tile_rows, raster_rows)

    assert payload == {
        "valid_QuebecTrees/bbox_map": 0.1,
        "valid_QuebecTrees/bbox_mar": 0.2,
        "valid_QuebecTrees/bbox_map50": 0.3,
        "valid_QuebecTrees/bbox_map75": 0.4,
        "valid_QuebecTrees/bbox_mar50": 0.5,
        "valid_QuebecTrees/bbox_mar75": 0.6,
        "valid_QuebecTrees/bbox_rf1": 0.7,
    }


def test_benchmark_wandb_tags_append_to_training_tags():
    from synthforest.training.benchmarking.runner import _merged_wandb_tags

    assert _merged_wandb_tags(
        training_tags=["dinov3", "detector", "fasterrcnn"],
        benchmark_tags=["real-eval", "detector", "canopyrs"],
    ) == ["dinov3", "detector", "fasterrcnn", "real-eval", "canopyrs"]


def test_benchmark_finetune_metadata_reads_sampling_summary(tmp_path):
    from synthforest.training.benchmarking.runner import _fine_tuning_metadata

    run_folder = tmp_path / "run"
    metadata_dir = run_folder / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "train_sampling.json").write_text(
        json.dumps(
            {
                "total_samples": 12,
                "selected_samples": 4,
                "fraction": 0.25,
                "per_dataset": {
                    "selvamask": {"total_samples": 4, "selected_samples": 1},
                    "bamforests": {"total_samples": 8, "selected_samples": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    train_config = DetectorTrainingConfig(
        train_dataset_names=["selvamask", "bamforests"],
        model_type="fasterrcnn",
        dinov3_use_fake_backbone=True,
        checkpoint_initialization=CheckpointInitializationConfig(
            checkpoint_path="synthetic/checkpoints/best.pth"
        ),
        train_sampling=TrainSamplingConfig(fraction=0.25, seed=17, min_samples=1),
    )

    metadata = _fine_tuning_metadata(
        train_config=train_config,
        run_folder=run_folder,
    )

    assert metadata["real_data_fraction"] == pytest.approx(0.25)
    assert metadata["seed"] == 17
    assert metadata["selected_samples"] == 4
    assert metadata["total_samples"] == 12
    assert set(metadata["train_sampling"]["per_dataset"]) == {"selvamask", "bamforests"}
    assert (
        _fine_tuning_metadata(
            train_config=DetectorTrainingConfig(
                train_dataset_names=["synthetic"],
                model_type="fasterrcnn",
                dinov3_use_fake_backbone=True,
            ),
            run_folder=run_folder,
        )
        is None
    )


def test_benchmark_wandb_logs_finetune_fraction(tmp_path):
    from synthforest.training.benchmarking.runner import _log_wandb_metrics

    class FakeRun:
        def __init__(self):
            self.logged = None
            self.step = None

        def log(self, payload, step):
            self.logged = payload
            self.step = step

    config = BenchmarkConfig(
        model_type="fasterrcnn",
        checkpoint_path=str(tmp_path / "best.pth"),
        datasets=["QuebecTrees"],
    )
    config.wandb.step = 5
    run = FakeRun()

    _log_wandb_metrics(
        wandb_run=run,
        config=config,
        tile_rows=[],
        segmentation_rows=[],
        raster_rows=[],
        summary={
            "num_products": 1,
            "num_tile_metric_rows": 0,
            "num_segmentation_metric_rows": 0,
            "num_raster_metric_rows": 0,
            "num_skipped_edge_tiles": 0,
            "fine_tuning": {
                "real_data_fraction": 0.25,
                "total_samples": 12,
                "selected_samples": 4,
            },
        },
        metrics_path=tmp_path / "wandb.json",
    )

    assert run.step == 5
    assert run.logged["finetuning/real_data_fraction"] == pytest.approx(0.25)
    assert run.logged["finetuning/total_samples"] == pytest.approx(12)
    assert run.logged["finetuning/selected_samples"] == pytest.approx(4)


def test_benchmark_wandb_resume_logs_next_step(tmp_path):
    from synthforest.training.benchmarking.runner import _log_wandb_metrics

    class FakeRun:
        def __init__(self):
            self.logged = None
            self.step = 1
            self.logged_step = None

        def log(self, payload, step):
            self.logged = payload
            self.logged_step = step

    config = BenchmarkConfig(
        model_type="fasterrcnn",
        checkpoint_path=str(tmp_path / "best.pth"),
        datasets=["QuebecTrees"],
    )
    config.wandb.step = 0
    run = FakeRun()

    _log_wandb_metrics(
        wandb_run=run,
        config=config,
        tile_rows=[],
        segmentation_rows=[],
        raster_rows=[],
        summary={
            "num_products": 1,
            "num_tile_metric_rows": 0,
            "num_segmentation_metric_rows": 0,
            "num_raster_metric_rows": 0,
            "num_skipped_edge_tiles": 0,
        },
        metrics_path=tmp_path / "wandb.json",
        wandb_resumed=True,
    )

    assert run.logged_step == 2


def test_benchmark_tile_dataset_can_drop_black_edge_tiles(tmp_path):
    pytest.importorskip("torch")
    from synthforest.training.benchmarking.inference import BenchmarkTileDataset

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    Image.new("RGB", (16, 16), color=(80, 120, 70)).save(tiles / "full.tif")
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(tiles / "edge.tif")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "full.tif", "width": 16, "height": 16},
                    {"id": 2, "file_name": "edge.tif", "width": 16, "height": 16},
                ],
                "annotations": [],
                "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = BenchmarkTileDataset(
        tile_dir=tiles,
        truth_coco_path=truth_path,
        drop_edge_tiles=True,
        edge_tile_black_pixel_threshold=0.5,
    )

    assert dataset.evaluated_file_names == ["full.tif"]
    assert dataset.edge_filter_stats == {
        "num_evaluated_tiles": 1,
        "num_skipped_edge_tiles": 1,
    }


def test_benchmark_tile_dataset_loads_uncertainty_height_channels(tmp_path):
    torch = pytest.importorskip("torch")
    rasterio = pytest.importorskip("rasterio")
    from synthforest.training.benchmarking.inference import BenchmarkTileDataset

    product = tmp_path / "product"
    tiles = product / "tiles" / "valid"
    height_dir = product / "height_maps" / "valid"
    tiles.mkdir(parents=True)
    height_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(80, 120, 70)).save(tiles / "tile_a.tif")
    with rasterio.open(
        height_dir / "tile_a.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=2,
        dtype="float32",
    ) as dataset:
        dataset.write(
            np.stack(
                [
                    np.full((16, 16), 12.0, dtype=np.float32),
                    np.full((16, 16), 2.0, dtype=np.float32),
                ],
                axis=0,
            )
        )
    truth_path = product / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "tile_a.tif", "width": 16, "height": 16}],
                "annotations": [],
                "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
            }
        ),
        encoding="utf-8",
    )
    training_config = DetectorTrainingConfig(
        train_dataset_names=["unused"],
        model_type="fasterrcnn",
        dinov3_use_fake_backbone=True,
    )
    training_config.height_fusion.enabled = True
    training_config.height_fusion.use_uncertainty = True
    training_config.height_fusion.generate_missing = False

    dataset = BenchmarkTileDataset(
        tile_dir=tiles,
        truth_coco_path=truth_path,
        training_config=training_config,
    )
    image, valid_mask, _ = dataset[0]

    assert image.shape == (5, 16, 16)
    assert torch.allclose(image[3], torch.full((16, 16), 12.0))
    assert torch.allclose(image[4], torch.full((16, 16), 2.0))
    assert valid_mask.all()


def test_benchmark_inference_filters_predictions_by_alpha_mask(tmp_path):
    torch = pytest.importorskip("torch")
    from synthforest.training.benchmarking.inference import (
        run_product_inference,
    )
    from synthforest.training.config import (
        DetectorTrainingConfig,
        ValidMaskConfig,
    )

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    rgba = np.full((16, 16, 4), 100, dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[:, 8:, 3] = 0
    Image.fromarray(rgba, mode="RGBA").save(tiles / "edge.tif")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "edge.tif", "width": 16, "height": 16}],
                "annotations": [],
                "categories": [{"id": 1, "name": "tree"}],
            }
        ),
        encoding="utf-8",
    )

    class FixedModel(torch.nn.Module):
        def forward(self, images):
            return [
                {
                    "boxes": torch.tensor(
                        [[0.0, 0.0, 8.0, 16.0], [8.0, 0.0, 16.0, 16.0]],
                        device=images[0].device,
                    ),
                    "scores": torch.tensor([0.9, 0.8], device=images[0].device),
                    "labels": torch.tensor([1, 1], device=images[0].device),
                }
                for _ in images
            ]

    training_config = DetectorTrainingConfig(
        train_dataset_names=["unused"],
        model_type="fasterrcnn",
        dinov3_use_fake_backbone=True,
        valid_mask=ValidMaskConfig(prediction_min_valid_fraction=0.6),
    )
    predictions, _, stats = run_product_inference(
        model=FixedModel(),
        tile_dir=tiles,
        truth_coco_path=truth_path,
        device=torch.device("cpu"),
        batch_size=1,
        num_workers=0,
        training_config=training_config,
    )

    assert predictions[0].boxes_xyxy == [[0.0, 0.0, 8.0, 16.0]]
    assert stats["num_predictions_before_mask_filter"] == 2
    assert stats["num_predictions_filtered_invalid"] == 1
    assert stats["prediction_invalid_fraction"] == pytest.approx(0.5)


def test_benchmark_cli_smoke_tile_only(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from synthforest.training.config import (
        DetectorTrainingConfig,
        TrainingStageConfig,
    )
    from synthforest.training.models import build_dinov3_detector

    real_root = tmp_path / "real"
    product_root = real_root / "quebec_trees" / "20210902_sblz1_p4rtk_rgb"
    tiles = product_root / "tiles" / "valid"
    tiles.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(80, 120, 70)).save(tiles / "tile_a.tif")
    (product_root / "20210902_sblz1_p4rtk_rgb_coco_gr0p03_valid.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "tile_a.tif", "width": 64, "height": 64}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [12, 12, 24, 24],
                        "area": 576,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
            }
        ),
        encoding="utf-8",
    )
    train_config = DetectorTrainingConfig(
        data_root_path=str(tmp_path / "unused"),
        train_dataset_names=["unused"],
        train_output_path=str(tmp_path / "train_out"),
        run_name="smoke",
        model_type="fasterrcnn",
        dinov3_use_fake_backbone=True,
        dinov3_embed_dim=8,
        backbone_out_channels=8,
        batch_size=1,
        grad_accumulation_steps=1,
        stages=[
            TrainingStageConfig(
                max_iterations=1,
                eval_interval=1,
                checkpoint_interval=1,
                train_log_interval=1,
            )
        ],
        dataloader_num_workers=0,
        use_amp=False,
        anchor_sizes=[8, 16, 32],
        box_predictions_per_image=5,
    )
    train_config.augmentation.resize_min_sizes_train = [64]
    train_config.augmentation.resize_min_size_eval = 64
    train_config.augmentation.max_size = 64
    run_folder = tmp_path / "runs" / "fasterrcnn" / "smoke"
    checkpoint_path = run_folder / "checkpoints" / "best.pth"
    checkpoint_path.parent.mkdir(parents=True)
    model = build_dinov3_detector(train_config)
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": model.state_dict(),
            "training_config": train_config.to_dict(),
            "iteration": 0,
        },
        checkpoint_path,
    )

    benchmark_config = BenchmarkConfig(
        model_type="fasterrcnn",
        train_config_path=None,
        run_folder=None,
        checkpoint_name="best.pth",
        checkpoint_path=None,
        real_data_root=str(real_root),
        output_folder=str(tmp_path / "benchmarks"),
        benchmark_name="smoke_benchmark",
        datasets=["QuebecTrees"],
        fold_name="valid",
        tile_only=True,
        batch_size=1,
        num_workers=0,
        max_products=1,
        max_tiles=1,
    )
    benchmark_config.wandb.enabled = False
    benchmark_config_path = tmp_path / "benchmark.yaml"
    benchmark_config.save_yaml(benchmark_config_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "synthforest.cli",
            "evaluate",
            "crowns",
            "--config",
            str(benchmark_config_path),
            "--run-folder",
            str(run_folder),
            "--checkpoint-name",
            "best.pth",
        ],
        check=True,
    )

    summary = json.loads(
        (run_folder / "benchmarks" / "smoke_benchmark" / "metrics" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["num_products"] == 1
    assert summary["num_tile_metric_rows"] == 2
    assert "prediction_filter_stats" in summary
    assert (
        run_folder / "benchmarks" / "smoke_benchmark" / "metrics" / "tile_level_metrics.csv"
    ).exists()
