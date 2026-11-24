from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from synthforest.training.benchmarking.config import BenchmarkConfig, parse_iou_thresholds
from synthforest.training.benchmarking.metrics import (
    RF1_50_95_IOU_THRESHOLDS,
    raster_level_bbox_metrics,
    tile_level_bbox_metrics,
    tile_level_segmentation_metrics,
    weighted_average_raster_metrics,
)
from synthforest.training.benchmarking.writers import (
    aggregate_gpkg_with_nms,
    load_prediction_coco,
    merge_coco_jsons,
    write_preaggregated_gpkg,
    write_prediction_coco,
    write_segmentation_prediction_coco,
    write_segmentation_truth_coco,
    write_subset_truth_coco,
)
from synthforest.training.checkpoint import load_checkpoint_model
from synthforest.training.coco_masks import (
    inspect_coco_mask_coverage,
)
from synthforest.training.config import DetectorTrainingConfig
from synthforest.training.layout import BenchmarkRunLayout, source_run_name
from synthforest.training.real_datasets import get_dataset_spec


def run_benchmark(config: BenchmarkConfig) -> Path:
    config.validate()
    checkpoint_path = config.resolved_checkpoint_path()
    if config.train_config_path is not None:
        train_config_path = Path(config.train_config_path)
        if not train_config_path.exists():
            raise FileNotFoundError(f"Training config does not exist: {train_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    layout, resumed_existing_benchmark = _resolve_benchmark_layout(config)
    prediction_source_layouts = _prediction_source_layouts(
        config=config,
        output_layout=layout,
    )

    model = None
    device = None
    reusable_metadata_layout = _complete_reusable_prediction_layout(
        config=config,
        source_layouts=prediction_source_layouts,
    )
    if (
        reusable_metadata_layout is not None
        and reusable_metadata_layout.training_config.exists()
        and reusable_metadata_layout.checkpoint_metadata.exists()
    ):
        train_config = DetectorTrainingConfig.from_yaml(reusable_metadata_layout.training_config)
        checkpoint_metadata = _read_json(reusable_metadata_layout.checkpoint_metadata)
        if checkpoint_metadata is None:
            raise ValueError(
                f"Invalid checkpoint metadata: {reusable_metadata_layout.checkpoint_metadata}"
            )
        print(
            "All requested product predictions are reusable; skipping detector model "
            "construction and checkpoint weight loading."
        )
    else:
        import torch

        device = torch.device(
            config.device
            if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model, train_config, checkpoint_metadata = load_checkpoint_model(
            checkpoint_path=checkpoint_path,
            device=device,
            config_path=config.train_config_path,
        )
    if train_config.model_type != config.model_type:
        raise ValueError(
            "Benchmark model_type does not match the checkpoint: "
            f"config={config.model_type!r}, "
            f"checkpoint={train_config.model_type!r}."
        )
    config.wandb.tags = _merged_wandb_tags(
        training_tags=train_config.wandb.tags,
        benchmark_tags=config.wandb.tags,
    )
    fine_tuning_metadata = _fine_tuning_metadata(
        train_config=train_config,
        run_folder=config.resolved_run_folder(),
    )
    layout.create()
    output_dir = layout.root
    fold_dir = layout.fold_predictions(config.fold_name)
    fold_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(layout.benchmark_config)
    train_config.save_yaml(layout.training_config)
    _write_json(layout.checkpoint_metadata, checkpoint_metadata)
    wandb_run, wandb_resumed = _init_wandb(
        config=config,
        output_dir=output_dir,
        training_run_name=_training_run_name(
            config,
            checkpoint_metadata,
        ),
        checkpoint_metadata=checkpoint_metadata,
        metadata_path=layout.wandb_metadata,
    )

    iou_thresholds = parse_iou_thresholds(config.eval_iou_threshold)
    raster_iou_thresholds = sorted(
        {
            *RF1_50_95_IOU_THRESHOLDS,
            *parse_iou_thresholds(config.raster_eval_iou_threshold),
        }
    )
    all_tile_rows: list[dict[str, Any]] = []
    all_segmentation_rows: list[dict[str, Any]] = []
    segmentation_coverage_rows: list[dict[str, Any]] = []
    all_raster_rows: list[dict[str, Any]] = []
    raster_coverage_rows: list[dict[str, Any]] = []
    product_count = 0
    skipped_edge_tile_count = 0
    reused_prediction_products = 0
    inferred_prediction_products = 0
    prediction_filter_stats = {
        "num_predictions_before_mask_filter": 0,
        "num_predictions_filtered_invalid": 0,
    }

    for dataset_name in config.datasets:
        dataset = get_dataset_spec(dataset_name)
        dataset.verify_fold(
            root_output_path=config.real_data_root,
            fold=config.fold_name,
            include_raster_inputs=False,
        )
        dataset_output_dir = fold_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        product_preds_cocos: list[Path] = []
        product_truth_cocos: list[Path] = []
        product_segm_preds_cocos: list[Path] = []
        product_segm_truth_cocos: list[Path] = []
        dataset_raster_rows: list[dict[str, Any]] = []
        products = list(
            dataset.iter_fold(
                root_output_path=config.real_data_root,
                fold=config.fold_name,
                include_raster_inputs=not config.tile_only,
            )
        )
        if config.max_products is not None:
            products = products[: config.max_products]

        for product in products:
            product_count += 1
            product_output_dir = dataset_output_dir / product.product_name
            product_output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Benchmarking {dataset_name}/{product.location}/{product.product_name} "
                f"({config.fold_name})"
            )
            reusable_prediction_path = _find_reusable_product_prediction(
                source_layouts=prediction_source_layouts,
                fold_name=config.fold_name,
                dataset_name=dataset_name,
                product_name=product.product_name,
                prefer_segmentation=(
                    config.evaluate_segmentation_when_available
                    and train_config.model_type == "maskrcnn"
                ),
            )
            if config.reuse_predictions and reusable_prediction_path is not None:
                predictions, evaluated_names, edge_filter_stats = load_prediction_coco(
                    reusable_prediction_path,
                    tile_dir=product.tile_dir,
                    truth_coco_path=product.truth_coco_path,
                )
                reused_prediction_products += 1
                print(f"Reusing saved predictions: {reusable_prediction_path}")
            else:
                if model is None or device is None:
                    raise RuntimeError(
                        "A required reusable prediction disappeared after benchmark "
                        "initialization; refusing to silently construct a model mid-run."
                    )
                from synthforest.training.benchmarking.inference import (
                    run_product_inference,
                )

                predictions, evaluated_names, edge_filter_stats = run_product_inference(
                    model=model,
                    tile_dir=product.tile_dir,
                    truth_coco_path=product.truth_coco_path,
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    max_tiles=config.max_tiles,
                    drop_edge_tiles=config.drop_edge_tiles,
                    edge_tile_black_pixel_threshold=config.edge_tile_black_pixel_threshold,
                    edge_tile_black_pixel_tolerance=config.edge_tile_black_pixel_tolerance,
                    avoid_fork=config.wandb.enabled,
                    training_config=train_config,
                    mask_probability_threshold=config.mask_probability_threshold,
                    source_gsd_m=(
                        dataset.ground_resolution
                        if dataset.ground_resolution is not None
                        else train_config.height_fusion.source_gsd_m
                    ),
                )
                inferred_prediction_products += 1
            skipped_edge_tile_count += edge_filter_stats["num_skipped_edge_tiles"]
            for key in prediction_filter_stats:
                prediction_filter_stats[key] += int(edge_filter_stats.get(key, 0))

            eval_truth_coco = product.truth_coco_path
            truth_image_names = _coco_image_names(product.truth_coco_path)
            if set(evaluated_names) != truth_image_names:
                eval_truth_coco = write_subset_truth_coco(
                    product.truth_coco_path,
                    product_output_dir / f"{product.product_name}_subset_truth_coco.json",
                    evaluated_names,
                )
            if reusable_prediction_path is not None and config.reuse_predictions:
                preds_coco = product_output_dir / f"{product.product_name}_preds_coco.json"
                if Path(reusable_prediction_path).resolve() != preds_coco.resolve():
                    shutil.copy2(reusable_prediction_path, preds_coco)
            else:
                preds_coco = write_prediction_coco(
                    truth_coco_path=eval_truth_coco,
                    predictions=predictions,
                    output_path=product_output_dir / f"{product.product_name}_preds_coco.json",
                )

            tile_metrics = tile_level_bbox_metrics(
                preds_coco_path=preds_coco,
                truth_coco_path=eval_truth_coco,
                max_dets=[1, 10, 100, dataset.tile_level_eval_max_dets],
                ground_resolution=dataset.ground_resolution,
                iou_thresholds=iou_thresholds,
            )
            tile_metrics.update(
                {
                    "dataset_name": dataset_name,
                    "location": product.location,
                    "product_name": product.product_name,
                    **edge_filter_stats,
                }
            )
            all_tile_rows.append(tile_metrics)
            product_preds_cocos.append(preds_coco)
            product_truth_cocos.append(Path(eval_truth_coco))

            segm_truth_coco: Optional[Path] = None
            segmentation_requested = (
                config.evaluate_segmentation_when_available
                and train_config.model_type == "maskrcnn"
            )
            if segmentation_requested:
                segm_truth_coco, mask_coverage = write_segmentation_truth_coco(
                    eval_truth_coco,
                    product_output_dir / f"{product.product_name}_segm_truth_coco.json",
                )
            else:
                mask_coverage = inspect_coco_mask_coverage(
                    eval_truth_coco,
                    validate_decoding=False,
                )
            segmentation_eligible, skip_reason = _segmentation_benchmark_eligibility(
                enabled=segmentation_requested,
                mask_coverage=mask_coverage,
                predictions=predictions,
            )
            segmentation_coverage_rows.append(
                {
                    "dataset_name": dataset_name,
                    "location": product.location,
                    "product_name": product.product_name,
                    "segmentation_eligible": segmentation_eligible,
                    "segmentation_skip_reason": skip_reason,
                    "num_evaluated_images": edge_filter_stats["num_evaluated_tiles"],
                    "num_filtered_segmentations": (
                        mask_coverage.total_annotations - mask_coverage.valid_segmentations
                    ),
                    **mask_coverage.to_dict(),
                }
            )
            if segmentation_eligible:
                if segm_truth_coco is None:
                    raise RuntimeError(
                        "Segmentation evaluation was marked eligible without segm truth."
                    )
                segm_preds_coco = write_segmentation_prediction_coco(
                    truth_coco_path=segm_truth_coco,
                    predictions=predictions,
                    output_path=(
                        product_output_dir / f"{product.product_name}_segm_preds_coco.json"
                    ),
                )
                segm_metrics = tile_level_segmentation_metrics(
                    preds_coco_path=segm_preds_coco,
                    truth_coco_path=segm_truth_coco,
                    max_dets=[
                        1,
                        10,
                        100,
                        dataset.tile_level_eval_max_dets,
                    ],
                    ground_resolution=dataset.ground_resolution,
                    iou_thresholds=iou_thresholds,
                )
                segm_metrics.update(
                    {
                        "dataset_name": dataset_name,
                        "location": product.location,
                        "product_name": product.product_name,
                        **edge_filter_stats,
                    }
                )
                all_segmentation_rows.append(segm_metrics)
                product_segm_preds_cocos.append(segm_preds_coco)
                product_segm_truth_cocos.append(segm_truth_coco)

            if not config.tile_only:
                raster_input_issue = _raster_input_issue(product)
                if raster_input_issue is not None:
                    raster_coverage_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "location": product.location,
                            "product_name": product.product_name,
                            "raster_eligible": False,
                            "raster_skip_reason": raster_input_issue,
                        }
                    )
                    if config.require_raster_inputs:
                        raise FileNotFoundError(
                            "Raster benchmark requested, but this product's raster inputs "
                            f"are unavailable ({raster_input_issue}): "
                            f"{dataset_name}/{product.product_name}"
                        )
                    print(
                        "Skipping raster-level metrics because truth/AOI GPKGs are "
                        f"unavailable ({raster_input_issue}): "
                        f"{dataset_name}/{product.product_name}"
                    )
                    continue
                raster_coverage_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "location": product.location,
                        "product_name": product.product_name,
                        "raster_eligible": True,
                        "raster_skip_reason": None,
                    }
                )
                pre_gpkg = write_preaggregated_gpkg(
                    predictions,
                    product_output_dir / f"{product.product_name}_pre_aggregated.gpkg",
                )
                agg_gpkg = aggregate_gpkg_with_nms(
                    preaggregated_gpkg_path=pre_gpkg,
                    output_path=product_output_dir / f"{product.product_name}_aggregated.gpkg",
                    score_threshold=config.score_threshold,
                    nms_threshold=config.nms_threshold,
                )
                raster_metrics = raster_level_bbox_metrics(
                    preds_gpkg_path=agg_gpkg,
                    truth_gpkg_path=product.truth_gpkg_path,
                    aoi_gpkg_path=product.aoi_gpkg_path,
                    ground_resolution=dataset.ground_resolution or 0.1,
                    iou_thresholds=raster_iou_thresholds,
                )
                raster_metrics.update(
                    {
                        "dataset_name": dataset_name,
                        "location": product.location,
                        "product_name": product.product_name,
                        "nms_iou_threshold": config.nms_threshold,
                        "nms_score_threshold": config.score_threshold,
                    }
                )
                all_raster_rows.append(raster_metrics)
                dataset_raster_rows.append(raster_metrics)

        merged_preds = merge_coco_jsons(
            product_preds_cocos,
            dataset_output_dir / "merged_preds_coco.json",
        )
        merged_truths = merge_coco_jsons(
            product_truth_cocos,
            dataset_output_dir / "merged_truths_coco.json",
        )
        dataset_tile_metrics = tile_level_bbox_metrics(
            preds_coco_path=merged_preds,
            truth_coco_path=merged_truths,
            max_dets=[1, 10, 100, dataset.tile_level_eval_max_dets],
            ground_resolution=dataset.ground_resolution,
            iou_thresholds=iou_thresholds,
        )
        dataset_tile_metrics.update(
            {
                "dataset_name": dataset_name,
                "location": dataset_name,
                "product_name": "average_over_rasters",
            }
        )
        all_tile_rows.append(dataset_tile_metrics)

        if product_segm_preds_cocos:
            merged_segm_preds = merge_coco_jsons(
                product_segm_preds_cocos,
                dataset_output_dir / "merged_segm_preds_coco.json",
            )
            merged_segm_truths = merge_coco_jsons(
                product_segm_truth_cocos,
                dataset_output_dir / "merged_segm_truths_coco.json",
            )
            dataset_segm_metrics = tile_level_segmentation_metrics(
                preds_coco_path=merged_segm_preds,
                truth_coco_path=merged_segm_truths,
                max_dets=[
                    1,
                    10,
                    100,
                    dataset.tile_level_eval_max_dets,
                ],
                ground_resolution=dataset.ground_resolution,
                iou_thresholds=iou_thresholds,
            )
            dataset_segm_metrics.update(
                {
                    "dataset_name": dataset_name,
                    "location": dataset_name,
                    "product_name": "average_over_mask_eligible_rasters",
                    "num_eligible_products": len(product_segm_preds_cocos),
                    "num_total_products": len(products),
                }
            )
            all_segmentation_rows.append(dataset_segm_metrics)

        if dataset_raster_rows:
            avg_raster = weighted_average_raster_metrics(dataset_raster_rows)
            avg_raster.update(
                {
                    "dataset_name": dataset_name,
                    "location": dataset_name,
                    "product_name": "average_over_rasters",
                }
            )
            all_raster_rows.append(avg_raster)

        _write_csv(layout.metrics_dir / "tile_level_metrics.csv", all_tile_rows)
        _write_csv(
            layout.metrics_dir / "tile_level_bbox_metrics.csv",
            all_tile_rows,
        )
        _write_csv(
            layout.metrics_dir / "tile_level_segm_metrics.csv",
            all_segmentation_rows,
        )
        _write_csv(
            layout.metrics_dir / "segmentation_coverage.csv",
            segmentation_coverage_rows,
        )
        if all_raster_rows:
            _write_csv(
                layout.metrics_dir / "raster_level_metrics.csv",
                all_raster_rows,
            )
        _write_csv(
            layout.metrics_dir / "raster_coverage.csv",
            raster_coverage_rows,
        )

    summary = {
        "output_dir": str(output_dir),
        "fold_name": config.fold_name,
        "datasets": config.datasets,
        "tile_level_metrics_csv": str(layout.metrics_dir / "tile_level_metrics.csv"),
        "tile_level_bbox_metrics_csv": str(layout.metrics_dir / "tile_level_bbox_metrics.csv"),
        "tile_level_segm_metrics_csv": str(layout.metrics_dir / "tile_level_segm_metrics.csv"),
        "segmentation_coverage_csv": str(layout.metrics_dir / "segmentation_coverage.csv"),
        "raster_level_metrics_csv": (
            str(layout.metrics_dir / "raster_level_metrics.csv") if all_raster_rows else None
        ),
        "raster_coverage_csv": str(layout.metrics_dir / "raster_coverage.csv"),
        "num_products": product_count,
        "num_tile_metric_rows": len(all_tile_rows),
        "num_segmentation_metric_rows": len(all_segmentation_rows),
        "num_segmentation_eligible_products": sum(
            bool(row["segmentation_eligible"]) for row in segmentation_coverage_rows
        ),
        "num_segmentation_skipped_products": sum(
            not bool(row["segmentation_eligible"]) for row in segmentation_coverage_rows
        ),
        "num_segmentation_eligible_images": sum(
            int(row["num_evaluated_images"])
            for row in segmentation_coverage_rows
            if bool(row["segmentation_eligible"])
        ),
        "num_segmentation_skipped_images": sum(
            int(row["num_evaluated_images"])
            for row in segmentation_coverage_rows
            if not bool(row["segmentation_eligible"])
        ),
        "num_raster_metric_rows": len(all_raster_rows),
        "num_raster_eligible_products": sum(
            bool(row["raster_eligible"]) for row in raster_coverage_rows
        ),
        "num_raster_skipped_products": sum(
            not bool(row["raster_eligible"]) for row in raster_coverage_rows
        ),
        "num_reused_prediction_products": reused_prediction_products,
        "num_inferred_prediction_products": inferred_prediction_products,
        "resumed_existing_benchmark": resumed_existing_benchmark,
        "num_skipped_edge_tiles": skipped_edge_tile_count,
        "drop_edge_tiles": config.drop_edge_tiles,
        "edge_tile_black_pixel_threshold": config.edge_tile_black_pixel_threshold,
        "valid_mask": train_config.to_dict()["valid_mask"],
        "valid_mask_policy_source": checkpoint_metadata["valid_mask_policy_source"],
        "model_type": train_config.model_type,
        "mask_probability_threshold": (
            config.mask_probability_threshold
            if config.mask_probability_threshold is not None
            else train_config.mask_probability_threshold
        ),
        "prediction_filter_stats": prediction_filter_stats,
        "prediction_invalid_fraction": (
            prediction_filter_stats["num_predictions_filtered_invalid"]
            / prediction_filter_stats["num_predictions_before_mask_filter"]
            if prediction_filter_stats["num_predictions_before_mask_filter"] > 0
            else 0.0
        ),
        "tile_only": config.tile_only,
        "eval_iou_thresholds": iou_thresholds,
        "raster_eval_iou_thresholds": raster_iou_thresholds,
    }
    if fine_tuning_metadata is not None:
        summary["fine_tuning"] = fine_tuning_metadata
    _write_json(layout.metrics_dir / "summary.json", summary)
    with open(
        layout.metrics_dir / "summary.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    _log_wandb_metrics(
        wandb_run=wandb_run,
        config=config,
        tile_rows=all_tile_rows,
        segmentation_rows=all_segmentation_rows,
        raster_rows=all_raster_rows,
        summary=summary,
        metrics_path=layout.metrics_dir / "wandb.json",
        wandb_resumed=wandb_resumed,
    )
    _finish_wandb(wandb_run)
    return output_dir


def _resolve_benchmark_layout(
    config: BenchmarkConfig,
) -> tuple[BenchmarkRunLayout, bool]:
    default_layout = BenchmarkRunLayout.resolve(
        output_root=config.output_folder,
        model_type=config.model_type,
        benchmark_name=config.resolved_benchmark_name(),
        run_folder=config.resolved_run_folder(),
    )
    if not config.resume_existing_benchmark:
        return default_layout, False

    if config.prediction_source is not None:
        source_layout = BenchmarkRunLayout(Path(config.prediction_source))
        if not source_layout.root.exists():
            raise FileNotFoundError(
                f"Configured prediction_source does not exist: {source_layout.root}"
            )
        if not _benchmark_layout_is_compatible(
            source_layout,
            config,
            allow_dataset_superset=False,
        ):
            raise ValueError(
                "Configured prediction_source is not compatible with this checkpoint, "
                "fold, dataset list, or inference filtering settings: "
                f"{source_layout.root}"
            )
        print(f"Continuing existing benchmark directory: {source_layout.root}")
        return source_layout, True

    if default_layout.root.exists():
        if _benchmark_layout_is_compatible(
            default_layout,
            config,
            allow_dataset_superset=False,
        ):
            print(f"Continuing existing benchmark directory: {default_layout.root}")
            return default_layout, True

    if config.benchmark_name is not None:
        return default_layout, False

    candidates = _compatible_benchmark_layouts(
        config,
        allow_dataset_superset=False,
    )
    if candidates:
        selected = max(candidates, key=_benchmark_layout_mtime)
        print(f"Continuing compatible existing benchmark directory: {selected.root}")
        return selected, True
    return default_layout, False


def _prediction_source_layouts(
    *,
    config: BenchmarkConfig,
    output_layout: BenchmarkRunLayout,
) -> list[BenchmarkRunLayout]:
    if not config.reuse_predictions:
        return []
    layouts: list[BenchmarkRunLayout] = []
    if output_layout.root.exists():
        layouts.append(output_layout)
    if config.prediction_source is not None:
        layouts.append(BenchmarkRunLayout(Path(config.prediction_source)))
    layouts.extend(
        _compatible_benchmark_layouts(
            config,
            allow_dataset_superset=True,
        )
    )
    unique: list[BenchmarkRunLayout] = []
    seen: set[Path] = set()
    for layout in layouts:
        resolved = layout.root.resolve()
        if resolved in seen:
            continue
        unique.append(layout)
        seen.add(resolved)
    return sorted(unique, key=_benchmark_layout_mtime, reverse=True)


def _complete_reusable_prediction_layout(
    *,
    config: BenchmarkConfig,
    source_layouts: list[BenchmarkRunLayout],
) -> Optional[BenchmarkRunLayout]:
    if not config.reuse_predictions or not source_layouts:
        return None
    for dataset_name in config.datasets:
        dataset = get_dataset_spec(dataset_name)
        products = list(
            dataset.iter_fold(
                root_output_path=config.real_data_root,
                fold=config.fold_name,
                include_raster_inputs=not config.tile_only,
            )
        )
        if config.max_products is not None:
            products = products[: config.max_products]
        for product in products:
            if (
                _find_reusable_product_prediction(
                    source_layouts=source_layouts,
                    fold_name=config.fold_name,
                    dataset_name=dataset_name,
                    product_name=product.product_name,
                    prefer_segmentation=False,
                )
                is None
            ):
                return None
    for layout in source_layouts:
        if layout.training_config.exists() and layout.checkpoint_metadata.exists():
            return layout
    return None


def _compatible_benchmark_layouts(
    config: BenchmarkConfig,
    *,
    allow_dataset_superset: bool,
) -> list[BenchmarkRunLayout]:
    if config.resolved_run_folder() is not None:
        parent = config.resolved_run_folder() / "benchmarks"
    else:
        parent = Path(config.output_folder) / config.model_type
    if not parent.exists():
        return []
    layouts: list[BenchmarkRunLayout] = []
    for config_path in parent.glob("*/config/benchmark.yaml"):
        layout = BenchmarkRunLayout(config_path.parents[1])
        if _benchmark_layout_is_compatible(
            layout,
            config,
            allow_dataset_superset=allow_dataset_superset,
        ):
            layouts.append(layout)
    return layouts


def _benchmark_layout_is_compatible(
    layout: BenchmarkRunLayout,
    config: BenchmarkConfig,
    *,
    allow_dataset_superset: bool,
) -> bool:
    if not layout.benchmark_config.exists():
        return False
    try:
        with open(layout.benchmark_config, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except (OSError, TypeError, yaml.YAMLError):
        return False
    if raw.get("model_type") != config.model_type:
        return False
    if raw.get("fold_name", "test") != config.fold_name:
        return False
    saved_datasets = [str(value) for value in raw.get("datasets", [])]
    if allow_dataset_superset:
        if not set(config.datasets).issubset(saved_datasets):
            return False
    elif saved_datasets != list(config.datasets):
        return False

    current_checkpoint = config.resolved_checkpoint_path().resolve()
    checkpoint_metadata = _read_json(layout.checkpoint_metadata)
    if checkpoint_metadata is not None and checkpoint_metadata.get("checkpoint_path"):
        if Path(str(checkpoint_metadata["checkpoint_path"])).resolve() != current_checkpoint:
            return False
    else:
        saved_checkpoint_name = str(raw.get("checkpoint_name", "best.pth"))
        if saved_checkpoint_name != config.checkpoint_name:
            return False

    scalar_fields = (
        "max_tiles",
        "drop_edge_tiles",
        "edge_tile_black_pixel_threshold",
        "edge_tile_black_pixel_tolerance",
        "mask_probability_threshold",
    )
    defaults = BenchmarkConfig(
        model_type=config.model_type,
        checkpoint_path=str(current_checkpoint),
        datasets=list(config.datasets),
    )
    for field_name in scalar_fields:
        saved_value = raw.get(field_name, getattr(defaults, field_name))
        if saved_value != getattr(config, field_name):
            return False

    saved_real_root = Path(str(raw.get("real_data_root", defaults.real_data_root))).resolve()
    if saved_real_root != Path(config.real_data_root).resolve():
        return False
    saved_train_config = raw.get("train_config_path")
    if _normalized_optional_path(saved_train_config) != _normalized_optional_path(
        config.train_config_path
    ):
        return False
    return (layout.predictions_dir / config.fold_name).exists()


def _normalized_optional_path(value: Any) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).resolve()


def _benchmark_layout_mtime(layout: BenchmarkRunLayout) -> float:
    for path in (layout.metrics_dir / "summary.json", layout.benchmark_config, layout.root):
        try:
            return path.stat().st_mtime
        except OSError:
            continue
    return 0.0


def _find_reusable_product_prediction(
    *,
    source_layouts: list[BenchmarkRunLayout],
    fold_name: str,
    dataset_name: str,
    product_name: str,
    prefer_segmentation: bool,
) -> Optional[Path]:
    for layout in source_layouts:
        product_dir = layout.fold_predictions(fold_name) / dataset_name / product_name
        candidates: list[Path] = []
        if prefer_segmentation:
            candidates.append(product_dir / f"{product_name}_segm_preds_coco.json")
        candidates.append(product_dir / f"{product_name}_preds_coco.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _coco_image_names(path: str | Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(image["file_name"]) for image in payload.get("images", [])}


def _raster_input_issue(product) -> Optional[str]:
    if product.truth_gpkg_path is None or product.aoi_gpkg_path is None:
        return "missing_registered_truth_or_aoi_gpkg"
    if not product.truth_gpkg_path.exists():
        return "missing_truth_gpkg_file"
    if not product.aoi_gpkg_path.exists():
        return "missing_aoi_gpkg_file"
    return None


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _segmentation_benchmark_eligibility(
    *,
    enabled: bool,
    mask_coverage,
    predictions,
) -> tuple[bool, Optional[str]]:
    predictions_have_masks = all(prediction.segmentations is not None for prediction in predictions)
    if not enabled:
        return False, "disabled_by_config"
    if not predictions_have_masks:
        return False, "model_has_no_mask_predictions"
    if not mask_coverage.is_complete and int(mask_coverage.valid_segmentations) <= 0:
        return (
            False,
            f"ground_truth_mask_coverage_{mask_coverage.status}",
        )
    return True, None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _read_json(path: str | Path) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, OSError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fine_tuning_metadata(
    *,
    train_config,
    run_folder: Optional[Path],
) -> Optional[dict[str, Any]]:
    initialization = getattr(train_config, "checkpoint_initialization", None)
    if initialization is None:
        return None
    sampling = train_config.train_sampling
    metadata: dict[str, Any] = {
        "real_data_fraction": float(sampling.fraction),
        "seed": sampling.seed if sampling.seed is not None else train_config.seed,
        "min_samples": int(sampling.min_samples),
        "train_dataset_names": list(train_config.train_dataset_names),
        "source_checkpoint_path": initialization.checkpoint_path,
    }
    if run_folder is not None:
        sampling_path = Path(run_folder) / "metadata" / "train_sampling.json"
        if sampling_path.exists():
            with open(sampling_path, "r", encoding="utf-8") as f:
                sampling_metadata = json.load(f)
            metadata["train_sampling"] = sampling_metadata
            for key in ("total_samples", "selected_samples"):
                value = sampling_metadata.get(key)
                if isinstance(value, (int, float)):
                    metadata[key] = int(value)
    return metadata


def _training_run_name(
    config: BenchmarkConfig,
    checkpoint_metadata: dict[str, Any],
) -> str:
    return source_run_name(
        config.resolved_checkpoint_path(),
        checkpoint_metadata.get("run_name"),
    )


def _merged_wandb_tags(
    *,
    training_tags: list[str],
    benchmark_tags: list[str],
) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in [*training_tags, *benchmark_tags]:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def _init_wandb(
    config: BenchmarkConfig,
    output_dir: Path,
    training_run_name: str,
    checkpoint_metadata: dict[str, Any],
    metadata_path: Path,
):
    if not config.wandb.enabled:
        return None, False
    try:
        import wandb
    except ImportError:
        message = (
            "benchmark wandb.enabled is true, but wandb is not installed. "
            "Install wandb or set benchmark wandb.enabled: false."
        )
        if config.wandb.required:
            raise RuntimeError(message)
        print(f"{message} Continuing without wandb logging.")
        return None, False

    wandb_config = config.to_dict()
    wandb_config["training_run_name"] = training_run_name
    wandb_config["checkpoint_metadata"] = checkpoint_metadata
    previous_metadata = _read_json(metadata_path)
    resume_run_id = None
    if config.wandb.resume_existing and previous_metadata is not None:
        candidate_id = previous_metadata.get("id")
        if candidate_id:
            resume_run_id = str(candidate_id)
    init_project = config.wandb.project
    init_entity = config.wandb.entity
    init_name = training_run_name
    if resume_run_id is not None and previous_metadata is not None:
        init_project = str(previous_metadata.get("project") or config.wandb.project)
        init_entity = previous_metadata.get("entity")
        init_name = str(previous_metadata.get("name") or training_run_name)
    try:
        init_kwargs = {
            "project": init_project,
            "entity": init_entity,
            "name": init_name,
            "tags": config.wandb.tags,
            "mode": config.wandb.mode,
            "config": wandb_config,
            "dir": str(output_dir),
            "job_type": "real_eval",
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
            print(f"Resuming existing W&B benchmark run: {resume_run_id}")
        run = wandb.init(
            **init_kwargs,
        )
    except Exception as exc:
        message = f"benchmark wandb initialization failed: {exc}"
        if config.wandb.required:
            raise RuntimeError(message) from exc
        print(f"{message} Continuing without wandb logging.")
        return None, False

    _write_json(
        metadata_path,
        {
            "project": init_project,
            "entity": init_entity,
            "name": init_name,
            "id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
            "step": config.wandb.step,
            "resumed": resume_run_id is not None,
        },
    )
    return run, resume_run_id is not None


def _log_wandb_metrics(
    wandb_run,
    config: BenchmarkConfig,
    tile_rows: list[dict[str, Any]],
    segmentation_rows: list[dict[str, Any]],
    raster_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metrics_path: Path,
    wandb_resumed: bool = False,
) -> None:
    if wandb_run is None:
        return
    payload = _build_wandb_metric_payload(
        config=config,
        tile_rows=tile_rows,
        raster_rows=raster_rows,
        segmentation_rows=segmentation_rows,
    )
    payload["benchmark/num_products"] = summary["num_products"]
    payload["benchmark/num_tile_metric_rows"] = summary["num_tile_metric_rows"]
    payload["benchmark/num_segmentation_metric_rows"] = summary["num_segmentation_metric_rows"]
    payload["benchmark/num_raster_metric_rows"] = summary["num_raster_metric_rows"]
    payload["benchmark/num_skipped_edge_tiles"] = summary["num_skipped_edge_tiles"]
    fine_tuning = summary.get("fine_tuning")
    if isinstance(fine_tuning, dict):
        _maybe_add_float(
            payload,
            "finetuning/real_data_fraction",
            fine_tuning.get("real_data_fraction"),
        )
        _maybe_add_float(
            payload,
            "finetuning/total_samples",
            fine_tuning.get("total_samples"),
        )
        _maybe_add_float(
            payload,
            "finetuning/selected_samples",
            fine_tuning.get("selected_samples"),
        )
    log_step = _wandb_log_step(
        wandb_run=wandb_run,
        configured_step=config.wandb.step,
        wandb_resumed=wandb_resumed,
    )
    wandb_run.log(payload, step=log_step)
    _write_json(metrics_path, payload)


def _wandb_log_step(wandb_run, configured_step: int, wandb_resumed: bool) -> int:
    configured_step = int(configured_step)
    if not wandb_resumed:
        return configured_step
    current_step = getattr(wandb_run, "step", None)
    try:
        return int(current_step) + 1
    except (TypeError, ValueError):
        return configured_step + 1


def _build_wandb_metric_payload(
    config: BenchmarkConfig,
    tile_rows: list[dict[str, Any]],
    raster_rows: list[dict[str, Any]],
    segmentation_rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float]:
    payload: dict[str, float] = {}
    tile_average_rows = {
        row["dataset_name"]: row
        for row in tile_rows
        if row.get("product_name") == "average_over_rasters"
    }
    raster_average_rows = {
        row["dataset_name"]: row
        for row in raster_rows
        if row.get("product_name") == "average_over_rasters"
    }
    segmentation_average_rows = {
        row["dataset_name"]: row
        for row in (segmentation_rows or [])
        if row.get("product_name") == "average_over_mask_eligible_rasters"
    }

    for dataset_name in config.datasets:
        tile_row = tile_average_rows.get(dataset_name)
        if tile_row is None:
            continue
        prefix = f"{config.fold_name}_{dataset_name}"
        _maybe_add_float(payload, f"{prefix}/bbox_map", tile_row.get("AP"))
        _maybe_add_float(payload, f"{prefix}/bbox_mar", tile_row.get("AR"))
        _maybe_add_float(payload, f"{prefix}/bbox_map50", tile_row.get("AP50"))
        _maybe_add_float(payload, f"{prefix}/bbox_map75", tile_row.get("AP75"))
        _maybe_add_float(payload, f"{prefix}/bbox_mar50", tile_row.get("AR50"))
        _maybe_add_float(payload, f"{prefix}/bbox_mar75", tile_row.get("AR75"))

        segmentation_row = segmentation_average_rows.get(dataset_name)
        if segmentation_row is not None:
            _maybe_add_float(
                payload,
                f"{prefix}/segm_map",
                segmentation_row.get("AP"),
            )
            _maybe_add_float(
                payload,
                f"{prefix}/segm_mar",
                segmentation_row.get("AR"),
            )
            _maybe_add_float(
                payload,
                f"{prefix}/segm_map50",
                segmentation_row.get("AP50"),
            )
            _maybe_add_float(
                payload,
                f"{prefix}/segm_map75",
                segmentation_row.get("AP75"),
            )

        raster_row = raster_average_rows.get(dataset_name)
        if raster_row is not None:
            _maybe_add_float(payload, f"{prefix}/bbox_rf1", raster_row.get("f1"))
            _maybe_add_float(
                payload,
                f"{prefix}/bbox_rf1_75",
                raster_row.get("rf1_75"),
            )
            _maybe_add_float(
                payload,
                f"{prefix}/bbox_rf1_50_95",
                raster_row.get("rf1_50_95"),
            )
    return payload


def _maybe_add_float(payload: dict[str, float], key: str, value: Any) -> None:
    if isinstance(value, (int, float)):
        payload[key] = float(value)


def _finish_wandb(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()
