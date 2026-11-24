"""Raster NMS search adapted from CanopyRS for this repository's benchmark artifacts."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from synthforest.training.benchmarking.config import (
    NmsSearchConfig,
    parse_iou_thresholds,
)
from synthforest.training.benchmarking.metrics import (
    RF1_50_95_IOU_THRESHOLDS,
    raster_f1_for_iou_thresholds,
    raster_level_bbox_metrics,
    weighted_average_raster_metrics,
)
from synthforest.training.benchmarking.runner import run_benchmark
from synthforest.training.benchmarking.writers import (
    aggregate_gpkg_with_nms,
    load_prediction_coco,
    write_preaggregated_gpkg,
)
from synthforest.training.layout import BenchmarkRunLayout
from synthforest.training.real_datasets import get_dataset_spec


def run_nms_search(config: NmsSearchConfig) -> Path:
    config.validate()
    search_output = config.resolved_search_output()
    search_output.mkdir(parents=True, exist_ok=True)
    config.save_yaml(search_output / "config" / "search.yaml")

    # Reuse a compatible validation benchmark when present. If it is absent,
    # run_benchmark creates the tile predictions once; every grid point below
    # consumes those same saved predictions.
    prediction_benchmark_config = config.as_prediction_benchmark_config()
    prediction_benchmark_output = run_benchmark(prediction_benchmark_config)
    prediction_layout = BenchmarkRunLayout(prediction_benchmark_output)

    objective_iou_thresholds = parse_iou_thresholds(config.objective_iou_threshold)
    report_iou_thresholds = sorted(
        {
            *RF1_50_95_IOU_THRESHOLDS,
            *parse_iou_thresholds(config.raster_eval_iou_threshold),
            *objective_iou_thresholds,
        }
    )

    product_inputs: list[dict[str, Any]] = []
    skipped_products: list[dict[str, str]] = []
    preaggregated_dir = search_output / "pre_aggregated_predictions"
    for dataset_name in config.datasets:
        dataset = get_dataset_spec(dataset_name)
        dataset.verify_fold(
            root_output_path=config.real_data_root,
            fold="valid",
            include_raster_inputs=False,
        )
        products = list(
            dataset.iter_fold(
                root_output_path=config.real_data_root,
                fold="valid",
                include_raster_inputs=True,
            )
        )
        if config.max_products is not None:
            products = products[: config.max_products]
        for product in products:
            raster_input_issue = _raster_input_issue(product)
            if raster_input_issue is not None:
                skipped_products.append(
                    {
                        "dataset_name": dataset_name,
                        "location": product.location,
                        "product_name": product.product_name,
                        "reason": raster_input_issue,
                    }
                )
                continue
            prediction_path = (
                prediction_layout.fold_predictions("valid")
                / dataset_name
                / product.product_name
                / f"{product.product_name}_preds_coco.json"
            )
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Validation prediction COCO was not produced for NMS search: {prediction_path}"
                )
            predictions, _, _ = load_prediction_coco(
                prediction_path,
                tile_dir=product.tile_dir,
                truth_coco_path=product.truth_coco_path,
            )
            preaggregated_path = (
                preaggregated_dir
                / dataset_name
                / product.location
                / f"{product.product_name}_pre_aggregated.gpkg"
            )
            write_preaggregated_gpkg(predictions, preaggregated_path)
            product_inputs.append(
                {
                    "dataset_name": dataset_name,
                    "location": product.location,
                    "product_name": product.product_name,
                    "preaggregated_gpkg_path": str(preaggregated_path),
                    "truth_gpkg_path": str(product.truth_gpkg_path),
                    "aoi_gpkg_path": str(product.aoi_gpkg_path),
                    "ground_resolution": float(dataset.ground_resolution or 0.1),
                }
            )

    if not product_inputs:
        raise ValueError(
            "No selected validation products have registered truth and AOI GPKGs. "
            "Choose raster-capable datasets in the NMS search config."
        )

    tasks: list[dict[str, Any]] = []
    for nms_iou_threshold in config.nms_iou_thresholds:
        for nms_score_threshold in config.nms_score_thresholds:
            for product_input in product_inputs:
                tasks.append(
                    {
                        **product_input,
                        "nms_iou_threshold": float(nms_iou_threshold),
                        "nms_score_threshold": float(nms_score_threshold),
                        "report_iou_thresholds": report_iou_thresholds,
                        "objective_iou_thresholds": objective_iou_thresholds,
                        "output_root": str(search_output / "aggregated_predictions"),
                    }
                )

    print(
        "Starting validation NMS search with "
        f"{len(tasks)} product/parameter tasks and {config.n_workers} worker(s)."
    )
    result_rows = _run_search_tasks(tasks, n_workers=config.n_workers)
    result_rows.extend(_average_search_rows(result_rows))
    results_path = _write_csv(
        search_output / "metrics" / "nms_search_results.csv",
        result_rows,
    )
    _write_csv(
        search_output / "metrics" / "raster_coverage.csv",
        [
            *[
                {
                    "dataset_name": row["dataset_name"],
                    "location": row["location"],
                    "product_name": row["product_name"],
                    "raster_eligible": True,
                    "reason": None,
                }
                for row in product_inputs
            ],
            *[
                {
                    **row,
                    "raster_eligible": False,
                }
                for row in skipped_products
            ],
        ],
    )

    global_average_rows = [
        row
        for row in result_rows
        if row.get("dataset_name") == "all_datasets"
        and row.get("product_name") == "average_over_rasters"
    ]
    if not global_average_rows:
        raise RuntimeError("NMS search did not produce an all-raster average.")
    best = sorted(
        global_average_rows,
        key=lambda row: (
            -float(row["objective_f1"]),
            -float(row.get("rf1_75", float("-inf"))),
            float(row["nms_score_threshold"]),
            float(row["nms_iou_threshold"]),
        ),
    )[0]

    optimal_payload = {
        "source_note": (
            "Adapted from CanopyRS validation-set raster NMS search for "
            "synthforest prediction artifacts."
        ),
        "checkpoint_path": str(config.resolved_checkpoint_path()),
        "prediction_benchmark_output": str(prediction_benchmark_output),
        "datasets": list(config.datasets),
        "fold_name": "valid",
        "objective_iou_thresholds": objective_iou_thresholds,
        "nms_algorithm": config.nms_algorithm,
        "nms_threshold": float(best["nms_iou_threshold"]),
        "score_threshold": float(best["nms_score_threshold"]),
        "objective_f1": float(best["objective_f1"]),
        "rf1_75": float(best["rf1_75"]),
        "rf1_50_95": float(best["rf1_50_95"]),
    }
    optimal_path = search_output / "optimal_nms.yaml"
    optimal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(optimal_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(optimal_payload, file, sort_keys=False)

    summary = {
        **optimal_payload,
        "search_output": str(search_output),
        "results_csv": str(results_path),
        "num_parameter_combinations": (
            len(config.nms_iou_thresholds) * len(config.nms_score_thresholds)
        ),
        "num_raster_products": len(product_inputs),
        "num_skipped_products": len(skipped_products),
    }
    _write_json(search_output / "metrics" / "summary.json", summary)
    with open(search_output / "metrics" / "summary.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(summary, file, sort_keys=False)

    print(
        "Optimal raster NMS parameters: "
        f"IoU={optimal_payload['nms_threshold']}, "
        f"score={optimal_payload['score_threshold']}, "
        f"objective RF1={optimal_payload['objective_f1']:.6f}."
    )
    return search_output


def _run_search_tasks(
    tasks: list[dict[str, Any]],
    *,
    n_workers: int,
) -> list[dict[str, Any]]:
    if n_workers == 1:
        return [_evaluate_nms_task(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_evaluate_nms_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    "NMS search task failed for "
                    f"{task['dataset_name']}/{task['product_name']} at "
                    f"IoU={task['nms_iou_threshold']}, "
                    f"score={task['nms_score_threshold']}."
                ) from exc
            completed += 1
            if completed == len(tasks) or completed % max(1, len(tasks) // 20) == 0:
                print(f"NMS search progress: {completed}/{len(tasks)} tasks")
    return results


def _evaluate_nms_task(task: dict[str, Any]) -> dict[str, Any]:
    parameter_tag = (
        f"nmsiou_{_float_tag(task['nms_iou_threshold'])}"
        f"_score_{_float_tag(task['nms_score_threshold'])}"
    )
    output_path = (
        Path(task["output_root"])
        / task["dataset_name"]
        / task["location"]
        / task["product_name"]
        / parameter_tag
        / f"{task['product_name']}_aggregated.gpkg"
    )
    aggregate_gpkg_with_nms(
        preaggregated_gpkg_path=task["preaggregated_gpkg_path"],
        output_path=output_path,
        score_threshold=float(task["nms_score_threshold"]),
        nms_threshold=float(task["nms_iou_threshold"]),
    )
    metrics = raster_level_bbox_metrics(
        preds_gpkg_path=output_path,
        truth_gpkg_path=task["truth_gpkg_path"],
        aoi_gpkg_path=task["aoi_gpkg_path"],
        ground_resolution=float(task["ground_resolution"]),
        iou_thresholds=[float(value) for value in task["report_iou_thresholds"]],
    )
    metrics.update(
        {
            "dataset_name": task["dataset_name"],
            "location": task["location"],
            "product_name": task["product_name"],
            "nms_iou_threshold": float(task["nms_iou_threshold"]),
            "nms_score_threshold": float(task["nms_score_threshold"]),
            "objective_iou_thresholds": [
                float(value) for value in task["objective_iou_thresholds"]
            ],
            "objective_f1": raster_f1_for_iou_thresholds(
                metrics,
                [float(value) for value in task["objective_iou_thresholds"]],
            ),
        }
    )
    return metrics


def _average_search_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    grouped_by_dataset: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["nms_iou_threshold"]),
            float(row["nms_score_threshold"]),
        )
        grouped.setdefault(key, []).append(row)
        grouped_by_dataset.setdefault((*key, str(row["dataset_name"])), []).append(row)

    averaged: list[dict[str, Any]] = []
    for (nms_iou, nms_score), group in grouped.items():
        average = weighted_average_raster_metrics(group)
        average.update(
            {
                "dataset_name": "all_datasets",
                "location": "all_datasets",
                "product_name": "average_over_rasters",
                "nms_iou_threshold": nms_iou,
                "nms_score_threshold": nms_score,
            }
        )
        averaged.append(average)
    for (nms_iou, nms_score, dataset_name), group in grouped_by_dataset.items():
        average = weighted_average_raster_metrics(group)
        average.update(
            {
                "dataset_name": dataset_name,
                "location": dataset_name,
                "product_name": "average_over_rasters",
                "nms_iou_threshold": nms_iou,
                "nms_score_threshold": nms_score,
            }
        )
        averaged.append(average)
    return averaged


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
                    for key, value in row.items()
                }
            )
    return path


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    return path


def _float_tag(value: float) -> str:
    return f"{float(value):.4g}".replace(".", "p")


def _raster_input_issue(product) -> str | None:
    if product.truth_gpkg_path is None or product.aoi_gpkg_path is None:
        return "missing_registered_truth_or_aoi_gpkg"
    if not product.truth_gpkg_path.exists():
        return "missing_truth_gpkg_file"
    if not product.aoi_gpkg_path.exists():
        return "missing_aoi_gpkg_file"
    return None
