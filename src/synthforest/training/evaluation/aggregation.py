from __future__ import annotations

import math


def macro_average_metrics(
    metrics_by_dataset: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not metrics_by_dataset:
        raise ValueError("Real validation requires at least one evaluated dataset.")
    count_keys = {
        "num_images",
        "num_predictions",
        "num_predictions_before_mask_filter",
        "num_predictions_filtered_invalid",
        "num_targets",
        "segm_num_images",
        "segm_num_truths",
        "segm_num_preds",
    }
    metric_keys = sorted(set().union(*(metrics.keys() for metrics in metrics_by_dataset.values())))
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        values = [
            float(metrics[key])
            for metrics in metrics_by_dataset.values()
            if key in metrics
            and isinstance(metrics[key], (int, float))
            and math.isfinite(float(metrics[key]))
            and (key in count_keys or float(metrics[key]) >= 0.0)
        ]
        if not values:
            continue
        aggregate[key] = (
            float(sum(values)) if key in count_keys else float(sum(values) / len(values))
        )
    return aggregate
