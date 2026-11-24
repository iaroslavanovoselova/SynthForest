#!/usr/bin/env python3
"""Download only missing CanopyRS raster-evaluation GeoPackages.

The Hugging Face lookup follows CanopyRS's preprocessed-dataset implementation,
but copies each requested file into this repository's local product layout.
Tiles, COCO annotations, imagery, and unselected datasets are not downloaded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from synthforest.training.benchmarking.config import NmsSearchConfig
from synthforest.training.config import DEFAULT_REAL_VALIDATION_ROOT
from synthforest.training.real_datasets import get_dataset_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only missing truth and AOI GPKGs required for CanopyRS-style "
            "raster evaluation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional NMS-search config supplying datasets, fold, and data root. "
            "Explicit CLI values override it."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--fold", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hf-root", default="CanopyRS")
    parser.add_argument("--revision", default="gpkg")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing and unregistered files without downloading anything.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help=(
            "Use the locally configured Hugging Face token. CanopyRS datasets are "
            "public, so downloads are anonymous by default to avoid stale-token errors."
        ),
    )
    return parser.parse_args()


def resolve_request(
    args: argparse.Namespace,
) -> tuple[list[str], str, Path]:
    config = None
    if args.config is not None:
        config = NmsSearchConfig.from_yaml(args.config, validate=False)

    datasets = args.datasets
    if datasets is None and config is not None:
        datasets = list(config.datasets)
    if not datasets:
        raise ValueError("Specify --datasets or provide an NMS-search --config.")

    fold = args.fold or (config.fold_name if config is not None else "valid")
    data_root = args.data_root
    if data_root is None:
        data_root = Path(
            config.real_data_root if config is not None else DEFAULT_REAL_VALIDATION_ROOT
        )
    return list(datasets), str(fold), Path(data_root)


def download_missing_gpkgs(
    *,
    datasets: list[str],
    fold: str,
    data_root: Path,
    hf_root: str,
    revision: str,
    dry_run: bool,
    local_files_only: bool,
    use_auth_token: bool,
) -> int:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    downloaded = 0
    existing = 0
    missing = 0
    unregistered_products = 0
    errors: list[str] = []

    for dataset_name in datasets:
        dataset = get_dataset_spec(dataset_name)
        fold_inputs = dataset.raster_level_eval_inputs.get(fold, {})
        if not fold_inputs:
            print(f"[NO RASTER INPUTS] {dataset_name}: fold={fold}")
            continue

        product_locations = {
            product_name: location
            for location, product_names in dataset.products_per_location.items()
            for product_name in product_names
        }
        hf_dataset_name = dataset.hf_dataset_name or dataset.dataset_name
        repo_id = f"{hf_root}/{hf_dataset_name}"
        for product_name, raster_inputs in fold_inputs.items():
            truth_name = raster_inputs.get("ground_truth_gpkg")
            aoi_name = raster_inputs.get("aoi_gpkg")
            if not truth_name or not aoi_name:
                unregistered_products += 1
                print(
                    f"[UNREGISTERED] {dataset_name}/{product_name}: "
                    f"no truth/AOI GPKG filenames (HF repo: {repo_id})"
                )
                continue
            try:
                location = product_locations[product_name]
            except KeyError:
                errors.append(f"{dataset_name}/{product_name}: product has no registered location")
                continue

            product_root = data_root / location / product_name
            for role, filename in (
                ("truth", str(truth_name)),
                ("aoi", str(aoi_name)),
            ):
                target = product_root / filename
                if target.exists():
                    existing += 1
                    print(f"[EXISTS] {dataset_name}/{product_name} {role}: {target}")
                    continue

                missing += 1
                print(
                    f"[MISSING] {dataset_name}/{product_name} {role}: "
                    f"{repo_id}@{revision}/{filename} -> {target}"
                )
                if dry_run:
                    continue

                try:
                    source = _download_hf_file(
                        repo_id=repo_id,
                        revision=revision,
                        filename=filename,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                    )
                    _copy_atomically(source, target)
                except Exception as exc:
                    errors.append(f"{dataset_name}/{product_name} {role} ({filename}): {exc}")
                    continue
                downloaded += 1
                print(f"[DOWNLOADED] {target}")

    print(
        "Summary: "
        f"downloaded={downloaded}, existing={existing}, missing={missing}, "
        f"unregistered_products={unregistered_products}, errors={len(errors)}"
    )
    if errors:
        print("Errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


def _download_hf_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    local_files_only: bool,
    use_auth_token: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required. Install the project requirements first."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_files_only=local_files_only,
            token=True if use_auth_token else False,
        )
    )


def _copy_atomically(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    datasets, fold, data_root = resolve_request(args)
    print(f"Datasets : {', '.join(datasets)}")
    print(f"Fold     : {fold}")
    print(f"Data root: {data_root}")
    print(f"HF source: {args.hf_root}/<dataset>@{args.revision}")
    if args.dry_run:
        print("Mode     : dry run")
    exit_code = download_missing_gpkgs(
        datasets=datasets,
        fold=fold,
        data_root=data_root,
        hf_root=args.hf_root,
        revision=args.revision,
        dry_run=args.dry_run,
        local_files_only=args.local_files_only,
        use_auth_token=args.use_auth_token,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
