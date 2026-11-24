#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from synthforest.huggingface_upload import (
    DatasetUploadTarget,
    format_bytes,
    repair_lfs_gitattributes,
    resolve_upload_targets,
    upload_target,
)
from synthforest.paths import project_root

PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the synthetic scene and tile datasets to one Hugging Face "
            "dataset repository. The command is a local dry run unless --execute "
            "is explicitly supplied."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repository as NAMESPACE/NAME.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Existing repository branch to upload to. Default: main.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="LOCAL_PATH[=PATH_IN_REPO]",
        help=(
            "Override the four default sources. Repeat for multiple folders. "
            "Relative local paths are resolved from the project root."
        ),
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repository if it does not already exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make a newly created repository private. Requires --create-repo.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Contact Hugging Face and upload files. Without this flag, no "
            "Hugging Face API is imported or contacted."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.private and not args.create_repo:
        print("Error: --private requires --create-repo.", file=sys.stderr)
        return 2
    if args.create_repo and args.revision != "main":
        print(
            "Error: a newly created repository only has the main branch. "
            "Create other revisions before using this uploader.",
            file=sys.stderr,
        )
        return 2

    try:
        targets = resolve_upload_targets(
            project_root=PROJECT_ROOT,
            source_specs=args.source,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    mode = "UPLOAD" if args.execute else "DRY RUN"
    print(f"{mode}: {len(targets)} folder(s) -> hf://datasets/{args.repo_id}@{args.revision}")
    for target in targets:
        print_target(target)
    total_files = sum(target.file_count for target in targets)
    total_bytes = sum(target.size_bytes for target in targets)
    print(f"\nTotal: {total_files:,} files, {format_bytes(total_bytes)}")
    print("Remote files absent locally will not be deleted.")

    if not args.execute:
        print(
            "\nNo Hugging Face API calls or uploads were made. Re-run with "
            "--execute after checking this mapping."
        )
        return 0

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        print(
            "Error: huggingface-hub is required for --execute. "
            "Install the project requirements first.",
            file=sys.stderr,
        )
        return 2

    api = HfApi()
    try:
        account = api.whoami()
        account_name = account.get("name") or account.get("fullname") or "<unknown>"
        print(f"\nAuthenticated to Hugging Face as {account_name}.")
        try:
            api.repo_info(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
            )
            print(f"Verified existing dataset repository {args.repo_id}.")
        except RepositoryNotFoundError:
            if not args.create_repo:
                raise RuntimeError(
                    f"Dataset repository {args.repo_id!r} does not exist or is "
                    "not accessible. Create it first or pass --create-repo."
                )
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="dataset",
                private=args.private,
                exist_ok=True,
            )
            print(f"Created dataset repository {args.repo_id}.")
    except Exception as exc:
        print(
            f"Error: Hugging Face preflight failed; no files were uploaded: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        repair = repair_lfs_gitattributes(
            api,
            repo_id=args.repo_id,
            revision=args.revision,
        )
    except Exception as exc:
        print(
            "Error: could not prepare the repository's .gitattributes; "
            f"no dataset folders were uploaded by this invocation: {exc}",
            file=sys.stderr,
        )
        return 1

    if repair.changed:
        print(
            "Compacted repository .gitattributes from "
            f"{format_bytes(repair.old_size_bytes)} to "
            f"{format_bytes(repair.new_size_bytes)} "
            f"({repair.removed_rules:,} redundant rules removed)."
        )
    else:
        print("Repository .gitattributes already has a compact *.tif LFS rule.")

    for index, target in enumerate(targets, start=1):
        print(
            f"\n[{index}/{len(targets)}] Uploading {target.local_path} to {target.path_in_repo}/..."
        )
        try:
            result = upload_target(
                api,
                target,
                repo_id=args.repo_id,
                revision=args.revision,
            )
        except Exception as exc:
            print(
                f"Error: upload failed at {target.path_in_repo!r}: {exc}\n"
                "Re-run the same command to resume; later folders were not attempted.",
                file=sys.stderr,
            )
            return 1
        commit_url = getattr(result, "commit_url", None)
        print(f"Uploaded {target.path_in_repo}." + (f" {commit_url}" if commit_url else ""))

    print(
        f"\nUploaded all {len(targets)} folders to https://huggingface.co/datasets/{args.repo_id}."
    )
    return 0


def print_target(target: DatasetUploadTarget) -> None:
    print(
        f"\n{target.local_path}\n"
        f"  destination: {target.path_in_repo}/\n"
        f"  data:        {target.file_count:,} files, "
        f"{format_bytes(target.size_bytes)}\n"
        f"  largest:     {format_bytes(target.largest_file_bytes)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
