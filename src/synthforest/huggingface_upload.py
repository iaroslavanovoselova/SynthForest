from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

LFS_ATTRIBUTES = "filter=lfs diff=lfs merge=lfs -text"
REQUIRED_LFS_PATTERNS = ("*.tif",)
IGNORED_UPLOAD_PATTERNS = (".gitattributes", "*/.gitattributes")

# Local folders below $SYNTHFOREST_OUTPUT_ROOT and their location in the dataset
# repository. The repository root corresponds to $SYNTHFOREST_OUTPUT_ROOT/synthforest.
DEFAULT_DATA_SOURCES = (
    ("synthforest/crown_delineation/scenes", "crown_delineation/scenes"),
    ("synthforest/crown_delineation/tiles", "crown_delineation/tiles"),
    ("synthforest/canopy_height/scenes", "canopy_height/scenes"),
    ("synthforest/canopy_height/tiles", "canopy_height/tiles"),
)


@dataclass(frozen=True)
class DatasetUploadTarget:
    local_path: Path
    path_in_repo: str
    file_count: int
    size_bytes: int
    largest_file_bytes: int


def resolve_upload_targets(
    *,
    project_root: Path,
    source_specs: Iterable[str] | None = None,
) -> list[DatasetUploadTarget]:
    """Resolve and validate local-folder to Hub-folder mappings."""
    project_root = project_root.expanduser().resolve()
    if source_specs:
        raw_mappings = [_parse_source_spec(spec) for spec in source_specs]
    else:
        output_root = Path(os.environ.get("SYNTHFOREST_OUTPUT_ROOT", project_root / "outputs"))
        raw_mappings = [
            (str(output_root / local_path), remote_path)
            for local_path, remote_path in DEFAULT_DATA_SOURCES
        ]

    mappings: list[tuple[Path, str]] = []
    for raw_local_path, raw_repo_path in raw_mappings:
        local_path = Path(raw_local_path).expanduser()
        if not local_path.is_absolute():
            local_path = project_root / local_path
        local_path = local_path.resolve()
        path_in_repo = normalize_repo_path(raw_repo_path)
        mappings.append((local_path, path_in_repo))

    _validate_unique_mappings(mappings)
    targets: list[DatasetUploadTarget] = []
    for local_path, path_in_repo in mappings:
        file_count, size_bytes, largest_file_bytes = summarize_source(local_path)
        targets.append(
            DatasetUploadTarget(
                local_path=local_path,
                path_in_repo=path_in_repo,
                file_count=file_count,
                size_bytes=size_bytes,
                largest_file_bytes=largest_file_bytes,
            )
        )
    return targets


def normalize_repo_path(value: str) -> str:
    """Validate a non-root relative POSIX path in a Hub repository."""
    cleaned = str(value).strip().replace("\\", "/").strip("/")
    path = PurePosixPath(cleaned)
    if not cleaned or cleaned == ".":
        raise ValueError("Hugging Face path_in_repo must not be empty or the repo root.")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid Hugging Face path_in_repo: {value!r}")
    return path.as_posix()


def summarize_source(folder: Path) -> tuple[int, int, int]:
    """Validate a source tree and return file count, size, and largest file size."""
    if not folder.is_dir():
        raise ValueError(f"Dataset source does not exist or is not a directory: {folder}")

    file_count = 0
    size_bytes = 0
    largest_file_bytes = 0
    symlinks: list[str] = []
    special_files: list[str] = []
    git_directories: list[str] = []

    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        current = Path(dirpath)
        for dirname in list(dirnames):
            path = current / dirname
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
                dirnames.remove(dirname)
            elif dirname == ".git":
                git_directories.append(relative)
                dirnames.remove(dirname)

        for filename in filenames:
            path = current / filename
            relative = path.relative_to(folder).as_posix()
            if filename == ".gitattributes":
                continue
            if path.is_symlink():
                symlinks.append(relative)
                continue
            if not path.is_file():
                special_files.append(relative)
                continue
            stat = path.stat()
            file_count += 1
            size_bytes += stat.st_size
            largest_file_bytes = max(largest_file_bytes, stat.st_size)

    errors: list[str] = []
    if symlinks:
        errors.append(_format_path_error("symbolic links", symlinks))
    if special_files:
        errors.append(_format_path_error("special files", special_files))
    if git_directories:
        errors.append(
            _format_path_error(
                ".git directories (ignored by Hugging Face uploads)",
                git_directories,
            )
        )
    if errors:
        raise ValueError(f"Unsafe dataset source {folder}:\n" + "\n".join(errors))
    if file_count == 0:
        raise ValueError(f"Dataset source contains no regular files: {folder}")
    return file_count, size_bytes, largest_file_bytes


def upload_target(
    api: Any,
    target: DatasetUploadTarget,
    *,
    repo_id: str,
    revision: str,
) -> Any:
    """Upload one validated folder without deleting any existing Hub files."""
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(target.local_path),
        path_in_repo=target.path_in_repo,
        revision=revision,
        commit_message=f"Upload {target.path_in_repo}",
        ignore_patterns=list(IGNORED_UPLOAD_PATTERNS),
    )


@dataclass(frozen=True)
class GitattributesRepair:
    changed: bool
    old_size_bytes: int
    new_size_bytes: int
    removed_rules: int


def repair_lfs_gitattributes(
    api: Any,
    *,
    repo_id: str,
    revision: str,
) -> GitattributesRepair:
    """Compact redundant TIFF LFS rules in a dataset repository.

    The Hub can add one LFS rule per uploaded file when an extension is not
    covered by a wildcard. For datasets containing many TIFFs, the repository's
    own `.gitattributes` can then exceed the Hub's 10 MiB regular-file limit.
    """
    attributes_path = Path(
        api.hf_hub_download(
            repo_id=repo_id,
            filename=".gitattributes",
            repo_type="dataset",
            revision=revision,
        )
    )
    existing = attributes_path.read_text(encoding="utf-8")
    compacted, removed_rules = compact_lfs_gitattributes(existing)
    repair = GitattributesRepair(
        changed=compacted != existing,
        old_size_bytes=len(existing.encode("utf-8")),
        new_size_bytes=len(compacted.encode("utf-8")),
        removed_rules=removed_rules,
    )
    if repair.changed:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=compacted.encode("utf-8"),
            path_in_repo=".gitattributes",
            revision=revision,
            commit_message="Compact TIFF LFS rules",
        )
    return repair


def compact_lfs_gitattributes(
    content: str,
    *,
    required_patterns: Iterable[str] = REQUIRED_LFS_PATTERNS,
) -> tuple[str, int]:
    """Replace path-specific LFS rules with compact extension-wide rules."""
    patterns = tuple(dict.fromkeys(required_patterns))
    required_suffixes = {
        pattern[1:] for pattern in patterns if pattern.startswith("*.") and "/" not in pattern
    }
    rule_suffix = f" {LFS_ATTRIBUTES}"
    kept_lines: list[str] = []
    removed_rules = 0

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.endswith(rule_suffix):
            kept_lines.append(line)
            continue

        pattern = stripped[: -len(rule_suffix)]
        is_required_pattern = pattern in patterns
        is_concrete_required_path = (
            not any(character in pattern for character in "*?[")
            and PurePosixPath(pattern).suffix in required_suffixes
        )
        if is_required_pattern or is_concrete_required_path:
            removed_rules += 1
            continue
        kept_lines.append(line)

    while kept_lines and not kept_lines[-1]:
        kept_lines.pop()
    kept_lines.extend(f"{pattern} {LFS_ATTRIBUTES}" for pattern in patterns)
    return "\n".join(kept_lines) + "\n", removed_rules


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _parse_source_spec(value: str) -> tuple[str, str]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Dataset --source must not be empty.")
    if "=" in raw:
        local_path, path_in_repo = raw.split("=", 1)
    else:
        local_path = raw
        path_in_repo = Path(raw).name
    if not local_path.strip():
        raise ValueError(f"Dataset --source has an empty local path: {value!r}")
    return local_path.strip(), path_in_repo.strip()


def _validate_unique_mappings(mappings: list[tuple[Path, str]]) -> None:
    local_paths = [local_path for local_path, _ in mappings]
    if len(set(local_paths)) != len(local_paths):
        raise ValueError("The same local dataset source was specified more than once.")

    repo_paths = [PurePosixPath(repo_path) for _, repo_path in mappings]
    for index, path in enumerate(repo_paths):
        for other in repo_paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(
                    "Hugging Face destination paths must not overlap: "
                    f"{path.as_posix()!r} and {other.as_posix()!r}"
                )


def _format_path_error(label: str, paths: list[str]) -> str:
    displayed = ", ".join(repr(path) for path in paths[:5])
    suffix = f", and {len(paths) - 5} more" if len(paths) > 5 else ""
    return f"found {len(paths)} {label}: {displayed}{suffix}"
