import pytest

from synthforest.huggingface_upload import (
    LFS_ATTRIBUTES,
    DatasetUploadTarget,
    compact_lfs_gitattributes,
    normalize_repo_path,
    repair_lfs_gitattributes,
    resolve_upload_targets,
    summarize_source,
    upload_target,
)


def test_resolve_default_upload_targets(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHFOREST_OUTPUT_ROOT", raising=False)
    for local_path in (
        "outputs/synthforest/crown_delineation/scenes",
        "outputs/synthforest/crown_delineation/tiles",
        "outputs/synthforest/canopy_height/scenes",
        "outputs/synthforest/canopy_height/tiles",
    ):
        folder = tmp_path / local_path
        folder.mkdir(parents=True)
        (folder / "data.bin").write_bytes(b"data")

    targets = resolve_upload_targets(project_root=tmp_path)

    assert [target.path_in_repo for target in targets] == [
        "crown_delineation/scenes",
        "crown_delineation/tiles",
        "canopy_height/scenes",
        "canopy_height/tiles",
    ]
    assert sum(target.file_count for target in targets) == 4


def test_custom_source_mapping_and_summary(tmp_path):
    folder = tmp_path / "custom"
    folder.mkdir()
    (folder / "one.bin").write_bytes(b"123")
    (folder / "two.bin").write_bytes(b"12345")

    targets = resolve_upload_targets(
        project_root=tmp_path,
        source_specs=["custom=data/custom"],
    )

    assert targets == [
        DatasetUploadTarget(
            local_path=folder.resolve(),
            path_in_repo="data/custom",
            file_count=2,
            size_bytes=8,
            largest_file_bytes=5,
        )
    ]


def test_overlapping_remote_paths_are_rejected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "data.bin").write_bytes(b"first")
    (second / "data.bin").write_bytes(b"second")

    with pytest.raises(ValueError, match="must not overlap"):
        resolve_upload_targets(
            project_root=tmp_path,
            source_specs=["first=data", "second=data/nested"],
        )


def test_symlinks_are_rejected(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "data.bin").write_bytes(b"data")
    (folder / "data-link.bin").symlink_to("data.bin")

    with pytest.raises(ValueError, match="symbolic links"):
        summarize_source(folder)


def test_local_gitattributes_are_excluded_from_summary(tmp_path):
    folder = tmp_path / "source"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "data.bin").write_bytes(b"data")
    (folder / ".gitattributes").write_text("root rule")
    (nested / ".gitattributes").write_text("nested rule")

    assert summarize_source(folder) == (1, 4, 4)


@pytest.mark.parametrize("value", ["", ".", "/", "../data", "data/../other"])
def test_invalid_repo_paths_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_repo_path(value)


def test_upload_does_not_request_remote_deletions(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    target = DatasetUploadTarget(
        local_path=folder,
        path_in_repo="scenes_detector",
        file_count=1,
        size_bytes=4,
        largest_file_bytes=4,
    )
    api = _FakeApi()

    result = upload_target(
        api,
        target,
        repo_id="forest-team/synthetic-data",
        revision="main",
    )

    assert result == "commit"
    assert api.kwargs == {
        "repo_id": "forest-team/synthetic-data",
        "repo_type": "dataset",
        "folder_path": str(folder),
        "path_in_repo": "scenes_detector",
        "revision": "main",
        "commit_message": "Upload scenes_detector",
        "ignore_patterns": [".gitattributes", "*/.gitattributes"],
    }
    assert "delete_patterns" not in api.kwargs


def test_compact_lfs_gitattributes_replaces_individual_tiff_rules():
    rule = LFS_ATTRIBUTES
    existing = (
        f"*.bin {rule}\n"
        f"data/first.tif {rule}\n"
        f"data/nested/second.tif {rule}\n"
        f"keep/special.dat {rule}\n"
    )

    compacted, removed = compact_lfs_gitattributes(existing)

    assert removed == 2
    assert compacted == (f"*.bin {rule}\nkeep/special.dat {rule}\n*.tif {rule}\n")
    assert compact_lfs_gitattributes(compacted) == (compacted, 1)


def test_repair_lfs_gitattributes_uploads_only_compacted_metadata(tmp_path):
    attributes = tmp_path / ".gitattributes"
    attributes.write_text(
        f"*.bin {LFS_ATTRIBUTES}\ntiles/a.tif {LFS_ATTRIBUTES}\n",
        encoding="utf-8",
    )
    api = _FakeAttributesApi(attributes)

    repair = repair_lfs_gitattributes(
        api,
        repo_id="forest-team/synthetic-data",
        revision="main",
    )

    assert repair.changed
    assert repair.removed_rules == 1
    assert api.download_kwargs == {
        "repo_id": "forest-team/synthetic-data",
        "filename": ".gitattributes",
        "repo_type": "dataset",
        "revision": "main",
    }
    assert api.upload_kwargs == {
        "repo_id": "forest-team/synthetic-data",
        "repo_type": "dataset",
        "path_or_fileobj": (f"*.bin {LFS_ATTRIBUTES}\n*.tif {LFS_ATTRIBUTES}\n").encode(),
        "path_in_repo": ".gitattributes",
        "revision": "main",
        "commit_message": "Compact TIFF LFS rules",
    }


class _FakeApi:
    def __init__(self):
        self.kwargs = None

    def upload_folder(self, **kwargs):
        self.kwargs = kwargs
        return "commit"


class _FakeAttributesApi:
    def __init__(self, attributes_path):
        self.attributes_path = attributes_path
        self.download_kwargs = None
        self.upload_kwargs = None

    def hf_hub_download(self, **kwargs):
        self.download_kwargs = kwargs
        return str(self.attributes_path)

    def upload_file(self, **kwargs):
        self.upload_kwargs = kwargs
