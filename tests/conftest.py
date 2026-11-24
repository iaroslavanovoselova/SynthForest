from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configured_external_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of workstation paths and shell configuration."""

    roots = {
        "SYNTHFOREST_DATA_ROOT": tmp_path / "data",
        "SYNTHFOREST_MODEL_ROOT": tmp_path / "models",
        "SYNTHFOREST_ASSET_ROOT": tmp_path / "assets",
        "SYNTHFOREST_OUTPUT_ROOT": tmp_path / "outputs",
    }
    for name, path in roots.items():
        path.mkdir()
        monkeypatch.setenv(name, str(path))

    source_root = Path(__file__).resolve().parents[1] / "src"
    inherited = os.environ.get("PYTHONPATH")
    pythonpath = str(source_root) if not inherited else f"{source_root}{os.pathsep}{inherited}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
