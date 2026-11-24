"""Consistency checks between the released configurations and the paper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from synthforest.cli import COMMANDS, main
from synthforest.commands.blender_launcher import botaniq_problem
from synthforest.config_io import load_yaml
from synthforest.paths import PathConfigurationError, config_path, expand_environment
from synthforest.training.benchmarking.config import BenchmarkConfig, NmsSearchConfig
from synthforest.training.config import (
    DetectorTrainingConfig,
    HeightRasterBenchmarkConfig,
    HeightTrainingConfig,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_cli_exposes_the_paper_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    assert {
        ("generate", "asset-pool"),
        ("generate", "textures"),
        ("generate", "scenes"),
        ("prepare", "downsample"),
        ("prepare", "tiles"),
        ("prepare", "split"),
        ("prepare", "height-maps"),
        ("train", "height"),
        ("train", "crowns"),
        ("finetune", "crowns"),
        ("evaluate", "height"),
        ("evaluate", "crowns"),
    } <= set(COMMANDS)
    assert main(["--help"]) == 0
    assert "train crowns" in capsys.readouterr().out


def test_config_identifiers_resolve_from_any_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert config_path("height/train_syn20") == CONFIGS / "height" / "train_syn20.yaml"
    assert config_path("crowns/training/synthetic_hf").is_file()


def test_unresolved_environment_variable_is_actionable() -> None:
    with pytest.raises(PathConfigurationError, match="MISSING_SYNTHFOREST_ROOT"):
        expand_environment("${MISSING_SYNTHFOREST_ROOT}/data", source="test")


def test_training_configs_pass_strict_schemas() -> None:
    for path in sorted((CONFIGS / "crowns").glob("training*/*.yaml")):
        DetectorTrainingConfig.from_yaml(path)
    for path in sorted((CONFIGS / "crowns" / "finetuning").glob("*.yaml")):
        DetectorTrainingConfig.from_yaml(path)
    for path in sorted((CONFIGS / "height").glob("train_*.yaml")):
        HeightTrainingConfig.from_yaml(path)


def test_benchmark_configs_pass_strict_schemas() -> None:
    benchmark = BenchmarkConfig.from_yaml(
        CONFIGS / "crowns/benchmark/maskrcnn.yaml", validate=False
    )
    benchmark.run_folder = "/portable/run"
    benchmark.validate()
    nms = NmsSearchConfig.from_yaml(CONFIGS / "crowns/nms/maskrcnn.yaml", validate=False)
    nms.run_folder = "/portable/run"
    nms.validate()
    for path in sorted((CONFIGS / "height" / "benchmark").glob("*.yaml")):
        config = HeightRasterBenchmarkConfig.from_yaml(path, validate=False)
        config.run_folder = "/portable/run"
        config.validate()


def test_generation_configs_resolve_with_hydra_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYNTHFOREST_OUTPUT_ROOT")
    for path in sorted((CONFIGS / "generation").glob("*.yaml")):
        config = OmegaConf.load(path)
        config.pop("hydra", None)
        resolved = OmegaConf.to_container(config, resolve=True)
        assert isinstance(resolved, dict) and resolved, path


@pytest.mark.parametrize(
    ("relative_path", "iterations", "learning_rate"),
    [
        ("height/train_syn20.yaml", 15_000, 5.95e-5),
        ("crowns/training/synthetic.yaml", 8_000, 1.4e-4),
        ("crowns/training/synthetic_hf.yaml", 8_000, 1.0e-4),
        ("crowns/training/synthetic_mt.yaml", 8_000, 5.0e-5),
        ("crowns/training/synthetic_hf_mt.yaml", 8_000, 5.0e-5),
        ("crowns/training/real_bamforest.yaml", 6_500, 1.0e-4),
        ("crowns/finetuning/real_bamforest.yaml", 6_500, 1.0e-4),
    ],
)
def test_paper_schedules(relative_path: str, iterations: int, learning_rate: float) -> None:
    _, config = load_yaml(CONFIGS / relative_path)
    stage = config["stages"][0]
    assert stage["max_iterations"] == iterations
    key = "decoder_lr" if relative_path.startswith("height/") else "detector_lr"
    assert stage["optimizer"][key] == pytest.approx(learning_rate)
    assert config["batch_size"] * config["grad_accumulation_steps"] == 4


def test_paper_transfer_settings() -> None:
    hf = DetectorTrainingConfig.from_yaml(CONFIGS / "crowns/training/synthetic_hf_mt.yaml")
    assert hf.height_fusion.levels == ["p8", "p16"]
    assert (hf.height_fusion.clip_min_m, hf.height_fusion.clip_max_m) == (0.0, 60.0)
    assert hf.height_fusion.std_m == 30.0
    assert hf.height_fusion.target_gsd_m == pytest.approx(0.20)
    assert hf.semi_supervised.ema_decay == pytest.approx(0.999)
    assert hf.semi_supervised.pseudo_labels.score_threshold == pytest.approx(0.55)
    assert hf.semi_supervised.unlabeled_real.dataset_names == ["SelvaMask", "BAMForest"]

    syn20 = yaml.safe_load((CONFIGS / "height/train_syn20.yaml").read_text(encoding="utf-8"))
    assert syn20["domain_adaptation"]["methods"] == ["HM", "PDA", "FDA"]
    assert [Path(path).name for path in syn20["domain_adaptation"]["reference_root_paths"]] == [
        "france",
        "neon",
        "sao_paulo",
        "virtual_shizuoka",
    ]


def test_experiment_chain_uses_final_checkpoints() -> None:
    for path in sorted(CONFIGS.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "best.pth" not in text, path
    for path in sorted((CONFIGS / "crowns" / "finetuning").glob("*.yaml")):
        config = DetectorTrainingConfig.from_yaml(path)
        assert config.domain_adaptation.enabled
        assert config.domain_adaptation.methods == ["HM", "PDA"]
        assert config.checkpoint_initialization.checkpoint_path.endswith("maskrcnn_syn_hf_mt.pth")


def test_scene_generation_reports_missing_botaniq(tmp_path: Path) -> None:
    message = botaniq_problem("create_tree_scatter_scene", [])
    assert message is not None and "Botaniq" in message
    message = botaniq_problem("create_botaniq_asset_pool", [])
    assert message is not None and "botaniq_full" in message

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text("{}", encoding="utf-8")
    assert botaniq_problem("create_tree_scatter_scene", [f"scatter.asset_pool_dir={pool}"]) is None
    assert botaniq_problem("convert_scenes_to_tiles", []) is None
