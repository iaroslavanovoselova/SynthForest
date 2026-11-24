from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingRunLayout:
    """Canonical paths for one training or sweep run."""

    root: Path

    @classmethod
    def resolve(
        cls,
        *,
        output_root: str | Path,
        model_type: str,
        run_name: str,
        sweep_phase: str | None = None,
    ) -> "TrainingRunLayout":
        parent = Path(output_root) / model_type
        if sweep_phase is not None:
            parent /= sweep_phase
        return cls(parent / run_name)

    def create(self) -> "TrainingRunLayout":
        for path in (
            self.config_dir,
            self.checkpoints_dir,
            self.stage_checkpoints_dir,
            self.branch_checkpoints_dir,
            self.metrics_dir,
            self.metadata_dir,
            self.wandb_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def resolved_config(self) -> Path:
        return self.config_dir / "resolved.yaml"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def stage_checkpoints_dir(self) -> Path:
        return self.checkpoints_dir / "stages"

    @property
    def branch_checkpoints_dir(self) -> Path:
        return self.checkpoints_dir / "branches"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best.pth"

    @property
    def best_bbox_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_bbox.pth"

    @property
    def best_segm_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_segm.pth"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last.pth"

    def stage_checkpoint(self, stage_name: str, kind: str) -> Path:
        if kind not in {"best", "last"}:
            raise ValueError("Stage checkpoint kind must be 'best' or 'last'.")
        return self.stage_checkpoints_dir / stage_name / f"{kind}.pth"

    def branch_checkpoint(self, iteration: int) -> Path:
        return self.branch_checkpoints_dir / f"iter_{int(iteration):08d}.pth"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def train_metrics(self) -> Path:
        return self.metrics_dir / "train.jsonl"

    @property
    def validation_metrics(self) -> Path:
        return self.metrics_dir / "validation.jsonl"

    @property
    def latest_validation_metrics(self) -> Path:
        return self.metrics_dir / "validation_latest.json"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def initialization_metadata(self) -> Path:
        return self.metadata_dir / "initialization.json"

    @property
    def train_sampling_metadata(self) -> Path:
        return self.metadata_dir / "train_sampling.json"

    @property
    def semi_supervised_metadata(self) -> Path:
        return self.metadata_dir / "semi_supervised.json"

    @property
    def trainability_metadata(self) -> Path:
        return self.metadata_dir / "trainability.json"

    @property
    def wandb_dir(self) -> Path:
        return self.root / "wandb"

    @property
    def benchmarks_dir(self) -> Path:
        return self.root / "benchmarks"

    def benchmark_root(self, benchmark_name: str) -> Path:
        return self.benchmarks_dir / benchmark_name

    @property
    def post_training_benchmark_metadata(self) -> Path:
        return self.metadata_dir / "post_training_benchmark.json"


@dataclass(frozen=True)
class BenchmarkRunLayout:
    """Canonical paths for one real-data benchmark."""

    root: Path

    @classmethod
    def resolve(
        cls,
        *,
        benchmark_name: str,
        output_root: str | Path | None = None,
        model_type: str | None = None,
        run_folder: str | Path | None = None,
    ) -> "BenchmarkRunLayout":
        if run_folder is not None:
            return cls(Path(run_folder) / "benchmarks" / benchmark_name)
        if output_root is None or model_type is None:
            raise ValueError(
                "BenchmarkRunLayout.resolve requires output_root and model_type "
                "when run_folder is not set."
            )
        return cls(Path(output_root) / model_type / benchmark_name)

    def create(self) -> "BenchmarkRunLayout":
        for path in (
            self.config_dir,
            self.metadata_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.wandb_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def benchmark_config(self) -> Path:
        return self.config_dir / "benchmark.yaml"

    @property
    def training_config(self) -> Path:
        return self.config_dir / "training.yaml"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def checkpoint_metadata(self) -> Path:
        return self.metadata_dir / "checkpoint.json"

    @property
    def wandb_metadata(self) -> Path:
        return self.metadata_dir / "wandb.json"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    def fold_predictions(self, fold: str) -> Path:
        return self.predictions_dir / fold

    @property
    def wandb_dir(self) -> Path:
        return self.root / "wandb"


@dataclass(frozen=True)
class EvaluationRunLayout:
    """Canonical paths for one standalone checkpoint evaluation."""

    root: Path

    @classmethod
    def resolve(
        cls,
        *,
        output_root: str | Path,
        model_type: str,
        source_run: str,
        evaluation_name: str,
    ) -> "EvaluationRunLayout":
        return cls(Path(output_root) / model_type / source_run / evaluation_name)

    def create(self) -> "EvaluationRunLayout":
        for path in (self.config_dir, self.metadata_dir, self.metrics_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def training_config(self) -> Path:
        return self.config_dir / "training.yaml"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def checkpoint_metadata(self) -> Path:
        return self.metadata_dir / "checkpoint.json"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def metrics_file(self) -> Path:
        return self.metrics_dir / "metrics.json"


def source_run_name(checkpoint_path: str | Path, configured_name: str | None) -> str:
    if configured_name:
        return configured_name
    checkpoint = Path(checkpoint_path)
    for parent in checkpoint.parents:
        if parent.name == "checkpoints":
            return parent.parent.name
    return checkpoint.parent.name or checkpoint.stem


def training_run_folder_from_checkpoint(checkpoint_path: str | Path) -> Path | None:
    checkpoint = Path(checkpoint_path)
    for parent in checkpoint.parents:
        if parent.name == "checkpoints":
            return parent.parent
    return None


def validate_checkpoint_name(checkpoint_name: str | Path) -> str:
    text = str(checkpoint_name).strip()
    if not text:
        raise ValueError("checkpoint_name must not be empty.")
    checkpoint = Path(text)
    if checkpoint == Path("."):
        raise ValueError("checkpoint_name must identify a checkpoint file.")
    if checkpoint.is_absolute():
        raise ValueError("checkpoint_name must be relative to the checkpoints folder.")
    if any(part == ".." for part in checkpoint.parts):
        raise ValueError("checkpoint_name must not contain parent-directory segments.")
    return text
