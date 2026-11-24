from __future__ import annotations

from typing import Any


def dataloader_worker_kwargs(num_workers: int, *, avoid_fork: bool) -> dict[str, Any]:
    worker_count = int(num_workers)
    kwargs: dict[str, Any] = {"num_workers": worker_count}
    if worker_count > 0 and avoid_fork:
        # wandb starts background threads/processes; forking dataloader workers after
        # that can deadlock before the first batch is produced.
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    return kwargs
