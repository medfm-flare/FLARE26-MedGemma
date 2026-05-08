from typing import Any

import torch
from erbium.api import get_all_gpu_info
from rich.console import Console
from rich.table import Table

from mle.engine import check_dataset, check_preprocessed_dataset
from mle.vars import ExpConfig


def check_environment(config: ExpConfig) -> dict[str, Any]:
    gpus = get_all_gpu_info()
    return {
        "dataset": check_dataset(config), "preprocessed_dataset": check_preprocessed_dataset(config), "gpus": gpus,
        "cuda": torch.version.cuda
    }


def print_environment_check_results(results: dict[str, Any], *, console: Console = Console()) -> None:
    table = Table(title="Available GPUs")
    table.add_column("Name (ID)", justify="left")
    table.add_column("Total Memory (GB)", justify="center", style="cyan")
    table.add_column("Utilization (%)", justify="center", style="magenta")
    table.add_column("Memory Utilization (%)", justify="center", style="green")
    for info in results["gpus"].values():
        table.add_row(
            f"{info.name} ({info.device_id})", info.total_memory_gb, info.utilization_percent,
            info.memory_utilization_percent
        )
    console.print(table)
    console.print(f"Dataset availability: {results["dataset"]}")
    console.print(f"CUDA version: {results["cuda"]}")


def check_satisfied_or_throw(results: dict[str, Any], dataset: bool, preprocessed_dataset: bool, cuda: bool) -> None:
    if dataset and not results["dataset"].startswith("OK"):
        raise RuntimeError(f"Dataset check failed: {results["dataset"]}")
    if preprocessed_dataset and not results["preprocessed_dataset"].startswith("OK"):
        raise RuntimeError(f"Preprocessed dataset check failed: {results["preprocessed_dataset"]}")
    if cuda and not results["cuda"]:
        raise RuntimeError("CUDA not available")
