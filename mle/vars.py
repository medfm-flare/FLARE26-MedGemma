from dataclasses import dataclass
from os import makedirs
from os.path import exists

from rich.console import Console


@dataclass
class ExpConfig(object):
    experiment_name: str
    root_dir: str
    dataset_name: str = "FLARE-MLLM-2D"
    _input_dir: str | None = None
    _output_dir: str | None = None

    @property
    def input_dir(self) -> str:
        return self._input_dir or f"{self.root_dir}/input"

    @property
    def dataset_dir(self) -> str:
        return f"{self.input_dir}/{self.dataset_name}"

    @property
    def preprocessed_dataset_dir(self) -> str:
        return f"{self.output_dir}/Preprocessed-{self.dataset_name}"

    @property
    def output_dir(self) -> str:
        return self._output_dir or f"{self.root_dir}/output"

    def initialize(self, *, console: Console = Console()) -> None:
        if not exists(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist")
        if not exists(self.input_dir):
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")
        if not exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} does not exist")
        if not exists(self.output_dir):
            makedirs(self.output_dir)


def erbium_config(experiment_name: str, *, root_dir: str | None = None) -> ExpConfig:
    return ExpConfig(experiment_name, root_dir or f"/workspace")


def slurm_config(experiment_name: str, username: str, *, root_dir: str | None = None) -> ExpConfig:
    return ExpConfig(experiment_name, root_dir or f"/scratch/{username}")
