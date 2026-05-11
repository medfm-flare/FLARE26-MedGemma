from typing import Sequence

from rich.console import Console
from mle.vars import ExpConfig


def infer(config: ExpConfig, tasks: Sequence[str], use_wandb: bool, smoke_test: bool, *, console: Console = Console(),
          **kwargs) -> None:
    """
    This is a template entrypoint for inference. You MUST NOT change its signature, but you may add functions and
    classes to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param tasks: the tasks to evaluate on
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    ...
