from rich.console import Console

from mle.vars import ExpConfig


# these constants MUST NOT be removed but may be modified
DEFAULT_NUM_EPOCHS: int = 1
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_LEARNING_RATE: float = 1e-4


def train(config: ExpConfig, num_epochs: int, batch_size: int, learning_rate: float, use_wandb: bool, smoke_test: bool,
          *, console: Console = Console(), **kwargs) -> None:
    """
    This is a template entrypoint for training. You MUST NOT change its signature, but you may add functions and classes
    to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    :param learning_rate: the learning rate for training
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    ...