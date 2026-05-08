from mle.vars import ExpConfig


def check_dataset(config: ExpConfig) -> str:
    """
    This function checks the availability of the dataset.
    :param config: experiment configuration
    :return: a string indicating the availability of the dataset: if available, it must start with "OK" followed by optional details; otherwise not available
    """
    ...


def check_preprocessed_dataset(config: ExpConfig) -> str:
    """
    This function checks the availability of the preprocessed dataset.
    :param config: experiment configuration
    :return: a string indicating the availability of the preprocessed dataset: see `check_dataset` for requirements
    """
    ...
