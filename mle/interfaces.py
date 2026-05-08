from mg_flare.vars import ExpConfig
from mg_flare.sanity import check_environment, print_environment_check_results, check_satisfied_or_throw
from mg_flare.engine import preprocess as _preprocess, train as _train, evaluate as _evaluate


def preprocess(config: ExpConfig, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, True, False, False)
    _preprocess(config, **kwargs)


def train(config: ExpConfig, num_epochs: int, batch_size: int, learning_rate: float, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, False, True, True)
    _train(config, num_epochs, batch_size, learning_rate, **kwargs)


def evaluate(config: ExpConfig, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, False, True, True)
    _evaluate(config, **kwargs)
