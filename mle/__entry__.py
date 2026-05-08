from argparse import ArgumentParser
from json import load
from yaml import safe_load

from mg_flare.interfaces import preprocess, train, evaluate
from mg_flare.vars import erbium_config, slurm_config


def __entry__() -> None:
    parser = ArgumentParser(prog="mg-flare", description="MedGemma Baselines",
                            epilog="GitHub: https://github.com/ATATC/MedGemma-FLARE")
    parser.add_argument("-c", "--config", choices=["slurm", "erbium"], default="erbium", help="Configuration to use")
    parser.add_argument("--suser", help="SLURM username")
    parser.add_argument("--custom_args", default=None, help="Custom arguments to pass to the engine")
    subparsers = parser.add_subparsers(dest="system", required=True)
    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess the dataset")
    preprocess_parser.add_argument("--assistant_content_style", choices=["string", "list"], default="string")
    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("tasks", nargs="+", help="Tasks to evaluate, e.g. `classification`")
    args = parser.parse_args()
    if args.slurm:
        return
    config = erbium_config() if args.config == "erbium" else slurm_config(args.suser)
    config.initialize()
    custom_args = {}
    if args.custom_args:
        with open(args.custom_args) as f:
            if args.custom_args.endswith(".json"):
                custom_args.update(load(f))
            elif args.custom_args.endswith(".yaml"):
                custom_args.update(safe_load(f))
            else:
                raise ValueError(f"Unsupported custom arguments file type: {args.custom_args}, expected JSON or YAML")
    match args.system:
        case "preprocess":
            preprocess(config, **custom_args)
        case "train":
            train(config, args.num_epochs, args.batch_size, args.learning_rate, **custom_args)
        case "evaluate":
            evaluate(config, **custom_args)
