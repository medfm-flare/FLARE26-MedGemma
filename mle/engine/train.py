from __future__ import annotations

import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

from mle.vars import ExpConfig

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    from datasets import Dataset, load_from_disk
except Exception:  # pragma: no cover
    Dataset = None
    load_from_disk = None

try:
    from peft import LoraConfig, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    prepare_model_for_kbit_training = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None
    EarlyStoppingCallback = None

try:
    from trl import SFTConfig, SFTTrainer
except Exception:  # pragma: no cover
    SFTConfig = None
    SFTTrainer = None

try:
    from erbium.api import ResourceMonitor
except Exception:  # pragma: no cover
    ResourceMonitor = None


# these constants MUST NOT be removed but may be modified
DEFAULT_NUM_EPOCHS: int = 1
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_LEARNING_RATE: float = 2e-4

MODEL_ID = "google/medgemma-1.5-4b-it"

TASK_INSTRUCTIONS = {
    "disease_diagnosis_classification": (
        "You are given a medical image. Answer the classification question using only the provided image. "
        "If options are provided, return only the correct option text or class label."
    ),
    "multi_label_classification": (
        "You are given a medical image. Identify all applicable findings or labels. "
        "Return labels separated by semicolons. Return an empty string if none apply."
    ),
    "report_generation": (
        "You are given a medical image. Generate a concise radiology-style report with relevant findings and impression."
    ),
    "detection": "You are given a medical image. Return detections exactly in the coordinate format requested by the question.",
    "cell_counting": "You are given a medical image. Return only the integer count requested.",
    "regression": "You are given a medical image. Return only the requested numeric measurement.",
}


def train(
    config: ExpConfig,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_wandb: bool,
    smoke_test: bool,
    *,
    console: Console = Console(),
    **kwargs,
) -> None:
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
    missing = dependency_gaps()
    if missing:
        raise RuntimeError("Missing training dependencies: " + ", ".join(missing))
    if not torch.cuda.is_available() and not bool(kwargs.get("allow_cpu", False)):
        raise RuntimeError("CUDA is required for MedGemma fine-tuning. Use a Slurm GPU job or pass allow_cpu=true for tiny dry runs.")

    output_dir = Path(kwargs.get("model_output_dir") or kwargs.get("output_dir") or Path(config.output_dir) / f"{config.experiment_name}-medgemma15-lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(kwargs.get("seed", 42))
    seed_everything(seed)
    configure_wandb(config, output_dir, use_wandb, kwargs)

    monitor = ResourceMonitor(str(output_dir)) if ResourceMonitor is not None and bool(kwargs.get("resource_monitor", True)) else None
    if monitor is not None:
        monitor.start()

    try:
        model_name_or_path = str(kwargs.get("model_name_or_path", MODEL_ID))
        image_size = int(kwargs.get("image_size", 512 if smoke_test else 896))
        resize_mode = str(kwargs.get("resize_mode", "square"))
        max_images_per_sample = int(kwargs.get("max_images_per_sample", 1))
        max_length = int(kwargs.get("max_length", 0))
        max_train_samples = optional_int(kwargs.get("max_train_samples", 8 if smoke_test else None))
        max_eval_samples = optional_int(kwargs.get("max_eval_samples", 4 if smoke_test else 256))
        per_device_eval_batch_size = int(kwargs.get("per_device_eval_batch_size", 1))
        gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1 if smoke_test else 16))
        max_steps = int(kwargs.get("max_steps", 2 if smoke_test else -1))
        if smoke_test:
            console.print("Smoke test mode: limiting training samples, steps, and evaluation workload.")

        console.print(f"Loading converted FLARE-MLLM-2D data from {config.preprocessed_dataset_dir}")
        train_dataset, eval_dataset = load_splits(Path(config.preprocessed_dataset_dir), max_train_samples, max_eval_samples)
        console.print(f"Loaded {len(train_dataset)} training row(s)" + (f" and {len(eval_dataset)} validation row(s)" if eval_dataset else ""))

        dtype = choose_dtype()
        attn_implementations = choose_attention_backends(str(kwargs.get("attn_implementation", "auto")))
        quant_config = make_quant_config(bool(kwargs.get("load_in_4bit", True)), dtype)

        console.print(f"Loading {model_name_or_path} with attention={attn_label(attn_implementations[0])}")
        model = load_model_with_attention_fallback(
            model_name_or_path,
            attn_implementations,
            console,
            torch_dtype=dtype,
            device_map=None if torch.cuda.is_available() else "cpu",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", True))
        if gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

        target_modules: Any = kwargs.get("target_modules", "all-linear")
        if isinstance(target_modules, str) and "," in target_modules:
            target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
        modules_to_save = split_csv(kwargs.get("modules_to_save", ""))
        peft_config = LoraConfig(
            r=int(kwargs.get("lora_rank", 8 if smoke_test else 16)),
            lora_alpha=int(kwargs.get("lora_alpha", 8 if smoke_test else 16)),
            lora_dropout=float(kwargs.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save or None,
        )

        collator = ImageSFTCollator(
            processor=processor,
            image_size=image_size,
            resize_mode=resize_mode,
            max_images_per_sample=max_images_per_sample,
            max_length=max_length if max_length > 0 else None,
            mask_prompt_tokens=bool(kwargs.get("mask_prompt_tokens", True)),
        )

        eval_strategy = "steps" if eval_dataset is not None else "no"
        early_stopping_patience = int(kwargs.get("early_stopping_patience", 0))
        use_early_stopping = eval_dataset is not None and early_stopping_patience > 0
        if use_early_stopping and EarlyStoppingCallback is None:
            raise RuntimeError("EarlyStoppingCallback is unavailable; install or upgrade transformers.")

        training_args = make_sft_config_compat(
            output_dir=str(output_dir),
            run_name=str(kwargs.get("run_name", config.experiment_name)),
            num_train_epochs=float(num_epochs),
            max_steps=max_steps,
            per_device_train_batch_size=int(kwargs.get("per_device_train_batch_size", batch_size)),
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=float(learning_rate),
            weight_decay=float(kwargs.get("weight_decay", 0.0)),
            warmup_ratio=float(kwargs.get("warmup_ratio", 0.03)),
            lr_scheduler_type=str(kwargs.get("lr_scheduler_type", "linear")),
            optim=str(kwargs.get("optim", "paged_adamw_8bit" if quant_config is not None else "adamw_torch_fused")),
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            max_grad_norm=float(kwargs.get("max_grad_norm", 0.3)),
            logging_steps=int(kwargs.get("logging_steps", 1 if smoke_test else 10)),
            save_strategy="steps",
            save_steps=int(kwargs.get("save_steps", 100000 if smoke_test else 200)),
            save_total_limit=int(kwargs.get("save_total_limit", 1 if smoke_test else 3)),
            eval_strategy=eval_strategy,
            eval_steps=int(kwargs.get("eval_steps", 100000 if smoke_test else 200)),
            load_best_model_at_end=use_early_stopping,
            metric_for_best_model="eval_loss" if use_early_stopping else None,
            greater_is_better=False if use_early_stopping else None,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            label_names=["labels"],
            report_to="wandb" if use_wandb else "none",
            push_to_hub=bool(kwargs.get("push_to_hub", False)),
            hub_model_id=kwargs.get("hub_model_id"),
            dataloader_num_workers=int(kwargs.get("dataloader_num_workers", 0 if smoke_test else 0)),
            dataloader_pin_memory=bool(kwargs.get("dataloader_pin_memory", False)),
            group_by_length=False,
            packing=False,
            max_seq_length=max_length if max_length > 0 else None,
        )

        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(kwargs.get("early_stopping_threshold", 0.0)),
            ))

        trainer = make_sft_trainer_compat(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collator,
            callbacks=callbacks,
        )

        console.print("Starting MedGemma LoRA fine-tuning")
        trainer.train(resume_from_checkpoint=kwargs.get("resume_from_checkpoint"))
        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        console.print(f"Saved final adapter and processor to {final_dir}")
    finally:
        if monitor is not None:
            stop_resource_monitor(monitor, console)


def dependency_gaps() -> list[str]:
    gaps = []
    if torch is None:
        gaps.append("torch")
    if Image is None or ImageOps is None:
        gaps.append("Pillow")
    if Dataset is None or load_from_disk is None:
        gaps.append("datasets")
    if LoraConfig is None or prepare_model_for_kbit_training is None:
        gaps.append("peft")
    if AutoModelForImageTextToText is None or AutoProcessor is None or BitsAndBytesConfig is None:
        gaps.append("transformers")
    if SFTConfig is None or SFTTrainer is None:
        gaps.append("trl")
    return gaps


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out > 0 else None


def split_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_wandb(config: ExpConfig, output_dir: Path, use_wandb: bool, kwargs: dict[str, Any]) -> None:
    if use_wandb:
        os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
        os.environ.setdefault("WANDB_PROJECT", str(kwargs.get("wandb_project", "medgemma15-flare-mllm-2d")))
        os.environ.setdefault("WANDB_RUN_NAME", str(kwargs.get("wandb_run_name", config.experiment_name)))
        for env_name, key in (("WANDB_ENTITY", "wandb_entity"), ("WANDB_MODE", "wandb_mode"), ("WANDB_TAGS", "wandb_tags")):
            if kwargs.get(key):
                os.environ.setdefault(env_name, str(kwargs[key]))
        os.environ.setdefault("WANDB_LOG_MODEL", str(kwargs.get("wandb_log_model", "checkpoint")))
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")


def make_sft_config_compat(**kwargs):
    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    allowed = set(params) - {"self"}
    config = dict(kwargs)
    if "max_seq_length" in config and "max_seq_length" not in allowed and "max_length" in allowed:
        config["max_length"] = config.pop("max_seq_length")
    if "eval_strategy" in config and "eval_strategy" not in allowed and "evaluation_strategy" in allowed:
        config["evaluation_strategy"] = config.pop("eval_strategy")
    if not accepts_kwargs:
        config = {key: value for key, value in config.items() if key in allowed}
    return SFTConfig(**config)


def make_sft_trainer_compat(**kwargs):
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    allowed = set(params) - {"self"}
    config = dict(kwargs)
    if "processing_class" in config and "processing_class" not in allowed and "tokenizer" in allowed:
        config["tokenizer"] = config.pop("processing_class")
    if not accepts_kwargs:
        config = {key: value for key, value in config.items() if key in allowed}
    return SFTTrainer(**config)


def choose_dtype():
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def choose_attention_backends(requested: str) -> list[str | None]:
    requested = requested.strip().lower()
    if requested == "auto":
        return ["sdpa", "eager", None]
    if requested in {"default", "none", "transformers-default"}:
        return [None]
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return ["flash_attention_2", "sdpa", "eager", None]
        except Exception:
            return ["sdpa", "eager", None]
    return unique_attention_backends([requested, "sdpa", "eager", None])


def unique_attention_backends(backends: Sequence[str | None]) -> list[str | None]:
    out = []
    for backend in backends:
        if backend not in out:
            out.append(backend)
    return out


def attn_label(attn_implementation: str | None) -> str:
    return attn_implementation or "transformers-default"


def load_model_with_attention_fallback(model_name_or_path: str, attn_implementations: Sequence[str | None], console: Console, **kwargs):
    last_error: Exception | None = None
    for index, attn_implementation in enumerate(attn_implementations):
        model_kwargs = dict(kwargs)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            if index > 0:
                console.print(f"Retrying model load with attention={attn_label(attn_implementation)}")
            return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
        except ValueError as exc:
            last_error = exc
            message = str(exc)
            if "attn" not in message.lower() and "attention" not in message.lower():
                raise
            console.print(f"Attention backend {attn_label(attn_implementation)} is unsupported here: {message}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("No attention backends were provided for model loading.")


def make_quant_config(load_in_4bit: bool, dtype: Any):
    if not load_in_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def maybe_json_load(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_number}")
            rows.append(row)
    return rows


def arrow_safe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def arrow_safe_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    keys = sorted({key for row in rows for key in row})
    return [{key: arrow_safe_value(row.get(key)) for key in keys} for row in rows]


def find_dataset_sources(data_dir: Path, train_name: str = "train", val_name: str = "validation") -> tuple[Path, Path | None, str]:
    for base in (data_dir / "hf_dataset", data_dir):
        train_dir = base / train_name
        val_dir = base / val_name
        if train_dir.exists():
            return train_dir, val_dir if val_dir.exists() else None, "hf"
    train_jsonl = data_dir / f"{train_name}.jsonl"
    val_jsonl = data_dir / f"{val_name}.jsonl"
    if train_jsonl.exists():
        return train_jsonl, val_jsonl if val_jsonl.exists() else None, "jsonl"
    raise FileNotFoundError(f"Could not find train split under {data_dir}")


def load_splits(data_dir: Path, max_train_samples: int | None, max_eval_samples: int | None):
    train_src, val_src, kind = find_dataset_sources(data_dir)
    if kind == "hf":
        train_dataset = load_from_disk(str(train_src))
        eval_dataset = load_from_disk(str(val_src)) if val_src else None
    else:
        train_dataset = Dataset.from_list(arrow_safe_records(read_jsonl(train_src)))
        eval_dataset = Dataset.from_list(arrow_safe_records(read_jsonl(val_src))) if val_src else None
    if max_train_samples:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if eval_dataset is not None and max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    return train_dataset, eval_dataset


def stop_resource_monitor(monitor: Any, console: Console) -> None:
    for method_name in ("stop", "close", "terminate", "shutdown"):
        method = getattr(monitor, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # pragma: no cover - depends on Erbium monitor implementation
                console.print(f"Warning: failed to stop resource monitor with {method_name}(): {exc}")
            return
    console.print("Warning: ResourceMonitor has no stop/close/terminate/shutdown method; skipping explicit cleanup.")


def get_image_paths(row: dict[str, Any]) -> list[str]:
    images = maybe_json_load(row.get("images"), default=[])
    if isinstance(images, list):
        paths = [str(path).strip() for path in images if str(path).strip()]
        if paths:
            return paths
    if isinstance(images, str) and images.strip():
        return [images.strip()]
    for key in ("image_path", "image", "volume_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    raise KeyError(f"No image path found for uid={row.get('uid', '<unknown>')}")


def load_image(path: str, image_size: int, resize_mode: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if image_size and image_size > 0:
        resample = Image.Resampling.BICUBIC
        if resize_mode == "square":
            image = image.resize((image_size, image_size), resample)
        elif resize_mode == "longest":
            image.thumbnail((image_size, image_size), resample)
        elif resize_mode != "none":
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
    return image


def load_row_images(row: dict[str, Any], image_size: int, resize_mode: str, max_images_per_sample: int):
    paths = get_image_paths(row)
    if max_images_per_sample and max_images_per_sample > 0:
        paths = paths[:max_images_per_sample]
    return [load_image(path, image_size, resize_mode) for path in paths]


def normalize_task_type(row: dict[str, Any]) -> str:
    task = row.get("task_type") or row.get("task") or row.get("subtask") or "disease_diagnosis_classification"
    task = str(task).strip().lower().replace(" ", "_").replace("-", "_")
    return {
        "classification": "disease_diagnosis_classification",
        "cls": "disease_diagnosis_classification",
        "multi_label": "multi_label_classification",
        "multilabel": "multi_label_classification",
        "count": "cell_counting",
        "counting": "cell_counting",
        "report": "report_generation",
    }.get(task, task)


def build_prompt(row: dict[str, Any]) -> str:
    task = normalize_task_type(row)
    prompt = as_text(row.get("prompt") or row.get("question") or "")
    choices = maybe_json_load(row.get("choices"), default=[])
    if not isinstance(choices, list):
        choices = []
    parts = [TASK_INSTRUCTIONS.get(task, "Answer the medical imaging question using the provided image.")]
    if prompt:
        parts.append(prompt)
    if choices and "options:" not in prompt.lower():
        parts.append("Options: " + "; ".join(str(choice) for choice in choices))
    return "\n\n".join(parts)


def extract_answer(row: dict[str, Any]) -> str:
    answer = maybe_json_load(row.get("raw_answer"), default=None)
    if answer is None:
        answer = row.get("answer", "")
    if isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def make_messages(num_images: int, prompt: str, answer: str, system_prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def find_subsequence(sequence, pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    target = torch.tensor(pattern, device=sequence.device, dtype=sequence.dtype)
    length = len(pattern)
    for index in range(0, sequence.numel() - length + 1):
        if torch.equal(sequence[index:index + length], target):
            return index
    return -1


@dataclass
class ImageSFTCollator:
    processor: Any
    image_size: int = 896
    resize_mode: str = "square"
    max_images_per_sample: int = 1
    max_length: int | None = None
    system_prompt: str = "You are an expert medical imaging assistant."
    mask_prompt_tokens: bool = True

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        batch_images = []
        for example in examples:
            images = load_row_images(example, self.image_size, self.resize_mode, self.max_images_per_sample)
            messages = make_messages(len(images), build_prompt(example), extract_answer(example), self.system_prompt)
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip()
            else:
                text = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip()
            texts.append(text)
            batch_images.append(images)

        proc_kwargs = {"text": texts, "images": batch_images, "return_tensors": "pt", "padding": True}
        if self.max_length:
            proc_kwargs.update({"truncation": True, "max_length": self.max_length})
        batch = self.processor(**proc_kwargs)
        labels = batch["input_ids"].clone()
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        for token_id in image_token_ids(tokenizer):
            labels[labels == token_id] = -100

        if self.mask_prompt_tokens:
            marker_lists = [tokenizer.encode(marker, add_special_tokens=False) for marker in ("<start_of_turn>model\n", "model\n")]
            for row_index in range(labels.shape[0]):
                for marker_tokens in marker_lists:
                    found = find_subsequence(batch["input_ids"][row_index], marker_tokens)
                    if found >= 0:
                        labels[row_index, :found + len(marker_tokens)] = -100
                        break

        batch["labels"] = labels
        return batch


def image_token_ids(tokenizer: Any) -> set[int]:
    token_ids = set()
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for key in ("boi_token", "image_token"):
        token = special_map.get(key)
        if token:
            try:
                token_ids.add(int(tokenizer.convert_tokens_to_ids(token)))
            except Exception:
                pass
    token_ids.add(262144)
    return {token_id for token_id in token_ids if token_id >= 0}
