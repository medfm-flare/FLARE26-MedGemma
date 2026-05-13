# MedGemma-FLARE-2D

## Usage

This implementation fine-tunes `google/medgemma-1.5-4b-it` on FLARE-MLLM-2D and evaluates the six requested FLARE
tasks:

- disease diagnostic classification: balanced accuracy
- cell counting: mean absolute error
- detection: F1 with IoU > 0.5 matching
- multi-label classification: example-level F1
- regression: mean absolute error
- report generation: GREEN score using `ATATC/GREEN`

The dataset directory must be available at:

```text
{INPUT_DIR}/{DATASET_NAME}
```

For the local dataset path in the prompt, use:

```text
INPUT_DIR=/scratch/atatc/input
DATASET_NAME=FLARE-MLLM-2D
```

### Install

If you are working inside a fork of MLE, you can install it directly from GitHub.

```shell
pip install git+https://github.com/your-username/your-forked-repo
```

If you cloned MLE and are working locally, upload the source files to "/workspace/app" and install it from there.

```shell
cd /workspace/app
pip install -e .
```

On a Slurm cluster, install inside a virtual environment on scratch or another writable project filesystem. Do not
install packages globally.

### Preprocess

Preprocessing converts FLARE-MLLM-2D JSON annotations into MedGemma SFT JSONL files and, when `datasets` is installed,
an Arrow dataset under `{OUTPUT_DIR}/Preprocessed-FLARE-MLLM-2D/hf_dataset`.

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  preprocess
```

Useful preprocessing custom args:

```yaml
# preprocess.yaml
tasks:
  - disease_diagnosis_classification
  - cell_counting
  - detection
  - multi_label_classification
  - regression
  - report_generation
allow_missing_images: false
include_unanswered: false
no_hf_dataset: false
```

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  --custom_args preprocess.yaml \
  preprocess
```

### Train

Training uses QLoRA/LoRA SFT by default. Run it on a CUDA GPU, not on a shared login node.

Example:

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  --custom_args train-medgemma.yaml \
  train \
  --num_epochs 1 \
  --batch_size 1 \
  --learning_rate 2e-4
```

Example training custom args:

```yaml
# train-medgemma.yaml
model_name_or_path: google/medgemma-1.5-4b-it
image_size: 896
resize_mode: square
max_images_per_sample: 1
gradient_accumulation_steps: 16
max_eval_samples: 256
load_in_4bit: true
lora_rank: 16
lora_alpha: 16
lora_dropout: 0.05
attn_implementation: auto
gradient_checkpointing: true
save_steps: 200
eval_steps: 200
save_total_limit: 3
```

The final adapter is saved by default to:

```text
{OUTPUT_DIR}/{EXPERIMENT_NAME}-medgemma15-lora/final
```

### Smoke Test Mode

Pass `--smoke_test` to run a tiny end-to-end check before committing cluster time to a full run. The engine keeps the
same preprocessing, training, inference, and evaluation code paths but applies small defaults:

- preprocessing limits rows per source JSON to 32 unless `max_rows_per_json` is set
- training limits to 8 train rows, 4 validation rows, 2 optimizer steps, smaller LoRA rank, and infrequent checkpointing
- inference limits to 4 rows, batch size 1, 512px images, and 32 generated tokens
- evaluation limits to 4 rows and skips the heavy GREEN scorer by default; set `skip_green_score: false` to force it

Example:

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output-smoke" \
  --custom_args train-medgemma.yaml \
  --smoke_test \
  train \
  --num_epochs 1 \
  --batch_size 1 \
  --learning_rate 2e-4
```

### Run Inference

Inference generates predictions with the fine-tuned adapter when available, then writes:

- `{OUTPUT_DIR}/{EXPERIMENT_NAME}-infer/{split}_predictions.jsonl`
- `{OUTPUT_DIR}/{EXPERIMENT_NAME}-infer/inference_details.json`

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  --custom_args infer-medgemma.yaml \
  infer \
  classification \
  cell_counting \
  detection \
  multi_label_classification \
  regression \
  report_generation
```

Example inference custom args:

```yaml
# infer-medgemma.yaml
split: validation
model_name_or_path: google/medgemma-1.5-4b-it
image_size: 896
resize_mode: square
max_images_per_sample: 1
batch_size: 1
max_new_tokens: 256
temperature: 0.0
```

### Infer With Base Model

To infer with `google/medgemma-1.5-4b-it` without loading any fine-tuned adapter, set `base_model: true` in custom args.
This is intentionally controlled through `custom_args`; the CLI parser is left unchanged by the engine implementation.

```yaml
# infer-base.yaml
base_model: true
split: validation
model_name_or_path: google/medgemma-1.5-4b-it
image_size: 896
resize_mode: square
max_images_per_sample: 1
batch_size: 1
max_new_tokens: 256
temperature: 0.0
```

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  --custom_args infer-base.yaml \
  infer \
  classification \
  cell_counting \
  detection \
  multi_label_classification \
  regression \
  report_generation
```

### Evaluate Predictions

Evaluation computes metrics from prediction files and does not load MedGemma. By default it looks for predictions at
`{OUTPUT_DIR}/{EXPERIMENT_NAME}-infer/{split}_predictions.jsonl`; pass `predictions` to score a different file.

```yaml
# eval-medgemma.yaml
split: validation
predictions: /path/to/predictions.jsonl  # optional if using the default inference output path
allow_missing_predictions: false
iou_threshold: 0.5
green_model_name: StanfordAIMI/GREEN-radllama2-7b
green_batch_size: 8
green_max_length: 2048
```

```shell
python -m mle \
  -d FLARE-MLLM-2D \
  --root_dir "$PWD" \
  --input_dir "/scratch/atatc/input" \
  --output_dir "$PWD/output" \
  --custom_args eval-medgemma.yaml \
  evaluate \
  classification \
  cell_counting \
  detection \
  multi_label_classification \
  regression \
  report_generation
```

Prediction files can be JSONL rows with `uid` and `prediction`, a JSON list of such rows, or a JSON object mapping
`uid` to prediction text.
