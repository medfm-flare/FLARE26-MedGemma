#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
USERNAME="${USERNAME:-atatc}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma-smoke}"

# Run this script inside an interactive allocation, for example:
#   salloc --account=rrg-jma --gpus-per-node=h100:1 --mem=32G --cpus-per-task=8 --time=1:00:00
#
# If your cluster accepts the older spelling, the user's requested allocation was:
#   salloc --account=rrg-jma --gres=gpu:h100:1 --mem=32G --cpus-per-task=8 --time=1:00:00

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: This smoke test runs model workloads and must be started inside a Slurm allocation." >&2
  echo "Start one first, then rerun:" >&2
  echo "  salloc --account=rrg-jma --gpus-per-node=h100:1 --mem=32G --cpus-per-task=8 --time=1:00:00" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

# DATA_ROOT must contain ${DATASET_NAME} as a child directory.
DATA_ROOT="${DATA_ROOT:-/scratch/${USERNAME}/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USERNAME}/outout/medgemma-flare-2d-smoke}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d-smoke/${SLURM_JOB_ID}}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma15-lora}"
SMOKE_DIR="${SMOKE_DIR:-${OUTPUT_ROOT}/smoke-${SLURM_JOB_ID}}"
REPORT_DIR="${REPORT_DIR:-${SMOKE_DIR}/reports}"
LOG_DIR="${LOG_DIR:-${SMOKE_DIR}/logs}"
CONFIG_DIR="${CONFIG_DIR:-${SMOKE_DIR}/configs}"

PREPROCESS_MAX_ROWS="${PREPROCESS_MAX_ROWS:-32}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
TRAIN_MAX_EVAL_SAMPLES="${TRAIN_MAX_EVAL_SAMPLES:-4}"
INFER_MAX_SAMPLES="${INFER_MAX_SAMPLES:-4}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-896}"
SMOKE_TASKS="${SMOKE_TASKS:-classification cell_counting detection multi_label_classification regression report_generation}"

mkdir -p "$OUTPUT_ROOT" "$SCRATCH_BASE" "$MODEL_OUTPUT_DIR" "$SMOKE_DIR" "$REPORT_DIR" "$LOG_DIR" "$CONFIG_DIR"

echo "Smoke resource test"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-unknown}}"
echo "Start:      $(date)"
echo "Repo:       $PROJECT_ROOT"
echo "Data:       $DATA_ROOT/$DATASET_NAME"
echo "Output:     $OUTPUT_ROOT"
echo "Smoke dir:  $SMOKE_DIR"
echo "---"

# Optional environment activation. Set one of these before running if needed:
#   export CONDA_ENV=medgemma
export VENV_PATH=/scratch/atatc/venv
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="${TMPDIR:-${SCRATCH_BASE}/tmp}"
export WANDB_DISABLED=true
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${REPORT_DIR}/nvidia-smi-start.txt"
else
  echo "WARNING: nvidia-smi was not found; GPU memory monitoring will be skipped." >&2
fi

read -r -a TASK_LIST <<< "$SMOKE_TASKS"
SUMMARY_CSV="${REPORT_DIR}/smoke-summary.csv"
SUMMARY_MD="${REPORT_DIR}/smoke-summary.md"
printf "stage,exit_code,elapsed_seconds,max_rss_kb,peak_gpu_mem_mib,log_path\n" > "$SUMMARY_CSV"

detect_time_command() {
  local candidate
  for candidate in "${TIME_CMD:-}" /usr/bin/time /bin/time gtime; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -v true >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

TIME_CMD="$(detect_time_command || true)"
if [[ -z "$TIME_CMD" ]]; then
  echo "WARNING: no GNU-compatible 'time -v' command was found; CPU MaxRSS will be reported as NA." >&2
fi

monitor_gpu() {
  local output_path="$1"
  : > "$output_path"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "$output_path" 2>/dev/null || true
    sleep "${GPU_POLL_SECONDS:-5}"
  done
}

peak_gpu_mib() {
  local gpu_log="$1"
  if [[ ! -s "$gpu_log" ]]; then
    echo "NA"
    return 0
  fi
  awk -F',' '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
      if ($4 + 0 > max) max = $4 + 0
    }
    END {
      if (max == "") print "NA"; else print max
    }
  ' "$gpu_log"
}

max_rss_kb() {
  local time_log="$1"
  local rss
  rss="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$time_log" | tail -1)"
  echo "${rss:-NA}"
}

diagnose_failure() {
  local stage="$1"
  local log_path="$2"
  local diag_path="${REPORT_DIR}/${stage}.diagnosis.txt"

  {
    echo "Diagnosis hints for ${stage}"
    echo "Generated: $(date)"
    echo
    if grep -Eiq "CUDA out of memory|out of memory|oom-kill|Killed|No space left on device|ModuleNotFoundError|Permission denied|Segmentation fault|Bus error|NCCL|RuntimeError: CUDA error" "$log_path"; then
      grep -Ein "CUDA out of memory|out of memory|oom-kill|Killed|No space left on device|ModuleNotFoundError|Permission denied|Segmentation fault|Bus error|NCCL|RuntimeError: CUDA error" "$log_path" | tail -30
    else
      echo "No common Slurm/Python failure pattern matched. Last 80 log lines:"
      tail -80 "$log_path"
    fi
  } > "$diag_path"
}

run_stage() {
  local stage="$1"
  shift
  local log_path="${LOG_DIR}/${stage}.log"
  local time_log="${REPORT_DIR}/${stage}.time.txt"
  local gpu_log="${REPORT_DIR}/${stage}.gpu.csv"
  local start_ts
  local end_ts
  local elapsed
  local exit_code
  local monitor_pid=""

  echo
  echo "== ${stage} =="
  echo "Command: $*"

  monitor_gpu "$gpu_log" &
  monitor_pid="$!"
  start_ts="$(date +%s)"
  set +e
  if [[ -n "$TIME_CMD" ]]; then
    "$TIME_CMD" -v "$@" > "$log_path" 2> >(tee "$time_log" >> "$log_path")
  else
    : > "$time_log"
    "$@" > "$log_path" 2>&1
  fi
  exit_code=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi

  printf "%s,%s,%s,%s,%s,%s\n" \
    "$stage" \
    "$exit_code" \
    "$elapsed" \
    "$(max_rss_kb "$time_log")" \
    "$(peak_gpu_mib "$gpu_log")" \
    "$log_path" >> "$SUMMARY_CSV"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Stage ${stage} failed with exit code ${exit_code}; see ${log_path}" >&2
    diagnose_failure "$stage" "$log_path"
    if [[ "${CONTINUE_ON_FAILURE:-1}" != "1" ]]; then
      exit "$exit_code"
    fi
  else
    echo "Stage ${stage} completed."
  fi
}

PREPROCESS_CONFIG="${CONFIG_DIR}/preprocess-smoke.yaml"
cat > "$PREPROCESS_CONFIG" <<YAML
tasks:
  - disease_diagnosis_classification
  - cell_counting
  - detection
  - multi_label_classification
  - regression
  - report_generation
allow_missing_images: false
include_unanswered: false
max_rows_per_json: ${PREPROCESS_MAX_ROWS}
no_hf_dataset: false
YAML

TRAIN_CONFIG="${CONFIG_DIR}/train-smoke.yaml"
cat > "$TRAIN_CONFIG" <<YAML
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${MODEL_OUTPUT_DIR}
image_size: ${IMAGE_SIZE}
resize_mode: square
max_images_per_sample: 1
gradient_accumulation_steps: ${SMOKE_GRADIENT_ACCUMULATION_STEPS:-1}
max_train_samples: ${TRAIN_MAX_SAMPLES}
max_eval_samples: ${TRAIN_MAX_EVAL_SAMPLES}
load_in_4bit: true
lora_rank: ${SMOKE_LORA_RANK:-8}
lora_alpha: ${SMOKE_LORA_ALPHA:-8}
lora_dropout: 0.05
attn_implementation: auto
gradient_checkpointing: true
save_steps: 100000
eval_steps: 100000
save_total_limit: 1
dataloader_num_workers: ${SMOKE_DATALOADER_NUM_WORKERS:-0}
seed: ${SEED:-42}
YAML

INFER_CONFIG="${CONFIG_DIR}/infer-smoke.yaml"
PREDICTIONS_OUT="${SMOKE_DIR}/validation_predictions.jsonl"
cat > "$INFER_CONFIG" <<YAML
split: ${SMOKE_SPLIT:-validation}
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${MODEL_OUTPUT_DIR}
eval_output_dir: ${SMOKE_DIR}/infer
predictions_out: ${PREDICTIONS_OUT}
image_size: ${IMAGE_SIZE}
resize_mode: square
max_images_per_sample: 1
batch_size: ${SMOKE_INFER_BATCH_SIZE:-1}
max_samples: ${INFER_MAX_SAMPLES}
max_new_tokens: ${SMOKE_MAX_NEW_TOKENS:-64}
temperature: 0.0
load_in_4bit: true
iou_threshold: 0.5
green_model_name: StanfordAIMI/GREEN-radllama2-7b
green_batch_size: ${SMOKE_GREEN_BATCH_SIZE:-1}
green_max_length: ${SMOKE_GREEN_MAX_LENGTH:-1024}
YAML

EVAL_CONFIG="${CONFIG_DIR}/evaluate-smoke.yaml"
cat > "$EVAL_CONFIG" <<YAML
split: ${SMOKE_SPLIT:-validation}
predictions: ${PREDICTIONS_OUT}
eval_output_dir: ${SMOKE_DIR}/evaluate
max_samples: ${EVAL_MAX_SAMPLES}
allow_missing_predictions: false
iou_threshold: 0.5
green_model_name: StanfordAIMI/GREEN-radllama2-7b
green_batch_size: ${SMOKE_GREEN_BATCH_SIZE:-1}
green_max_length: ${SMOKE_GREEN_MAX_LENGTH:-1024}
YAML

run_stage preprocess \
  python -m mle \
    -n "$EXPERIMENT_NAME" \
    -d "$DATASET_NAME" \
    --config slurm \
    --suser "$USERNAME" \
    --root_dir "$PROJECT_ROOT" \
    --input_dir "$DATA_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    --custom_args "$PREPROCESS_CONFIG" \
    preprocess

run_stage finetune \
  python -m mle \
    -n "$EXPERIMENT_NAME" \
    -d "$DATASET_NAME" \
    --config slurm \
    --suser "$USERNAME" \
    --root_dir "$PROJECT_ROOT" \
    --input_dir "$DATA_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    --custom_args "$TRAIN_CONFIG" \
    train \
    --num_epochs "${SMOKE_NUM_EPOCHS:-1}" \
    --batch_size "${SMOKE_TRAIN_BATCH_SIZE:-1}" \
    --learning_rate "${SMOKE_LEARNING_RATE:-2e-4}"

run_stage infer \
  python -m mle \
    -n "$EXPERIMENT_NAME" \
    -d "$DATASET_NAME" \
    --config slurm \
    --suser "$USERNAME" \
    --root_dir "$PROJECT_ROOT" \
    --input_dir "$DATA_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    --custom_args "$INFER_CONFIG" \
    evaluate \
    "${TASK_LIST[@]}"

run_stage evaluate \
  python -m mle \
    -n "$EXPERIMENT_NAME" \
    -d "$DATASET_NAME" \
    --config slurm \
    --suser "$USERNAME" \
    --root_dir "$PROJECT_ROOT" \
    --input_dir "$DATA_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    --custom_args "$EVAL_CONFIG" \
    evaluate \
    "${TASK_LIST[@]}"

{
  echo "# Smoke Resource Summary"
  echo
  echo "- Job ID: \`${SLURM_JOB_ID}\`"
  echo "- Node: \`$(hostname)\`"
  echo "- Generated: \`$(date)\`"
  echo "- Samples: preprocess max rows/json=${PREPROCESS_MAX_ROWS}, train=${TRAIN_MAX_SAMPLES}, infer=${INFER_MAX_SAMPLES}, eval=${EVAL_MAX_SAMPLES}"
  echo
  echo "| Stage | Exit | Elapsed | Max RSS | Peak GPU | Log |"
  echo "|---|---:|---:|---:|---:|---|"
  tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r stage exit_code elapsed max_rss peak_gpu log_path; do
    echo "| ${stage} | ${exit_code} | ${elapsed}s | ${max_rss} KB | ${peak_gpu} MiB | \`${log_path}\` |"
  done
  echo
  echo "## Smoke-Scale Resource Floor"
  echo
  echo "| Stage | Suggested CPU Mem | Observed GPU Mem | Suggested Time Floor |"
  echo "|---|---:|---:|---:|"
  awk -F',' '
    NR > 1 {
      stage = $1
      elapsed = $3
      rss = $4
      gpu = $5
      if (rss == "NA" || rss == "") {
        mem = "NA"
      } else {
        mem_gb = int(((rss / 1024 / 1024) * 1.5) + 0.999)
        if (mem_gb < 4) mem_gb = 4
        mem = mem_gb "G"
      }
      if (gpu == "NA" || gpu == "") {
        gpu_text = "NA"
      } else {
        gpu_gb = int(((gpu / 1024) * 1.25) + 0.999)
        if (gpu_gb < 1) gpu_gb = 1
        gpu_text = gpu_gb "G"
      }
      time_seconds = int(elapsed * 2)
      if (time_seconds < 600) time_seconds = 600
      hours = int(time_seconds / 3600)
      mins = int((time_seconds % 3600) / 60)
      if (mins == 0 && hours == 0) mins = 10
      printf "| %s | %s | %s | %02d:%02d:00 |\n", stage, mem, gpu_text, hours, mins
    }
  ' "$SUMMARY_CSV"
  echo
  echo "## Interpreting This"
  echo
  echo "- The suggested CPU memory is Max RSS with 50% headroom and a 4G floor."
  echo "- The observed GPU memory is peak GPU memory with 25% headroom. Pick the smallest GPU/MIG profile above this value."
  echo "- The suggested time floor is only for smoke-scale runs. Full runs need scaling based on dataset size, epochs, and generation length."
  echo "- If any stage failed, read the matching \`${REPORT_DIR}/<stage>.diagnosis.txt\` file first."
  echo
  echo "## Slurm Accounting"
  echo
  seff "${SLURM_JOB_ID}" 2>/dev/null || echo "seff is not available yet for this interactive allocation."
  echo
  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID%15,JobName%28,Partition%12,State%12,ExitCode%10,Elapsed%12,TotalCPU%12,ReqMem%12,MaxRSS%12,AllocTRES%40 \
    --noheader 2>/dev/null || echo "sacct is not available yet for this interactive allocation."
} | tee "$SUMMARY_MD"

echo
echo "Smoke tests finished."
echo "Summary: ${SUMMARY_MD}"
echo "Raw CSV: ${SUMMARY_CSV}"
