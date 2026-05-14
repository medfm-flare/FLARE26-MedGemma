#!/bin/bash
#SBATCH --job-name=medgemma1-finetune
#SBATCH --account=rrg-jma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
USERNAME="${USERNAME:-atatc}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USERNAME}/app/MedGemma-FLARE-2D}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma1}"

# DATA_ROOT must contain ${DATASET_NAME} as a child directory.
DATA_ROOT="${DATA_ROOT:-/scratch/${USERNAME}/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USERNAME}/output/medgemma-flare-2d-output}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d}"
SLURM_HOME="${SLURM_HOME:-/scratch/${USERNAME}}"
export HOME="$SLURM_HOME"
export NETRC="${NETRC:-${SLURM_HOME}/.netrc}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma1-lora}"

cd "$PROJECT_ROOT"
mkdir -p logs logs/configs "$OUTPUT_ROOT" "$SCRATCH_BASE"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-0}"
echo "Start:  $(date)"
echo "Repo:   $PROJECT_ROOT"
echo "Data:   $DATA_ROOT/$DATASET_NAME"
echo "Output: $OUTPUT_ROOT"
echo "Model:  $MODEL_OUTPUT_DIR"
echo "---"

# Optional environment activation. Set one of these before sbatch if needed:
#   export CONDA_ENV=medgemma
module load arrow
module load opencv
module load python/3.12
module load cuda
export VENV_PATH=/scratch/atatc/venv
source "$VENV_PATH/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="${TMPDIR:-${SCRATCH_BASE}/tmp/${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_BASE}/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${SCRATCH_BASE}/xdg_config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CONFIG_HOME}/matplotlib}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_BASE}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${SCRATCH_BASE}/wandb_config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${SCRATCH_BASE}/wandb_data}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  for token_file in "${HF_TOKEN_FILE:-}" "${SLURM_HOME}/.cache/huggingface/token" "${SLURM_HOME}/.huggingface/token"; do
    if [[ -n "$token_file" && -r "$token_file" ]]; then
      export HF_TOKEN="$(< "$token_file")"
      break
    fi
  done
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Hugging Face authentication is required. Export HF_TOKEN or run 'huggingface-cli login' on the cluster." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WANDB_FLAG=()
if [[ "${USE_WANDB:-1}" == "1" || "${USE_WANDB:-true}" == "true" ]]; then
  export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
  if [[ -z "${WANDB_MODE:-}" ]]; then
    WANDB_CLUSTER_NAME="${CC_CLUSTER:-${SLURM_CLUSTER_NAME:-$(hostname -f 2>/dev/null || hostname)}}"
    if [[ "${WANDB_CLUSTER_NAME,,}" == *trillium* ]]; then
      export WANDB_MODE=offline
      echo "Detected Trillium ($WANDB_CLUSTER_NAME); using WANDB_MODE=offline. Sync later with: wandb sync $WANDB_DIR"
    else
      export WANDB_MODE=online
    fi
  else
    export WANDB_MODE
  fi
  export WANDB_PROJECT="${WANDB_PROJECT:-medgemma1-flare-mllm-2d}"
  unset WANDB_DISABLED
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
  WANDB_FLAG=(--wandb)
else
  export WANDB_DISABLED=true
fi

CONFIG_PATH="logs/configs/train_${SLURM_JOB_ID:-manual}.yaml"
cat > "$CONFIG_PATH" <<YAML
model_name_or_path: google/medgemma-4b-it
model_output_dir: ${MODEL_OUTPUT_DIR}
image_size: ${IMAGE_SIZE:-896}
resize_mode: square
max_images_per_sample: 1
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS:-2}
max_eval_samples: ${MAX_EVAL_SAMPLES:-256}
load_in_4bit: true
lora_rank: ${LORA_RANK:-16}
lora_alpha: ${LORA_ALPHA:-16}
lora_dropout: 0.05
attn_implementation: auto
gradient_checkpointing: true
save_steps: ${SAVE_STEPS:-200}
eval_steps: ${EVAL_STEPS:-200}
save_total_limit: ${SAVE_TOTAL_LIMIT:-3}
dataloader_num_workers: ${DATALOADER_NUM_WORKERS:-4}
seed: ${SEED:-42}
YAML

python -m mle \
  -n "$EXPERIMENT_NAME" \
  -d "$DATASET_NAME" \
  --config slurm \
  --suser "$USERNAME" \
  --root_dir "$PROJECT_ROOT" \
  --input_dir "$DATA_ROOT" \
  --output_dir "$OUTPUT_ROOT" \
  --custom_args "$CONFIG_PATH" \
  "${WANDB_FLAG[@]}" \
  train \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --learning_rate "${LEARNING_RATE:-2e-4}"

echo "---"
echo "End: $(date)"

# Write a post-run usage report for this job.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPORT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/logs"
  mkdir -p "$REPORT_DIR"
  SEFF_REPORT_PATH="${REPORT_DIR}/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID}_usage.txt"

  {
    echo "Usage report for Slurm job ${SLURM_JOB_ID}"
    echo "Generated: $(date)"
    echo
    echo "== seff =="
    seff "${SLURM_JOB_ID}" || echo "seff report is not available yet."
    echo
    echo "== sacct =="
    sacct -j "${SLURM_JOB_ID}" \
      --format=JobID%15,JobName%28,Partition%12,State%12,ExitCode%10,Elapsed%12,TotalCPU%12,ReqMem%12,MaxRSS%12,AllocTRES%40 \
      --noheader || echo "sacct report is not available yet."
  } | tee "${SEFF_REPORT_PATH}"
fi
