#!/bin/bash
#SBATCH --job-name=medgemma-finetune
#SBATCH --account=rrg-jma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
USERNAME="${USERNAME:-atatc}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USERNAME}/app/MedGemma-FLARE-2D}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma}"

# DATA_ROOT must contain ${DATASET_NAME} as a child directory.
DATA_ROOT="${DATA_ROOT:-/scratch/${USERNAME}/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USERNAME}/output/medgemma-flare-2d-output}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma15-lora}"

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
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

WANDB_FLAG=()
if [[ "${USE_WANDB:-0}" == "1" || "${USE_WANDB:-false}" == "true" ]]; then
  export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
  export WANDB_PROJECT="${WANDB_PROJECT:-medgemma15-flare-mllm-2d}"
  mkdir -p "$WANDB_DIR"
  WANDB_FLAG=(--wandb)
else
  export WANDB_DISABLED=true
fi

CONFIG_PATH="logs/configs/train_${SLURM_JOB_ID:-manual}.yaml"
cat > "$CONFIG_PATH" <<YAML
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${MODEL_OUTPUT_DIR}
image_size: ${IMAGE_SIZE:-896}
resize_mode: square
max_images_per_sample: 1
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS:-16}
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
  --batch_size "${BATCH_SIZE:-1}" \
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
