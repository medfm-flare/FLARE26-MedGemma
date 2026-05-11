#!/bin/bash
#SBATCH --job-name=medgemma-infer
#SBATCH --account=rrg-jma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USERNAME}/app/MedGemma-FLARE-2D}"
USERNAME="${USERNAME:-atatc}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma}"

# DATA_ROOT must contain ${DATASET_NAME} as a child directory.
DATA_ROOT="${DATA_ROOT:-/scratch/${USERNAME}/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USERNAME}/output/medgemma-flare-2d-output}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma15-lora}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}-infer}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${INFER_OUTPUT_DIR}/${INFER_SPLIT:-validation}_predictions.jsonl}"

cd "$PROJECT_ROOT"
mkdir -p logs logs/configs "$OUTPUT_ROOT" "$SCRATCH_BASE" "$INFER_OUTPUT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-0}"
echo "Start:  $(date)"
echo "Repo:   $PROJECT_ROOT"
echo "Data:   $DATA_ROOT/$DATASET_NAME"
echo "Output: $OUTPUT_ROOT"
echo "Model:  $MODEL_OUTPUT_DIR"
echo "Preds:  $PREDICTIONS_OUT"
echo "---"

# Optional environment activation. Set one of these before sbatch if needed:
#   export CONDA_ENV=medgemma
export VENV_PATH=/scratch/atatc/venv
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

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  for token_file in "${HF_TOKEN_FILE:-}" "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
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
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-medgemma15-flare-mllm-2d}"
  unset WANDB_DISABLED
  mkdir -p "$WANDB_DIR"
  WANDB_FLAG=(--wandb)
else
  export WANDB_DISABLED=true
fi

read -r -a TASK_LIST <<< "${TASKS:-classification cell_counting detection multi_label_classification regression report_generation}"

# The repo's `mle infer` entry point is currently a stub, so this uses the
# evaluation engine's generation path and writes predictions_out.
CONFIG_PATH="logs/configs/infer_${SLURM_JOB_ID:-manual}.yaml"
cat > "$CONFIG_PATH" <<YAML
split: ${INFER_SPLIT:-validation}
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${MODEL_OUTPUT_DIR}
eval_output_dir: ${INFER_OUTPUT_DIR}
predictions_out: ${PREDICTIONS_OUT}
image_size: ${IMAGE_SIZE:-896}
resize_mode: square
max_images_per_sample: 1
batch_size: ${INFER_BATCH_SIZE:-1}
max_new_tokens: ${MAX_NEW_TOKENS:-256}
temperature: ${TEMPERATURE:-0.0}
load_in_4bit: true
iou_threshold: ${IOU_THRESHOLD:-0.5}
green_model_name: StanfordAIMI/GREEN-radllama2-7b
green_batch_size: ${GREEN_BATCH_SIZE:-8}
green_max_length: ${GREEN_MAX_LENGTH:-2048}
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
  evaluate \
  "${TASK_LIST[@]}"

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
