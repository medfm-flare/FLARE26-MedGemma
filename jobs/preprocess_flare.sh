#!/bin/bash
#SBATCH --job-name=flare-preprocess
#SBATCH --partition=gpu
#SBATCH --account=rrg-jma
#SBATCH --constraint=nvidia_h100_80gb_hbm3_1g.10gb
#SBATCH --gres=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Submit from the repository root, or set PROJECT_ROOT explicitly.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
USERNAME="${USERNAME:-atatc}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma}"

# DATA_ROOT should contain $DATASET_NAME as a child directory.
DATA_ROOT="${DATA_ROOT:-/project/rrg-jma/${USERNAME}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/rrg-jma/${USERNAME}/medgemma-flare-2d-output}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d}"

cd "$PROJECT_ROOT"
mkdir -p logs logs/configs "$OUTPUT_ROOT" "$SCRATCH_BASE"

# Schedule a post-job efficiency report after Slurm accounting finalizes.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPORT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/logs"
  mkdir -p "$REPORT_DIR"
  SEFF_REPORT_PATH="${REPORT_DIR}/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID}_usage.txt"
  SEFF_WRAP=$(
    cat <<'SEFF'
set -euo pipefail
: "${SEFF_TARGET_JOB_ID:?missing job id}"
: "${SEFF_REPORT_PATH:?missing report path}"
mkdir -p "$(dirname "$SEFF_REPORT_PATH")"
{
  echo "Usage report for Slurm job ${SEFF_TARGET_JOB_ID}"
  echo "Generated: $(date)"
  echo
  echo "== seff =="
  seff "${SEFF_TARGET_JOB_ID}"
  echo
  echo "== sacct =="
  sacct -j "${SEFF_TARGET_JOB_ID}" \
    --format=JobID%15,JobName%28,Partition%12,State%12,ExitCode%10,Elapsed%12,TotalCPU%12,ReqMem%12,MaxRSS%12,AllocTRES%40 \
    --noheader
} | tee "${SEFF_REPORT_PATH}"
SEFF
  )
  sbatch \
    --job-name="${SLURM_JOB_NAME:-job}-seff" \
    --account=rrg-jma \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=1G \
    --output="${REPORT_DIR}/%x_%j.out" \
    --error="${REPORT_DIR}/%x_%j.err" \
    --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,SEFF_TARGET_JOB_ID="${SLURM_JOB_ID}",SEFF_REPORT_PATH="${SEFF_REPORT_PATH}" \
    --wrap="$SEFF_WRAP" >/dev/null
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-0}"
echo "Start:  $(date)"
echo "Repo:   $PROJECT_ROOT"
echo "Data:   $DATA_ROOT/$DATASET_NAME"
echo "Output: $OUTPUT_ROOT"
echo "---"

# Optional environment activation:
# export CONDA_ENV=medgemma
# export VENV_PATH=/project/rrg-jma/${USERNAME}/envs/medgemma
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-medgemma15-flare-mllm-2d}"
unset WANDB_DISABLED
export TMPDIR="${TMPDIR:-${SCRATCH_BASE}/tmp/${SLURM_JOB_ID}}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$TMPDIR"

if ! python - <<'PY'
import netrc
import os
import sys

if os.environ.get("WANDB_API_KEY"):
    sys.exit(0)

netrc_path = os.environ.get("NETRC") or os.path.expanduser("~/.netrc")
try:
    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
except (FileNotFoundError, netrc.NetrcParseError):
    auth = None

sys.exit(0 if auth and auth[2] else 1)
PY
then
  echo "WandB is not authenticated. Run 'wandb login' on the cluster or export WANDB_API_KEY before submitting." >&2
  exit 1
fi

CONFIG_PATH="logs/configs/preprocess_${SLURM_JOB_ID}.yaml"
cat > "$CONFIG_PATH" <<'YAML'
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
  --wandb \
  preprocess

echo "---"
echo "End: $(date)"
