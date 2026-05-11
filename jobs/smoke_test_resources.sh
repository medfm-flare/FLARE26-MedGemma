#!/bin/bash
# Submit tiny dependent smoke jobs and collect resource diagnostics for the
# preprocess, fine-tune, infer, and evaluate pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
USERNAME="${USERNAME:-atatc}"
CPU_ACCOUNT="${CPU_ACCOUNT:-def-jma-ab}"
GPU_ACCOUNT="${GPU_ACCOUNT:-rrg-jma}"
DATASET_NAME="${DATASET_NAME:-FLARE-MLLM-2D}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flare-medgemma-smoke-${RUN_ID}}"

# DATA_ROOT must contain ${DATASET_NAME} as a child directory.
DATA_ROOT="${DATA_ROOT:-/project/rrg-jma/${USERNAME}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/rrg-jma/${USERNAME}/medgemma-flare-2d-smoke/${RUN_ID}}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USERNAME}/medgemma-flare-2d-smoke/${RUN_ID}}"
SMOKE_DIR="${SMOKE_DIR:-${PROJECT_ROOT}/logs/smoke_resources/${RUN_ID}}"
GENERATED_DIR="${SMOKE_DIR}/generated"

GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_1g.10gb}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_CPUS="${GPU_CPUS:-2}"
GPU_MEM="${GPU_MEM:-16G}"
CPU_CPUS="${CPU_CPUS:-2}"
CPU_MEM="${CPU_MEM:-8G}"

PREPROCESS_TIME="${PREPROCESS_TIME:-00:30:00}"
TRAIN_TIME="${TRAIN_TIME:-01:00:00}"
INFER_TIME="${INFER_TIME:-00:45:00}"
EVALUATE_TIME="${EVALUATE_TIME:-00:45:00}"
SUMMARY_TIME="${SUMMARY_TIME:-00:10:00}"

SMOKE_MAX_ROWS_PER_JSON="${SMOKE_MAX_ROWS_PER_JSON:-24}"
SMOKE_MAX_TRAIN_SAMPLES="${SMOKE_MAX_TRAIN_SAMPLES:-4}"
SMOKE_MAX_EVAL_SAMPLES="${SMOKE_MAX_EVAL_SAMPLES:-2}"
SMOKE_MAX_INFER_SAMPLES="${SMOKE_MAX_INFER_SAMPLES:-2}"
SMOKE_MAX_EVALUATE_SAMPLES="${SMOKE_MAX_EVALUATE_SAMPLES:-2}"
SMOKE_IMAGE_SIZE="${SMOKE_IMAGE_SIZE:-512}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-64}"
SMOKE_NO_EXTRACT_ARCHIVES="${SMOKE_NO_EXTRACT_ARCHIVES:-true}"
SMOKE_TASKS="${SMOKE_TASKS:-classification cell_counting detection multi_label_classification regression report_generation}"
SMOKE_SPLIT="${SMOKE_SPLIT:-validation}"

mkdir -p "$SMOKE_DIR" "$GENERATED_DIR"

PREPROCESS_SCRIPT="${GENERATED_DIR}/01_preprocess.sbatch"
TRAIN_SCRIPT="${GENERATED_DIR}/02_train.sbatch"
INFER_SCRIPT="${GENERATED_DIR}/03_infer.sbatch"
EVALUATE_SCRIPT="${GENERATED_DIR}/04_evaluate.sbatch"
SUMMARY_SCRIPT="${GENERATED_DIR}/05_summarize.sbatch"
JOB_IDS_FILE="${SMOKE_DIR}/job_ids.env"

cat > "$PREPROCESS_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=smoke-preprocess
#SBATCH --account=${CPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_CPUS}
#SBATCH --mem=${CPU_MEM}
#SBATCH --time=${PREPROCESS_TIME}
#SBATCH --output=${SMOKE_DIR}/%x_%j.out
#SBATCH --error=${SMOKE_DIR}/%x_%j.err

set -euo pipefail

cd "${PROJECT_ROOT}"
mkdir -p logs/configs "${OUTPUT_ROOT}" "${SCRATCH_BASE}" "${SMOKE_DIR}"

if [[ -n "\${CONDA_ENV:-}" ]]; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV"
elif [[ -n "\${VENV_PATH:-}" ]]; then
  source "\$VENV_PATH/bin/activate"
fi

export HF_HOME="\${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="\${TMPDIR:-${SCRATCH_BASE}/tmp/\${SLURM_JOB_ID}}"
export WANDB_DISABLED=true
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$TMPDIR"

CONFIG_PATH="${SMOKE_DIR}/preprocess_\${SLURM_JOB_ID}.yaml"
cat > "\$CONFIG_PATH" <<'YAML'
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
max_rows_per_json: ${SMOKE_MAX_ROWS_PER_JSON}
no_extract_archives: ${SMOKE_NO_EXTRACT_ARCHIVES}
YAML

echo "Stage: preprocess"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node:   \$(hostname)"
echo "Start:  \$(date)"
/usr/bin/time -v -o "${SMOKE_DIR}/preprocess_time_\${SLURM_JOB_ID}.txt" \
  python -m mle \
    -n "${EXPERIMENT_NAME}" \
    -d "${DATASET_NAME}" \
    --config slurm \
    --suser "${USERNAME}" \
    --root_dir "${PROJECT_ROOT}" \
    --input_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --custom_args "\$CONFIG_PATH" \
    preprocess
echo "End: \$(date)"
EOF

cat > "$TRAIN_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=smoke-train
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${SMOKE_DIR}/%x_%j.out
#SBATCH --error=${SMOKE_DIR}/%x_%j.err

set -euo pipefail

cd "${PROJECT_ROOT}"
mkdir -p logs/configs "${OUTPUT_ROOT}" "${SCRATCH_BASE}" "${SMOKE_DIR}"

if [[ -n "\${CONDA_ENV:-}" ]]; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV"
elif [[ -n "\${VENV_PATH:-}" ]]; then
  source "\$VENV_PATH/bin/activate"
fi

export HF_HOME="\${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="\${TMPDIR:-${SCRATCH_BASE}/tmp/\${SLURM_JOB_ID}}"
export WANDB_DISABLED=true
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$TMPDIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "${SMOKE_DIR}/train_gpu_\${SLURM_JOB_ID}.csv" &
  GPU_MONITOR_PID=\$!
  trap 'kill "\${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT
fi

CONFIG_PATH="${SMOKE_DIR}/train_\${SLURM_JOB_ID}.yaml"
cat > "\$CONFIG_PATH" <<YAML
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma15-lora
image_size: ${SMOKE_IMAGE_SIZE}
resize_mode: square
max_images_per_sample: 1
max_train_samples: ${SMOKE_MAX_TRAIN_SAMPLES}
max_eval_samples: ${SMOKE_MAX_EVAL_SAMPLES}
per_device_eval_batch_size: 1
gradient_accumulation_steps: 1
load_in_4bit: true
lora_rank: 8
lora_alpha: 8
lora_dropout: 0.05
attn_implementation: auto
gradient_checkpointing: true
save_steps: 10
eval_steps: 10
save_total_limit: 1
dataloader_num_workers: 0
dataloader_pin_memory: false
resource_monitor: true
seed: 42
YAML

echo "Stage: train"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node:   \$(hostname)"
echo "GPU:    ${GPU_TYPE}:${GPU_COUNT}"
echo "Start:  \$(date)"
/usr/bin/time -v -o "${SMOKE_DIR}/train_time_\${SLURM_JOB_ID}.txt" \
  python -m mle \
    -n "${EXPERIMENT_NAME}" \
    -d "${DATASET_NAME}" \
    --config slurm \
    --suser "${USERNAME}" \
    --root_dir "${PROJECT_ROOT}" \
    --input_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --custom_args "\$CONFIG_PATH" \
    train \
    --num_epochs 1 \
    --batch_size 1 \
    --learning_rate 2e-4
echo "End: \$(date)"
EOF

cat > "$INFER_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=smoke-infer
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${INFER_TIME}
#SBATCH --output=${SMOKE_DIR}/%x_%j.out
#SBATCH --error=${SMOKE_DIR}/%x_%j.err

set -euo pipefail

cd "${PROJECT_ROOT}"
mkdir -p logs/configs "${OUTPUT_ROOT}" "${SCRATCH_BASE}" "${SMOKE_DIR}" "${OUTPUT_ROOT}/${EXPERIMENT_NAME}-infer"

if [[ -n "\${CONDA_ENV:-}" ]]; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV"
elif [[ -n "\${VENV_PATH:-}" ]]; then
  source "\$VENV_PATH/bin/activate"
fi

export HF_HOME="\${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="\${TMPDIR:-${SCRATCH_BASE}/tmp/\${SLURM_JOB_ID}}"
export WANDB_DISABLED=true
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$TMPDIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "${SMOKE_DIR}/infer_gpu_\${SLURM_JOB_ID}.csv" &
  GPU_MONITOR_PID=\$!
  trap 'kill "\${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT
fi

read -r -a TASK_LIST <<< "${SMOKE_TASKS}"
CONFIG_PATH="${SMOKE_DIR}/infer_\${SLURM_JOB_ID}.yaml"
cat > "\$CONFIG_PATH" <<YAML
split: ${SMOKE_SPLIT}
model_name_or_path: google/medgemma-1.5-4b-it
model_output_dir: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-medgemma15-lora
infer_output_dir: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-infer
predictions_out: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-infer/${SMOKE_SPLIT}_predictions.jsonl
image_size: ${SMOKE_IMAGE_SIZE}
resize_mode: square
max_images_per_sample: 1
batch_size: 1
max_samples: ${SMOKE_MAX_INFER_SAMPLES}
max_new_tokens: ${SMOKE_MAX_NEW_TOKENS}
temperature: 0.0
load_in_4bit: true
YAML

echo "Stage: infer"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node:   \$(hostname)"
echo "GPU:    ${GPU_TYPE}:${GPU_COUNT}"
echo "Start:  \$(date)"
/usr/bin/time -v -o "${SMOKE_DIR}/infer_time_\${SLURM_JOB_ID}.txt" \
  python -m mle \
    -n "${EXPERIMENT_NAME}" \
    -d "${DATASET_NAME}" \
    --config slurm \
    --suser "${USERNAME}" \
    --root_dir "${PROJECT_ROOT}" \
    --input_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --custom_args "\$CONFIG_PATH" \
    infer \
    "\${TASK_LIST[@]}"
echo "End: \$(date)"
EOF

cat > "$EVALUATE_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=smoke-evaluate
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${EVALUATE_TIME}
#SBATCH --output=${SMOKE_DIR}/%x_%j.out
#SBATCH --error=${SMOKE_DIR}/%x_%j.err

set -euo pipefail

cd "${PROJECT_ROOT}"
mkdir -p logs/configs "${OUTPUT_ROOT}" "${SCRATCH_BASE}" "${SMOKE_DIR}" "${OUTPUT_ROOT}/${EXPERIMENT_NAME}-eval"

if [[ -n "\${CONDA_ENV:-}" ]]; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV"
elif [[ -n "\${VENV_PATH:-}" ]]; then
  source "\$VENV_PATH/bin/activate"
fi

export HF_HOME="\${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TMPDIR="\${TMPDIR:-${SCRATCH_BASE}/tmp/\${SLURM_JOB_ID}}"
export WANDB_DISABLED=true
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$TMPDIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "${SMOKE_DIR}/evaluate_gpu_\${SLURM_JOB_ID}.csv" &
  GPU_MONITOR_PID=\$!
  trap 'kill "\${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT
fi

read -r -a TASK_LIST <<< "${SMOKE_TASKS}"
CONFIG_PATH="${SMOKE_DIR}/evaluate_\${SLURM_JOB_ID}.yaml"
cat > "\$CONFIG_PATH" <<YAML
split: ${SMOKE_SPLIT}
predictions: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-infer/${SMOKE_SPLIT}_predictions.jsonl
eval_output_dir: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}-eval
max_samples: ${SMOKE_MAX_EVALUATE_SAMPLES}
allow_missing_predictions: false
iou_threshold: 0.5
green_model_name: StanfordAIMI/GREEN-radllama2-7b
green_batch_size: 1
green_max_length: 1024
YAML

echo "Stage: evaluate"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node:   \$(hostname)"
echo "GPU:    ${GPU_TYPE}:${GPU_COUNT}"
echo "Start:  \$(date)"
/usr/bin/time -v -o "${SMOKE_DIR}/evaluate_time_\${SLURM_JOB_ID}.txt" \
  python -m mle \
    -n "${EXPERIMENT_NAME}" \
    -d "${DATASET_NAME}" \
    --config slurm \
    --suser "${USERNAME}" \
    --root_dir "${PROJECT_ROOT}" \
    --input_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --custom_args "\$CONFIG_PATH" \
    evaluate \
    "\${TASK_LIST[@]}"
echo "End: \$(date)"
EOF

cat > "$SUMMARY_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=smoke-summary
#SBATCH --account=${CPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=${SUMMARY_TIME}
#SBATCH --output=${SMOKE_DIR}/%x_%j.out
#SBATCH --error=${SMOKE_DIR}/%x_%j.err

set -euo pipefail

SMOKE_DIR="${SMOKE_DIR}"
JOB_IDS_FILE="${JOB_IDS_FILE}"
SUMMARY_PATH="\${SMOKE_DIR}/resource_summary.md"
source "\$JOB_IDS_FILE"
ALL_JOB_IDS="\${PREPROCESS_JOB_ID},\${TRAIN_JOB_ID},\${INFER_JOB_ID},\${EVALUATE_JOB_ID}"

{
  echo "# Smoke Resource Summary"
  echo
  echo "- Generated: \$(date)"
  echo "- Dataset: ${DATASET_NAME}"
  echo "- Experiment: ${EXPERIMENT_NAME}"
  echo "- Output root: ${OUTPUT_ROOT}"
  echo "- GPU smoke request: ${GPU_TYPE}:${GPU_COUNT}, cpus=${GPU_CPUS}, mem=${GPU_MEM}"
  echo "- CPU smoke request: cpus=${CPU_CPUS}, mem=${CPU_MEM}"
  echo "- Sample caps: preprocess max_rows_per_json=${SMOKE_MAX_ROWS_PER_JSON}, train=${SMOKE_MAX_TRAIN_SAMPLES}, infer=${SMOKE_MAX_INFER_SAMPLES}, evaluate=${SMOKE_MAX_EVALUATE_SAMPLES}"
  echo "- Archive extraction during smoke preprocess: disabled=${SMOKE_NO_EXTRACT_ARCHIVES}"
  echo
  echo "## Job IDs"
  echo
  echo "- preprocess: \${PREPROCESS_JOB_ID}"
  echo "- train: \${TRAIN_JOB_ID}"
  echo "- infer: \${INFER_JOB_ID}"
  echo "- evaluate: \${EVALUATE_JOB_ID}"
  echo
  echo "## sacct"
  echo
  echo '~~~text'
  sacct -j "\$ALL_JOB_IDS" --format=JobID%15,JobName%28,Account%15,State%18,ExitCode%10,Elapsed%12,Timelimit%12,ReqMem%10,MaxRSS%12,AllocTRES%60 --noheader || true
  echo '~~~'
  echo
  echo "## seff"
  for job_id in "\${PREPROCESS_JOB_ID}" "\${TRAIN_JOB_ID}" "\${INFER_JOB_ID}" "\${EVALUATE_JOB_ID}"; do
    echo
    echo "### \${job_id}"
    echo '~~~text'
    seff "\$job_id" || true
    echo '~~~'
  done
  echo
  echo "## /usr/bin/time summaries"
  for path in "\${SMOKE_DIR}"/*_time_*.txt; do
    [[ -e "\$path" ]] || continue
    echo
    echo "### \$(basename "\$path")"
    echo '~~~text'
    grep -E "Elapsed|Maximum resident set size|Percent of CPU|Exit status" "\$path" || cat "\$path"
    echo '~~~'
  done
  echo
  echo "## GPU memory traces"
  for path in "\${SMOKE_DIR}"/*_gpu_*.csv; do
    [[ -e "\$path" ]] || continue
    echo
    echo "### \$(basename "\$path")"
    echo '~~~text'
    tail -n 12 "\$path"
    echo '~~~'
  done
  echo
  echo "## Failure pattern scan"
  echo
  echo '~~~text'
  grep -Eih "CUDA out of memory|out of memory|oom-kill|Killed|TIMEOUT|No space left on device|ModuleNotFoundError|Permission denied|Bus error|Segmentation fault|DependencyNeverSatisfied|RuntimeError: CUDA" "\${SMOKE_DIR}"/*.out "\${SMOKE_DIR}"/*.err 2>/dev/null || echo "No common failure patterns found in smoke logs."
  echo '~~~'
  echo
  echo "## How to use this"
  echo
  echo "- If a GPU smoke job reports CUDA OOM, retry with GPU_TYPE=nvidia_h100_80gb_hbm3_2g.20gb, then 3g.40gb, then h100."
  echo "- If MaxRSS is close to ReqMem, set production --mem to at least 1.5x the observed MaxRSS."
  echo "- If elapsed time is close to Timelimit, set production --time to at least 2x the observed smoke elapsed and scale for the full sample count."
  echo "- If the 10 GB MIG smoke succeeds, keep production defaults conservative until full-size seff reports confirm lower needs."
  echo "- If preprocessing fails because image files are still zipped, rerun with SMOKE_NO_EXTRACT_ARCHIVES=false."
} > "\$SUMMARY_PATH"

echo "Wrote \$SUMMARY_PATH"
EOF

chmod +x "$PREPROCESS_SCRIPT" "$TRAIN_SCRIPT" "$INFER_SCRIPT" "$EVALUATE_SCRIPT" "$SUMMARY_SCRIPT"

echo "Smoke run: $RUN_ID"
echo "Generated scripts: $GENERATED_DIR"
echo "Logs and summary: $SMOKE_DIR"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not submitting. Generated scripts are ready for inspection."
  exit 0
fi

PREPROCESS_JOB_ID="$(sbatch --parsable "$PREPROCESS_SCRIPT")"
TRAIN_JOB_ID="$(sbatch --parsable --dependency=afterany:${PREPROCESS_JOB_ID} "$TRAIN_SCRIPT")"
INFER_JOB_ID="$(sbatch --parsable --dependency=afterany:${TRAIN_JOB_ID} "$INFER_SCRIPT")"
EVALUATE_JOB_ID="$(sbatch --parsable --dependency=afterany:${INFER_JOB_ID} "$EVALUATE_SCRIPT")"

cat > "$JOB_IDS_FILE" <<EOF
PREPROCESS_JOB_ID=${PREPROCESS_JOB_ID}
TRAIN_JOB_ID=${TRAIN_JOB_ID}
INFER_JOB_ID=${INFER_JOB_ID}
EVALUATE_JOB_ID=${EVALUATE_JOB_ID}
EOF

SUMMARY_JOB_ID="$(sbatch --parsable --dependency=afterany:${EVALUATE_JOB_ID} "$SUMMARY_SCRIPT")"

cat >> "$JOB_IDS_FILE" <<EOF
SUMMARY_JOB_ID=${SUMMARY_JOB_ID}
EOF

echo "Submitted smoke jobs:"
echo "  preprocess: ${PREPROCESS_JOB_ID}"
echo "  train:      ${TRAIN_JOB_ID}"
echo "  infer:      ${INFER_JOB_ID}"
echo "  evaluate:   ${EVALUATE_JOB_ID}"
echo "  summary:    ${SUMMARY_JOB_ID}"
echo
echo "Monitor with:"
echo "  squeue -j ${PREPROCESS_JOB_ID},${TRAIN_JOB_ID},${INFER_JOB_ID},${EVALUATE_JOB_ID},${SUMMARY_JOB_ID}"
echo
echo "After the summary job finishes, read:"
echo "  ${SMOKE_DIR}/resource_summary.md"
