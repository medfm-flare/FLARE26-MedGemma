# Machine Learning Engineer

## Getting Started

This codebase is a template codebase for a general machine learning workflow to apply a model onto a dataset:
preprocess $\to$ train $\to$ evaluate. Before actually running experiments, you need to install [this](SKILL.md) skill
and let an agent implement the engine.

For example, suppose we want to fine-tune MedGemma 1.5 on the FLARE-MLLM-2D dataset:

> Fine-tune MedGemma 1.5 (https://huggingface.co/google/medgemma-1.5-4b-it) on the FLARE-MLLM-2D
> dataset (https://huggingface.co/datasets/FLARE-MedFM/FLARE-MLLM-2D). For evaluation, please report:
> 
> - Balanced accuracy for the disease diagnostic classification
> - Mean Absolute Error (MAE) for cell counting
> - F1 score matching via IoU > 0.5 for detection
> - F1 score for multi-label classification
> - Mean absolute error (MAE) for regression
> - GREEN score for report generation
>
> For the GREEN score computation, use the implementation in https://github.com/ATATC/GREEN.

## Usage

### Erbium

#### Environment Setup

```shell
pip install git+https://github.com/ProjectNeura/MLE
```

### SLURM

#### Environment Setup

```shell
module load python/3.12
module load arrow
module load cuda
virtualenv /scratch/${USER}/venv
source /scratch/${USER}/venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index simpleitk  # because building wheels for simpleitk is too slow
pip install git+https://github.com/ProjectNeura/MLE
```

Use [dra-config](https://github.com/ATATC/dra-config) skills to generate the job script or use the following template.

```shell
#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 || true
module load python/3.12 || true
module load arrow || true
module load cuda || true

python -m mle ...
```

### Custom Arguments

You can have a JSON or YAML file with the arguments you want to pass to the engine.

Suppose you have "path/to/args.json", simply add a flag to the command like:

```shell
python -m mle
```