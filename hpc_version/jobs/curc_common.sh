#!/bin/bash
# Shared CURC environment for T5 jobs. Source after the SBATCH header.
# Partition / gres are chosen at sbatch time (see submit_pipeline.sh).

CONDA_BASE="${CONDA_BASE:-/projects/$USER/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV"

export PYTHONHASHSEED="${EXPERIMENT_SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
export T5_ABLATION_MODE="${T5_ABLATION_MODE:-nonneg}"
export T5_MODEL_NAME="${T5_MODEL_NAME:-t5-small}"
export USE_FULL_TEST_SETS="${USE_FULL_TEST_SETS:-1}"
export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"
export PROJECT="${PROJECT:-/projects/$USER}"
export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Seed=$EXPERIMENT_SEED model=$T5_MODEL_NAME ablation=$T5_ABLATION_MODE full_test=$USE_FULL_TEST_SETS env=$CONDA_ENV"
