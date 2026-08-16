#!/bin/bash
# Shared CURC environment for foundation-model jobs. Source after the SBATCH header.

CONDA_BASE="${CONDA_BASE:-/projects/$USER/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || {
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV'"
    exit 1
}

export PYTHONHASHSEED="${EXPERIMENT_SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
export FOUNDATION_MODEL_NAME="${FOUNDATION_MODEL_NAME:-${PYTHIA_MODEL_NAME:-EleutherAI/pythia-1.4b}}"
export PYTHIA_MODEL_NAME="${PYTHIA_MODEL_NAME:-$FOUNDATION_MODEL_NAME}"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"
export PROJECT="${PROJECT:-/projects/$USER}"
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "Seed=$EXPERIMENT_SEED model=$FOUNDATION_MODEL_NAME variant=$MONOTONIC_VARIANT env=$CONDA_ENV"
if [[ "$FOUNDATION_MODEL_NAME" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: $FOUNDATION_MODEL_NAME is a gated HuggingFace repo."
    echo "Accept the license on huggingface.co and export HF_TOKEN=hf_..."
    exit 1
fi
