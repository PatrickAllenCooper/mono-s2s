#!/bin/bash
# Shared CURC environment for foundation-model jobs. Source after the SBATCH header.

CONDA_ENV="${CONDA_ENV:-mono_s2s}"
if [ -f "${SLURM_SUBMIT_DIR}/jobs/activate_conda.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/jobs/activate_conda.sh"
elif [ -f "${SLURM_SUBMIT_DIR}/activate_conda.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/activate_conda.sh"
else
    source "$(dirname "$0")/activate_conda.sh"
fi

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
