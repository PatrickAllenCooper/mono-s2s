#!/bin/bash
# Llama-3.2-1B and 3.2-3B (3 seeds) then Llama-3.1-8B (1-seed screen) on H200.
set -euo pipefail
cd "$(dirname "$0")"
export CURC_PARTITION="${CURC_PARTITION:-ah200}"
export CONDA_ENV="${CONDA_ENV:-mono_s2s_cu128}"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-gated_updown}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "1. Accept the licenses on huggingface.co for"
  echo "   meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.1-8B"
  echo "2. Create a read-scoped access token"
  echo "3. export HF_TOKEN=hf_..."
  exit 1
fi

for SEED in 42 1337 2024; do
  echo "=== Llama-3.2-1B seed $SEED ==="
  FOUNDATION_MODEL_NAME=meta-llama/Llama-3.2-1B EXPERIMENT_SEED=$SEED ./submit_pipeline.sh
done

for SEED in 42 1337 2024; do
  echo "=== Llama-3.2-3B seed $SEED ==="
  FOUNDATION_MODEL_NAME=meta-llama/Llama-3.2-3B EXPERIMENT_SEED=$SEED ./submit_pipeline.sh
done

echo "=== Llama-3.1-8B seed 42 (screening) ==="
FOUNDATION_MODEL_NAME=meta-llama/Llama-3.1-8B EXPERIMENT_SEED=42 ./submit_pipeline.sh
