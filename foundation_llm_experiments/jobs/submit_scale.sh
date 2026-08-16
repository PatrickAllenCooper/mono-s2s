#!/bin/bash
# Pythia-2.8B (3 seeds) then Pythia-6.9B (1-seed screen) on H200.
set -euo pipefail
cd "$(dirname "$0")"
export CURC_PARTITION="${CURC_PARTITION:-ah200}"
export CONDA_ENV="${CONDA_ENV:-mono_s2s_cu128}"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"

for SEED in 42 1337 2024; do
  echo "=== pythia-2.8b seed $SEED ==="
  PYTHIA_MODEL_NAME=EleutherAI/pythia-2.8b EXPERIMENT_SEED=$SEED ./submit_pipeline.sh
done

echo "=== pythia-6.9b seed 42 (screening) ==="
PYTHIA_MODEL_NAME=EleutherAI/pythia-6.9b EXPERIMENT_SEED=42 ./submit_pipeline.sh
