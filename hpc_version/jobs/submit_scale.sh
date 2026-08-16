#!/bin/bash
# T5-base then T5-large, 3 seeds each, on RTX Pro 6000 (96GB).
# Requires the mono_s2s_cu128 conda env (see hpc_version/README.md).
set -euo pipefail
cd "$(dirname "$0")"
export CURC_PARTITION="${CURC_PARTITION:-artxpro6000}"
export USE_FULL_TEST_SETS=1
export CONDA_ENV="${CONDA_ENV:-mono_s2s_cu128}"

SEEDS="${SCALE_SEEDS:-42 1337 2024}"
MODELS="${SCALE_MODELS:-t5-base t5-large}"

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    echo "=== $MODEL seed $SEED ==="
    T5_MODEL_NAME=$MODEL EXPERIMENT_SEED=$SEED ./submit_pipeline.sh
  done
done
