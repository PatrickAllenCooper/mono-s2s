#!/bin/bash
# Attribution-ablation arms on seed 42 (extend seeds if results are close).
set -euo pipefail
cd "$(dirname "$0")"
export EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
export T5_MODEL_NAME="${T5_MODEL_NAME:-t5-small}"
export USE_FULL_TEST_SETS="${USE_FULL_TEST_SETS:-1}"
export CURC_PARTITION="${CURC_PARTITION:-aa100}"

for MODE in sign_frozen abs_init_free; do
  echo "=== ablation $MODE ==="
  T5_ABLATION_MODE=$MODE ./submit_pipeline.sh
done
