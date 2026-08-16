#!/bin/bash
# Five-seed T5-small validation at full test-set size.
# Re-evaluates seed 42 as well so every seed uses USE_FULL_TEST_SETS=1.
set -euo pipefail
cd "$(dirname "$0")"
export USE_FULL_TEST_SETS=1
export T5_MODEL_NAME="${T5_MODEL_NAME:-t5-small}"
export T5_ABLATION_MODE="${T5_ABLATION_MODE:-nonneg}"
export CURC_PARTITION="${CURC_PARTITION:-aa100}"

for SEED in 42 1337 2024 8888 12345; do
  echo "=== seed $SEED ==="
  EXPERIMENT_SEED=$SEED ./submit_pipeline.sh
done
